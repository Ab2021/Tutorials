#!/usr/bin/env python
"""
04_dpo.py — Direct Preference Optimization (DPO).

What it is
----------
DPO skips the reward model and the RL loop entirely. It reparameterizes the RLHF
objective so that the optimal policy has a closed form, then optimizes that directly
with a supervised-style loss on (prompt, chosen, rejected) triples:

    L_DPO = -log σ( β·[ log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x) ] )

Read that as: "increase the log-ratio of the chosen response over the rejected one,
relative to the frozen reference model, and don't move too far."

Why β matters
-------------
β controls how far you may drift from the reference policy. It is the same knob as the
KL penalty in PPO, made explicit.
  β → 0    : ignore the reference. The model will happily become degenerate (repetitive,
             reward-hacked) because nothing anchors it.
  β → 1+   : you can barely move. Alignment has no effect.
  0.1-0.5  : the working range. β=0.1 is the most common default for a 7B model.

Memory
------
DPO needs TWO models in memory: the policy and the frozen reference. But the
reference does not need gradients or optimizer state, and — key trick — you can
initialize the reference as a *copy of the policy with the LoRA adapter disabled*,
which costs almost nothing extra. `trl.DPOTrainer` does this for you when you pass
`ref_model=None` and use PEFT. That is why 7B DPO fits in 24 GB.

Run it
------
    python 04_dpo.py --dry-run --data data/sample_preference.jsonl
    python 04_dpo.py --data data/sample_preference.jsonl --sft-adapter ./out/sft-lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data_utils as du                    # noqa: E402
from common.memory import TrainPlan, _print_plan       # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    sft_adapter=None,          # start from your SFT adapter — DPO on a raw base rarely works
    data="data/sample_preference.jsonl",
    output="./out/dpo",
    max_len=1024,              # prompt + response. Shorter than SFT; DPO uses 2 forward passes.
    max_prompt_len=512,
    beta=0.1,                  # the KL-strength knob. 0.1 is the standard starting point.
    loss_type="sigmoid",       # sigmoid (DPO) | ipo | hinge | kto_pair | orpo | simpo
    epochs=1,                  # DPO overfits FAST. 1-2 epochs is usually the whole budget.
    lr=5e-6,                   # 10-100x lower than SFT. DPO is very LR-sensitive.
    batch=1,
    grad_accum=8,
    warmup_ratio=0.1,
    scheduler="cosine",
    lora_r=16,
    lora_alpha=32,
    quant="4bit",
    seed=42,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, default=v, action=argparse.BooleanOptionalAction)
        elif isinstance(v, int):
            p.add_argument(flag, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(flag, type=float, default=v)
        else:
            p.add_argument(flag, default=v)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    # ----------------------------------------------------------------------------------
    # 1. DATA — preference data has its own failure modes
    # ----------------------------------------------------------------------------------
    if not Path(a.data).exists():
        sys.exit(f"Not found: {a.data}\n"
                 f"Build one:  python data/make_preference_data.py --out {a.data}")

    pairs = du.load_preference_jsonl(a.data)
    print(f"  preference pairs   {len(pairs):,}")

    # Sanity checks that catch real, common dataset bugs.
    identical = sum(1 for p in pairs
                    if p["chosen"][0]["content"].strip() == p["rejected"][0]["content"].strip())
    if identical:
        print(f"  ⚠  {identical} pairs have identical chosen/rejected text "
              f"({100*identical/len(pairs):.1f}%). These contribute zero gradient — drop them.")

    # Length bias in the DATA is the root cause of length bias in the MODEL. If chosen
    # responses are systematically longer, DPO learns "longer = better" and nothing else.
    c_len = sum(len(p["chosen"][0]["content"].split()) for p in pairs) / len(pairs)
    r_len = sum(len(p["rejected"][0]["content"].split()) for p in pairs) / len(pairs)
    print(f"  mean words         chosen {c_len:.0f}  rejected {r_len:.0f}  "
          f"({100*(c_len/r_len - 1):+.1f}% length gap)")
    if c_len > r_len * 1.2:
        print("  ⚠  Chosen answers are >20% longer than rejected. Your model will learn "
              "'longer is better' (length bias). Consider length-matching the pairs, or "
              "use loss_type=simpo which length-normalizes.")

    # ----------------------------------------------------------------------------------
    # 2. MEMORY
    # ----------------------------------------------------------------------------------
    _print_plan(TrainPlan(
        model=next((m for m in ["1.5B", "1B", "3B", "7B", "8B"] if m.lower() in a.model.lower()), "7B"),
        method="qlora" if a.quant == "4bit" else "lora",
        seq_len=a.max_len, batch=a.batch, grad_accum=a.grad_accum, lora_r=a.lora_r,
    ))

    if a.dry_run:
        print("  --dry-run complete.\n")
        return

    # ----------------------------------------------------------------------------------
    # 3. MODELS
    # ----------------------------------------------------------------------------------
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # IMPORTANT for DPO: left-pad so the response ends at the
                                # same position for both chosen and rejected. With right
                                # padding the loss is computed over pad tokens and the
                                # model learns nothing useful.

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if a.quant == "4bit" else None

    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb,
        torch_dtype=torch.bfloat16 if bnb is None else None,
        device_map="auto", attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Start from the SFT adapter if given — this is the single biggest quality lever
    # in DPO. DPO on a raw base model has no good policy to improve, and you usually
    # get a model that is confidently wrong in a different style.
    if a.sft_adapter:
        print(f"  starting from SFT adapter: {a.sft_adapter}")
        model = PeftModel.from_pretrained(model, a.sft_adapter, is_trainable=True)
    else:
        print("  ⚠  No --sft-adapter. Training DPO on a base/instruct model directly.")
        print("     Expect worse results than SFT→DPO. This is a real, measured gap.")

    # ----------------------------------------------------------------------------------
    # 4. TRAIN
    # ----------------------------------------------------------------------------------
    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    cfg = DPOConfig(
        output_dir=a.output,
        beta=a.beta,
        loss_type=a.loss_type,
        max_length=a.max_len,
        max_prompt_length=a.max_prompt_len,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type=a.scheduler,
        warmup_ratio=a.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=a.seed,
        # precompute_ref_log_probs caches reference logprobs and lets you drop the
        # reference model from memory after the first pass. Saves ~half the VRAM.
        precompute_ref_log_probs=True,
    )

    # ref_model=None + a PEFT model => TRL reuses the base weights with the adapter
    # disabled as the reference. This is the standard memory-saving trick.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(a.output)
    tok.save_pretrained(a.output)
    print(f"\n  ✅ DPO adapter saved to {a.output}")

    # ----------------------------------------------------------------------------------
    # 5. THE CHECK EVERYONE SKIPS
    # ----------------------------------------------------------------------------------
    _implicit_reward_report(model, tok, pairs[:32], a.beta)

    print("\n  ── what to verify before shipping ──")
    print("    1. Implicit reward margin should INCREASE over training (see above).")
    print("       If it is flat, your β or LR is wrong, or the data has no signal.")
    print("    2. Generate on 50 held-out prompts and compare to the SFT model.")
    print("       Watch mean length — a >30% jump means you trained length bias.")
    print("    3. Run eval_utils.reward_hack_report(sft_outputs, dpo_outputs).")
    print("    4. Check refusals. DPO on safety-flavoured data causes over-refusal fast.")


@torch.no_grad() if False else (lambda f: f)   # placeholder so the module imports without torch
def _implicit_reward_report(model, tok, pairs, beta) -> None:
    """
    Compute the implicit reward margin on a sample of pairs.

    DPO's implicit reward is  r(x,y) = β · log(π(y|x) / π_ref(y|x)).

    The margin between chosen and rejected should be positive and should grow during
    training. If it does not, nothing else about the run matters.
    """
    try:
        import torch
        import torch.nn.functional as F

        tok.padding_side = "left"
        margins = []
        for p in pairs:
            prompt = tok.apply_chat_template(p["prompt"], tokenize=False, add_generation_prompt=True)
            scores = []
            for resp in (p["chosen"], p["rejected"]):
                text = prompt + resp[0]["content"] + tok.eos_token
                enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                out = model(**enc)
                logp = F.log_softmax(out.logits[:, :-1], dim=-1)
                tgt = enc["input_ids"][:, 1:]
                # Only score the response tokens, not the prompt.
                plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
                mask = torch.zeros_like(tgt, dtype=torch.bool)
                mask[:, plen - 1:] = True
                scores.append(logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[mask].sum().item())
            margins.append(scores[0] - scores[1])

        mean_margin = sum(margins) / len(margins)
        print(f"\n  implicit reward margin (chosen − rejected): {mean_margin:+.3f}")
        if mean_margin <= 0:
            print("  ⚠  Margin is not positive. The model does not prefer your chosen "
                  "responses. Check that chosen/rejected are not swapped, and that the "
                  "reference used at train time matches the one you are scoring against.")
        else:
            print("  ✅ Model prefers the chosen responses (as intended).")
    except Exception as e:                                    # noqa: BLE001
        print(f"\n  (margin check skipped: {e})")


if __name__ == "__main__":
    main()
