#!/usr/bin/env python
"""
05_orpo.py — Odds-Ratio Preference Optimization (ORPO).

What makes ORPO different
-------------------------
Every other alignment method is a SECOND stage: you SFT first, then align a copy of
that model. ORPO does both in one pass, on one dataset, with ONE model in memory.

    L_ORPO = L_SFT + λ · L_OR
           = -log p(y_w | x)                      ← supervised fine-tuning term
             - λ · log σ( log(odds(y_w)/odds(y_l)) )  ← odds-ratio preference term

where  odds(y|x) = p(y|x) / (1 − p(y|x)).

Key consequences:
  * **No reference model.** The odds ratio is computed against the model itself — the
    SFT term is what stops it drifting, so no frozen copy is needed. That is the whole
    trick, and it is why ORPO needs roughly half the memory of DPO and a quarter of PPO.
  * **No separate SFT stage.** You point it at (prompt, chosen, rejected) data and it
    learns format AND preference together.
  * **λ (lambda)** replaces β as the strength knob. Typical 0.1-1.0; the paper's default
    is 0.1 for many setups, and it is far less sensitive than DPO's β.

When to use ORPO
----------------
  ✅ You have preference pairs but no good SFT checkpoint
  ✅ You are memory-constrained (single 24GB card, 7-8B model)
  ✅ You want one training job instead of two
  ❌ You already have a strong SFT model and only need light alignment — DPO will
     usually beat ORPO there, because ORPO's SFT term will keep re-teaching format
     instead of concentrating on preference.

Run it
------
    python 05_orpo.py --dry-run --data data/sample_preference.jsonl
    python 05_orpo.py --data data/sample_preference.jsonl --lambda-orpo 0.1
    python 05_orpo.py --data data/sample_preference.jsonl --model Qwen/Qwen2.5-7B   # from BASE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data_utils as du                    # noqa: E402
from common.memory import TrainPlan, _print_plan, sniff_size   # noqa: E402

DEFAULTS = dict(
    # NOTE: ORPO is designed to run on a BASE model. Using an -Instruct model wastes
    # the method's main advantage and can double-align (over-refusing).
    model="Qwen/Qwen2.5-1.5B",
    data="data/sample_preference.jsonl",
    output="./out/orpo",
    max_len=1024,
    max_prompt_len=512,
    lambda_orpo=0.1,       # weight of the odds-ratio term
    epochs=1,
    lr=8e-6,               # ORPO wants a low LR; it is doing SFT + preference at once
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

    if not Path(a.data).exists():
        sys.exit(f"Not found: {a.data}\n"
                 f"Build one:  python data/make_preference_data.py --out {a.data}")

    pairs = du.load_preference_jsonl(a.data)
    print(f"  preference pairs   {len(pairs):,}")

    # ORPO trains on the chosen response with SFT loss, so it needs the SFT-quality
    # signal too. A preference set where the chosen responses are low quality will
    # teach the model to produce low-quality text confidently.
    c_len = sum(len(p["chosen"][0]["content"].split()) for p in pairs) / len(pairs)
    r_len = sum(len(p["rejected"][0]["content"].split()) for p in pairs) / len(pairs)
    print(f"  mean words         chosen {c_len:.0f}  rejected {r_len:.0f}")
    print("  ℹ  ORPO's SFT term trains on 'chosen'. Chosen responses must be GOOD "
          "enough to imitate, not merely better than rejected.")
    if c_len > r_len * 1.25:
        print("  ⚠  Large length gap in the data — ORPO will learn length bias too.")

    # ----------------------------------------------------------------------------------
    # MEMORY — this is ORPO's headline advantage. Compare against 04_dpo.py's output.
    # ----------------------------------------------------------------------------------
    print("\n  ── memory advantage over DPO ──")
    for label, method, extra in [
        ("ORPO  (this script)", "qlora", "1 model  — no reference, no reward, no value"),
        ("DPO   (04_dpo.py)", "qlora", "2 models — policy + frozen reference"),
        ("PPO   (RLHF)", "qlora", "4 models — policy + ref + reward + value"),
    ]:
        plan = TrainPlan(
            model=sniff_size(a.model, "7B"),
            method=method, seq_len=a.max_len, batch=a.batch, grad_accum=a.grad_accum, lora_r=a.lora_r,
        )
        print(f"    {label:<22} {plan.total_gb():>6.1f} GB   {extra}")
    print()

    if a.dry_run:
        print("  --dry-run complete.\n")
        return

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import ORPOConfig, ORPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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

    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    cfg = ORPOConfig(
        output_dir=a.output,
        beta=a.lambda_orpo,          # TRL exposes ORPO's λ as `beta`
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
    )

    # Note the absence of ref_model — that is the point of ORPO.
    trainer = ORPOTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(a.output)
    tok.save_pretrained(a.output)

    print(f"\n  ✅ ORPO adapter saved to {a.output}")
    print("\n  ── interpreting the two loss terms ──")
    print("    If the run looks like it barely learned preferences, RAISE --lambda-orpo")
    print("    (toward 0.5-1.0). If the model became repetitive or degenerate, LOWER it.")
    print("    If output quality dropped overall, your 'chosen' data is not good enough")
    print("    to imitate — ORPO's SFT term is faithfully teaching it.")
    print("\n  Both terms are logged separately as 'rewards/chosen', 'rewards/rejected'")
    print("  and 'nll_loss'. Watch the chosen-rejected gap; it should widen.")


if __name__ == "__main__":
    main()
