#!/usr/bin/env python
"""
01_sft_lora.py — Supervised fine-tuning with LoRA.

The workhorse script. If you only ever run one file from this repo, run this one.

What it does
------------
Loads a base model in 4-bit (or bf16), attaches LoRA adapters, trains on an
instruction dataset with the prompt tokens masked out of the loss, evaluates on a
held-out split, and saves the adapter.

Why LoRA by default
-------------------
Full fine-tuning of a 7B model needs ~112 GB of optimizer+gradient+weight state and
will OOM on anything under an 80GB card with realistic sequence lengths. LoRA trains
~1-2% of the parameters, fits the same model in 12-24 GB, and matches full FT quality
on most instruction-following tasks. See CS-23 for where it *doesn't* match.

Run it
------
    python 01_sft_lora.py --dry-run                       # config + token stats, no GPU
    python 01_sft_lora.py --data data/sample_sft.jsonl
    python 01_sft_lora.py --data mine.jsonl --epochs 2 --lr 1e-4 --output ./out/v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data_utils as du                      # noqa: E402
from common.memory import TrainPlan, _print_plan         # noqa: E402

# --------------------------------------------------------------------------------------
# Defaults. These are the *safe* defaults, chosen so a first run does not fail.
# Deviate deliberately, not accidentally.
# --------------------------------------------------------------------------------------
DEFAULTS = dict(
    model="Qwen/Qwen2.5-1.5B-Instruct",   # small enough to iterate fast; swap freely
    data="data/sample_sft.jsonl",
    eval_data=None,                        # if None, split `data` by val_split
    val_split=0.05,
    output="./out/sft-lora",
    format="alpaca",
    max_seq_len=2048,
    epochs=2,                              # >3 epochs on a small set = overfitting
    lr=2e-4,                               # LoRA wants a much higher LR than full FT
    batch=2,
    grad_accum=8,                          # effective batch = 2*8 = 16 examples/step
    warmup_ratio=0.03,
    scheduler="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    lora_r=16,                             # rank; 8-32 covers most instruction tasks
    lora_alpha=32,                         # convention: alpha = 2*r
    lora_dropout=0.05,
    lora_targets="all",                    # all | attention  (see --help for why)
    quant="4bit",                          # 4bit | 8bit | none
    grad_checkpointing=True,
    packing=False,                         # turn on if your examples are short
    seed=42,
    push_to_hub=None,
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
    p.add_argument("--dry-run", action="store_true",
                   help="Validate data + print VRAM estimate, do not load the model.")
    p.add_argument("--train-on-prompt", action="store_true",
                   help="Do NOT mask the prompt. Almost always wrong; see data_utils.")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    # ----------------------------------------------------------------------------------
    # 1. DATA — always inspect before training. Most "training failed" is "data was bad".
    # ----------------------------------------------------------------------------------
    if not Path(a.data).exists():
        sys.exit(f"Dataset not found: {a.data}\n"
                 f"Generate a sample first:  python data/make_instruction_data.py --out {a.data}")

    raw = du.load_jsonl(a.data, fmt=a.format)
    if not raw:
        sys.exit("Dataset parsed to zero rows — check --format against your file schema.")

    n_val = max(1, int(len(raw) * a.val_split)) if not a.eval_data else 0
    train_rows = raw[:-n_val] if n_val else raw
    val_rows = du.load_jsonl(a.eval_data, fmt=a.format) if a.eval_data else (raw[-n_val:] if n_val else [])

    print(f"  dataset            {a.data}")
    print(f"  format             {a.format}")
    print(f"  train / val rows   {len(train_rows)} / {len(val_rows)}")

    # Token stats are computed *after* the tokenizer loads, so in dry-run mode we
    # report character-level proxies instead. Cheap and catches most problems.
    char_lens = sorted(sum(len(m["content"]) for m in row) for row in train_rows)
    print(f"  chars/example      p50={char_lens[len(char_lens)//2]}  "
          f"p95={char_lens[int(len(char_lens)*0.95)]}  max={char_lens[-1]}")
    print(f"  approx tokens      {sum(char_lens)//4:,}  (÷4 chars/token heuristic)")

    roles = {m["role"] for row in train_rows for m in row}
    if "assistant" not in roles:
        sys.exit("No 'assistant' turns found. This looks like a pretraining corpus, not an SFT set.\n"
                 "       For raw-text training use 03_continued_pretraining.py instead.")

    # ----------------------------------------------------------------------------------
    # 2. MEMORY — budget before you rent.
    # ----------------------------------------------------------------------------------
    method = "full" if a.quant == "none" and a.lora_targets == "none" else (
        "qlora" if a.quant == "4bit" else "lora")
    _print_plan(TrainPlan(
        model=next((m for m in ["1.5B", "1B", "3B", "7B", "8B", "13B", "70B"]
                    if m.lower() in a.model.lower()), "7B"),
        method=method, seq_len=a.max_seq_len, batch=a.batch, grad_accum=a.grad_accum,
        lora_r=a.lora_r, grad_checkpointing=a.grad_checkpointing,
    ))

    if a.dry_run:
        print("  --dry-run: config validated. Remove the flag to actually train.\n")
        return

    # ----------------------------------------------------------------------------------
    # 3. IMPORTS — deferred so --dry-run works on a machine with no torch installed.
    # ----------------------------------------------------------------------------------
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    # ----------------------------------------------------------------------------------
    # 4. TOKENIZER — set pad_token first. Many base models have no pad token, and the
    #    default behaviour (padding with eos) corrupts the loss on padded positions.
    # ----------------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # right-padding for training; LEFT for generation

    if not getattr(tok, "chat_template", None):
        print("  ⚠  tokenizer has no chat_template — falling back to ChatML in data_utils.")

    # ----------------------------------------------------------------------------------
    # 5. BUILD EXAMPLES with loss masking.
    # ----------------------------------------------------------------------------------
    def build(rows):
        out = [du.build_masked_example(tok, m, max_len=a.max_seq_len,
                                       train_on_prompt=a.train_on_prompt) for m in rows]
        before = len(out)
        out = [e for e in out if du.has_supervision(e)]
        if len(out) < before:
            print(f"  ⚠  dropped {before - len(out)} examples with no supervised tokens "
                  f"(usually truncation cut off the answer — raise --max-seq-len)")
        return out

    train_ex = build(train_rows)
    val_ex = build(val_rows) if val_rows else []
    if not train_ex:
        sys.exit("Every example lost its supervision after truncation. Raise --max-seq-len.")

    stats = du.token_stats(train_ex)
    print(f"  token stats        {stats}")
    if stats["supervised_token_frac"] < 0.10 and not a.train_on_prompt:
        print(f"  ⚠  only {stats['supervised_token_frac']:.1%} of tokens carry loss. "
              f"Your prompts are long relative to answers; that is fine but slow.")

    if a.packing:
        train_ex = du.pack_examples(train_ex, a.max_seq_len, tok.pad_token_id)
        print(f"  packed to          {len(train_ex)} full-length sequences")

    train_ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex) if val_ex else None

    # ----------------------------------------------------------------------------------
    # 6. MODEL + QUANTIZATION
    # ----------------------------------------------------------------------------------
    bnb = None
    if a.quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # NF4 > fp4 for normally-distributed weights
            bnb_4bit_compute_dtype=torch.bfloat16, # compute in bf16, store in 4-bit
            bnb_4bit_use_double_quant=True,        # quantize the quant constants: ~0.37 bits/param saved
        )
    elif a.quant == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if bnb is None else None,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "sdpa",
    )
    model.config.use_cache = False          # incompatible with gradient checkpointing

    if bnb is not None:
        # Casts layernorms to fp32 and enables input grads — required for stable
        # training through a quantized base. Skipping this is a common silent failure.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_checkpointing)

    # ----------------------------------------------------------------------------------
    # 7. LoRA CONFIG
    # ----------------------------------------------------------------------------------
    targets = _resolve_targets(model, a.lora_targets)
    lora_cfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none",                    # training biases adds params for ~no gain
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA targets       {targets}")
    print(f"  trainable params   {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")

    # ----------------------------------------------------------------------------------
    # 8. TRAINING
    # ----------------------------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=a.output,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type=a.scheduler,
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        max_grad_norm=a.max_grad_norm,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=a.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=100,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=os.environ.get("REPORT_TO", "none"),   # set REPORT_TO=wandb to log
        run_name=Path(a.output).name,
        seed=a.seed,
        optim="paged_adamw_8bit" if bnb is not None else "adamw_torch",
        group_by_length=True,       # batches similar lengths together: less padding
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=du.IGNORE_INDEX),
    )

    trainer.train(resume_from_checkpoint=_latest_ckpt(a.output))
    trainer.save_model(a.output)
    tok.save_pretrained(a.output)
    print(f"\n  ✅ adapter saved to {a.output}")

    _final_report(trainer, a, train_ex)

    # ----------------------------------------------------------------------------------
    # 9. OPTIONAL PUBLISH
    # ----------------------------------------------------------------------------------
    if a.push_to_hub:
        model.push_to_hub(a.push_to_hub, private=True)
        tok.push_to_hub(a.push_to_hub, private=True)
        print(f"  ✅ pushed to https://huggingface.co/{a.push_to_hub}")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_targets(model, which: str) -> list[str]:
    """
    `all` targets every linear projection (q,k,v,o,gate,up,down). `attention` targets
    only q,v — the original LoRA paper's choice. `all` is ~4x more trainable params
    and measurably better on instruction/domain tasks; `attention` is cheaper and
    sometimes better for small classification-style adaptations.
    """
    if which == "attention":
        return ["q_proj", "v_proj"]
    linear_names = {n.split(".")[-1] for n, m in model.named_modules()
                    if m.__class__.__name__ == "Linear"}
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = [n for n in preferred if n in linear_names]
    if found:
        return found
    # Fall back to whatever linear layers actually exist (other architectures).
    return sorted(linear_names)[:8]


def _latest_ckpt(outdir: str):
    p = Path(outdir)
    if not p.exists():
        return None
    cks = sorted([d for d in p.glob("checkpoint-*") if d.is_dir()],
                 key=lambda d: int(d.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def _final_report(trainer, a, examples) -> None:
    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    evals = [h["eval_loss"] for h in hist if "eval_loss" in h]

    print("\n  ── training report ──────────────────────────────")
    print(f"  steps              {trainer.state.global_step}")
    print(f"  final train loss   {losses[-1]:.4f}" if losses else "  no loss recorded")
    if evals:
        print(f"  final eval loss    {evals[-1]:.4f}")
        if losses and evals[-1] > losses[-1] * 1.15:
            print("  ⚠  eval loss is well above train loss → overfitting. "
                  "Reduce --epochs or add data.")
    print("\n  Next steps:")
    print(f"    • chat with it:   python 15_serve_vllm.py --adapter {a.output}")
    print(f"    • merge + export: python 09_merge_and_export.py --adapter {a.output} --gguf")
    print("    • evaluate:       run eval_utils.reward_hack_report() against your baseline")


if __name__ == "__main__":
    main()
