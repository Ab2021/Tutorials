#!/usr/bin/env python
"""
02_sft_unsloth.py — fine-tune an LLM with Unsloth (2-4x faster, far less VRAM).

What Unsloth actually does
--------------------------
It is not a new training algorithm. It is a set of hand-written Triton kernels plus a
custom autograd graph that removes the work a generic trainer does and does not need:

  1. **Manual backprop of the LoRA graph.** For a LoRA layer, `y = Wx + (alpha/r)*BAx`,
     the gradients w.r.t. A and B are a closed-form two-matmul expression. Unsloth writes
     them by hand instead of building the full autograd graph — less memory, fewer kernels.
  2. **No attention matrix materialisation.** Same idea as FlashAttention: never write the
     n x n scores to HBM. This is the single biggest win at long sequence lengths, and it
     is why the speedup grows with `max_seq_length`.
  3. **Fused RoPE and fused cross-entropy.** Rotary embedding and the LM head loss each
     become one kernel instead of several.
  4. **No 4-bit dequantize -> compute -> requantize round trip** on the frozen base weights.
     In a naive QLoRA loop the base is dequantized for every forward pass; Unsloth keeps
     the computation in a form that avoids that traffic.

The honest framing of the marketing numbers
-------------------------------------------
"2x faster, 60% less VRAM" is measured against a *naive* HuggingFace baseline. Against a
well-configured `transformers + peft + flash-attn-2` run the gap is much smaller — **1.2-1.4x**
on the architectures and sequence lengths Unsloth supports well, dropping to 1.0-1.2x (inside
run-to-run noise) on long sequences and on architectures where its kernels do not apply. The
1.6x figure this line used to carry is the favourable tail of the range, not the centre, and it
disagreed with CS-16 §4.7.2 and CH-16 §1 row 2 — which say 1.2-1.4x. Those are right. Unsloth's
real, durable advantages are:
  * it works on a single consumer GPU with a small max_seq_length where nothing else fits,
  * the setup is 10 lines instead of 100, so you make fewer mistakes,
  * it is genuinely excellent for single-GPU LoRA/QLoRA on the architectures it supports.
Its real constraints are: single-GPU focus (multi-GPU/FSDP is not its strength), and
supported-architecture coverage rather than "any model on the Hub".

Run it
------
    python 02_sft_unsloth.py --dry-run --data data/sample_sft.jsonl
    python 02_sft_unsloth.py --data data/sample_sft.jsonl --out out/unsloth-lora
    python 02_sft_unsloth.py --data data/sft.jsonl --model Qwen/Qwen2.5-7B-Instruct \
        --max-seq-len 4096 --r 32 --out out/unsloth-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data_utils as du                       # noqa: E402
from common.memory import TrainPlan, _print_plan, sniff_size   # noqa: E402

DEFAULTS = {
    "model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_len": 2048,
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 1,
    "lr": 2e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "seed": 3407,
    "save_steps": 100,
}

# Unsloth's own pre-quantized 4-bit uploads. Using these instead of quantizing at load
# time is faster to start and avoids a class of "wrong quantization config" mistakes.
MODEL_MAP = {
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen2.5-3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "qwen2.5-1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "gemma-2-2b": "unsloth/gemma-2-2b-it-bnb-4bit",
    "phi-3.5-mini": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
}


# --------------------------------------------------------------------------------------
# The one thing that most often ruins an Unsloth run
# --------------------------------------------------------------------------------------
CHAT_TEMPLATE_TRAP = """
  ── THE #1 SILENT FAILURE IN UNSLOTH RUNS ──

  Unsloth derives the instruction/response split from the tokenizer's chat template.
  Three ways this goes wrong, all of which train happily and produce a bad model:

  1. NO CHAT TEMPLATE AT ALL. `tokenizer.chat_template is None` means every token is
     trained on, including the prompt. The model learns to generate USER turns. Your loss
     curve looks perfect. The model is ruined.

  2. TEMPLATE MISMATCH. You train with Llama-3's template and serve with ChatML (or the
     reverse — vLLM's default, or Ollama's Modelfile). The special tokens are different
     ids and the model sees a format it never trained on.

  3. `train_on_responses_only` WITH THE WRONG MARKER STRINGS. It searches for the literal
     substrings of the assistant header to build the mask. Get them wrong and it either
     masks everything (loss = 0, model learns nothing) or nothing (trains on the prompt).

  The check is cheap and it is the difference between a working run and a wasted one:
  print the template, apply it to one real example, and confirm the mask boundaries by
  decoding the tokens that are NOT masked with IGNORE_INDEX.
"""


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="SFT jsonl (alpaca/sharegpt/openai/completion)")
    p.add_argument("--out", default="out/unsloth-lora")
    p.add_argument("--model", default=None, help="HF id, an Unsloth 4-bit id, or a key from MODEL_MAP")
    p.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    p.add_argument("--r", type=int, default=DEFAULTS["r"])
    p.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=DEFAULTS["lora_dropout"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    p.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--no-4bit", action="store_true",
                   help="Load in bf16 instead of 4-bit (needs the VRAM; better quality)")
    p.add_argument("--full-finetune", action="store_true",
                   help="Train all weights, not a LoRA adapter (needs ~4x the VRAM)")
    p.add_argument("--packing", action="store_true",
                   help="Pack short examples into full-length sequences (big speedup)")
    p.add_argument("--assistant-only-loss", action="store_true", default=True)
    p.add_argument("--no-assistant-only-loss", dest="assistant_only_loss",
                   action="store_false")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate data, print the plan, print the chat template, train nothing")
    p.add_argument("--merge", action="store_true",
                   help="Save a merged 16-bit model at the end (needed for llama.cpp)")
    return p.parse_args()


def resolve_model(name: str | None) -> str:
    if name is None:
        return DEFAULTS["model"]
    key = name.lower()
    if key in MODEL_MAP:
        print(f"  model alias        {name} → {MODEL_MAP[key]}")
        return MODEL_MAP[key]
    return name


def main() -> None:
    a = parse_args()
    model_id = resolve_model(a.model)

    print("\n  ── plan ──")
    size = sniff_size(model_id, "7B")
    # TrainPlan takes `model` and `batch`, not `size`/`batch_size`, and `method` is a
    # required field — this call used to raise TypeError before printing anything,
    # which broke the `--dry-run` command CH-13 §6 tells readers to run first.
    # Unsloth loads the base in 4-bit by default, so the memory model needs to be told
    # which of the three regimes this run is actually in.
    method = ("full" if a.full_finetune else ("lora" if a.no_4bit else "qlora"))
    plan = TrainPlan(
        model=size,
        method=method,
        seq_len=a.max_seq_len,
        batch=a.batch_size,
        grad_accum=a.grad_accum,
        lora_r=(0 if a.full_finetune else a.r),
    )
    _print_plan(plan)
    print(f"  method             {method}"
          f"{'  (4-bit base)' if method == 'qlora' else ''}")
    print(f"  epochs             {a.epochs}")
    print(f"  lr                 {a.lr:g}   (LoRA wants ~1e-4..3e-4; full FT wants ~1e-5..2e-5)")
    print(f"  packing            {a.packing}")
    print(f"  assistant-only     {a.assistant_only_loss}")

    if a.full_finetune and a.lr > 1e-4:
        print("\n  ⚠  Full fine-tuning at lr={:g} will very likely diverge. Full FT needs a".format(a.lr))
        print("     learning rate roughly 10-20x SMALLER than LoRA. Use --lr 2e-5.")

    if a.lora_dropout and not a.full_finetune:
        # Warn HERE, in the plan block, not only at model-construction time. The whole
        # point of --dry-run is to surface problems before you rent the GPU, and a warning
        # that only fires on the training path is invisible to the one command every reader
        # runs first. CH-16 §1 row 4 / §4.2, CS-16 §6.4.
        print(f"\n  ⚠  --lora-dropout {a.lora_dropout} is non-zero. Unsloth's fused kernels")
        print("     only apply at dropout 0, so peft will silently use the generic autograd")
        print("     path and you will lose roughly 15-35% of the speedup you installed")
        print("     Unsloth for — no error, no warning from the library. Pass")
        print("     --lora-dropout 0 unless you have measured that dropout helps here.")
    if a.max_seq_len > 4096 and not a.no_4bit:
        print(f"\n  ⚠  max_seq_len={a.max_seq_len} with 4-bit: activation memory grows with")
        print("     sequence length. If you OOM, halve max_seq_len before touching batch size —")
        print("     it is the cheaper lever, and Unsloth's attention never materialises the")
        print("     n x n matrix anyway, so you are paying for activations, not attention.")

    # ----------------------------------------------------------------------------------
    # Data — validate before touching the GPU.
    #
    # We load the real tokenizer here even on --dry-run. It is a small download and it is
    # the only way to get TRUE masked-token counts. An estimate from word counts is off
    # by 20-40% across tokenizers, which is exactly the margin that decides whether your
    # responses survive truncation.
    # ----------------------------------------------------------------------------------
    convs = du.load_jsonl(a.data)
    print(f"\n  ── data ──")
    print(f"  rows               {len(convs)}")

    tokenizer = _load_tokenizer(model_id)
    if tokenizer is None:
        print("\n  ⚠  could not load the tokenizer — skipping the data analysis.")
        print("     Fix that before training: an unanalysed dataset is how you end up")
        print("     training on prompts, or on nothing at all.")
    else:
        _print_template(tokenizer)
        examples = [du.build_masked_example(tokenizer, c, max_len=a.max_seq_len)
                    for c in convs]
        stats = du.token_stats(examples)
        _print_data_report(stats, a)

    if a.dry_run:
        print("\n  --dry-run: no training performed.")
        print(f"\n     python 02_sft_unsloth.py --data {a.data} --out {a.out}")
        return

    _train(a, model_id)


def _load_tokenizer(model_id: str):
    # The tokenizer is genuinely required even for --dry-run: the chat-template check is
    # this script's whole reason to exist, and it cannot be done without the real
    # tokenizer. So report a missing install as an instruction, not a traceback.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"\n  ⚠  transformers is not installed, so the chat-template check "
              f"(this script's main job) cannot run.\n     pip install transformers  ({e})")
        return None
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠  tokenizer load failed: {e}")
        return None


def _print_data_report(stats: dict, a) -> None:
    """
    Interpret the numbers the way they need to be interpreted.

    The two that matter most are `without_supervision` and `supervised_token_frac`, and
    both are invisible without actually building the mask.
    """
    print(f"  total tokens       {stats['tokens_total']:,}")
    print(f"  length p50/p95/max {stats['len_p50']} / {stats['len_p95']} / "
          f"{stats['len_max']}")
    print(f"  supervised frac    {stats['supervised_token_frac']:.1%}"
          "   ← fraction of tokens that actually contribute to the loss")

    if stats["without_supervision"]:
        print(f"\n  ⚠  {stats['without_supervision']}/{stats['examples']} rows have NO "
              "supervised tokens at all.")
        print("     These are examples the model learns NOTHING from — the prompt is so")
        print("     long that the assistant turn was truncated away entirely. They are")
        print("     pure wasted compute and they will be counted in your epoch loss as")
        print("     zeros. Filter them out with du.has_supervision().")

    if stats["supervised_token_frac"] < 0.10:
        print(f"\n  ⚠  Only {stats['supervised_token_frac']:.1%} of tokens are supervised.")
        print("     Long prompts with short answers: most of the FLOPs go into tokens you")
        print("     explicitly mask out. This is normal for RAG-style data but it means you")
        print("     need proportionally more steps to see the same learning signal.")

    if stats["len_p95"] < a.max_seq_len * 0.5 and not a.packing:
        print(f"\n  ℹ  p95 length ({stats['len_p95']}) is well under max_seq_len "
              f"({a.max_seq_len}).")
        print("     Most of every batch is padding. --packing would cut wall-clock a lot.")

    if stats["len_max"] >= a.max_seq_len:
        print(f"\n  ℹ  Some rows hit the {a.max_seq_len}-token ceiling. In SFT, truncation")
        print("     cuts the END of the response — the exact part you want the model to")
        print("     learn to produce. Raise --max-seq-len, or filter the long rows.")


def _print_template(tok) -> None:
    """
    Print the template and prove the mask boundaries BEFORE spending GPU hours.

    This is the highest-value block in the file. Every silent Unsloth failure listed in
    this script's docstring is visible here first, for the cost of one small download.
    """
    tpl = getattr(tok, "chat_template", None)
    print(f"\n  ── chat template ──")
    if not tpl:
        print("  ⚠  tokenizer.chat_template is None. Training will include the PROMPT")
        print("     tokens in the loss, teaching the model to generate user turns.")
        print("     Set the template before training, e.g.:")
        print('       tok.chat_template = "{% for m in messages %}<|im_start|>{{ m.role }}'
              '\\n{{ m.content }}<|im_end|>\\n{% endfor %}"')
        return

    head = " ".join(tpl.split())[:150]
    print(f"  template           {head}{'...' if len(tpl) > 150 else ''}")

    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}]
    try:
        text = du.render(tok, msgs)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠  template failed to render a basic 2-message conversation: {e}")
        return
    print(f"  rendered           {text!r}")
    print("  ↑ Those assistant header/terminator substrings are exactly what")
    print("    train_on_responses_only searches for. If they look wrong, the mask will be")
    print("    wrong — and the loss curve will still look perfectly healthy.")

    # Show the mask directly rather than making the reader infer it.
    ex = du.build_masked_example(tok, msgs, max_len=512)
    n_sup = sum(1 for l in ex["labels"] if l != du.IGNORE_INDEX)
    print(f"\n  mask check         {n_sup}/{len(ex['labels'])} tokens supervised "
          f"({n_sup/max(len(ex['labels']),1):.0%})")
    if n_sup == 0:
        print("  ⚠  NOTHING IS SUPERVISED. The mask is inverted or the template has no")
        print("     assistant header. Training this would reduce the loss on nothing.")
    elif n_sup / max(len(ex["labels"]), 1) > 0.9:
        print("  ⚠  Almost everything is supervised — the prompt is probably included in")
        print("     the loss. Confirm against the rendered text above before launching.")


def _train(a, model_id: str) -> None:
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only, get_chat_template
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e}\n\n"
            "  pip install unsloth\n\n"
            "  Install unsloth BEFORE upgrading torch/transformers/trl — it pins compatible\n"
            "  versions, and upgrading them afterwards is the most common way to break it."
        )

    print(f"\n  loading {model_id} (4-bit={not a.no_4bit})...")
    kwargs = dict(
        model_name=model_id,
        max_seq_length=a.max_seq_len,
        dtype=None,                       # auto: bf16 on Ampere+, else fp16
        load_in_4bit=not a.no_4bit,
    )
    if a.full_finetune:
        # Unsloth supports full fine-tuning, but it needs the whole model in memory plus
        # optimizer states. Realistically: a 7B full FT does not fit on 24GB.
        kwargs["full_finetuning"] = True
        kwargs["load_in_4bit"] = False

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if not a.full_finetune:
        # `--lora-dropout` is warned about in the plan block in main(), which --dry-run
        # also reaches. Repeating the warning here would print it twice on a real run.
        model = FastLanguageModel.get_peft_model(
            model,
            r=a.r,
            lora_alpha=a.lora_alpha,
            lora_dropout=a.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            # "unsloth" checkpointing is the memory win: it offloads the saved activations
            # to CPU RAM in a compressed form instead of recomputing them, so it saves
            # memory WITHOUT the usual ~30% throughput penalty of standard checkpointing.
            use_gradient_checkpointing="unsloth",
            random_state=a.seed,
        )

    # ---- dataset --------------------------------------------------------------------
    convs = du.load_jsonl(a.data)
    eos = tokenizer.eos_token or ""

    def to_text(messages: list[dict]) -> dict:
        return {"text": du.render(tokenizer, messages) + eos}

    ds = Dataset.from_list([to_text(c) for c in convs])
    print(f"  dataset            {len(ds)} rows")

    sft_kwargs = dict(
        output_dir=a.out,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        warmup_ratio=DEFAULTS["warmup_ratio"],
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        fp16=not _is_bf16(),
        bf16=_is_bf16(),
        logging_steps=1,
        save_steps=DEFAULTS["save_steps"],
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=DEFAULTS["weight_decay"],
        lr_scheduler_type="linear",
        seed=a.seed,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=a.max_seq_len,
        packing=a.packing,
    )
    try:
        cfg = SFTConfig(**sft_kwargs)
    except TypeError:
        sft_kwargs.pop("max_seq_length", None)      # newer TRL moved this onto SFTConfig
        cfg = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)

    # ---- the critical step: mask the prompt out of the loss -------------------------
    if a.assistant_only_loss:
        try:
            # These marker strings differ per model family. get_chat_template normalises
            # the tokenizer to a known template first, which is why it is called here.
            if "llama" in model_id.lower():
                tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
                trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                                     train_dataset=ds, args=cfg)
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            print("\n  ✓ assistant-only loss enabled (prompt tokens masked to IGNORE_INDEX)")
        except Exception as e:                                # noqa: BLE001
            print(f"\n  ⚠  train_on_responses_only FAILED: {e}")
            print("     Do NOT ignore this. Without it the model trains on the prompt and")
            print("     learns to generate user turns. Check the template printed above and")
            print("     pass the correct instruction_part/response_part for your model.")
            sys.exit(1)

    print("\n  training...")
    t = trainer.train()
    print(f"  ✅ trained in {t.metrics.get('train_runtime', 0)/60:.1f} min")

    model.save_pretrained(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"  ✅ adapter saved to {a.out}")

    if a.merge:
        # Use Unsloth's own merge. Doing it with peft.merge_and_unload() on a model that
        # was loaded 4-bit produces a subtly wrong 16-bit model — Unsloth's path handles
        # the dequantization correctly, and this is exactly the trap 09_merge_and_export.py
        # warns about.
        merged = Path(a.out).parent / (Path(a.out).name + "-merged")
        print(f"\n  merging to 16-bit → {merged}")
        try:
            model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
            print(f"  ✅ merged model at {merged}")
            print("     Use this directory (not the adapter) for llama.cpp / GGUF export.")
        except Exception as e:                                # noqa: BLE001
            print(f"  ⚠  merge failed: {e}")
            print("     Fall back to 09_merge_and_export.py, but reload the base in bf16 —")
            print("     merging a 4-bit-loaded model with peft.merge_and_unload() silently")
            print("     bakes the quantization error into the merged weights.")

    print("\n  ── before you trust this model ──")
    print("     1. Generate on 10 held-out prompts and READ them.")
    print("     2. Confirm it did not learn to emit user turns (the template trap).")
    print("     3. If you merged, verify the merged model matches the adapter:")
    print(f"          python 09_merge_and_export.py --base {model_id} --adapter {a.out} \\")
    print("              --out <dir> --verify")
    print(f"     4. Serve and benchmark: python 15_serve_vllm.py --model <dir> --benchmark")


def _is_bf16() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:                                         # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
