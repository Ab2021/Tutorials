#!/usr/bin/env python
"""
12_multimodal_vlm.py — fine-tune a vision-language model (Qwen2-VL and friends).

What is actually being trained
------------------------------
A VLM is three parts bolted together:
    [vision encoder]  →  [projector / merger]  →  [language model]
       (ViT)                (MLP, or a            (the LLM you already
                            resampler)             know how to fine-tune)

Only the last two are normally trained. The vision encoder is almost always frozen, and
that is not laziness — it is correct:
  * The ViT was trained on far more images than you have. Fine-tuning it on 5k examples
    destroys its features (catastrophic forgetting again, this time in vision).
  * It is a large fraction of the parameters, so unfreezing it multiplies your VRAM.
  * Your task is almost never "learn to see". It is "learn to describe / extract / point
    at what is already being seen". That mapping lives in the projector and the LLM.

So the default, and the right default, is: freeze the vision tower, train the projector
(if it is trainable) and LoRA the LLM.

The resolution trap
-------------------
VLMs resize images to a fixed token budget. A document scan or a dense screenshot shrunk
to 336px loses the text you are asking the model to read. Qwen2-VL uses dynamic resolution
— it emits a variable number of vision tokens based on the image's aspect ratio and size,
which means:

    more pixels  →  more vision tokens  →  more VRAM and a longer sequence

This is the #1 cause of "it worked on my demo images and OOMs on real documents". It is
also why OCR-style VLM tasks are the most expensive per example in this entire handbook.

Run it
------
    python 12_multimodal_vlm.py --data data/vlm.jsonl --dry-run
    python 12_multimodal_vlm.py --data data/vlm.jsonl --out out/qwen2vl-lora
    python 12_multimodal_vlm.py --data data/vlm.jsonl --max-pixels 1003520 --min-pixels 200704
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix

DEFAULTS = {
    "model": "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "max_seq_len": 2048,
    "r": 16,
    "lora_alpha": 16,
    "batch_size": 1,          # VLM batches are large; 1-2 is normal
    "grad_accum": 8,
    "epochs": 1,
    "lr": 2e-4,
    # Qwen2-VL pixel bounds. max_pixels caps vision tokens per image; min_pixels stops
    # very small images from becoming unreadable. 28*28 is one "patch".
    "max_pixels": 28 * 28 * 1280,     # ≈1.0M px
    "min_pixels": 28 * 28 * 256,      # ≈0.2M px
}

# LLM-side LoRA targets. Note what is ABSENT: any `visual.*` module. This is the
# deliberate choice described above.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="jsonl of {image, conversations}")
    p.add_argument("--out", default="out/vlm-lora")
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    p.add_argument("--r", type=int, default=DEFAULTS["r"])
    p.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    p.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--max-pixels", type=int, default=DEFAULTS["max_pixels"])
    p.add_argument("--min-pixels", type=int, default=DEFAULTS["min_pixels"])
    p.add_argument("--train-vision", action="store_true",
                   help="Unfreeze the vision encoder (usually a mistake — read the docstring)")
    p.add_argument("--image-col", default="image")
    p.add_argument("--conv-col", default="conversations")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = _load(a.data)

    print(f"\n  ── vision-language fine-tuning ──")
    print(f"  base model         {a.model}")
    print(f"  examples           {len(rows)}")
    print(f"  batch x accum      {a.batch_size} x {a.grad_accum} = "
          f"{a.batch_size * a.grad_accum} effective")
    print(f"  max_pixels         {a.max_pixels:,}  "
          f"(≈{a.max_pixels // (28*28):,} vision patches)")

    _check_data(rows, a)
    _estimate_vision_tokens(a)
    _explain_freezing(a)

    if a.dry_run:
        print("\n  --dry-run: no training performed.")
        return
    _train(a, rows)


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"  No such file: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------
def _check_data(rows: list[dict], a) -> None:
    problems: dict[str, int] = {}

    def flag(k: str, detail: str | None = None) -> None:
        problems[k] = problems.get(k, 0) + 1
        if detail and problems[k] <= 3:
            print(f"    ⚠  {k}: {detail}")

    missing_files, sizes = [], []
    for i, r in enumerate(rows):
        img = r.get(a.image_col)
        convs = r.get(a.conv_col) or r.get("messages")

        if not img:
            flag("no_image_field", f"row {i}")
            continue
        if isinstance(img, str) and not img.startswith(("http://", "https://")):
            p = Path(img)
            if not p.exists():
                flag("image_file_not_found", f"row {i}: {img}")
            else:
                sizes.append(p.stat().st_size)
                if p.stat().st_size == 0:
                    flag("zero_byte_image", f"row {i}")

        if not convs:
            flag("no_conversations", f"row {i}")
            continue

        if isinstance(convs, list) and convs and isinstance(convs[0], dict):
            roles = [c.get("role") or c.get("from") for c in convs]
            if not any(x in ("assistant", "gpt", "model") for x in roles):
                flag("no_assistant_turn", f"row {i}")
            if not any(x in ("user", "human") for x in roles):
                flag("no_user_turn", f"row {i}")

            # The placeholder must be in a HUMAN turn, and it must be the literal
            # placeholder token — not the English word "image", which appears in ordinary
            # sentences ("tell me about this image") and would make this check useless.
            human = " ".join(
                str(c.get("content") or c.get("value", ""))
                for c in convs
                if (c.get("role") or c.get("from")) in ("user", "human"))
            # Different families use different placeholders; accept the known set.
            placeholders = ("<image>", "<|vision_start|>", "<image>\n", "<img>")
            if not any(ph in human for ph in placeholders):
                flag("placeholder_not_in_user_turn", f"row {i}")

    if sizes:
        sizes.sort()
        print(f"\n  ── data ──")
        print(f"  image bytes p50    {sizes[len(sizes)//2]:,}")
        print(f"  image bytes max    {sizes[-1]:,}")
        big = sum(1 for s in sizes if s > 2_000_000)
        if big:
            print(f"  ℹ  {big} images over 2 MB. Large images are downscaled to "
                  f"--max-pixels anyway,")
            print("     so you are paying disk and load time for pixels the model never")
            print("     sees. Pre-resize your corpus — it is the cheapest speedup available.")

    if problems:
        print(f"\n  ⚠  {sum(problems.values())} issue(s): "
              f"{dict(sorted(problems.items(), key=lambda kv: -kv[1]))}")
    else:
        print("\n  ✅ structure looks valid")

    print("\n  ── the data format people get wrong ──")
    print("    Each row needs (a) the image path/URL and (b) a conversation containing the")
    print("    literal <image> placeholder in the HUMAN turn. Omit the placeholder and the")
    print("    image is silently never inserted — training runs, loss falls, and the model")
    print("    learns to answer from the text alone. That failure is invisible in the loss")
    print("    curve and only shows up when you ask about a picture.")


def _estimate_vision_tokens(a) -> None:
    """
    Vision tokens are the hidden cost. Make it visible before the OOM.
    """
    patches = a.max_pixels // (28 * 28)
    # Qwen2-VL merges 2x2 patch groups into one LLM token, so /4.
    merged = patches // 4
    print(f"\n  ── vision token budget ──")
    print(f"    patches (28x28)  {patches:,}")
    print(f"    LLM tokens       ~{merged:,}   (2x2 merge)")
    print(f"    + text           up to {a.max_seq_len - merged:,} remaining")
    if merged > a.max_seq_len * 0.5:
        print(f"  ⚠  Vision tokens consume over half of max_seq_len={a.max_seq_len}.")
        print("     Your text (the question AND the answer you want the model to learn)")
        print("     has the remainder. For OCR/document tasks, raise --max-seq-len or")
        print("     lower --max-pixels — but lowering it is exactly what makes small text")
        print("     unreadable. This is the central tradeoff of document VLMs.")
    print(f"    KV cache         vision tokens are cached like any other token; at")
    print(f"                     batch {a.batch_size} and {merged:,} vision tokens/image,")
    print("                     long-context VLM training is memory-bandwidth bound.")


def _explain_freezing(a) -> None:
    print(f"\n  ── what is trainable ──")
    print(f"    vision encoder   {'UNFROZEN ⚠' if a.train_vision else 'frozen ✓'}")
    print(f"    projector        trainable")
    print(f"    LLM              LoRA r={a.r}, alpha={a.lora_alpha}")
    print(f"    targets          {TARGET_MODULES}")
    if a.train_vision:
        print("\n  ⚠  --train-vision is on. Expect roughly 3-5x the VRAM and a serious")
        print("     risk of destroying the pretrained visual features unless your image")
        print("     dataset is very large (100k+). Unfreeze only if your domain is")
        print("     genuinely far from natural images — medical, satellite, microscopy —")
        print("     AND you have the data to support it. Start frozen; unfreeze only if")
        print("     a frozen run plateaus well below your target.")


def _train(a, rows: list[dict]) -> None:
    try:
        from unsloth import FastVisionModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        sys.exit(f"  pip install unsloth trl  ({e})")

    print(f"\n  loading {a.model}...")
    model, processor = FastVisionModel.from_pretrained(
        a.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=a.max_seq_len,
    )
    # Set the pixel bounds on the processor, not the model — this is where the
    # resolution/token tradeoff is actually configured.
    try:
        processor.image_processor.max_pixels = a.max_pixels
        processor.image_processor.min_pixels = a.min_pixels
    except AttributeError:
        print("  ⚠  could not set pixel bounds on this processor. Check the model's")
        print("     image processor attribute names — they differ across VLM families.")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=a.train_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=a.r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        random_state=3407,
    )

    from PIL import Image

    def to_conv(r: dict) -> dict:
        convs = r.get(a.conv_col) or r.get("messages") or []
        out = []
        for c in convs:
            role = c.get("role") or c.get("from") or "user"
            role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(role, role)
            content = c.get("content") or c.get("value") or ""
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
        img = r[a.image_col]
        return {"messages": out, "image": _open_image(img)}

    def _open_image(src):
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            import requests
            from io import BytesIO
            return Image.open(BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
        return Image.open(src).convert("RGB")

    ds = Dataset.from_list([to_conv(r) for r in rows])
    print(f"  dataset            {len(ds)} rows")

    def collate(batch):
        texts, images = [], []
        for b in batch:
            texts.append(processor.apply_chat_template(
                b["messages"], add_generation_prompt=False, tokenize=False))
            images.append([b["image"]])
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=a.max_seq_len)
        labels = enc["input_ids"].clone()
        # Mask everything that is not an assistant turn. For VLMs this is fiddly because
        # vision tokens are interleaved; the simple version below masks pad and the
        # image-token ids. Verify on one example before trusting it.
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tok in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                labels[labels == tid] = -100
        enc["labels"] = labels
        return enc

    trainer = SFTTrainer(
        model=model, tokenizer=processor.tokenizer, train_dataset=ds,
        data_collator=collate,
        args=SFTConfig(
            output_dir=a.out,
            per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum,
            num_train_epochs=a.epochs,
            learning_rate=a.lr,
            warmup_ratio=0.03,
            logging_steps=5,
            save_steps=200,
            optim="adamw_8bit",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,   # MUST be False: our collator needs 'image'
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    print("  training...")
    trainer.train()
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    print(f"  ✅ saved to {a.out}")

    print("\n  ── before you ship this ──")
    print("     1. Test on the SMALLEST text in your hardest images. Aggressive --max-pixels")
    print("        silently makes fine print unreadable, and the model will then")
    print("        CONFIDENTLY HALLUCINATE plausible values. Check the failure mode, not")
    print("        just the accuracy.")
    print("     2. Verify the mask: decode the tokens that are NOT -100 for one example and")
    print("        confirm you are training on the assistant answer, not the question.")
    print("     3. Test aspect ratios your training set did not cover. VLMs generalise")
    print("        worse across aspect ratio than across content.")
    print("     4. Serving a VLM costs several times an equivalent text model per request")
    print("        — the image tokens dominate. Benchmark before committing.")
    print("     5. If the task is pure OCR, a dedicated OCR pipeline is often cheaper AND")
    print("        more accurate than a VLM. Reach for a VLM when you need REASONING over")
    print("        the image, not transcription.")


if __name__ == "__main__":
    main()
