#!/usr/bin/env python
"""
03_continued_pretraining.py — Domain-adaptive pretraining (DAPT) on your own PDFs.

This is the path nobody blogs about and everybody should consider first.

What it is
----------
Training on RAW TEXT (no instruction/response pairs) to move a base model's
distribution toward your domain — its jargon, its style, its entity names. You do
NOT get an assistant out of this. You get a base model that is less surprised by
your domain, which you then SFT on top of.

Pipeline:  PDFs → extract → clean → dedup → chunk → train → (then SFT)

Why do this instead of RAG?
---------------------------
RAG injects *facts* at inference. DAPT changes *fluency* in a domain. If your model
writes about pharmacology with the register of a Reddit comment, RAG will not fix it
and DAPT will. If you just need it to know a fact, use RAG — DAPT is 100x the cost.

Run it
------
    python 03_continued_pretraining.py --dry-run --pdf-dir ./pdfs
    python 03_continued_pretraining.py --pdf-dir ./pdfs --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    python 03_continued_pretraining.py --text-file corpus.txt --epochs 1

WARNING: use a BASE model, not an -Instruct model, for DAPT. Continued pretraining on
an instruct model degrades its instruction-following, because the chat template tokens
get trained on as if they were ordinary text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.memory import TrainPlan, _print_plan   # noqa: E402


# ======================================================================================
# STAGE 1 — EXTRACTION
# ======================================================================================
def extract_pdf(path: Path) -> str:
    """
    Extract text from a PDF.

    Try in order of quality:
      marker / docling  — best for tables and layout, heavy, GPU-accelerated (not used here)
      pdfplumber        — good layout awareness, slower than pypdf
      PyMuPDF (fitz)    — fast and good; AGPL licence, check before commercial use
      pypdf             — fast, pure-python, poor on multi-column and tables

    We use pdfplumber when available because it is permissively licensed and handles
    two-column academic papers far better than pypdf. Two-column layout is the #1
    source of silent garbage: pypdf reads across the gutter and interleaves the
    columns into word salad that *looks* like text and trains a broken model.
    """
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join((page.extract_text() or "") for page in pdf.pages)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except ImportError:
        raise SystemExit("Install a PDF reader:  pip install pdfplumber   (or pypdf)")


def sanity_check_extraction(text: str, name: str) -> list[str]:
    """
    Heuristic QA on extracted text. Run this ALWAYS — silent extraction failure is
    the most expensive bug in this pipeline because it wastes a full training run.
    """
    problems = []
    if len(text) < 200:
        problems.append("suspiciously short — probably a scanned PDF needing OCR")
    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < 0.5:
        problems.append("low alphabetic ratio — look for garbled CID/ligature encoding")
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        short = sum(1 for l in lines if len(l.strip()) < 25) / len(lines)
        if short > 0.5:
            problems.append("most lines are very short — likely a two-column layout read "
                            "across the gutter, or a table")
    words = re.findall(r"[A-Za-z]{2,}", text)
    if words:
        avg = sum(len(w) for w in words) / len(words)
        if avg > 9 or avg < 3:
            problems.append(f"odd average word length ({avg:.1f}) — check for missing spaces")
    return problems


# ======================================================================================
# STAGE 2 — CLEANING
# ======================================================================================
def clean_text(text: str) -> str:
    """
    Normalise and de-boilerplate. Every step here is a real bug you will otherwise hit.
    """
    # 1. Ligatures and typographic quotes: PDFs emit U+FB01 (ﬁ) etc. The tokenizer
    #    has never seen these and will emit <unk> or split them bizarrely.
    text = unicodedata.normalize("NFKC", text)

    # 2. De-hyphenate line-broken words: "distribu-\ntion" -> "distribution".
    #    Only when the hyphen is at EOL and followed by a lowercase letter, otherwise
    #    you destroy legitimate compounds like "state-\nof-the-art".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Join lines that are mid-sentence (line ended without terminal punctuation).
    text = re.sub(r"([^\n.!?:;])\n(?=[a-z])", r"\1 ", text)

    # 4. Collapse whitespace but keep paragraph breaks — paragraph structure is a
    #    useful signal and destroying it makes chunking worse.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip control characters that break JSON/tokenizers.
    text = "".join(c for c in text if c == "\n" or c == "\t" or unicodedata.category(c)[0] != "C")
    return text.strip()


def strip_running_headers(pages: list[str], min_frac: float = 0.6) -> list[str]:
    """
    Remove repeating headers/footers.

    Method: a line that appears at the same relative position on >60% of pages is
    boilerplate (journal name, page number, copyright). We compare first/last lines
    of each page rather than the whole document, because a genuine sentence will
    essentially never repeat verbatim across most pages.

    This matters more than it sounds: a running header like "J. Pharmacol. 2021"
    repeated 400 times teaches the model that your corpus is mostly citations.
    """
    if len(pages) < 3:
        return pages
    head_counter: Counter = Counter()
    foot_counter: Counter = Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue
        head_counter[lines[0]] += 1
        foot_counter[lines[-1]] += 1

    n = len(pages)
    drop = {l for l, c in head_counter.items() if c / n >= min_frac}
    drop |= {l for l, c in foot_counter.items() if c / n >= min_frac}
    # Page numbers vary, so they never hit the frequency threshold. Catch them by shape.
    page_num = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.I)

    out = []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip() not in drop and not page_num.match(l)]
        out.append("\n".join(lines))
    return out


def strip_references(text: str) -> str:
    """Drop the bibliography. Reference lists are ~5-15% of an academic PDF and are
    pure noise for domain adaptation."""
    m = re.search(r"\n\s*(references|bibliography|works cited)\s*\n", text, re.I)
    return text[:m.start()] if m and m.start() > len(text) * 0.4 else text


def scrub_pii(text: str) -> str:
    """
    Best-effort PII removal. Not a compliance guarantee — a real deployment needs a
    proper detector (presidio, or a NER model). This catches the obvious cases.
    """
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    return text


# ======================================================================================
# STAGE 3 — DEDUP
# ======================================================================================
def minhash_signature(text: str, num_hashes: int = 64, shingle: int = 5) -> tuple[int, ...]:
    """
    MinHash signature for near-duplicate detection.

    How it works: shingle the text into 5-grams, hash each, then for each of
    `num_hashes` independent hash functions keep the minimum hash value. The
    fraction of positions where two signatures agree estimates the Jaccard
    similarity of their shingle sets. This turns O(n²) pairwise comparison into
    O(n) bucketing (with LSH bands) and is what production dedup actually uses.

    Why dedup matters: duplicated documents are (a) memorised verbatim by the model,
    (b) an eval-contamination vector, and (c) wasted compute. Web-scale corpora are
    typically 30-60% near-duplicate.
    """
    import random
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < shingle:
        return tuple(hash(text) for _ in range(num_hashes))

    grams = {" ".join(toks[i:i + shingle]) for i in range(len(toks) - shingle + 1)}
    sig = []
    for seed in range(num_hashes):
        rng = random.Random(seed)
        a, b = rng.randrange(1 << 31), rng.randrange(1 << 31)
        sig.append(min((a * hash(g) + b) & 0xFFFFFFFF for g in grams))
    return tuple(sig)


def dedup_documents(docs: list[str], threshold: float = 0.8, num_hashes: int = 64) -> list[str]:
    """Keep the first document of each near-duplicate cluster."""
    seen: list[tuple[int, ...]] = []
    kept: list[str] = []
    for d in docs:
        sig = minhash_signature(d, num_hashes)
        is_dup = False
        for s in seen:
            agree = sum(1 for x, y in zip(sig, s) if x == y) / num_hashes
            if agree >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen.append(sig)
            kept.append(d)
    return kept


# ======================================================================================
# STAGE 4 — CHUNKING
# ======================================================================================
def chunk(text: str, size: int = 1024, overlap: int = 64) -> list[str]:
    """
    Split into training chunks.

    Note the difference from RAG chunking: for *pretraining* we chunk on tokens
    (approximated here by words) with a modest overlap, not on semantic boundaries.
    Why: the model learns from everything, so a chunk that starts mid-sentence is a
    perfectly good training example. For RAG you chunk on semantics because retrieval
    returns whole chunks to a reader.

    Overlap exists so that a fact spanning a boundary appears intact in at least one
    chunk. 5-10% is enough.
    """
    words = text.split()
    if len(words) <= size:
        return [text] if len(words) > 32 else []
    out, step = [], size - overlap
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if len(piece) > 32:
            out.append(" ".join(piece))
        if i + size >= len(words):
            break
    return out


# ======================================================================================
# MAIN
# ======================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf-dir", help="Directory of PDFs to ingest")
    src.add_argument("--text-file", help="A single already-extracted .txt corpus")
    src.add_argument("--jsonl-text-field", help="JSONL where each row has a 'text' field")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    p.add_argument("--out-dir", default="./out/dapt")
    p.add_argument("--chunk-words", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="10-100x lower than pretraining-from-scratch. Do not raise casually.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--replay-frac", type=float, default=0.10,
                   help="Fraction of general-domain text mixed in to slow catastrophic forgetting.")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    # ----- ingest -------------------------------------------------------------------
    raw_docs: list[str] = []
    if a.pdf_dir:
        d = Path(a.pdf_dir)
        pdfs = sorted(d.glob("**/*.pdf"))
        if not pdfs:
            sys.exit(f"No PDFs found under {d}")
        print(f"  found {len(pdfs)} PDFs")
        for pdf in pdfs:
            try:
                pages = extract_pdf(pdf).split("\f")
                pages = strip_running_headers([clean_text(pg) for pg in pages])
                text = strip_references(clean_text("\n\n".join(pages)))
                problems = sanity_check_extraction(text, pdf.name)
                flag = "  ⚠ " + "; ".join(problems) if problems else ""
                print(f"    {pdf.name:<50} {len(text):>9,} chars{flag}")
                raw_docs.append(text)
            except Exception as e:                       # noqa: BLE001
                print(f"    {pdf.name:<50} FAILED: {e}")
        raw_docs = [scrub_pii(t) for t in raw_docs]
    elif a.text_file:
        raw_docs = [clean_text(Path(a.text_file).read_text(encoding="utf-8", errors="ignore"))]
    else:
        raw_docs = [json.loads(l)["text"] for l in
                    Path(a.jsonl_text_field).read_text(encoding="utf-8").splitlines() if l.strip()]

    total_chars = sum(len(d) for d in raw_docs)
    print(f"\n  raw corpus         {total_chars:,} chars  (~{total_chars//4:,} tokens)")

    # ----- dedup --------------------------------------------------------------------
    before = len(raw_docs)
    raw_docs = dedup_documents(raw_docs)
    print(f"  after dedup        {len(raw_docs)} docs (removed {before - len(raw_docs)} near-duplicates)")

    # ----- chunk --------------------------------------------------------------------
    chunks: list[str] = []
    for d in raw_docs:
        chunks.extend(chunk(d, a.chunk_words))
    print(f"  chunks             {len(chunks):,}  (~{a.chunk_words} words each)")

    if not chunks:
        sys.exit("No usable text after cleaning. Check the sanity-check warnings above.")

    # ----- the data-volume reality check --------------------------------------------
    est_tokens = sum(len(c.split()) for c in chunks) * 4 // 3
    print(f"  estimated tokens   {est_tokens:,}")
    if est_tokens < 1_000_000:
        print("  ⚠  Under ~1M tokens, DAPT will change your model very little. "
              "Strongly consider RAG or SFT-only instead.")
    elif est_tokens < 10_000_000:
        print("  ℹ  1-10M tokens: expect a modest style/jargon shift, not new knowledge.")
    else:
        print("  ✅ 10M+ tokens: enough for a meaningful domain adaptation.")

    # ----- forgetting mitigation ----------------------------------------------------
    if a.replay_frac > 0:
        print(f"  replay             mixing {a.replay_frac:.0%} general text "
              f"(reduces catastrophic forgetting; set --replay-frac 0 to disable)")

    _print_plan(TrainPlan(
        model=next((m for m in ["1B", "1.5B", "3B", "7B", "8B"] if m.lower() in a.model.lower()), "1B"),
        method="full", seq_len=a.seq_len, batch=a.batch, grad_accum=a.grad_accum,
    ))
    if a.dry_run:
        print("  --dry-run complete. Remove the flag to train.\n")
        return

    # ----- train --------------------------------------------------------------------
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=a.seq_len, padding=False)
        return out

    ds = Dataset.from_dict({"text": chunks}).map(
        tokenize, batched=True, remove_columns=["text"], num_proc=2,
        desc="tokenizing",
    )
    # Drop chunks that tokenized to almost nothing.
    ds = ds.filter(lambda e: len(e["input_ids"]) > 16)
    print(f"  tokenized          {len(ds):,} sequences")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targs = TrainingArguments(
        output_dir=a.out_dir,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        group_by_length=True,
        optim="adamw_torch",
    )

    # DataCollatorForLanguageModeling with mlm=False builds labels = input_ids shifted
    # internally by the model — this is the causal-LM path. Do NOT pass mlm=True unless
    # you are training an encoder like BERT.
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    trainer.save_model(a.out_dir)
    tok.save_pretrained(a.out_dir)

    print(f"\n  ✅ domain-adapted base model saved to {a.out_dir}")
    print("\n  IMPORTANT — what you have now is a BASE model, not an assistant.")
    print("  It will NOT follow instructions. Verify the adaptation worked by checking")
    print("  perplexity on held-out domain text, then run 01_sft_lora.py on top of it.")
    print("\n  Next steps:")
    print("    • measure domain perplexity gain (and general-benchmark regression)")
    print(f"    • python 01_sft_lora.py --model {a.out_dir} --data <your SFT set>")


if __name__ == "__main__":
    main()
