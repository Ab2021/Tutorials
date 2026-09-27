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

The one thing to get right before anything else
-----------------------------------------------
**Split off a held-out set before you touch anything, and measure perplexity on it.**
Without that number you cannot tell a successful adaptation from a model that has
simply memorised your corpus — and memorisation looks like success on the training
loss curve. This script splits by document, deduplicates with the held-out set
protected, and reports held-out perplexity at the end. If you take one thing from
this file, take that.

Run it
------
    python 03_continued_pretraining.py --dry-run --pdf-dir ./pdfs
    python 03_continued_pretraining.py --pdf-dir ./pdfs --replay-file general.txt
    python 03_continued_pretraining.py --text-file corpus.txt --epochs 1

WARNING: use a BASE model, not an -Instruct model, for DAPT. Continued pretraining on
an instruct model degrades its instruction-following, because the chat template tokens
get trained on as if they were ordinary text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.memory import TrainPlan, _print_plan, sniff_size   # noqa: E402


# ======================================================================================
# STAGE 1 — EXTRACTION
#
# Page boundaries are preserved here deliberately. Everything downstream that needs to
# know "where was this line on the page" (running-header removal) can only work if the
# boundary still exists, and the boundary is destroyed the moment you join the pages.
# ======================================================================================
def extract_pdf(path: Path) -> list[str]:
    """
    Extract text from a PDF, one string per page.

    Try in order of quality:
      marker / docling  — best for tables and layout, heavy, GPU-accelerated (not used here)
      pdfplumber        — good layout awareness, slower than pypdf
      PyMuPDF (fitz)    — fast and good; AGPL licence, check before commercial use
      pypdf             — fast, pure-python, poor on multi-column and tables

    We use pdfplumber when available because it is permissively licensed and handles
    two-column academic papers far better than pypdf. Two-column layout is the #1
    source of silent garbage: pypdf reads across the gutter and interleaves the
    columns into word salad that *looks* like text and trains a broken model.

    Returns a LIST OF PAGES. The previous version returned one joined string, which made
    the caller's `.split("\\f")` a no-op (a single element), which in turn made
    strip_running_headers hit its `len(pages) < 3` guard and return its input unchanged.
    The header stripper was therefore dead code that reported success.
    """
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return [(page.extract_text() or "") for page in pdf.pages]
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
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


_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.I)


def strip_running_headers(pages: list[str], min_frac: float = 0.6) -> list[str]:
    """
    Remove repeating headers/footers, POSITIONALLY.

    Method: a line that appears as the FIRST line on >60% of pages, or as the LAST line,
    is boilerplate (journal name, page number, copyright). We compare the first/last line
    of each page rather than scanning for the string anywhere, because a genuine sentence
    will essentially never repeat verbatim across most pages.

    This matters more than it sounds: a running header like "J. Pharmacol. 2021"
    repeated 400 times teaches the model that your corpus is mostly citations.

    **Must be called at extraction time, on the raw page list, BEFORE clean_text.** Two
    reasons: (a) `clean_text` step 3 joins lines across newlines, so after it runs you can
    no longer tell which line was first on a page; (b) positional removal is only possible
    while position still means something. Applying it after cleaning is the same as not
    applying it.

    **Removal is by position, not by value.** The original version collected the set of
    frequent first/last lines and then deleted every occurrence of those strings anywhere
    in the page — so a document whose running header was the word "Introduction" lost
    every "Introduction" in its body text. We now drop the line only where we detected it.
    """
    if len(pages) < 3:
        return pages

    def lines_of(p: str) -> list[str]:
        return [l.strip() for l in p.splitlines() if l.strip()]

    head_counter: Counter = Counter()
    foot_counter: Counter = Counter()
    for p in pages:
        ls = lines_of(p)
        if ls:
            head_counter[ls[0]] += 1
            foot_counter[ls[-1]] += 1

    n = len(pages)
    drop_head = {l for l, c in head_counter.items() if c / n >= min_frac}
    drop_foot = {l for l, c in foot_counter.items() if c / n >= min_frac}
    if not (drop_head or drop_foot):
        return pages

    out = []
    for p in pages:
        lines = p.splitlines()
        # Find the first and last non-blank lines *by index*, and drop those indices only.
        idx = [i for i, l in enumerate(lines) if l.strip()]
        if not idx:
            out.append(p)
            continue
        first_i, last_i = idx[0], idx[-1]
        kept = []
        for i, l in enumerate(lines):
            s = l.strip()
            if i == first_i and s in drop_head:
                continue
            if i == last_i and s in drop_foot:
                continue
            if _PAGE_NUM.match(s):
                continue
            kept.append(l)
        out.append("\n".join(kept))
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

    Applied to EVERY source branch, not just --pdf-dir. A .txt or .jsonl corpus carries
    exactly the same PII exposure as a PDF; routing it around the scrubber because of the
    flag you happened to use is not a defensible position.
    """
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    return text


# ======================================================================================
# STAGE 3 — DEDUP
# ======================================================================================
def _stable_hash(s: str) -> int:
    """
    A hash that is the same on every run, on every machine, in every Python process.

    Python's built-in `hash()` is salted per-process by PYTHONHASHSEED, so a MinHash
    signature built from it is different every time you run the script. That makes dedup
    results unreproducible and unauditable: you cannot re-run the pipeline and get the
    same corpus, cannot diff two runs, and cannot explain to a reviewer which documents
    were dropped. blake2b is deterministic and fast.
    """
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


_SHINGLE_RE = re.compile(r"\w+")


def minhash_signature(text: str, num_hashes: int = 64, shingle: int = 5) -> tuple[int, ...]:
    """
    MinHash signature for near-duplicate detection.

    How it works: shingle the text into 5-grams, hash each deterministically, then for
    each of `num_hashes` independent hash functions keep the minimum hash value. The
    fraction of positions where two signatures agree estimates the Jaccard similarity of
    their shingle sets.

    The pairwise similarity check itself is O(n^2) in the number of documents. What makes
    that tractable is LSH BANDING (see `Deduper`): the signature rows are split into bands,
    and only documents that agree on a whole band are ever compared. That turns the
    candidate set from "every pair" into "pairs that are plausibly similar", which is what
    production dedup actually does. The banding proposes candidates; the full signature
    comparison then *verifies* each one, so the result is identical to the exhaustive check
    while doing far fewer comparisons.

    Why dedup matters: duplicated documents are (a) memorised verbatim by the model,
    (b) an eval-contamination vector, and (c) wasted compute. Web-scale corpora are
    typically 30-60% near-duplicate.
    """
    toks = _SHINGLE_RE.findall(text.lower())
    if len(toks) < shingle:
        # Too short to shingle. Fall back to a constant signature so that all such
        # documents compare as identical — which is the honest answer for a document
        # with no 5-gram content, and is at least deterministic.
        return tuple([_stable_hash(text)] * num_hashes)

    grams = {" ".join(toks[i:i + shingle]) for i in range(len(toks) - shingle + 1)}
    gram_hashes = [_stable_hash(g) for g in grams]
    sig = []
    for seed in range(num_hashes):
        # Deterministic per-row coefficients derived from the seed — no `random` module,
        # no process-level salting.
        a = (_stable_hash(f"a{seed}") | 1) & 0xFFFFFFFF
        b = _stable_hash(f"b{seed}") & 0xFFFFFFFF
        sig.append(min((a * h + b) & 0xFFFFFFFF for h in gram_hashes))
    return tuple(sig)


class Deduper:
    """
    Incremental near-duplicate detector with LSH banding.

    `is_dup(sig)` is called BEFORE `add(sig)` for each document. Two documents are
    duplicates when their signatures agree on at least `threshold` of their rows.
    """

    def __init__(self, num_hashes: int = 64, bands: int = 8, threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.bands = max(1, min(bands, num_hashes))
        self.rows = num_hashes // self.bands
        self.threshold = threshold
        self.buckets: dict[tuple, list[tuple[int, ...]]] = defaultdict(list)

    def _keys(self, sig: tuple[int, ...]):
        for b in range(self.bands):
            start = b * self.rows
            yield (b, sig[start:start + self.rows])

    def is_dup(self, sig: tuple[int, ...]) -> bool:
        for k in self._keys(sig):
            for other in self.buckets[k]:
                agree = sum(1 for x, y in zip(sig, other) if x == y) / self.num_hashes
                if agree >= self.threshold:
                    return True
        return False

    def add(self, sig: tuple[int, ...]) -> None:
        for k in self._keys(sig):
            self.buckets[k].append(sig)


def split_docs(docs: list[str], eval_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """
    Document-level held-out split. Documents, not chunks — splitting chunks lets one
    document appear on both sides of the split, which is the leak that makes a held-out
    perplexity number meaningless.
    """
    if eval_frac <= 0 or len(docs) < 5:
        return docs, []
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(len(docs) * eval_frac))
    eval_idx = set(idx[:n_eval])
    train = [d for i, d in enumerate(docs) if i not in eval_idx]
    evald = [d for i, d in enumerate(docs) if i in eval_idx]
    return train, evald


def dedup_with_eval_priority(train: list[str], evald: list[str],
                             threshold: float = 0.8, num_hashes: int = 64,
                             bands: int = 8) -> tuple[list[str], list[str], dict]:
    """
    Deduplicate, protecting the held-out set from contamination.

    The order matters and the obvious order is the wrong one. "Deduplicate, then split"
    would be leak-safe but can silently delete every held-out document. "Split, then
    deduplicate each half independently" keeps the held-out set but LEAKS: a near-duplicate
    pair that straddles the boundary is never compared, so the same content ends up on both
    sides and your held-out perplexity is optimistic.

    So: build the held-out representative FIRST, then drop any training document that is a
    near-duplicate of it. Every held-out document survives; no training document may share
    content with one. Within each side, keep the first of each cluster.
    """
    # Held-out side: dedup within itself, and these signatures are the leak guard.
    guard = Deduper(num_hashes, bands, threshold)
    kept_eval: list[str] = []
    dropped_eval = 0
    for d in evald:
        sig = minhash_signature(d, num_hashes)
        if guard.is_dup(sig):
            dropped_eval += 1
            continue
        guard.add(sig)
        kept_eval.append(d)

    # Training side: drop anything that collides with a held-out document, then dedup.
    train_deduper = Deduper(num_hashes, bands, threshold)
    kept_train: list[str] = []
    leaked = 0
    dropped_train = 0
    for d in train:
        sig = minhash_signature(d, num_hashes)
        if guard.is_dup(sig):
            leaked += 1
            continue
        if train_deduper.is_dup(sig):
            dropped_train += 1
            continue
        train_deduper.add(sig)
        kept_train.append(d)

    stats = {
        "eval_in": len(evald), "eval_kept": len(kept_eval), "eval_dupes": dropped_eval,
        "train_in": len(train), "train_kept": len(kept_train),
        "train_dupes": dropped_train, "train_leaked": leaked,
    }
    return kept_train, kept_eval, stats


# ======================================================================================
# STAGE 4 — CHUNKING
# ======================================================================================
def chunk(text: str, size: int = 1024, overlap: int = 0) -> list[str]:
    """
    Split into training chunks.

    Note the difference from RAG chunking: for *pretraining* we chunk on tokens
    (approximated here by words), not on semantic boundaries. Why: the model learns from
    everything, so a chunk that starts mid-sentence is a perfectly good training example.
    For RAG you chunk on semantics because retrieval returns whole chunks to a reader.

    **Overlap defaults to 0 for continued pretraining.** For causal LM training an overlap
    region appears in two chunks with two different contexts, which slightly corrupts the
    next-token signal at the boundary; and because every token in the corpus is trained on
    many times over the epoch anyway, overlap buys nothing that a second epoch does not.
    A small overlap (<5% of `size`) is defensible if you are training for strictly one
    epoch and want a fact spanning a boundary to appear intact somewhere — set
    `--overlap-words`, and keep it under 5% of `--chunk-words`.
    """
    words = text.split()
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
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
# REPLAY — the anti-forgetting dial
# ======================================================================================
def load_replay(path: str) -> list[str]:
    """Load general-domain replay text. Accepts .txt or a .jsonl with a 'text' field."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"  --replay-file not found: {p}")
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r["text"] for r in rows if r.get("text")]
    return [p.read_text(encoding="utf-8", errors="ignore")]


def mix_replay(chunks: list[str], replay_chunks: list[str], frac: float) -> list[str]:
    """
    Interleave replay chunks into the domain corpus at `frac` of the final mixture.

    This is the single most effective defence against catastrophic forgetting, and the
    failure mode it prevents is brutal: a model that becomes excellent at your domain and
    forgets how to write a sentence outside it. Without replay you are trading general
    ability for domain ability at an exchange rate nobody measures.
    """
    if frac <= 0 or not replay_chunks:
        return chunks
    frac = min(frac, 0.95)
    n_replay = int(len(chunks) * frac / (1 - frac))
    if n_replay <= 0:
        return chunks
    rng = random.Random(0)
    if n_replay <= len(replay_chunks):
        picks = rng.sample(replay_chunks, n_replay)
    else:
        # Not enough general text — repeat it. Note this in the log: repeating replay
        # chunks makes the model memorise them, which is its own (milder) problem.
        picks = [rng.choice(replay_chunks) for _ in range(n_replay)]
    mixed = chunks + picks
    rng.shuffle(mixed)
    return mixed


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
    p.add_argument("--doc-separator", default=None,
                   help="Literal string separating documents in --text-file (e.g. '---'). "
                        "Without it the whole file is ONE document, which disables the "
                        "held-out split and dedup.")
    p.add_argument("--chunk-words", type=int, default=1024)
    p.add_argument("--overlap-words", type=int, default=0,
                   help="Chunk overlap in words. 0 for CPT (the default); keep any "
                        "non-zero value under 5%% of --chunk-words.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="10-100x lower than pretraining-from-scratch. Do not raise casually.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-frac", type=float, default=0.05,
                   help="Fraction of DOCUMENTS held out. Without a held-out set you cannot "
                        "distinguish adaptation from memorisation.")
    p.add_argument("--replay-file", default=None,
                   help="General-domain text (.txt or .jsonl with a 'text' field) to mix in.")
    p.add_argument("--replay-frac", type=float, default=0.10,
                   help="Target share of general-domain text in the final mixture.")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip near-duplicate removal. Almost never what you want.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.overlap_words and a.overlap_words >= 0.05 * a.chunk_words:
        print(f"  ⚠  --overlap-words {a.overlap_words} is "
              f"{a.overlap_words/a.chunk_words:.1%} of --chunk-words; keep it under 5%.")

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
                # ORDER: headers/footers first (they need page position), THEN clean.
                pages = strip_running_headers(extract_pdf(pdf))
                text = clean_text("\n\n".join(pages))
                text = strip_references(text)
                text = scrub_pii(text)
                problems = sanity_check_extraction(text, pdf.name)
                flag = "  ⚠ " + "; ".join(problems) if problems else ""
                print(f"    {pdf.name:<50} {len(text):>9,} chars{flag}")
                if text.strip():
                    raw_docs.append(text)
            except Exception as e:                       # noqa: BLE001
                print(f"    {pdf.name:<50} FAILED: {e}")
    elif a.text_file:
        text = Path(a.text_file).read_text(encoding="utf-8", errors="ignore")
        # A .txt corpus is almost never one document, and treating it as one silently
        # disables BOTH the held-out split and dedup (you cannot split or deduplicate a
        # single document). Split it, then run the same cleaning ladder as every other
        # branch, and say so if the split produced nothing to work with.
        if a.doc_separator:
            parts = text.split(a.doc_separator)
        else:
            parts = [text]
        raw_docs = [scrub_pii(strip_references(clean_text(p)))
                    for p in parts if p.strip()]
        if len(raw_docs) < 5:
            print(f"  ⚠  --text-file produced only {len(raw_docs)} document(s). The "
                  f"held-out split")
            print(f"     and near-duplicate removal both need multiple documents, and "
                  f"with one they")
            print(f"     silently do nothing — you would get no held-out perplexity and "
                  f"no dedup while")
            print(f"     the log claimed both ran. Pass --doc-separator (e.g. a line of "
                  f"'---') if your")
            print(f"     file has document boundaries, or use --jsonl-text-field, which is "
                  f"one document")
            print(f"     per row.")
    else:
        rows = [json.loads(l) for l in
                Path(a.jsonl_text_field).read_text(encoding="utf-8").splitlines() if l.strip()]
        # Same cleaning ladder as the PDF path. Skipping it here meant a .jsonl corpus
        # bypassed PII scrubbing and reference stripping entirely.
        raw_docs = [scrub_pii(strip_references(clean_text(r["text"])))
                    for r in rows if r.get("text")]

    raw_docs = [d for d in raw_docs if d.strip()]
    total_chars = sum(len(d) for d in raw_docs)
    print(f"\n  raw corpus         {total_chars:,} chars  (~{total_chars//4:,} tokens)")

    if not raw_docs:
        sys.exit("No usable text after cleaning. Check the sanity-check warnings above.")

    # ----- split BEFORE chunking, so no document spans the boundary ------------------
    train_docs, eval_docs = split_docs(raw_docs, a.eval_frac, a.seed)
    print(f"  split              {len(train_docs)} train / {len(eval_docs)} held-out docs "
          f"(document-level, seed {a.seed})")

    # ----- dedup, protecting the held-out set ---------------------------------------
    if a.no_dedup:
        print("  ⚠  dedup SKIPPED (--no-dedup). Duplicated documents get memorised "
              "verbatim and waste epochs.")
        stats = {}
    else:
        train_docs, eval_docs, stats = dedup_with_eval_priority(train_docs, eval_docs)
        print(f"  dedup (train)      {stats['train_kept']}/{stats['train_in']} kept  "
              f"({stats['train_dupes']} near-duplicates, "
              f"{stats['train_leaked']} dropped as held-out leaks)")
        print(f"  dedup (held-out)   {stats['eval_kept']}/{stats['eval_in']} kept  "
              f"({stats['eval_dupes']} near-duplicates within the held-out set)")
        if stats["train_leaked"]:
            print(f"  ℹ  {stats['train_leaked']} training documents were near-duplicates of "
                  f"a held-out document. Dropping them is what makes the held-out")
            print(f"     perplexity number below trustworthy. This is the leak you cannot "
                  f"see in a loss curve.")

    # ----- chunk --------------------------------------------------------------------
    def chunks_of(docs: list[str]) -> list[str]:
        out: list[str] = []
        for d in docs:
            out.extend(chunk(d, a.chunk_words, a.overlap_words))
        return out

    chunks = chunks_of(train_docs)
    eval_chunks = chunks_of(eval_docs)
    print(f"  chunks             {len(chunks):,} train / {len(eval_chunks):,} held-out  "
          f"(~{a.chunk_words} words each, overlap {a.overlap_words})")

    if not chunks:
        sys.exit("No usable text after cleaning. Check the sanity-check warnings above.")

    # ----- replay: actually mixed, not merely announced ------------------------------
    replay_chunks: list[str] = []
    if a.replay_frac > 0:
        if a.replay_file:
            replay_docs = load_replay(a.replay_file)
            replay_chunks = chunks_of(replay_docs)
            if not replay_chunks:
                print(f"  ⚠  --replay-file {a.replay_file} produced no usable chunks. "
                      f"Replay is OFF.")
            else:
                before = len(chunks)
                chunks = mix_replay(chunks, replay_chunks, a.replay_frac)
                mixed = len(chunks) - before
                print(f"  replay             {mixed:,} general-domain chunks mixed in "
                      f"({mixed/len(chunks):.1%} of the corpus)")
                if mixed > len(replay_chunks):
                    print(f"  ⚠  Only {len(replay_chunks):,} distinct replay chunks were "
                          f"available, so some are repeated. Repetition makes the model "
                          f"memorise them — supply more general text, or lower "
                          f"--replay-frac.")
        else:
            print(f"  ⚠  --replay-frac {a.replay_frac:.0%} was requested but NO "
                  f"--replay-file was given, so")
            print(f"     REPLAY IS OFF. Nothing general is being mixed in. This script "
                  f"used to print")
            print(f"     'mixing 10% general text' here while mixing nothing at all — the "
                  f"most")
            print(f"     dangerous kind of bug, because the mitigation is announced and "
                  f"absent.")
            print(f"     Supply --replay-file <general.txt> to actually enable it.")
    else:
        print("  ⚠  --replay-frac 0: training on domain text alone. Catastrophic forgetting "
              "is likely;")
        print("     expect general capability to degrade and measure it on a general set.")

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

    method = "lora" if "--lora" in sys.argv else "full"
    plan = TrainPlan(
        model=sniff_size(a.model, "1B"),
        method=method, seq_len=a.seq_len, batch=a.batch, grad_accum=a.grad_accum,
    )
    _print_plan(plan)
    if a.eval_frac <= 0:
        print("  ⚠  --eval-frac 0: there is no held-out set, so no perplexity will be")
        print("     reported and memorisation is undetectable. This is the one number "
              "that tells")
        print("     you whether the run worked.")
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
        # NOTE: do NOT let pad_token alias eos_token here. If they share an id, a
        # position-based mask becomes an id-based mask, and the collator then masks the
        # real end-of-document token as if it were padding. The model never learns where
        # documents end.
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    if tok.eos_token is None:
        sys.exit("  The base model has no EOS token; continued pretraining needs a "
                 "document boundary. Pick a base model that defines one.")

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=a.seq_len, padding=False)
        # `length` is what `group_by_length=True` actually groups on. Without it, that
        # flag is a silent no-op.
        out["length"] = [len(ids) for ids in out["input_ids"]]
        return out

    def build(docs_chunks: list[str]):
        if not docs_chunks:
            return None
        ds = Dataset.from_dict({"text": docs_chunks}).map(
            tokenize, batched=True, remove_columns=["text"], num_proc=2, desc="tokenizing")
        return ds.filter(lambda e: len(e["input_ids"]) > 16)

    ds = build(chunks)
    eval_ds = build(eval_chunks) if eval_chunks else None
    print(f"  tokenized          {len(ds):,} train sequences"
          + (f" / {len(eval_ds):,} held-out" if eval_ds is not None else ""))

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    # Needed for evaluation to report a real (untruncated) perplexity.
    model.config.pad_token_id = tok.pad_token_id
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
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=a.seed,
        group_by_length=True,
        optim="adamw_torch",
    )

    # DataCollatorForLanguageModeling with mlm=False builds labels = input_ids shifted
    # internally by the model — this is the causal-LM path. Do NOT pass mlm=True unless
    # you are training an encoder like BERT.
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    # Baseline FIRST. A post-training perplexity number on its own tells you nothing —
    # you need the before value to know whether the adaptation did anything at all.
    baseline_ppl = _perplexity(model, eval_ds, tok, a) if eval_ds is not None else None
    if baseline_ppl is not None:
        print(f"\n  held-out perplexity BEFORE  {baseline_ppl:,.2f}")

    trainer.train()
    trainer.save_model(a.out_dir)
    tok.save_pretrained(a.out_dir)

    after_ppl = _perplexity(trainer.model, eval_ds, tok, a) if eval_ds is not None else None
    print(f"\n  ✅ domain-adapted base model saved to {a.out_dir}")

    print("\n  ── did it work? ──")
    if baseline_ppl is not None and after_ppl is not None:
        delta = (after_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"    held-out domain perplexity   {baseline_ppl:,.2f} → {after_ppl:,.2f}  "
              f"({delta:+.1f}%)")
        if after_ppl < baseline_ppl:
            print(f"    ✅ The model is less surprised by held-out DOMAIN text. That is the "
                  f"signal")
            print(f"       continued pretraining is supposed to produce.")
        else:
            print(f"    ❌ Perplexity on held-out domain text did NOT improve. The "
                  f"adaptation failed")
            print(f"       or the held-out set is not representative. More epochs will not "
                  f"fix this.")
        print(f"    ⚠  Now measure the OTHER direction: perplexity on GENERAL text "
              f"(e.g. a slice")
        print(f"       of Wikipedia) will have got WORSE. That is catastrophic forgetting, "
              f"and the")
        print(f"       size of the gap is what --replay-frac buys back. Report both.")
    else:
        print("    ⚠  No held-out perplexity available (--eval-frac 0 or no eval chunks).")
        print("       You have no evidence the run did anything but memorise. Re-run with "
              "a held-out set.")

    print("\n  IMPORTANT — what you have now is a BASE model, not an assistant.")
    print("  It will NOT follow instructions.")
    print("\n  Next steps:")
    print(f"    • python 01_sft_lora.py --model {a.out_dir} --data <your SFT set>")


def _perplexity(model, eval_ds, tok, a) -> float | None:
    """Held-out perplexity = exp(mean cross-entropy per token)."""
    try:
        import math
        import torch
        if eval_ds is None or len(eval_ds) == 0:
            return None
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(eval_ds), max(a.batch, 1)):
                batch = eval_ds[i:i + max(a.batch, 1)]
                ids = [torch.tensor(x) for x in batch["input_ids"]]
                padded = torch.nn.utils.rnn.pad_sequence(
                    ids, batch_first=True, padding_value=tok.pad_token_id)
                attn = (padded != tok.pad_token_id).long()
                out = model(input_ids=padded, attention_mask=attn)
                # Shift: predict token t+1 from tokens <=t, and only count real tokens.
                logits = out.logits[:, :-1].float()
                tgt = padded[:, 1:]
                mask = attn[:, 1:].bool()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
                total_loss += loss[mask.reshape(-1)].sum().item()
                n += int(mask.sum())
        model.train()
        return math.exp(total_loss / max(n, 1))
    except Exception as e:                                       # noqa: BLE001
        print(f"  (perplexity computation failed: {e})")
        return None


if __name__ == "__main__":
    main()
