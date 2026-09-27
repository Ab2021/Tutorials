#!/usr/bin/env python
"""
10_embedding_finetune.py — fine-tune a sentence embedding model for retrieval.

Embedding fine-tuning is a different problem from LLM fine-tuning
----------------------------------------------------------------
An LLM is trained to generate. An embedding model is trained to place texts in a space
where a distance function means something. So the objective is not cross-entropy over
tokens — it is a **contrastive** loss over PAIRS, and the entire quality of your result is
determined by the quality of your negatives. Not by the learning rate, not by the base
model, not by the epochs. By the negatives.

The loss families
-----------------
  * **MultipleNegativesRankingLoss (MNRL)** — the workhorse. For a batch of (anchor,
    positive) pairs, every OTHER pair's positive acts as a negative for this anchor. The
    batch IS your negative pool, so larger batch = harder problem = better embeddings.
    Needs only positives; this is why it is the default.
  * **TripletLoss** — explicit (anchor, positive, negative). You control the negatives,
    which is better when you have genuinely hard ones, at the cost of having to mine them.
  * **ContrastiveLoss** — pairs with a binary similar/dissimilar label.
  * **CoSENT / AnglE** — ranking losses over all pairs in a batch. Often better than
    cosine-similarity losses on small data.

The finding that matters most for evaluation
--------------------------------------------
There is a famous result in the source material: a 768-dimensional model scored ~88% on a
similarity task while a 384-dimensional model scored ~51%, and the natural reading —
"more dimensions is better" — is BACKWARDS as an explanation. Embedding quality is driven
by the training objective, not the vector width. A model with an un-contrastive objective
produces **anisotropic** embeddings: everything clustered in a narrow cone, so every pair
looks similarly similar and cosine similarity carries almost no signal. That is what
"51%" means. Dimension is a red herring; the objective is the cause.

Run it
------
    python 10_embedding_finetune.py --data data/pairs.jsonl --validate
    python 10_embedding_finetune.py --data data/pairs.jsonl --out out/embed --dry-run
    python 10_embedding_finetune.py --data data/pairs.jsonl --out out/embed --mine-hard-negatives
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix

DEFAULTS = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "loss": "mnrl",
    "epochs": 3,
    "batch_size": 32,      # matters a LOT for MNRL — see the docstring
    "lr": 2e-5,
    "max_len": 256,
    "warmup_ratio": 0.1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="jsonl with anchor/positive and optionally negative")
    p.add_argument("--out", default="out/embed")
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--loss", default=DEFAULTS["loss"],
                   choices=["mnrl", "triplet", "contrastive", "cosent"])
    p.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    p.add_argument("--anchor-col", default="anchor")
    p.add_argument("--positive-col", default="positive")
    p.add_argument("--negative-col", default="negative")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-queries", default=None,
                   help="jsonl of {query, relevant:[...]} for recall@k before/after")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = _load(a.data)
    print(f"\n  ── embedding fine-tuning ──")
    print(f"  base model         {a.model}")
    print(f"  pairs              {len(rows)}")
    print(f"  loss               {a.loss}")
    print(f"  batch size         {a.batch_size}")

    _validate(rows, a)
    _explain_loss(a)

    if a.eval_queries and Path(a.eval_queries).exists():
        _baseline_eval(a)

    if a.dry_run or a.validate:
        print("\n  --dry-run: no training performed.")
        return
    _train(a, rows)


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"  No such file: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------
def _validate(rows: list[dict], a) -> None:
    problems: dict[str, int] = {}

    def flag(k: str, detail: str | None = None) -> None:
        problems[k] = problems.get(k, 0) + 1
        if detail and problems[k] <= 2:
            print(f"    ⚠  {k}: {detail}")

    has_neg = 0
    for i, r in enumerate(rows):
        anc = r.get(a.anchor_col) or r.get("query") or r.get("question")
        pos = r.get(a.positive_col) or r.get("passage") or r.get("answer")
        if not anc or not pos:
            flag("missing_anchor_or_positive", f"row {i}: keys={sorted(r)[:6]}")
            continue
        if str(anc).strip() == str(pos).strip():
            # An identical pair teaches "everything is similar to itself" — zero signal
            # and it actively hurts, because it is trivially satisfied.
            flag("anchor_equals_positive", f"row {i}")
        if len(str(pos).split()) < 3:
            flag("positive_too_short", f"row {i}: {len(str(pos).split())} words")
        if r.get(a.negative_col):
            has_neg += 1

    print(f"\n  ── data ──")
    print(f"  with negatives     {has_neg}/{len(rows)}")

    if a.loss == "mnrl" and has_neg == 0:
        print("\n  ℹ  MNRL needs no explicit negatives — it uses the other pairs in the")
        print("     batch. That is its main advantage, and also its constraint:")
        print(f"     with batch_size={a.batch_size} each anchor sees only {a.batch_size - 1}")
        print("     negatives per step. Small batches make MNRL weak. Push the batch size")
        print("     up (gradient caching / larger GPU) before you push the epochs up.")
    if a.loss in ("triplet", "cosent") and has_neg == 0:
        print(f"\n  ⚠  --loss {a.loss} expects explicit negatives and none were found.")
        print("     It will fall back to in-batch negatives, which defeats the point.")

    if problems:
        print(f"\n  ⚠  {sum(problems.values())} issue(s): {problems}")
    else:
        print("  ✅ structure looks fine")

    print("\n  ⚠  THE THING THAT DECIDES YOUR RESULT:")
    print("     Negative quality, not hyperparameters. 'Random' negatives (unrelated")
    print("     documents) are easy — the model learns a coarse topic split and stops.")
    print("     What you want are HARD negatives: passages that are topically similar but")
    print("     not the answer. Mine them from your own retriever's top-k misses.")
    print("     A model trained only on easy negatives scores well on your eval set and")
    print("     badly in production, because production queries are the hard ones.")


def _explain_loss(a) -> None:
    print(f"\n  ── what {a.loss} is doing ──")
    if a.loss == "mnrl":
        print("    For each (anchor, positive) in the batch, the other positives serve as")
        print("    negatives. Loss = cross-entropy over cosine similarities scaled by a")
        print("    temperature:")
        print("      L = -log( exp(sim(a,p+)/τ) / Σ_j exp(sim(a,p_j)/τ) )")
        print("    In-batch negatives are a free, automatic hard-negative miner — with a")
        print("    large enough batch, some of those other positives ARE near-misses.")
    elif a.loss == "triplet":
        print("    L = max(0, sim(a,n) - sim(a,p) + margin)")
        print("    Only pairs that violate the margin contribute. If your margin is too")
        print("    small, most triplets contribute nothing and the model stops learning —")
        print("    watch the fraction of 'active' triplets.")
    elif a.loss == "cosent":
        print("    A ranking loss over all pairs in the batch; often more stable than a")
        print("    cosine loss when the dataset is small.")


# --------------------------------------------------------------------------------------
def _baseline_eval(a) -> None:
    """
    Measure the BASE model BEFORE fine-tuning. Without this you cannot tell whether your
    fine-tune helped or hurt — and it frequently HURTS on general-domain queries.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ⚠  sentence-transformers not installed; skipping baseline eval.")
        return

    queries = _load(a.eval_queries)
    model = SentenceTransformer(a.model)
    print(f"\n  ── baseline (before fine-tuning) ──")
    _score(model, queries)


def _score(model, queries) -> None:
    """
    Recall@k and MRR. Report these, not 'average cosine similarity' — similarity has no
    absolute meaning, only a ranking does.
    """
    import numpy as np

    corpus, q_texts, relevant = [], [], []
    seen: dict[str, int] = {}
    for q in queries:
        rel = q.get("relevant") or q.get("positives") or []
        if not rel:
            continue
        for r in rel:
            if r not in seen:
                seen[r] = len(corpus)
                corpus.append(r)
        q_texts.append(q.get("query") or q.get("question") or "")
        relevant.append([seen[r] for r in rel])

    if not corpus:
        print("    no relevant passages found in the eval file")
        return

    c_emb = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    q_emb = model.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
    sims = np.asarray(q_emb) @ np.asarray(c_emb).T

    for k in (1, 5, 10):
        if k > len(corpus):
            continue
        hits = 0
        for i, rel in enumerate(relevant):
            top = np.argsort(-sims[i])[:k]
            if any(r in top for r in rel):
                hits += 1
        print(f"    recall@{k:<3}        {hits/len(relevant):.3f}")

    rr = []
    for i, rel in enumerate(relevant):
        order = np.argsort(-sims[i])
        rank = next((int(np.where(order == r)[0][0]) + 1 for r in rel
                     if r in order), None)
        rr.append(1.0 / rank if rank else 0.0)
    print(f"    MRR            {sum(rr)/len(rr):.3f}")
    print("\n    ↑ Save these. After fine-tuning, re-run and compare. A fine-tune that")
    print("      improves in-domain recall while collapsing on general queries is a")
    print("      REGRESSION for a RAG system — it will retrieve well for your test set")
    print("      and badly for the questions users actually ask.")


def _train(a, rows: list[dict]) -> None:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        sys.exit("  pip install sentence-transformers")

    model = SentenceTransformer(a.model)
    model.max_seq_length = a.max_len

    examples = []
    for r in rows:
        anc = r.get(a.anchor_col) or r.get("query")
        pos = r.get(a.positive_col) or r.get("passage")
        neg = r.get(a.negative_col)
        if not anc or not pos:
            continue
        if neg:
            examples.append(InputExample(texts=[anc, pos, neg]))
        else:
            examples.append(InputExample(texts=[anc, pos]))

    print(f"\n  {len(examples)} training examples")

    if a.loss == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif a.loss == "triplet":
        train_loss = losses.TripletLoss(model)
    elif a.loss == "cosent":
        train_loss = losses.CoSENTLoss(model)
    else:
        train_loss = losses.ContrastiveLoss(model)

    loader = DataLoader(examples, shuffle=True, batch_size=a.batch_size)

    print("  training...")
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=int(a.epochs),
        warmup_steps=math.ceil(len(loader) * a.epochs * DEFAULTS["warmup_ratio"]),
        optimizer_params={"lr": a.lr},
        output_path=a.out,
        show_progress_bar=True,
    )
    print(f"\n  ✅ saved to {a.out}")

    if a.eval_queries and Path(a.eval_queries).exists():
        print("\n  ── after fine-tuning ──")
        _score(model, _load(a.eval_queries))

    print("\n  ── before you ship this ──")
    print("     1. Compare recall@k / MRR BEFORE vs AFTER, on data the model never saw.")
    print("     2. Test on OUT-OF-DOMAIN queries. Embedding fine-tuning forgets general")
    print("        language noticeably. If your index holds mixed content, this is the")
    print("        failure that hurts most and shows up last.")
    print("     3. You MUST re-embed the whole corpus with the new model. Old vectors and")
    print("        new query vectors live in different spaces; mixing them returns")
    print("        confident nonsense with no error.")
    print("     4. Check the anisotropy of the output: if the mean pairwise cosine")
    print("        similarity across random texts is >0.8, the space is collapsed and")
    print("        ranking will be poor regardless of your recall on the test set.")


if __name__ == "__main__":
    main()
