#!/usr/bin/env python
"""
11_bert_classification.py — fine-tune an encoder (BERT/RoBERTa/DeBERTa) for
classification and token classification (NER).

Encoder fine-tuning is NOT decoder fine-tuning, and the differences bite
------------------------------------------------------------------------
  1. **Learning rate.** 2e-5, not 2e-4. Encoders are far more sensitive; a decoder-style
     LR destroys the pretrained representations in a few hundred steps. This single number
     is the most common reason a BERT fine-tune "doesn't work".
  2. **Warmup matters more.** 6-10% of steps. Without it the first few large updates can
     wreck the encoder before the head has learned anything.
  3. **You replace the head, and that discards pretrained weights.** Calling
     `from_pretrained(num_labels=K)` keeps the *body* and reinitialises the classification
     head. That is correct and intended — but note the consequence: the head is random, so
     the loss starts at ln(K), and a few dozen steps of high loss can push the body around.
     Low LR + warmup is what protects it.
  4. **The pooler is often frozen by accident.** `bert.pooler` sits between the encoder and
     the head. Some recipes freeze it, some do not. Decide deliberately.
  5. **Token classification is a subword-alignment problem**, and most of the bugs live
     there, not in the model. See the alignment section below.

The label-remap trap
--------------------
If you filter your dataset (say, to drop three rare classes), the surviving label integers
do NOT get renumbered. You can end up with labels {0, 1, 3} while `num_labels=3`, and then
the model trains happily, evaluates to a plausible-looking accuracy, and is wrong about
everything class 3 would have been. The script checks for this explicitly.

Run it
------
    python 11_bert_classification.py --task classification --data data/sst2.jsonl --dry-run
    python 11_bert_classification.py --task classification --data data/sst2.jsonl --out out/bert-cls
    python 11_bert_classification.py --task ner --data data/conll.jsonl --out out/bert-ner
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
    "model": "bert-base-uncased",
    "lr": 2e-5,           # NOT 2e-4 — see the docstring
    "epochs": 3,
    "batch_size": 16,
    "max_len": 256,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["classification", "ner"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="out/bert")
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze the bottom N encoder layers (0 = train all)")
    p.add_argument("--freeze-pooler", action="store_true")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--tokens-col", default="tokens")
    p.add_argument("--tags-col", default="tags")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = _load(a.data)
    print(f"\n  ── {a.task} ──")
    print(f"  rows               {len(rows)}")
    print(f"  base model         {a.model}")
    print(f"  lr                 {a.lr:g}")
    if a.lr > 5e-5 and "bert" in a.model.lower():
        print("  ⚠  Learning rate > 5e-5 on an encoder is very likely too high.")
        print("     Encoders want 1e-5..5e-5. Decoder LRs (1e-4+) destroy them. If training")
        print("     loss drops while validation rises early, this is the cause.")

    if a.task == "classification":
        _check_classification(rows, a)
    else:
        _check_ner(rows, a)

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
def _check_classification(rows: list[dict], a) -> None:
    labels = [r.get(a.label_col) for r in rows]
    if any(l is None for l in labels):
        sys.exit(f"  Rows are missing the '{a.label_col}' field. "
                 f"Present keys: {sorted(rows[0])}")

    # Accept both string labels and ints — string labels are the safer input.
    distinct = sorted({str(l) for l in labels})
    counts = Counter(str(l) for l in labels)
    print(f"  classes            {len(distinct)}  {distinct[:12]}")

    # ---- the label-remap trap ------------------------------------------------------
    ints = []
    for l in labels:
        try:
            ints.append(int(l))
        except (TypeError, ValueError):
            break
    if len(ints) == len(labels) and ints:
        lo, hi = min(ints), max(ints)
        if hi >= len(distinct) or lo != 0:
            print(f"\n  ⚠  LABEL INDEX GAP: labels span {lo}..{hi} but there are "
                  f"{len(distinct)} distinct classes.")
            print("     If you filtered the dataset without renumbering, the surviving")
            print("     labels are non-contiguous (e.g. {0,1,3}). Training with")
            print(f"     num_labels={len(distinct)} on labels up to {hi} either crashes or")
            print("     silently trains a head that can never predict the high classes.")
            print("     Fix: remap labels to 0..K-1 BEFORE training.")
            sys.exit(1)

    # ---- class imbalance -----------------------------------------------------------
    if counts:
        maj = counts.most_common(1)[0]
        mnr = counts.most_common()[-1]
        ratio = maj[1] / max(mnr[1], 1)
        print(f"  majority/minority  {maj[0]}={maj[1]} / {mnr[0]}={mnr[1]}  ({ratio:.1f}x)")
        if ratio > 10:
            print(f"  ⚠  {ratio:.0f}x class imbalance.")
            print("     Accuracy becomes a misleading metric — a model that always predicts")
            print(f"     '{maj[0]}' scores {maj[1]/len(labels):.0%} and learns nothing.")
            print("     Report macro-F1 and per-class recall instead, and consider class")
            print("     weights in the loss. Do NOT compare runs on accuracy alone.")

    lens = [len(str(r.get(a.text_col, "")).split()) for r in rows]
    lens.sort()
    print(f"  words p50/p95/max  {lens[len(lens)//2]} / {lens[int(len(lens)*0.95)]} / "
          f"{lens[-1]}")
    over = sum(1 for l in lens if l > a.max_len * 0.75)
    if over:
        print(f"  ℹ  ~{over} rows likely exceed max_len={a.max_len} and will be truncated.")
        print("     Encoder truncation silently drops the END of the text — which for a")
        print("     long review is often where the verdict is. Check this before trusting")
        print("     your metrics.")


def _check_ner(rows: list[dict], a) -> None:
    if a.tokens_col not in rows[0] or a.tags_col not in rows[0]:
        sys.exit(f"  NER needs '{a.tokens_col}' and '{a.tags_col}'. "
                 f"Present keys: {sorted(rows[0])}")

    tags = Counter(t for r in rows for t in r[a.tags_col])
    print(f"  tag set            {len(tags)}  {sorted(tags)[:12]}")

    # ---- alignment sanity ----------------------------------------------------------
    bad = [i for i, r in enumerate(rows)
           if len(r[a.tokens_col]) != len(r[a.tags_col])]
    if bad:
        print(f"\n  ⚠  {len(bad)} rows where len(tokens) != len(tags). First: row {bad[0]}")
        print("     This is the single most common NER data bug. It crashes inside the")
        print("     tokenizer's word_ids alignment or, worse, silently mislabels.")
        sys.exit(1)

    # ---- the BIO scheme check ------------------------------------------------------
    has_bio = any(t.startswith("B-") or t.startswith("I-") for t in tags)
    if not has_bio and any("-" in t for t in tags):
        print("  ⚠  Tags contain '-' but no B-/I- prefixes. Unexpected tagging scheme —")
        print("     confirm whether this is BIO, BIOES, or something else before training.")

    # ---- subword alignment demonstration -------------------------------------------
    print("\n  ── subword alignment (where NER bugs live) ──")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True)
    ex = rows[0]
    enc = tok(ex[a.tokens_col], is_split_into_words=True,
              truncation=True, max_length=a.max_len)
    word_ids = enc.word_ids()
    print(f"  words (first 8)    {ex[a.tokens_col][:8]}")
    print(f"  tags  (first 8)    {ex[a.tags_col][:8]}")
    print(f"  subword tokens     {tok.convert_ids_to_tokens(enc['input_ids'])[:14]}")
    print(f"  word_ids           {word_ids[:14]}")
    print("  ↑ word_ids maps each subword back to its source word (None for [CLS]/[SEP]).")
    print("    The alignment rule that matters:")
    print("      * Label the FIRST subword of each word with the word's tag.")
    print("      * Label all CONTINUATION subwords with -100 (ignored by the loss).")
    print("    If you instead copy the tag onto every subword, a word split into 3 pieces")
    print("    contributes 3x the loss and the model learns to over-predict long entities.")
    print("    If you label only the first and use the word's tag on continuations for a")
    print("    B- tag, you create invalid B-I-B sequences that seqeval will count as errors.")

    n_sub = sum(len(tok(word, add_special_tokens=False)["input_ids"])
                for word in ex[a.tokens_col])
    print(f"\n  expansion          {len(ex[a.tokens_col])} words → {n_sub} subwords "
          f"({n_sub/max(len(ex[a.tokens_col]),1):.2f}x)")
    print("    This ratio is also your sequence-length multiplier: max_len counted in")
    print("    WORDS is not max_len counted in subwords.")


# --------------------------------------------------------------------------------------
def _train(a, rows: list[dict]) -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoModelForTokenClassification, AutoTokenizer,
                                  DataCollatorForTokenClassification, Trainer,
                                  TrainingArguments)
    except ImportError as e:
        sys.exit(f"  pip install torch transformers datasets evaluate seqeval  ({e})")

    if a.task == "classification":
        _train_classification(a, rows, np, torch, Dataset, AutoTokenizer,
                              AutoModelForSequenceClassification, Trainer, TrainingArguments)
    else:
        _train_ner(a, rows, np, torch, Dataset, AutoTokenizer,
                   AutoModelForTokenClassification, DataCollatorForTokenClassification,
                   Trainer, TrainingArguments)


def _freeze(a, model, torch) -> None:
    """Freeze the bottom N encoder layers and/or the pooler."""
    if a.freeze_layers:
        # `bert.encoder.layer` is the standard attribute across BERT/RoBERTa/DeBERTa.
        for name in ("bert", "roberta", "deberta"):
            base = getattr(model, name, None)
            if base is not None and hasattr(base, "encoder"):
                for layer in base.encoder.layer[:a.freeze_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False
                print(f"  froze              bottom {a.freeze_layers} encoder layers")
                break
    if a.freeze_pooler:
        for name in ("bert", "roberta", "deberta"):
            base = getattr(model, name, None)
            if base is not None and hasattr(base, "pooler") and base.pooler is not None:
                for p in base.pooler.parameters():
                    p.requires_grad = False
                print("  froze              pooler")
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable          {trainable:,} / {total:,} ({trainable/total:.1%})")
    if trainable / total < 0.05:
        print("  ⚠  Under 5% trainable. Fine if deliberate — but with a random head and")
        print("     most of the body frozen you are close to training a linear probe.")


def _train_classification(a, rows, np, torch, Dataset, AutoTokenizer, AutoModel, Trainer,
                          TrainingArguments) -> None:
    labels = sorted({str(r[a.label_col]) for r in rows})
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"\n  label map          {lab2id}")

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = Dataset.from_list([
        {"text": str(r[a.text_col]), "label": lab2id[str(r[a.label_col])]} for r in rows])

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=a.max_len, padding=False)

    ds = ds.map(encode, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)

    model = AutoModel.from_pretrained(a.model, num_labels=len(labels))
    print(f"  ⚠  num_labels={len(labels)} reinitialises the classification head. It is")
    print("     random at step 0, so the initial loss should be ≈ln({})={:.3f}. If it is"
          .format(len(labels), np.log(len(labels))))
    print("     far from that, something upstream is wrong.")
    _freeze(a, model, torch)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        from sklearn.metrics import accuracy_score, f1_score
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "macro_f1": f1_score(p.label_ids, preds, average="macro")}

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr, warmup_ratio=DEFAULTS["warmup_ratio"],
        weight_decay=DEFAULTS["weight_decay"], logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        report_to="none", seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=split["train"],
                      eval_dataset=split["test"], compute_metrics=metrics)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"\n  ✅ saved to {a.out}")
    _closing_advice(a, "macro_f1")


def _train_ner(a, rows, np, torch, Dataset, AutoTokenizer, AutoModelTok,
               Collator, Trainer, TrainingArguments) -> None:
    tags = sorted({t for r in rows for t in r[a.tags_col]})
    tag2id = {t: i for i, t in enumerate(tags)}
    print(f"\n  tag map            {len(tags)} tags")
    print(f"  ⚠  num_labels={len(tags)} — this must match the tag map EXACTLY, including")
    print("     every B-/I- variant and the outside tag. Off-by-one here trains a model")
    print("     whose predictions are silently shifted by one class.")

    tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True)

    def align(examples):
        enc = tok(examples[a.tokens_col], is_split_into_words=True,
                  truncation=True, max_length=a.max_len)
        all_labels = []
        for i, tags_row in enumerate(examples[a.tags_col]):
            word_ids = enc.word_ids(batch_index=i)
            prev, out = None, []
            for wid in word_ids:
                if wid is None:
                    out.append(-100)               # [CLS], [SEP], padding
                elif wid != prev:
                    out.append(tag2id[tags_row[wid]])   # first subword: the real tag
                else:
                    out.append(-100)               # continuation: ignored by the loss
                prev = wid
            all_labels.append(out)
        enc["labels"] = all_labels
        return enc

    ds = Dataset.from_list([{a.tokens_col: r[a.tokens_col], a.tags_col: r[a.tags_col]}
                            for r in rows])
    ds = ds.map(align, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)

    model = AutoModelTok.from_pretrained(a.model, num_labels=len(tags))
    _freeze(a, model, torch)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        # Strip the ignored positions BEFORE scoring, or seqeval chokes on -100.
        true_l, true_p = [], []
        for pr, lb in zip(preds, p.label_ids):
            mask = lb != -100
            true_l.append([tags[i] for i in lb[mask]])
            true_p.append([tags[i] for i in pr[mask]])
        try:
            from seqeval.metrics import classification_report, f1_score
            print("\n" + classification_report(true_l, true_p))
            return {"f1": f1_score(true_l, true_p)}
        except ImportError:
            # Token-level accuracy is NOT entity-level F1 and is much more forgiving:
            # most tokens are 'O', so a model predicting all-O scores high here.
            flat_t = [t for row in true_l for t in row]
            flat_p = [t for row in true_p for t in row]
            return {"token_accuracy": float(np.mean([a_ == b_ for a_, b_ in
                                                     zip(flat_t, flat_p)]))}

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=DEFAULTS["warmup_ratio"],
        weight_decay=DEFAULTS["weight_decay"], logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, report_to="none", seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=split["train"],
                      eval_dataset=split["test"], compute_metrics=metrics,
                      data_collator=Collator(tok))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"\n  ✅ saved to {a.out}")
    _closing_advice(a, "entity F1")


def _closing_advice(a, metric: str) -> None:
    print(f"\n  ── before you trust this model ──")
    print(f"     1. Report {metric}, not accuracy. A trivial baseline usually scores")
    print("        better on accuracy than a real model does on F1.")
    print("     2. Compute the majority-class baseline EXPLICITLY. If your model does not")
    print("        beat it, it learned nothing. This is the check people skip.")
    print("     3. Look at the confusion matrix, not just the headline number. Which")
    print("        classes are being merged tells you whether the problem is data or model.")
    print("     4. Check calibration if you will threshold the outputs.")
    print(f"     5. Encoder inference is cheap — compare against a prompted LLM on")
    print("        cost-per-correct-answer. For narrow classification a 110M encoder")
    print("        usually wins on both latency and cost.")


if __name__ == "__main__":
    main()
