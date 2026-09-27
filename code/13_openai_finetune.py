#!/usr/bin/env python
"""
13_openai_finetune.py — fine-tune a hosted OpenAI model, with the cost arithmetic done
before you spend anything.

What is fundamentally different from open-weight fine-tuning
------------------------------------------------------------
You do not get weights. You get an endpoint name. That single fact drives everything:

  * You cannot merge, quantize, prune, distill, or serve it yourself.
  * You cannot run it offline, on-prem, or in a VPC you control.
  * You cannot inspect what changed. There is no loss curve you can fully trust and no
    way to diff the weights against the base.
  * You pay per token, forever. Training cost is a one-off; INFERENCE cost recurs. At
    volume the endpoint bill dwarfs the training bill, and this is the number people
    forget to model.
  * You cannot undo it, but you can stop using it — the base model remains available.

So the decision is not "is fine-tuning better than prompting". It is: *does the behaviour
need to be baked in, and is the ongoing per-token premium worth it* versus a smaller
open model you host yourself.

The three jobs fine-tuning is genuinely good at
-----------------------------------------------
  1. **Style and format compliance** that prompting cannot pin down reliably.
  2. **Narrow classification / extraction** where a smaller tuned model beats a larger
     prompted one on cost-per-correct-answer.
  3. **Distilling a large model's behaviour into a cheaper one** (generate with the big
     model, fine-tune the small one on its outputs).

The job it is usually the WRONG tool for
----------------------------------------
**Teaching the model facts.** Retrieval does that better, cheaper, and updatably. If your
problem is "the model doesn't know about our product", fine-tuning is a slow, expensive,
non-updatable answer. And if you only need guaranteed JSON *shape*, use structured
outputs / a schema — that is a decoding constraint, not a weight change.

Run it
------
    python 13_openai_finetune.py --data data/sft.jsonl --validate
    python 13_openai_finetune.py --data data/sft.jsonl --estimate --epochs 3
    python 13_openai_finetune.py --data data/sft.jsonl --upload --suffix pharma-v1
    python 13_openai_finetune.py --data data/sft.jsonl --train --model gpt-4o-mini-2024-07-18
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix

# --------------------------------------------------------------------------------------
# Prices. These are the figures the API charges and they are also the figures that go
# stale fastest. Treat every number here as "verify before you commit budget", and prefer
# --estimate output over any hard-coded figure you read in a blog post (including this doc).
# --------------------------------------------------------------------------------------
# USD per 1M tokens. Training is billed per token PER EPOCH. Inference is billed on both
# input and output tokens, every call, forever.
PRICES = {
    # model family            train        input     cached_in   output
    "gpt-4o-mini":         (3.00,        0.30,     0.15,       1.20),
    "gpt-4o":              (25.00,       3.75,     1.875,      15.00),
    "gpt-4.1-mini":        (3.00,        0.40,     0.20,       1.60),
    "gpt-4.1":             (25.00,       2.00,     1.00,       8.00),
    "gpt-3.5-turbo":       (8.00,        3.00,     1.50,       6.00),
}

# Validation limits. Hitting these at upload time is the classic first-run failure.
LIMITS = {
    "max_file_bytes": 512 * 1024 * 1024,
    "max_examples": 50_000,
    "max_tokens_per_example": 16_384,   # context-dependent; verify for your base model
    "min_examples": 10,
    "max_epochs": 50,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="jsonl in chat or completion format")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18",
                   help="Base model id (the dated snapshot, not the alias)")
    p.add_argument("--suffix", default=None, help="Name for the fine-tuned model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--mode", choices=["chat", "completion"], default="chat")
    p.add_argument("--validate", action="store_true", help="Local pre-flight (no API calls)")
    p.add_argument("--estimate", action="store_true", help="Cost arithmetic, no API calls")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--inference-volume", type=int, default=100_000,
                   help="Monthly inference calls, for the cost projection")
    p.add_argument("--avg-output-tokens", type=int, default=250)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = _load(a.data)

    if a.validate or not (a.upload or a.train or a.estimate):
        _validate(rows, a)
    if a.estimate or not (a.upload or a.train):
        _estimate(rows, a)
    if a.upload:
        _upload(a)
    if a.train:
        _train(a)


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"  No such file: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------
# Validation — the stage that catches the failures that would otherwise cost money
# --------------------------------------------------------------------------------------
def _validate(rows: list[dict], a) -> None:
    print(f"\n  ── validating {len(rows)} examples ──")
    problems: dict[str, int] = {}

    def flag(k: str, detail: str | None = None) -> None:
        problems[k] = problems.get(k, 0) + 1
        if detail and problems[k] <= 3:
            print(f"    ⚠  {k}: {detail}")

    if len(rows) < LIMITS["min_examples"]:
        flag("too_few_examples", f"{len(rows)} < {LIMITS['min_examples']}")
    if len(rows) > LIMITS["max_examples"]:
        flag("too_many_examples", f"{len(rows)} > {LIMITS['max_examples']}")

    tot_in = tot_out = 0
    for i, r in enumerate(rows):
        msgs = _to_messages(r, a.mode)
        if msgs is None:
            flag("malformed_shape", f"row {i}: keys={sorted(r)[:6]}")
            continue

        if a.mode == "chat":
            roles = [m.get("role") for m in msgs]
            if "assistant" not in roles:
                flag("no_assistant_message", f"row {i}: roles={roles}")
            if roles[-1] != "assistant":
                # Legal for some training modes, but it means the sample has no
                # completion to learn from under the default chat behaviour.
                flag("last_message_not_assistant", f"row {i}: roles={roles[-2:]}")
            for m in msgs:
                if m.get("role") not in ("system", "user", "assistant", "tool"):
                    flag("unknown_role", f"row {i}: {m.get('role')!r}")
                if not str(m.get("content", "")).strip():
                    flag("empty_content", f"row {i}: role={m.get('role')}")

        words_in = sum(len(str(m.get("content", "")).split()) for m in msgs[:-1])
        words_out = len(str(msgs[-1].get("content", "")).split())
        tot_in += words_in
        tot_out += words_out
        if words_out < 3:
            flag("very_short_completion", f"row {i}: {words_out} words")

    # The near-duplicate check matters more than it looks: OpenAI's own guidance is that
    # duplicate examples waste epochs and push the model toward memorisation.
    seen: dict[str, int] = {}
    for r in rows:
        msgs = _to_messages(r, a.mode)
        if not msgs:
            continue
        key = " ".join(str(m.get("content", "")) for m in msgs)[:400].lower()
        key = "".join(ch for ch in key if ch.isalnum())[:200]
        seen[key] = seen.get(key, 0) + 1
    dupes = sum(v - 1 for v in seen.values() if v > 1)
    if dupes:
        flag("duplicate_examples", f"{dupes} duplicate-ish rows")

    raw = Path(a.data).stat().st_size
    if raw > LIMITS["max_file_bytes"]:
        flag("file_too_large", f"{raw/1e9:.2f} GB > {LIMITS['max_file_bytes']/1e9:.2f} GB")

    # Rough token estimate: ~1.33 tokens per word for English.
    est_in = int(tot_in * 1.33)
    est_out = int(tot_out * 1.33)
    est_train = int((est_in + est_out) * a.epochs)

    print(f"\n  examples           {len(rows)}")
    print(f"  file size          {raw/1e6:.2f} MB")
    print(f"  est. tokens/epoch  {est_in + est_out:,}  (in {est_in:,} / out {est_out:,})")
    print(f"  est. train tokens  {est_train:,}  ({a.epochs} epochs)")

    if problems:
        print(f"\n  ⚠  {sum(problems.values())} issue(s): "
              f"{dict(sorted(problems.items(), key=lambda kv: -kv[1]))}")
        print("     Fix these BEFORE uploading. A bad training file produces a bad model")
        print("     and you pay for the run either way — the API does not refund a model")
        print("     that learned from malformed data.")
    else:
        print("\n  ✅ no structural problems found")
    print("\n  ⚠  Structural validation is not quality validation. Nothing above can tell")
    print("     you whether the ASSISTANT ANSWERS ARE GOOD. Read 20 rows yourself.")
    print("     Fine-tuning on mediocre completions reliably produces a mediocre model.")


def _to_messages(r: dict, mode: str):
    """Accept the shapes people actually have on disk."""
    if mode == "completion":
        if "prompt" in r and "completion" in r:
            return [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]}]
        return None
    if "messages" in r and isinstance(r["messages"], list):
        return r["messages"]
    if "instruction" in r and "output" in r:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        ins = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        msgs += [{"role": "user", "content": ins},
                 {"role": "assistant", "content": r["output"]}]
        return msgs
    if "prompt" in r and "completion" in r:
        return [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]
    return None


# --------------------------------------------------------------------------------------
# Cost — the part that decides whether this is a good idea at all
# --------------------------------------------------------------------------------------
def _estimate(rows: list[dict], a) -> None:
    key = next((k for k in PRICES if k in a.model), None)
    if key is None:
        print(f"\n  ⚠  No price on file for '{a.model}'. Known families: {list(PRICES)}")
        print("     Add it to PRICES, or check the current pricing page — costs change.")
        return
    train_usd, in_usd, _cached, out_usd = PRICES[key]

    tot_in = tot_out = 0
    for r in rows:
        msgs = _to_messages(r, a.mode)
        if not msgs:
            continue
        tot_in += sum(len(str(m.get("content", "")).split()) for m in msgs[:-1])
        tot_out += len(str(msgs[-1].get("content", "")).split())
    est_in, est_out = int(tot_in * 1.33), int(tot_out * 1.33)

    print(f"\n  ── cost estimate ({key}) ──")
    print(f"  price              train ${train_usd}/1M  in ${in_usd}/1M  out ${out_usd}/1M")

    # Training is billed on ALL tokens per epoch — input and output alike.
    train_tokens = (est_in + est_out) * a.epochs
    train_cost = train_tokens / 1e6 * train_usd
    print(f"\n  TRAINING (one-off)")
    print(f"    tokens           {train_tokens:,}  ({a.epochs} epochs x "
          f"{est_in + est_out:,})")
    print(f"    cost             ${train_cost:,.2f}")

    # Inference — the recurring term, and usually the larger one.
    calls = a.inference_volume
    inf_in = int(calls * est_in / max(len(rows), 1))
    inf_out = int(calls * a.avg_output_tokens)
    inf_cost = inf_in / 1e6 * in_usd + inf_out / 1e6 * out_usd
    print(f"\n  INFERENCE (recurring, at {calls:,} calls/month)")
    print(f"    input tokens     {inf_in:,}  → ${inf_in/1e6*in_usd:,.2f}")
    print(f"    output tokens    {inf_out:,}  → ${inf_out/1e6*out_usd:,.2f}")
    print(f"    monthly          ${inf_cost:,.2f}")
    print(f"    annualised       ${inf_cost*12:,.2f}")

    months = (inf_cost and train_cost / inf_cost) or 0
    print(f"\n  TOTALS")
    print(f"    month 1          ${train_cost + inf_cost:,.2f}")
    print(f"    year 1           ${train_cost + inf_cost*12:,.2f}")
    print(f"    training is      {train_cost/(train_cost+inf_cost*12)*100:.1f}% of the "
          f"year-1 bill")
    print(f"    → training pays back in {months:.1f} months of inference")

    print("\n  ── read this before committing ──")
    print(f"    * The training cost is a ONE-OFF. The inference cost is FOREVER. At")
    print(f"      {calls:,} calls/month the recurring term dominates — a hosted tuned")
    print(f"      endpoint carries a premium over the base model on every single call.")
    print("    * Compare against self-hosting an open model of the same size. A 7B on a")
    print("      rented GPU is a fixed hourly cost regardless of volume; at high volume")
    print("      that usually wins, and it also lets you quantize and distil.")
    print("    * More epochs is not more better. 3 epochs is the common starting point;")
    print("      beyond ~5 you are usually paying to overfit.")
    print("    * Prices quoted here will go stale. Verify before you budget.")


# --------------------------------------------------------------------------------------
def _upload(a) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("  Set OPENAI_API_KEY first.")
    from openai import OpenAI
    client = OpenAI()

    print(f"\n  uploading {a.data}...")
    with open(a.data, "rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"  ✅ file id {resp.id}")
    print("\n  ℹ  The API validates on upload and reports per-line errors with line numbers.")
    print("     Line numbers refer to YOUR file — fix the data and re-upload rather than")
    print("     dropping the offending rows, unless they are genuinely junk.")
    print(f"\n     python 13_openai_finetune.py --data {a.data} --train "
          f"--model {a.model}" + (f" --suffix {a.suffix}" if a.suffix else ""))


def _train(a) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("  Set OPENAI_API_KEY first.")
    from openai import OpenAI
    client = OpenAI()

    files = client.files.list(purpose="fine-tune")
    fid = files.data[0].id if files.data else None
    if fid is None:
        sys.exit("  No uploaded fine-tune file found. Run --upload first.")
    print(f"  using file {fid}")

    job = client.fine_tuning.jobs.create(
        training_file=fid,
        model=a.model,
        suffix=a.suffix,
        hyperparameters={"n_epochs": a.epochs},
    )
    print(f"  ✅ job {job.id} created (status {job.status})")
    print(f"\n     Watch it:  python -c \"from openai import OpenAI; "
          f"print(OpenAI().fine_tuning.jobs.retrieve('{job.id}').status)\"")
    print("\n  ── what to watch in the returned metrics ──")
    print("    * training_loss falling while validation_loss RISES = overfitting. Stop")
    print("      early; more epochs will make it worse, not better.")
    print("    * training_token_accuracy near 1.0 on a small dataset usually means the")
    print("      model memorised the training set, not that it learned the task.")
    print("    * Both flat? The task is probably too hard for this base model, or the")
    print("      data is inconsistent (the same input mapped to different outputs).")
    print("\n  ⚠  A fine-tuned model is a NEW model id. Your old prompt may need adjusting:")
    print("     models sometimes follow instructions slightly differently after tuning.")
    print("     Re-run your eval suite against both the base and the tuned model.")


if __name__ == "__main__":
    main()
