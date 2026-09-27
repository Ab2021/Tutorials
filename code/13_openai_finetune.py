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

The comparison that decides it
------------------------------
A fine-tuned model costs **2x the base model's per-token inference price** on the
`gpt-4.1` family (1.5x on `gpt-4o`). You pay that premium on *every* call, forever. So the
fine-tune only wins if the prompt tokens you can delete are worth more than the markup you
pay on everything else. This script computes that comparison explicitly rather than
showing you a training bill that looks like ₹70 and is actually ₹0.28.

Run it
------
    python 13_openai_finetune.py --data data/sft.jsonl                      # validate + estimate
    python 13_openai_finetune.py --data data/sft.jsonl --validate           # pre-flight only
    python 13_openai_finetune.py --data data/sft.jsonl --estimate --epochs 3
    python 13_openai_finetune.py --data data/sft.jsonl --upload --suffix pharma-v1
    python 13_openai_finetune.py --data data/sft.jsonl --train --file-id file-abc --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix

# --------------------------------------------------------------------------------------
# Prices. USD per 1,000,000 tokens.
#
# **DATED SNAPSHOT.** Every figure below is copied from the handbook's CS-18 case study,
# which dates its own tables. Prices change on the provider's schedule, not ours. Verify
# against the live pricing page before you commit budget, and prefer the provider's own
# estimate endpoint over anything hard-coded here (including this file).
#
#   train     supervised fine-tuning, billed per token PER EPOCH. There is no hourly term
#             for SFT — wall-clock duration does not enter the bill at all.
#   base_in   the untouched base model's input price    ─┐ these two are what decide
#   base_out  the untouched base model's output price   ─┘ whether to tune at all
#   ft_in     the fine-tuned model's input price
#   ft_out    the fine-tuned model's output price
#   ft_cached the fine-tuned input price on a prompt-cache hit
#
# `None` means "not verified for this handbook". The estimator prints `unknown` and
# declines to invent a number — a wrong price is worse than a missing one, because a
# missing price makes you go and check and a wrong one makes you commit.
#
# The original version of this file carried the base price in the `cached` slot for some
# rows and the fine-tuned price in the `input` slot for others, so the recurring bill was
# understated by up to 2x on exactly the model the video demos. Naming the fields is the
# fix; a tuple you have to remember the layout of is a bug waiting to happen.
# --------------------------------------------------------------------------------------
PRICES: dict[str, tuple] = {
    # family          train    base_in  base_out  ft_in   ft_out   ft_cached
    "gpt-4.1-nano":   (1.50,   0.10,    0.40,     0.20,   0.80,    0.10),
    "gpt-4.1-mini":   (3.00,   0.40,    1.60,     0.80,   3.20,    0.40),
    "gpt-4.1":        (None,   2.00,    8.00,     4.00,   16.00,   2.00),
    "gpt-4o-mini":    (3.00,   0.15,    0.60,     0.30,   1.20,    0.15),
    "gpt-4o":         (25.00,  2.50,    10.00,    3.75,   15.00,   1.875),
    "gpt-3.5-turbo":  (8.00,   3.00,    6.00,     None,   None,    None),
}

# Validation limits. Hitting these at upload time is the classic first-run failure.
LIMITS = {
    "max_file_bytes": 512 * 1024 * 1024,
    "max_examples": 50_000,
    # 16385 = 16384 + 1, the 2024 cookbook constant for a 16k-context model family. It is
    # STALE for the gpt-4.1 line-up, where the context is far larger. Kept because it is
    # what the reference implementation uses and truncation here is SILENT — this script
    # checks against it and prints a histogram so you can see the loss either way.
    "max_tokens_per_example": 16_385,
    "min_examples": 10,
    "max_epochs": 50,
    "max_default_epochs": 25,
}

# Per-message overhead. The tokenizer does not bill you for content alone: every message
# carries role/delimiter overhead, and the reply carries priming tokens. On short examples
# this is a fifth of the bill — measured at +19.6% on the video's own dataset. Omitting it
# under-counts, and under-counting is the direction that gets a budget approved and then
# blown.
OVERHEAD_TOKENS_PER_MESSAGE = 3
OVERHEAD_TOKENS_PER_REPLY = 3

# Tokens per word for English. Crude, but stated so you can argue with it.
TOKENS_PER_WORD = 1.33

# The roles a chat training file may legally use. `developer` replaced `system` as the
# higher-precedence instruction role in the newer chat API; a validator that rejects it
# rejects valid files.
VALID_ROLES = ("system", "developer", "user", "assistant", "tool")

# Keys a message object may carry. Anything else is a typo the API will reject at upload —
# better to catch it locally, where the error names your line.
VALID_MESSAGE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id", "refusal")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="jsonl in chat or completion format")
    p.add_argument("--model", default="gpt-4.1-nano-2025-04-14",
                   help="Base model id (the dated snapshot, not the alias)")
    p.add_argument("--suffix", default=None, help="Name for the fine-tuned model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--mode", choices=["chat", "completion"], default="chat")
    p.add_argument("--validate", action="store_true", help="Local pre-flight (no API calls)")
    p.add_argument("--estimate", action="store_true", help="Cost arithmetic, no API calls")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--file-id", default=None,
                   help="Training file id from a prior --upload. REQUIRED for --train: "
                        "guessing 'the newest file on the account' trains the wrong data "
                        "on any account that has uploaded before.")
    p.add_argument("--validation-file-id", default=None,
                   help="Held-out file id. Without one you cannot see overfitting.")
    p.add_argument("--seed", type=int, default=None, help="Makes the run reproducible.")
    p.add_argument("--metadata", default=None,
                   help='JSON object, e.g. \'{"base":"gpt-4.1-nano","data_version":"v7"}\'')
    p.add_argument("--inference-volume", type=int, default=100_000,
                   help="Monthly inference calls, for the cost projection")
    p.add_argument("--avg-output-tokens", type=int, default=250)
    p.add_argument("--ft-prompt-ratio", type=float, default=1.0,
                   help="Fraction of the base prompt the TUNED model still needs. 1.0 "
                        "(default) means 'no prompt shrink', which is the honest "
                        "assumption and makes the comparison say 'the fine-tune loses'.")
    p.add_argument("--cache-hit-rate", type=float, default=0.0,
                   help="Fraction of prompt tokens served from cache (0.0-1.0). The "
                        "single largest inference lever you control; off by default so "
                        "the estimate never flatters you.")
    a = p.parse_args()
    a.metadata = _parse_metadata(a.metadata)
    return a


def _parse_metadata(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"  --metadata is not valid JSON: {e}")
    if not isinstance(obj, dict):
        sys.exit("  --metadata must be a JSON object, e.g. '{\"data_version\":\"v7\"}'")
    return obj


def main() -> None:
    a = parse_args()
    rows = _load(a.data)

    # Each flag does exactly one thing. A bare invocation runs the two safe, free stages.
    acted = False
    if a.validate:
        _validate(rows, a)
        acted = True
    if a.estimate:
        _estimate(rows, a)
        acted = True
    if not acted and not (a.upload or a.train):
        _validate(rows, a)
        _estimate(rows, a)

    if a.upload:
        # Uploading an unvalidated file is one flag away, and a bad training file produces
        # a bad model that you pay for either way. So: never upload without validating,
        # and refuse outright on the problems that are fatal rather than cosmetic.
        blocking = _validate(rows, a, report_only=False)
        if blocking:
            sys.exit(f"  ✋ Refusing to upload: {blocking} blocking problem(s) above. "
                     f"Fix the data, or pass --validate to inspect without uploading.")
        _upload(a)
    if a.train:
        _train(a)


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"  No such file: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------
# Message normalisation
# --------------------------------------------------------------------------------------
def _content_of(m: dict) -> str:
    """
    `str(m.get("content", ""))` renders `content: null` as the four characters "None",
    which is non-empty and therefore passes an emptiness check. A tool-call row with a
    null content is a normal, valid row — it must count as empty content, not as the
    word "None".
    """
    c = m.get("content")
    return "" if c is None else str(c)


def _to_messages(r: dict, mode: str):
    """Accept the shapes people actually have on disk. Returns None if unrecognisable."""
    if not isinstance(r, dict):
        return None
    if mode == "completion":
        if "prompt" in r and "completion" in r:
            return [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]}]
        return None
    if "messages" in r:
        if isinstance(r["messages"], list):
            return r["messages"]
        return None
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


def _shape_diagnosis(r: dict, mode: str) -> str:
    """
    Name the actual problem. The previous version printed `keys=['messages']` — i.e. it
    listed the very key it was implicitly calling missing, which is a worse-than-useless
    diagnostic.
    """
    if not isinstance(r, dict):
        return f"row is a {type(r).__name__}, not an object"
    if mode == "completion":
        return f"need both 'prompt' and 'completion'; got keys={sorted(r)[:6]}"
    if "messages" in r:
        return (f"'messages' is a {type(r['messages']).__name__}, not a list "
                f"(a JSON string of JSON is a common cause)")
    return f"keys={sorted(r)[:6]} match no accepted shape (messages / instruction+output / prompt+completion)"


def _count_tokens(msgs: list[dict]) -> tuple[int, int]:
    """
    Return (content_tokens, overhead_tokens). Overhead is billed but invisible in the
    text, and it is a fixed cost per message — so it hurts short examples most.
    """
    content = sum(len(_content_of(m).split()) for m in msgs)
    overhead = len(msgs) * OVERHEAD_TOKENS_PER_MESSAGE + OVERHEAD_TOKENS_PER_REPLY
    return int(content * TOKENS_PER_WORD), overhead


# --------------------------------------------------------------------------------------
# Validation — the stage that catches the failures that would otherwise cost money
# --------------------------------------------------------------------------------------
def _validate(rows: list[dict], a, report_only: bool = True) -> int:
    """Returns the number of BLOCKING problems (0 = safe to upload)."""
    print(f"\n  ── validating {len(rows)} examples ──")
    problems: dict[str, int] = {}
    blocking: set[str] = set()

    def flag(k: str, detail: str | None = None, fatal: bool = False) -> None:
        problems[k] = problems.get(k, 0) + 1
        if fatal:
            blocking.add(k)
        if detail and problems[k] <= 3:
            print(f"    ⚠  {k}: {detail}")

    if len(rows) < LIMITS["min_examples"]:
        flag("too_few_examples", f"{len(rows)} < {LIMITS['min_examples']}", fatal=True)
    if len(rows) > LIMITS["max_examples"]:
        flag("too_many_examples", f"{len(rows)} > {LIMITS['max_examples']}", fatal=True)

    per_example_tokens: list[int] = []
    tot_content = tot_overhead = 0

    for i, r in enumerate(rows):
        msgs = _to_messages(r, a.mode)
        if msgs is None:
            flag("malformed_shape", f"row {i}: {_shape_diagnosis(r, a.mode)}", fatal=True)
            continue
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            flag("malformed_shape",
                 f"row {i}: 'messages' must be a non-empty list of objects", fatal=True)
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
                role = m.get("role")
                if role not in VALID_ROLES:
                    flag("unknown_role", f"row {i}: {role!r} (valid: {list(VALID_ROLES)})",
                         fatal=True)
                extra = set(m) - set(VALID_MESSAGE_KEYS)
                if extra:
                    flag("message_unrecognized_key",
                         f"row {i}: {sorted(extra)} on the {role!r} message", fatal=True)
                if not _content_of(m).strip() and not m.get("tool_calls"):
                    flag("empty_content", f"row {i}: role={role}")

        c_tok, o_tok = _count_tokens(msgs)
        tot_content += c_tok
        tot_overhead += o_tok
        per_example_tokens.append(c_tok + o_tok)

        # The completion count must come from an actual assistant turn. Taking `msgs[-1]`
        # unconditionally counts a trailing user turn as the completion, which both
        # inflates the token total and points at the wrong message.
        last = msgs[-1]
        if last.get("role") in ("assistant", "tool"):
            words_out = len(_content_of(last).split())
            if words_out < 3:
                flag("very_short_completion", f"row {i}: {words_out} words")

    # Silent truncation: the highest-value silent failure in the whole pipeline. You are
    # billed for every token you sent and the model never sees the tail of the example.
    over = [n for n in per_example_tokens if n > LIMITS["max_tokens_per_example"]]
    if over:
        flag("examples_over_token_cap",
             f"{len(over)} examples exceed {LIMITS['max_tokens_per_example']:,} tokens "
             f"(max {max(over):,}) — these are TRUNCATED, not rejected",
             fatal=False)

    # The near-duplicate check matters more than it looks: the provider's own guidance is
    # that duplicate examples waste epochs and push the model toward memorisation.
    seen: dict[str, int] = {}
    for r in rows:
        msgs = _to_messages(r, a.mode)
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            continue
        key = " ".join(_content_of(m) for m in msgs)[:400].lower()
        key = "".join(ch for ch in key if ch.isalnum())[:200]
        seen[key] = seen.get(key, 0) + 1
    dupes = sum(v - 1 for v in seen.values() if v > 1)
    if dupes:
        flag("duplicate_examples", f"{dupes} duplicate-ish rows")

    raw = Path(a.data).stat().st_size
    if raw > LIMITS["max_file_bytes"]:
        flag("file_too_large",
             f"{raw/1e9:.2f} GB > {LIMITS['max_file_bytes']/1e9:.2f} GB", fatal=True)

    if a.epochs > LIMITS["max_default_epochs"]:
        flag("epochs_above_default_ceiling",
             f"{a.epochs} > {LIMITS['max_default_epochs']}; the auto-epoch policy caps "
             f"here, and beyond ~5 epochs you are usually paying to overfit")

    total = tot_content + tot_overhead
    print(f"\n  examples           {len(rows)}")
    print(f"  file size          {raw/1e6:.2f} MB")
    print(f"  tokens / epoch     {total:,}  (content {tot_content:,} + overhead "
          f"{tot_overhead:,})")
    print(f"  train tokens       {total * a.epochs:,}  ({a.epochs} epochs)")
    if tot_content:
        print(f"  overhead share     {tot_overhead/total*100:.1f}%  ← billed, invisible in "
              f"your text, and worst on short examples")
    if per_example_tokens:
        s = sorted(per_example_tokens)
        print(f"  tokens / example   p50 {s[len(s)//2]:,}   p95 {s[int(len(s)*0.95)]:,}   "
              f"max {s[-1]:,}")

    if problems:
        print(f"\n  ⚠  {sum(problems.values())} issue(s): "
              f"{dict(sorted(problems.items(), key=lambda kv: -kv[1]))}")
        if blocking:
            print(f"     ✋ BLOCKING (these make the file invalid, not just suboptimal): "
                  f"{sorted(blocking)}")
        print("     Fix these BEFORE uploading. A bad training file produces a bad model")
        print("     and you pay for the run either way — the API does not refund a model")
        print("     that learned from malformed data.")
    else:
        print("\n  ✅ no structural problems found")
    print("\n  ⚠  Structural validation is not quality validation. Nothing above can tell")
    print("     you whether the ASSISTANT ANSWERS ARE GOOD. Read 20 rows yourself.")
    print("     Fine-tuning on mediocre completions reliably produces a mediocre model.")

    return len(blocking) if not report_only else 0


# --------------------------------------------------------------------------------------
# Cost — the part that decides whether this is a good idea at all
# --------------------------------------------------------------------------------------
def resolve_price(model: str) -> str | None:
    """
    Map a model id to a price row. Longest match wins.

    The naive test — `next(k for k in PRICES if k in model)` — is decided by *dict
    insertion order*, so `gpt-4.1` (listed before `gpt-4.1-mini`) shadows
    `gpt-4.1-mini-2025-04-14` and `gpt-4.1-nano-2025-04-14` resolves to $25.00/M train
    and $4.00/M input: a 16.7x overestimate on the model the video actually demos, with
    no warning. Sort by length and take the longest hit.
    """
    hay = model.lower()
    hits = [k for k in PRICES if hay.startswith(k)]
    if not hits:
        # Tolerate a vendored prefix or a decorated id, but still take the longest hit.
        hits = [k for k in PRICES
                if re.search(rf"(?<![a-z0-9.]){re.escape(k)}(?![a-z0-9])", hay)]
    return max(hits, key=len) if hits else None


def _estimate(rows: list[dict], a) -> None:
    key = resolve_price(a.model)
    print(f"\n  ── cost estimate (--model {a.model}) ──")
    if key is None:
        print(f"  ⚠  No price on file for '{a.model}'. Known families: {sorted(PRICES)}")
        print("     Add it to PRICES, or check the current pricing page — costs change.")
        return
    if not a.model.lower().startswith(key):
        print(f"  ⚠  '{a.model}' matched '{key}' by substring, not by prefix. Confirm this "
              f"is the family you meant.")
    train_usd, base_in, base_out, ft_in, ft_out, ft_cached = PRICES[key]

    tot_content = tot_overhead = 0
    for r in rows:
        msgs = _to_messages(r, a.mode)
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            continue
        c, o = _count_tokens(msgs)
        tot_content += c
        tot_overhead += o
    billed = tot_content + tot_overhead
    if not billed:
        print("  ⚠  No usable rows — nothing to estimate.")
        return

    print(f"  price row          {key}")
    print(f"  training           {_fmt(train_usd)} /1M tokens (per epoch)")
    print(f"  base inference     in {_fmt(base_in)}  out {_fmt(base_out)} /1M")
    print(f"  tuned inference    in {_fmt(ft_in)}  out {_fmt(ft_out)} /1M")

    # ---- Training: a one-off ----------------------------------------------------------
    print(f"\n  TRAINING (one-off)")
    if train_usd is None:
        print(f"    tokens           {billed * a.epochs:,}  ({a.epochs} epochs)")
        print(f"    cost             unknown — no verified training price for {key}.")
        print(f"                     Check the pricing page. Not guessing: a missing price")
        print(f"                     makes you look it up, a wrong one makes you commit.")
        train_cost = None
    else:
        train_tokens = billed * a.epochs
        train_cost = train_tokens / 1e6 * train_usd
        print(f"    tokens           {train_tokens:,}  ({a.epochs} epochs x {billed:,})")
        print(f"    cost             ${train_cost:,.2f}")
        print(f"                     = {train_tokens:,} / 1e6 x ${train_usd}")

    # ---- Inference: the recurring term, and usually the larger one ---------------------
    if ft_in is None or ft_out is None:
        print(f"\n  INFERENCE — no verified fine-tuned inference price for {key}. Stopping.")
        return

    calls = a.inference_volume
    avg_prompt = billed / max(len(rows), 1)
    calls_per_example = avg_prompt
    monthly_prompt_tokens = calls * calls_per_example
    monthly_out_tokens = calls * a.avg_output_tokens

    cache = max(0.0, min(1.0, a.cache_hit_rate))
    billed_in = monthly_prompt_tokens * (1 - cache)
    cached_in = monthly_prompt_tokens * cache
    ft_cached_rate = ft_cached if ft_cached is not None else ft_in

    ft_monthly = (billed_in / 1e6 * ft_in
                  + cached_in / 1e6 * ft_cached_rate
                  + monthly_out_tokens / 1e6 * ft_out)
    base_monthly = (monthly_prompt_tokens / 1e6 * base_in
                    + monthly_out_tokens / 1e6 * base_out)

    print(f"\n  INFERENCE (recurring, at {calls:,} calls/month)")
    print(f"    prompt tokens    {monthly_prompt_tokens:,.0f}  "
          f"(avg {calls_per_example:,.0f}/call)")
    print(f"    output tokens    {monthly_out_tokens:,.0f}  "
          f"(@ {a.avg_output_tokens}/call)")
    if cache:
        print(f"    cache hit rate   {cache*100:.0f}%  "
              f"→ {cached_in:,.0f} tokens at the cached rate")
    print(f"    TUNED   monthly  ${ft_monthly:,.2f}   annual ${ft_monthly*12:,.2f}")
    print(f"    BASE    monthly  ${base_monthly:,.2f}   annual ${base_monthly*12:,.2f}")
    print(f"    premium          ${ft_monthly - base_monthly:,.2f}/month  "
          f"({(ft_monthly/base_monthly - 1)*100:+.0f}%)")

    # ---- The comparison that actually decides it ---------------------------------------
    print(f"\n  ── the decision ──")
    savings = base_monthly - ft_monthly
    # A tuned model usually lets you DELETE prompt tokens: that is the whole economic case
    # for style/format tuning. --ft-prompt-ratio models how many the tuned model still
    # needs. At the default 1.0 no prompt is deleted, and the fine-tune always loses —
    # which is the correct answer for a dataset that is not mostly few-shot examples.
    pf_ratio = a.ft_prompt_ratio
    tuned_prompt = monthly_prompt_tokens * pf_ratio
    ft_monthly_shrunk = (tuned_prompt / 1e6 * ft_in
                         + monthly_out_tokens / 1e6 * ft_out)
    real_savings = base_monthly - ft_monthly_shrunk
    print(f"    At --ft-prompt-ratio {pf_ratio:g}, the tuned model sends "
          f"{tuned_prompt:,.0f} prompt tokens")
    print(f"    (vs {monthly_prompt_tokens:,.0f} for the base). Monthly: base "
          f"${base_monthly:,.2f} vs tuned ${ft_monthly_shrunk:,.2f}")
    if real_savings > 0:
        if train_cost is None:
            print(f"    → The tuned model SAVES ${real_savings:,.2f}/month. Training cost "
                  f"unknown, so payback cannot be computed.")
        else:
            payback = train_cost / real_savings
            print(f"    → The tuned model SAVES ${real_savings:,.2f}/month. Training "
                  f"(${train_cost:,.2f}) pays back in")
            print(f"      {payback:.1f} months ({payback/12:.1f} years) of inference.")
    else:
        print(f"    ❌ The tuned model COSTS ${-real_savings:,.2f}/month MORE than the "
              f"base model at this prompt ratio. It never pays back.")
        # Deleting prompt tokens is the only lever, so show how many would be needed.
        # Solve  tuned_prompt*ft_in + out*ft_out  <  monthly_prompt*base_in + out*base_out
        budget = (monthly_prompt_tokens / 1e6 * base_in
                  + monthly_out_tokens / 1e6 * base_out
                  - monthly_out_tokens / 1e6 * ft_out)
        need = budget / (ft_in / 1e6) if ft_in else 0.0
        if need <= 0:
            print(f"       The output-token markup alone (${ft_out} vs ${base_out} /1M) "
                  f"exceeds the base")
            print(f"       model's entire bill. No prompt is short enough — this fine-tune "
                  f"cannot win.")
        else:
            print(f"       To break even you would need to cut the prompt from "
                  f"{monthly_prompt_tokens:,.0f}")
            print(f"       to {need:,.0f} tokens/call "
                  f"({need/monthly_prompt_tokens*100:.0f}% of current). "
                  f"Run with --ft-prompt-ratio")
            print(f"       {need/monthly_prompt_tokens:.3f} to model that scenario.")

    print(f"\n  TOTALS")
    if train_cost is not None:
        print(f"    month 1          ${train_cost + ft_monthly_shrunk:,.2f}")
        print(f"    year 1           ${train_cost + ft_monthly_shrunk*12:,.2f}")
        year = train_cost + ft_monthly_shrunk * 12
        if year:
            print(f"    training is      {train_cost/year*100:.1f}% of the year-1 bill")
        print(f"    base year 1      ${base_monthly*12:,.2f}  (no training cost, so this "
              f"is the bar the tuned model must beat)")

    print("\n  ── read this before committing ──")
    print("    * The training cost is a ONE-OFF. The inference premium is FOREVER. At")
    print(f"      {calls:,} calls/month the recurring term dominates — a hosted tuned")
    print("      endpoint carries a markup over the base model on every single call.")
    print("    * The fine-tune wins ONLY if the prompt you can delete is worth more than")
    print("      the markup you pay on everything else. Run --ft-prompt-ratio against your")
    print("      real prompt lengths before you believe any training bill.")
    print("    * Compare against self-hosting an open model of the same size. A 7B on a")
    print("      rented GPU is a fixed hourly cost regardless of volume; at high volume")
    print("      that usually wins, and it also lets you quantize and distil.")
    print("    * More epochs is not more better. 3 is the common starting point; beyond")
    print("      ~5 you are usually paying to overfit.")
    print("    * Prices quoted here are a DATED SNAPSHOT. Verify before you budget.")


def _fmt(v) -> str:
    return "unknown" if v is None else f"${v:g}"


# --------------------------------------------------------------------------------------
def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("  Set OPENAI_API_KEY first.")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("  pip install openai")
    return OpenAI()


def _upload(a) -> None:
    client = _client()

    print(f"\n  uploading {a.data}...")
    with open(a.data, "rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"  ✅ file id {resp.id}")
    print("\n  ℹ  The API validates on upload and reports per-line errors with line numbers.")
    print("     Line numbers refer to YOUR file — fix the data and re-upload rather than")
    print("     dropping the offending rows, unless they are genuinely junk.")
    print(f"\n  ⚠  This id is NOT remembered. Pass it explicitly to the training step —")
    print(f"     otherwise you will train on whatever file happens to be newest on the")
    print(f"     account, which on a shared account is somebody else's dataset.")
    print(f"\n     python 13_openai_finetune.py --data {a.data} --train \\")
    print(f"         --file-id {resp.id} --model {a.model}"
          + (f" --suffix {a.suffix}" if a.suffix else "")
          + (f" --validation-file-id <held-out-file-id>" if a.validation_file_id else ""))


def _train(a) -> None:
    client = _client()

    if not a.file_id:
        # Refuse to guess. Listing what IS available is more useful than a bare error.
        try:
            files = client.files.list(purpose="fine-tune")
            avail = [(f.id, getattr(f, "created_at", None), getattr(f, "filename", "?"))
                     for f in files.data][:10]
        except Exception as e:                                  # noqa: BLE001
            avail, _ = [], e
        sys.exit(
            "  --file-id is required. Without it this would train on files.data[0] — the\n"
            "  newest file on the account, which is not necessarily the one you just\n"
            "  uploaded and validated.\n"
            + (f"  Recent fine-tune files:\n" +
               "".join(f"    {i}  {c}  {n}\n" for i, c, n in avail)
               if avail else f"  (could not list files: {avail})\n")
            + f"  Re-run with:  --file-id <id>")

    kwargs = dict(
        training_file=a.file_id,
        model=a.model,
        hyperparameters={
            "n_epochs": a.epochs,
            # Both of these are what the auto-epoch policy actually uses. Set them and you
            # get the policy; omit them and you get your raw --epochs. Leaving validation
            # out entirely is how an overfitting run goes undetected.
            "batch_size": "auto",
            "learning_rate_multiplier": "auto",
        },
    )
    if a.suffix:
        kwargs["suffix"] = a.suffix
    if a.validation_file_id:
        kwargs["validation_file"] = a.validation_file_id
    if a.seed is not None:
        kwargs["seed"] = a.seed
    if a.metadata:
        kwargs["metadata"] = a.metadata

    job = _create_job(client, kwargs)
    print(f"  ✅ job {job.id} created (status {job.status})")
    if a.validation_file_id:
        print(f"     validation file {a.validation_file_id} → you will get a "
              f"validation_loss curve")
    else:
        print("     ⚠  No --validation-file-id. You will get no validation loss, so you")
        print("        cannot see overfitting — only that training loss went down, which")
        print("        it always does.")
    if a.seed is None:
        print("     ⚠  No --seed. This run is not reproducible.")
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


def _create_job(client, kwargs: dict):
    """
    The supervised fine-tuning request shape changed within the openai 2.x line: the flat
    `hyperparameters=` argument was superseded by a nested
    `method={"type": "supervised", "supervised": {"hyperparameters": {...}}}`. Rather than
    pin one and break on the other, try the current shape and fall back. The SDK version
    you have installed decides which one this machine needs; the log tells you which.
    """
    hparams = kwargs.pop("hyperparameters", {})
    try:
        return client.fine_tuning.jobs.create(
            method={"type": "supervised", "supervised": {"hyperparameters": hparams}},
            **kwargs,
        )
    except TypeError as e:
        # Older SDK: `method` is not a parameter.
        print(f"  ℹ  nested `method=` not accepted by this SDK ({e}); using the flat shape")
        return client.fine_tuning.jobs.create(hyperparameters=hparams, **kwargs)


if __name__ == "__main__":
    main()
