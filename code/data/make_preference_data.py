#!/usr/bin/env python
"""
data/make_preference_data.py — build a (prompt, chosen, rejected) preference dataset.

How preference data is actually made
------------------------------------
Three routes, in increasing order of cost and quality:

  1. **Model-vs-model** (`--mode model-vs-model`): sample two responses from different
     models (or the same model at different temperatures), have a judge pick a winner.
     This is how UltraFeedback and most open preference sets were built. Cheap, scales,
     and inherits the judge's biases.

  2. **Corrupted/edited** (`--mode corrupt`): take one good response and programmatically
     degrade it (drop the reasoning, truncate, add an error, break the format). The
     original is `chosen`, the corrupted is `rejected`. This is the best option when you
     have a curated set of good answers — the signal is clean and unambiguous.

  3. **Human annotation** (`--mode template`): the gold standard, ~$1-5 per pair, and the
     only option for genuinely subjective qualities (tone, helpfulness, safety).

The mistake that ruins DPO runs
-------------------------------
**Length bias.** Every preference pair teaches two things at once — "this CONTENT is
better" and "this LENGTH is better". If the lengths differ systematically, the model
takes the shortcut and learns the length rule instead of the quality one.

Both directions are failures, and the second is the one people miss:

  * `chosen` longer  → the model learns "longer = better" and pads every answer
    forever. This is the classic DPO pathology.
  * `rejected` longer → the model learns "shorter = better" and under-answers on
    questions that genuinely need a long response. This one is easy to miss because
    the average looks fine.

`--mode corrupt` is especially exposed, because several corruptions change length as a
side effect: on the seed set, `verbose` inflates by +161% and `hedge` by +92%, while
`truncate` shrinks by -62%. Only `wrong` (+4%) is close to neutral. So the default
`--match-lengths` rebalances each corrupted response back to its chosen's length —
trimming padding off the long ones and adding content-free filler to the short ones.
That turns `truncate` into a more realistic corruption than it was: not short, but
long-winded and uninformative.

`--only-neutral` is the belt-and-braces version: it DROPS pairs that remain
imbalanced rather than fixing them. With `--match-lengths` on (the default) it is
usually a no-op, which is the intended end state. Run it with `--no-match-lengths` to
see how much of your raw corruption signal is really just length.

Fix the DATA, not the loss function.

Run it
------
    python data/make_preference_data.py --template --out data/sample_preference.jsonl
    python data/make_preference_data.py --mode corrupt --sft data/sample_sft.jsonl --out data/pp.jsonl
    python data/make_preference_data.py --mode corrupt --sft data/sft.jsonl --no-match-lengths --only-neutral --out data/pp.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: F401  — applies the UTF-8 console fix on Windows (see common/__init__.py)


# --------------------------------------------------------------------------------------
# Corruptions — each one models a REAL failure mode you want the model to learn to avoid
# --------------------------------------------------------------------------------------
def corrupt_truncate(text: str) -> str:
    """Cut the answer off mid-thought. Teaches completeness."""
    cut = max(5, int(len(text.split()) * 0.4))
    return " ".join(text.split()[:cut]) + "..."


def corrupt_drop_reasoning(text: str) -> str:
    """Keep only the conclusion, dropping the reasoning. Teaches 'show your work'."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sents) <= 2:
        return text
    return " ".join(sents[-1:])


def corrupt_hallucinate(text: str) -> str:
    """
    Confidently assert a fabricated specific. Teaches the model to avoid invented
    numbers — this is the highest-value corruption for a production assistant.
    """
    fake = ["approximately 73.2%", "in 1987", "roughly 4.7 million", "by a factor of 12"]
    return text + f" This is supported by data showing {random.choice(fake)} of cases."


def corrupt_hedge(text: str) -> str:
    """Bury the answer in hedging. Teaches directness."""
    return ("It depends on a lot of factors and I really can't say for certain, but "
            "possibly, in some cases, one might consider that " + text[0].lower() + text[1:])


def corrupt_verbose(text: str) -> str:
    """Pad without adding information. Teaches concision."""
    return ("That's a really great and important question, and it's one that many people "
            "ask. There are several things to consider here. " + text +
            " I hope this helps! Let me know if you'd like me to elaborate further on "
            "any aspect of this.")


def corrupt_format(text: str) -> str:
    """Wrap the answer in chatty prose. Teaches exact-format compliance."""
    return f"Sure thing! Here you go:\n\n{text}\n\nLet me know if you need anything else!"


def corrupt_wrong(text: str) -> str:
    """Negate the key claim. Teaches factual accuracy (use sparingly — risks teaching
    the model that fluent-but-wrong is common)."""
    swaps = [("increases", "decreases"), ("inhibits", "activates"), ("safe", "unsafe"),
             ("should", "should not"), ("is", "is not")]
    for a, b in swaps:
        if a in text:
            return text.replace(a, b, 1)
    return text


FILLER = [
    "As is well known, this is an important consideration in this area.",
    "There are several factors to consider here, and context matters a great deal.",
    "This is a topic that many people have written about at considerable length.",
    "It is worth bearing in mind that individual circumstances vary widely.",
    "Broadly speaking, the general principles apply in the majority of situations.",
]


def _rebalance(rejected: str, target_words: int, tol: float = 0.20) -> str:
    """
    Pull a corrupted response back to roughly the length of its `chosen`, so the pair
    teaches a QUALITY distinction rather than a LENGTH one.

    Both directions matter, and the second is the one people forget:
      * Too long  (verbose / format / hedge / hallucinate) — trim the added padding off,
        preferring a sentence boundary so the text still reads naturally.
      * Too short (truncate / drop_reasoning) — pad with content-free filler. This turns
        them into a MORE realistic corruption than the original: long-winded but
        uninformative, which is exactly how a padded model answer fails in production.

    The point is not cosmetic. A model trained on length-imbalanced pairs learns the
    length shortcut and stops attending to content.
    """
    w = rejected.split()
    lo, hi = target_words * (1 - tol), target_words * (1 + tol)
    if lo <= len(w) <= hi:
        return rejected

    if len(w) > hi:
        # Trim, but break at a sentence boundary if one falls inside the band.
        sents = re.split(r"(?<=[.!?])\s+", " ".join(w[: int(hi * 1.3)]))
        out: list[str] = []
        for s in sents:
            if len(" ".join(out + [s]).split()) > hi:
                break
            out.append(s)
        return " ".join(out) if len(" ".join(out).split()) >= lo else " ".join(w[: int(hi)])

    i = 0
    while len(w) < lo:
        w.extend(FILLER[i % len(FILLER)].split())
        i += 1
    return " ".join(w)


CORRUPTIONS = {
    "truncate": corrupt_truncate,
    "drop_reasoning": corrupt_drop_reasoning,
    "hallucinate": corrupt_hallucinate,
    "hedge": corrupt_hedge,
    "verbose": corrupt_verbose,
    "format": corrupt_format,
    "wrong": corrupt_wrong,
}

# Why this matters: a preference pair teaches TWO things at once — "this content is
# better" AND "this LENGTH is better". If the corruption changes the length, the model
# may learn the length shortcut instead of the quality distinction you intended.
#
# Do NOT guess which corruptions are length-neutral — measure. On the template seed set
# the observed deltas are roughly: verbose +161%, hedge +92%, format +48%, hallucinate
# +40%, wrong +4%, truncate -62%, drop_reasoning -67%. Only `wrong` is close to neutral.
# `--only-neutral` therefore filters on the MEASURED per-pair delta, not on this table.
LENGTH_TOLERANCE_PCT = 25.0


# --------------------------------------------------------------------------------------
# Template mode
# --------------------------------------------------------------------------------------
# NOTE ON CONSTRUCTION: chosen and rejected are deliberately of SIMILAR LENGTH, and the
# rejected answer is plausible and fluent rather than obviously bad. This is the realistic
# case and the useful one — the model must learn a quality distinction, not "longer wins".
# Most real preference datasets fail here: chosen is systematically longer, and the model
# learns to pad. If your own data has a large length gap, fix the DATA before training.
TEMPLATE_PAIRS = [
    ("My 2-year-old has had a fever of 38.5°C for one day. Should I be worried?",
     "A fever of 38.5°C lasting one day is common in young children and usually reflects "
     "the body responding to a mild infection. Judge by how the child looks and behaves "
     "rather than the number alone: if they are drinking, producing wet nappies and "
     "responding to you, monitoring at home is reasonable. Seek same-day review if the "
     "child is floppy or hard to wake, has a rash that does not fade under pressure, is "
     "breathing fast, or the fever persists beyond three days.",
     "A fever of 38.5°C for a single day is not usually something to worry about, since "
     "children run fevers often and most are caused by minor viral illnesses that resolve "
     "on their own. Keep them comfortable, make sure they are drinking enough, and give "
     "paracetamol if they seem distressed. Most parents find it settles within a day or "
     "two without any specific treatment being needed."),

    ("Explain why my cholesterol medication is taken in the evening.",
     "Many statins work by blocking HMG-CoA reductase, the rate-limiting enzyme in "
     "cholesterol synthesis, and the body synthesises most of its cholesterol overnight. "
     "Taking the dose in the evening therefore puts peak drug levels where enzyme "
     "activity is highest. This matters most for short-acting statins such as "
     "simvastatin; long half-life agents like atorvastatin and rosuvastatin are far less "
     "time-sensitive, so for those, consistency matters more than the hour.",
     "Statins are usually taken in the evening because that is the standard advice and "
     "the way they have traditionally been prescribed. Cholesterol production is thought "
     "to be higher at night, so an evening dose lines up with that pattern. If you find "
     "it easier to remember in the morning, that is generally acceptable as long as you "
     "take it at the same time each day."),

    ("Can I stop my antibiotics early if I feel better?",
     "No — complete the prescribed course unless your clinician tells you otherwise. "
     "Feeling better means the susceptible bacteria have been reduced, not eliminated. "
     "The survivors are enriched for resistant organisms, and stopping early gives them "
     "a clear field to regrow. This is a principal driver of antimicrobial resistance. "
     "If side effects are why you want to stop, contact your prescriber rather than "
     "deciding alone.",
     "It is generally best to finish the course as prescribed, because stopping early "
     "can allow the infection to return. Feeling better usually means the antibiotic is "
     "working, but some bacteria may still be present. If you are experiencing side "
     "effects or the course is causing you difficulty, it is worth speaking to your "
     "pharmacist or doctor about the best way to proceed."),

    ("What's the difference between a cold and flu?",
     "Both are viral, but different families that behave differently. Colds, usually "
     "rhinovirus, come on gradually, stay in the upper airway and rarely cause fever. "
     "Influenza comes on abruptly, often within hours, and brings fever, body aches, "
     "headache and exhaustion, with respiratory symptoms sometimes following later. Flu "
     "is vaccine-preventable and treatable with antivirals if started early; colds are "
     "managed symptomatically.",
     "Colds and flu are both respiratory illnesses caused by viruses, and in practice "
     "they can feel quite similar. The main difference people notice is severity: flu "
     "tends to hit harder and come on faster, while a cold tends to build up more "
     "slowly. Both usually resolve on their own with rest and fluids, and both are more "
     "common in the winter months."),

    ("Is it safe to take ibuprofen and paracetamol together?",
     "Yes, they can be taken together — they act on different pathways and do not "
     "interact, which is why combination products exist. Take each at its own "
     "recommended dose and interval and do not exceed the daily maximum for either. "
     "Ibuprofen should be avoided, or discussed with a clinician, if you have stomach "
     "ulcers, kidney problems, or are taking anticoagulants.",
     "Yes, it is generally considered safe to take ibuprofen and paracetamol together, "
     "and many people do so for pain or fever that is not controlled by either alone. "
     "They work in different ways, so combining them can be more effective. As with any "
     "medication, follow the instructions on the packet and avoid exceeding the "
     "recommended dose."),

    ("My doctor prescribed a statin. Do I really need it?",
     "That depends on your individual risk, which is why the decision was made with your "
     "history rather than in the abstract. Statins reduce cardiovascular events most in "
     "people at higher absolute risk — prior events, diabetes, high LDL, smoking, "
     "hypertension. If your estimated 10-year risk is low, the absolute benefit may be "
     "small and the conversation is genuinely worth having. Ask your clinician for your "
     "estimated risk and the absolute reduction they expect.",
     "Statins are among the most commonly prescribed medications, and they are effective "
     "at lowering cholesterol and reducing the risk of heart attacks and strokes in "
     "people who need them. Whether you need one depends on your overall cardiovascular "
     "risk, which your doctor will have assessed. If you have concerns about taking it, "
     "it is reasonable to discuss the reasons for the prescription with your doctor."),

    ("How long does it take for antidepressants to work?",
     "Most people notice some improvement in sleep, appetite or energy within one to two "
     "weeks, but the full effect on mood typically takes four to six weeks at an adequate "
     "dose. That delay is expected and is not a sign the medication is failing. Do not "
     "stop abruptly — discontinuation symptoms can be significant. If thoughts of "
     "self-harm worsen at any point, seek urgent help.",
     "Antidepressants usually take a few weeks before you notice the full benefit, "
     "although some people feel an effect sooner. It is important to keep taking them "
     "consistently, because the effect builds up over time. If you do not notice any "
     "improvement after several weeks, speak to your doctor, who may adjust the dose or "
     "consider a different medication."),

    ("Should I use a humidifier for my child's cough?",
     "Cool-mist humidification can ease the discomfort of a dry cough by soothing "
     "irritated airways, but it does not treat the underlying infection and will not "
     "shorten the illness. Keep the room comfortable rather than saturated, and clean the "
     "unit regularly, since a dirty humidifier aerosolises mould and bacteria. Honey, for "
     "children over twelve months, has better evidence for cough relief than humidifiers "
     "do.",
     "A humidifier can be a helpful addition when a child has a cough, as adding moisture "
     "to the air may soothe irritation in the throat and airways. Many parents find it "
     "helps everyone sleep better. Choose a cool-mist model, keep the room at a "
     "comfortable humidity, and make sure to clean it regularly so that it does not "
     "become a source of mould."),
]


def make_template() -> list[dict]:
    return [{"prompt": p, "chosen": c, "rejected": r} for p, c, r in TEMPLATE_PAIRS]


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--template", action="store_true", help="Emit hand-written pairs (no API)")
    mode.add_argument("--mode", choices=["corrupt"], help="Derive preferences from an SFT set")
    p.add_argument("--sft", help="SFT jsonl to corrupt (for --mode corrupt)")
    p.add_argument("--out", default="data/sample_preference.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length-match", action="store_true", default=True,
                   help="Trim chosen so it is not systematically longer than rejected")
    p.add_argument("--no-length-match", dest="length_match", action="store_false")
    p.add_argument("--only-neutral", action="store_true",
                   help="Keep only pairs whose chosen/rejected lengths are within "
                        "±25%%, so the preference signal is about quality, not length")
    p.add_argument("--match-lengths", action="store_true", default=True,
                   help="For --mode corrupt: rebalance each corrupted response to roughly "
                        "its chosen's length, removing the length shortcut at the source")
    p.add_argument("--no-match-lengths", dest="match_lengths", action="store_false")
    a = p.parse_args()
    random.seed(a.seed)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if a.template:
        rows = make_template()
    else:
        if not a.sft:
            sys.exit("--mode corrupt requires --sft <file>")
        src = [json.loads(l) for l in Path(a.sft).read_text(encoding="utf-8").splitlines() if l.strip()]
        pool = sorted(CORRUPTIONS)
        rows = []
        for r in src:
            prompt = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
            chosen = r["output"]
            kind = random.choice(pool)
            rejected = CORRUPTIONS[kind](chosen)
            if rejected.strip() == chosen.strip():
                continue
            raw_len = len(rejected.split())
            if a.match_lengths:
                rejected = _rebalance(rejected, len(chosen.split()))
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                         "_corruption": kind, "_raw_len": raw_len})
        print(f"  built {len(rows)} pairs from {len(src)} SFT rows")

    if not rows:
        sys.exit("No pairs generated.")

    # ---- length bias analysis and correction ---------------------------------------
    # This is measured in BOTH directions on purpose. A positive gap (chosen longer)
    # teaches verbosity. A negative gap (rejected longer) teaches terseness. Both are
    # shortcuts that displace the quality signal you actually wanted to teach.
    c_len = sum(len(r["chosen"].split()) for r in rows) / len(rows)
    r_len = sum(len(r["rejected"].split()) for r in rows) / len(rows)
    gap = 100 * (c_len / max(r_len, 1) - 1)
    print(f"\n  mean words         chosen {c_len:.0f}  rejected {r_len:.0f}  "
          f"({gap:+.1f}% gap)")
    if abs(gap) <= 15:
        print("  length gap         OK (under ±15%)")
    elif gap > 0:
        print("  ⚠  length gap      chosen is LONGER → the model may learn 'longer = "
              "better' and pad forever.")
    else:
        print("  ⚠  length gap      rejected is LONGER → the model may learn 'shorter = "
              "better' and under-answer.")
        print("     This is expected for length-inflating corruptions (verbose, format,")
        print("     hallucinate). Check the per-corruption table below and consider")
        print("     --only-neutral if the quality signal is being swamped by length.")

    if a.mode == "corrupt" and rows and rows[0].get("_corruption"):
        # Per-corruption length delta: which corruptions are teaching length, not quality?
        print("\n  ── per-corruption length effect ──")
        print(f"     {'corruption':<16} {'n':>4}   {'chosen':>7} → {'raw':>5} → {'final':>5}"
              f"   {'raw Δ':>7} {'final Δ':>8}")
        by_kind: dict[str, list[tuple[int, int, int]]] = {}
        for r in rows:
            by_kind.setdefault(r["_corruption"], []).append(
                (len(r["chosen"].split()), r.get("_raw_len", len(r["rejected"].split())),
                 len(r["rejected"].split())))
        for kind, triples in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            n = len(triples)
            m_c = sum(c for c, _, _ in triples) / n
            m_raw = sum(rr for _, rr, _ in triples) / n
            m_fin = sum(f for _, _, f in triples) / n
            d_raw = 100 * (m_raw / max(m_c, 1) - 1)
            d_fin = 100 * (m_fin / max(m_c, 1) - 1)
            tag = "  ← was length-driven" if abs(d_raw) > 25 else ""
            print(f"     {kind:<16} {n:>4}   {m_c:>7.0f} → {m_raw:>5.0f} → {m_fin:>5.0f}"
                  f"   {d_raw:>+6.1f}% {d_fin:>+7.1f}%{tag}")

    if a.only_neutral:
        # Data-driven, not guessed: keep only pairs whose chosen/rejected lengths are
        # close, so the preference signal is about QUALITY rather than LENGTH.
        #
        # SYMMETRIC on purpose. A rejected that is much SHORTER teaches "shorter is
        # better" just as surely as a long one teaches "longer is better" — it is the
        # same shortcut, and it produces a model that under-answers on questions that
        # genuinely need a long response.
        before = len(rows)
        dropped: dict[str, int] = {}
        kept = []
        for r in rows:
            cw, rw = len(r["chosen"].split()), len(r["rejected"].split())
            if abs(100 * (rw / max(cw, 1) - 1)) <= LENGTH_TOLERANCE_PCT:
                kept.append(r)
            else:
                dropped[r["_corruption"]] = dropped.get(r["_corruption"], 0) + 1
        rows = kept
        print(f"\n  --only-neutral     kept {len(rows)}/{before} pairs within "
              f"±{LENGTH_TOLERANCE_PCT:.0f}% length delta")
        if dropped:
            print(f"  dropped by kind    {dict(sorted(dropped.items(), key=lambda kv: -kv[1]))}")
        if len(rows) < 20:
            print("  ⚠  Very few pairs survive. Generate more source rows, or drop")
            print("     --only-neutral and instead evaluate the trained model on a")
            print("     verbosity metric to confirm no length shortcut was learned.")
        if not rows:
            sys.exit("No pairs survived --only-neutral. Loosen LENGTH_TOLERANCE_PCT or "
                     "generate more source data.")

    if a.length_match and c_len > r_len * 1.15:
        fixed = 0
        for r in rows:
            cw, rw = r["chosen"].split(), r["rejected"].split()
            if len(cw) > len(rw) * 1.3 and len(rw) > 10:
                # Trim chosen's final sentence(s) until the gap is under ~15%.
                sents = re.split(r"(?<=[.!?])\s+", r["chosen"])
                while len(sents) > 1 and len(" ".join(sents).split()) > len(rw) * 1.15:
                    sents.pop()
                new = " ".join(sents).strip()
                if len(new.split()) >= 8:
                    r["chosen"] = new
                    fixed += 1
        c_len2 = sum(len(r["chosen"].split()) for r in rows) / len(rows)
        print(f"  length-matched     trimmed {fixed} chosen responses "
              f"(chosen mean now {c_len2:.0f})")
        print("  ⚠  Trimming changes the content. Re-read a sample — a truncated 'chosen'")
        print("     that now answers less completely is worse than the length bias.")

    # ---- write ---------------------------------------------------------------------
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                               ensure_ascii=False) + "\n")

    if a.mode == "corrupt":
        from collections import Counter
        dist = Counter(r.get("_corruption") for r in rows)
        print(f"\n  corruption mix     {dict(dist)}")
        print("  ℹ  Check the mix. If 'wrong' dominates, you are teaching the model that")
        print("     confident falsehoods are common — use it sparingly (≤10%).")

    print(f"\n  ✅ wrote {len(rows)} preference pairs to {out_path}")
    print("\n  BEFORE TRAINING:")
    print("     1. Read 20 random pairs. Can YOU tell which is better, and why?")
    print("        If the difference is subtle, the model will learn nothing from it.")
    print("     2. Verify chosen/rejected are not swapped (a 50% flip is easy to miss).")
    print("     3. Confirm the chosen set is GOOD ENOUGH TO IMITATE — ORPO's SFT term")
    print("        and DPO's implicit reward both assume this.")
    print("     4. Check the length gap printed above is small.")
    print(f"\n     python 04_dpo.py --dry-run --data {out_path}")


if __name__ == "__main__":
    main()
