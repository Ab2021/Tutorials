#!/usr/bin/env python
"""
data/make_instruction_data.py — build an SFT dataset from your own documents.

Two modes:

  1. --from-docs   : chunk your text and synthesise instruction/response pairs with an
                     LLM. Fast, cheap, and the standard modern approach (see CS-09).
  2. --template    : no LLM at all — emit a small hand-written seed set you can grow.
                     Useful for testing the training pipeline before spending money.

The critical quality rule
-------------------------
Generated data is only as good as its filter. A raw LLM dump will contain refusals,
hallucinated specifics, repetitive phrasing, and answers that ignore the source. The
filtering stage below is not optional; skipping it is why "we fine-tuned on synthetic
data and it got worse" happens.

Run it
------
    python data/make_instruction_data.py --template --out data/sample_sft.jsonl
    python data/make_instruction_data.py --from-docs corpus.txt --n 500 --out data/mine.jsonl
    python data/make_instruction_data.py --from-docs corpus.txt --n 500 --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: F401  — applies the UTF-8 console fix on Windows (see common/__init__.py)

# --------------------------------------------------------------------------------------
# Prompt template for generation. Note what it asks for — diversity in *length* matters
# as much as diversity in task type, because a model trained only on 3-sentence answers
# learns to always answer in 3 sentences.
# --------------------------------------------------------------------------------------
GEN_PROMPT = """You are creating training data for a domain-specific assistant.

Below is an excerpt from a source document. Create ONE instruction/response pair that
teaches a model something genuinely useful from this excerpt.

Rules:
- The instruction must be a realistic thing a user would ask. Vary the phrasing and type
  (explain / list / compare / summarise / extract / classify / troubleshoot / calculate).
- The response must be answerable ONLY from the excerpt. Do not add outside knowledge.
- If the excerpt is too thin to support a good question, reply exactly: SKIP
- Vary response length naturally: some answers 1 sentence, some a full paragraph.
- Do NOT start every instruction with "What is..." or every answer with "Sure!"

Reply as strict JSON on a single line, no markdown fence:
{{"instruction": "...", "input": "", "output": "..."}}

EXCERPT:
{chunk}
"""


# --------------------------------------------------------------------------------------
# Mode 1: template (no API key, no cost — for testing the pipeline)
# --------------------------------------------------------------------------------------
TEMPLATE_PAIRS = [
    ("Explain the mechanism of action of {topic} in two or three sentences.",
     "{topic} acts by binding to its target receptor and modulating downstream signalling. "
     "This produces a measurable change in cellular behaviour within minutes to hours, "
     "depending on the pathway involved."),
    ("List three key considerations when working with {topic}.",
     "1. Dose and timing matter more than the specific agent chosen.\n"
     "2. Interactions with the existing regimen should be checked first.\n"
     "3. Monitoring should continue after the acute phase, not just during it."),
    ("Summarise the main risks associated with {topic} in one paragraph.",
     "The principal risks of {topic} fall into three groups: immediate reactions, "
     "cumulative dose-dependent effects, and interactions with concurrent therapy. "
     "Most are manageable with appropriate monitoring."),
    ("Compare {topic} with the alternative approach. Which is preferred and why?",
     "{topic} is generally preferred when rapid onset is required; the alternative is "
     "better tolerated over long horizons. The deciding factor is usually the expected "
     "duration of treatment rather than the acute response."),
    ("A colleague asks whether {topic} is appropriate for a patient with reduced renal "
     "function. What do you tell them?",
     "Exercise caution. Reduced renal clearance prolongs exposure, so start at a lower "
     "dose and titrate slowly, with more frequent monitoring of renal function."),
    ("Extract the dosage information for {topic} and present it as a short list.",
     "- Standard starting dose: as per local protocol\n"
     "- Titration: step up no faster than the recommended interval\n"
     "- Maximum: do not exceed without specialist input"),
    ("Is the following statement supported by the text? 'The benefits of {topic} outweigh "
     "the risks in all patients.' Answer yes or no and justify briefly.",
     "No. The text supports benefit in specific populations and explicitly notes "
     "contraindications, so a universal claim overstates the evidence."),
    ("Rewrite this for a patient with no medical background: '{topic} modulates the "
     "downstream signalling cascade.'",
     "{topic} changes how your cells respond to signals. In plain terms, it turns the "
     "volume down on a process that is running too high."),
    ("What follow-up would you recommend after starting {topic}?",
     "Review within two to four weeks to assess response and tolerance, then at longer "
     "intervals once stable. Escalate early if symptoms worsen rather than improve."),
    ("Give one situation where {topic} should NOT be used.",
     "It should be avoided where the alternative pathway is already compromised, since "
     "the compensatory reserve is insufficient to cover the added load."),
]

TOPICS = [
    "Metformin", "Atorvastatin", "Ezetimibe", "ACE inhibitors", "beta blockers",
    "NSAIDs", "proton pump inhibitors", "SSRIs", "inhaled corticosteroids", "opioids",
    "anticoagulation", "insulin titration", "thyroid replacement", "diuretics",
    "bisphosphonates", "antihistamines", "corticosteroid tapering", "antibiotic courses",
]


def make_template(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        tmpl, ans = TEMPLATE_PAIRS[i % len(TEMPLATE_PAIRS)]
        topic = TOPICS[i % len(TOPICS)]
        rows.append({
            "instruction": tmpl.format(topic=topic),
            "input": "",
            "output": ans.replace("{topic}", topic),
        })
    random.shuffle(rows)
    return rows


# --------------------------------------------------------------------------------------
# Mode 2: generate from documents with an LLM
# --------------------------------------------------------------------------------------
def chunk_text(text: str, words: int = 400, overlap: int = 40) -> list[str]:
    w = text.split()
    out = []
    for i in range(0, len(w), words - overlap):
        piece = w[i:i + words]
        if len(piece) >= 60:
            out.append(" ".join(piece))
        if i + words >= len(w):
            break
    return out


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model=model, temperature=0.9, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


def call_anthropic(prompt: str, model: str = "claude-sonnet-5") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model=model, max_tokens=600, temperature=0.9,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


def call_local(prompt: str, model: str) -> str:
    """Generate with a local HF model — zero marginal cost, good for bulk generation."""
    from transformers import pipeline
    gen = pipeline("text-generation", model=model, device_map="auto")
    out = gen(prompt, max_new_tokens=400, do_sample=True, temperature=0.9,
              return_full_text=False)[0]["generated_text"]
    return out


# --------------------------------------------------------------------------------------
# Filtering — the stage that decides whether this works
# --------------------------------------------------------------------------------------
REFUSAL_MARKERS = ["i cannot", "i can't", "as an ai", "i'm sorry", "i am unable",
                   "i don't have access", "it is not possible to determine"]

BAD_OPENERS = ["sure!", "certainly!", "of course!", "great question"]


def quality_filter(rows: list[dict], source_chunks: set[str] | None = None) -> tuple[list[dict], dict]:
    """
    Returns (kept, report). Every rule here exists because it caught a real failure.
    """
    kept, reasons = [], {"too_short": 0, "too_long": 0, "refusal": 0, "bad_opener": 0,
                         "duplicate": 0, "echoes_instruction": 0, "placeholder": 0}
    seen: set[str] = set()

    for r in rows:
        ins, out = r.get("instruction", "").strip(), r.get("output", "").strip()
        if not ins or not out:
            continue
        if len(out.split()) < 8:
            reasons["too_short"] += 1
            continue
        if len(out.split()) > 400:
            reasons["too_long"] += 1
            continue
        if any(m in out.lower() for m in REFUSAL_MARKERS):
            reasons["refusal"] += 1
            continue
        if any(out.lower().startswith(b) for b in BAD_OPENERS):
            reasons["bad_opener"] += 1
            continue
        if re.search(r"\{[a-z_]+\}|\[INSERT|XXX|TODO|\.\.\.\.", out):
            reasons["placeholder"] += 1
            continue
        # The model restating the question instead of answering it.
        if ins.lower()[:40] in out.lower():
            reasons["echoes_instruction"] += 1
            continue
        # Dedup on the instruction — near-identical questions teach nothing extra.
        key = re.sub(r"\W+", "", ins.lower())[:80]
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"instruction": ins, "input": r.get("input", ""), "output": out})

    return kept, reasons


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-docs", help="Text file (or .txt corpus) to generate from")
    src.add_argument("--template", action="store_true", help="Emit a template seed set (no API)")
    p.add_argument("--out", default="data/sample_sft.jsonl")
    p.add_argument("--n", type=int, default=200, help="Target number of pairs")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    p.add_argument("--model", default=None, help="Override the provider's default model")
    p.add_argument("--chunk-words", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- template mode --------------------------------------------------------------
    if a.template:
        rows = make_template(max(a.n, 40))
        _write(rows, out_path)
        print(f"  ✅ wrote {len(rows)} template pairs to {out_path}")
        print("  ℹ  These are deliberately generic — good for smoke-testing the training")
        print("     pipeline, not for producing a useful model.")
        return

    # ---- doc mode -------------------------------------------------------------------
    text = Path(a.from_docs).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, a.chunk_words)
    print(f"  corpus             {len(text):,} chars → {len(chunks)} chunks")
    if not chunks:
        sys.exit("Source text too short to chunk.")

    call = {"openai": call_openai, "anthropic": call_anthropic,
            "local": (lambda pr: call_local(pr, a.model or "Qwen/Qwen2.5-7B-Instruct"))}[a.provider]
    if a.provider == "openai" and a.model:
        call = lambda pr: call_openai(pr, a.model)                      # noqa: E731
    if a.provider == "anthropic" and a.model:
        call = lambda pr: call_anthropic(pr, a.model)                   # noqa: E731

    raw: list[dict] = []
    skipped = 0
    # Cost estimate — print it BEFORE spending money.
    est_tokens = len(chunks) * (a.chunk_words * 4 // 3 + 200)
    print(f"  provider           {a.provider}")
    print(f"  est. input tokens  ~{est_tokens:,}  (plus output) — know your price before running")

    for i, ch in enumerate(chunks):
        if len(raw) >= a.n:
            break
        try:
            resp = call(GEN_PROMPT.format(chunk=ch))
        except Exception as e:                                  # noqa: BLE001
            print(f"    chunk {i}: API error {e}")
            continue
        resp = re.sub(r"^```(?:json)?|```$", "", resp.strip(), flags=re.M).strip()
        if resp.upper().startswith("SKIP"):
            skipped += 1
            continue
        try:
            row = json.loads(resp)
            raw.append(row)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", resp, re.S)
            if m:
                try:
                    raw.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(chunks)} chunks → {len(raw)} raw pairs")

    kept, report = quality_filter(raw)
    print(f"\n  generated          {len(raw)} raw  ({skipped} chunks were SKIP)")
    print(f"  after filtering    {len(kept)}")
    print(f"  rejected           {report}")

    if not kept:
        sys.exit("Nothing survived filtering. Inspect the raw generations — usually the "
                 "prompt is too weak or the source text is too thin.")

    random.shuffle(kept)
    _write(kept, out_path)

    print(f"\n  ✅ wrote {len(kept)} pairs to {out_path}")
    print("\n  ⚠  BEFORE TRAINING:")
    print("     1. READ 20 RANDOM ROWS YOURSELF. No filter replaces reading your data.")
    print("     2. Check the response-length distribution — a narrow band means your")
    print("        model will learn to answer at exactly that length.")
    print("     3. Decontaminate against your eval set (see common/eval_utils.py).")
    print("     4. If the data came from a commercial model, check its ToS on training")
    print("        competing models.")
    print(f"\n     python 01_sft_lora.py --dry-run --data {out_path}")


def _write(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
