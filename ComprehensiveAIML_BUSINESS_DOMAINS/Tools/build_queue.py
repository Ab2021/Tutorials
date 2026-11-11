#!/usr/bin/env python3
"""Build the authoring/repair work queue from measured repository state.

Idempotent: re-running recomputes gaps from the files on disk, so a resumed
session never re-does completed work and never trusts a stale claim.

    python Tools/build_queue.py            # writes .tmp_parts/queue.jsonl + task briefs
    python Tools/build_queue.py --summary   # print counts only
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".authoring_state"
BRIEFS = ROOT / "Tools" / "assignments" / "tasks"

sys.path.insert(0, str(ROOT / "Tools"))
from manuscript_lint import lint  # noqa: E402

CASE_ROWS = re.compile(r"^\|\s*(PLAN-[\w-]+|CASE-[\w-]+)\s*\|(.+)$", re.M)


def slug(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[`*:]", "", s)
    s = re.sub(r"\b(planned|proposed|shared brief|draft)\b.*$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s).strip()
    return "_".join(w.capitalize() for w in s.split())


def units() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() and (p / "Case_Studies").is_dir() and (p / "Cheat_Sheets").is_dir():
            if "Tools" in p.parts or any(part.startswith(".") for part in p.relative_to(ROOT).parts):
                continue
            out.append(p)
    return out


def promised_rows(idx: Path) -> list[tuple[str, str, str | None]]:
    """Return (plan_id, title, linked_target_or_None), deduplicated by title."""
    if not idx.exists():
        return []
    t = idx.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for line in t.splitlines():
        m = re.match(r"^\|\s*(PLAN-[\w-]+|CASE-[\w-]+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        title_cell = cells[1]
        link = re.match(r"^\[([^\]]+)\]\(\.?/?([^)]+\.md)\)$", title_cell)
        target = None
        if link:
            title, target = link.group(1), link.group(2)
            if target.startswith("../"):
                target = None
        else:
            title = title_cell
        if "REF-" in title_cell:
            continue
        title = re.sub(r"\s*[\(\[]\s*(planned|proposed|draft|shared brief)[^\)\]]*[\)\]]\s*$", "", title, flags=re.I).strip()
        if not title:
            continue
        key = re.sub(r"\W+", "", title).casefold()
        if not key or key in seen:
            continue
        # Reject cells that are not a real topic: bare dates, ids, or filler words.
        words = [w for w in re.split(r"\W+", title) if w]
        if len(words) < 2 or re.fullmatch(r"[\d\W_]+", title):
            continue
        if re.fullmatch(r"planned|tbd|todo|placeholder|n/?a", title.strip(), re.I):
            continue
        if re.fullmatch(r"\d{4}[\s\-_]\d{2}[\s\-_]\d{2}", title.strip()):
            continue
        # A cell that lists several topics (semicolons, or a topic inventory) is not a
        # single case decision: it cannot become one manuscript filename.
        if title.count(";") >= 1 or len(words) > 12:
            continue
        seen.add(key)
        rows.append((cells[0], title, target))
    return rows



def h2_count(idx: Path) -> int:
    if not idx.exists():
        return 0
    t = idx.read_text(encoding="utf-8", errors="replace")
    return len([l for l in t.splitlines() if re.match(r"^## \d", l)])


def main() -> int:
    summary_only = "--summary" in sys.argv
    if not summary_only:
        BRIEFS.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    paths_seen: set[str] = set()
    seq = 0

    def add(kind: str, prio: int, path: str, unit: str, brief: str, extra: dict | None = None):
        nonlocal seq
        if path in paths_seen and kind.startswith(("create", "repair")):
            return
        paths_seen.add(path)
        seq += 1
        tid = f"T{seq:04d}"
        rec = {"id": tid, "kind": kind, "priority": prio, "path": path, "unit": unit,
               "brief": brief, "status": "pending"}
        if extra:
            rec.update(extra)
        tasks.append(rec)
        if not summary_only:
            (BRIEFS / f"{tid}.md").write_text(brief, encoding="utf-8")

    # ---- scan existing manuscripts -------------------------------------------------
    for p in sorted(ROOT.rglob("*.md")):
        if "Tools" in p.parts or any(part.startswith(".") for part in p.relative_to(ROOT).parts):
            continue
        if p.name in ("INDEX.md", "AUTHORING_PROGRESS.md"):
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("00_") or rel in ("README.md",):
            continue
        r = lint(p)
        unit = str(p.parent.parent.relative_to(ROOT)).replace("\\", "/") if p.parent.name in ("Case_Studies", "Cheat_Sheets") else str(p.parent.relative_to(ROOT)).replace("\\", "/")
        if r["kind"] not in ("case", "sheet"):
            continue
        if r["ok"]:
            continue
        padded = r.get("long_repeat_groups", 0) > 0
        prio = 1 if padded else 2
        if padded:
            diagnosis = """This file was inflated with repeated prose (for example `Operational note 879: ...`, or an
identical interview Answer/Reasoning/Follow-up trio copied a dozen times). Physical line count is
not the deliverable: replace every repeated block with genuine, distinct analysis."""
        elif r["lines"] < 1000 or r["distinct_prose"] < 700:
            diagnosis = """This file is too thin to serve its teaching role. It needs more substantive sections' worth of
real reasoning — additional mechanisms, distinct worked numbers, distinct failure modes, and
individually answered questions — not filler."""
        else:
            diagnosis = """The prose substance is acceptable; the structural contract is not. Fix the listed structural
defects by adding or correcting the required sections, and keep the existing verified analysis
intact where it is sound. Do not pad to compensate: if you add material it must be new substance."""
        brief = f"""# Repair task {rel}

The course acceptance lint rejects this existing manuscript. It is NOT complete and must not be
represented as complete. Rewrite it as a substantive manuscript on the SAME topic; keep its title
and its role in the unit.

## Measured defects (from `python Tools/manuscript_lint.py`)
{chr(10).join('- ' + x for x in r['problems'])}

## Current measurements
{json.dumps({k: r[k] for k in ('kind','lines','distinct_prose','dup_ratio','long_repeat_groups')}, indent=1)}

## Diagnosis
{diagnosis}

## What to do
1. Read `00_AGENT_AUTHORING_CONTRACT.md` and obey it exactly.
2. Keep the topic and the required heading set for its kind
   ({r['kind']}: {'15 case sections' if r['kind']=='case' else '12 sheet sections'}), in order.
3. Replace every repeated block with genuine analysis: real mechanisms, distinct numbers,
   distinct failure modes, distinct interview questions each with its own answer.
4. Re-derive every number and re-state its units and validity condition.
5. Lint until exit 0:
   `python Tools/manuscript_lint.py {rel}`

## Boundaries
- Rewrite only this file. Do not touch indexes (integration task covers them).
- Never invent a measured company result; label constructed values `ILLUSTRATIVE`.
"""
        add("repair", prio, rel, unit, brief, {"defects": r["problems"]})

    # ---- unit-level gaps -----------------------------------------------------------
    for u in units():
        urel = str(u.relative_to(ROOT)).replace("\\", "/")
        cases_dir, sheets_dir = u / "Case_Studies", u / "Cheat_Sheets"
        existing_cases = sorted(f.name for f in cases_dir.glob("*.md") if f.name != "INDEX.md" and not f.name.startswith("."))
        existing_sheets = sorted(f.name for f in sheets_dir.glob("*.md") if f.name != "INDEX.md" and not f.name.startswith("."))
        has_ci = any(f.endswith("Common_Issues.md") for f in existing_sheets)

        nums = [int(m.group(1)) for f in existing_sheets for m in [re.match(r"(\d+)_", f)] if m]
        next_sheet = max(nums + [0]) + 1

        if not has_ci:
            ci = f"{next_sheet:02d}_25_Common_Issues.md"
            brief = f"""# Common Issues task: {urel}

Create `{urel}/Cheat_Sheets/{ci}` — the mandatory final troubleshooting artifact for this unit.

## Requirements
- Same 12 numbered H2 headings as every sheet, in contract order (see
  `00_AGENT_AUTHORING_CONTRACT.md` section 4).
- At least **25 distinct, complete** issues. Each issue row uses a stable id `CI-01`, `CI-02`, ... and
  has: category (VIOLATED_ASSUMPTION | NUMERICAL_EDGE | SCALE_FAILURE | CONCEPT_CONFUSION),
  symptom, competing causes, discriminating check, correction, prevention, verification.
- The issues must be specific to this unit's subject matter, not generic "monitor the model" advice.
  Derive them from the unit's own topic map in `{urel}/INDEX.md` and its existing files:
  sheets present: {', '.join(existing_sheets) or 'none'}; cases present: {', '.join(existing_cases) or 'none'}.
- >=1,000 physical lines, >=700 distinct prose lines, no long line repeated more than twice.
- >=12 answered diagnostic questions in section 11.

## Acceptance
`python Tools/manuscript_lint.py {urel}/Cheat_Sheets/{ci}` exits 0.
"""
            add("create_ci", 3, f"{urel}/Cheat_Sheets/{ci}", urel, brief)

        # promised case rows without a file
        rows = promised_rows(cases_dir / "INDEX.md")
        cn = [int(m.group(1)) for f in existing_cases for m in [re.match(r"(\d+)_", f)] if m]
        nxt = max(cn + [0]) + 1
        for pid, title, target in rows:
            if target and (cases_dir / target).exists():
                continue
            s = slug(title)
            if any(s[:28].lower() in f.lower() for f in existing_cases):
                continue
            if target:
                fname = target
            else:
                fname = f"{nxt:02d}_{s[:64]}.md"
                nxt += 1
            brief = f"""# Case task: {urel}

Create `{urel}/Case_Studies/{fname}` for the catalogs row `{pid}` — "{title}".

## Requirements
- 15 numbered H2 sections exactly as listed in `00_AGENT_AUTHORING_CONTRACT.md` section 2,
  with `### 5a`-`### 5e` and `### Interview pack` under section 14.
- One bounded consequential decision with a named owner, decision grain, cut-off timestamp,
  baseline, and a real "defer / no-action" alternative.
- >=3 checked numeric derivations in fenced ```text blocks using
  FORMULA -> WHERE -> CONSTRAINT -> NUMERICAL EXAMPLE -> INTERPRETATION, plus one boundary
  counterexample. Recompute the arithmetic and show intermediate values.
- >=4 distinct failure mechanisms with discriminating checks.
- 12-18 distinct interview questions, each with its own Answer / Reasoning / Follow-up / Red flag /
  Scoring. At least half must require calculation, selection, or diagnosis.
- Section 15 Quick Reference Card last.
- Evidence honesty: no invented company architecture or measured results; label constructed
  figures ILLUSTRATIVE and cite nothing you cannot support.

## Context
Read `{urel}/INDEX.md` (topic map) and any sibling case in the same folder for house style, but do not
copy their prose.

## Acceptance
`python Tools/manuscript_lint.py {urel}/Case_Studies/{fname}` exits 0.
"""
            add("create_case", 4, f"{urel}/Case_Studies/{fname}", urel, brief)

        # promised sheet rows without a file
        for pid, title, target in promised_rows(sheets_dir / "INDEX.md"):
            if target and (sheets_dir / target).exists():
                continue
            s = slug(title)
            if any(s[:28].lower() in f.lower() for f in existing_sheets):
                continue
            if target:
                fname = target
            else:
                fname = f"{next_sheet:02d}_{s[:64]}.md"
                next_sheet += 1
            brief = f"""# Cheat-sheet task: {urel}

Create `{urel}/Cheat_Sheets/{fname}` for the catalogs row `{pid}` — "{title}".

## Requirements
- 12 numbered H2 sections exactly as listed in `00_AGENT_AUTHORING_CONTRACT.md` section 3,
  with `### 4.1`-`### 4.4`.
- Every substantive result uses the five-part anatomy FORMULA -> WHERE -> CONSTRAINT ->
  NUMERICAL EXAMPLE -> INTERPRETATION, and ends with an explicit validity condition.
  For a non-mathematical topic use the mechanism/invariant block instead (preconditions ->
  state transition -> invariant and justification -> worked trace -> failure boundary).
- >=3 independent arithmetic checks in section 5 with a sensitivity note.
- >=4 failure entries with competing causes and discriminating checks in section 8.
- 10-15 answered questions in section 11, at least half computational or diagnostic.
- Section 12 Quick Reference Card (5-10 rows) is the LAST content.

## Context
Read `{urel}/INDEX.md` for the unit's scope; do not duplicate a sibling sheet's topic.

## Acceptance
`python Tools/manuscript_lint.py {urel}/Cheat_Sheets/{fname}` exits 0.
"""
            add("create_sheet", 4, f"{urel}/Cheat_Sheets/{fname}", urel, brief)

        # collection index completeness
        if h2_count(cases_dir / "INDEX.md") < 7:
            rel = f"{urel}/Case_Studies/INDEX.md"
            brief = f"""# Index task: {rel}

Rewrite this collection index to the mandated **7 sections**, in order:
1 Collection metadata | 2 Case catalog (11 columns: Case ID, linked title, domain, case type,
decision, primary pattern, up to two secondary patterns, evidence, stage, last reviewed,
result/limitation) | 3 Canonical reusable patterns | 4 Similarity/contrast map |
5 Questions/future-case backlog | 6 Review/retirement log | 7 Pre-publish checks.

Rules:
- Link ONLY files that exist on disk. List every existing case file in `{urel}/Case_Studies/`.
- A promised decision with no file keeps a `planned` row with its original `PLAN-` id and no link.
- Never delete a commitment silently; record any re-scoping in section 5 with a reason.

## Acceptance
Every `./*.md` link resolves; 7 numbered H2 sections present.
"""
            add("index_cases", 5, rel, urel, brief)

        if h2_count(sheets_dir / "INDEX.md") < 11:
            rel = f"{urel}/Cheat_Sheets/INDEX.md"
            brief = f"""# Index task: {rel}

Rewrite this collection index to the mandated **11 sections**, in order:
1 Overview | 2 Topic ladder (foundation/core/advanced/final) | 3 Quick lookup |
4 Beginner/Practitioner/Advanced reading paths | 5 Prerequisite skills | 6 Exclusions/handoffs |
7 Master symbol glossary (symbol, meaning, units, shape/type, domain, first use) |
8 Card directory | 9 Common Issues locator | 10 Sources/versions/maintenance | 11 Acceptance checklist.

Rules:
- Link ONLY files that exist on disk in `{urel}/Cheat_Sheets/`.
- The glossary must contain real symbols actually used by the existing sheets.
- Planned cards stay `planned` without a link.

## Acceptance
Every `./*.md` link resolves; 11 numbered H2 sections present.
"""
            add("index_sheets", 5, rel, urel, brief)

        # unit topic-map status honesty
        idx = u / "INDEX.md"
        if idx.exists():
            t = idx.read_text(encoding="utf-8", errors="replace")
            if "| planned |" in t:
                rel = f"{urel}/INDEX.md"
                brief = f"""# Unit index task: {rel}

Update the "Detailed topic map" / "Topics to develop" status column so it reports reality:
- `authored` + a relative link **only** for topics actually covered by an existing file in
  `{urel}/Case_Studies/` or `{urel}/Cheat_Sheets/`.
- `planned` for the rest.
Do not change topic names, do not add new sections, do not claim review.

Existing sheets: {', '.join(existing_sheets) or 'none'}
Existing cases: {', '.join(existing_cases) or 'none'}
"""
                add("unit_index", 6, rel, urel, brief)

    if summary_only:
        import collections
        print(len(tasks), "tasks")
        print(collections.Counter(t["kind"] for t in tasks))
        return 0

    with open(TMP / "queue.jsonl", "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    import collections
    print(len(tasks), "tasks queued")
    for k, v in sorted(collections.Counter(t["kind"] for t in tasks).items()):
        print(f"  {k}: {v}")
    for p in sorted({t["priority"] for t in tasks}):
        print(f"  priority {p}: {sum(1 for t in tasks if t['priority'] == p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
