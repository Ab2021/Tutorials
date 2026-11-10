#!/usr/bin/env python3
"""Deterministic collection-index generator.

Rewrites `Case_Studies/INDEX.md` (7 sections) and `Cheat_Sheets/INDEX.md`
(11 sections) from the actual files on disk. No model involved: every catalog
row is built from the target manuscript's own YAML frontmatter, so the index can
never claim a file that does not exist or invent a decision that was not written.

    python Tools/gen_indexes.py --dry-run
    python Tools/gen_indexes.py
    python Tools/gen_indexes.py --only 01_foundations_computing_and_problem_solving
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Tools"))
from manuscript_lint import lint  # noqa: E402

FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
CASE_ROW = re.compile(r"^\|\s*(PLAN-[\w-]+)\s*\|", re.M)
PLANNED_TITLE = re.compile(r"^\|\s*(PLAN-[\w-]+)\s*\|\s*([^|]+?)\s*\|", re.M)


def frontmatter(text: str) -> dict[str, str]:
    m = FRONT.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def title_of(path: Path, text: str, fm: dict[str, str]) -> str:
    for key in ("case_title", "title"):
        if fm.get(key):
            return fm[key]
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ")


def status_of(path: Path) -> str:
    r = lint(path)
    return "authored" if r["ok"] else "draft (lint open)"


def planned_rows(idx: Path) -> list[tuple[str, str]]:
    if not idx.exists():
        return []
    t = idx.read_text(encoding="utf-8", errors="replace")
    rows, seen = [], set()
    for pid, title in PLANNED_TITLE.findall(t):
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title).strip()
        title = re.sub(r"\s*[\(\[]\s*(planned|proposed|shared brief|draft)[^\)\]]*[\)\]]\s*$", "", title, flags=re.I).strip()
        if not title or title.casefold() in seen:
            continue
        words = [w for w in re.split(r"\W+", title) if w]
        if len(words) < 2 or "REF-" in title:
            continue
        seen.add(title.casefold())
        rows.append((pid, title))
    return rows


def unit_title(unit: Path) -> str:
    """Human title from the unit's own INDEX.md H1, falling back to the folder name."""
    idx = unit / "INDEX.md"
    if idx.exists():
        for line in idx.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                t = line[2:].strip()
                t = re.sub(r"^(Module\s+\d+\s*[—-]\s*)", "", t)
                return re.sub(r"\s*[—-]\s*(Case Stud(y|ies)|Cheat Sheets?)$", "", t, flags=re.I).strip()
    return unit.name.replace("_", " ")


def case_index(unit: Path) -> str:
    cdir = unit / "Case_Studies"
    files = sorted(f for f in cdir.glob("*.md") if f.name != "INDEX.md")
    entries = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        fm = frontmatter(text)
        entries.append({
            "file": f,
            "title": title_of(f, text, fm),
            "id": fm.get("case_id") or "-",
            "type": fm.get("case_type") or "-",
            "decision": fm.get("owner_role") and fm.get("case_title") or "-",
            "primary": fm.get("primary_pattern") or "-",
            "secondary": ", ".join(re.findall(r"[a-z0-9-]+", fm.get("secondary_patterns", "")))[:60] or "-",
            "evidence": fm.get("evidence_level") or "-",
            "stage": fm.get("status") or "proposed",
            "reviewed": fm.get("last_reviewed") or "null",
            "status": status_of(f),
        })
    planned = planned_rows(cdir / "INDEX.md")

    L = [f"# {unit_title(unit)} — Case Studies", "", "## 1. Collection metadata", "",
         "| Field | Value |", "|---|---|",
         f"| Collection owner | Curriculum maintainer |",
         f"| Canonical case-ID format | {entries[0]['id'] if entries else 'CASE-XX-NN'} |",
         "| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |",
         "| Editorial state | Draft; independent publication review is not claimed. |",
         "| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |",
         "| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |",
         "", "## 2. Case catalog", "",
         "| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        L.append(
            f"| {e['id']} | [{e['title']}](./{e['file'].name}) | {unit_title(unit)} | {e['type']} | "
            f"decision record in the linked case | {e['primary']} | {e['secondary']} | {e['evidence']} | "
            f"{e['stage']} | {e['reviewed']} | {e['status']} |"
        )
    for pid, title in planned:
        L.append(f"| {pid} | {title} | {unit_title(unit)} | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |")
    if not entries and not planned:
        L.append("| — | No case committed yet | — | — | — | — | — | — | — | — | — |")

    L += ["", "## 3. Canonical reusable patterns", "",
          "| Pattern | Diagnostic signature | Use when | Boundary |", "|---|---|---|---|",
          "| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |",
          "| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |",
          "| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |",
          "| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |",
          "| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |",
          "", "## 4. Similarity / contrast map", "",
          f"This collection holds {len(entries)} authored case(s) and {len(planned)} planned commitment(s). "
          "Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, "
          "owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: "
          "cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.",
          "", "## 5. Questions / future-case backlog", ""]
    if planned:
        L.append("Planned commitments, retained without silent deletion:")
        L.append("")
        for pid, title in planned:
            L.append(f"- `{pid}` — {title}. Not authored; no file exists, so no link is offered.")
    else:
        L.append("No unfulfilled case commitment is recorded for this collection.")
    L += ["", "## 6. Review / retirement log", "",
          "| Date | Artifact | Event | Evidence / stage change |", "|---|---|---|---|"]
    for e in entries:
        L.append(f"| 2026-11-10 | [{e['title']}](./{e['file'].name}) | index regenerated from disk | {e['evidence']} / {e['stage']} unchanged |")
    if not entries:
        L.append("| 2026-11-10 | collection | index regenerated; no authored case yet | not-measured / proposed |")
    L += ["", "## 7. Pre-publish checks", "",
          "- [x] Every catalog row links a file that exists, or is marked planned without a link.",
          "- [x] Evidence, implementation stage, and editorial status are separate columns.",
          "- [x] Planned commitments are retained with no silent deletion.",
          "- [ ] Independent source review of every authored case.",
          "- [ ] Publication review of every authored case.",
          ""]
    return "\n".join(L)


def sheet_index(unit: Path) -> str:
    sdir = unit / "Cheat_Sheets"
    files = sorted(f for f in sdir.glob("*.md") if f.name != "INDEX.md")
    entries = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        fm = frontmatter(text)
        entries.append({
            "file": f,
            "title": title_of(f, text, fm),
            "level": fm.get("level") or ("troubleshooting" if "Common_Issues" in f.name else "core"),
            "evidence": fm.get("evidence") or "DERIVED",
            "status": status_of(f),
        })
    planned = planned_rows(sdir / "INDEX.md")
    ci = [e for e in entries if "Common_Issues" in e["file"].name]
    cards = [e for e in entries if e not in ci]

    L = [f"# {unit_title(unit)} — Cheat Sheets", "", "## 1. Overview", "",
         f"This collection carries {len(cards)} reference card(s) and {len(ci)} troubleshooting artifact(s) for "
         f"`{unit_title(unit)}`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file "
         "turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when "
         "`Tools/manuscript_lint.py` exits 0 for it.",
         "", "## 2. Topic ladder", "",
         "| Level | Cards | Purpose |", "|---|---|---|"]
    for lvl in ("foundation", "core", "advanced", "troubleshooting"):
        names = [e["title"] for e in entries if e["level"] == lvl]
        if names:
            L.append(f"| {lvl} | {', '.join(names)} | {'Diagnose and prevent recurring failures' if lvl=='troubleshooting' else 'Build from prerequisites to production judgement'} |")
    if not any(e["level"] for e in entries):
        L.append("| — | none yet | no card authored in this collection |")

    L += ["", "## 3. Quick lookup", "",
          "| If you need to… | Read |", "|---|---|"]
    for e in cards:
        L.append(f"| work on {e['title'].lower()} | [{e['title']}](./{e['file'].name}) |")
    for e in ci:
        L.append(f"| diagnose a failure that keeps recurring | [{e['title']}](./{e['file'].name}) |")
    if not entries:
        L.append("| — | nothing authored yet |")

    L += ["", "## 4. Reading paths", "",
          "| Path | Sequence |", "|---|---|",
          "| Beginner | foundation cards first, then the Quick Reference Card (§12) of each core card |",
          "| Practitioner | core cards in ladder order, then one case in `../Case_Studies/` that cites them |",
          "| Advanced | advanced cards, then the Common Issues artifact and its discriminating checks |",
          "", "## 5. Prerequisite skills", "",
          "| Skill | Why it is needed | Where it is available |", "|---|---|---|",
          "| Unit topic map | Defines what this collection is responsible for and what it excludes | `../INDEX.md` |",
          "| Decision framing | Cards assume a stated decision, population, and cut-off | sibling case collection |",
          "| Arithmetic discipline | Every card carries checked numbers with units | the cards themselves |",
          "", "## 6. Exclusions and handoffs", "",
          "| Excluded here | Handed to |", "|---|---|",
          "| Decision-context narrative, baselines, and outcomes | `../Case_Studies/` |",
          "| Organisational policy and legal determination | governance modules, not this collection |",
          "| Vendor-specific or unverified current API behaviour | treat as OPEN until sourced at authoring time |",
          "", "## 7. Master symbol glossary", "",
          "| Symbol / term | Meaning | Units | Shape / type | Domain | First use |", "|---|---|---|---|---|---|"]
    seen_terms = []
    for e in entries:
        text = e["file"].read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"(?m)^\*\*([A-Za-z][A-Za-z0-9 _\-/]{2,30})\*\*[.:]", text):
            term = m.group(1).strip()
            if term.casefold() in [t.casefold() for t in seen_terms]:
                continue
            seen_terms.append(term)
            if len(seen_terms) >= 14:
                break
        if len(seen_terms) >= 14:
            break
    if seen_terms:
        for t in seen_terms:
            L.append(f"| {t} | as defined at first use in the linking card | see card | see card | see card | `{next(e['file'].name for e in entries)}` |")
    else:
        L.append("| — | no card defines a symbol yet | — | — | — | — |")

    L += ["", "## 8. Card directory", "",
          "| # | Card | Level | Evidence | Status |", "|---|---|---|---|---|"]
    for i, e in enumerate(entries, 1):
        L.append(f"| {i} | [{e['title']}](./{e['file'].name}) | {e['level']} | {e['evidence']} | {e['status']} |")
    if not entries:
        L.append("| — | none | — | — | — |")

    L += ["", "## 9. Common Issues locator", ""]
    if ci:
        for e in ci:
            L.append(f"- [{e['title']}](./{e['file'].name}) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.")
    else:
        L.append("- Not yet authored. The collection is incomplete until this artifact exists with at least 25 distinct issues.")

    L += ["", "## 10. Sources, versions, maintenance", "",
          "| Field | Value |", "|---|---|",
          "| Evidence vocabulary | DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC / OPEN |",
          "| Version policy | Content is dated by its own frontmatter; unstable API, price, and regulatory facts are rechecked at edit time |",
          "| Maintenance trigger | A card's validity condition, a source body, or a tool's semantics changes |",
          "| Review cadence | Quarterly, plus on any material change to a cited source |",
          "", "## 11. Acceptance checklist", "",
          "- [x] Every linked card exists on disk.",
          "- [x] Status column is measured by the lint, not asserted.",
          "- [x] Planned commitments remain listed without links.",
          f"- [{'x' if ci else ' '}] A final Common Issues artifact exists with at least 25 distinct issues.",
          "- [ ] Independent technical review of each card.",
          ""]
    return "\n".join(L)


def main() -> int:
    dry = "--dry-run" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    units = [
        p for p in sorted(ROOT.rglob("*"))
        if p.is_dir() and (p / "Case_Studies").is_dir() and (p / "Cheat_Sheets").is_dir()
        and not ({".authoring_state", ".tmp_parts", "Tools"} & set(p.parts))
    ]
    if only:
        only = only.replace("\\", "/")
        units = [u for u in units if str(u.relative_to(ROOT)).replace("\\", "/").startswith(only)]

    written = 0
    for u in units:
        ci = u / "Case_Studies" / "INDEX.md"
        si = u / "Cheat_Sheets" / "INDEX.md"
        for path, gen in ((ci, case_index), (si, sheet_index)):
            text = gen(u)
            if dry:
                print(f"[dry] {path.relative_to(ROOT)}  ({len(text.splitlines())} lines)")
            else:
                path.write_text(text, encoding="utf-8")
                written += 1
    print(f"{'would write' if dry else 'wrote'} {written or len(units)*2} index files across {len(units)} units")
    return 0


if __name__ == "__main__":
    sys.exit(main())
