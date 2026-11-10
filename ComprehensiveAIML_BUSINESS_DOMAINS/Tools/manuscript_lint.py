#!/usr/bin/env python3
"""Manuscript acceptance lint for ComprehensiveAIML_BUSINESS_DOMAINS.

Enforces the anti-padding contract in 00_AGENT_AUTHORING_CONTRACT.md:
headings, order, distinct-content floors, no repeated long prose, no placeholders.

Usage:
    python manuscript_lint.py FILE [FILE ...]     # exit 1 if any file fails
    python manuscript_lint.py --tree DIR          # lint every .md below DIR
    python manuscript_lint.py --json FILE         # machine-readable result
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

MIN_PHYSICAL_LINES = 950
MIN_DISTINCT_PROSE = 680
MAX_LONG_LINE_REPEAT = 2
MAX_DUP_LONG_RATIO = 0.15
LONG_LINE_CHARS = 70
INDEX_MIN_LINES = 40

CASE_HEADINGS = [
    "## 1. The One-Line Framing",
    "## 2. Context & Background",
    "## 3. The Business Question",
    "## 4. Assumptions & Constraints",
    "## 5. Data, Tools & Method",
    "## 6. Analysis Walkthrough",
    "## 7. The Finding",
    "## 8. System/Solution Design",
    "## 9. Evaluation & Validation",
    "## 10. Action Taken",
    "## 11. Outcome & Results",
    "## 12. Failure Modes & What Went Wrong",
    "## 13. Transferable Lesson",
    "## 14. Connections & Cross-References",
    "## 15. Quick Reference Card",
]

SHEET_HEADINGS = [
    "## 1. File Header",
    "## 2. Concept Map / Mental Model",
    "## 3. Foundational Definitions",
    "## 4. Core Formulas with Full Derivation",
    "## 5. Worked Numerical Example",
    "## 6. Visual / Geometric Intuition",
    "## 7. Comparison / Selection Table",
    "## 8. Failure Modes & Edge Cases",
    "## 9. Complexity & Cost Analysis",
    "## 10. Key Theorems / Results Reference",
    "## 11. Interview / Exam Questions",
    "## 12. Quick Reference Card",
]

PLACEHOLDER = re.compile(
    r"(?i)\b(TODO|TBD|FIXME|<placeholder>|content omitted|lorem ipsum|\[insert)\b"
)
FENCE = re.compile(r"^```")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _normalise(heading: str) -> str:
    h = heading.rstrip().rstrip("#").strip()
    h = re.sub(r"\s+", " ", h)
    return h.casefold()


def classify(path: Path, text: str) -> str:
    low = path.as_posix().casefold()
    if low.endswith("index.md") or path.name == "AUTHORING_PROGRESS.md":
        return "index"
    for line in text.splitlines()[:40]:
        if line.strip().casefold().startswith("case_id:"):
            return "case"
    if "Common_Issues" in path.name or "/Cheat_Sheets/" in path.as_posix():
        return "sheet"
    return "other"


def ordered_present(found: list[str], required: list[str]) -> tuple[bool, list[str], list[str]]:
    """Return (ok, missing, misordered) using casefolded exact matches."""
    norm_found = [_normalise(h) for h in found]
    missing = [r for r in required if _normalise(r) not in norm_found]
    positions = []
    for r in required:
        target = _normalise(r)
        if target in norm_found:
            positions.append(norm_found.index(target))
    misordered = []
    for a, b in zip(positions, positions[1:]):
        if b < a:
            misordered.append(required[positions.index(b)] if b in positions else required[-1])
    ok = not missing and positions == sorted(positions)
    return ok, missing, misordered


def lint(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    body: list[str] = []
    headings: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            m = HEADING.match(line)
            if m:
                headings.append(line.strip())
                continue
        body.append(line)

    prose = [l.strip() for l in body if l.strip()]
    long_lines = [l for l in prose if len(l) >= LONG_LINE_CHARS]
    counter = Counter(long_lines)
    dup_groups = {k: v for k, v in counter.items() if v > MAX_LONG_LINE_REPEAT}
    dup_lines = sum(v - 1 for v in counter.values() if v > 1)
    dup_ratio = (dup_lines / len(long_lines)) if long_lines else 0.0

    kind = classify(path, text)
    problems: list[str] = []
    warnings: list[str] = []

    h2 = [h for h in headings if h.startswith("## ")]
    if kind == "case":
        ok, missing, misordered = ordered_present(h2, CASE_HEADINGS)
        if missing:
            problems.append(f"missing case headings: {missing}")
        if not ok and not missing:
            problems.append("case headings out of order")
        if len([h for h in h2 if re.match(r"^## \d+\.", h)]) != 15:
            problems.append("expected exactly 15 numbered H2 sections")
        if not re.search(r"(?m)^#{3,4}\s*(?:\d+[\.\d]*[a-z]?\.?\s*)?Interview pack\b", text):
            problems.append("missing '### Interview pack' subsection under section 14")
    elif kind == "sheet":
        ok, missing, misordered = ordered_present(h2, SHEET_HEADINGS)
        if missing:
            problems.append(f"missing sheet headings: {missing}")
        if not ok and not missing:
            problems.append("sheet headings out of order")
        base = [h for h in h2 if re.match(r"^## \d+\.", h)]
        if len(base) != 12:
            problems.append("expected exactly 12 numbered H2 sections")
        if "Common_Issues" in path.name:
            issues = len(re.findall(r"(?m)^\|\s*(?:`)?CI-\d+", text))
            if issues < 25:
                problems.append(f"Common Issues entries found {issues}, need >=25 (IDs starting CI-)")
    elif kind == "index":
        if len(lines) < INDEX_MIN_LINES:
            warnings.append(f"index shorter than {INDEX_MIN_LINES} lines ({len(lines)})")

    if kind in ("case", "sheet"):
        if len(lines) < MIN_PHYSICAL_LINES:
            problems.append(f"physical lines {len(lines)} < {MIN_PHYSICAL_LINES}")
        if len(prose) < MIN_DISTINCT_PROSE:
            problems.append(f"distinct prose lines {len(prose)} < {MIN_DISTINCT_PROSE}")
        if dup_groups:
            worst = sorted(dup_groups.items(), key=lambda kv: -kv[1])[:3]
            problems.append(
                f"{len(dup_groups)} long line(s) repeated >{MAX_LONG_LINE_REPEAT}x, worst: "
                + "; ".join(f"x{v}: {k[:60]!r}" for k, v in worst)
            )
        if dup_ratio > MAX_DUP_LONG_RATIO:
            problems.append(f"duplicate long-line ratio {dup_ratio:.3f} > {MAX_DUP_LONG_RATIO}")

    for m in PLACEHOLDER.finditer(text):
        problems.append(f"placeholder marker {m.group(0)!r}")
        break

    if kind == "case":
        for marker in ("FORMULA", "CONSTRAINT", "NUMERICAL EXAMPLE"):
            if text.count(marker) < 3:
                problems.append(f"fewer than 3 '{marker}' derivation blocks in walkthrough")
                break

    return {
        "file": str(path),
        "kind": kind,
        "lines": len(lines),
        "distinct_prose": len(prose),
        "dup_ratio": round(dup_ratio, 4),
        "long_repeat_groups": len(dup_groups),
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
    }


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    targets: list[Path] = []
    if "--tree" in argv:
        root = Path(argv[argv.index("--tree") + 1])
        targets = sorted(p for p in root.rglob("*.md") if ".tmp_parts" not in p.parts)
    else:
        targets = [Path(a) for a in argv if not a.startswith("--")]
    if not targets:
        print(__doc__)
        return 2

    results = [lint(p) for p in targets if p.exists()]
    missing = [str(p) for p in targets if not p.exists()]
    for m in missing:
        results.append({"file": m, "ok": False, "problems": ["file not found"], "warnings": []})

    if as_json:
        print(json.dumps(results, indent=2))
    else:
        fails = 0
        for r in results:
            if r["ok"]:
                print(
                    f"PASS {r['file']}  ({r['kind']}, {r['lines']} lines, "
                    f"{r['distinct_prose']} prose, dup {r['dup_ratio']:.3f})"
                )
            else:
                fails += 1
                print(f"FAIL {r['file']}  ({r.get('kind')}, {r.get('lines')} lines, dup {r.get('dup_ratio')})")
                for p in r["problems"]:
                    print(f"   - {p}")
            for w in r.get("warnings", []):
                print(f"   ~ {w}")
        print(f"\n{len(results) - fails}/{len(results)} passed")
        return 1 if fails else 0
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
