#!/usr/bin/env python3
"""Per-module coverage report.

Answers the only question that matters at the end: for each of the eleven parts,
are the submodules' manuscripts actually accepted, and what is still missing?

    python Tools/coverage_report.py
    python Tools/coverage_report.py --part 1
    python Tools/coverage_report.py --md > COVERAGE_STATUS.md
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Tools"))
from manuscript_lint import lint  # noqa: E402

SKIP = {".authoring_state", ".tmp_parts", "Tools", "Templates"}


def units() -> list[Path]:
    return [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_dir()
        and (p / "Case_Studies").is_dir()
        and (p / "Cheat_Sheets").is_dir()
        and not (SKIP & set(p.parts))
    ]


def part_key(unit: Path) -> int:
    head = unit.relative_to(ROOT).parts[0]
    try:
        return int(head.split("_")[0])
    except (ValueError, IndexError):
        return 0


def measure(unit: Path) -> dict:
    cases = sorted(f for f in (unit / "Case_Studies").glob("*.md") if f.name != "INDEX.md")
    sheets = sorted(f for f in (unit / "Cheat_Sheets").glob("*.md") if f.name != "INDEX.md")
    case_ok = sum(lint(f)["ok"] for f in cases)
    sheet_ok = sum(lint(f)["ok"] for f in sheets)
    ci = [f for f in sheets if f.name.endswith("Common_Issues.md")]
    ci_ok = sum(lint(f)["ok"] for f in ci)
    return {
        "cases": len(cases),
        "cases_ok": case_ok,
        "sheets": len(sheets),
        "sheets_ok": sheet_ok,
        "ci": len(ci),
        "ci_ok": ci_ok,
        "complete": bool(cases) and case_ok == len(cases) and ci_ok == 1 and sheet_ok == len(sheets),
    }


def main() -> int:
    want = None
    if "--part" in sys.argv:
        want = int(sys.argv[sys.argv.index("--part") + 1])
    md = "--md" in sys.argv

    by_part: dict[int, list[tuple[Path, dict]]] = defaultdict(list)
    for u in units():
        k = part_key(u)
        if want is not None and k != want:
            continue
        by_part[k].append((u, measure(u)))

    totals = {"cases": 0, "cases_ok": 0, "sheets": 0, "sheets_ok": 0, "ci": 0, "ci_ok": 0, "complete": 0, "units": 0}
    rows = []
    for k in sorted(by_part):
        agg = defaultdict(int)
        done = 0
        for u, m in by_part[k]:
            for key in totals:
                if key in m:
                    agg[key] += int(m[key]) if isinstance(m[key], bool) else m[key]
            done += m["complete"]
        head = next(iter(by_part[k]))[0].relative_to(ROOT).parts[0]
        rows.append((k, head, agg, done))
        for key in totals:
            if key in agg:
                totals[key] += agg[key]
        totals["complete"] += done
        totals["units"] += len(by_part[k])

    if md:
        print("# Coverage status\n")
        print("Generated from the filesystem: a manuscript counts only when "
              "`Tools/manuscript_lint.py` exits 0 for it.\n")
        print("| Part | Units complete | Cases accepted | Sheets accepted | Common Issues |")
        print("|---|---|---|---|---|")
        for k, head, agg, done in rows:
            print(f"| {head} | {done}/{len(by_part[k])} | {agg['cases_ok']}/{agg['cases']} | "
                  f"{agg['sheets_ok']}/{agg['sheets']} | {agg['ci_ok']}/{agg['ci']} |")
        print(f"| **Total** | **{totals['complete']}/{totals['units']}** | **{totals['cases_ok']}/{totals['cases']}** | "
              f"**{totals['sheets_ok']}/{totals['sheets']}** | **{totals['ci_ok']}/{totals['ci']}** |")
        return 0

    for k, head, agg, done in rows:
        print(f"{head}")
        print(f"   units complete {done}/{len(by_part[k])}   cases {agg['cases_ok']}/{agg['cases']}   "
              f"sheets {agg['sheets_ok']}/{agg['sheets']}   CI {agg['ci_ok']}/{agg['ci']}")
    print(f"TOTAL units {totals['complete']}/{totals['units']}  cases {totals['cases_ok']}/{totals['cases']}  "
          f"sheets {totals['sheets_ok']}/{totals['sheets']}  CI {totals['ci_ok']}/{totals['ci']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
