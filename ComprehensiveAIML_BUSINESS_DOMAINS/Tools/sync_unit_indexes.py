#!/usr/bin/env python3
"""Reconcile each unit's topic-map Status column with the files that actually exist.

A topic is marked `authored` only when an existing manuscript in that unit's
collections covers it (title-token overlap against the topic name). Everything
else becomes `planned`. Never invents a link.

    python Tools/sync_unit_indexes.py --dry-run
    python Tools/sync_unit_indexes.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".authoring_state", ".tmp_parts", "Tools", "Templates"}
STOP = {"and", "the", "for", "with", "of", "in", "to", "a", "an", "on", "data", "systems", "system", "ai"}


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.casefold()) if w not in STOP and len(w) > 2}


def units() -> list[Path]:
    return [
        p for p in sorted(ROOT.rglob("*"))
        if p.is_dir() and (p / "Case_Studies").is_dir() and (p / "Cheat_Sheets").is_dir()
        and not (SKIP & set(p.parts))
    ]


def coverage(unit: Path) -> list[tuple[Path, set[str]]]:
    out = []
    for sub in ("Case_Studies", "Cheat_Sheets"):
        for f in sorted((unit / sub).glob("*.md")):
            if f.name == "INDEX.md":
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), f.stem)
            out.append((f, tokens(title + " " + f.stem)))
    return out


def main() -> int:
    dry = "--dry-run" in sys.argv
    touched = 0
    for unit in units():
        idx = unit / "INDEX.md"
        if not idx.exists():
            continue
        text = idx.read_text(encoding="utf-8", errors="replace")
        cov = coverage(unit)
        if not cov:
            continue

        def repl(m: re.Match) -> str:
            num, topic, _status = m.group(1), m.group(2), m.group(3)
            tt = tokens(topic)
            if not tt:
                return m.group(0)
            best = None
            for f, ft in cov:
                if not ft:
                    continue
                overlap = len(tt & ft) / len(tt)
                if overlap >= 0.5 and (best is None or overlap > best[0]):
                    best = (overlap, f)
            if best:
                f = best[1]
                rel = f"{f.parent.name}/{f.name}"
                return f"| {num} | {topic} | [authored]({rel}) |"
            return f"| {num} | {topic} | planned |"

        new = re.sub(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$", repl, text, flags=re.M)
        if new != text:
            touched += 1
            if dry:
                print(f"[dry] {idx.relative_to(ROOT)}")
            else:
                idx.write_text(new, encoding="utf-8")
    print(f"{'would touch' if dry else 'touched'} {touched} unit indexes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
