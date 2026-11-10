#!/usr/bin/env python3
"""Repair relative Markdown links whose target exists but whose path is wrong.

Authors routinely overshoot with `../../` when a sibling lives one level up. This
tool rewrites a link only when exactly one file with the same basename exists in a
small neighbourhood (the linking file's directory, the unit directory, or the unit
directory's parent). Ambiguous or genuinely absent targets are left alone and
reported, never guessed.

    python Tools/fix_links.py --check     # report only
    python Tools/fix_links.py --write     # rewrite what is unambiguous
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[([^\]]*)\]\((?!https?://|mailto:|#)([^)#]+?)(#[^)]*)?\)")
SKIP = {".authoring_state", ".tmp_parts", "Tools", ".git"}


def markdown_files():
    for p in ROOT.rglob("*.md"):
        if not (set(p.relative_to(ROOT).parts) & SKIP):
            yield p


def candidates(linker: Path, base: str) -> list[Path]:
    """Files named `base` near the linking file, most specific first."""
    unit = linker.parent
    # a collection file lives in <unit>/<Collection>/; the unit is one or two levels up
    dirs = [unit, unit.parent, unit.parent.parent]
    found: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for name in (base,):
            cand = d / name
            if cand.is_file():
                found.append(cand)
        # also search that directory's immediate subdirectories
        for sub in d.iterdir():
            if sub.is_dir() and sub not in dirs:
                cand = sub / base
                if cand.is_file():
                    found.append(cand)
    uniq = []
    for f in found:
        if f.resolve() not in {u.resolve() for u in uniq}:
            uniq.append(f)
    return uniq


def scan(write: bool) -> int:
    broken = fixed = unresolved = 0
    for md in markdown_files():
        text = md.read_text(encoding="utf-8", errors="replace")
        changed = False
        for m in list(LINK.finditer(text)):
            label, target, anchor = m.group(1), m.group(2).strip(), m.group(3) or ""
            if not target.endswith(".md"):
                continue
            resolved = (md.parent / target).resolve()
            if resolved.is_file():
                continue
            broken += 1
            base = Path(target).name
            cands = candidates(md, base)
            if len(cands) != 1:
                unresolved += 1
                print(f"UNRESOLVED {md.relative_to(ROOT)}: {target} ({len(cands)} candidates)")
                continue
            rel = os.path.relpath(cands[0], md.parent).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            text = text.replace(m.group(0), f"[{label}]({rel}{anchor})")
            changed = True
            fixed += 1
            print(f"{'FIX' if write else 'WOULD FIX'} {md.relative_to(ROOT)}: {target} -> {rel}")
        if write and changed:
            md.write_text(text, encoding="utf-8")
    print(f"\nbroken={broken} rewritable={fixed} unresolved={unresolved}")
    return 0


def main() -> int:
    if "--write" not in sys.argv and "--check" not in sys.argv:
        print(__doc__)
        return 2
    return scan("--write" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
