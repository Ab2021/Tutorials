#!/usr/bin/env python3
"""Integration verifier: link integrity, heading contract, and coverage accounting.

    python Tools/verify_course.py            # summary + failures
    python Tools/verify_course.py --links    # only link check
    python Tools/verify_course.py --tree 11_ # scope to one subtree
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Tools"))
from manuscript_lint import lint  # noqa: E402

LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\s]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#")


def _is_link_target(target: str) -> bool:
    """A real Markdown link target, not a prose parenthetical such as `(0.05)`."""
    if not target or target.startswith(SKIP_PREFIX):
        return False
    return "/" in target or target.endswith((".md", ".txt", ".json", ".png", ".svg", ".csv"))


def md_files(scope: Path | None) -> list[Path]:
    base = scope or ROOT
    return [
        p
        for p in sorted(base.rglob("*.md"))
        if ".authoring_state" not in p.parts and ".tmp_parts" not in p.parts and "Tools" not in p.parts
    ]


def check_links(files: list[Path]) -> list[tuple[str, str, str]]:
    broken = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        # strip fenced code so example links inside code are ignored
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for raw in LINK.findall(text):
            target = raw.split("#", 1)[0]
            if not _is_link_target(target):
                continue
            if target.startswith("/"):
                resolved = ROOT / target.lstrip("/")
            else:
                from urllib.parse import unquote

                resolved = (p.parent / unquote(target)).resolve()
            if not resolved.exists():
                broken.append((str(p.relative_to(ROOT)), raw, str(resolved)))
    return broken


def main() -> int:
    scope = None
    if "--tree" in sys.argv:
        scope = ROOT / sys.argv[sys.argv.index("--tree") + 1]
    files = md_files(scope)

    broken = check_links(files)
    print(f"files: {len(files)}  broken links: {len(broken)}")
    for b in broken[:25]:
        print(f"  {b[0]}: {b[1]}")

    if "--links" in sys.argv:
        return 1 if broken else 0

    results = [lint(p) for p in files if p.name != "AUTHORING_PROGRESS.md"]
    manuscripts = [r for r in results if r["kind"] in ("case", "sheet")]
    passing = [r for r in manuscripts if r["ok"]]
    print(
        f"manuscripts: {len(manuscripts)}  pass: {len(passing)}  "
        f"fail: {len(manuscripts) - len(passing)}"
    )
    from collections import Counter

    print("failures by reason:", Counter(
        p.split(":")[0][:45] for r in manuscripts if not r["ok"] for p in r["problems"]
    ).most_common(6))
    return 0 if not broken and len(passing) == len(manuscripts) else 1


if __name__ == "__main__":
    sys.exit(main())
