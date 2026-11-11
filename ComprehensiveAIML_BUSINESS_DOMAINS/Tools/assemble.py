#!/usr/bin/env python3
"""Assemble a manuscript from ordered part files, then lint the result.

Agents cannot emit a 1,000+ line manuscript in one response, so they write
ordered parts to the OS temp directory and call this to produce the real file.
The target is written once and atomically, so an interrupted run never leaves a
truncated manuscript in the repository.

    python Tools/assemble.py <batch> <target.md>            # assemble + lint
    python Tools/assemble.py <batch> <target.md> --no-lint
    python Tools/assemble.py --status <batch>               # list parts

Part files live in `<temp>/authoring/<batch>/` and are ordered by filename
(`part01.md`, `part02.md`, ...). Non-markdown files are ignored.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def parts_dir(batch: str) -> Path:
    return Path(tempfile.gettempdir()) / "authoring" / batch


def assemble(batch: str, target: Path, lint: bool = True) -> int:
    d = parts_dir(batch)
    files = sorted(p for p in d.glob("*.md"))
    if not files:
        print(f"no part files in {d}")
        return 2

    chunks, total = [], 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        total += text.count("\n")
        chunks.append(text)
    body = "".join(chunks)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".assemble-", dir=target.parent) as scratch:
        staging = Path(scratch) / target.name
        staging.write_text(body, encoding="utf-8")
        if lint:
            result = subprocess.run(
                [sys.executable, str(TOOLS / "manuscript_lint.py"), str(staging)],
                cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
            )
            sys.stdout.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
            if result.returncode:
                print(f"rejected candidate; target unchanged: {target}")
                return result.returncode
        os.replace(staging, target)
    print(f"assembled {len(files)} parts -> {target} ({total} lines)")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    if "--status" in sys.argv:
        d = parts_dir(args[0])
        files = sorted(p for p in d.glob("*.md"))
        if not files:
            print(f"no parts in {d}")
            return 0
        for p in files:
            print(f"{p.name:12s} {p.stat().st_size:7d} B  {p.read_text(encoding='utf-8').count(chr(10)):5d} lines")
        print(f"total lines: {sum(p.read_text(encoding='utf-8').count(chr(10)) for p in files)}")
        return 0
    if len(args) < 2:
        print("usage: assemble.py <batch> <target.md> [--no-lint]")
        return 2
    return assemble(args[0], (ROOT / args[1]).resolve(), lint="--no-lint" not in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
