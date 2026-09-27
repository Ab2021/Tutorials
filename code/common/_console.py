"""
_console — one definition of the Windows console fix, reachable from both
`import common` and from a direct `python common/memory.py`.

The scripts print box-drawing and warning glyphs (✅ ⚠ ─ → §). Windows consoles
default to cp1252, which cannot encode any of them, so the first status line
raises UnicodeEncodeError and the script dies — a genuinely confusing way to
fail, because the code is fine and only the terminal is wrong.

Why this is a separate module rather than a function in `__init__.py`:
`memory.py` lives INSIDE the package, and the README's quickstart tells you to
run it as a file (`python common/memory.py --table`). Run that way it is
`__main__`, so the package `__init__` never executes and the fix never applies —
which is exactly the case that used to crash. Keeping the fix in a leaf module
lets both entry points share one implementation instead of drifting apart.
"""

from __future__ import annotations

import sys


def force_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 when they are not already.

    Safe to call more than once, and safe to call when the fix is impossible:
    `sys.stdout.reconfigure` needs Python 3.7+ and a TextIOWrapper, so a
    redirected stream, a pipe, or an exotic wrapper is left alone rather than
    raising. Losing the fix is acceptable; crashing while applying it is not.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc and enc.lower().replace("-", "") not in ("utf8", "utf8mb4"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass
