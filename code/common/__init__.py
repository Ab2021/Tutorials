"""
common — shared utilities for the Fine-Tuning Handbook codebase.

Importing this package applies a small but important fix: the scripts print
box-drawing and warning glyphs (✅ ⚠ → §), and Windows consoles default to cp1252,
which cannot encode them. Without this, every script crashes on Windows the moment it
prints a status line — which is a confusing way to fail.

`sys.stdout.reconfigure` needs Python 3.7+ and a TextIOWrapper; both are wrapped.
"""

from __future__ import annotations

import sys


def _force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc and enc.lower().replace("-", "") not in ("utf8", "utf8mb4"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                # Redirected to a file, a pipe, or an exotic wrapper — leave it alone.
                pass


_force_utf8_console()
