"""
common — shared utilities for the Fine-Tuning Handbook codebase.

Importing this package applies a small but important fix: the scripts print
box-drawing and warning glyphs (✅ ⚠ → §), and Windows consoles default to
cp1252, which cannot encode them. Without this, every script crashes on Windows
the moment it prints a status line — which is a confusing way to fail.

The implementation lives in `._console` so that `common/memory.py` can apply the
same fix when it is run directly as a file (see that module's docstring).
"""

from __future__ import annotations

from ._console import force_utf8

# Kept as a module-level name because it documents the side effect at the import
# site; the real work is in `_console.force_utf8`.
_force_utf8_console = force_utf8

_force_utf8_console()
