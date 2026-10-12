"""Structural gate for ai-system-design-mastery.

Checks, per markdown file:
  - section count under the 12/13-section house shape
  - numbered Q&A count (>= 12 for content files)
  - pushback count (>= 3 for content files)
  - Q numbering is contiguous from 1
  - placeholder strings

Usage:  python _check_structure.py [--verbose]
"""

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai-system-design-mastery")

# Files exempt from the Q&A/pushback bar, with the reason.
EXEMPT = {
    "README.md": "index",
    "00-meta/01-how-to-use-and-template.md": "meta",
    "00-meta/02-source-topic-map.md": "meta",
}

# Two shapes in use:  **Q5. …**  (content files)  and  ### Q5. …  (question banks)
Q_RE = re.compile(r"^(?:\*\*Q|#{2,4} Q)(\d+)\b", re.M)
# Pushbacks appear in four shapes across the repo:
#   **Q5 (pushback). ...            **Q10. *Pushback:* "..."
#   **Q8 (medium). *Pushback:* ...  ... ⚡  /  a "### Pushback round" section
PB_ANY_RE = re.compile(r"⚡|pushback", re.I)
Q_LINE_RE = re.compile(r"^(?:\*\*Q\d+|#{2,4} Q\d+).*$", re.M)
SEC_RE = re.compile(r"^## ", re.M)
PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|lorem ipsum)\b", re.I)


def main():
    verbose = "--verbose" in sys.argv
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".md"):
                files.append(os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/"))
    files.sort()

    problems = []
    for rel in files:
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        is_cheatsheet = rel.startswith("Cheat-Sheets/")
        exempt = rel in EXEMPT

        qs = [int(n) for n in Q_RE.findall(text)]
        # A pushback is any of:
        #   - a Q-line flagged as one:  **Q5 (pushback). / **Q10. *Pushback:* "…"
        #   - a P-item in a "### Pushback round" section:  **P1. …**
        #   - a bare ⚡ marker
        q_lines = Q_LINE_RE.findall(text)
        pb_in_q = sum(1 for ln in q_lines if PB_ANY_RE.search(ln))
        p_items = len(re.findall(r"^\*\*P\d+\b", text, re.M))
        bolts = len(re.findall(r"⚡", text))
        pbs = pb_in_q + p_items + max(0, bolts - pb_in_q)
        secs = len(SEC_RE.findall(text))
        ph = PLACEHOLDER_RE.findall(text)

        notes = []
        if qs and qs != list(range(1, len(qs) + 1)):
            notes.append(f"Q numbering not contiguous: {qs}")
        if not exempt and not is_cheatsheet:
            if len(qs) < 12:
                notes.append(f"only {len(qs)} Q&A (bar: 12)")
            if pbs < 3:
                notes.append(f"only {pbs} pushback(s) (bar: 3)")
        if secs < 9:
            notes.append(f"only {secs} sections")
        if ph:
            notes.append(f"placeholder text: {set(ph)}")

        if notes:
            problems.append((rel, notes))
        if verbose:
            print(f"{rel:70s} secs={secs:2d} Q={len(qs):2d} pb={pbs}")

    print()
    print(f"files checked : {len(files)}")
    if problems:
        print(f"files with issues : {len(problems)}")
        for rel, notes in problems:
            print(f"  {rel}")
            for n in notes:
                print(f"      - {n}")
    else:
        print("structural issues : NONE")


if __name__ == "__main__":
    main()
