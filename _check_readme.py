"""README index <-> disk cross-check, both directions."""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai-system-design-mastery")
os.chdir(ROOT)

disk = set()
for dp, dn, fns in os.walk("."):
    dn[:] = [d for d in dn if not d.startswith(".")]
    for fn in fns:
        if fn.endswith(".md"):
            r = os.path.relpath(os.path.join(dp, fn), ".").replace(os.sep, "/")
            if r != "README.md":
                disk.add(r)

rd = open("README.md", encoding="utf-8").read()
linked = set()
for m in re.finditer(r"\]\(([^)]+\.md)(?:#[^)]*)?\)", rd):
    t = m.group(1)
    if t.startswith("http"):
        continue
    norm = os.path.normpath(t).replace(os.sep, "/")
    linked.add(norm)

missing = sorted(f for f in disk if f not in linked)
extra = sorted(l for l in linked if l not in disk)
print("disk md (excl README):", len(disk))
print("distinct md linked   :", len(linked))
print("on disk, not in README:", len(missing))
for m in missing:
    print("   MISSING ROW:", m)
print("in README, not on disk:", len(extra))
for e in extra:
    print("   DEAD ROW:", e)
