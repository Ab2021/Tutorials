"""Re-derive the README headline counts from disk."""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai-system-design-mastery")

Q_RE = re.compile(r"^(?:\*\*Q|#{2,4} Q)(\d+)\b", re.M)
P_RE = re.compile(r"^\*\*P\d+\b", re.M)

files = 0
lines = 0
qa = 0
pb = 0
for dp, dn, fns in os.walk(ROOT):
    dn[:] = [d for d in dn if not d.startswith(".")]
    for fn in sorted(fns):
        if not fn.endswith(".md"):
            continue
        p = os.path.join(dp, fn)
        t = open(p, encoding="utf-8").read()
        n = len(t.splitlines())
        lines += n
        if fn != "README.md":
            files += 1
        qa += len(Q_RE.findall(t))
        pb += len(P_RE.findall(t))

print(f"content files (excl README) : {files}")
print(f"total md files              : {files + 1}")
print(f"total lines (all md)        : {lines:,}")
print(f"numbered Q&A items          : {qa:,}")
print(f"pushback-round items (P*)   : {pb:,}")
print(f"Q&A + pushbacks             : {qa + pb:,}")
