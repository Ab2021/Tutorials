import subprocess
import sys
from pathlib import Path

md = Path(sys.argv[1]).read_text(encoding="utf-8")
lines = md.splitlines()

i = next(k for k, l in enumerate(lines) if l.startswith("```python"))
j = i + 1
src = []
while not lines[j].startswith("```"):
    src.append(lines[j])
    j += 1
src = "\n".join(src)

k = next(x for x in range(j, len(lines)) if lines[x].strip().lower().startswith("expected output"))
k = next(x for x in range(k, len(lines)) if lines[x].startswith("```text"))
m = k + 1
exp = []
while not lines[m].startswith("```"):
    exp.append(lines[m])
    m += 1
exp = "\n".join(exp)

out = subprocess.run([sys.executable, "-I", "-c", src],
                     capture_output=True, text=True)
got = out.stdout.rstrip("\n")
exp = exp.rstrip("\n")
print("MATCH" if got == exp else "MISMATCH")
if got != exp:
    import difflib
    for line in difflib.unified_diff(exp.splitlines(), got.splitlines(),
                                     "expected", "actual", lineterm=""):
        print(line)
