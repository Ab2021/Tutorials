import ast
import io
import subprocess
import sys
from pathlib import Path

md = Path(sys.argv[1]).read_text(encoding="utf-8")
lines = md.splitlines()

blocks = []
i = 0
while i < len(lines):
    if lines[i].startswith("```python"):
        j = i + 1
        body = []
        while not lines[j].startswith("```"):
            body.append(lines[j])
            j += 1
        blocks.append((i + 1, "\n".join(body)))
        i = j
    i += 1

print("python blocks:", len(blocks))
for start, src in blocks:
    ast.parse(src)
    proc = subprocess.run([sys.executable, "-I", "-c", src],
                          capture_output=True, text=True)
    print("=" * 60)
    print("block at line", start, "rc", proc.returncode)
    print(proc.stdout, end="")
    if proc.stderr:
        print("STDERR:", proc.stderr)
