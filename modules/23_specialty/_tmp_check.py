import re
import subprocess
import sys
import tempfile
from pathlib import Path

md = Path(sys.argv[1]).read_text(encoding="utf-8")
start = md.index("## 7.")
end = md.index("## 8.")
section = md[start:end]

pattern = re.compile(r"```python\n(.*?)```\n\nExpected output:\n\n```text\n(.*?)```", re.S)
blocks = pattern.findall(section)
print(f"found {len(blocks)} code blocks")
ok = True
for i, (code, expected) in enumerate(blocks, 1):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    proc = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True)
    actual = proc.stdout.replace("\r\n", "\n").strip()
    exp = expected.strip()
    match = actual == exp and proc.returncode == 0
    ok = ok and match
    print(f"block {i}: returncode={proc.returncode} match={match}")
    if not match:
        print("--- expected ---")
        print(exp)
        print("--- actual ---")
        print(actual)
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr)
print("ALL MATCH" if ok else "MISMATCH")
