#!/usr/bin/env python3
"""Resumable dispatch helper for authoring batches.

- `--next N`   print prompts for the N highest-priority unfinished batches
- `--verify`   re-lint every queued artifact on disk and update state
- `--state`    print progress summary

State lives in `.tmp_parts/state.json` and is rebuilt from the filesystem by
`--verify`, so a resumed session never trusts a prior run's claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".tmp_parts"
sys.path.insert(0, str(ROOT / "Tools"))
from manuscript_lint import lint  # noqa: E402


def load(name: str):
    p = TMP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(name: str, obj) -> None:
    (TMP / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def verify() -> dict:
    tasks = {t["id"]: t for t in (json.loads(l) for l in (TMP / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    state = load("state.json") or {"tasks": {}}
    done = failed = missing = 0
    for tid, t in tasks.items():
        path = ROOT / t["path"]
        if not path.exists():
            state["tasks"][tid] = "missing"
            missing += 1
            continue
        if t["kind"] in ("index_cases", "index_sheets", "unit_index"):
            txt = path.read_text(encoding="utf-8", errors="replace")
            need = {"index_cases": 7, "index_sheets": 11}.get(t["kind"])
            if need:
                got = len([l for l in txt.splitlines() if l.startswith("## ") and l[3:4].isdigit()])
                ok = got >= need
            else:
                ok = True
            state["tasks"][tid] = "done" if ok else "partial"
            done += ok
            failed += not ok
            continue
        r = lint(path)
        state["tasks"][tid] = "done" if r["ok"] else "failed"
        done += r["ok"]
        failed += not r["ok"]
    save("state.json", state)
    return {"done": done, "failed": failed, "missing": missing, "total": len(tasks)}


def next_batches(n: int) -> list[dict]:
    batches = load("batches.json")["batches"]
    state = (load("state.json") or {}).get("tasks", {})
    claimed = load("claimed.json") or {}
    tasks = {t["id"]: t for t in (json.loads(l) for l in (TMP / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    out = []
    for b in batches:
        if all(state.get(t) == "done" for t in b["tasks"]):
            continue
        if b["batch"] in claimed:
            continue
        prio = min(tasks[t]["priority"] for t in b["tasks"])
        out.append((prio, b))
    out.sort(key=lambda x: x[0])
    return [b for _, b in out[:n]]


def prompt_for(batch: dict) -> str:
    tasks = {t["id"]: t for t in (json.loads(l) for l in (TMP / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    lines = [
        f"You are authoring batch {batch['batch']} for the course at D:/Biomedical/Train/ComprehensiveAIML_BUSINESS_DOMAINS",
        "",
        "STEP 1 (mandatory, do not skip): read `00_AGENT_AUTHORING_CONTRACT.md` at the course root, then run",
        "`python Tools/manuscript_lint.py --help`-style inspection if needed — the contract explains every rule.",
        "",
        "STEP 2: read each brief below and complete ALL of them, in order.",
        "",
        "## Briefs",
    ]
    for tid in batch["tasks"]:
        t = tasks[tid]
        lines.append(f"- `Tools/assignments/tasks/{tid}.md`  ({t['kind']} -> `{t['path']}`)")
    lines += [
        "",
        "## Hard rules",
        "- Write ONLY the files named in those briefs. Never touch other directories, root `INDEX.md`,",
        "  `README.md`, `00_*` files, `COVERAGE_LEDGER.*`, `Templates/`, or `Tools/`.",
        "- Every manuscript you write must pass `python Tools/manuscript_lint.py <that file>` with exit 0.",
        "  Run the lint yourself after each file; fix and re-run until it passes. Do not report a file as",
        "  done while its lint fails.",
        "- NO padding. Physical length without distinct content is rejected: no repeated sentences, no",
        "  numbered filler, no copied interview answer trios. See contract section 0.",
        "- Derive and recompute every number you print. Label constructed values ILLUSTRATIVE.",
        "- When an index brief asks you to rewrite an index, link only files that exist after your work.",
        "",
        "## Progress accounting",
        "Append one row per finished artifact to `<unit>/AUTHORING_PROGRESS.md` (create if missing):",
        "| Timestamp (UTC) | Artifact | Status | Lint | Note |",
        "",
        "## Final message",
        "Report: each file written, its lint line (`PASS ...`), and anything you could not finish with the",
        "reason. Never claim a file passed if you did not see the PASS line.",
    ]
    return "\n".join(lines)


def main() -> int:
    if "--verify" in sys.argv:
        print(json.dumps(verify(), indent=1))
        return 0
    if "--state" in sys.argv:
        s = (load("state.json") or {}).get("tasks", {})
        from collections import Counter
        print(Counter(s.values()))
        return 0
    if "--next" in sys.argv:
        n = int(sys.argv[sys.argv.index("--next") + 1])
        chosen = next_batches(n)
        if "--claim" in sys.argv:
            claimed = load("claimed.json") or {}
            for b in chosen:
                claimed[b["batch"]] = "claimed"
            save("claimed.json", claimed)
        for b in chosen:
            print("=" * 70)
            print(prompt_for(b))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
