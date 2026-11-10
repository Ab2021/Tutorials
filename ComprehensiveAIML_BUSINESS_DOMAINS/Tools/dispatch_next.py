#!/usr/bin/env python3
"""Continuous dispatcher: keep N authoring agents running until the queue drains.

The kernel-side `agent()` calls live in the eval cell; this module only decides
*what* to hand out. Keeping the decision logic here (and state on disk) makes the
whole run resumable after an interruption.

    python Tools/dispatch_next.py --slots 8     # claims and prints next N batches
    python Tools/dispatch_next.py --reclaim     # release claims whose files are still untouched
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".authoring_state"
sys.path.insert(0, str(ROOT / "Tools"))


def load(name, default):
    p = TMP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def save(name, obj):
    (TMP / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def main() -> int:
    batches = load("batches.json", {"batches": []})["batches"]
    queue = {t["id"]: t for t in (json.loads(l) for l in (TMP / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    state = load("state.json", {"tasks": {}})["tasks"]
    claimed = load("claimed.json", {})

    if "--reclaim" in sys.argv:
        # a claim is released if none of its target files exist yet
        released = []
        for b in batches:
            if b["batch"] not in claimed:
                continue
            if all((ROOT / queue[t]["path"]).exists() for t in b["tasks"]):
                continue
            released.append(b["batch"])
            claimed.pop(b["batch"], None)
        save("claimed.json", claimed)
        print(f"released {len(released)} claims: {released[:10]}")
        return 0

    slots = int(sys.argv[sys.argv.index("--slots") + 1]) if "--slots" in sys.argv else 8
    ready = []
    for b in batches:
        if b["batch"] in claimed:
            continue
        if all(state.get(t) == "done" for t in b["tasks"]):
            continue
        blocked = [t for t in b["tasks"] if state.get(t) == "done"]
        if blocked and len(blocked) == len(b["tasks"]):
            continue
        prio = min(queue[t]["priority"] for t in b["tasks"])
        ready.append((prio, b["batch"], b))
    ready.sort(key=lambda x: (x[0], x[1]))
    chosen = ready[:slots]
    for _, name, b in chosen:
        claimed[name] = "claimed"
    save("claimed.json", claimed)

    from run_batch import prompt_for  # local import keeps this script tiny

    for _, _, b in chosen:
        print("=" * 70)
        print(prompt_for(b))
    print(f"\n[{len(chosen)} claimed, {len(ready) - len(chosen)} still queued]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
