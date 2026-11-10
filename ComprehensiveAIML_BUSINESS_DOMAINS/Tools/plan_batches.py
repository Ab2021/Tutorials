#!/usr/bin/env python3
"""Group the queue into conflict-free agent batches.

Ownership invariant: a collection directory (…/Case_Studies or …/Cheat_Sheets)
belongs to exactly one batch, and all of its tasks — manuscripts *and* index
rewrites — go to that batch. Concurrent agents therefore never write the same
file, and an index is never rewritten while its own manuscripts are in flight.

    python Tools/plan_batches.py --summary
    python Tools/plan_batches.py            # writes .tmp_parts/batches.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".authoring_state"

MAX_MS_PER_BATCH = 1


def load_queue() -> list[dict]:
    path = TMP / "queue.jsonl"
    if not path.exists():
        sys.exit("run Tools/build_queue.py first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def collection_dir(task: dict) -> str:
    """Directory a task writes into."""
    p = task["path"]
    parts = p.split("/")
    if "Case_Studies" in parts:
        return "/".join(parts[: parts.index("Case_Studies") + 1])
    if "Cheat_Sheets" in parts:
        return "/".join(parts[: parts.index("Cheat_Sheets") + 1])
    return p.rsplit("/", 1)[0]


def unit_of(dirpath: str) -> str:
    return dirpath.rsplit("/", 1)[0] if dirpath.split("/")[-1] in ("Case_Studies", "Cheat_Sheets") else dirpath


def main() -> int:
    tasks = load_queue()

    # collection dir -> {manuscripts: [...], light: [...]}
    bydir: "OrderedDict[str, dict]" = OrderedDict()
    for t in sorted(tasks, key=lambda x: (x["priority"], x["path"])):
        d = collection_dir(t)
        slot = bydir.setdefault(d, {"manuscripts": [], "light": [], "unit": unit_of(d)})
        (slot["light"] if t["kind"] == "unit_index" else slot["manuscripts"]).append(t)

    # unit_index tasks must live with the unit's collections
    units: "OrderedDict[str, list[str]]" = OrderedDict()
    for d, slot in bydir.items():
        units.setdefault(slot["unit"], []).append(d)

    batches: list[dict] = []
    seq = 0
    for unit, dirs in units.items():
        # keep whole directories together; pack consecutive dirs up to the cap
        pending: list[dict] = []
        for d in sorted(dirs):
            slot = bydir[d]
            # a dir larger than the cap becomes its own chunk sequence
            items = slot["manuscripts"]
            base = len(pending)
            if base and base + len(items) > MAX_MS_PER_BATCH:
                seq += 1
                batches.append(_mk(seq, pending, unit))
                pending = []
            # split an oversized dir deterministically; light tasks follow the last chunk
            chunks = [items[i:i + MAX_MS_PER_BATCH] for i in range(0, len(items), MAX_MS_PER_BATCH)] or [[]]
            for idx, chunk in enumerate(chunks):
                last = idx == len(chunks) - 1
                group = list(chunk) + (slot["light"] if last else [])
                if not group:
                    continue
                if len(pending) + len(group) > MAX_MS_PER_BATCH and pending:
                    seq += 1
                    batches.append(_mk(seq, pending, unit))
                    pending = []
                pending.extend(group)
                if len(pending) >= MAX_MS_PER_BATCH:
                    seq += 1
                    batches.append(_mk(seq, pending, unit))
                    pending = []
        if pending:
            seq += 1
            batches.append(_mk(seq, pending, unit))

    # A split directory legitimately spans batches, but no two batches may own
    # the same file: index files and unit INDEX.md are exclusive, and each
    # manuscript belongs to exactly one task.
    seen: dict[str, str] = {}
    clashes = []
    for b in batches:
        for tid in b["tasks"]:
            t = next(x for x in tasks if x["id"] == tid)
            if t["path"] in seen:
                clashes.append((t["path"], seen[t["path"]], b["batch"]))
            seen[t["path"]] = b["batch"]

    (TMP / "batches.json").write_text(
        json.dumps({"batches": batches, "clashes": clashes}, indent=1), encoding="utf-8"
    )

    print(f"{len(batches)} batches, {len(seen)} collection dirs, clashes={len(clashes)}")
    print(Counter(len(b["tasks"]) for b in batches))
    if clashes:
        for c in clashes[:5]:
            print("  CLASH", c)
        return 1
    if "--summary" in sys.argv:
        for b in batches[:12]:
            print(b["batch"], len(b["tasks"]), [t for t in b["directories"]][:2])
    return 0


def _mk(seq: int, items: list[dict], unit: str) -> dict:
    return {
        "batch": f"B{seq:04d}",
        "unit": unit,
        "directories": sorted({collection_dir(t) for t in items}),
        "tasks": [t["id"] for t in items],
    }


if __name__ == "__main__":
    sys.exit(main())
