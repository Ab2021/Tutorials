"""Population stability index (PSI) for one numeric feature.

Usage: python drift.py baseline.csv current.csv column
CSV files must have a header row.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path


def load(path: str, column: str) -> list[float]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [float(row[column]) for row in csv.DictReader(fh) if row[column] not in ("", None)]


def psi(base: list[float], cur: list[float], bins: int = 10) -> float:
    base_sorted = sorted(base)
    cuts = [base_sorted[int(len(base_sorted) * i / bins)] for i in range(1, bins)]
    def dist(vals):
        counts = [0] * (bins)
        for v in vals:
            idx = 0
            while idx < len(cuts) and v > cuts[idx]:
                idx += 1
            counts[idx] += 1
        total = max(len(vals), 1)
        return [max(c / total, 1e-6) for c in counts]
    b, c = dist(base), dist(cur)
    return sum((ci - bi) * math.log(ci / bi) for bi, ci in zip(b, c))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("column")
    args = ap.parse_args()
    value = psi(load(args.baseline, args.column), load(args.current, args.column))
    print(f"PSI {args.column}: {value:.4f}")
    print("review" if value >= 0.2 else "ok")
