"""Window-function semantics in pure Python, standard library only.

Run: python window_functions.py
Computes ROW_NUMBER, RANK, LAG, and a running SUM over partitions of a small
claims table. The point is to make the SQL semantics executable so that the
partition and frame rules can be checked.
"""

from collections import defaultdict

CLAIMS = [
    {"patient": "P1", "claim_date": "2026-01-04", "amount": 120.0},
    {"patient": "P1", "claim_date": "2026-02-11", "amount": 80.0},
    {"patient": "P1", "claim_date": "2026-02-11", "amount": 45.0},
    {"patient": "P1", "claim_date": "2026-03-02", "amount": 300.0},
    {"patient": "P2", "claim_date": "2026-01-19", "amount": 500.0},
    {"patient": "P2", "claim_date": "2026-04-07", "amount": 75.0},
]


def window(rows, partition_key, order_key, func, reverse=False):
    """Apply func to each partition, ordered by order_key.

    func receives (position, row, previous_row, running_total) and returns a
    dict of computed columns.
    """
    partitions = defaultdict(list)
    for row in rows:
        partitions[row[partition_key]].append(row)

    out = []
    for key in sorted(partitions):
        ordered = sorted(partitions[key], key=lambda r: (r[order_key], r["amount"]), reverse=reverse)
        running = 0.0
        previous = None
        for position, row in enumerate(ordered, start=1):
            running += row["amount"]
            computed = func(position, row, previous, running)
            out.append({**row, **computed})
            previous = row
    return out


def row_number(position, row, previous, running):
    return {"row_number": position}


def rank_with_ties(position, row, previous, running):
    # RANK leaves gaps after ties; DENSE_RANK does not. Ties here are on date.
    if previous is not None and previous["claim_date"] == row["claim_date"]:
        rank = previous["_rank"]
    else:
        rank = position
    row["_rank"] = rank
    return {"rank": rank}


def lag_amount(position, row, previous, running):
    return {"prev_amount": None if previous is None else previous["amount"]}


def running_sum(position, row, previous, running):
    return {"running_amount": round(running, 2)}


def main():
    rows = window(CLAIMS, "patient", "claim_date", row_number, reverse=True)
    rows = window(rows, "patient", "claim_date", rank_with_ties)
    rows = window(rows, "patient", "claim_date", lag_amount)
    rows = window(rows, "patient", "claim_date", running_sum)

    print("CLAIMS WITH WINDOW COLUMNS")
    header = f"  {'patient':8s} {'date':12s} {'amount':>7s} {'rn_desc':>8s} {'rank':>4s} {'prev':>7s} {'cum':>8s}"
    print(header)
    for r in rows:
        prev = "" if r["prev_amount"] is None else f"{r['prev_amount']:.1f}"
        print(f"  {r['patient']:8s} {r['claim_date']:12s} {r['amount']:7.1f} "
              f"{r['row_number']:8d} {r['rank']:4d} {prev:>7s} {r['running_amount']:8.1f}")

    latest = [r for r in rows if r["row_number"] == 1]
    print()
    print("LATEST CLAIM PER PATIENT (row_number = 1)")
    for r in latest:
        print(f"  {r['patient']}: {r['claim_date']} amount {r['amount']:.1f}")

    total = sum(r["amount"] for r in latest)
    print()
    print(f"  total of latest claims: {total:.1f}")
    print(f"  total of all claims:    {sum(r['amount'] for r in CLAIMS):.1f}")
    print("  the difference is the fan-out you get if you forget the deduplication filter.")


if __name__ == "__main__":
    main()
