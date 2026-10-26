reports = [
    ("R001", "drugA", "nausea", "2024-01-10", "site1"),
    ("R002", "drugA", "nausea", "2024-01-10", "site1"),
    ("R003", "drugA", "nausea", "2024-02-01", "site2"),
    ("R004", "drugB", "nausea", "2024-02-02", "site2"),
]
seen = {}
duplicates = []
for rid, drug, event, date, site in reports:
    key = (drug, event, date, site)
    if key in seen:
        duplicates.append((rid, seen[key]))
    else:
        seen[key] = rid
print(f"Reports: {len(reports)}")
print(f"Duplicate pairs: {duplicates}")
print(f"Unique reports after exact dedup: {len(seen)}")
print()
print("Exact duplicates are easy. Case-level duplicates with different dates or spellings")
print("require probabilistic matching and manual review.")
