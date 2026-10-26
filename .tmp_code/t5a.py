WINDOWS = [
    ("r01", "p01", "train"), ("r01", "p01", "train"), ("r01", "p01", "test"),
    ("r02", "p02", "train"), ("r02", "p02", "test"),
    ("r03", "p03", "test"), ("r03", "p03", "test"),
    ("r04", "p04", "train"), ("r05", "p05", "test"),
    ("r06", "p06", "train"), ("r06", "p06", "test"),
]
train_records = {r for r, p, s in WINDOWS if s == "train"}
test_records = {r for r, p, s in WINDOWS if s == "test"}
train_patients = {p for r, p, s in WINDOWS if s == "train"}
test_patients = {p for r, p, s in WINDOWS if s == "test"}

print(f"Windows: {len(WINDOWS)}")
print(f"Train windows: {sum(1 for _, _, s in WINDOWS if s == 'train')}")
print(f"Test windows:  {sum(1 for _, _, s in WINDOWS if s == 'test')}")
print(f"Record overlap:  {sorted(train_records & test_records)}")
print(f"Patient overlap: {sorted(train_patients & test_patients)}")
print()
print(f"Share of test windows from a leaked record: "
      f"{sum(1 for r, p, s in WINDOWS if s == 'test' and r in train_records) / sum(1 for _, _, s in WINDOWS if s == 'test'):.0%}")
