IMAGES = [
    ("p01", "s01", "train"), ("p01", "s01", "train"), ("p01", "s02", "test"),
    ("p02", "s03", "train"), ("p02", "s03", "train"),
    ("p03", "s04", "test"), ("p03", "s04", "test"),
    ("p04", "s05", "train"), ("p05", "s06", "test"),
    ("p06", "s07", "train"), ("p06", "s08", "test"),
    ("p07", "s09", "train"), ("p08", "s10", "test"),
]

train_patients = {p for p, s, split in IMAGES if split == "train"}
test_patients = {p for p, s, split in IMAGES if split == "test"}
train_studies = {s for p, s, split in IMAGES if split == "train"}
test_studies = {s for p, s, split in IMAGES if split == "test"}

print(f"Images: {len(IMAGES)}")
print(f"Train images: {sum(1 for _, _, x in IMAGES if x == 'train')}")
print(f"Test images:  {sum(1 for _, _, x in IMAGES if x == 'test')}")
print(f"Train patients: {len(train_patients)}  Test patients: {len(test_patients)}")
print(f"Train studies:  {len(train_studies)}  Test studies:  {len(test_studies)}")
print()
print(f"Patient overlap: {sorted(train_patients & test_patients)}")
print(f"Study overlap:   {sorted(train_studies & test_studies)}")
print()
n_leak_patients = len(train_patients & test_patients)
n_test_patients = len(test_patients)
print(f"Share of test patients also in train: {n_leak_patients/n_test_patients:.0%}")
print("A patient-level split would move every image for a leaked patient to one side.")
