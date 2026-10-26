ACTIVITIES = ["train", "evaluate", "internal_demo", "release_weights", "publish", "deploy", "redistribute_raw"]
CODES = {"yes": "Y", "conditional": "C", "no": "N", "silent": "?"}

LICENCES = {
    "PhysioNet DUA (MIMIC-IV)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "conditional",
        "release_weights": "no", "publish": "conditional", "deploy": "no",
        "redistribute_raw": "no",
    },
    "Stanford AIMI RUA (CheXpert)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "no", "publish": "conditional", "deploy": "no",
        "redistribute_raw": "no",
    },
    "CC BY 4.0": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "yes", "publish": "yes", "deploy": "yes",
        "redistribute_raw": "yes",
    },
    "Commercial EHR licence": {
        "train": "conditional", "evaluate": "yes", "internal_demo": "conditional",
        "release_weights": "no", "publish": "conditional", "deploy": "conditional",
        "redistribute_raw": "no",
    },
    "Public domain (US government)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "yes", "publish": "yes", "deploy": "yes",
        "redistribute_raw": "yes",
    },
}

def matrix(licences):
    width = max(len(name) for name in licences) + 2
    short = {"train": "train", "evaluate": "eval", "internal_demo": "demo", "release_weights": "weights", "publish": "publish", "deploy": "deploy", "redistribute_raw": "redist"}
    header = f"{'Licence':<{width}}" + "".join(f"{short[a]:>10}" for a in ACTIVITIES)
    print(header)
    print("-" * len(header))
    for name, perms in licences.items():
        row = f"{name:<{width}}"
        row += "".join(f"{CODES[perms[a]]:>10}" for a in ACTIVITIES)
        print(row)

def commercial_blockers(licences):
    out = {}
    for name, perms in licences.items():
        blocked = [a for a in ("train", "deploy", "release_weights") if perms[a] in ("no", "silent")]
        out[name] = blocked
    return out

print("LICENCE COMPATIBILITY MATRIX")
print("Y = permitted, C = conditional, N = prohibited, ? = silent")
print()
matrix(LICENCES)
print()
print("BLOCKERS FOR A COMMERCIAL PRODUCT")
for name, blocked in commercial_blockers(LICENCES).items():
    if blocked:
        print(f"  {name}: {', '.join(blocked)}")
    else:
        print(f"  {name}: none")
