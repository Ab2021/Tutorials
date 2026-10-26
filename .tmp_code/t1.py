from dataclasses import dataclass

GATES = ("decision", "population", "access", "license", "benchmark")
LABELS = {
    "decision": "Decision",
    "population": "Population",
    "access": "Access",
    "license": "License",
    "benchmark": "Benchmark",
}

@dataclass(frozen=True)
class Candidate:
    name: str
    scores: dict
    evidence: dict

    def vetoes(self):
        return tuple(g for g in GATES if self.scores[g] == 0)

    def total(self):
        return sum(self.scores.values())

CANDIDATES = [
    Candidate(
        "MIMIC-IV v3.1",
        {"decision": 2, "population": 1, "access": 2, "license": 2, "benchmark": 1},
        {
            "decision": "Sepsis-3 computable from vitals, labs, antibiotics, cultures",
            "population": "Single tertiary academic ICU; no community-hospital test",
            "access": "PhysioNet credentialed; CITI; DUA",
            "license": "Research use under DUA; commercial use needs separate terms",
            "benchmark": "No frozen patient-level split published",
        },
    ),
    Candidate(
        "eICU-CRD v2.0",
        {"decision": 2, "population": 2, "access": 1, "license": 2, "benchmark": 1},
        {
            "decision": "Sepsis-3 partially computable; lab coverage varies by site",
            "population": "About 200 US hospitals; voluntary participation",
            "access": "PhysioNet credentialed; application review",
            "license": "Research use under DUA; commercial use needs separate terms",
            "benchmark": "No frozen split; site identifiers present",
        },
    ),
    Candidate(
        "Epic Cosmos",
        {"decision": 2, "population": 2, "access": 0, "license": 1, "benchmark": 1},
        {
            "decision": "Phenotypes vendor-mediated; raw labs not exported",
            "population": "About 2,358 hospitals (vendor figure)",
            "access": "Contributing organizations only; no row-level export",
            "license": "Contributor agreement; query outputs only",
            "benchmark": "No reproducible test set",
        },
    ),
    Candidate(
        "National claims",
        {"decision": 0, "population": 2, "access": 1, "license": 1, "benchmark": 1},
        {
            "decision": "No vitals or labs at 6-hour granularity",
            "population": "Commercially insured or Medicare coverage",
            "access": "Licensed contract",
            "license": "Field-of-use limits; no model release",
            "benchmark": "Billing labels only",
        },
    ),
    Candidate(
        "Prospective partner site",
        {"decision": 2, "population": 2, "access": 0, "license": 2, "benchmark": 2},
        {
            "decision": "Full EHR at bedside; prospective collection",
            "population": "Three community hospitals in target setting",
            "access": "No agreement inside 12-month window",
            "license": "Contract negotiable",
            "benchmark": "Prospective silent-mode evaluation possible",
        },
    ),
]

def report(rows):
    header = f"{'Candidate':<24}" + "".join(f"{LABELS[g]:>11}" for g in GATES) + f"{'Veto':>16}"
    print(header)
    print("-" * len(header))
    for c in rows:
        cells = "".join(f"{c.scores[g]:>11}" for g in GATES)
        veto = ",".join(c.vetoes()) if c.vetoes() else "none"
        print(f"{c.name:<24}{cells}{veto:>16}")

print("ALL CANDIDATES")
report(CANDIDATES)
print()
print("ADMISSIBLE SET")
for c in CANDIDATES:
    if not c.vetoes():
        print(f"  {c.name}: total {c.total()}/10")
print()
print("VETOED")
for c in CANDIDATES:
    if c.vetoes():
        first = c.vetoes()[0]
        print(f"  {c.name}: {first} -> {c.evidence[first]}")
