"""Cardiology scores and prevalence-adjusted PPV. Standard library only."""

from dataclasses import dataclass


@dataclass
class Patient:
    age: int
    sex: str                 # "M" or "F"
    chf: bool = False
    hypertension: bool = False
    diabetes: bool = False
    stroke: bool = False
    vascular: bool = False


def cha2ds2_vasc(p):
    """2010 Chest derivation; as scored in the 2023 ACC/AHA/ACCP/HRS AF guideline."""
    pts = 0
    pts += 1 if p.chf else 0                  # C
    pts += 1 if p.hypertension else 0         # H
    pts += 2 if p.age >= 75 else 1 if p.age >= 65 else 0   # A2 / A
    pts += 1 if p.diabetes else 0             # D
    pts += 2 if p.stroke else 0               # S2
    pts += 1 if p.vascular else 0             # V
    pts += 1 if p.sex == "F" else 0           # Sc
    return pts


def cha2ds2_va(p):
    """2024 ESC variant: the sex category is removed, so women score one point less."""
    return cha2ds2_vasc(p) - (1 if p.sex == "F" else 0)


def has_bled(uncontrolled_htn, abnormal_renal, abnormal_liver, stroke,
             bleeding, labile_inr, age, drugs, alcohol):
    """Pisters et al., Chest 2010. Maximum 9."""
    s = 0
    s += 1 if uncontrolled_htn else 0
    s += 1 if abnormal_renal else 0
    s += 1 if abnormal_liver else 0
    s += 1 if stroke else 0
    s += 1 if bleeding else 0
    s += 1 if labile_inr else 0
    s += 1 if age > 65 else 0
    s += 1 if drugs else 0
    s += 1 if alcohol else 0
    return s


def predict(sens, spec, prevalence, n=100_000):
    """2x2 counts at a stated prevalence, with sens and spec held fixed."""
    diseased = round(n * prevalence)
    healthy = n - diseased
    tp = round(diseased * sens)
    fp = round(healthy * (1.0 - spec))
    fn = diseased - tp
    tn = healthy - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "ppv": tp / (tp + fp), "npv": tn / (tn + fn)}


print("== Patient ==")
p = Patient(age=71, sex="M", hypertension=True, diabetes=True)
print("age 71 M, hypertension, diabetes")
print("CHA2DS2-VASc (2023 ACC/AHA):", cha2ds2_vasc(p))
print("CHA2DS2-VA  (2024 ESC)     :", cha2ds2_va(p))
print("HAS-BLED (SBP 168, age >65):",
      has_bled(True, False, False, False, False, False, p.age, False, False))

print()
print("== PPV at fixed sensitivity 0.95 and specificity 0.95 ==")
print(f"{'prevalence':>10} {'diseased':>9} {'TP':>6} {'FP':>7} "
      f"{'PPV':>8} {'NPV':>8}")
for prev in (0.005, 0.01, 0.05, 0.20):
    r = predict(0.95, 0.95, prev)
    print(f"{prev:>9.1%} {round(100_000 * prev):>9} {r['tp']:>6} "
          f"{r['fp']:>7} {r['ppv']:>7.1%} {r['npv']:>8.2%}")

print()
print("== Specificity sensitivity analysis at prevalence 0.5% ==")
for spec in (0.99, 0.95, 0.90):
    r = predict(0.95, spec, 0.005)
    print(f"spec {spec:.2f}: FP={r['fp']:>6} PPV={r['ppv']:>6.2%}")
