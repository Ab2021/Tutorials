"""Resume audit: keyword coverage and bullet quality, standard library only.

Run: python resume_audit.py
Inputs are inline so the output is reproducible. Replace RESUME_TEXT and
REQUIREMENTS with your own text to audit a real resume.
"""

import re

# (requirement, postings_mentioning, status)
# status is one of "strong", "partial", "weak", "absent".
REQUIREMENTS = [
    ("SQL", 3, "partial"),
    ("Python", 3, "partial"),
    ("Claims data", 2, "absent"),
    ("Causal inference", 2, "absent"),
    ("Dashboarding", 2, "absent"),
    ("Clinical data standards", 3, "strong"),
    ("Stakeholder communication", 3, "weak"),
    ("Study design", 2, "absent"),
]

STATUS_SCORE = {"strong": 1.0, "partial": 0.5, "weak": 0.25, "absent": 0.0}

WEAK_OPENERS = (
    "responsible for",
    "assisted",
    "helped",
    "worked on",
    "worked with",
    "participated",
    "supported",
    "involved in",
    "duties included",
    "used",
    "ensured",
)

BEFORE = [
    "Responsible for statistical programming for clinical trials",
    "Created SDTM and ADaM datasets",
    "Worked with cross-functional teams",
    "Performed quality control on datasets",
    "Used SAS and SQL daily",
    "Supported audits",
    "Automated reporting processes",
    "Trained new team members",
    "Ensured compliance with CDISC standards",
    "Reduced programming time by 20 percent",
]

AFTER = [
    "Programmed 23 Phase I-III oncology studies in SAS, delivering 100 percent of analysis "
    "datasets on or before the data cut across four sponsor audits with zero quality findings",
    "Built SDTM and ADaM datasets for a 640-patient Phase II study, reducing the median "
    "programming cycle from 12 to 8 days by templating the derivation code",
    "Automated a monthly safety reporting process in Python, cutting preparation from 3 days "
    "to 4 hours and eliminating two manual transcription steps",
    "Designed 42 edit checks for a 14-site registry; query rate at database lock fell from "
    "7.9 percent to 1.8 percent between pilot and primary study",
    "Trained four new programmers on CDISC conventions; all four passed their first internal "
    "quality review without rework",
    "Built a claims-based readmission cohort of 48,000 adults as a portfolio project, "
    "validating the model on the most recent 12 months with AUPRC improving from 0.21 to 0.29",
]


def coverage(reqs):
    weights = [p for _, p, _ in reqs]
    scores = [STATUS_SCORE[s] * p for _, p, s in reqs]
    return sum(scores) / sum(weights)


def weak_opener(bullet):
    low = bullet.lower().strip()
    for opener in WEAK_OPENERS:
        if low.startswith(opener):
            return opener
    return None


def has_number(bullet):
    return bool(re.search(r"\d", bullet))


def audit(bullets, label):
    weak = [(b, weak_opener(b)) for b in bullets if weak_opener(b)]
    numbered = [b for b in bullets if has_number(b)]
    mean_words = sum(len(b.split()) for b in bullets) / len(bullets)
    print(f"{label}: {len(bullets)} bullets")
    print(f"  weak openers: {len(weak)}")
    for bullet, opener in weak:
        print(f"    - starts with '{opener}': {bullet[:60]}")
    print(f"  bullets containing a number: {len(numbered)}/{len(bullets)}")
    print(f"  mean words per bullet: {mean_words:.1f}")


def main():
    print("KEYWORD COVERAGE")
    print(f"  {'requirement':26s} {'postings':>8s} {'status':>8s} {'score':>6s}")
    for label, count, status in REQUIREMENTS:
        print(f"  {label:26s} {count:8d} {status:>8s} {STATUS_SCORE[status]:6.2f}")
    cov = coverage(REQUIREMENTS)
    print(f"  weighted coverage: {cov:.3f}")
    print(f"  verdict at 0.60: {'pass' if cov >= 0.60 else 'rewrite the dialect'}")

    print()
    audit(BEFORE, "BEFORE")
    print()
    audit(AFTER, "AFTER")

    print()
    print("NOTE")
    print(f"  coverage above scores the BEFORE resume: {cov:.3f}")
    print("  the AFTER resume adds claims and validation terms, which this")
    print("  table does not re-score; re-run with new statuses to compare.")


if __name__ == "__main__":
    main()
