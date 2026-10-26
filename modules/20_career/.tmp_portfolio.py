"""Portfolio scoring and repository readiness check, standard library only.

Run: python portfolio_score.py
Creates a temporary directory, populates it with a mock portfolio repository,
and scores it. Nothing is written outside the temporary directory.
"""

import tempfile
from pathlib import Path

# (dimension, weight, score before, score after), scores on a 0-3 scale.
RUBRIC = [
    ("Decision framing", 0.20, 1, 3),
    ("Evidence standard", 0.25, 1, 3),
    ("Reproducibility", 0.20, 0, 2),
    ("Communication", 0.20, 1, 2),
    ("Judgment", 0.15, 1, 3),
]

REQUIRED = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "Makefile",
    "src",
    "tests",
    "results",
]

THRESHOLDS = {"credible": 2.0, "panel_ready": 2.5}


def weighted_score(rows, index):
    total = 0.0
    for row in rows:
        total += row[1] * row[index]
    return total


def check_repo(root):
    """Return (present, missing) for the baseline RAP checklist."""
    present, missing = [], []
    for name in REQUIRED:
        (present if (root / name).exists() else missing).append(name)
    return present, missing


def build_mock_repo(root):
    (root / "README.md").write_text("# Mock portfolio repository\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pandas==2.2.2\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "cohort.py").write_text("print('cohort')\n", encoding="utf-8")
    (root / "tests" / "test_cohort.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


def main():
    before = weighted_score(RUBRIC, 2)
    after = weighted_score(RUBRIC, 3)

    print("PORTFOLIO SCORE (0-3 weighted)")
    print(f"  {'dimension':20s} {'weight':>6s} {'before':>6s} {'after':>6s}")
    for label, weight, b, a in RUBRIC:
        print(f"  {label:20s} {weight:6.2f} {b:6d} {a:6d}")
    print(f"  {'weighted total':20s} {sum(w for _, w, _, _ in RUBRIC):6.2f} "
          f"{before:6.2f} {after:6.2f}")

    for stage, score in (("before", before), ("after", after)):
        verdict = "not credible"
        if score >= THRESHOLDS["panel_ready"]:
            verdict = "panel-ready"
        elif score >= THRESHOLDS["credible"]:
            verdict = "credible"
        print(f"  {stage:6s} verdict: {verdict}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_mock_repo(root)
        present, missing = check_repo(root)
        print()
        print("REPOSITORY READINESS (baseline RAP checklist)")
        print(f"  present: {len(present)}/{len(REQUIRED)} -> {', '.join(present)}")
        print(f"  missing: {len(missing)}/{len(REQUIRED)} -> {', '.join(missing)}")
        print(f"  readiness: {len(present) / len(REQUIRED):.2f}")

    print()
    print("GAP TO PANEL-READY")
    print(f"  current after score: {after:.2f}")
    print(f"  points needed: {max(0.0, THRESHOLDS['panel_ready'] - after):.2f}")
    print("  cheapest route: raise reproducibility from 2 to 3 (+0.20),")
    print("  then communication from 2 to 3 (+0.20).")


if __name__ == "__main__":
    main()
