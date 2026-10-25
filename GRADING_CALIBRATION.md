# Grading Calibration Guide

> **Purpose:** Make grading consistent across instructors, cohorts, and reviewers.
>
> **Status:** Protocol and anchor descriptors. Calibration has not yet been run with external graders; record the first calibration session in `review/`.

---

## 1. Principles

- Grade against the rubric, not against the best submission.
- Use evidence from the artifact, not effort or polish.
- Score each criterion independently before summing.
- Use whole points only; avoid false precision.
- Record one concrete reason for every score below “competent.”
- Reconcile differences; do not average them away.

## 2. Scale

| Score | Label | Meaning |
|---:|---|---|
| 1 | Novice | Missing or misleading; unsafe if used |
| 2 | Developing | Partial; major gaps |
| 3 | Competent | Meets the standard with support |
| 4 | Expert | Independent, defensible, decision-ready |

A score of 3 is a good result. A 4 requires evidence of independent judgment, not extra length.

## 3. Anchor artifacts

Prepare five anchor submissions per major assessment:

1. Clear novice.
2. High novice / low developing.
3. Solid competent.
4. High competent.
5. Expert.

Remove names and identifiers. Keep the rubric visible. Ask graders to score independently, then compare.

## 4. Calibration session

1. **Brief** (15 min): review rubric, anchors, and common biases.
2. **Independent scoring** (60–90 min).
3. **Compare** (30 min): criterion-level differences.
4. **Discuss** differences > 1 point.
5. **Re-score** one artifact (20 min).
6. **Agree** on adjustments to the rubric or anchors.
7. **Record** the memo and inter-rater agreement.

Target: exact agreement on 70% of criterion scores; within 1 point on 90%. If not achieved, revise the rubric or anchors.

## 5. Midterm evidence dossier anchors

| Criterion | Novice | Competent | Expert |
|---|---|---|---|
| Decision and workflow | “Predict readmission” | Named user and action | Comparator, capacity, and value |
| Data and label | Convenience sample | Explicit cohort and label | Provenance, missingness, bias quantified |
| Design and estimand | No protocol | Clear estimand and validation | External/impact plan and sensitivity |
| Analysis and validation | AUROC only | Calibration and utility | Subgroup and decision analysis |
| Communication and reproducibility | Technical | Clear memo | Executable appendix and audit trail |

## 6. Track playbook anchors

| Criterion | Novice | Competent | Expert |
|---|---|---|---|
| Sector depth | Generic | Accurate workflows and incentives | Insight beyond public summaries |
| Use-case selection | Technology-first | Decision-first | Portfolio logic and sequencing |
| Evidence and regulation | Names rules | Maps evidence to pathway | Anticipates reviewer objections |
| Economics and adoption | No numbers | ROI and adoption plan | Sensitivity, budget impact, procurement |
| Roadmap and communication | Vague | Milestones and owners | Credible 12-month plan with stop criteria |

## 7. Case critique anchors

| Criterion | Novice | Competent | Expert |
|---|---|---|---|
| Decision framing | Summary only | Clear decision and user | Comparator and incentives |
| Evidence appraisal | Accepts claims | Identifies design and validation issues | Quantifies uncertainty and bias |
| Bias and limitations | Generic caveats | Specific mechanisms | Links to harm and equity |
| Alternatives and judgment | No alternative | Plausible alternative | Decision-ready recommendation |
| Communication | Jargon | Clear one-pager | Persuasive to a mixed audience |

## 8. Capstone anchors

| Criterion | Novice | Competent | Expert |
|---|---|---|---|
| Problem framing | Model-first | Decision and user | Workflow, capacity, and value |
| Data | No provenance | Access and quality | Linkage, missingness, bias |
| Validation | Random split | Temporal/external | Prospective or impact plan |
| Calibration/utility | AUC only | Calibration and decision curve | Subgroup and net benefit |
| Regulatory | Ignores | Triage and evidence gap | PCCP/change-control plan |
| Deployment | Notebook | Architecture and monitoring | Incident, rollback, cost |
| Communication | Technical | Clear | Executive and clinical |

## 9. Common grading biases

- **Halo:** polished writing raises technical scores.
- **Length:** more pages are not more evidence.
- **Complexity:** a transformer is not automatically better than logistic regression.
- **Hindsight:** grading the result, not the decision process.
- **Charity:** filling gaps the learner did not state.
- **Severity:** penalizing a competent, simple solution.

## 10. Feedback template

```markdown
# Assessment feedback

Artifact:
Reviewer:
Date:

## Scores
| Criterion | Score | Evidence |
|---|---:|---|

## What works
[2–3 specific strengths]

## What must change
[1–3 specific, actionable issues]

## What would raise the score
[one concrete example]

## Safety/ethical flags
[none or specific]

## Reassessment
[ ] No [ ] Revise and resubmit by [date]
```

## 11. Reassessment policy

- One revision allowed for labs and case critiques.
- Midterm dossier: corrections within 7 days.
- Capstone: no revision after the defense; the defense is the final gate.
- A safety-critical error requires correction even if the numeric score is high.

## 12. Record keeping

- Retain rubrics, scores, feedback, and calibration memos.
- Remove unnecessary personal data.
- Keep assessment data separate from public course files.
- Report aggregate results, not individual learner data, in course reviews.
