# Pilot Protocol — 5–10 Learner Lab and Course Feasibility Study

> **Purpose:** Test whether the labs actually run, whether the workload is feasible, and whether learners can meet the outcomes before the course is described as validated.
>
> **Status:** Protocol only. No pilot has been conducted.

---

## 1. Questions

1. Can a learner run every required lab from a clean environment?
2. How long does each lab actually take?
3. Where do learners get stuck: setup, data access, concepts, or tooling?
4. Does the workload fit the stated weekly hours?
5. Do learners meet the competency standard on the anchor assessments?
6. Are the materials accessible and inclusive?
7. What breaks on Windows, macOS, Linux, and low-bandwidth connections?

## 2. Design

- **Sample:** 5–10 learners, deliberately mixed.
- **Inclusion:** at least two with clinical background, two with software/ML background, two with hardware/device interest, at least one disabled or neurodivergent learner if they consent.
- **Duration:** 6–8 weeks for a compressed pilot, or 12 weeks for a fuller pilot.
- **Format:** remote or hybrid; one live session per week; asynchronous labs.
- **Compensation:** fair, if institutional policy allows.
- **Ethics:** informed consent, voluntary withdrawal, no employment consequences.

## 3. Pilot scope

Minimum:

- Modules 00, 01A, 02, 03, 06, 14, 14A, 14B, 19.
- Labs: 01A.1, 03.1, 06.3, 11.1, 14A.1, 14B.2, 14B.7, 19 charter.
- Assessments: one case critique, one midterm dossier, one capstone pitch.

Optional:

- One specialty elective from 23.
- One hardware/robotics lab from 24.
- One cloud/streaming lab from 26.

## 4. Schedule

| Week | Activity | Data collected |
|---:|---|---|
| 0 | Consent, baseline survey, environment check | Skills, expectations, accessibility needs |
| 1 | Module 00 + charter | Time, clarity, confidence |
| 2 | 01A + unit lab | Lab completion, errors, time |
| 3 | 02/03 + cohort lab | Setup failures, data access time |
| 4 | 06 + calibration lab | Concept quiz, lab time |
| 5 | 14/14A + device teardown | Regulatory quiz, lab time |
| 6 | 14B + quantization/HIL | Target measurements, blockers |
| 7 | 19 + capstone pitch | Rubric scores, feedback |
| 8 | Interviews and report | Qualitative themes |

## 5. Measures

### 5.1 Feasibility

| Measure | Target |
|---|---|
| Lab completion without instructor rescue | ≥ 80% |
| Clean-environment setup success | ≥ 80% first attempt |
| Median lab time within estimated range | ≥ 80% of labs |
| Data-access blockage | < 20% of learners |
| Critical safety issue | 0 |

### 5.2 Learning

| Measure | Target |
|---|---|
| Pre/post knowledge gain | ≥ 20 percentage points |
| Anchor assessment competent | ≥ 70% |
| Calibration/decision-curve competency | ≥ 70% |
| Regulatory triage competency | ≥ 70% |
| Assumptions-ledger competency | ≥ 70% |

### 5.3 Experience

- Workload rating (1–5; target ≤ 4).
- Clarity rating (target ≥ 4).
- Accessibility rating (target ≥ 4).
- Would recommend (target ≥ 80%).
- Qualitative interviews.

## 6. Data collection

- Pre/post survey.
- Lab timing log.
- Error log with screenshots.
- Screen recording or think-aloud (optional, consented).
- Assessment scores.
- Weekly pulse survey.
- Exit interview.
- Accessibility feedback.
- Instructor observation notes.

## 7. Analysis

- Descriptive statistics for timing and completion.
- Paired pre/post comparison.
- Thematic analysis of interviews and error logs.
- Rubric score distribution.
- Failure taxonomy: setup, data, concept, tool, time, accessibility.
- Report both successes and failures; do not hide negative results.

## 8. Success criteria

The pilot supports “pilot-tested” if:

- No critical safety or legal issue.
- ≥ 80% of required labs complete without rescue.
- ≥ 70% of learners meet competent on anchor assessments.
- Workload is within 20% of the stated estimate.
- Accessibility issues are documented with a remediation plan.

It does **not** support “validated” unless external reviewers also sign off and corrections are closed.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Learner lacks data access | Synthetic fallback |
| Laptop/GPU limits | Cloud or CPU-only path |
| Hardware lab unavailable | Simulator path |
| Privacy concern | De-identified/synthetic data |
| Power imbalance | Independent facilitator, anonymous survey |
| Small sample | Report as feasibility, not efficacy |
| Instructor bias | External reviewer and pre-specified measures |

## 10. Consent and ethics

- Plain-language information sheet.
- Voluntary participation; withdraw anytime.
- No impact on grades, employment, or clinical role.
- Store data securely; remove identifiers.
- Report aggregate findings.
- Follow institutional IRB/ethics requirements if applicable.

## 11. Report template

```markdown
# Pilot report

## Design
## Participants
## Lab feasibility
## Learning outcomes
## Workload
## Accessibility
## Failures and fixes
## Qualitative themes
## Limitations
## Recommendation
[ ] Not ready [ ] Ready for another pilot [ ] Ready for external review [ ] Pilot-tested
```

## 12. What not to claim

- Do not call a 5-person pilot “validated.”
- Do not generalize learning gains to all learners.
- Do not hide labs that failed.
- Do not publish learner quotes without consent.
