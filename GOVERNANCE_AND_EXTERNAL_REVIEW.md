# Governance and External Review Plan

> **Purpose:** Define how the course is reviewed, corrected, approved, and maintained before it is described as validated.
>
> **Status:** Plan and templates only. No external reviewer has yet been convened or has signed off. This distinction must remain visible in all public descriptions.

---

## 1. Why this exists

A course about regulated, safety-critical work cannot be validated by its authors alone. It needs:

- Clinical review for plausibility and harm.
- Regulatory/quality review for legal and standards accuracy.
- Hardware/robotics review for physical safety and feasibility.
- Privacy/ethics review for data and equity.
- Learner pilot evidence for whether the labs actually run.

This document defines the review board, its powers, its evidence requirements, and its sign-off.

## 2. Review board composition

| Role | Required expertise | Authority |
|---|---|---|
| Chair | Independent senior academic or industry leader | Runs review, resolves conflicts, signs final report |
| Clinician | Board-certified physician with informatics experience | Clinical claims, workflows, harm, usability |
| Regulatory/quality lead | FDA/EU device or pharma quality experience | Device classification, QMS, PCCP, GCP, claims |
| Hardware/robotics engineer | Medical device or surgical robotics experience | Sensors, firmware, electrical safety, HIL, robot safety |
| Privacy/security/ethics reviewer | HIPAA/GDPR, security, or research ethics | Data flows, consent, fairness, security |
| Patient/community representative | Lived experience or patient advocacy | Accessibility, language, burden, trust |
| Learner representative | Recent learner or pilot participant | Teachability, workload, lab feasibility |
| Legal counsel (advisory) | Health/device law | Regulatory claims, liability language |

**Minimum for a formal review:** chair, clinician, regulatory/quality lead, hardware/robotics engineer, privacy/ethics reviewer. Patient and learner representatives are strongly recommended.

## 3. Terms of reference

The board will:

1. Review the course scope, outcomes, competency map, and assessment blueprint.
2. Sample at least 25% of modules, including all modules with regulatory, device, robotics, or clinical safety claims.
3. Review all high-risk claims in `REGULATORY_VERIFICATION_LOG.md`.
4. Review the capstone briefs for feasibility, safety, and ethics.
5. Review the instructor kit and grading calibration.
6. Observe or review a pilot delivery.
7. Issue a written report with critical, major, and minor findings.
8. Re-review corrections within 60 days.
9. Sign a final statement that distinguishes:
   - **Reviewed:** the board read the material.
   - **Approved:** no critical findings remain.
   - **Validated:** pilot evidence and corrections are complete.
10. Recommend a review cadence and owners.

The board does **not**:

- Provide legal advice.
- Certify regulatory compliance.
- Approve a real medical device.
- Replace institutional IRB, legal, or quality review.
- Guarantee learner competence.

## 4. Conflict of interest

Each reviewer completes a disclosure covering:

- Employment and consulting.
- Equity and advisory roles.
- Grants and speaking fees.
- Products or companies mentioned in the course.
- Relationships with authors or institutions.

A reviewer with a material conflict recuses from the affected module or claim. Disclosures are recorded in the review file.

## 5. Review package

Provide reviewers with:

- `README.md`, `COURSE_OUTLINE.md`, `COMPETENCY_MAP.md`
- `_course_spec.md`
- All modules and research briefs
- `REGULATORY_VERIFICATION_LOG.md`
- `INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`
- `PILOT_PROTOCOL.md`
- Link verification and DOI check results
- A one-page change summary
- A claim-to-source index
- A review rubric and issue log template

## 6. Review rubric

| Criterion | Weight | Novice / fail | Competent | Expert |
|---|---:|---|---|---|
| Clinical accuracy and safety | 25% | Errors or unsafe simplifications | Accurate and appropriately caveated | Anticipates edge cases and harm |
| Regulatory accuracy | 20% | Confuses guidance, law, and proposal | Correctly distinguishes and dates | Identifies jurisdictional gaps |
| Technical accuracy | 20% | Wrong methods or claims | Correct and reproducible | Explains trade-offs and failure modes |
| Assessment validity | 15% | No link to outcomes | Outcomes mapped and assessed | Evidence of calibration |
| Equity and accessibility | 10% | Generic statement | Specific mechanisms and mitigations | Community-informed design |
| Teachability and workload | 10% | Unrealistic | Feasible with support | Piloted and adjusted |

**Decision rule:** any unresolved critical finding blocks “approved.” A critical finding is one that could cause clinical harm, illegal data use, a false regulatory claim, or a lab that cannot run safely.

## 7. Review schedule

| Stage | Timing | Output |
|---|---|---|
| Desk review | Week 0 | Reviewer assignments, COI, package |
| Independent review | Weeks 1–3 | Module and claim findings |
| Panel meeting | Week 4 | Consolidated report |
| Author response | Weeks 5–6 | Corrections and rationale |
| Re-review | Weeks 7–8 | Closure or escalation |
| Pilot | Weeks 9–14 | Learner and lab evidence |
| Final sign-off | Week 16 | Reviewed/approved/validated statement |

## 8. Issue log template

| ID | Module/claim | Severity | Finding | Evidence | Owner | Due | Status |
|---|---|---|---|---|---|---|---|
| R-001 | 14 §9.3 | Critical | FDA page moved; claim not verified | URL check | Reg lead | Week 2 | Open |
| R-002 | 14B §7 | Major | Calibration set not defined for PTQ | Lab review | Author | Week 4 | Open |
| R-003 | 24 oncology | Minor | Endpoint example lacks source | Citation check | Author | Week 6 | Open |

Severity:

- **Critical:** harm, illegality, false regulatory claim, unsafe lab.
- **Major:** material inaccuracy, missing evidence, unteachable section.
- **Minor:** clarity, citation, formatting, consistency.

## 9. Sign-off template

```markdown
# Course review sign-off

Course version:
Review dates:
Reviewers:
Conflict-of-interest status:

## Scope reviewed
[modules, claims, labs, assessments]

## Findings
Critical: [n] open / [n] closed
Major: [n] open / [n] closed
Minor: [n] open / [n] closed

## Statement
[ ] Reviewed only
[ ] Approved for pilot
[ ] Approved for delivery
[ ] Validated with pilot evidence

## Conditions
[list]

## Signatures
Chair:
Clinician:
Regulatory/quality:
Hardware/robotics:
Privacy/ethics:
Date:
```

## 10. What cannot be claimed yet

Until sign-off:

- Do not say “clinically validated.”
- Do not say “regulatory approved.”
- Do not say “accredited.”
- Do not say “reviewed by clinicians” unless named reviewers consented.
- Do not say “pilot-tested” unless the pilot is complete.

## 11. Maintenance

- Review regulatory claims every 6 months.
- Review the full course annually.
- Re-run the link checker quarterly.
- Re-run the pilot after major changes.
- Keep a public change log and a private reviewer file.
- Retire or replace modules that no longer meet the evidence standard.

## 12. Review file location

Create a private, access-controlled folder outside the public package:

```
review/
├── membership.md
├── coi/
├── package/
├── findings/
├── signoff.md
└── pilot/
```

Do not put reviewer personal data or confidential findings in the public course folder.
