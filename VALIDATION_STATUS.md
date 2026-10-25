# Validation Status

> **Purpose:** Single-page truth about what is complete, what is drafted, and what is still pending. Do not describe the course as validated until every P0 item is closed.
>
> **Version:** 2026.2.1
> **Date:** 24 October 2026

## Status legend

| Status | Meaning |
|---|---|
| Complete | Artifact exists and has been checked internally |
| Drafted | Artifact exists but has not been independently reviewed |
| Pending | Not yet done |
| Not claimed | Explicitly outside the course’s claims |

## P0 — before calling the course “complete”

| # | Item | Artifact | Status | Evidence / blocker |
|---:|---|---|---|---|
| 1 | Formal competency map | `COMPETENCY_MAP.md` | Drafted | Desk-built crosswalk; external validation pending |
| 2 | External reviewers: clinician, regulatory/quality, hardware/robotics | `GOVERNANCE_AND_EXTERNAL_REVIEW.md` | Pending | Terms of reference and templates exist; no reviewers convened |
| 3 | Regulatory claims verified by counsel/professional | `REGULATORY_VERIFICATION_LOG.md` | Pending | 54 claims desk-checked; none professionally signed off |
| 4 | Instructor answer keys, sample solutions, starter notebooks, grading calibration | `INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`, `labs/` | Drafted | 5 starter notebooks; anchor answer outlines; calibration protocol not yet run |
| 5 | Pilot with 5–10 learners and lab execution | `PILOT_PROTOCOL.md` | Pending | Protocol only; no pilot conducted |

## P1 — depth

| # | Item | Artifact | Status |
|---:|---|---|---|
| 6 | Specialty electives: oncology, cardiology, pathology, critical care, pediatrics/OB, public health | `23_specialty_electives.md` | Drafted |
| 7 | Hands-on ROS 2, firmware, EMC, HIL labs | `24_hardware_robotics_labs.md` | Drafted |
| 8 | Trial operations, HTA/market access, global regulatory | `25_trial_operations_hta_global_regulatory.md` | Drafted |
| 9 | Cloud/Kubernetes/streaming production labs | `26_cloud_streaming_production_labs.md` | Drafted |

## Reference stack verification

| Check | Result |
|---|---|
| `labs/mlops` Python tests | 5 passed |
| Python modules compile | Yes |
| YAML manifests parse | Yes |
| Terraform applied | No; no AWS credentials/tooling in authoring environment |
| Docker/Kubernetes run | No; no Docker/kubectl in authoring environment |
| Kafka run | No; no Docker in authoring environment |

The code is a teaching scaffold, not a deployment. Run `terraform validate/plan`, `docker build`, `kubectl apply --dry-run=server`, and the Kafka lab before using it in a class.

## What may be claimed now

- The package is a **comprehensive industry-oriented course outline and reading package**.
- It contains **30 modules including advanced electives**, labs, capstone briefs, and source packs.
- Its web sources and DOIs have been **desk-checked** with documented residual exceptions.
- It has **not** been clinically validated, regulatory-approved, accredited, or pilot-tested.

## What may not be claimed

- “Clinically validated.”
- “Regulatory approved.”
- “Accredited.”
- “Reviewed by clinicians” unless named reviewers consent.
- “Pilot-tested” unless the pilot is complete and reported.
- “All of biomedical data science is covered.”

## Exit criteria for “validated”

1. All P0 items complete.
2. External review report closed with no critical findings.
3. Regulatory log signed by a qualified reviewer for Tier 1 claims.
4. Pilot report meets success criteria.
5. Corrections propagated to modules, assessments, and source logs.
6. A dated version tag and public changelog entry exist.
