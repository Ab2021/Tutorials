# Pharma, Healthcare and Life Sciences — Cross-Domain Cheat Sheets

## 1. Overview

This collection teaches reusable evidence-to-action controls that apply across commercial, clinical/RWE/safety, and manufacturing/quality workflows. It assumes basic ratios and timestamp reasoning. Readiness means a learner can define a decision grain, prevent future-information use, size a review queue, and distinguish recommendation from authorized action. Source filenames assigned elsewhere establish scope only; unavailable source bodies remain OPEN.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | Editorial status |
|---|---|---|---|---|---|
| 1 | [01 Evidence-to-Action Controls](./01_Evidence_to_Action_Controls.md) | Audit a recommendation’s time, evidence, capacity, and authority boundary. | Ratios; event timestamps | `r=a/n`, `B=max(0,lambda-mu)t` | draft; below course 1,000-line publication gate |
| Final | Common Issues | Cross-domain diagnostic issues. | Completed focused sheets | competing-cause diagnosis | not authored |

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| Can this output act now? | 01 §4 | Output could affect a workflow | A score does not grant authority. |
| Is the queue sustainable? | 01 §5, §9 | Human review is required | Precision does not measure backlog. |
| Did a feature leak future state? | 01 §3, §10 | Evaluating a historical decision | Later updates are not prediction-time inputs. |

## 4. Reading paths — three options

- **Beginner:** 01 §§1–5; identify a unit, denominator, and timestamp.
- **Practitioner:** 01 §§1–10; design a recommendation-to-review path and capacity check.
- **Advanced:** 01 §§1–11, then the domain-specific collections; add authority, recovery, and evidence-ledger review.

## 5. Prerequisite skills

| Skill | Required capability | Reference / boundary |
|---|---|---|
| Ratio arithmetic | Preserve numerator, denominator, and units | 01 §4–5 |
| Event-time reasoning | Establish input availability at decision time | 01 §3, §10 |
| Experimental reasoning | Separate observed association from causal effect | 01 §4 |
| Workflow control | Name an owner and authorized state transition | 01 §2, §10 |

## 6. What this collection does NOT cover

It does not provide clinical diagnosis, legal or regulatory interpretation, disease-specific treatment guidance, or organization-specific deployment claims. Clinical evidence details hand off to `../02_clinical_rwe_safety_and_document_intelligence/`; commercial intervention design to `../01_commercial_kda_and_omnichannel/`; batch and quality operations to `../03_manufacturing_quality_and_supply_chain/`.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / type | Domain | First appearance |
|---|---|---|---|---|
| `a` | evidence-complete actionable records | count | nonnegative integer | [01 §4](./01_Evidence_to_Action_Controls.md#41-eac-01) |
| `n` | eligible records | count | positive integer for rate | [01 §4](./01_Evidence_to_Action_Controls.md#41-eac-01) |
| `r` | actionable rate | dimensionless fraction | `[0,1]` | [01 §4](./01_Evidence_to_Action_Controls.md#41-eac-01) |
| `B` | queued reviews | reviews | nonnegative | [01 §4](./01_Evidence_to_Action_Controls.md#42-eac-02) |
| `lambda`, `mu` | arrival and completion rates | reviews/hour | nonnegative | [01 §4](./01_Evidence_to_Action_Controls.md#42-eac-02) |
| `t_d` | decision-time cutoff | timestamp | each keyed event | [01 §3](./01_Evidence_to_Action_Controls.md#3-foundational-definitions) |

## 8. Quick Reference Card directory

| File / card | Core rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [01 §12](./01_Evidence_to_Action_Controls.md#12-quick-reference-card) | no future input; no unauthorized transition | evidence-backed workflow | named owner and timestamp | source, time, or authority unknown |

## 9. Common Issues locator

No final Common Issues artifact has been authored. The draft sheet contains 25 HYPOTHETICAL diagnostic entries in §8, but they are not a substitute for the required final 12-section troubleshooting artifact.

## 10. Sources, versions and maintenance

| File | Evidence / source basis | Status | Open evidence | Review trigger |
|---|---|---|---|---|
| 01 | DERIVED and ILLUSTRATIVE; source bodies OPEN | draft | primary workflow evidence and independent editorial review | authority policy, source access, or workflow changes |

The visible source assignments are in `../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md` under pharma commercial, clinical, and operations. They identify planned outcomes, not accessible source bodies or measured results.

## 11. Acceptance checklist

- [ ] All focused sheets meet the 12-section and 1,000-substantive-line gate.
- [ ] A final Common Issues artifact has at least 25 distinct complete issues.
- [x] The current draft retains the 12 ordered sheet sections and checked illustrative arithmetic.
- [x] Draft evidence labels distinguish DERIVED, ILLUSTRATIVE, and OPEN claims.
- [ ] Independent technical and source review is recorded.
- [x] Current local navigation points only to existing artifacts.
