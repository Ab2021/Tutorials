# ML Product Sense — Cheat Sheets

## 1. Overview
This collection is the reusable decision layer for deciding whether an ML product should exist, what it should optimize, and how users and operators retain control. No individual sheet has been authored in this directory. The catalog preserves the original twelve product-sense topics and the source-plan additions; a title is not coverage.

## 2. Topic ladder
| Level | Topic | Outcome | Prerequisite | Status |
|---|---|---|---|---|
| 1 | User problem; baseline; ML fit | State the unit, decision, non-ML baseline, and falsifiable benefit | basic product workflow | Planned |
| 2 | UX; metrics; experiments; human review | Define a north-star metric, guardrails, and escalation path | level 1 | Planned |
| 3 | Abuse; rollout; business case; stakeholder communication; interviews | Defend a constrained launch or rejection | levels 1–2 | Planned |
| Source extension | recommendations, search, fraud, NLP, forecasting, agents, CV, causal experiments, PM interviews | Apply product framing without treating a proxy as product value | relevant domain method | Planned |
| Final | Common Issues | Diagnose recurring product-decision errors | all applicable topics | Not authored |

## 3. Quick topic lookup
| Question | Intended topic | Exact decision | Failure warning |
|---|---|---|---|
| Should this be ML at all? | ML fit | compare rule/manual/ML alternatives | accuracy cannot justify a workflow with no action |
| What metric can ship? | North-star and guardrails | declare numerator, denominator, window, slices, and harm guardrail | engagement can hide cost or harm |
| When should humans intervene? | Human review | allocate scarce review capacity by uncertainty and consequence | review volume is not review quality |
| Can an observed lift be attributed to the feature? | Experiments / causal extension | choose randomized or explicitly limited observational design | before/after movement is confounded |

## 4. Reading paths
- **Beginner:** user problem → baseline → ML fit. Objective: write a decision contract before selecting a model.
- **Practitioner:** beginner path → UX → metrics → experiments → human review → rollout. Assumes ability to define event data and a baseline.
- **Advanced:** all original topics plus source-plan extensions; add abuse, business-case, and stakeholder constraints rather than optimizing a single proxy.

## 5. Prerequisite skills
Problem decomposition; basic descriptive statistics; ratios and denominators; experiment vocabulary; stakeholder mapping; privacy/security escalation routes; and ability to distinguish an observed association from an intervention effect.

## 6. What this collection does NOT cover
It does not provide a domain model implementation, legal determination, company benchmark, or claim that product patterns transfer between regulated, consumer, and internal workflows. Detailed training, serving, and reliability mechanics belong in neighboring collections.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Domain | First intended topic |
|---|---|---|---|---|
| `N` | eligible decision opportunities | count | nonnegative, frozen population | Metrics |
| `C` | completed desired actions | count | subset of `N` under a declared window | Metrics |
| `C/N` | completion rate | proportion | defined denominator and exposure policy | Metrics |
| `Δ` | treatment-minus-control estimand | outcome unit | randomized or stated causal assumptions | Experiments |
| review capacity | qualified decisions available per period | reviews/time | staffing and queue constraints | Human review |

## 8. Quick Reference Card directory
No authored card exists. Future cards will link here only after their 12-section sheets and final cards exist.

## 9. Common Issues locator
No Common Issues artifact exists. Required future artifact: at least 25 complete, distinct product-decision issues with diagnostic signals, competing causes, bounded correction, mitigation risk, and verification.

## 10. Empirical sources, versions and maintenance
The [product-sense source brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-product-sense) establishes visible-topic scope only; source bodies were not provided. No empirical results are registered. Revisit planned artifacts when a metric definition, action policy, population, review capacity, experiment design, source access, or applicable policy changes.

## 11. Collection acceptance checklist
- [x] The index retains the original topic ladder and source-plan extension.
- [x] Unauthored material is not linked as a completed sheet.
- [ ] Each topic has a 12-section, >=1000-line substantive sheet.
- [ ] Each sheet has checked examples, four diagnostics, and 10–15 answered questions.
- [ ] A final 25-issue Common Issues sheet exists.
- [ ] Source evidence and independent review are complete.
