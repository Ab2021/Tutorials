# Evaluation, Reliability and Incidents — Cheat Sheets

## 1. Overview
This collection covers evidence design and operational reliability across predictive models, assistants, streaming, edge systems, and incidents. No local individual sheet is authored. The index preserves original evaluation/reliability topics and the two reference-informed briefs without treating reference narratives or their figures as evidence.

## 2. Topic ladder
| Level | Topic | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Evaluation design; offline/online gaps; human evaluation | define a versioned evaluand, population, comparator, and acceptance rule | metrics and sampling basics | Planned |
| 2 | Drift; SLOs; tracing; red teaming | select observable signals and a bounded response | level 1; system boundaries | Planned |
| 3 | Canaries; incident command; postmortems; continuous improvement | contain, replay, and learn without fabricating causality | levels 1–2 | Planned |
| Source extension IR008 | manifests, contamination, judges, over-refusal, sandbox scoring | decide whether a release has sufficient evidence | evaluation design | Planned |
| Source extension IR030 | hash flooding, BFS memory, regex backtracking, recursive depth, ORM N+1, overflow, ties, floating termination | identify the violated invariant and replay it safely | algorithm contracts | Planned |
| Final | Common Issues | diagnose cross-cutting evaluation and recovery failures | all rows | Not authored |

## 3. Quick topic lookup
| Question | Intended topic | Use condition | Failure warning |
|---|---|---|---|
| Does a candidate pass? | Evaluation matrix / IR008 | suites, versions, and critical failures are frozen | a judge score is not deployment safety |
| Is a metric regression real? | Offline/online gaps | cohorts and eligibility match | different populations invalidate direct comparison |
| How much unreliability is tolerable? | SLO/error budget | user journey and window are explicit | availability alone can conceal bad outputs |
| What failed during an incident? | IR030/postmortem | invariant and replay oracle exist | a plausible story is not root-cause evidence |

## 4. Reading paths
- **Beginner:** evaluation design → offline/online gaps → human evaluation. Objective: separate measurement definition from score calculation.
- **Practitioner:** beginner path → drift → SLOs → tracing → canaries → incident command.
- **Advanced:** all topics plus IR008 and IR030, using adversarial bounded traces and explicit recovery contracts.

## 5. Prerequisite skills
Ratios and uncertainty; cohort manifests; time windows; logs/traces; version control; fault isolation; basic algorithmic time/space reasoning; and the difference between monitoring evidence and a causal outcome claim.

## 6. What this collection does NOT cover
It does not certify a particular framework, judge model, SLO target, production incident, or safety claim. Detailed product value and ML-system implementation are handoffs to the sibling collections. Underlying interview-reference source bodies remain reference-only where specified.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Valid domain | First intended topic |
|---|---|---|---|---|
| evaluand | exact system, version, population, task, and window being judged | structured manifest | immutable per comparison | Evaluation design |
| `n` | eligible evaluated units | count | nonnegative | Evaluation matrix |
| `f` | critical failures | count | subset of `n`, rubric-defined | Evaluation matrix |
| SLI | measurable service-level indicator | ratio/time/count | tied to a user journey | SLOs |
| error budget | allowed bad-event amount over window | count/proportion | declared window and objective | SLOs |
| invariant | condition that must hold through operation | boolean contract | operation-specific | Incident postmortem |

## 8. Quick Reference Card directory
No authored cards exist. This directory will contain only links to completed 12-section sheets.

## 9. Common Issues locator
No final Common Issues artifact exists. A completed artifact must contain at least 25 distinct complete issues and cannot count the reference-informed incident units as locally authored or observed incidents.

## 10. Empirical sources, versions and maintenance
`SRC-IR008` and `SRC-IR030` are [reference-informed planned briefs](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md), not evidence of local execution. The local source mapping is [ds-practical-sources](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-practical-sources). Revisit when evaluation manifests, task taxonomy, judge calibration, load profile, algorithm implementation, replay oracle, or source access changes.

## 11. Collection acceptance checklist
- [x] Original topics and both source briefs are visible in the ladder.
- [x] No uncreated sheet/card is misrepresented as linked content.
- [ ] Individual 12-section sheets with checked examples and interview packs exist.
- [ ] IR008 and IR030 commitments have focused coverage.
- [ ] Final Common Issues has >=25 complete entries.
- [ ] Independent correctness, source, and navigation review is complete.
