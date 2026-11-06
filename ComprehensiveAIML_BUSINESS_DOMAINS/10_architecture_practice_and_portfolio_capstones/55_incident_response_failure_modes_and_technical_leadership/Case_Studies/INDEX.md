# Module 55 Case Studies â€” Incident Response, Failure Modes and Technical Leadership

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. The catalog includes a draft illustrative case and planned briefs, not reviewed publications. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Training-data breach incident | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | robot fleet rollback | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | Proposed home | hybrid | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | `atomic-release-bundle` | `rollout-regression`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR028 | Bound request amplification before expensive parsing and validation | Proposed home | incident | Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily? | `bounded-work` | `resource-admission`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Supporting curriculum; shared brief | incident | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | `bounded-work` | `correctness-contract`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR030 | Diagnose production failures from violated algorithm invariants | Proposed home | incident | Which invariant failed, how can it be contained, and which replay would prevent recurrence? | `incident-postmortem` | `bounded-work`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR005 | [REF-IR005](../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults |
| PLAN-2026-IR028 | [REF-IR028](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md) | The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation |
| PLAN-2026-IR029 | [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims |
| PLAN-2026-IR030 | [REF-IR030](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md) | Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->

## Collection metadata

Owner: incident response curriculum maintainers. Review cadence: on substantive changes to source material, technology semantics or case assumptions. Last reviewed: not reviewed. Index version: draft. Streaming case identifiers use `CASE-STREAM-NNN`. This index is the collection source of truth. Audience: incident responders and system-design interview candidates.

## Case catalog

| Case ID | Case title | Domain | Case type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Content status | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-STREAM-001 | [Recovering a hot partition without losing customer-event ordering](01_hot_partition_consumer_lag_incident.md) | Real-time analytics | streaming-systems | How should operators contain partition lag without losing or duplicating effects? | Incident diagnosis and recovery | Partition-aware capacity planning | illustrative | proposed | draft | not reviewed | No actual incident or measured outcome; rates and timeline are hypothetical. |

## Canonical reusable pattern vocabulary

| Pattern | Mechanism and diagnostic signature | Preconditions and counterexample | Related case / method |
|---|---|---|---|
| Incident diagnosis and recovery | Distinguish competing causes, contain safely, reconcile and validate recovery | Requires observable boundaries; do not infer a cause from lag alone | CASE-STREAM-001; [SRE SLO methods](https://sre.google/workbook/implementing-slos/) |
| Partition-aware capacity planning | Compare arrival and service by partition rather than relying on aggregate averages | Useful with partition-affine work; a global sink bottleneck requires a different remedy | CASE-STREAM-001; [Kafka consumer groups](https://kafka.apache.org/documentation/#intro_consumers) |

## Cross-case similarity and contrast map

| Pattern | Supporting case | Contrasting case | Theory / methodology | Boundary learned |
|---|---|---|---|---|
| Partition-aware capacity planning | CASE-STREAM-001 is an illustrative exercise, not empirical support | No comparable completed case established | [Kafka consumer groups](https://kafka.apache.org/documentation/#intro_consumers) | Added group members do not split a single partition. |

## Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Is order-level independence safe? | CASE-STREAM-001 | Business invariants and transition histories | Audit cross-order state | High | Application owner | Before repartitioning |
| Can upstream safely throttle? | CASE-STREAM-001 | Acceptance and buffering semantics | Sustained overload and reconciliation test | High | Producer owner | Before containment rollout |
| Can cutover and rollback preserve effects? | CASE-STREAM-001 | Failure-injection and replay records | Fenced migration rehearsal | High | Streaming platform lead | Before migration |

## Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-STREAM-001 | 2026-11-07 | Initial draft catalog entry | Illustrative teaching exercise | Drafted; not reviewed | Streaming platform lead |

## Pre-publish index checks

- [ ] Complete substantive review of the draft and its source references.
- [ ] Confirm source links and compare claims to current deployed dependency versions.
- [ ] Validate all case assumptions and production test gates before operational use.
- [ ] Establish a comparable case or explicitly retain the missing contrast.
- [ ] Confirm case IDs, editorial state, evidence labels and actual file links before publication.
