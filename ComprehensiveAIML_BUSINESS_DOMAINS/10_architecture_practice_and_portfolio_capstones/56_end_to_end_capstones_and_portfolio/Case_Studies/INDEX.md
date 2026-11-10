# Module 56 Case Studies — End-to-End Capstones and Portfolio

## 1. Collection metadata

**Audience:** portfolio authors, reviewers, and system-design candidates. **Owner:** capstone curriculum maintainers. **Index version:** draft. **Review cadence:** whenever evidence, workload, versions, or scope changes. This index distinguishes illustrative evidence from document state. Source names in the visible directory are inventory, not project artifacts.

## 2. Case catalog

| Case ID | Case title | Domain | Case type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-CAPSTONE-001 | [Choose a release gate for a local signal-to-explanation demonstrator](01_local_signal_demo_release_gate.md) | edge portfolio demonstrator | hybrid | Is evidence sufficient for a trace-checked claim? | evidence-bounded release gate | end-to-end contract; bounded-work | illustrative | proposed | not reviewed | Standalone constructed-oracle smoke checks state and arithmetic; no app, UI visual, HIL, or operational result. |
| PLAN-CAPSTONE-002 | Choose from six cross-domain capstones | cross-domain | analytical | Which scope is feasible with actual evidence? | scope selection | milestone evidence | not-measured | proposed | not reviewed | planned; no file |
| PLAN-CAPSTONE-003 | Complete architecture review | portfolio review | analytical | Does a proposed architecture meet stated contracts? | architecture review | failure-path review | not-measured | proposed | not reviewed | planned; no file |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Preconditions / counterexample | Related case |
|---|---|---|---|
| evidence-bounded release gate | map claim tier to fixture, oracle, manifest, and limitation | no claim may outrun the weakest unverified arrow; a screenshot is counterexample | CASE-CAPSTONE-001 |
| end-to-end contract | name input, state, output, owner, and recovery boundary | needs a defined unit/identity; component success alone is insufficient | CASE-CAPSTONE-001 |
| bounded-work | validate/admit work under length, queue, and identity limits | limits must preserve a declared failure response, not silently discard evidence | CASE-CAPSTONE-001 |

## 4. Similarity / contrast map

| Pattern | Supporting case | Contrast | Transfer boundary |
|---|---|---|---|
| evidence tier | CASE-CAPSTONE-001 uses constructed values and proposed validation | an observed deployment would need workload/window/denominator evidence | no clinical or production inference |
| recovery contract | duplicate/frame replay is explicit | a non-idempotent external effect needs a different oracle | identity/transaction boundary must be named |

## 5. Questions and future-case backlog

| Open question | Origin | Needed evidence | Proposed action | Priority | Owner | Trigger |
|---|---|---|---|---|---|---|
| Is hardware protocol valid? | CASE-CAPSTONE-001 | versioned HIL traces and setup manifest | run safe hardware test | high | firmware/service owners | hardware available |
| Can retrieval citations support displayed claims? | CASE-CAPSTONE-001 | corpus provenance and query audit | construct evaluation suite | high | retrieval owner | corpus changes |
| Which capstone scope is feasible? | PLAN-CAPSTONE-002 | time, hardware, source, and milestone inventory | compare options | high | learner/mentor | planning cycle |
| Does review find unowned failure paths? | PLAN-CAPSTONE-003 | interface and recovery contracts | architecture review | high | technical lead | release candidate |

## 6. Review and retirement log

| Case ID | Date | Change | Evidence trigger | Action | Owner |
|---|---|---|---|---|---|
| CASE-CAPSTONE-001 | 2026-11-09 | initial illustrative draft | visible source inventory only | draft; not reviewed | capstone technical lead |
| CASE-CAPSTONE-001 | 2026-11-10 | expanded source/frame/queue/retrieval/tier recovery analysis and embedded synthetic oracle | illustrative documentation smoke only | remains draft DESIGN-only; not reviewed | capstone technical lead |

## 7. Pre-publish checks

- [ ] Every case has 15 substantive sections and meets the course line/depth gate.
- [ ] Case catalog links, IDs, evidence, implementation stage, and editorial state are reconciled.
- [ ] Primary sources, versions, real traces/HIL evidence, and numerical claims are independently reviewed.
- [ ] Proposed validation is not described as executed validation.
- [ ] Planned cases are authored or intentionally retained as planned with an honest reason.
