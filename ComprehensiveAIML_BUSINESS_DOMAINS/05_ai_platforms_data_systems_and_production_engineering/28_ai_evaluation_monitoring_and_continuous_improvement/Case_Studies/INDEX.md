# Module 28 Case Studies — AI Evaluation, Monitoring and Continuous Improvement

**Status:** planning catalog; no authored `CASE-*` files. Future cases must follow the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | evaluation design, human/judge assessment, monitoring, and regression gates |
| Evidence boundary | proposed illustrative protocols; a score is not a deployment result |
| Review trigger | task manifest, scorer, judge, rater protocol, policy, model, or traffic changes |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-28-01 | RAG quality regression | evaluation | llm-agentic | gate, roll back, or investigate | evaluation-mismatch | rollout-regression | not-measured | proposed | — | needs manifest and task-completion evidence |
| PLAN-28-02 | Agent tool-use failure analysis | evaluation | incident | isolate contract failure | correctness-contract | idempotent-side-effects | not-measured | proposed | — | needs trace and sandbox evidence |
| PLAN-2026-IR001 | Gate retraining on corpus provenance and long-tail quality | data/evaluation | hybrid | admit or quarantine documents | distribution-shift | data-leakage; evaluation-mismatch | not-measured | proposed | 2026-11-06 (planning) | source names/costs unverified |
| PLAN-2026-IR008 | Block assistant releases on auditable quality and safety gates | evaluation | llm-agentic | determine release eligibility | evaluation-mismatch | data-leakage; rollout-regression | not-measured | proposed | 2026-11-06 (planning) | judge score is not deployment guarantee |

## 3. Canonical reusable patterns

- **Evaluation mismatch:** define the decision outcome and show why a proxy score is or is not adequate.
- **Rollout regression:** compare candidate and baseline on a frozen manifest and production guardrails.
- **Correctness contract:** score tool behavior against typed input, authorization, state, output, and side-effect rules.

## 4. Similarity / contrast map

| Case | Similarity | Critical distinction |
|---|---|---|
| RAG regression and IR008 | manifests, slices, promotion gates | judge calibration does not establish user utility |
| tool failure and RAG | traceable evidence and recovery | a correct answer can still have an unsafe tool side effect |

## 5. Questions and future-case backlog

1. What population and critical slice are excluded by an aggregate score?
2. Which contamination check applies before a benchmark becomes a gate?
3. How is judge–human disagreement measured and adjudicated?
4. Which failure blocks release even if the average metric improves?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no executed evaluation or monitored outcome |

## 7. Pre-publish checks

- [ ] Link a 15-section `CASE-*` with a versioned manifest, baseline, uncertainty, failure path, and Section 15 card.
- [ ] Reconcile evidence label, stage, and editorial status; distinguish proposed protocol from executed test.
