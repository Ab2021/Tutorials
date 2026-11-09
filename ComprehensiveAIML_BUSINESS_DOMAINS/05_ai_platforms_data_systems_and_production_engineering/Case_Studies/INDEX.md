# Part 05 Case Studies — AI Platforms, Data Systems and Production Engineering

**Collection status:** planning catalog; no part-level `CASE-*` artifact exists.  
**Contract:** [course standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). Part cases must integrate at least two Part 05 modules without duplicating their canonical module case.

## 1. Collection metadata

| Field | Value |
|---|---|
| Scope | Cross-module platform, data-system, and production-engineering decisions |
| Case types | analytical, predictive-ml, llm-agentic, streaming-systems, incident, or genuinely integrated hybrid |
| Evidence / stage | new designs are illustrative and proposed until accessible primary evidence says otherwise |
| Maintenance trigger | material change to workload, source evidence, hardware, dependency semantics, or cost model |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-P05-00A | Size an inference cluster | hardware/serving | hybrid | select capacity and admission policy | resource-admission | cost-latency-tradeoff | not-measured | proposed | — | canonical home: Module 23 |
| PLAN-P05-00B | Profile memory-bound training | hardware/training | incident | select bottleneck-removal change | resource-accounting | correctness-contract | not-measured | proposed | — | canonical home: Module 23 |
| PLAN-P05-01 | Train a large model on limited cluster capacity | training/platform | hybrid | select parallelism, checkpoint, and capacity plan | resource-admission | atomic-release-bundle | not-measured | proposed | — | canonical home: Module 24 |
| PLAN-P05-02 | Recover from checkpoint failures | training/platform | incident | restore a consistent generation | atomic-release-bundle | correctness-contract | not-measured | proposed | — | canonical home: Module 24 |
| PLAN-P05-03 | Multi-model inference gateway | serving/platform | llm-agentic | route eligible requests | eligibility-before-ranking | cost-latency-tradeoff | not-measured | proposed | — | canonical home: Module 25 |
| PLAN-P05-04 | Reduce p99 LLM latency | serving/hardware | hybrid | select a latency intervention | resource-admission | correctness-contract | not-measured | proposed | — | canonical home: Module 25 |
| PLAN-P05-05 | Deploy a credit model safely | MLOps | predictive-ml | promote or halt a release | atomic-release-bundle | rollout-regression | not-measured | proposed | — | canonical home: Module 26 |
| PLAN-P05-06 | Feature freshness incident response | data/MLOps | incident | contain and recover stale features | materialized-view | correctness-contract | not-measured | proposed | — | canonical home: Module 26 |
| PLAN-P05-07 | Broken upstream schema incident | data reliability | incident | quarantine, replay, or roll forward | correctness-contract | idempotent-side-effects | not-measured | proposed | — | canonical home: Module 27 |
| PLAN-P05-08 | Silent label corruption investigation | data reliability | incident | decide retraining eligibility | distribution-shift | evaluation-mismatch | not-measured | proposed | — | canonical home: Module 27 |
| PLAN-P05-09 | RAG quality regression | evaluation/serving | llm-agentic | gate release and choose rollback | evaluation-mismatch | rollout-regression | not-measured | proposed | — | canonical home: Module 28 |
| PLAN-P05-10 | Agent tool-use failure analysis | evaluation/platform | incident | isolate failing contract | correctness-contract | idempotent-side-effects | not-measured | proposed | — | canonical home: Module 28 |
| PLAN-P05-11 | Internal AI platform design | API/platform | hybrid | choose platform boundaries | authorization-scope | resource-admission | not-measured | proposed | — | canonical home: Module 29 |
| PLAN-P05-12 | Cloud-cost anomaly response | FinOps/platform | incident | contain spend without unsafe degradation | cost-latency-tradeoff | resource-admission | not-measured | proposed | — | canonical home: Module 29 |

## 3. Canonical reusable patterns

- **Atomic release bundle:** promote and recover model, data/feature, policy, code, and configuration versions as a compatible unit.
- **Correctness contract:** name the invariant, owner, input/output grain, and recovery behavior before tuning a metric.
- **Resource admission:** protect a finite capacity or budget under peak load and partial failure.
- **Evaluation mismatch:** distinguish proxy improvement from the decision outcome it is meant to support.

## 4. Similarity / contrast map

| Cluster | Similarity | Critical distinction |
|---|---|---|
| Training capacity and inference latency | both need capacity arithmetic and cancellation/failure assumptions | training optimizes sustained progress; serving has deadline and isolation constraints |
| Release, feature, and data incidents | all need versioned provenance and rollback | a replay-safe schema fix does not prove labels remain valid |
| Evaluation and agent cases | both use gated evidence | a judge score is not task completion or authorization evidence |
| Platform and FinOps cases | both allocate shared resources | cost controls must not weaken tenancy or safety boundaries |

## 5. Questions and future-case backlog

1. Which two modules genuinely change the decision, rather than merely appearing in an architecture diagram?
2. What primary evidence could change the illustrative status of each proposed case?
3. What fault-replay, recovery, and sensitivity evidence is required before an action is called successful?
4. Which module catalog owns the canonical detailed case when topics overlap?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | Part catalog | reconciled as planned | no authored part case and no measured outcome evidence |

## 7. Pre-publish checks

- [ ] Assign `CASE-*` only to an extant 15-section case with Section 15 last.
- [ ] Link each cross-module case to its module sheets and canonical module-case relationship.
- [ ] Check units, calculations, sources, failure/recovery paths, and 12–18 answered interview questions.
- [ ] Reconcile this catalog's evidence, stage, and editorial status against actual files.
