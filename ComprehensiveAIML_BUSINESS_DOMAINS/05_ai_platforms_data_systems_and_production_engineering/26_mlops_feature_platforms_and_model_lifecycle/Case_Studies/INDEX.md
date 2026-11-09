# Module 26 Case Studies — MLOps, Feature Platforms and Model Lifecycle

**Status:** planning catalog; no authored `CASE-*` files. Future cases must meet the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | reproducible ML release, feature, deployment, and lifecycle operations |
| Evidence boundary | proposed designs and illustrative calculations only until primary release artifacts are available |
| Review trigger | release manifest, feature contract, policy, model, tool, index, or deployment semantics change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-26-01 | Deploy a credit model safely | MLOps | predictive-ml | promote, hold, or roll back | atomic-release-bundle | rollout-regression | not-measured | proposed | — | needs domain controls and real release evidence |
| PLAN-26-02 | Feature freshness incident response | MLOps/data | incident | contain and restore fresh features | materialized-view | correctness-contract | not-measured | proposed | — | needs lineage and replay evidence |
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | MLOps | hybrid | promote/recover compatible versions | atomic-release-bundle | rollout-regression; human-in-the-loop | not-measured | proposed | 2026-11-06 (planning) | prompt-only rollback is insufficient |

## 3. Canonical reusable patterns

- **Atomic release bundle:** version the model, features, code, configuration, prompt, tools, index, and policy that jointly determine behavior.
- **Rollout regression:** compare staged traffic to a declared baseline while preserving a safe halt and recovery path.
- **Materialized view:** manage feature freshness, point-in-time meaning, ownership, and replay separately from model code.

## 4. Similarity / contrast map

| Case | Shared method | Critical distinction |
|---|---|---|
| release and feature incident | provenance, ownership, rollback | restoring a model cannot repair an invalid feature snapshot |
| credit release and IR005 | compatible bundles and approval | domain decision governance needs its own verified requirements |

## 5. Questions and future-case backlog

1. Which artifacts must share a compatibility gate?
2. What shadow execution is side-effect free, and what cannot it validate?
3. How do rollback and data/feature replay avoid reintroducing a bad state?
4. Which release metric is a guardrail rather than a promotion objective?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no executed promotion or recovery drill |

## 7. Pre-publish checks

- [ ] Create a linked 15-section `CASE-*` file with bundle manifest, compatibility test, rollback drill, and failure path.
- [ ] Separate proposed validation from results; reconcile source, evidence, stage, and editorial status.
