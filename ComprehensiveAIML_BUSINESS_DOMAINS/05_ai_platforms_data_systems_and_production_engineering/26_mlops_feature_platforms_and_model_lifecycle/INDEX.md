# Module 26 â€” MLOps, Feature Platforms and Model Lifecycle

> **Part:** [Part 05 â€” AI Platforms, Data Systems and Production Engineering](../INDEX.md)  
> **Prerequisites:** 06,11,25  
> **Purpose:** Operationalize reproducible training, deployment, feature management, approvals, rollback, and ownership.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | experiment tracking and registries | planned |
| 2 | feature stores | planned |
| 3 | CI/CD/CT for ML | planned |
| 4 | deployment patterns | planned |
| 5 | rollback and lifecycle controls | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [ML Release and Rollback Checklist](Cheat_Sheets/01_ML_Release_and_Rollback_Checklist.md) — release bundles, promotion gates and rollback controls; visible source inventory only.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Implementation practices, production patterns, domain uses, multi-agent operations, reliability, governance, cost, scaling, pilot rollout and checklists](../../00_SOURCE_COVERAGE_AGENTS.md#agent-production) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 14 / cheat sheet 14 — practical MLOps workflows](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-mlops) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 17 — evaluation, monitoring, quantization and reproducibility implementation](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-operations) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
