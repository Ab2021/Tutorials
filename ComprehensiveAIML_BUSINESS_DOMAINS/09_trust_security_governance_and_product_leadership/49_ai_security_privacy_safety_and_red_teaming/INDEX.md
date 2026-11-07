# Module 49 â€” AI Security, Privacy, Safety and Red Teaming

> **Part:** [Part 09 â€” Trust, Security, Governance and Product Leadership](../INDEX.md)  
> **Prerequisites:** 16,27,28,30,31  
> **Purpose:** Defend models, data, agents, and users against misuse, leakage, supply-chain compromise, unsafe outputs, and adversarial behaviour.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | AI threat modelling | planned |
| 2 | prompt injection and tool abuse | partial: [enterprise agent approval gate](../../06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/Case_Studies/01_enterprise_agent_approval_gate.md) is a hypothetical, unreviewed case |
| 3 | data/model supply chain | planned |
| 4 | red teaming | planned |
| 5 | incident response | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | How should tenant adapters be trained, selected, versioned and rolled back? | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix |
| PLAN-2026-IR008 | Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring | Does a candidate pass the required suites with enough evidence to justify promotion? | Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Execution surfaces, identity, trust, approval gates, sandboxing and blast radius](../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals) | A hypothetical approval-gate case is an unreviewed partial draft; remaining focused work is planned |
| [Fine-tuning Module 13 — safety, alignment and interpretability](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
