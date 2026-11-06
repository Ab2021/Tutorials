# Module 19 â€” Transformers, LLMs and Foundation-Model Internals

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 05,17,18  
> **Purpose:** Understand tokenisation, attention, scaling, context, and inference so model choices are grounded in mechanics.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | tokenisation and embeddings | planned |
| 2 | attention and transformer blocks | planned |
| 3 | pretraining and scaling | planned |
| 4 | context architectures | planned |
| 5 | inference behaviour | planned |

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
| PLAN-2026-IR007 | Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests |
| PLAN-2026-IR009 | Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery | Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget? | Held-out domain and general quality, full training cost, achieved utilization and restart/reproducibility drills |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Explicit criteria, few-shot prompting, structured output/tool use, validation/retry feedback, batch work, multi-pass review and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-prompting) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 01 / cheat sheet 01 — attention, blocks, scaling and MoE](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 03 / cheat sheet 02 — objectives and scaling laws](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-pretraining) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 11 / cheat sheet 10 — context extension](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-context) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 16 / cheat sheet 13 — alternative architectures, pruning and efficiency](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-alternative-architectures) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 00 — LLM foundations and modern training pipeline](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-foundations) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
