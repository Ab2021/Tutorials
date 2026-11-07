# Module 19 â€” Transformers, LLMs and Foundation-Model Internals

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 05,17,18  
> **Purpose:** Understand tokenisation, attention, scaling, context, and inference so model choices are grounded in mechanics.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | tokenisation and embeddings | planned |
| 2 | attention and transformer blocks | Submodules 1.1–1.4 drafted; focused sheets and independent review remain planned |
| 3 | pretraining and scaling | Submodules 3.1–3.3 drafted; focused sheets, empirical validation and bounded case remain planned |
| 4 | context architectures | planned |
| 5 | inference behaviour | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored source submodules

The visible Module 00 and Module 01 source titles are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). These course lessons are new drafts; the original source lesson bodies were not available for inspection.

- [0.1 — What Is a Large Language Model?](01_What_Is_a_Large_Language_Model.md) — draft; token-sequence probability, NLL, objective boundaries, and method selection.
- [0.2 — Modern LLM Training Pipeline](02_Modern_LLM_Training_Pipeline.md) — draft; data lineage, training stages, evaluation, release, and operational gates.
- [1.1 — Attention Mechanism](03_Attention_Mechanism.md) — draft; scaled dot-product derivation, tensor shapes, masking, cost and failure diagnostics.
- [1.2 — Transformer Block Architecture](04_Transformer_Block_Architecture.md) — draft; residual/norm ordering, FFN structure, parameter accounting and system boundaries.
- [1.3 — Scaling and Parameter Counting](05_Scaling_and_Parameter_Counting.md) — draft; dense parameter accounting, compute approximations, empirical scaling limits and controlled sweep design.
- [1.4 — Mixture of Experts](06_Mixture_of_Experts.md) — draft; total versus active capacity, top-k routing, capacity, dispatch and load diagnostics.
- [3.1 — Training Objectives](3.1_Training_Objectives.md) — draft; cross-entropy, autoregressive/masked/contrastive objective boundaries and compute proxy.
- [3.2 — Scaling Laws](3.2_Scaling_Laws.md) — draft; fit scope, compute/data allocation, exponent fitting, uncertainty and extrapolation limits.
- [3.3 — Pretraining Hyperparameters](3.3_Pretraining_Hyperparameters.md) — draft; optimizer/batch/schedule definitions, controlled tuning, numerical stability and resume state.
These are module lessons, not substitutes for the separately planned focused cheat sheets, cases, or independently reviewed publication artifacts.

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

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list. Each technical title is a teaching commitment; linked briefs distinguish authored draft lessons from unwritten sheets, cases, and validation. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) for cheat sheets and cases; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Explicit criteria, few-shot prompting, structured output/tool use, validation/retry feedback, batch work, multi-pass review and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-prompting) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 01 / cheat sheet 01 — attention, blocks, scaling and MoE](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers) | Submodule 1.1 lesson drafted; focused sheets for attention, blocks, scaling and MoE, remaining submodules, a bounded case, review and empirical validation remain planned |
| [Fine-tuning Module 03 / cheat sheet 02 — objectives and scaling laws](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-pretraining) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 11 / cheat sheet 10 — context extension](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-context) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 16 / cheat sheet 13 — alternative architectures, pruning and efficiency](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-alternative-architectures) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 00 — LLM foundations and modern training pipeline](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-foundations) | Submodule lessons 0.1 and 0.2 drafted; focused sheets, a bounded case, independent review and empirical validation remain planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
