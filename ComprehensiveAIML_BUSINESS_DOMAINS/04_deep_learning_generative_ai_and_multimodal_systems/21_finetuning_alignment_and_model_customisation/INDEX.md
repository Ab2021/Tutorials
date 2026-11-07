# Module 21 â€” Fine-Tuning, Alignment and Model Customisation

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 19,20  
> **Purpose:** Choose between prompting, retrieval, fine-tuning, preference optimization, distillation, and adapters based on evidence.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | curation and quality | Submodules 2.1–2.4 drafted; focused sheets, case and independent review remain planned |
| 2 | SFT and PEFT | [LoRA](Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md) and [QLoRA](Cheat_Sheets/02_QLoRA_Quantized_Base_Adapter_Finetuning.md) drafts; SFT and broader scope planned |
| 3 | preference optimisation | planned |
| 4 | distillation and model merging | planned |
| 5 | regression evaluation | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

- [RoPE and position interpolation](Cheat_Sheets/03_RoPE_and_Position_Interpolation.md) — unreviewed draft on long-context adaptation, rotary geometry and serving boundaries; remaining context-extension topics and cases are planned.

## Authored source submodules

The visible Module 02 source titles are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). These course lessons are new drafts; the original source lesson bodies were not available for inspection.

- [2.1 — Data Sources and Scale](2.1_Data_Sources_and_Scale.md) — draft; governance, volume accounting, mixture planning, splits and data-value limits.
- [2.2 — Data Curation Pipeline](2.2_Data_Curation_Pipeline.md) — draft; governed acquisition, parsing, filtering, deduplication, split integrity and lineage.
- [2.3 — Tokenization](2.3_Tokenization.md) — draft; vocabulary/ID compatibility, Unicode, special tokens, token budgets and multilingual evaluation.
- [2.4 — Data Formatting for Training](2.4_Data_Formatting_for_Training.md) — draft; chat serialization, target masks, truncation, packing and batch validation.


## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | How should tenant adapters be trained, selected, versioned and rolled back? | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix |
| PLAN-2026-IR009 | Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery | Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget? | Held-out domain and general quality, full training cost, achieved utilization and restart/reproducibility drills |
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | When do repository-aware completion or language adapters justify their added complexity? | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Should a speech adapter be promoted, and to which evaluated product and language domains? | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Fine-tuning source provenance, hidden directories and contribution artifacts](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-source-artifacts) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 07 / cheat sheet 05 — SFT, reward modeling and PPO](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-alignment-core) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 07 — DPO, GRPO, process supervision and constitutional feedback](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-alignment-grpo) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 08 / cheat sheet 07 — PEFT and model merging](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-peft) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 15 / cheat sheet 08 — continual learning and forgetting](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-continual) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning cheat sheet 15 — 50 Common LLM Fine-Tuning Issues target](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-common-issues) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 02 — data curation and tokenization](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-data) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 15 — continued pretraining, sampling, distillation and multi-task adaptation](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-advanced-offline) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 15 — KTO, ORPO, SimPO, RLVR, DAPO, REINFORCE++ and self-play](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-advanced-rl) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 17 — merging, context extension and end-to-end production implementation](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-composition) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 17 — adapter, SFT, memory, distributed and preference implementation](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-training) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
