# Module 23 â€” GPU Architecture, Accelerators and AI Hardware

> **Part:** [Part 05 â€” AI Platforms, Data Systems and Production Engineering](../INDEX.md)  
> **Prerequisites:** 04,19  
> **Purpose:** Match training and inference workloads to compute, memory, networking, accelerators, and energy constraints.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | GPU memory and parallelism | Unreviewed H100/roofline cheat-sheet drafts; Module 06.1–6.4 mapped lessons drafted |
| 2 | TPU/NPU/accelerator landscape | planned |
| 3 | interconnects and clusters | planned |
| 4 | precision and quantisation | planned |
| 5 | cost-energy-performance | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored source submodules

The visible Module 06 titles are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). These lessons are new drafts; the original source lesson bodies were not available.

- [6.1 — GPU Architecture Primer](6.1_GPU_Architecture_Primer.md) — draft; execution hierarchy, memory/bandwidth, roofline reasoning and benchmarking.

- [6.2 — Kernel Optimization and Fusion](6.2_Kernel_Optimization_and_Fusion.md) — draft; profiling, memory traffic, tiling, fusion tradeoffs, numerical validation and benchmarking.
- [6.3 — FlashAttention](6.3_FlashAttention.md) — draft; tiled exact attention, online softmax, IO/memory accounting, mask correctness and kernel validation.
- [6.4 — Mixed-Precision Training](6.4_Mixed_Precision_Training.md) — draft; range versus precision, accumulation policy, loss scaling, overflow and validation.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR007 | Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [NVIDIA hardware, CUDA operations, nvidia-smi, MIG, container toolkit, Kubernetes, DCGM, DGX/Base Command and TensorRT](../../00_SOURCE_COVERAGE_AGENTS.md#agent-nca-infrastructure) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA source containers, indexes and collation utilities](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-source-artifacts) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 01 / cheat sheet 01 — H100 architecture](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-h100) | H100 architecture sheet is an unreviewed partial draft; device validation and remaining focused work are planned |
| [CUDA Module 02 / cheat sheet 02 — programming foundations](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-programming) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 03 / cheat sheet 03 — asynchronicity and mbarrier](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-barriers) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 04 / cheat sheet 04 — TMA and tensor maps](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-tma) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 05 / cheat sheet 05 — cp.async.bulk and asynchronous movement](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-cp-async) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 06 / cheat sheet 06 — WGMMA fundamentals and descriptors](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-wgmma1) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 07 / cheat sheet 07 — WGMMA pipelines, FP8 and sparsity](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-wgmma2) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 08 / cheat sheet 08 — kernel design](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-kernel) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA Module 09 / cheat sheet 09 — named CUTLASS and flash.cu walkthroughs](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-walkthrough) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CUDA cheat sheet 13 — 50 Common Issues coverage target](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#cuda-issues) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 06 — GPU internals, fusion, FlashAttention and precision](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-gpu-internals) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
