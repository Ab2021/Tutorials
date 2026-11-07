# Module 25 â€” Model Serving, Inference Optimisation and Cost

> **Part:** [Part 05 â€” AI Platforms, Data Systems and Production Engineering](../INDEX.md)  
> **Prerequisites:** 04,19,23  
> **Purpose:** Deliver models with predictable latency, quality, cost, and availability across batch, online, and edge workloads.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | serving architectures | Source Module 10 submodule 10.4 mapped as an unreviewed draft |
| 2 | batching, caching and routing | Source Module 10 submodule 10.1 mapped as an unreviewed draft; remaining visible Module 10 submodules are planned |
| 3 | quantisation and compilation | Source Module 10 submodule 10.2 mapped as an unreviewed draft; compilation remains planned |
| 4 | autoscaling | planned |
| 5 | cost-quality-latency optimisation | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored source submodules

Visible Module 10 source titles 10.1–10.4 are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). These are new drafts because the original source lesson bodies were unavailable.

- [10.1 — Inference Optimization: KV Cache and Batching](10.1_Inference_Optimization_KV_Cache_and_Batching.md) — draft; KV payload accounting, prefill/decode, scheduling, admission, isolation and SLO validation.
- [10.2 — Quantization for Inference](10.2_Quantization_for_Inference.md) — draft; storage/compute distinction, compatibility, calibration, validation and rollback gates.
- [10.3 — Speculative Decoding](10.3_Speculative_Decoding.md) — draft; target/draft roles, acceptance semantics, correctness, workload slices and fallback gates.
- [10.4 — Serving Architecture and Deployment](10.4_Serving_Architecture_and_Deployment.md) — draft; serving-plane contracts, authorization/routing, progressive delivery and rollback.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR002 | Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets | Which eligible provider and fallback path satisfy residency, capability, deadline and spend requirements? | Constraint violations, successful-task cost, deadline misses and fallback correctness on fault replays |
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | How should tenant adapters be trained, selected, versioned and rolled back? | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix |
| PLAN-2026-IR007 | Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Infrastructure substrate, model asset, serving systems, interaction contracts and long-context stress](../../00_SOURCE_COVERAGE_AGENTS.md#agent-models-serving) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 10 / cheat sheet 09 — inference and deployment](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-inference) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [On-device SLM/LLM, GGUF, llama.cpp, Gemma3, KV cache, FAISS retrieval, medical RAG, QLoRA, ONNX embeddings and sovereign deployment](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-edge-llm) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
