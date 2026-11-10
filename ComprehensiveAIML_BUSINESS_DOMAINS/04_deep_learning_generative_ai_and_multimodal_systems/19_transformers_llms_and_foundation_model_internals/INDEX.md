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
| 4 | context architectures | [authored](Case_Studies/01_code_assistance_model_choice.md) |
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

Visible Module 11 source titles 11.1–11.5 are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). Lessons 11.1 and 11.3–11.5 are new drafts; 11.2 reuses the existing unreviewed [RoPE and position interpolation](../21_finetuning_alignment_and_model_customisation/Cheat_Sheets/03_RoPE_and_Position_Interpolation.md) sheet. Original source bodies were unavailable.

- [11.1 — Why Context Length Matters](11.1_Why_Context_Length_Matters.md) — draft; usable context budget, attention growth, retrieval alternatives and position-aware evaluation.
- [11.2 — RoPE and Position Interpolation](../21_finetuning_alignment_and_model_customisation/Cheat_Sheets/03_RoPE_and_Position_Interpolation.md) — mapped to existing unreviewed sheet; does not imply source-body inspection.
- [11.3 — Long Context Training Strategies](11.3_Long_Context_Training_Strategies.md) — draft; position/length curriculum, data controls, short-context regression and long-context gates.
- [11.4 — Memory-Efficient Long-Context Inference](11.4_Memory_Efficient_Long_Context_Inference.md) — draft; cache payload/peak distinction, admission, isolation and tail-latency validation.
- [11.5 — Evaluation of Long-Context Models](11.5_Evaluation_of_Long_Context_Models.md) — draft; position-aware evidence suite, retrieval comparison and quality/cost gates.

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

## Local coverage and artifact identities

This ledger preserves all five original topics, both original case rows, all three original sheet rows, the thirteen existing lessons, and the source-added syllabus. Editorial state is **draft**, independently of the proposed implementation stage and not-measured outcome evidence. Source filenames below identify visible inventory entries, not inspected original source bodies. No clipped filename is reconstructed.

| Order | Commitment and measurable outcome | Lesson home | Sheet identity / required focused scope | Case identity / bounded decision |
|---:|---|---|---|---|
| 1 | Tokenisation and embeddings: reconcile bytes, token IDs, vocabulary rows and usable context | 0.1, plus local tokenisation lesson | SH19-02 token budgeting; SH19-04 sequence probability; tokenisation/embedding compatibility | CASE-19-01 original code-assistance model choice; CASE-19-05 whether weight changes are needed |
| 2 | Attention and transformer blocks: derive shapes, dense counts, routed counts and overload boundaries | 1.1–1.4 | SH19-01 original transformer anatomy; focused attention, blocks, parameter counting and MoE sheets | CASE-19-06 dense versus routed feed-forward blocks |
| 3 | Pretraining and scaling: distinguish objectives, scoped fits and token-indexed schedules | 3.1–3.3 | Independent objectives, scaling and hyperparameter sheets | CASE-19-07 fixed-compute model/token choice; CASE-19-04 IR009 |
| 4 | Context architectures: distinguish usable context, positional extension, training and task evidence | 11.1, external 11.2, 11.3–11.5 | SH19-02 token budgeting; focused extension training, memory and positional-evaluation sheets; external RoPE reference unchanged | CASE-19-02 original long-context quality diagnosis |
| 5 | Inference behaviour: distinguish sampling policy, prefill/decode, admission and exact speculation | Local inference-behaviour lesson | SH19-03 original model selection matrix; focused admission/cache and speculative-correctness sheets | CASE-19-03 IR007 latency/admitted throughput |
| 6 | ft-foundations: factor sequence likelihood and trace training-stage artifact lineage | 0.1, 0.2 | Independent probability and training-stage contract sheets | CASE-19-05 unchanged model versus weight update |
| 7 | claude-prompting: explicit criteria and few-shot prompting | Local prompting lesson | Independent criteria and few-shot sheets | CASE-19-08 extraction prompt and feedback budget |
| 8 | claude-prompting: structured output/tool use and validation/retry feedback | Local prompting lesson | Independent schema/tool-contract and bounded-feedback sheets | CASE-19-08 |
| 9 | claude-prompting: batch processing, multi-pass review and domain4 practice questions | Local prompting lesson | Independent batch/review sheets; answered practice inside the local lesson and interview packs | CASE-19-08 |
| 10 | ft-alternative-architectures: Mamba selective state, RWKV recurrence and RetNet retention | Local architecture-specific lessons | Independent Mamba, RWKV and RetNet sheets | CASE-19-09 streaming-memory architecture choice |
| 11 | ft-alternative-architectures: structured/unstructured pruning and sparsification | Local pruning lesson | Independent structured/unstructured pruning and sparsification reasoning | CASE-19-09 |
| Final | Diagnose cross-topic failures without confusing symptoms with causes | All lessons | Final Common Issues artifact: at least 25 complete issues | Failure and recovery analysis in each case |

### Provenance and publication boundary

- Inventory: [`directory.txt`](../../../Templates/directory.txt), lines 223–231 (prompting), 842–855 (source sheets), 857–864 (foundations/architecture), 870–873 (pretraining), 919–924 (context), 951–953 (alternative architectures/pruning).
- Focused commitments: [prompting](../../00_SOURCE_COVERAGE_AGENTS.md#claude-prompting), [foundations](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-foundations), [transformers](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers), [pretraining](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-pretraining), [context](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-context), [alternatives](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-alternative-architectures).
- Supporting interview references: [IR007](../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir007-balance-interactive-inference-latency-and-admitted-throughput) and [IR009](../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir009-allocate-a-domain-training-budget-with-restart-headroom). Their scenarios and assumed figures are not measurements of the proposed local designs.
- Every case/sheet must carry its own inventory link, precise sources, numerical inputs, evidence gaps, and independent editorial state. This ledger does not certify that a planned identity has an authored artifact.
- Original source bodies remain unavailable; new teaching content may cover the visible technical titles using primary publications without pretending to reproduce inaccessible lessons. Runtime, quality, deployment and review outcomes remain not measured/not independently reviewed.
