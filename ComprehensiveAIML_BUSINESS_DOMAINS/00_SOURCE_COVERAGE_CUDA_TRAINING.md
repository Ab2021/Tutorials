# CUDA, LLM training, fine-tuning and speech — exhaustive source coverage plan

**Document status: planned.** This is a topic architecture and future-content brief, not a completed lesson library, implemented training system, checked benchmark, or evidence of deployment. Every example below is **ILLUSTRATIVE**: inputs are constructed teaching values, calculations are authoring targets, and validation is proposed, not performed. Source-specific API, instruction, hardware, framework and model names are study targets requiring authoritative versioned revalidation, not assertions about inspected implementations.

## Source scope and provenance

This plan covers every parsed visible inventory node under **CUDA**, **finetuning_course**, and **Finetune_LLM**: **322 nodes** (141 CUDA; 180 finetuning_course; 1 Finetune_LLM). The snapshot derives from the **first fenced TRAIN tree** in [TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md), matching the supplied tree. It is a directory/screenshot reconstruction, not an inspection of lesson bodies, source programs, exercises, benchmark logs, citations, or hidden directory contents. A topic title proves visibility only, not authoring depth or correctness.

The permanent machine-readable [source crosswalk](00_SOURCE_TOPIC_CROSSWALK.json) contains each exact `source_path`, existing course-relative `destination`, explicit HTML `plan_anchor` in this document, `plan_file`, and one disposition. For this slice, dispositions are **planned-topic** (230), **supporting-artifact** (26), **source-container** (37), and **unresolved-source** (29). Containers organize scope; README/index, collation and replication/contribution artifacts are not technical lessons. Hidden directories and clipped filenames remain unresolved even when a visible prefix can support a limited topic brief.

## Authoring contract and publication boundary

- Follow the [Case Study Content Master Template](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md): preserve the 15-section case structure and seven-section collection index; one consequential decision per case; source basis, baseline, assumptions, input/output/data-time boundaries, alternatives, traceable design, validation, uncertainty, failure boundaries, action and honest outcome status. Future cases here start as proposed illustrative designs with outcome **not-measured**.
- Follow the [Cheat Sheet Content Master Template](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md): preserve all 12 individual sections, the 11-section collection index and final quick-reference card; foundations→core→advanced ladder; definitions/units/shapes; valid derivations and mechanism traces; selection, failures, cost/complexity and checked interview answers before publication.
- Apply the [course authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md). This is the shared authoring contract, not an authored lesson or evidence that a publication gate has passed.
- Every substantive equation needs validity, derivation and the five-part **FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION** block with a boundary/counterexample. Every systems mechanism needs preconditions, state transitions/invariants, justification, a worked trace and failure boundary. Specify decimal GB versus binary GiB, FLOP accounting, probability/log conventions, token denominators, precision and numerical tolerances.
- In each topic register below, every technical filename is a **separate proposed focused-sheet minimum**, not a promise that a module-sized survey suffices. Where one visible title contains independent methods, split those methods as explicitly directed in the brief. Source cheat sheets are families to decompose and cross-link, not substitute giant summaries for the method sheets. Reuse an existing focused sheet when scope truly matches; do not duplicate derivations across module and cheat-family sources.
- Each group supplies a learning outcome, mathematical/mechanistic plan, concrete numerical exercise, alternatives/failure discriminators and a bounded case decision/validation protocol. Authors must additionally supply each template's glossary, interview answers, final cards, sources, cost assumptions and acceptance checklist. A source row is a coverage commitment, not a completed artifact. All proposed sheet titles below are plain text, not links to nonexistent lessons.
- H100/Hopper/Ampere, PTX/TMA/WGMMA, CUTLASS/flash.cu, MPI/PMIx/SLURM/NCCL, modern model families and framework behavior require original manuals/papers and pinned source revisions at authoring time. No hardware capability, bitfield, valid launch configuration, performance advantage, price or current research frontier is certified by this plan.
- Both **50 Common Issues** source files establish future issue-count targets only. Authors must produce 50 distinct substantive entries per promised collection, including diagnostic, fix and verification; no issues are asserted completed here.

## Reading and decomposition order

CUDA: prerequisites → H100 → programming → barriers → TMA/copy → WGMMA → kernel design → pinned-code walkthroughs → node/cluster communication. Fine-tuning: foundations/transformers/data → pretraining/memory/parallelism → post-training/PEFT → evaluation/deployment → long context, advanced objectives/architectures and implementation. VLM and speech each retain their own data, training, evaluation and deployment ladders. Module numbers in the source and destination module numbers are different namespaces.

## Focused group navigation

- [CUDA Module 00 — MPI and profiling prerequisites](#cuda-prerequisites)
- [CUDA Module 01 / cheat sheet 01 — H100 architecture](#cuda-h100)
- [CUDA Module 02 / cheat sheet 02 — programming foundations](#cuda-programming)
- [CUDA Module 03 / cheat sheet 03 — asynchronicity and mbarrier](#cuda-barriers)
- [CUDA Module 04 / cheat sheet 04 — TMA and tensor maps](#cuda-tma)
- [CUDA Module 05 / cheat sheet 05 — cp.async.bulk and asynchronous movement](#cuda-cp-async)
- [CUDA Module 06 / cheat sheet 06 — WGMMA fundamentals and descriptors](#cuda-wgmma1)
- [CUDA Module 07 / cheat sheet 07 — WGMMA pipelines, FP8 and sparsity](#cuda-wgmma2)
- [CUDA Module 08 / cheat sheet 08 — kernel design](#cuda-kernel)
- [CUDA Module 09 / cheat sheet 09 — named CUTLASS and flash.cu walkthroughs](#cuda-walkthrough)
- [CUDA Module 10 / cheat sheet 10 — node topology, P2P and MPI](#cuda-multigpu1)
- [CUDA Module 11 / cheat sheet 11 — cluster orchestration and parallelisms](#cuda-multigpu2)
- [CUDA cheat sheet 13 — 50 Common Issues coverage target](#cuda-issues)
- [Fine-tuning Module 00 — LLM foundations and modern training pipeline](#ft-foundations)
- [Fine-tuning Module 01 / cheat sheet 01 — attention, blocks, scaling and MoE](#ft-transformers)
- [Fine-tuning Module 02 — data curation and tokenization](#ft-data)
- [Fine-tuning Module 03 / cheat sheet 02 — objectives and scaling laws](#ft-pretraining)
- [Fine-tuning Module 04 / cheat sheet 03 — training memory and accumulation](#ft-single-gpu)
- [Fine-tuning Module 05 / cheat sheet 04 — ZeRO and multidimensional parallelism](#ft-parallelism)
- [Fine-tuning Module 06 — GPU internals, fusion, FlashAttention and precision](#ft-gpu-internals)
- [Fine-tuning Module 07 / cheat sheet 05 — SFT, reward modeling and PPO](#ft-alignment-core)
- [Fine-tuning Module 07 — DPO, GRPO, process supervision and constitutional feedback](#ft-alignment-grpo)
- [Fine-tuning Module 08 / cheat sheet 07 — PEFT and model merging](#ft-peft)
- [Fine-tuning Module 09 / cheat sheet 11 — evaluation and debugging](#ft-evaluation)
- [Fine-tuning Module 10 / cheat sheet 09 — inference and deployment](#ft-inference)
- [Fine-tuning Module 11 / cheat sheet 10 — context extension](#ft-context)
- [Fine-tuning Module 12 — VLM, code and tool-use training](#ft-specialized)
- [Fine-tuning Module 13 — safety, alignment and interpretability](#ft-safety)
- [Fine-tuning Module 14 / cheat sheet 14 — practical MLOps workflows](#ft-mlops)
- [Fine-tuning Module 15 — continued pretraining, sampling, distillation and multi-task adaptation](#ft-advanced-offline)
- [Fine-tuning Module 15 — KTO, ORPO, SimPO, RLVR, DAPO, REINFORCE++ and self-play](#ft-advanced-rl)
- [Fine-tuning Module 15 / cheat sheet 08 — continual learning and forgetting](#ft-continual)
- [Fine-tuning Module 16 / cheat sheet 13 — alternative architectures, pruning and efficiency](#ft-alternative-architectures)
- [Fine-tuning Module 16 — tokenizer training, curriculum and multilingual learning](#ft-tokenizers-multilingual)
- [Fine-tuning Module 17 — environment, hardware and data implementation](#ft-playbook-data)
- [Fine-tuning Module 17 — adapter, SFT, memory, distributed and preference implementation](#ft-playbook-training)
- [Fine-tuning Module 17 — evaluation, monitoring, quantization and reproducibility implementation](#ft-playbook-operations)
- [Fine-tuning Module 17 — merging, context extension and end-to-end production implementation](#ft-playbook-composition)
- [Fine-tuning Module 18 — VLM architecture, data and training](#ft-vlm-training)
- [Fine-tuning Module 18 — OCR, charts, multimodal preferences and video](#ft-vlm-docs-video)
- [Fine-tuning Module 18 — evaluation, production and research boundaries](#ft-vlm-eval-deploy)
- [Fine-tuning Module 19 — DeepSpeed, FSDP2 and Megatron-LM](#ft-distributed-frameworks)
- [Fine-tuning Module 19 — resilience, profiling, cluster management and costs](#ft-distributed-resilience)
- [Fine-tuning Module 19 — ZeRO++, DualPipe and research frontiers](#ft-distributed-overlap)
- [Fine-tuning Module 20 — speech foundations, ASR/STT, Whisper and streaming](#ft-speech-asr)
- [Fine-tuning Module 20 — TTS, voice cloning and native-language training](#ft-speech-tts)
- [Fine-tuning Module 20 / cheat sheet 12 — speech-LLM integration, evaluation and deployment](#ft-speech-production)
- [Fine-tuning cheat sheet 15 — 50 Common LLM Fine-Tuning Issues target](#ft-common-issues)
- [CUDA source containers, indexes and collation utilities](#cuda-source-artifacts)
- [Fine-tuning source provenance, hidden directories and contribution artifacts](#ft-source-artifacts)

## Topic groups and exact-node registers

<a id="cuda-prerequisites"></a>

### CUDA Module 00 — MPI and profiling prerequisites

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Trace ranks, collective participation and GPU profiling boundaries before attempting a distributed run.

**Focused sheets, derivations and mechanisms.** Separate proposed sheets for MPI communicator/rank/message matching and profiling clocks, warm-up, synchronization, sampling and counter interpretation. Derive speedup S=T1/Tp and efficiency S/p only for equivalent work.

**Numerical exercise to author.** ILLUSTRATIVE inputs: fixed work takes 800 ms on one rank and 260 ms on four ranks; compute speedup and parallel efficiency, then account for 20 ms profiling overhead per measured run.

**Alternatives and failure diagnostics.** Compare MPI with single-process peer access; distinguish deadlock from a slow rank and asynchronous launch time from completed kernel time.

**Bounded case-study decision.** Choose whether a four-rank prototype is worth advancing beyond a single-GPU baseline.

**Proposed validation; no result measured.** Plan identical-work traces, per-rank barriers only at measurement boundaries, warm-up exclusion, repeated timings and rank-failure diagnosis. Hidden MPI_Basics and Profiling_Tools contents remain unresolved.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/12_MPI_and_Profiling.md` | 12 MPI and Profiling | `planned-topic` |
| `CUDA/Module_00_Prerequisites` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_00_Prerequisites/MPI_Basics` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_00_Prerequisites/Profiling_Tools` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_00_Prerequisites/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-h100"></a>

### CUDA Module 01 / cheat sheet 01 — H100 architecture

**Status:** mostly planned; [H100 architecture and resource limits](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/Cheat_Sheets/01_H100_Architecture_and_Resource_Limits.md) is an unreviewed partial draft, not a device-validated benchmark or a reproduction of the unavailable source files. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Explain how a workload traverses the GigaThread engine, GPCs, streaming multiprocessors, SMSPs, warp schedulers/dispatch, registers, shared memory/L1, L2/HBM3, TMA and tensor cores, and where NVLink4 fits.

**Focused sheets, derivations and mechanisms.** Propose separate sheets for chip hierarchy/scheduling; register-limited occupancy; HBM3/L2 and shared-memory/L1 reuse; NVLink4; TMA and tensor-core capability boundaries. Derive resident warps from register, shared-memory, block and architectural ceilings rather than treating occupancy as throughput.

**Numerical exercise to author.** ILLUSTRATIVE, not H100 specifications: assume 65,536 32-bit registers per SM, 256 threads/block and 64 registers/thread; compute the register-only limit of four resident blocks, then apply an assumed 48 KiB/block and 96 KiB shared-memory ceiling.

**Alternatives and failure diagnostics.** Compare occupancy tuning with data reuse and larger tiles; diagnose register spilling, bandwidth saturation, shared-memory pressure and dispatch stalls separately.

**Bounded case-study decision.** Choose register/tile budgets for one dense multiplication workload before purchasing more GPUs.

**Proposed validation; no result measured.** Plan compiler resource reports, correctness checks against a library baseline, achieved bandwidth/occupancy measurements and tile sweeps. Revalidate every H100 specification and hardware feature against dated NVIDIA manuals.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/01_H100_Architecture.md` | 01 H100 Architecture | `planned-topic` |
| `CUDA/Module_01_H100_Architecture` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_01_H100_Architecture/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_01_H100_Architecture/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_01_H100_Architecture/01_specifications_and_overview.md` | 01 specifications and overview | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/02_giga_thread_engine_and_gpcs.md` | 02 giga thread engine and gpcs | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/03_HBM3_memory_and_L2_cache.md` | 03 HBM3 memory and L2 cache | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/04_nvlink4_interconnect.md` | 04 nvlink4 interconnect | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/05_streaming_multiprocessors.md` | 05 streaming multiprocessors | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/06_smsp_compute_units.md` | 06 smsp compute units | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/07_warp_scheduler_and_dispatch.md` | 07 warp scheduler and dispatch | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/08_registers_and_occupancy.md` | 08 registers and occupancy | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/09_shared_memory_and_L1_cache.md` | 09 shared memory and L1 cache | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/10_tma_unit_overview.md` | 10 tma unit overview | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/11_tensor_cores_overview.md` | 11 tensor cores overview | `planned-topic` |
| `CUDA/Module_01_H100_Architecture/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-programming"></a>

### CUDA Module 02 / cheat sheet 02 — programming foundations

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Construct valid thread-block clusters and distributed shared-memory access, and reason about inline PTX state spaces, FP8/FP16/BF16 types and pointer arithmetic.

**Focused sheets, derivations and mechanisms.** Split cluster sizing/DSMEM synchronization, inline-PTX operands/state-space conversions, and representation/address calculation into separate sheets. Derive byte address=base+element_size*(row*stride+column); state alignment, lifetime and ownership invariants.

**Numerical exercise to author.** ILLUSTRATIVE inputs: a row-major BF16 matrix has 128 rows, stride 272 elements and base address 4096 bytes; calculate address for row 3 column 7, then compare a 2-byte versus 4-byte interpretation.

**Alternatives and failure diagnostics.** Compare cluster-local DSMEM with global-memory exchange and compiler intrinsics with inline PTX; diagnose misalignment, wrong address spaces, out-of-range cluster ranks and lossy conversions.

**Bounded case-study decision.** Decide whether a small clustered reduction requires DSMEM or a simpler global-memory implementation.

**Proposed validation; no result measured.** Plan boundary-size tests, race/memory checking, precision-reference comparisons and architecture/compiler-version guards; no unsupported cluster size is asserted.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/02_Programming_Fundamentals.md` | 02 Programming Fundamentals | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_02_Programming_Fundamentals/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_02_Programming_Fundamentals/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_02_Programming_Fundamentals/01_thread_block_clusters.md` | 01 thread block clusters | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/02_cluster_sizes_and_dsmem.md` | 02 cluster sizes and dsmem | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/03_inline_ptx.md` | 03 inline ptx | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/04_ptx_state_spaces.md` | 04 ptx state spaces | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/05_data_types_fp8_fp16_bf16.md` | 05 data types fp8 fp16 bf16 | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/06_pointer_arithmetic.md` | 06 pointer arithmetic | `planned-topic` |
| `CUDA/Module_02_Programming_Fundamentals/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-barriers"></a>

### CUDA Module 03 / cheat sheet 03 — asynchronicity and mbarrier

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Trace producer/consumer memory hazards, mbarrier fundamentals/API/lifecycle, arrive/wait patterns and advanced synchronization without reusing storage prematurely.

**Focused sheets, derivations and mechanisms.** Separate sheets: asynchronous dependency/hazard graphs; mbarrier initialization, expected arrivals/transaction accounting and phase lifecycle; advanced synchronization scope. State the happens-before invariant for each buffer phase and distinguish instruction issue from transaction completion.

**Numerical exercise to author.** ILLUSTRATIVE inputs: two producers each transfer 8 KiB into a shared tile and four consumers read it; write a two-phase event trace with 16 KiB expected completion and show why an 8 KiB expectation permits unsafe reuse.

**Alternatives and failure diagnostics.** Compare synchronous barriers, completion groups and mbarrier; distinguish missing arrival, incorrect byte accounting, phase mismatch and stale read.

**Bounded case-study decision.** Choose the synchronization protocol for one double-buffered copy/compute stage.

**Proposed validation; no result measured.** Plan deterministic sentinel buffers, delayed producer schedules, phase-wrap tests and version-specific PTX/manual checks; require both numerical correctness and absence of premature reads.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/03_Asynchronicity_and_Barriers.md` | 03 Asynchronicity and Barriers | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/01_philosophy_of_asynchronicity.md` | 01 philosophy of asynchronicity | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/02_memory_hazards.md` | 02 memory hazards | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/03_mbarrier_fundamentals.md` | 03 mbarrier fundamentals | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/04_mbarrier_api_and_lifecycle.md` | 04 mbarrier api and lifecycle | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/05_arrive_and_wait_patterns.md` | 05 arrive and wait patterns | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/06_advanced_synchronization.md` | 06 advanced synchronization | `planned-topic` |
| `CUDA/Module_03_Asynchronicity_and_Barriers/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-tma"></a>

### CUDA Module 04 / cheat sheet 04 — TMA and tensor maps

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Build and inspect tensor-map descriptors while explaining TMA hardware, shared-memory swizzling, L2 cache promotion, interleaving and physical layouts.

**Focused sheets, derivations and mechanisms.** Propose separate descriptor creation/stride/alignment, swizzle-bank mapping, L2-promotion policy, and interleaving-layout sheets. Derive logical-to-byte mappings and bank-conflict examples before introducing opaque descriptor fields; specify legal values from current APIs.

**Numerical exercise to author.** ILLUSTRATIVE inputs: a 128×64 FP16 tile, 128-byte row stride and assumed 32 banks of 4 bytes; compute its 16 KiB footprint and compare bank indices for a declared unswizzled mapping and a stated XOR swizzle.

**Alternatives and failure diagnostics.** Compare TMA with ordinary/coalesced loads; diagnose descriptor rank/stride errors, incompatible swizzle and interleave, alignment faults and cache pollution.

**Bounded case-study decision.** Choose a tensor map and shared-memory layout for one repeated tile load.

**Proposed validation; no result measured.** Plan CPU address-map oracles, padding/edge tests, descriptor API-version records, bank-conflict counters and equal-work cache-policy experiments; do not invent valid descriptor encodings.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/04_TMA_and_Tensor_Map.md` | 04 TMA and Tensor Map | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_04_TMA_and_Tensor_Map/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_04_TMA_and_Tensor_Map/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_04_TMA_and_Tensor_Map/01_tma_hardware_overview.md` | 01 tma hardware overview | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/02_tensor_map_descriptor.md` | 02 tensor map descriptor | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/03_swizzling_for_shared_memory.md` | 03 swizzling for shared memory | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/04_L2_cache_promotion.md` | 04 L2 cache promotion | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/05_interleaving_and_layouts.md` | 05 interleaving and layouts | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/06_creating_tensor_maps_in_code.md` | 06 creating tensor maps in code | `planned-topic` |
| `CUDA/Module_04_TMA_and_Tensor_Map/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-cp-async"></a>

### CUDA Module 05 / cheat sheet 05 — cp.async.bulk and asynchronous movement

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Distinguish Hopper versus Ampere asynchronous movement and choose structured/unstructured bulk transfer, completion, L2 hints, multicast, prefetching and asynchronous reduction.

**Focused sheets, derivations and mechanisms.** Split cp.async.bulk syntax/address contracts, structured tensor versus unstructured byte transfers, synchronization/completion mechanisms, L2 hint/multicast policy, and prefetch/reduction semantics into focused sheets. Derive transfer-compute overlap bounds max(Tcopy,Tcompute) versus Tcopy+Tcompute under explicit independence assumptions.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 64 KiB transfer at assumed effective 1 TB/s and 120 ns compute; calculate ideal serialized and overlapped stage times, then add 30 ns synchronization overhead.

**Alternatives and failure diagnostics.** Compare regular loads, Ampere-style async copy and Hopper bulk pathways; diagnose incomplete transfers, bad scopes, multicast participation errors, over-prefetching and reduction precision.

**Bounded case-study decision.** Choose whether bulk copy and multicast help one shared-tile workload.

**Proposed validation; no result measured.** Plan feature/version checks, reference data comparisons, completion-order traces and copy/compute timeline measurements with multicast enabled and disabled.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/05_cp_async_bulk.md` | 05 cp async bulk | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_05_cp_async_bulk/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_05_cp_async_bulk/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_05_cp_async_bulk/01_innovations_in_async_data_movement.md` | 01 innovations in async data movement | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/02_cp_async_bulk_instruction.md` | 02 cp async bulk instruction | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/03_hopper_vs_ampere_async.md` | 03 hopper vs ampere async | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/04_unstructured_vs_structured_bulk.md` | 04 unstructured vs structured bulk | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/05_sync_and_completion_mechanisms.md` | 05 sync and completion mechanisms | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/06_L2_cache_hinting_and_multicasting.md` | 06 L2 cache hinting and multicasting | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/07_prefetching_and_async_reductions.md` | 07 prefetching and async reductions | `planned-topic` |
| `CUDA/Module_05_cp_async_bulk/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-wgmma1"></a>

### CUDA Module 06 / cheat sheet 06 — WGMMA fundamentals and descriptors

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Explain the warp-to-warpgroup shift, wgmma.mma_async, matrix-A fetching, register management and WGMMA descriptor construction.

**Focused sheets, derivations and mechanisms.** Split warpgroup participation/instruction contracts, matrix-A register/shared-memory choices, and register/descriptor layouts into separate sheets. Derive GEMM work 2MNK and shared-memory address progression; document descriptor bitfields only from a pinned authoritative specification.

**Numerical exercise to author.** ILLUSTRATIVE inputs: M=64, N=128, K=32, FP16 operands and FP32 accumulators; calculate 524,288 floating-point operations and 32 KiB logical output, then budget accumulator storage across an explicitly assumed participant count.

**Alternatives and failure diagnostics.** Compare register/shared and shared/shared operands with a library GEMM; diagnose divergent participation, descriptor misencoding, register pressure and missing async ordering.

**Bounded case-study decision.** Choose the operand layout for one supported WGMMA tile.

**Proposed validation; no result measured.** Plan shape legality checks, independent GEMM reference errors, layout/address traces and compiler/register reports before any throughput claim.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/06_WGMMA_Part1.md` | 06 WGMMA Part1 | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_06_WGMMA_Part1/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_06_WGMMA_Part1/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_06_WGMMA_Part1/01_paradigm_shift_warp_to_warpgroup.md` | 01 paradigm shift warp to warpgroup | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1/02_wgmma_mma_async_instruction.md` | 02 wgmma mma async instruction | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1/03_fetching_matrix_A.md` | 03 fetching matrix A | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1/04_register_management.md` | 04 register management | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1/05_wgmma_descriptor.md` | 05 wgmma descriptor | `planned-topic` |
| `CUDA/Module_06_WGMMA_Part1/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-wgmma2"></a>

### CUDA Module 07 / cheat sheet 07 — WGMMA pipelines, FP8 and sparsity

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Trace WGMMA groups, output storage, FP8 support/quirks/packing/K-major layouts and sparse WGMMA without confusing nominal with useful work.

**Focused sheets, derivations and mechanisms.** Propose distinct pipeline group/commit/wait, accumulator-to-output layout, FP8 scaling/packing/K-major, and sparse metadata/eligibility sheets. Derive quantization scale and saturation boundaries and compare dense work with eligible nonzero work under declared sparsity.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 4 pipeline groups, two shared buffers, 64×128 FP32 output and assumed FP8 range ±240 with source maximum magnitude 30; calculate output bytes and scale 8, then trace buffer reuse and one saturated outlier of magnitude 40.

**Alternatives and failure diagnostics.** Compare FP16/BF16 with scaled FP8 and dense with supported structured sparse operations; diagnose phase reuse, output permutation, overflow, K-layout mismatch and invalid sparse metadata.

**Bounded case-study decision.** Choose whether FP8 or sparse WGMMA is acceptable for one quality-constrained matrix workload.

**Proposed validation; no result measured.** Plan layerwise error/saturation measurements, dense-equivalent correctness, invalid-metadata tests and group-depth sweeps; revalidate FP8 formats and sparse instruction support.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/07_WGMMA_Part2.md` | 07 WGMMA Part2 | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_07_WGMMA_Part2/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_07_WGMMA_Part2/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_07_WGMMA_Part2/01_pipelining_with_wgmma_groups.md` | 01 pipelining with wgmma groups | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2/02_storing_wgmma_output.md` | 02 storing wgmma output | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2/03_fp8_paradigm_and_support.md` | 03 fp8 paradigm and support | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2/04_fp8_quirks_packing_k_major.md` | 04 fp8 quirks packing k major | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2/05_sparse_wgmma.md` | 05 sparse wgmma | `planned-topic` |
| `CUDA/Module_07_WGMMA_Part2/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-kernel"></a>

### CUDA Module 08 / cheat sheet 08 — kernel design

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Use roofline/arithmetic intensity, warp specialization, circular pipelines, persistent scheduling, epilogues and multikernel optimization to select a bounded kernel design.

**Focused sheets, derivations and mechanisms.** Separate sheets for roofline accounting; producer/consumer warp specialization; circular-buffer pipeline state; persistent scheduling/load balance; epilogue fusion; and multikernel launch/memory tradeoffs. Derive attainable rate min(Ppeak,B*I) with explicit bytes counted and no assumed peak achievement.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 2 TFLOP work, 0.5 TB transferred, assumed 60 TFLOP/s compute and 2 TB/s bandwidth; derive I=4 FLOP/byte and 8 TFLOP/s bandwidth ceiling, then estimate the effect of removing 0.1 TB intermediate traffic.

**Alternatives and failure diagnostics.** Compare persistent with conventional scheduling and fused with split epilogues; diagnose underfilled waves, buffer stalls, load imbalance and increased register pressure.

**Bounded case-study decision.** Choose fusion and pipeline depth for one fixed GEMM-plus-activation shape distribution.

**Proposed validation; no result measured.** Plan equal-numerics baselines, size-distribution benchmarks, end-to-end rather than kernel-only timings, resource counters and tail-latency reports.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/08_Kernel_Design.md` | 08 Kernel Design | `planned-topic` |
| `CUDA/Module_08_Kernel_Design` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_08_Kernel_Design/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_08_Kernel_Design/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_08_Kernel_Design/01_roofline_model_and_arithmetic_intensity.md` | 01 roofline model and arithmetic intensity | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/02_warp_specialization.md` | 02 warp specialization | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/03_pipelining_and_circular_buffering.md` | 03 pipelining and circular buffering | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/04_persistent_kernel_scheduling.md` | 04 persistent kernel scheduling | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/05_epilogue_design.md` | 05 epilogue design | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/06_multikernel_optimizations.md` | 06 multikernel optimizations | `planned-topic` |
| `CUDA/Module_08_Kernel_Design/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-walkthrough"></a>

### CUDA Module 09 / cheat sheet 09 — named CUTLASS and flash.cu walkthroughs

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Reconstruct pipeline state and tile ownership through named sm90_pipeline_hpp, cooperative pipelining, MMA RS/SS layouts, tile scheduler, Stream-K/group GEMM, sm90_epilogue, sm90_builder_inl and flash_cu walkthroughs.

**Focused sheets, derivations and mechanisms.** Plan one code-reading sheet per visible walkthrough: CUTLASS sm90_pipeline.hpp; cooperative pipelining; MMA register/shared (RS) and shared/shared (SS) layouts; tile scheduler; Stream-K and grouped GEMM (split independently); SM90 epilogue; sm90_builder.inl; flash.cu attention. These names are study targets, not claims of a checked source revision. Trace producer/consumer descriptors and pipeline phases, and derive online-softmax rescaling for the attention walkthrough.

**Numerical exercise to author.** ILLUSTRATIVE inputs: two GEMMs with tile counts 5 and 11 scheduled onto four workers; draw static versus pooled/Stream-K assignments. For attention logits [1,2], [3,4], compute block maxima and the online normalization update with dimensionless scores.

**Alternatives and failure diagnostics.** Compare library execution with a custom kernel and static with Stream-K/group scheduling; diagnose descriptor/layout disagreement, epilogue indexing, phase bugs, partial reduction errors and softmax overflow.

**Bounded case-study decision.** Choose whether a pinned CUTLASS component can replace one bespoke pipeline.

**Proposed validation; no result measured.** Plan commit/version and exact source-path records, call-graph/state traces, small-shape reference tests, uneven tile cases and attention error tests. The tree alone supplies no code body or implementation facts.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/09_Code_Walkthrough.md` | 09 Code Walkthrough | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_09_Code_Walkthrough/01_sm90_pipeline_hpp.md` | 01 sm90 pipeline hpp | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/02_cooperative_pipelining.md` | 02 cooperative pipelining | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/03_mma_rs_ss_layouts.md` | 03 mma rs ss layouts | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/04_tile_scheduler.md` | 04 tile scheduler | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/05_stream_k_and_group_gemm.md` | 05 stream k and group gemm | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/06_sm90_epilogue.md` | 06 sm90 epilogue | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/07_sm90_builder_inl.md` | 07 sm90 builder inl | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/08_flash_cu_walkthrough.md` | 08 flash cu walkthrough | `planned-topic` |
| `CUDA/Module_09_Code_Walkthrough/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-multigpu1"></a>

### CUDA Module 10 / cheat sheet 10 — node topology, P2P and MPI

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Locate single-GPU bottlenecks and compare intranode NVLink mesh, NVSwitch fabric, DGX H100 topology, CUDA P2P programming and MPI.

**Focused sheets, derivations and mechanisms.** Split node/fabric topology mapping, peer-memory visibility and transfer ordering, and MPI communication contracts into separate sheets. Derive latency-bandwidth time alpha+n/beta and distinguish one-way, aggregate and bisection bandwidth.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 64 MiB peer transfer, 5 microseconds startup and assumed 100 GB/s effective link; calculate transfer time using 1 MiB=2^20 bytes, then compare with 20 ms compute.

**Alternatives and failure diagnostics.** Compare P2P, host-staged copies and MPI transport; diagnose topology asymmetry, inaccessible peers, oversubscription, rank-device mapping and host staging.

**Bounded case-study decision.** Choose GPU placement and transfer path for one two-GPU training shard.

**Proposed validation; no result measured.** Plan topology capture, peer-access checks, per-direction bandwidth/latency matrices, correctness checks and compute/communication overlap traces; DGX/NVLink versions require authoritative revalidation.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/10_MultiGPU_Part1.md` | 10 MultiGPU Part1 | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_10_MultiGPU_Part1/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_10_MultiGPU_Part1/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_10_MultiGPU_Part1/01_single_gpu_compute_bottleneck.md` | 01 single gpu compute bottleneck | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/02_intranode_nvlink_mesh.md` | 02 intranode nvlink mesh | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/03_nvswitch_and_scaling_fabric.md` | 03 nvswitch and scaling fabric | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/04_dgx_h100_node_topology.md` | 04 dgx h100 node topology | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/05_cuda_p2p_programming.md` | 05 cuda p2p programming | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/06_mpi_basics.md` | 06 mpi basics | `planned-topic` |
| `CUDA/Module_10_MultiGPU_Part1/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-multigpu2"></a>

### CUDA Module 11 / cheat sheet 11 — cluster orchestration and parallelisms

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Trace SLURM allocation, PMIx process management and NCCL initialization/communication primitives, then distinguish data, tensor, pipeline and expert parallelism.

**Focused sheets, derivations and mechanisms.** Split SLURM/rank-device placement; PMIx launch lifecycle; NCCL initialization and collectives; data-parallel gradient synchronization; tensor sharding; pipeline scheduling; and expert routing into independent sheets. Derive ring all-reduce traffic 2(p−1)n/p under the stated ring model and pipeline bubble fractions under a declared schedule.

**Numerical exercise to author.** ILLUSTRATIVE inputs: eight ranks, 1 GiB gradient payload, assumed 25 GB/s effective link and 10 microseconds per ring step; calculate bandwidth and startup terms. Draw a four-stage pipeline with eight microbatches and separately account for expert all-to-all bytes.

**Alternatives and failure diagnostics.** Compare each parallelism alone and hybrid layouts; diagnose communicator mismatch, PMIx launch failure, SLURM resource mismatch, collective order deadlock, imbalance and expert hotspots.

**Bounded case-study decision.** Choose a rank topology for one model whose parameters exceed a single device.

**Proposed validation; no result measured.** Plan launch/rank inventory, collective correctness, single-node to multinode scaling, synthetic imbalance cases and a bounded rank-loss exercise; no completed scale-out is implied.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/11_MultiGPU_Part2.md` | 11 MultiGPU Part2 | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/Module_11_MultiGPU_Part2/code` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_11_MultiGPU_Part2/exercises` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `CUDA/Module_11_MultiGPU_Part2/01_cluster_orchestration_slurm.md` | 01 cluster orchestration slurm | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/02_pmix_process_management.md` | 02 pmix process management | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/03_nccl_initialization.md` | 03 nccl initialization | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/04_data_parallelism.md` | 04 data parallelism | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/05_tensor_parallelism.md` | 05 tensor parallelism | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/06_pipeline_parallelism.md` | 06 pipeline parallelism | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/07_expert_parallelism.md` | 07 expert parallelism | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/08_nccl_communication_primitives.md` | 08 nccl communication primitives | `planned-topic` |
| `CUDA/Module_11_MultiGPU_Part2/README.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="cuda-issues"></a>

### CUDA cheat sheet 13 — 50 Common Issues coverage target

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Build a future 50-entry diagnostic collection that distinguishes errors across the visible CUDA topics rather than listing generic GPU advice.

**Focused sheets, derivations and mechanisms.** Organize distinct issue candidates around memory/addressing, barrier phases, TMA descriptors/swizzling, cp.async completion, WGMMA layouts/FP8/sparse metadata, scheduling, MPI/NCCL and profiling; each published entry needs symptom, competing causes, discriminating evidence, fix, verification and prevention. The source count is a target, not 50 supplied or completed issues.

**Numerical exercise to author.** ILLUSTRATIVE inputs: expected async completion 32 KiB, actual producer payloads 16 KiB and 8 KiB; calculate the 8 KiB deficit and contrast a stalled barrier with a stale-buffer race.

**Alternatives and failure diagnostics.** Compare minimal deterministic reproductions with full-workload traces; do not fix a hang by removing synchronization or hiding a failed rank.

**Bounded case-study decision.** Choose the first diagnostic experiment for one stalled tiled kernel, not a wholesale rewrite.

**Proposed validation; no result measured.** Plan one reproducible fault per issue, negative controls for competing causes and before/after correctness measurements; count only substantive authored issues.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA/CUDA_Cheat_Sheets/13_50_Common_Issues.md` | 13 50 Common Issues | `planned-topic` |

<a id="ft-foundations"></a>

### Fine-tuning Module 00 — LLM foundations and modern training pipeline

**Status:** partially drafted. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md). Course lessons 0.1 and 0.2 are now authored as drafts; focused sheets, a bounded case, independent review and empirical validation remain open.

**Learning outcome.** Distinguish a language model, training objective, pretraining, supervised adaptation, preference optimization and serving, and identify the artifact handed off at each stage.

**Focused sheets, derivations and mechanisms.** Propose separate LLM probability factorization and training-stage contract sheets. Derive sequence log-likelihood from conditional token probabilities; trace dataset, tokenizer, checkpoint, adapter and evaluation lineage.

**Numerical exercise to author.** ILLUSTRATIVE inputs: token probabilities 0.5, 0.25 and 0.1 for a three-token sequence; compute joint probability and mean negative log-likelihood in nats/token.

**Alternatives and failure diagnostics.** Compare prompting, retrieval, continued pretraining and SFT; diagnose confusing a pretrained next-token objective with verified task reliability.

**Bounded case-study decision.** Decide whether a small domain task needs weight changes at all.

**Proposed validation; no result measured.** Plan unchanged-model and prompting baselines, held-out task criteria and explicit licensing/data-access checks before proposing training.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_00_Foundations` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_00_Foundations/0.1_What_is_a_Large_Language_Model.md` | 0.1 What is a Large Language Model | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/01_What_Is_a_Large_Language_Model.md); source body unavailable |
| `finetuning_course/Module_00_Foundations/0.2_Modern_LLM_Training_Pipeline.md` | 0.2 Modern LLM Training Pipeline | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/02_Modern_LLM_Training_Pipeline.md); source body unavailable |

<a id="ft-transformers"></a>

### Fine-tuning Module 01 / cheat sheet 01 — attention, blocks, scaling and MoE

**Status:** module lessons partially drafted. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md). Draft lessons cover submodules 1.1–1.4; focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Compute tensor shapes and parameter/compute budgets for attention, transformer blocks and mixture-of-experts routing.

**Focused sheets, derivations and mechanisms.** Split attention softmax/scaling; block residual/norm/MLP architecture; parameter counting; and mixture-of-experts capacity/routing into independent sheets. Derive QK^T/sqrt(d_k), per-layer parameter sums and active-versus-total MoE parameters.

**Numerical exercise to author.** ILLUSTRATIVE inputs: batch 2, sequence 128 tokens, hidden width 512, 8 heads and 4× MLP expansion; derive Q/K/V shapes, attention-score elements and dense-layer parameter counts excluding biases. For MoE, assume 8 experts and top-2 routing.

**Alternatives and failure diagnostics.** Compare multihead variants and dense versus MoE blocks; diagnose shape errors, softmax instability, routing overload and misleading active-parameter comparisons.

**Bounded case-study decision.** Choose dense versus routed feed-forward blocks for a bounded model-budget exercise.

**Proposed validation; no result measured.** Plan small tensor oracle comparisons, parameter-count reconciliation, router-load histograms and equal-token cost/quality comparisons.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/01_Transformer_Architecture_Attention.md` | 01 Transformer Architecture Attention | `planned-topic` |
| `finetuning_course/Module_01_Transformer_Architecture` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_01_Transformer_Architecture/1.1_Attention_Mechanism.md` | 1.1 Attention Mechanism | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/03_Attention_Mechanism.md); source body unavailable |
| `finetuning_course/Module_01_Transformer_Architecture/1.2_Transformer_Block_Architecture.md` | 1.2 Transformer Block Architecture | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/04_Transformer_Block_Architecture.md); source body unavailable |
| `finetuning_course/Module_01_Transformer_Architecture/1.3_Scaling_and_Parameter_Counting.md` | 1.3 Scaling and Parameter Counting | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/05_Scaling_and_Parameter_Counting.md); source body unavailable |
| `finetuning_course/Module_01_Transformer_Architecture/1.4_Mixture_of_Experts.md` | 1.4 Mixture of Experts | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/06_Mixture_of_Experts.md); source body unavailable |

<a id="ft-data"></a>

### Fine-tuning Module 02 — data curation and tokenization

**Status:** all four mapped module lessons are draft. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md). Focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Specify data sources/scale, curation, tokenization and training formatting with license, deduplication and leakage boundaries.

**Focused sheets, derivations and mechanisms.** Separate data acquisition/filter/dedup sheets; tokenizer segmentation/vocabulary contracts; and training packing/chat-template/loss-mask sheets. Derive token-weighted sampling and contamination estimates; trace how formatting changes supervised labels.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 100,000 documents, 20% removed by licensing, then 10% of the remainder by deduplication; 400 tokens/document and 2,048-token packed sequences. Calculate retained tokens and the ideal full-pack count before boundary losses.

**Alternatives and failure diagnostics.** Compare domain-only with mixed replay data and packing with padding; diagnose hidden duplicates, train/test overlap, template mismatch and loss on prompt/padding tokens.

**Bounded case-study decision.** Choose a legally usable, bounded domain SFT dataset and formatting policy.

**Proposed validation; no result measured.** Plan source/license records, document/time/entity splits, dedup audits, decoded token/label inspection and a manually checked training batch.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_02_Data_Curation_Tokenization` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_02_Data_Curation_Tokenization/2.1_Data_Sources_and_Scale.md` | 2.1 Data Sources and Scale | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/2.1_Data_Sources_and_Scale.md); source body unavailable |
| `finetuning_course/Module_02_Data_Curation_Tokenization/2.2_Data_Curation_Pipeline.md` | 2.2 Data Curation Pipeline | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/2.2_Data_Curation_Pipeline.md); source body unavailable |
| `finetuning_course/Module_02_Data_Curation_Tokenization/2.3_Tokenization.md` | 2.3 Tokenization | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/2.3_Tokenization.md); source body unavailable |
| `finetuning_course/Module_02_Data_Curation_Tokenization/2.4_Data_Formatting_for_Training.md` | 2.4 Data Formatting for Training | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/2.4_Data_Formatting_for_Training.md); source body unavailable |

<a id="ft-pretraining"></a>

### Fine-tuning Module 03 / cheat sheet 02 — objectives and scaling laws

**Status:** all three mapped module lessons are draft. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md). Focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Compare training objectives, fit scoped scaling-law estimates and choose learning-rate/batch/token schedules without extrapolating beyond evidence.

**Focused sheets, derivations and mechanisms.** Separate autoregressive/masked/other stated objectives, compute-optimal scaling fits and hyperparameter schedules. Derive cross-entropy and a declared compute proxy C≈6ND for dense training; distinguish fitted scaling coefficients from universal constants.

**Numerical exercise to author.** ILLUSTRATIVE inputs: N=1 billion parameters, D=20 billion tokens and assumed sustained 100 TFLOP/s; calculate the 1.2×10^20 FLOP proxy and idealized runtime, excluding evaluation/checkpoint/communication overhead.

**Alternatives and failure diagnostics.** Compare from-scratch and continued training; diagnose mismatched tokenization in scaling comparisons, undertrained models, unstable warm-up and extrapolated fits.

**Bounded case-study decision.** Choose one of two model/token budgets under a fixed compute ceiling.

**Proposed validation; no result measured.** Plan small proxy sweeps, held-out loss, general-domain regression, all-in cost accounting and uncertainty around extrapolated quality.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/02_Scaling_Laws_Pretraining_Objectives.md` | 02 Scaling Laws Pretraining Objectives | `planned-topic` |
| `finetuning_course/Module_03_Pretraining_Scaling_Laws` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_03_Pretraining_Scaling_Laws/3.1_Training_Objectives.md` | 3.1 Training Objectives | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/3.1_Training_Objectives.md); source body unavailable |
| `finetuning_course/Module_03_Pretraining_Scaling_Laws/3.2_Scaling_Laws.md` | 3.2 Scaling Laws | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/3.2_Scaling_Laws.md); source body unavailable |
| `finetuning_course/Module_03_Pretraining_Scaling_Laws/3.3_Pretraining_Hyperparameters.md` | 3.3 Pretraining Hyperparameters | `draft` — mapped course lesson at [module 19](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/3.3_Pretraining_Hyperparameters.md); source body unavailable |

<a id="ft-single-gpu"></a>

### Fine-tuning Module 04 / cheat sheet 03 — training memory and accumulation

**Status:** partially drafted. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md). Submodules 4.1 and 4.4 are new mapped drafts; 4.2–4.3 filenames remain clipped/unresolved; focused sheets, case, review and empirical validation remain open.

**Learning outcome.** Account for training memory, sequence-length-dependent activations, visible recomputation/checkpointing topics and gradient accumulation.

**Focused sheets, derivations and mechanisms.** Separate state-memory accounting, activation sequence scaling, checkpoint/recompute schedules and accumulation/effective-batch semantics. Derive memory from parameter/gradient/optimizer precision and batch_effective=microbatch×accumulation×data_parallel_size. Clipped filenames are not reconstructed.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 1 billion parameters with 2-byte weights, 2-byte gradients and two 4-byte optimizer moments; calculate 12 GB state before optional master weights/activations. With microbatch 2, accumulation 8 and one rank, derive 16 sequences/update.

**Alternatives and failure diagnostics.** Compare checkpointing, smaller microbatches and offload; diagnose accumulation scaling errors, unexpectedly quadratic attention memory and allocator versus live-tensor memory.

**Bounded case-study decision.** Choose a memory policy for a 24 GB illustrative device without silently changing the effective batch.

**Proposed validation; no result measured.** Plan measured peak-memory timelines, gradient-equivalence checks, sequence-length sweeps and a checkpointing throughput tradeoff report. Two clipped source filenames stay unresolved-source.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/03_Memory_Compute_SingleGPU_Optimization.md` | 03 Memory Compute SingleGPU Optimization | `planned-topic` |
| `finetuning_course/Module_04_Single_GPU_Training` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_04_Single_GPU_Training/4.1_Memory_Components_During_Training.md` | 4.1 Memory Components During Training | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/4.1_Memory_Components_During_Training.md); source body unavailable |
| `finetuning_course/Module_04_Single_GPU_Training/4.2_Activation_Memory_and_the_Sequence_Lengt… [TRUNCATED/UNVERIFIED]` | OPEN: clipped filename; teach only visible prefix: UNVERIFIED] | `unresolved-source` |
| `finetuning_course/Module_04_Single_GPU_Training/4.3_Activation_Recomputation_Gradient_Checkpo… [TRUNCATED/UNVERIFIED]` | OPEN: clipped filename; teach only visible prefix: UNVERIFIED] | `unresolved-source` |
| `finetuning_course/Module_04_Single_GPU_Training/4.4_Gradient_Accumulation.md` | 4.4 Gradient Accumulation | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/4.4_Gradient_Accumulation.md); source body unavailable |

<a id="ft-parallelism"></a>

### Fine-tuning Module 05 / cheat sheet 04 — ZeRO and multidimensional parallelism

**Status:** six visible module lessons are draft. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md). Submodules 5.1–5.6 are mapped drafts; clipped 5.7, focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Distinguish data, ZeRO, tensor, context, pipeline, expert and visible 5D configuration-selection topics by what is partitioned and communicated.

**Focused sheets, derivations and mechanisms.** Give data parallelism, ZeRO stages, tensor parallelism, context parallelism, pipeline schedules, expert routing and multidimensional configuration separate sheets. Derive sharded optimizer/gradient/parameter memory per stage and logical parallel-group products with explicit nesting/replication assumptions.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 32 GPUs, tensor degree 2, pipeline degree 4 and data degree 4; compute ranks/group and fit 16 GB unsharded optimizer state under a four-way optimizer shard. Separately declare whether context/expert groups reuse or multiply those axes.

**Alternatives and failure diagnostics.** Compare ZeRO sharding with replicated optimizer state and each parallelism with the simplest feasible plan; diagnose communication domination, incorrect process groups, bubble waste and expert imbalance.

**Bounded case-study decision.** Choose one feasible configuration for a model exceeding one-device memory.

**Proposed validation; no result measured.** Plan process-group/rank diagrams, equivalent global batch, numerical agreement, memory accounting and equal-token scaling trials; the clipped 5D filename remains unresolved-source.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/04_Distributed_Training_Parallelism.md` | 04 Distributed Training Parallelism | `planned-topic` |
| `finetuning_course/Module_05_Distributed_Training` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_05_Distributed_Training/5.1_Data_Parallelism.md` | 5.1 Data Parallelism | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.1_Data_Parallelism.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.2_ZeRO_Zero_Redundancy_Optimizer.md` | 5.2 ZeRO Zero Redundancy Optimizer | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.2_ZeRO_Zero_Redundancy_Optimizer.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.3_Tensor_Parallelism.md` | 5.3 Tensor Parallelism | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.3_Tensor_Parallelism.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.4_Context_Parallelism.md` | 5.4 Context Parallelism | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.4_Context_Parallelism.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.5_Pipeline_Parallelism.md` | 5.5 Pipeline Parallelism | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.5_Pipeline_Parallelism.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.6_Expert_Parallelism.md` | 5.6 Expert Parallelism | `draft` — mapped course lesson at [module 24](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/5.6_Expert_Parallelism.md); source body unavailable |
| `finetuning_course/Module_05_Distributed_Training/5.7_5D_Parallelism_and_Configuration_Selection… [TRUNCATED/UNVERIFIED]` | OPEN: clipped filename; teach only visible prefix: UNVERIFIED] | `unresolved-source` |

<a id="ft-gpu-internals"></a>

### Fine-tuning Module 06 — GPU internals, fusion, FlashAttention and precision

**Status:** partially drafted. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md). Submodules 6.1–6.4 are new mapped drafts; focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Relate GPU architecture to kernel fusion, FlashAttention and mixed-precision training, linking to the detailed CUDA groups rather than replacing them.

**Focused sheets, derivations and mechanisms.** Separate GPU primer, fusion/launch-memory accounting, FlashAttention exact tiled attention and mixed-precision/loss-scaling sheets. Derive online softmax max/sum updates and memory saved by avoiding the full attention matrix; distinguish exact algorithm from finite-precision arithmetic.

**Numerical exercise to author.** ILLUSTRATIVE inputs: batch 1, 16 heads, 4,096 tokens and 2-byte attention scores; compute the 512 MiB materialized-score footprint. Trace an online-softmax update for score blocks [1,2] and [3,4].

**Alternatives and failure diagnostics.** Compare fused with separate kernels and BF16/FP16 with FP32 accumulation; diagnose loss overflow, precision-dependent drift and counting kernel speedup as end-to-end speedup.

**Bounded case-study decision.** Choose a supported attention kernel and precision policy for one sequence-length regime.

**Proposed validation; no result measured.** Plan reference attention comparisons, gradient checks, peak-memory profiles, overflow counters and equal-work end-to-end timings.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_06_GPU_Internals` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_06_GPU_Internals/6.1_GPU_Architecture_Primer.md` | 6.1 GPU Architecture Primer | `draft` — mapped course lesson at [module 23](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/6.1_GPU_Architecture_Primer.md); source body unavailable |
| `finetuning_course/Module_06_GPU_Internals/6.2_Kernel_Optimization_and_Fusion.md` | 6.2 Kernel Optimization and Fusion | `draft` — mapped course lesson at [module 23](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/6.2_Kernel_Optimization_and_Fusion.md); source body unavailable |
| `finetuning_course/Module_06_GPU_Internals/6.3_FlashAttention.md` | 6.3 FlashAttention | `draft` — mapped course lesson at [module 23](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/6.3_FlashAttention.md); source body unavailable |
| `finetuning_course/Module_06_GPU_Internals/6.4_Mixed_Precision_Training.md` | 6.4 Mixed Precision Training | `draft` — mapped course lesson at [module 23](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/6.4_Mixed_Precision_Training.md); source body unavailable |

<a id="ft-alignment-core"></a>

### Fine-tuning Module 07 / cheat sheet 05 — SFT, reward modeling and PPO

**Status:** partially drafted. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md). Submodules 7.1–7.4 are new mapped drafts; focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Explain why post-training matters and distinguish SFT, reward-model training and RLHF with PPO by data, objectives and failure boundaries.

**Focused sheets, derivations and mechanisms.** Separate SFT masking/objective, pairwise reward-model likelihood and PPO clipped surrogate/KL control sheets. Derive sequence versus token loss normalization, pairwise logistic loss and the clipped importance-ratio objective; define the reference policy.

**Numerical exercise to author.** ILLUSTRATIVE inputs: reward-model score difference 1.2, old action probability 0.2, new probability 0.25, advantage 2 and PPO clipping epsilon 0.2; calculate pairwise preference probability and unclipped/clipped surrogate terms.

**Alternatives and failure diagnostics.** Compare SFT-only with reward-model/PPO and offline preference training; diagnose reward hacking, KL drift, label noise and mixing token/sequence reward units.

**Bounded case-study decision.** Choose whether one narrowly specified assistant behavior needs PPO beyond a supervised baseline.

**Proposed validation; no result measured.** Plan held-out prompts, blinded preference labels, reward/real-quality agreement, safety regression and equal-budget comparisons; no reward gain is treated as user benefit without validation.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/05_Post_Training_SFT_RLHF_PPO.md` | 05 Post Training SFT RLHF PPO | `planned-topic` |
| `finetuning_course/Module_07_Post_Training_Alignment/7.1_Why_Post_Training_Matters.md` | 7.1 Why Post Training Matters | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.1_Why_Post_Training_Matters.md); source body unavailable |
| `finetuning_course/Module_07_Post_Training_Alignment/7.2_Supervised_Fine_Tuning.md` | 7.2 Supervised Fine Tuning | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.2_Supervised_Fine_Tuning.md); source body unavailable |
| `finetuning_course/Module_07_Post_Training_Alignment/7.3_Reward_Modeling.md` | 7.3 Reward Modeling | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.3_Reward_Modeling.md); source body unavailable |
| `finetuning_course/Module_07_Post_Training_Alignment/7.4_RLHF_with_PPO.md` | 7.4 RLHF with PPO | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.4_RLHF_with_PPO.md); source body unavailable |

<a id="ft-alignment-grpo"></a>

### Fine-tuning Module 07 — DPO, GRPO, process supervision and constitutional feedback

**Status:** partially drafted. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md). Submodules 7.5, 7.6 and 7.8 are new mapped drafts; the clipped 7.7 filename remains unresolved; focused sheets, bounded case, independent review and empirical validation remain open.

**Learning outcome.** Differentiate DPO, Group Relative Policy Optimization, visible reasoning/process-supervision topics and Constitutional AI/RLAIF from the reward-model/PPO baseline.

**Focused sheets, derivations and mechanisms.** Create separate sheets for DPO log-ratio preference objective; GRPO within-group baselines; outcome versus process supervision; and constitutional/AI-generated feedback reliability. Derive DPO log odds and grouped reward standardization with a stated zero-variance convention.

**Numerical exercise to author.** ILLUSTRATIVE inputs: four sampled rewards [0,1,1,2], explicit population-standard-deviation convention and DPO beta 0.1 with preferred/rejected policy/reference log-ratios 2 and -1; compute normalized advantages and logistic preference loss.

**Alternatives and failure diagnostics.** Compare human labels with AI feedback and outcome with step-level supervision; diagnose zero-variance groups, length bias, verifier leakage and self-reinforcing judge mistakes.

**Bounded case-study decision.** Choose DPO or a grouped-policy experiment for one verifiable reasoning task.

**Proposed validation; no result measured.** Plan independent verifier tests, held-out problem families, label-source audits, token-normalized cost and expert review of process labels. The truncated reasoning filename remains unresolved-source.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/06_Preference_Optimization_DPO_Variants.md` | 06 Preference Optimization DPO Variants | `planned-topic` |
| `finetuning_course/Module_07_Post_Training_Alignment` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_07_Post_Training_Alignment/7.5_Direct_Preference_Optimization.md` | 7.5 Direct Preference Optimization | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.5_Direct_Preference_Optimization.md); source body unavailable |
| `finetuning_course/Module_07_Post_Training_Alignment/7.6_GRPO_Group_Relative_Policy_Optimization.md` | 7.6 GRPO Group Relative Policy Optimization | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.6_GRPO_Group_Relative_Policy_Optimization.md); source body unavailable |
| `finetuning_course/Module_07_Post_Training_Alignment/7.7_Reasoning_Training_and_Process_Supervision… [TRUNCATED/UNVERIFIED]` | OPEN: clipped filename; teach only visible prefix: UNVERIFIED] | `unresolved-source` |
| `finetuning_course/Module_07_Post_Training_Alignment/7.8_Constitutional_AI_and_RLAIF.md` | 7.8 Constitutional AI and RLAIF | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/7.8_Constitutional_AI_and_RLAIF.md); source body unavailable |

<a id="ft-peft"></a>

### Fine-tuning Module 08 / cheat sheet 07 — PEFT and model merging

**Status:** partial, unreviewed drafts; mapped source submodules 8.1, 8.2, 8.4 and 8.5 are covered (8.2 by the existing LoRA sheet); 8.3 remains unresolved. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md). Draft sheets: [LoRA](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md) and [quantized-base adapters](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/Cheat_Sheets/02_QLoRA_Quantized_Base_Adapter_Finetuning.md); neither verifies unavailable source contents.

**Learning outcome.** Select full versus parameter-efficient adaptation, LoRA, visible QLoRA/quantization-aware fine-tuning, advanced PEFT and model merging/weight averaging.

**Focused sheets, derivations and mechanisms.** Separate LoRA low-rank update/scaling, quantization-plus-adapter memory/error, advanced PEFT comparison and merging/weight-averaging compatibility sheets. Derive LoRA trainable parameters r(d_in+d_out), quantization de/reconstruction and weighted model averages only for compatible coordinates.

**Numerical exercise to author.** ILLUSTRATIVE inputs: a 4,096×4,096 projection, LoRA rank 16 and alpha 32; calculate 131,072 adapter parameters versus 16,777,216 dense parameters and scale alpha/r=2. Compare 4-bit raw base storage with 16-bit storage, explicitly excluding metadata.

**Alternatives and failure diagnostics.** Compare LoRA, full fine-tuning and quantization-aware choices; diagnose rank underfit, incompatible base/tokenizer merges, quantization outliers and task interference.

**Bounded case-study decision.** Choose an adapter/merge strategy for one domain task under a fixed memory budget.

**Proposed validation; no result measured.** Plan equal-data rank sweeps, base/task regression, quantization-error checks and merge-versus-unmerged output comparisons; clipped QLoRA filename stays unresolved-source.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/07_Parameter_Efficient_Finetuning_PEFT.md` | 07 Parameter Efficient Finetuning PEFT | `planned-topic` |
| `finetuning_course/Module_08_PEFT_Model_Merging` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_08_PEFT_Model_Merging/8.1_Why_PEFT_and_the_Fine_Tuning_Spectrum.md` | 8.1 Why PEFT and the Fine Tuning Spectrum | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/8.1_Why_PEFT_and_the_Fine_Tuning_Spectrum.md); source body unavailable |
| `finetuning_course/Module_08_PEFT_Model_Merging/8.2_LoRA_Low_Rank_Adaptation.md` | 8.2 LoRA Low Rank Adaptation | `draft` — mapped to existing [LoRA cheat sheet](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md); original source body unavailable |
| `finetuning_course/Module_08_PEFT_Model_Merging/8.3_QLoRA_and_Quantization_Aware_Fine_Tuning… [TRUNCATED/UNVERIFIED]` | OPEN: clipped filename; teach only visible prefix: UNVERIFIED] | `unresolved-source` |
| `finetuning_course/Module_08_PEFT_Model_Merging/8.4_Advanced_PEFT_Methods.md` | 8.4 Advanced PEFT Methods | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/8.4_Advanced_PEFT_Methods.md); source body unavailable |
| `finetuning_course/Module_08_PEFT_Model_Merging/8.5_Model_Merging_and_Weight_Averaging.md` | 8.5 Model Merging and Weight Averaging | `draft` — mapped course lesson at [module 21](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/8.5_Model_Merging_and_Weight_Averaging.md); source body unavailable |

<a id="ft-evaluation"></a>

### Fine-tuning Module 09 / cheat sheet 11 — evaluation and debugging

**Status:** partial, unreviewed drafts for mapped source submodules 9.1–9.3; visible submodules 9.4–9.6 remain planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md).

**Learning outcome.** Interpret training/loss metrics, perplexity, comprehensive benchmarks, LLM-as-judge/human evaluation, training-failure diagnostics and infrastructure profiling.

**Focused sheets, derivations and mechanisms.** Separate loss/perplexity, benchmark contamination/slicing, judge calibration/human agreement, optimization-failure triage and infrastructure benchmark sheets. Derive perplexity exp(mean token NLL), paired differences and uncertainty under a stated sampling unit.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 1,000 held-out tokens with summed NLL 1,500 nats; compute perplexity exp(1.5). On 200 independent paired prompts, compare 120 candidate wins, 60 losses and 20 ties with a stated tie convention.

**Alternatives and failure diagnostics.** Compare judge scores with blinded human judgments and training loss with held-out behavior; diagnose tokenization mismatch, leakage, NaNs, data-order bugs and profiling perturbation.

**Bounded case-study decision.** Choose whether one candidate checkpoint should replace its baseline.

**Proposed validation; no result measured.** Plan frozen test data, confidence intervals by prompt/domain, judge calibration, compute/latency profiling and explicit non-inferiority guardrails; no benchmark pass is invented.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/11_Evaluation_Debugging_Training.md` | 11 Evaluation Debugging Training | `planned-topic` |
| `finetuning_course/Module_09_Evaluation_Debugging` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_09_Evaluation_Debugging/9.1_Training_Metrics_and_Loss_Analysis.md` | 9.1 Training Metrics and Loss Analysis | `draft` — mapped course lesson at [module 28](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/9.1_Training_Metrics_and_Loss_Analysis.md); source body unavailable |
| `finetuning_course/Module_09_Evaluation_Debugging/9.2_Perplexity_and_Language_Model_Metrics.md` | 9.2 Perplexity and Language Model Metrics | `draft` — mapped course lesson at [module 28](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/9.2_Perplexity_and_Language_Model_Metrics.md); source body unavailable |
| `finetuning_course/Module_09_Evaluation_Debugging/9.3_Evaluation_Benchmarks_Comprehensive.md` | 9.3 Evaluation Benchmarks Comprehensive | `draft` — mapped course lesson at [module 28](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/9.3_Evaluation_Benchmarks_Comprehensive.md); source body unavailable |
| `finetuning_course/Module_09_Evaluation_Debugging/9.4_LLM_as_Judge_and_Human_Evaluation.md` | 9.4 LLM as Judge and Human Evaluation | `planned-topic` |
| `finetuning_course/Module_09_Evaluation_Debugging/9.5_Debugging_Common_Training_Failures.md` | 9.5 Debugging Common Training Failures | `planned-topic` |
| `finetuning_course/Module_09_Evaluation_Debugging/9.6_Infrastructure_Benchmarking_and_Profiling.md` | 9.6 Infrastructure Benchmarking and Profiling | `planned-topic` |

<a id="ft-inference"></a>

### Fine-tuning Module 10 / cheat sheet 09 — inference and deployment

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md).

**Learning outcome.** Size KV cache/batches, compare inference quantization and speculative decoding, and choose a bounded serving architecture.

**Focused sheets, derivations and mechanisms.** Separate KV-cache/prefill/decode accounting, batching/scheduling, quantization quality/error, speculative acceptance/correctness and deployment interface sheets. Derive KV bytes=2×layers×tokens×KV_heads×head_dim×bytes_per_value×batch; state grouped-query assumptions.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 32 layers, 4,096 cached tokens, 8 KV heads, dimension 128, BF16 at 2 bytes and batch 4; calculate 2 GiB KV storage before fragmentation. For 10 drafted tokens with 7 accepted, distinguish observed acceptance fraction from expected speedup.

**Alternatives and failure diagnostics.** Compare unquantized/quantized and standard/speculative decoding; diagnose KV exhaustion, queue starvation, draft mismatch, unsupported kernels and quality regressions.

**Bounded case-study decision.** Choose a serving policy for a declared context/load SLO.

**Proposed validation; no result measured.** Plan offered/accepted load, time-to-first-token and inter-token latency distributions, memory peaks, exact-distribution checks where required and held-out quality gates.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/09_Inference_Deployment_Serving.md` | 09 Inference Deployment Serving | `planned-topic` |
| `finetuning_course/Module_10_Inference_Deployment` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_10_Inference_Deployment/10.1_Inference_Optimization_KV_Cache_and_Batching.md` | 10.1 Inference Optimization KV Cache and Batching | `planned-topic` |
| `finetuning_course/Module_10_Inference_Deployment/10.2_Quantization_for_Inference.md` | 10.2 Quantization for Inference | `planned-topic` |
| `finetuning_course/Module_10_Inference_Deployment/10.3_Speculative_Decoding.md` | 10.3 Speculative Decoding | `planned-topic` |
| `finetuning_course/Module_10_Inference_Deployment/10.4_Serving_Architecture_and_Deployment.md` | 10.4 Serving Architecture and Deployment | `planned-topic` |

<a id="ft-context"></a>

### Fine-tuning Module 11 / cheat sheet 10 — context extension

**Status:** partial draft. [RoPE and position interpolation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/Cheat_Sheets/03_RoPE_and_Position_Interpolation.md) is authored but unreviewed; the remaining training, inference, evaluation and case-study scope is planned. **Existing destination:** [foundation-model internals](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md).

**Learning outcome.** Explain context utility, RoPE/position interpolation, long-context training, memory-efficient inference and evaluation without equating larger windows with reliable retrieval.

**Focused sheets, derivations and mechanisms.** Split positional rotation/interpolation, extension training schedules, long-context memory policies and positional/task evaluation into separate sheets. Derive RoPE rotation angles and interpolation p_new=p_old×L_old/L_new under the declared scheme; explain aliasing and out-of-distribution positions.

**Numerical exercise to author.** ILLUSTRATIVE inputs: original 4,096-token window, target 16,384 and position 12,000; calculate interpolated position 3,000 and compare dense attention element counts at both lengths.

**Alternatives and failure diagnostics.** Compare retrieval/chunking with extension and interpolation variants; diagnose lost-in-the-middle behavior, position-frequency mismatch and OOM from quadratic components.

**Bounded case-study decision.** Choose retrieval versus context extension for one long-document task.

**Proposed validation; no result measured.** Plan length/position/noise sweeps, answer evidence checks, short-context regression, quality-cost plots and architecture-specific positional-source revalidation.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/10_Context_Length_Extension.md` | 10 Context Length Extension | `planned-topic` |
| `finetuning_course/Module_11_Context_Length_Extension` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_11_Context_Length_Extension/11.1_Why_Context_Length_Matters.md` | 11.1 Why Context Length Matters | `planned-topic` |
| `finetuning_course/Module_11_Context_Length_Extension/11.2_RoPE_and_Position_Interpolation.md` | 11.2 RoPE and Position Interpolation | `partial-draft` |
| `finetuning_course/Module_11_Context_Length_Extension/11.3_Long_Context_Training_Strategies.md` | 11.3 Long Context Training Strategies | `planned-topic` |
| `finetuning_course/Module_11_Context_Length_Extension/11.4_Memory_Efficient_Long_Context_Inference.md` | 11.4 Memory Efficient Long Context Inference | `planned-topic` |
| `finetuning_course/Module_11_Context_Length_Extension/11.5_Evaluation_of_Long_Context_Models.md` | 11.5 Evaluation of Long Context Models | `planned-topic` |

<a id="ft-specialized"></a>

### Fine-tuning Module 12 — VLM, code and tool-use training

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md).

**Learning outcome.** Separate vision-language, code-language and agentic/tool-use training by data contract, supervision and evaluation.

**Focused sheets, derivations and mechanisms.** Create independent introductory sheets for image/text alignment, code completion/execution supervision and tool-call schema/action training; link advanced VLM to Module 18 and speech to Module 20. Derive masked multimodal/assistant-token loss and executable tool-call state invariants.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 200 image tokens, 300 prompt tokens and 100 answer tokens in a 600-token sequence; calculate a response-only loss mask. For 100 code tasks, distinguish 70 syntactically valid from 45 functionally correct outputs.

**Alternatives and failure diagnostics.** Compare frozen vision encoders, text-only coding adaptation and constrained tools; diagnose image-label mismatch, repository leakage, invalid tool arguments and unauthorized actions.

**Bounded case-study decision.** Choose whether one receipt-reading, code-generation or tool-use task needs specialized training; author three separate bounded cases, not one combined deployment.

**Proposed validation; no result measured.** Plan document/image splits, repository/time splits or sandboxed tool tasks as appropriate; each case needs its own task-success and safety protocol.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_12_Multimodal_Specialized` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_12_Multimodal_Specialized/12.1_Vision_Language_Models.md` | 12.1 Vision Language Models | `planned-topic` |
| `finetuning_course/Module_12_Multimodal_Specialized/12.2_Code_Language_Models.md` | 12.2 Code Language Models | `planned-topic` |
| `finetuning_course/Module_12_Multimodal_Specialized/12.3_Agentic_and_Tool_Use_Training.md` | 12.3 Agentic and Tool Use Training | `planned-topic` |

<a id="ft-safety"></a>

### Fine-tuning Module 13 — safety, alignment and interpretability

**Status:** planned. **Existing destination:** [09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md](09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md).

**Learning outcome.** Distinguish alignment principles, safety training, mechanistic interpretability and hallucination mitigation from empirical evidence of safety.

**Focused sheets, derivations and mechanisms.** Separate alignment/objective disagreement, safety data/refusal boundaries, mechanistic attribution/intervention and hallucination/grounding diagnosis sheets. Trace causal intervention versus correlational activation analysis and derive false-refusal/unsafe-compliance rates under defined labels.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 100 benign prompts with 8 refusals and 100 prohibited prompts with 12 unsafe completions; calculate false-refusal and unsafe-compliance rates and identify missing uncertainty and threat-model assumptions.

**Alternatives and failure diagnostics.** Compare training, retrieval, policy controls and human escalation; diagnose over-refusal, benchmark contamination, ungrounded attribution and hidden unsafe tool effects.

**Bounded case-study decision.** Choose one safety intervention for a specified prohibited-action boundary.

**Proposed validation; no result measured.** Plan held-out adversarial and benign slices, independent human adjudication, intervention ablations and regression gates; interpretability signals alone do not certify safety.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_13_Safety_Alignment_Interpretability` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_13_Safety_Alignment_Interpretability/13.1_Alignment_Principles_and_the_Alignment_Problem.md` | 13.1 Alignment Principles and the Alignment Problem | `planned-topic` |
| `finetuning_course/Module_13_Safety_Alignment_Interpretability/13.2_Safety_Training_Techniques.md` | 13.2 Safety Training Techniques | `planned-topic` |
| `finetuning_course/Module_13_Safety_Alignment_Interpretability/13.3_Mechanistic_Interpretability.md` | 13.3 Mechanistic Interpretability | `planned-topic` |
| `finetuning_course/Module_13_Safety_Alignment_Interpretability/13.4_Hallucination_Causes_and_Mitigation.md` | 13.4 Hallucination Causes and Mitigation | `planned-topic` |

<a id="ft-mlops"></a>

### Fine-tuning Module 14 / cheat sheet 14 — practical MLOps workflows

**Status:** mostly planned; [checkpoint and recovery](05_ai_platforms_data_systems_and_production_engineering/Cheat_Sheets/01_Distributed_Training_Checkpoint_and_Recovery.md) is an unreviewed partial draft. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md).

**Learning outcome.** Design experiments/ablations, tracking/reproducibility, checkpoints/fault tolerance, post-training production handoffs and compute/cost plans.

**Focused sheets, derivations and mechanisms.** Split experimental controls, artifact lineage, checkpoint/resume state, promotion pipeline and all-in cost modeling into separate sheets. Derive expected lost work≈checkpoint_interval/2 under uniform failure time and expose serialization/recovery overhead.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 8 GPUs at illustrative 3 currency-units/GPU-hour, 12 training hours, 2 evaluation hours and 10% failed-work overhead; calculate compute-only cost before data/storage/labor. Compare 20-minute checkpoint intervals and 2-minute writes.

**Alternatives and failure diagnostics.** Compare faster runs with reproducible/economical runs; diagnose incomplete optimizer/RNG restore, untracked data revisions and unpriced evaluation or failed work.

**Bounded case-study decision.** Choose a checkpoint interval and promotion gate for one training job.

**Proposed validation; no result measured.** Plan a seeded rerun tolerance, artifact manifest, forced interruption/resume, held-out promotion criteria and a dated price ledger; no real vendor price is asserted.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/14_Data_MLOps_Production_Workflows.md` | 14 Data MLOps Production Workflows | `planned-topic` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows/14.1_Experiment_Design_and_Ablations.md` | 14.1 Experiment Design and Ablations | `planned-topic` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows/14.2_Experiment_Tracking_and_Reproducibility.md` | 14.2 Experiment Tracking and Reproducibility | `planned-topic` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows/14.3_Checkpointing_and_Fault_Tolerance.md` | 14.3 Checkpointing and Fault Tolerance | `planned-topic` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows/14.4_Post_Training_Production_Pipeline.md` | 14.4 Post Training Production Pipeline | `planned-topic` |
| `finetuning_course/Module_14_MLOps_Practical_Workflows/14.5_Cost_Estimation_and_Compute_Planning.md` | 14.5 Cost Estimation and Compute Planning | `planned-topic` |

<a id="ft-advanced-offline"></a>

### Fine-tuning Module 15 — continued pretraining, sampling, distillation and multi-task adaptation

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Distinguish continued pretraining/domain adaptation, rejection-sampling fine-tuning/Best-of-N, online/iterative preference optimization, distillation and multi-task instruction tuning.

**Focused sheets, derivations and mechanisms.** Plan one sheet per method, not an advanced-method survey: domain/general mixture objectives; Best-of-N selection bias and acceptance; iterative data/policy feedback; temperature-scaled distillation; and multi-task sampling/loss weighting. Derive sample acceptance and teacher/student KL under declared supports.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 2,000 prompts, 8 candidates/prompt, 25% pass a verifier and 200 generated tokens/candidate; calculate candidate tokens and expected accepted candidates, separating training examples from generation cost. For task weights 0.7/0.3, compute allocations from 10 million training tokens.

**Alternatives and failure diagnostics.** Compare SFT/replay with continued pretraining, teacher sampling with distillation and fixed offline preferences with iterative updates; diagnose verifier bias, teacher contamination and task interference.

**Bounded case-study decision.** Choose distillation versus sampled supervised adaptation for one bounded task and budget.

**Proposed validation; no result measured.** Plan disjoint prompt families, teacher/student and baseline comparisons, general-capability regressions and full generation-plus-training costs.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_15_Advanced_Post_Training/15.1_Continued_Pretraining_and_Domain_Adaptation.md` | 15.1 Continued Pretraining and Domain Adaptation | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.2_Rejection_Sampling_Fine_Tuning_and_Best_of_N.md` | 15.2 Rejection Sampling Fine Tuning and Best of N | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.3_Online_and_Iterative_Preference_Optimization.md` | 15.3 Online and Iterative Preference Optimization | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.4_Knowledge_Distillation_for_LLMs.md` | 15.4 Knowledge Distillation for LLMs | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.9_Multi_Task_Instruction_Tuning_at_Scale.md` | 15.9 Multi Task Instruction Tuning at Scale | `planned-topic` |

<a id="ft-advanced-rl"></a>

### Fine-tuning Module 15 — KTO, ORPO, SimPO, RLVR, DAPO, REINFORCE++ and self-play

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** State the distinct data and reward assumptions of KTO, ORPO, SimPO, reinforcement learning from verifiable rewards, DAPO, REINFORCE++ and self-play/self-improvement.

**Focused sheets, derivations and mechanisms.** Require independent proposed sheets for KTO utility/reference treatment, ORPO odds-ratio plus supervised objective, SimPO reference-free margin, RLVR verifier contract, DAPO objective/sampling details, REINFORCE++ baseline/variance treatment and self-play feedback. Derive each objective from its original versioned paper; do not silently equate different normalizations. Link GRPO separately.

**Numerical exercise to author.** ILLUSTRATIVE inputs: four completions with rewards [0,0,1,1], lengths [20,40,20,80] tokens and log-probability sums [-20,-32,-18,-64]; compare sequence versus per-token scores and centered rewards. For a verifier with 2 false accepts in 100 invalid outputs, calculate empirical false-accept rate.

**Alternatives and failure diagnostics.** Compare these methods with SFT/DPO/GRPO under equal sample/token budgets; diagnose reward hacking, length bias, zero signal, verifier loopholes and self-play collapse.

**Bounded case-study decision.** Choose exactly one advanced objective for a constrained, verifiable task rather than adding every RL method to a pipeline.

**Proposed validation; no result measured.** Plan a separately authored bounded comparison for each objective, independently audited verifiers, held-out problem generators, seed variation, safety/general regressions and total rollout cost. Research names are targets for authoritative revalidation.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_15_Advanced_Post_Training/15.5_KTO_ORPO_SimPO_Advanced_Alignment.md` | 15.5 KTO ORPO SimPO Advanced Alignment | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.6_RLVR_Reinforcement_from_Verifiable_Rewards.md` | 15.6 RLVR Reinforcement from Verifiable Rewards | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.7_DAPO_REINFORCE_Plus_Plus_and_Advanced_RL.md` | 15.7 DAPO REINFORCE Plus Plus and Advanced RL | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.8_Self_Play_and_Self_Improvement.md` | 15.8 Self Play and Self Improvement | `planned-topic` |

<a id="ft-continual"></a>

### Fine-tuning Module 15 / cheat sheet 08 — continual learning and forgetting

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Quantify catastrophic forgetting and compare continual learning/replay with adapter isolation and compatible model merging.

**Focused sheets, derivations and mechanisms.** Create separate continual-learning retention/replay and merge-versus-adapter-composition sheets, cross-linking Module 08 for merge mechanics. Derive task-by-time accuracy matrices, average retained performance and backward transfer with a fixed sign convention.

**Numerical exercise to author.** ILLUSTRATIVE inputs: task A accuracy before/after new-task training 0.84/0.72 and task B 0.40/0.80; compute the 12-percentage-point retention loss and evaluate a replay candidate yielding A=0.82 and B=0.77.

**Alternatives and failure diagnostics.** Compare replay, separate adapters, regularization and merging; diagnose hidden task-distribution shifts, memory-budget inequivalence and incompatible weight coordinates.

**Bounded case-study decision.** Choose whether to promote a domain update given a declared old-task regression limit of 3 percentage points.

**Proposed validation; no result measured.** Plan versioned old/new task sets, multiple update orders, replay-budget accounting, uncertainty and adapter/base compatibility tests. No actual retention result is supplied.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/08_Model_Merging_Continual_Learning.md` | 08 Model Merging Continual Learning | `planned-topic` |
| `finetuning_course/Module_15_Advanced_Post_Training` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_15_Advanced_Post_Training/15.10_Continual_Learning_and_Catastrophic_Forgetting.md` | 15.10 Continual Learning and Catastrophic Forgetting | `planned-topic` |

<a id="ft-alternative-architectures"></a>

### Fine-tuning Module 16 / cheat sheet 13 — alternative architectures, pruning and efficiency

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md).

**Learning outcome.** Distinguish Mamba, RWKV and RetNet mechanisms from attention and analyze model pruning/sparsification without universal efficiency claims.

**Focused sheets, derivations and mechanisms.** Propose independent Mamba selective-state update, RWKV recurrence, RetNet retention, structured/unstructured pruning and sparsification sheets. Derive each architecture-specific state recurrence from its original source, compare state/activation scaling, and separate parameter sparsity from actual kernel speedup.

**Numerical exercise to author.** ILLUSTRATIVE inputs: sequence 8,192 tokens, model width 1,024 and a hypothetical recurrent state of 1,024×16 FP32 values; calculate 64 KiB state versus one FP16 8,192×8,192 attention-score matrix of 128 MiB, explicitly noting these are non-equivalent partial memory components.

**Alternatives and failure diagnostics.** Compare transformer, recurrent/state-space and pruned models under matched task and token budgets; diagnose lost long-range quality, unsupported sparse kernels and misleading partial-memory comparisons.

**Bounded case-study decision.** Choose an architecture or pruning candidate for one streaming-memory constraint.

**Proposed validation; no result measured.** Plan fixed-data/task comparisons, throughput and full-memory measurements, long-range tests and hardware/kernel compatibility records; source-specific claims require dated primary references.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/13_Advanced_Architectures_Efficiency.md` | 13 Advanced Architectures Efficiency | `planned-topic` |
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency/16.1_Alternative_Architectures_Mamba_RWKV_RetNet.md` | 16.1 Alternative Architectures Mamba RWKV RetNet | `planned-topic` |
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency/16.2_Model_Pruning_and_Sparsification.md` | 16.2 Model Pruning and Sparsification | `planned-topic` |

<a id="ft-tokenizers-multilingual"></a>

### Fine-tuning Module 16 — tokenizer training, curriculum and multilingual learning

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md).

**Learning outcome.** Train and evaluate tokenizers, design curriculum/data ordering and measure multilingual LLM quality without reducing native-language performance to aggregate English metrics.

**Focused sheets, derivations and mechanisms.** Separate tokenizer-from-scratch vocabulary/normalization/segmentation, curriculum/order controls and multilingual sampling/transfer sheets. Derive byte/token fertility, coverage and length-weighted language sampling; trace embedding/token-ID compatibility after vocabulary changes.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 10,000 native-language characters become 8,000 old tokens versus 5,000 new tokens; compute characters/token and a 2,048-token window coverage change. With languages at 90%/10% corpus share, calculate a declared square-root resampling distribution.

**Alternatives and failure diagnostics.** Compare a new tokenizer with vocabulary extension and adapter-only multilingual adaptation; diagnose broken Unicode normalization, changed token IDs, curriculum confounding and minority-language regressions.

**Bounded case-study decision.** Choose whether tokenizer replacement is justified for one low-resource language.

**Proposed validation; no result measured.** Plan native-reviewed held-out text/tasks, code-switch and script slices, tokenizer round trips, embedding migration checks, equal-byte/token-cost reporting and randomized curriculum controls.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency/16.3_Tokenizer_Training_from_Scratch.md` | 16.3 Tokenizer Training from Scratch | `planned-topic` |
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency/16.4_Curriculum_Learning_and_Data_Ordering.md` | 16.4 Curriculum Learning and Data Ordering | `planned-topic` |
| `finetuning_course/Module_16_Advanced_Architectures_Efficiency/16.5_Multilingual_LLM_Training.md` | 16.5 Multilingual LLM Training | `planned-topic` |

<a id="ft-playbook-data"></a>

### Fine-tuning Module 17 — environment, hardware and data implementation

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Translate environment/hardware selection, data-pipeline implementation and domain data synthesis/curation into an executable, auditable training-input contract.

**Focused sheets, derivations and mechanisms.** Separate versioned environment/device capability selection, streaming/packing data implementation and synthetic-data provenance/filtering sheets. Derive data throughput required to avoid compute starvation and trace source→tokenizer→pack→loss mask with deterministic IDs.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 4 GPUs consume 12,000 tokens/s each, input shards deliver 30,000 tokens/s and preprocessing raises delivery to 60,000 tokens/s; compute the original deficit and new headroom. For 100,000 synthetic records with 15% rejection, calculate retained records.

**Alternatives and failure diagnostics.** Compare local cached with streamed data and real with synthesized examples; diagnose dataloader stalls, nondeterministic shard duplication, synthetic contamination and dependency mismatch.

**Bounded case-study decision.** Choose a data delivery and environment configuration for one planned SFT run.

**Proposed validation; no result measured.** Plan environment lock/capability records, batch inspection, shard-disjointness checks, throughput traces and independent human review of a sampled synthetic subset.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_17_Implementation_Playbook/17.1_Environment_Setup_and_Hardware_Selection.md` | 17.1 Environment Setup and Hardware Selection | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.2_Data_Pipeline_Implementation.md` | 17.2 Data Pipeline Implementation | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.15_Domain_Data_Synthesis_and_Curation.md` | 17.15 Domain Data Synthesis and Curation | `planned-topic` |

<a id="ft-playbook-training"></a>

### Fine-tuning Module 17 — adapter, SFT, memory, distributed and preference implementation

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Implement distinct LoRA/QLoRA, SFT loop, memory optimization, distributed setup and DPO/GRPO workflows using the corresponding mathematical contracts.

**Focused sheets, derivations and mechanisms.** Require separate implementation sheets for each workflow, reusing rather than redefining Module 04/05/07/08 objectives. Trace trainable-parameter selection, label masking, optimizer-step/accumulation timing, sharded groups and policy/reference or group reward inputs.

**Numerical exercise to author.** ILLUSTRATIVE inputs: microbatch 2, accumulation 8, data degree 4 and 1,024 answer tokens/sequence; calculate 64 sequences and 65,536 supervised answer tokens/update, then audit one optimizer step.

**Alternatives and failure diagnostics.** Compare full with adapter training, unsharded with distributed execution and DPO with GRPO; diagnose frozen/accidentally trainable tensors, loss normalization errors and wrong reference policy.

**Bounded case-study decision.** Choose one minimal correct adaptation loop before enabling distributed or memory optimizations.

**Proposed validation; no result measured.** Plan tiny-data overfit, reference loss/gradient checks, one-versus-many-rank equivalence and memory/time profiles; these are future implementation requirements, not executed recipes.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_17_Implementation_Playbook/17.3_LoRA_and_QLoRA_Implementation.md` | 17.3 LoRA and QLoRA Implementation | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.4_SFT_Training_Loop_Best_Practices.md` | 17.4 SFT Training Loop Best Practices | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.5_Memory_Optimization_Playbook.md` | 17.5 Memory Optimization Playbook | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.6_Distributed_Training_Setup.md` | 17.6 Distributed Training Setup | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.7_Preference_Optimization_DPO_GRPO.md` | 17.7 Preference Optimization DPO GRPO | `planned-topic` |

<a id="ft-playbook-operations"></a>

### Fine-tuning Module 17 — evaluation, monitoring, quantization and reproducibility implementation

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md).

**Learning outcome.** Wire evaluation pipelines, training debugging/monitoring, quantization/inference optimization and reproducibility tracking into a run-to-promotion audit trail.

**Focused sheets, derivations and mechanisms.** Plan independent evaluation runner, failure-monitoring, quantized-serving export and experiment-lineage sheets. Derive step/token-normalized metrics, explicit alert thresholds and artifact compatibility invariants; link inference and evaluation foundations.

**Numerical exercise to author.** ILLUSTRATIVE inputs: evaluate 500 prompts at 3 checkpoints with 2 decoding seeds; compute 3,000 generations. With 0.02 illustrative currency-units/generation, estimate 60 units before annotation and retries.

**Alternatives and failure diagnostics.** Compare fixed versus adaptive evaluation and quantized versus original exports; diagnose metric denominator drift, stale checkpoint labels, NaNs and export/tokenizer mismatch.

**Bounded case-study decision.** Choose whether a quantized adapter export may enter a limited evaluation deployment.

**Proposed validation; no result measured.** Plan manifest reconciliation, fixed prompt/seed records, baseline comparisons, failure injection and export-versus-training-runtime output tolerances.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_17_Implementation_Playbook/17.8_Evaluation_Pipeline_Implementation.md` | 17.8 Evaluation Pipeline Implementation | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.9_Debugging_and_Training_Monitoring.md` | 17.9 Debugging and Training Monitoring | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.10_Quantization_and_Inference_Optimization.md` | 17.10 Quantization and Inference Optimization | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.11_MLOps_Experiment_Tracking_Reproducibility.md` | 17.11 MLOps Experiment Tracking Reproducibility | `planned-topic` |

<a id="ft-playbook-composition"></a>

### Fine-tuning Module 17 — merging, context extension and end-to-end production implementation

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Compose model merging/adapters, context extension and an end-to-end production pipeline without skipping independent compatibility and release gates.

**Focused sheets, derivations and mechanisms.** Separate adapter composition/merge compatibility, context-extension implementation and release orchestration sheets. Trace base/tokenizer/adapter identity, positional configuration, evaluation artifacts and reversible promotion state.

**Numerical exercise to author.** ILLUSTRATIVE inputs: base revision A, two adapters trained on A and a candidate base B with different token IDs; construct a compatibility matrix and disqualify unsupported combinations. For a 4× context extension, estimate 16× dense attention-score elements.

**Alternatives and failure diagnostics.** Compare isolated adapters with merges and long-context training with retrieval; diagnose incompatible adapter bases, positional mismatch and promotion without rollback artifacts.

**Bounded case-study decision.** Choose a release unit for one adapted model rather than an unbounded production transformation.

**Proposed validation; no result measured.** Plan base/adapter/tokenizer hash checks, short/long-context regression, staged load tests and rollback drills using preserved prior artifacts.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_17_Implementation_Playbook` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_17_Implementation_Playbook/17.12_Model_Merging_and_Adapter_Composition.md` | 17.12 Model Merging and Adapter Composition | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.13_Context_Length_Extension_Implementation.md` | 17.13 Context Length Extension Implementation | `planned-topic` |
| `finetuning_course/Module_17_Implementation_Playbook/17.14_End_to_End_Production_Pipeline.md` | 17.14 End to End Production Pipeline | `planned-topic` |

<a id="ft-vlm-training"></a>

### Fine-tuning Module 18 — VLM architecture, data and training

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md).

**Learning outcome.** Reason across VLM components, vision-encoder fine-tuning, multimodal data, instruction/alignment training and LoRA/frozen-component strategies.

**Focused sheets, derivations and mechanisms.** Split architecture/component interfaces, encoder gradients/resolution, multimodal dataset/packing, instruction tuning and efficient frozen/adapter/full strategies into independent sheets. Derive image patch counts and projected-token/loss-mask dimensions.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 448×448-pixel images with 14×14-pixel patches yield 1,024 patches/image before special tokens; with 2 images, 256 text tokens and batch 4, calculate 9,216 input tokens before any pooling.

**Alternatives and failure diagnostics.** Compare frozen encoder/projector training, LoRA and full VLM adaptation; diagnose image-text mispairing, resolution-driven memory growth and silent frozen gradients.

**Bounded case-study decision.** Choose which VLM components to train for one bounded document/image instruction task.

**Proposed validation; no result measured.** Plan image/source/entity-disjoint splits, image-text pairing audits, ablations over trainable components, task quality and full memory/throughput measurements.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.1_VLM_Architecture_and_Component_Overview.md` | 18.1 VLM Architecture and Component Overview | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.2_Vision_Encoder_Finetuning.md` | 18.2 Vision Encoder Finetuning | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.3_VLM_Data_Pipeline_and_Multimodal_Datasets.md` | 18.3 VLM Data Pipeline and Multimodal Datasets | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.4_VLM_Instruction_Tuning_and_Alignment.md` | 18.4 VLM Instruction Tuning and Alignment | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.5_Efficient_VLM_Training_LoRA_and_Frozen_Strategies.md` | 18.5 Efficient VLM Training LoRA and Frozen Strategies | `planned-topic` |

<a id="ft-vlm-docs-video"></a>

### Fine-tuning Module 18 — OCR, charts, multimodal preferences and video

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md).

**Learning outcome.** Distinguish document understanding/OCR/chart reasoning, multimodal preference optimization/RLHF and video-language temporal understanding.

**Focused sheets, derivations and mechanisms.** Create separate OCR/layout, chart-value reasoning, multimodal preference objective and video temporal sampling/alignment sheets. Derive normalized box coordinates, numeric chart-error measures and frame/token budgets; trace where preferences may reward plausible but unsupported text.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 1,200×1,600-pixel page with box (120,160,360,480); compute normalized coordinates. For 60-second video sampled at 2 frames/s and 256 tokens/frame, calculate 120 frames and 30,720 visual tokens before text.

**Alternatives and failure diagnostics.** Compare OCR-plus-text pipelines with end-to-end VLM and sparse frames with temporal models; diagnose layout leakage, chart-unit mistakes, temporal-order blindness and hallucinated visual evidence.

**Bounded case-study decision.** Author separate decisions for a chart-reading system and a temporal event task rather than one omnibus VLM case.

**Proposed validation; no result measured.** Plan document-family splits, exact numeric/unit checks, counterfactual image swaps, reordered-frame tests and independent visual preference review.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.6_Document_Understanding_OCR_and_Chart_Models.md` | 18.6 Document Understanding OCR and Chart Models | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.7_Multimodal_Preference_Optimization_and_RLHF.md` | 18.7 Multimodal Preference Optimization and RLHF | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.9_Video_Language_Models_and_Temporal_Understanding.md` | 18.9 Video Language Models and Temporal Understanding | `planned-topic` |

<a id="ft-vlm-eval-deploy"></a>

### Fine-tuning Module 18 — evaluation, production and research boundaries

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md).

**Learning outcome.** Evaluate and deploy a VLM while keeping 2025 research-frontier labels historical until primary-source review.

**Focused sheets, derivations and mechanisms.** Separate VLM benchmark/metric validity, serving image preprocessing/token accounting and research replication/source review sheets. Derive task-specific denominators and visual-token capacity; document unsupported modalities rather than treating all VLMs as interchangeable.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 200 held-out documents, 180 correct field extractions and 20 erroneous outputs; compute exact-match rate and stratify an explicitly supplied 50-document rotated-page subset separately. For 10 concurrent requests at 1,024 visual tokens each, account for 10,240 visual tokens before text.

**Alternatives and failure diagnostics.** Compare pipeline and VLM baselines under identical document splits; diagnose contamination, OCR-normalization scoring artifacts, unsupported image resolution and serving/training preprocessing mismatch.

**Bounded case-study decision.** Choose whether one evaluated VLM should replace an OCR baseline for a narrow field-extraction task.

**Proposed validation; no result measured.** Plan exact match/numeric error/grounding metrics, human adjudication, latency/memory under load and source/version records for every frontier claim; no 2025 claim is presumed current.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.8_VLM_Evaluation_Benchmarks_and_Metrics.md` | 18.8 VLM Evaluation Benchmarks and Metrics | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.10_VLM_Production_Deployment_and_Optimization.md` | 18.10 VLM Production Deployment and Optimization | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/18.11_Research_Frontiers_CV_VLM_2025.md` | 18.11 Research Frontiers CV VLM 2025 | `planned-topic` |
| `finetuning_course/Module_18_CV_VLM_Finetuning_Research/INDEX.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="ft-distributed-frameworks"></a>

### Fine-tuning Module 19 — DeepSpeed, FSDP2 and Megatron-LM

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Compare the modern distributed landscape, advanced DeepSpeed configuration, PyTorch-native FSDP2 and Megatron-LM 3D parallelism through explicit state/communication contracts.

**Focused sheets, derivations and mechanisms.** Require independent DeepSpeed state/offload configuration, FSDP2 shard/reshard/gradient behavior and Megatron tensor/pipeline/data layout sheets, plus a landscape selection map. Derive per-rank state memory and collective payloads for a pinned version/configuration.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 8 billion parameters, 2-byte weights, 2-byte gradients and 8 bytes optimizer moments/parameter on eight ranks; compute 96 GB unsharded state and a 12 GB ideal fully sharded component sum before activations/gathers/fragmentation.

**Alternatives and failure diagnostics.** Compare framework-native strategies, not generic brand rankings; diagnose configuration defaults, gather peaks, process-group mismatch and incompatible checkpoint formats.

**Bounded case-study decision.** Choose one framework/configuration for a model-memory and topology constraint.

**Proposed validation; no result measured.** Plan pinned versions, identical model/data/global batch, state/memory audits, numerical equivalence, checkpoint portability limitations and equal-token throughput measurements.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.1_Modern_Distributed_Training_Landscape.md` | 19.1 Modern Distributed Training Landscape | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.2_DeepSpeed_Advanced_Configuration.md` | 19.2 DeepSpeed Advanced Configuration | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.3_FSDP2_PyTorch_Native_Distributed.md` | 19.3 FSDP2 PyTorch Native Distributed | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.4_Megatron_LM_and_3D_Parallelism.md` | 19.4 Megatron LM and 3D Parallelism | `planned-topic` |

<a id="ft-distributed-resilience"></a>

### Fine-tuning Module 19 — resilience, profiling, cluster management and costs

**Status:** mostly planned; [checkpoint and recovery](05_ai_platforms_data_systems_and_production_engineering/Cheat_Sheets/01_Distributed_Training_Checkpoint_and_Recovery.md) is an unreviewed partial draft. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Integrate fault tolerance/elasticity, at-scale profiling, SLURM/Kubernetes management, compute planning and multinode debugging/monitoring.

**Focused sheets, derivations and mechanisms.** Separate elastic membership/data replay, distributed profiling, scheduler placement, cost/capacity and multinode failure-triage sheets. Derive useful-token throughput, recovery-time/lost-work and cost per accepted checkpoint, with actual completion rather than allocated GPU-hours as denominator.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 16 GPUs, 10 hours allocated, 8 hours useful, 0.5 hour recovery and illustrative 2.5 currency-units/GPU-hour; calculate paid compute cost and useful fraction, keeping recovery overlap assumptions explicit.

**Alternatives and failure diagnostics.** Compare fixed allocation with elasticity and SLURM with Kubernetes only under stated workload needs; diagnose stragglers, data replay/omission, rank hangs, scheduler eviction and network failures.

**Bounded case-study decision.** Choose checkpoint/recovery and placement policies for one 16-GPU job.

**Proposed validation; no result measured.** Plan deliberate rank loss, deterministic data-ID accounting after resume, per-rank timelines, scheduler manifests, cost ledger and independent bottleneck confirmation.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.5_Fault_Tolerance_and_Elastic_Training.md` | 19.5 Fault Tolerance and Elastic Training | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.6_At_Scale_Profiling_and_Performance_Optimization.md` | 19.6 At Scale Profiling and Performance Optimization | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.8_Cluster_Management_SLURM_Kubernetes.md` | 19.8 Cluster Management SLURM Kubernetes | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.9_Cost_Optimization_and_Compute_Planning.md` | 19.9 Cost Optimization and Compute Planning | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.10_Multi_Node_Debugging_and_Distributed_Monitoring.md` | 19.10 Multi Node Debugging and Distributed Monitoring | `planned-topic` |

<a id="ft-distributed-overlap"></a>

### Fine-tuning Module 19 — ZeRO++, DualPipe and research frontiers

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md).

**Learning outcome.** Explain communication overlap, ZeRO++ and DualPipe separately, and bound historical 2025 distributed-training research claims.

**Focused sheets, derivations and mechanisms.** Create independent communication/computation overlap, ZeRO++ communication-reduction mechanics, DualPipe scheduling and research replication sheets. Derive exposed communication=max(0,Tcomm−Toverlap) only under a stated dependency model; record the precise quantization/grouping/schedule variant from original sources.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 80 ms compute, 50 ms communication, 35 ms legally overlappable communication and 5 ms scheduling overhead; compare serialized 130 ms with an estimated 100 ms step under that simplified model.

**Alternatives and failure diagnostics.** Compare no overlap, ordinary overlap, ZeRO++ and DualPipe at equal numerics/batch; diagnose contention, invalid dependencies, quantized-communication error and pipeline imbalance.

**Bounded case-study decision.** Choose whether one communication-bound run merits an overlap/schedule change.

**Proposed validation; no result measured.** Plan timeline traces, numerical-error budgets, weak/strong scaling, failure recovery and source/commit pinning; original paper results are not reproduced results.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_19_Distributed_Training_Infrastructure` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.7_Communication_Overlap_ZeRO_Plus_Plus_and_DualPipe.md` | 19.7 Communication Overlap ZeRO Plus Plus and DualPipe | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.11_Research_Frontiers_Distributed_Training_2025.md` | 19.11 Research Frontiers Distributed Training 2025 | `planned-topic` |
| `finetuning_course/Module_19_Distributed_Training_Infrastructure/INDEX.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="ft-speech-asr"></a>

### Fine-tuning Module 20 — speech foundations, ASR/STT, Whisper and streaming

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md).

**Learning outcome.** Explain speech/ASR/STT foundations, compare ASR architecture families, adapt Whisper and design streaming/real-time recognition.

**Focused sheets, derivations and mechanisms.** Separate waveform/features and sample-rate contracts, ASR architecture/objective mechanisms, Whisper fine-tuning and streaming/chunk/state sheets. Derive duration from samples/rate, word error rate (S+D+I)/N and real-time factor runtime/audio duration; explain partial versus final transcript latency.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 30 seconds mono audio at 16 kHz, 16-bit PCM; calculate 480,000 samples and 960,000 payload bytes. For N=100 reference words, S=8, D=3 and I=4, compute WER=15%; 6-second processing gives RTF=0.2.

**Alternatives and failure diagnostics.** Compare frozen base, full/adapter Whisper adaptation and streaming alternatives; diagnose resampling errors, speaker/call leakage, timestamp drift and hallucinations on silence.

**Bounded case-study decision.** Choose whether a domain Whisper adapter meets an explicitly declared ASR/STT quality/latency constraint.

**Proposed validation; no result measured.** Plan speaker/call/time-disjoint splits, independent human transcripts, domain/accent/noise/code-switch slices, paired WER uncertainty and chunk latency tests.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.1_Speech_Processing_Landscape_and_Foundations.md` | 20.1 Speech Processing Landscape and Foundations | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.2_ASR_Architecture_Deep_Dive.md` | 20.2 ASR Architecture Deep Dive | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.3_Whisper_Finetuning_Complete_Guide.md` | 20.3 Whisper Finetuning Complete Guide | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.4_Streaming_Real_Time_ASR.md` | 20.4 Streaming Real Time ASR | `planned-topic` |

<a id="ft-speech-tts"></a>

### Fine-tuning Module 20 — TTS, voice cloning and native-language training

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md).

**Learning outcome.** Distinguish modern TTS architecture, TTS training/fine-tuning, zero-shot voice cloning, multilingual/low-resource speech and a native-language data/training pipeline.

**Focused sheets, derivations and mechanisms.** Require separate TTS text/acoustic/vocoder architecture, alignment/training, voice-cloning consent/speaker conditioning, multilingual sampling and native-language normalization/lexicon sheets. Derive duration/frame counts and language/speaker sampling; do not present similarity as consent or identity verification.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 5 hours consented speech at 80 acoustic frames/s yields 1,440,000 frames; a 90%/10% speaker distribution with a declared balanced sampler gives equal sampling rather than equal unique data. Compare 20 utterances rated by 5 native speakers with explicit ordinal-rating limits.

**Alternatives and failure diagnostics.** Compare recorded prompts, non-cloned stock voices, full TTS training and zero-shot cloning; diagnose pronunciation/tokenization errors, speaker leakage, alignment collapse and abusive impersonation risks.

**Bounded case-study decision.** Choose an authorized native-language voice strategy for one constrained speech product.

**Proposed validation; no result measured.** Plan explicit recorded consent/usage rights, speaker-disjoint tests, native-reviewed pronunciation/intelligibility, blinded naturalness/similarity studies and misuse safeguards; no cloned identity is inferred from filenames.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.5_TTS_Architecture_and_Modern_Models.md` | 20.5 TTS Architecture and Modern Models | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.6_TTS_Training_and_Finetuning.md` | 20.6 TTS Training and Finetuning | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.7_Voice_Cloning_and_Zero_Shot_TTS.md` | 20.7 Voice Cloning and Zero Shot TTS | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.8_Multilingual_Speech_and_Low_Resource_Languages.md` | 20.8 Multilingual Speech and Low Resource Languages | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.9_Native_Language_Training_Pipeline.md` | 20.9 Native Language Training Pipeline | `planned-topic` |

<a id="ft-speech-production"></a>

### Fine-tuning Module 20 / cheat sheet 12 — speech-LLM integration, evaluation and deployment

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md).

**Learning outcome.** Evaluate speech-LLM/foundation integration and production speech deployment while marking 2025–2026 research labels as source-review targets.

**Focused sheets, derivations and mechanisms.** Split speech-to-text-to-LLM versus integrated speech-model interfaces, speech benchmarks/metric design, production buffering/latency/capacity and research-source audit into separate sheets. Cheat sheet 12 is a family: also link VLM/code/tool-training briefs rather than substituting a speech-only survey. Derive end-to-end latency sums and quality/latency bottlenecks with shared-resource caveats.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 300 ms endpointing, 200 ms ASR, 400 ms model response and 250 ms TTS first audio; calculate 1,150 ms serial first-response latency, then identify which stages can actually overlap. With 8 audio-seconds/s service capacity and 6 audio-seconds/s arrival, compute 75% offered utilization.

**Alternatives and failure diagnostics.** Compare cascaded and integrated speech systems; diagnose accumulated transcription errors, queue buildup, benchmark normalization artifacts and unmeasured native-language quality.

**Bounded case-study decision.** Choose a cascade or integrated interface for one latency-constrained language route, without claiming all routes are supported.

**Proposed validation; no result measured.** Plan WER/CER, intelligibility/naturalness, task completion, route-specific latency, concurrency overload tests and native-speaker review. Research-frontier claims require primary date/version evidence.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/12_Multimodal_Speech_Specialized.md` | 12 Multimodal Speech Specialized | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.10_Speech_LLM_Integration_and_Foundation_Models.md` | 20.10 Speech LLM Integration and Foundation Models | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.11_Speech_Evaluation_Benchmarks_and_Metrics.md` | 20.11 Speech Evaluation Benchmarks and Metrics | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.12_Production_Speech_Systems_and_Deployment.md` | 20.12 Production Speech Systems and Deployment | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/20.13_Research_Frontiers_Speech_2025_2026.md` | 20.13 Research Frontiers Speech 2025 2026 | `planned-topic` |
| `finetuning_course/Module_20_Speech_Processing_ASR_TTS_STT/INDEX.md` | Source navigation/status — reconcile with this focused group | `supporting-artifact` |

<a id="ft-common-issues"></a>

### Fine-tuning cheat sheet 15 — 50 Common LLM Fine-Tuning Issues target

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Plan 50 distinct issues across data, tokenization, memory, loss, distributed groups, preference/RL, adapters, context, multimodal/speech and export.

**Focused sheets, derivations and mechanisms.** Use the template’s 12-section troubleshooting adaptation and one canonical issue register. Each issue must include discriminating diagnostic, cause, fix, verification and prevention; related failures link to focused method sheets. The visible 50 count is a future coverage target, not completed issue content.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 512 prompt tokens, 128 answer tokens and 384 padding tokens per training sample; calculate 128/1,024=12.5% intended supervised positions and trace an accidental all-token loss mask.

**Alternatives and failure diagnostics.** Compare a label-mask bug with optimization failure and data mismatch; never suppress NaNs, failed ranks or missing labels as a fix.

**Bounded case-study decision.** Choose the first experiment for a decreasing-loss model whose target-task quality does not improve.

**Proposed validation; no result measured.** Plan controlled fault injection, minimal examples, counter-diagnostics and fixed-data before/after comparisons; only actual authored substantive entries count toward 50.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/15_50_Common_LLM_Finetuning_Issues.md` | 15 50 Common LLM Finetuning Issues | `planned-topic` |

<a id="cuda-source-artifacts"></a>

### CUDA source containers, indexes and collation utilities

**Status:** planned. **Existing destination:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md).

**Learning outcome.** Preserve navigation and source provenance without mistaking maintenance utilities or collated text for authored GPU lessons.

**Focused sheets, derivations and mechanisms.** Supporting-artifact briefs: README and 00_INDEX provide navigation/status; all_content.txt is a collation snapshot; collate_md.py is a collation utility; replicate.py and replicate_structure.py are source-maintenance/structure targets whose actual behavior is unknown until inspected. Keep originals and generated text distinguishable; no code behavior is inferred from a filename.

**Numerical exercise to author.** ILLUSTRATIVE inputs: 12 module containers and 13 numbered cheat topics; reconcile a topic ledger with the visible tree without treating a README or a second collation of a topic as an additional completed lesson.

**Alternatives and failure diagnostics.** Compare canonical topic records with duplicate collated renderings; diagnose stale source indexes, duplicate counts and accidental publication of generated material as measured evidence.

**Bounded case-study decision.** Choose the canonical source-of-truth and import boundary for a future CUDA authoring batch.

**Proposed validation; no result measured.** Plan inventory-to-authoring reconciliation, source hashes/version records when bodies become available, and link/status review; unknown code and exercise directories remain OPEN, not silently populated.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `CUDA` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/CUDA_Cheat_Sheets` | Source scope/container — not a completed lesson | `source-container` |
| `CUDA/CUDA_Cheat_Sheets/00_INDEX.md` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `CUDA/all_content.txt` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `CUDA/collate_md.py` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `CUDA/README.md` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `CUDA/replicate.py` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `CUDA/replicate_structure.py` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |

<a id="ft-source-artifacts"></a>

### Fine-tuning source provenance, hidden directories and contribution artifacts

**Status:** planned. **Existing destination:** [04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Learning outcome.** Separate visible training topics from navigation, all_content collation, replication/contribution artifacts and unseen collections.

**Focused sheets, derivations and mechanisms.** Supporting-artifact briefs: INDEX/00_INDEX are navigation; all_content.txt is a collation artifact; replicate_structure.py is a structure-maintenance study target, not proven executable training code; OSS_Contribution_Roadmap_1Month.md is a contribution-planning artifact, not a model-training method. The finetuning_course and cheat-family containers scope the topic ladder. Finetune_LLM and Tutorials-overall1 have hidden contents, so no lessons or exact filenames can be inferred.

**Numerical exercise to author.** ILLUSTRATIVE inputs: a hypothetical four-week contribution plan with 5 author-days/week and 2 review-days/week; calculate 12 implementation days after review, while clearly separating this planning exercise from training compute or completed contributions.

**Alternatives and failure diagnostics.** Compare source-artifact inventory with technical-topic inventory; diagnose treating generated collations, contribution plans or invisible folders as completed coverage.

**Bounded case-study decision.** Choose an import/authoring boundary and contribution slice based only on visible evidence.

**Proposed validation; no result measured.** Plan source/body access before promoting hidden-folder coverage; retain unresolved source rows and exact clipped path strings, then reconcile against a future authoritative listing rather than guessing suffixes.

**Exact source coverage and visible focused-topic titles:**

| Exact source node | Role / proposed focused-sheet topic | Disposition |
|---|---|---|
| `Finetune_LLM` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `finetuning_course` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/LLM_Finetuning_Cheat_Sheets` | Source scope/container — not a completed lesson | `source-container` |
| `finetuning_course/LLM_Finetuning_Cheat_Sheets/00_INDEX.md` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `finetuning_course/Tutorials-overall1` | OPEN: directory contents not visible; no hidden topics invented | `unresolved-source` |
| `finetuning_course/all_content.txt` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `finetuning_course/INDEX.md` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `finetuning_course/OSS_Contribution_Roadmap_1Month.md` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |
| `finetuning_course/replicate_structure.py` | Navigation, collation or maintenance artifact — see artifact brief | `supporting-artifact` |

## Unresolved-source and provenance boundary

The exact registers above retain every uncertain node. CUDA has invisible MPI_Basics/Profiling_Tools directories and invisible code/exercises directories; these are obligations to inspect future source bodies, not invented implementations or completed exercises. `Finetune_LLM` and `finetuning_course/Tutorials-overall1` have no visible contents. Fine-tuning clipped filenames remain exact inventory strings, including their ellipsis and `[TRUNCATED/UNVERIFIED]` marker: activation memory/sequence length, activation recomputation/gradient checkpointing, 5D configuration selection, reasoning/process supervision and QLoRA/quantization-aware fine-tuning are teachable only to their visible topic boundary. No missing suffix or exact source filename is reconstructed.

The source research-frontier year labels (CV/VLM 2025, distributed training 2025, speech 2025–2026) describe visible historical source targets, not a claim of present completeness. CUTLASS-like source walkthrough names and `flash.cu` likewise require the actual repository/revision and file provenance before asserting their internals. The supplied visible tree has no hidden lesson content, precise hardware support matrix or measured performance artifacts to transfer.

**Publication boundary:** resolving source uncertainty, writing full lessons/cases, executing numerical checks and validation protocols, completing issue entries and collecting primary citations remain authoring work. This plan intentionally does not fabricate those deliverables; it specifies what each visible source topic must become and where it belongs in the existing course.
