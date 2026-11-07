# AI Platforms, Data Systems and Production Engineering — Cheat Sheets

**Status:** The authored checkpoint sheet is a draft and unreviewed; remaining reference families are planned. Apply the [course authoring and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master. A title or reference card is not a completed teaching file.

## 1. Overview

This collection serves platform and ML infrastructure engineers choosing durable, measurable production systems. Its current draft addresses training recovery; other platform areas remain planned. Readiness means identifying correctness and resource assumptions, tracing failures, and testing a full recovery against the deployed stack.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Advanced | [Distributed training checkpoint and recovery](01_Distributed_Training_Checkpoint_and_Recovery.md) | Design and restore a consistent committed generation; requires training, optimizer and distributed-rank fundamentals | Draft; unreviewed |
| Planned | GPU architecture and accelerators | GPU performance model, precision guide, hardware sizing | Not written here |
| Planned | Distributed training and data pipelines | Parallelism selection and throughput checklist | Not written here |
| Planned | Model serving and inference optimization | Inference optimization, service-level objectives, capacity formulas | Not written here |
| Planned | MLOps and lifecycle | Reference architecture, release checklist, model-card template | Not written here |
| Planned | Data quality and reliability | Data observability, ML SLOs, incident template | Not written here |
| Planned | AI evaluation and monitoring | Evaluation matrix, human-rater guide, metric tree | Not written here |
| Planned | API platforms and FinOps | API review, Kubernetes AI workloads, cost metrics | Not written here |
| Planned | Common Issues | Cross-topic, independently diagnosable issue locator | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Is a set of shards restartable? | [Checkpoint mechanisms](01_Distributed_Training_Checkpoint_and_Recovery.md#4-core-formulas-with-full-derivation) | Require a consistent, durable and validated published generation, not a directory listing. |
| How do I estimate checkpoint time? | [Worked example](01_Distributed_Training_Checkpoint_and_Recovery.md#5-worked-numerical-example) | Payload divided by effective throughput excludes fixed and contention costs. |
| Why did recovery diverge? | [Failure modes](01_Distributed_Training_Checkpoint_and_Recovery.md#8-failure-modes--edge-cases) | Inspect state, counters, cursor, RNG and framework/version compatibility. |

## 4. Reading paths

- **Beginner:** Learn optimizer steps, gradient accumulation, distributed ranks and durable storage first; the foundational platform sheets remain planned. Then read the checkpoint mental model and definitions.
- **Practitioner:** Read the complete checkpoint sheet and design a restore drill for the actual storage and training stack.
- **Advanced:** Trace multi-rank publication and fencing, elastic resharding, input progress and partial failure; calculate recovery and capacity under measured conditions. The planned platform references will extend this path.

## 5. Prerequisite skills

Understand optimizer state, stochastic training and random-number-generator state, data-loader progress, distributed synchronization, throughput units, and the actual storage system's durability and publication semantics. A successful write is not automatically an atomic committed generation.

## 6. What this collection does not cover

The checkpoint draft is not a framework-independent implementation recipe, benchmark, incident report or guarantee of identical numerical trajectory after an elastic restart. Other listed platform areas and the final Common Issues file remain planned in this collection; consult relevant module indexes for separate drafts.

## 7. Master symbol glossary

| Symbol | Meaning and units | Scope / warning | Reference |
|---|---|---|---|
| C | Non-overlapped checkpoint pause, seconds | Assumed constant in the interval approximation | [Checkpoint formula](01_Distributed_Training_Checkpoint_and_Recovery.md#4-core-formulas-with-full-derivation) |
| λ | Failure rate, inverse seconds | Stationary Poisson approximation is not a correlated-outage model | [Checkpoint formula](01_Distributed_Training_Checkpoint_and_Recovery.md#4-core-formulas-with-full-derivation) |
| S, B | Unique checkpoint payload in GiB; effective aggregate bandwidth in GiB/s | Actual throughput and overhead require measurement | [Worked example](01_Distributed_Training_Checkpoint_and_Recovery.md#5-worked-numerical-example) |
| RPO / RTO | Recovery point / recovery time objectives | An objective is not proof of an achieved worst-case guarantee | [Definitions](01_Distributed_Training_Checkpoint_and_Recovery.md#3-foundational-definitions) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Checkpoint and recovery](01_Distributed_Training_Checkpoint_and_Recovery.md#12-quick-reference-card) | Transfer time is unique bytes divided by effective throughput. | Publish only a complete consistent generation under actual backend durability and fencing semantics. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no completed collection-wide issue count is claimed. The draft's [failure modes and edge cases](01_Distributed_Training_Checkpoint_and_Recovery.md#8-failure-modes--edge-cases) give symptoms, discriminators, bounded response and verification for checkpoint recovery.

## 10. Sources, versions and maintenance

The checkpoint draft is illustrative and unreviewed. The original inventoried fault-tolerance and checkpoint lessons are unavailable in this workspace; their contents were not verified. Check the installed release's [PyTorch distributed checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) and storage provider semantics before implementation; review on framework, topology, storage or source availability changes. The [distributed-training module](../24_distributed_training_data_pipelines_and_ml_compute/INDEX.md) retains broader planned work.

## 11. Collection acceptance checklist

- [ ] All planned platform topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each authored file retains all 12 sections, checked calculations or state traces, diagnostic failures, answered interview questions and its final card.
- [ ] Empirical claims are attributed; product versions, open evidence gaps and review triggers are documented.
- [ ] Links, learning paths, glossary, issue locations and statuses are reconciled against completed files.
