# Module 24 Case Studies — Distributed Training, Data Pipelines and ML Compute

**Status:** planned catalog; no authored `CASE-*` files. Follow the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | distributed training, data pipelines, and ML compute |
| Evidence boundary | illustrative/proposed unless accessible primary evidence is cited |
| Review trigger | topology, dataset, checkpoint backend, scheduler, or cost-model change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-24-01 | Train a large model on limited cluster capacity | training | hybrid | select a feasible plan | resource-admission | cost-latency-tradeoff | not-measured | proposed | — | requires achieved utilization and quality evidence |
| PLAN-24-02 | Recover from checkpoint failures | training | incident | restore consistent progress | atomic-release-bundle | correctness-contract | not-measured | proposed | — | requires full recovery drill |
| PLAN-2026-IR001 | Gate retraining on corpus provenance and long-tail quality | data/training | hybrid | admit or quarantine documents | distribution-shift | data-leakage; evaluation-mismatch | not-measured | proposed | 2026-11-06 (planning) | classifier cannot establish authorship |
| PLAN-2026-IR009 | Allocate a domain-training budget with restart headroom | training | hybrid | scratch versus continued training | cost-latency-tradeoff | data-leakage; atomic-release-bundle | not-measured | proposed | 2026-11-06 (planning) | figures are assumptions, not legal advice |

## 3. Canonical reusable patterns

- **Atomic release bundle:** publish a complete, versioned checkpoint generation with compatible optimizer, RNG, data-cursor, and topology state.
- **Resource admission:** admit a run only when accelerator, storage, network, and recovery headroom fit the declared constraint.
- **Distribution shift:** audit corpus provenance and coverage before treating new data as training-eligible.

## 4. Similarity / contrast map

| Case | Similarity | Contrast / transfer boundary |
|---|---|---|
| PLAN-24-01 and IR009 | capacity, quality, and restart cost interact | a proxy sweep cannot establish achieved full-run utilization |
| PLAN-24-02 and IR001 | both require provenance and durable state | checkpoint consistency does not validate corpus quality |

## 5. Questions and future-case backlog

1. Which state must be atomically committed for a restart to be scientifically comparable?
2. What measured bandwidth, failure model, and interruption window changes the chosen checkpoint cadence?
3. Which corpus exclusion changes a held-out slice and why?
4. Which alternate parallelism layout is feasible under the same memory and network constraints?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no primary run artifacts or executed recovery drill |

## 7. Pre-publish checks

- [ ] Create a 15-section `CASE-*` file and retain its final card as Section 15.
- [ ] Check state ownership, bytes/rates, recovery arithmetic, quality comparator, and fault-replay path.
- [ ] Reconcile this catalog with actual sources, evidence labels, stage, and editorial state.
