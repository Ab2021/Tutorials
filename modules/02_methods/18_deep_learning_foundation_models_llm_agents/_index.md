# Module 18 — Deep Learning, Foundation Models, and LLM Agents

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/18_deep_learning_foundation_models_llm_agents.md`](../../../course/18_deep_learning_foundation_models_llm_agents.md)
> **Folder:** `modules/02_methods/18_deep_learning_foundation_models_llm_agents/`
> **Version:** 2026.5

---

## 1. Module purpose

This module places neural networks, time-series foundation models, LLMs, RAG, and agents in the supply chain stack and teaches how to evaluate them without benchmark illusion.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 07 |
| Unlocks | 19, 24 |
| Assessment | Foundation-model evaluation (Week 21) |
| Survey file | `18_deep_learning_foundation_models_llm_agents.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_neural-network-fundamentals.md`](01_neural-network-fundamentals.md) | Neural Network Fundamentals | Layers, activations, loss, regularisation, and what the model learns. | 1,200–1,800 | planned |
| 2 | [`02_sequence-architectures.md`](02_sequence-architectures.md) | Sequence Architectures | RNN, CNN, and transformer encoder patterns for time series. | 1,200–1,800 | planned |
| 3 | [`03_global-local-models.md`](03_global-local-models.md) | Global versus Local Models | Cross-learning across series and the data required to earn it. | 1,200–1,800 | planned |
| 4 | [`04_neural-forecasting-zoo.md`](04_neural-forecasting-zoo.md) | The Neural Forecasting Zoo | DeepAR, N-BEATS, N-HiTS, TFT, PatchTST, and when each is a candidate. | 1,200–1,800 | planned |
| 5 | [`05_ts-foundation-models.md`](05_ts-foundation-models.md) | Time-Series Foundation Models | Pretraining, zero-shot use, fine-tuning, and the claims vendors make. | 1,200–1,800 | planned |
| 6 | [`06_zero-shot-evaluation.md`](06_zero-shot-evaluation.md) | Evaluating Zero-Shot and Fine-Tuned Models | Local baselines, horizon effects, and cost per forecast. | 1,200–1,800 | planned |
| 7 | [`07_benchmark-contamination.md`](07_benchmark-contamination.md) | Benchmark Contamination and Leakage | Pretraining exposure and why a public score may not transfer. | 1,200–1,800 | planned |
| 8 | [`08_llm-extraction.md`](08_llm-extraction.md) | LLMs for Extraction and Supplier Communication | Contracts, emails, and documents with human verification. | 1,200–1,800 | planned |
| 9 | [`09_rag-agents.md`](09_rag-agents.md) | RAG over Contracts and Policies; Agents and Guardrails | Retrieval, tool use, planning, memory, and human approval. | 1,200–1,800 | planned |
| 10 | [`10_multimodal.md`](10_multimodal.md) | Multimodal: Shelf Images, Damage, and Video | Vision models for availability, compliance, and quality inspection. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Neural Network Fundamentals ->
02 Sequence Architectures ->
03 Global versus Local Models ->
04 The Neural Forecasting Zoo ->
05 Time-Series Foundation Models ->
06 Evaluating Zero-Shot and Fine-Tuned Models ->
07 Benchmark Contamination and Leakage ->
08 LLMs for Extraction and Supplier Communication ->
09 RAG over Contracts and Policies; Agents and Guardrails ->
10 Multimodal: Shelf Images, Damage, and Video
```

## 5. What this module does not cover

- It does not cover model training at scale or GPU infrastructure engineering.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`18_deep_learning_foundation_models_llm_agents.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
