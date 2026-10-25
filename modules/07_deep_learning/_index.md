# Module 07 — Deep Learning, Multimodal Models, and LLMs

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/07_deep_learning_multimodal_llms.md`](../../course/07_deep_learning_multimodal_llms.md)
> **Folder:** `modules/07_deep_learning/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06 |
| Unlocks | 09, 10, 11, 14B |
| Assessment | Foundation-model evaluation dossier (Week 14) |
| Survey file | `07_deep_learning_multimodal_llms.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_neural-network-fundamentals.md`](01_neural-network-fundamentals.md) | Neural Network Fundamentals | Layers, activations, loss functions, backpropagation, and optimization. | 1,200-1,800 | complete |
| 2 | [`02_training-dynamics.md`](02_training-dynamics.md) | Training Dynamics | Initialization, learning rates, regularization, overfitting, and debugging. | 1,200-1,800 | complete |
| 3 | [`03_cnns-and-medical-vision.md`](03_cnns-and-medical-vision.md) | CNNs and Medical Vision | Convolutions, receptive fields, transfer learning, and medical imaging specifics. | 1,200-1,800 | complete |
| 4 | [`04_sequence-models.md`](04_sequence-models.md) | Sequence Models | RNNs, temporal convolution, and attention for signals and text. | 1,200-1,800 | complete |
| 5 | [`05_transformers.md`](05_transformers.md) | Transformers | Self-attention, positional encoding, scaling, and compute cost. | 1,200-1,800 | complete |
| 6 | [`06_medical-foundation-models.md`](06_medical-foundation-models.md) | Medical Foundation Models | Pretraining corpora, adaptation, and the evidence needed before clinical use. | 1,200-1,800 | complete |
| 7 | [`07_llm-basics.md`](07_llm-basics.md) | LLM Basics | Tokenization, next-token prediction, instruction tuning, and decoding. | 1,200-1,800 | complete |
| 8 | [`08_prompting-and-rag.md`](08_prompting-and-rag.md) | Prompting and Retrieval-Augmented Generation | Prompt design, retrieval, grounding, and citation of sources. | 1,200-1,800 | complete |
| 9 | [`09_agents-and-tools.md`](09_agents-and-tools.md) | Agents and Tools | Tool use, planning, memory, and the failure modes of autonomous clinical agents. | 1,200-1,800 | complete |
| 10 | [`10_evaluating-generative-models.md`](10_evaluating-generative-models.md) | Evaluating Generative Models | Task metrics, human review, hallucination rates, and red teaming. | 1,200-1,800 | complete |
| 11 | [`11_compute-latency-and-cost.md`](11_compute-latency-and-cost.md) | Compute, Latency, and Cost | GPU economics, context length, batching, and serving trade-offs. | 1,200-1,800 | complete |
| 12 | [`12_multi-agent-orchestration.md`](12_multi-agent-orchestration.md) | Multi-Agent Orchestration | Stateful multi-agent graphs, LangGraph, agent memory tiers, and prior auth workflows. | 1,200-1,800 | complete |
| 13 | [`13_agentic-ai-safety-and-guardrails.md`](13_agentic-ai-safety-and-guardrails.md) | Agentic AI Safety and Guardrails | Clinical red-teaming, indirect prompt injection defense, and post-market discrepancy monitoring. | 1,200-1,800 | complete |
| 14 | [`14_multimodal-fusion-architectures.md`](14_multimodal-fusion-architectures.md) | Multimodal Fusion Architectures | Cross-attention fusion, modality dropout, BioViL-T, and missing-modality robustness. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Neural Network Fundamentals ->
02 Training Dynamics ->
03 CNNs and Medical Vision ->
04 Sequence Models ->
05 Transformers ->
06 Medical Foundation Models ->
07 LLM Basics ->
08 Prompting and Retrieval-Augmented Generation ->
09 Agents and Tools ->
10 Evaluating Generative Models ->
11 Compute, Latency, and Cost
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 0 |
| complete | 11 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

