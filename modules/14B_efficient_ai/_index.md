# Module 14B — Efficient AI and On-Device Inference

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
> **Folder:** `modules/14B_efficient_ai/`
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
| Prerequisites | 07, 14A |
| Unlocks | 15, 24 |
| Assessment | Efficient-model dossier (Week 25) |
| Survey file | `14B_efficient_ai_distillation_quantization_on_device_inference.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | `01_the-deployment-pareto-frontier.md` | The Deployment Pareto Frontier | Accuracy, calibration, latency, memory, energy, and cost as joint constraints. | 1,200-1,800 | planned |
| 2 | `02_target-hardware-and-performance.md` | Target Hardware and Performance | Profiling, roofline reasoning, and what each accelerator can actually do. | 1,200-1,800 | planned |
| 3 | `03_the-compression-family.md` | The Compression Family | Distillation, quantization, pruning, low-rank, and when to use each. | 1,200-1,800 | planned |
| 4 | `04_knowledge-distillation.md` | Knowledge Distillation | Soft targets, temperature, feature matching, and distillation failures. | 1,200-1,800 | planned |
| 5 | `05_quantization.md` | Quantization | Affine and symmetric schemes, PTQ, QAT, calibration, and STE. | 1,200-1,800 | planned |
| 6 | `06_pruning-and-sparsity.md` | Pruning and Sparsity | Structured versus unstructured, lottery tickets, and hardware support. | 1,200-1,800 | planned |
| 7 | `07_inference-optimization.md` | Inference Optimization | Graph fusion, operator fallback, KV cache, and batching. | 1,200-1,800 | planned |
| 8 | `08_on-device-architecture.md` | On-Device Architecture | Runtimes, model formats, and the deployment pipeline. | 1,200-1,800 | planned |
| 9 | `09_medical-and-regulatory-constraints.md` | Medical and Regulatory Constraints | Change control, PCCP, and what compression does to a cleared claim. | 1,200-1,800 | planned |
| 10 | `10_validation-on-target-hardware.md` | Validation on Target Hardware | p99 latency, energy per inference, thermal throttling, and HIL testing. | 1,200-1,800 | planned |
| 11 | `11_economics-of-compression.md` | Economics of Compression | Cloud cost, device BOM, battery life, and the break-even calculation. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Deployment Pareto Frontier ->
02 Target Hardware and Performance ->
03 The Compression Family ->
04 Knowledge Distillation ->
05 Quantization ->
06 Pruning and Sparsity ->
07 Inference Optimization ->
08 On-Device Architecture ->
09 Medical and Regulatory Constraints ->
10 Validation on Target Hardware ->
11 Economics of Compression
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 11 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

