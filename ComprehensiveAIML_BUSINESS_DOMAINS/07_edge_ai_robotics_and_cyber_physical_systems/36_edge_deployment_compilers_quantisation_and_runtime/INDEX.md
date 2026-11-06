# Module 36 â€” Edge Deployment, Compilers, Quantisation and Runtime

> **Part:** [Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems](../INDEX.md)  
> **Prerequisites:** 21,23,25,35  
> **Purpose:** Compile and optimise models for on-device execution using graph optimisation, runtimes, quantisation, and benchmarking.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | ONNX/TFLite/TensorRT runtimes | planned |
| 2 | PTQ and QAT | planned |
| 3 | compiler optimisation | planned |
| 4 | benchmarking | planned |
| 5 | OTA deployment | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [INT8/INT4, PTQ/QAT, per-channel scales, AWQ/SmoothQuant, structured/unstructured pruning, distillation, mixed precision and NAS](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-compression) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [TFLite Micro/CMSIS-NN, ONNX Runtime, TVM/AutoTVM/Ansor, llama.cpp/GGUF, Edge Impulse, model conversion and runtime gaps](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-runtimes) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [MLIR, TVM Relay/TIR, graph/loop optimization, microTVM/AOT, TOSA/IREE, RISC-V/RVV, quantization compilation and pass debugging](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-compilers) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.