# Module 22 â€” Multimodal Models, Video, Voice and World Models

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 17,18,19  
> **Purpose:** Combine visual, textual, audio, temporal, and action data for perception-rich applications while measuring cross-modal reliability.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | VLM architectures | planned |
| 2 | video understanding | planned |
| 3 | voice agents | planned |
| 4 | vision-language-action | planned |
| 5 | world models and limitations | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Which claim fields can be extracted reliably, and which bundles require adjuster review? | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [P&C computer vision, medical imaging, clinical vision and multimodal documents](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#deep-vision) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 12 — VLM, code and tool-use training](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-specialized) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 18 — evaluation, production and research boundaries](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-vlm-eval-deploy) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 18 — VLM architecture, data and training](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-vlm-training) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 18 — OCR, charts, multimodal preferences and video](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-vlm-docs-video) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
