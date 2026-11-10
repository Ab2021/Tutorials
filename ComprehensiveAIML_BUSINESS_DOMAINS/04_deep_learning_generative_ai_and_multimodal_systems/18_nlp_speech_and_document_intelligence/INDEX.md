# Module 18 â€” NLP, Speech and Document Intelligence

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 05,11  
> **Purpose:** Build language, voice, and document systems around retrieval, extraction, transcription, and human verification.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | text classification and NER | [authored](Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) |
| 2 | ASR and TTS | [authored](Cheat_Sheets/03_asr-transcription-evaluation.md) |
| 3 | document parsing and OCR | planned |
| 4 | embeddings | [authored](Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) |
| 5 | language evaluation | [authored](Cheat_Sheets/03_asr-transcription-evaluation.md) |
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
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | When do repository-aware completion or language adapters justify their added complexity? | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support |
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Which claim fields can be extracted reliably, and which bundles require adjuster review? | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Which customer objections can be reported with reliable speaker attribution and evidence? | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Should a speech adapter be promoted, and to which evaluated product and language domains? | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Speech-to-text/TTS in insurance and healthcare](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#deep-speech) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 20 / cheat sheet 12 — speech-LLM integration, evaluation and deployment](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-speech-production) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 16 — tokenizer training, curriculum and multilingual learning](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-tokenizers-multilingual) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 20 — speech foundations, ASR/STT, Whisper and streaming](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-speech-asr) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 20 — TTS, voice cloning and native-language training](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-speech-tts) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
