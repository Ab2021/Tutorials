# Module 09 — Clinical NLP

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
> **Folder:** `modules/09_clinical_nlp/`
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
| Prerequisites | 07 |
| Unlocks | 10, 18 |
| Assessment | Clinical NLP validation report (Week 16) |
| Survey file | `09_clinical_nlp.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_clinical-text-sources.md`](01_clinical-text-sources.md) | Clinical Text Sources | Notes, reports, messages, and forms; structure and provenance. | 1,200-1,800 | complete |
| 2 | [`02_documentation-behaviour.md`](02_documentation-behaviour.md) | Documentation Behaviour | Copy-forward, templates, late entries, and what they do to text data. | 1,200-1,800 | complete |
| 3 | [`03_task-taxonomy.md`](03_task-taxonomy.md) | Task Taxonomy | Extraction, assertion, classification, linking, summarization, and question answering. | 1,200-1,800 | complete |
| 4 | [`04_classical-methods.md`](04_classical-methods.md) | Classical Methods | Rules, dictionaries, regex, and statistical models that still win in production. | 1,200-1,800 | complete |
| 5 | [`05_transformers-for-clinical-text.md`](05_transformers-for-clinical-text.md) | Transformers for Clinical Text | Domain pretraining, long documents, and efficient inference. | 1,200-1,800 | complete |
| 6 | [`06_annotation.md`](06_annotation.md) | Annotation | Guidelines, adjudication, inter-annotator agreement, and annotation cost. | 1,200-1,800 | complete |
| 7 | [`07_evaluation.md`](07_evaluation.md) | Evaluation | Span-level metrics, assertion accuracy, and clinician review designs. | 1,200-1,800 | complete |
| 8 | [`08_deidentification.md`](08_deidentification.md) | De-identification | PHI categories, automated de-identification, and residual risk measurement. | 1,200-1,800 | complete |
| 9 | [`09_summarization-and-grounding.md`](09_summarization-and-grounding.md) | Summarization and Grounding | Abstractive summarization, faithful citation, and hallucination detection. | 1,200-1,800 | complete |
| 10 | [`10_llm-safety-in-clinical-text.md`](10_llm-safety-in-clinical-text.md) | LLM Safety in Clinical Text | Prompt injection, leakage, unsafe advice, and human-in-the-loop design. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Clinical Text Sources ->
02 Documentation Behaviour ->
03 Task Taxonomy ->
04 Classical Methods ->
05 Transformers for Clinical Text ->
06 Annotation ->
07 Evaluation ->
08 De-identification ->
09 Summarization and Grounding ->
10 LLM Safety in Clinical Text
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
| complete | 10 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

