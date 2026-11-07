# NLP and LLM Playbook — Cheat Sheets

**Status:** The RAG sheet is a draft and unreviewed; other references remain planned. Follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked cheat-sheet master. A card does not replace the assumptions, evidence and evaluation in a full case.

## 1. Overview

This collection helps practitioners design and evaluate NLP and language-model systems. The current draft focuses on retrieval-augmented generation (RAG), evidence selection, authorization and diagnostics. Readiness means calculating a retrieval metric and context budget, distinguishing retrieval failure from generation failure, and respecting access boundaries before retrieval results reach a model or user.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [RAG retrieval, chunking and evidence](13_RAG_Retrieval_Augmented_Generation.md) | Reason about authorized retrieval, context selection and grounded evaluation; requires tokenization, vectors, ranking metrics and access control | Draft; unreviewed |
| Planned | Prompt and schema guide | State output constraints and validation boundaries | Not written |
| Planned | LLM production incident checklist | Diagnose and contain failures across the model and retrieval pipeline | Not written |
| Planned | Common Issues | Distinct collection-wide diagnostics | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Did retrieval find relevant evidence? | [Worked example](13_RAG_Retrieval_Augmented_Generation.md#5-worked-numerical-example) | Define eligible relevant items, rank cutoff and denominator; missing labels can bias the result. |
| How much evidence fits in context? | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) | Reserve instruction, query and output space; token counts depend on the actual tokenizer. |
| Why does a response lack support? | [Failure modes](13_RAG_Retrieval_Augmented_Generation.md#8-failure-modes--edge-cases) | Check authorization, indexing, retrieval, evidence selection and generation separately. |

## 4. Reading paths

- **Beginner:** Review tokenization, vectors, nearest-neighbor retrieval and document access control, then read the draft's definitions. Prompt and schema fundamentals remain planned.
- **Practitioner:** Read the complete RAG draft; reproduce the retrieval and token-budget arithmetic, then define evaluation and authorization for a concrete corpus.
- **Advanced:** Test index freshness, permission changes, duplicate and conflicting sources, ranking tradeoffs, unsupported answers and end-to-end cost. The incident reference remains planned.

## 5. Prerequisite skills

Understand tokenization, embedding similarity and ranking, precision and recall at a specified cutoff, document/version identity, access control, and basic query/response service contracts. Interpret quality metrics only for their defined query and relevance population.

## 6. What this collection does not cover

The RAG draft does not establish factual correctness, source authorization or production quality for an unspecified deployment. It is not a complete NLP or LLM curriculum, vendor implementation guide, prompt/schema handbook or finished Common Issues reference.

## 7. Master symbol glossary

| Symbol | Meaning and units | Scope / warning | Reference |
|---|---|---|---|
| R, H, n | Eligible relevant items, retrieved relevant items, and items returned; counts | Define eligibility and the evaluation cutoff; guard zero denominators | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) |
| L, C, O | Document length, chunk length and overlap; tokens | Tokenizer and chunking rules must match the implementation | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) |
| W, B | Total context window and evidence budget; tokens | Reserve other prompt and output components | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) |
| K | Rank-fusion constant | Distinct from top-k retrieval cutoff | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) |
| D, V, b | Chunk count, vector dimension and bytes per component | Vector payload excludes index and metadata overhead | [Core formulas](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [RAG retrieval and evidence](13_RAG_Retrieval_Augmented_Generation.md#12-quick-reference-card) | Measure retrieval and grounded response quality against an eligible evidence set. | Authorization, relevance labels and the evaluation population must be defined. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no collection-wide issue count is claimed. Consult the draft's [failure modes](13_RAG_Retrieval_Augmented_Generation.md#8-failure-modes--edge-cases) for retrieval, evidence and access-boundary diagnostics.

## 10. Sources, versions and maintenance

The visible inventory names `DS_Domain/NLP_LLM_Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md`, but the original file was unavailable; its contents are not verified. The [draft source registry](13_RAG_Retrieval_Augmented_Generation.md) identifies references whose inspection remains open. Review on changes to the corpus, access policy, tokenizer, model, index, retrieval pipeline or evaluation population.

## 11. Collection acceptance checklist

- [ ] Planned references and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet retains all 12 sections, checked calculations or traces, failure diagnostics, answered interviews and a final card.
- [ ] Authorization, evidence provenance, source limitations and evaluation population are explicit.
- [ ] Links, glossary, reading paths, issue locations and editorial statuses match actual files.
