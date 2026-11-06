# NLP and LLM Playbook - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in NLP and LLM Playbook. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Prompt and schema guide | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | [RAG retrieval, chunking and evidence](13_RAG_Retrieval_Augmented_Generation.md) | Authorized retrieval, context selection, evaluation and diagnostics; source: `DS_Domain/NLP_LLM_Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md`. | draft; unreviewed |
| 3 | LLM production incident checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

The RAG sheet is a draft, not an independently reviewed or published reference. The other references and the collection-wide Common Issues artifact remain planned; the index still needs the full collection structure before publication. Its card is in [Section 12](13_RAG_Retrieval_Augmented_Generation.md#12-quick-reference-card). Prerequisites are tokenization, vectors, ranking metrics and access control; the sheet's source registry identifies references whose inspection remains open.

RAG notation: `R` is the count of eligible relevant items, `H` is the count of retrieved relevant items, and `n` is the count returned; all are nonnegative integer counts. `L`, `C` and `O` are document length, chunk length and overlap in tokens. `W` and `B` are total window and available evidence budget in tokens. `K` is the nonnegative rank-fusion constant, distinct from top-`k`. `D`, `V` and `b` are chunk count, vector dimension and bytes per component, respectively. Definitions and constraints are in the [sheet](13_RAG_Retrieval_Augmented_Generation.md#4-core-formulas-with-full-derivation).

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.