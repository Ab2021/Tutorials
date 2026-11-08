# Module 20 â€” RAG, Context Engineering and Enterprise Knowledge

> **Part:** [Part 04 â€” Deep Learning, Generative AI and Multimodal Systems](../INDEX.md)  
> **Prerequisites:** 06,14,18,19  
> **Purpose:** Build grounded knowledge systems using ingestion, retrieval, reranking, citations, permissions, and evaluation.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | document ingestion | planned |
| 2 | chunking embeddings and hybrid search | planned |
| 3 | reranking and context assembly | planned |
| 4 | grounding and citations | planned |
| 5 | permission-aware retrieval | [Permission-aware retrieval draft](Cheat_Sheets/01_Permission_Aware_Retrieval.md); broader topic planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [Permission-Aware Retrieval](Cheat_Sheets/01_Permission_Aware_Retrieval.md) — eligibility, provenance and authorization failure containment; visible source inventory only.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR003 | Authorized retrieval; effective policy versions; claim-level citations; embedding migrations | When should a support assistant answer, abstain or escalate a policy question? | Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels |
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | Which chunking and index design retrieves every required authorized span within the service budget? | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency |
| PLAN-2026-IR013 | Effective intervals; recorded-time history; jurisdiction rules; hybrid retrieval; reranking and claim verification | Which sources are eligible before ranking, and when is legal-owner review required? | Boundary-date and amendment replays, eligible recall, wrong-version citations and expert-reviewed abstentions |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [legal contract analysis](../../00_SOURCE_COVERAGE_AGENTS.md#scenario-08) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [LangChain RAG retrieval pipelines](../../00_SOURCE_COVERAGE_AGENTS.md#lang-rag) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
