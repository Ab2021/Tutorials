# Enterprise Knowledge, Support and Sales - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in Enterprise Knowledge, Support and Sales. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Enterprise RAG controls | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | Support quality rubric | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 3 | CRM integration checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR003 | Authorized retrieval; effective policy versions; claim-level citations; embedding migrations | Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels. Boundary: Citations and entailment checks require validation; never transfer the example refund window into a new business. | [REF-IR003](../../../../../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md) | planned; not written |
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix. Boundary: Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements. | [REF-IR004](../../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | planned; not written |
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency. Boundary: Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements. | [REF-IR012](../../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | planned; not written |
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability. Boundary: Human approval and source maintenance remain necessary; the example word target is not a compliance standard. | [REF-IR015](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
