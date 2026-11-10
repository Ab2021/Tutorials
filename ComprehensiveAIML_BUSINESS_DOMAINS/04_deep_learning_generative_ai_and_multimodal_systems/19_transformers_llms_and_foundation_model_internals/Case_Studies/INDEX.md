# Module 19 Case Studies — Transformers, LLMs and Foundation-Model Internals

## 1. Collection metadata

This is a draft catalog for bounded foundation-model decisions. It retains the two original
case threads and the IR007/IR009 planning commitments without claiming a run, organisation,
deployment, benchmark, or measured result. Visible source names identify syllabus inventory;
they do not establish access to the original lesson bodies.

| Field | Value |
|---|---|
| Collection state | Draft; no completed local case manuscript |
| Evidence / implementation state | not-measured / proposed |
| Scope | Model choice, context-quality diagnosis, inference admission, and training-budget allocation |
| Review trigger | Completed canonical case, source access, or changed model/serving assumption |

## 2. Case catalog

| Case ID | Linked title / canonical home | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-19-01 | Choose a model for code assistance — `01_code_assistance_model_choice.md` (planned) | Developer tooling | llm-agentic | Which model and retrieval boundary should be evaluated against a manual baseline | `evaluation-mismatch` | `authorization-scope`, `cost-latency-tradeoff` | not-measured | proposed | never | No manuscript or measured comparison |
| CASE-19-02 | Diagnose long-context quality loss — `02_long_context_quality_diagnosis.md` (planned) | Long-context evaluation | analytical | Whether quality loss is context, retrieval, position, or evaluation-contract failure | `evaluation-mismatch` | `distribution-shift`, `resource-admission` | not-measured | proposed | never | No manuscript or measured diagnosis |
| PLAN-2026-IR007 | Interactive inference latency and admitted throughput (planned) | LLM inference | hybrid | Which scheduling and KV-memory policy meets isolated admitted load | `resource-admission` | `cost-latency-tradeoff`, `authorization-scope` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR009 | Domain-training budget and restart headroom (planned shared brief) | Model training | hybrid | Whether continued training or from-scratch training is justified | `cost-latency-tradeoff` | `data-leakage`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 | Planning brief only |

## 3. Canonical reusable patterns

| Pattern | Meaning | Boundary |
|---|---|---|
| `evaluation-mismatch` | A token metric or synthetic benchmark can disagree with the real task contract. | Validate the specified task, slice, and comparator. |
| `resource-admission` | Requests are accepted only when bounded cache, latency, and isolation resources exist. | Offered load is not admitted throughput. |
| `cost-latency-tradeoff` | A faster or larger alternative changes cost, quality, and tail behaviour together. | A single average hides tail and restart costs. |
| `authorization-scope` | Code/context input must be authorised at retrieval and serving boundaries. | Better completion does not justify exposure. |
| `atomic-release-bundle` | Model, tokenizer, prompt/configuration, and evaluation evidence must move together. | A weight-only rollback may not restore behaviour. |

## 4. Similarity and contrast map

| Item | Shared reasoning | Distinct check |
|---|---|---|
| CASE-19-01 | Model selection needs a task-true baseline. | Repository/time split and authorised-context contract. |
| CASE-19-02 | Context behaviour requires position-aware evidence. | Separate missing evidence from position, truncation, and retrieval errors. |
| PLAN-2026-IR007 | Serving choice changes task latency. | Measure prefill, decode, admission, cache headroom, and target equivalence. |
| PLAN-2026-IR009 | Quality/cost claims require a complete budget. | Include held-out general quality, restart, and reproducibility gates. |

## 5. Questions and future-case backlog

1. Define units for token, request, sequence, cache byte, prefill time, and decode time before comparing alternatives.
2. Test a no-retrieval, authorised-retrieval, and abstain baseline before attributing code-task quality to a model.
3. Make effective context length a task-specific measured curve rather than a nominal context-window claim.
4. Reject scaling extrapolations outside their fitted data, compute, and objective regime.
5. Treat speculative decoding as a correctness-preserving algorithm only if its acceptance and target-equivalence contracts are tested.

## 6. Review and retirement log

| Item | State | Required action |
|---|---|---|
| CASE-19-01 | Planned | Author canonical case with independent task, security, quality, cost, and latency evidence. |
| CASE-19-02 | Planned | Author canonical case with a checked position/retrieval diagnostic trace. |
| PLAN-2026-IR007/009 | Planning only | Preserve source-reference limits and do not transfer assumed reference workload figures. |
| Collection | Draft | Reconcile catalog links, claim ledgers, and review state after each manuscript. |

## 7. Pre-publish checks

- [ ] Each canonical case has all 15 ordered sections and a final Section 15 card.
- [ ] The decision owner, baseline, data cut-off, model/tokenizer/configuration versions, and authority boundaries are explicit.
- [ ] Numerical memory, throughput, or budget examples are recomputed with units and invalidating boundaries.
- [ ] Each case includes 12–18 substantive answered interview questions and failure recovery paths.
- [ ] Catalog links target only real canonical manuscripts; shared decision families are linked, not duplicated.
