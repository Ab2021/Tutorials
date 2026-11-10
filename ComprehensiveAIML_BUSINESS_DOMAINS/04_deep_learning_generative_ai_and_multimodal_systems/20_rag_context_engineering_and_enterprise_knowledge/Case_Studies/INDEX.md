# Module 20 Case Studies — RAG, Context Engineering and Enterprise Knowledge

## 1. Collection metadata

This draft collection retains two original decision cases and three reference-informed decision
briefs. A case is canonical only in its named module home; a shared brief must not be duplicated.
All listed systems, workloads, and outcomes remain proposed and not measured. Visible source
titles are inventory evidence only; their original bodies were not inspected.

| Field | Value |
|---|---|
| Scope | Grounded, authorised enterprise knowledge decisions |
| Editorial state | Draft collection; no completed local case manuscript |
| Evidence / implementation boundary | not-measured / proposed |
| Review trigger | New primary evidence, policy-version contract, or completed case manuscript |

## 2. Case catalog

| Case ID | Linked title / canonical home | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-20-01 | Policy assistant with citations — `01_policy_assistant_with_citations.md` (planned) | Enterprise policy support | llm-agentic | Answer, abstain, or escalate a policy question | `eligibility-before-ranking` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | never | No manuscript, deployment, or outcome |
| CASE-20-02 | Knowledge-base migration evaluation — `02_knowledge_base_migration_evaluation.md` (planned) | Knowledge-system migration | analytical | Whether a migrated corpus preserves authorised answer evidence | `materialized-view` | `evaluation-mismatch`, `snapshot-consistency` | not-measured | proposed | never | No manuscript, deployment, or outcome |
| PLAN-2026-IR003 | Applicable-policy answer/abstain decision (planned) | Policy retrieval | llm-agentic | Whether an answer has authorised applicable support | `evaluation-mismatch` | `authorization-scope`, `materialized-view` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR012 | Chunk and index selection (planned) | Retrieval infrastructure | llm-agentic | Which design retrieves all required authorised spans | `materialized-view` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR013 | Effective-date source eligibility (planned) | Time-aware retrieval | llm-agentic | Which sources are eligible before ranking | `eligibility-before-ranking` | `snapshot-consistency`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 | Planning brief only |

## 3. Canonical reusable patterns

| Pattern | Meaning | Hard boundary |
|---|---|---|
| `eligibility-before-ranking` | Filter by authority, permission, jurisdiction, and effective-time rules before scoring. | A high score cannot make an ineligible source valid. |
| `authorization-scope` | Retrieval and displayed evidence are constrained by the requester and purpose. | A citation must not disclose an unauthorised span. |
| `materialized-view` | Index state is a derived view with version, deletion, and freshness obligations. | Index presence does not prove source currency. |
| `snapshot-consistency` | Evaluation and answer construction need a declared source/version time. | A mixed-time answer cannot claim one policy state. |
| `evaluation-mismatch` | Retrieval metrics do not establish grounded answer correctness. | Test required-span coverage and human review outcomes. |

## 4. Similarity and contrast map

| Item | Shared reasoning | Distinct validation |
|---|---|---|
| CASE-20-01 | An answer needs authorised evidence. | Validate abstention and citation applicability at question time. |
| CASE-20-02 | Derived retrieval state can lose source evidence. | Compare migration snapshots, deletions, and source-to-chunk lineage. |
| PLAN-2026-IR012 | Chunking changes evidence recall. | Use all-required-span success, not only top-k relevance. |
| PLAN-2026-IR013 | Eligibility precedes ranking. | Replay boundary dates and amendments with maintained metadata. |

## 5. Questions and future-case backlog

1. Define the requester, policy effective time, jurisdiction, and decision authority before selecting a retrieval metric.
2. Specify an exact filtered baseline and an all-required-span oracle before selecting an approximate index.
3. Test deletion, permission-revocation, and policy-amendment propagation as correctness transitions.
4. Separate citation presence from entailment, applicability, and authorised disclosure.
5. Do not make legal or compliance determinations from a proposed retrieval design; route those to the authorised owner.

## 6. Review and retirement log

| Item | State | Required action |
|---|---|---|
| CASE-20-01 | Planned | Write the canonical case and independently check its eligibility/time worked trace. |
| CASE-20-02 | Planned | Write the canonical case and independently check its migration reconciliation example. |
| PLAN-2026-IR003/012/013 | Planning only | Preserve source-access limits and the planned/non-measured boundary. |
| Collection | Draft | Reconcile links and catalog fields after each canonical manuscript is complete. |

## 7. Pre-publish checks

- [ ] Each real case has all 15 ordered sections, a bounded owner and baseline, and Section 15 last.
- [ ] Claims distinguish illustrative, derived, sourced, and not-measured evidence.
- [ ] Time, permission, version, and deletion semantics are explicit in data and evaluation contracts.
- [ ] Quantitative examples include units, recalculation, and an invalidating boundary.
- [ ] Interview packs contain 12–18 checked answers; catalog links point only to real canonical files.
