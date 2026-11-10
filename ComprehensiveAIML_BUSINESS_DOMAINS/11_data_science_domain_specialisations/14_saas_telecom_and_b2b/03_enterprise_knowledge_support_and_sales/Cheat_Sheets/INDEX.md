# Enterprise Knowledge, Support and Sales — Cheat Sheets

> **Collection state:** navigation and reuse register. This collection has no locally authored sheet yet. The linked canonical artifacts are external teaching homes, not evidence that the proposed enterprise deployment, tenant count, or results exist.

## 1. Overview
This collection supports a knowledge-platform owner deciding how to ingest, authorize, retrieve, generate, route, audit, and retire enterprise answers. It assumes basic search, probability, document access control, and event-time reasoning. Readiness means a learner can state an answer contract, trace evidence and permissions at request time, calculate retrieval measures, reject unsafe use, and specify a validation plan; it does not mean that any proposed assistant is production-ready.

## 2. Topic ladder
| Level | File / canonical teaching home | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | [RAG: evidence before answers](../../../02_data_science_method_playbooks/05_nlp_llm_playbook/Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md) | Trace document-to-citation retrieval and diagnose missing evidence. | Tokens, ranked lists, ACLs | external draft |
| 2 | Same, Sections 4–5 | Calculate eligible recall and reason about chunk/context budgets. | Level 1 | external draft |
| 3 | Same, Sections 7–9 | Select filters, reranking, abstention and monitor cost/latency boundaries. | Levels 1–2 | external draft |
| 4 | Local planned: tenant adapter lifecycle (PLAN-2026-IR004) | Specify tenant-scoped adapter selection, pinning and rollback. | Model/version lifecycle | planned |
| 5 | Local planned: structure-aware retrieval (PLAN-2026-IR012) | Evaluate all-required-span success, deletion propagation and filtered recall. | Levels 1–3 | planned |
| 6 | Local planned: reviewed coaching registry (PLAN-2026-IR015) | Publish only eligible, reviewed, current scripts. | Evidence/version governance | planned |
| Final | Common Issues | Diagnose recurring retrieval, policy, CRM and review failures. | All applicable sheets | not authored |

## 3. Quick topic lookup
| Question | Teaching home | Section | Use condition | Failure warning |
|---|---|---|---|---|
| Why did a relevant answer disappear? | RAG sheet | 2, 8 | Trace every retrieval stage | Reranking cannot restore an absent candidate |
| Is a citation enough? | RAG sheet | 3, 7 | Claim has a source span | Citation is not entailment or authorization |
| How is retrieval quality measured? | RAG sheet | 4–5 | Gold eligible spans exist | Do not mix unauthorized items into denominator |
| When should the assistant abstain? | RAG sheet | 7–8 | Evidence conflicts or is missing | Refusal must not reveal protected existence |
| How do we handle policy versions? | Local planned IR003 | planned | Time-effective policy matters | A current index may contain superseded content |

## 4. Reading paths — three options
- **Beginner:** RAG Sections 1–3, 5, and 12; objective: distinguish document, chunk, candidate, citation, and authorization.
- **Practitioner:** Beginner path plus Sections 4, 7, 8, and 9; objective: define an evaluation set and answer/abstain contract.
- **Advanced:** Practitioner path plus planned IR003, IR004, IR012, and IR015 when authored; objective: defend lifecycle, tenant isolation, freshness, and human-release boundaries.

## 5. Prerequisite skills
| Area | Concrete skill | Reference |
|---|---|---|
| Search | Interpret precision, recall, ranking and candidate sets | RAG sheet Sections 3–5 |
| Security | Separate authentication, authorization and policy evaluation time | RAG sheet Sections 2–3 |
| Data engineering | Track document IDs, versions, offsets, deletions and CDC | planned IR012 |
| Operations | Define a request trace and a rollback condition | RAG sheet Sections 8–9 |

## 6. What this collection does NOT cover
It does not establish legal compliance, a universal model choice, proprietary CRM behavior, autonomous financial/HR actions, or safety of tool execution. Contract interpretation and policy approval remain owner decisions. Fine-tuning mechanics belong to the model-training curriculum; local IR004 is a proposed application boundary, not that curriculum’s replacement.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Domain | First appearance |
|---|---|---|---|---|
| q | request/question | text event | authenticated request | RAG sheet §2 |
| d | source document/version | versioned object | corpus | RAG sheet §3 |
| c | chunk with stable source locator | text object | derived from d | RAG sheet §3 |
| E(q) | eligible evidence set for q | set | policy and time scoped | RAG sheet §4 |
| k | retrieved candidate count | count | non-negative integer | RAG sheet §4 |
| ACL | authorization policy | rule set | request-time principal/resource | RAG sheet §2 |

## 8. Quick Reference Card directory
| Card | Rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [RAG card](../../../02_data_science_method_playbooks/05_nlp_llm_playbook/Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md#12-quick-reference-card) | authorize → retrieve → support → answer/abstain | evidence-backed QA | policy decision is current | source or permission freshness is unknown |

## 9. Common Issues locator
No local Common Issues artifact exists. Do not infer that a generic RAG failure list covers tenant adapters, CRM writes, reviewed scripts, or local policy semantics. Required final count: at least 25 complete distinct issues after the local sheets are authored.

## 10. Empirical sources, versions and maintenance
| ID | Source / claim scope | Status | Review trigger |
|---|---|---|---|
| CAN-RAG-13 | Canonical RAG teaching artifact; illustrative examples, no local benchmark | external draft | corpus, ACL, embedding, or evaluation change |
| REF-IR003 | grounded policy-RAG interview reference; examples are not transferable facts | source reference only | policy/version contract change |
| REF-IR004 | multi-tenant adapter reference; not evidence of isolation | source reference only | base/adaptor lifecycle change |
| REF-IR012 | chunking/index reference; not a local SLO | source reference only | tokenizer/index/delete semantics change |
| REF-IR015 | reviewed scripts reference; not a compliance determination | source reference only | product evidence/review workflow change |

## 11. Collection acceptance checklist
- [x] Eleven required index sections and real canonical links are present.
- [x] Proposed source commitments retain uncertainty and remain visibly planned.
- [ ] Local focused sheets satisfy the 12-section, 1,000-line contract.
- [ ] A local Common Issues file contains at least 25 complete issues.
- [ ] Local numerical examples, interview packs, and link checks are authored and independently reviewed.
