# Module 54 Case Studies — ML and AI System Design Interviews

## 1. Collection metadata
Audience: senior system-design candidates and reviewers. Owner: interview curriculum maintainers. Index state: draft. Reference narratives are prompts, not organization evidence. Evidence, implementation stage, and editorial status are tracked separately.

## 2. Case catalog
| Case ID | Case title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-54-001 | Design content moderation | platform ML | predictive-ml | select decision/data/evaluation design | decision contract | human review | not-measured | proposed | not reviewed | planned; no file |
| PLAN-54-002 | Design enterprise RAG assistant | enterprise knowledge | llm-agentic | select grounded retrieval/service contract | decision contract | retrieval evaluation | not-measured | proposed | not reviewed | planned; no file |
| PLAN-2026-IR020 | Distributed service correctness | search/routing | hybrid | preserve guarantees under sharding/outage | correctness-contract | admission; materialized-view | not-measured | proposed | not reviewed | planned; reference-only |
| PLAN-2026-IR026 | Data-structure operation mix | serving/DSA | hybrid | select structure with resource bounds | correctness-contract | bounded-work | not-measured | proposed | not reviewed | planned; reference-only |
| PLAN-2026-IR034 | Stateful serving consistency | cache/typeahead | hybrid | define refresh/failure guarantees | correctness-contract | snapshot-consistency | not-measured | proposed | not reviewed | planned; reference-only |

## 3. Canonical reusable patterns
| Pattern | Mechanism | Boundary |
|---|---|---|
| decision contract | define owner/input/action/objective/guardrail | no metric substitutes for decision |
| correctness contract | state invariant and recovery boundary | local correctness need not survive partial failure |
| bounded-work | set admission/work/output limits and response | rejected work needs explicit behavior |

## 4. Similarity / contrast map
| Pattern | Supporting planned case | Contrast | Lesson |
|---|---|---|---|
| human review | moderation | RAG citation review | reviewers need capacity and an oracle |
| consistency | IR020/034 | IR026 local operation selection | distributed semantics change the proof |

## 5. Questions and future-case backlog
| Question | Needed evidence | Proposed case | Owner | Trigger |
|---|---|---|---|---|
| Which false decision is most costly? | stakeholder cost model | moderation | product owner | requirements review |
| What constitutes grounded output? | corpus/query/citation oracle | enterprise RAG | knowledge owner | corpus change |
| What fails under shard loss? | failure replay/reference comparison | IR020 | platform lead | architecture change |

## 6. Review and retirement log
| Case ID | Date | Change | Evidence | Action |
|---|---|---|---|---|
| all | 2026-11-09 | catalog normalized to seven sections | planned reference inventory | no case promoted |

## 7. Pre-publish checks
- [ ] Every case has all 15 sections, a checked numerical/trace task, >=12 answered interview questions, and final card.
- [ ] Catalog links, case IDs, evidence labels, stages, and editorial states reflect actual files.
- [ ] Primary sources support every factual organization/model/vendor claim.
- [ ] Planned reference examples remain clearly non-empirical until independently authored.
