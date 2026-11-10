# Module 06 Case Studies — Data Modelling, Databases, Lakehouse and Governance

## 1. Collection Metadata

| Field | Value |
| --- | --- |
| Collection | Module 06 case studies |
| Teaching home | [Module 06](../INDEX.md) |
| Purpose | Apply data modelling, database, lakehouse, reconciliation, and governance choices to bounded decisions. |
| Collection status | Mixed: one authored draft and remaining planned briefs. |
| Authoring contract | [Course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). |
| Evidence posture | Evidence, implementation stage, and editorial status are separate. “Draft” does not mean deployed or measured. |
| Source limit | Reference-informed items remain proposals unless an authored canonical case records accessible evidence. |

## 2. Case Catalog

The catalog retains the original planned case identity and every reference-informed addition. `CASE` IDs identify authored canonical cases; `PLAN` IDs remain briefs, not completed cases.

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CASE-2026-DS06-001 | [Model an e-commerce lakehouse](01_model_an_ecommerce_lakehouse.md) | E-commerce analytics, reconciliation, governance | streaming-systems | Whether a narrow contract-gated ledger is warranted over nightly overwrite extracts | reconcilable-event-ledger | contract-gated-serving; late-data-recovery | illustrative / estimated | proposed | not independently reviewed | Draft gives a $100 constructed ledger oracle and payload estimate; no deployment, source contract, cost, or outcome measurement. |
| PLAN-02 | repair a broken customer 360 pipeline | Customer identity and data quality | incident | Design the decision, architecture, metrics, risks, and validation plan | planned | — | not-measured | proposed | not reviewed | Original row retained; no authored canonical case. |
| PLAN-2026-IR012 | Preserve answer evidence through ingestion and filtered retrieval | Authorized retrieval | llm-agentic | Which chunking and index design retrieves every required authorized span within the service budget? | materialized-view | authorization-scope; evaluation-mismatch | not-measured | proposed | planning reviewed 2026-11-06 | Supporting curriculum brief; not a completed case. |
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely | Search/index operations | hybrid | How should search remain correct across indexing, deletion, ranking and shard failures? | materialized-view | authorization-scope; snapshot-consistency | not-measured | proposed | planning reviewed 2026-11-06 | Supporting curriculum brief; not a completed case. |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work | Hierarchical data access | incident | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | bounded-work | authorization-scope; snapshot-consistency | not-measured | proposed | planning reviewed 2026-11-06 | Proposed home; not a completed case. |

**Editorial status:** `CASE-2026-DS06-001` is a draft, not independently reviewed. All `PLAN` rows are planned/proposed. No catalog row asserts production implementation or measured result.

## 3. Canonical Reusable Patterns

| Pattern | Mechanism | Preconditions | Avoid / counterexample | Case or plan |
| --- | --- | --- | --- | --- |
| reconcilable-event-ledger | Admit one signed business effect per stable identity and preserve lineage to immutable receipts. | Stable source/effect identity, explicit conflict policy, signed amount semantics. | Do not use if sources cannot distinguish replay from correction. | CASE-2026-DS06-001 |
| contract-gated-serving | Publish a decision table only with declared grain, cutoff, quality status, and owner. | Named consumer, measurable gate, refusal behavior. | A generic “certified” label without denominator or authority is not a gate. | CASE-2026-DS06-001 |
| late-data-recovery | Preserve named snapshots and publish corrections as linked restatements. | Event and knowledge time, versioned input manifests. | Overwriting “latest” cannot reproduce a close. | CASE-2026-DS06-001 |
| materialized-view | Build a derived read model from controlled source state. | Freshness, authorization, and deletion semantics are explicit. | A fast index cannot restore omitted or unauthorized source content. | PLAN-2026-IR012; PLAN-2026-IR021 |
| bounded-work | Bound traversal/index work before exposing a request path. | Defined workload, authorization checks, truncation behavior. | One SQL statement or an outer limit does not necessarily bound work. | PLAN-2026-IR031 |

## 4. Similarity / Contrast Map

| Item | Similarity | Contrast / boundary | Reusable question |
| --- | --- | --- | --- |
| E-commerce ledger vs customer 360 | Both require stable identity and time-aware joins. | Finance effect accounting is not a universal customer profile. | What grain and purpose make a join permissible? |
| E-commerce ledger vs retrieval ingestion | Both use immutable intake and delete propagation. | Retrieval quality is candidate/authorization coverage, not net-sales reconciliation. | What output is certified, and by what oracle? |
| E-commerce ledger vs search index evolution | Both need snapshot consistency and safe correction. | Search ranking correctness differs from signed financial effect correctness. | Can a named version be replayed and compared? |
| E-commerce ledger vs hierarchy API | Both require bounded work and authorization. | Graph traversal has request work bounds; ledger has close/reconciliation bounds. | Which boundary prevents silent wrong output? |

## 5. Questions and Future-Case Backlog

| ID | Question / proposed case | Why not closed | Required evidence or test | Owner / status |
| --- | --- | --- | --- | --- |
| B-01 | Complete original row 2: repair a broken customer 360 pipeline. | No canonical manuscript exists. | Identity-link, consent, delete, and fan-out diagnosis with bounded action. | Curriculum / planned |
| B-02 | Verify payment source identity and settlement semantics for a real adoption decision. | Case inputs are illustrative. | Source contracts and synthetic correction/replay oracle. | Payment Ops / OPEN |
| B-03 | Define Controller-approved close, restatement, and materiality policy. | No policy artifact available. | Signed policy and exception workflow. | Finance / OPEN |
| B-04 | Measure batch-plus-audit versus ledger total cost and recovery behavior. | No representative workload or price data. | Controlled load, scan, compaction, labor, and recovery study. | Platform / OPEN |
| PLAN-2026-IR012 | Preserve answer evidence through ingestion and filtered retrieval. | Reference-informed brief only. | All-required-span, ACL, freshness, and latency evaluation. | proposed |
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely. | Reference-informed brief only. | Boolean/phrase/top-k/delete/index-lag oracles. | proposed |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work. | Reference-informed brief only. | Traversal, authorization, scan, and truncation comparison. | proposed |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
| --- | --- | --- |
| PLAN-2026-IR012 | [REF-IR012](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | Reranking cannot recover a missing candidate; source embedding limits and tenant count are not platform requirements. |
| PLAN-2026-IR021 | [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | Exact global top-k needs a valid distributed protocol; source figures are scenario claims, not search-cluster benchmarks. |
| PLAN-2026-IR031 | [REF-IR031](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion-work bound. |

## 6. Review / Retirement Log

| Date | Item | Event | Evidence / decision | Editorial state |
| --- | --- | --- | --- | --- |
| 2026-11-06 | PLAN-2026-IR012/021/031 | Planning review recorded in inherited catalog. | Reference-informed briefs retained without completion claim. | proposed |
| 2026-11-10 | CASE-2026-DS06-001 | Initial authored case registered. | Illustrative ledger arithmetic and proposed validation only. | draft |
| OPEN | CASE-2026-DS06-001 | Independent content, source, and local-policy review. | Needed before reviewed/published label. | draft |
| OPEN | Any PLAN item | Assign canonical CASE ID only when manuscript is authored. | Do not retire plan solely because a related topic exists. | proposed |

Retirement rule: retire or supersede a case only after named consumers, links, and its distinctive decision outcome are migrated to a canonical replacement. No item is retired here.

## 7. Pre-Publish Checks

| Check | CASE-2026-DS06-001 state | Collection action |
| --- | --- | --- |
| 15 ordered case sections; Section 15 last | Present in draft. | Recheck after every substantive edit. |
| 5a–5e, bounded owner/decision/baseline | Present in draft. | Preserve on review. |
| Worked arithmetic, sensitivity, sizing, recovery | Present; illustrative and manually checked. | Independently recompute before publication. |
| 12–18 answered interview questions | 13 present. | Check answer/support/scoring consistency. |
| Source/evidence/stage/editorial status separate | Present. | Do not upgrade absent source evidence. |
| Ledger/replay/delete/reconciliation validation | Proposed except authoring arithmetic smoke. | Execute only under authorized local test conditions. |
| Link integrity and planned-row retention | Links and PLAN rows are recorded; external reference contents remain unverified here. | Inspect before publication. |
| Collection seven-section structure | Present. | Keep catalog at 11 columns and retain source additions. |

**Open source limits:** primary payment/order schemas, settlement definitions, privacy/deletion policy, retention rules, measured workload, vendor behavior, and pricing are unavailable to this illustrative case. These limits constrain every technology and outcome claim in this collection.
