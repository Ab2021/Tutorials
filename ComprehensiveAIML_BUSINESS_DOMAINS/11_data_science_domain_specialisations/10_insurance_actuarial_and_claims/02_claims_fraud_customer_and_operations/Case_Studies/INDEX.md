# Claims, Fraud, Customer and Operations - Case Study Collection

> **Mandatory authoring contract:** Every authored case must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-10-02-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-10-02-01 | FNOL triage | Claims, Fraud, Customer and Operations | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-10-02-02 | Claims-fraud detection | Claims, Fraud, Customer and Operations | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-10-02-03 | Damage-image estimate | Claims, Fraud, Customer and Operations | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-10-02-04 | Customer communications | Claims, Fraud, Customer and Operations | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-10-02-05 | Claims backlog optimisation | Claims, Fraud, Customer and Operations | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

## 3. Canonical reusable patterns

| Tag | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| data-leakage | Information unavailable at decision time inflates performance | Unexpectedly strong predictive result | Feature is truly available before the cutoff |
| label-delay | Outcome truth arrives late or changes | Fraud, claims, retention, downstream operational labels | Immediate, stable labels |
| distribution-shift | Operating population differs from training/evaluation | New market, season, channel, or rollout | Change is sampling noise only |
| evaluation-mismatch | Offline metric does not represent the business decision | Ranking, forecast, LLM, agent, or multi-objective products | Metric captures decision and guardrails |
| cost-latency-tradeoff | Quality, cost, and response time conflict | Online, real-time, or edge operation | Cost/latency are non-binding |
| human-in-the-loop | Qualified review changes risk/quality trade-off | High-impact, uncertain, or regulated work | Review is infeasible or ineffective |
| rollout-regression | Release degrades a live guardrail despite tests | Canary, A/B, feature-flag, or migration work | No rollout comparison exists |
| model-monitoring | Production signals reveal drift or degradation | Deployed model/system with feedback | No meaningful feedback signal |
| causal-confounding | Observational relationship is not intervention effect | Attribution, uplift, promotion, policy decisions | Valid randomization addresses concern |
| incident-postmortem | Operational impact requires causal reconstruction | Actual incident or replay exercise | No failure event/replay exists |

## 4. Planned cross-case map

| Pattern | Supporting planned cases | Theory/method links | Boundary to test |
|---|---|---|---|
| evaluation-mismatch | All planned cases | Relevant method and evaluation modules | Whether proxy metric changes decision quality |
| distribution-shift | Market, season, policy, or user changes | Forecasting, MLOps, and domain indexes | Population and operational conditions |
| human-in-the-loop | High-impact operational cases | Governance, UX, and domain workflows | Review capacity and override quality |

## 5. Backlog and review

Before drafting any listed case, replace the PLAN ID with a CASE ID, add a relative file link, choose one primary pattern and no more than two secondary patterns, state evidence/stage independently, and set a concrete review trigger. The full 15-section integrated case-study outline is mandatory for each drafted case.

## 6. Pre-publish checks

- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | Supporting curriculum; shared brief | hybrid | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | `atomic-release-bundle` | `rollout-regression`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR011 | Compare document VLMs with OCR for adjuster assistance | Supporting curriculum; shared brief | hybrid | Which claim fields can be extracted reliably, and which bundles require adjuster review? | `human-in-the-loop` | `evaluation-mismatch`, `delayed-selective-labels` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR014 | Separate extraction correctness from routing economics | Supporting curriculum; shared brief | llm-agentic | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | `evaluation-mismatch` | `idempotent-side-effects`, `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR015 | Publish reviewed coaching scripts from current product evidence | Supporting curriculum; shared brief | llm-agentic | Which generated scripts are fit for an approved, versioned coaching registry? | `human-in-the-loop` | `eligibility-before-ranking`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR016 | Distinguish speech, speaker and extraction errors in call analytics | Supporting curriculum; shared brief | hybrid | Which customer objections can be reported with reliable speaker attribution and evidence? | `evaluation-mismatch` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work | Supporting curriculum; shared brief | incident | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | `bounded-work` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR036 | Evaluate incremental investigator value from graph fraud signals | Supporting curriculum; shared brief | predictive-ml | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | `delayed-selective-labels` | `human-in-the-loop`, `data-leakage` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR005 | [REF-IR005](../../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults |
| PLAN-2026-IR011 | [REF-IR011](../../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing |
| PLAN-2026-IR014 | [REF-IR014](../../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes |
| PLAN-2026-IR015 | [REF-IR015](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | Human approval and source maintenance remain necessary; the example word target is not a compliance standard |
| PLAN-2026-IR016 | [REF-IR016](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores |
| PLAN-2026-IR031 | [REF-IR031](../../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound |
| PLAN-2026-IR036 | [REF-IR036](../../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
