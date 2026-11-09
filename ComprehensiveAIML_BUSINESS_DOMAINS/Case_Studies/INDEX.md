# Cross-Domain Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Owner | Curriculum maintainer; individual authors unassigned |
| Audience and boundary | Cross-domain architecture and decision practice; focused methods remain in module collections |
| Index version / review cadence | 3.0-draft / quarterly and on evidence or dependency change |
| Substantive case review | Not performed; six authored illustrative design exercises await independent review |
| Case identity | `CASE-XD-NNN` identifies an authored case; `PLAN-XD-NNN` remains the preserved original brief identity |
| Source of truth | This INDEX for cross-domain briefs; source-specific topics are in the root source crosswalk |
| Authoring authority | [Case master template](../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) and [course numerical standard](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) |
| Editorial-state convention | `draft` is separate from evidence and implementation stage; no draft is a deployment claim |

## 2. Case catalog

Every row is **editorial state: draft**, **stage: proposed**, **evidence: illustrative / not-measured outcome**. Links lead to authored design exercises; their constructed values and proposed tests are not organization measurements.

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-XD-001 (PLAN-XD-001) | [Enterprise knowledge agent with controlled actions](01_enterprise_knowledge_agent_controlled_actions.md) | Enterprise knowledge | llm-agentic | Which retrieved answers and tool actions may be released under exact-user permissions and approval constraints? | authorization-scope | human-in-the-loop; idempotent-side-effects | illustrative; outcome not measured | proposed | not reviewed | Derived queue/recovery arithmetic; authorization, approval, and action results require replay |
| CASE-XD-002 (PLAN-XD-002) | [Predictive-maintenance-to-robotics operating loop](02_predictive_maintenance_robotics_operating_loop.md) | Industrial operations | hybrid | When should a detected fault cause review, maintenance scheduling or a controlled physical stop? | safe-state-boundary | delayed-feedback; cost-latency-tradeoff | illustrative; outcome not measured | proposed | not reviewed | Fault labels, control timing, and safety evidence unavailable; HIL validation remains required |
| CASE-XD-003 (PLAN-XD-003) | [Regulated healthcare multimodal assistant](03_regulated_healthcare_multimodal_assistant.md) | Healthcare | llm-agentic | Which document/signal-supported responses can aid a clinician, and which require abstention or expert escalation? | human-in-the-loop | authorization-scope; evaluation-mismatch | illustrative; outcome not measured | proposed | not reviewed | No clinical benefit claim; site/time/slice review and clinician validation remain required |
| CASE-XD-004 (PLAN-XD-004) | [Real-time financial crime platform](04_real_time_financial_crime_platform.md) | Payments and investigations | hybrid | Which events should be blocked or queued for investigation at fixed review capacity? | delayed-feedback | evaluation-mismatch; idempotent-side-effects | illustrative; outcome not measured | proposed | not reviewed | No fraud reduction claim; capacity-aware replay and mature-label protocol required |
| CASE-XD-005 (PLAN-XD-005) | [Sustainable supply-chain control tower](05_sustainable_supply_chain_control_tower.md) | Supply chain | hybrid | Which inventory/transport exception actions improve service within capacity, cost and emissions constraints? | constrained-allocation | human-in-the-loop; evaluation-mismatch | illustrative; outcome not measured | proposed | not reviewed | Scenario assumptions only; feasibility, emissions factors, and planner trial require validation |
| CASE-XD-006 (PLAN-XD-006) | [Global model-platform incident](06_global_model_platform_incident.md) | AI platform operations | incident | Which containment and recovery sequence restores useful responses without incompatible state or duplicate effects? | compatible-recovery | idempotent-side-effects; cost-latency-tradeoff | illustrative; outcome not measured | proposed | not reviewed | Hypothetical incident exercise; restore/replay and duplicate-effect testing remain required |

Each authored case keeps the original bounded decision, source uncertainty, full data/decision/workload assumptions, pass/fail rules, and a labeled illustrative source basis. Illustrative values must not become claimed results.

## 3. Canonical reusable patterns

| Pattern | Mechanism / diagnostic signature | Preconditions | Avoid condition |
|---|---|---|---|
| authorization-scope | Eligibility filters precede retrieval, ranking or execution | Explicit user/resource policy | An unverified policy cannot establish access |
| human-in-the-loop | Qualified review changes the permitted action | Reviewer authority, time and information | Review is unavailable or becomes a rubber stamp |
| idempotent-side-effects | Stable operation identity prevents repeated external effects | Durable deduplication and reconciliation boundary | Downstream behavior cannot support the claimed guarantee |
| safe-state-boundary | Physical actuation is gated by verified operating limits | Device/control interfaces and safe-state plan | Model accuracy is the only safety evidence |
| delayed-feedback | Action truth arrives after the decision and may be selected | Availability timestamps and adjudication accounting | Mature, immediate labels need no such mechanism |
| evaluation-mismatch | A proxy improves while the real decision may worsen | Defined business outcome and guardrails | Proxy already measures the bounded decision directly |
| cost-latency-tradeoff | Quality competes with response/resource budget | Comparable workload and explicit constraints | Unmeasured tail behavior is treated as a guarantee |
| constrained-allocation | Explicit resources and feasibility govern selection | Valid objective, constraints and fallback | Infeasible recommendations can execute unchecked |
| compatible-recovery | Recover model/data/prompt/tool/schema as a coherent unit | Version ownership and tested restore/replay paths | Rolling back one component violates the state contract |

This deliberately small cross-domain vocabulary is provisional; focused collections may define additional mechanisms. No deprecated synonyms yet. Case relationships and method homes are below.

## 4. Similarity and contrast map

These are proposed teaching comparisons, not empirical support.

| Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned |
|---|---|---|---|---|
| human-in-the-loop | XD-001, XD-003 | XD-002 | [Agents](../06_llm_agents_knowledge_systems_and_ai_augmented_development/INDEX.md) / [edge control](../07_edge_ai_robotics_and_cyber_physical_systems/INDEX.md) | Human review latency may be incompatible with physical deadlines |
| delayed-feedback | XD-004 | XD-005 | [Decision science](../02_data_analytics_and_decision_science/INDEX.md) | Mature retrospective labels differ from action-time information |
| compatible-recovery | XD-006 | XD-001 | [Platforms](../05_ai_platforms_data_systems_and_production_engineering/INDEX.md) | Restored internal state does not undo an external effect |
| evaluation-mismatch | XD-003, XD-004 | XD-005 | [Domain systems](../08_business_domains_and_decision_intelligence/INDEX.md) | Model quality, investigator value and operational feasibility need different protocols |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Which actions remain safe across retries? | XD-001, XD-006 | Exact approval payloads, downstream operation contracts, crash/replay artifacts | Bound side effects at each failure window | High | Platform/security reviewer | Before drafting deployment claims |
| What validates physical or clinical safety? | XD-002, XD-003 | Expert workflow, device/site conditions, failure and review evidence | Hardware-in-loop or bounded expert validation; not a substitute for applicable requirements | High | Domain and safety reviewers | Before recommending operational use |
| Does offline value survive delayed feedback and interference? | XD-004, XD-005 | Mature labels, action logs, cohorts and intervention protocol | Capacity-aware replay then controlled operational evaluation | High | Decision-science reviewer | Before reporting business benefit |
| Which published sources support these designs? | All | Accessible primary sources and claim ledger | Select illustrative/published/reproduced basis separately for each case | High | Case author | On author assignment |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| All PLAN-XD briefs | 2026-11-06 | Added bounded decisions, evidence limits and numerical/validation work; adopted master structure | Course coverage reconciliation | Planning update only; no case reviewed or published | Curriculum maintainer |
| CASE-XD-001 through CASE-XD-006 | 2026-11-09 | Authored illustrative draft cases, stable CASE IDs, decision ledgers, quantitative traces, recovery probes, and interview packs | Course authoring completion pass | Draft only; no source verification, experiment, pilot, or independent review is claimed | Curriculum maintainer |

[Reference-informed case and cheat-sheet plan](../00_REFERENCE_INFORMED_CASE_PLAN.md): 36 example files inform separate future briefs. They are references only, not completed cases or established results. Preserve their own stable planning IDs and source limitations.

## 7. Pre-publish checks

- [x] Six preserved planning IDs crosswalked to six stable CASE IDs; authored files and missing outcomes are explicit.
- [x] Decision, type, evidence, editorial state, and implementation stage are recorded independently.
- [x] Each draft has all 15 sections, a source/claim boundary, a hand-derived checked numerical walkthrough, sensitivity, and proposed reproducibility inputs.
- [x] Each draft specifies data/workload cutoffs, baseline, alternatives, uncertainty, falsification protocol, failure diagnostics, recovery reasoning, and 15 answered interview questions; Section 15 is last.
- [ ] Execute the listed replay, recovery, capacity, hardware-in-loop, clinical, financial-crime, supply-chain, and review validation protocols where applicable.
- [ ] Verify authoritative primary sources, links/anchors, external requirements, pattern relationships, independent review records, and every master-template publication gate.
