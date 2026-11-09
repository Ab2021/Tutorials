# Module 30 Cheat Sheets — Agentic Systems, Workflows and Orchestration

## 1. Overview
For engineers familiar with transactions and distributed failures, this collection supports decisions about request ownership, agent state, isolation, recovery and side effects. The boundary sheet is an unreviewed draft; production readiness requires independent review and workload-specific evidence.

## 2. Topic ladder
| Level | Topic and outcome | Prerequisite / skill | Status |
|---|---|---|---|
| Foundation | [01 Agent boundaries and request ownership](01_agent_boundaries_and_request_ownership.md): assign decision, state, retry and effect ownership to a bounded request path. | Basic model/tool interactions and HTTP identity. | Draft; not reviewed |
| Core | Workflow selection: select an orchestration style and explain tradeoffs. | Agent-pattern concepts. | Planned |
| Core | Handoff protocol: define ownership and validate a handoff under failure. | Workflow and identity basics. | Planned |
| Production | [04 Sessions, state and lifecycle](04_agent_sessions_state_isolation_and_lifecycle.md): trace ownership and isolation. | Authentication, transactions and access control. | Draft; not reviewed |
| Production | [05 Durable workflows](05_durable_workflows_and_idempotent_execution.md): analyze crash windows and bounded backlog. | Sheet 04, queues and rate arithmetic. | Draft; not reviewed; source contents unavailable |

## 3. Quick topic lookup
| Question | Exact section | Use and warning |
|---|---|---|
| Where does a model proposal become an authorized action? | [Boundaries](01_agent_boundaries_and_request_ownership.md#4-core-formulas-with-full-derivation) | Require adapter-side canonical arguments and policy; schema validity is insufficient. |
| Who owns session state? | [Definitions](04_agent_sessions_state_isolation_and_lifecycle.md#3-foundational-definitions) | Separate context from authoritative state. |
| Can a timed-out write be retried? | [Failures](05_durable_workflows_and_idempotent_execution.md#8-failure-modes--edge-cases) | Inspect provider status and key contract; timeout is not failure proof. |

## 4. Reading paths
- **Beginner:** Read [01](01_agent_boundaries_and_request_ownership.md), then learn authentication and database transactions; read [session definitions](04_agent_sessions_state_isolation_and_lifecycle.md#3-foundational-definitions) and its [mental model](04_agent_sessions_state_isolation_and_lifecycle.md#2-concept-map--mental-model).
- **Practitioner:** Read sheets [01](01_agent_boundaries_and_request_ownership.md), 04, then [05](05_durable_workflows_and_idempotent_execution.md), tracing ownership, checkpoints and external effects.
- **Advanced:** Read all three drafts, then use their failures and cards to reason through replay, tenant isolation and reconciliation; the workflow and handoff guides are planned extensions.

## 5. Prerequisite skills
Understand HTTP authentication, tenant authorization, transactions, leases, queue rates and retry ambiguity. Start with the [session header](04_agent_sessions_state_isolation_and_lifecycle.md#1-file-header) and its linked concurrency reference; then apply queue arithmetic in [05](05_durable_workflows_and_idempotent_execution.md#4-core-formulas-with-full-derivation).

## 6. What this collection does NOT cover
The workflow-selection and handoff-protocol guides remain planned. The boundary draft does not establish any framework or provider contract; neither it nor the two production-focused drafts prove exactly-once external effects. Tool integration details require separate study. Reference-informed briefs below are proposals, not completed sheets.

## 7. Master symbol glossary
| Symbol / term | Meaning; unit / type / domain | First use |
|---|---|---|
| Run / effect key | Versioned attempt / stable intended business operation identifier; opaque strings, tenant-scoped. | [01 definitions](01_agent_boundaries_and_request_ownership.md#3-foundational-definitions) |
| Session | Tenant-scoped durable interaction container; identifier, not authorization. | [04 definitions](04_agent_sessions_state_isolation_and_lifecycle.md#3-foundational-definitions) |
| B | Pending backlog; nonnegative scalar jobs. | [05 definitions](05_durable_workflows_and_idempotent_execution.md#3-foundational-definitions) |
| lambda, mu | Arrival and completion rates; jobs per second, nonnegative scalars under the stated model. | [05 definitions](05_durable_workflows_and_idempotent_execution.md#3-foundational-definitions) |

## 8. Quick Reference Card directory
| Card | Rule / use | Condition and avoid |
|---|---|---|
| [01 card](01_agent_boundaries_and_request_ownership.md#12-quick-reference-card) | Bind model proposals to an authorized adapter and reconcile the same effect key. | A timeout is ambiguity, not failure proof. |
| [04 card](04_agent_sessions_state_isolation_and_lifecycle.md#12-quick-reference-card) | Trace state ownership and isolation. | An identifier is not proof of authorization. |
| [05 card](05_durable_workflows_and_idempotent_execution.md#12-quick-reference-card) | Reconcile ambiguous effects before unsafe retry. | Provider semantics and key retention govern effect safety. |

## 9. Common Issues locator
The final Common Issues artifact has not been authored; its count is unknown. Use the draft [session failures](04_agent_sessions_state_isolation_and_lifecycle.md#8-failure-modes--edge-cases) for isolation and ownership, and [workflow failures](05_durable_workflows_and_idempotent_execution.md#8-failure-modes--edge-cases) for a lost response: look up the stable operation key before retry. These are hypothetical diagnostics, not incident counts.

## 10. Empirical sources, versions and collection maintenance
Original source directories were unavailable and not inspected; the crosswalk identifies source titles only. Both sheets are drafts without independent review; their examples are not production measurements. Consult [standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [template](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md); recheck provider contracts, retention, authorization and source versions on change. No review date is asserted. Each sheet requires all 12 sections and a final card.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

### Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery. Boundary: A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults. | [REF-IR005](../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | planned; not written |
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests. Boundary: External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions. | [REF-IR006](../../../../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->

## 11. Collection acceptance checklist
- [ ] Author the three planned guides and final Common Issues artifact with at least 25 substantive issues.
- [ ] Independently review both drafts, their 12-section structure, calculations, recovery claims and sources.
- [ ] Verify current provider contracts, evidence, links and anchors before publication.
- [ ] Record review dates and distinguish illustrative traces from observed incidents.
