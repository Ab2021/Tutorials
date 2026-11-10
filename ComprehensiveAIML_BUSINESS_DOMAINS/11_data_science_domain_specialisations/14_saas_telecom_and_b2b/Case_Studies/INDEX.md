# SaaS, Telecom and B2B — Cross-Domain Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Scope | Decisions that genuinely compare or integrate two or more family domains |
| Canonical ID | `CASE-14-XD-NNN` |
| Editorial state | No local case is authored; this is a truthful navigation catalog |
| Evidence policy | Separate illustrative, published, simulated, observed, estimated, and not-measured claims |
| Review trigger | Change to a cross-domain decision contract, evidence source, or shared identity/policy boundary |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-14-XD-001 | Unified account-health intervention | SaaS + telecom customer operations | predictive-ml | Should a common risk signal route a customer to product or service remediation? | evaluation-mismatch | label-delay; human-in-the-loop | not-measured | proposed | not reviewed | Planned only; shared identity and causal outcome must be established. |
| PLAN-14-XD-002 | Authorized support-answer handoff | Enterprise knowledge + SaaS support | llm-agentic | When should a cited answer resolve a ticket versus hand off to a human? | authorization-scope | evaluation-mismatch; human-in-the-loop | not-measured | proposed | not reviewed | Planned only; no support outcome is claimed. |
| PLAN-14-XD-003 | Incident communication across service channels | Telecom operations + B2B customer success | streaming-systems | Which affected accounts receive which verified outage update under freshness constraints? | bounded-work | snapshot-consistency; rollout-regression | not-measured | proposed | not reviewed | Planned only; no operator workflow or delivery rate is asserted. |

## 3. Canonical reusable patterns
| Pattern | Meaning | Use when | Disqualifier |
|---|---|---|---|
| authorization-scope | Output is conditioned on request-time principal, resource, and policy | Knowledge or CRM data crosses roles/tenants | Static metadata is treated as a current decision |
| evaluation-mismatch | Offline score omits the actual decision or guardrail | Predictive, ranking, or generation output triggers action | Metric has no action-level definition |
| label-delay | Outcome maturity trails the intervention | Churn, support resolution, or restoration outcome is delayed | Immediate proxy is accepted as truth |
| human-in-the-loop | Qualified review is part of the control | Ambiguous, high-impact, or low-confidence work | Review capacity/quality is not measured |
| snapshot-consistency | A multi-record decision must use a stated coherent view | Incident status, eligibility, or entitlement can change mid-request | Read time and version are unspecified |

## 4. Similarity and contrast map
| Planned case | Shares | Differs from | Required comparison |
|---|---|---|---|
| XD-001 | Delayed outcomes and intervention capacity | Product use versus service degradation causes | Compare no action, domain-specific action, and shared triage under identical outcome window |
| XD-002 | Evidence, access control, review | Answer quality is not ticket resolution | Separate supported-answer rate from resolved-without-reopen rate |
| XD-003 | Event freshness and customer communications | Delivery success is not service restoration | Evaluate status correctness, entitlement scope, and recovery update separately |

## 5. Questions and future-case backlog
A future author must identify a single owner, input/output grain, baseline, bounded action, evidence source, and validation protocol. Questions not yet resolved: whether identities can be joined lawfully and correctly; which actions are reversible; what event-time policy governs cross-system truth; and how customer impact is measured without proxy-only conclusions.

## 6. Review and retirement log
| Date | Scope | Evidence observed | Decision | Recheck trigger |
|---|---|---|---|---|
| 2026-11-09 | Cross-domain catalog | No local source ledger, implementation, or outcome artifact exists. | Keep all entries as `PLAN`; do not present them as cases. | A full 15-section case with a source/evidence ledger and checked example. |

## 7. Pre-publish checks
- [ ] Replace a `PLAN` ID only after creating and linking a 15-section case with Section 15 last.
- [ ] Record owner, action, unit, baseline, time window, and evidence label at claim level.
- [ ] Show successful and failed/recovery paths for systems cases.
- [ ] Include 12–18 answered interview prompts, numerical checks, and sensitivity where quantitative.
- [ ] Verify that any cross-domain join, permission, and event-time claim is supported or explicitly illustrative.
