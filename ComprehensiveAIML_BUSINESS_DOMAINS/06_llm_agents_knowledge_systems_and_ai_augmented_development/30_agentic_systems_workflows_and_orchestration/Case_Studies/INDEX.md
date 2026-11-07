# Module 30 Case Studies — Agentic Systems, Workflows and Orchestration

## 1. Collection metadata

Owner: enterprise agent security curriculum maintainers. Last reviewed: not reviewed. Review cadence: annually and on material threat, policy, or connector changes. Index version: 1.1. Canonical IDs: `CASE-2026-NNN`. Source of truth: this index and its linked case files. Audience: engineers, operators, security reviewers, and interview candidates. Editorial state, implementation stage, and evidence are independent. Apply the [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) and [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md).

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Key result or limitation | Document state |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-030 | [Approval gate for an enterprise agent](01_enterprise_agent_approval_gate.md) | Enterprise agent security | llm-agentic | How should an assistant gate ticket actions despite untrusted documents? | `exact-action-approval` | `least-privilege-execution`, `idempotent-side-effects` | not-measured | proposed | not reviewed | No deployment or outcome measured; illustrative calculations only. | draft |

Planned, not case records: claims operations agent; research workflow with approval gates. Reference-informed briefs [PLAN-2026-IR005 and PLAN-2026-IR006](../../../00_REFERENCE_INFORMED_CASE_PLAN.md) remain plans; the latter shares recovery concepts but is not a published comparison.

## 3. Canonical reusable pattern vocabulary

| Pattern | Mechanism and diagnostic signature | Preconditions | Avoid / counterexample | Related case and theory | Deprecated synonyms |
|---|---|---|---|---|---|
| `exact-action-approval` | Reviewer authorizes a canonical immutable action, not a general model intention; stale hashes deny. | Independent authorization and trusted execution binding. | Cannot safely use if connector can change semantics without revalidation. | CASE-2026-030; [OWASP prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/). | none |
| `least-privilege-execution` | Restrict identities, resources, operations and duration; audit denied access. | Enforceable downstream authorization. | No protection against abuse of legitimately overbroad permissions. | CASE-2026-030; [NIST definition](https://csrc.nist.gov/glossary/term/least_privilege). | none |
| `idempotent-side-effects` | Stable business operation key and sink reconciliation limit duplicate effects after uncertain responses. | Sink deduplication or queryable receipts. | Cannot claim exactly once with opaque non-idempotent endpoints. | CASE-2026-030; PLAN-2026-IR006 is a planned contrast. | none |

## 4. Cross-case similarity and contrast map

| Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned |
|---|---|---|---|---|
| Exact action approval | CASE-2026-030 is a proposed example, not empirical support. | Procurement recovery PLAN-2026-IR006 remains planned. | [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/). | A gate blocks unapproved effects only if execution uses the reviewed intent and an independent authorization check. |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Can reviewers recognize semantic manipulation without excessive delay? | CASE-2026-030 | Blinded, representative adjudicated reviews. | Controlled reviewer study with adversarial and benign packets. | High | Security research lead | Before pilot. |
| Does the actual ticket connector deduplicate and expose authoritative receipts? | CASE-2026-030 | Connector contract and fault-injection evidence. | Replay/reconciliation test. | High | Integration lead | Before any write access. |
| Can policy evaluation remain consistent at execution? | CASE-2026-030 | Revocation and race tests against the policy service. | Expiry and entitlement-change tests. | High | Authorization owner | Before pilot. |
| How do procurement approvals differ? | PLAN-2026-IR006 | Actual sink and approval contract. | Develop planned procurement case. | Medium | Curriculum maintainer | When primary artifacts are available. |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-2026-030 | 2026-11-07 | Draft added; no substantive review claimed. | Illustrative security design and source inventory. | Updated; review pending. | Curriculum maintainer |

## 7. Pre-publish index checks

- [x] Unique case ID and draft link distinguished from planned briefs.
- [x] Tags describe the proposed mechanism; evidence, stage, and editorial state are separate.
- [x] No measured impact is claimed for this case.
- [ ] Substantive case and index review, including source and link verification, completed.
- [ ] Connector-specific security and recovery claims validated before any deployment.
- [x] Unreviewed dates are not represented as review dates; no retired case is hidden.
- [x] Catalog contains one authored draft; planning briefs are not counted as cases.
