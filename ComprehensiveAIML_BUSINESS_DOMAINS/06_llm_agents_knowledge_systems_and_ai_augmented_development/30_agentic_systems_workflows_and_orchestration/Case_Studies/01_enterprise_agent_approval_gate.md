---
case_id: "CASE-2026-030"
case_title: "Approval gate for an enterprise agent facing untrusted instructions"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Enterprise operations and agent execution security"
primary_pattern: "exact-action-approval"
secondary_patterns: ["least-privilege-execution", "idempotent-side-effects"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Enterprise agent security lead"
source_index: "INDEX.md"
review_trigger: "Change to action semantics, sink guarantees, policy, model, or adversarial evidence"
---
# Approval gate for an enterprise agent facing untrusted instructions

## 1. The One-Line Framing
Given one employee request and retrieved ticket evidence, decide whether to execute one exact ticket update for an operations owner, under a hard authorization boundary, compared with a human performing the update manually.
The task is system operation: every proposed write needs a per-action authorization decision.
The primary condition is no unauthorized side effect, with legitimate completion and reviewer effort as guardrails.
The primary pattern is `exact-action-approval`; secondary patterns are `least-privilege-execution` and `idempotent-side-effects`.
**Source declaration:** The enterprise, people, workflow, action records, quantities, calculations, policies, and design are hypothetical, not a deployed system or an observed incident.
Primary security references in Section 5 explain general risks and principles; they do not report this design or its effectiveness.
A reviewer-facing answer is not enough: a model can generate a plausible explanation while a different downstream write executes.
A conservative architecture can reduce privilege exposure but cannot guarantee an authorized action is semantically appropriate.
The finding is a proposed decision, not a measured improvement.

## 2. Context & Background
A hypothetical internal operations desk processes employee requests to update existing service tickets.
An assistant reads the request and authorized ticket context, then proposes a ticket status change or a private internal comment.
The scenario excludes external email, payments, file deletion, user provisioning, and administrative changes.
The employee can request a change but cannot automatically authorize the assistant to perform it.
The operations reviewer evaluates the proposed action, the relevant evidence, and whether it is appropriate.
A separate integration owner operates the connector; security owns abuse detection and incident escalation.
A policy owner defines which roles may request, review, and execute each action.
The manual baseline is a human viewing the ticket in the system and entering a change with their own authorized identity.
That baseline may take time and may itself contain human mistakes; no comparative error or delay has been measured.
The proposal adds model-assisted preparation and a gate; it does not replace the ticket platform's access control.
The assumed pilot would cover one English-language internal support queue in one tenant, during staffed hours.
The hypothetical queue is not a claim about any identifiable enterprise or jurisdiction.
There are no known prior attempts, historical incidents, or verified legal obligations for this fictional deployment.
Compliance counsel must establish applicable retention, labor, privacy, and notification requirements before implementation.
The decision is made at design time before any pilot; all validation and release steps below remain prospective.
A ticket's comments and attachments can contain adversarial text from external or internal participants.
Retrieved text is evidence about the ticket, never a source of new execution authority.
The trusted reviewer interface, policy decision point, execution gateway, and ticket platform are separate security boundaries.

## 3. The Business Question
**Illustrative stakeholder question:** Can we let an assistant prepare ticket changes while preventing document content from gaining authority to cause unauthorized writes?
Decision owner: the operations service owner, with security and ticket-platform owners having veto over launch.
Trigger: a proposal to grant an assistant the ability to update tickets.
Deadline: before granting any write capability, not an asserted calendar commitment.
Options: retain manual work; provide read-only draft suggestions; use a deterministic review-and-execute gate; or permit a model-controlled tool to write directly.
Value is useful completed work with appropriate evidence; downside is incorrect changes, data leakage, cleanup, and loss of trust.
Objective: increase safely completed eligible requests without eroding human oversight or isolation.
Hard scenario constraints: no unreviewed write, no cross-tenant access, no bypass of ticket ACLs, and no execution after revocation.
Scope: one write to one existing ticket per approval; batch actions and chained writes are excluded.
Out of scope: automated decisions on employee discipline or emergency response, downstream ticket automations not inventoried, and autonomous remediation.
Approval count is a workload proxy, not a business value measure; a high approval rate can reward rubber-stamping.
The default if control evidence is inadequate is manual operation or read-only drafts.

## 4. Assumptions & Constraints
| ID | Assumption or scenario constraint | Type and verification | Status / consequence | Fallback / owner and trigger |
|---|---|---|---|---|
| A-01 | The ticket platform exposes a conditional update and stable revision. | Integration: inspect contract and test concurrent writes. | Unverified; otherwise TOCTOU risk. | Manual update; integration owner before launch. |
| A-02 | Sink accepts a scoped idempotency key and exposes an authoritative lookup. | Reliability: fault test response loss. | Unverified; exactly-once cannot be asserted. | Freeze uncertain writes; operations owner before launch. |
| A-03 | Reviewers can identify source manipulation and have time to decide. | Human workflow: blinded exercise and observation. | Unverified; rubber-stamping can authorize harmful changes. | Manual flow; operations owner before pilot. |
| A-04 | Access policy can be checked at execution and revocation is timely. | Identity: revoke between approval and send. | Unverified; stale privilege permits unauthorized write. | Deny and manual escalation; identity owner on policy change. |
| A-05 | Ticket updates have bounded downstream automations. | Business: inventory triggers and external notifications. | Unverified; one write may cause many side effects. | Prohibit such tickets until mapped; platform owner. |
| A-06 | Receipt records are retained and queryable across outages. | Integration: restore and query test. | Unverified; unknown outcome cannot be resolved safely. | Manual incident reconciliation; platform owner. |
| C-01 | No write without an exact, current, independently authorized approval. | Scenario hard rule, not a legal assertion. | Proposed requirement; breach blocks release. | Deny; policy owner. |
| C-02 | Tenant and ticket access must be enforced at retrieval and execution. | Scenario hard rule; test native ticket ACL. | Proposed requirement; breach blocks release. | Disable connector; security owner. |
| C-03 | An approval cannot bypass an explicit platform or policy denial. | Scenario hard rule; test deny precedence. | Proposed requirement; bypass blocks release. | Manual authorized path; policy owner. |
| C-04 | Treat personal data and source text as sensitive. | Scenario constraint; determine retention with privacy owner. | Review needed; long retention increases exposure. | Minimize and quarantine; privacy owner. |
The assumptions most likely to reverse the design are sink deduplication and receipts, current authorization, and reviewer competence.
No market prices, jurisdiction-specific legal conclusions, or real model capacities are assumed.

## 5. Data, Tools & Method
### 5a. Data Specification
Unit: a single proposed state-changing ticket operation, identified by tenant, ticket, action ID, and immutable proposal version.
A request can lead to zero or multiple proposals; each proposal is approved separately.
The intended evaluation window is a future four-week shadow phase followed, only if authorized, by a limited pilot.
No real request corpus, dates, population counts, or run artifacts are available.
Inputs: authenticated requester ID, tenant, request time in UTC, ticket ID and revision, authorized evidence references, action type, structured payload, and policy version.
A ticket comment and its attachments are untrusted content even if retrieved through a trusted API.
At decision time the gate uses the latest platform revision, current entitlements, and an unexpired approval bound to the same payload.
An outcome label is an adjudicated judgment of whether the exact proposed effect was authorized and appropriate under the applicable policy at execution time.
A later cleanup event is not itself proof that the original action was unauthorized.
Include eligible requests submitted during the selected period; document refusals, duplicates, timeouts, and all attempted actions.
Exclude direct manual operations only from the agent-action denominator, not from the separate baseline cohort.
Join proposals to approvals by tenant, action ID, proposal version and digest; do not join by ticket ID alone.
Join the eventual receipt by scoped idempotency key and action ID, checking one-to-one cardinality.
Repeated proposals for a ticket are correlated; partition test sets by ticket and time and preserve incident clusters for analysis.
Adjudication labels must mature after reconciliation; unresolved effects remain unknown, not automatically safe.
Training data and model fitting are not claimed; an agent model is treated as an untrusted proposal generator.
### 5b. Tools & Stack
The design is product-neutral: identity provider, ticket API, retrieval gateway, model endpoint, policy service, transactional approval ledger, executor, audit store and reviewer UI.
All are proposed components, not an assertion that any product supports the required semantics.
The model and prompt versions, schema version, retrieval index version, gateway release, policy version and connector version are recorded per action.
A restricted sandbox runs any attachment parser with resource and network limits; this design does not execute document code.
There is no shell or arbitrary HTTP tool available to the agent.
The connector, not the model, obtains short-lived scoped credentials from an authorized secrets service.
The exact model, provider, library versions, cloud region, storage engine and hardware are not selected.
Those choices require privacy, isolation, performance and contract review before an implementation can be specified.
### 5c. Method Summary
Input request and ACL-filtered context -> model-generated structured proposal -> deterministic canonicalization -> policy validation -> reviewer approval -> independent fresh authorization -> conditional, idempotent sink write -> reconciliation and audit.
The deterministic policy service is authoritative for permissions; the model's explanation and labels are not.
A high-risk or ambiguous proposal is rejected or routed for manual handling; approval does not override a policy denial.
| Candidate | Evidence | Strength | Limitation | Decision | Revisit |
|---|---|---|---|---|---|
| Manual ticket update | Hypothetical baseline; no measurement. | Existing user access control and human context. | Workload and mistakes unmeasured. | Retain as fallback. | When comparable data exists. |
| Draft-only assistant | Considered, not tested. | No model-initiated sink write. | Reviewer still must safely inspect and enter data. | Safest initial comparator. | If automation benefit is demonstrated. |
| Prompt-only tool access | Considered, not tested. | Low implementation overhead. | Model instructions are not an authorization boundary. | Reject for writes. | Never without independent enforcement. |
| Exact-action gate | Proposed, not tested. | Independent action binding, authorization, and recovery. | Cost, review errors and sink constraints. | Conditional design choice. | If A-01 to A-06 fail. |
### 5d. Data provenance, quality, exclusions and leakage controls
Only a ticket-platform retrieval service may fetch a ticket, and it evaluates requester and tenant access before release.
Attach source identifier, ticket revision, retrieval time, document version and classification to each snippet.
Do not place retrieved text in privileged instructions; label it as untrusted data in the model context.
A snippet can still contain instructions or misleading facts; labels and structured outputs are not a complete defense [SRC-01, SRC-03].
Quarantine executable attachments and preserve a content hash; do not send them to a parser with ambient network access.
Restrict review packets and audit logs to role-authorized viewers; redact secrets and unnecessary personal data.
Version retention rules and allow legal holds to supersede deletion only after proper authorization.
Deletion must reach derived caches and evaluation datasets; evidence hashes may themselves be sensitive if low-entropy content is guessable.
Evaluation splits must keep near-duplicate payloads and revisions of the same ticket in one partition.
The final holdout remains unseen during prompt and detector tuning; changed attack families require a new holdout.
There is no accessible source code or dataset for the directory inventory's `code/` entry; no contents are inferred.
### 5e. Reproducibility and method-selection ledger
| ID | Source or artifact | Version / accessed | Supports | Limits |
|---|---|---|---|---|
| SRC-01 | [OWASP GenAI Security Project, LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | 2025 page, modified 2025-04-17; accessed 2026-11-07. | Direct and indirect injection, least privilege, human approval, external content segregation. | General guidance, not proof this hypothetical gate works. |
| SRC-02 | [NIST CSRC least privilege glossary](https://csrc.nist.gov/glossary/term/least_privilege) | Live glossary; accessed 2026-11-07; cites SP 800-53 Rev. 5. | Minimum required authorizations and resources for each entity. | Does not dictate a ticket design or establish legal applicability. |
| SRC-03 | [OpenAI, Safety in building agents](https://developers.openai.com/api/docs/guides/agent-builder-safety) | Live guide; accessed 2026-11-07. | Untrusted content, constrained structured data flow and approval are defenses in depth, not guarantees. | Vendor documentation is not evidence of deployment or efficacy here; product-specific details may change. |
| INV-01 | `Train/Templates/directory.txt`, lines 86–94 | Repository snapshot, accessed 2026-11-07. | Topic inventory: execution surfaces, identity, approvals, isolation, blast radius. | Does not expose module content or `code/` contents. |
| RUN-01 | No run exists. | Not executed. | None. | Calculations below can be checked independently; no system behavior is reproduced. |
| CFG-01 | Proposed model/prompt/policy/index/connector bundle. | Versions not selected. | Future experiment reproducibility. | Capture immutable IDs and access-controlled artifacts before execution. |
Published security claims are general; all scenario facts, designs, thresholds, and numerical inputs are illustrative.
Do not treat this file as a claim that NIST or OWASP certified the proposed implementation.

## 6. Analysis Walkthrough
The decisive tradeoff is architectural: let the model prepare an intent, but do not let it manufacture authorization or select a different write after review.
No system has been exercised; the following is a worked trace and arithmetic under hypothetical assumptions, not a measured result.
| Step | Decision and input snapshot | Operation | Intermediate output | Interpretation and evidence |
|---|---|---|---|---|
| 1 | User U-17 asks to close ticket T-42, revision 8. | Check user may view ticket; retrieve permitted comments. | Authorized snapshot at revision 8. | The source text may still be hostile; illustrative. |
| 2 | Ticket comment contains a request to send all records elsewhere. | Treat text as data; deny external destination tool. | Candidate output cannot itself authorize any export. | Independent privileges constrain damage, not model behavior; SRC-01/02. |
| 3 | Model proposes status=closed, ticket T-42, revision 8. | Canonicalize typed fields and hash complete intent. | Digest H1; display exact effect and evidence. | Digest is binding only if execution verifies the same bytes and semantics. |
| 4 | Reviewer R-9 approves H1 with policy P3. | Check role, separation of duties, MFA session, expiry; persist approval. | Approval references H1 and expiry. | A human can err; policy cannot be waived by clicking. |
| 5 | Executor consumes approval for H1. | Recheck role, requester access, policy, expiry, ticket revision, and action hash. | Allowed only if all remain current. | Deny on stale policy, changed revision or missing authority. |
| 6 | Sink receives operation key K1 with conditional revision 8. | Record send intent; make scoped idempotent request; reconcile receipt. | Confirmed receipt or UNKNOWN, never assumed success. | Sink guarantees are unverified; illustrative. |
The intent digest is not evidence of source truth or reviewer comprehension.
The highest-leverage step is the fresh execution-side comparison of approved intent to the actual sink call, including destination and revision.
If the sink does not support idempotency or authoritative receipts, the recommended branch changes to draft-only/manual rather than blind retry.
**Worked reviewer capacity:** 240 eligible requests per day × 30% producing one write proposal each = 72 approvals per day.
At 2.5 minutes per approval the expected active review work is 72 × 2.5 = 180 minutes, or 3 reviewer-hours per day.
At 4 productive review hours per reviewer-day, this consumes 3 / 4 = 0.75 reviewer-days before breaks, variance, escalations, and coverage.
At 50% writes the workload becomes 240 × 0.50 × 2.5 / 60 = 5 reviewer-hours, or 1.25 reviewer-days.
At 5 minutes per review and 30% writes the workload is 72 × 5 / 60 = 6 reviewer-hours, or 1.5 reviewer-days.
These are capacity arithmetic, not queueing predictions; no arrival distribution, coverage, or service-level guarantee is implied.
**Worked exposure:** if a scoped identity can edit at most 1 ticket per approval and 10 approvals remain valid, the gate's maximum pending approved action count is 10, not necessarily 10 actual side effects.
A ticket may trigger multiple downstream automations, so the real consequence can exceed the number of ticket writes; A-05 must be verified.
**Illustrative economics:** at an assumed fully loaded $60/reviewer-hour, 3 hours/day costs $180/day in review labor only.
At 20 working days, this is $3,600/month; model, storage, engineering, egress, ticket API, failed work and overhead are unpriced.
The $60 figure is a teaching assumption, not a sourced market price or procurement estimate.
A decision-grade estimate must replace every assumed unit price with dated local contracts and include all omitted costs.
| Sensitivity | Original | Alternative | Consequence | Limit |
|---|---|---|---|---|
| Write fraction | 30%, 3 hours/day. | 50%, 5 hours/day. | Staffing requirement rises by 2 hours/day. | Assumes identical average review duration. |
| Review time | 2.5 minutes, 3 hours/day. | 5 minutes, 6 hours/day. | Review work doubles. | No queueing or approval-quality model. |
| Sink semantics | Idempotent and queryable, unverified. | Neither property available. | No unattended replay after uncertainty; manual path. | An application ledger alone cannot prove no external effect. |
| Authorization | Revocation visible at send. | Stale policy cache. | Fail closed or avoid automated execution. | No revocation propagation time measured. |
No alternative was actually tried; no negative or positive experiment result is asserted.

## 7. The Finding
Conditionally choose an exact-action approval gate only if the ticket platform, identity system, and reviewers can meet the stated prerequisites; otherwise retain manual or draft-only handling. No relative safety, throughput, or cost outcome has been measured against the manual baseline. The numerical examples describe workload and costs under constructed assumptions, not business impact. Before granting writes, validate action binding, current authorization, connector recovery, downstream effects, and independent reviewer performance.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
No model training is planned by this exercise; data preparation is for evaluation and policy release only.
Proposed offline flow: authorized, redacted ticket samples -> provenance and duplicate checks -> grouped temporal split -> attack and benign scenario curation -> blinded adjudication -> versioned test bundle -> security and operations release decision.
Maintain a separate restricted adversarial corpus; do not expose holdout payloads to prompt authors.
Freeze action schema, policy, model, prompt, retrieval index, parser, reviewer UI and connector versions as one tested release bundle.
Each bundle has an immutable manifest and compatibility rules for pending approvals.
Test rollback before rollout; a rollback cannot automatically undo a write already committed to the ticket platform.
A scheduled policy or index refresh invalidates approval when it can alter action meaning or authorization.
No hardware, parallelism or checkpoint numbers are asserted; a future implementation must measure its workload.
### 8b. Online / Serving / Real-Time Path
```text
[Employee request + identity] -> [AuthN + ACL] -> [ticket retrieval: untrusted content]
                                      |                     |
                                      v                     v
                               [deny + audit]        [model: proposal only]
                                                            |
                                                            v
                          [schema + policy checks; canonical intent + digest]
                                      |                     |
                                      v                     v
                              [reject/manual]   [reviewer UI + immutable approval ledger]
                                                            |
                                                            v
                         [executor: recheck identity, policy, expiry, revision, digest]
                                      |                     |
                                      v                     v
                               [deny + audit]   [conditional ticket write with key K]
                                                            |
                                      +---------------------+------------------+
                                      v                                        v
                              [confirmed receipt]                     [unknown: reconcile, freeze]
                                      |                                        |
                                      +---------------------> [audit + incident owner]
```
The model cannot call the ticket write API, access the executor credential, change the authorization result, or approve its own proposal.
The retrieval gateway enforces source permissions before data reaches the model; execution repeats the platform-side check.
The approval UI renders validated, canonical destination, operation, ticket, before/after values and evidence provenance rather than a model-authored summary alone.
A separate identity provider authenticates the reviewer; approval is a signed or integrity-protected server-side record, not a bearer token in model context.
The policy service checks both requester eligibility and the reviewer's current permission at execution, including conflict-of-interest rules.
A fresh ticket revision is checked in the same conditional sink operation; a separate preflight check alone does not close the race.
Any semantic change, policy change that affects authorization, revision change, or approval expiry returns the action to review.
A proposed timeout must be bounded; on timeout the system returns pending or manual handling, never an invented confirmation.
At cold start, unavailable policy keys or entitlement state cause deny; no last-known permissive state is accepted.
During peak load, backpressure stops new agent writes and preserves the manual workflow; do not silently skip review.
If the sink outcome is unknown, freeze this action, reconcile by operation key and receipt, and avoid a fresh key or blind resend.
### 8c. Key Design Decisions
| ID | Choice | Alternative | Evidence / tradeoff | Failure boundary / revisit |
|---|---|---|---|---|
| DEC-01 | Approve canonical exact intent. | Approve whole conversation. | Proposed; finer control but more reviewer work. | UI or executor mismatch; any schema change. |
| DEC-02 | Execute with restricted service identity plus delegated authority checks. | User's broad credential in model. | Proposed least privilege [SRC-02]; integration cost. | Native platform cannot enforce scope; connector change. |
| DEC-03 | Recheck at send, use conditional revision. | Only check at proposal creation. | Proposed protection against stale approval. | Sink lacks conditional operation; API change. |
| DEC-04 | Stable action key and explicit UNKNOWN. | Retry with new key on timeout. | Proposed recovery discipline; may reduce availability. | Sink cannot deduplicate/query; fault test. |
| DEC-05 | Human review all writes in scope. | Risk-scored automatic approval. | Proposed hard scenario constraint; added latency. | Review fatigue or outage; study and capacity change. |
### 8d. Architecture artifacts and operational readiness
**Canonical intent fields:** tenant ID; ticket ID; expected revision; operation enum; typed payload; requester ID; action ID; proposal version; policy version; schema version; expiry; and any downstream-effect category.
Canonicalization defines stable field ordering, Unicode normalization rules, typed numeric representation, absent versus null semantics, and rejection of unknown fields.
Digest must cover every field that can change the effect, including destination and version; select a modern collision-resistant hash through organizational cryptographic policy.
The server must build both reviewer display and executor request from the same validated typed intent, rather than separately interpreting model prose.
Approval record: digest, immutable action ID/version, reviewer identity, entitlement snapshot reference, authenticated session strength, decision, decision time, expiry, policy version, and audit event ID.
Expiry uses server time; do not accept client-provided timestamps as authority.
The record must be integrity protected and stored transactionally with the state transition to APPROVED.
A reviewer can reject or request a corrected proposal; correction creates a new immutable version and requires a new approval.
A signed approval is not transferable across tenants or destinations, and possession is insufficient without fresh authorization.
**State transitions:** DRAFT -> VALIDATED -> AWAITING_APPROVAL -> APPROVED -> DISPATCHING -> CONFIRMED.
Additional terminal states are REJECTED, EXPIRED, CANCELLED and FAILED; UNKNOWN remains unresolved until reconciliation.
Only server-side transactional compare-and-swap can claim APPROVED -> DISPATCHING for a particular action ID.
A worker crash after claiming does not itself prove whether the sink received the request.
The stable key is derived from a server-generated action ID and tenant, with a unique constraint; it is never regenerated for retries of that action.
If the sink's deduplication retention window expires before resolution, do not retry automatically.
A confirmed receipt must be verified against target, payload digest, action key, and actual platform revision where supported.
If sink receipt does not contain enough information, reconcile the resulting state against authoritative audit history; if ambiguous, hold for a human.
A compensation is a new, separately authorized operation; do not assume it reverses notifications, automation or externally observed effects.
**Identity:** user, reviewer, retrieval service, model service and executor each have distinct principals and scoped permissions.
The requester cannot approve their own action under the illustrative separation-of-duties policy.
Emergency access requires a separately governed manual break-glass procedure; this proposal does not create an automatic override.
Revoke executor credentials and block egress at the gateway when a suspected bypass or cross-tenant leak occurs.
**Tenant isolation:** tenant-scoped storage, keys, authorization decisions, network paths and audit queries support isolation; test every boundary for cross-tenant access.
Tenant ID comes from authenticated server context rather than a model-generated value.
Avoid copying tenant secrets into model prompts or logs; short-lived scoped credentials stay in the executor.
**Sandbox:** parsers run without privileged mounts, ambient credentials or unrestricted network; enforce CPU, memory, file count and wall-time caps.
Sandboxing limits parser compromise, but it does not make parsed text trustworthy or validate business intent.
**Observability:** log action ID, digest, tenant pseudonym, policy decision, reviewer, version bundle, state transition, sink receipt and trace ID.
Keep source content out of routine logs; protect and separately retain minimal forensics evidence according to privacy policy.
Alert on write without matching approval, digest mismatch, cross-tenant denial, unknown outcomes, queue age, abnormal reject rates and reviewer anomalies.
Incident owner receives action IDs and safe links rather than secrets or attacker payloads in alert bodies.
**Service indicators:** proposed monthly SLI is eligible requests reaching either confirmed completion or an explicit safe manual handoff within one business day, divided by all eligible requests.
Illustrative target is 95% over a calendar month, measured at the gateway; security deny events remain in the denominator with their manual-handoff outcome.
Security invariant violations are a separate zero-tolerance release stop, not absorbed into the availability objective.
A proposed 5% monthly error budget reaching exhaustion freezes feature expansion and invokes service-owner review; it never relaxes authorization.
**Resilience:** short, bounded retries with jitter apply only to idempotent reads and verified same-key writes within sink guarantee windows.
Circuit breakers stop repeated sink failures; queue depth and oldest age trigger backpressure and manual routing.
Timeouts, retry budgets, load limits, escalation time and staffed coverage require measured contracts before launch.
**Capacity illustration:** 240 requests/day and 72 proposed writes/day are daily averages, not peak rates.
If a 24-KiB redacted approval and audit envelope is retained per write, 72 × 24 = 1,728 KiB/day.
At 365 days this is 630,720 KiB, approximately 616 MiB before indexes, replication, backups and deletion overhead.
If three copies are retained, logical payload alone is about 1,848 MiB; compression and encryption metadata are not included.
A burst of 20 reviews arriving in 10 minutes is not handled by daily averages; measure reviewer service-time distribution and queue age.
A 10-fold arrival increase raises incoming work to 720 reviews/day, or 30 reviewer-hours and 7.5 reviewer-days at four productive review hours per reviewer-day, under unchanged assumptions.
No end-to-end p99 is calculated from component percentiles; measure latency from request acceptance to safe disposition.
**Deployment:** begin with draft-only shadow evaluation; no writes or hidden automatic approval.
Only after explicit owner approvals, run a small opt-in cohort with monitored manual fallback and independent review.
A canary must include policy, model, index, UI, executor, parser and connector versions together.
Rollback first blocks new sends, then drains or expires approvals; incompatible schema or policy versions require a new approval.
Do not replay old approved messages across a bundle change unless semantics, permissions, expiry and exact intent are revalidated.
**Recovery:** restore the approval ledger and executor checkpoint, then reconcile against sink audit records before resuming dispatch.
Recovery point objective and recovery time objective are unselected; set them after measuring data-loss and manual-work consequences.
If the ledger is restored to before a sink write, reconcile by stable action key rather than assuming the missing ledger record means no effect.
The security lead decides incident containment; ticket owners correct state using independently authorized changes.

### Execution contract: field-level design
The following fields and rules are a proposed implementation contract, not a claim about a particular vendor API.
The authorization decision is computed from authenticated context and canonical intent; neither is trusted merely because the model emitted it.
| Field | Source of truth | Validation / security significance |
|---|---|---|
| `tenant_id` | Authenticated server session. | Match resource tenant; reject a client-supplied override. |
| `requester_id` | Identity provider subject. | Verify active status and permission for the requested operation. |
| `reviewer_id` | Authenticated review session. | Verify active reviewer role, scope and separation of duties. |
| `action_id` | Gateway-generated random identifier. | Unique within tenant; bind retries and receipts to this identity. |
| `proposal_version` | Monotonic server-assigned version. | Changes when any intent or material evidence changes. |
| `ticket_id` | Selected through an authorized retrieval. | Confirm it belongs to the server-bound tenant and allowed queue. |
| `expected_revision` | Ticket platform snapshot. | The sink must compare it atomically when applying the change. |
| `operation` | Strict action enum. | Reject arbitrary commands, unknown actions and unrecognized enum values. |
| `status_value` | Allowed finite status enum. | Check transition is permitted from the observed current state. |
| `comment_body` | Validated proposal text. | Enforce size, classification and permitted audience; bind every byte. |
| `comment_visibility` | Platform enum, displayed to reviewer. | Reject public visibility because only internal comments are in scope. |
| `policy_version` | Policy registry. | Bind evaluated policy to the decision and invalidate on material change. |
| `schema_version` | Versioned gateway contract. | Refuse unknown schemas and reapprove after semantic incompatibility. |
| `evidence_refs` | Retrieval gateway's provenance registry. | Include source identity and material revision without granting source authority. |
| `review_deadline` | Server-calculated time. | Check against trusted time at execution, not client-supplied time. |
| `automation_version` | Ticket-platform configuration registry. | A material downstream change invalidates the reviewed effect. |
| `intent_digest` | Server over canonical validated intent. | Recompute from executor input and require equality to approval. |
| `idempotency_key` | Executor derived from action identity and tenant. | Reuse for a retry only under the sink's verified deduplication contract. |
| `receipt_id` | Authenticated sink response or authoritative lookup. | Match the key and target before recording confirmed completion. |
| `trace_id` | Gateway tracing subsystem. | Link logs across services without granting authorization. |

### Canonicalization and interpretation rules
Reject duplicate JSON object keys before parsing; different parsers can otherwise disagree on the value the reviewer approved.
Reject unknown fields rather than silently discarding a field that a downstream connector could interpret.
Enforce a single schema version before computing a digest; never hash raw model JSON and then execute a differently parsed object.
Use strict UTF-8 decoding and reject malformed byte sequences instead of replacing them differently in the UI and connector.
Define Unicode normalization in the schema and keep the normalized text identical in review and execution.
Display escaped control characters and bidirectional overrides so visually misleading content can be recognized.
Preserve a lossless authorized view of the canonical value for reviewers rather than relying only on a shortened preview.
If a comment is too long for full inspection, require an explicit expansion or a safe manual route.
Use integer or enum representations for status values; reject strings that loosely coerce to an allowed enum.
If future action fields include money, represent currency and integer minor units explicitly rather than binary floating-point amounts.
Distinguish absent, empty, and null values according to the sink contract; do not collapse them by convenience.
Include both operation type and resource identity in the hash, even when a payload alone happens to be unique.
Bind the requester's identity because an action authorized for one user is not automatically authorized for another.
Bind the policy and action-schema versions so that unchanged bytes cannot silently acquire new meaning.
Bind evidence revisions when policy requires the reviewer to assess a particular ticket state or document.
Bind the internal-comment visibility so a private note cannot become a public message after approval.
Represent expiration in a single UTC format generated by the server; reject invalid or ambiguous time representations.
Set a maximum field and total intent size before hashing to avoid resource exhaustion.
Perform canonicalization once in a trusted component and persist its result as immutable action data.
Use the same typed object as input to rendering, policy checks, digest calculation and sink request construction.
On any discrepancy between components, stop execution and retain the mismatch for investigation.
Hash equality binds contents; it does not authenticate a reviewer or establish that content is truthful.
A keyed integrity mechanism or signed server record protects the approval ledger from unauthorized modification.
Protect signing keys separately from the model and reviewer browser, with audited rotation and revocation.
Document how old approval signatures are treated when the corresponding verification key is rotated.
Reject a pending approval if the system cannot verify its signature or establish its signing-key validity.

### State machine, transactions, and uncertainty
| Current state | Event | Guard | Next state and required record |
|---|---|---|---|
| DRAFT | Proposal accepted for validation. | Schema parses and requester context exists. | VALIDATED; immutable proposal version stored. |
| DRAFT | Invalid or out-of-scope action. | Unknown action or missing required field. | REJECTED; structured reason, no sink call. |
| VALIDATED | Review requested. | Policy permits submission and evidence is available. | AWAITING_APPROVAL; review packet version stored. |
| AWAITING_APPROVAL | Reviewer rejects. | Valid reviewer session and matching action. | REJECTED; reviewer and reason recorded. |
| AWAITING_APPROVAL | Reviewer approves. | Authority, separation and digest checks pass. | APPROVED; immutable approval stored transactionally. |
| AWAITING_APPROVAL | Deadline passes. | Trusted server time is available. | EXPIRED; no write may be dispatched. |
| APPROVED | Ticket or material evidence changes. | Current revision differs from reviewed revision. | CANCELLED; create a new proposal if still wanted. |
| APPROVED | Permission is revoked. | Fresh policy returns deny. | CANCELLED; retain the reason and decision version. |
| APPROVED | Executor claims work. | Compare-and-swap succeeds; approval still valid. | DISPATCHING; durable operation key and worker claim recorded. |
| DISPATCHING | Sink returns authenticated matching receipt. | Receipt identifies the action and intended target. | CONFIRMED; receipt and resulting revision recorded. |
| DISPATCHING | Sink returns definite conditional conflict. | Platform guarantees no write for that response. | FAILED; current revision and safe retry instructions recorded. |
| DISPATCHING | Transport result is ambiguous. | No authoritative proof of commit or non-commit. | UNKNOWN; block new-key retries and open reconciliation. |
| UNKNOWN | Lookup proves matching commit. | Authoritative receipt matches intent and operation key. | CONFIRMED; record reconciliation evidence. |
| UNKNOWN | Lookup proves no commit within sink contract. | Deduplication remains valid and all authorization rechecks pass. | Eligible for controlled same-key retry. |
| UNKNOWN | Lookup is inconclusive. | Receipt or audit history is unavailable or insufficient. | Remain UNKNOWN; owner must investigate manually. |
| Any nonterminal state | Emergency disable. | Authorized security operator invokes stop. | Hold; retain data and forbid new sends. |
The ledger transaction and the external ticket update are not a single atomic transaction unless the sink explicitly provides that property.
An outbox can reliably publish a dispatch request after a ledger commit, but it does not by itself make an external side effect exactly once.
A worker lease can prevent simultaneous local claims, but lease expiry cannot prove that the previous worker did not reach the sink.
The server must establish a stable action identifier before the first external attempt and retain it across recoveries.
An operation key must not be reused for a different intent, even if the first action ultimately fails.
The sink's interpretation of key scope, retention, conflicting payloads and concurrent duplicates must be contractually understood.
If the sink returns a success for a repeated key, the gateway still checks that it corresponds to the originally approved intent.
A lookup indicating “not found” is only decisive if the sink specifies sufficient consistency and retention for that conclusion.
An eventually consistent lookup can temporarily return not found after a successful write; immediate replay could duplicate the effect.
Set the reconciliation delay from measured sink behavior and the actual contract, not from an assumed universal timeout.
The proposed architecture cannot claim exactly-once external side effects when sink semantics are unknown.
A cancellation after dispatch must distinguish an unreceived request from an already committed effect.
No state transition should erase a prior approval, denial, dispatch attempt or reconciliation event from the audit trail.
Events can be redacted or deleted only under an authorized retention policy that preserves required accountability.
The event timestamp records when the gateway observed an event; the platform's commit time may be different.
If clocks disagree, order causally related events using transaction sequence numbers and verified receipt links where possible.
Concurrent revisions of the same ticket may each have valid approvals, but only the matching conditional revision can commit.
Reviewing a replacement proposal creates a new digest and approval; old approval must not be transferred by ticket ID.
An approver's rejection is not a business-policy override and does not authorize a different action.
If the user withdraws the original request, cancel pending work and recheck withdrawal before dispatch.
If withdrawal occurs after a confirmed write, invoke a separately authorized correction rather than claiming the action never happened.
Manual operators need a visible distinction between FAILED, CANCELLED, UNKNOWN and CONFIRMED to avoid unnecessary corrections.
Only CONFIRMED has a verified sink receipt; a user-facing “accepted” message must not be mistaken for completion.

### Trust boundaries and access-control matrix
| Boundary or principal | Allowed capability | Forbidden or conditional capability |
|---|---|---|
| Requester | Submit a request and see authorized disposition. | Cannot create an approval by writing a prompt or ticket comment. |
| Retrieval gateway | Read ticket data after source ACL evaluation. | Cannot authorize writes or disclose restricted attachments. |
| Model process | Produce a proposal in the allowed schema. | Cannot obtain sink credentials or invoke an unrestricted network endpoint. |
| Attachment parser | Extract inert content under sandbox limits. | Cannot access production secrets, host mounts or arbitrary network destinations. |
| Policy service | Return a signed or authenticated allow/deny with version. | Cannot change the reviewed intent or silently approve itself. |
| Reviewer interface | Render exact intent and collect a decision from an authenticated reviewer. | Cannot directly invoke the ticket update using a browser-supplied approval. |
| Approval ledger | Persist immutable transitions and approval provenance. | Cannot confer execution rights without current policy evaluation. |
| Executor | Send a current, approved, scoped intent to the intended sink. | Cannot modify effect-bearing fields or choose a new target. |
| Ticket platform | Enforce its own authorization and conditional update. | Must not accept a gateway assertion as a substitute for platform access checks. |
| Audit reader | Inspect events permitted by their investigative role. | Cannot browse raw source text or secrets without separate permission. |
| Security operator | Pause dispatch and revoke compromised credentials. | Cannot use the emergency stop as a ticket-write authorization bypass. |
| Integration maintainer | Release a reviewed connector bundle. | Cannot silently reinterpret pending approved schema versions. |
The service account's queue scope is an independent containment layer, not a substitute for requester-level authorization.
Short-lived credentials reduce exposure time but do not make excessive permissions safe during their validity.
An internal comment may be visible to all authorized ticket staff; classification still matters even when the comment is not public.
Read access can leak sensitive data before any write, so approval of side effects does not replace retrieval authorization.
The model endpoint's data-handling terms must be verified before sending even authorized ticket content.
Tool output is also untrusted data: a compromised connector response may attempt to instruct the model.
The policy service must not accept an authorization result embedded in a ticket, model output or browser field.
The reviewer interface must make the source of each statement distinguishable from the policy decision.
When the reviewer cannot access a required source, the action is not ready for informed approval.
Approval of an intent does not confer new permission on the requester or on the reviewer.
A reviewer cannot enlarge the executor's platform scope by approving an out-of-scope action.
External notifications and automation triggered by an otherwise internal status change belong in the effect inventory.
The allowed action list must be revisited when ticket platform configuration or data classification changes.
Rate limits belong to the server and connector, not just the agent prompt, because prompt instructions can be ignored.
Admin tools, bulk export, arbitrary URLs and shell access remain unavailable in this bounded case.
Approvals must be impossible to inject into an execution queue directly from a retrieved document.
The reviewer UI must reject cross-site request forgery and bind the decision to the authenticated session.
Session renewal does not extend the approval deadline unless a new approval is explicitly obtained.
Machine-to-machine calls require authenticated transport and narrowly scoped service authorization.
An attacker with ticket-edit rights should not be assumed to have policy-administration rights.
If the same administrator controls the model, policy and audit store, the trust separation is weakened and must be recorded.
Use separate administrative roles and independent security audit where the organization can support them.
Periodic entitlement reviews should find dormant reviewer roles and overbroad executor grants.
Privileged break-glass access is investigated independently and must not silently produce an agent approval.

### Review packet and adjudication details
The packet starts with the requester's original request, clearly labeled as a request rather than an instruction to the reviewer.
Show the ticket identifier, queue and tenant using values resolved by the platform rather than model-generated labels.
Show the current ticket revision and its timestamp so the reviewer can recognize a potentially stale snapshot.
Show the proposed operation as a verb and the exact before and after values.
Show internal versus public visibility prominently for comment actions.
Show the full intended comment in a safe, inert renderer, with a means to inspect any truncated text.
Show evidence references with source owner, retrieval time and source revision where available.
Clearly label external or user-authored snippets as untrusted even when they appear in an internal ticket.
Show the relevant deterministic policy decision and policy version without claiming policy approval proves business correctness.
Show the identity of the requester and the independent reviewer, subject to privacy and access controls.
Display the approval expiry and explain that expiration requires a fresh review.
Present reject and request-revision as first-class choices, not visually hidden alternatives to approve.
Do not use an urgent or manipulative instruction from a document to modify the interface's priority.
Do not render document-provided buttons, links or scripts as trusted review controls.
Provide the reviewer an authorized route to inspect the original ticket when a summary omits context.
If evidence access fails, surface that absence explicitly rather than substituting a model's confident summary.
Require a reason code for rejection or escalation, with optional minimal free text under retention policy.
Record the exact packet version that was displayed, not only the version that was generated.
If display rendering fails or fields cannot be verified, deny approval and route the request manually.
Test reviewers on adversarial examples that preserve plausible business context, not only obvious “ignore instructions” strings.
Include benign urgent requests in the test set so reviewers are not rewarded for rejecting every time-sensitive item.
Include cases in which source facts are false but the requested action is structurally valid.
Include cases in which a correct business goal is paired with an unauthorized ticket or operation.
Train adjudicators on independent policy snapshots without exposing the held-out attack payloads to prompt developers.
Sample both approved and rejected proposals when estimating reviewer errors to avoid verification bias.
Measure disagreement on both authorization and business appropriateness; they are related but distinct judgments.
Reviewers' confidence is not an independent ground-truth label and must not be used alone to adjudicate correctness.
Escalate uncertain cases to a separate role; do not force a binary approve/reject when evidence is incomplete.
Monitor workload distribution by reviewer to find concentration that could produce fatigue or coercion.
An unusually fast approval is a triage signal, not proof of malicious conduct or an automatic personnel finding.
Protect reviewers from attacker-controlled harassment embedded in ticket text while retaining evidence needed for security analysis.
When a reviewer's role changes, cancel or reassess pending approvals according to policy rather than relying on a nightly batch.
Correlate reviewer exceptions with ticket type and action type to detect systematic interface or policy ambiguity.
Retain decision examples for calibration only with approved access, redaction and retention controls.
An adversarial reviewer study should report the sampling frame and confidence limitations, not a single decontextualized accuracy percentage.

### Cost, throughput, and blast-radius extensions
All values in this subsection are additional hypothetical arithmetic, not vendor prices, observed measurements or deployment estimates.
Assume for a planning exercise that 72 proposed writes arrive evenly across an eight-hour staffed day.
The mean arrival rate under that simplifying assumption is 72 / 8 = 9 proposals per hour.
At 2.5 minutes of active handling, one reviewer has a theoretical maximum of 60 / 2.5 = 24 reviews per active hour.
At four productive review hours per eight-hour day, that reviewer can handle 4 × 24 = 96 reviews per day in this simplified model.
Daily utilization of that illustrative productive capacity is 72 / 96 = 75%, before variation, escalations and service interruptions.
With 120 reviews per day at the same handling time, the requirement is 120 × 2.5 / 60 = 5 productive hours.
That exceeds the assumed four-hour productive allocation for one reviewer by one hour per day.
Adding one reviewer can increase capacity, but coverage, scheduling and training determine whether the queue actually meets a deadline.
Do not treat average utilization as a queueing delay estimate; bursts and heavy-tailed review times can dominate waits.
If 10% of 72 reviews are escalated and each escalation adds 12 minutes, escalation work is 72 × 0.10 × 12 = 86.4 minutes per day.
Adding that escalation work to the original 180 minutes gives 266.4 minutes, or 4.44 reviewer-hours per day.
At the illustrative $60 per reviewer-hour, 4.44 hours costs $266.40 per day before all non-labor costs.
For 20 working days, that illustrative labor component is $5,328 per month.
The escalation arithmetic assumes every escalated case consumes 12 additional minutes; shared or parallel work would change the estimate.
If 5% of 72 requests are regenerated once, the expected added proposal count is 3.6 per day under the simplifying average.
At 2.5 extra review minutes each, this adds 9 reviewer-minutes per day, excluding model and connector costs.
Do not round expected daily averages into a claim that a fractional actual ticket occurs.
At 72 writes per day, a 1% duplicate-effect rate would correspond to 0.72 duplicate effects per day on average if applied uniformly.
That number is a hypothetical conversion, not an observed rate or an acceptable target.
If one ticket closure triggers three independent downstream notifications, one mistaken write can initiate three messages.
Those messages may be copied or acted upon, so the count of initial notifications is not a complete harm bound.
If ten approvals are pending and each is restricted to one ticket, at most ten gateway-authorized ticket writes remain pending.
This bound excludes direct platform bypasses, broader stolen credentials and uncounted downstream automation.
If each pending approval expires after 30 minutes, its exposure window is at most that duration only when time and revocation checks work.
The 30-minute value is an illustrative design choice; no measured reviewer delay supports adopting it.
If average processing exceeds the expiry, repeated reapproval increases workload and may incentivize unsafe shortcuts.
If four service instances each hold a queue-scoped credential, compromise of one instance must not expose credentials for other tenants.
Credential rotation frequency must be chosen from actual issuance capability and incident-response needs, not this arithmetic.
For storage, 616 MiB per year of primary logical audit envelopes excludes the source documents and retrieval index.
Three logical copies yield about 1,848 MiB per year before backup history and storage-system metadata.
If retention is shortened or increased, privacy, investigation and legal-hold decisions must be made by the appropriate owners.
Cloud storage price, network egress, compute, model tokens and sink API fees remain unpriced until dated contracts are selected.
Compute cost per useful outcome must divide total attributable spend by correctly completed eligible requests, not by all attempted tool calls.
Manual comparison costs must include equivalent reviewer and remediation work for the same population and period.
An unresolved write imposes investigation cost even when the system correctly prevents a duplicate retry.
An aggressive safety gate can reduce throughput; its business tradeoff must be measured rather than asserted.

## 9. Evaluation & Validation
All protocols in this section are **planned and not executed**; no run ID, baseline value, observed confidence interval, or deployment result exists.
The primary safety metric is unauthorized side effects per 1,000 attempted agent-originated writes, counting confirmed effects that violate applicable authorization at execution.
The denominator includes allowed, denied, failed, timed-out and unresolved write attempts; list unresolved cases separately and adjudicate before declaring a pass.
Report attack and legitimate-request denominators separately. The attack-control rate is confirmed unauthorized effects divided by adjudicated adversarial attempts; its denominator excludes ordinary requests because it asks whether a hostile input crosses the boundary. The legitimate-service rate is correctly completed or safely handed-off eligible legitimate requests divided by all eligible legitimate requests; denials, expiries, and unresolved actions stay in this denominator. Report an unknown-outcome count and rate separately rather than placing UNKNOWN in either numerator.
An attacker may submit many near-identical drafts, so report both attempt-weighted and ticket-cluster-weighted attack results, with the cluster definition and source family. A detector that rejects every request can look good on the attack denominator while failing the legitimate-service denominator.
For every rate, record numerator event definition, denominator population, calendar window, tenant scope, adjudication status, comparator, and whether it was executed. No proposed zero-failure result is an assurance claim.
A second metric is policy-bypassing execution count, with a target of zero in the test corpus; zero is a release criterion, not proof of zero real-world risk.
A useful-work metric is the fraction of eligible requests correctly completed or safely handed to manual work within one business day.
Measure review duration, reviewer disagreement, false denial, time to containment, sensitive-data disclosure, duplicate sink effects, and cost per correctly completed request.
A proposed four-week shadow cohort compares the existing manual flow and draft suggestions on the same eligible population where privacy and consent permit.
Do not interpret shadow approvals as actual writes or as a production safety rate.
After shadow evidence, a pilot could randomize eligible requests by ticket, not by individual repeated event, to manual versus gated assistance.
Record eligibility, exclusions, assigned cohort, crossover, missing outcomes, interference, sample-ratio checks and a prespecified stopping rule.
A shared reviewer pool and downstream ticket automations may cause interference; use cluster-aware intervals or a design that separates pools.
Choose practical thresholds and sample size before exposure; no calculated power is available because baseline rates and effect sizes are unknown.
Reviewers adjudicating safety are blinded to treatment where feasible and use a versioned rubric with policy and evidence snapshots.
Two independent reviewers classify a sample; disagreements go to a senior adjudicator, with all changes logged.
The model must not judge its own safety without calibration against independent human labels.
Stratify by direct/indirect injection, tenant, language, ticket age, attachment, permitted action, access boundary and sink failure.
The final attack holdout must be created independently of the prompt tuning process and include benign hard negatives.
Replay tests use a non-production sink with safe canaries; no real exfiltration or customer notification is needed to test the control.
Attempt to mutate the approved payload at every boundary and verify that the sink observes no write on mismatch.
Fault-inject timeout before send, response loss after commit, duplicate dispatch, reviewer revocation, policy change and restored stale state.
For zero events in N independent representative trials, the rough one-sided 95% upper bound is approximately 3/N; independence and representative sampling are strong assumptions.
For example, zero failures in 300 independent trials gives an approximate upper bound of 1% per trial, not proof of impossibility.
Correlated payloads, adaptive adversaries, and selection bias invalidate a naive interpretation of that bound.
Release requires zero known invariant violations, complete reconciliation of unknown outcomes, security sign-off, supported sink guarantees and an independently reviewed rollback plan.
Actual thresholds for acceptable useful-work rate, latency and staffing require stakeholder agreement before the study.
A failing safety test blocks writes; an inconclusive study or unavailable sink evidence returns to draft-only or manual handling.

## 10. Action Taken
No action taken as of 2026-11-07; this is an illustrative design exercise, not a real organization decision.
Recommendation: evaluate a draft-only assistant, then consider gated ticket writes only after the prerequisites and release tests above are satisfied.
No reviewer, security owner or platform owner has approved deployment; no effective scope, release date, or configuration exists.
Proposed implementers are the integration, identity, security and operations teams, with an owner-assigned runbook before any pilot.
Security can stop new sends, revoke the executor credential and isolate the gateway; operations routes requests to manual work.
Rollback cannot undo already committed effects and must reconcile unresolved operations before any resumption.
Autonomous external messaging, multi-ticket actions and self-approval are excluded.

## 11. Outcome & Results
**Outcome not yet measured.** Figures elsewhere are constructed teaching values, not measurements of any named organization or evidence of realized impact.
| Metric | Baseline / window / N | Treatment / window / N | Change | Uncertainty | Evidence and limitation |
|---|---|---|---|---|---|
| Unauthorized effects per 1,000 attempted writes | Manual comparison not measured. | Not deployed; N unavailable. | Not estimable. | Not estimable. | CLM-01, not-measured; unresolved outcomes require adjudication. |
| Useful safe disposition within one business day | Not measured. | Not measured. | Not estimable. | Not estimable. | CLM-02, not-measured; requester experience may differ by cohort. |
| Review labor and end-to-end costs | Not measured. | Not measured. | Not estimable. | Not estimable. | CLM-03, not-measured; Section 6 gives assumptions only. |
| Duplicate side effects | Not measured. | Not measured. | Not estimable. | Not estimable. | CLM-04, not-measured; sink guarantees unknown. |
Security lead owns measurement design; operations owns service metrics; platform owner owns sink reconciliation.
Proposed measurement window is four weeks of shadow activity, followed only if authorized by a defined pilot window.
Decision trigger: all invariants and prerequisites pass, with adequate reviewer capacity and no unresolved write outcomes.
| Claim ID | Exact scope | Evidence | Supports / does not establish | Review trigger |
|---|---|---|---|---|
| CLM-01 | No measured unauthorized effect rate exists for this design. | Not-measured; Sections 6 and 9. | Conditional recommendation only / no effectiveness. | First real adjudicated evaluation. |
| CLM-02 | No measured useful-work outcome exists. | Not-measured; Section 9. | Need comparator / no productivity claim. | Cohort study. |
| CLM-03 | 72 × 2.5 minutes = 3 hours/day under stated assumptions. | Illustrative; Section 6. | Staffing sensitivity / no actual cost. | Workload or labor price changes. |
| CLM-04 | Unknown sink result cannot safely be treated as no effect. | Design reasoning, A-02 and Section 8. | Freeze and reconcile / no exactly-once claim. | Connector contract and fault test. |

## 12. Failure Modes & What Went Wrong
No incident is claimed. This is a prospective fault model: every row is a distinct hypothesis to test with synthetic tickets and a non-production sink. “Deny” means no automated ticket write; it does not mean that the requester’s underlying need was resolved.

### 12a. Failure diagnosis and recovery matrix
| ID | Trigger and discriminating check | What can go wrong | Immediate safe state | Repair and regression oracle | Accountable owner |
|---|---|---|---|---|---|
| F-01 | A retrieved comment asks for an export; inspect proposed operation and network trace. | Indirect prompt injection is treated as execution authority. | Reject proposal; preserve source revision. | Keep content untrusted and expose only the ticket-action enum; no export call is possible. | Security lead |
| F-02 | Requester selects a ticket outside their ACL; compare authenticated subject at retrieval and send. | A client or model claim substitutes for access control. | Deny without returning ticket data. | Query platform ACL twice; denied subject receives neither content nor a write. | Identity owner |
| F-03 | Ticket ID belongs to another tenant; compare authenticated tenant to platform tenant. | Cross-tenant read or write through an unbound identifier. | Deny and investigate repeated attempts. | Tenant comes from server context; a foreign identifier is unreachable. | Security lead |
| F-04 | Parser exits abnormally or attempts network access. | Attachment parsing reaches host resources or credentials. | Quarantine attachment and terminate sandbox. | Remove mounts, credentials and egress; canary file and network accesses fail. | Integration owner |
| F-05 | Look-alike identifiers or bidi controls appear in a review packet. | Reviewer sees a different resource from the one serialized. | Hold action and show escaped canonical identifiers. | Normalize and render controls; changing either identifier changes the digest. | UI owner |
| F-06 | Document markup produces an active control or priority styling. | Source text manipulates the reviewer interface. | Render inertly and remove active content. | Enforce a strict renderer and content policy; scripts and links cannot act. | UI owner |
| F-07 | Model says a user is privileged; compare against identity decision. | Generated prose is accepted as authorization evidence. | Deny the unsupported proposal. | Use only authenticated identity and policy; model text cannot change allow/deny. | Policy owner |
| F-08 | Executor payload differs from the approved typed object. | Approval substitution changes target, operation, payload, or visibility. | Stop dispatch and cancel approval. | Recompute digest over executor input; every effect-field mutation denies. | Executor owner |
| F-09 | Ticket revision changes after review; sink returns conditional conflict. | A stale action overwrites newer ticket state. | Mark FAILED, retain conflict details, require a new proposal. | Atomic `expected_revision` condition rejects revision eight after revision nine exists. | Ticket-platform owner |
| F-10 | Reviewer role is revoked before dispatch. | Historical approval is mistaken for current authority. | CANCELLED, not CONFIRMED. | Fresh entitlement check denies revoked reviewer. | Identity owner |
| F-11 | Requester access is revoked while work waits. | Agent continues acting for a user no longer entitled. | CANCELLED and manual handoff. | Execution rechecks requester access as well as reviewer access. | Identity owner |
| F-12 | Material policy version changes while an approval is pending. | Unchanged bytes acquire an unauthorized policy interpretation. | Cancel affected approvals. | Current policy decision must match; changed material rule requires review again. | Policy owner |
| F-13 | Server time is after the approval deadline or clock health is unknown. | An old intent executes after its validity window. | EXPIRED or deny on time uncertainty. | Trusted-time check immediately precedes dispatch; no sink request follows expiry. | Platform operations |
| F-14 | Two tabs or workers race to claim one action ID. | Duplicate local dispatch is created before the sink sees a key. | One worker claims; loser records no send. | Transactional compare-and-swap permits one APPROVED-to-DISPATCHING transition. | Executor owner |
| F-15 | Sink commits but response is lost; receipt lookup finds matching key. | A timeout is falsely labeled failure and blindly resent. | UNKNOWN until lookup. | Reconcile same action key and matching digest to CONFIRMED without a new operation. | Integration owner |
| F-16 | Worker crashes before known transmission; transport cannot prove non-receipt. | System fabricates success or sends a new key. | UNKNOWN and frozen. | Retain action/key before dispatch; retry only the same key under verified semantics. | Executor owner |
| F-17 | Key is older than sink deduplication retention. | Same-key replay may no longer deduplicate. | Remain UNKNOWN; prohibit automatic replay. | Compare key age to contract; route evidence to human reconciliation. | Integration owner |
| F-18 | Sink has neither receipt lookup nor authoritative action history. | Commit status cannot be distinguished from absence. | Disable automated confirmation and retry. | Exclude this sink from unattended writes; manual reconciliation is the only recovery. | Service owner |
| F-19 | Ticket automation inventory changes after approval. | An internal update now creates an unreviewed downstream effect. | Cancel action class until re-reviewed. | Bind material automation category/version; changed effect inventory blocks dispatch. | Ticket-platform owner |
| F-20 | Retrieved runbook contradicts policy registry or has changed revision. | Poisoned source is treated as policy. | Quarantine source and re-review. | Policy comes from its registry; source revision cannot alter authorization. | Policy owner |
| F-21 | Search returns restricted ticket material. | Retrieval leaks data before the approval gate. | Stop retrieval path and notify data owner. | ACL-filter at query and render; restricted text never reaches model context. | Retrieval owner |
| F-22 | Summary lacks a contradictory source; inspect authorized original. | Reviewer approves from incomplete evidence. | Escalate rather than force approval. | Packet exposes provenance and authorized original; missing evidence is visible. | Operations owner |
| F-23 | Fast approvals cluster by reviewer; sample decisions and durations. | Fatigue, coercion, or poor UI creates rubber-stamping. | Pause that review queue and audit recent decisions. | Capacity limits and blinded calibration detect unsafe throughput; speed alone proves nothing. | Operations owner |
| F-24 | Session or MFA provenance is anomalous, then account is revoked. | A compromised reviewer creates apparently valid approvals. | Revoke session and suspend its pending actions. | Fresh session/role check blocks dispatch; investigate rather than infer intent. | Security lead |
| F-25 | Requester and reviewer resolve to the same stable subject. | Separation of duties exists only in the UI. | Reject and assign an independent reviewer. | Policy denies self-approval even when both roles are present. | Policy owner |
| F-26 | Executor secret appears in model trace or is used from a non-executor path. | Model environment can bypass the gate. | Revoke credential and block connector egress. | Secrets remain only in executor; model runtime cannot read or use the credential. | Security lead |
| F-27 | Policy service returns unavailable rather than explicit deny. | Availability pressure converts unknown authorization into allow. | Deny automated send; manual handling. | No permissive cache is accepted; outage produces zero automated writes. | Policy owner |
| F-28 | Ledger restore predates a sink commit. | Missing ledger state is interpreted as no external effect. | Freeze workers and reconcile action keys. | Sink audit identifies committed key; record is repaired without a new send. | Platform operations |
| F-29 | Sink audit has a write with no approval chain. | Direct bypass, delayed audit, or restoration loss is prematurely misclassified. | Disable writes; preserve ledger, sink, and credential evidence. | Reconcile event ordering before attribution; every confirmed write eventually has a verifiable chain. | Security lead |
| F-30 | Secret canary appears in routine logs or alerts. | Observability multiplies source-data exposure. | Restrict access and rotate any exposed secret. | Logs retain IDs and decisions, not raw source; canary is absent from routine stores. | Privacy owner |
| F-31 | Valid record is replayed under a different tenant context. | Action ID or approval is not tenant-bound. | Reject and isolate client path. | Tenant is in digest, key scope, and authenticated context; cross-tenant replay fails. | Executor owner |
| F-32 | Old approval is decoded by a new schema or connector. | Identical bytes take on new semantics. | Hold or expire pending work. | Compatibility proof is required; otherwise generate a new proposal and approval. | Release owner |
| F-33 | Trusted time source reports unacceptable skew. | Expiry check uses a stale local clock. | Stop sends until clock health returns. | Bound skew monitoring makes an out-of-range clock fail closed. | Platform operations |
| F-34 | Schema review finds an effect-bearing field absent from digest. | Payload or destination can mutate while hash still matches. | Disable action type and audit prior effects. | Schema completeness test mutates every effect-bearing field and expects denial. | Security lead |
| F-35 | Model emits unknown fields or an unsupported operation. | Connector interprets output beyond allowed schema. | Reject before ledger approval. | Strict enum and unknown-field rejection keep it from executor. | Model integration owner |
| F-36 | Injection is split across languages, comments, or attachments. | Detection misses composition and reviewer follows content. | Keep only exact allowed effect eligible for review. | Evaluate mixed-language cases; policy prevents disallowed effect even if detector misses. | Security lead |
| F-37 | Issued token reaches queues outside intended scope. | Least privilege is asserted but not enforced downstream. | Revoke and replace token. | Platform denies an unrelated queue write using the actual issued grant. | Integration owner |
| F-38 | Queue depth or oldest age exceeds staffed boundary. | Backlog drives review bypass or unsafe speed. | Throttle intake and route to manual triage. | Admission control preserves approval; overload never silently skips it. | Operations owner |
| F-39 | User-facing status says complete without a matching receipt. | Safe denial, accepted request, or unknown outcome is mislabeled success. | Correct disposition and retain investigation state. | Only verified matching receipt creates CONFIRMED. | Service owner |
| F-40 | Comment contains data prohibited for its audience. | An approved private note still leaks to authorized ticket readers. | Block comment and contain disclosure. | Classification/output rule runs independently of approval; prohibited secret cannot be added. | Privacy owner |
| F-41 | Review serialization and captured sink call differ. | UI/executor use separate interpretations of intent. | Stop dispatch and invalidate affected approvals. | One typed object supplies display and request; property test finds no divergence. | Executor owner |
| F-42 | Rollback cannot interpret pending newer records. | Recovery replays or drops work with changed meaning. | Freeze pending work. | Bundle compatibility test requires reapproval for incompatible records. | Release owner |
| F-43 | Response claims success but lacks authenticated, matching receipt. | Forged receipt produces false audit certainty. | Mark UNKNOWN and investigate. | Authenticated transport plus receipt/action/key comparison blocks confirmation. | Integration owner |
| F-44 | One principal submits many plausible drafts. | Review capacity is exhausted without a policy bypass. | Rate-limit principal and preserve priority manual route. | Quotas bound queue share; flood cannot create unbounded review work. | Operations owner |
| F-45 | Material evidence revision changes after review. | Correctly bound action rests on obsolete facts. | Cancel and require fresh review when policy marks evidence material. | Evidence revision is bound where needed; changed source forces re-review. | Policy owner |

### 12b. Recovery procedure for an uncertain external effect
1. Stop further dispatch for the action ID; preserve its tenant, canonical intent, digest, version bundle, approval, key, worker lease, and observed timestamps.
2. Move it to UNKNOWN, not FAILED or CONFIRMED. FAILED is appropriate only when the sink authoritatively says the conditional write did not occur.
3. Query the sink with the original tenant-scoped idempotency key and inspect an authenticated receipt or authoritative audit record.
4. Confirm only if target, operation, payload digest, key, and resulting revision agree with the approved intent. A generic “200” response is insufficient evidence.
5. If the sink proves no commit while its deduplication window is still valid, repeat fresh authorization checks and make at most a controlled same-key retry.
6. If lookup is inconclusive, key retention elapsed, or receipt semantics are unknown, retain UNKNOWN and assign human reconciliation. Do not generate a corrective ticket update automatically.
7. Any correction is a new proposal with its own expected revision, current authorization, review, key, and receipt chain.

### 12c. Planned recovery and boundary drills
These drills use synthetic tickets and a non-production sink. They are planned validation work, not observed control results.
| Drill | Setup | Pass condition | Failure decision |
|---|---|---|---|
| D-01 Retrieval isolation | Two users have different access to one synthetic ticket. | Denied user’s model trace and packet contain no restricted text. | Stop retrieval release. |
| D-02 Principal substitution | Authenticated user A is paired with client claim B. | Policy uses A and rejects inconsistent identity. | Fix identity binding. |
| D-03 Self-approval | Requester also has reviewer role. | Same stable subject cannot approve; independent reviewer can. | Fix separation policy. |
| D-04 Payload mutation | Change one approved private-comment character before send. | Digest mismatch causes no sink call. | Disable write path. |
| D-05 Stale revision | Approve revision 8, independently move ticket to revision 9. | Atomic conditional write conflicts; revision 9 is intact. | Require sink conditional update. |
| D-06 Duplicate claim | Deliver one approval event to two workers. | One claim and at most one sink request use the stable key. | Repair ledger claim transaction. |
| D-07 Commit then response loss | Test sink commits then discards response. | UNKNOWN reconciles to matching CONFIRMED without a second key. | Repair receipt query path. |
| D-08 No receipt | Sink offers no receipt or history after dispatch timeout. | No retry or automatic confirmation occurs. | Keep draft-only/manual operation. |
| D-09 Revocation and expiry | Revoke role and separately advance trusted clock before send. | Both paths cancel/expire with zero sink calls. | Repair fresh authorization/time checks. |
| D-10 Automation change | Add synthetic notification after approval. | Material inventory change invalidates approval before notification. | Expand effect inventory. |
| D-11 Restore race | Restore ledger to before a known test commit. | Reconciliation finds the key and prevents replay. | Freeze recovery release. |
| D-12 Capacity overload | Submit more synthetic drafts than admission and review bounds. | Bounded queue/manual route; no approval bypass. | Adjust staffing/admission plan. |

### Planned recovery and boundary drills
These drills are prospective, use synthetic tickets and a non-production sink, and are not assertions of observed behavior.
Each must capture the release bundle, trace, policy snapshot, initial state, exact sink requests and final state.
An independent test operator checks both the absence of forbidden effects and the accuracy of the reported disposition.

**Drill 01 — authenticated retrieval.**
- Set up two users with different access to one synthetic ticket.
- Ask each to retrieve the same identifier using their own authenticated session.
- Expect only the authorized user to receive ticket content, including snippets and metadata.
- Inspect model traces to ensure the denied user's context never contains restricted text.

**Drill 02 — requester substitution.**
- Prepare an otherwise valid action as user A, then send an execution request claiming user B.
- Preserve the authenticated server context for A and the altered client claim for B.
- Expect the policy service to use A's identity and reject the inconsistent request.
- Inspect the audit trail for the server-derived requester rather than a model-produced name.

**Drill 03 — conflicting reviewer.**
- Give the requester a reviewer role and generate an eligible write.
- Attempt approval with the same stable identity in a second session.
- Expect the separation-of-duties rule to reject it despite the role grant.
- Then use an independent authorized reviewer to confirm that ordinary approval remains possible.

**Drill 04 — scope of evidence.**
- Provide one accessible ticket comment and one restricted attachment with related content.
- Construct a request whose answer would be easier if the restricted attachment were included.
- Expect retrieval to exclude the attachment even when the model requests it repeatedly.
- Check that reviewer evidence links do not reveal the attachment's contents or existence beyond policy.

**Drill 05 — indirect destination change.**
- Put a synthetic instruction in an accessible comment asking the agent to send a secret elsewhere.
- Permit only ticket-status and internal-comment actions in the tool registry.
- Expect no outbound message or HTTP request even if the model attempts to propose one.
- Inspect the network boundary as well as the structured proposal validation result.

**Drill 06 — ambiguous action target.**
- Create two synthetic ticket IDs that look similar but are distinct canonical identifiers.
- Ask the assistant to operate on one while the untrusted comment mentions the other.
- Expect the review interface to show the exact canonical ticket and its authorized context.
- Mutating the identifier after review must invalidate the approval digest.

**Drill 07 — effect-bearing text.**
- Prepare an allowed private comment with a benign body and approve its digest.
- Change one character of the body before the executor computes its outgoing request.
- Expect digest mismatch and no sink write.
- Verify that audit records store the original digest and the denied mutation event.

**Drill 08 — absent versus null.**
- Compare an intent where an optional field is absent with one where it is explicitly null.
- Define whether the sink treats these values as equivalent before choosing canonical encoding.
- Expect the validator to reject ambiguous representations if effect equivalence is unknown.
- Confirm neither representation can borrow an approval for a semantically different update.

**Drill 09 — reviewer display completeness.**
- Create an action with every permitted effect-bearing field populated.
- Compare the typed canonical intent, the reviewer screen, and the final serialized sink call.
- Expect all material fields to be visible to the reviewer and identical at execution.
- Require a design fix before launch if display truncation hides a destination or payload.

**Drill 10 — approval expiry.**
- Approve an action with a short test-only time-to-live using a controlled server clock.
- Pause the worker until the expiration instant has passed.
- Expect the state to become EXPIRED and no request to reach the sink.
- Repeat with clock health unavailable and require a safe denial rather than stale time.

**Drill 11 — policy revocation.**
- Approve an action under a policy that permits the selected operation.
- Revoke that permission before the worker claims the approved action.
- Expect a fresh policy check to deny the write and invalidate the pending approval.
- Preserve both policy versions to explain why the earlier approval became unusable.

**Drill 12 — conditional update race.**
- Approve a change against ticket revision eight.
- Update that ticket independently to revision nine before dispatch.
- Expect the ticket platform's conditional write to fail atomically.
- Confirm the action returns for a new proposal and no revision-nine content is overwritten.

**Drill 13 — duplicate worker claim.**
- Deliver the same approved-action event to two executor workers concurrently.
- Both attempt the transactional APPROVED-to-DISPATCHING transition.
- Expect at most one successful claim for that action identifier.
- Inspect the sink call count and verify the losing worker has no independent credential path.

**Drill 14 — committed write with lost response.**
- Configure the test sink to commit the update and then discard its response.
- Expect the executor to classify the action UNKNOWN rather than failed or confirmed.
- Query the sink using the original key and validate the returned effect and target.
- Confirm reconciliation moves the record to CONFIRMED without creating another operation.

**Drill 15 — lost request before commit.**
- Stop the test connection before the sink receives the action.
- Expect UNKNOWN unless the transport and sink can authoritatively prove non-receipt.
- Retry with the original key only within the tested deduplication guarantee.
- Check that the final sink history contains one effect, or that the action stays unresolved.

**Drill 16 — receipt unavailable.**
- Simulate a sink that neither returns a receipt nor offers a queryable action history.
- Cause a timeout after dispatch without revealing whether the sink committed.
- Expect automatic retry and automated confirmation to remain disabled.
- Require a documented human reconciliation or retention of the unresolved state.

**Drill 17 — deduplication window.**
- Approve and dispatch an action using a sink with a known test deduplication window.
- Delay reconciliation until after that contractual window expires.
- Expect the gateway to refuse automatic replay even with the original operation key.
- Verify operations receives the action context needed to investigate manually.

**Drill 18 — ledger restoration.**
- Commit a test sink effect, then restore the approval ledger to a snapshot predating confirmation.
- Keep executor dispatch disabled while comparing platform audit against stable action keys.
- Expect the committed operation to be reconciled instead of sent as a new action.
- Resume processing only after discrepancies and unresolved outcomes are assigned owners.

**Drill 19 — model and parser isolation.**
- Place a canary credential and a private network endpoint outside the sandbox.
- Submit a crafted document that asks the parser or model to access the canary.
- Expect neither the parser nor the model runtime to obtain the executor credential.
- Verify denied network and filesystem events in the sandbox audit without logging the secret.

**Drill 20 — downstream automation.**
- Configure a synthetic ticket closure to trigger an external notification in the test environment.
- Approve closure while the recorded automation inventory says no notification is expected.
- Expect the material automation change to invalidate the pending approval.
- Confirm the notification is not emitted until its downstream effect is explicitly reviewed.

**Drill 21 — authorization outage.**
- Make the policy endpoint unavailable immediately before a pending write.
- Expect the executor to stop without accepting a cached permissive decision.
- Verify that the requester sees pending or manual handling, not a false success.
- Restore the service and require a fresh decision before any dispatch.

**Drill 22 — reviewer session compromise.**
- Approve an action in a synthetic reviewer session, then revoke the account before dispatch.
- Expect entitlement and session revocation to invalidate the pending execution.
- Inspect the audit chain for the original approval and the subsequent revocation.
- Confirm the incident operator can suspend other pending actions from the same account.

**Drill 23 — rollback compatibility.**
- Produce pending approvals using a newer schema and release bundle.
- Roll the gateway back to a version that cannot establish compatible semantics.
- Expect those approvals to be held or expired, not decoded optimistically.
- Confirm any new proposal goes through the current reviewer and policy checks.

**Drill 24 — resource exhaustion.**
- Submit a burst larger than the configured intake and reviewer capacity using synthetic users.
- Measure queue depth, oldest age, per-principal throttling and manual-routing behavior.
- Expect bounded admission and no bypass of approval under load.
- Check that priority handling does not silently starve ordinary authorized requests.

**Drill 25 — log privacy and audit integrity.**
- Include a synthetic secret and injection payload in the source document.
- Execute a denied proposal and inspect routine logs, alerts and restricted forensic storage.
- Expect the routine logs to contain identifiers and decisions but not the secret.
- Confirm authorized investigators can still correlate the trace with the immutable decision chain.

## 13. Transferable Lesson
The initial hypothesis is that exact-action approval plus independent least-privilege enforcement can reduce the consequences of a model following an untrusted instruction.
This remains an unvalidated design hypothesis, not a demonstrated safety improvement.
The mechanism is separating recommendation, authority, approval, and execution, with the exact effect bound across each transition.
Required preconditions include fresh identity and policy, a trusted reviewer UI, sink-side conditional writes and a recoverable side-effect protocol.
A counterexample is a sink that accepts opaque irreversible commands without receipts or deduplication: the safe answer may be manual handling.
Another counterexample is a compromised authorized reviewer: exact binding records the action but does not make the action legitimate.
A portability test asks whether every effect, destination, dependency and actor can be identified and rechecked before a side effect.
Transfer to procurement would require binding supplier, amount, currency, budget, tax, bank account, and purchase-order version, plus procurement-specific approvals.
Do not transfer the ticket thresholds, retention assumptions or reviewer staffing to procurement without new evidence.

## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
| Relationship | Target | Shared mechanism | Boundary |
|---|---|---|---|
| Uses method | [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | Untrusted content, least privilege, human oversight and adversarial testing. | General guidance does not prove this implementation. |
| Uses method | [NIST least privilege](https://csrc.nist.gov/glossary/term/least_privilege) | Minimize authorization and resources for each principal. | Does not specify action approval or sink reconciliation. |
| Contrasts with | Procurement workflow recovery, PLAN-2026-IR006, planned in [collection index](INDEX.md). | Exact version approval and replay safety. | Planned brief only; no empirical comparison. |
| Extends / superseded-by | No completed directly comparable case or successor is asserted. | Not applicable. | Add an actual link only when available. |

### 14b. What this case leaves open
| Question | Evidence gap | Resolving test | Priority / trigger | Owner |
|---|---|---|---|---|
| Can people reliably spot manipulated evidence under load? | No reviewer study or workload sample. | Blinded representative and adversarial review study. | High / before pilot. | Operations research lead. |
| Does the connector support stable deduplication, conditional updates and authoritative receipts? | API contract not selected. | Contract inspection and failure injection. | High / before writes. | Integration owner. |
| How fast do authorization changes propagate? | No identity implementation or measurement. | Revocation race and outage tests. | High / before pilot. | Identity owner. |
| What downstream automations does a ticket update invoke? | Trigger inventory absent. | Map and exercise notifications, workflows and integrations. | High / before action scope approval. | Ticket platform owner. |

### 14c. Intense interview pack — questions with checked answers
Practice as a two-minute framing, ten-minute method discussion, then a 30–45-minute design and debugging discussion.
For every question, the scoring scale is 0 incorrect, 1 partial, 2 correct with assumptions, and 3 defended tradeoff.

#### Q-01 / intermediate / Framing
- Candidate prompt: What decision is being made and for whom?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Distinguish authorizing an effect from generating an answer.
- Strong answer: The operations owner chooses whether to permit one exact ticket update; manual operation is the baseline.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Draft-only is safer where write guarantees are absent.
- Follow-up: What changes when no ticket write is required?
- Red flag: A model confidence score is not authority.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Sections 1–3.

#### Q-02 / intermediate / Threat modeling
- Candidate prompt: Can a retrieved ticket comment authorize a tool call?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Identify the source trust boundary.
- Strong answer: No; content can influence a proposal but cannot grant permissions or bypass policy, even if it looks like a system instruction.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Quarantine suspicious material without relying exclusively on detection.
- Follow-up: What if the comment is from an internal employee?
- Red flag: Treating source origin as automatic authority.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: SRC-01 and Section 8.

#### Q-03 / intermediate / Least privilege
- Candidate prompt: Which principal receives the ticket write credential?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Consider model, requester and executor separately.
- Strong answer: Only the restricted executor holds a short-lived scoped credential; it verifies delegated requester and reviewer authorization.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A native delegated ticket-platform credential may work if equally constrained and independently checked.
- Follow-up: What if the sink only offers a tenant-wide admin token?
- Red flag: Giving the token to the model.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: SRC-02 and Section 8.

#### Q-04 / intermediate / Binding
- Candidate prompt: Why is approval of a conversation insufficient?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Focus on what the sink actually receives.
- Strong answer: A conversation does not bind target, operation, payload and version; approve a canonical intent and verify its digest at send.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A signed structured command is acceptable with the same semantic binding.
- Follow-up: What if a schema upgrade changes interpretation?
- Red flag: Comparing only the model explanation.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Sections 6 and 8.

#### Q-05 / intermediate / Concurrency
- Candidate prompt: How do you prevent a stale reviewed update overwriting a later ticket edit?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Identify the check-to-use race.
- Strong answer: Require a sink-side conditional write against the approved ticket revision; on conflict generate a new proposal and approval.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A platform transaction may provide equivalent atomicity.
- Follow-up: What if the sink cannot condition on revision?
- Red flag: A separate preflight read guarantees safety.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: A-01 and Section 8.

#### Q-06 / intermediate / Recovery
- Candidate prompt: A timeout occurs after send; what happens next?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: The send may or may not have committed.
- Strong answer: Mark UNKNOWN, freeze action and query an authoritative receipt using the same scoped key; retry only under verified sink semantics.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Manual reconciliation is valid if the receipt is unavailable.
- Follow-up: What happens after dedupe retention expires?
- Red flag: Retry with a new key immediately.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: A-02 and Section 8.

#### Q-07 / intermediate / Sizing
- Candidate prompt: At 240 requests/day, 30% writes and 2.5 minutes/review, how much review time?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Count writes first and keep units.
- Strong answer: 240 × 0.30 = 72 reviews; 72 × 2.5 = 180 minutes = 3 hours/day.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A distributional queueing model is needed for waiting times.
- Follow-up: What if 50% need writes?
- Red flag: Treating three hours as a guarantee of same-day completion.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 6.

#### Q-08 / intermediate / Sensitivity
- Candidate prompt: What happens if review duration doubles to five minutes?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Keep arrival and write fraction fixed.
- Strong answer: 72 × 5 / 60 = 6 reviewer-hours/day, double the original three.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Staffing buffers depend on arrival variability and coverage.
- Follow-up: What if each reviewer has four productive hours/day?
- Red flag: Ignoring breaks or claiming peak coverage from averages.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 6.

#### Q-09 / advanced / Data
- Candidate prompt: What is the unit of evaluation and how should splits avoid leakage?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Account for correlated revisions.
- Strong answer: One action proposal is the unit; group all revisions of a ticket and related attack payloads within one time-aware partition.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Cluster by broader incident if multiple tickets share an attack.
- Follow-up: How would you handle an unresolved sink outcome?
- Red flag: Joining only by ticket ID or labeling unknown as safe.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 5a and Section 9.

#### Q-10 / advanced / Evaluation
- Candidate prompt: Does zero failures in 300 trials prove safety?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Specify the approximation and assumptions.
- Strong answer: No; the approximate one-sided 95% upper bound is 3/300 = 1% under independent representative trials, and adaptive attacks weaken it.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Use an exact binomial bound when assumptions are suitable.
- Follow-up: What if the payloads are near duplicates?
- Red flag: Declaring zero production risk.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 9.

#### Q-11 / advanced / Debugging
- Candidate prompt: A sink write has no approval record: what do you do?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Consider delayed logging, direct bypass and restore loss.
- Strong answer: Stop writes, preserve evidence, compare sink audit, ledger restore history and service-credential access, and treat the cause as unresolved.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Continue read-only work if separately isolated.
- Follow-up: What if the audit pipeline is merely delayed?
- Red flag: Assuming either attack or harmless delay before reconciliation.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: F-29 and Section 8.

#### Q-12 / advanced / Authorization
- Candidate prompt: What if a reviewer is revoked after approving but before dispatch?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Separate historical approval from present authority.
- Strong answer: Execution must recheck the reviewer and requester entitlements and deny when required authority has been revoked.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A manually governed exception would be a new separately authorized workflow.
- Follow-up: What if the authorization service is unavailable?
- Red flag: Treating the historical approval as irrevocable authority.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: A-04 and Section 8.

#### Q-13 / advanced / Human factors
- Candidate prompt: Is a human click sufficient to prevent injection?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Consider presentation, workload and reviewer access.
- Strong answer: No; display exact structured effect, trustworthy source provenance and counterevidence, then measure comprehension and fatigue.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: For low-risk read-only tasks a human review may not be necessary under a different policy.
- Follow-up: What if reviewers approve every item in one second?
- Red flag: Counting clicks as proof of informed judgment.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: F-23 and Section 9.

#### Q-14 / advanced / Isolation
- Candidate prompt: What does a parser sandbox protect and not protect?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Distinguish code execution from semantic manipulation.
- Strong answer: It limits parser access to host resources and network but does not make extracted instructions trustworthy.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: A managed parser can be used if its isolation and data terms are independently verified.
- Follow-up: What if an image contains a hidden instruction?
- Red flag: Equating successful parsing with trustworthy content.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: SRC-01 and Section 8.

#### Q-15 / advanced / Rollout
- Candidate prompt: Can old approvals be replayed after rolling back the gateway?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Check schema, policy and sink compatibility.
- Strong answer: Freeze sends and reconcile; only reuse approvals after proving exact semantic compatibility, freshness and authorization, otherwise obtain new approval.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Expire all pending approvals for a simpler conservative rollback.
- Follow-up: What if the sink already committed an action?
- Red flag: Assuming rollback undoes external side effects.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 8d.

#### Q-16 / advanced / Transfer
- Candidate prompt: Would the same gate automatically secure a purchase?
- Given facts: use only the hypothetical system and inputs specified in this case.
- Clarify or assume: Compare business effect fields and downstream consequences.
- Strong answer: No; procurement requires amount, currency, supplier, account, tax, budget and order revision plus appropriate authorities and sink semantics.
- Reasoning check: distinguish the authorized intent, actual effect and evidence status.
- Acceptable alternative: Manual procurement is preferable if those constraints cannot be bound.
- Follow-up: How does a payment reversal change the risk?
- Red flag: Copying ticket thresholds or assuming compensation erases consequences.
- Scoring: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff.
- Answer support: Section 13.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Enterprise internal-ticket agent execution security. |
| Problem type | Authorize or deny one exact proposed side effect under untrusted content. |
| Method | Independent policy, exact-intent human approval, constrained executor and recoverable sink; manual baseline. |
| Primary result | Not measured; no deployment or evaluation claimed. |
| Key lesson | Bind execution to current authorized exact intent; binding does not cure reviewer error or unsafe sink semantics. |
| Tags | `exact-action-approval`; `least-privilege-execution`; `idempotent-side-effects`. |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-2026-030 / 1.0 / 2026-11-07. |
| Evidence level | Not-measured outcome; general security sources published; scenario calculations illustrative. |
| Implementation stage | Proposed; document draft and unreviewed. |
| Baseline / primary metric | Manual update / unauthorized effects per 1,000 attempted agent-originated writes; window proposed, not executed. |
| Key limitation | Sink guarantees, reviewer performance and actual outcome are unknown. |
| Review trigger / owner | Model, policy, connector or threat change / enterprise agent security lead. |
