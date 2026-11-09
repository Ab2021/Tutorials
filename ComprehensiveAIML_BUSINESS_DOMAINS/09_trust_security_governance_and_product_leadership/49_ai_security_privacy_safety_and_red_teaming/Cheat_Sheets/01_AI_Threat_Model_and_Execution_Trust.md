---
collection: "module-49-security"
file_id: "01"
topic: "AI threat modelling and execution trust"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A new tool, identity provider, model, data class, or external effect changes the trust boundary."
notation_style: "plain-text"
prerequisites: ["../../INDEX.md", "../INDEX.md"]
pattern_tags: ["asset-threat-control", "authorization-scope", "approval-payload-binding"]
source_status: "DERIVED / ILLUSTRATIVE / SOURCED; visible source bodies unavailable"
owner_role: "AI security architect"
source_index: "INDEX.md"
---

## 1. File Header

# AI Threat Model and Execution Trust

> **Purpose:** Turn one AI workflow into an asset-to-action threat model that distinguishes model text from authorized effect.
> **Prerequisites:** Basic authentication, authorization, API request/response semantics, and the module index.
> **Key outcome:** Given a workflow, enumerate attack paths, quantify a bounded residual-risk priority, and select controls that prevent unauthorized external effects.
> **This file does NOT cover:** Penetration testing of a particular product, legal compliance determinations, or evidence that any named provider is secure.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED mechanisms; ILLUSTRATIVE arithmetic and traces; SOURCED framework vocabulary. Source lesson bodies were unavailable.
> **Retrieval cue:** If a model can call a tool or read untrusted content, start at Section 4 and then Section 8.

## 2. Concept Map / Mental Model

```text
[user, document, connector, operator]
                 |
                 v
        [identity + tenant + purpose]
                 |
                 v
 [model context assembly] <--- untrusted instructions/data
                 |
                 v
       [proposed tool intent]
                 |
      schema, authorization, policy
                 |
                 v
 [effect preview bound to payload digest]
                 |
          human approval when required
                 |
                 v
 [narrow executor + effect journal]
                 |
                 v
[receipt, monitoring, revocation, recovery]
```

A language model emits a proposal, not authority. The decisive boundary is the executor: it must authenticate the caller, authorize the exact effect, enforce a resource limit, and journal the result. Prompt text may influence a proposal but must not alter the executor's independent policy decision.

Threat modelling starts with a decision unit. For an export assistant, the unit is one requested export with tenant, subject, fields, destination, purpose, expiry, and approval state. “The chatbot” is too broad to test.

**Takeaway:** Map data and authority separately; data may be read without gaining permission to act.

## 3. Foundational Definitions

| Term | Meaning | Type / unit | Boundary or nonexample |
|---|---|---|---|
| Asset | Something whose confidentiality, integrity, availability, or authorized use matters | data, credential, model, action | A public formatting instruction is not automatically an asset |
| Principal | Entity making a request | user, workload, service | A tenant name in a prompt is not identity |
| Capability | Narrow authority to invoke one operation under constraints | signed/tokenized permission | A model tool description is not a capability |
| Execution surface | Component able to create an external effect | API, database write, browser, shell, message bus | Text generation alone is not an external effect |
| Trust boundary | Place where assumptions about identity, integrity, or policy change | interface/state transition | Internal network location alone is not a sufficient boundary |
| Attack path | Ordered route from entry condition to unwanted outcome | directed graph | A vulnerability list with no reachable path is incomplete |
| Control | Mechanism that prevents, detects, contains, or recovers from a path | technical or procedural | A policy PDF without enforcement is not a preventive control |
| Blast radius | Maximum scope an accepted request can affect | records, currency, tenants, time | Average usage is not an upper bound |
| Approval binding | Cryptographic or durable association between reviewed preview and executed payload | digest + policy/version + expiry | Approving a conversation transcript is not enough |
| Residual risk | Exposure after stated controls and assumptions | prioritized ordinal/expected loss | It is not proof of safety |

A threat is an actor exploiting a condition to violate an objective. It is not synonymous with a model mistake. A hallucinated explanation is an integrity problem only when a consumer treats it as evidence; a fabricated tool argument becomes an authorization problem when an executor accepts it.

### Threat-model canvas: make each path falsifiable

Use the following canvas once for each effect-bearing operation. A completed row is a hypothesis that a reviewer can challenge, not a claim that the listed safeguard works.

| Field | Export example | Evidence needed | Decision if unknown |
|---|---|---|---|
| Actor | authenticated analyst; malicious document author; compromised workload | identity issuer and session trace | do not enable export |
| Asset | record contents, destination identity, capability token, receipt | classification and retention record | limit to synthetic data |
| Entry point | user text, retrieved page, tool result, retry endpoint | interface inventory | block unlisted route |
| Trust boundary | identity-to-session; model-to-executor; approval-to-commit | contract and trace | retain manual baseline |
| Harm | unauthorized disclosure, wrong-tenant disclosure, duplicate effect | impact owner statement | escalate to risk owner |
| Precondition | executor accepts a proposal or an alternate endpoint exists | negative test | remove capability |
| Preventive control | scope, quota, payload binding, egress deny | configuration/contract | no promotion |
| Detection | denial event, receipt, quota counter, canary egress log | observable correlation ID | add instrumentation |
| Containment | revoke capability, disable route, freeze queue | tested recovery run | keep action read-only |
| Recovery | reconcile receipts, notify owner under policy, remediate state | incident procedure | define before pilot |

#### Asset: data content

Confidentiality questions are not limited to raw document text. A title, count, embedding, file path, query term, recipient address, or timing pattern can be sensitive in context. Inventory the minimum fields necessary for the effect. A design that removes record bodies but logs every recipient and query may merely move the exposure.

Integrity questions ask who can change the content, classification, or mapping between record identifier and returned record. If a model proposes identifiers but the executor resolves them against a current server-side store, record resolution is a security decision. The resolution must use the authorization-time tenant and policy, not a model-supplied filter.

Availability questions ask whether a malicious request can consume reviewer time, quota, storage, or downstream API capacity. An approval gate is an availability surface: adversaries can submit plausible-looking requests that create a human backlog. Rate limits and queue admission should be considered alongside confidentiality controls.

#### Asset: authority

A credential is not the only authority asset. A browser session, service-to-service identity, approval record, feature flag, deployment role, signed callback URL, or message-bus permission can cause an effect. Enumerate these separately because their revocation and audit properties differ.

For each authority asset, record issuer, subject, audience, expiry, scope, storage location, rotation path, revocation path, and the endpoint that consumes it. “Stored in a secret manager” does not answer who can mint or replay it.

Do not put authority in model-visible context unless the model must invoke a narrow broker. A model that can read a bearer token can leak it in output, send it through a tool, or cause its use in an unintended sequence. Brokered execution can retain the credential outside model context.

#### Asset: policy and evidence

Policy version is an integrity asset. An approval made under policy version `P1` should not silently authorize an effect after a material rule changes to `P2`. Bind the relevant policy version to the approval and choose an explicit rule: either honor the approved snapshot within a short expiry or require reapproval on policy change.

Evidence is also an asset because altered receipts can hide abuse and over-retained receipts can expose sensitive information. Identify who can append, correct, redact, export, and delete journal records. An immutable append-only store can still require a controlled redaction process for privacy obligations; “immutable” is not a universal retention answer.

#### Entry point: user instruction

Treat every user-controlled string as data until a deterministic parser maps a small allowed subset into a task field. A request such as “send the report to finance” may be legitimate but does not itself prove which finance group, which tenant, which report version, or which purpose is authorized.

Ambiguity is a threat condition. If a natural-language request maps to multiple destinations, the safe behavior is to ask a bounded clarification or retain the manual workflow. Guessing a recipient turns uncertainty into an external effect.

#### Entry point: retrieved content

Retrieved content has two separate properties: whether it is relevant and whether it is trusted to supply facts. Relevance scoring does not grant instruction authority. A high-ranked document can be malicious, stale, mislabeled, or legitimately authored for a different audience.

Carry provenance through retrieval: source ID, tenant, access decision, timestamp, content version, and trust label. The model can be told that content is untrusted, but the decisive control is downstream: an executor must reject effects unsupported by authenticated task fields and policy.

#### Entry point: tool result

A tool result can be syntactically valid while malicious or wrong. For example, a directory lookup may return a recipient whose display name resembles the intended one. Parse tool output into typed fields, validate source identity and freshness, and show a bounded human-readable preview when a person must decide.

If a tool returns an instruction-like field, do not concatenate it into the agent's control prompt. Preserve it as data with an explicit purpose. The same rule applies to error messages: an untrusted server can return text crafted to influence the next model action.

#### Entry point: retry and callback

Retries are attacker-controlled in many distributed environments because clients, proxies, and workers may redeliver. The threat model must specify which identifier defines “same intent,” where it is stored, how long it is retained, and what the executor returns after a timeout.

Callbacks and webhooks introduce a second initiator. Verify origin with a current, scoped authentication mechanism; bind callback data to the original operation and state; and reject callbacks that advance an expired or cancelled operation. Network origin alone is not identity.

#### Trust boundary: model to executor

The model-to-executor contract should expose an intent, not a general-purpose command line. Prefer a small operation set such as `draft_ticket`, `lookup_allowed_record`, or `submit_export_for_approval` over `run_shell` or arbitrary HTTP. Each typed operation has its own policy and resource budget.

Validation order matters. Parse first to eliminate ambiguity; authenticate the caller; resolve authoritative server-side objects; authorize purpose/scope; enforce budget; require approval if policy says so; commit once; then create a receipt. A model-generated “approved=true” field is ignored unless it maps to an approval service record.

#### Threat path walkthrough: indirect instruction to export

1. An analyst asks for a summary of a retrieved document.
2. The document contains a string requesting an export to an external address.
3. The model proposes an export tool call after reading the string.
4. The tool wrapper submits the proposal to the policy decision point.
5. The policy receives tenant and purpose from the authenticated session, not the document.
6. The destination is not in the permitted class, so the policy returns denial.
7. The executor writes a denial receipt with correlation ID and reason class.
8. Monitoring aggregates the event without storing the document body.
9. The analyst sees a bounded explanation and can use an approved workflow.

The path is unsafe if step 5 reads a tenant or destination from model text, if step 6 is bypassed by a second endpoint, or if step 7 cannot establish whether a commit already happened. Each is a separate test objective.

#### Threat path walkthrough: compromised authorized analyst

1. An authenticated analyst legitimately has access to tenant A.
2. Their session is compromised or their request is malicious.
3. They request a large export to an allowed destination.
4. Scope check passes because tenant and destination are valid.
5. Quantity threshold routes request to human approval.
6. The preview states count, fields, recipient, purpose, expiry, and policy version.
7. Reviewer rejects or reduces scope when purpose is unsupported.
8. If abuse is suspected, capability is revoked and prior receipts are reviewed.

This path shows why tenant scoping alone is insufficient. It constrains cross-tenant harm but not misuse within a granted tenant. Purpose limitation, quota, approval, anomaly detection, and recovery can reduce within-scope harm.

**Takeaway:** A usable threat model has a named asset, an entry condition, an enforceable boundary, an observable signal, and a recovery decision for every important path.

**Takeaway:** Name the asset, principal, effect, and violated objective before proposing a control.

## 4. Core Formulas with Full Derivation

### 4.1 Base: expected-loss priority

Validity before use: use this only for comparative triage when likelihood is a scenario probability over the same time window and impact is expressed in the same loss unit. It does not estimate adversarial frequency from a small test suite.

Derivation: if loss is I when a path succeeds and 0 otherwise, then expected loss is E[L] = P(success) * I. A control reducing conditional success probability from p to p_c changes expected loss by (p - p_c) * I.

```text
FORMULA [TM-01]: R = p * I
WHERE: R expected loss [currency/window]; p path-success probability [0,1];
       I loss conditional on success [currency/event].
CONSTRAINT: same population/window; one outcome per event; p is a scenario input,
            not a measured attack rate; correlated paths must not be summed blindly.
NUMERICAL EXAMPLE: ILLUSTRATIVE p=0.04 per reviewed month; I=75,000 currency/event;
                   R=0.04*75,000=3,000 currency/month.
INTERPRETATION: A control reducing p to 0.01 has illustrative avoided exposure of
                (0.04-0.01)*75,000=2,250 currency/month before control cost.
```

Boundary/counterexample: if a single credential compromise enables ten exports, multiplying one-event loss by a marginal probability understates correlated loss. Model the campaign or set an explicit cap.

### 4.2 General: constrained effect budget

Validity before use: the executor must count every committed effect under one stable authorization scope and reject before commitment. Retries must carry the same operation identifier.

Derivation: a scope permits at most q items per call, c calls per authorization, and b bytes per item. The maximum payload-only effect is q*c*b; this is a ceiling, not storage or network capacity.

```text
FORMULA [TM-02]: B_max = q * c * b
WHERE: B_max maximum permitted payload [bytes/authorization]; q items/call;
       c calls/authorization; b maximum bytes/item.
CONSTRAINT: all writes are counted; no alternate endpoint bypasses the quota;
            metadata, retries, and compression are excluded from this payload bound.
NUMERICAL EXAMPLE: ILLUSTRATIVE q=20 records/call, c=3 calls, b=4096 bytes/record;
                   B_max=20*3*4096=245,760 bytes=240 KiB.
INTERPRETATION: An authorization cannot legitimately export 100 records under this scope.
```

Independent check: 60 records multiplied by 4 KiB/record is 240 KiB. If c becomes 5, the ceiling becomes 400 KiB; a change request must re-evaluate approval and monitoring thresholds.

### 4.3 Variant: payload binding invariant

Preconditions: canonical serialization, a collision-resistant digest selected by the security program, trusted clock, server-side approval record, and an executor that recomputes the digest.

```text
INVARIANT [TM-03]: execute(payload, approval) is permitted only when
 digest(canonical(payload)) = approval.payload_digest
 AND now < approval.expiry
 AND caller, tenant, purpose, policy_version, and remaining_budget match approval.
```

State transition: `proposed -> policy_checked -> previewed -> approved -> executing -> committed|failed`. Only the executor may enter `committed`; a retry with the same operation ID must return the prior receipt rather than repeat the effect.

Worked trace: request `E-17` proposes 12 records to destination A. The reviewer approves digest `d1`. A later instruction changes destination to B, producing `d2`. Because `d2 != d1`, execution is rejected and a new preview is required. The expected signal is `APPROVAL_PAYLOAD_MISMATCH`, not silent substitution.

Failure boundary: a digest cannot tell whether the reviewer understood a misleading preview; field labels, recipient rendering, and policy semantics still need review.

### 4.4 Limit and counterexample: model confidence is not authorization

A high confidence score can rank a proposal but cannot prove identity, tenant entitlement, purpose, or approval. A system that authorizes when confidence exceeds 0.9 converts a statistical signal into a privilege-escalation channel. Keep model-derived scores advisory unless a separate deterministic policy explicitly permits their use.

### 4.5 Authorization-time versus execution-time state

Validity before use: use this distinction whenever data, roles, policy, recipient groups, or quotas can change between a request and an effect. It is especially important for queued work, approvals, retries, and long-running agents.

At authorization time, the system establishes what was permitted under a named policy and identity context. At execution time, it decides whether that permission remains usable. A safe design explicitly chooses which fields are snapshotted and which must be rechecked. For example, an approval can snapshot a record set and recipient, while membership in a disabled-user list must be checked again immediately before commit.

```text
FORMULA [TM-04]: permit_commit = A_snapshot AND V_current AND Q_remaining
WHERE: A_snapshot is approval/policy match at authorization time [Boolean];
       V_current is current validity of revocation, expiry, and mandatory rechecks [Boolean];
       Q_remaining is remaining quota >= requested effect [Boolean].
CONSTRAINT: Booleans must be evaluated by a trusted executor; a snapshot is not a
            substitute for revocation, and a current check is not a substitute for
            preserving what the reviewer approved.
NUMERICAL EXAMPLE: ILLUSTRATIVE approval expires at 10:30, execution arrives 10:29,
                   recipient remains allowed, and 8 of 60 records remain under quota
                   for a request of 10 records. A_snapshot=true and V_current=true,
                   but Q_remaining=false, so permit_commit=false.
INTERPRETATION: A valid approval cannot overdraw its own budget.
```

Boundary: rechecking every dynamic field may make a workflow non-deterministic and impossible to reproduce. State which current checks are mandatory, record their result in the receipt, and route changed business facts to a new approval rather than silently substituting them.

### 4.6 Per-effect budget conservation

Validity before use: the same central ledger must observe every committing endpoint and each effect must have a nonnegative declared cost. This is a mechanism for a bounded authorization; it is not a prediction of business loss.

Derivation: starting allowance Q_0 decreases by each committed amount x_j. Rejecting before commit whenever the next amount exceeds the remaining allowance preserves nonnegativity.

```text
FORMULA [TM-05]: Q_after = Q_before - x ; require 0 <= x <= Q_before
WHERE: Q_before and Q_after are remaining records, bytes, currency, or actions
       [same unit]; x is the next committed effect [same unit].
CONSTRAINT: one unit and one ledger per scope; failed reservations must be released;
            concurrent commits require serialization or an atomic conditional update.
NUMERICAL EXAMPLE: ILLUSTRATIVE Q_0=60 records. Commit x_1=20 gives Q_1=40;
                   commit x_2=20 gives Q_2=20; request x_3=25 is denied because
                   25>20; request x_3=15 gives Q_3=5.
INTERPRETATION: Total committed records are 55 and cannot exceed 60.
```

Independent check: 20+20+15=55 and 60-55=5. A race where two workers each read Q=20 and both commit x=15 violates the invariant unless the ledger update is atomic. The security test is not “does the UI show 20?” but “can concurrent commits leave Q negative?”

### 4.7 Review-queue stability boundary

Validity before use: this is a fluid capacity estimate with an average arrival rate lambda and average service rate mu. It ignores variability, priority classes, breaks, abandonment, and correlated attack bursts; use measured queueing analysis before operational commitments.

```text
FORMULA [TM-06]: rho = lambda / mu ; stable fluid backlog requires rho < 1
WHERE: rho utilization [dimensionless]; lambda review arrivals [requests/hour];
       mu completed reviews [requests/hour].
CONSTRAINT: rates refer to the same queue and interval; capacity must include actual
            availability, escalation, and rework, not only nominal headcount.
NUMERICAL EXAMPLE: ILLUSTRATIVE lambda=40 requests/hour; one reviewer completes
                   3600/45=80 requests/hour; rho=40/80=0.5.
INTERPRETATION: Average spare capacity is 40 requests/hour in this simplified model.
```

Counterexample: with 90-second reviews, mu=40 requests/hour and rho=1.0. A positive backlog has no draining surplus, so adding buffer space only postpones the operating failure. If a hostile workload can produce bursts, admission control and priority rules are security controls as well as operations controls.

### Control-to-path selection record

| Path condition | Candidate control | Why it is selected | Residual condition | Test oracle |
|---|---|---|---|---|
| Model proposes external recipient | server-side recipient policy | model text is non-authoritative | authorized insider may still misuse allowed group | disallowed recipient produces denial receipt |
| Retrieved content carries instruction | provenance label plus executor policy | content relevance does not imply command authority | model can still waste tokens or confuse user | hostile document cannot cause commit |
| Approval queue is overloaded | threshold, admission, capacity plan | preserve review for high-consequence effects | threshold policy can be wrong | queue and denial reason reconciled |
| Retry arrives after timeout | operation ID/effect journal | distinguish prior commit from uncommitted attempt | journal outage needs recovery protocol | duplicate delivery has one receipt |
| Sandbox code probes network | default-deny egress | reduce reachable destinations | allowed endpoint can still carry sensitive data | canary destination denied |
| Policy changes after approval | policy version and expiry | make change visible | urgent revocation needs current check | revoked approval is denied |

**Takeaway:** A control is selected because it blocks a specific transition under stated assumptions, not because its name appears in a security checklist.
**Takeaway:** Quantify bounded exposure, but make authorization a deterministic check over authenticated attributes and the exact effect.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE workflow; arithmetic verified manually. No organization, incident, or deployment is represented.

A research assistant can search a tenant-approved index and propose an export. The baseline allows a service account to execute any model-produced JSON that passes a schema. The proposed design issues short-lived, tenant-scoped capabilities and requires approval above 20 records or for external destinations.

| Input | Value | Meaning |
|---|---:|---|
| Allowed records/call q | 20 records | executor quota |
| Calls/approval c | 3 calls | operation budget |
| Maximum record b | 4 KiB | payload cap |
| Proposed records | 55 records | model request |
| Internal review time | 45 s | illustrative human queue service time |
| Requests/hour | 40 requests/h | arrival rate during a peak |
| Reviewers | 1 reviewer | assumed available capacity |

Step 1: the proposed 55-record export cannot be a single call because q=20. It requires three calls: 20 + 20 + 15 = 55 records. This exactly fits c=3; a fourth call is rejected even if each call is schema-valid.

Step 2: maximum payload is TM-02: 20*3*4096=245,760 bytes. The 55-record request has at most 55*4096=225,280 bytes=220 KiB. Its payload fits, but fit is not authorization; destination and purpose must still match.

Step 3: reviewer service capacity is 3600/45=80 requests/h under constant service and no breaks. At 40 requests/h, utilization is 40/80=0.50. This is a capacity estimate, not a latency guarantee.

Sensitivity: if service time increases to 90 s because previews contain more context, capacity falls to 40 requests/h and utilization reaches 1.00. Queues become unstable under any sustained burst in the simple deterministic model. The mitigation is not to waive approval; it is to reduce eligible review volume, add reviewers, or improve bounded previews while preserving the payload-binding invariant.

Oracle inputs: `(q,c,b)=(20,3,4096)` must return `245760`; `(records,calls)=(55,3)` must accept only partitions totaling 55 with each partition <=20; `(destination=d2, approved_digest=d1)` must reject.

**Takeaway:** A request may fit a technical budget yet require a human or policy decision; capacity changes do not justify bypassing a security boundary.

## 6. Visual / Geometric Intuition

Think of authority as an intersection, not a label:

```text
[authenticated principal]
           intersect
[tenant membership]
           intersect
[purpose and data class]
           intersect
[operation allow-list]
           intersect
[quantity/time budget]
           intersect
[approved exact payload when required]
                     = executable authority
```

Every extra intersection narrows the feasible action set. A broad service credential resembles a large rectangle: many accidental and malicious paths fit inside. A scoped, expiring capability is a smaller rectangle. Approval binding removes the gap between “reviewed proposal” and “executed effect.”

A data-flow diagram answers where bytes travel. An authority-flow diagram answers who can cause an effect. Draw both. A retrieval document can be untrusted data while the connector that fetched it is trusted to authenticate; neither property transfers to tool arguments generated from the document.

**Takeaway:** The safest design is not one with the most filters; it is one whose allowed action region is small, explicit, and independently enforced.

## 7. Comparison / Selection Table

| Design | Use when | Prevents | Does not prevent | Disqualifier / fallback |
|---|---|---|---|---|
| Prompt-only policy | Demonstration with no effects | Some accidental model behavior | adversarial instructions or credential misuse | Never use as an execution control |
| Schema validation | Arguments have stable types/ranges | malformed requests | authorized-looking harmful requests | add authorization and quotas |
| Static service credential | tightly bounded internal automation | repeated login overhead | tenant confusion, excess privilege | replace with workload identity/scoped token |
| Tenant-scoped capability | multi-tenant operations | cross-tenant operation by default | compromised authorized principal | add purpose, expiry, budget, revocation |
| Human approval | irreversible or high-impact effects | unreviewed high-impact execution | reviewer deception, queue overload | bind preview and use thresholds |
| Sandbox | untrusted code or documents | host impact within declared isolation | secret exposure through allowed channels | remove secrets and restrict egress |
| Kill switch | active abuse or unknown blast radius | continued use after activation | prior committed effects | pair with receipts and recovery plan |

Decision guide: choose deterministic authorization at every execution surface. Add approval when the residual consequence exceeds a policy threshold or when automated evidence is insufficient. Use a sandbox for untrusted code but do not treat it as an authorization system. The simplest credible baseline is a read-only assistant with no executor.

**Takeaway:** Controls compose by covering different links in a path; no single control replaces identity, authorization, containment, and recovery.

## 8. Failure Modes & Edge Cases

### F-01 Prompt injection changes a tool argument

**Competing causes:** a retrieved document embeds instructions; a benign user request is ambiguously parsed; a tool wrapper concatenates untrusted text into a command.

**Discriminating check:** preserve the untrusted-content label through context assembly and compare proposed arguments with a policy-derived allow-list. The expected signal is a proposed action unsupported by authenticated task fields.

**Bounded fix:** separate data from instructions, use typed tool arguments, and authorize against server-side attributes. Do not claim that delimiter text or a refusal prompt guarantees prevention.

**Verification:** inject a document requesting an external export; the model may mention it, but the executor must reject the unapproved destination.

### F-02 Tenant identity arrives from model text

**Competing causes:** tenant ID was copied from a prompt; session routing selected the wrong tenant; a cache key omitted tenant scope.

**Discriminating check:** log tenant claim source. Only identity middleware may set the authoritative tenant ID. A mismatch between prompt and session must be rejected or treated as data, never silently reconciled.

**Bounded fix:** derive tenant scope from authenticated session and include it in cache, retrieval filter, capability, approval, and effect journal keys.

**Verification:** two sessions with identical user text but different tenants retrieve disjoint document IDs and cannot exchange capabilities.

### F-03 Approval covers a preview but not the committed effect

**Competing causes:** mutable request after approval; canonicalization mismatch; retry reconstructs arguments differently.

**Discriminating check:** recompute canonical payload digest immediately before commit and compare all approval attributes. Expected signal: `APPROVAL_PAYLOAD_MISMATCH`.

**Bounded fix:** immutable approval record, idempotency key, expiry, and a single executor that performs the comparison.

**Verification:** mutate recipient after approval; no side effect occurs and the audit log identifies the rejected operation.

### F-04 Sandbox has network egress to sensitive systems

**Competing causes:** default route remains; metadata endpoint is reachable; proxy allow-list is overbroad.

**Discriminating check:** egress test from sandbox using canary destinations and DNS logs. Expected signal is deny before connection establishment for unapproved destinations.

**Bounded fix:** default-deny egress, separate credentials, resource quotas, read-only mounts, and short-lived isolated workers.

**Verification:** a test program cannot resolve or reach a prohibited endpoint; authorized fixture access remains available.

### F-05 Quota is bypassed through retries or alternate endpoints

**Competing causes:** quotas count requests rather than committed items; retry lacks operation ID; batch and single-record APIs have separate counters.

**Discriminating check:** submit a timed-out operation twice and compare receipts, counters, and resulting records. Expected outcome is one committed effect and one receipt lineage.

**Bounded fix:** central effect journal and quota accounting at commit, not at model invocation.

**Verification:** replay returns the original receipt without increasing item count.

### F-06 Safety evaluation passes while tool harm remains

**Competing causes:** suite tests only text; sandbox simulator omits effectful endpoint; success metric ignores policy denials.

**Discriminating check:** include execution traces and assert that prohibited intents cannot produce committed receipts. Compare textual refusal score with executor-denial rate.

**Bounded fix:** evaluate end-to-end tasks with a controlled executor and explicit harmful-effect oracle.

**Verification:** a model can produce a persuasive but unsafe tool proposal without causing an effect.

**Takeaway:** Diagnose the boundary that failed, not merely the model output that preceded it.

## 9. Complexity & Cost Analysis

Threat modelling cost is mainly modeling and review effort, not a classifier runtime. Let A be assets, E interfaces, and P paths considered. A workshop that enumerates each asset-interface pair has O(AE) starting surface; path exploration can grow combinatorially, so cap depth and prioritize reachable high-impact paths.

For every tool call, deterministic checks commonly add O(k) work where k is the number of authorization predicates plus O(1) lookup for a capability and idempotency record under indexed storage assumptions. Payload hashing is O(n) in payload bytes. This cost is appropriate when it prevents a larger external effect; measure it in the deployed environment before setting an SLO.

Illustrative sizing: 40 approval requests/h at 45 s/request needs 30 reviewer-minutes/h, or 0.5 reviewer-equivalents under uninterrupted service. Add breaks, variance, escalation, and quality sampling before staffing. A security design that hides human work creates an unpriced backlog.

Retention cost: effect journals should record the minimum necessary request identity, policy decision, payload digest, receipt, and timestamps. Do not log sensitive payloads merely for audit convenience; the log becomes an asset with its own access and retention controls.

**Takeaway:** Price prevention, review, logging, and recovery together; moving risk into an unstaffed queue is not a lower-cost control.

## 10. Key Theorems / Results Reference

| Result | Use | Conditions and limit | Source |
|---|---|---|---|
| NIST AI RMF functions | Organize governance activities around Govern, Map, Measure, Manage | Framework is voluntary guidance, not a product certification | NIST AI 100-1, January 2023, accessed 2026-11-09: https://doi.org/10.6028/NIST.AI.100-1 |
| Least privilege | Limit authority to necessary operations and scope | Requires correct asset and operation inventory; does not detect misuse within granted scope | Saltzer & Schroeder, 1975; apply as design principle, not empirical performance claim |
| Complete mediation | Check authorization on every access/effect | Caches and alternate paths must not bypass check | Saltzer & Schroeder, 1975 |
| OWASP prompt injection guidance | Treat untrusted instructions as hostile input | Guidance does not prove a particular defense succeeds | OWASP Prompt Injection Prevention Cheat Sheet, accessed 2026-11-09: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html |

**Takeaway:** Frameworks supply questions and principles; they do not substitute for a tested system-specific threat model.

## 11. Interview / Exam Questions

### Q1. Why is schema validation insufficient for a tool call?
**Answer:** It proves shape, not entitlement or consequence. A valid `recipient` string can still identify an unauthorized external destination. Authorize authenticated attributes and the exact effect after parsing.
**Reasoning:** Syntax is a necessary precondition; policy is a separate predicate.
**Follow-up:** Add an allow-listed recipient group and a purpose claim.
**Acceptable alternative:** A formally verified schema may reduce parser bugs but still needs authorization.
**Red flag:** “The model only emits valid JSON.”
**Scoring:** 0 conflates syntax and policy; 1 names authorization; 2 explains effect binding.

### Q2. Compute the payload ceiling for q=20, c=3, b=4096.
**Answer:** 245,760 bytes, or 240 KiB.
**Reasoning:** TM-02 gives 20*3*4096.
**Follow-up:** If c=0, no effect is permitted.
**Acceptable alternative:** State decimal 245.76 kB only if the unit convention is labeled.
**Red flag:** Reporting 60 KiB.
**Scoring:** 0 wrong multiplication; 1 correct bytes; 2 correct binary conversion and interpretation.

### Q3. A prompt says “use tenant B”; the authenticated session is tenant A. What happens?
**Answer:** Treat the prompt claim as untrusted data and reject a request for B unless a separately authenticated authorized delegation exists.
**Reasoning:** Identity originates from the session boundary, not model text.
**Follow-up:** Inspect cache and retrieval keys for tenant scope.
**Acceptable alternative:** Ask for a new authenticated session.
**Red flag:** Let the model choose the tenant.
**Scoring:** 0 accepts text; 1 rejects; 2 traces all enforcement points.

### Q4. What does payload binding prevent, and what does it not prevent?
**Answer:** It prevents execution of a different canonical payload than the approved one. It does not ensure the preview was understandable or the underlying authorization policy was correct.
**Reasoning:** Digest equality protects integrity of the transition, not semantic quality.
**Follow-up:** Require clear rendering and policy version in the approval record.
**Acceptable alternative:** Signed immutable request object.
**Red flag:** “A hash makes it safe.”
**Scoring:** 0 magic-hash claim; 1 names mismatch; 2 states semantic limit.

### Q5. A test suite shows 0 harmful tool effects in 100 trials. Is risk zero?
**Answer:** No. The suite may miss adaptive or correlated attacks and its sampling process defines the conclusion. Report the suite, oracle, and uncertainty; do not infer population safety.
**Reasoning:** Absence in a finite nonrandom suite is bounded evidence.
**Follow-up:** Add adversarial, benign, and permission-boundary slices.
**Acceptable alternative:** Use a confidence interval only with stated sampling assumptions.
**Red flag:** “100% secure.”
**Scoring:** 0 zero-risk claim; 1 finite-sample caution; 2 names coverage and oracle.

### Q6. Which control contains a compromised model that can propose arbitrary shell commands?
**Answer:** A narrow executor with command allow-list, scoped credentials, resource limits, default-deny egress, and an effect journal; not prompt instructions alone.
**Reasoning:** Containment sits below the model proposal.
**Follow-up:** Make the task read-only if no effect is required.
**Acceptable alternative:** Replace shell with typed domain operations.
**Red flag:** Trust the system prompt.
**Scoring:** 0 prompt-only; 1 mentions sandbox; 2 composes authorization and containment.

### Q7. Why does an idempotency key matter to security as well as reliability?
**Answer:** It prevents a replayed or ambiguous retry from multiplying an authorized effect and provides a receipt lineage for investigation.
**Reasoning:** A quota is meaningful only if repeated delivery does not evade it.
**Follow-up:** Bind the key to principal, tenant, operation, and payload digest.
**Acceptable alternative:** Durable effect journal with uniqueness constraint.
**Red flag:** Retry until success without recording prior commit.
**Scoring:** 0 reliability only; 1 duplicate prevention; 2 binds identity and payload.

### Q8. A sandboxed task exfiltrates data through a permitted webhook. What failed?
**Answer:** The egress/purpose policy or credential scope, not necessarily sandbox isolation. Review destination allow-list, proxy identity, payload limits, and data classification.
**Reasoning:** Isolation does not make an allowed channel safe.
**Follow-up:** Revoke credential and inspect effect journal.
**Acceptable alternative:** Disable webhook pending review.
**Red flag:** Rebuild the container only.
**Scoring:** 0 container-only fix; 1 names egress; 2 separates containment from authorization.

### Q9. When should a human approval gate be used?
**Answer:** When consequence, uncertainty, irreversibility, or policy requires a human decision and the review can see a bounded exact effect. It is not a default substitute for deterministic low-risk automation.
**Reasoning:** Humans must receive a meaningful decision, not an unbounded conversation.
**Follow-up:** Specify timeout, escalation, and queue capacity.
**Acceptable alternative:** Dual control for high-impact actions.
**Red flag:** “Approve every token.”
**Scoring:** 0 universal answer; 1 consequence threshold; 2 adds bounded preview and operations.

### Q10. Change the constraint: approval latency must be under 30 seconds but reviews take 45 seconds. What changes?
**Answer:** Do not quietly remove approval. Narrow the action to a deterministic low-risk scope, pre-authorize bounded capability under policy, add reviewer capacity, or keep the manual baseline.
**Reasoning:** The binding constraint reverses feasibility of the original workflow.
**Follow-up:** Measure latency distribution and false-negative risk before changing policy.
**Acceptable alternative:** Asynchronous completion if the business process permits it.
**Red flag:** Raise the threshold without risk analysis.
**Scoring:** 0 bypasses control; 1 names a tradeoff; 2 offers tested alternatives.

**Takeaway:** Strong answers distinguish evidence, proposal, authorization, and committed effect under a changed constraint.

### Q11. An approval is valid for 60 records, but a worker has already committed 55. May it commit a 10-record retry?
**Answer:** No. The current budget check fails even if the payload digest, principal, and expiry still match.
**Reasoning:** TM-04 requires both the authorization snapshot and `Q_remaining`; TM-05 makes the remaining allowance a conserved state, not a UI hint.
**Follow-up:** Ask whether the retry has the original operation ID. If it does, return its prior receipt; if it does not, treat it as a new request and evaluate it independently.
**Acceptable alternative:** Reserve capacity atomically before execution and release a reservation only on a proven non-commit.
**Red flag:** “The approval was for 60, so another 10 is close enough.”
**Scoring:** 0 permits overdraw; 1 denies it; 2 explains atomicity and idempotency.

### Q12. A reviewer approves a canonical payload, then the recipient directory changes display names. What must be rechecked?
**Answer:** Re-resolve the recipient identifier and recheck its current authorization/revocation status at commit; do not silently replace the approved identifier with a newly matched display name.
**Reasoning:** Approval binding preserves the reviewed payload, while execution-time validity protects against changed eligibility. These are different facts.
**Follow-up:** Decide whether a changed recipient attribute is a hard denial or requires a new preview under policy.
**Acceptable alternative:** Bind a durable directory object ID and fail closed if it no longer resolves.
**Red flag:** Re-run fuzzy name matching at execution and send to its new best match.
**Scoring:** 0 relies only on the old preview; 1 performs a current check; 2 preserves the snapshot and prevents substitution.

### Q13. A denial receipt appears, but a downstream partner says it received the export. What is the first investigation branch?
**Answer:** Treat the executor receipt and partner assertion as conflicting evidence. Freeze the relevant capability, locate the idempotency/effect-journal record by correlation ID, and determine whether another endpoint or an out-of-band route committed the effect.
**Reasoning:** A denial record proves only what that path recorded. It cannot prove global non-delivery when alternate execution surfaces exist.
**Follow-up:** Compare commit timestamps, destination delivery identifiers, queue acknowledgements, and policy decision IDs while preserving the minimum necessary evidence.
**Acceptable alternative:** Disable all equivalent exporters pending scope assessment when the potential impact justifies it.
**Red flag:** Delete the denial receipt or assume the partner is mistaken.
**Scoring:** 0 treats one log as complete proof; 1 scopes the investigation; 2 identifies alternate-path and evidence-integrity risks.

### Q14. Why is a capability with tenant scope but no purpose scope incomplete?
**Answer:** It can prevent an action in the wrong tenant while still permitting an inappropriate action within the correct tenant, such as exporting a sensitive report for an unrelated purpose.
**Reasoning:** Tenant is an ownership boundary; purpose is a contextual authorization constraint. Neither replaces the other.
**Follow-up:** Define who asserts purpose, which purposes are enumerated, and which server-side policy consumes it.
**Acceptable alternative:** A narrowly named operation can encode part of the purpose if its parameters and policy make the scope unambiguous.
**Red flag:** “Same tenant means any use is allowed.”
**Scoring:** 0 conflates tenancy with authorization; 1 names purpose; 2 relates purpose to policy and audit.

### Q15. A red-team prompt causes the model to emit forbidden shell text, but the executor denies it. Is the test a pass?
**Answer:** It is a narrow pass for the observed effect oracle and a finding about model susceptibility. It is not evidence that the entire system is secure or that the denial path is available under all races, endpoints, and identities.
**Reasoning:** The model proposal and committed effect have different security meanings. A robust evaluation records both without collapsing them.
**Follow-up:** Test alternate endpoints, malformed inputs, retry paths, and denial observability; evaluate whether repeated proposals can exhaust review or logging capacity.
**Acceptable alternative:** Classify the result as “contained unsafe proposal” with the precise suite and harness version.
**Red flag:** Report “injection solved.”
**Scoring:** 0 calls it universal safety; 1 distinguishes proposal and effect; 2 identifies coverage limits.

### 11.1 Runnable synthetic threat and trace oracle

**Status:** NOT EXECUTED. This is a runnable, dependency-free Python 3 oracle specification for constructed teaching inputs. It checks arithmetic and deterministic transitions only; it does not test a real identity provider, digest implementation, sandbox, model, network, policy engine, or compliance posture. The expected outputs below are derived from the declared inputs and have not been executed in this document.

```python
# File: illustrative_execution_trust_oracle.py
# Evidence: ILLUSTRATIVE / NOT EXECUTED
from dataclasses import dataclass
from hashlib import sha256

def digest(payload: str) -> str:
    return sha256(payload.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class Approval:
    tenant: str
    purpose: str
    payload_digest: str
    policy_version: str
    remaining_records: int
    expiry_minute: int

def decide(payload, tenant, purpose, policy_version, now_minute, requested_records,
           approval, operation_id, journal):
    if operation_id in journal:
        return journal[operation_id]
    if tenant != approval.tenant or purpose != approval.purpose:
        return "DENY_SCOPE"
    if policy_version != approval.policy_version:
        return "DENY_POLICY_VERSION"
    if now_minute >= approval.expiry_minute:
        return "DENY_EXPIRED"
    if digest(payload) != approval.payload_digest:
        return "DENY_APPROVAL_PAYLOAD_MISMATCH"
    if requested_records < 0 or requested_records > approval.remaining_records:
        return "DENY_BUDGET"
    receipt = f"COMMIT:{operation_id}:{requested_records}"
    journal[operation_id] = receipt
    return receipt

payload_a = '{"destination":"tenant-a-archive","records":[1,2,3]}'
approval = Approval("tenant-a", "research-export", digest(payload_a), "P7", 60, 30)
journal = {}

assert 20 * 3 * 4096 == 245760
assert decide(payload_a, "tenant-a", "research-export", "P7", 10, 55,
              approval, "op-17", journal) == "COMMIT:op-17:55"
assert decide(payload_a, "tenant-a", "research-export", "P7", 10, 55,
              approval, "op-17", journal) == "COMMIT:op-17:55"
assert decide(payload_a, "tenant-a", "research-export", "P7", 10, 10,
              approval, "op-18", journal) == "DENY_BUDGET"
payload_b = '{"destination":"external.invalid","records":[1,2,3]}'
assert decide(payload_b, "tenant-a", "research-export", "P7", 10, 3,
              approval, "op-19", journal) == "DENY_APPROVAL_PAYLOAD_MISMATCH"
assert decide(payload_a, "tenant-b", "research-export", "P7", 10, 3,
              approval, "op-20", journal) == "DENY_SCOPE"
assert decide(payload_a, "tenant-a", "research-export", "P8", 10, 3,
              approval, "op-21", journal) == "DENY_POLICY_VERSION"
assert decide(payload_a, "tenant-a", "research-export", "P7", 30, 3,
              approval, "op-22", journal) == "DENY_EXPIRED"
print("illustrative oracle assertions passed")
```

The oracle has one deliberately narrow design choice: a duplicate operation returns its original journal receipt before it reevaluates the supplied arguments. This models an operation identifier as the identity of a prior attempted effect. A production protocol must authenticate who is allowed to query that receipt and bind the idempotency key to a canonical request; otherwise an attacker could probe operation existence.

#### Oracle input ledger and expected outcomes

| Test ID | Constructed input | Expected deterministic result | Security claim it can support | What it cannot establish |
|---|---|---|---|---|
| ORC-01 | `q=20,c=3,b=4096` | `245760` payload bytes | arithmetic for TM-02 | actual payload limit or storage use |
| ORC-02 | `op-17`, matching tenant/purpose/digest, `55<=60` | `COMMIT:op-17:55` | stated trace reaches one commit branch | delivery, data correctness, or authorization in a real service |
| ORC-03 | repeat `op-17` | same receipt | intended idempotent branch in this toy oracle | distributed exactly-once behavior |
| ORC-04 | new `op-18`, request 10 after 55 | `DENY_BUDGET` | bounded ledger arithmetic | atomic cross-worker reservation |
| ORC-05 | payload destination differs | `DENY_APPROVAL_PAYLOAD_MISMATCH` | digest mismatch transition | collision resistance or preview semantics |
| ORC-06 | `tenant-b` | `DENY_SCOPE` | tenant equality predicate | identity issuer correctness |
| ORC-07 | policy `P8` | `DENY_POLICY_VERSION` | snapshot-version comparison | policy adequacy |
| ORC-08 | minute `30`, expiry `30` | `DENY_EXPIRED` | chosen strict-expiry boundary | clock synchronization behavior |

Run protocol, if a practitioner elects to execute it: save exactly the code block as UTF-8, use a declared Python 3 interpreter, record interpreter version and file digest, run `python illustrative_execution_trust_oracle.py`, and preserve only the exit status and constructed test identifiers. Do not paste real destinations, credentials, tenant identifiers, or payloads into this teaching harness.

#### Trace oracle: input-to-output contract

| Field | Contract |
|---|---|
| Input identity | authenticated tenant and purpose supplied by middleware, not the model |
| Input proposal | canonical payload string whose digest was previewed |
| State | immutable approval plus durable operation-to-receipt journal |
| Transition | reject duplicate as prior receipt; otherwise enforce scope, policy, expiry, digest, then budget before commit |
| Output | one of a bounded denial code or a receipt identifier |
| Invariant | a new commit occurs only when every predicate is true and the requested amount is no greater than remaining budget |
| Recovery | ambiguous transport result queries journal by authorized operation identity; it never blindly retries a new operation |

**Takeaway:** An executable toy oracle is useful because it makes a claimed invariant falsifiable; it does not turn the constructed workflow into evidence about a deployed system.

### 11.2 Design-review drill: reconstruct a threatened execution surface

Use this drill before proposing a control. The answers are intentionally concrete enough for a reviewer to reject.

1. **Name the effect.** “Export records” is insufficient. State `send 0–20 classified record bodies to a resolved destination using one tenant-scoped operation`.
2. **Name the commit owner.** It is an executor service, not the model, UI, queue, or reviewer.
3. **Locate authority sources.** Session identity, policy decision, approval record, quota ledger, and executor credential are separate sources.
4. **Enumerate alternate routes.** Batch export, single-record export, retry worker, administrative endpoint, callback, and direct downstream API are each candidates until excluded.
5. **Define the denial receipt.** Include operation ID, authenticated principal pseudonym or reference, tenant, policy decision ID, reason class, timestamp, and payload digest; exclude raw sensitive payload by default.
6. **Choose recovery semantics.** A client timeout is “unknown outcome,” not “safe to repeat.” Require journal lookup or an idempotent replay.
7. **Set a human boundary.** Specify which non-derivable risk requires approval, exactly what preview the reviewer sees, timeout behavior, and escalation owner.
8. **Write negative tests.** Mutated recipient, revoked session, exhausted quota, replay, stale policy, malformed canonicalization, and alternate endpoint all require separate tests.

Worked micro-trace, ILLUSTRATIVE and NOT EXECUTED:

```text
T0 session S-4 authenticates tenant=A, purpose=research-export.
T1 model proposes {records:[11,12], destination:tenant-a-archive}.
T2 executor resolves record IDs under tenant A and canonicalizes payload P.
T3 policy says approval required; preview shows 2 records, destination, purpose, P7, expiry.
T4 reviewer approves digest H(P), remaining budget=3 records.
T5 worker receives operation O-9; current revocation check passes.
T6 executor atomically consumes 2 records of budget and commits one outbound request.
T7 journal stores receipt R-9 with H(P), O-9, policy P7 and result class.
T8 duplicate delivery of O-9 returns R-9; it cannot consume remaining budget again.
```

At T6, a database transaction, conditional write, or a comparable single-writer protocol is needed. A read-modify-write sequence without exclusion is not equivalent: two workers can each observe three remaining records, each commit two, and leave four records committed against a three-record approval. The threat is a state race, even if every individual request appears authorized in isolation.

### 11.3 Threat-model questions that force a decision

| Reviewer question | Good evidence | Decision consequence if absent |
|---|---|---|
| Which interface actually commits the effect? | endpoint inventory and trace correlation | retain read-only design |
| Which fields are authoritative and where were they asserted? | middleware contract and logs with provenance | do not accept model-derived identity |
| Can the same action reach a downstream service another way? | route inventory plus negative tests | scope blast radius to all routes |
| What turns an unknown timeout into a known outcome? | durable journal and lookup semantics | do not issue blind retry |
| Which quantity is bounded, and at which transition? | atomic ledger design and oracle | do not cite a UI limit |
| What can a reviewer understand and change? | preview artifact and rejection/edit flow | approval is not a meaningful control |
| What does revocation invalidate immediately? | policy and execution contract | expire or reapprove queued work |
| What evidence survives an incident without retaining payload bodies? | receipt schema and retention design | add minimum viable forensic trail |

**Takeaway:** A review should fail a vague “secure agent” statement until it names the committed effect, authoritative fields, state owner, and denial/recovery evidence.
### 8.1 Failure investigation playbooks

The following playbooks deepen the earlier failure entries. They are diagnostic procedures, not claims that the failures occurred in a particular system. Every scenario is HYPOTHETICAL / ILLUSTRATIVE and NOT EXECUTED.

#### F-07 Canonicalization changes the approved meaning

**Trigger/environment:** The preview service serializes a recipient list in display order, while the executor serializes it in normalized identifier order. Both compute a digest, but the policy reviewer was shown display labels.

**Symptom:** Benign requests are denied after approval, or an implementation “fixes” the mismatch by hashing only selected fields.

**Competing causes:** Different Unicode normalization; omitted default field; changed list ordering; server resolution replacing a requested object; a genuine attempted payload change.

**Discriminating diagnostic:** Persist the exact canonical-byte representation version and field schema with the approval. Reconstruct it from the same durable input. The expected signal for a format mismatch is differing bytes with semantically identical typed fields; the expected signal for a semantic change is one different authoritative typed field.

**Root-cause evidence:** A digest mismatch alone proves no root cause. Compare parser version, canonicalization version, field-presence rules, and resolved IDs without exposing content unnecessarily.

**Immediate mitigation:** Deny commit and ask for a fresh preview. Never fall back to a partial digest match.

**Correction:** Define one canonical typed representation owned by a versioned library; make preview and executor consume that library or versioned output.

**Prevention:** Add fixtures for reordered keys, null versus absent values, Unicode, repeated fields, and destination aliases. Include canonicalization version in approval records.

**Verification:** For every fixture, preview and executor produce identical bytes and digest; intentionally changed authoritative fields produce denial. This test remains a proposed test until executed in the actual components.

**Mitigation risk and escalation:** A strict change can deny pending work after a library upgrade. Expire affected approvals and route high-priority requests to an authorized manual procedure; do not silently reinterpret them.

#### F-08 Revocation is recorded but queued work still commits

**Trigger/environment:** A capability is revoked after approval while a worker already holds a queued message.

**Symptom:** A receipt shows a commit timestamp after a revocation timestamp.

**Competing causes:** Clock skew; revocation propagation delay; executor checks a cache with excessive time-to-live; an alternate credential was used; the record was revoked after rather than before the true commit.

**Discriminating diagnostic:** Compare monotonic sequencing where available, authoritative revocation version, executor decision time, downstream commit acknowledgement, and clock source. Do not order distributed events solely by a user-facing wall clock.

**Root-cause evidence:** Establish whether the executor queried current validity at T5/T6, which revocation version it observed, and which credential/route committed the action.

**Immediate mitigation:** Disable the relevant effect route, invalidate remaining queued operations, and preserve receipt/journal metadata according to incident procedure.

**Correction:** Make current revocation a mandatory execution predicate for high-consequence operations; bound cache freshness and explicitly test stale-cache behavior.

**Prevention:** Include a revocation-race test in the release suite and define maximum acceptable propagation delay as a scenario constraint, not a presumed property.

**Verification:** In a controlled synthetic environment, revoke between approval and commit and assert denial. A passing synthetic result does not establish propagation behavior in production.

**Mitigation risk and escalation:** Aggressive global invalidation can halt legitimate work. Escalate to the service owner and risk owner to choose scoped versus global disablement.

#### F-09 A receipt is complete-looking but cannot support reconciliation

**Trigger/environment:** The executor logs “success” with a request timestamp but no operation ID, payload digest, policy decision, or downstream acknowledgement reference.

**Symptom:** Operations cannot determine whether a timeout produced zero, one, or multiple effects.

**Competing causes:** Log sampling; independent downstream success without a local commit; journal retention expired; a retry created a new request identity.

**Discriminating diagnostic:** Attempt a one-to-one join from client operation to approval, policy decision, executor decision, quota update, and downstream delivery result. The expected healthy property is exactly one bounded lineage per committed operation.

**Root-cause evidence:** Missing correlation is an observability gap, not evidence of no effect.

**Immediate mitigation:** Stop blind retries; query downstream using an authorized business identifier if one exists; mark outcome unknown until reconciliation.

**Correction:** Define a receipt schema before launch. Store minimal linkage fields and outcomes with access controls; keep raw payload out unless separately justified.

**Prevention:** Require chaos/timeout drills to produce reconcilable receipts before enabling an effect.

**Verification:** Inject a synthetic post-commit response loss. Operators must determine the single committed state without duplicate execution. This is a proposed exercise, not an executed result.

**Mitigation risk and escalation:** Longer retention increases privacy exposure. Set retention by documented need and involve privacy/security owners where policy requires.

#### F-10 Human approval becomes an attacker-controlled denial-of-service queue

**Trigger/environment:** Every ambiguous request is escalated, and untrusted inputs can create many plausible-looking approval cards.

**Symptom:** Queue age rises, urgent requests wait behind low-value requests, reviewers begin rubber-stamping or bypassing the queue.

**Competing causes:** Legitimate seasonal demand; malicious flooding; preview rendering regression raising service time; staffing outage; one tenant dominating shared capacity.

**Discriminating diagnostic:** Partition arrivals by tenant, effect class, source, and verdict; compare service-time distributions before and after preview changes. The signal of a flood is a shift concentrated in a controllable entry class, not merely a high total count.

**Root-cause evidence:** Capture bounded request metadata and denial reason classes. Avoid retaining malicious content bodies simply because they are diagnostically interesting.

**Immediate mitigation:** Apply admission limits, prioritize high-consequence queues, and preserve the manual baseline for urgent legitimate work.

**Correction:** Separate low-risk deterministic workflows from high-impact approvals; use quotas per authenticated principal/tenant and make reviewer capacity explicit.

**Prevention:** Capacity-test with adversarial arrival patterns and train reviewers on rejection/escalation rather than speed alone.

**Verification:** A proposed load harness should show protected-class requests retain their service objective while excess requests receive deterministic admission outcomes. It must not use production identities or data.

**Mitigation risk and escalation:** Rate limits can suppress legitimate incident response. Define an authenticated emergency path with dual control and post-event review.

#### F-11 Policy evaluates the correct request but the wrong data snapshot

**Trigger/environment:** The model supplies record IDs, and the executor re-resolves them after records have changed classification or membership.

**Symptom:** The receipt is internally consistent, yet it does not explain which version of the data was exported.

**Competing causes:** Expected live-data semantics; stale cache; identifier reuse; unauthorized record substitution; policy design omitted snapshot choice.

**Discriminating diagnostic:** Record record-version identifiers or a deterministic snapshot reference in the preview and receipt. Compare the authorization-time set to execution-time resolution.

**Root-cause evidence:** A changed record set can be a valid business change or a security-impacting substitution; classification and authorization policy determine which.

**Immediate mitigation:** Deny if a declared immutable set changed; otherwise re-preview the changed set under an explicit live-data policy.

**Correction:** State per operation whether data identity and classification are snapshotted, revalidated, or both.

**Prevention:** Include deletion, reclassification, and identifier-reuse cases in the threat model.

**Verification:** In a constructed trace, change one record classification between approval and execution and assert the declared policy branch. No generic answer is correct without that policy.

**Mitigation risk and escalation:** Excessive snapshotting can retain data longer than necessary. Limit references and expiry; consult data governance owners for actual retention decisions.

### 9.1 Costed control choices with explicit boundaries

The following is a constructed planning comparison, not a price quote or staffing recommendation. It prevents a common error: claiming a control is “cheap” after omitting reviewer, latency, storage, recovery, and false-denial costs.

| Option | Direct mechanism | Illustrative recurring inputs | Benefit measured here | Missing cost / disqualifier |
|---|---|---|---|---|
| Read-only assistant | no executor | model/infrastructure only | removes external-effect path | may not meet workflow objective |
| Typed executor + policy | deterministic scope check | predicate lookup, journal write, engineering maintenance | can deny unsupported effect | cannot decide ambiguous business purpose alone |
| Approval-bound executor | typed executor plus reviewer | reviewer minutes, queue, preview design, escalation | adds human decision for bounded action | unsafe if review is not comprehensible or capacity is absent |
| General service credential | low apparent integration cost | hidden blast-radius/recovery cost | rapid prototype | disqualified for unbounded tenant/effect authority |

Illustrative reviewer labor calculation:

```text
FORMULA [TM-07]: H = (n * s) / 3600
WHERE: H reviewer-hours per hour of arrivals [reviewer-hour/hour];
       n requests/hour; s seconds/request.
CONSTRAINT: each request receives one review; no breaks, rework, batching,
            context-switching, or simultaneous review are included.
NUMERICAL EXAMPLE: ILLUSTRATIVE n=40 requests/hour, s=45 seconds/request;
                   H=(40*45)/3600=0.5 reviewer-hour/hour.
INTERPRETATION: nominally half of one continuously available reviewer is consumed;
                staffing must include nonproductive time and demand variation.
```

Derivation: each request takes `s` seconds; `n` requests consume `n*s` reviewer-seconds; dividing by 3,600 seconds/hour converts to reviewer-hours per hour. The units cancel correctly: `(requests/hour * seconds/request) / (seconds/hour) = reviewer-hours/hour`.

Sensitivity: at 80 requests/hour and 60 seconds/request, `H=4800/3600=1.333...` reviewer-hours/hour. One nominal reviewer cannot keep up even before breaks. At 20 requests/hour and 30 seconds/request, `H=0.1667` reviewer-hours/hour. This calculation does not predict queue percentile latency because arrivals and service time variability are omitted.

**Takeaway:** Compare controls on the consequence they bound and the workload they create; a nominally inexpensive approval gate can be operationally infeasible.

### 10.1 Results reference: claims to use carefully

| Result or principle | Precise use in this sheet | Preconditions | Invalid inference |
|---|---|---|---|
| Least privilege | limit each executor credential and operation to the smallest stated need | inventory of effects and credentials is materially complete | small scopes prove no abuse |
| Complete mediation | check policy at the effect boundary, including alternate routes | every relevant commit route uses the same enforcement point | one gateway check covers hidden/direct APIs |
| Fail closed | deny when an essential authorization fact is unavailable | availability tradeoff and recovery path are specified | every system error should automatically deny without considering safe degradation |
| Separation of duties | distinguish requester, reviewer, and executor where a policy requires independent judgment | identities and authority are not colluded or reused | two UI clicks by one principal are dual control |
| Idempotency | map repeated delivery of the same operation to one effect/result lineage | durable operation identity and atomic commit/journal relationship | generic retry is safe |

No result above certifies an architecture. They are design heuristics and properties to test against a particular contract.

### 3.1 Authority provenance mini-ledger

Before implementation, fill a ledger like this for every field that can influence an effect.

| Field | Example value | Authoritative issuer | Model may see? | Model may propose? | Executor must verify | Retention boundary |
|---|---|---|---|---|---|---|
| Tenant | `tenant-a` | authenticated session middleware | yes, as context label | no | match capability and request route | receipt reference only |
| Purpose | `research-export` | policy-controlled workflow | yes | may select only enumerated task | policy allow-list | policy decision ID |
| Record IDs | `[11,12]` | server-side resolution under tenant | optionally | yes, as candidates | existence, classification, access | digest/version reference |
| Destination | archive object ID | directory/policy service | display label only | may propose candidate | resolved ID and allow-list | resolved ID/pseudonym |
| Approval | `A-91` | approval service | no bearer token | no | status, expiry, digest, policy version | durable decision record |
| Budget | 60 records | quota ledger | remaining count may be shown | no | atomic remaining value | receipt delta |
| Executor credential | workload identity | identity issuer | no | never | audience and scope | credential metadata, not secret |

The ledger catches a subtle defect: a field can be visible to a model without being authoritative, and a model can propose a field without being permitted to assert it. Treating those verbs as synonyms is a common source of prompt-to-privilege escalation.

**Takeaway:** For each effectful field, record who asserts it, who can propose it, who verifies it, and what evidence remains after the operation.
### 11.4 Conformance scenarios for a proposed executor

These scenarios are a test-design checklist. They are not test results. A team should replace identifiers and values with isolated synthetic fixtures, execute in its own authorized environment, and record versioned evidence. A “pass” in one scenario does not prove resilience to unenumerated paths.

| Scenario ID | Preconditions | Stimulus | Required oracle | Failure interpretation |
|---|---|---|---|---|
| CT-01 | valid session, allowed operation | matching typed request | one commit receipt with all lineage fields | a denial may indicate policy/configuration; an uncorrelated success is observability failure |
| CT-02 | valid session, forbidden destination | model proposes destination in untrusted content | denial before downstream effect | text refusal alone is insufficient |
| CT-03 | two tenant sessions | identical proposed record ID | each resolution uses own tenant scope | common result set indicates possible cross-tenant leakage |
| CT-04 | approved payload | mutate one authoritative field | digest/payload denial and no commit | acceptance is an approval-binding failure |
| CT-05 | approval near expiry | execute at and after expiry boundary | declared strict boundary behavior | ambiguous clock semantics require design clarification |
| CT-06 | sufficient budget | deliver same operation twice | one effect lineage and stable receipt | two commits indicate idempotency/atomicity defect |
| CT-07 | budget nearly exhausted | concurrent distinct operations | total committed amount never exceeds allowance | negative balance or excess receipt is a race |
| CT-08 | queued approved operation | revoke capability before commit | no commit after current check | commit needs route/cache investigation |
| CT-09 | sandboxed executor | attempt denied egress to canary | denied connection and bounded audit event | success means containment boundary is absent or misconfigured |
| CT-10 | downstream timeout after possible commit | client retries by protocol | journal resolves outcome without second effect | blind retry is recovery failure |
| CT-11 | policy change | execute prior approval | explicit snapshot/reapproval rule observed | silent use of changed semantics is policy-integrity ambiguity |
| CT-12 | reviewer threshold | adversarial burst of low-value requests | admission/priority behavior remains explicit | unbounded backlog is an availability control failure |

#### Reading an oracle failure without overclaiming

When CT-04 accepts a mutated payload, first determine whether the test truly reached the committing executor. A browser/UI test that observes a “success” toast does not establish a downstream commit. Gather the test operation ID, approval ID, canonical bytes/digest version, executor decision, quota update, and destination response reference. Then classify the failure:

- **Test-harness defect:** the fixture did not mutate an authoritative field, used an expired approval, or invoked a mock route. Fix the harness and document the prior invalid result.
- **Preview/executor disagreement:** two canonicalization implementations disagree. Treat this as a high-priority integrity defect; do not loosen comparison.
- **Alternate-path bypass:** the test called an endpoint not covered by the broker. Expand the interface inventory and constrain or retire the bypass.
- **Authorization defect:** the executor accepted a digest mismatch or used model-controlled attributes. Disable the effect surface or bound it to manual operation pending repair.
- **Observability defect:** the effect may have occurred but receipts cannot establish it. Contain and reconcile rather than claiming a pass or failure of the external action.

This classification avoids two opposite errors. One is blaming the model when deterministic enforcement failed. The other is declaring a control successful when the test observed only a simulated or pre-commit path.

#### Minimal evidence packet for one scenario

| Packet item | Why it exists | Avoid including |
|---|---|---|
| scenario ID and source revision | connects conclusion to exact test design | vague “security test passed” statement |
| synthetic fixture manifest | makes inputs and tenant/destination classes reproducible | real customer data or credentials |
| policy and canonicalization versions | explains which rules produced the decision | copied secrets or broad configuration dumps |
| correlation/operation/approval references | reconstructs one path | raw payload unless separately approved |
| expected and observed decision class | permits a bounded pass/fail judgment | unstated assumptions |
| downstream observation method | distinguishes executor result from external effect | claims that a mock is a production delivery |
| limitation and recheck trigger | stops inappropriate generalization | boilerplate “may vary” without a trigger |

### 8.2 Edge-boundary decision table

| Edge condition | Unsafe shortcut | Bounded decision | Why |
|---|---|---|---|
| Recipient name is ambiguous | choose the first directory match | request an identifier/clarification or retain manual routing | display text is not identity |
| Policy service unavailable | use model confidence | fail closed for high-consequence effects; offer read-only or queued workflow only if explicitly designed | confidence does not replace policy |
| Approval expired during outage | extend it implicitly | require new approval or use a predeclared emergency procedure | expiry is part of the reviewed scope |
| Journal unavailable after downstream commit | retry with a new ID | hold as unknown and reconcile downstream | retry can multiply effect |
| Data classification changes | keep old approval silently | recheck/re-preview according to declared snapshot rule | policy must state time semantics |
| Review backlog grows | waive approval globally | admission control, capacity adjustment, or narrowed deterministic operation | bypass changes risk, not just latency |
| Sandbox needs an external dependency | permit all egress | allow one authenticated destination with data/purpose limits | isolation and network policy are separate |

**Takeaway:** Edge cases are where a stated trust boundary either has a deterministic behavior or quietly disappears.
### 7.1 Selection sequence for an effect-bearing feature

Use this sequence to make the baseline explicit rather than adding controls after a model is already connected to broad credentials.

1. **Can the user objective be met without an external effect?** If yes, select a read-only answer or a draft. This removes the execution surface rather than trying to harden it.
2. **Can the effect be expressed as one typed, narrow operation?** If no, split the workflow. A general shell or generic HTTP client usually hides policy-relevant distinctions.
3. **What authenticated facts are necessary?** Specify principal, tenant, purpose, operation, data class, destination, quantity, and time. If a fact cannot be authenticated, do not use it to grant privilege.
4. **What is the strongest credible non-human policy?** Enforce allow-lists, scope, quotas, revocation, and idempotency at the executor before considering approval.
5. **What uncertainty remains that a reviewer can genuinely decide?** Add approval only when it receives an exact, bounded preview and has timeout, edit, reject, and escalation behavior.
6. **How will an unknown outcome recover?** Define journal lookup, reconciliation ownership, and the point at which operators freeze further effects.
7. **Which negative path would reverse confidence?** Add one oracle for each boundary: scope, payload, budget, revocation, alternate route, and downstream ambiguity.

Worked selection: an assistant that drafts an export message selects step 1: no external send and no export capability. An assistant that must send a report selects step 2: `submit_export_for_approval`, not `http_request`. It derives recipient from an authenticated directory lookup, applies a record cap, and creates an approval preview. This is a proposed design pattern; it is not a statement about any existing assistant.

#### Selection anti-patterns and their replacements

| Anti-pattern | Why it fails | Replace with |
|---|---|---|
| “The model was instructed not to do that.” | instruction is not an enforcement decision | typed executor and independent policy |
| “The payload passed JSON schema.” | shape does not establish entitlement | server-side resolution and authorization |
| “The reviewer saw the chat.” | chat can change after review | canonical preview plus digest/version binding |
| “The service account is internal.” | network placement does not define allowed effect | workload identity, audience, scope, and budget |
| “Retry until it works.” | unknown outcome can become duplicate effect | idempotency and reconciliation |
| “Sandboxed means safe.” | allowed egress/credentials can still carry harm | remove secrets, restrict egress, authorize destination |

**Takeaway:** Start by removing unnecessary effects; for necessary effects, make every authority-bearing transition explicit and testable.
## 12. Quick Reference Card

| Field | Reference |
|---|---|
| Decision unit | One effect-bearing request with tenant, purpose, payload, budget, and receipt |
| First question | What asset can be read, changed, spent, disclosed, or executed? |
| Non-negotiable boundary | Model output is a proposal; executor independently authorizes exact effect |
| Core invariant | Recompute payload digest and match identity, scope, policy, expiry, and budget before commit |
| Sizing rule | `B_max = q*c*b`; count committed effects, not model calls |
| Top diagnostics | tenant source mismatch; approval digest mismatch; denied egress; replayed operation ID |
| Avoid | prompt-only controls, broad credentials, transcript-only approval, unbounded retries |
| Evidence boundary | Illustrative examples are not incident rates or deployment results |
