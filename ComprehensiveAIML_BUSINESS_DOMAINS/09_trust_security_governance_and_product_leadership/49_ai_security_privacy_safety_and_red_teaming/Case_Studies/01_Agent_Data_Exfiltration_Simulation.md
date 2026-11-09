---
case_id: "CASE-49-001"
case_title: "Approval-bound export for a retrieval assistant"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "AI security / controlled data export"
primary_pattern: "authorization-scope"
secondary_patterns: ["approval-payload-binding", "evaluation-mismatch"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "Security and data-platform owner"
source_index: "INDEX.md"
review_trigger: "Any change to export tool, identity, data class, approval policy, or executor."
---

# Approval-Bound Export for a Retrieval Assistant

## 1. The One-Line Framing

Given one tenant-scoped research request, decide whether to permit a bounded data export for an authenticated analyst, subject to exact-payload approval and a 240 KiB authorization ceiling, compared with a baseline that executes schema-valid model JSON. This is an **illustrative design exercise**: no company, model, incident, test run, or outcome is represented. The primary pattern is authorization scope; the baseline fails because valid JSON does not establish identity, purpose, recipient, or effect budget.

## 2. Context & Background

An internal assistant retrieves approved knowledge-base documents and drafts research notes. An analyst may need a small export to a case workspace. The workflow is request → retrieval → proposed export → policy/approval → executor → receipt. The security owner owns export policy; the data steward owns data classification; the platform team owns executor correctness; a reviewer owns approval decisions.

The baseline has a service account and schema validator. It cannot show whether a document injected a recipient, whether a model chose another tenant, or whether retries duplicated a committed export. Scale, jurisdiction, regulatory obligations, and actual data sensitivity are unknown; this scenario uses constructed values only. The required boundary is that the assistant may propose, but an executor independently decides and journals every effect.

## 3. The Business Question

| Decision contract | Detail |
|---|---|
| Question | Should the team permit bounded assistant-proposed exports, or retain a read-only/manual baseline? |
| Decision owner | Security and data-platform owner |
| Trigger/deadline | A pilot needs a controlled export workflow before access expands |
| Actions | read-only baseline; schema-only export; scoped capability plus approval; defer pilot |
| Objective | enable eligible export without unauthorized disclosure |
| Hard constraints | tenant isolation; no unapproved destination; receipt for every commit |
| Scope | one export request, one tenant, one capability, one review window |
| Out of scope | legal approval, production staffing, and real incident response |

The business value is reduced manual copying for eligible work. The downside is an unauthorized disclosure or a review queue that blocks legitimate work. “Model quality” is not the objective because a fluent export proposal can still be unauthorized.

## 4. Assumptions & Constraints

| ID | Assumption/constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Session tenant is authenticated | identity | negative cross-tenant test | unverified | scope can be forged | read-only mode | platform |
| A-02 | Executor sees every effect path | architecture | endpoint inventory/replay | unverified | alternate bypass | disable export | platform |
| A-03 | Canonical payload is stable | data | digest test vectors | unverified | approval mismatch ambiguity | manual export | security |
| C-01 | >20 records requires approval | policy scenario | policy review | illustrative | incorrect routing | manual review | data steward |
| C-02 | Export cap is 240 KiB | containment scenario | quota oracle | illustrative | excess disclosure | deny request | security |

The two assumptions most likely to reverse the design are A-01 and A-02. A scoped capability cannot repair a tenant identity source that is model-controlled, and a perfect executor cannot protect an uninstrumented alternate API.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit is one export request keyed by operation ID. Fields are authenticated user ID, tenant ID, purpose, requested record IDs, destination, payload digest, policy version, approval ID, and receipt. No real data is used. The illustrative test set contains hostile documents, benign requests, and deterministic mock records. Point-in-time rule: policy and tenant scope are captured before the model receives a tool result.

### 5b. Tools & Stack

Proposed components are an identity middleware, policy decision point, approval service, narrow export executor, effect journal, and controlled test sink. Product/version behavior is not asserted. A model may generate a typed proposal but receives no direct export credential.

### 5c. Method Summary

`authenticated request -> provenance-labelled context -> typed proposal -> policy check -> preview/digest -> approval if threshold -> executor recomputation -> receipt`.

| Candidate | Strength | Limitation | Status |
|---|---|---|---|
| Manual/read-only baseline | least external effect | no assisted export | considered |
| Schema-only tool | rejects malformed arguments | accepts authorized-looking harm | rejected by reasoning |
| Scoped capability + approval | binds actor/scope/effect | needs operational review | proposed |

### 5d. Provenance, quality, exclusions and leakage controls

All records and rates are illustrative. Retrieved hostile text is labelled untrusted. Tenant ID originates only from identity middleware. Test logs retain payload digest and minimal receipt fields, not payload content. Unknown production retention, data classification, and authorization policy are explicit gaps.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | NIST AI RMF, https://doi.org/10.6028/NIST.AI.100-1 | 2023 | risk vocabulary | public | not case evidence |
| SRC-02 | OWASP prompt injection sheet | accessed 2026-11-09 | threat method | public | not a tested control |
| RUN-01 | illustrative oracle inputs below | 2026-11-09 | arithmetic/trace | reproducible by hand | not executed system run |

## 6. Analysis Walkthrough

The decisive design claim is narrow: an export is not committed unless authenticated attributes, budget, destination, policy version, expiry, and payload digest satisfy the executor. That claim is a proposed invariant, not a measured result.

```text
FORMULA [CASE-49-001-01]: B_max = q*c*b
WHERE: q records/call; c calls/approval; b bytes/record; B_max bytes/approval.
CONSTRAINT: central executor counts commits; payload-only capacity; stable operation ID.
NUMERICAL EXAMPLE: q=20, c=3, b=4096 bytes; B_max=245760 bytes=240 KiB.
INTERPRETATION: a request cannot legitimately exceed 60 records or 240 KiB under this scope.
```

Trace: request `OP-9` proposes 55 records to destination A. Three partitions 20,20,15 fit the call count and 55*4096=225,280 bytes fits the cap. Reviewer approves digest `d1`. A retrieved document changes destination to B, creating `d2`. The executor recomputes `d2`, observes `d2 != d1`, rejects, and commits no receipt. This is the proposed oracle.

Sensitivity: if calls/approval rises from 3 to 5, the cap rises from 240 KiB to 400 KiB. The design decision changes: policy owner must reassess whether the original approval context supports the enlarged blast radius. If executor sees only batch API but a single-record API exists, conclusion is invalid; disable export until endpoint coverage is restored.

What we tried first: no implementation was tested. Schema-only execution is a considered baseline, not a historical failed test. It is rejected analytically because schema validity lacks authorization semantics.

## 7. The Finding

The proposed pilot should use a narrow executor with tenant-scoped, expiring capability and exact-payload approval for exports above the stated threshold; it should not use the schema-only baseline. The constructed quota limits one approval to 240 KiB payload-only under its assumptions. Outcome is **not measured**: the recommendation depends on unverified identity provenance, complete endpoint inventory, canonicalization, and operational review capacity. If those assumptions cannot be demonstrated, retain the manual/read-only baseline.

## 8. System/Solution Design

```text
Identity -> request broker -> model/retrieval (untrusted content labelled)
                               -> typed export proposal
Policy + capability -> preview/digest -> approval service
                               -> narrow executor -> effect journal -> receipt
                         denial/revocation -> audit and recovery queue
```

The executor accepts only typed record identifiers and a server-side destination class. It recomputes the canonical digest; it does not execute model-supplied shell text. Each operation ID is idempotent: a retry returns the first receipt. The approval record includes principal, tenant, purpose, policy version, digest, budget, expiry, and reviewer identity. Failure path: any mismatch yields a denial receipt and no external effect; suspected compromise revokes capability and disables executor routing.

## 9. Evaluation & Validation

| Metric | Definition | Comparator/window | Guardrail | Status |
|---|---|---|---|---|
| Unauthorized commit | prohibited committed receipts / prohibited tests | schema-only vs proposed simulator | zero critical commits | proposed, not executed |
| False block | unnecessary denials / benign eligible tests | same manifest | review burden | proposed |
| Payload bound | committed bytes <= B_max | every receipt | no bypass endpoint | proposed |
| Review capacity | reviewed requests/hour | illustrative 8-hour day | queue growth | estimated only |

Proposed validation includes cross-tenant attempts, hostile retrieval text, changed destination after approval, duplicate delivery, expired approval, quota boundary, sandbox egress, and alternate endpoint discovery. No metric result is claimed. A test passes only if the journal verifies the expected receipt/denial, not if the model text sounds safe.

## 10. Action Taken

No production action was taken. The recommended next action is a bounded simulator implementation, endpoint inventory, identity review, and policy-owner approval of thresholds before any pilot. The decision owner must explicitly accept or reject the residual risk; engineering cannot infer authorization from this document.

## 11. Outcome & Results

**Not measured.** The only checked outputs are hand-derived arithmetic: 20*3*4096=245,760 bytes and 55*4096=225,280 bytes. No security effectiveness, throughput, reviewer behavior, user value, incident reduction, or deployment result is known. A future result must state population, numerator, denominator, window, comparator, uncertainty, and source artifact.

## 12. Failure Modes & What Went Wrong

| Failure | Competing causes | Discriminating signal | Bounded correction | Verification |
|---|---|---|---|---|
| Cross-tenant proposal | prompt identity; session routing; cache key | tenant-source lineage mismatch | identity-derived scope everywhere | negative tenant replay |
| Approval mismatch | mutable payload; canonicalization bug | digest differs before commit | immutable record/recompute | mutated destination denied |
| Duplicate export | timeout; new retry ID; alternate endpoint | two receipts for same intent | idempotency/effect journal | replay returns one receipt |
| Egress bypass | default route; proxy allow-list | connection to canary destination | default-deny egress | sandbox test denied |
| Queue overload | slow review; burst arrivals | arrival rate exceeds service | narrow scope/add capacity | measured queue test |

## 13. Transferable Lesson

The reusable mechanism is not “add human review.” It is bind every consequential effect to authenticated scope, a deterministic policy decision, an exact payload when review is needed, a quota, and a durable receipt. In another domain, replace records/export with payments, deployments, messages, or actuator commands; re-derive the asset, budget, approver, and recovery plan. The counterexample is a low-risk read-only query, where an approval queue adds cost without reducing an external effect.

## 14. Connections & Cross-References

Related theory: [Threat Model and Execution Trust](../Cheat_Sheets/01_AI_Threat_Model_and_Execution_Trust.md) and [Prompt Injection and Red-Team Evaluation](../Cheat_Sheets/02_Prompt_Injection_and_Red_Team_Evaluation.md). The existing [enterprise approval-gate case](../../../06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/Case_Studies/01_enterprise_agent_approval_gate.md) is a hypothetical, unreviewed reference, not evidence for this case.

1. **What proves the model cannot export?** Nothing; the executor must deny unauthorized effects. *Reasoning:* models propose. *Follow-up:* inspect alternate endpoints. *Alternative:* remove export capability. *Red flag:* system prompt. *Score:* 0–2 for proposal/boundary/trace.
2. **Compute the cap.** 240 KiB. *Reasoning:* q*c*b. *Follow-up:* c=5 gives 400 KiB. *Alternative:* bytes only. *Red flag:* count calls only. *Score:* arithmetic and interpretation.
3. **Why bind a digest?** It detects changed payload between review and commit. *Follow-up:* semantic-preview limit. *Alternative:* signed immutable request. *Red flag:* approve transcript. *Score:* integrity plus limitation.
4. **How test cross-tenant isolation?** Replay two authenticated tenants with identical prompt and assert disjoint retrieval/capability/receipt scope. *Follow-up:* cache keys. *Alternative:* isolated environment. *Red flag:* prompt tenant. *Score:* 0–2.
5. **What if review latency doubles?** Capacity halves; narrow automation or add capacity, do not silently waive approval. *Follow-up:* queue distribution. *Alternative:* manual baseline. *Red flag:* bypass. *Score:* tradeoff.
6. **Why is schema validity insufficient?** It lacks entitlement and purpose. *Follow-up:* authorize server-side fields. *Alternative:* typed contract plus policy. *Red flag:* valid JSON equals safe. *Score:* 0–2.
7. **How recover after uncertain timeout?** Query effect journal by stable operation ID; do not resend blindly. *Follow-up:* receipt loss. *Alternative:* reconciliation job. *Red flag:* retry with new ID. *Score:* 0–2.
8. **What does a sandbox not solve?** Authorized egress and credential misuse. *Follow-up:* canary egress. *Alternative:* default deny. *Red flag:* container equals isolation. *Score:* 0–2.
9. **Which result blocks release?** Any critical prohibited commit under the defined oracle. *Follow-up:* preserve inconclusives. *Alternative:* manual-only scope. *Red flag:* average score. *Score:* 0–2.
10. **Changed constraint: no human reviewer.** Restrict to deterministic low-consequence action or retain manual baseline. *Follow-up:* policy owner. *Alternative:* asynchronous review. *Red flag:* larger quota. *Score:* 0–2.
11. **What evidence is missing?** real policy, endpoint inventory, identity validation, workload, and test execution. *Follow-up:* owner per artifact. *Alternative:* no pilot. *Red flag:* infer from document. *Score:* 0–2.
12. **Why journal denials?** To detect attack pressure and prove policy path, while minimizing sensitive data. *Follow-up:* retention. *Alternative:* aggregate signals. *Red flag:* log all payloads. *Score:* 0–2.

## 15. Quick Reference Card

| Field | Case reference |
|---|---|
| Decision | allow bounded export or retain read-only/manual baseline |
| Evidence | illustrative design; arithmetic hand-checked only |
| Baseline flaw | schema-valid JSON lacks authorization |
| Core invariant | executor matches scope, budget, expiry, policy, and payload digest before commit |
| Numerical bound | 20 records/call × 3 calls × 4096 bytes = 240 KiB |
| Key oracle | changed payload yields denial and no committed receipt |
| Do not claim | incident reduction, security efficacy, or production outcome |
