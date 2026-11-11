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
source_basis: "illustrative defensive simulation"
as_of_date: "2026-11-11"
owner_role: "Security and data-platform owner"
source_index: "INDEX.md"
review_trigger: "Any change to export tool, identity, data class, approval policy, or executor."
---

# Approval-Bound Export for a Retrieval Assistant

This is an **ILLUSTRATIVE** defensive design and bounded simulation, not a report of a company, deployment, incident, model test, or measured outcome. Constructed identities, records, thresholds, and arithmetic are used solely to reason about prevention, detection, containment, privacy, and recovery. The manuscript intentionally excludes instructions for obtaining, transmitting, concealing, or scaling unauthorized data access.

## 1. The One-Line Framing

For a tenant-scoped research assistant, permit an export only when an independent executor can prove the authenticated principal, permitted purpose, selected records, approved destination class, expiry, byte budget, and exact payload digest still match at commit time.

The decision is not whether a model can produce valid JSON. The decision is whether a consequential release remains within a verifiable authorization envelope after untrusted retrieved text and model output have influenced a proposal.

A safe default is denial with a receipt when any required binding is absent, stale, ambiguous, or inconsistent. A denial is preferable to a silent broadening of tenant, recipient, record set, or budget.

## 2. Context & Background

The illustrative service helps an authenticated analyst summarize documents already visible to that analyst. A separate controlled export path may place a bounded selection into an approved case workspace. Retrieval content can be adversarial even when it is stored in a trusted-looking knowledge base.

The workflow is request, retrieval, proposed effect, policy evaluation, optional human approval, independent execution, receipt, and review. The model is a proposer; it is never an authorization authority or an effect executor.

A schema-only interface is inadequate because syntax does not establish who requested an effect, why the effect is permitted, which tenant owns the records, whether the destination remains eligible, or whether a retry repeats a previous commit.

The relevant assets are record content, tenant boundaries, destination eligibility, approval integrity, capability integrity, receipt integrity, and the availability of the review path. Confidentiality failure is important, but integrity failure can create the same disclosure by redirecting an otherwise permitted payload.

The illustrated organization has no asserted production topology, classification catalog, staffing level, legal interpretation, or incident history. Those unknowns are gates for a real deployment rather than facts supplied by this exercise.

## 3. The Business Question

| Decision element | Illustrative decision contract |
|---|---|
| Decision owner | Security and data-platform owner who can accept or reject the residual risk |
| Trigger | A pilot proposes assistant-supported transfer into an approved case workspace |
| Eligible action | Export a narrow, policy-eligible record set after server-side checks |
| Baseline | Read-only assistance followed by a manual transfer outside the assistant path |
| Rejected shortcut | Execute a schema-valid model proposal without independent authorization |
| Security objective | Prevent an unauthorized tenant, record, destination, purpose, or volume from committing |
| Privacy objective | Minimize retained payload content while preserving accountable receipts |
| Availability objective | Deny safely without creating an unbounded reviewer queue |
| Decision horizon | One request, one tenant, one capability, and one defined review window |
| Non-goal | Proving that the proposed design has reduced incidents in a real environment |

The business benefit is avoiding repetitive copying for eligible research work. The countervailing cost is review latency and engineering effort needed to make the effect path complete and observable.

The decision owner must be able to distinguish a secure denial from a failed user experience. A denied request can be correct when the evidence needed to bind the effect is unavailable.

## 4. Assumptions & Constraints

| Identifier | Assumption or constraint | Why it matters | Defensive fallback |
|---|---|---|---|
| A-01 | Tenant identity originates in authenticated middleware, not a model field | A forged tenant field would defeat every downstream filter | Disable export and retain read-only assistance |
| A-02 | The executor is the only component able to commit an export | A bypass path would escape quotas and receipts | Remove routing to the unverified path |
| A-03 | Record classification is available before policy evaluation | Unknown sensitivity cannot be safely routed | Deny classification-unknown records |
| A-04 | Destination class is resolved server side | A textual URL or label is not reliable authority | Permit only registered destination identifiers |
| A-05 | Canonicalization is deterministic for the approved payload | Digest comparison is meaningless if serialization changes | Require a stable server-side canonical form |
| A-06 | Operation identifiers are durable across retries | Duplicate commits increase disclosure volume | Return the prior receipt for a repeated identifier |
| A-07 | Clock source is trusted for expiry evaluation | An expired approval must not remain usable | Deny when trusted time is unavailable |
| A-08 | Policy version is retained with the decision | A later policy cannot silently reinterpret old approval | Reject decisions with an unavailable version |
| C-01 | Requests over 20 records require approval in this scenario | Human review is reserved for larger releases | Use a lower automatic threshold of zero if uncertain |
| C-02 | One approval authorizes at most three commits | Partitioning cannot silently expand the release | Re-approval is required after the third commit |
| C-03 | The payload-only ceiling is 240 KiB in this scenario | The design needs a finite blast-radius bound | Deny when byte measurement is unavailable |
| C-04 | Approval expiry is 15 minutes after issuance | Review intent should not persist indefinitely | Require a newly evaluated request |
| C-05 | Receipts retain metadata rather than full payload content | Audit should not create a second sensitive store | Store a digest, counts, and policy evidence only |

Assumptions A-01 and A-02 are architectural gates. A strong digest cannot compensate for a model-controlled tenant, and a perfect policy decision cannot constrain an undocumented alternate writer.

A real owner should replace each illustrative constraint with a documented policy source, an accountable approver, a testable enforcement location, and a recovery decision. Until then, the safe status is proposed rather than approved.

## 5. Data, Tools & Method

### 5a. Data Specification

The grain is one requested export operation identified by a server-generated operation identifier. The decision timestamp is the policy evaluation time; the commit timestamp is separately recorded because an approval can expire between those events.

The population is a constructed set of eligible and ineligible requests in a defensive simulator. No person, company dataset, credential, customer record, production log, or external destination is represented.

The primary key is `operation_id`. Supporting keys are authenticated `principal_id`, `tenant_id`, registered `destination_id`, immutable `approval_id` when required, and per-record opaque `record_id` values.

Each record has an illustrative classification, tenant, byte count, and eligibility state. Each request carries a purpose code, requested record identifiers, declared use window, and policy snapshot reference.

Units are records, bytes, KiB, minutes, requests, commits, denials, and receipts. One KiB is 1,024 bytes. The 240 KiB capacity in this manuscript is therefore 245,760 bytes, not 240,000 decimal bytes.

The model never receives a capability secret, a broad destination address, a raw authorization rule, a reviewer identity, or an executor credential. It can receive a constrained preview such as record count, class summary, and a policy-safe reason for denial.

Not available at decision time are future retries, future policy changes, reviewer intent beyond the approved payload, undiscovered endpoints, and content that is not included in the server-side record selection. Those absences are part of the decision boundary, not missing model context.

| Field | Source of truth | Valid form | Excluded from model authority |
|---|---|---|---|
| principal_id | authenticated session middleware | immutable internal identifier | free-text requester name |
| tenant_id | session claim verified by broker | one server-resolved tenant | tool argument tenant field |
| purpose_code | controlled request form | enumerated business purpose | narrative rationale invented by model |
| record_ids | server-side search result selection | opaque identifiers within tenant | arbitrary retrieval citations |
| classification | data catalog snapshot | policy-recognized label | document self-description |
| destination_id | registered workspace catalog | immutable approved identifier | pasted destination text |
| policy_version | policy decision service | retained version reference | model-selected rule version |
| approval_id | approval service | signed, expiring server record | reviewer name in prompt text |
| payload_digest | executor canonicalization | fixed algorithm output | model-computed checksum |
| byte_budget | capability claim | nonnegative integer bytes | inferred token count |
| receipt_id | effect journal | append-only server identifier | client-generated confirmation |

### 5b. Tools & Stack

The proposed stack contains an identity middleware, request broker, retrieval service, typed proposal validator, policy decision point, preview builder, approval service, narrow executor, effect journal, and controlled simulation sink.

Identity middleware supplies principal and tenant attributes. The request broker makes those attributes immutable for the request lifecycle and rejects attempts to supply them through untrusted content.

The retrieval service labels source provenance so that instructions embedded in documents are treated as data, not as authority. The typed proposal validator accepts a small effect vocabulary and rejects opaque command strings.

The policy decision point evaluates classification, purpose, destination class, record count, bytes, tenancy, and approval requirement. The preview builder produces the exact server-side selection that a reviewer can assess without exposing a broad retrieval transcript.

The approval service binds a reviewer decision to a digest, principal, tenant, purpose, destination class, budget, policy version, and expiry. The narrow executor recomputes every material value rather than trusting the proposal or approval display.

The effect journal records a minimum necessary receipt. The controlled simulation sink returns harmless acknowledgments and deliberately has no channel to external systems, thereby keeping the exercise bounded and defensive.

### 5c. Method Summary

The method is: authenticate request, resolve tenant and destination server side, select eligible records, create a typed proposal, evaluate policy, canonicalize a preview, obtain approval when required, recompute constraints in the executor, commit only to the controlled sink, and journal a receipt.

The central invariant is that every committed effect has evidence for actor, tenant, purpose, destination, record set, byte budget, policy version, expiry, and idempotency. A failure in any evidence item produces a denial receipt rather than a best-effort commit.

The simulation changes one defensive variable at a time. Examples include altered destination binding, stale approval, cross-tenant selection, byte counter disagreement, repeated operation identifier, unavailable policy version, and missing receipt persistence.

The comparison is conceptual: a schema-only baseline versus the proposed constrained path. No empirical effectiveness rate is claimed because the simulator described here has not been executed.

### 5d. Provenance, Quality, Exclusions, Leakage

All values, traces, rates, names, and thresholds in this case are ILLUSTRATIVE. The sources below provide vocabulary and publicly available guidance; they are not evidence that the proposed control has been deployed or tested.

Retrieved content is untrusted input regardless of where it was indexed. A document may contain imperative language, misleading metadata, or a false claim of approval, none of which may alter a server-side authorization decision.

Quality checks focus on referential integrity: record tenant must equal session tenant, every selected record must remain eligible, the byte total must match canonical content, and approval evidence must reference the same canonical selection.

The case excludes endpoint-specific network configuration, secrets management implementation, legal retention schedules, workforce procedures, actual identity providers, model tuning, vendor guarantees, and production incident response details. Those omissions prevent this illustrative work from being mistaken for operational instructions.

Leakage reduction means retaining a digest and bounded metadata in the journal rather than copying content into logs. Diagnostic views should redact identifiers according to a documented access policy and should never become an alternative retrieval channel.

### 5e. Reproducibility & Method-Selection Ledger

| Artifact | Role in reasoning | Reproducible by | Limitation |
|---|---|---|---|
| ILL-INPUT-01 | constructed request with 55 records | hand arithmetic from stated inputs | not a real request |
| ILL-INPUT-02 | constructed stale-approval trace | deterministic state comparison | not an executed service |
| ILL-INPUT-03 | constructed tenant-mismatch trace | equality check on opaque labels | labels carry no real data |
| ILL-ORACLE-01 | commit allowed only when all bindings match | review of stated invariant | not a formal proof |
| ILL-ORACLE-02 | repeated operation returns first receipt | idempotency state reasoning | storage behavior unmeasured |
| SRC-01 | NIST AI RMF risk vocabulary | public publication | not case-specific evidence |
| SRC-02 | OWASP prompt injection guidance | public publication | not a measured control result |
| SRC-03 | NIST privacy framework vocabulary | public publication | not a retention policy |

The proposed method was selected over schema-only execution because schema validity answers structural questions while the decision requires identity, scope, purpose, policy, budget, and durable-effect questions. It was selected over permanent broad credentials because a finite capability gives the owner a revocation and containment surface.

The method should be rejected for a real pilot if endpoint inventory, tenant provenance, canonicalization, policy version retention, or receipt durability cannot be independently demonstrated. That is a go/no-go criterion, not a request for compensating model prompts.
## 6. Analysis Walkthrough

The walkthrough is a defensive proof sketch for a proposed invariant. It asks whether a constructed request remains bounded when the model or retrieved material proposes a different record set, destination, volume, or timing than the authorized preview.

```text
FORMULA [CASE-49-001-01]: B_max = q × c × b
WHERE: q is records per commit [records/commit]; c is commits per approval [commits/approval]; b is maximum bytes per record [bytes/record]; B_max is maximum payload bytes per approval [bytes/approval].
CONSTRAINT: The executor centrally counts successful commits, measures canonical payload bytes, and denies a fourth commit; all factors are nonnegative integers.
NUMERICAL EXAMPLE: ILLUSTRATIVE q=20 records/commit; c=3 commits/approval; b=4,096 bytes/record. First q×c=20×3=60 records/approval. Then B_max=60×4,096=245,760 bytes=240 KiB.
INTERPRETATION: Under these assumptions, one approval cannot authorize more than 60 records or 240 KiB of payload. The result is invalid if another writer bypasses the central counter.
```

The boundary case is a 61st eligible record. Even if each record is below 4,096 bytes, the maximum count rule denies it because count and byte constraints are independent. A lower byte total does not create permission for a fourth partition.

```text
FORMULA [CASE-49-001-02]: D_match = 1 if H(C(request)) = H(C(approval)); otherwise D_match = 0
WHERE: C is deterministic server-side canonicalization; H is a fixed collision-resistant digest function; D_match is a Boolean eligibility condition with no unit.
CONSTRAINT: The same fields, ordering, encoding, destination class, tenant, purpose, and policy version must be included in both canonical forms.
NUMERICAL EXAMPLE: ILLUSTRATIVE canonical request contains tenant=T-17, destination=W-A, and 55 ordered record identifiers, yielding digest d_req. Approval stores d_appr=d_req. Changing destination to W-B changes canonical input, so d_req≠d_appr and D_match=0.
INTERPRETATION: Review covers the exact committed selection only when canonicalization includes every effect-relevant field. A display-only digest is insufficient.
```

The counterexample is a digest that omits destination class. In that defective design, a payload could retain the same digest while being routed to a different class of workspace. The formula then reports a match while the security property is false, so destination must be part of `C`.

```text
FORMULA [CASE-49-001-03]: R_rate = n_review / (n_review + n_auto)
WHERE: n_review is requests requiring human approval [requests/window]; n_auto is requests permitted without review [requests/window]; R_rate is a unitless fraction in [0,1].
CONSTRAINT: The numerator and denominator cover the same eligible population and evaluation window; denied requests are tracked separately rather than silently excluded.
NUMERICAL EXAMPLE: ILLUSTRATIVE over one eight-hour window, n_review=18 and n_auto=42. Denominator=18+42=60 requests. R_rate=18/60=0.30, or 30% of eligible requests requiring review.
INTERPRETATION: The number describes planned reviewer demand, not an observed queue outcome. It cannot predict staffing without service-time and arrival-rate evidence.
```

The failure boundary is a window with 18 review requests but zero reviewers available. The fraction still equals 30%, yet approval latency is undefined; an availability control must deny or hold safely rather than infer a review decision.

```text
FORMULA [CASE-49-001-04]: E_exposure = min(B_cap, B_selected) × I_commit
WHERE: B_cap is authorized byte cap [bytes]; B_selected is canonical selected payload size [bytes]; I_commit is 1 for a committed effect and 0 for a denial; E_exposure is an upper-bound proxy [bytes].
CONSTRAINT: This is a containment estimate only; it assumes the sink receives exactly the canonical payload and no duplicate writer exists.
NUMERICAL EXAMPLE: ILLUSTRATIVE B_cap=245,760 bytes, B_selected=225,280 bytes, and I_commit=1. min(245,760,225,280)=225,280 bytes, so E_exposure=225,280 bytes. If the executor denies, I_commit=0 and E_exposure=0 bytes in this model.
INTERPRETATION: A cap bounds a single authorized effect but does not establish that a recipient is trustworthy or that all copied data can be recovered.
```

A cap is not a guarantee of zero harm. The same byte total can represent different sensitivity classes, and a misclassified single record may have disproportionate impact. Policy must therefore combine class restrictions with quantitative ceilings.

### Defensive simulation catalogue

Each scenario below is a bounded oracle specification, not a procedure for obtaining data. The desired result is either a safe denial with a diagnostic receipt or a permitted commit to the controlled sink under all stated bindings.

| Simulation condition | Expected independent check | Safe expected state | Evidence retained |
|---|---|---|---|
| Retrieved text asks for a different tenant | Compare record tenant to session tenant | Denied before preview | tenant mismatch code and operation identifier |
| Proposal names an unregistered workspace | Resolve destination only from catalog | Denied before approval | unknown destination class code |
| Approval is older than 15 minutes | Evaluate trusted expiry clock | Denied before commit | expiry timestamp and policy version |
| Reviewer approved 55 records but proposal adds one | Recompute ordered selection digest | Denied before commit | digest mismatch indicator |
| Selected records total 245,761 bytes | Measure canonical bytes in executor | Denied before commit | measured byte count and cap |
| Fourth partition uses a valid approval | Read centralized successful-commit counter | Denied before commit | prior receipt identifiers |
| Retry repeats a committed operation identifier | Lookup durable idempotency record | Return first receipt; no second commit | original receipt reference |
| Retry follows an earlier denial | Lookup denial state and re-evaluate only through a new request | No implicit elevation | original denial reason |
| Policy version cannot be loaded | Require exact retained policy reference | Denied before commit | missing policy version code |
| Classification changes after preview | Re-read current classification snapshot | Denied or require fresh approval | old and current classification references |
| Capability is revoked after approval | Check revocation state at commit | Denied before commit | revocation epoch |
| Journal append fails after sink acknowledgement | Treat effect state as uncertain and contain | Escalate reconciliation, no silent success | reconciliation marker |
| Clock source is unavailable | Refuse expiry comparison | Denied before commit | trusted-time unavailable code |
| Model proposal has malformed identifiers | Validate typed opaque identifier format | Denied before policy | validation error class |
| Identity middleware returns no tenant | Require authenticated tenant binding | Denied before retrieval selection | missing tenant evidence |
| Same user requests a different purpose | Compare purpose to approved canonical form | Denied before commit | purpose mismatch code |
| Destination catalog marks workspace suspended | Resolve current destination state | Denied before commit | suspended destination state |
| Record catalog has an unknown label | Apply classification-unknown rule | Denied before preview | unknown classification count |
| Receipt lookup is unavailable during retry | Avoid assuming no prior effect | Hold for reconciliation | idempotency lookup outage marker |
| A bypass endpoint is discovered in inventory | Compare writer inventory to enforced path | Disable export feature pending closure | endpoint coverage exception |

A scenario is passed only if the controlled sink observes no effect for a denial state and the journal contains enough metadata to explain the decision without logging payload content. A response that merely changes model phrasing does not count as a control.

The simulator should label each test input as constructed, run only against the controlled sink, and prohibit live credentials, external transmission, and realistic sensitive payloads. This maintains useful defensive pressure without turning evaluation into an exfiltration exercise.

## 7. The Finding

The proposed finding is conditional: a bounded export can be considered for a future pilot only when the executor independently enforces authenticated tenant scope, purpose, destination class, exact selection binding, byte and commit limits, expiry, revocation, and durable idempotency.

The schema-only baseline is analytically rejected because well-formed fields do not provide independent evidence of authority. A model-generated `tenant_id`, recipient label, or approval reference is an untrusted claim until server-side sources verify it.

The ILLUSTRATIVE 240 KiB bound follows from 20 records per commit, three commits per approval, and 4,096 bytes per record. It is not a recommended production threshold and is not evidence of a tolerable disclosure amount.

No outcome has been measured. The finding changes to read-only/manual handling if a complete writer inventory, trustworthy tenant provenance, deterministic canonicalization, current classification, or receipt durability cannot be shown.

## 8. System/Solution Design

```text
Authenticated session -> request broker -> retrieval with provenance labels
                                      -> typed proposal validator
Policy decision point -> canonical preview -> approval service when required
                                      -> narrow executor -> controlled sink
                                      -> minimal effect journal -> receipt and review queue
```

The request broker freezes authenticated principal and tenant attributes for the operation. It treats a tenant string in a retrieved document, model output, or client-provided tool parameter as untrusted narrative rather than identity evidence.

The typed proposal expresses only `record_ids`, `purpose_code`, and an abstract destination request. It does not carry credentials, raw network targets, executable text, or a free-form instruction channel for downstream effect systems.

The policy decision point resolves record ownership, current classification, purpose eligibility, destination class, approval threshold, and required retention handling. Its result names the exact policy version and denial reason category without revealing a policy bypass path.

The canonical preview sorts opaque record identifiers under a documented deterministic rule and includes tenant, principal, purpose, destination class, classification summary, byte budget, policy version, and expiry-relevant fields. The approval is bound to that server-produced preview.

The approval service captures reviewer identity, decision time, expiry, and the digest of the preview. It cannot approve a broader selection by editing text; any material change creates a new preview and requires a new decision.

The narrow executor recomputes tenant ownership, eligibility, classification, byte count, policy version, expiry, revocation status, commit count, and digest immediately before the controlled effect. It never delegates these checks back to the model.

The effect journal stores operation identifier, decision state, receipt identifier, policy reference, counts, byte total, digest, timestamps, and redacted diagnostic class. It deliberately does not store payload text, raw retrieved passages, or destination secrets.

### Control-to-threat binding

| Threat hypothesis | Preventive control | Detective signal | Containment decision | Recovery evidence |
|---|---|---|---|---|
| Untrusted document changes tenant | authenticated tenant binding | tenant mismatch denials | disable affected request path | session and policy references |
| Untrusted document changes destination | registered destination resolution | destination mismatch denials | revoke pending capability | destination catalog version |
| Model expands record selection | digest-bound preview | digest mismatch count | deny and require fresh request | preview and approval identifiers |
| Retry duplicates a commit | durable idempotency state | repeated operation lookup | return original receipt | receipt lineage |
| Approval is replayed after expiry | trusted expiry evaluation | stale approval denials | revoke approval family if anomalous | expiry and trusted-time evidence |
| Policy changes after review | exact policy version binding | policy version disagreement | deny until re-evaluated | old and current policy references |
| Record classification becomes stricter | commit-time catalog recheck | classification drift event | hold and re-approve if eligible | classification snapshot references |
| Broad writer is exposed | endpoint inventory control | inventory coverage gap | turn off export routing | endpoint attestation record |
| Journal durability is uncertain | commit-state reconciliation | ambiguous effect marker | suspend related capability | sink acknowledgement correlation |
| Reviewer account is unavailable | separation of duties and queue routing | unavailable-reviewer signal | retain safe hold or manual path | workflow status reference |
| Destination is later suspended | current destination state check | suspension event | revoke uncommitted capabilities | catalog state transition |
| Byte meter disagrees with preview | executor measurement authority | byte discrepancy event | deny without approximation | measured byte total |

### Privacy and security control inventory

| Control area | Defensive rule | Privacy consequence | Security consequence | Owner role |
|---|---|---|---|---|
| Data minimization | Journal metadata rather than payload content | reduces secondary sensitive store | limits diagnostic replay detail | privacy steward |
| Purpose limitation | Bind purpose code into approval digest | blocks reuse for unrelated work | prevents purpose swapping | policy owner |
| Tenant isolation | Resolve tenant from session only | avoids cross-tenant disclosure | rejects forged tenant claims | identity owner |
| Destination governance | Use registered destination classes | restricts recipient context | blocks arbitrary routing | workspace owner |
| Least privilege | Give model no export credential | reduces credential exposure | eliminates direct model execution | platform owner |
| Time limitation | Expire approvals and capabilities | narrows long-lived access | limits replay window | security owner |
| Integrity | Recompute canonical digest at commit | preserves review meaning | blocks selection mutation | executor owner |
| Audit minimization | Redact receipt diagnostics | avoids logging sensitive text | retains accountable decision trail | audit owner |
| Revocation | Check current revocation epoch | stops pending use after concern | contains suspected compromise | security owner |
| Separation of duties | Reviewer differs from executor service | adds independent judgment | reduces single-control failure | workflow owner |
| Availability safety | Deny or hold when evidence service fails | avoids speculative release | prevents fail-open behavior | operations owner |
| Retention governance | Set receipt retention by policy | avoids indefinite metadata storage | preserves needed investigation period | records owner |

The design does not claim that any individual control is sufficient. The security property arises from independent sources of truth that meet again at the executor, coupled with an ability to revoke, contain, and explain an uncertain effect.
## 9. Evaluation & Validation

Evaluation is proposed and not executed. Its purpose is to falsify the stated defensive invariant in a controlled simulator, not to measure model creativity, demonstrate a live exploit, or generate transferable operational instructions.

| Metric | Numerator | Denominator | Population and window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|---|
| Prohibited commit rate | prohibited commits | prohibited constructed requests | all negative simulator cases in one declared run | schema-only conceptual baseline | target is zero committed effects | proposed, not executed |
| Eligible commit preservation | permitted commits | eligible constructed requests | all positive simulator cases in one declared run | manual review design | inspect false denials | proposed, not executed |
| Digest-binding detection | mutated proposals denied | mutated proposals presented | digest-mutation cases in one declared run | preview without binding | no accepted mutation | proposed, not executed |
| Tenant-boundary detection | tenant-mismatch requests denied | tenant-mismatch requests presented | constructed cross-label cases | proposal-supplied tenant baseline | no sink effect | proposed, not executed |
| Byte-cap enforcement | over-cap requests denied | over-cap requests presented | constructed size-boundary cases | preview-only measurement | no approximate acceptance | proposed, not executed |
| Idempotency preservation | retries returning original receipt | retries of committed identifiers | constructed retry cases | stateless retry handling | no duplicate sink acknowledgment | proposed, not executed |
| Expiry enforcement | stale approvals denied | stale approvals presented | constructed clock-boundary cases | no expiry check | trusted time required | proposed, not executed |
| Policy-version enforcement | obsolete-policy attempts denied | obsolete-policy attempts presented | constructed version-drift cases | latest-policy substitution | exact version evidence | proposed, not executed |
| Classification-drift handling | changed-class requests held or denied | changed-class requests presented | constructed catalog-drift cases | preview-only classification | no stale-class commit | proposed, not executed |
| Endpoint coverage | known writers mediated | inventoried writers | architecture inventory snapshot | partial inventory | any gap is release blocker | proposed, not executed |
| Receipt completeness | terminal states with minimum receipt | terminal decisions | all simulator operations | ad hoc logging | no payload text retained | proposed, not executed |
| Review-load estimate | requests needing review | eligible requests | ILLUSTRATIVE eight-hour planning window | all-manual process | not a staffing conclusion | estimated only |

For each metric, a result must state the test manifest version, the number of constructed cases, the simulator version, the controlled-sink state, and whether the test actually ran. A blank result is unknown, not a zero.

Uncertainty is structural rather than statistical because no execution is asserted. Potential false assurance includes incomplete threat coverage, inaccurate mock semantics, unknown production dependencies, and a sink that fails to model an alternate writer.

A release gate should require independent review of identity provenance, writer inventory, policy source, canonicalization specification, idempotency storage, receipt retention, and recovery owner. Passing unit-style cases alone is insufficient when the deployment boundary is unknown.

### Validation matrix for defensive claims

| Claim | Evidence required | Invalidating observation | Conservative decision |
|---|---|---|---|
| Tenant cannot be selected by retrieved text | immutable session-to-executor trace | any untrusted field changes effective tenant | retain read-only mode |
| Approval covers the committed selection | canonical form and digest comparison | omitted effect-relevant field | require redesign |
| Byte ceiling is enforceable | commit-time byte oracle | alternate writer or approximate meter | deny export |
| Retries do not duplicate effects | durable receipt lookup trace | second sink acknowledgment for one identifier | suspend feature |
| Revocation reaches pending work | commit-time revocation trace | revoked capability still commits | disable capability issue |
| Policy remains interpretable | retained version artifact | unavailable historical policy | deny stale approval |
| Classification remains current | commit-time catalog reference | changed classification ignored | require fresh review |
| Receipt is privacy-minimized | receipt schema inspection | payload content in normal receipt | remediate logging first |
| Destination remains approved | current catalog lookup | suspended destination accepts work | block routing |
| Review has separation of duties | role-binding evidence | same actor controls decision and effect | redesign workflow |
| Failures do not fail open | dependency outage cases | unknown evidence permits effect | change default to denial |
| Recovery can locate an effect | receipt-to-sink correlation | ambiguous commit state unowned | block pilot |

## 10. Action Taken

No production action was taken and no implementation is claimed. The only action in this manuscript is the definition of a proposed bounded defensive simulation and its review gates.

Before a pilot is considered, the responsible owners would need to choose whether the baseline remains read-only, whether a controlled simulation is authorized, and whether the evidence prerequisites are complete. Those decisions cannot be inferred from a model output or from this illustrative case.

The recommended sequence is governance-oriented rather than operational: verify the complete effect inventory, document authoritative identity and classification sources, approve the policy semantics, review the canonicalization specification, define receipt retention, assign a recovery owner, and assess the simulation in a controlled sink.

A stop condition is any unknown writer, model-held effect credential, unbound destination, non-deterministic preview, fail-open dependency, or inability to reconcile an ambiguous commit. The stop condition is intentionally stronger than a desire to preserve convenience.

| Decision point | Accountable role | Required evidence | Safe disposition if absent |
|---|---|---|---|
| Permit a simulator | security owner | bounded sink and synthetic manifest | do not run the simulation |
| Permit a pilot | product and security owners | all architectural gates reviewed | retain read-only baseline |
| Set classification policy | data steward | current catalog semantics | deny unknown classes |
| Set approval threshold | policy owner | documented risk rationale | require review for all exports |
| Approve destination class | workspace owner | registered destination controls | deny unregistered destination |
| Accept receipt retention | privacy and records owners | minimization and retention rationale | retain no new payload logs |
| Own recovery | incident owner | reconciled receipt-to-sink mapping | block pilot |
| Accept residual risk | named decision owner | explicit exception record | do not infer acceptance |

## 11. Outcome & Results

**Not measured.** No company result, security efficacy rate, reviewer behavior, throughput value, user value, incident reduction, deployment result, or simulated test run is represented.

The checked arithmetic is ILLUSTRATIVE: 20 records/commit multiplied by three commits/approval gives 60 records/approval; 60 multiplied by 4,096 bytes/record gives 245,760 bytes, which equals 240 KiB; 55 multiplied by 4,096 bytes/record gives 225,280 bytes, which is below that cap by 20,480 bytes.

The last subtraction is 245,760 minus 225,280 equals 20,480 bytes, or 20 KiB. This slack does not authorize another 4,096-byte record if the 60-record count limit has already been reached, and it does not prove that the selected records are low sensitivity.

A future outcome report must identify its population, numerator, denominator, observation window, comparator, uncertainty, exclusions, source artifact, and whether the result came from simulation, staging, or production. Without those fields, the appropriate label is unknown.

| Reportable future result | Minimum evidence | Prohibited shortcut |
|---|---|---|
| Denial accuracy | reviewed manifest and adjudication rationale | calling every denial correct by default |
| Commit preservation | eligible-case definition and terminal receipts | excluding difficult eligible requests |
| Review burden | arrival window and service-time evidence | extrapolating from one reviewer anecdote |
| Privacy impact | data-flow and retention assessment | treating a digest as automatically anonymous |
| Security impact | threat coverage and endpoint evidence | inferring safety from prompt wording |
| Recovery quality | reconciled ambiguous-effect exercises | claiming recovery from a receipt alone |
| Availability | dependency outage observations | treating a denial as an outage-free experience |
| User value | defined task outcome and comparator | equating model fluency with value |

## 12. Failure Modes & What Went Wrong

The following are distinct failure mechanisms rather than generic advice to monitor a model. Each check is chosen to distinguish competing causes and to guide a bounded correction without providing an attack recipe.

| Failure mechanism | Competing causes | Discriminating check | Bounded correction | Verification |
|---|---|---|---|---|
| Tenant mismatch reaches proposal | identity mapping defect or untrusted field influence | compare immutable session tenant with effective executor tenant | block export and repair authority path | negative constructed tenant case denies |
| Digest mismatch is missed | omitted canonical field or stale preview cache | inspect canonical field inventory and recompute server side | include omitted field and require fresh approval | destination mutation denies |
| Byte cap is exceeded | approximate meter or alternate writer | compare executor count with sink acknowledgment size | deny on disagreement and close bypass | one-byte-over case denies |
| Duplicate effect follows retry | idempotency record is non-durable or key changes | correlate operation identifier across receipt and sink | persist stable operation state | retry returns original receipt |
| Expired approval commits | clock drift or expiry omitted from executor | compare trusted time and approval issue time at commit | fail closed on time uncertainty | post-expiry case denies |
| Revoked capability commits | revocation cache staleness or missing check | inspect commit-time revocation epoch | require fresh revocation evaluation | revoked-case denial is recorded |
| Classification drift is ignored | catalog snapshot frozen too long or update event lost | compare preview and commit classification references | hold for reevaluation | changed-class case does not commit |
| Suspended destination accepts payload | catalog state not checked or stale resolver | compare current destination state at executor | block suspended destinations | suspension case denies |
| Journal lacks terminal state | write ordering ambiguity or storage outage | reconcile sink acknowledgement with receipt state | mark uncertain and invoke recovery owner | ambiguous state is not reported as success |
| Payload appears in normal logs | diagnostic overcollection or error serializer | schema scan of receipt and error paths | redact and purge per policy | sampled receipts contain metadata only |
| Reviewer approval is too broad | preview hides material fields or role workflow weak | compare reviewer view to canonical form | reduce preview ambiguity and split roles | reviewer scenario catches mutation |
| Model holds a writer credential | architecture shortcut or secret propagation defect | credential inventory and runtime boundary review | remove credential and rotate affected secret | model path cannot call writer |
| Policy version cannot be reconstructed | retention gap or mutable rule reference | load referenced version from evidence store | deny stale approvals and retain versions | historical decision is interpretable |
| Unknown endpoint can write | inventory incompleteness or new integration | compare code and deployment inventory to mediation list | disable export until mediated | all known writers have executor evidence |
| Fail-open occurs on dependency outage | availability preference overrides policy | outage simulation observes permitted effect | change default to deny or safe hold | outage produces no sink effect |
| Receipt metadata identifies unnecessarily | excessive identifiers or weak access control | privacy review of receipt schema and readers | minimize fields and restrict access | receipt access review passes |
| Manual fallback bypasses policy | undocumented process or emergency pressure | trace fallback through same decision record | make fallback governed and logged | fallback has accountable receipt |
| Approval queue becomes unavailable | staffing shortfall or workflow service outage | distinguish no reviewer from service fault | safe hold with escalation window | no implicit auto-approval occurs |
| Sink acknowledgement is forged or ambiguous | weak correlation or simulator defect | verify authenticated sink correlation token | quarantine state pending reconciliation | no final success before correlation |
| Canonical ordering changes between services | library difference or unspecified encoding | cross-implementation canonical test vectors | publish deterministic canonical specification | identical input yields identical digest |

A broad pattern connects these failures: a control becomes brittle when it trusts a value merely because it appears near an approval. The discriminating checks therefore emphasize source, time, completeness, and reconciliation rather than intent claims.

### Detection and containment playbook boundaries

Detection begins with receipt anomalies: unexpected denial clusters, mismatched tenant evidence, repeated operation identifiers, stale approvals, byte disagreements, missing policy versions, revocation conflicts, or uncertain journal state. A diagnostic signal is not itself proof of disclosure.

Containment means disable the relevant export routing, revoke pending capabilities where supported, preserve minimal decision evidence, prevent automatic retry, and route uncertain states to the assigned recovery owner. It does not mean inspecting unrelated payloads or widening access to diagnose the event.

Triage should distinguish a confirmed committed effect, a denied request, a duplicate request returning an existing receipt, and an ambiguous state where sink and journal disagree. Ambiguous state requires reconciliation because treating it as either success or no-op can create harmful follow-on actions.

Recovery requires a bounded record of operation identifier, receipt state, policy version, destination class, digest, count, byte total, timestamps, and accountable owner. Payload reconstruction is not presumed necessary or appropriate for normal recovery.

### Defensive boundary analyses

The following boundary analyses make the containment design inspectable. Each identifies the authority source, the non-authoritative lookalike, and the conservative state transition; none describes a route for moving data.

#### Session Tenant Binding

**Condition:** The boundary is crossed when the session has no verified tenant.

**Authoritative evidence:** The executor relies on identity middleware assertion and request-broker trace.

**Defensive decision:** It must deny before record selection.

**Non-authority:** The system must not treat a second tenant claim in a proposal as equivalent evidence.

**Recovery note:** The minimal receipt should name the session tenant binding decision category without recording protected payload content.

**Detection signal:** A rising count of `session_tenant_binding` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of session tenant binding must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of session tenant binding uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a session tenant binding review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Principal Status

**Condition:** The boundary is crossed when the authenticated principal is suspended after preview.

**Authoritative evidence:** The executor relies on current account-state lookup.

**Defensive decision:** It must deny at commit and retain the suspension reason class.

**Non-authority:** The system must not treat a display name that still appears valid as equivalent evidence.

**Recovery note:** The minimal receipt should name the principal status decision category without recording protected payload content.

**Detection signal:** A rising count of `principal_status` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of principal status must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of principal status uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a principal status review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Purpose Eligibility

**Condition:** The boundary is crossed when the stated business purpose is no longer eligible for the selected class.

**Authoritative evidence:** The executor relies on current purpose-to-class policy decision.

**Defensive decision:** It must deny and require a newly evaluated purpose.

**Non-authority:** The system must not treat a convincing natural-language explanation as equivalent evidence.

**Recovery note:** The minimal receipt should name the purpose eligibility decision category without recording protected payload content.

**Detection signal:** A rising count of `purpose_eligibility` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of purpose eligibility must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of purpose eligibility uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a purpose eligibility review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Record Ownership

**Condition:** The boundary is crossed when one selected opaque identifier belongs to another tenant.

**Authoritative evidence:** The executor relies on per-record authoritative ownership comparison.

**Defensive decision:** It must deny the entire constructed request rather than silently substituting data.

**Non-authority:** The system must not treat a search ranking that groups records together as equivalent evidence.

**Recovery note:** The minimal receipt should name the record ownership decision category without recording protected payload content.

**Detection signal:** A rising count of `record_ownership` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of record ownership must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of record ownership uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a record ownership review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Record Deletion

**Condition:** The boundary is crossed when a selected record was deleted after the preview.

**Authoritative evidence:** The executor relies on commit-time existence lookup.

**Defensive decision:** It must deny because the approved object no longer exists.

**Non-authority:** The system must not treat a cached retrieval excerpt as equivalent evidence.

**Recovery note:** The minimal receipt should name the record deletion decision category without recording protected payload content.

**Detection signal:** A rising count of `record_deletion` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of record deletion must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of record deletion uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a record deletion review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Legal Hold Indicator

**Condition:** The boundary is crossed when a current hold marks a record non-exportable.

**Authoritative evidence:** The executor relies on catalog hold state at commit.

**Defensive decision:** It must deny and route only to the governed case process.

**Non-authority:** The system must not treat an old classification snapshot as equivalent evidence.

**Recovery note:** The minimal receipt should name the legal hold indicator decision category without recording protected payload content.

**Detection signal:** A rising count of `legal_hold_indicator` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of legal hold indicator must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of legal hold indicator uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a legal hold indicator review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Classification Escalation

**Condition:** The boundary is crossed when classification becomes more restrictive between decision and commit.

**Authoritative evidence:** The executor relies on versioned catalog comparison.

**Defensive decision:** It must hold for fresh policy and approval.

**Non-authority:** The system must not treat the previous reviewer decision as equivalent evidence.

**Recovery note:** The minimal receipt should name the classification escalation decision category without recording protected payload content.

**Detection signal:** A rising count of `classification_escalation` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of classification escalation must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of classification escalation uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a classification escalation review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Classification Uncertainty

**Condition:** The boundary is crossed when the catalog cannot resolve a record class.

**Authoritative evidence:** The executor relies on explicit unknown-class status.

**Defensive decision:** It must deny instead of assigning a convenience default.

**Non-authority:** The system must not treat a model summary of the record as equivalent evidence.

**Recovery note:** The minimal receipt should name the classification uncertainty decision category without recording protected payload content.

**Detection signal:** A rising count of `classification_uncertainty` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of classification uncertainty must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of classification uncertainty uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a classification uncertainty review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Destination Registration

**Condition:** The boundary is crossed when the requested workspace identifier is absent from the catalog.

**Authoritative evidence:** The executor relies on server-side destination lookup.

**Defensive decision:** It must deny before a preview is shown.

**Non-authority:** The system must not treat a pasted workspace label as equivalent evidence.

**Recovery note:** The minimal receipt should name the destination registration decision category without recording protected payload content.

**Detection signal:** A rising count of `destination_registration` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of destination registration must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of destination registration uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a destination registration review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Destination Suspension

**Condition:** The boundary is crossed when a previously registered workspace is suspended.

**Authoritative evidence:** The executor relies on current destination lifecycle state.

**Defensive decision:** It must deny and notify the governing owner through the defined workflow.

**Non-authority:** The system must not treat the fact that the approval is recent as equivalent evidence.

**Recovery note:** The minimal receipt should name the destination suspension decision category without recording protected payload content.

**Detection signal:** A rising count of `destination_suspension` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of destination suspension must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of destination suspension uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a destination suspension review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Destination Tenant Binding

**Condition:** The boundary is crossed when the workspace tenant differs from the request tenant.

**Authoritative evidence:** The executor relies on catalog tenant comparison.

**Defensive decision:** It must deny as a recipient-scope mismatch.

**Non-authority:** The system must not treat a similarly named workspace as equivalent evidence.

**Recovery note:** The minimal receipt should name the destination tenant binding decision category without recording protected payload content.

**Detection signal:** A rising count of `destination_tenant_binding` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of destination tenant binding must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of destination tenant binding uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a destination tenant binding review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Destination Retention Posture

**Condition:** The boundary is crossed when the workspace lacks the retention posture required for the record class.

**Authoritative evidence:** The executor relies on destination-class policy attributes.

**Defensive decision:** It must deny until an eligible destination is selected.

**Non-authority:** The system must not treat the workspace owner’s informal assurance as equivalent evidence.

**Recovery note:** The minimal receipt should name the destination retention posture decision category without recording protected payload content.

**Detection signal:** A rising count of `destination_retention_posture` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of destination retention posture must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of destination retention posture uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a destination retention posture review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Canonical Field Completeness

**Condition:** The boundary is crossed when a field affecting recipient context is absent from canonicalization.

**Authoritative evidence:** The executor relies on published field inventory compared with executor inputs.

**Defensive decision:** It must block release of the binding design.

**Non-authority:** The system must not treat a digest that matches an incomplete form as equivalent evidence.

**Recovery note:** The minimal receipt should name the canonical field completeness decision category without recording protected payload content.

**Detection signal:** A rising count of `canonical_field_completeness` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of canonical field completeness must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of canonical field completeness uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a canonical field completeness review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Canonical Encoding Stability

**Condition:** The boundary is crossed when two services encode the same identifier differently.

**Authoritative evidence:** The executor relies on cross-service fixed test vectors.

**Defensive decision:** It must deny on digest disagreement and reconcile specifications.

**Non-authority:** The system must not treat a string comparison in one component as equivalent evidence.

**Recovery note:** The minimal receipt should name the canonical encoding stability decision category without recording protected payload content.

**Detection signal:** A rising count of `canonical_encoding_stability` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of canonical encoding stability must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of canonical encoding stability uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a canonical encoding stability review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Digest Algorithm Availability

**Condition:** The boundary is crossed when the required digest implementation is unavailable.

**Authoritative evidence:** The executor relies on executor dependency health and algorithm identifier.

**Defensive decision:** It must safe hold because exact binding cannot be verified.

**Non-authority:** The system must not treat a cached digest from another request as equivalent evidence.

**Recovery note:** The minimal receipt should name the digest algorithm availability decision category without recording protected payload content.

**Detection signal:** A rising count of `digest_algorithm_availability` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of digest algorithm availability must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of digest algorithm availability uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a digest algorithm availability review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Approval Signature Validation

**Condition:** The boundary is crossed when approval evidence has an invalid or unverifiable signature.

**Authoritative evidence:** The executor relies on approval service verification result.

**Defensive decision:** It must deny without attempting to repair the record.

**Non-authority:** The system must not treat a reviewer name included in text as equivalent evidence.

**Recovery note:** The minimal receipt should name the approval signature validation decision category without recording protected payload content.

**Detection signal:** A rising count of `approval_signature_validation` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of approval signature validation must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of approval signature validation uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a approval signature validation review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Approval Expiry

**Condition:** The boundary is crossed when trusted time is later than the approval expiry.

**Authoritative evidence:** The executor relies on trusted-clock evaluation with recorded timestamps.

**Defensive decision:** It must deny and require a fresh decision.

**Non-authority:** The system must not treat client device time as equivalent evidence.

**Recovery note:** The minimal receipt should name the approval expiry decision category without recording protected payload content.

**Detection signal:** A rising count of `approval_expiry` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of approval expiry must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of approval expiry uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a approval expiry review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Approval Reviewer Role

**Condition:** The boundary is crossed when the approving account no longer has the required role.

**Authoritative evidence:** The executor relies on role assertion at decision record creation.

**Defensive decision:** It must invalidate the decision according to policy.

**Non-authority:** The system must not treat a historical email indicating seniority as equivalent evidence.

**Recovery note:** The minimal receipt should name the approval reviewer role decision category without recording protected payload content.

**Detection signal:** A rising count of `approval_reviewer_role` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of approval reviewer role must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of approval reviewer role uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a approval reviewer role review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Capability Revocation

**Condition:** The boundary is crossed when the capability is revoked after review.

**Authoritative evidence:** The executor relies on commit-time revocation epoch lookup.

**Defensive decision:** It must deny and preserve revocation correlation.

**Non-authority:** The system must not treat the approval’s original status as equivalent evidence.

**Recovery note:** The minimal receipt should name the capability revocation decision category without recording protected payload content.

**Detection signal:** A rising count of `capability_revocation` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of capability revocation must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of capability revocation uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a capability revocation review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Capability Audience

**Condition:** The boundary is crossed when a capability is presented to a different executor audience.

**Authoritative evidence:** The executor relies on audience claim comparison.

**Defensive decision:** It must deny as a token-routing error.

**Non-authority:** The system must not treat a matching operation label as equivalent evidence.

**Recovery note:** The minimal receipt should name the capability audience decision category without recording protected payload content.

**Detection signal:** A rising count of `capability_audience` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of capability audience must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of capability audience uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a capability audience review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Byte Meter Precision

**Condition:** The boundary is crossed when the preview meter and executor meter use different byte conventions.

**Authoritative evidence:** The executor relies on canonical byte-count comparison.

**Defensive decision:** It must deny on nonzero disagreement until unified.

**Non-authority:** The system must not treat token count or character count as equivalent evidence.

**Recovery note:** The minimal receipt should name the byte meter precision decision category without recording protected payload content.

**Detection signal:** A rising count of `byte_meter_precision` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of byte meter precision must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of byte meter precision uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a byte meter precision review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Record Count Enforcement

**Condition:** The boundary is crossed when partitioning would exceed the approved commit count.

**Authoritative evidence:** The executor relies on central count of successful receipts.

**Defensive decision:** It must deny the extra partition.

**Non-authority:** The system must not treat a client-side count as equivalent evidence.

**Recovery note:** The minimal receipt should name the record count enforcement decision category without recording protected payload content.

**Detection signal:** A rising count of `record_count_enforcement` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of record count enforcement must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of record count enforcement uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a record count enforcement review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Idempotency Key Stability

**Condition:** The boundary is crossed when a retry changes serialization but represents the same operation.

**Authoritative evidence:** The executor relies on server-generated operation identifier lookup.

**Defensive decision:** It must return existing terminal state.

**Non-authority:** The system must not treat a newly generated client request identifier as equivalent evidence.

**Recovery note:** The minimal receipt should name the idempotency key stability decision category without recording protected payload content.

**Detection signal:** A rising count of `idempotency_key_stability` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of idempotency key stability must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of idempotency key stability uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a idempotency key stability review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Idempotency Store Outage

**Condition:** The boundary is crossed when the terminal-state store cannot be read.

**Authoritative evidence:** The executor relies on explicit store-health response.

**Defensive decision:** It must hold rather than guess that no commit occurred.

**Non-authority:** The system must not treat an absence of cached receipt data as equivalent evidence.

**Recovery note:** The minimal receipt should name the idempotency store outage decision category without recording protected payload content.

**Detection signal:** A rising count of `idempotency_store_outage` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of idempotency store outage must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of idempotency store outage uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a idempotency store outage review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Policy Version Retention

**Condition:** The boundary is crossed when the approval references a policy version no longer available.

**Authoritative evidence:** The executor relies on version archive lookup.

**Defensive decision:** It must deny or re-evaluate under a documented new decision.

**Non-authority:** The system must not treat a mutable current policy page as equivalent evidence.

**Recovery note:** The minimal receipt should name the policy version retention decision category without recording protected payload content.

**Detection signal:** A rising count of `policy_version_retention` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of policy version retention must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of policy version retention uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a policy version retention review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Policy Decision Freshness

**Condition:** The boundary is crossed when policy input age exceeds its permitted evaluation window.

**Authoritative evidence:** The executor relies on decision timestamp and freshness rule.

**Defensive decision:** It must re-evaluate before any effect.

**Non-authority:** The system must not treat the model’s recent conversation context as equivalent evidence.

**Recovery note:** The minimal receipt should name the policy decision freshness decision category without recording protected payload content.

**Detection signal:** A rising count of `policy_decision_freshness` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of policy decision freshness must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of policy decision freshness uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a policy decision freshness review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Writer Inventory Coverage

**Condition:** The boundary is crossed when an integration can write without the narrow executor.

**Authoritative evidence:** The executor relies on attested writer inventory compared with deployment routes.

**Defensive decision:** It must disable exports until every writer is mediated.

**Non-authority:** The system must not treat an undocumented operational convention as equivalent evidence.

**Recovery note:** The minimal receipt should name the writer inventory coverage decision category without recording protected payload content.

**Detection signal:** A rising count of `writer_inventory_coverage` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of writer inventory coverage must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of writer inventory coverage uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a writer inventory coverage review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Controlled Sink Correlation

**Condition:** The boundary is crossed when the sink returns an acknowledgment that cannot be tied to the operation.

**Authoritative evidence:** The executor relies on authenticated correlation reference.

**Defensive decision:** It must mark terminal state uncertain and reconcile.

**Non-authority:** The system must not treat a generic success message as equivalent evidence.

**Recovery note:** The minimal receipt should name the controlled sink correlation decision category without recording protected payload content.

**Detection signal:** A rising count of `controlled_sink_correlation` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of controlled sink correlation must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of controlled sink correlation uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a controlled sink correlation review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Journal Write Ordering

**Condition:** The boundary is crossed when a receipt write races with an external acknowledgment.

**Authoritative evidence:** The executor relies on ordered state transition and reconciliation marker.

**Defensive decision:** It must avoid declaring final success automatically.

**Non-authority:** The system must not treat a best-effort log line as equivalent evidence.

**Recovery note:** The minimal receipt should name the journal write ordering decision category without recording protected payload content.

**Detection signal:** A rising count of `journal_write_ordering` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of journal write ordering must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of journal write ordering uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a journal write ordering review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Receipt Access Control

**Condition:** The boundary is crossed when a support role can read receipt metadata without need.

**Authoritative evidence:** The executor relies on access review against support duties.

**Defensive decision:** It must restrict access and review past readers.

**Non-authority:** The system must not treat the claim that receipts hold no payload as equivalent evidence.

**Recovery note:** The minimal receipt should name the receipt access control decision category without recording protected payload content.

**Detection signal:** A rising count of `receipt_access_control` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of receipt access control must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of receipt access control uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a receipt access control review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Receipt Retention

**Condition:** The boundary is crossed when metadata remains beyond an approved retention interval.

**Authoritative evidence:** The executor relies on retention policy reference and deletion evidence.

**Defensive decision:** It must apply governed deletion or legal-hold exception.

**Non-authority:** The system must not treat an indefinite troubleshooting preference as equivalent evidence.

**Recovery note:** The minimal receipt should name the receipt retention decision category without recording protected payload content.

**Detection signal:** A rising count of `receipt_retention` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of receipt retention must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of receipt retention uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a receipt retention review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Diagnostic Redaction

**Condition:** The boundary is crossed when an exception serializer includes a source passage.

**Authoritative evidence:** The executor relies on schema-based log inspection.

**Defensive decision:** It must redact and prevent recurrence before broad testing.

**Non-authority:** The system must not treat a manual promise not to inspect logs as equivalent evidence.

**Recovery note:** The minimal receipt should name the diagnostic redaction decision category without recording protected payload content.

**Detection signal:** A rising count of `diagnostic_redaction` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of diagnostic redaction must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of diagnostic redaction uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a diagnostic redaction review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Queue Degradation

**Condition:** The boundary is crossed when human review service has no available reviewer.

**Authoritative evidence:** The executor relies on workflow state distinguishes capacity from system fault.

**Defensive decision:** It must safe hold or governed manual route.

**Non-authority:** The system must not treat automatic approval after a timeout as equivalent evidence.

**Recovery note:** The minimal receipt should name the queue degradation decision category without recording protected payload content.

**Detection signal:** A rising count of `queue_degradation` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of queue degradation must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of queue degradation uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a queue degradation review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Recovery Ownership

**Condition:** The boundary is crossed when an uncertain effect lacks a named accountable owner.

**Authoritative evidence:** The executor relies on receipt metadata contains no recovery assignment.

**Defensive decision:** It must block the pilot until ownership is assigned.

**Non-authority:** The system must not treat a shared mailbox with no decision authority as equivalent evidence.

**Recovery note:** The minimal receipt should name the recovery ownership decision category without recording protected payload content.

**Detection signal:** A rising count of `recovery_ownership` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of recovery ownership must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of recovery ownership uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a recovery ownership review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

#### Incident Boundary

**Condition:** The boundary is crossed when a suspected disclosure signal arrives after a commit.

**Authoritative evidence:** The executor relies on correlate minimal receipt evidence with controlled sink state.

**Defensive decision:** It must invoke the approved incident process without widening access.

**Non-authority:** The system must not treat an attempt to reconstruct payload from routine logs as equivalent evidence.

**Recovery note:** The minimal receipt should name the incident boundary decision category without recording protected payload content.

**Detection signal:** A rising count of `incident_boundary` denials is a review trigger, not evidence that a release occurred.

**Accountability:** The owner of incident boundary must decide whether the condition is isolated, systemic, or a release blocker.

**Privacy limit:** Investigation of incident boundary uses receipt metadata and authoritative references; it does not justify collecting additional payload text.

**Closure criterion:** Close a incident boundary review only when the authoritative evidence, terminal receipt state, and controlled-sink correlation agree.

## 13. Transferable Lesson

The reusable pattern is not merely adding human approval. It is binding a consequential effect to authenticated scope, an independently evaluated policy, an exact representation of what is approved, finite budgets, current revocation, idempotent execution, privacy-minimized evidence, and an accountable recovery path.

In a payment workflow, substitute payee and amount for destination and bytes; in a deployment workflow, substitute environment and artifact digest; in a messaging workflow, substitute recipient group and message classification. The same question remains: can the executor independently prove that the effect is still inside the authorized envelope?

The counterexample is a low-risk read-only query with no external effect and no sensitive response persistence. For that case, an approval queue may add delay without materially reducing an effect risk; provenance and output handling may be the more relevant controls.

A second counterexample is a task that cannot be represented deterministically enough to bind a preview to an effect. The correct response is not a vague approval but a different workflow, such as manual handling or a narrower action vocabulary.

| Transfer test | Export case expression | Payment analogue | Deployment analogue | Messaging analogue |
|---|---|---|---|---|
| Actor binding | authenticated principal | account holder | release approver | message sender |
| Scope binding | tenant and records | payee and amount | service and environment | recipient group and content class |
| Exactness binding | canonical payload digest | transaction digest | artifact digest | rendered message digest |
| Budget binding | bytes and commit count | currency and transfer count | rollout percentage | recipient count |
| Time binding | approval expiry | payment window | change window | send window |
| Revocation | capability epoch | payment hold | deployment freeze | campaign cancellation |
| Idempotency | operation receipt | transaction reference | deployment run identifier | message dispatch identifier |
| Privacy evidence | minimal receipt | minimized ledger metadata | change audit fields | delivery audit fields |
| Recovery | sink-journal reconciliation | settlement reconciliation | runtime state reconciliation | delivery reconciliation |
## 14. Connections & Cross-References

Related theory includes [Threat Model and Execution Trust](../Cheat_Sheets/01_AI_Threat_Model_and_Execution_Trust.md) and [Prompt Injection and Red-Team Evaluation](../Cheat_Sheets/02_Prompt_Injection_and_Red_Team_Evaluation.md). These references provide vocabulary and defensive evaluation ideas; neither is evidence for a deployment described here.

The case also connects to privacy-by-design practices: data minimization, purpose limitation, storage limitation, access control, and accountable records. A minimal receipt can improve recoverability while avoiding a duplicate repository of sensitive payload content.

It connects to reliability engineering through idempotency, durable terminal state, timeouts, and reconciliation. Those mechanisms are security-relevant because uncertain retries and split-brain records can expand an effect even when the initial authorization was valid.

It connects to product leadership because an approval threshold expresses a tradeoff between workflow friction and blast-radius control. That tradeoff must be owned explicitly; it cannot be delegated to a language model or hidden inside a convenience API.

### Interview pack

Each answer is specific to this ILLUSTRATIVE case. Scores use a 0–3 scale: 0 is unsafe or materially incorrect, 1 identifies a term without applying it, 2 applies the core reasoning with a minor omission, and 3 identifies the decision boundary and a concrete defensive consequence.

#### Question 1 — Why is schema validation insufficient for an export proposal?

**Answer:** Schema validation can establish that fields have an expected shape, but it cannot establish that the tenant, recipient, purpose, record set, or budget is authorized.

**Reasoning:** A model can produce syntactically valid identifiers sourced from untrusted context. Authority must originate in authenticated middleware, policy, and the executor rather than in a well-formed proposal.

**Follow-up:** Name one field that must be server-resolved even if the schema marks it required.

**Red flag:** “Valid JSON means the tool request is safe.”

**Scoring:** 3 requires separating structure from authorization and naming an independent authority source.

#### Question 2 — Compute the illustrative maximum payload when q=20 records per commit, c=3 commits per approval, and b=4,096 bytes per record.

**Answer:** The maximum is 20×3×4,096=245,760 bytes, which is 240 KiB.

**Reasoning:** First compute 20×3=60 records per approval, then multiply 60×4,096=245,760 bytes. The equality to 240 KiB uses 1 KiB=1,024 bytes.

**Follow-up:** Does 20 KiB of unused byte capacity permit a fourth commit?

**Red flag:** Treating KiB as 1,000 bytes or ignoring the separate commit limit.

**Scoring:** 3 requires correct arithmetic, units, and the count-limit caveat.

#### Question 3 — A document says that its tenant is T-99 while the authenticated session is T-17. What should determine the effective tenant?

**Answer:** The authenticated session binding must determine the effective tenant; the document text is untrusted content.

**Reasoning:** Letting retrieved text select tenant would make the model and document an identity source. The executor must compare each selected record against the session-derived tenant.

**Follow-up:** What is the safe decision if the authenticated tenant is missing?

**Red flag:** Allowing the model to choose the more plausible tenant label.

**Scoring:** 3 requires denial on absent authenticated tenant evidence.

#### Question 4 — Why must destination class be included in the canonical approval digest?

**Answer:** Destination class changes the recipient context of the effect, so omitting it permits a payload match without a recipient match.

**Reasoning:** The digest is meaningful only if every effect-relevant field participates in deterministic canonicalization. A payload-only digest cannot prove where the payload will be sent.

**Follow-up:** What happens when the destination catalog marks the workspace suspended after approval?

**Red flag:** Treating destination as display metadata rather than authorization scope.

**Scoring:** 3 requires a commit-time catalog check and safe denial for suspension.

#### Question 5 — A 55-record payload uses 4,096 bytes per record. Is it within the illustrative cap, and by how much?

**Answer:** It uses 55×4,096=225,280 bytes and is 245,760−225,280=20,480 bytes, or 20 KiB, below the byte cap.

**Reasoning:** The arithmetic checks bytes first and then converts the difference using 1,024 bytes per KiB. Eligibility also requires that the operation stays within the separate three-commit bound.

**Follow-up:** Why does this calculation not prove that the release is safe?

**Red flag:** Concluding that a below-cap size bypasses classification or approval checks.

**Scoring:** 3 requires both intermediate multiplication and the non-quantitative policy caveat.

#### Question 6 — How should the executor handle a retry that uses an operation identifier with an existing committed receipt?

**Answer:** It should return the original receipt and create no second controlled-sink effect.

**Reasoning:** Durable idempotency makes the operation identifier a stable reference to the terminal decision. Re-executing because a client is uncertain can duplicate the release.

**Follow-up:** What is the safe state if receipt lookup is unavailable during the retry?

**Red flag:** Automatically retrying because the prior effect cannot be found.

**Scoring:** 3 requires a safe hold or reconciliation path for lookup uncertainty.

#### Question 7 — Diagnose a case where the reviewer approved 55 records but the commit request contains 56.

**Answer:** The executor must recompute the canonical selection digest and deny because the approved and requested selections differ.

**Reasoning:** The material issue is not whether the added record appears harmless; it is that the human decision did not cover the changed selection. A new preview and decision are required.

**Follow-up:** Which other fields besides record identifiers must be canonicalized?

**Red flag:** Deleting the extra record automatically without recording a mismatch.

**Scoring:** 3 requires naming destination, tenant, purpose, and policy version as effect-relevant bindings.

#### Question 8 — What is the review-rate formula for 18 reviewed and 42 automatically permitted eligible requests?

**Answer:** The illustrative review rate is 18/(18+42)=18/60=0.30, or 30%.

**Reasoning:** The numerator and denominator share one eligible population and one stated window. Denied requests should be reported separately so that a high denial rate is not hidden.

**Follow-up:** Why does 30% not determine how many reviewers are needed?

**Red flag:** Treating a fraction of requests as a staffing conclusion without arrival and service-time data.

**Scoring:** 3 requires correct fraction, population definition, and capacity limitation.

#### Question 9 — What does a missing historical policy version imply for a pending approval?

**Answer:** The pending approval should be denied or held for re-evaluation because its original decision cannot be interpreted reliably.

**Reasoning:** Substituting today’s policy could broaden or narrow a decision without evidence. Retained policy versions make approvals auditable rather than mutable narratives.

**Follow-up:** What receipt evidence should identify the relevant policy?

**Red flag:** Silently applying the latest policy and claiming the original approval remains valid.

**Scoring:** 3 requires exact version binding and a conservative disposition.

#### Question 10 — A classification changes from internal to restricted after preview. What is the defensible executor decision?

**Answer:** Hold or deny the request and require a current evaluation; the stale preview must not commit under the earlier classification.

**Reasoning:** Classification is part of policy scope. A correct earlier review does not remain correct when a material attribute changes before effect.

**Follow-up:** What distinguishes catalog drift from a model hallucination in this case?

**Red flag:** Assuming the approval remains valid because the record identifiers did not change.

**Scoring:** 3 requires a commit-time authoritative catalog comparison.

#### Question 11 — Why should receipts avoid full payload content even though they support recovery?

**Answer:** Full payload logging can create a second sensitive store with broader readers and longer retention than the original system.

**Reasoning:** Recovery usually needs operation identifiers, digest, counts, bytes, policy evidence, terminal state, and sink correlation. Those fields support reconciliation without duplicating content.

**Follow-up:** Is a digest automatically anonymous?

**Red flag:** Claiming that auditability always justifies retaining payload text.

**Scoring:** 3 requires minimizing data and rejecting an automatic anonymity claim.

#### Question 12 — Calculate the containment proxy when B_cap=245,760 bytes, B_selected=225,280 bytes, and I_commit=0.

**Answer:** E_exposure=min(245,760,225,280)×0=225,280×0=0 bytes in the illustrative proxy.

**Reasoning:** The indicator models whether the controlled effect committed. It does not prove that no information was exposed through another path or before the executor.

**Follow-up:** Give one condition that would invalidate this proxy.

**Red flag:** Treating a zero proxy as proof of no security event.

**Scoring:** 3 requires the zero arithmetic and a limitation such as an alternate writer.

#### Question 13 — How should an ambiguous state be handled when the sink acknowledges but the journal append fails?

**Answer:** Treat the state as uncertain, prevent automatic repeat, preserve minimal correlation evidence, and assign reconciliation to the recovery owner.

**Reasoning:** Declaring either success or no-op without correlation can create duplicate effects or conceal a committed effect. Ambiguity is a distinct terminal state until reconciled.

**Follow-up:** Which fields enable reconciliation without retaining payload text?

**Red flag:** Retrying immediately because the journal has no receipt.

**Scoring:** 3 requires no automatic retry and a named ownership path.

#### Question 14 — What is the product leadership decision when a reviewer queue is unavailable?

**Answer:** Choose a documented safe hold, read-only fallback, or governed manual path; never convert queue unavailability into implicit approval.

**Reasoning:** Availability pressure can undermine a control if the system fails open. The policy owner and product owner must explicitly decide the service experience and residual risk.

**Follow-up:** What evidence distinguishes a staffing shortage from an approval-service outage?

**Red flag:** “Users need the feature, so allow low-risk requests automatically” without a policy change.

**Scoring:** 3 requires a safe disposition and explicit accountable ownership.

#### Question 15 — Which four facts must be independently recomputed immediately before commit?

**Answer:** A defensible minimum is tenant ownership, current classification eligibility, canonical digest match, and byte/count budget; expiry, revocation, destination state, and policy version are also required in this design.

**Reasoning:** Recomputing turns an approval from a trusted message into an enforceable condition. The executor is the last point that can stop a changed effect.

**Follow-up:** Why is recomputing only the digest insufficient?

**Red flag:** Listing model confidence or prompt wording as a commit-time authority source.

**Scoring:** 3 requires independent sources of truth and recognition of non-digest constraints.

### Cross-reference map

| Topic | Relevant case mechanism | Question it helps answer |
|---|---|---|
| Prompt injection | provenance-labelled retrieval | why document instructions lack authority |
| Authorization | authenticated scope and policy | who may request which bounded effect |
| Privacy engineering | minimized receipts and retention | how audit avoids a duplicate data store |
| Distributed systems | idempotency and reconciliation | how retries avoid repeated effects |
| Incident readiness | revocation and uncertain-state handling | how to contain before full diagnosis |
| Product governance | named risk owner and thresholds | who decides tradeoffs and exceptions |
| Evaluation design | constructed negative and positive cases | what a defensive simulator can falsify |
| Data governance | classification and purpose limitation | which records may enter the workflow |

## 15. Quick Reference Card

| Field | Defensive reference |
|---|---|
| Core decision | Commit only when independent evidence binds actor, scope, effect, time, and budget |
| Trust boundary | Model and retrieved text may propose; they do not authorize or execute |
| Tenant source | Authenticated middleware, frozen by the request broker |
| Destination source | Server-side registered destination catalog |
| Approval binding | Canonical digest of the effect-relevant preview |
| Illustrative cap | 20 records/commit × 3 commits/approval × 4,096 bytes/record = 240 KiB |
| Commit-time checks | tenant, classification, destination, digest, budget, expiry, revocation, policy version |
| Retry rule | Same operation identifier returns the original receipt and no second effect |
| Fail-safe rule | Missing, stale, ambiguous, or unavailable evidence produces denial or safe hold |
| Receipt contents | operation and receipt identifiers, digest, counts, bytes, policy reference, timestamps, state |
| Receipt exclusions | payload content, raw retrieval text, broad destination secrets, credentials |
| Detection focus | mismatches, stale approvals, duplicate identifiers, drift, uncertain journal state |
| Containment focus | disable routing, revoke pending capability, stop automatic retry, preserve minimal evidence |
| Recovery focus | reconcile controlled sink and journal using durable identifiers and accountable owner |
| Outcome status | Not measured; all scenario values and arithmetic are ILLUSTRATIVE |
| Release blocker | unknown writer, model-held credential, non-deterministic preview, fail-open dependency, or unreconciled effect state |
