# Model Context Protocol: Contracts, Execution Boundaries, and Recovery

## Section 1 — File Header

> **Sheet ID:** MCP-01. **Level:** 3, advanced production. **Editorial status:** draft; unreviewed.
> **Purpose:** Trace and secure a Model Context Protocol (MCP) session, and diagnose uncertain tool execution without mistaking a protocol response for a business guarantee.
> **Prerequisites:** JSON objects and RPC request/response IDs; HTTP authentication; basic process isolation, retry and side-effect semantics. See the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md).
> **Outcome:** Given a client/server trace, authorization context and a side-effecting call, identify the failed boundary, choose a safe recovery, and calculate a bounded request-budget estimate.
> **Excludes:** Implementing a particular SDK; detailed OAuth provider configuration; proving the safety of arbitrary tools. Those require version-specific implementation and security review.
> **Evidence:** Protocol facts are SOURCED to pinned 2025-06-18 specification pages below. Scenarios and numbers are ILLUSTRATIVE, not production observations. Calculations are DERIVED from stated assumptions; no benchmark was run.
> **Version caveat:** This sheet deliberately uses the 2025-06-18 protocol, not an assertion about the newest version. Later specifications and SDK releases may differ. Recheck normative text when upgrading.
> **Retrieval cue:** For a timeout after a write, start at Sections 6 and 8; for missing tools, start at Sections 2 and 8.

The decision is not simply whether a tool is discoverable.
The decision is whether a particular authenticated principal can safely perform a particular operation on a particular resource at this time.
Protocol compatibility is necessary for interoperability, but it is not sufficient for authority, isolation, or safe execution.
The source inventory names the original lesson and related MCP materials, but the original `Train/Agents_Stack_Infra` files are unavailable in this workspace.
Consequently, the crosswalk is a scope locator, not evidence of their contents; this sheet relies on the cited primary specification instead.

**Takeaway:** Separate interoperability, authorization, execution, and evidence of effect before troubleshooting.

## Section 2 — Concept Map / Mental Model

```text
User and application policy
        | desired task and permitted scope (control)
        v
Host / model / orchestration
        | selects candidate action (proposal, not authorization)
        v
MCP client -- initialization and negotiated capabilities --> MCP server
        |                        JSON-RPC messages                  |
        |                                                    validates request
        |                                                    authenticates caller
        |                                                    authorizes resource
        |                                                            |
        |                                                            v
        |                                                   executor / downstream API
        |                                                            |
        |<-- result or error; may be absent if connection fails -----|
        v
Host treats output as data; verifies effect where needed
```

The arrows show data and control flow, not a claim that the model executes operating-system operations.
A host may create separate clients for multiple servers; that does not imply shared server state or shared authority.
The server may itself perform an operation, or it may delegate to an executor or external service.
The implementation's security boundary must be explicitly identified in either case.
Discovery advertises interfaces; it does not grant an identity permission to use them.
A successful JSON-RPC exchange is not proof that an external business process committed.
A disconnected stream does not establish that an already-dispatched write was cancelled.
The governing invariant is that every effect must be authorized at the actual enforcement point under the current principal and resource scope.

**Takeaway:** Follow the action across each trust boundary instead of treating MCP as the executor.

## Section 3 — Foundational Definitions

| Term | Definition and example | Boundary / operational check |
|---|---|---|
| Host | Application coordinating model interactions and MCP clients. | A model-generated call proposal is not itself an execution authorization. |
| Client | Protocol participant connected to a server, typically controlled by a host. | Check its negotiated version and capabilities per connection. |
| Server | Protocol participant exposing supported primitives to a client. | It may call a separate executor; being a server does not imply sandboxing. |
| Tool | Callable server capability with a name and input contract. | Treat tool descriptions and results as untrusted content; validate inputs and enforce permissions independently. |
| Resource | Server-provided readable context identified by URI. | Resource content is data, not an instruction overriding host policy. |
| Prompt | Server-provided reusable message template. | A template is not inherently trusted system policy. |
| Capability | Negotiated support for a protocol feature. | Feature support is not user-specific authorization. |
| Session | Logically related interactions beginning at initialization. | A session ID is not a substitute for user authentication or authorization. |
| JSON-RPC request ID | Correlation identifier for one request and its response. | It is not automatically a durable business idempotency key. |
| Principal | Authenticated user, service or delegated identity evaluated by policy. | A valid session ID alone is not proof of principal identity. |
| Executor | Component that actually acts on a filesystem, database, API or other resource. | It needs its own least-privilege and isolation controls. |
| Idempotency key | Application-defined durable key binding repeated attempts to one intended effect. | Effective only if the execution boundary atomically stores and enforces it. |
| Unknown outcome | Client cannot determine whether a side effect occurred. | Reconcile against authoritative state before reissuing an unsafe write. |
| Stream resumption | Optional recovery of server events after a disconnect. | It is not a guarantee of tool-effect deduplication. |

A tool schema defines accepted shape; application policy must still check semantics and authorization.
For instance, a syntactically valid path may resolve outside an allowed directory after symlink resolution.
A resource URI identifies content but is not evidence that the content is accurate, safe, or fresh.
A root communicated by a client is useful context, not a replacement for server-side access enforcement.
A notification has no JSON-RPC response ID; lack of a response is expected rather than proof that an effect completed.

**Takeaway:** Protocol identifiers and capabilities describe communication, not business authority or effect uniqueness.

## Section 4 — Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**Validity:** Each logical call is attempted once; a call has one response or an explicitly recorded missing response.
This is a counting identity, not a probability of business success.
Define the response-observation rate as observed responses divided by sent requests.
Partition requests into those with an observed response and those without one; the categories cover all requests by construction.

```text
FORMULA [M1]: r = R / N
WHERE: N > 0 is sent request count; R is observed response count; 0 <= R <= N.
CONSTRAINT: Each request is counted once in a closed observation window.
NUMERICAL EXAMPLE: N = 20 requests; R = 18 responses; r = 18/20 = 0.90.
INTERPRETATION: 90% had observed responses; two outcomes require further classification.
```

The numerator includes success and error responses unless separately filtered.
A response may describe a tool error; a missing response may follow a completed side effect.
Thus response observation is not equivalent to successful authorized business completion.

### 4.2 General Case

**Validity:** Fixed number of logical calls; each failure is independently retried at most once; both attempts have the same failure probability `p`.
This model concerns transport attempts, not tool effects and not a production reliability claim.
The first attempt always occurs; a second occurs only if the first fails.
Therefore expected attempts per logical call equal one plus the probability of reaching the retry.
The final failure event requires both attempts to fail; independence permits multiplication.

```text
FORMULA [M2]: E[A] = 1 + p;  P(final failure) = p * p
WHERE: A is attempts per logical call; p is per-attempt failure probability in [0,1].
CONSTRAINT: At most one retry; independent, identically distributed attempt failures.
NUMERICAL EXAMPLE: p = 0.10; E[A] = 1.10 attempts; P(final failure) = 0.01.
INTERPRETATION: One retry adds 10% expected attempt load while modelled failure falls to 1%.
```

For `N` logical calls, expected attempts are `N * E[A]` by linearity of expectation.
Linearity does not require independence across different calls; the squared failure expression does require independence within each call.
A shared outage violates that independence and may make the second attempt fail whenever the first fails.
The policy must not retry a side-effecting operation merely because the transport failed.

### 4.3 Key Special Cases / Variants

| Variant | Changed condition | Appropriate inference | Avoid when |
|---|---|---|---|
| No retry | One attempt per logical call | Expected attempts = 1; failure probability = p. | A safe read has a recoverable transient failure and latency permits retry. |
| One independent retry | M2 assumptions hold | Expected attempts = 1+p; final failure = p². | A write might already have committed or failures are correlated. |
| Perfectly correlated failures | Retry fails exactly when first attempt fails | Final failure remains p despite retry load. | Assuming a new attempt implies an independent failure domain. |
| Durable deduplicated write | Executor atomically binds key to effect and result | Repeated submissions with the same key can return one recorded effect. | Key scope, retention or payload consistency is not enforced. |
| Reconciliation before retry | Authoritative state can resolve ambiguous write | Determine prior result, then decide whether a new action is appropriate. | State lookup itself is stale or cannot uniquely identify the operation. |

A deduplication mechanism is an application design, not a guarantee implied by MCP.
The same key used for different payloads should be rejected rather than silently returning an unrelated prior result.
The retention period must exceed the actual replay horizon if old attempts can reappear.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

When `p = 0`, the retry path is never entered and expected attempts equal one.
When `p = 1`, every call receives two attempts and still fails under the model.
For a persistent outage, repeated immediate retries amplify traffic without improving availability.
For side effects, even `p = 0.1` does not establish safety: a request may commit and lose its response.
A request ID may be newly generated on retry, so matching responses cannot by itself prevent duplicate execution.
For an expired deduplication record, an apparently repeated key can become a new effect unless retention and recovery are designed together.
A zero-denominator response rate is undefined when no requests were sent; do not report it as zero or 100%.

**Takeaway:** Retry arithmetic is useful only when its failure model and side-effect safety conditions are explicit.

## Section 5 — Worked Numerical Example

**Evidence:** ILLUSTRATIVE, constructed workload; arithmetic is derived here, not measured.
**Decision:** Assess a proposed retry budget for a read-only MCP integration before any live load test.
**Inputs:** 1,000 logical read requests in a window; per-attempt failure probability 0.10; one retry at most.
**Assumptions:** Failures within each call are independent; retries remain authorized; all attempts have equal cost; no concurrency limit or time-varying outage.
**Step 1:** First attempts total 1,000.
**Step 2:** Expected first-attempt failures are 1,000 × 0.10 = 100.
**Step 3:** Those failures receive 100 expected additional attempts.
**Step 4:** Expected total attempts are 1,100, or 1.10 attempts per logical request.
**Step 5:** Expected final failures are 1,000 × 0.10 × 0.10 = 10.
**Step 6:** Expected successful logical requests are 1,000 − 10 = 990.
**Independent check:** 900 expected first-attempt successes plus 100 × 0.90 = 90 retry successes gives 990.
**Units check:** requests × dimensionless probability yields requests; attempts per request × requests yields attempts.
**Wrong path:** Treating 1,100 attempts as 1,100 independent user successes double-counts retries.
**Sensitivity:** With perfectly correlated failures at the same marginal 10%, final failures remain 100 rather than 10, while expected attempts remain 1,100.
**Production interpretation:** Size and test downstream rate limits for the additional attempts, including bursts; do not treat an expectation as a capacity or latency guarantee.
**Write variant:** If these were writes and a lost response could follow commit, neither the probability result nor the read retry policy makes re-execution safe.
**Evidence gap:** Actual error correlations, latency distribution, failure modes, and per-operation idempotency have not been measured.

**Takeaway:** A retry can improve an idealized read success rate and still increase load or duplicate writes in reality.

## Section 6 — Visual / Geometric Intuition

```text
Time flows downward; actors are named columns, not a scale plot.

Client                    MCP server                    Downstream
  | POST with tools/call       |                             |
  |--------------------------->| authorize and dispatch      |
  |                            |---------------------------->|
  |                            |                       COMMIT X
  |                            |<----------------------------|
  |          network loss      X                             |
  |<----------- no result -----|                             |
  |                            |                             |
  | query operation status --------------------------------->|
  |<----------------------- committed X ---------------------|
  | report original result; do not execute X again           |
```

Caption: Illustrative uncertain-response window for a side-effecting request; `X` is one business operation.
The network break occurs after commit but before the client observes the result.
Cancellation sent after this point is not a rollback.
The authoritative lookup is an application-specific capability, not an MCP guarantee.
If no lookup exists and deduplication is absent, escalate the ambiguous result rather than claiming exactly-once execution.
If a status store is eventually consistent, a temporary absence is not proof that the write did not commit.

**Takeaway:** A timeout locates uncertainty at the client, not necessarily failure at the executor.

## Section 7 — Comparison / Selection Table

| Option | Preconditions | Guarantee and cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|
| Direct application API | One known integration; application owns contract | Application-specific semantics; custom adapter maintenance. | Simple stable one-off integration. | Common discovery and multi-client interoperability are core requirements. | ILLUSTRATIVE design comparison; not benchmarked. |
| MCP over stdio | Client can launch trusted local child process | Local process transport; JSON-RPC on stdin/stdout; isolation remains external. | Local integrations with controlled process lifecycle. | Remote shared service or untrusted binary without isolation. | [Transport specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports). |
| MCP Streamable HTTP | Client and server support specified version and network security | Shared endpoint, POST and optional SSE behavior; identity and network controls add work. | Remote independently operated servers. | Deployment cannot authenticate, authorize, or protect endpoint. | Same pinned transport specification. |
| Broad shell tool | Sandboxed execution and strict authority are feasible | Flexible actions with large execution surface; policy cannot be inferred from a text prompt. | Controlled administrative tasks where a constrained API is insufficient. | User-controlled inputs can reach privileged commands or tenant boundaries. | HEURISTIC, not a measured comparison. |
| Narrow purpose-built tool | Clearly specified operation and resource scope | Smaller authority surface; more explicit contracts to maintain. | Frequent known business operations. | Required action genuinely cannot be expressed within the bounded contract. | HEURISTIC, not a measured comparison. |

Selection priority: reject any option that cannot satisfy identity, tenant isolation, and data-handling requirements.
Then choose the simplest integration that meets the interoperability requirement.
Prefer a narrow tool when it can express the intended task without a broad execution environment.
Choose local stdio only when the process provenance, effective OS identity, and lifecycle are acceptable.
Choose Streamable HTTP only with authenticated transport, resource authorization, and deployable network controls.
If the policies conflict with desired flexibility, escalate the required privilege rather than hiding it behind a protocol abstraction.
No throughput, latency, or cost comparison has been measured for these options.

**Takeaway:** Select on authority and operational requirements first, and treat transport choice as separate from execution safety.

## Section 8 — Failure Modes & Edge Cases

All cases below are HYPOTHETICAL teaching scenarios, not incident reports.

### F-01: Missing tools after connection

**Symptom:** Initialization succeeds but the expected tool is absent.
**Competing causes:** Server did not advertise tools, wrong server endpoint, stale discovery cache, pagination not exhausted, or principal-specific filtering.
**Discriminating check:** Record server identity, negotiated capabilities, `tools/list` response and pagination cursor under the same principal.
**Interpretation:** A missing tool in a complete response from the intended server is a discovery/configuration issue, not proof of a model defect.
**Negative check:** If the tool appears in the full current list, inspect host routing and tool selection instead.
**Mitigation:** Route to a known safe alternative only if independently authorized.
**Correction:** Fix registration, endpoint, list pagination or cache invalidation as supported by evidence.
**Verification:** Reinitialize, rediscover and compare the allowed tool list for both permitted and denied identities.

### F-02: Timeout after a write

**Symptom:** Client times out; downstream state may or may not have changed.
**Competing causes:** Dispatch failed, executor is running, commit succeeded but response was lost, or status is delayed.
**Discriminating check:** Correlate operation key with executor audit and authoritative downstream state, respecting consistency delay.
**Mitigation:** Freeze automatic retry of the affected side effect; preserve request metadata without credentials.
**Correction:** Reconcile the outcome; use durable deduplication or a business-specific compensation process where appropriate.
**Verification:** Inject a dropped response after commit in a controlled test and confirm exactly one effect for the supported key horizon.
**Limitation:** A test demonstrates behavior under its injection conditions, not a universal exactly-once guarantee.

### F-03: Cross-tenant access

**Symptom:** A valid request returns another tenant's resource.
**Competing causes:** Tenant derived from client-supplied arguments, pooled credentials, cache key missing tenant, or session identity confusion.
**Discriminating check:** Trace authenticated principal and tenant from ingress through policy decision, cache lookup and downstream access.
**Mitigation:** Disable the affected capability and revoke exposed credentials where appropriate.
**Correction:** Derive tenant from authenticated context and enforce resource ownership at the actual data boundary.
**Verification:** Attempt cross-tenant reads and writes, including cached and resumed requests, using separate test identities.
**Negative check:** If authorization decisions and cache keys are correct, inspect downstream data partitioning before concluding the incident is resolved.

### F-04: Prompt injection in a tool result

**Symptom:** The model proposes an unrelated privileged operation after reading retrieved content.
**Competing causes:** Malicious content, an overly broad host prompt, unsafe tool selection, or authorization failure.
**Discriminating check:** Compare the original user task, retrieved bytes, model proposal and policy decision without executing the proposed action.
**Mitigation:** Stop the dangerous operation and quarantine the offending content where operationally justified.
**Correction:** Treat tool output as untrusted data; require independent policy checks and user approval for sensitive actions.
**Verification:** Run an adversarial fixture that includes an instruction to exfiltrate data and confirm the executor denies it.
**Limitation:** A single injection fixture cannot prove immunity to all prompt attacks.

### F-05: Corrupted stdio framing

**Symptom:** Client reports malformed JSON-RPC messages immediately after server startup.
**Competing causes:** Diagnostic logs on stdout, multiline frames, incompatible launcher, or binary encoding mismatch.
**Discriminating check:** Capture raw stdout and stderr separately in a controlled environment, redacting secrets.
**Mitigation:** Stop the process rather than parsing arbitrary extra output as protocol content.
**Correction:** Send diagnostics to stderr and protocol messages only to stdout; ensure newline-delimited UTF-8 framing.
**Verification:** Start a clean server and parse each stdout line as one valid JSON-RPC message.
**Negative check:** If framing is clean, examine version negotiation and process termination separately.

### F-06: HTTP session expires

**Symptom:** A request using a previously assigned session ID receives HTTP 404.
**Competing causes:** Session expired, server intentionally terminated it, request reached the wrong state store, or deployment lost state.
**Discriminating check:** Inspect server session lifecycle and routing telemetry; do not log raw session tokens unnecessarily.
**Mitigation:** Stop using the stale session and initialize a new one as specified for this version.
**Correction:** Repair routing or state-store availability if the expiry was unintended.
**Verification:** Confirm new initialization and capability discovery, then reconcile any in-flight side effects before replay.
**Limitation:** A new session does not establish the outcome of the old session's unfinished write.

### F-07: Retry storm

**Symptom:** Attempt rate rises while useful completions fall.
**Competing causes:** Downstream outage, aggressive retries, small timeouts, throttling, or unbounded client concurrency.
**Discriminating check:** Compare logical request rate, attempt rate, status classes, retry count and downstream saturation on the same time axis.
**Mitigation:** Bound concurrency and retry budgets; pause retries for unsafe writes.
**Correction:** Address the failing dependency; use bounded backoff and jitter where retries are safe.
**Verification:** Controlled failure injection should show bounded attempts and eventual recovery without duplicate effects.
**Negative check:** If retries are not elevated, investigate queueing or payload growth rather than changing the retry policy blindly.

### F-08: Browser-origin attack on local HTTP server

**Symptom:** A browser-origin request reaches a local MCP endpoint without intended user action.
**Competing causes:** Missing Origin validation, broad network binding, absent authentication, or permissive proxy configuration.
**Discriminating check:** Exercise allowed and disallowed Origin values and inspect listener binding and authentication decisions.
**Mitigation:** Stop external exposure and revoke potentially exposed credentials.
**Correction:** Validate Origin, bind local services to loopback where appropriate, and apply authentication and authorization.
**Verification:** Test DNS-rebinding and untrusted-origin cases in an isolated environment.
**Caveat:** The 2025-06-18 transport text specifies Origin validation; implementation details depend on deployment and version.

### F-09: Replayed approval with changed arguments

**Symptom:** An approved action executes with a different destination, amount or resource.
**Competing causes:** Approval bound only to tool name, mutable payload, expired approval accepted, or identity mismatch.
**Discriminating check:** Compare approved canonical action, principal, target, parameters and expiry with the exact dispatched operation.
**Mitigation:** Suspend the write and require a fresh approval.
**Correction:** Bind approvals to an immutable action representation, identity, policy version and expiry; enforce at dispatch.
**Verification:** Change each protected field in a test and confirm the old approval is rejected.
**Limitation:** Approval does not replace downstream authorization or idempotent execution.

### F-10: Resource content becomes stale

**Symptom:** A host answers from outdated resource content even though the source changed.
**Competing causes:** Unsupported subscription, missed notification, stale host cache, or independently updated source.
**Discriminating check:** Compare resource version or source timestamp with cache metadata and negotiated subscription support.
**Mitigation:** Avoid presenting cached content as current where freshness is material.
**Correction:** Use supported change notifications plus bounded refresh or an authoritative reread policy.
**Verification:** Update a controlled source and check whether the client refreshes within its stated freshness objective.
**Limitation:** Notifications are not a substitute for a correctness and freshness contract.

**Takeaway:** Diagnose each symptom at the layer where evidence distinguishes competing causes, then verify a bounded correction.

### Production diagnostic playbook: lifecycle and protocol

The following short investigations are HYPOTHETICAL.
They are distinct checks, not evidence that any implementation was exercised.

#### D-01 — Initialization fails before discovery

**Signal:** The client receives an initialization error or cannot parse the response.
**Boundary:** Initialization precedes ordinary capability use.
**Hypothesis A:** Client and server do not share an acceptable protocol version.
**Hypothesis B:** The process or intermediary produced an invalid response.
**Check:** Compare the requested version, server-selected version and client-supported set.
**Check:** For stdio, separately inspect stderr and the raw protocol output.
**Expected distinction:** A well-formed response selecting an unsupported version differs from corrupted framing.
**Immediate action:** Disconnect rather than inventing a compatible version.
**Durable action:** Install compatible components or explicitly configure supported versions.
**Verification:** Repeat initialization with a pinned supported pair and inspect negotiated capabilities.
**Risk:** Downgrading solely to remove an error may remove required security or feature behavior.
**Escalation:** Seek compatibility guidance if no mutually supported version exists.

#### D-02 — Unexpected server capability

**Signal:** A server advertises an ability that the host was not configured to use.
**Boundary:** Advertisement says what the server supports, not what a principal may do.
**Hypothesis A:** A different server or deployment answered the request.
**Hypothesis B:** The intended server was upgraded or configured differently.
**Check:** Compare authenticated endpoint, server implementation identity and deployment manifest.
**Check:** Compare the negotiated feature set to the host's allowed feature policy.
**Expected distinction:** An identity mismatch points to routing; a matching identity with a changed manifest points to configuration drift.
**Immediate action:** Do not invoke the unexpected capability.
**Durable action:** Correct routing or review and approve the changed configuration.
**Verification:** Reinitialize and compare the full capability set with the approved baseline.
**Risk:** Blindly rejecting all changes can block a legitimate migration; review rather than silently enabling.
**Escalation:** Treat unexplained server identity changes as a security event.

#### D-03 — HTTP request has no protocol version

**Signal:** A server behaves as an older protocol version after an otherwise successful handshake.
**Boundary:** The pinned HTTP transport specifies a protocol-version header on subsequent requests.
**Hypothesis A:** A proxy strips the header.
**Hypothesis B:** The client never emits it or negotiated another version.
**Check:** Compare sanitized outbound and inbound header metadata at the proxy boundary.
**Check:** Compare the header value against the initialize response.
**Expected distinction:** Present at client and absent at server identifies an intermediary problem.
**Immediate action:** Stop operations that depend on unsupported interpretation.
**Durable action:** Preserve the header across the trusted path and pin the intended negotiated version.
**Verification:** Exercise initialization and subsequent request handling with the proxy enabled.
**Risk:** A proxy configuration fix must not allow arbitrary client-supplied identity headers to bypass authentication.
**Escalation:** Escalate version ambiguity if sensitive operations have already run.

#### D-04 — Stream disconnect during read

**Signal:** A Streamable HTTP response stream closes before the final response is observed.
**Boundary:** Stream loss is not automatically cancellation.
**Hypothesis A:** A network intermediary closed an idle connection.
**Hypothesis B:** The server failed, expired the session, or completed without delivering the response.
**Check:** Correlate request ID, connection events and server-side completion telemetry.
**Check:** Determine whether the server implements resumable events for that stream.
**Expected distinction:** A completed server operation with an interrupted stream differs from an unstarted request.
**Immediate action:** Avoid inferring cancellation from disconnection.
**Durable action:** Configure suitable timeouts and supported resumption behavior.
**Verification:** Drop the stream in a test and observe recovery without duplicating a side effect.
**Risk:** A replayed stream event must not itself trigger an unsafe repeated application action.
**Escalation:** Use the uncertain-outcome procedure if the operation had side effects.

#### D-05 — Cancellation does not stop a job

**Signal:** Work continues after a client has requested cancellation.
**Boundary:** Cancellation is a protocol signal, not a transactional rollback guarantee.
**Hypothesis A:** The executor had already committed before receiving cancellation.
**Hypothesis B:** The server failed to propagate cancellation to the downstream worker.
**Check:** Compare cancellation receipt, dispatch and commit timestamps in the same operation trace.
**Check:** Review whether the downstream operation supports cancellation and at which stages.
**Expected distinction:** Commit preceding cancellation is different from a dropped cancellation message.
**Immediate action:** Reconcile the actual result and prevent an unreviewed duplicate.
**Durable action:** Document cancellation points and propagate cancellation where the executor supports it.
**Verification:** Inject cancellation before dispatch, during work and after commit in separate cases.
**Risk:** Interrupting halfway through a non-transactional operation may leave partial state.
**Escalation:** Route partial state to a documented compensating or manual procedure.

#### D-06 — Tool discovery changes during a session

**Signal:** A previously listed tool disappears or its schema changes.
**Boundary:** Cached discovery is a snapshot, not an immutable authorization contract.
**Hypothesis A:** Server deployment changed the advertised tool.
**Hypothesis B:** The authenticated principal or policy scope changed.
**Check:** Compare server version, tool-list responses and principal under both observations.
**Check:** Inspect whether list-change notification support was negotiated.
**Expected distinction:** A change for all principals suggests deployment; a change for one suggests policy or identity.
**Immediate action:** Stop using a stale schema for the affected tool.
**Durable action:** Refresh discovery according to supported notifications and cache policy.
**Verification:** Test removal, addition and schema change against a controlled server.
**Risk:** Discovery refresh alone does not invalidate an already-approved action safely.
**Escalation:** Reapprove actions whose semantics or target changed.

#### D-07 — Concurrent replies appear out of order

**Signal:** The second request returns before the first.
**Boundary:** Response correlation is by JSON-RPC ID, not arrival position.
**Hypothesis A:** Downstream work durations differ.
**Hypothesis B:** The client matches replies using a FIFO queue instead of IDs.
**Check:** Record request IDs and corresponding response IDs, excluding sensitive arguments.
**Check:** Force different test durations and confirm the correct caller receives each result.
**Expected distinction:** Out-of-order replies with correct ID mapping are not necessarily a failure.
**Immediate action:** Stop processing results if correlation cannot be established.
**Durable action:** Correct correlation and isolate outstanding request state.
**Verification:** Exercise concurrent requests with intentionally reversed completion order.
**Risk:** Reusing an outstanding request ID can misassociate results.
**Escalation:** Investigate any result delivered across tenant or user contexts.

#### D-08 — Initialization succeeds but authorization fails

**Signal:** A tool is listed but invocation is denied.
**Boundary:** Discovery and policy evaluation are different operations.
**Hypothesis A:** The principal lacks the required resource permission.
**Hypothesis B:** The token is expired, scoped for another audience, or not forwarded correctly.
**Check:** Inspect the authenticated principal, token validation result and resource-policy decision.
**Check:** Compare the requested action and resource with the approved scope.
**Expected distinction:** An explicit deny differs from failure to establish an identity.
**Immediate action:** Do not replace user credentials with a broad service credential to make the call pass.
**Durable action:** Request appropriate access or correct the token audience and delegation setup.
**Verification:** Test both authorized and unauthorized identities for the same target.
**Risk:** A broad fallback identity can bypass tenant separation.
**Escalation:** Escalate unexplained access grants, not just denials.

#### D-09 — Resource URI points outside approved scope

**Signal:** A resource URI resolves to content outside the intended repository or tenant.
**Boundary:** Identifier syntax is not proof of ownership or safe resolution.
**Hypothesis A:** Path traversal or symbolic links bypassed a prefix-based check.
**Hypothesis B:** An indirect identifier resolved under the wrong tenant context.
**Check:** Resolve and authorize the canonical target at the executor's actual access boundary.
**Check:** Compare authenticated tenant, target owner and approved root.
**Expected distinction:** A path containing benign-looking segments can still resolve outside scope.
**Immediate action:** Deny access and preserve a sanitized audit record.
**Durable action:** Use resource-aware authorization and safe path-resolution primitives.
**Verification:** Test traversal, symlinks, encoded separators and cross-tenant identifiers.
**Risk:** A check performed before resolution may be invalidated by a later filesystem change.
**Escalation:** Treat confirmed unauthorized reads as a potential data exposure.

#### D-10 — Tool error is confused with protocol failure

**Signal:** The model receives a response but the intended action did not succeed.
**Boundary:** Transport status, JSON-RPC error and tool-level result are distinct layers.
**Hypothesis A:** The server returned a valid protocol response containing an operation error.
**Hypothesis B:** The HTTP or JSON-RPC exchange itself failed.
**Check:** Inspect transport status, JSON-RPC envelope and tool result independently.
**Check:** Compare any downstream status with the application's success criterion.
**Expected distinction:** HTTP success does not establish tool or business success.
**Immediate action:** Report the relevant failure without claiming completion.
**Durable action:** Make error categories actionable without exposing sensitive internals.
**Verification:** Exercise malformed input, authorization denial and downstream rejection separately.
**Risk:** Automatically retrying every error may repeat unsafe operations.
**Escalation:** Reconcile if a business effect remains ambiguous.

#### D-11 — Duplicate key carries different payload

**Signal:** A retried write returns the result of a different requested change.
**Boundary:** A deduplication key must be bound to the intended operation, not just a caller-supplied string.
**Hypothesis A:** A client accidentally reused a key for another payload.
**Hypothesis B:** The executor did not store or compare a canonical payload fingerprint.
**Check:** Compare the stored principal, target, operation and canonical payload with the retry.
**Check:** Inspect the deduplication store's atomic conflict behavior.
**Expected distinction:** Same key and same payload is a replay; same key and different payload is a conflict.
**Immediate action:** Reject the conflicting operation without performing either a new write or returning a misleading result.
**Durable action:** Bind each key to a canonical operation and define explicit conflict semantics.
**Verification:** Submit the same key concurrently with identical and differing payloads.
**Risk:** Incorrect canonicalization may treat semantically different requests as identical.
**Escalation:** Reconcile all effects if a conflict was previously accepted.

#### D-12 — Deduplication window expires

**Signal:** An old request replay creates a second business effect.
**Boundary:** An idempotency record provides protection only while it is retained and enforced.
**Hypothesis A:** The replay arrived after the key's retention period.
**Hypothesis B:** A regional failover lost the deduplication record.
**Check:** Compare original execution time, replay time and record retention policy.
**Check:** Verify replication and durability assumptions for the deduplication store.
**Expected distinction:** Expiry and data loss require different corrections.
**Immediate action:** Stop replay and identify the affected business operations.
**Durable action:** Align retention with the maximum replay and recovery horizon.
**Verification:** Test a replay near expiry and during a controlled failover.
**Risk:** Longer retention increases storage and privacy obligations.
**Escalation:** Engage the business owner if duplicate effects require compensation.

#### D-13 — Cross-user session reuse

**Signal:** A session initialized by one principal is accepted under another.
**Boundary:** A session ID correlates interactions but is not a standalone authorization token.
**Hypothesis A:** Session state is keyed only by the identifier.
**Hypothesis B:** A proxy or client pool accidentally shares headers across users.
**Check:** Compare authenticated identity at initialization and each subsequent request.
**Check:** Inspect connection and header reuse across isolated test accounts.
**Expected distinction:** A shared client pool can expose state even if the server's access checks are correct.
**Immediate action:** Terminate affected sessions and restrict access.
**Durable action:** Bind session state to authenticated identity and prevent unsafe header reuse.
**Verification:** Replay one user's session identifier under another identity.
**Risk:** Logging session identifiers for debugging may itself expose sensitive material.
**Escalation:** Investigate potential data exposure and credential compromise.

#### D-14 — Wrong OAuth token audience

**Signal:** A server accepts a token issued for a downstream API and forwards it unchanged.
**Boundary:** Each resource server must validate the token intended for itself.
**Hypothesis A:** The server treats any cryptographically valid bearer token as sufficient.
**Hypothesis B:** Client and server disagree about the intended resource or audience.
**Check:** Inspect issuer, audience, expiry and scope using a safe token-validation path.
**Check:** Compare the MCP server's accepted token with its downstream credential flow.
**Expected distinction:** A valid signature does not establish a valid audience.
**Immediate action:** Reject wrongly scoped tokens and disable token passthrough.
**Durable action:** Use appropriately issued tokens and separately authorized downstream access.
**Verification:** Submit tokens with valid signatures but incorrect audiences and scopes.
**Risk:** Reusing a broad downstream token can bypass policy and auditing.
**Escalation:** Rotate exposed credentials and investigate misuse.

#### D-15 — Confused deputy in an authorization proxy

**Signal:** An unfamiliar client gains access following a user's earlier third-party consent.
**Boundary:** Consent to one proxy's upstream identity is not necessarily consent to every MCP client.
**Hypothesis A:** The proxy failed to obtain per-client consent.
**Hypothesis B:** Redirect URI validation or state binding was insufficient.
**Check:** Trace client registration, redirect URI, user consent and authorization callback.
**Check:** Compare the consenting client with the client that receives the authorization code.
**Expected distinction:** A remembered upstream consent cookie does not prove fresh consent to a new MCP client.
**Immediate action:** Revoke affected grants and stop the vulnerable authorization path.
**Durable action:** Apply per-client consent, strict redirect validation and secure state handling.
**Verification:** Attempt the documented malicious-client flow in an isolated test environment.
**Risk:** Revoking grants can interrupt legitimate access; coordinate recovery with users.
**Escalation:** Use the security incident procedure for confirmed code or token exposure.

#### D-16 — Tool description changes maliciously

**Signal:** The same named tool suddenly asks the model to disclose secrets in its description.
**Boundary:** Tool metadata is external input and must not override host policy.
**Hypothesis A:** The server deployment was compromised.
**Hypothesis B:** A legitimate update added unsafe or misleading instructions.
**Check:** Compare the current metadata against an approved baseline and deployment history.
**Check:** Inspect the server binary provenance and authorized change records.
**Expected distinction:** A metadata change does not itself prove malicious execution, but warrants investigation.
**Immediate action:** Disable or quarantine the affected integration.
**Durable action:** Review server provenance, restrict tool authority and monitor semantic changes.
**Verification:** Run an adversarial metadata fixture and confirm secrets remain inaccessible.
**Risk:** Hash pinning detects byte changes but does not prove unchanged code is trustworthy.
**Escalation:** Investigate any prior unauthorized tool invocations.

#### D-17 — Cache leaks another principal's resource

**Signal:** Two users receive the same cached response despite different access scopes.
**Boundary:** A shared cache is part of the authorization data path.
**Hypothesis A:** The cache key omits principal, tenant or policy version.
**Hypothesis B:** Authorization is checked only when the cache entry is created.
**Check:** Compare cache keys and policy decisions across the two users.
**Check:** Test access after a permission change or revocation.
**Expected distinction:** An identical public resource can be shared; a protected resource needs scope-aware enforcement.
**Immediate action:** Disable or purge the affected cache.
**Durable action:** Include appropriate authorization context in keys and revalidate access.
**Verification:** Run cross-tenant and permission-revocation tests with warm caches.
**Risk:** Overly granular keys can reduce cache efficiency and increase memory use.
**Escalation:** Treat confirmed protected-content disclosure as a security incident.

#### D-18 — Trace logs contain secrets

**Signal:** Tokens or sensitive tool arguments appear in centralized diagnostics.
**Boundary:** Observability systems are separate data stores with their own access and retention.
**Hypothesis A:** Full request headers were logged by a proxy.
**Hypothesis B:** Tool arguments or results were logged without field-level redaction.
**Check:** Inspect configured log fields and a sanitized representative event.
**Check:** Identify all copies, consumers and retention policies.
**Expected distinction:** A correlation ID can often support tracing without storing a bearer token.
**Immediate action:** Restrict access, revoke exposed tokens and stop additional logging.
**Durable action:** Implement allowlisted fields, redaction and retention controls.
**Verification:** Send synthetic secrets and confirm they do not enter logs, traces or error reports.
**Risk:** Deleting one log copy may not remove exports or backups.
**Escalation:** Apply the organization's exposure and notification procedures.

#### D-19 — Unbounded resource content

**Signal:** A resource read causes memory pressure or exhausts a model context budget.
**Boundary:** Valid protocol content can still exceed application resource budgets.
**Hypothesis A:** The server returned an unexpectedly large document.
**Hypothesis B:** The host repeatedly loads duplicate or irrelevant resources.
**Check:** Measure returned bytes, decoded size and context allocation per request.
**Check:** Compare repeated resource identifiers and actual task requirements.
**Expected distinction:** One oversized object requires different mitigation from repeated redundant reads.
**Immediate action:** Stop or bound ingestion before further allocations.
**Durable action:** Apply size limits, pagination where supported and relevance-aware selection.
**Verification:** Test maximum-size, oversized and repeated-resource inputs.
**Risk:** Truncation may omit essential qualifiers; label incomplete context explicitly.
**Escalation:** Ask the data owner for a narrower authoritative resource when needed.

#### D-20 — Stale approval after policy change

**Signal:** An action approved under a previous policy is dispatched after access was revoked.
**Boundary:** Approval and authorization are time-dependent and separately enforced.
**Hypothesis A:** The executor does not recheck current policy.
**Hypothesis B:** The approval cache outlives its intended expiry.
**Check:** Compare approval time, policy version, revocation time and dispatch time.
**Check:** Inspect which principal and resource were evaluated at the final enforcement point.
**Expected distinction:** A valid historical approval can coexist with a current policy denial.
**Immediate action:** Suspend dispatch and revoke stale approvals where supported.
**Durable action:** Recheck current authorization and bind approval to an expiry and action.
**Verification:** Approve an action, revoke access, then attempt dispatch in a controlled test.
**Risk:** Reauthorization can delay time-critical operations; define a safe escalation path.
**Escalation:** Route exceptional access requests to the designated authority.

### Boundary-focused review of the complete request path

The following review questions describe proposed checks, not executed tests.
They give the operator an end-to-end method for separating protocol defects from application and executor defects.

#### A. Before starting a local server

1. Identify the executable path and how it was installed.
2. Identify the human or automation principal that approved that executable.
3. Record the expected binary version and software provenance.
4. Determine whether the executable can modify files or launch child processes.
5. Inspect inherited environment variables for credentials and unnecessary authority.
6. Pass only the environment variables that are required for the integration.
7. Specify the working directory explicitly rather than inheriting a sensitive directory accidentally.
8. Decide whether a dedicated operating-system identity is necessary.
9. Restrict filesystem access independently of any model-facing tool description.
10. Restrict outbound network access independently of the declared tool schema.
11. Limit process resources where the operating system supports effective controls.
12. Define how the parent detects an unexpected child exit.
13. Separate stdout protocol traffic from stderr diagnostics.
14. Define how stderr is redacted before aggregation.
15. Determine how shutdown handles child processes that ignore a closed stdin.
16. Confirm that restarting a process does not silently replay a side effect.
17. Decide which data may remain on disk after process exit.
18. Record how the binary and its dependencies will receive security updates.
19. Distinguish server process health from the health of each downstream service.
20. Document the owner who can stop the integration during an incident.

#### B. Before exposing a remote endpoint

1. Identify the endpoint's public hostname and network exposure.
2. Define which clients and users may connect to the service.
3. Protect credentials in transit using an appropriate authenticated transport.
4. Validate the expected server identity in the client.
5. Define where authentication terminates when a reverse proxy is present.
6. Ensure the backend cannot be reached through an unauthenticated alternate route.
7. Define how authenticated identity is conveyed from proxy to server.
8. Prevent an external caller from forging trusted identity headers.
9. Validate Origin on connections according to the pinned transport specification.
10. Avoid binding a local-only endpoint to all interfaces without an explicit need.
11. Apply network policy in addition to application authentication.
12. Document token issuance, intended audience, expiry and revocation behavior.
13. Decide how separate tenants obtain separate authority scopes.
14. Keep the MCP-facing token distinct from any downstream service token.
15. Define rate limits by principal and operation, not merely by source IP.
16. Decide which requests can safely be retried at each proxy layer.
17. Ensure intermediary timeouts are visible to the operator.
18. Identify whether intermediaries buffer or terminate event streams.
19. Define how deployment rollouts affect in-flight sessions.
20. Record the security owner for endpoint and credential changes.

#### C. Initializing a session

1. Record the version proposed by the client.
2. Record the version returned by the server.
3. Reject or disconnect when the selected version is unsupported by the client.
4. Capture client and server implementation identifiers without recording credentials.
5. Compare advertised capabilities with the host's approved capability policy.
6. Do not assume support for optional features that were not negotiated.
7. Send the initialized notification after a successful initialize response.
8. Respect the lifecycle's documented exceptions to pre-initialization messaging.
9. If the server assigns an HTTP session ID, carry it in subsequent requests as specified.
10. Protect the session ID against disclosure in traces and exception messages.
11. Associate session state with the authenticated principal.
12. Send the negotiated protocol version header on subsequent HTTP requests.
13. Do not treat a successful handshake as evidence that downstream credentials are valid.
14. Do not treat a session ID as a durable operation identifier.
15. Record whether resource subscriptions and list-change notifications are supported.
16. Record whether server-initiated requests are expected in the deployment.
17. Configure request timeouts with a finite maximum even where progress resets a timer.
18. Decide how the application handles an unexpected connection loss during initialization.
19. Keep compatibility behavior specific to the negotiated version.
20. Re-run discovery after a new session when prior session state is no longer valid.

#### D. Discovering and selecting a tool

1. Authenticate the server before consuming its advertised tool list.
2. Retrieve all pages needed to understand the available set.
3. Track list freshness and whether change notifications are supported.
4. Review the name, input schema and intended side-effect class.
5. Treat annotations and descriptions as claims requiring policy interpretation.
6. Identify the actual executor and downstream identity for each tool.
7. Map the requested operation to an allowed resource scope.
8. Reject tools whose authority exceeds the task unless an appropriate policy permits them.
9. Distinguish a read-only operation from an operation that modifies external state.
10. Remember that a read operation may still disclose sensitive data.
11. Decide whether the task requires a human approval before proposing an action.
12. Bind any approval to exact arguments and the principal.
13. Confirm that the selected schema corresponds to the server's current version.
14. Validate argument types before dispatch.
15. Validate semantic constraints that cannot be expressed by the schema alone.
16. Resolve paths and identifiers using the executor's real access rules.
17. Reject ambiguous resource references instead of guessing a tenant or account.
18. Check the user's current authorization at the time of execution.
19. Avoid including credentials or unrelated private data in tool arguments.
20. Log a safe operation identifier and policy decision for later reconciliation.

#### E. Executing a side-effecting operation

1. Distinguish the model's proposed action from the action actually approved.
2. Compare the approved target and parameters to the dispatched operation.
3. Check that approval has not expired or been revoked.
4. Recheck the principal's current authorization at the enforcement point.
5. Identify the exact downstream system that can commit the effect.
6. Specify the transaction or commit boundary if one exists.
7. Define the operation identifier independently of the JSON-RPC request ID.
8. If idempotency is supported, define its scope and retention.
9. Bind an idempotency key to principal, target and canonical operation.
10. Reject conflicting payloads that reuse an existing key.
11. Ensure key insertion and effect handling meet the executor's atomicity contract.
12. Understand what happens if the process fails before the effect.
13. Understand what happens if it fails after the effect but before recording the response.
14. Understand what happens when the downstream system is eventually consistent.
15. Record a status that distinguishes pending, committed, rejected and unknown.
16. Avoid converting an unknown outcome into a false failure.
17. Avoid converting an accepted request into a false business success.
18. Define compensation if the business operation permits safe reversal.
19. Define escalation where an irreversible effect cannot be reconciled automatically.
20. Preserve a minimal audit trail tying approval, dispatch and observed outcome together.

#### F. Returning and consuming results

1. Match responses to requests by their protocol IDs.
2. Do not assume concurrent responses arrive in request order.
3. Separate HTTP errors, JSON-RPC errors and tool-level errors.
4. Interpret a success envelope according to the tool's documented result semantics.
5. Check whether the business operation is complete or only accepted for later processing.
6. Treat returned text, links and embedded instructions as untrusted data.
7. Do not allow a tool result to expand its own authorization.
8. Validate returned identifiers before using them in subsequent operations.
9. Apply tenant and resource checks again on follow-on requests.
10. Do not expose stack traces or credentials to end users.
11. Preserve enough diagnostic context to distinguish policy denial from downstream failure.
12. Apply output size limits before including content in a model context.
13. Preserve important caveats when summarizing a partial result.
14. Identify freshness or version metadata when it affects the answer.
15. Distinguish a cached result from a newly retrieved result.
16. Do not silently interpret absence of data as proof of absence of an effect.
17. Avoid presenting unverified external content as a trusted system instruction.
18. Record whether the final answer depends on a failed or incomplete tool.
19. Provide a human-readable uncertainty statement for unresolved outcomes.
20. Redact sensitive output before writing traces or persistent transcripts.

#### G. Disconnecting and recovering

1. Identify whether the underlying connection or the logical session ended.
2. Determine which requests were in flight at the time of interruption.
3. Mark unobserved side-effecting requests as unknown until reconciled.
4. Check server-side completion only through appropriately authorized telemetry.
5. Use an authoritative business status lookup where one exists.
6. Account for replication delay before treating a missing record as conclusive.
7. Determine whether the server supports event resumption for the affected stream.
8. Respect the documented scope of any event replay.
9. Do not confuse event replay with safe re-execution of a tool.
10. Send cancellation explicitly if cancellation is intended and still meaningful.
11. Do not treat cancellation as proof of rollback.
12. Start a new initialization when the specified expired-session response requires it.
13. Authenticate and authorize the new session independently.
14. Refresh capability discovery rather than assuming old session state survived.
15. Reconcile pending writes before submitting replacement actions.
16. Reuse an idempotency key only under its documented binding and retention contract.
17. Back off safely if many clients are reconnecting simultaneously.
18. Bound reconnection attempts to protect a recovering dependency.
19. Escalate irreversible or ambiguous operations to the business owner.
20. Verify that the recovered application state matches downstream state.

#### H. Reviewing observable evidence

1. Count logical user operations separately from transport attempts.
2. Record the number of observed responses separately from successful effects.
3. Distinguish client timeout rate from confirmed downstream failure rate.
4. Group diagnostics by server version and negotiated protocol version.
5. Separate authentication failures from authorization denials.
6. Record tool name and operation class without recording unnecessary arguments.
7. Measure discovery failures independently from invocation failures.
8. Capture downstream dependency saturation where the operator has access.
9. Track retry attempts and retry exhaustion per logical operation.
10. Track unresolved unknown outcomes as an operational backlog.
11. Measure reconciliation delay from first uncertainty to authoritative resolution.
12. Record the age of pending approvals and their expiry behavior.
13. Measure queue time separately from executor service time.
14. Avoid inferring end-to-end percentile latency by summing component percentiles.
15. Partition error rates by tenant only when privacy and volume permit meaningful analysis.
16. Record the observation window and denominator for every reported rate.
17. Document sampling and missing telemetry before comparing periods.
18. Separate deliberately injected test faults from actual incidents.
19. Report whether a claimed result was observed, reproduced or only hypothesized.
20. Retain raw evidence only for the period justified by policy and legal requirements.

#### I. Threat-model walk-through: repository search and controlled edit

1. A user requests a search of a repository and a proposed edit.
2. The host chooses a narrow search capability rather than unrestricted shell where feasible.
3. The server authenticates the requesting principal.
4. The authorization layer identifies the user's allowed repository and branch.
5. The search executor enforces those resource boundaries.
6. Search results can contain adversarial text committed by another contributor.
7. The host treats that text as evidence about repository content, not as policy.
8. The model proposes a patch but does not itself apply it.
9. The host presents the exact patch and target to the approval mechanism if required.
10. The approval binds the repository, branch, target revision and patch content.
11. Before writing, the executor checks that the target revision has not changed.
12. If the revision changed, the executor rejects or requires an explicit rebase and reapproval.
13. The write executor verifies current principal permissions again.
14. The operation records a durable identifier for reconciliation.
15. A network failure after the write creates an unknown client outcome.
16. The host queries the authoritative repository state before reapplying the patch.
17. A duplicate request with an enforceable idempotency key returns the prior outcome.
18. Without such a guarantee, the host asks an operator to resolve ambiguity.
19. The final response reports the observed state and any unresolved uncertainty.
20. No part of this example implies that MCP itself supplies a patch executor or approval system.

#### J. Version migration decision

1. Inventory every client and server version that participates in the integration.
2. Identify the currently pinned protocol date for each connection.
3. Review the target specification's lifecycle and transport changes.
4. Review the target version's authorization and security requirements separately.
5. Determine which optional capabilities each deployed client uses.
6. Check whether the selected SDK implements those capabilities in the target version.
7. Identify any old HTTP+SSE integration that requires a compatibility plan.
8. Avoid assuming that a 2024 transport endpoint implements the newer transport.
9. Define an explicit supported version intersection for the migration.
10. Test the unsupported-version path as well as the successful handshake.
11. Test authenticated requests through every production proxy.
12. Test session expiry, reconnection and in-flight request behavior.
13. Test tool discovery and schema compatibility under both allowed and denied identities.
14. Test cancellation and uncertain writes without using real production side effects.
15. Test output handling for malformed and adversarial content.
16. Monitor version-specific error categories during a controlled rollout.
17. Define rollback criteria before enabling the new integration.
18. Preserve operation reconciliation across rollback and restart.
19. Update the source ledger and operational runbook to the adopted version.
20. Do not mark the migration safe merely because initialization succeeds.

#### K. Security ownership and escalation

1. The application owner defines which user tasks are permitted.
2. The identity owner defines authentication and token lifecycle.
3. The server owner publishes capability and deployment changes.
4. The executor owner enforces resource-level access and effect semantics.
5. The downstream system owner defines authoritative business state.
6. The security owner reviews trust boundaries and abuse cases.
7. The privacy owner defines retention and handling of sensitive content.
8. The incident commander coordinates containment when a boundary fails.
9. The reviewer should know who may disable a tool without disabling unrelated services.
10. The reviewer should know how to revoke a compromised session or credential.
11. The reviewer should know how to identify other sessions sharing the same authority.
12. The reviewer should know whether an action is reversible.
13. The reviewer should know where approval evidence is stored.
14. The reviewer should know where executor audit records are stored.
15. The reviewer should know how long idempotency records survive.
16. The reviewer should know which telemetry may contain personal data.
17. The reviewer should know whether downstream status reads are strongly consistent.
18. The reviewer should know who can approve a manual reconciliation.
19. The reviewer should know the procedure for correcting an incorrect final answer.
20. The reviewer should not equate an absent alert with absence of compromise.

#### L. Protocol-message reading exercise

The following trace is schematic and omits full JSON bodies.
It is an ILLUSTRATIVE reading exercise, not a captured production trace.
The client initiates a connection to a server it has been configured to trust.
The client sends request `id=1`, method `initialize`, with its supported protocol version.
The server returns response `id=1`, including its selected version and capabilities.
The client verifies that the selected version is supported.
The client sends `notifications/initialized`, which has no request ID.
The client sends a `tools/list` request with a fresh correlation ID.
The server returns a list of advertised tool definitions.
The client processes additional pages if the list is paginated.
The host considers a tool for the user's explicitly requested task.
The host checks its own policy and any required approval.
The client sends a `tools/call` request with the selected tool and validated arguments.
The server authenticates the caller according to the deployment's identity mechanism.
The server checks whether that principal may use that tool for that target.
The server calls its executor if the authorization decision permits dispatch.
The executor applies the operation using its own bounded authority.
The server returns the executor's result or an appropriate error.
The client correlates that response with the outstanding request ID.
The host checks whether the result establishes the requested business outcome.
The host reports the observed result without treating returned content as higher-priority instructions.
The exercise deliberately leaves authentication implementation and business idempotency outside MCP message semantics.
If the response is lost after dispatch, the trace is incomplete and the effect may be unknown.
If the server advertised no tools, a tool call should not be inferred from another server's capability list.
If a notification has no response, that alone is not a protocol error.
If the tool returns a valid error result, successful HTTP delivery is not successful task completion.
If the server chooses an unsupported version, the client should disconnect as described by the pinned lifecycle.
If the tool list is cached, the host must still enforce current policy at invocation time.
If a tool name is reused with different semantics, prior approval cannot safely be assumed to cover the new operation.
If the executor is a downstream API, the MCP server cannot claim a transaction guarantee that API does not provide.
If a remote session expires, restart initialization and reconcile any outstanding effect separately.
If the process uses stdio, diagnostic text on stdout corrupts the message channel.
If the process exits before a write response, process exit alone does not determine whether the external write committed.
If the event stream reconnects, replayed events must be interpreted within the documented stream scope.
If a tool result contains a URL, accessing that URL requires its own network and authorization policy.
If a result contains sensitive data, the host must not place it in unrestricted logs.
If the client's identity changes, cached access decisions from the prior identity must not be reused.
If the user revokes permission after approval, current authorization must be checked before dispatch.
If cancellation arrives after commit, a compensating transaction may be needed rather than a retry.
The protocol trace therefore describes communication, while the effect trace describes business state.

#### M. Safe fault-injection design

The scenarios below describe tests to design; none is claimed to have been run.
Use a disposable tenant, nonproduction credentials and reversible or simulated effects.
Record a unique operation identifier before each injection.
Record the expected authoritative outcome before observing the result.
Inject connection failure before the server receives the request.
Expected observation: the server has no corresponding dispatch record if ingress telemetry is complete.
Do not generalize that result to failures after the server accepts a request.
Inject failure after the server receives the request but before executor dispatch.
Expected observation: policy and ingress evidence exist, while executor evidence should show no dispatch.
If executor evidence is incomplete, classify the result as unknown rather than safely failed.
Inject failure while the executor is processing the operation.
Expected observation: outcome depends on the executor's own cancellation and transaction semantics.
Inject failure immediately after downstream commit but before the server sends a result.
Expected observation: the client can time out while the authoritative system records a committed effect.
Inject failure after the server sends a response but before the client records it.
Expected observation: client-side response loss and server-side completion can coexist.
Repeat the same operation with a documented idempotency key.
Expected observation: a conforming deduplication design returns or identifies the existing effect.
Repeat with the same key and a different payload.
Expected observation: the executor rejects the conflict instead of returning a misleading success.
Repeat after the advertised key retention period in an isolated environment.
Expected observation: behavior must match the documented retention boundary, not an assumed infinite guarantee.
Run two simultaneous attempts with the same key.
Expected observation: the atomicity contract determines whether only one effect is committed.
Interrupt the idempotency store between key registration and business commit.
Expected observation: recovery behavior must not claim a completed effect without authoritative evidence.
Inject a stale resource lookup before retrying a write.
Expected observation: the client should not treat a temporarily missing record as a definitive failure.
Inject a session 404 after a request has been sent.
Expected observation: a new session is initialized, but the previous operation still requires reconciliation.
Inject malformed tool output containing instruction-like content.
Expected observation: the host treats it as untrusted data and policy rejects unrelated privileged actions.
Inject a denied principal into an otherwise healthy session.
Expected observation: a valid session does not override the resource-level denial.
Inject a proxy header-stripping fault.
Expected observation: diagnostics distinguish client omission from intermediary mutation.
Inject a server restart while a stream is active.
Expected observation: recovery respects actual server support for event replay.
Record each result with protocol version, SDK version, deployment topology and injection point.
Record negative results and inconclusive instrumentation, not only successful checks.
State clearly whether the test establishes one observed execution or a broader formal guarantee.
Retain synthetic identifiers rather than real credentials in shared reports.
Escalate any test that unexpectedly reaches a real external effect.

#### N. Contract review questions for implementers

The contract owner should be able to answer each question before a write-capable integration is enabled.
Which principal initiates the request?
How is that principal authenticated at the server?
Which tenant and resource does the action target?
Where is the final authorization decision enforced?
Which arguments determine the business effect?
Which arguments are covered by approval?
How is the exact approved payload compared with the dispatched payload?
Which identity does the downstream service see?
Is that identity narrower than the MCP server's overall authority?
What is the durable business operation identifier?
Is an idempotency key supported by the component that commits the effect?
What happens when the key is reused with different parameters?
How long is the deduplication record retained?
Can a delayed event arrive after that retention period?
Can two concurrent retries race before the deduplication record is stored?
What happens if the deduplication store is unavailable?
What happens if the server crashes before dispatch?
What happens if it crashes after dispatch but before a response?
How is a committed effect distinguished from an accepted request?
Is the status lookup authoritative or eventually consistent?
How is an unknown outcome displayed to the user?
Who may resolve an ambiguous business operation?
What evidence is retained for that resolution?
Which data must be redacted from the audit record?
Can a session be reassigned to another user?
How are permissions rechecked after token expiry or revocation?
Which failures are safe to retry automatically?
How are retry attempts bounded across clients and proxies?
Can a server notification change the host's policy?
Can tool output cause a new privileged operation without independent authorization?
Does a resource URI resolve within the permitted tenant and path?
How are list changes propagated to clients with cached schemas?
What happens when the session expires while the action is running?
What happens if a cancellation request arrives after commit?
Can a compensating operation itself fail or require approval?
What recovery path exists if no safe compensation exists?
Who owns downstream capacity and rate limits?
Which metric distinguishes logical operations from protocol attempts?
Which protocol and SDK versions were actually tested?
Which assumptions remain open and require explicit risk acceptance?

**Playbook takeaway:** An MCP request is only as safe as the identities, policy checks, execution semantics and recovery evidence across its entire path.

## Section 9 — Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let `n` be the number of advertised tools, `b` the total serialized bytes, and `a` the number of attempts.
A client that consumes every listed tool must process at least the returned `b` bytes; parsing work is at least proportional to input read under a conventional sequential parser.
A simple linear scan over `n` tool names takes O(n) comparisons; an implementation-specific index can change lookup cost and add memory and maintenance overhead.
Pagination adds network round trips; total listing cost depends on page size, latency, and server behavior.
A request's end-to-end duration includes queueing, network, server policy and downstream work; protocol framing alone does not predict it.
A retry increases attempt count and can intensify contention at the downstream service.
Recovery can require durable operation-status lookups; a session restart is not a replacement for such a lookup.
Stored audit and idempotency state have retention, indexing, privacy and deletion costs.
These are operation-model statements, not measured performance claims.

| Operation | Size and model | Time / communication | Additional space / caveat |
|---|---|---|---|
| Parse tool listing | `b` UTF-8 bytes | At least O(b) input read for a full listing. | Representation and validation overhead are implementation dependent. |
| Linear name lookup | `n` tool names | O(n) comparisons worst case. | O(1) auxiliary space excluding stored listing. |
| Execute call | `a` attempts | Up to `a` requests plus downstream work. | Executor work and network latency dominate in many workloads but are not measured here. |
| Reconcile write | `k` candidate records | Depends on downstream index and consistency. | An absent record may be inconclusive in an eventually consistent store. |

### 9b. Back-of-Envelope Sizing

**Illustrative workload:** 100,000 logical read requests per day, each with independent per-attempt failure probability 0.10 and at most one retry.
**Formula:** Use M2, `E[A] = 1+p`, under the stated independence and one-retry assumptions.
**Expected attempts:** 100,000 × 1.10 = 110,000 attempts/day.
**Expected retry attempts:** 10,000/day; this is a mean, not a peak-rate bound.
**Illustrative price:** $0.001 per attempt, with no minimum fee, storage cost, taxes, labor or egress.
**Attempt charge:** 110,000 × $0.001 = $110/day.
**Without retry:** 100,000 × $0.001 = $100/day.
**Incremental expected attempt charge:** $10/day.
**Expected final successes under M2:** 99,000/day; attempt-only charge per expected success is approximately $0.001111.
**Interpretation:** This ratio divides expected charge by expected successes; it is not the expected value of a random per-success ratio.
**Sensitivity:** At a 0.20 first-attempt failure probability the expected attempt count becomes 120,000/day and attempt charge becomes $120/day.
**Operational budget:** Peak load, concurrency, timeout policy, retention and recovery need separate measurement; a daily mean does not size a burst.
**Environment:** No hardware, SDK version or network topology has been measured; price is invented only to illustrate the calculation.
**Cost limitation:** Pricing ignores downstream work, model tokens, operational staff, security controls and failed-outcome costs.

**Takeaway:** Estimate logical work and attempt amplification separately; measure the bottleneck and full economic cost before capacity decisions.

## Section 10 — Key Theorems / Results Reference

| Result | Statement and scope | Essential condition | Does not imply | Source |
|---|---|---|---|---|
| Initialization order | The client initiates initialization before normal operations and sends `notifications/initialized` after successful initialization. | MCP 2025-06-18 lifecycle; note the specification's stated ping/logging exceptions. | Authentication or authorization has been correctly implemented. | [Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle). |
| HTTP session expiry | A server may terminate a session; a client receiving 404 for its session starts a new initialization without that session ID. | Streamable HTTP session behavior in 2025-06-18. | Any old side effect did not commit. | [Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports). |
| Event resumption | A server may support SSE event IDs and replay on a resumed stream. | Optional server support and the documented stream restrictions. | Exactly-once business effects or guaranteed recovery. | [Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports). |
| Expected attempts | One unconditional attempt plus one attempt taken with probability p gives 1+p. | At most one retry; finite expected count. | Independent failures or safe repeated writes. | DERIVED in M2. |
| Independent final failure | Two independent failures, each probability p, have joint probability p². | Independence and identical marginal probabilities. | The formula holds during a common persistent outage. | DERIVED in M2. |

### Source ledger and version limits

1. [MCP lifecycle, 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle): initialization, version negotiation, capabilities, shutdown and timeout recommendations. Accessed for this draft; no independent editorial review is claimed.
2. [MCP transports, 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports): stdio framing, Streamable HTTP, session management and optional resumption. This replaces the older 2024-11-05 HTTP+SSE transport in the pinned version; compatibility requires explicit handling.
3. [MCP security best practices, 2025-11-25 documentation](https://modelcontextprotocol.io/docs/2025-11-25/tutorials/security/security_best_practices): confused-deputy and token-passthrough risks; later-version guidance, not silently treated as a 2025-06-18 normative requirement.
4. [Course source crosswalk](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-tools-mcp): records the designated teaching home and original source filenames, not proof that unavailable source text was reviewed.
5. [JSON-RPC 2.0 specification](https://www.jsonrpc.org/specification): background on request/response and notifications; MCP adds its own versioned lifecycle and transport requirements.

**Recheck trigger:** Changing protocol version, SDK major version, transport, authorization design, or server deployment requires reviewing the applicable specification and implementation behavior.
**Takeaway:** Versioned protocol requirements and derived reliability identities apply only within their stated scopes.

## Section 11 — Interview / Exam Questions

The following are constructed practice questions, not claims about any employer's interviews.
For each, score 0 for an incorrect answer, 1 for a partial answer, 2 for a correct answer with assumptions, and 3 for a defended tradeoff and relevant boundary.

### Level 1 — Definition / Recall

**Q1 — MCP client versus executor.** A host discovers a `delete_record` tool. Who actually deletes the record?
**Answer:** The server implementation or its delegated downstream executor performs the effect; MCP carries the request and result.
Discovery alone does not establish that the caller is authorized.
**Alternative:** A server can be its own executor, but it still needs an enforcement boundary.
**Follow-up:** If the model proposes deletion, the host and server must independently enforce applicable policy.
**Trap:** Saying that the protocol or model guarantees safe execution.
**Support:** Sections 2–3.

**Q2 — Session versus identity.** Is a session ID sufficient proof of a user identity?
**Answer:** No; authenticate the caller and bind session access to the authenticated principal and authorization context.
A session identifier correlates protocol interactions, not independent authority.
**Alternative:** A deployment may use a separate authenticated token alongside the session ID.
**Follow-up:** A stolen session ID must not authorize cross-user use.
**Trap:** Treating secrecy of the ID as the whole authorization model.
**Support:** Sections 3 and 8.

**Q3 — Tool versus resource.** How does a resource differ from a tool?
**Answer:** A resource supplies readable context; a tool is an invocable server capability.
A document read as a resource is not automatically an instruction to the host.
**Alternative:** A tool can also retrieve data, but its invocation and contract are distinct from a resource read.
**Follow-up:** Retrieved content remains untrusted regardless of which primitive delivered it.
**Trap:** Equating a resource URI with data integrity or permission.
**Support:** Section 3.

### Level 2 — Derivation / Apply

**Q4 — Retry calculation.** For 1,000 reads with p=0.1 and one independent retry, calculate expected attempts and final failures.
**Answer:** M2 gives 1,000 × 1.1 = 1,100 attempts and 1,000 × 0.1² = 10 final failures.
**Sanity check:** 900 first successes plus 90 retry successes gives 990 successes.
**Alternative:** Under perfectly correlated failures, expect 100 final failures.
**Follow-up:** The result does not determine peak concurrent load.
**Trap:** Counting retries as new logical requests or applying the calculation to unsafe writes.
**Support:** Sections 4–5.

**Q5 — Handshake trace.** Give a normal initialization sequence for the pinned version.
**Answer:** Client sends `initialize`; server replies with version, capabilities and implementation info; client sends `notifications/initialized`; normal negotiated operations follow.
The client checks whether it supports the server's selected version.
**Alternative:** Protocol-specified ping and logging exceptions have their stated timing rules.
**Follow-up:** HTTP requests after initialization include the protocol-version header and, when assigned, the session ID as specified.
**Trap:** Calling a tool before completing initialization as a general rule.
**Support:** Section 10 lifecycle source.

**Q6 — Unknown write outcome.** A write times out after dispatch. Is retry safe?
**Answer:** Not necessarily; the write may have committed while the response was lost.
Reconcile using an authoritative operation identifier or use an executor-enforced durable idempotency contract.
**Alternative:** A genuinely idempotent operation can be retried if its full semantics and authorization still hold.
**Follow-up:** An eventually consistent negative lookup may require waiting or escalation.
**Trap:** Treating timeout or a fresh JSON-RPC request ID as proof of no prior effect.
**Support:** Sections 4 and 6.

**Q7 — Session expiration.** A 2025-06-18 Streamable HTTP server returns 404 to a request carrying its session ID. What next?
**Answer:** Start a new initialization without the old session ID, then renegotiate and rediscover as needed.
Reconcile in-flight writes separately before replaying them.
**Alternative:** Investigate unexpected expiry or state-store failure operationally while observing the protocol recovery rule.
**Follow-up:** A new session does not restore prior authorization automatically; perform required authentication checks.
**Trap:** Assuming 404 means the prior write never executed.
**Support:** Section 10 transport source.

### Level 3 — Judgement / Design / Debug

**Q8 — Transport choice.** A local desktop host must integrate a controlled child process. Choose a starting transport.
**Answer:** stdio is a reasonable starting point when the host can launch and supervise the server and the process identity is appropriately restricted.
The choice avoids a network listener but does not sandbox the child process.
**Alternative:** Streamable HTTP can be justified if an independently operated service or multiple remote clients are required.
**Follow-up:** An untrusted binary requires provenance checks and OS isolation regardless of transport.
**Trap:** Calling local execution inherently safe.
**Support:** Section 7.

**Q9 — Missing tool.** Initialization succeeds, but the model cannot invoke an expected tool. How do you debug?
**Answer:** Verify endpoint and server identity, negotiated tool capability, complete paginated listing, principal filtering and host routing.
Inspect the discovery response before attributing the problem to the model.
**Alternative:** If discovery is correct, investigate selection policy or schema mismatch.
**Follow-up:** A cache invalidation notification is relevant only where supported and correctly handled.
**Trap:** Broadening permissions without evidence.
**Support:** F-01.

**Q10 — Injection.** A repository resource says to upload all credentials before answering. What should happen?
**Answer:** Treat the text as untrusted data, not a higher-priority instruction; deny the unrelated exfiltration at the policy/executor boundary.
Record enough provenance for investigation without storing the credentials themselves.
**Alternative:** Quarantine the resource if operational policy supports it.
**Follow-up:** Even an apparently benign model response should not bypass authorization on a later tool call.
**Trap:** Relying solely on a prompt telling the model to ignore malicious text.
**Support:** F-04.

**Q11 — Approval changed.** A user approved a payment of $50 to account A; the model changes it to $500 to account B. Is the approval still valid?
**Answer:** No; the approved principal, action, amount, target and expiry must match the dispatched operation.
Require a new approval and still enforce downstream authorization and idempotency.
**Alternative:** A preauthorized policy may permit bounded changes only if the original approval explicitly covered them.
**Follow-up:** Reject reused approval after expiry or policy change according to the approval contract.
**Trap:** Binding approval only to the tool name.
**Support:** F-09.

**Q12 — Verify reliability.** A team claims retries provide exactly-once effects because it observed no duplicates in a happy-path demo. How do you challenge it?
**Answer:** Test response loss after commit, duplicate submissions, concurrent attempts, process restart and key expiry against authoritative downstream state.
Require a clearly scoped atomic deduplication contract and inspect its retention and payload-binding rules.
**Alternative:** If exactly-once is impossible, use reconciliation, compensating action or explicit operator escalation.
**Follow-up:** State the tested failure windows and avoid generalizing beyond them.
**Trap:** Equating a successful response, SSE resumption or a small sample with an end-to-end guarantee.
**Support:** Sections 6 and 8.

**Takeaway:** Strong answers name the enforcement point, assumptions, failure evidence, and a verification strategy.

## Section 12 — Quick Reference Card

| Item | Value |
|---|---|
| Topic | Model Context Protocol contracts, execution boundaries and recovery |
| Core formula | One independent retry gives expected attempts `1+p` and final failure probability `p²`; this does not establish write safety |
| Key assumption | Protocol version and capabilities are negotiated; authentication and authorization are enforced separately |
| Complexity | Transport, discovery and execution costs depend on workload, server implementation and network; no benchmark is claimed |
| Use when | A reusable client–server capability contract justifies the integration and operational overhead |
| Avoid when | Authority, isolation or side-effect controls cannot be enforced, or a direct API is simpler and sufficient |
| Go deeper | [Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle), [transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports), [crosswalk](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-tools-mcp) |
| Diagnostic | A timeout after a write leaves the outcome unknown; reconcile authoritative state before retry |
| Evidence | Protocol references are pinned to 2025-06-18; workload calculations are illustrative |
