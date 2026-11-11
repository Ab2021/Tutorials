---
title: "Tool Schema and Side-Effect Boundaries"
topic: "Typed tool contracts, admission control, idempotency keys, receipts, and compensation limits"
collection: "tools-mcp-protocols-and-enterprise-integrations"
level: "core"
evidence: "DERIVED"
created: "2026-11-09"
last_reviewed: null
content_status: "draft"
file_id: "02"
version: "2.0"
notation_style: "plain-text fences"
prerequisites: ["01_model_context_protocol.md", "JSON Schema basics", "HTTP idempotency", "RBAC/ABAC"]
source_status: "OPEN - source bodies unavailable; visible titles define scope only"
owner_role: "curriculum maintainer"
source_index: "INDEX.md"
review_trigger: "protocol revision, executor change, authorization model change, or downstream API contract change"
---

# Tool Schema and Side-Effect Boundaries

## 1. File Header

**Scope.**
A tool is a capability contract with three separable layers: a typed interface (what arguments are
legal), an authority contract (who may cause which effect on which resource), and an effect contract
(what the outside world is allowed to observe after the call).
This sheet treats those layers as independent and shows where each one fails.
Discovery is a description of the interface; it never grants authority and never proves an outcome.

**Outcome for the reader.**
Given a trace of an agent call, the reader should be able to classify the effect, decide whether the
call is admissible, compute the duplicate-effect risk of a retry, size the recovery lane, and choose
between reconciliation, compensation, and escalation.
The sheet also supplies the numbers needed to defend a retention window and an approval lifetime.

**Evidence legend used throughout.**

| Marker | Meaning in this file |
|---|---|
| DERIVED | Follows from stated preconditions by algebra or state-machine reasoning |
| VERIFIED | Arithmetic re-checked line by line in this document |
| ILLUSTRATIVE | A constructed workload, not an observed deployment |
| HEURISTIC | A rule of thumb with a stated failure regime |
| OPEN | Contested or dependent on environment; no single answer |

No number in this sheet is a measured production result.
Where a rate, latency, or cost appears it is an ILLUSTRATIVE input chosen to make the arithmetic
visible; the formulas are the transferable part, not the constants.

**Notation and units.**

| Symbol | Meaning | Units | Shape / range |
|---|---|---|---|
| `p` | principal invoking the tool | identity record | must resolve to one tenant |
| `r` | resolved resource set | set of record identifiers | may be empty |
| `t` | authorization evaluation time | seconds, epoch | monotonic per executor |
| `x` | typed argument object | JSON object | validated against pinned schema |
| `s` | business state read at admission | snapshot | read time recorded |
| `k` | operation key | opaque string, 16-64 bytes | stable across retries |
| `h` | canonical argument digest | 32-byte hex | must cover effect fields |
| `lambda` | arrival rate of distinct intents | intents/second | > 0 |
| `mu` | execution completion rate | intents/second | > 0 |
| `rho` | retry probability per attempt | dimensionless | [0,1) |
| `f` | fanout: records touched per intent | records/intent | >= 1 |
| `B` | backlog | intents | >= 0 |

**Prerequisites the reader is assumed to hold.**
Reading a JSON Schema and knowing which keywords validate versus which merely annotate.
Knowing that HTTP `PUT` and `DELETE` are defined idempotent while `POST` is not.
Knowing the difference between authentication (who) and authorization (may this principal do this).
Knowing that a queue is a buffer with a finite service rate, not a guarantee of eventual delivery.

**Deliberate exclusions.**
Model-side prompt engineering, retrieval quality, and evaluation of answer correctness are out of
scope and belong to the neighbouring sheets in this unit.
Provider-specific billing tiers are out of scope; §9 gives a cost *model* with pluggable unit prices.
This sheet does not assert that any particular vendor implements the controls it recommends.

**How to read it under time pressure.**
Sections 3 and 12 are the lookup surface; sections 4 and 10 carry the arguments; section 8 is the
diagnostic catalogue to consult when a trace looks wrong but no exception was raised.
Section 5 is the one fully worked end-to-end numeric pass, and it re-checks its own arithmetic.

**Takeaway.**
A syntactically valid call is a proposal, not an effect.
Safety comes from three independent checks - typed shape, enforced authority, and a durable ledger
that binds one intended effect to one key - and from an outcome record that survives a lost response.

## 2. Concept Map / Mental Model

**The five planes.**
Every tool invocation crosses five planes, and each plane can fail without the others noticing.

```text
 P1  proposal      model emits tool name + arguments inside a context window
      |            untrusted text; may be wrong, stale, injected, or malformed
      v
 P2  admission     host validates schema, resolves resources, evaluates policy,
      |            and claims the operation key in a durable ledger
      v
 P3  dispatch      executor authenticates to the downstream system with its own
      |            credential, re-checks scope, and applies a deadline
      v
 P4  outcome       downstream commits, rejects, or leaves the result unknowable
      |            to the caller (lost response, timeout, partial batch)
      v
 P5  record        receipt persists; the caller reconciles, compensates, or
                   escalates; audit gets a redacted terminal record
```

A failure at P1 is a prompt problem.
A failure at P2 is a contract problem: the schema admitted something the policy would have refused,
or the policy was evaluated against a stale snapshot.
A failure at P3 is a credential or scope problem, usually a confused deputy.
A failure at P4 is an evidence problem and it is *normal*: distributed calls lose responses.
A failure at P5 is an accounting problem, and it is the one that turns a transient error into a
duplicate payment, a double booking, or a silently abandoned refund.

**Why the model cannot be the control point.**
The model proposes arguments in natural-language context, which means the arguments are derived from
text that may include retrieved documents, web pages, or another tool's output.
That makes the proposal plane adversarial by construction: content that reaches the context can
influence the arguments.
Anything the model can assert - "this is a read-only call", "the user approved it", "the amount is
small" - must therefore be re-established from data the model does not control.

**The three registries.**
A workable implementation keeps three separate stores, because they change on different clocks.

| Registry | Holds | Changes when | Failure if missing |
|---|---|---|---|
| Schema registry | pinned input/output schemas per tool version | a tool version ships | silent argument drift; old clients break |
| Policy store | rules binding principal, action, resource, and conditions | access review or regulation changes | stale grants, cross-tenant reads |
| Operation ledger | key, digest, state, receipt reference, timestamps | every admitted call | duplicate effects on retry |

Collapsing these three into one table is the most common architectural shortcut and the most common
source of "the retry looked new to us" incidents: schema version bumps then masquerade as new
operations, and a policy rollback silently re-permits revoked principals.

**The effect ladder.**
Classify every tool on a six-rung ladder before writing its schema.
The ladder drives retention, approval, and reconciliation requirements, so it must be decided by a
human owner, not inferred from a model description.

```text
 R0  pure read, no state change, no metered side channel        -> cacheable, no key
 R1  read with observable access log (PII read)                 -> audit required
 W2  write, reversible by the same system (draft, label)        -> key + idempotent retry
 W3  write, reversible only by compensation (refund, cancel)    -> key + receipt + saga
 W4  write, externally visible and hard to reverse (payment)    -> key + approval + reconcile
 W5  irreversible broadcast (email blast, public post, delete)  -> approval + no auto-retry
```

Two rules follow immediately.
First, a tool whose effect class is unknown defaults to W4: pessimistic defaults cost latency,
optimistic defaults cost money.
Second, a tool may not span classes: a "get or create customer" call is W2 even though most
invocations read, because the minority path creates.

**Invariants to hold in the implementation.**
These are the properties worth writing as tests rather than as prose.

| ID | Invariant | How it breaks in practice |
|---|---|---|
| I1 | Every admitted effect has exactly one ledger row per `(tool, key)` | upsert semantics replace instead of reject |
| I2 | The digest covers every field that changes the effect | new optional field added without digest update |
| I3 | Authority is re-evaluated at the executor, not only at the host | executor trusts a host-supplied scope claim |
| I4 | State moves forward only through declared transitions | admin tool force-writes `COMMITTED` |
| I5 | A receipt is written before the caller is told "ok" | success returned, receipt write fails, caller retries |
| I6 | `UNKNOWN` is representable and never coerced to `FAILED` | enum lacks the value, so code picks `FAILED` |

Invariant I6 is the cheapest to fix and the most valuable: a durable `UNKNOWN` state converts an
unanswerable question ("did it happen?") into a decidable one ("ask the sink with the key").

**One call, annotated.**
A healthy trace looks like this, and the interesting part is what each line proves.

```text
 12:00:01.104  P1  tool=payments.refund args={order:8841, amount_minor:4500, ccy:USD}
 12:00:01.109  P2  schema=refund/3.2 ok; resolved target merchant=M-77 (from order, not from args)
 12:00:01.114  P2  policy=deny: amount 4500 > refund_limit 4000 for role support_agent
 12:00:01.115  P2  outcome=REJECTED code=POLICY_AMOUNT_LIMIT receipt=-
```

The trace proves the schema passed and the policy saw a resource resolved server-side.
It proves nothing about the downstream payment network, because nothing was dispatched.
Contrast with the ambiguous trace, where the last line is missing:

```text
 12:04:10.220  P3  dispatch attempt=1 deadline=3.0s key=rf_8841_a
 12:04:13.230  P4  no response; connection reset by peer
 12:04:13.240  P5  state=UNKNOWN key=rf_8841_a  -> reconcile with sink
```

Both traces are "errors" to a naive caller, yet only the second requires a sink query, and only the
first can be retried with a fresh decision.

**Counter-argument worth internalising.**
"It is simpler to let the model decide whether a call is safe."
That works until the context contains text written by an attacker, at which point the model's
judgement is exactly the thing being manipulated.
The defensible split is: the model chooses *which* capability to use and with which arguments; the
host and executor decide whether the resulting effect is permitted.
This keeps the model's flexibility where it is valuable and removes it where it is dangerous.

**Where the mental model stops.**
The five-plane view assumes a single executor per tool.
In federated deployments, where an MCP client relays to several servers with different operators, the
planes multiply and the ledger must be keyed by server identity as well as tool name; otherwise two
servers offering `create_ticket` share a keyspace and the second call is rejected as a duplicate of
an effect it never performed.

**Takeaway.**
Distinguish proposal, admission, dispatch, outcome evidence, and accounting; assign each a different
owner, a different test, and a different failure budget.

## 3. Foundational Definitions

Definitions are given with the boundary that makes them useful, because most tool-safety bugs come
from a word being used at two levels of abstraction in the same design review.

**Schema versus contract.**
A schema is the machine-checkable shape of the arguments: types, required fields, enumerations,
numeric bounds.
A contract is the schema plus the guarantees the executor makes about the effect, the receipt, and the
error taxonomy.
A tool with a published schema and no stated effect guarantees is an interface without a contract.

**Tool annotations are hints, not facts.**
MCP-style servers may annotate a tool with `readOnlyHint`, `destructiveHint`, `idempotentHint`, and
`openWorldHint`, and clients are expected to treat those annotations as untrusted unless the server
itself is trusted.
The defaults are pessimistic: absence of annotations means *not* read-only, *possibly* destructive,
*not* idempotent, and open to external systems.
A client that reads `readOnlyHint: true` as an authorization fact has converted a hint into a grant.

| Term | Meaning | Units / type | Boundary that matters |
|---|---|---|---|
| Tool | named capability with typed input and declared effect class | descriptor | a name is not a namespace; versions differ |
| Schema | accepted argument shape | versioned JSON Schema | validates shape, not intent, not rights |
| Contract | schema + effect + receipt + error guarantees | prose plus tests | absent guarantees are not implicit |
| Canonical payload | normalised subset of arguments that determines the effect | derived object | excludes display-only fields |
| Digest `h` | cryptographic hash of the canonical payload | 32 bytes | only as complete as its field set |
| Operation key `k` | stable identifier of one intended business effect | opaque string | must not be a per-attempt UUID |
| Request ID | correlation token for one transport attempt | opaque string | dies with the attempt |
| Receipt | authoritative record of what the sink committed | durable record | not a UI toast, not a log line |
| `UNKNOWN` | outcome not yet decidable | durable state | must be representable |
| Reconciliation | querying the sink with `k` to learn the truth | control action | requires sink support for `k` |
| Compensation | a new effect that offsets a committed one | new operation | not a rollback |
| Confused deputy | executor misusing its own authority on a caller's behalf | failure class | prevented by per-caller scope |
| Confirmation fatigue | approvers rubber-stamping prompts | human factor | the real cost of noisy approvals |

**Canonical payload, precisely.**
The canonical payload is the set of arguments that, if changed, would change what the world looks
like afterwards.
For a refund it is the order, the amount, the currency, and the destination account.
For a ticket creation it is the queue, the requester, and the body.
For a search it is the query and filters - but a search has no effect, so it needs no key at all.
Display fields such as `reason_text`, `display_label`, and `requested_by_name` belong in the audit
record and not in the digest, because including them makes two retries of the same intent look like
a conflict.

**Digest coverage is a property, not an adjective.**
Write it as a ratio: the number of effect-relevant fields covered by the digest divided by the total
number of effect-relevant fields.
A digest with coverage below 1.0 cannot detect the changes it omits, and no amount of retry logic
recovers that.

**Operation key lifetime.**
A key is created once per intended effect by the component that *owns the intent* - the planner, not
the transport retry loop.
Every retry of that intent reuses the key; every genuinely new intent gets a new key.
The pair `(tool, key)` is the ledger's uniqueness constraint, which is why the executor must include
the tool and, in federated setups, the server identity.

**Receipt versus response.**
The response is the bytes that reached the caller.
The receipt is the record that survived.
They are equal only when nothing failed; the design must remain correct when the response is lost,
duplicated, delayed, or reordered.

**Idempotent versus replayable.**
Idempotent means repeating the call with the same key leaves the state unchanged after the first
effect.
Replayable means the executor can reproduce the original receipt for a repeated key, which is
stronger and is what clients actually need: a duplicate call should return the first outcome rather
than merely "not create a second one".

**Approval is an effect, not a message.**
An approval is a signed assertion that a specific principal permitted a specific canonical payload at
a specific time.
If it is not bound to the digest, it approves a class of actions rather than an action, and payload
substitution after approval becomes a live attack.

**Blast radius.**
The number of external records an undetected defect can alter before someone notices.
Fanout multiplies it, retry amplifies it, and weak reconciliation hides it.
It is the quantity to bound when choosing between automated retry and human review.

**Poison effect.**
An effect that will fail on every replay because of a permanent condition - a closed account, a
deleted record, a rejected currency.
Retrying a poison effect converts one failure into a retry storm; the discriminator is that the error
is deterministic and identical across attempts.

**Takeaway.**
Each term above names one type, one storage location, and one failure mode; if two teams disagree on
where the term lives, they will disagree about who owns the incident.
