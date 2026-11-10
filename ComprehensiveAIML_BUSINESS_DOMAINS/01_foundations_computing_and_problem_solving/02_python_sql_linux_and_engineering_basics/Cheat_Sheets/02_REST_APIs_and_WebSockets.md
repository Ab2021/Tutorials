---
collection: "python-sql-linux-engineering-basics"
file_id: "02"
topic: "REST APIs and WebSockets"
level: 2
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Any change to the team's transport choice, gateway limits, retry policy,
  idempotency-key retention window, or WebSocket resume protocol. Re-verify RFC 9110
  conditional-request semantics and RFC 6455 handshake/close rules if cited behaviour
  is relied on operationally."
notation_style: "plain-text"
prerequisites:
  - "01_Python_Async_and_Concurrency.md (tasks, cancellation, bounded concurrency, queue depth)"
  - "Ability to read a JSON schema and an HTTP request line with headers"
pattern_tags: ["interaction-contract", "idempotency", "streaming", "backpressure", "recovery"]
source_status: "OPEN — mapped source body was unavailable; claims below are derived from
  published protocol specifications plus labelled illustrative arithmetic."
owner_role: "Backend / platform engineering"
source_index: "INDEX.md"
---

# REST APIs and WebSockets

> **Purpose:** Choose, specify and defend the *interaction contract* between two processes —
> when request/response HTTP is the honest default, when a long-lived bidirectional socket
> buys enough to justify its operational cost, and how either transport must be wrapped so
> that retries, reconnects, concurrency and slow readers cannot silently corrupt state.
> The sheet's distinctive angle is **effect semantics**: the transport carries bytes, but the
> business decides what a duplicate byte means.
>
> **Prerequisites:** Python async and concurrency (tasks, cancellation, bounded queues,
> in-flight counts) and enough HTTP literacy to read a request line, status code and header
> block. No framework knowledge is assumed.
>
> **Key outcome:** Given a product interaction (a status page that refreshes, a chat room, an
> order submission, a 200 000-subscriber price ticker) you can (a) select a transport with a
> stated disqualifier, (b) write the clause that makes the mutation safe to retry, (c) size the
> in-flight concurrency, outbound buffer and deduplication store with units, and (d) diagnose
> which of four competing causes produced a duplicated order, a stale write, a 1006 close or a
> stalled consumer.
>
> **This file does NOT cover:** application-layer authentication protocols in depth (OAuth 2.0
> flows, token issuance — see the security material in this domain); HTTP caching/CDN
> configuration as a standalone topic; and the design of an event broker's own durability or
> replication guarantees. Those are handoffs, not omissions to be improvised here.
>
> **Level:** 2 — Core Methods, with production diagnostics in Sections 8–9.
>
> **Version / reviewed:** 2.0, drafted 2026-11-09, **not reviewed**. Review of this rewrite is
> OPEN; nothing below has been certified by a second reader.
>
> **Evidence:** Structural claims about HTTP semantics are `SOURCED` to RFC 9110 (HTTP
> Semantics) and RFC 9112 (HTTP/1.1), and about the WebSocket protocol to RFC 6455. Queueing
> results are `DERIVED` from stated premises. Every workload number is `ILLUSTRATIVE`
> arithmetic — there is no measured production trace behind it and none is implied. Items that
> could not be pinned to a specification are marked `OPEN`.
>
> **Retrieval cue:** Duplicate charge → Section 8, F-01. Stale overwrite → F-02. Socket
> memory climbing → F-03. Events lost or repeated across a reconnect → F-04. Queue latency
> rising with no CPU saturation → Section 4.2 and Section 6(c). "Should we use WebSockets?"
> → Section 7.

## 1. File Header

This section fixes the vocabulary and the boundary of the sheet, because "REST API" and
"WebSocket" are used loosely in industry and the looseness is where projects lose money.

**Scope of the decision.** Two independent programs must exchange state. There are exactly four
things that can go wrong in a way no amount of application code will fix, and they are all
transport-shaped:

1. **Ambiguity after a timeout.** The caller stopped waiting. The callee may or may not have
   applied the effect. Neither side knows which, and no acknowledgement can resolve it
   retroactively.
2. **Duplication.** Retries, reconnects, broker redelivery and load-balancer replay all create
   copies. Copies are cheap for reads and expensive for writes.
3. **Disorder.** Bytes are ordered inside one TCP connection. They are not ordered across a
   reconnect, across two connections from the same user, or across two application retries.
4. **Unbounded buffering.** When a producer is faster than a consumer, something must absorb
   the difference or refuse it. "Absorb forever" is a memory-failure plan.

Everything else — status codes, header design, JSON shape, versioning — is convention built on
top of those four. The sheet therefore spends its derivations on ambiguity windows, retry
amplification, queue depth and buffer sizing rather than on naming conventions.

**The reader's task.** A reader who has finished this sheet should be able to take an
interaction and produce a one-page contract containing, at minimum:

| Clause | Question it answers | Failure if absent |
|---|---|---|
| Transport + directionality | Who may initiate, and for how long does the channel live? | Long-polling bolted onto a socket, or a socket used for one-shot reads |
| Identity + authorization scope | Which principal, on which tenant, for which action, checked when? | Handshake-only authorization; revocation has no effect |
| Request/response schema + validation | What is well-formed, and what error does malformed input produce? | Silent coercion; unactionable 500s |
| Effect semantics | What does a repeated identical request do? | Duplicate charge, duplicate shipment |
| Concurrency semantics | What happens when two writers target one record? | Lost update |
| Timeout + retry policy | Who retries, how many times, with what backoff and budget? | Retry storm, amplification through layers |
| Ordering + resume policy | What is guaranteed in order, over what scope, after reconnect? | Gaps and duplicates that look like product bugs |
| Buffering + shedding policy | What happens to a slow consumer? | Process-wide memory exhaustion |
| Observability contract | Which counters distinguish offered/accepted/processed/failed? | Every incident becomes archaeology |

**What "REST" is and is not.** REST, in Fielding's sense, is an architectural style whose
constraints include client–server separation, statelessness of requests, cacheability, a uniform
interface, and a layered system. Most production "REST APIs" are HTTP+JSON RPC with resource-ish
URLs; they implement *some* of the style, typically statelessness and a uniform verb set, and
skip hypermedia. This sheet teaches the HTTP+JSON contract that practitioners actually ship, and
says so rather than pretending to teach Fielding compliance. The practical payoffs people want
from "REST" — cacheability, idempotent retries, uniform error handling — are obtainable from
RFC 9110 semantics without hypermedia controls; the payoffs that are *not* obtainable without
hypermedia (driving a client through unknown transitions) are out of scope. `SOURCED`
conceptually to Fielding's 2000 dissertation; the operational advice below is `DERIVED` from
RFC 9110.

**What a WebSocket is.** RFC 6455 defines a protocol that begins as an HTTP/1.1 `GET` with
`Upgrade: websocket`, and on acceptance becomes a framed, full-duplex, message-oriented channel
over the same TCP connection. It is *not* "a faster HTTP". It removes the request/response
constraint and the per-request header cost; in exchange it makes the server responsible for
connection state, liveness detection, per-message authorization, flow control and resume. Those
responsibilities are the entire cost of the technology and the reason most "we need WebSockets"
requests are better served by Server-Sent Events or plain polling.

**The one-sentence version of the whole sheet.** A transport moves bytes; a contract decides
what a repeated byte means, what a missing byte means, and what happens to a byte nobody can
consume — and the second half is where systems actually fail.

**Takeaway:** Choose a transport for its directionality and lifetime; choose a contract for its
effect, ordering and buffering semantics. Confusing the two is the root error this sheet exists
to prevent.

## 2. Concept Map / Mental Model

The map reads top to bottom: an interaction requirement is classified, a transport is selected
with an explicit disqualifier, and then — independently of the transport — the effect contract is
specified. The right-hand column is the part teams skip.

```text
                 [ Product interaction requirement ]
                 who initiates? how long? how many listeners?
                 does a repeat mean the same thing twice?
                                   |
                                   v
                 +-----------------------------------+
                 | Directionality / lifetime filter  |
                 | one-shot? server-push-only?       |
                 | bidirectional & stateful?         |
                 +-----------------------------------+
                    |            |             |
        one-shot    |            | push-only   |  bidirectional
                    v            v             v
              HTTP request    HTTP + SSE     WebSocket
              /response       (text/event-   (RFC 6455
              (RFC 9110)       stream)        upgrade)
                    |            |             |
                    +------------+-------------+
                                   |
                                   v
                 +-----------------------------------+
                 | EFFECT CONTRACT (transport-blind) |
                 | idempotency key / conditional     |
                 | write / sequence number / cursor  |
                 +-----------------------------------+
                          |                  |
                 writes   |                  |  streams
                          v                  v
                 [ dedup at the       [ ordered sequence +
                   effect owner ]       replay buffer or
                          |             explicit gap signal ]
                          v                  v
                 +-----------------------------------+
                 | RESOURCE & BUFFER POLICY          |
                 | in-flight cap, queue watermark,   |
                 | per-connection byte budget,       |
                 | shedding rule, close code         |
                 +-----------------------------------+
                                   |
                                   v
                 [ observability: offered / accepted / processed /
                   rejected / timed-out / disconnected / retried,
                   keyed by principal and route ]
                                   |
                                   v
                 [ feedback: does the *effect* match the intent?
                   reconcile from the system of record, not the log ]
```

**How to read the map.** Three arrows carry almost all the information:

- **The vertical arrow through "effect contract" is transport-independent.** This is the central
  claim of the sheet. You can put an idempotency key on a `POST` and you can put a sequence number
  on a socket frame; both are the same idea — a client-chosen token that lets the *effect owner*
  recognise a repeat. Swapping transports does not change the design of this layer, and swapping
  it does not fix a missing one.
- **The fork after "effect contract" is where the two styles genuinely diverge.** A request is a
  discrete, independently completable act; it can be retried by anyone holding the same token. A
  stream is a *position in a sequence*; recovering it means knowing where you were, which
  requires either a server-side replay buffer or an explicit "you missed events" signal. Request
  recovery is idempotent re-execution; stream recovery is cursor rewind.
- **The bottom feedback arrow is not the access log.** A 200 response proves the server accepted
  and answered. It does not prove the downstream ledger was written. Reconciliation must query
  the system of record for the *effect*, which is a different data source from the API's own
  response log.

**The central invariant.** *For every client intent there exists at most one committed effect,
and every committed effect is reachable by every authorized reader that is entitled to it.* The
first half is idempotency; the second half is ordered replay. A design that satisfies only the
first half will double-charge nobody but will silently lose events; a design that satisfies only
the second half will deliver every event and apply some of them twice.

**What changes between variants.**

| Variant | What changes relative to the base | What stays identical |
|---|---|---|
| HTTP request/response | Baseline: one request, one response, connection may be reused | Effect contract, retry budget, observability |
| HTTP + conditional GET (`If-None-Match`) | Response may be `304` with no body | Cacheability reasoning, version identity |
| HTTP long-poll | Server holds the request open until data or timeout | Still one request per delivery; resume via cursor |
| SSE | One long-lived response, server→client only, `id:`/`Last-Event-ID` resume | Sequence + gap semantics |
| WebSocket | Full duplex, framed messages, both sides push, connection is state | Sequence + gap semantics; per-message authorization |
| Webhook | Receiver is the server; retries are the *sender's* problem | Idempotency key on the receiving end |
| gRPC streaming | HTTP/2 multiplexed streams, typed schema, flow-control windows | Flow control = backpressure; ordering per stream |

**The most important boundary.** All of the above run over TCP (or QUIC for HTTP/3), and TCP
guarantees ordered, reliable delivery *of bytes within one connection instance*. It guarantees
nothing across a reconnect, nothing about message boundaries (that is the application protocol's
job), nothing about whether your application consumed a byte it acknowledged, and nothing about
ordering between two connections. Every "we lost events on reconnect" incident lives in the gap
between "TCP is reliable" and "your session is not".

**Second-order map: where a single request spends its time.**

```text
client                network              gateway            service            store
  |  build req            |                    |                  |                 |
  |------ send ---------->|                    |                  |                 |
  |                       |-- queue/wait ----->|                  |                 |
  |                       |                    |-- authz lookup ->|                 |
  |                       |                    |-- route -------->|                 |
  |                       |                    |                  |-- validate ---->|
  |                       |                    |                  |-- write ------->|
  |                       |                    |                  |<-- commit ------|
  |<----- response -------|<-------------------|<-----------------|                 |
  |                       |                    |                  |                 |
  |<=========== client timeout window =========>|                  |                 |
  |  ambiguity begins here if the timeout fires before the commit marker returns     |
```

The diagram's point is the bottom band: the client's timeout is a property of the *client*, and
the commit happens at the *store*. When the timeout fires, the request does not stop existing; it
continues to execute server-side. That is precisely why a timeout is not a failure signal and why
`POST` retries need a token.

**Takeaway:** Pick the transport from the directionality/lifetime table, then design effects and
recovery as if the transport did not exist — because the failure that hurts is always the one the
transport was never going to prevent.

## 3. Foundational Definitions

Eight terms carry the sheet. Each is defined formally, given a plain-language reading, given a
notation with units and domain, and then distinguished from the near-neighbour it is most
often confused with. The "check" column is what an operator can actually observe.

### 3.1 Idempotency and idempotency key

**Definition.** An operation is *idempotent* if applying it twice produces the same resulting
state as applying it once. Formally, for operation $f$ over state $S$: $f(f(S)) = f(S)$. An
*idempotency key* is a client-generated token, sent with a mutating request, that the server
stores alongside the created effect so that a later request bearing the same token returns the
recorded outcome instead of creating a second effect. RFC 9110 identifies `GET`, `HEAD`,
`OPTIONS`, `TRACE` and — with server-side care — `PUT` and `DELETE` as idempotent methods, and
explicitly notes that `POST` is neither safe nor idempotent by default.

**Intuition.** "Set the light to red" is idempotent; "toggle the light" is not. "Set the light to
red" is *also* safe to retry because the retry carries enough information to be a no-op. An
idempotency key adds that same property to "charge the card", whose payload cannot express
"already done".

**Notation.** $k$ = key, a string with a scope (principal), a hash of the request payload, and a
retention window $T_{\text{idem}}$ in seconds. Domain: keys must be unique per principal within
the window; reuse with a *different* payload must be rejected, not honoured.

**Example / nonexample.** Example: `POST /v1/transfers` with `Idempotency-Key: 7c1f…` and a body
of 100.00 USD; the retry with the same key returns the original transfer ID and the original
`201` body. Nonexample: a server that de-duplicates on the *payload* instead of a key will refuse
a legitimate second identical transfer.

**Distinction that matters.** *Idempotent* is a property of an operation's semantics; an
*idempotency key* is a mechanism that makes a non-idempotent operation behave idempotently for
one client intent. Method-level idempotency (`PUT`) and key-level idempotency (`POST` + key)
are not the same guarantee: `PUT` is idempotent only if the body is the complete target state.

**Operational check.** Send the identical request twice and compare (a) the created record
count and (b) the recorded response bodies. Two records ⇒ the key is not being persisted at the
effect owner. One record but two different responses ⇒ the key is stored, but the recorded
response is being regenerated rather than replayed.

### 3.2 Safe method, side effect, and effect owner

**Definition.** A method is *safe* if it is intended only for retrieval and imposes no state
change. A *side effect* is any durable consequence of handling a request. The *effect owner* is
the single component whose write is the authoritative consequence — the ledger, the order table,
the queue with a transactional outbox.

**Intuition.** The effect owner is the only place where de-duplication can actually be correct,
because it is the only place that commits. A gateway that de-duplicates is preventing duplicate
*network invocations*; it is not preventing duplicate *effects* if a second path (a retry from a
different gateway, a queue redelivery, a manual replay) reaches the store.

**Notation.** For request $r$ with key $k$, the effect owner maintains a row
$(k, \text{hash}(r), \text{status}, \text{response}, t_{\text{created}})$. Uniqueness on $k$.

**Example / nonexample.** Example: an outbox row written in the same transaction as the order row.
Nonexample: an in-memory `set` of recently-seen keys in the gateway — correct until restart,
correct on one replica, and therefore wrong under both conditions that matter.

**Check.** Restart every gateway replica, then replay a key. If a second effect appears, the
store was not authoritative.

### 3.3 Conditional request / optimistic concurrency

**Definition.** A conditional request includes a validator — `ETag` compared via `If-Match`, or a
`Last-Modified` date compared via `If-Unmodified-Since` — such that the server applies the
change only if the target still matches. RFC 9110 defines the outcome of a failed precondition as
`412 Precondition Failed`. This is optimistic concurrency control: no lock is held between
read and write; a conflict is detected at write time.

**Intuition.** "Overwrite the row I read, but only if nobody changed it since." The alternative —
pessimistic locking — holds a lock for the round-trip, which is unsafe across a network because
a client can die holding it.

**Notation.** $v$ = version token (opaque). Read returns $(s, v)$. Write sends `If-Match: v`. Server
applies iff stored version $=v$; else `412`. Version may be a counter, a content hash, or the
`ETag` of the serialized representation.

**Example / nonexample.** Example: two analysts open one record at version 7; the first write with
`If-Match: 7` succeeds and the record becomes version 8; the second write with `If-Match: 7`
receives `412` and must re-read. Nonexample: `PUT` that silently overwrites — that is last-writer-wins,
which is a *business* decision someone must own.

**Distinction.** `ETag` on a `GET` enables cache validation (saves bytes); `If-Match` on a `PUT`
enables concurrency control (prevents lost updates). Same header family, different purpose.

**Check.** Force two concurrent writes from a script; assert exactly one `2xx` and one `412`.

### 3.4 Sequence number, cursor, and replay window

**Definition.** For a stream, a *sequence number* is a monotonic per-scope identifier assigned by
the server to each event. A *cursor* is the last number a client has durably processed. A
*replay window* is the retention interval $T_{\text{replay}}$ during which the server can resend
events at or after an arbitrary cursor.

**Intuition.** Request recovery re-executes an operation; stream recovery rewinds a position.
Without a number there is no position, and "reconnect" degenerates into "hope".

**Notation.** $e_i$ event $i$; cursor $c$; window $T_{\text{replay}}$ [s]; events retained
$R = \lambda_{\text{ev}} T_{\text{replay}}$ [events] for event rate $\lambda_{\text{ev}}$ [events/s].

**Example / nonexample.** Example: SSE `id:` field plus `Last-Event-ID` on reconnect; the server
replays from the next unacknowledged id. Nonexample: incrementing a counter inside the client on
receipt — after a gap, the client's counter cannot tell it what it missed.

**Distinction.** TCP sequence numbers are per-connection and invisible to the application; they
do not survive reconnect. Application sequence numbers must be assigned where the event becomes
durable, not where it is sent.

**Check.** Kill the client for $>T_{\text{replay}}$ and reconnect; assert the server reports a gap
explicitly rather than silently resuming.

### 3.5 Backpressure, bounded queue, and shedding

**Definition.** *Backpressure* is the mechanism by which a consumer's inability to keep up is
communicated to the producer so the producer slows or refuses. A *bounded queue* is a buffer with a
fixed maximum occupancy and a declared behaviour at the bound. *Shedding* is the deliberate
discard (or rejection) of work that cannot be served within budget.

**Intuition.** Latency is a queue in disguise: if arrivals exceed departures, the excess does not
disappear, it accumulates as waiting time. Shedding converts an unbounded latency increase (bad
for everyone) into a bounded, visible rejection (bad for some, observable).

**Notation.** $\lambda$ arrival rate [msg/s], $\mu$ service rate [msg/s], $W$ per-message service
time [s], $L$ mean in-flight [msg], $B_{\max}$ queue capacity [msg].

**Example / nonexample.** Example: per-connection outbound buffer capped at $B_{\max}=256$ frames;
on overflow, coalesce or close with code 1013 ("try again later"). Nonexample: `asyncio.Queue()`
with no `maxsize` in front of a slow writer.

**Distinction.** Backpressure is the *signal*; a bounded queue is the *structure*; shedding is the
*policy at the bound*. HTTP/2 and QUIC implement flow control at the byte level, which is real
backpressure — but only up to the application's own reading behaviour.

**Check.** Stall the consumer and watch the producer's write latency; if it does not rise while
queue depth rises, the queue is unbounded and the process will OOM rather than slow down.

### 3.6 Head-of-line blocking and message framing

**Definition.** *Head-of-line (HOL) blocking* occurs when a unit of work that cannot proceed
prevents later units from proceeding despite their being independently serviceable. *Framing* is
the rule that lets a receiver find message boundaries in a byte stream.

**Intuition.** A single TCP connection carrying many logical messages means one lost segment
delays every message behind it. HTTP/2 reduces this *across requests* via multiplexing but not
within one stream; WebSocket has exactly one ordered message stream per connection, so a slow
consumer blocks only its own connection — which is why "one connection per user session" is often
safer than "one connection per app".

**Notation.** Framing on WebSocket: an opcode, a FIN/continuation bit, and a 7-bit length with
16- or 64-bit extension; client→server frames are masked with a 4-byte key.

**Example / nonexample.** Example: two independent dashboards over one WebSocket share fate — one
stalled tab blocks the other. Nonexample: assuming HTTP/1.1 pipelining gives concurrency; it gives
ordering, and the responses must still come back in order.

**Check.** Introduce artificial delay on one logical channel and measure whether a second channel
on the same connection slows.

### 3.7 Timeout, deadline, and cancellation propagation

**Definition.** A *timeout* is a local limit on waiting. A *deadline* is an absolute time by which
a caller needs an answer, propagated so that downstream work can stop early. *Cancellation* is the
act of stopping work no longer wanted. In gRPC these are first-class (`grpc-timeout`); in HTTP they
must be constructed from a header or a context.

**Intuition.** A timeout is a local promise to stop waiting. It is not a promise that the remote
side stops working — that requires propagation, and even then a cancellation that arrives after
the commit is a no-op.

**Notation.** $T_{\text{budget}}$ remaining budget [s], decremented by each hop's observed latency;
a hop starting with $T_{\text{budget}} \le 0$ must fail fast.

**Example / nonexample.** Example: gateway enforces 2 000 ms; it forwards `grpc-timeout: 1800m` after
subtracting 200 ms of its own elapsed time. Nonexample: three services each with an independent
5 s timeout behind a 3 s client timeout — the client gives up, the services keep working, and the
work is wasted and possibly duplicated by the retry.

**Check.** Compare the count of commits at the store against the count of `5xx`/timeouts at the
client for the same window; unexplained commits are the ambiguity.

### 3.8 Close code, liveness, and zombie connection

**Definition.** In RFC 6455, a closing handshake carries a two-byte status code. Codes `1000`
(normal), `1001` (going away), `1008` (policy violation), `1009` (message too big), `1011`
(internal error), `1013` (try again later) are defined; `1006` is *reserved for local reporting only*
and means "closed without a close frame" — it is never sent on the wire. *Liveness* is the
ability to distinguish "idle" from "dead". A *zombie* connection is one whose TCP state is open but
whose peer or path is gone.

**Intuition.** `1006` is the protocol telling you that it could not tell you anything. It is a
diagnosis of ignorance, not a cause. If your dashboards are dominated by `1006`, your liveness
design is missing, not your close handling.

**Notation.** Heartbeat interval $h$ [s], liveness deadline $T_{\text{dead}} = n h$ [s] for $n$
missed heartbeats; expected detection time uniform in $(0, T_{\text{dead}})$ under a naive scheme.

**Example / nonexample.** Example: server sends a ping every 25 s and closes the connection if no
pong within 60 s, then the client reconnects with a cursor. Nonexample: relying on TCP keepalive at
the OS default of two hours to detect a half-open connection through a NAT that dropped the mapping.

**Check.** Kill the network path (not the process) and time how long until the server frees the
connection and the client knows.

| Term | Formal meaning | Units / shape | Valid domain | Common confusion | Where it is checked |
|---|---|---|---|---|---|
| Idempotent operation | $f(f(S)) = f(S)$ | Boolean property of the operation | Any state model with a defined $f$ | "retried" vs "idempotent" | Replay test at the store |
| Idempotency key | Client token identifying one intent | String, scoped per principal | Unique within $T_{\text{idem}}$ | De-duplicating on payload | Duplicate submit, count records |
| Effect owner | Component whose commit is authoritative | Component | Exactly one per effect | Gateway, log, or cache | Replay after all replicas restart |
| Safe method | Retrieval with no intended state change | HTTP method | Defined set in RFC 9110 | "GET is harmless" (it can be logged, counted, cached) | Side-effect audit |
| Conditional request | Apply only if validator matches | Header pair (`If-Match`/`ETag`) | Non-empty validator | Cache validation vs concurrency control | Concurrent-write test |
| Sequence number | Server-assigned monotonic per-scope id | Integer | Monotonic, gap-free per scope | TCP sequence number | Reconnect and compare ids |
| Cursor | Last durably processed position | Integer | $\le$ latest committed id | Client-side counter | Resume test |
| Replay window | Retention interval for resending | Seconds, events | $R = \lambda_{\text{ev}} T_{\text{replay}}$ | "We keep logs, so we can replay" | Reconnect after $>T_{\text{replay}}$ |
| Backpressure | Consumer-slow signal to producer | Mechanism | Requires a bounded path | "We added a queue" | Producer latency under stalled consumer |
| Bounded queue | Fixed-capacity buffer with policy | Capacity [items] | $B_{\max} < \infty$ | "The broker buffers it" | Memory vs depth under stall |
| HOL blocking | One stuck unit blocks later ones | Mechanism | Any shared ordered channel | "HTTP/2 fixed HOL" (only partly) | Delay one channel, measure another |
| Deadline | Absolute latest acceptable completion | Timestamp / duration | $\ge$ now | Local timeout | Compare hop budgets |
| Close code | Two-byte reason in closing handshake | Integer | 1000–1015, 3000–4999 | `1006` as a cause | Server-side close-frame log |

**Takeaway:** The eight definitions reduce to two questions — "can I recognise a repeat?" (3.1,
3.2, 3.4) and "can I refuse work I cannot serve?" (3.5, 3.8) — and the rest of the sheet is the
arithmetic and diagnostics those questions imply.

## 4. Core Formulas with Full Derivation

Section 4 has one job: replace hand-waving with arithmetic. Every result below is derived from a
definition, given the five-part block, and then broken at its boundary. All numbers are
`ILLUSTRATIVE`.

### 4.1 Base case — Little's law and the in-flight count

**Validity before use.** The system must be in steady state over the observation window, the
count must be taken at an instant or averaged consistently, and every unit that enters must
eventually leave (no leak, no infinite retention). The measured quantity is a *time-average*
of occupancy against a *time-average* arrival rate; it is not a guarantee about any percentile.

**Derivation.** Consider a window $[0,T]$. Let $A(T)$ be the number of units that entered and
$L$ the average number present. Each unit $j$ contributes its residence time $W_j$ to the
integral of occupancy, so $\int_0^T L(t)\,dt = \sum_{j} W_j$ (up to edge effects for units
straddling the boundary). Dividing by $T$ gives $\bar L = \frac{1}{T}\sum_j W_j$. Writing
$\lambda = A(T)/T$ and $\bar W = \frac{1}{A(T)}\sum_j W_j$ yields
$\bar L = \lambda \bar W$. This is Little's law; it holds for any stable system with a
consistent counting convention, regardless of arrival distribution — which is exactly why it is
the right first tool and exactly why it cannot tell you anything about variance.

```text
FORMULA [L-01]: L = lambda * W
WHERE: L mean number in system [requests]; lambda mean arrival/completion rate
       [requests/s] in steady state; W mean time in system [s].
CONSTRAINT: steady state over the window; consistent in/out boundary;
            no leak (units are not retained beyond W); time-averages only.
NUMERICAL EXAMPLE: lambda = 500 req/s, W = 0.040 s
                   L = 500 * 0.040 = 20 requests in flight.
INTERPRETATION: 20 concurrent workers (plus headroom) can serve this load at
                this mean latency; 20 is not a limit, it is a measurement.
```

**Where it misleads.** Teams use $L$ as a concurrency setting. If mean latency is 40 ms and you
provision exactly 20 slots, you have provisioned the mean, and any burst or latency tail turns
into queueing. Provision from a target at a high percentile, not from the mean: if you want the
$p_{99}$ latency to stay at 40 ms and you size for the $p_{99}$ arrival rate, you get a much
larger $L$. Concretely, if $p_{99}$ arrival is 800 req/s during the peak minute and the $p_{99}$
service time is 90 ms, then $L_{99} = 800 \times 0.090 = 72$ slots are occupied at the moment
that matters.

**Boundary.** If $\lambda$ exceeds capacity, no steady state exists, $L$ grows without bound, and
Little's law still "holds" trivially (both sides diverge) while telling you nothing useful. That
is why $L$ must be read alongside queue depth and rejection rate.

### 4.2 General case — queueing delay under utilisation

**Validity before use.** The M/M/1 form below assumes Poisson arrivals, exponentially distributed
service, a single server, infinite buffer, and first-come-first-served discipline. Those
assumptions are routinely violated (bursty arrivals, multi-server pools, bounded buffers with
shedding, priority classes), so the result is used as a *shape*, not an oracle.

**Derivation.** With arrival rate $\lambda$ and service rate $\mu$, define utilisation
$\rho = \lambda/\mu < 1$. The mean number in an M/M/1 system is $L = \rho/(1-\rho)$ (standard
result for the birth–death chain with $\pi_n = (1-\rho)\rho^n$, giving
$\mathbb{E}[n] = \rho/(1-\rho)$). Applying [L-01] gives mean time in system
$W = L/\lambda = 1/(\mu-\lambda)$. Subtracting the mean service time $1/\mu$ leaves the mean
*waiting* time $W_q = \rho/(\mu-\lambda)$.

```text
FORMULA [Q-01]: W_q = rho / (mu - lambda)          with rho = lambda/mu < 1
WHERE: W_q mean queueing (waiting, excluding service) time [s];
       lambda arrival rate [req/s]; mu service rate [req/s]; rho dimensionless.
CONSTRAINT: Poisson arrivals, exponential service, one server, infinite
            buffer, FCFS; rho < 1 (no steady state at or above 1).
NUMERICAL EXAMPLE: mu = 500 req/s (W = 2 ms service), lambda = 450 req/s
                   rho = 0.90; W_q = 0.90 / (500 - 450) = 0.018 s = 18 ms
                   so mean end-to-end = 18 + 2 = 20 ms.
INTERPRETATION: at 90% utilisation, waiting time is 9x the service time;
                the same system at rho = 0.50 gives W_q = 0.50/250 = 2 ms.
```

**The shape that matters.** $W_q$ is proportional to $\rho/(1-\rho)$: 0.50→1, 0.80→4, 0.90→9,
0.95→19, 0.99→99. Moving from 80% to 90% utilisation *doubles* waiting time while adding only 10
percentage points of throughput. This single ratio explains most "the API got slow and we changed
nothing" incidents: traffic rose 12%, utilisation crossed a knee, latency doubled, and the change
looked discontinuous because it was non-linear, not because something broke.

```text
FORMULA [Q-02]: W(multi-server) ≈ (1/mu) + C * rho^(sqrt(2(m+1))-1) / (m(1-rho)) * (1/mu)
WHERE: m identical servers; C depends on service-time variability
       (C = 1 for exponential, ->0 for deterministic); rho = lambda/(m*mu).
CONSTRAINT: approximation for M/M/m and M/D/m; identical servers, common
            queue, no priority classes; error grows at very high rho.
NUMERICAL EXAMPLE: m = 4, mu = 125 req/s each (capacity 500),
                   lambda = 450 -> rho = 0.90;
                   exponent sqrt(2*5)-1 = sqrt(10)-1 = 2.1623;
                   0.90^2.1623 = 0.7963; W_q ≈ 0.7963/(4*0.10*125)
                   = 0.7963/50 = 0.01593 s = 15.9 ms.
INTERPRETATION: pooling servers barely helps at fixed rho - the curve is
                set by utilisation, not by m; m helps because it raises mu.
```

**Why the multi-server number is close to the single-server one.** The approximation is dominated
by the $(1-\rho)^{-1}$ term; increasing $m$ at constant $\rho$ does not flatten the knee. What
pooling buys is *variance reduction* — with $m$ servers and a common queue, a burst can be
absorbed by idle peers instead of waiting behind one slow request, which matters at high
service-time variability (high $C$), not at low.

**Boundary.** At $\rho \to 1$ every formula here diverges. That is not a modelling artifact: it is
the mathematical statement that an infinite-buffer system at capacity has unbounded waiting. Real
systems avoid the divergence by *shedding* — which is why Section 3.5 is a prerequisite for
reading this section honestly.

### 4.3 Variants — retries, backoff, tokens, dedup, buffering

Each variant below changes one assumption of the base model. Read the "what changes" line first.

**(a) Retry amplification.** *What changes:* a failed attempt may be repeated $r$ times, so one
client intent can generate more than one server-observed request.

**Derivation.** Assume each attempt succeeds independently with probability $q$, and let
$p = 1-q$ be the per-attempt failure probability (this independence is the strong assumption).
The number of attempts $N \in \{1, \dots, r+1\}$; $\Pr(N > i) = p^i$. Using
$\mathbb{E}[N] = \sum_{i \ge 0}\Pr(N > i)$ for a positive integer variable:

```text
FORMULA [R-01]: E[N] = sum_{i=0}^{r} p^i = (1 - p^(r+1)) / (1 - p) = (1 - p^(r+1))/q
WHERE: N attempts per intent; p per-attempt failure probability; r retries
       (so r+1 attempts max); q = 1 - p success probability.
CONSTRAINT: independent attempts, fixed r, no jitter interaction, failure
            detectable (timeouts treated as failures); no circuit breaker.
NUMERICAL EXAMPLE: p = 0.20, r = 2 -> E[N] = (1 - 0.008)/0.80 = 1.24
                   attempts per intent; success probability 0.992.
                   With r = 5: E[N] = (1 - 0.000064)/0.80 = 1.24992,
                   success probability 0.999936.
INTERPRETATION: retries buy reliability with sharply diminishing returns -
                going from r=2 to r=3 raises success from 0.992 to 0.9984
                (+0.64 percentage points) for 0.008 extra attempts per
                intent; r=5 buys another 0.15 points for 0.002.
```

**The load consequence.** If 20% of intents fail and each gets 2 retries, server-observed load is
$1.248\times$ the intent rate — but only *after* the failures happen, in a burst, at the worst
moment. Two layers each retrying 2 times compound to $1.248^2 \approx 1.56\times$ observed load and
up to 9 attempts per intent in the worst case. This is why retry policy must be owned by exactly
one layer, usually the outermost that knows the user's intent.

**(b) Exponential backoff with cap.** *What changes:* retries are spaced, so a struggling
dependency is not hammered.

```text
FORMULA [B-01]: d_i = min(d_cap, d_0 * 2^i), i = 0,1,...,r
WHERE: d_i delay before attempt i+1 [s]; d_0 initial delay [s];
       d_cap maximum delay [s]; i retry index (0-based).
CONSTRAINT: no jitter shown (add uniform or decorrelated jitter in practice);
            timer resolution and queueing add overhead to the nominal delay.
NUMERICAL EXAMPLE: d_0 = 0.1 s, d_cap = 30 s, r = 8:
                   delays 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6 s
                   cumulative 51.1 s of waiting before the 10th attempt.
INTERPRETATION: 8 retries is not "more robust"; it is 51 s of a caller
                holding a request open. Choose r from the user's deadline.
```

**Why jitter is not cosmetic.** Without jitter, all clients that failed together retry together;
the retry wave is synchronised and can exceed the original load. Full jitter
(`sleep = uniform(0, d_i)`) spreads the wave; its cost is a wider latency distribution. The
correct statement is: *backoff reduces rate, jitter decorrelates* — neither alone gives you both.

**(c) Token bucket admission.** *What changes:* a steady rate is enforced with a burst allowance.

**Derivation.** A bucket holds at most $C$ tokens (the burst size) and refills at rate $\rho_t$
tokens/s. Each request removes one token; a request arriving at an empty bucket waits
$(1 - \text{available})/\rho_t$ seconds. Starting empty, $n$ consecutive requests need
$\lceil n/\rho_t \rceil$ seconds in the limit.

```text
FORMULA [T-01]: wait(n) = max(0, (n - c_avail) / rho_t)
WHERE: n requests to admit [count]; c_avail tokens currently available
       [count]; rho_t refill rate [tokens/s]; C bucket capacity [tokens].
CONSTRAINT: single bucket, no priorities, refill continuous, cost 1/request.
NUMERICAL EXAMPLE: rho_t = 50 req/s, C = 20, bucket empty, n = 30:
                   wait = (30 - 0)/50 = 0.6 s for all 30 to be admitted.
                   A burst of 20 is admitted immediately when full.
INTERPRETATION: the bucket converts a rate limit into a bounded queue of
                at most C/r outliers; capacity C is the burst you promise.
```

**(d) Deduplication store sizing.** *What changes:* the idempotency layer now has state, and that
state has a size and a cost.

```text
FORMULA [D-01]: S = m * T_idem * b
WHERE: S store size [bytes]; m mutating requests per second [req/s];
       T_idem retention window [s]; b bytes per record including key,
       payload hash, status and response snapshot [bytes/record].
CONSTRAINT: one record per distinct intent, no compaction, no replication,
            uniform rate; excludes indexes and WAL overhead.
NUMERICAL EXAMPLE: m = 200 req/s, T_idem = 86400 s (24 h), b = 200 B:
                   records = 200 * 86400 = 17,280,000
                   S = 17,280,000 * 200 = 3,456,000,000 B = 3.219 GiB.
INTERPRETATION: a 24 h window at 200 writes/s costs ~3.2 GiB before indexes;
                halving the window halves the store, and halving it again is
                usually safe if clients retry within minutes.
```

**How to choose $T_{\text{idem}}$.** It must exceed the *maximum plausible* client retry horizon:
the longest a client (or a queue, or a human clicking twice, or a batch replay) might resubmit the
same intent. A 24 h window that costs 3.2 GiB is cheap; a 7-day window at 2 000 writes/s would be
$2000 \times 604800 \times 200 = 2.4192\times 10^{11}$ B = 225.3 GiB, which is a different
conversation. The window is a *customer-facing promise* and belongs in the API documentation.

**(e) Outbound buffer and slow-consumer memory.** *What changes:* each connection now owns a
buffer, so memory scales with connections × buffer, not with load.

```text
FORMULA [M-01]: M = N * (B_max * s + o)
WHERE: M resident buffer memory [bytes]; N concurrent connections;
       B_max frames queued per connection [frames]; s mean frame size
       [bytes/frame]; o per-connection overhead [bytes/conn].
CONSTRAINT: buffers are full (worst case for memory), uniform frame size,
            no kernel socket-buffer accounting, no allocator overhead.
NUMERICAL EXAMPLE: N = 50,000, B_max = 256, s = 24 B (small JSON delta),
                   o = 24 KiB: M = 50,000*(256*24 + 24576) = 50,000*30,720
                   = 1,536,000,000 B = 1.431 GiB.
INTERPRETATION: a "small" 256-frame buffer is 1.4 GiB at 50k connections;
                B_max = 1024 would be 5.5 GiB and is a capacity decision.
```

**(f) Stream replay retention.** *What changes:* recovery requires the server to hold history.

```text
FORMULA [P-01]: R = lambda_ev * T_replay ; S_replay = R * s
WHERE: R retained events [events]; lambda_ev event rate [events/s];
       T_replay replay window [s]; s bytes per retained event [bytes];
       S_replay retention storage [bytes].
CONSTRAINT: uniform rate, fixed payload, payload-only storage, no
            per-subscriber cursors or index overhead.
NUMERICAL EXAMPLE: lambda_ev = 2,000 ev/s, T_replay = 300 s, s = 512 B:
                   R = 2,000*300 = 600,000 events
                   S_replay = 600,000*512 = 307,200,000 B = 293.0 MiB.
INTERPRETATION: a 5-minute replay window at 2k ev/s is affordable; a 24 h
                window would be 2,000*86,400*512 = 88,473,600,000 B = 82.4 GiB,
                so long outages need a different recovery path
                (snapshot + cursor) than "replay everything".
```

**(g) Fan-out bandwidth.** *What changes:* one published event becomes $N$ transmissions.

```text
FORMULA [F-01]: B = N * r * s   ;  B_bits = 8 * B
WHERE: B aggregate egress [B/s]; N subscribers; r events per second per
       subscriber [ev/s]; s bytes per event [bytes]; B_bits [bit/s].
CONSTRAINT: every subscriber receives every event; no coalescing, no
            compression, no per-subscriber filtering; egress only.
NUMERICAL EXAMPLE: N = 20,000 subscribers, r = 2 ev/s, s = 512 B:
                   B = 20,000*2*512 = 20,480,000 B/s = 163.84 Mbit/s.
INTERPRETATION: a 20k-subscriber ticker needs ~164 Mbit/s sustained, which is
                why delta encoding (s -> s/10) is a capacity feature.
```

### 4.4 Asymptotic and limiting behaviour, and counterexamples

This subsection is the honesty check on Sections 4.1–4.3: each result is stated with the way it
breaks.

| Result | Required assumption | Diagnostic that shows a break | Violating example | Correct alternative |
|---|---|---|---|---|
| $L=\lambda W$ [L-01] | Steady state, no leak | Occupancy grows while rate is flat | Connection objects never released after `1006` | Fix the leak; then re-measure $L$ |
| $W_q=\rho/(\mu-\lambda)$ [Q-01] | Poisson arrivals, exponential service, FCFS | Arrivals bursty; $W_q$ observed $\gg$ prediction at low $\rho$ | Marketing campaign releases 10 000 requests in 200 ms at $\rho=0.4$ | Use a burst/G/D/1 model or measure the tail directly |
| $E[N]$ [R-01] | Independent attempts | Retries correlate with the same failure | Retrying a `503` while the dependency is down | Circuit breaker; retry only on transient, idempotent operations |
| Backoff [B-01] | Timer honoured, no queueing | Observed delay $\gg$ nominal | Event loop blocked 2 s while sleeping 100 ms | Wall-clock deadlines, not sleep counts |
| Token bucket [T-01] | Single bucket, unit cost | Large requests cost the same as small | 10 MB uploads at 1 token each | Cost-weighted buckets (bytes, not requests) |
| Dedup store [D-01] | One record per intent | Key reuse with different payload | Client generates the key at render time and reuses it | Bind key to payload hash; reject mismatch |
| Buffer memory [M-01] | Bounded $B_{\max}$ | Queue depth unbounded, memory tracks load | `asyncio.Queue()` with default `maxsize=0` | Set `maxsize`; define overflow policy |
| Replay [P-01] | Uniform event rate | Recovery after outage exceeds window | Client offline 40 min with $T_{\text{replay}}=300$ s | Snapshot + resume, or explicit "gap" error |
| Fan-out [F-01] | Every subscriber gets everything | Egress exceeds link capacity at peak | 20k × 2 ev/s × 4 KiB = 1.31 Gbit/s | Coalesce, filter, or delta-encode |

**The four limits worth memorising.**

1. **$\rho \to 1$.** Waiting time diverges as $\rho/(1-\rho)$. Any capacity plan whose headroom is
   "we are at 92%, it's fine" is a plan to make latency non-linear in traffic. The only real
   protections are admission control (shed), queue bounds, and load reduction — not more servers
   added *after* the knee.
2. **$r \to \infty$ in [R-01].** $\mathbb{E}[N] \to 1/q$. Retries cannot make a broken dependency
   work; they can only convert a hard failure into a slow failure plus load amplification. The
   ceiling is $1/q$, which is finite for any $q>0$ but reached only by unbounded waiting.
3. **$B_{\max} \to \infty$ in [M-01].** Memory becomes proportional to *backlog*, not to
   connections. This converts a latency problem into an OOM kill, which is strictly worse because
   it takes every connection down at once. An unbounded queue is an outage scheduler.
4. **$N \to \infty$ in [F-01].** Fan-out cost is linear in subscribers, so a per-subscriber
   payload reduction (delta, compression, batching) is often the only lever before link capacity.

**A counterexample that catches most teams.** "We use WebSockets for real-time, so we don't need
idempotency — messages are ordered." Ordering is not at-most-once. A reconnect after a `1006`
close replays every message the client did not acknowledge; if the client acknowledged at the
application level *after* applying the effect, the replay is harmless, but if it acknowledged on
receipt *before* applying, the replay duplicates effects. The protocol's ordering guarantee is
about delivery order within a connection; it says nothing about how many times you applied
something.

**Takeaway:** Every formula in this section is a ratio of "what arrives" to "what can be served".
The base case tells you the steady state, the general case shows the knee, the variants price the
mitigations, and the limits say which mitigation is the only one that still works when the
assumption fails.

## 5. Worked Numerical Example

**Example title:** Order-submission service, illustrative workload — sizing concurrency,
timeouts, idempotency storage and dedup correctness in one pass.

**Evidence label:** `ILLUSTRATIVE`. Every input below is a constructed teaching workload. The
arithmetic is `DERIVED` and independently checked; no figure is a measured production trace, and
no company is implied.

**Workflow and decision.** A checkout service accepts `POST /v1/orders`. The decision on the table
is: *how many workers, what timeout, how long to keep idempotency records, and whether the
current retry policy is safe* — with a go/no-go on the design rather than a vague "seems fine".

**Input data (all illustrative).**

| Quantity | Symbol | Value | Unit | Note |
|---|---|---|---|---|
| Peak order intent rate | $\lambda$ | 400 | intents/s | Sustained peak minute |
| Mean service time | $W$ | 0.050 | s | Validation + ledger write |
| $p_{99}$ service time | $W_{99}$ | 0.120 | s | Measured-style assumption |
| Per-attempt failure probability | $p$ | 0.15 | — | 5xx + timeouts from downstream |
| Retries configured | $r$ | 2 | attempts | Client-side |
| Client timeout | $T_c$ | 2.0 | s | Hard user-facing limit |
| Idempotency window | $T_{\text{idem}}$ | 86 400 | s | 24 h |
| Dedup record size | $b$ | 220 | B | Key + hash + status + body |
| Mean payload request | — | 1 800 | B | JSON order |
| Mean response | — | 900 | B | JSON acknowledgement |

**Simplifications, stated.** Steady state at peak; failures independent; retries issued by the
client only (no gateway retry); one region; ledger write is the effect owner and commits
atomically with the dedup row; no caching of order reads.

### Step 1 — In-flight concurrency at the mean [L-01]

```text
L = lambda * W = 400 * 0.050 = 20 orders in flight (mean).
```

### Step 2 — In-flight concurrency at the percentile that sizes the pool

Use the peak rate and the $p_{99}$ service time, because the pool must hold the *slow* requests
too, not only the average ones:

```text
L_99 = 400 * 0.120 = 48 orders in flight.
Headroom 30% -> 48 * 1.30 = 62.4 -> 63 workers.
```

The gap between 20 and 63 is the whole lesson of Section 4.1: the mean tells you what is typical,
the percentile tells you what you must survive.

### Step 3 — Queueing delay at the chosen capacity [Q-01]

The pool has 63 workers, so capacity is set by *mean* service time even though the pool was sized
on the $p_{99}$:

```text
Capacity mu = 63 / 0.050 = 1260 req/s.
rho = 400 / 1260 = 0.3175.
W_q = rho / (mu - lambda) = 0.3175 / (1260 - 400) = 0.3175 / 860 = 0.000369 s
    = 0.369 ms mean waiting time.
```

Mean total = $0.369 + 50 = 50.37$ ms — inside the 2 s client timeout with a 39.7× margin. That
margin is what absorbs a 120 ms request without the pool stalling. This is the *intended* steady
state. Section 4.2's non-linearity is why the design must also state what happens when $\lambda$
rises: at 1 100 req/s, $\rho = 0.873$, $W_q = 0.873/160 = 5.46$ ms; at 1 200 req/s,
$\rho = 0.952$, $W_q = 0.952/60 = 15.87$ ms — 2.9× worse for 9.1% more traffic.

### Step 4 — Retry amplification and observed load [R-01]

```text
E[N] = (1 - p^(r+1)) / (1 - p) = (1 - 0.15^3) / 0.85
     = (1 - 0.003375) / 0.85 = 0.996625 / 0.85 = 1.17250 attempts/intent.
```

Observed load on the service:

```text
lambda_obs = 400 * 1.17250 = 469.0 req/s.
Effective utilisation: 469.0 / 1260 = 0.3722  (was 0.3175 without retries).
Revised W_q = 0.3722/(1260 - 469) = 0.3722/791 = 0.4705 ms.
```

So retries cost 17.25% extra traffic and raise mean wait from 0.369 ms to 0.471 ms — acceptable
here. Now the same arithmetic with a second retrying layer, which is the common production
mistake:

```text
Two layers, each r = 2: observed multiplier 1.17250^2 = 1.37476
lambda_obs = 400 * 1.37476 = 549.9 req/s; rho = 0.4364
Worst-case attempts per intent = 3 * 3 = 9, not 3.
```

That is 37.5% extra load for a success probability improvement of
$0.996625^2 = 0.99326$ versus $0.996625$ — about 0.33 percentage points, bought with 150 req/s of
extra load on a service that is already failing 15% of the time. The decision: **retries at one
layer only**.

### Step 5 — Timeout budget versus total retry time [B-01]

```text
Nominal delays (d_0 = 0.100 s, cap = 30 s): 0.100, 0.200 s.
Cumulative backoff before attempt 3: 0.100 + 0.200 = 0.300 s.
Worst-case wall clock with full 2.0 s timeouts on every attempt:
   3 * 2.0 + 0.300 = 6.300 s per intent.
```

Against a 2.0 s client deadline, 4.3 s of that is spent *after the user has already given up*.
The corrective arithmetic is to size each hop from the remaining budget:

```text
remaining_budget(hop k) = D - sum(elapsed so far)
Attempt 1 timeout = 2.0 s. If it fails at t = 1.4 s with a connection reset,
backoff 0.1 s -> attempt 2 starts at t = 1.5 s with 0.5 s remaining,
so its timeout must be 0.5 s, not 2.0 s.
If a retry cannot be given at least a minimum useful timeout (say 0.3 s),
do not retry at all - fail fast and let the client reconcile.
```

This is what "deadline propagation" means arithmetically: the retry's timeout is the *remaining*
budget, not a fresh copy of the original. The same rule applies across services: a gateway with a
2 000 ms budget that has spent 200 ms must forward a 1 800 ms deadline, not its own 5 s default.

### Step 6 — Idempotency store size [D-01]

```text
m = 400 intents/s (mutating, deduplicated)
S = 400 * 86 400 * 220 = 7,603,200,000 B
7,603,200,000 / 1,073,741,824 = 7.0812 GiB
```

Add indexes and WAL overhead at a stated 30%: $7.0812 \times 1.30 = 9.206$ GiB. Decision: a
24 h window is affordable at this rate; document the window publicly and set the client's maximum
retry horizon below it (e.g. 6 h), so that a client can never legitimately need a record the
server has already evicted.

### Step 7 — Bandwidth for the read path

Checkout confirmation also pushes an order-status change to the customer's open page. Assume
12 000 concurrent viewers, each receiving 1 status event per 8 s at 300 B (after delta encoding):

```text
lambda_ev per viewer = 1/8 = 0.125 ev/s
B = 12 000 * 0.125 * 300 = 450,000 B/s = 3.60 Mbit/s
```

Comfortable. Before delta encoding, at 1 400 B per full event: $12\,000 \times 0.125 \times 1400 =
2\,100\,000$ B/s = 16.8 Mbit/s — still feasible, but 4.7× the egress, which is why payload shape
is a capacity lever rather than a cosmetic one.

### Step 8 — Decision and its consequence

| Decision | Value | Consequence if wrong |
|---|---|---|
| Worker pool | 63 (peak-rate × $p_{99}$ × 1.3) | 48 ⇒ no headroom; 20 ⇒ queueing at $p_{99}$ |
| Client timeout | 2 000 ms, propagated with remaining budget | Fixed per-hop timeouts ⇒ wasted work, duplicate retries |
| Retry policy | One layer, 1 retry, transient-only, deadline-bounded | Two layers ⇒ 37.5% amplification, 9 attempts worst case |
| Idempotency window | 86 400 s, documented; client horizon 21 600 s | Too short ⇒ duplicate orders after late retry |
| Dedup location | Ledger transaction (effect owner) | Gateway memory ⇒ duplicates after restart or across replicas |
| Status push | SSE with `id:` and `Last-Event-ID`; 300 B deltas | WebSocket ⇒ duplex complexity with no product need |

### Independent checks

1. **Dimension check.** [L-01] gives intents/s × s = dimensionless count ✓. [D-01] gives
   (intents/s)(s)(B) = B ✓. [F-01] gives subscribers × (ev/s) × (B/ev) = B/s ✓.
2. **Round-trip check.** 7 603 200 000 B ÷ 220 B = 34 560 000 records; 34 560 000 ÷ 86 400 s =
   400 records/s ✓ matches $\lambda$.
3. **Limit check.** As $p \to 1$, [R-01] gives $\mathbb{E}[N] \to 1/(1-p) \to \infty$ — consistent
   with "retries cannot fix a dead dependency". As $r \to 0$, $\mathbb{E}[N] = 1$ ✓.
4. **Monotonicity check.** $W_q$ must increase in $\lambda$: 400→0.369 ms, 1 100→5.46 ms,
   1 200→15.9 ms. Monotone increasing ✓, and convex, as the model requires.
5. **Cross-formula check.** Step 3's $\rho$ recomputed from Step 4's observed load matches the
   stated 0.372 within rounding (469.0/1260 = 0.37222 vs 0.372) ✓.

### The wrong path, made explicit

A plausible and wrong analysis:

```text
WRONG: "Mean latency is 50 ms and we have 400 req/s, so 400*0.05 = 20 workers.
        Add 10%: 22 workers."
```

Why it is wrong: it sizes the pool at the mean and ignores the service-time distribution. With 22
workers, capacity is $22/0.05 = 440$ req/s; at 400 req/s, $\rho = 0.909$,
$W_q = 0.909/40 = 22.7$ ms, and — worse — the 22nd worker is often occupied by a 120 ms request
while three shorter requests wait. The observable result is a $p_{99}$ far above $p_{99}$ service
time, which the team then misdiagnoses as "the database is slow". Correction: size from
peak-rate × $p_{99}$-service × headroom (Step 2), then verify with a load test that the measured
$p_{99}$ is within the target.

### Production-scale change

At 10× the rate (4 000 intents/s) nothing here scales linearly in effort: the dedup store becomes
70.8 GiB before overhead (Step 6 × 10), the worker pool becomes 630 (fine, but the ledger's write
capacity becomes the bottleneck long before the pool does), and the retry amplification of 172
req/s becomes 1 720 req/s of extra load — which is the number that turns a 15% failure into a
cascade. The design change at that scale is not "more workers"; it is admission control in front
of the ledger plus a queue with a declared max depth.

### What the example does not establish

It does not establish that the ledger can commit 469 writes/s (no measured write capacity is
assumed), does not model connection-pool limits, does not account for the gateway's own queue, and
says nothing about the tail beyond the $p_{99}$ input. Every one of those is a measurement the
reader must take before treating the numbers as a capacity plan.

**Takeaway:** The example's value is not its numbers but its *shape*: mean latency sizes nothing,
percentiles size pools, retry arithmetic sizes load, deadlines bound retries, and the dedup window
is a storage cost you choose and then publish.

## 6. Visual / Geometric Intuition

Four diagrams carry the intuition: the timeout ambiguity window, the utilisation knee, the
WebSocket lifecycle state machine, and the backpressure flow. Each is schematic, not to scale, and
each has a named key point.

### 6.1 The ambiguity window (why a timeout is not a failure)

```text
 time ->
 t0                      t_commit                t_response        t_deadline
  |                          |                       |                  |
  |--- request sent ------->|                       |                  |
  |                          |<-- effect applied --->|                  |
  |                          |        (ledger commit)|                  |
  |                          |                       |-- response ----->|
  |                                                                     |
  |<=========== client waits ===========>|                             |
  |                                      |                             |
  |                    client timeout fires at t = 2.0 s              |
  |                                      |                             |
  |       AMBIGUITY: did the ledger commit before t=2.0 s?            |
```

**Key point:** the commit at $t_{\text{commit}}$ and the deadline at $t_{\text{deadline}}$ are
independent events. When $t_{\text{commit}} < t_{\text{deadline}} < t_{\text{response}}$, the
client observes a timeout for an operation that *succeeded*. Any retry from the client will be a
second intent unless it carries a token that lets the ledger recognise the first.

**Interpretation.** The width of the ambiguity band is $t_{\text{response}} - t_{\text{deadline}}$,
and it is not a network property — it is whatever the server's slow tail happens to be. This is
why "we retry on timeout" without an idempotency key is a duplicate generator, and why the correct
diagnostic is to count commits in the ledger against successful responses at the client; the
difference is the ambiguity band made visible.

### 6.2 The utilisation knee

```text
 W_q / (1/mu)   relative waiting time vs utilisation (M/M/1)
   99 |                                                        *
      |                                                     *
   19 |                                              *
      |                                        *
    9 |                              *
      |                       *
    4 |              *
      |        *
    1 |*
      +----+----+----+----+----+----+----+----+----+----+----> rho
         0.50 0.60 0.70 0.80 0.85 0.90 0.95 0.97 0.98 0.99 1.00

  rho:    0.50  0.80  0.90  0.95  0.99
  W_q*mu: 1.0   4.0   9.0   19.0  99.0
```

**Key point:** the curve is $\rho/(1-\rho)$; it has no knee in the mathematical sense — it is
smooth and everywhere increasing — but it *looks* like a knee because the eye is linear and the
function is not. Between 0.50 and 0.80 waiting quadruples; between 0.90 and 0.99 it multiplies by
eleven for nine points of utilisation.

**Interpretation.** The "knee" is a perceptual artifact of linear plotting, and that is precisely
why capacity reviews mislead: a chart of CPU utilisation versus time looks calm at 92% while
waiting time is nine times service time. Plot *latency* against utilisation, logged on the y-axis,
and the relationship becomes a straight line with slope 1 near the boundary — the signature of a
queue.

### 6.3 WebSocket lifecycle as a state machine

```text
        [CONNECTING]
             |  TCP established
             v
        [HANDSHAKE] -- 4xx/invalid key --> [CLOSED_NEVER_OPENED]
             |  101 Switching Protocols
             v
        [OPEN] <-------------------------------+
          |  |                                  |
   send   |  |  recv (data/ping/pong)           | resume(cursor) OK
          |  |                                  |
          v  v                                  |
        [DRAINING] -- close frame sent --> [CLOSING] -- peer close --> [CLOSED]
             |                                  |
             |  no close frame (path died)       |
             +------------> [CLOSED 1006] ------+
                                   |
                                   v
                            [RECONNECT with backoff + jitter]
                                   |
                     cursor within T_replay? --no--> [GAP: resync needed]
                                   |yes
                                   +-------------> [OPEN]
```

**Key point:** `1006` is a terminal state reached *without* a closing handshake, and the only exit
is a new connection with a cursor. The state machine has two exits from `OPEN` that look identical
in a dashboard (both "disconnected") but require different remedies: a `1000`/`1001` close is
expected and resumable; a `1006` means you do not know whether the peer processed your last frame.

**Interpretation.** Drawing the state machine forces the two questions teams skip: *who assigns the
cursor* (must be the server, at durability) and *what is the maximum resumable absence* (must be
$T_{\text{replay}}$, published). A client that cannot answer "what do I do after `1006`" has an
undefined protocol, not a bug.

### 6.4 Backpressure flow, and what the queue does to latency

```text
 WITHOUT BOUND (unbounded queue)
 producer --> [ . . . . . . . . . . . . . . . . . . ] --> consumer(slow)
  1000/s                    grows without limit             100/s
  latency(t) = backlog(t)/100  ->  rises forever
  failure mode: memory exhaustion at t = O(memory / (900 B/s * frame))

 WITH BOUND (maxsize = 100) and shedding
 producer --> [ max 100 ] --full--> REJECT (429/1013) or COALESCE
  1000/s          |                    |
                  v                    v
              consumer(100/s)      client retries later
  latency <= 100/100 = 1 s  (bounded and *stated*)
  failure mode: visible, per-client, recoverable
```

**Key point:** the bounded queue converts an unbounded latency into a bounded latency plus a
visible rejection. Neither is "good"; the bounded one is *observable and per-client*, which is the
property that makes an incident diagnosable.

**Interpretation.** The arithmetic in the diagram is the whole design: with $\mu=100$ frames/s and
$B_{\max}=100$ frames, worst-case queueing delay is $B_{\max}/\mu = 1$ s. Choosing $B_{\max}$
*is* choosing the maximum latency you will impose before refusing. A team that has not chosen
$B_{\max}$ has not chosen a latency bound; it has chosen to find out at 3 a.m.

### 6.5 Frame layout, for the one time you need to debug it

```text
 RFC 6455 base framing (server -> client, unmasked)
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-------+-+-------------+-------------------------------+
 |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
 |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
 |N|V|V|V|       |S|             |   (if payload len == 126/127) |
 | |1|2|3|       |K|             |                               |
 +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
 |     Extended payload length continued, if len == 127          |
 + - - - - - - - - - - - - - - - +-------------------------------+
 | Masking-key (4 bytes), present only if MASK bit is 1          |
 +-------------------------------+-------------------------------+
 |                     Payload Data                              |
 +---------------------------------------------------------------+
```

**Key point:** `MASK` must be 1 for client→server frames and 0 for server→client. A server that
receives an unmasked client frame must fail the connection (`1002` protocol error). The masking
is not encryption — it defeats cache-poisoning intermediaries by making attacker-controlled bytes
non-deterministic on the wire. Worked example: masking key `37 fa 21 3d` applied to the ASCII
payload `Hello` (`48 65 6c 6c 6f`) gives `7f 9f 4d 51 58` (XOR each byte with the key cyclically).
`VERIFIED` by hand: $0x48 \oplus 0x37 = 0x7f$, $0x65 \oplus 0xfa = 0x9f$,
$0x6c \oplus 0x21 = 0x4d$, $0x6c \oplus 0x3d = 0x51$, $0x6f \oplus 0x37 = 0x58$.

**Interpretation.** You need this layout exactly twice in a career: when writing a minimal client
in a language with no library, and when a proxy is mangling frames. Having it on one page saves a
day; treating it as everyday knowledge is wasted effort.

### 6.6 The handshake, byte for byte

```text
 Client:                                Server:
 GET /chat HTTP/1.1
 Host: example.internal
 Upgrade: websocket
 Connection: Upgrade
 Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
 Sec-WebSocket-Version: 13
                                        HTTP/1.1 101 Switching Protocols
                                        Upgrade: websocket
                                        Connection: Upgrade
                                        Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

 Accept = base64( SHA1( key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11" ) )
```

**Key point:** `Sec-WebSocket-Key` is a 16-byte nonce, base64-encoded — not a credential. The
`Accept` value is a SHA-1 of the key concatenated with the fixed GUID, which proves the peer
understood the protocol rather than echoing a cached response. `VERIFIED`: for the key
`dGhlIHNhbXBsZSBub25jZQ==` the derived accept value is `s3pPLMBiTxaQ9kYGzzhZRbK+xOo=`.

**Interpretation.** Because the handshake is an ordinary HTTP `GET`, it traverses proxies and
authenticators like any other request — which is both the reason WebSockets deploy easily and the
reason *authorization done at handshake time is done exactly once for a long-lived session*.
Section 8, F-06 is the consequence.

**Takeaway:** The pictures say one thing each: a timeout is not a verdict, a "knee" is a plotting
illusion over a real divergence, `1006` is the state where you know nothing, and a queue bound is
a latency budget you set on purpose.

## 7. Comparison / Selection Table

The honest baseline is **plain HTTP polling**, not "REST". Polling is trivially deployable, needs no
new protocol, and — with conditional requests — is often good enough. Any more complex option must
beat it on a stated axis.

### 7.1 Transport comparison

| Option | Preconditions | Guarantee it actually provides | Cost model | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| HTTP polling (fixed interval) | None; any HTTP client | Eventually sees state, with lag up to the interval | Requests/s = viewers / interval; mostly wasted when idle | Low-frequency state (< 1 change/min), simple clients, caches | Interactive latency needed; poll rate would exceed change rate by >10× | `DERIVED` |
| HTTP conditional polling (`If-None-Match` → `304`) | ETag support | Same lag, near-zero bytes when unchanged | 304 responses are header-only (~200–300 B) | Dashboards that change rarely but must look live | Change rate high enough that most polls are 200 | `DERIVED` (RFC 9110) |
| HTTP long-poll | Server can hold requests; timeout discipline | Push-like latency, one message per request | One held connection per viewer + re-establish per event | Legacy clients, firewalls blocking upgrades | High event rate per viewer (reconnect churn dominates) | `DERIVED` |
| HTTP + SSE (`text/event-stream`) | HTTP/1.1 chunked or HTTP/2; proxy must not buffer | Server→client ordered stream with `id:`/`Last-Event-ID` resume | One long-lived response per viewer; text framing | Notifications, tickers, progress, log tails | Truly bidirectional need; binary payloads | `SOURCED` (WHATWG HTML spec) |
| WebSocket | Upgrade path end-to-end; stateful LB | Full-duplex framed messages; ordered within a connection | One persistent connection + per-connection server memory | Chat, collaborative editing, game state, bidirectional control | Only push needed; client cannot resume; connections ≫ traffic | `SOURCED` (RFC 6455) |
| Webhook (outbound HTTP callback) | Receiver exposes endpoint; retries owned by sender | At-least-once delivery with sender-defined retry | Sender retry load; receiver must dedupe | Third-party integrations, async job completion | Receiver cannot dedupe; ordering required | `DERIVED` |
| gRPC bidirectional streaming | HTTP/2 end-to-end; typed schemas; codegen | Multiplexed streams with flow control and deadlines | Streams per connection; HPACK header compression | Service-to-service, typed contracts, internal fleets | Browser clients without a proxy; simple public APIs | `DERIVED` (gRPC docs) |
| Message queue + worker (async job) | Durable broker; job id | Decouples acceptance from completion; retries survive restart | Broker storage + consumer capacity | Long-running work (> 1 s), spiky load | Caller needs the result synchronously | `DERIVED` |

### 7.2 The axis that actually decides it: who initiates, and for how long

```text
Q1: Does the client need to send unsolicited messages at arbitrary times?
    NO  -> Q2
    YES -> Q4

Q2: Is the required update latency < 1 s?
    NO  -> HTTP polling (with ETag) ; done.
    YES -> Q3

Q3: Is the stream one-way (server -> client)?
    YES -> SSE (id/Last-Event-ID resume, no upgrade complexity) ; done.
    NO  -> (impossible: one-way was answered yes) -- unreachable

Q4: Do both directions carry independent, low-latency messages?
    NO  -> HTTP request/response (client sends when it has something to say)
    YES -> Q5

Q5: Can you afford per-connection state, liveness detection, per-message
    authorization, bounded outbound buffers and a resume protocol?
    NO  -> re-scope the product to polling/SSE, or accept a degraded
           experience; do not ship a half-owned socket.
    YES -> WebSocket.
```

**What the flow chart refuses to say.** It never says "WebSockets are faster". For a 1 change/minute
dashboard with 500 viewers, polling costs $500/60 \approx 8.3$ req/s of header-only traffic;
a WebSocket would cost 500 persistent connections, 500 liveness timers, and a state machine — for
no product benefit. Conversely, for a collaborative editor, polling cannot represent "the other
person is typing" without a 10× traffic increase, and the socket is not an optimisation but the
only workable design.

### 7.3 Selection under conflicting constraints

```text
IF a hard constraint fails (no upgrade path through a required proxy, no
   stateful LB, binary incompatible client) -> disqualify that option outright.
ELSE IF the client cannot implement resume/dedupe -> prefer SSE or polling,
   because a WebSocket without resume turns every disconnect into data loss.
ELSE IF per-viewer event rate > 1/s and viewers > 10k -> check egress first:
   B = N*r*s; if B exceeds the link budget, fix payload/coalescing before
   choosing a transport.
ELSE IF latency target < 200 ms and one-way -> SSE.
ELSE IF latency target < 200 ms and bidirectional -> WebSocket.
ELSE -> HTTP request/response; it is the cheapest thing that can possibly work.
If two constraints conflict (e.g. bidirectional need + no per-connection state
budget) -> escalate as a product decision, not a transport decision.
```

### 7.4 Mutation-safety mechanisms compared

| Mechanism | What it prevents | What it does not prevent | Complexity | Choose when |
|---|---|---|---|---|
| `PUT` full replacement | Duplicate effect of the same body | Lost update (needs a validator) | Low | The client can send the complete target state |
| Idempotency key on `POST` | Duplicate effect for one intent | Two *different* intents that happen to be equal | Medium: persistent store + retention | Creating resources with server-assigned IDs |
| Conditional write (`If-Match`) | Lost update | Duplicate creation | Low–medium | Concurrent edits to an existing record |
| Client-generated ID | Duplicate creation | Conflicts on the same ID with different bodies | Low | Create-with-known-identity (e.g. UUIDv7 orders) |
| Transactional outbox | Effect committed but event lost (dual-write) | Duplicate consumers (needs consumer dedupe) | Medium–high | Any "write then publish" flow |
| Consumer dedupe on message ID | Duplicate delivery from the broker | Duplicate *effects* if the dedupe is not with the write | Medium | At-least-once delivery is the only option |
| Exactly-once illusion (log + idempotent sink) | Duplicates at the sink | Duplicates in side effects outside the sink | High | Streaming pipelines with transactional sinks |

**Takeaway:** Polling is the baseline that must be beaten on a stated axis; SSE wins every one-way
case that polling loses; a WebSocket is justified only by genuine bidirectionality *plus* a
willingness to own the session — and no transport choice can substitute for one of the four
mutation-safety mechanisms.

## 8. Failure Modes & Edge Cases

Eight failures are developed blow by blow. For each: the trigger and symptom as an operator sees
it, the competing causes that produce that *same* symptom, the discriminating check that separates
them, the bounded fix, the prevention that makes the class not recur, and its provenance. The
point of listing competing causes is that "check the logs" is not a diagnosis — most of these
symptoms have a decoy cause that a competent team will reach for first and be wrong about.

### F-01 Duplicate orders after a timeout

| Field | Detail |
|---|---|
| Trigger/symptom | Customer support reports two identical orders, seconds apart, both charged. Client saw one error toast. |
| Category | `VIOLATED_ASSUMPTION` (retry assumed to be a no-op) |
| Provenance | `HYPOTHETICAL` teaching scenario; mechanism is `DERIVED` from RFC 9110's non-idempotent `POST` semantics |

**Competing causes.**

1. *Client retried without a key.* The client library retried on timeout; the server had no token,
   so attempt 2 was a second intent. This is the modal cause and is invisible in server logs
   because both requests are individually valid.
2. *Gateway retried upstream.* An ingress with `retry_non_idempotent` style configuration replayed
   a `POST` after an upstream reset. Server logs show two requests with identical bodies and
   different correlation IDs.
3. *User double-clicked.* Genuinely two intents, milliseconds apart, from the same session. The
   distinguishing feature is that both requests arrive *before* any response returns.
4. *Queue redelivery.* The order was accepted, published to a queue, the consumer acked late, and
   the broker redelivered; the consumer created a second order row because its handler was not
   idempotent on message ID.

**Discriminating checks.**

```text
CHECK 1: Do the two requests carry the same Idempotency-Key header value?
         same  -> the key was ignored or not persisted at the effect owner (cause 1 or 2)
         absent-> the key was never sent; go to CHECK 2
CHECK 2: Compare client-observed timestamps of the two intents with the
         response arrival time of the first.
         both before first response -> double submission (cause 3)
         second after first response -> retry of some kind (cause 1, 2, or 4)
CHECK 3: Count order rows per (customer, sku, minute) and compare with the
         count of distinct message IDs processed.
         rows > message IDs -> consumer created two rows for one message (cause 4)
CHECK 4: Restart every gateway replica, replay the same key once, recount.
         second row appears -> the dedupe was in memory, not at the store.
```

**Bounded fix.** Add or repair a persistent idempotency record written in the *same transaction* as
the order, keyed on `Idempotency-Key` (or a client-generated order ID for create-with-identity),
with a unique constraint on the key. Return the recorded response on replay, including the
original status code. For cause 4, deduplicate on message ID inside the consumer's transaction.

**Prevention and verification.**

- Contract test: submit the same key twice against a freshly restarted replica; assert one row and
  byte-identical responses.
- Gateway configuration test: assert non-idempotent methods are not retried, or are retried only
  with a key.
- Retention test: a key older than $T_{\text{idem}}$ must be *rejected as expired*, not silently
  treated as new — otherwise the guarantee has a cliff instead of a boundary.
- Verification is at the ledger (row count per intent), never at the API response log.

**Residual risk.** A key reused with a *different* payload must return a conflict (RFC-style `422`
or `409`), not the original response; otherwise the record becomes a lie.

### F-02 Lost update / stale overwrite

| Field | Detail |
|---|---|
| Trigger/symptom | Two support agents edit the same ticket; the second save silently erases the first agent's notes. No error appears anywhere. |
| Category | `VIOLATED_ASSUMPTION` (concurrency assumed absent) |
| Provenance | `DERIVED` from RFC 9110 `If-Match`/`412` semantics |

**Competing causes.**

1. *No validator at all.* `PUT` overwrites unconditionally — last-writer-wins by accident rather
   than by decision.
2. *Validator exists but is not checked.* The server accepts `If-Match` and ignores it (a common
   bug: the header is parsed for a middleware that only handles `If-None-Match` caching).
3. *Validator is too coarse.* The `ETag` is computed over a list endpoint or the whole collection,
   so any unrelated change invalidates every write and clients retry blindly — producing lost
   updates under load or constant `412` storms.
4. *Read-modify-write split across two requests.* The client reads, shows a form, and writes the
   whole object minutes later. Even a correct `If-Match` protects only the target row; field-level
   edits by another user are overwritten with stale values from the form.
5. *Cache revalidation mistaken for concurrency control.* The `304` path is tested; the `412` path
   was never exercised, so a validation bug is invisible.

**Discriminating checks.**

```text
CHECK 1: Send PUT with If-Match: "wrong-version". Expect 412.
         got 200 -> validator ignored (cause 2)
CHECK 2: Send the same PUT twice with the same validator, no intervening change.
         both 200 -> validator not bound to state, i.e. constant (cause 3)
         first 200 then 412 -> correct behaviour, look elsewhere
CHECK 3: Inspect the audit trail: does the overwritten row's version increase by
         exactly 1 per successful write? A jump > 1 means unversioned writes exist
         on another path (cause 1).
CHECK 4: Replay the user's edit sequence with a 5-minute gap and an external
         field change in between; check whether the untouched field survived.
         no -> form-level overwrite (cause 4); fix with PATCH+field masks.
```

**Bounded fix.** Make the version a stored column, increment it in the same statement that writes
(`UPDATE ... SET v = v + 1 WHERE id = ? AND v = ?`), and return `412` on zero rows affected. For
form submissions, use a patch document scoped to the fields the user touched, and carry the
version the fields were read at.

**Prevention and verification.** A concurrency test that fires $n$ parallel writers at one row and
asserts exactly one success per version step; a schema test that every mutable resource exposes a
validator; and a rule that `PATCH` bodies are field-scoped, not `PUT`-shaped.

**Residual risk.** `412` handling is user-visible behaviour. If the client cannot merge, the correct
product behaviour is to show a conflict, not to retry automatically — an automatic retry of a
conflicting write is a lost update with extra steps.

### F-03 Slow consumer drives server memory to exhaustion

| Field | Detail |
|---|---|
| Trigger/symptom | At 03:00 the realtime service is OOM-killed; restart clears it. Connection count is normal. Heap grows at ~40 MiB/min. |
| Category | `SCALE_FAILURE` |
| Provenance | `DERIVED` from [M-01]; workload `ILLUSTRATIVE` |

**Competing causes.**

1. *Unbounded per-connection queue.* `asyncio.Queue()` (default `maxsize=0`) grows whenever the
   socket write blocks, so memory tracks backlog.
2. *Application-level fan-out loop that awaits a slow writer inline.* One slow client blocks the
   broadcast loop; other clients' pending events accumulate in *their* queues.
3. *Genuine leak.* Connection objects retained by a registry that never removes on abnormal close
   (typically `1006` paths that skip the cleanup coroutine).
4. *Compression buffer.* `permessage-deflate` enabled with a large window keeps per-connection
   zlib state; at 50 k connections this is tens of MiB even with small payloads.
5. *Kernel socket buffers counted as "ours".* Not usually the cause of process heap growth, but a
   frequent red herring: `ss -m` shows large send-queue bytes while the process heap is flat.

**Discriminating checks.**

```text
CHECK 1: Correlate heap growth with (a) connection count, (b) sum of per-connection
         queue depths, (c) sum of socket send-queue bytes.
         heap tracks queue depths -> cause 1 or 2
         heap tracks connection count -> cause 3 or 4
         heap flat, send-q large -> not our heap (cause 5)
CHECK 2: Dump the top-10 connections by queue depth. If 3-5 connections hold
         90% of the backlog -> slow-consumer concentration (cause 1/2).
CHECK 3: Compare objects-per-connection after 1 000 connect/disconnect cycles
         including forced 1006 closes. Growth -> cause 3.
CHECK 4: Disable permessage-deflate; re-measure baseline RSS per connection.
         Step change -> cause 4.
```

**Bounded fix.** Set `maxsize` on every per-connection queue; on overflow apply the declared
policy — for market-style data *coalesce* (keep only the newest value per key), for event streams
*close* with `1013` and let the client resume with a cursor. Cap total memory with a per-connection
byte budget (frames × max frame size), not a frame count alone, since frame sizes vary.

**Prevention and verification.** A load test with a deliberately stalled consumer population
(e.g. 5% of connections stop reading) asserting (a) RSS bound, (b) other connections' latency
unchanged, (c) a counter increment for every shed event. Alert on
`queue_depth_p99 / maxsize > 0.5` before it becomes an OOM.

### F-04 Events lost or duplicated across reconnect

| Field | Detail |
|---|---|
| Trigger/symptom | After a laptop sleep, a dashboard shows a stale price for 20 minutes, then jumps. Another client shows a duplicated trade. |
| Category | `CONCEPT_CONFUSION` ("TCP is reliable" applied to a session) |
| Provenance | `DERIVED` from RFC 6455 close/`1006` semantics and cursor reasoning |

**Competing causes.**

1. *No cursor.* Reconnect starts a fresh subscription; whatever happened during the gap is simply
   not sent.
2. *Cursor assigned client-side on receipt.* Events applied before the crash were counted as
   received (or vice versa); resume point is wrong in one direction or the other.
3. *Cursor accepted outside the replay window.* Server silently resumes at "now", creating a gap
   the client cannot detect.
4. *At-least-once delivery without idempotent application.* Replay delivers events the client
   already applied; the client applies them again (duplicate trade in the UI).
5. *Server restarts and restarts sequence numbers.* Client's cursor is now ahead of the server's
   latest, so the server sends nothing and the client believes it is current.

**Discriminating checks.**

```text
CHECK 1: On reconnect, log (client cursor, server latest id, replay window start).
         cursor < window_start -> the server cannot honour it (cause 3); the
         protocol must say so explicitly (gap error), not silently resume.
CHECK 2: Compare the count of applied events in the client with the count derived
         from the server's sequence range for the same interval.
         applied > derived -> duplicates (cause 4)
         applied < derived -> gap (cause 1/2/3)
CHECK 3: Force a server restart; check whether sequence numbers continue or
         reset. Reset -> cause 5; fix with a durable sequence source (or an
         epoch/instance id alongside the number, and reject cursors from a
         previous epoch).
CHECK 4: Interrupt the client (SIGKILL) mid-apply and restart; check whether the
         cursor advanced past the last *applied* event. It must not (cause 2).
```

**Bounded fix.** Server assigns monotonically increasing per-scope IDs at durability; the client
persists its cursor *after* applying effects (write-then-ack, not ack-then-write); the server
rejects out-of-window cursors with an explicit resync-required signal; the client treats every
delivered event as idempotent by ID.

**Prevention and verification.** A "chaos reconnect" test: kill the client at random points, kill
the server, and assert the reconstructed state equals the state derived from the full event log.
That equality is the only real verification of a resume protocol.

**Residual risk.** Cursor-after-apply means at-least-once, not exactly-once; exactly-once delivery
is not achievable end-to-end across a network, and any design claiming it has pushed the problem
into an idempotent sink.

### F-05 Retry storm / metastable failure

| Field | Detail |
|---|---|
| Trigger/symptom | A brief dependency blip turns into 40 minutes of elevated errors; load on the dependency is *higher* than normal. Recovery happens only after traffic is drained. |
| Category | `SCALE_FAILURE` (amplification + synchronised retries) |
| Provenance | `DERIVED` from [R-01], [B-01]; pattern is `HEURISTIC` from practice |

**Competing causes.**

1. *No jitter.* Every client that failed at the same instant retries at the same instant.
2. *Multiple retrying layers.* Client × gateway × SDK × database driver each retry, multiplying
   attempts (Section 5, Step 4).
3. *Retries on non-transient errors.* Retrying `400`, `401`, `403`, `409`, `422` or a deterministic
   business rejection adds load and never succeeds.
4. *Retry budget absent.* Retries are a fixed count, so a 1% error rate and a 90% error rate
   produce the same per-request multiplier — the opposite of what is wanted.
5. *Cached error responses mistaken for success by an intermediary* (a negative cache returning a
   stale success for a `GET` that would now fail), making the incident's traffic pattern
   uninterpretable.

**Discriminating checks.**

```text
CHECK 1: Plot dependency request rate against dependency error rate. If both rise
         together, amplification is real; if only errors rise, look elsewhere.
CHECK 2: Compare attempts per intent at each hop (client log, gateway log,
         service log) for the same trace id. Product > 2 -> multiple retrying
         layers (cause 2).
CHECK 3: Histogram retry timestamps modulo 100 ms. A spike -> synchronised
         retries (cause 1).
CHECK 4: List status codes that were retried. Any 4xx other than 408/429 ->
         cause 3.
CHECK 5: Compare retry count with the fraction of requests that could possibly
         benefit (errors eligible for retry). Fixed count regardless of error
         rate -> cause 4.
```

**Bounded fix.** Exactly one retrying layer; retries only on connection errors, timeouts, `429`,
`502`, `503`, `504`; full jitter; a *retry budget* (retries ≤ 10% of successful requests) rather
than a fixed count; and a circuit breaker that opens on sustained failure so the dependency can
drain. Add `Retry-After` respect for `429`/`503`.

**Prevention and verification.** A fault-injection test that returns `503` from the dependency for
60 s and asserts the dependency's own request rate never exceeds $1.2\times$ baseline; a dashboard
panel of attempts per intent; and an alert on budget exhaustion.

**Residual risk.** A circuit breaker introduces its own failure mode: an open circuit on one
instance while others are healthy produces uneven load. Keep the open threshold short and
half-open probes frequent.

### F-06 Authorization checked only at handshake, or only at route

| Field | Detail |
|---|---|
| Trigger/symptom | A user's permission is revoked; their open socket keeps receiving another tenant's data for hours. Or: an endpoint accepts a valid token whose subject is not the owner of the resource. |
| Category | `VIOLATED_ASSUMPTION` (authentication at connect ≠ authorization per message) |
| Provenance | `DERIVED` from RFC 6455's handshake design; consequence is `HEURISTIC` |

**Competing causes.**

1. *Handshake-only authn.* The token is validated once; the connection then outlives the token's
   expiry and any revocation.
2. *Subscription authorized at subscribe time, topic membership changed later.* The socket keeps a
   subscription it is no longer entitled to.
3. *Object-level authorization missing (IDOR).* The route is authorized by role but the resource
   ID in the path is not checked for ownership.
4. *Tenant confusion in a shared channel.* A broadcast channel keyed only on `symbol` rather than
   `(tenant, symbol)` delivers cross-tenant data to everyone subscribed to that symbol.
5. *Service-to-service calls skipping user context.* An internal RPC authenticates as the service,
   losing the end-user's permission, so the service can read anything.

**Discriminating checks.**

```text
CHECK 1: Revoke a permission mid-session and observe the next message.
         delivered -> no per-message check (cause 1/2)
CHECK 2: With a valid low-privilege token, request another user's resource id.
         200 -> object-level check missing (cause 3)
CHECK 3: Inspect channel keys in the subscribe path. Key without tenant ->
         cause 4. Confirm by subscribing two tenants to the same symbol and
         diffing payloads.
CHECK 4: Read the internal RPC's authorization context; absent user id ->
         cause 5.
```

**Bounded fix.** Authorize every privileged action at the point of effect with the acting
principal; treat subscription as an authorization decision and re-evaluate it on permission change
(push a revocation event to the session, or re-check at a bounded interval, stated in the contract);
scope every channel key by tenant; carry the end-user subject through internal calls.

**Prevention and verification.** An authorization test matrix (principal × resource × action)
executed against the socket path, not only the REST path; a "revocation within $T$ seconds"
assertion; and a tenancy test that asserts zero cross-tenant payload bytes.

### F-07 Unbounded request bodies, decompression bombs and frame-size abuse

| Field | Detail |
|---|---|
| Trigger/symptom | A worker OOMs on a single request; request body is 3 MiB compressed, 2 GiB expanded. Or a 2 GiB single-frame WebSocket message stalls a reader. |
| Category | `NUMERICAL_EDGE` (size limit missing or applied after the cost) |
| Provenance | `DERIVED` |

**Competing causes.**

1. *No `Content-Length` limit; server reads to EOF.* Memory follows attacker input.
2. *Limit applied after decompression.* The compressed byte cap passes, the expanded buffer is
   unbounded.
3. *Per-message limit absent on the socket.* RFC 6455 allows 64-bit payload lengths; a compliant
   server that does not cap them will attempt to allocate.
4. *Aggregate limit absent.* Individual bodies are 1 MiB, but 20 000 concurrent uploads are 20 GiB.
5. *Parser recursion.* Deeply nested JSON exhausts the stack before any size check.

**Discriminating checks.**

```text
CHECK 1: Send Content-Length = 10 GB with a tiny body; observe whether the
         server allocates on the header. Allocates -> cause 1.
CHECK 2: Send a 1 MiB gzip of 1 GiB of zeros; measure peak RSS. Peak ~1 GiB ->
         cause 2 (decompress-then-check).
CHECK 3: Send a single 100 MiB frame; observe buffer allocation and any 1009.
         No close with 1009 -> cause 3.
CHECK 4: Load test with many concurrent medium uploads and compare RSS with the
         single-request test. Superlinear -> cause 4.
CHECK 5: Send 10 000-deep nested JSON; stack trace shows parser recursion ->
         cause 5.
```

**Bounded fix.** Reject on declared length before reading; decompress with a hard output cap and
abort past it; cap frame size (`max_size`) and close with `1009` past it; cap aggregate concurrent
body bytes as a semaphore, not just per request; cap nesting depth in the parser.

**Prevention and verification.** Fuzz with length and nesting generators; assert a `413`/`1009` for
each oversize class; add an alert on peak request-body bytes per instance.

### F-08 Versioning and breaking-change drift

| Field | Detail |
|---|---|
| Trigger/symptom | A mobile client released 18 months ago starts failing after a "backwards-compatible" field change. |
| Category | `VIOLATED_ASSUMPTION` (additive change assumed safe) |
| Provenance | `HEURISTIC`; rationale below is `DERIVED` |

**Competing causes.**

1. *Field removal or rename.* The old client deserialises `null` and crashes.
2. *Type change* (number → string, int → float). Silently accepted by many parsers, wrong downstream.
3. *Semantic change.* Same field, same type, different unit or meaning (cents → dollars; local →
   UTC). The worst class because no schema check catches it.
4. *Stricter validation.* A field that used to be optional becomes required; old clients stop working.
5. *Enum growth.* A new enum value the old client cannot map; behaviour depends on default branch.

**Discriminating checks.**

```text
CHECK 1: Diff the schema registry for the window; classify each change as
         removal/rename, type, semantic, strictness, enum-growth.
CHECK 2: Replay a recorded corpus of requests from the oldest supported client
         version against the new server. Failures localise the class.
CHECK 3: For semantic changes, search for unit-conversion tests; absent tests
         are the gap.
CHECK 4: Check client behaviour on an unknown enum value (default branch vs
         exception) before shipping enum growth.
```

**Bounded fix.** Additive-only within a major version; never reuse a field name with a new meaning;
introduce new fields alongside deprecated ones with a stated removal date; treat unknown enum
values as "unknown" on the client; version at the representation level and document the support
window.

**Prevention and verification.** A contract test that runs the oldest-supported client against the
candidate server on every build; a schema-compatibility gate in CI that rejects removals; a
recorded-corpus replay as a release gate.

**Takeaway:** each of these eight symptoms has at least one plausible decoy cause that competent
teams pick first — which is why the discriminating check, not the fix, is the valuable part of
each entry.

## 9. Complexity & Cost Analysis

Two cost questions are asked about APIs: "how does this scale?" (algorithmic, Section 9a) and "what
will it cost to run?" (sizing and money, Sections 9b–9d). They are different units and must not be
added together.

### 9a. Algorithmic complexity

Define the size variables once. $n$ = items in a collection; $p$ = page size; $k$ = number of
keys in a request; $c$ = concurrent connections; $m$ = message size in bytes; $h$ = header bytes;
$D$ = distinct idempotency keys retained; $e$ = events retained for replay.

| Operation | Input sizes | Time | Space | I/O or communication | Dominant bottleneck / assumption |
|---|---|---|---|---|---|
| Idempotency check (indexed unique key) | $D$ retained keys | $O(\log D)$ per lookup | $O(D)$ total store | 1 index probe + 1 row read | B-tree index residency; hash index gives expected $O(1)$ |
| Idempotency check (unindexed scan) | $D$ | $O(D)$ | $O(1)$ extra | full scan | Immediate red flag: lookup cost grows with window length |
| Conditional write | 1 row | $O(\log n)$ to locate + $O(1)$ compare-and-set | $O(1)$ | 1 write + 1 version read | Row lock contention at hot keys |
| Offset pagination | $n$, offset $o$, page $p$ | $O(o + p)$ | $O(p)$ | reads $o+p$ rows | Cost grows with page depth; `OFFSET 1000000` is a scan |
| Keyset pagination | $n$, page $p$ | $O(p \log n)$ | $O(p)$ | 1 index range scan | Needs a stable total order; unstable sort breaks it |
| JSON serialisation | message of $m$ bytes | $O(m)$ | $O(m)$ (or streaming $O(1)$) | — | Allocation per request dominates at high RPS |
| Response compression | $m$ bytes | $O(m)$ time, but with a large constant | $O(m)$ window | trades CPU for egress | Ratio depends on payload redundancy |
| SSE fan-out | $c$ subscribers, event size $s$ | $O(c)$ per event | $O(c \cdot B_{\max})$ | $O(c \cdot s)$ bytes/event | Egress is linear in subscribers |
| WebSocket fan-out | $c$ subscribers | $O(c)$ per event | $O(c \cdot B_{\max})$ | same as SSE, plus framing | Per-connection memory is the scaling wall |
| Replay from cursor | $e$ retained, $d$ behind | $O(d)$ | $O(1)$ streaming | $O(d \cdot s)$ bytes burst | Burst capacity, not steady state, sizes it |
| Heartbeat sweep | $c$ connections | $O(c)$ per interval | $O(1)$ amortised | 1 frame each | Timer-wheel beats one timer per connection |
| Dedup garbage collection | $D$ expired | $O(D_{\text{expired}})$ | $O(1)$ | batched deletes | Delete rate must track insert rate |
| Handshake | 1 connection | $O(1)$ (SHA-1 over 60 B + base64) | $O(1)$ | 1 round trip | Negligible vs TLS; do not model it as a cost |

**The two complexity traps in this topic.**

1. *Retention turns an $O(1)$ check into a growing one.* An idempotency lookup is $O(\log D)$, and
   $D = m T_{\text{idem}}$ grows linearly with the window. A 7-day window at 2 000 writes/s is
   $1.2096\times 10^{9}$ rows; the lookup is still logarithmic, but the *store* is
   $1.2096\times10^{9} \times 200\ \text{B} = 2.4192\times10^{11}$ B = 225.3 GiB and the GC deletes
   2 000 rows/s forever. Cost scales with the window even though latency does not.
2. *Pagination depth is a hidden linear cost.* `OFFSET 500000 LIMIT 20` reads 500 020 rows to
   return 20. Keyset pagination makes the same request $O(p \log n)$. The tell is that latency
   grows with page number while RPS is constant.

### 9b. Back-of-envelope sizing, with units

**Workload (illustrative).** 20 000 concurrent subscribers; 2 events/s each; 512 B per event;
heartbeat every 25 s; 400 req/s of mutating REST traffic at peak; 24 h idempotency window; 220 B
per dedup record; 300 s replay window.

```text
[1] Egress:            B = 20 000 * 2 * 512 = 20 480 000 B/s
                       = 19.53 MiB/s = 163.84 Mbit/s
[2] Ingress (acks):    20 000 * (1/25) * 6 B = 4 800 B/s  (pongs, tiny)
[3] Connection memory: M = 20 000 * (256*512 + 24 576)
                       = 20 000 * 155 648 = 3 112 960 000 B = 2.899 GiB
                       (256-frame buffer of full 512 B frames)
[4] Replay store:      shared broadcast stream retained once:
                       40 000 ev/s * 300 s * 512 B = 6 144 000 000 B
                       = 5.722 GiB  (40 000 = 20 000 subscribers * 2 ev/s)
                       per-subscriber state is only a cursor: 20 000 * 8 B
                       = 160 000 B, negligible.
[5] Dedup store:       400 * 86 400 * 220 = 7 603 200 000 B = 7.081 GiB
[6] Handshake CPU:     20 000 handshakes over a 10 min reconnect storm
                       = 33.3/s; at ~50 microseconds CPU each = 1.7 ms/s
                       = 0.17% of one core. Not a bottleneck.
```

**Note on [4], which is the interesting one.** Storing replay history *per subscriber* and storing
the *broadcast stream once* consume the same bytes here — 12 000 000 events either way, because
each of the 20 000 subscribers receives 600 of the 12 000 000 broadcast events. The difference is
in the write path, not in storage: a per-subscriber queue requires $c$ writes per published event
(40 000 × 20 000 = $8\times10^{8}$ writes/s, impossible), while a shared append-only log requires
one. Per-subscriber queues also cannot serve a resume cursor older than their own depth, so the
replay window silently becomes "whatever we happened to buffer". The correct architecture is a
single append-only broadcast log plus one 8-byte cursor per subscriber: appends scale with the
event rate, resumes scale with the subscriber count, and the replay window is a property of the log
rather than of each connection's luck. This is a data-model decision, and no transport choice can
substitute for it.

**Bottleneck identification.** Egress at 163.84 Mbit/s is likely the first hard limit; per-connection
memory at 2.9 GiB is the second; dedup store growth is a slow cost rather than a wall. Confirm by
measuring NIC egress and process RSS separately under load — do not infer from total CPU.

### 9c. Cost model in money (illustrative unit prices)

Costs below use *stated hypothetical* unit prices so that the arithmetic is reproducible. They are
not provider quotes and should be replaced with current prices when used for a real decision.

Let $P_{\text{egress}}$ = price per GiB egress, $P_{\text{store}}$ = price per GiB-month,
$P_{\text{compute}}$ = price per vCPU-hour.

```text
FORMULA [C-01]: Cost_month = (E_GiB * P_egress) + (S_GiB * P_store)
                            + (H_vCPU * P_compute)
WHERE: E_GiB monthly egress [GiB/month]; S_GiB stored GiB [GiB];
       H_vCPU compute hours [vCPU-h/month]; prices in currency/GiB or /h.
CONSTRAINT: steady-state rates, no free tier, no committed-use discount,
            excludes support, inter-zone replication, and labour.
NUMERICAL EXAMPLE: E_GiB = 20 480 000 B/s * 2 592 000 s/month / 2^30
                   = 49 438.48 GiB/month; at P_egress = 0.09 -> 4 449.46
                   S_GiB = 7.081 (dedup) + 5.722 (replay) = 12.803 GiB
                   at P_store = 0.10 -> 1.28
                   H_vCPU = 8 vCPU * 730 h = 5 840 h at 0.05 -> 292.00
                   Total = 4 742.74 currency units/month; egress is 93.8%.
INTERPRETATION: egress dominates; a 50% payload reduction saves far more
                than any compute optimisation, which is the usual shape for
                push-heavy services.
```

### 9d. Where the money actually goes, and the levers

| Cost | Driver | Lever | Boundary |
|---|---|---|---|
| Egress | subscribers × events/s × bytes | Delta encoding, coalescing, compression, filtering by entitlement | Compression costs CPU and latency; coalescing loses intermediate states |
| Per-connection memory | connections × buffer × frame size | Smaller $B_{\max}$, byte budget, coalescing | Too small ⇒ more shedding/`1013` closes |
| Dedup storage | writes/s × window × record size | Shorter documented window; hash-only records | Window is a client-facing promise |
| Replay storage | event rate × window × size | Snapshot + cursor for long absences | Snapshots require a materialised state per entity |
| Compute | connections × per-connection CPU | Fewer, larger frames; efficient serialisation | Batching increases latency |
| Request-serving | RPS × per-request bytes | `304`s, caching, pagination with keyset | Cache staleness is a product decision |
| Operational | incidents × MTTR | Observability of the seven counters | Monitoring is itself a cost; instrument the decision points |

**Takeaway:** complexity tells you what grows with the window and the subscriber count; sizing tells
you which resource hits the wall first (usually egress, then per-connection memory); money follows
egress and window length, and the levers that reduce them are payload shape and retention policy —
both of which are design decisions, not infrastructure purchases.

## 10. Key Theorems / Results Reference

Section 10 is `CONDITIONAL` for a systems sheet, and it is *relevant* here because the topic rests
on two specification-defined guarantees and three results from queueing and protocol theory.
Nothing below is decorative; each row has an overclaim attached to it.

| Result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Source |
|---|---|---|---|---|---|
| Little's law | For a stable system in steady state, time-average occupancy equals arrival rate times mean time in system: $L=\lambda W$. | Consistent in/out boundary; no leak; steady state over the window. | Concurrency sizing, capacity reasoning from latency and rate. | Nothing about the distribution, the tail, or transients. | Standard queueing result; derivation in 4.1 |
| M/M/1 waiting time | $W_q = \rho/(\mu-\lambda)$ for Poisson arrivals, exponential service, one server, infinite buffer, FCFS. | Markovian arrival/service; $\rho<1$; FCFS; infinite buffer. | Explains non-linear latency near capacity; gives the $\rho/(1-\rho)$ shape. | Not a general latency predictor; bursty or heavy-tailed service invalidates it quantitatively. | Standard queueing theory |
| M/M/m approximation | $W_q \approx \frac{C\rho^{\sqrt{2(m+1)}-1}}{m(1-\rho)}\cdot\frac{1}{\mu}$. | $m$ identical servers, common queue, no priorities; accuracy degrades at $\rho\to1$. | Why pooling raises capacity but not the shape of the knee at fixed $\rho$. | Does not mean "more servers fixes latency at fixed utilisation". | Queueing approximation, labelled `HEURISTIC` |
| HTTP method semantics | `GET`/`HEAD`/`OPTIONS`/`TRACE` are safe; `GET`/`HEAD`/`OPTIONS`/`TRACE`/`PUT`/`DELETE` are idempotent; `POST`/`PATCH` are neither by default. | Method used per its definition; server implements the semantics. | Which methods may be retried automatically, and where a key is required. | Does not make a `POST` retry-safe in practice if the server ignores keys; does not make `PUT` safe when concurrent. | RFC 9110 §9.2 |
| Conditional request outcome | A failed precondition yields `412 Precondition Failed`; a matched `If-None-Match` on `GET` yields `304 Not Modified`. | Validators are correct and compared with the specified weak/strong rules. | Concurrency control and cache validation. | A `304` proves nothing about the *current* server state beyond the validator. | RFC 9110 §13 |
| WebSocket opening handshake | The server returns `101` with `Sec-WebSocket-Accept = base64(SHA1(key + GUID))`. | HTTP/1.1 request with `Upgrade: websocket`, version 13. | Verifies protocol understanding, not identity; traverses HTTP infrastructure. | The handshake is *not* authentication, and success does not authorize any subsequent message. | RFC 6455 §4 |
| WebSocket framing | Frames carry FIN/opcode/length; client→server frames MUST be masked; a server receiving an unmasked client frame MUST fail the connection. | Compliant peer; correct masking. | Explains `1002` protocol errors and proxy corruption symptoms. | Masking is not encryption and provides no confidentiality. | RFC 6455 §5 |
| Close codes | `1000`–`1015` are defined; `1005` and `1006` MUST NOT be sent on the wire, and `1006` is for local reporting of an abnormal closure. | Compliant implementation. | `1006` means "no close frame received" — a diagnosis of missing information, not a cause. | Does not identify whether the peer crashed, the path died, or a proxy reset. | RFC 6455 §7.4 |
| At-least-once is the achievable end-to-end guarantee | Across an unreliable network with retries, exactly-once *delivery* is not achievable; exactly-once *effect* requires idempotence at the sink. | Independent failures, no reliable end-to-end transaction. | Design for idempotent effects rather than perfect delivery. | Does not forbid exactly-once *processing* within a transactional boundary (e.g. a broker with transactional sinks). | `DERIVED` argument; consistent with distributed-systems literature |
| TCP ordering scope | TCP delivers bytes in order and without gaps *within one connection instance*. | Same connection, no reset. | Ordering must be reconstructed by the application across reconnects. | Says nothing about message boundaries, cross-connection order, or application-level duplication. | RFC 9293 (TCP) |

**How to use this table in an argument.** When someone says "TCP guarantees delivery so we don't
need sequence numbers", the rebuttal is row "TCP ordering scope": the guarantee is scoped to a
connection instance and to bytes, and a reconnect is a different instance. When someone says
"we use a message queue, so exactly-once is handled", the rebuttal is row "at-least-once": the
queue can give you exactly-once *within its own transactional boundary*, but your side effect
outside that boundary still needs an idempotency key.

**What no theorem here gives you.** None of these results says anything about your *business*
correctness. A protocol-compliant `201` is not a payment; a `1000` close is not a completed
session; a replayed event is not a re-applied effect unless you made it so.

**Takeaway:** three results (Little, M/M/1, at-least-once) and three specification scopes (method
semantics, handshake, close codes) cover every argument in this sheet — and each row's "does not
imply" column is the sentence to quote in a design review.

## 11. Interview / Exam Questions

Twelve questions across recall, application and judgement. Each has a checked answer, the
reasoning that earns credit, a follow-up, an accepted alternative, and a red flag. At least seven
require calculation, selection or diagnosis.

### Q1 — Definition / Recall. Is `POST` idempotent?

**Prompt.** "You retry a `POST` after a timeout. Is that safe?"

**Answer.** `POST` is neither safe nor idempotent by default (RFC 9110 §9.2.1). Retrying it is safe
*only* if the server implements an idempotency mechanism — a client-supplied key persisted with the
effect, a client-generated resource ID that the store enforces unique, or an equivalent
natural-key constraint. The transport retry does not make it safe; the server's effect semantics
do. `PUT` and `DELETE` are idempotent by definition, which is a different guarantee: it holds only
when `PUT`'s body is the complete target state and `DELETE`'s second call is tolerated (a `404` on
the second delete is acceptable behaviour for an idempotent operation).

**Follow-up.** "What about `PATCH`?" — Not defined as idempotent; a patch like `{"op":"increment"}`
is explicitly not. A patch of the form "set field to value" is idempotent *by accident*.

**Red flag.** "Retrying `POST` is fine because TCP won't duplicate the packet." TCP retransmission
is invisible and does not duplicate at the application layer; the *application* retry after a
timeout is a new request on a new or reused connection. Conflating the two is the classic error.

### Q2 — Recall. What does `1006` mean?

**Prompt.** "Your dashboards show 92% `1006` closes. What is happening?"

**Answer.** `1006` is a *reserved* code that must never be sent on the wire; it is reported locally
when a connection closes without receiving a close frame (RFC 6455 §7.1.5). So a 92% `1006` rate
means: 92% of connections end without a closing handshake. It is a diagnosis of *missing
information*, not a cause. Candidates: half-open connections through a NAT/load balancer idle
timeout, client process crash, network path loss, proxy reset, or server crash.

**Follow-up.** "How would you reduce it?" — Application-level ping/pong with a deadline shorter
than the middlebox idle timeout, so that the *server* detects death first and closes with a real
code; and client-side close on page unload where possible.

**Red flag.** "1006 means the client disconnected abruptly." It means *no close frame was
received* — the peer may have sent one that never arrived. The distinction decides whether you look
at the client or at the path.

### Q3 — Recall. Why does a `304` not mean "the resource is unchanged"?

**Prompt.** "Our client got `304 Not Modified`, so we know the record is unchanged — right?"

**Answer.** `304` means the *validator you sent* still matches what the server has for that
representation under the specified comparison rules (RFC 9110 §13.1.2, §13.2.2). It does not mean
nothing changed anywhere: other representations, other fields not covered by the entity tag, or
the underlying domain state can have changed. Weak entity tags (`W/"..."`) explicitly permit
semantic-equivalent-but-not-byte-identical bodies.

**Follow-up.** "Where would this bite?" — A conditional `GET` used as a cheap polling mechanism for
a *derived* value whose inputs changed but whose serialised form did not.

**Red flag.** Treating `ETag` as a version of the *row* rather than of the *representation*.

### Q4 — Apply. Size a worker pool.

**Prompt.** 400 requests/s at peak, mean service 50 ms, $p_{99}$ service 120 ms. How many workers,
and what is the mean queueing delay at that size?

**Answer.** Size from peak rate × $p_{99}$ service × headroom:
$400 \times 0.120 \times 1.30 = 62.4 \rightarrow 63$ workers. Capacity
$\mu = 63/0.050 = 1260$ req/s (mean service), $\rho = 400/1260 = 0.3175$,
$W_q = \rho/(\mu-\lambda) = 0.3175/860 = 0.369$ ms. Total mean ≈ 50.37 ms. Sanity check: the pool is
5× the mean in-flight count of 20, so it has ample margin at the mean and just enough at the
$p_{99}$.

**Accepted alternative.** "Size for the $p_{99}$ arrival rate too, and verify with a load test" —
better than a pure formula answer, provided they state that $W_q$ from M/M/1 is a shape, not a
prediction.

**Follow-up.** "Traffic rises 10%. What happens?" — $\lambda = 440$, $\rho = 0.349$,
$W_q = 0.349/820 = 0.426$ ms. Almost nothing, because utilisation is low. The same 10% at $\rho=0.90$
takes $W_q$ from 9× to 11× service time. **The lesson is that headroom is what makes the system
insensitive to traffic growth, and the formula quantifies it.**

**Red flag.** $400 \times 0.050 = 20$ workers. That is the mean in-flight count, and it leaves no
room for the 120 ms requests that occupy a worker for 2.4× longer.

### Q5 — Apply. Compute retry amplification across two layers.

**Prompt.** Client retries twice, gateway retries twice. Per-attempt failure probability 15%. What
is the load multiplier and the worst-case attempts?

**Answer.** Per-layer $\mathbb{E}[N] = (1-0.15^3)/0.85 = 1.1725$. Two independent layers:
$1.1725^2 = 1.3748$, so observed load is 37.5% above intent rate. Worst case attempts per intent is
$3 \times 3 = 9$. Success probability $0.996625^2 = 0.99326$ — a 0.33-percentage-point improvement
over one layer for 20 percentage points of extra load, on a dependency that is already failing 15%
of the time.

**Follow-up.** "What is the right architecture?" — One retrying layer, a retry budget rather than
a fixed count, jitter, and a circuit breaker. Justification: $\mathbb{E}[N]$ is bounded by $1/q$,
so retries cannot exceed $1/q$-fold amplification in expectation, but *synchronised* retries can
exceed it transiently and that transient is what causes the outage.

**Red flag.** Adding retries to improve reliability without computing the load multiplier.

### Q6 — Apply / derive. Why is the idempotency window a storage decision?

**Prompt.** 2 000 mutating requests/s, 220 B per record. What does a 7-day window cost, and what
would you do instead?

**Answer.** $2000 \times 604800 = 1.2096\times10^{9}$ records;
$\times 220 = 2.66112\times10^{11}$ B = 247.8 GiB (payload only). Plus indexing/WAL overhead at
30% ⇒ ≈ 322 GiB. The alternative is not "make it smaller arbitrarily": the window is a *promise* to
clients, so choose the window from the maximum plausible retry horizon, publish it, and add
reconciliation for intents older than the window.

**Follow-up.** "The client retries after 8 days. What happens?" — With a 7-day window the record is
gone, so the retry creates a second effect. Correct behaviour: reject keys older than the window
with an explicit error class, or require a client-generated ID that the store enforces as a
permanent unique key (no window at all, at the cost of unbounded key storage or a natural-key
design).

**Red flag.** Choosing the window from storage convenience without publishing it.

### Q7 — Diagnose. Duplicate orders, three plausible causes.

**Prompt.** Two identical orders, same customer, 1.4 s apart. What do you check first?

**Answer.** First establish whether the two requests carried the same idempotency key. If yes, the
key was ignored or not persisted at the effect owner (or was stored in gateway memory and lost on a
different replica). If no, determine whether the second request was issued by the client library
(retry on timeout), by the gateway (upstream replay), or by the user (double submit) — the timing
relative to the *first response* separates them: a second intent issued before any response is a
double-submit; one issued after a response is impossible as a retry of a successful call, so it is
a second user action; one issued after a *timeout* without a response is a retry. Then check the
queue consumer path for redelivery without dedupe.

**Follow-up.** "How would you prove the fix?" — Restart all gateway replicas, replay the same key,
assert one row and an identical response body; plus a message-ID dedupe test in the consumer.

**Red flag.** "Add a uniqueness constraint on (customer, sku, amount)." That prevents the symptom
by forbidding legitimate repeat purchases: two identical orders a week apart are valid.

### Q8 — Diagnose. Memory grows, connections normal.

**Prompt.** Realtime service RSS grows 40 MiB/min; connections are flat at 20 000; CPU normal. Where
do you look?

**Answer.** Correlate heap with (a) sum of per-connection queue depths, (b) connection count,
(c) socket send-queue bytes. Heap tracking queue depth ⇒ unbounded per-connection queues or a
fan-out loop awaiting a slow writer inline. Heap tracking connection count ⇒ leaked connection
objects on abnormal close (`1006` skips cleanup) or per-connection compression state. Heap flat but
`ss -m` shows large send-queues ⇒ kernel buffers, not the process heap.

**Follow-up.** "What bound would you set?" — $B_{\max}$ from a latency target:
$B_{\max} = \text{target wait} \times \mu$. For 1 s max wait at 100 frames/s per connection,
$B_{\max} = 100$ frames.

**Red flag.** "Increase the heap / restart on a cron." Both convert a bounded design bug into an
unbounded operational cost.

### Q9 — Judgement. WebSocket or SSE?

**Prompt.** A trading dashboard: 5 000 users, 30 price updates/s, no client→server data besides
subscription changes. What do you choose?

**Answer.** SSE. The data is one-way; subscription changes are infrequent and fit an ordinary
`POST`. Egress: $5000 \times 30 \times s$; at 160 B deltas, $24\,000\,000$ B/s = 192 Mbit/s — a
capacity conversation, not a protocol one. SSE gives ordered delivery, `id:`/`Last-Event-ID` resume
through ordinary HTTP, no custom framing, and no per-message authorization burden. WebSocket would
add duplex framing, masking, ping/pong, close-code handling, and a stateful load balancer for zero
product benefit.

**Accepted alternative.** WebSocket if there is a credible near-term requirement for client→server
low-latency messages (order entry from the same surface). Then the justification is future
bidirectionality, stated honestly.

**Follow-up.** "The proxy buffers responses." — Disable buffering (`X-Accel-Buffering: no`-style
configuration at the proxy) and verify with a heartbeat event; a buffered SSE stream looks like a
dead connection.

**Red flag.** "WebSocket is faster." It has less per-message overhead; it is not faster in any way
that matters at 30 updates/s, and it is more expensive to operate.

### Q10 — Judgement. Lost update vs duplicate.

**Prompt.** Two admins edit one record; the second save erases the first's change. Duplicate
protection? Versioning? Both?

**Answer.** This is a lost update, not a duplicate: two *distinct* intents, both applied, one
overwriting the other. Idempotency keys do nothing here. The fix is a validator: a stored version,
`If-Match` on write, `412` on mismatch, and a client that re-reads or merges. If the UI saves whole
objects, also switch to field-scoped patches, because a row validator cannot protect fields the
user never saw.

**Follow-up.** "What if the client cannot merge?" — Show the conflict to the user. Silent
last-writer-wins is a product decision that must be made explicitly and labelled in the contract.

**Red flag.** "Add an idempotency key." It solves a different problem and will pass the test while
the lost update continues.

### Q11 — Judgement / verification. How do you prove the resume protocol works?

**Prompt.** You claim "no events are lost or duplicated on reconnect". What test establishes it?

**Answer.** State space, not a happy-path test. Kill the client at random points (including between
apply and cursor-persist), kill the server, expire the replay window, and then compare the
reconstructed client state against the state derived from the authoritative event log for the same
interval. Equality of *state* is the assertion; counting acks is not. Add: reconnect with a cursor
older than the window and assert an explicit gap signal rather than a silent resume; restart the
server and assert the sequence does not reset (or that the epoch invalidates old cursors).

**Follow-up.** "Delivery is at-least-once. Is the test still valid?" — Yes, and it must be, because
the assertion is on *state after idempotent application*, not on delivery count. A test that asserts
"exactly one delivery" is testing something unachievable.

**Red flag.** "We log every event, so we can always replay." Log retention is not a resume protocol:
the cursor, the window boundary, and the gap signal are the protocol.

### Q12 — Changed constraint. The dependency goes down for 10 minutes.

**Prompt.** Your retry policy is 3 retries with 100 ms base backoff, no jitter, no breaker. The
dependency is down for 10 minutes. What happens, and what would you change before the next
incident?

**Answer.** Without a breaker, every request spends
$0.1 + 0.2 + 0.4 = 0.7$ s in backoff plus 4 timeouts, then fails: callers hold resources longer,
thread/connection pools fill, and the failure spreads upstream. Without jitter, all callers retry
in phase, producing load spikes at 100/300/700 ms after each failure wave. During a 10-minute
outage the dependency receives continuous load and cannot recover its caches or connections when it
comes back — the classic metastable pattern. Changes: jitter, a retry budget, a circuit breaker with
half-open probes, respect `Retry-After`, and a bulkhead so that the failing dependency cannot
consume the whole pool.

**Follow-up.** "How do you verify?" — Fault-inject 60 s of `503` and assert the dependency's
observed request rate never exceeds $1.2\times$ baseline, and that the breaker opens within a stated
number of failures.

**Red flag.** "Increase retries to 5." $\mathbb{E}[N] \to 1/q$; if the dependency is fully down,
$q \approx 0$ and retries only add load and latency.

### Q13 — Recall / selection. Which HTTP status for a duplicate key?

**Prompt.** A client submits an idempotency key that already exists but with a *different* payload.
What do you return?

**Answer.** A `4xx` conflict — `422 Unprocessable Content` or `409 Conflict` — with a stable error
code identifying key reuse. Returning the original response would be a lie: the recorded outcome
does not describe the request that was sent. The contract must state this explicitly, because
clients otherwise assume "same key ⇒ same response" unconditionally.

**Follow-up.** "Same key, same payload?" — Return the recorded response, including its original
status code, and mark it as a replay in a header or field so that operators can distinguish new
work from replay.

**Red flag.** "Return `200` with an empty body." It hides the conflict and teaches clients nothing.

### Q14 — Judgement. Is "one connection per app tab" a good design?

**Prompt.** Each browser tab opens its own WebSocket. 5 tabs per user, 50 000 users. What breaks?

**Answer.** Connection count becomes 250 000 rather than 50 000: per-connection memory
$250\,000 \times (\text{buffer} + \text{overhead})$ — using Section 4.3(e) with 256 frames × 512 B
and 24 KiB overhead, $250\,000 \times 155\,648$ B = 36.24 GiB, which is a different deployment. Tab
suspension (mobile background) also produces `1006` storms on wake. The alternative is one
connection per browser profile shared across tabs (a service worker or a shared worker), or SSE with
per-tab subscriptions to a shared stream.

**Follow-up.** "What does that cost you?" — A shared worker adds coordination complexity and a
single point of failure per profile; the tradeoff is stated rather than assumed.

**Red flag.** "Connections are cheap." They are cheap per unit and expensive in aggregate, which is
the definition of a scale wall.

**Takeaway:** the interview set has one theme: every answer that says "add a mechanism" must also
say what the mechanism costs and what it does *not* protect.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Choosing and specifying an interaction contract between processes: REST request/response versus streamed/duplex channels, with retry, concurrency and buffering semantics attached. |
| Core rule | Transport is chosen by directionality and lifetime; *effect semantics* are chosen separately — one idempotency token per intent, persisted at the effect owner, replayed from the store (not from gateway memory). |
| Core formula | `L = lambda * W` (steady state, no leak; sizes concurrency from a *percentile*, not the mean) and `W_q = rho/(mu - lambda)` with `rho < 1` (waiting time diverges as `rho -> 1`, so headroom is the only protection). |
| Key assumption most likely violated | That a retry is a no-op. It is one only if the effect owner has a persistent, unique record of the original intent — otherwise every retry is a second intent. Second most violated: that "TCP is reliable" extends across a reconnect. |
| Complexity / cost | Idempotency lookup `O(log D)` with `D = m*T_idem` growing with the window; fan-out `O(c)` and egress `8*N*r*s` bit/s; per-connection memory `N*(B_max*s + o)`. Worked: 20k × 2 ev/s × 512 B = 163.84 Mbit/s; 50k conns × (256 frames × 24 B + 24 KiB) = 1.536 GB ≈ 1.43 GiB. |
| Use when | HTTP: independently completable operations, cacheable reads, anything a proxy must traverse. SSE: one-way low-latency push with resume. WebSocket: genuine bidirectional low-latency messages *and* an owned session lifecycle. |
| Avoid when | WebSocket when only server push is needed, when the client cannot implement cursor resume, or when per-connection state is unaffordable. Idempotency keys as a substitute for versioning (they solve duplication, not lost updates). Retries at more than one layer. |
| Diagnostic order | (1) does the duplicate carry the same key? (2) was the key persisted at the store, across restarts? (3) do versions increment by exactly 1 per write? (4) does heap track queue depth or connection count? (5) does `1006` dominate closes? |
| Operational guardrail | Bound every queue; set `B_max` from a latency target; publish `T_idem` and `T_replay`; one retrying layer with jitter, a retry budget and a breaker; alert on attempts-per-intent and on `queue_depth_p99 / max_size`. |
| Go deeper | `01_Python_Async_and_Concurrency.md` (bounded concurrency, cancellation, queue depth) — then the security material in this domain for token lifetime and revocation, which this sheet deliberately excludes. |

**Evidence note for the card.** The formulas are `DERIVED`; the protocol claims are `SOURCED` to
RFC 9110/9111/6455 and the WHATWG SSE specification; all workload figures used to illustrate them
(`ILLUSTRATIVE`) are constructed teaching inputs, not measured production traces. Review status:
**OPEN** — this rewrite has not been independently reviewed.

