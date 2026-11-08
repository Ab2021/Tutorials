---
collection: "python-sql-linux-engineering-basics"
file_id: "02"
topic: "REST APIs and WebSockets"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "mapped source title; original body unavailable"
---

# REST APIs and WebSockets

> **Source mapping:** `Agents_Stack_Infra/Module_00_Prerequisites/02_rest_apis_and_websockets.md` in [`Train/Templates/directory.txt`](../../../../Templates/directory.txt). Original body unavailable; new mapped draft.

## 1. Choose the interaction contract

Use request/response HTTP for independently completable operations with clear resource/action semantics. Use WebSockets only when a bidirectional long-lived session materially improves the product; then own connection lifecycle, backpressure, resume and authorization. Neither transport makes an operation reliable or idempotent by itself.

## 2. HTTP contract

Define method, path, authenticated principal, request schema, response schema, validation errors, stable error codes, timeouts, pagination, concurrency/version behavior and idempotency. A `POST` retry after a transport timeout is ambiguous unless the server supports an idempotency key or client reconciliation. Never treat a `200` as proof that an asynchronous downstream effect completed unless the contract says so.

A resource update needs explicit concurrency semantics: conditional version/ETag, compare-and-swap, or a documented last-writer rule. Silent overwrite is a business decision, not a transport default.

## 3. WebSocket contract

A session needs authenticated handshake, per-message authorization/schema validation, message IDs/ordering policy, bounded inbound/outbound queues, heartbeat/idle timeout, close codes, reconnect/resume semantics and replay/deduplication rules. TCP order exists within one connection, not across reconnects or application retries. Slow clients need bounded buffering or deliberate shedding; unbounded queues convert latency into memory failure.

## 4. Capacity and timeout trace

At 100 messages/s with a 50 ms mean processing time, mean in-flight processing is `100*.05=5` under stable assumptions. This is a measurement/sizing input, not a queue limit. Track offered, accepted, processed, rejected, timed-out, disconnected and retried messages by principal/route.

## 5. Failure modes

| Symptom | Check | Mitigation |
|---|---|---|
| Duplicate mutation | Same idempotency/message ID creates multiple effects | Deduplicate at effect owner; reconcile |
| Stale overwrite | Version mismatch absent from writes | Conditional update and conflict response |
| Slow-client memory growth | Outbound queue depth and write latency | Bound queue; drop/disconnect under declared policy |
| Reconnect loses/duplicates events | Resume cursor/message ID trace | Durable sequence/replay contract |
| Authorization only at handshake | Tenant/action mismatch on message | Authorize every privileged action |

## 6. Validation

Test malformed messages, duplicate retries, disconnect during mutation, stale write, slow consumer, reconnect with resume, authorization changes mid-session and server restart. Verify persisted effects, not only transport responses.

## 7. Interview checks

1. **Why use idempotency keys?** To make retry effect semantics explicit.
2. **Does WebSocket reconnect preserve application state?** Not without an explicit resume protocol.
3. **Why bound queues?** To contain memory and tail failure.
4. **What makes an update safe under concurrency?** A declared version/conflict policy.
5. **What does a timeout mean?** Caller uncertainty, not necessarily operation failure.

## 8. Quick reference

| Need | Contract |
|---|---|
| Retriable mutation | Idempotency/reconciliation |
| Concurrent update | Version/conditional write rule |
| Long-lived stream | Auth, IDs, ordering, bounded queues, resume |
| Release | Failure replays plus effect-level verification |

**Limitations:** no API, traffic trace or source body was available. The capacity value is illustrative.
