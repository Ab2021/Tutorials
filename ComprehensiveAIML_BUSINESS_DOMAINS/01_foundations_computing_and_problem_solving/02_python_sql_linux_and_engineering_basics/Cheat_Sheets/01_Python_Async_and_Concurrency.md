---
collection: "python-sql-linux-engineering-basics"
file_id: "01"
topic: "Python async and concurrency"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "mapped source title; original body unavailable"
review_trigger: "Python runtime, transport library, cancellation policy, workload, deployment or SLO changes"
---

# Python Async and Concurrency

> **Source mapping:** `Agents_Stack_Infra/Module_00_Prerequisites/01_python_async_and_concurrency.md` in [`Train/Templates/directory.txt`](../../../../Templates/directory.txt). Original source text was unavailable; this is a new mapped draft.

## 1. Purpose and boundary

Select async I/O, threads, processes or sequential execution from the dominant wait/compute behavior and operational constraints. Async concurrency overlaps nonblocking waits on one event loop; it does not make CPU-bound Python work parallel and does not remove capacity limits.

## 2. Mental model

```text
request -> validate -> await network/storage -> resume -> CPU transform -> response
                         ^ event loop schedules another ready task while waiting
```

A coroutine progresses only when it reaches an awaitable suspension point. Blocking CPU work or synchronous I/O on the event-loop thread stalls unrelated tasks. Cancellation is a request to unwind work; resource cleanup and downstream side effects require explicit ownership.

## 3. Definitions

| Term | Meaning | Common error |
|---|---|---|
| Concurrency | Multiple in-flight tasks with overlapping wait | Assuming simultaneous CPU execution |
| Parallelism | Work executing at the same time on separate execution resources | Assuming async supplies it |
| Event loop | Scheduler advancing ready coroutines | Calling blocking code inside it |
| Backpressure | Bounded admission when downstream cannot keep up | Unlimited task creation |
| Cancellation | Cooperative interruption signal | Treating it as automatic remote rollback |
| Idempotency | Repeating an operation preserves intended effect | Retrying a non-idempotent side effect |

## 4. Capacity calculation

Under a stable system, Little's law gives average in-flight work:

```text
FORMULA: L = lambda * W
WHERE: L is mean in-flight requests, lambda requests/s, W mean time in system s.
CONSTRAINT: stable measured interval; W includes queue/service time for declared boundary.
EXAMPLE: lambda=20 requests/s, W=.25 s -> L=5 requests.
```

This is not a concurrency limit. Tail latency, connection pools, downstream quotas, memory/request, cancellation and burstiness determine the configured semaphore/admission limit. Set one only from measured headroom and a failure policy.

## 5. Selection table

| Dominant work | Default | Why | Boundary |
|---|---|---|---|
| Many network/storage waits | Async I/O with bounded concurrency | Overlaps waits without one thread/request | Libraries must be nonblocking |
| CPU parsing/ML inference | Worker processes or external service | Uses parallel compute resources | Serialization, memory and startup cost |
| Blocking SDK with small concurrency | Thread offload/bounded pool | Isolates blocking call | Thread safety and pool exhaustion |
| Simple low-rate flow | Sequential | Lowest failure surface | Does not meet concurrency need |

## 6. Cancellation trace

Five tasks start; task 3 exceeds its timeout. Correct flow: timeout triggers cancellation request; task 3 catches cancellation only to release owned resource/re-raise; remaining tasks continue; request owner awaits/records every task outcome. Creating tasks without retaining/awaiting them creates orphan effects and hides exceptions.

Use structured ownership (`TaskGroup` or equivalent) so child failure/cancellation has documented sibling semantics. Give every outbound side effect a timeout, idempotency key or compensating/reconciliation path; a client disconnect does not prove a remote operation stopped.

## 7. Failure modes

| Symptom | Discriminating signal | Bounded mitigation |
|---|---|---|
| Event loop stalls | Loop lag rises while synchronous CPU/I/O span is active | Move bounded work to worker/service; profile first |
| Orphan tasks | Unobserved exception/pending task at request end | Structured ownership and outcome logging |
| Connection exhaustion | Waiters/pool utilization reach bound | Bound admission and set queue timeout |
| Duplicate side effect after retry | Same idempotency key creates multiple effects | Enforce dedupe/effect journal; reconcile |
| Timeout leaks resource | Open socket/file/task count grows after timeout | `finally` cleanup and cancellation tests |

## 8. Bounded design exercise

A tool gateway has network calls plus CPU-bound document parsing. Use async I/O for bounded network fan-out; route parsing to bounded workers; reserve queue/time budgets; attach request/idempotency IDs; and test disconnect, timeout and retry paths. Do not use an event-loop executor for unbounded parsing merely because the API is async.

## 9. Validation

Exercise: event-loop blocking probe; five-task timeout trace; transport disconnect; connection-pool saturation; duplicate retry; containerized dependency lockfile rebuild. Record in-flight count, queue wait, success/error/cancel funnels, resource cleanup and downstream effects.

## 10. Interview checks

1. **Why is async not CPU parallelism?** One event-loop thread cannot run blocking CPU tasks concurrently.
2. **What does L=5 mean?** Average in-flight work for the declared stable workload, not safe maximum.
3. **Why bound fan-out?** To protect pools, memory and downstream services.
4. **What does cancellation guarantee?** Local cooperative unwinding only; remote effects need protocol support.
5. **What makes a retry safe?** Idempotency or reconciliation, not hope.
6. **How detect blocking?** Event-loop lag correlated with synchronous spans.
7. **When choose processes?** CPU-bound work where serialization/memory tradeoff is justified.
8. **Why await all tasks?** To observe failures and complete cleanup.
9. **What is backpressure?** Deliberate admission/queue limiting under saturation.
10. **What promotes the design?** Measured tail/error/effect correctness under failure replay.

## 11. Quick Reference Card

| Decision | Rule |
|---|---|
| Network waits | Async with bounded concurrency |
| CPU work | Worker/service, not event-loop thread |
| Retry | Idempotency/effect ownership required |
| Timeout | Cleanup plus downstream uncertainty handling |
| Capacity | Measure tail/headroom; Little's law is only a sizing input |

**Limitations:** No gateway, workload, dependency implementation or source lesson body was available. Numerical values are illustrative.
