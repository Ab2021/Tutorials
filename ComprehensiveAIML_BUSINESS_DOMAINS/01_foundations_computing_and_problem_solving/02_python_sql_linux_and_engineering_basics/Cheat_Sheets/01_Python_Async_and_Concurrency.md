---
collection: "python-sql-linux-engineering-basics"
file_id: "01"
topic: "Python async and concurrency"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: "CPython minor version, event-loop implementation, transport client library, cancellation policy, retry budget, workload mix, or SLO changes"
notation_style: "plain-text"
prerequisites:
  - "Python functions, exceptions, context managers, generators"
  - "Processes versus threads versus coroutines as OS/runtime concepts"
  - "Basic request/response networking and file-descriptor readiness"
pattern_tags: ["concurrency", "asyncio", "backpressure", "cancellation", "capacity"]
source_status: "OPEN"
owner_role: "Platform engineering / backend curriculum maintainer"
source_index: "INDEX.md"
---

# Python Async and Concurrency

> **Purpose:** Choose between sequential code, threads, processes and `asyncio`, size a bounded-concurrency pool from a latency budget instead of a guess, and reason about cancellation, retries and idempotency as *effect-ownership* problems rather than syntax problems.
> **Prerequisites:** You can read a Python function with `try/except/finally`, you know that a thread is scheduled by the OS while a coroutine is scheduled by a library, and you can read a request-rate number with units.
> **Key outcome:** Given arrival rate, service-time distribution summary, a concurrency limit, a retry policy and a timeout budget, you can compute steady-state in-flight work, the throughput ceiling implied by the pool, the extra load retries inject, the fan-out probability that at least one of $N$ parallel calls is slow, and the per-million-request cost of the design — then state which of those numbers would invalidate the choice.
> **This file does NOT cover:** Distributed-systems consensus, database transaction isolation, asyncio-based web framework internals, or GPU/accelerator parallelism. Handoffs: protocol semantics to [REST APIs and WebSockets](02_REST_APIs_and_WebSockets.md); packaging and runtime isolation to [Docker and containers](04_Docker_and_Containers.md).
> **Level:** 2 — Core Methods, with a production-diagnostics tail.
> **Version / reviewed:** 2.0 / not reviewed (review OPEN).
> **Evidence:** Every formula here is DERIVED from stated premises or is a named theorem with its hypotheses stated. Every workload number is ILLUSTRATIVE teaching input; arithmetic on those inputs is checked by hand or by an independent expression, but no company, cluster or benchmark is being described. Behavioural claims about CPython are labelled where they depend on version.
> **Retrieval cue:** Loop stalls under load → Section 8 (`ISSUE`-style rows F1–F8). "How many workers do I need?" → Section 4.1 and Section 5. "Should this be async at all?" → Section 7. "Why did my retry storm take down the dependency?" → Section 8, row F5.

## 1. File Header

Concurrency is the discipline of having more unfinished work than execution resources, and then deciding what the runtime is allowed to do while one unit of work is not runnable. Python gives you four honest answers and one dishonest one:

| Answer | What is actually overlapped | What it costs | What silently breaks it |
|---|---|---|---|
| Sequential loop | Nothing | Wall clock equals sum of latencies | Nothing; it is the baseline |
| `asyncio` coroutines | Waiting on file descriptors, timers, futures, queues | One interpreter thread; every library on the path must cooperate | Any synchronous call inside a coroutine |
| Threads | Waiting, plus a little bytecode interleaving | Stack memory, lock discipline, GIL contention | Assuming threads give CPU speedup |
| Processes | CPU execution, memory space, GIL | Serialization, startup, RAM, IPC | Assuming cheap shared state |
| "Just make it async" | Nothing by itself | Rewrite risk, no capacity change | Being believed |

The dishonest answer is treating the `async`/`await` keyword as a performance feature. It is a *scheduling* feature. It converts "this thread is blocked in `recv()`" into "this coroutine is parked and the loop is free to run something else". If the work between two `await` points is arithmetic, the loop is blocked exactly as a thread would be, and the only thing you bought is the ability to have ten thousand of them without ten thousand stacks.

The decision this sheet supports is therefore not "should I use asyncio" in the abstract. It is a chain of five concrete questions, each of which has a number attached:

1. **What fraction of wall clock is spent not-runnable?** If a unit of work is 95 % waiting, overlapping the waiting buys up to roughly 20× on that unit's latency contribution; if it is 95 % arithmetic, it buys almost nothing and you need processes or more machines.
2. **What is the arrival rate, with units?** Requests per second, messages per second, rows per second. A rate without a window and a boundary is not an input.
3. **What does one unit of work cost in service time, and what does its tail look like?** Mean alone is insufficient because concurrency limits are set by tail behaviour and by downstream quotas.
4. **What is the smallest pool that keeps utilisation below the point where queueing dominates?** This is a calculation (Section 4.2, Section 5), not a vibe.
5. **When a unit of work is abandoned — timeout, client disconnect, deploy, shutdown — who owns the effect it may already have had?** This is the question most async rewrites skip, and it is the one that produces duplicate charges, orphaned jobs and unreconciled state.

Everything below builds those five answers. Sections 2 and 3 fix the vocabulary so that "concurrency", "parallelism", "throughput" and "capacity" stop sliding into each other. Section 4 derives the results that do the actual work — Little's law (L-01), the pool throughput ceiling and utilisation identity (C-01), retry amplification (R-01), the fan-out tail union (F-01), timeout budget closure (T-01) — plus the Erlang C queueing model (Q-01) for when the pool is nearly saturated. Section 5 runs one realistic workload end to end with every arithmetic step visible. Section 6 gives the pictures that make the algebra obvious. Section 7 turns the whole thing into a selection table with disqualifiers. Section 8 is the diagnostic core: eight failure modes, each with a competing-cause list, a *discriminating* check (one that separates the candidates rather than confirming your favourite), a bounded fix and a verification. Section 9 puts units and money on it. Section 10 collects the theorems and the exact hypotheses under which they hold. Section 11 is twelve answered interview questions with a red flag each. Section 12 is the card.

**Takeaway:** `async` is a way to stop paying for idle time; it is not a way to make arithmetic faster, and it does not by itself bound anything — bounding is a separate, numerical act that this sheet teaches.

## 2. Concept Map / Mental Model

Start with the single-threaded scheduler picture, because almost every async bug is a violation of something visible in it.

```text
   arrival stream  (lambda [units/s])
        |
        v
  +-------------------+     admission control: semaphore / queue / token bucket
  |  BOUNDED INTAKE   |     capacity c  ->  refuse or wait, never grow unbounded
  +-------------------+
        |
        v   (at most c units inside this box)
  +--------------------------------------------------------------+
  |                      EVENT LOOP (1 thread)                   |
  |                                                              |
  |   ready deque:  [ T7 ][ T2 ][ T9 ]   <- runnable coroutines   |
  |        |                                                     |
  |        +--> step task: run until it hits an awaitable that   |
  |             is NOT ready, then park it and take the next one |
  |                                                              |
  |   waiters:  fd readable? | timer expired? | future resolved? |
  |        ^                                                     |
  |        |  selector (epoll / kqueue / IOCP) reports readiness  |
  +--------------------------------------------------------------+
        |                                   |
        |  coroutine needs real CPU          |  coroutine needs a blocking
        |  (parsing, crypto, PIL, sklearn)   |  call with no async client
        v                                   v
  +---------------------+            +-----------------------------+
  | PROCESS POOL        |            | THREAD / EXECUTOR OFFLOAD   |
  | true parallelism    |            | isolates the block, still   |
  | pickle/IPC boundary |            | GIL-bound for pure Python   |
  +---------------------+            +-----------------------------+
        |                                   |
        +---------------+-------------------+
                        v
                 downstream effects  ---> owned by whom? idempotency key?
                                          compensator? reconciliation job?
```

Three rules fall straight out of that diagram, and they are the whole mental model.

**Rule 1 — a coroutine only yields at an `await` that actually suspends.** `await something()` where `something()` is a plain function returning an already-completed value never gives the loop a chance to run anyone else. Modern CPython optimises the fast path further (a coroutine that never suspends can be driven eagerly under an eager task factory), so "I put `await` in front of it" is not evidence of yielding. The unit of cooperation is the *suspension point*, not the keyword.

**Rule 2 — between two suspension points the loop is yours alone.** A 40 ms JSON parse, a synchronous database driver, `time.sleep`, `requests.get`, a `str` join over 200 MB — each of these runs to completion with every other task parked. The loop has no preemption. There is no timer that will take the CPU away. This is why one blocking call can convert a 5 000-connection server into a 1-connection server, and why "the loop is single-threaded" is a capacity statement, not a stylistic one.

**Rule 3 — anything that can grow must be bounded before it is created.** `asyncio.create_task` in a loop over an unbounded input is a memory leak with extra steps: each task holds a frame, captured locals, and whatever buffers it owns. The bound is a semaphore, a fixed worker set consuming from a queue, or backpressure propagated to the producer. "The downstream will handle it" is not a bound; it relocates the failure.

Now place the three execution mechanisms on the same map so their tradeoffs become mechanical:

```text
                        overlapped resource
                 waiting only          waiting + CPU
              +---------------------+---------------------+
  in-process  |  asyncio coroutines |  threads            |
              |  ~KiB per task      |  ~MiB stack (virtual)|
              |  cooperative        |  preemptive, GIL    |
              +---------------------+---------------------+
  out-of-     |  (async client to   |  processes          |
  process     |   a remote service) |  real parallel CPU  |
              |                     |  pickle/IPC cost    |
              +---------------------+---------------------+
```

The upper-left cell is where most data-plane Python lives. The lower-right cell is where most model training, PDF parsing farms and image pipelines live. The upper-right cell exists mainly for *legacy* blocking clients you cannot replace — it is a compatibility layer, not an architecture. The lower-left cell is the one people forget: if the expensive thing is already a service, your async client is just a client, and the capacity question has moved to that service.

A useful way to hold the whole sheet: **concurrency is a budget, and the budget is denominated in simultaneously-unfinished work.** Every design choice in Sections 4–9 is an answer to "how much unfinished work will I allow, and what happens at the moment I hit that number?"

**Takeaway:** the loop is a single-threaded scheduler over readiness; you get overlap only at genuine suspension points, you own the CPU between them, and any unbounded creation point is where the design will fail first.

## 3. Foundational Definitions

Definitions here are written so that a sentence using the term can be falsified. Where a term has a colloquial meaning that conflicts with the operational one, the conflict is named, because those conflicts are what produce bad sizing arguments.

### 3.1 The two axes: overlap and simultaneity

| Term | Operational meaning | Unit / shape | Nonexample (what it is not) |
|---|---|---|---|
| Concurrency | Number of units of work that have started and not finished, regardless of whether any CPU is executing them | count (dimensionless) | Not "speed"; a system can be highly concurrent and slow |
| Parallelism | Number of instruction streams executing at the same instant on distinct execution resources | count of hardware contexts | Not what `async` provides |
| Overlap | The property that unit B's waiting interval intersects unit A's running interval | boolean per pair | Not necessarily throughput gain if a shared resource serialises |
| Utilisation ($\rho$) | Fraction of a resource's capacity that is busy over a window | dimensionless, $0 \le \rho < 1$ for stable systems | Not "load average"; load average counts runnable+uninterruptible on Linux and is not normalised |
| Throughput ($X$) | Completed units per unit time at a stated boundary | units/s | Not arrival rate; completions lag arrivals |
| Latency | Time from a defined start event to a defined end event for one unit | seconds | Undefined without both endpoints |
| Service time ($S$) | Time a unit occupies the resource, excluding queueing | seconds | Not response time |
| Response time ($R$) | Service time plus all waiting before service begins | seconds | Not "server processing time" as reported by an app log |

The colloquial collapse to avoid: "async is concurrent, threads are parallel". Threads in CPython are concurrent *and* only parallel across I/O waits and C extensions that release the GIL; coroutines are concurrent and never parallel. The axis that matters for a design decision is *which resource is oversubscribed*, not which keyword appears in the source.

### 3.2 Runtime objects

| Term | Operational meaning | Detail that changes behaviour |
|---|---|---|
| Coroutine object | A suspendable computation created by calling an `async def` function | Calling it runs no body code; it must be awaited or scheduled, otherwise CPython emits a "never awaited" `RuntimeWarning` at collection |
| Task | A coroutine wrapped for independent scheduling by the loop, with its own result/exception/cancellation state | Task identity is what `asyncio.all_tasks()` reports; a bare coroutine is invisible to it |
| Future | A placeholder for a result that some other code will resolve | Awaiting a Future that nobody resolves parks the task forever — a silent liveness bug, not an error |
| Event loop | The object that owns the ready queue, the selector, the timer heap and the task registry | One loop per thread in practice; `asyncio.run()` creates, runs and closes one |
| Suspension point | A point where the running task yields control because its awaitable is not ready | Only these give other tasks a turn |
| Readiness | The condition (fd readable/writable, timer expired, future done) that makes a parked task runnable again | Timeouts are implemented as timers in the same loop, so a starved loop delays its own timeouts |
| Callback | A plain function the loop invokes directly from the ready queue | Callbacks have no suspension; a slow callback blocks everything and is invisible to `await`-based reasoning |

A subtle but load-bearing distinction: a **timeout is scheduled on the loop**, so a loop that is blocked for 2 s cannot fire a 200 ms timeout at 200 ms. The timer fires when the loop next runs. This is why "my timeout didn't work" is usually a blocking-call diagnosis, not a timeout-configuration diagnosis (Section 8, row F1).

### 3.3 The GIL and what it does and does not permit

| Statement | Status | Consequence for design |
|---|---|---|
| CPython's default build serialises Python bytecode execution across threads of one interpreter | True for the standard build through at least CPython 3.13 | Threads do not speed up pure-Python arithmetic |
| C extensions can release the GIL around long operations | True (e.g. many numeric/compression/crypto routines) | Threads *can* speed up such work; verify per library rather than assuming |
| The GIL serialises `await` scheduling | Not exactly — the loop is a Python object, so scheduling costs bytecode execution and is therefore GIL-bound | Very high task counts spend real time in scheduler overhead |
| Free-threaded builds remove the GIL | True as an experimental build option in recent CPython versions (3.13 era), with single-threaded performance and extension-compatibility caveats | Do not plan capacity on it without measuring your exact stack and version |
| `sys.setswitchinterval` (default 5 ms) bounds thread preemption granularity | True | A CPU-bound thread can hold the GIL for up to roughly that interval before another thread runs |

The practical reading: threads are a **waiting-overlap** mechanism in Python by default, and a CPU-parallelism mechanism only where the heavy work happens inside GIL-releasing C code. Processes are the reliable CPU-parallelism answer, at the price of a serialization boundary.

### 3.4 Flow-control vocabulary

| Term | Operational meaning | Why the distinction matters |
|---|---|---|
| Backpressure | A signal that a producer must slow down because a consumer is at capacity | Distinguished from *dropping* and from *buffering*; only backpressure preserves bounded memory |
| Admission control | A decision at the entry boundary to accept, queue-with-deadline, or reject work | The only place where "reject" is cheap |
| Bounded concurrency | A hard ceiling $c$ on units simultaneously inside a region | The number that Sections 4.2 and 5 are about |
| Semaphore | A counter with acquire/release; acquire waits while the counter is zero | In `asyncio`, a semaphore is *fair* in practice (FIFO waiters), which matters for starvation analysis |
| Queue with maxsize | A bounded buffer between producer and consumer stages | `maxsize<=0` means unbounded — the default you must not rely on |
| Head-of-line blocking | A single slow unit at the front preventing progress of units behind it on the same ordered channel | Multiplexed protocols hit this at the connection level, not the task level |
| Starvation | A unit that is runnable in principle never receives service | Caused by unfair scheduling, priority inversion, or unbounded retry of high-priority work |
| Thundering herd | Many waiters waking simultaneously to contend for one resource | Retry synchronisation and timer alignment both produce it |
| Jitter | Randomised delay added to a schedule to decorrelate contenders | The fix for herd effects; its parameters need justification, not folklore |

### 3.5 Lifecycle and effect vocabulary

| Term | Operational meaning | Nonexample |
|---|---|---|
| Cancellation | A cooperative request that a task stop at its next suspension point by raising `asyncio.CancelledError` | Not a kill; not a rollback; not a guarantee that a remote call stopped |
| Structured concurrency | A discipline where the lifetime of child tasks is bounded by a lexical scope that also specifies sibling-failure semantics | Not "I kept a list of tasks somewhere" |
| Orphan task | A task whose outcome nobody awaits or logs | Not the same as a detached daemon; an orphan is an accident |
| Idempotency | Repeating an operation with the same key produces the same observable effect as one execution | Not "retry-safe by luck"; requires a dedupe store or a naturally idempotent operation |
| At-least-once delivery | The system guarantees delivery attempts continue until acknowledged, so duplicates are possible | Not at-most-once, under which loss is possible |
| Timeout budget | The maximum wall time a caller will wait, apportioned across the calls it makes | Not a per-call timeout chosen independently at each layer |
| Deadline propagation | Passing the remaining budget to downstream calls so they do not work past the caller's patience | Not the same as a hop-by-hop timeout |
| Compensation | A semantic undo of an effect that cannot be rolled back transactionally | Not a retry; it is a different operation |

The pairing to memorise: **cancellation is local and cooperative; effects are remote and durable.** Every design that conflates them eventually ships a duplicate.

### 3.6 Sizing vocabulary, in the exact units used later

| Symbol | Meaning | Units | Domain |
|---|---|---|---|
| $\lambda$ | arrival rate at a boundary over a stated window | units/s | $\lambda \ge 0$ |
| $\mu$ | service rate of the *whole* pool at concurrency $c$ | units/s | $\mu \le c/S$ |
| $S$ | mean service time per unit | s | $S > 0$ |
| $c$ | configured concurrency limit | count | integer $\ge 1$ |
| $L$ | mean number of units in the system (in-flight) | units | $0 \le L$ |
| $W$ | mean time in system per unit | s | $W > 0$ |
| $W_q$ | mean queue wait | s | $W_q \ge 0$ |
| $\rho$ | utilisation $=\lambda S/c$ | dimensionless | $0 \le \rho < 1$ |
| $p$ | per-call probability of a slow/error outcome | dimensionless | $0 \le p \le 1$ |
| $N$ | fan-out width (parallel calls per unit) | count | integer $\ge 1$ |
| $r$ | retry attempts per failed call, or retry fraction | count or dimensionless | $\ge 0$ |

**Takeaway:** if a sentence about concurrency cannot be restated with one of these symbols and a unit, it is probably a preference rather than a fact.

## 4. Core Formulas with Full Derivation

Four results carry the whole sheet, plus one queueing model that tells you when the simple ones stop being adequate. Each is derived, given the five-part block, and then bounded by a case where it is wrong. Every numeric input below is **ILLUSTRATIVE**; the arithmetic is checked.

### 4.1 Base result — Little's law converts a rate and a time into in-flight work

**Validity first.** Little's law applies to a system in steady state over an observation window, with a well-defined boundary (what counts as "in" and what counts as "arrived"), where the mean arrival rate and mean time-in-system exist and the window is long enough that end effects are negligible. It does not require Poisson arrivals, exponential service, independence, or a single server. It says nothing about tails, and it does not cap anything.

**Derivation.** Consider a window of length $T$. Let $A$ be the number of arrivals during $T$ and $D$ the number of departures; in steady state (and with the boundary defined so that nothing is created or destroyed inside) $A \approx D \approx \lambda T$. Each unit contributes its sojourn time to the integral of the in-flight count over the window. Summing all sojourn time gives $\sum_i W_i$, and integrating the count over time gives $\int_0^T L(t)\,dt$; these are the same area counted two ways. Dividing by $T$ and by $\lambda$:

```text
FORMULA [L-01]  Little's law
    L = lambda * W

WHERE
    L      mean units in the system inside the declared boundary   [units]
    lambda mean arrival rate at that boundary over the window       [units/s]
    W      mean time a unit spends inside the boundary              [s]
    identity: units/s * s = units

CONSTRAINT
    steady state over the window; boundary explicitly drawn;
    means exist; end effects negligible;
    no statement about distributions, tails, or capacity.

NUMERICAL EXAMPLE
    lambda = 20 requests/s, W = 0.250 s
    L = 20 * 0.250 = 5 requests in flight on average

INTERPRETATION
    5 is the *average* occupancy. If the pool size is 5, the system is
    exactly saturated on average, which means it spends real time above 5
    and therefore queues. A pool of 5 sized from this number alone is a
    coin flip on latency, not a design.
```

**Boundary / counterexample.** Take a window of 1 s on a system that receives 20 requests, all of which arrive in the first 100 ms and take 0.25 s each. Mean in-flight is 20 for 0.25 s and near 0 afterwards: the time-average $L \approx 5$ still holds, but at no instant was the occupancy 5. Conversely a system with 1 request/s and a 0.001 s service time has $L = 0.001$: Little's law says the average is a thousandth of a request, which is a true statement about a mean and a useless statement about a hard concurrency requirement. Little's law is a *conservation identity*, not a provisioning rule; the provisioning rule needs utilisation and tail behaviour.

### 4.2 General result — the pool throughput ceiling and the utilisation identity

**Validity first.** This result assumes a pool of $c$ workers (or $c$ in-flight permits) each processing one unit at a time for mean service time $S$, with work arriving independently of which worker is free. Service times may be arbitrarily distributed for the *mean* statement; the tail statement in 4.5 needs more.

**Derivation.** One worker completes one unit per $S$ seconds in expectation, so it has rate $1/S$. With $c$ independent workers the aggregate service rate is $c/S$, an upper bound on completions per second because it assumes workers never idle for lack of work. Arrivals supply work at rate $\lambda$; steady state requires $\lambda < c/S$. Dividing both sides by the capacity gives the dimensionless utilisation, which is the fraction of worker-time that must be occupied.

```text
FORMULA [C-01]  pool throughput ceiling and utilisation
    mu_max = c / S                [units/s]
    rho    = lambda * S / c       [dimensionless]

WHERE
    c   concurrency limit (workers, permits, connections)   [count]
    S   mean service time per unit                          [s/unit]
    c/S -> (count) / (s/unit) = units/s
    rho -> (units/s)(s/unit)/(count) = dimensionless

CONSTRAINT
    one unit per worker at a time; S measured on the same machine class
    and with the same downstream state; no retries in lambda; stable window.
    rho >= 1 means the backlog grows without bound in expectation.

NUMERICAL EXAMPLE
    c = 32, S = 0.120 s  ->  mu_max = 32 / 0.120 = 266.67 units/s
    lambda = 180 units/s ->  rho = 180 * 0.120 / 32 = 0.675

INTERPRETATION
    At 180/s the pool is 67.5% busy, so 32.5% of worker-time is idle.
    The headroom is what absorbs bursts and retries; a rho of 0.675 is
    a *starting* point, not an achievement, because the tail is not yet
    considered (4.5).
```

**Boundary / counterexample.** Set $\lambda = 266.67$: $\rho = 1$ and the backlog is a random walk with positive drift, so even though the *ceiling* equals the arrival rate, queue wait diverges. The ceiling formula also quietly assumes every one of the $c$ permits is usable; if each request holds a permit while waiting on a downstream quota of 10, the effective $c$ is 10 and the ceiling is 83.3 units/s, not 266.67. **Effective concurrency is the minimum of your semaphore and every shared resource behind it** — connection pool size, downstream rate limit, file descriptors, database connection cap, thread pool if any layer offloads.

### 4.3 Variants — retry amplification, tail fan-out, and the deadline triangle

These are the three variants that change answers in practice: retries change $\lambda$, fan-out changes what "one unit" means for latency, and timeouts decide whether work is abandoned while its effect continues.

**Variant A — retry amplification.** Validity: a fraction $f$ of calls fail per attempt, each failure is retried up to $k$ additional times with independent failure probability (a strong and often false assumption), and retried calls consume the same capacity as originals.

Derivation: attempt 1 happens for every request. Attempt 2 happens only for the $f$ that failed, so expected extra attempts $= f$. Attempt 3 $= f^2$, and so on. Summing the geometric series gives the load multiplier; the arrival rate seen by the *dependency* is the offered rate times that multiplier.

```text
FORMULA [R-01]  retry amplification
    A_k = sum_{i=0..k} f^i = (1 - f^(k+1)) / (1 - f)      for f < 1
    lambda_eff = lambda * A_k

WHERE
    f  per-attempt failure probability (of the class being retried)
    k  additional attempts allowed after the first
    A_k dimensionless multiplier on load at the dependency
    lambda_eff [units/s]

CONSTRAINT
    failures independent across attempts; no retry budget; retried work
    consumes full capacity. Correlated failures (a dependency outage)
    make f -> 1, so A_k -> k+1: exactly the storm case.

NUMERICAL EXAMPLE
    f = 0.05, k = 2 -> A = 1 + 0.05 + 0.0025 = 1.0525
    lambda = 180/s   -> lambda_eff = 180 * 1.0525 = 189.45 /s
    f = 0.80, k = 2 -> A = 1 + 0.80 + 0.64      = 2.44
    lambda = 180/s   -> lambda_eff = 439.2 /s

INTERPRETATION
    A 5% independent error rate costs 5.25% extra load: acceptable.
    An 80% correlated error rate nearly 2.5x's the load *at the moment the
    dependency is least able to serve it*. This is why retries must be
    budgeted (a capped fraction of traffic) and jittered, and why a
    dependency outage must trip a circuit breaker that stops retrying.
```

**Variant B — tail fan-out.** Validity: $N$ parallel calls per unit, each independently exceeding a latency threshold with probability $p$; the unit's latency is the maximum of the $N$ call latencies.

Derivation: the probability that *no* call exceeds the threshold is $(1-p)^N$ by independence, so the probability that at least one does is the complement. The maximum being the unit's latency follows from "the unit finishes when its last child finishes".

```text
FORMULA [F-01]  probability at least one child is slow
    P_max_slow = 1 - (1 - p)^N

WHERE
    p  per-call probability of exceeding threshold T          [dimensionless]
    N  parallel calls per unit (fan-out)                      [count]
    T  the latency threshold of interest                      [s]

CONSTRAINT
    independence across children (violated by shared downstream,
    shared connection pool, shared GIL/loop time); identical p;
    "slow" defined by one threshold, not a full distribution.

NUMERICAL EXAMPLE
    p = 0.01 (99th percentile), N = 10
    P = 1 - 0.99^10 = 1 - 0.904382 = 0.095618  -> 9.56%
    At N = 50: 1 - 0.99^50 = 1 - 0.605006 = 0.394994 -> 39.5%

INTERPRETATION
    A 1% per-call tail becomes a ~10% tail at fan-out 10 and a ~40% tail
    at fan-out 50. Fan-out converts a good p99 into a bad p90. Fix by
    hedging/replicating only where the budget allows, capping N, or
    changing the aggregation so the unit does not need the max.
```

**Variant C — the deadline triangle.** Validity: a caller with total budget $B$ makes sequential stages with per-stage timeouts $t_i$ plus its own overhead $\epsilon$; the caller must not wait past $B$.

Derivation: the caller's worst-case wall time is the sum of stage timeouts plus overhead. Requiring it to be at most $B$ gives the budget constraint; the per-stage allocation is a choice under that constraint, and the correct propagation is to pass *remaining* budget, not a fresh $t_i$ at each hop.

```text
FORMULA [T-01]  timeout budget closure
    sum_{i=1..m} t_i + epsilon <= B
    t_i <= B_remaining_at_stage_i

WHERE
    B caller budget [s]; t_i stage timeout [s]; epsilon local overhead [s]
    B_remaining = B - elapsed_before_stage_i

CONSTRAINT
    sequential stages; timeouts are enforced by a scheduler that is
    itself running (see 3.2: a blocked loop delays its own timers).

NUMERICAL EXAMPLE
    B = 2.000 s, m = 3 stages, epsilon = 0.050 s
    naive: t_i = 2.000 each -> worst case 6.050 s  (violates B)
    allocated: t1 = 0.400, t2 = 0.900, t3 = 0.650, epsilon = 0.050
    check: 0.400 + 0.900 + 0.650 + 0.050 = 2.000 s  (tight)

INTERPRETATION
    Per-call timeouts chosen independently do not compose: three 2 s
    timeouts are a 6 s worst case. A deadline must be apportioned and
    propagated, and the sum must be checked with epsilon included.
```

### 4.4 Limits and counterexamples — when the closed forms stop describing reality

The three closed forms above are *mean-value* statements. Near saturation, queueing dominates and the mean becomes misleading. The standard single-queue model with Poisson arrivals and exponential service (M/M/c) gives the queue-wait result below; it is a *model*, and its assumptions are usually violated by web traffic, but it is the cheapest way to see the shape of the cliff.

Validity: Poisson arrivals (rate $\lambda$), exponential service (mean $S$), $c$ identical servers, infinite waiting room, FIFO, no abandonment. Under these, the probability that an arrival must wait is given by Erlang C, and the mean wait follows.

```text
FORMULA [Q-01]  Erlang C and M/M/c mean queue wait
    a = lambda * S                      (offered load, Erlangs)
    rho = a / c
    C(c,a) = [ a^c / (c! (1 - rho)) ] /
             [ sum_{n=0..c-1} a^n / n! + a^c / (c! (1 - rho)) ]
    W_q = C(c,a) * S / (c - a)

WHERE
    a offered load [Erlangs, dimensionless]; rho dimensionless;
    C(c,a) probability an arrival waits [dimensionless];
    W_q mean queue wait [s]; S [s]; c [count]; lambda [1/s]

CONSTRAINT
    the five M/M/c assumptions above; rho < 1; infinite buffer.
    Violations (bursty arrivals, heavy-tailed S, finite queue, priority
    classes) change the constants, often by a large factor.

NUMERICAL EXAMPLE  (c = 32, S = 0.120 s, lambda swept; a = lambda*S)
    lambda=180 -> a=21.60, rho=0.6750 -> C=0.0245, W_q=0.282 ms
    lambda=240 -> a=28.80, rho=0.9000 -> C=0.4582, W_q=17.182 ms
    lambda=260 -> a=31.20, rho=0.9750 -> C=0.8393, W_q=125.894 ms

INTERPRETATION
    Going from 67.5% to 90% utilisation multiplies mean queue wait by
    about 61x (0.282 -> 17.182 ms) while throughput rises only 33%.
    From 90% to 97.5% it multiplies wait again by about 7.3x
    (17.182 -> 125.894 ms) for another 8.3% of throughput. This is the
    shape of the cliff, and it is why target utilisation is a policy
    number (commonly well below 1) rather than a maximisation goal.
```

Treat the specific $W_q$ values above as **ILLUSTRATIVE arithmetic of the model**, not predictions for your service: the exact values are sensitive to $c$, to the service-time distribution (a deterministic service time behaves very differently from exponential), and to whether failures are retried.

**The four limits to state explicitly:**

1. **$\rho \to 1$**: all mean-value models diverge, and a finite buffer converts the divergence into *rejection* instead of *delay*. Choose which one you prefer before the incident.
2. **$c$ is not yours alone**: effective $c = \min(c_{\text{semaphore}}, c_{\text{pool}}, c_{\text{downstream quota}}, c_{\text{fd limit}}, c_{\text{memory}})$. Sizing on the largest of these is the classic over-provisioning illusion.
3. **Little's law with retries**: $W$ must include retry time, otherwise $L$ is understated and your pool is smaller than the traffic implies.
4. **Independent-failure assumptions fail together**: during an incident $f \to 1$ and $p \to 1$ simultaneously across tenants, so both 4.3 variants collapse toward their worst case at the same moment.

**Takeaway:** use L-01 to translate between rate and occupancy, C-01 to find the ceiling and the headroom, R-01 and F-01 to see how failures and fan-out multiply load and tails, T-01 to make timeouts compose, and Q-01 only to remember the *shape* of the saturation cliff.

## 5. Worked Numerical Example

One workload, carried from service-time measurement to instance count to monthly bill, with every substitution written out. The workload is **ILLUSTRATIVE**: a document-enrichment gateway that authenticates a request, performs three downstream lookups, transforms the payload locally, and persists a result. No real deployment is being described; the arithmetic is checked and the model consequences are what matter.

### 5.1 Stated inputs

| Symbol | Value | Meaning | Source status |
|---|---|---|---|
| $\lambda$ | 120 req/s | sustained arrival rate over the peak hour | ILLUSTRATIVE |
| $S_{\text{cpu,loop}}$ | 0.0015 s | loop-thread CPU: parse headers, validate, serialise response | ILLUSTRATIVE |
| $S_{\text{cpu,worker}}$ | 0.0165 s | CPU transform per request (decompression, field mapping) | ILLUSTRATIVE |
| $S_{\text{call}}$ | 0.045 s | mean latency of one downstream lookup | ILLUSTRATIVE |
| $N$ | 3 | lookups per request, independent of one another | ILLUSTRATIVE |
| $S_{\text{gather}}$ | 0.004 s | extra local cost of fan-out (gather, error merge) | ILLUSTRATIVE |
| $q$ | 250 lookups/s | downstream quota for the caller's credential | ILLUSTRATIVE |
| $f$ | 0.08 | per-attempt failure probability on a downstream call | ILLUSTRATIVE |
| $k$ | 2 | retries allowed per failed call | policy input |
| $B$ | 2.000 s | end-to-end caller budget | policy input |
| $C_{\text{inst}}$ | $0.10 /h | illustrative hourly instance price | ILLUSTRATIVE |
| $\rho_{\text{target}}$ | 0.70 | target utilisation for the loop and worker pools | policy input |

### 5.2 Step 1 — service time, sequential versus fanned out

Sequential execution: the CPU transform and each lookup happen one after another.

```text
S_seq = S_cpu,loop + S_cpu,worker + N * S_call + 0
      = 0.0015 + 0.0165 + 3 * 0.045
      = 0.0180 + 0.135
      = 0.1530 s
```

Fanned out: the three lookups overlap, the CPU work does not overlap with itself, and gather adds a little.

```text
S_fan = S_cpu,loop + S_cpu,worker + max(S_call) + S_gather
      = 0.0015 + 0.0165 + 0.045 + 0.004
      = 0.0670 s
```

Latency improvement on the *waiting* portion, at the level of one request:

```text
ratio = S_seq / S_fan = 0.1530 / 0.0670 = 2.28358...
```

So fan-out is worth about 2.28× on per-request service time — not 3× — because the 18 ms of CPU and the 4 ms of gather overhead are paid regardless. **This is the first place an async rewrite delivers less than advertised**, and it is arithmetic, not opinion: the Amdahl-style limit here is that only the $N \times S_{\text{call}}$ term is parallelisable. If $S_{\text{cpu,worker}}$ had been 0.100 s instead of 0.0165 s, the ratio would be $(0.0015+0.100+0.135)/(0.0015+0.100+0.045+0.004)=0.2365/0.1505=1.57$ and the fan-out would barely be worth the complexity.

### 5.3 Step 2 — concurrency limit from utilisation, not from Little's law

Using C-01 with the fanned-out service time:

```text
c_min(rho) = ceil( lambda * S_fan / rho )

rho = 0.70: c = ceil(120 * 0.0670 / 0.70) = ceil(8.04 / 0.70) = ceil(11.4857) = 12
rho = 0.60: c = ceil(8.04 / 0.60) = ceil(13.4000) = 14
rho = 0.80: c = ceil(8.04 / 0.80) = ceil(10.0500) = 11
```

The offered load $a = \lambda S = 120 \times 0.067 = 8.04$ Erlangs. Note that 8.04 is *less than* the 12 permits chosen: the extra 3.96 permits exist purely to keep utilisation at 0.70 so that queueing does not dominate. **Little's law alone would have suggested a pool of 8**, which is exactly at capacity in mean terms and therefore guarantees a growing queue for any burst.

Check the ceiling at the chosen $c$:

```text
mu_max(12) = 12 / 0.0670 = 179.104 units/s   vs lambda = 120  -> rho = 0.6700
```

For headroom against bursts and against the retry amplification computed below, 16 permits is the operating choice:

```text
mu_max(16) = 16 / 0.0670 = 238.806 units/s   ->  rho = 8.04 / 16 = 0.5025
```

### 5.4 Step 3 — what the queue model says about the three candidate pools

Applying Q-01 with $c \in \{12, 14, 16, 20\}$ and $a = 8.04$:

| $c$ | $\rho = a/c$ | $C(c,a)$ = P(wait) | $W_q$ (ms) | Reading |
|---:|---:|---:|---:|---|
| 12 | 0.670 | 0.1439 | 2.43 | 14% of requests queue; mean wait 2.4 ms |
| 14 | 0.574 | 0.0407 | 0.46 | queueing nearly gone |
| 16 | 0.503 | 0.0094 | 0.08 | effectively no queue at the mean |
| 20 | 0.402 | 0.0003 | 0.00 | paying for idle permits |

The decision this table supports: **14 is the smallest pool where queue wait stops being a material fraction of the 67 ms service time** (0.46 ms is 0.7 %), and 16 buys burst absorption at 2 extra permits of cost. Choosing 20 is buying idle capacity with no measured tail benefit — that is the point where "more concurrency" becomes a cost with no mechanism behind it. These values are the *model's* arithmetic under M/M/c assumptions; a real service with a heavier-tailed downstream would show larger $W_q$ at the same $\rho$, so treat the table as a comparative shape, not a latency promise.

### 5.5 Step 4 — the downstream quota, which is the real binding constraint

Each request issues $N = 3$ lookups:

```text
lookups/s = lambda * N = 120 * 3 = 360 lookups/s
quota ceiling in units/s = q / N = 250 / 3 = 83.333 units/s
```

The pool of 16 permits permits 238.8 units/s. The quota permits 83.3 units/s. **The effective concurrency limit is not the semaphore.** Two consequences must be stated before any code is written:

1. At 120 req/s the caller is asking for $360/250 = 1.44\times$ the allowed lookup rate, so it will be throttled (or will error) on 44 % of its demand — a capacity failure that no amount of `async` fixes.
2. The design must reduce lookups per request, not increase permits. Caching is the cheapest lever; the required hit rate is:

```text
misses/s = lambda * N * (1 - h)  <=  q
360 * (1 - h) <= 250
1 - h <= 250/360 = 0.694444
h >= 0.305556   ->  30.56% cache hit rate
```

A 30.6 % hit rate on a metadata lookup is a modest target. Batching is the alternative: if one call can resolve 3 keys, the lookup count drops from 360/s to 120/s, well under quota, at the cost of coupling keys into batches (and of head-of-line effects when a batch contains one slow key). Either answer is a *design* answer; raising the semaphore is not.

### 5.6 Step 5 — retry amplification at the dependency

Applying R-01 with $f = 0.08$ and $k = 2$:

```text
A_3 = 1 + 0.08 + 0.08^2 = 1 + 0.08 + 0.0064 = 1.0864
lambda_eff(units)   = 120  * 1.0864 = 130.368 req/s
lambda_eff(lookups) = 360  * 1.0864 = 391.104 lookups/s
```

391 lookups/s against a 250/s quota means **retries alone push the caller 56 % over quota** even though the base rate was only 44 % over. Now the correlated-failure case, which is what actually happens during a dependency incident — failures are no longer independent (the retry succeeds only after the dependency recovers):

```text
f = 0.50: A = 1 + 0.5 + 0.25      = 1.750  -> 210.0 req/s, 630.0 lookups/s
f = 0.80: A = 1 + 0.8 + 0.64      = 2.440  -> 292.8 req/s, 878.4 lookups/s
```

At $f = 0.8$ the caller attempts 878 lookups/s against a 250/s allowance: a 3.5× storm aimed at a dependency that is already failing. **Bounded fix:** a retry budget expressed as a fraction of successful traffic (commonly a small percentage), full-jitter exponential backoff, and a circuit breaker that stops retrying when the observed failure rate crosses a threshold. Verification: replay the failure injection and confirm lookups/s at the dependency stays below quota during the injected outage.

### 5.7 Step 6 — the tail the fan-out creates

With per-call $p = 0.02$ at the selected latency threshold and $N = 3$:

```text
P(at least one child slow) = 1 - (1 - 0.02)^3 = 1 - 0.98^3 = 1 - 0.941192 = 0.058808
```

5.88 % of requests inherit the bad tail of their worst child. Widening fan-out to 10 and then 50 for comparison:

```text
N = 10: 1 - 0.98^10 = 1 - 0.817073 = 0.182927  -> 18.29%
N = 50: 1 - 0.98^50 = 1 - 0.364170 = 0.635830  -> 63.58%
```

The practical rule this produces: **fan-out is a latency tool that spends tail budget.** If the SLO is on p95 end-to-end, a fan-out of 3 with a 2 % per-call tail spends a meaningful share of it; a fan-out of 50 will define the SLO entirely. Hedging (issuing a second copy only after a delay) buys tail at the price of extra load — which is why hedging must be budgeted with R-01 like any retry.

### 5.8 Step 7 — the loop-thread CPU budget

The loop must be free to schedule. Its per-request work is 1.5 ms:

```text
rho_loop at 120 req/s = 120 * 0.0015 = 0.180
max lambda at rho_loop 0.70 = 0.70 / 0.0015 = 466.667 req/s
```

Contrast with the same computation if the 16.5 ms transform ran on the loop thread:

```text
rho_loop = 120 * 0.018 = 2.160 > 1   -> the loop cannot keep up at all
```

That single number is the argument for offloading the transform. It is also the answer to "why is my async service slower than the threaded one": the naive version put 18 ms of uninterruptible work in a scheduler that must also handle every other request's bookkeeping.

### 5.9 Step 8 — worker pool sizing for the CPU transform

```text
worker CPU demand = lambda * S_cpu,worker = 120 * 0.0165 = 1.980 cores
with 4 workers    : utilisation = 1.980 / 4 = 0.4950
worker ceiling    : 4 / 0.0165 = 242.42 req/s
```

Two workers would sit at 99 % utilisation — no burst headroom, and any retry or reconnect storm tips it over. Four workers at 49.5 % gives room for a doubled rate. Note that worker count is a *core* decision: if the host has 4 vCPUs and the loop plus OS need one, the worker pool is competing for the same cores, so the honest comparison is 3 workers against a 1.98-core demand (66 %), not 4 against 49.5 %.

### 5.10 Step 9 — timeout budget closure

Applying T-01 with $B = 2.000$ s, three stages, and $\epsilon = 0.050$ s:

```text
auth     t1 = 0.400
enrich   t2 = 0.900      (dominant: fan-out plus retry allowance)
persist  t3 = 0.650
overhead eps = 0.050
sum = 0.400 + 0.900 + 0.650 + 0.050 = 2.000 s   -> exactly B
```

The check that matters: the *naive* configuration where each stage independently chooses 2 s gives a worst case of $3 \times 2.000 + 0.050 = 6.050$ s, which violates the caller budget by 3×. Any downstream that receives a fresh 2 s timeout after the caller has already spent 1.4 s is being invited to work for the caller's benefit after the caller has left — the classic wasted-work pattern.

### 5.11 Step 10 — what happens under unbounded task creation

If instead of a semaphore the code did `create_task` per input row from a 50 000-row batch, with 3 KiB of task/frame overhead and a 16 KiB buffer each:

```text
per task          = 3 KiB + 16 KiB = 19 KiB
16 tasks          = 16 * 19 / 1024 = 0.297 MiB
500 tasks         = 500 * 19 / 1024 = 9.277 MiB
5 000 tasks       = 5 000 * 19 / 1024 = 92.773 MiB
50 000 tasks      = 50 000 * 19 / 1024 = 927.734 MiB
```

At 50 000 the process is a gigabyte of parked work that the downstream can only serve at 250 lookups/s — a 200-second queue of in-memory promises with no admission decision anywhere. The bounded version holds 16 in flight and blocks the producer, which is what makes memory a function of the limit instead of a function of the input size.

### 5.12 Step 11 — instance count and cost

Capacity per instance is set by the tighter of the permit ceiling (238.8/s), the loop ceiling (466.7/s) and the worker ceiling (242.4/s) — so 238.8/s. At 120 req/s, one instance runs at 50 % of that; availability requires independent failure domains, so two instances each carry 60 req/s and the pair survives one loss at 50 % of the survivor's ceiling.

```text
requests/hour at 120 req/s          = 120 * 3600        = 432 000
cost per request (1 instance)       = 0.10 / 432000     = 2.3148e-7 USD
cost per 1M requests (1 instance)   = 2.3148e-7 * 1e6   = 0.231481 USD
cost per 1M requests (2 instances)  = 0.231481 * 2      = 0.462963 USD
monthly (720 h, 2 instances)        = 2 * 0.10 * 720    = 144.00 USD
monthly requests                    = 120 * 86400 * 30  = 311 040 000
monthly cost per 1M                 = 144 / 311.04      = 0.463 USD
```

The two routes to the same number agree ($0.462963$ vs $0.463$), which is the arithmetic check. What the number *means*: the compute bill for this gateway is under half a dollar per million requests, which is why concurrency mistakes here are rarely visible on the bill — they show up as latency SLO breaches and as duplicate effects. **The expensive failure mode is not the idle permit; it is the incident.**

### 5.13 Sensitivity: which inputs actually move the answer

| Input changed | From → to | Effect on the decision | Verdict |
|---|---|---|---|
| $S_{\text{call}}$ | 0.045 → 0.090 s | $S_{\text{fan}}$ 0.067 → 0.112; $c_{\min}(0.70)$ 12 → 20; per-instance ceiling 238.8 → 142.9 | Changes instance count; fan-out still worth it |
| $S_{\text{cpu,worker}}$ | 0.0165 → 0.100 s | $S_{\text{seq}}$ 0.2365, $S_{\text{fan}}$ 0.1505, ratio 1.57; worker demand 12.0 cores | Async fan-out no longer the main lever; scale workers |
| $N$ | 3 → 1 | No overlap to buy: one call is already serial, so $S$ stays 0.067 s with or without fan-out | Removes the entire concurrency justification |
| quota $q$ | 250 → 1000 /s | quota ceiling 333.3 > 238.8 permits | Quota stops binding; permits become the constraint |
| $f$ | 0.08 → 0.30 | $A = 1.39$; lookups 500/s vs 250 quota | Retry budget becomes mandatory, not optional |
| $\rho_{\text{target}}$ | 0.70 → 0.90 | $c_{\min}$ 12 → 9; $W_q$ rises sharply per Q-01 | Cheaper and materially worse in the tail |

The one-line conclusion from the table: **the answer is dominated by $N$, the per-call latency, and the quota — not by the choice of `asyncio` versus threads.** The concurrency mechanism sets the achievable overlap; the workload and the quota set the requirement.

**Takeaway:** a full sizing pass is nine substitutions and two comparisons, and it produces numbers — 12 permits minimum, 16 for burst, 360→391 lookups/s against a 250/s quota, a 30.6 % cache requirement, a 5.9 % inherited tail — that can each be argued with and rechecked independently.

## 6. Visual / Geometric Intuition

The formulas in Section 4 all have a picture. Learning the picture is what lets you re-derive the number when the situation drifts instead of recalling a memo.

### 6.1 Two timelines that differ only in where the gaps go

```text
SEQUENTIAL (S = 0.153 s per request)
  t=0   10   20   30   40   50   60   70   80   90  100  110  120  130  140  150 ms
  |cpu1|-+------------ call A ------------+|cpu2|-+------ call B -----| ... continues
        ^ 45 ms of wall clock in which this request does nothing but wait

FANNED OUT (S = 0.067 s per request)
  |cpu1|---+call A--+
       |call B------|  <- three waits overlap; wall clock = max, not sum
       |call C--+   |
       |-+-gather-+
  0    10   20   30   40   50   60   70 ms
```

The area under each timeline is the request's in-flight time, and Section 4.1 says the average of that area is $L$. Fannning out does not remove the call wait; it **collapses three serial waits into one interval whose length is the slowest wait**. Geometrically, you are replacing $N$ adjacent rectangles with $N$ overlapping ones clipped to the tallest — which is exactly why the gain is capped by the ratio of waiting to total work.

### 6.2 The utilisation cliff as a curve, not a rule of thumb

Plot mean queue wait against utilisation for the M/M/c model with $c = 16$:

```text
W_q (ms, log-ish)
 10.0 |                                                        *
  3.0 |                                              *
  1.0 |                                    *
  0.5 |                          *
  0.1 |              *
  0.0 +----+----+----+----+----+----+----+----+----+----+----+----+--> rho
       0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.92 0.94 0.96 0.98 0.99
```

The shape is the entire reason "run at 95 % utilisation" is a bad policy without a latency requirement attached. The curve is flat where the system is comfortably inside its capacity and vertical where it is not, and the vertical region is narrow enough that *the difference between 90 % and 95 % is much larger than the difference between 60 % and 85 %*. There is no linear intuition to be had: doubling utilisation near the cliff does not double the wait, it multiplies it.

Second geometric fact worth internalising: for a *fixed* $\lambda$ and $S$, adding permits moves you left along the curve, but each permit costs idle capacity linearly while saving wait super-linearly — so there is a visible knee, and the sensible operating point is at or just past it. In Section 5.4 that knee is at $c \approx 14$: the step from 12→14 removes 81 % of the queue wait ($2.43 \to 0.46$ ms) and the step 14→16 removes a further 83 % of a much smaller number. Diminishing returns are visible in the table, which is what makes the choice defensible.

### 6.3 Little's law as conservation of area

```text
count in flight
 8 |            +--------+
 6 |     +------+        |
 4 |  +--+               +-----+
 2 |  |                        +----+
 0 +--+------------------------------> t
     ^ area of the shaded region = sum of sojourn times
       L = area / T  ,  W = area / (number of arrivals)  ,  L = lambda * W
```

Two ways of computing the same area give the law. The operational use: if you can measure two of $\{L, \lambda, W\}$ you have the third, and if the three disagree you have a *boundary definition* bug — usually a boundary that excludes retries from $\lambda$ or excludes queue time from $W$. That disagreement check is more useful than the formula, because it catches instrumentation that lies.

### 6.4 The fan-out tail as a union

```text
             unit SLO threshold T
  child 1  |------------------|            ok
  child 2  |-----------------------------|  ok
  child 3  |-----------------------------------|  SLOW  -> unit is slow
             ^ the unit's latency is the max of the children's
             P(unit slow) = P(any child slow) = 1 - (1-p)^N
```

Geometrically, the unit ends when the last child finishes, so the *maximum* statistic governs. Making $N$ larger extends the fan horizontally, which raises the chance that at least one stroke crosses the threshold — a union of events, not an average of them. This is why p99 fan-out arithmetic surprises people: averaging would suggest tails smooth out, and the geometry says the opposite.

### 6.5 Concurrency versus memory, on one axis

```text
parked tasks
     ^
     |                                            / unbounded create_task
     |                                          /
     |                                        /
     |                                ________/  bounded: flat at c
     |___________________________----
     +------------------------------------------> offered work
                 admission limit reached here
```

The bounded line is a design choice; the diagonal is what happens by default. The vertical distance between them is memory you either provision or lose.

### 6.6 The event loop as a single cashier

A useful everyday analogy with one precise mapping: a single cashier serving a queue. If a customer's request requires the cashier to walk to the back room and wait for a supplier (an `await` on I/O), the cashier *can* serve the next customer — that is async. If a customer hands over a 500-page form to be checked by hand (CPU work), the cashier stands there doing it while everyone waits — that is a blocking call on the loop. Adding cashiers is processes; adding a second queue for form-checking is a thread pool or a worker service. The analogy earns its place because it predicts the failure correctly: the shop's throughput collapses when the *cashier* does the slow thing, no matter how many customers are queued.

**Takeaway:** overlap is geometric (max, not sum), saturation is a cliff (not a ramp), tails add by union (not average), and unbounded creation is a diagonal you did not choose.

## 7. Comparison / Selection Table

Selection here is a filter chain, not a menu. Answer the questions in order and most options disqualify themselves before any preference enters.

```text
Q1  Is the dominant cost waiting on something external?   --no--> processes / more hosts
     |yes
Q2  Is there a maintained async client for that thing?    --no--> bounded thread offload
     |yes                                                            (compat layer)
Q3  Is there CPU work per unit above ~5 ms?               --yes-> offload to workers
     |no                                                             or a service
Q4  Can you state a bound c and a reject policy?          --no--> fix the bound first
     |yes
Q5  Do effects need idempotency/compensation?             --yes-> design the effect
                                                                     contract
```

### 7.1 Primary comparison

| Option | What is overlapped | Preconditions that must hold | Cost per unit of concurrency | Use when | Disqualify when |
|---|---|---|---|---|---|
| Sequential loop | nothing | none | ~0 | rate is far below capacity; simplicity is the SLO | latency budget needs overlap |
| `asyncio` + `gather` | I/O waits | every call on the path is non-blocking and cancellable | $\mathcal{O}(1)$ small object; ~KiB scale per task | many concurrent waits, high fan-out, chatty protocol | any sync library in the hot path; CPU >5 ms/unit |
| `asyncio` + `Semaphore` | I/O waits, bounded | as above plus a stated $c$ and a reject/queue policy | same plus a counter | production default for fan-out | $c$ chosen by guess; queue unbounded behind it |
| `asyncio.to_thread` / executor | blocking calls | thread-safe client; pool size stated | OS thread (~MiB virtual stack) | legacy SDK with no async client, small concurrency | called at high rate per request (pool starvation) |
| `concurrent.futures.ThreadPoolExecutor` | waits + GIL-releasing C | thread safety, no shared mutable state without locks | OS thread | CPU work inside GIL-releasing C (compression, hashing, some numeric libs) | pure-Python CPU bottleneck |
| `ProcessPoolExecutor` / multiprocessing | CPU, memory isolation | picklable inputs/outputs; memory per process affordable | process (tens of MiB+) plus IPC copy | parsing, image/PDF transform, feature computation | large payloads (IPC copy dominates); per-request spawn |
| External worker service (queue + consumers) | CPU and failure domains | durable queue, visibility timeout, idempotent consumer | infrastructure + operational surface | long tasks, retryable batch, independent scaling | request/response latency SLO below a few ms |
| Free-threaded build | CPU across threads | all extensions compatible; measured single-thread regression acceptable | thread | experimental; specialised | production plan without measurement |

### 7.2 Decision by symptom cluster

| Observed situation | Correct first move | Why not the obvious alternative |
|---|---|---|
| p99 latency high, CPU low, many waits | raise $c$ toward the Q-01 knee, or reduce $N$ | adding hosts adds idle hosts, not overlap |
| CPU pinned at 100 % on one core, latency high | find the blocking call with loop-lag instrumentation | increasing $c$ makes it worse (more parked work, same block) |
| Latency fine at low rate, falls over at 3× | check quota/connection-pool limits before permits | permits were never the binding constraint |
| Occasional 30 s hangs | audit every `await` on a Future nobody resolves; audit thread-pool starvation | timeouts configured per-layer hide it |
| Duplicate side effects after a deploy | idempotency keys plus effect journal | retries are not the root cause; effect ownership is |
| Memory grows with input size | add admission control at the creation point | GC tuning treats the symptom |
| Error rate spikes during incidents | retry budget plus circuit breaker | retries amplify load exactly when capacity is lowest |
| Shutdown loses in-flight work | structured shutdown with grace period and drain | `loop.close()` is not a drain |

### 7.3 Async versus threads, stated precisely

| Dimension | `asyncio` | Threads |
|---|---|---|
| Scheduling | cooperative, at `await` | preemptive (GIL switch interval, default 5 ms) |
| Preemption of a runaway loop | none — the loop is blocked | possible, other threads get the GIL |
| Memory per unit | small (frame + locals) | stack reservation per thread (platform-dependent; often MiB-scale virtual) |
| Context switches per second | high, but cheap and user-space | lower, but OS-scheduled |
| Debuggability | readable tracebacks per task; `Task` names help | stack traces of other threads are awkward |
| Shared state | single thread, no locks needed between awaits | requires locks; races are timing-dependent |
| Blocking library compatibility | fatal unless offloaded | works, but consumes a thread |
| Cancellation | cooperative, explicit, structured | interrupt-based (`Thread` has no safe cancel) |
| CPU parallelism | none | only through GIL release |
| Failure isolation | one task's exception is contained if awaited | one thread's exception is silent unless captured |

The row that decides most cases is *blocking library compatibility*: if the SDK you must call is synchronous and hot, threads are the honest answer for that call, and the async layer above must offload it with a bounded pool. The row people underestimate is *cancellation*: threads cannot be safely cancelled, which makes timeout-plus-cleanup strictly easier in async code — a genuine correctness advantage that has nothing to do with speed.

### 7.4 Choosing the bound $c$

| Basis for $c$ | Formula | Risk |
|---|---|---|
| Utilisation target | $c = \lceil \lambda S / \rho_{\text{target}} \rceil$ | needs a defensible $\rho$ and a service-time distribution check |
| Downstream quota | $c_{\text{eff}} = q \cdot S / N$ | quota may be per-credential and shared with other callers |
| Connection pool | $c \le$ pool size, or explicit queue wait | pool smaller than semaphore is a silent throttle |
| File descriptors | $c \le$ (soft ulimit − baseline) with margin | leak converts into a hard failure at the limit |
| Memory | $c \le$ budget / per-unit footprint | footprint grows with payload size |
| Measured knee | sweep $c$ and record p99 | measurement noise, and the knee moves with workload |
| SLO-derived | largest $c$ with $W_q$ budget met per Q-01 | model assumptions |

The practical procedure is: compute from utilisation, take the minimum against quota/pool/fd/memory, then verify by sweeping $c$ under a representative load and confirming p99 does not degrade. If the computed value and the swept knee disagree by more than a small factor, one of the inputs is wrong — most often $S$ measured under a different downstream state, or a hidden serialising resource.

**Takeaway:** the selection is decided by "what is blocking" and "what is the binding limit", and every option in the table is correct within a stated precondition and wrong outside it.

## 8. Failure Modes & Edge Cases

Each entry below follows the same discipline: **symptom → competing causes → discriminating check** (a check whose outcome differs between the candidates) **→ bounded fix → verification → residual limitation**. A discriminating check that returns the same answer for two causes is worthless; the point of the column is to separate them, so the checks are chosen to be *inconsistent* with the wrong hypothesis.

### F1 — `VIOLATED_ASSUMPTION` — Event loop stalls; latency collapses under modest load

**Symptom.** Throughput far below the C-01 ceiling while CPU of the *process* is moderate; p99 climbs in steps; timers fire late; health checks time out even though the service "is up".

**Competing causes.**
1. A blocking call inside a coroutine (sync HTTP client, `time.sleep`, a synchronous DB driver, a large `read` on a network mount).
2. Pure-Python CPU work per request large enough that $\lambda S_{\text{loop}}$ approaches 1.
3. Garbage-collection pauses from a large live heap (deep object graphs, many tasks).
4. A slow callback registered with `call_soon`/`add_done_callback` that never yields.

**Discriminating check.** Measure loop lag directly: a coroutine sleeps 10 ms and records actual elapsed minus 10 ms every 100 ms, logging the max per interval alongside the currently-running request span. Lag spikes aligned with specific spans confirm cause 1 or 2 and tell you *which* span. To separate them, profile while the stall reproduces: a blocking call appears as one long leaf in a single frame; CPU work appears distributed across many Python frames. For GC, log `gc.callbacks` start/stop with generation — a stall coinciding with a gen-2 collection and absent otherwise points at 3. For callbacks, wrap `call_soon` at debug time and log durations.

**Bounded fix.** Move the blocking call to `asyncio.to_thread` with a stated executor size (or replace the client); move CPU work to a process pool or an external service; for GC, reduce per-request allocation and tune thresholds only after measuring. Never "fix" it by raising the semaphore.

**Verification.** Re-run the lag probe under the same load: max lag below budget (for example under 20 ms) and p99 matching the service-time arithmetic. Then remove the fix and watch lag return — that step distinguishes a real fix from a lucky run.

**Residual limitation.** Lag probes cost a timer per interval and only sample; stalls shorter than the probe interval hide. Sampling profilers miss very short blocking calls at high rate.

### F2 — `CONCEPT_CONFUSION` — "We used async, so it is parallel" and the CPU-bound path is unchanged

**Symptom.** The rewrite shipped; I/O-heavy endpoints improved; the batch/transform endpoint is exactly as slow; adding `await` before a CPU function changed nothing.

**Competing causes.**
1. The work is CPU-bound, so the expectation was wrong, not the code.
2. The work is CPU-bound *inside a C call that releases the GIL*, so threads or processes would help while coroutines cannot.
3. The work is I/O but the "async" client is a synchronous client wrapped in `async def` — a shim that never suspends.

**Discriminating check.** Compare wall time against CPU time for one unit (`time.process_time()` versus `time.perf_counter()`); a unit at 95 % CPU can only be helped by CPU resources. Then prove the client suspends: run a ticker coroutine incrementing a counter alongside it and assert the counter advances during the call. A shim shows zero ticks.

**Bounded fix.** Processes or a worker service for CPU work; threads if and only if the heavy work is inside GIL-releasing C; replace shims with real async clients, or offload them explicitly.

**Verification.** Worker-pool utilisation plus per-unit CPU time; a throughput curve that scales with worker count up to physical cores and then flattens.

**Residual limitation.** Process pools pay serialization cost per task; with large payloads the IPC copy can erase the parallel gain (see F7).

### F3 — `SCALE_FAILURE` — Unbounded task creation; memory grows with input until the OOM killer arrives

**Symptom.** Memory proportional to input size rather than to concurrency limit; RSS climbs during batch ingestion; the process is killed without a Python traceback; latency degrades gradually before death.

**Competing causes.**
1. `create_task`/`gather` over an unbounded iterable with no semaphore.
2. A queue with `maxsize<=0` used as the buffer between producer and consumer.
3. Results accumulated in a list while tasks are still running (the list, not the tasks, is the leak).
4. Per-task buffers sized from payload size rather than from a constant.

**Discriminating check.** Sample `len(asyncio.all_tasks())` once per second and plot against input size: flat for a bounded design, diagonal for the unbounded one — conclusive for cause 1. Distinguish cause 3 by checking whether retained memory is dominated by payload objects or by frames/tasks (`tracemalloc` grouped by traceback).

**Bounded fix.** A semaphore or a fixed worker set consuming a bounded queue; write results incrementally to a sink; block the producer (backpressure) unless dropping is an explicit policy with its own metric.

**Verification.** Memory flat while input size varies by an order of magnitude; a 10× input load test holding RSS in budget; a test asserting task count never exceeds $c + \text{constant}$.

**Residual limitation.** Bounded admission converts an OOM into latency or rejection — the intended trade, but it must be visible as a metric rather than a mystery 504.

### F4 — `CONCEPT_CONFUSION` — Cancellation treated as rollback

**Symptom.** After timeouts or disconnects, downstream state shows partial effects: a charge without an order record, a created resource nobody references, a job still running after the caller left.

**Competing causes.**
1. Cancellation never reached the remote system (no protocol support; the request was cancelled locally mid-flight).
2. The remote received it and committed anyway (its own semantics).
3. Local code swallowed `CancelledError`, so the cleanup path never ran.
4. Cleanup ran, but a retry created a second effect because the key was not idempotent.

**Discriminating check.** Correlate the request ID at both ends: a downstream log entry for the ID rules out nothing by itself, but *multiple committed entries with different timestamps for one key* identifies cause 4; *no entry at all* points at 3 or at a request never sent. A static scan for `except:` / `except Exception` / `except BaseException` wrapping `await` identifies 3.

**Bounded fix.** Never swallow `CancelledError`; use `try/finally` for local cleanup and re-raise; attach idempotency keys to every mutating call; where the protocol cannot cancel, use a compensating action plus a reconciliation sweep.

**Verification.** Fault injection: cancel mid-call and assert exactly one committed effect per key over repeated runs, exactly one cleanup execution, and no task remaining in `all_tasks()` after the scope exits.

**Residual limitation.** Compensation is itself an effect that can fail; the reconciliation sweep is the backstop and must alert on the age of unreconciled items.

### F5 — `SCALE_FAILURE` — Retry storm: the policy removes the capacity it was meant to protect

**Symptom.** During a dependency degradation, request rate at the dependency rises 2–3× above baseline, its latency worsens, and the outage outlasts the trigger. Client, proxy and SDK retries all fire.

**Competing causes.**
1. Unbudgeted retries with no cap: R-01 with $f \to 1$ gives multiplier $k+1$.
2. Synchronised retries without jitter, producing a herd at fixed offsets.
3. Multi-layer retries multiplying (client × proxy × service × SDK), so effective $k$ is a product.
4. Retries on non-retryable classes (4xx), spending the budget on calls that cannot succeed.
5. No circuit breaker, so retries continue after the dependency is known dead.

**Discriminating check.** Measure the ratio of *attempts to logical requests* at the dependency during an incident. Near 1 rules out 1 and 3; near $k+1$ confirms one layer; near a product (9 = 3×3) identifies 3. Histogram attempt timestamps modulo the backoff base: a spike at a fixed offset confirms 2. Break attempts down by status class to detect 4.

**Bounded fix.** A retry budget expressed as a fraction of *successful* traffic, full-jitter exponential backoff, retries only on idempotent operations with idempotency keys, retries enabled at exactly one layer, and a circuit breaker with a stated threshold plus a half-open probe.

**Verification.** Failure injection at the dependency: the attempts-to-requests ratio stays within budget, non-retryable classes are attempted exactly once, and recovery time beats the unbounded policy in a side-by-side run.

**Residual limitation.** A budget penalises the unlucky request that needed its retry mid-outage; that is the deliberate trade and must be documented.

### F6 — `NUMERICAL_EDGE` — The semaphore exceeds the binding resource, so the limit is fiction

**Symptom.** Config says `max_concurrency = 100`; observed parallelism plateaus at 20; latency grows linearly past a certain rate with neither CPU saturation nor queueing at the service.

**Competing causes.**
1. Connection pool smaller than the semaphore, so 80 permit-holders wait for a connection.
2. Downstream quota (per-credential or per-tenant) below the semaphore-implied rate.
3. File-descriptor soft limit reached, surfacing as intermittent connection errors rather than a clean failure.
4. A lock or one shared session object serialising part of the path.
5. A synchronous section guarded by a lock (for example a token refresh) admitting one at a time.

**Discriminating check.** Compare four steady-state counters: permits held, pool connections checked out, downstream requests in flight (from the dependency's metrics), and open fds. Whichever plateaus first is binding. A lock is identified by a wait-time histogram inside the critical section, or by concurrency capping while none of the other three plateau.

**Bounded fix.** Set $c=\min(\text{all limits})$ and assert the invariant at startup (fail fast if semaphore > pool); raise quota deliberately or reduce requests per unit via $N$, batching or caching; raise the fd limit with margin and alert near it.

**Verification.** Startup assertion plus a load test pushed to the ceiling that shows the plateau at the intended number with the intended error class.

**Residual limitation.** Startup assertions cannot see limits owned by another team; every quota needs an owner, a value and an alert.

### F7 — `NUMERICAL_EDGE` — A process pool slower than sequential because of IPC and pickling

**Symptom.** The transform moved to processes, main-process CPU fell, but end-to-end latency rose; throughput worsened for small payloads and improved for large ones.

**Competing causes.**
1. Pickling/unpickling cost per task exceeds the compute saved.
2. Large payloads copied over a pipe, so bandwidth dominates.
3. Pool size above physical cores, adding context switches and cache thrash.
4. Per-request pool creation (fork/start per call) instead of a long-lived pool.
5. No shared memory where it would apply, so the copy is paid every task.

**Discriminating check.** Time the components separately: pickle, IPC transfer, compute, unpickle. If $(\text{pickle}+\text{IPC}+\text{unpickle})/\text{compute} > 0.3$ the boundary is wrong. Then compare pool size with `os.cpu_count()` and check whether process count grows with request count (cause 4).

**Bounded fix.** Batch small items into one task to amortise IPC; send paths/handles rather than payloads when the data is already on disk; use shared memory for large arrays; keep one long-lived pool sized to physical cores; move compute to the data when the data is large.

**Verification.** Latency distributions compared on the real payload-size distribution, not a microbenchmark on a toy payload.

**Residual limitation.** The correct boundary depends on the payload-size distribution, which drifts; a pool tuned for 1 MB payloads is wrong for 1 KB payloads.

### F8 — `VIOLATED_ASSUMPTION` — Shutdown and deploy lose in-flight work and produce duplicates

**Symptom.** Deploys produce a burst of 5xx; some requests are retried and duplicated; each instance turnover raises the error rate.

**Competing causes.**
1. No signal handling or drain: the process exits with tasks pending.
2. Drain exists, but the grace period is shorter than the p99 of in-flight work.
3. The readiness probe stops routing only after the process begins exiting, so new work arrives during the drain.
4. Effects committed before exit are never reconciled, and the client retry duplicates them.
5. Children ignore cancellation, so the drain never completes and the process is killed.

**Discriminating check.** Log pending-task count and names at exit and align with the error burst. Pending > 0 at exit indicates 1 or 2. A burst *before* the drain starts indicates 3. Duplicates with no error indicates 4. A "waiting for N tasks" line that never reaches zero before the kill indicates 5.

**Bounded fix.** On SIGTERM: flip readiness to false, stop accepting, await in-flight with a deadline, cancel the remainder and log the count; set the grace period above the in-flight p99 with margin; use idempotency keys so a retried commit deduplicates; make children honour cancellation.

**Verification.** A deploy under sustained load over a 30 s window: zero client-visible errors, zero duplicate effects per key, logged pending count zero at exit.

**Residual limitation.** A grace period long enough for every workload may exceed the orchestrator's hard kill timeout; long tasks belong in a durable queue, not the request path.

### 8.9 Cross-cutting edge cases worth naming before they bite

| Edge case | Mechanism | Check | Handling |
|---|---|---|---|
| Timeout below the loop's effective timer granularity | timers fire late under lag | compare configured versus observed timeout | keep timeouts well above measured lag; fix lag first |
| `gather(return_exceptions=True)` then ignoring results | failures become silent data loss | assert result types; count exception entries | classify results into a funnel; log counts |
| Permit acquired and released on different paths | permits leak until the pool is empty | gauge of permits held must return to zero at idle | `async with` or `try/finally` |
| Outer `gather` cancelled while children run | children left running if unstructured | count tasks after scope exit | TaskGroup or explicit cancel-and-await |
| Blocking DNS resolution | resolver runs in a small default thread pool | first-request and cache-miss latency spikes | async resolver or pre-resolve and cache with TTL |
| One tenant's fan-out consumes all permits | no fairness in a global semaphore | per-tenant in-flight gauge | per-tenant semaphore or weighted fair queue |
| Retry of a streamed upload | stream already consumed; retry sends nothing | bytes received at the receiver | require seekable input or buffer with a cap |
| Clock skew in deadline propagation | remaining budget computed from a remote clock | compare offsets across hosts | propagate a duration, or synchronise clocks |
| Loop closed with pending tasks | `RuntimeError` at shutdown or silent loss | log at close | await cancellation before closing |
| Exception in a fire-and-forget task | warning at GC time; work lost | a loop `exception_handler` | log with task names; never fire-and-forget silently |

**Takeaway:** every failure above reduces to one of three questions — *is the loop free?*, *is the bound real?*, *is the effect owned?* — and each discriminating check is designed to answer exactly one of them.

## 9. Complexity & Cost Analysis

Complexity here has four currencies, and a design is only sized when all four are stated: **scheduler time per step**, **memory per in-flight unit**, **wall clock per unit** (Section 4), and **money per unit of work**. Confusing any two of them produces the classic error of "it is async, so it scales".

### 9.1 Algorithmic costs of the concurrency primitives

These bounds are about the *bookkeeping*, not about the I/O, and they matter because at high rates the bookkeeping is a real fraction of the loop's CPU budget ($\rho_{\text{loop}}$ from Section 5.8).

| Operation | Time | Space | Assumption / note |
|---|---|---|---|
| Create a task | $\mathcal{O}(1)$ | one Task object + coroutine frame | allocation dominates; frame size grows with locals |
| `await` on a not-ready future | $\mathcal{O}(1)$ | one waiter entry | registering a callback on the future |
| Wake a ready task | $\mathcal{O}(1)$ amortised | one entry in the ready deque | deque append/popleft |
| Semaphore acquire (uncontended) | $\mathcal{O}(1)$ | — | counter decrement |
| Semaphore acquire (contended) | $\mathcal{O}(1)$ amortised | one waiter per parked acquirer | FIFO waiter list; no fairness proof in the language spec |
| Timer scheduling / cancellation | $\mathcal{O}(\log n)$ | one heap entry per timer | $n$ = live timers; every timeout is a timer |
| Ready-loop dispatch of $k$ runnable tasks | $\mathcal{O}(k)$ | — | each step runs until its next suspension |
| `gather` over $N$ children | $\mathcal{O}(N)$ setup | $N$ Task + result slots | plus the children's own costs |
| TaskGroup exit | $\mathcal{O}(k)$ | — | $k$ = children; includes cancellation and exception aggregation |
| Bounded queue put/get | $\mathcal{O}(1)$ amortised | capacity × item size | `maxsize` is the memory bound |
| Thread creation | $\mathcal{O}(1)$ syscall, constant factor large | stack reservation (platform-dependent, often MiB-scale virtual) | amortise by pooling |
| Process creation | $\mathcal{O}(1)$ syscall, constant factor larger | full interpreter address space | fork/start method changes cost and semantics |
| Pickling a task payload | $\mathcal{O}(b)$ | $\mathcal{O}(b)$ transient | $b$ = serialised bytes; the hidden tax on process pools |
| IPC transfer of $b$ bytes | $\mathcal{O}(b)$ | $\mathcal{O}(b)$ in flight | pipe or socket bandwidth bound |

Two consequences worth stating plainly:

1. **Every timeout costs a heap operation.** A per-request timeout is not free at 100 k req/s; it is 100 k heap insertions and removals per second, plus the timer callback. This is a real cost that shows up in $\rho_{\text{loop}}$, and it is one reason to prefer coarser timeouts at high rates (a single sweeping deadline checker over a sorted wheel) rather than one timer per request.
2. **Task creation is $\mathcal{O}(1)$ but not free.** At 100 k tasks/s the allocation rate dominates the young generation and drives GC frequency, which feeds back into loop lag (F1 cause 3). Reusing a fixed worker set with a queue trades $\mathcal{O}(1)$-per-task allocation for $\mathcal{O}(1)$-per-item queue operations — same asymptotics, very different constants.

### 9.2 Loop CPU cost model and the rate ceiling it imposes

The loop's per-request cost has three parts: application CPU before/between awaits ($C_{\text{app}}$), scheduler cost per suspension round trip ($C_{\text{sched}}$ in seconds, multiplied by the number of round trips $m$), and per-request fixed overhead (logging, context, metrics). Summing gives the loop's service demand, whose inverse is the loop's own ceiling:

```text
FORMULA [K-01]  loop-thread ceiling
    C_loop   = C_app + m * C_sched + C_fixed        [s/request]
    lambda_loop_max = rho_target / C_loop           [requests/s]

WHERE
    C_app    application CPU on the loop thread, per request   [s]
    m        await round trips per request that actually suspend [count]
    C_sched  loop cost per suspend/resume pair                 [s]
    C_fixed  per-request fixed overhead                        [s]
    rho_target chosen headroom for the loop (e.g. 0.70)

CONSTRAINT
    C_sched is version- and platform-dependent and must be measured on
    your interpreter; treating it as zero is the common error.

NUMERICAL EXAMPLE
    C_app = 0.000800 s, m = 6, C_sched = 0.000004 s, C_fixed = 0.000100 s
    C_loop = 0.000800 + 6*0.000004 + 0.000100 = 0.000924 s
    lambda_loop_max(0.70) = 0.70 / 0.000924 = 757.576 requests/s

INTERPRETATION
    At 120 req/s the loop is at 120*0.000924 = 0.1109 -> 11.1% busy.
    At 1 000 req/s it is at 0.924 -> past the headroom and heading for 1.
    The difference between 757 and 1080 (if C_sched were zero) is 30% of
    the ceiling, which is exactly the error made by ignoring scheduler cost.
```

`C_sched` is labelled **OPEN** for your environment until you measure it: run a coroutine that awaits a resolved future $10^6$ times and divide. Published figures for loop throughput exist, but version, platform and payload all move them, so a borrowed number is a guess wearing a citation.

### 9.3 Memory: what one unit of in-flight work actually costs

| Component | Order of magnitude | Notes |
|---|---|---|
| Task object + coroutine frame | hundreds of bytes to a few KiB | grows with local variables and closures |
| Per-request buffers | payload-dependent; often dominant | the reason memory tracks payload size, not task count |
| Connection/socket structures | small, but fd-limited | each open fd has kernel-side cost too |
| TLS session state | KiB-scale | matters at thousands of connections |
| Timer entry | tens of bytes per live timer | matters at high rate with per-request timeouts |
| Result accumulation | unbounded if you keep results | the most common real leak |

The useful arithmetic is the one from Section 5.11: at 19 KiB per in-flight unit, 50 000 parked units cost 928 MiB. Reverse it into a limit:

```text
c_max_mem = memory_budget / footprint_per_unit
          = 256 MiB / 19 KiB = 262144 / 19 = 13797 units
```

So a 256 MiB budget supports about 13 797 units of that size — comfortably more than the 16 permits the workload needs, which means for *this* workload memory is not binding and the quota is (Section 5.5). Stating that comparison explicitly is what stops a team from "solving" memory with a smaller semaphore when the real limit is a downstream quota, or vice versa.

### 9.4 Wall-clock cost per unit, and the fan-out speedup limit

The serial fraction sets the ceiling on overlap. With a fraction $s$ of the work that cannot be overlapped, the maximum speedup from unlimited concurrency is $1/s$ — an Amdahl-form bound that applies here because the same arithmetic governs "fraction of service time that is not overlappable".

```text
FORMULA [A-01]  overlap speedup ceiling
    S_max = 1 / (s + (1 - s)/N_inf) -> 1/s   as N_inf grows
    practical: S(N) = 1 / (s + (1 - s)/N)

WHERE
    s  non-overlappable fraction of service time  [dimensionless]
    N  achieved overlap (parallel calls in flight for one unit)
    S  speedup on per-unit service time           [dimensionless]

CONSTRAINT
    fixed work content; no resource contention at higher N;
    overheads of fan-out folded into s (as in 5.2, where they are).

NUMERICAL EXAMPLE  (Section 5.2 numbers; gather treated as a fan-out tax,
                   so it is excluded from s and added back afterwards)
    s     = (0.0015 + 0.0165) / 0.153 = 0.018 / 0.153 = 0.117647
    S_max = 1 / 0.117647 = 8.5000
    S(3)  = 1 / (0.117647 + 0.882353/3) = 1 / (0.117647 + 0.294118)
          = 1 / 0.411765 = 2.4286        (= 0.153 / 0.063; the 0.063 s
                                            ideal fan-out time)
    achieved = 0.153 / 0.067 = 2.2836     (0.067 includes 4 ms gather)
    tax      = 2.4286 - 2.2836 = 0.1450x, i.e. 6.0% of the ideal gain

INTERPRETATION
    Even with infinite parallel lookups this workload cannot go below
    1/8.5 = 11.8% of its sequential time, because 11.8% of it is local
    CPU. At fan-out 3 the ideal is 2.43x and the achieved gain is 2.28x;
    the difference is the fixed overhead fan-out itself introduces,
    which is why adding parallelism to a cheap call can make things
    slower rather than faster.
```

### 9.5 Money, in units

| Cost line | Formula | Worked value (from Section 5) | Unit |
|---|---|---|---|
| Compute per request | $C_{\text{inst}}/3600/\lambda_{\text{inst}}$ | $0.10/3600/120 = 2.3148\times10^{-7}$ | USD/request |
| Compute per million | previous × $10^6$ | $0.231481$ | USD/1M requests |
| Compute per million, 2 instances | × 2 | $0.462963$ | USD/1M requests |
| Idle cost of over-provisioning | $\Delta c/c \times$ base | 16→20 permits ≈ +25 % permits, ≈ 0 extra throughput | USD/month |
| Retry cost | $(A_k-1)\times$ base × attempts | $f=0.08$: 8.64 % extra calls → 8.64 % extra downstream load | requests/s |
| Cache cost to meet quota | hit rate $h$ required (5.5) | 30.56 % at 3 lookups/request | dimensionless |
| Incident cost | not modelled here | — | OPEN |

The last row is deliberate. The compute bill for this gateway is about half a dollar per million requests; the retry multiplier at $f=0.8$ is 2.44×, and a 20-minute outage at 120 req/s drops 144 000 requests. Any argument that trades reliability for a few percent of compute cost is trading dollars against orders of magnitude, and should be rejected on the arithmetic.

### 9.6 Sizing summary as a single checklist

```text
1  C_loop  = C_app + m*C_sched + C_fixed              -> lambda_loop_max
2  S_fan   = cpu + max(call) + gather                 -> per-unit service time
3  c_util  = ceil(lambda * S_fan / rho_target)        -> permits from utilisation
4  c_eff   = min(c_util, q*S/N, pool, fd, mem)        -> real limit
5  A_k     = sum f^i                                  -> dependency load
6  P_tail  = 1-(1-p)^N                                -> inherited tail
7  T_budget= sum t_i + eps <= B                       -> timeout closure
8  mem     = c_eff * footprint                        -> RSS budget
9  cost    = instances * price / requests             -> USD per 1M
```

Every row produces a number with a unit, and rows 4 and 7 are the ones that most often fail review because they require information owned by someone else (quota, downstream latency).

**Takeaway:** the loop has its own ceiling that is not the pool ceiling; memory is a function of the bound not the input; the overlap speedup is capped by the non-overlappable fraction; and the money is small enough that reliability, not compute, is the constraint worth optimising.

## 10. Key Theorems / Results Reference

Each entry states the result, the exact hypotheses (because the hypotheses are the useful part), what the result does *not* say, and the practical corollary.

### T-01 — Little's law $L=\lambda W$

**Statement.** For a system in steady state over an observation window, with a declared boundary, mean arrival rate $\lambda$, mean sojourn time $W$ and mean in-system count $L$, all finite, $L=\lambda W$ holds exactly.

**Hypotheses.** (a) steady state over the window; (b) boundary defined so units are neither created nor destroyed inside; (c) the three means exist; (d) window long enough that initial and terminal transients are negligible.

**What it does not say.** Nothing about distributions, variance, tails, blocking, or capacity. It does not require Poisson arrivals, exponential service, independence, or FIFO — this is why it survives in messy real systems.

**Corollary used in this sheet.** $L$ is an *occupancy* number, so a pool sized equal to $L$ sits at 100 % average utilisation and therefore queues (Section 4.1, Section 5.3). Provisioning uses C-01 with a target $\rho$ below 1.

**Evidence.** DERIVED (conservation of area under the in-flight count). Proof for general systems appears in standard queueing texts; the derivation in 4.1 is the discrete version.

### T-02 — Throughput ceiling and the utilisation identity

**Statement.** With $c$ workers each serving one unit at a time with mean service time $S$, the aggregate service rate is $\mu_{\max}=c/S$; stability requires $\lambda<\mu_{\max}$; utilisation $\rho=\lambda S/c$ is the fraction of worker-time that must be occupied.

**Hypotheses.** One unit per worker; $S$ measured on the same machine/downstream regime; $\lambda$ excludes retries; no hidden serialising resource.

**What it does not say.** It does not promise that $\lambda=\mu_{\max}$ yields timely service — at $\rho=1$ the backlog is a positive-drift random walk. It also does not bound latency; that is Q-01's job.

**Corollary.** Effective $c=\min$ over all shared limits (Section 4.4, limit 2; F6), so the formula must be applied with the *real* $c$, not the configured one.

**Evidence.** DERIVED (rate arithmetic). Any contradiction with measurement indicates that $S$ or $c$ is wrong, which is itself the diagnostic.

### T-03 — Retry amplification and the correlated-failure limit

**Statement.** With independent per-attempt failure probability $f$ and $k$ retries, expected attempts per logical request is $A_k=\sum_{i=0}^{k} f^i$, so dependency load is $\lambda A_k$; as $f\to1$, $A_k\to k+1$.

**Hypotheses.** Independence of failure across attempts; retries consume full capacity; no budget cap; the same failure class is retried.

**What it does not say.** It does not model the *feedback* case where retries cause the failure (retry-induced overload raises $f$), which is the actual storm mechanism; the formula gives the bound that makes the feedback plausible.

**Corollary.** Correlated failure is the design case, not the tail case: during a dependency incident $f$ is near 1, so any retry policy without a budget and a circuit breaker is a multiplier of order $k+1$ at exactly the wrong moment.

**Evidence.** DERIVED (geometric series) with ILLUSTRATIVE numbers in 5.6.

### T-04 — Tail fan-out union bound

**Statement.** For $N$ independent children each exceeding a threshold with probability $p$, the probability the maximum exceeds it is $1-(1-p)^N$, which is at least $p$ and increasing in $N$; for small $Np$ it is approximately $Np$.

**Hypotheses.** Independence. Shared downstream, shared connection pool, shared loop time and shared GC all violate it — and violation makes the true probability *higher* than the independent estimate, since failures cluster.

**What it does not say.** Nothing about the shape of the latency distribution beyond the one threshold, and nothing about means.

**Corollary.** Fan-out spends tail budget: at $p=0.02$, $N=3$ inherits 5.88 % and $N=50$ inherits 63.58 %. Tail SLOs and high fan-out are in direct tension; hedging is the priced way to buy the tail back.

**Evidence.** DERIVED, with arithmetic checked in 5.7.

### T-05 — Timeout composition (worst-case additivity)

**Statement.** For $m$ sequential stages with timeouts $t_i$ and local overhead $\epsilon$, the caller's worst-case wait is $\sum t_i+\epsilon$; a budget $B$ is respected iff that sum is at most $B$.

**Hypotheses.** Sequential stage execution; timeouts are enforced while the loop is healthy.

**What it does not say.** It says nothing about the *typical* case, and it does not create an obligation for the downstream to stop — a timeout is a local decision (Section 3.5).

**Corollary.** Timeouts must be apportioned and the remainder propagated; three independent 2 s timeouts compose into 6.05 s, which is why "each layer sets 2 s" is not a policy but an accident.

**Evidence.** DERIVED (arithmetic composition), checked in 5.10.

### T-06 — Amdahl-form overlap ceiling

**Statement.** If a fraction $s$ of a unit's service time cannot be overlapped, then with achieved overlap $N$ the speedup is $1/(s+(1-s)/N)$, bounded above by $1/s$ as $N\to\infty$.

**Hypotheses.** Fixed work content; no contention or overhead growth at higher $N$; overlap is perfect within the overlappable fraction.

**What it does not say.** It does not model queueing at the downstream when many units fan out simultaneously, which is exactly what turns a latency win into a capacity loss.

**Corollary.** Measure $s$ before promising a fan-out gain: at $s=0.144$ the ceiling is 6.95×, and at $s=0.65$ it is 1.54× and the rewrite is not worth the failure surface.

**Evidence.** DERIVED (algebra of the serial/parallel decomposition); the classical statement is Amdahl's law, applied here to service time rather than instruction throughput.

### T-07 — Erlang C (M/M/c waiting probability) and its mean wait

**Statement.** For Poisson arrivals at rate $\lambda$, exponential service with mean $S$, $c$ identical servers, infinite buffer and FIFO, the probability an arrival waits is $C(c,a)$ as given in Q-01, and the mean wait is $W_q=C(c,a)\,S/(c-a)$ with $a=\lambda S$.

**Hypotheses.** The five assumptions above, plus $\rho<1$.

**What it does not say.** Real web traffic is bursty and service times are heavy-tailed, so absolute values are unreliable; finite buffers replace delay with rejection; priorities invalidate FIFO. Use it for *shape* and for *relative* comparison of candidate $c$ values.

**Corollary.** The knee in $W_q$ versus $\rho$ is steep enough that 0.70 and 0.90 are qualitatively different operating points, so the utilisation target is a policy decision that must be written down with its latency consequence.

**Evidence.** DERIVED within the model (standard result); the table in 5.4 is ILLUSTRATIVE arithmetic of the model, not a measurement.

### T-08 — Conservation of effects under at-least-once delivery

**Statement.** If a message or request may be delivered or retried more than once, and the operation is not idempotent with respect to a stable key, then the number of committed effects is not bounded by the number of logical operations.

**Hypotheses.** At-least-once transport or client retry; effects are durable; no dedupe at the effect boundary.

**What it does not say.** It does not require exactly-once infrastructure; idempotency keys plus a dedupe store or a naturally idempotent operation (set-to-value, conditional write, upsert keyed by a deterministic ID) is sufficient.

**Corollary.** "We retry on timeout" is a commitment to duplicate effects unless the operation is keyed. Cancellation does not undo this (T-09). The reconciliation sweep is the backstop for effects whose outcome is unknown.

**Evidence.** DERIVED (counting argument).

### T-09 — Cancellation is local and cooperative

**Statement.** In `asyncio`, cancellation raises `CancelledError` at a suspension point in the cancelled task; it cannot interrupt code between suspension points, and it cannot reach a remote system that does not implement cancellation.

**Hypotheses.** The task is scheduled by a running loop; the coroutine reaches a suspension point; nothing swallows the exception.

**What it does not say.** It does not provide rollback, and it does not guarantee that a request already sent was not executed.

**Corollary.** Every mutating call needs an idempotency key or a compensator *before* timeouts are configured, because after the fact the only available answers are reconciliation and apology.

**Evidence.** DERIVED from the documented semantics of the runtime, with the version caveat that eager execution of non-suspending coroutines (available in recent CPython versions through a task factory) changes *when* bodies run but not the cooperative nature of cancellation.

### T-10 — Bounded admission converts unbounded memory into bounded latency or rejection

**Statement.** If work arrives faster than it can be admitted and the buffer is bounded at capacity $K$, then sustained overload results in either rejection or producer blocking; memory stays $\mathcal{O}(K)$ and cannot grow with input size.

**Hypotheses.** The bound is enforced at or before the creation of per-unit state; the producer can be blocked or can accept rejection.

**What it does not say.** It does not say which of rejection and delay is correct — that is a product decision — and it does not help if a downstream resource (quota) is the binding constraint, in which case admission must reflect the quota.

**Corollary.** Unbounded `create_task` over input of size $n$ is $\mathcal{O}(n)$ memory by construction (Section 5.11, F3); the fix is structural, not a GC tuning parameter.

**Evidence.** DERIVED (memory accounting).

**Takeaway:** the ten results above are all small pieces of arithmetic or counting, and the review habit that catches most real mistakes is checking which of their hypotheses the situation actually satisfies.

## 11. Interview / Exam Questions

Twelve questions, each with a checked answer, the reasoning, and the red flag that would make me doubt a candidate who gave it. Numbers are consistent with Sections 4–5, so a candidate can be walked to them.

### Q1 — "Async makes Python faster, right?"

**Checked answer.** No. Async overlaps *waiting*; it does not create CPU parallelism, because all coroutines run on one thread. On I/O-bound work with high per-unit wait, overlap reduces wall-clock service time (Section 5.2: 0.153 s → 0.067 s, a 2.28× improvement). On CPU-bound work, async is neutral at best and worse than sequential if the work blocks the loop, because it also delays every other task's bookkeeping and timers.

**Reasoning trail.** The only thing gained is that the loop can run other tasks while this one is parked. Between suspension points nothing else runs. So the gain is bounded by the wait fraction, and the loss appears when a coroutine does not suspend at all.

**Red flag.** "It uses the event loop so it's concurrent, therefore multi-core." That conflates concurrency with parallelism and predicts a speedup on CPU work, which is the failure in F2.

### Q2 — "You set `max_concurrency = 100`. How do you know 100 is right?"

**Checked answer.** I do not, until I compute it. With $S=0.0670$ s and a target $\rho=0.70$, $c=\lceil\lambda S/\rho\rceil$; at $\lambda=120$ that is $\lceil 8.04/0.70\rceil=\lceil 11.4857\rceil=12$. Then I take the minimum against every shared limit: the downstream quota (250 lookups/s at 3 lookups per request = 83.3 units/s), the connection pool size, the fd limit and the memory budget. The effective $c$ is the smallest; 100 would be a number with no mechanism behind it and would park 100 units against a resource that can serve fewer.

**Reasoning trail.** Two distinct numbers are being conflated by the question: the semaphore (a policy) and the binding resource (a fact). Only the second determines throughput.

**Red flag.** "100 because it's a common default" or "we benchmarked until it stopped improving" without stating the binding resource; benchmarking at one load level can miss the quota.

### Q3 — "At 120 req/s with 0.067 s service time, how many requests are in flight? Is that your pool size?"

**Checked answer.** $L=\lambda W=120\times0.067=8.04$, so about 8 on average. It is *not* the pool size: 8 permits means 100 % average utilisation, and any burst or retry pushes it over, so the queue grows without bound in expectation. The pool comes from a utilisation target (12 at $\rho=0.70$, 16 with burst headroom); Little's law only converts rate and time into an occupancy.

**Reasoning trail.** Little's law is a conservation identity; it says nothing about capacity. Using it for provisioning is the single most common sizing error in async designs.

**Red flag.** "About 8, so 8 workers." The candidate has the right formula and the wrong decision rule.

### Q4 — "Your service has 3 downstream calls per request. What happens to your p99 when you fan them out?"

**Checked answer.** Fan-out governs the maximum, so with per-call $p=0.02$ at the threshold, the unit inherits $1-(1-0.02)^3=1-0.941192=5.88\%$. At $N=10$ it is 18.29 %, at $N=50$ it is 63.58 %. The mean improves (0.153→0.067 s) while the tail worsens relative to the mean, which is why a fan-out that "improves latency" can breach a p99 SLO. Remedies: cap $N$, reduce the per-call tail, or hedge with a strictly bounded and budgeted duplicate call.

**Reasoning trail.** The unit finishes when the slowest child finishes, so the relevant statistic is the maximum; a union of tail events grows with $N$.

**Red flag.** "Averaging over 3 calls smooths the tail." It does the opposite, and the candidate who believes this will ship a fan-out of 20.

### Q5 — "A client times out. What has definitely happened downstream?"

**Checked answer.** Nothing definite. The timeout is a local decision taken by the loop's timer; the request may not have been sent, may be executing, or may have committed. What is certain is that you no longer know the outcome, which is why every mutating call needs an idempotency key or a compensator, and why cancellation must not be described as rollback (T-08, T-09, F4).

**Reasoning trail.** Cancellation is cooperative and local; effects are remote and durable. The only sound engineering response to an unknown outcome is a deduplicating key or a reconciliation sweep.

**Red flag.** "The timeout cancels the request so nothing happened," or "we call the rollback endpoint on timeout" without checking whether the rollback itself can fail or arrive before the original.

### Q6 — "Your retry policy is 2 retries. How much extra load can it add to the dependency?"

**Checked answer.** For independent failures at rate $f$, $A_3=1+f+f^2$. At $f=0.08$ that is 1.0864 (8.64 % extra); at $f=0.5$ it is 1.75; at $f=0.8$ it is 2.44. With 120 req/s and 3 lookups each, the base is 360 lookups/s, so $f=0.8$ means 878 lookups/s against a 250/s quota. Correlated failure is the design case, so the answer must also mention a retry budget as a fraction of successful traffic, full jitter, idempotency, one retrying layer, and a circuit breaker.

**Reasoning trail.** The multiplier is a geometric sum; correlation drives $f\to1$ precisely when the dependency is least able to absorb load.

**Red flag.** "Retries are cheap, they only fire on errors." That is the mechanism of the storm: the errors are the moment retries arrive.

### Q7 — "p99 is 800 ms, CPU is 30 %, and the pool semaphore is fully acquired. Diagnose."

**Checked answer.** Three families, separated by measurement. (a) The loop is blocked: measure loop lag and profile; a blocking call or heavy CPU span is the cause (F1). (b) The permits are held while waiting on a *different* smaller limit — connection pool or quota — so the semaphore is saturated without the service being busy (F6); compare permits held, pool checkout, downstream in-flight and fds, and whichever plateaus first is binding. (c) Queueing: at $\rho$ near 1 the mean wait explodes per Q-01, so check $\lambda S/c$ — 30 % CPU with a saturated semaphore is inconsistent with pure CPU shortage and points to (a) or (b).

**Reasoning trail.** "Semaphore full, CPU low" cannot be caused by CPU cost; it must be either non-CPU blocking on the loop, or waiting on a resource the CPU metrics do not show.

**Red flag.** "Increase the semaphore." That adds parked work against whichever resource is actually binding and increases the tail (F6, Section 7.2).

### Q8 — "When would you choose a process pool over `asyncio.to_thread`?"

**Checked answer.** When the work is pure-Python CPU and must use multiple cores (threads are GIL-serialised for bytecode), and when payload sizes are small enough that pickling plus IPC does not dominate. If the heavy work is inside GIL-releasing C, threads win on simplicity. If the work is long-running, retryable or needs independent scaling, an external worker service with a durable queue wins. Check the boundary cost: if (pickle + IPC + unpickle)/compute > 0.3, the process boundary is wrong (F7).

**Reasoning trail.** The choice is about *where the CPU time goes* and *what the serialization cost is*, not about which API looks modern.

**Red flag.** "Process pools are always faster for CPU work." They are faster only when the payload crossing the boundary is small relative to the compute.

### Q9 — "How do you make deploys not lose in-flight work?"

**Checked answer.** On SIGTERM: stop accepting (flip readiness), keep serving what is in flight, await it with a deadline that exceeds the in-flight p99, then cancel the remainder and log the count and names. Children must honour cancellation. Effects need idempotency keys so client retries do not duplicate. Verify with a deploy under load: zero client errors, zero duplicate effects, pending count zero at exit (F8).

**Reasoning trail.** The failure is a lifecycle problem, not a concurrency-primitive problem: the process leaves before its work does.

**Red flag.** "The orchestrator handles it" or "we call `loop.close()`". Closing a loop with pending tasks is not a drain.

### Q10 — "Explain backpressure and where you would put it in a request pipeline."

**Checked answer.** Backpressure is a signal that a producer must slow because a consumer is at capacity. It belongs at the earliest point where per-unit state is created — before `create_task`, before enqueuing, before reading the next batch — and it must reflect the *binding* resource, not just the local semaphore. Concretely: bounded queue with `maxsize`, semaphore sized to the effective $c$, and a rejection or blocking policy chosen explicitly. Without it, memory grows with input (T-10, F3).

**Reasoning trail.** Any unbounded buffer relocates the failure from "slow" to "out of memory". The bound's *value* must come from the quota/pool/memory analysis, not from a round number.

**Red flag.** "We use an unbounded queue so producers never block." That is a memory leak with a queue API.

### Q11 — "Walk me through the loop-thread CPU budget for a service at 1 000 req/s."

**Checked answer.** $C_{\text{loop}}=C_{\text{app}}+mC_{\text{sched}}+C_{\text{fixed}}$. With $C_{\text{app}}=0.8$ ms, $m=6$, $C_{\text{sched}}=4\,\mu$s and $C_{\text{fixed}}=0.1$ ms, $C_{\text{loop}}=0.924$ ms, so at 1 000 req/s the loop is at $\rho_{\text{loop}}=0.924$ — over the 0.70 headroom and close to the ceiling of 757.6 req/s at that target. Fix by reducing $C_{\text{app}}$ (move the transform off-loop), reducing $m$ (fewer round trips, batch the awaits), or running more processes each with its own loop.

**Reasoning trail.** The loop is a shared serial resource; its utilisation is a first-class capacity number even when request-level concurrency looks healthy.

**Red flag.** "The loop is asynchronous so it can handle any rate." Every task's bookkeeping and every timer is charged to that thread.

### Q12 — "Given a 2 s caller budget, three stages and 50 ms overhead, what are the timeouts?"

**Checked answer.** $t_1+t_2+t_3+0.05\le2.000$, so the stage timeouts must sum to at most 1.950 s. An example allocation is 0.400/0.900/0.650, summing to exactly 2.000 with overhead. Each downstream must receive the *remaining* budget, not a fresh per-stage timeout, otherwise three independent 2 s timeouts compose to 6.05 s (T-05). Verification: assert the sum in a test and assert that a stage receiving less than its own minimum refuses fast rather than truncating work silently.

**Reasoning trail.** Timeouts compose additively in the worst case; a deadline is a budget that must be apportioned and propagated.

**Red flag.** "Each layer sets 2 s so every layer has a chance." That is a 3× budget violation and a wasted-work generator.

**Takeaway:** interviewers are not checking whether you can name `asyncio.TaskGroup`; they are checking whether you can turn a workload into numbers and then state which number would change your mind.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Choosing among sequential, `asyncio`, threads, processes and external workers; sizing a bounded concurrency limit; owning effects across timeout, retry, cancellation and shutdown. |
| Core formula | $c=\lceil\lambda S/\rho_{\text{target}}\rceil$ (C-01), with stability requiring $\lambda<c/S$; $L=\lambda W$ (L-01) is an *occupancy* identity only and must never be used alone as a pool size. |
| Key assumption | Every call on the awaited path actually suspends and never blocks the loop; $S$ was measured under the same downstream regime as the capacity you are claiming. Break either and every number here is fiction. |
| Complexity | Task create/await/wake $\mathcal{O}(1)$; timer insert/cancel $\mathcal{O}(\log n)$; fan-out setup $\mathcal{O}(N)$; bounded queue $\mathcal{O}(1)$ per item; memory $\mathcal{O}(c\cdot\text{footprint})$, not $\mathcal{O}(\text{input})$. |
| Use when | The dominant cost is waiting on external I/O (network, storage, another service) and a maintained non-blocking client exists for it. |
| Avoid when | The work is pure-Python CPU (use processes or a service), a synchronous hot SDK cannot be replaced (bounded thread offload only), or no defensible bound $c$ and reject/queue policy can be stated. |
| Go deeper | [REST APIs and WebSockets](02_REST_APIs_and_WebSockets.md) for protocol and idempotency semantics; [Docker and containers](04_Docker_and_Containers.md) for runtime isolation, fd and memory limits, and graceful-stop configuration. |
| Diagnostic | When p99 climbs with the semaphore fully acquired and CPU low: compare permits held, connection-pool checkout, downstream in-flight and fd count — the first to plateau is the binding limit (F6); if none plateaus, the loop is blocked (F1). |
| Guardrail | Retry budget as a fraction of successful traffic + full jitter + idempotency keys + one retrying layer + circuit breaker; grace period above in-flight p99 with a SIGTERM drain that logs pending tasks at exit. |
| Evidence limit | All workload numbers in this sheet are ILLUSTRATIVE teaching inputs with checked arithmetic; queue-model values are arithmetic of M/M/c under its five assumptions. `C_sched` and free-threaded behaviour remain OPEN for any specific interpreter and platform until measured there. |

