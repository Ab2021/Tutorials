# Reliability Patterns for AI Systems

**TL;DR** — LLM APIs are measurably less reliable than traditional APIs, and they have a failure mode classical reliability engineering has no name for: **a fluent, confident, well-formed wrong answer.** Everything else — timeouts, rate limits, provider outages — is ordinary distributed-systems work. This playbook covers the six patterns that make an AI system survive its dependencies (retry with backoff and jitter, circuit breaker, bulkhead, layered and adaptive timeouts, graceful degradation, multi-provider failover and hedging), plus the two failure classes that are *specific to AI*: **semantic failure** (the output parses and is wrong) and **silent quality degradation** (every infrastructure metric is green and the answers got worse). The organising insight: **retry handles transient failure, the circuit breaker handles systemic failure, the bulkhead stops the blast radius, and degradation decides what you serve when nothing works — and none of the four addresses the failure that matters most, which is that the answer is wrong.**

> **60-second interview answer:** "The patterns split into two groups, and I'd lead with the split because it's the thing people miss. The first group is ordinary distributed-systems reliability: retry with exponential backoff and jitter, a circuit breaker with a failure threshold and a half-open probe, bulkheads so a batch job can't starve real-time traffic, layered timeouts with a p99-derived adaptive value, multi-provider failover, and a defined degradation ladder. Those protect me from the provider failing. The second group is AI-specific and it's the harder one: the output can be perfectly well-formed and confidently wrong. So on top of the infra patterns I need a semantic validation layer — grounding checks, citation validity, schema constraints, and a sampled judge — plus quality SLOs that alarm on their own, because the failure mode is that every infra metric stays green while the answers get worse. If I had to pick one number, it's the sampled quality score: nothing else catches that."

---

## 1. Purpose

This playbook defines how an AI system behaves when its dependencies fail — and when its own outputs fail without anything technically breaking.

**The two failure classes:**

| Class | Example | Detected by | Classical patterns apply? |
|---|---|---|---|
| **Infrastructure failure** | Provider 503, timeout, rate limit | Error codes, latency | **Yes** — retry, breaker, bulkhead |
| **Semantic failure** | Fluent, confident, wrong answer | Only eval or a judge | **No** — needs its own layer |

**The reason this playbook exists as a separate document:** most AI reliability discussions stop at group one. Group two is where the user-visible damage actually happens, because a 503 is a retry and a wrong answer is a decision.

**In scope:** retry, circuit breaking, bulkheading, timeouts, failover, hedging, degradation, semantic validation, and quality-degradation detection.

**Out of scope:** release-time quality gates (see [deployment and CI/CD](04-deployment-release-and-cicd.md)), capacity saturation (see [capacity planning](05-capacity-planning-and-autoscaling.md)), and disaster recovery / region loss (see [DR and failover](09-dr-failover-and-backup.md)).

---

## 2. Scope — the LLM-specific failure modes

| Failure mode | Cause | User impact | Pattern |
|---|---|---|---|
| **Rate limiting** | Quota exceeded, bursty traffic | Rejection / delay | Retry with `Retry-After`, bulkhead, quota planning |
| **Timeout** | Long generation, network | Slow or failed response | Layered + adaptive timeouts, hedging |
| **Provider outage** | Their infrastructure | Total failure on that provider | Circuit breaker, multi-provider failover |
| **Malformed output** | Generation error, truncation | Parse failure → 500 | Schema validation + bounded repair retry |
| **Context overflow** | Input exceeds the window | Hard failure | Pre-flight token count, truncation policy |
| **Semantic failure** | Model is confidently wrong | **Wrong decision** | Grounding checks, judge, abstention |
| **Quality degradation** | Silent model change, load, prompt drift | Slowly worse answers | Quality SLO + sampling + drift detection |
| **Partial agent failure** | Step N of 10 fails | Half-completed task | Durable checkpointing, idempotent steps |
| **Tool-call loop** | Retryable error + no ceiling | Cost and latency explosion | Step/cost ceilings, no-progress detector |
| **Model regression** | Provider updates under an alias | Quality drop with no deploy | Pinned snapshots, release gate |

**The three that have no classical analogue** are semantic failure, quality degradation, and partial agent failure. Everything else maps onto a known pattern.

### 2.1 Reliability tiers

| Tier | Availability | p99 latency | Examples | Patterns required |
|---|---|---|---|---|
| **Critical** | 99.99% | < 3 s | Payments, auth, medical | All patterns + active-active |
| **Standard** | 99.9% | < 10 s | Customer support, RAG | All patterns + multi-provider |
| **Best effort** | 99% | < 30 s | Background, batch | Retry + breaker + queue |

**Choose the tier explicitly.** *"99.99% on an LLM system means active-active across providers with different tokenisers and tool-call formats, which is a large engineering commitment and it changes answer formatting subtly. Most systems should be 99.9% and accept a degradation path rather than pay for four nines of a dependency you don't control. I'd say that out loud rather than let someone assume four nines is achievable by trying harder."*

---

## 3. SLIs and SLOs

### 3.1 Infrastructure SLIs

| SLI | Definition | Target | Alert |
|---|---|---|---|
| Availability | Successful / total requests | 99.9% | < 99.9% for 5 m |
| Error rate | 5xx + unhandled / total | ≤ 0.5% | > 5% for 5 m |
| **Retry rate** | Retried requests / total | ≤ 8% | > 15% for 10 m |
| **Retry amplification** | Total attempts / successful requests | ≤ 1.15 | > 1.30 |
| **Circuit-open rate** | Time with any breaker open | ≤ 0.1% | Any open > 2 m |
| Provider failover rate | Requests served by a non-primary | ≤ 15% steady | > 30% for 5 m |
| Timeout rate | Requests hitting the total timeout | ≤ 0.3% | > 2% for 5 m |
| Malformed-output rate | Parse/validation failures | ≤ 0.2% | > 1% for 5 m |
| Context-overflow rate | Rejected for input size | ≤ 0.1% | > 1% for 15 m |

### 3.2 Semantic SLIs — the ones that matter

| SLI | Definition | Target | Alert |
|---|---|---|---|
| **Sampled quality score** | LLM-judge or rubric on a 1% sample | ≥ 4.2/5.0 | < 3.8 for 30 m |
| **Faithfulness / groundedness** | Claims supported by retrieved evidence | ≥ 0.85 | < 0.80 for 1 h |
| **Abstention correctness** | Correct "I don't know" when evidence is absent | ≥ 0.95 | < 0.90 for 1 h |
| **Hallucination rate** | Unsupported claims / sampled answers | ≤ 0.03 | > 0.06 for 1 h |
| **Schema-valid output rate** | Outputs passing the contract | ≥ 0.995 | < 0.98 for 15 m |
| **Semantic-cache false-hit rate** | Wrong answer served from cache | ≤ 0.005 | > 0.01 for 1 h |
| **Trajectory success rate** (agents) | Tasks completing correctly | ≥ 0.85 | < 0.75 for 1 h |
| **Partial-completion rate** (agents) | Tasks ending in a partial state | ≤ 0.02 | > 0.05 for 1 h |

**The semantic SLO is the one to defend in an interview:** *"Infrastructure SLIs are necessary and insufficient. A system can be at 99.99% availability while producing answers that are 8% less faithful, and nothing in a standard monitoring stack will tell you. So the sampled quality score is a first-class SLI with its own alert, not a dashboard someone checks on Fridays."*

---

## 4. Architecture

```
══════════════ REQUEST PATH WITH ALL PATTERNS ══════════════

  request
     │
     ▼
┌─────────────────────────────────────────────┐
│  BULKHEAD (per workload class)              │
│  realtime 50 · critical 10 · batch 200      │
│  → queue full = reject fast, don't cascade  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  PRE-FLIGHT VALIDATION                      │
│  token count vs window · schema · PII scan  │
│  → fail fast BEFORE spending a call         │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  PROVIDER SELECTOR                          │
│  health · circuit state · latency · cost    │
└──────────────────┬──────────────────────────┘
                   ▼
      ┌────────────┴────────────┐
      ▼                         ▼
┌───────────────┐        ┌───────────────┐
│ CIRCUIT       │        │ CIRCUIT       │
│ BREAKER A     │        │ BREAKER B     │
│ 5 fail/30s    │        │ (per provider)│
└───────┬───────┘        └───────┬───────┘
        ▼                        ▼
┌───────────────┐        ┌───────────────┐
│ LAYERED       │        │ LAYERED       │
│ TIMEOUTS      │        │ TIMEOUTS      │
│ conn 5s       │        │ (adaptive p99  │
│ read 30s      │        │  × 1.2)       │
│ total 60s     │        │               │
└───────┬───────┘        └───────┬───────┘
        ▼                        ▼
┌───────────────┐        ┌───────────────┐
│ RETRY         │        │ HEDGE         │
│ backoff+jitter│        │ after 2.0s    │
│ max 3         │        │ → first wins  │
└───────┬───────┘        └───────┬───────┘
        └────────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  SEMANTIC VALIDATION  ← the AI-specific layer│
│  1. schema / parse                          │
│  2. citation validity                       │
│  3. grounding vs retrieved evidence         │
│  4. refusal correctness                     │
│  5. sampled judge (1%)                      │
└──────────────────┬──────────────────────────┘
                   ▼
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ RESPOND      │        │ DEGRADE /    │
│              │        │ ABSTAIN      │
└──────────────┘        └──────────────┘
```

---

## 5. Standard operating procedures

### SOP-1 — Retry with exponential backoff and jitter

```python
class RetryConfig:
    max_retries      = 3          # 4 total attempts
    base_delay       = 1.0        # seconds
    max_delay        = 60.0       # ceiling
    exponential_base = 2.0        # 1s, 2s, 4s, 8s...
    jitter           = 0.5        # ±50% randomisation

    def get_delay(self, attempt):
        delay = min(self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)
```

**The retryable / non-retryable split — get this right or you will make outages worse:**

| **Retryable** | Why |
|---|---|
| `RateLimitError` (429) | Respect `Retry-After`; do not retry before it |
| `TimeoutError` | Transient; but **only if the request is idempotent** |
| `ServiceUnavailableError` (503) | Provider-side transient |
| `ConnectionError` | Network blip |

| **NOT retryable** | Why |
|---|---|
| `AuthenticationError` (401/403) | Retrying makes it worse and can lock an account |
| `InvalidRequestError` (400) | Deterministic; the same input fails identically |
| `ContentPolicyViolation` | Deterministic refusal |
| `ContextLengthExceeded` | Deterministic; truncate instead |
| `QuotaExhausted` (hard cap) | Retrying burns the remaining quota faster |
| **Malformed JSON output** | Retry **once** with a repair prompt, then fail — not the standard retry path |
| **Semantic failure** | **Never retried** — see the box below |

**The most important rule in this section:**

```
   ⚠  NEVER BLINDLY RETRY A NON-IDEMPOTENT SIDE EFFECT.

   If the request issues a refund, sends a message, creates a
   ticket, or writes a record, a timeout does NOT mean the
   operation failed. It means you do not know.

   → Every side-effecting retry needs a deterministic
     idempotency key so the downstream service can deduplicate.
   → Without one, a retry after a timeout produces a duplicate.
   → Duplicates of a read are free. Duplicates of a refund are not.

   ⚠  NEVER RETRY A BAD ANSWER.

   A semantically wrong answer will be wrong again with the same
   prompt at temperature 0. Retrying burns cost and latency and
   returns the same thing. Semantic failures take ONE of four
   paths: repair (with a changed prompt), escalate to a stronger
   model, abstain, or fail. Never the retry loop.
```

**Jitter sizing:** *"Full jitter — a uniform random in `[0, delay]` — is the strongest de-correlator but it makes the tail longer. I use ±50% around the computed delay, which de-correlates within a herd while keeping a predictable upper bound. What I never do is retry without jitter, because N clients failing simultaneously will retry simultaneously and keep the dependency down."*

**Retry budget — the global ceiling:** *"Per-request retries can still produce a retry storm if every request retries. So I also cap the fleet-wide retry rate at 10% of traffic. Above that, the client stops retrying regardless of the per-request policy, because the system is telling you the dependency is down and retries are actively harmful."*

### SOP-2 — Circuit breaker

```python
class CircuitBreakerConfig:
    failure_threshold   = 5                 # consecutive failures
    recovery_timeout    = timedelta(seconds=30)
    half_open_max_calls = 3                 # probes allowed
    success_threshold   = 2                 # probes needed to close
```

```
                 failures >= 5
      ┌────────────────────────────────┐
      │                                ▼
 ┌─────────┐                     ┌──────────┐
 │ CLOSED  │                     │   OPEN   │
 │ normal  │                     │ reject   │
 │  ops    │                     │ instantly│
 └─────────┘                     └────┬─────┘
      ▲                               │ 30 s elapsed
      │                               ▼
      │                        ┌────────────┐
      │  2 successes           │ HALF_OPEN  │
      └────────────────────────│ max 3 probes│
                               └─────┬──────┘
                                     │ any failure
                                     └──────► OPEN
```

**One breaker per (provider, model, operation) — not one per provider.** *"A provider can be healthy for chat completions and failing for embeddings, or healthy on the small model and rate-limited on the frontier one. A single breaker per provider will either trip too late or trip everything when only one route is broken. So the breaker key is `provider:model:operation`, and that's why the failover decision is per-route."*

**The distinction to state in an interview:** *"Retry handles transient failure — it assumes the next attempt may succeed. The circuit breaker handles systemic failure — it assumes the dependency is unhealthy and that repeated attempts waste resources and slow recovery. They work together: a request fails, retries with backoff, and if the retries exhaust, the breaker records a failure. After five consecutive failures the breaker opens and requests fail fast instead of each one waiting through the full retry ladder. Without the breaker, during an outage every request waits 7 seconds of backoff before failing — so your latency p99 becomes your retry ladder length and your threads are all occupied waiting."*

**Threshold tuning by call volume:**

| Traffic | failure_threshold | recovery_timeout | Rationale |
|---|---|---|---|
| Low (< 1/s) | 3 | 60 s | Few samples; trip fast |
| Medium (1–50/s) | 5 | 30 s | The default |
| High (> 50/s) | 20 + 50% error rate | 15 s | Avoid tripping on a brief blip |
| Critical path | 3 | 90 s | Prefer failing fast over flapping |

### SOP-3 — Bulkhead

```python
bulkheads = {
    "critical": Bulkhead("critical", max_concurrent=10,  max_queued=20),
    "realtime": Bulkhead("realtime", max_concurrent=50,  max_queued=100),
    "batch":    Bulkhead("batch",    max_concurrent=200, max_queued=1000),
    "agents":   Bulkhead("agents",   max_concurrent=30,  max_queued=50),
}
```

**What bulkheading actually buys you:** a batch re-embedding job cannot consume the concurrency that interactive queries need. This is the single most valuable pattern for AI systems, because the workloads are wildly heterogeneous — a batch job and a user query are the same API call and completely different resource profiles.

**The bulkhead layout I would defend:**

| Bulkhead | Concurrency | Workload | Shed order |
|---|---|---|---|
| `critical` | 10 | Payments, auth-adjacent generation | Never shed |
| `realtime` | 50 | Interactive RAG, chat | Shed last |
| `agents` | 30 | Multi-step agent runs | Shed early |
| `batch` | 200 | Eval, backfill, re-embed | Shed first |

**Why agents get their own bulkhead:** *"An agent run holds 5 to 40 concurrency slots over 30 to 90 seconds. If agents share the realtime pool, one agent surge consumes the entire pool and takes interactive RAG down. Separate bulkhead, separate pool, separate capacity plan."*

**Queue-full behaviour must be explicit:** reject fast with a 429 and a `Retry-After`, do not queue unboundedly. *"An unbounded queue converts a load problem into a latency problem and then into a memory problem. The queue depth is a capacity signal, not a buffer."*

### SOP-4 — Layered and adaptive timeouts

```python
class TimeoutConfig:
    connection_timeout = 5.0     # TCP + TLS + first byte
    read_timeout       = 30.0    # between tokens
    total_timeout      = 60.0    # the whole operation

# Adaptive: derives the total from observed latency
class AdaptiveTimeout:
    initial  = 30.0
    min      = 10.0
    max      = 120.0
    percentile = 0.99
    window   = 1000              # observations

    # on each observation:
    #   keep the last 1000 latencies
    #   new_timeout = p99(latencies) × 1.2
    #   clamp to [min, max]
```

**The timeout budget must be nested correctly:**

```
   Total SLO: 3 s p95 end-to-end
   ┌──────────────────────────────────────────────────┐
   │ gateway 50 ms                                     │
   │ ├─ guardrail (input) 120 ms                       │
   │ ├─ retrieval 250 ms   [timeout 800 ms]            │
   │ ├─ rerank 300 ms      [timeout 1,500 ms]          │
   │ ├─ generation 2,000ms [timeout 12,000 ms]  ← p95  │
   │ │   └─ per-request total from adaptive p99 × 1.2  │
   │ └─ guardrail (output) 150 ms                      │
   │ post-checks 100 ms                                │
   └──────────────────────────────────────────────────┘

   THE RULE: inner timeouts MUST be shorter than outer ones,
   and the sum of the critical path plus slack must fit the
   outer budget. A 12 s generation timeout inside a 3 s SLO
   means the SLO is 3 s only because p95 happens to be under it —
   the timeout is not enforcing anything.
```

**The three timeout mistakes:**

| Mistake | Consequence | Fix |
|---|---|---|
| No timeout at all | A hung request holds a concurrency slot forever | Always set `total_timeout` |
| Inner timeout > outer | The outer fires first and you lose the error detail | Nest correctly |
| Static timeout on a variable workload | Too tight kills legitimate long answers; too loose pins capacity | **Adaptive from p99 × 1.2** |

**Adaptive timeout is the AI-specific contribution here**, because LLM latency varies enormously with output length. *"A fixed 30-second timeout is wrong for both a 20-token classification and a 4,000-token synthesis. So I derive it from the observed p99 with a 20% buffer, clamped between 10 and 120 seconds, recomputed from a rolling 1,000-observation window. That way the timeout tracks the workload instead of being a guess that's wrong half the time."*

**Per-class timeouts must be separate.** A classification call and a long-form synthesis call should not share a timeout. Classify the request first, then apply the class's timeout.

### SOP-5 — Multi-provider failover and hedging

```python
class ProviderManager:
    providers     = {"primary": ..., "secondary": ..., "tertiary": ...}
    priority      = ["primary", "secondary", "tertiary"]
    health        = {name: True}
    recheck_delay = 30   # seconds after marking unhealthy

    async def generate(request):
        for name in priority:
            if not health[name] or not breaker[name].can_execute():
                continue
            try:
                result = await providers[name].generate(request)
                breaker[name].record_success()
                return result
            except RetryableError:
                health[name] = False
                breaker[name].record_failure()
                schedule_recheck(name)     # after 30 s
                continue
        raise AllProvidersUnavailableError()
```

**Hedged requests — for latency, not for failure:**

```python
class HedgedRequest:
    hedge_delay = 2.0    # seconds

    async def generate(request):
        primary = start(providers[0])
        try:
            return await wait_for(primary, timeout=2.0)
        except TimeoutError:
            # Primary is SLOW, not failed — hedge
            hedges = [start(p) for p in providers[1:]]
            done, pending = await wait([primary, *hedges],
                                       return_when=FIRST_COMPLETED)
            for t in pending: t.cancel()
            for t in done:
                if t.exception() is None: return t.result()
            raise AllProvidersFailedError()
```

**Hedging arithmetic — why 2.0 s:**

```
   Hedging costs you a duplicate call whenever the primary is slow.
   It is worth it when:
     hedge_cost_per_request  <  p99_latency_saved × value_of_latency

   At a 2.0 s hedge delay on a 2.7 s p95:
     ~12% of requests get hedged
     cost = 0.12 × $0.012 = $0.0014 per request
     benefit = p99 drops from ~6.8 s to ~3.4 s

   At a 1.0 s delay: 30% hedged, cost triples, marginal latency gain.
   At a 4.0 s delay: 3% hedged, almost no benefit.

   → 2.0 s ≈ p95 of the primary, which hedges only the tail.
```

**The three-provider rule and why the secondary always takes traffic:**

| Provider | Steady share | Role |
|---|---|---|
| Primary | 70% | Provisioned/optimised path |
| Secondary | **25% — always, not 0%** | Keeps the failover path exercised |
| Tertiary | 5% + emergency burst | Last resort |

*"A failover provider you have never sent traffic to is one you have never tested. You will discover the tokeniser difference, the tool-call format difference, and their rate limit at 3 a.m. during an incident. Running 25% steady keeps the path warm and the differences known. The redundancy is real money and it is the cheapest reliability you will ever buy."*

**Cross-provider differences that break naive failover:**

| Difference | Failure it causes |
|---|---|
| Tokeniser | Truncation boundaries and cost differ |
| Tool-call format | Parse failure on the fallback |
| System-prompt sensitivity | Answer style changes visibly |
| `max_tokens` semantics | Different truncation behaviour |
| Rate-limit shape | RPM vs TPM — one may 429 where the other didn't |
| Refusal behaviour | Different content policy boundaries |

**Mitigation:** a normalisation layer between the provider and the application, so the application sees one shape. And a **cross-provider eval run in CI**, so a prompt that only works on the primary is caught before the incident, not during it.

### SOP-6 — Graceful degradation

```python
class DegradationLevel(Enum):
    FULL    = "full"      # RAG + frontier + verification
    REDUCED = "reduced"   # RAG + smaller model, no verification
    MINIMAL = "minimal"   # no retrieval, smallest model
    CACHED  = "cached"    # cache-only; abstain on a miss
    OFFLINE = "offline"   # honest error
```

| Level | What runs | Trigger | User impact |
|---|---|---|---|
| **FULL** | Everything | Normal | None |
| **REDUCED** | Retrieval + small model, no rerank, no verification | Secondary provider only | Slightly worse answers |
| **MINIMAL** | Small model, no retrieval, no tools | All providers degraded | Clearly worse; may hallucinate |
| **CACHED** | Semantic cache only; abstain on miss | Total provider outage | Only previously-seen questions |
| **OFFLINE** | Honest error with an ETA | Nothing available | Explicit failure |

**The two rules that make degradation safe:**

```
   RULE 1 — SAFETY IS NOT A DEGRADATION LEVEL.
     The ladder may remove: retrieval, reranking, verification,
     the frontier model, tools, memory, caching.
     The ladder may NOT remove: input guardrails, output guardrails,
     PII filtering, the action gate, permission checks.
     → Capability degrades. Safety does not.

   RULE 2 — NEVER SILENTLY SERVE A WORSE ANSWER.
     Every degraded response carries a marker:
       { "degraded": true, "level": "reduced", "reason": "primary_unavailable" }
     and the user-facing surface says so where it matters.
     → At MINIMAL, retrieval is off, so the model may hallucinate.
       A user who doesn't know that will trust the answer.
```

**Recovery is a ladder too, and it must be slow:**

```
   Recovering from CACHED → FULL in one step re-creates the surge.
   Step one level every 5 minutes, watching error rate and latency.
   If a step regresses, go back down and wait 10 minutes.
```

### SOP-7 — Semantic validation (the AI-specific layer)

This is the layer with no classical analogue, and it is where the real damage is prevented.

```
   VALIDATION STAGES, cheapest first

   1. SCHEMA / PARSE                          ~0 ms, free
      □ Valid JSON? Required fields present? Types correct?
      □ Enum values in range? Length under the cap?
      → fail: repair retry ONCE, then fail the request

   2. CONSTRAINTS                             ~1 ms, free
      □ No PII in the output (regex + classifier)
      □ No forbidden strings (competitor names, promises)
      □ Numeric values within plausible ranges
      → fail: block the response, log, alert

   3. CITATION VALIDITY                       ~5 ms, free
      □ Every citation ID resolves to a retrieved chunk
      □ Every quoted span appears in the cited chunk
      → fail: strip the citation, mark ungrounded, alert

   4. GROUNDING / FAITHFULNESS                ~200 ms, cheap model
      □ Are the claims entailed by the retrieved evidence?
      □ Is the answer length proportionate to the evidence?
      → fail: abstain or regenerate once

   5. REFUSAL CORRECTNESS                     ~200 ms
      □ Retriever found nothing AND the model answered → over-confident
      □ Retriever found strong evidence AND the model refused → over-cautious
      → fail: adjust the confidence signal

   6. SAMPLED JUDGE (1%)                      ~1 s, expensive
      □ Relevance, accuracy, helpfulness on a sample
      → feeds the quality SLI, not the request path
```

**The cost of the validation layer, and why it is worth it:**

| Stage | Latency added | Cost added | Runs on |
|---|---|---|---|
| Schema + constraints + citations | ~6 ms | ~0 | 100% of requests |
| Grounding check | ~200 ms | $0.0004 | 100% of requests |
| Refusal check | ~200 ms | $0.0003 | 100% of requests |
| Sampled judge | — (async) | $0.0009 | 1% of requests |
| **Total in-path** | **~410 ms** | **$0.0007** | |

*"410 milliseconds and seven hundredths of a cent per request. Against that, the alternative is shipping confident wrong answers. If your p95 budget is 3 seconds, 410 milliseconds is 13% of it and it is the best-spent 13% in the system. The mitigation when latency is tight is to run stages 1 to 3 synchronously and stages 4 to 5 asynchronously with a hold-and-release pattern for high-stakes requests only."*

### SOP-8 — Reliability incident runbook

```
SYMPTOM: availability or quality SLO breach

T+0   CLASSIFY (this determines everything that follows)
      □ Infrastructure or semantic?
        → error rate up = infrastructure
        → error rate flat, quality down = SEMANTIC. Different path.
      □ One provider, one model, or global?
      □ Did a release happen in the last 60 min? → roll back first,
        it's a 90-second fix and it eliminates the biggest hypothesis
      □ Is a breaker already open? → check the breaker dashboard

T+2m  INFRASTRUCTURE PATH
      □ 1. Confirm the breaker opened (it should have, automatically)
      □ 2. Confirm traffic failed over to the secondary
      □ 3. If failover did not happen → force it manually
      □ 4. If the secondary is also failing → degradation ladder
      □ 5. Reduce the retry budget to 0 (stop amplifying the outage)
      □ 6. Check the provider status page; do not debug their system

T+2m  SEMANTIC PATH
      □ 1. Freeze the release pipeline (prevent a second variable)
      □ 2. Check the sampled quality score by slice, not aggregate
      □ 3. Check faithfulness and citation-validity rates
      □ 4. Check whether the retrieval layer changed (recall drop
           looks exactly like a model quality drop)
      □ 5. Check whether the model snapshot actually served matches
           the manifest — the classic "provider changed it" case
      □ 6. If a slice is broken, consider an abstention policy for
           that slice while you investigate
      □ 7. DO NOT raise temperature or "try again" — a semantic
           failure is deterministic at temperature 0

T+15m ESCALATE
      □ Page the platform lead
      □ Open a status page if user-visible
      □ If semantic and unresolved in 30 min → consider rolling back
        to the last known-good release even without a root cause

T+RESOLVED
      □ Recover the ladder one level per 5 minutes
      □ Restore the retry budget gradually
      □ Post-incident: which SLO caught it? If it was the sampled
        judge, the judge earned its cost. If it was a user complaint,
        the SLO has a gap — add the signal that would have caught it.
```

---

## 6. Failure modes and mitigations

| # | Failure | Symptom | Root cause | Mitigation |
|---|---|---|---|---|
| 1 | **Retry storm** | Dependency stays down; load rises | No jitter, no global budget | Jitter + 10% fleet retry cap |
| 2 | **Retry of a bad answer** | Cost and latency up, quality flat | Semantic failure retried | Never retry semantic failures |
| 3 | **Duplicate side effect** | Two refunds | Timeout + retry, no idempotency key | Deterministic idempotency keys |
| 4 | **Breaker never trips** | Every request waits the full ladder | Threshold too high or key too coarse | Per-route breaker, tuned threshold |
| 5 | **Breaker flaps** | Repeated open/close cycles | Recovery timeout too short | 30 s min, 90 s on critical paths |
| 6 | **Failover provider untested** | Failover produces parse errors | Tokeniser/format difference | 25% steady traffic; cross-provider CI eval |
| 7 | **Hedging cost explosion** | Cost up 30%, latency barely moved | Hedge delay too short | Hedge at ~p95 (2.0 s) |
| 8 | **Batch starves realtime** | Interactive latency collapse | Shared concurrency pool | Bulkhead per workload class |
| 9 | **Unbounded queue** | Memory growth, then OOM | Queue used as a buffer | Bounded queue; reject at the limit |
| 10 | **Timeout not enforcing** | p99 exceeds the SLO silently | Inner timeout > outer budget | Nest timeouts; verify the sum fits |
| 11 | **Static timeout wrong for the workload** | Long answers killed; short ones pin capacity | One timeout for all classes | Adaptive p99 × 1.2, per class |
| 12 | **Degradation removes safety** | Guardrails off under load | Ladder included safety steps | Safety never in the ladder |
| 13 | **Silent degraded answers** | User trusts a retrieval-less answer | No `degraded` marker | Mark every degraded response |
| 14 | **Semantic failure invisible** | Users complain; dashboards green | No quality SLI | Sampled judge as a first-class SLI |
| 15 | **Hallucination during outage** | MINIMAL level invents facts | Retrieval off, no marker | Abstain at MINIMAL for factual queries |
| 16 | **Partial agent completion** | Half the actions applied | No durable checkpointing | Checkpoint per step; idempotent resume |
| 17 | **Tool-call loop** | 40 steps, $14 task | Retryable error with no ceiling | Step/cost ceilings + no-progress detector |
| 18 | **Model regression under an alias** | Quality drop, no deploy | Provider mutated a floating alias | Pin snapshots; alert on served-vs-manifest mismatch |
| 19 | **Recovery surge** | Second incident on recovery | Full restoration in one step | Ladder recovery, one level per 5 min |
| 20 | **Failover cascades to all providers** | All three circuits open | Synchronised hedging | Stagger hedges; per-provider budgets |

### 6.1 Why retrying a semantic failure is worse than useless

```
   Infrastructure failure:
     request → 503 → retry → 200 → correct answer    ✓ retry works

   Semantic failure:
     request → 200 → wrong answer
             → retry → 200 → SAME wrong answer      ✗ at temperature 0
             → retry → 200 → DIFFERENT wrong answer ✗ at temperature > 0

   The second case is worse than the first: you paid twice and now you
   have two plausible wrong answers and no basis to choose.

   The four legitimate responses to a semantic failure:
     1. REPAIR      change the prompt (add the retrieved evidence,
                    restate the constraint) and try once
     2. ESCALATE    send it to a stronger model or a reasoning model
     3. ABSTAIN     return "I don't have enough information"
     4. FAIL        return an honest error to the caller

   All four cost one call. The retry loop costs N calls and converges
   on nothing.
```

### 6.2 Detecting the model-regression failure

```
   THE SCENARIO
     A provider updates the model behind a floating alias, or
     deprecates a snapshot and silently routes to a newer one.
     Answer quality changes. Nothing in your stack changed.
     Error rate is flat. Latency is flat. Cost may even improve.

   THE DEFENCES, in layers
     1. PIN the snapshot in the release manifest — never an alias
     2. RECORD the served snapshot per request and compare against
        the manifest; alert on any mismatch (critical)
     3. RUN a frozen golden set daily against production, not just
        at release time — this catches drift between releases
     4. ALERT on the distribution of output tokens and refusal rate,
        because both shift before quality scores do
     5. KEEP a small held-out eval you never tune against, so
        drift in your own eval doesn't hide drift in the model

   The daily production golden set is the one that matters. Release
   gates only catch changes you made; a provider change happens on
   their schedule, not yours.
```

### 6.3 Partial failure in agent flows

An agent task that fails at step 7 of 10 has already taken 6 side effects. This is the reliability problem with no classical analogue.

| Property | Requirement |
|---|---|
| **Durable checkpointing** | State persisted after every step, so a crash resumes rather than restarts |
| **Idempotent steps** | Every side-effecting step carries a deterministic key |
| **Partial-completion reporting** | The result says exactly what succeeded and what did not |
| **Compensating actions** | Where possible, a rollback or reversal for completed steps |
| **No silent restart** | A resumed task must not redo completed steps |
| **Explicit final state** | `completed` / `partial` / `failed` — never an ambiguous "done" |

**The reporting contract that matters:** *"A partial result must enumerate what was done and what was not. The worst outcome is not a failure — it is a task that reports success having completed six of ten steps. So the terminal state is explicit, the action log is in the payload, and the caller can see exactly where it stopped."*

---

## 7. Capacity and cost of reliability

### 7.1 What each pattern costs

| Pattern | Cost | Latency | Worth it when |
|---|---|---|---|
| Retry (3 attempts) | Up to 4x on failures | +7 s worst case | Transient errors are > 0.1% |
| Circuit breaker | ~0 | −(retry ladder) on outage | Always |
| Bulkhead | Capacity fragmentation | ~0 | Heterogeneous workloads |
| Adaptive timeout | ~0 | Protects the tail | Always |
| Multi-provider | **25% redundancy spend** | +0 | Standard tier and above |
| Hedging (2.0 s) | ~12% duplicate calls | −50% p99 | Latency-sensitive |
| Semantic validation | $0.0007 + 410 ms | +410 ms p95 | Always |
| Sampled judge (1%) | $0.0009/request | 0 (async) | Always |
| Degradation ladder | ~0 | 0 | Always |
| Durable checkpointing | Write per step | ~5 ms | Agents |

**Total reliability overhead, typical:** ~4% of inference cost, ~450 ms of p95 latency, plus the 25% multi-provider redundancy.

**The justification sentence:** *"Four percent of inference cost and 450 milliseconds of p95 in exchange for surviving a provider outage and not shipping confident wrong answers. If someone asks me to cut the reliability budget, the semantic validation is the last thing I give up, not the first — because an outage is visible and a wrong answer is not."*

### 7.2 The availability arithmetic

```
   Single provider, no patterns:
      provider availability 99.9% → system 99.9%

   Single provider + retry + breaker, 2-minute MTTR on degradation:
      effective ≈ 99.95%

   Two providers, one hot standby, failover in 10 s:
      P(both down) = 0.001 × 0.001 = 1e-6 → 99.9999%
      BUT: correlated failures (shared dependency, shared region,
      shared network) dominate. Assume 50% correlation:
      effective ≈ 99.99%

   Two providers + degradation ladder (serves cached/minimal):
      perceived availability ≈ 99.99% even when both are down

   THE POINT: the degradation ladder buys more perceived availability
   than the second provider does, and it costs almost nothing.
```

**The correlated-failure caveat, worth stating:** *"Two providers looks like 99.9999% on paper and is really about 99.99%, because providers share underlying cloud regions, DNS, and network paths. If both your providers run in us-east-1, one region event takes both out. So provider diversity without region diversity is mostly cosmetic."*

---

## 8. Metrics and alerts

### 8.1 Alerts

```yaml
reliability_alerts:
  # ── Infrastructure ───────────────────────────────────────
  - name: error_rate_high
    condition: error_rate > 0.05
    window: 5m
    severity: critical
    action: page, check_breakers

  - name: retry_amplification
    condition: total_attempts / successful_requests > 1.30
    window: 10m
    severity: warning
    note: the dependency is degrading before it fails

  - name: retry_budget_exhausted
    condition: fleet_retry_rate > 0.10
    window: 5m
    severity: critical
    action: disable_client_retries
    note: stop amplifying the outage

  - name: circuit_open
    condition: any_breaker_state == OPEN
    duration: 2m
    severity: critical
    action: page, verify_failover

  - name: circuit_flapping
    condition: breaker_state_transitions > 6 in 30m
    severity: warning
    action: lengthen_recovery_timeout

  - name: failover_rate_high
    condition: non_primary_traffic_share > 0.30
    window: 5m
    severity: warning

  - name: all_providers_degraded
    condition: available_providers < 2
    severity: critical
    action: enable_degradation_ladder

  - name: timeout_rate
    condition: timeout_rate > 0.02
    window: 5m
    severity: warning
    note: check whether the adaptive timeout drifted up

  - name: bulkhead_queue_full
    condition: bulkhead_rejections > 0.01 of that class
    window: 5m
    severity: warning
    note: that class needs more capacity or a lower shed threshold

  - name: malformed_output_rate
    condition: parse_failures / total > 0.01
    window: 5m
    severity: warning
    note: usually a prompt change or a provider format change

  # ── Semantic (the AI-specific ones) ─────────────────────
  - name: quality_score_drop
    condition: sampled_quality_score < 3.8
    window: 30m
    severity: critical
    action: freeze_pipeline, investigate
    note: infra metrics will be GREEN — this is the only signal

  - name: faithfulness_drop
    condition: faithfulness < 0.80
    window: 1h
    severity: critical

  - name: hallucination_rate
    condition: unsupported_claim_rate > 0.06
    window: 1h
    severity: critical

  - name: citation_invalidity
    condition: invalid_citations / total_citations > 0.005
    severity: warning
    note: fabricated citations; strip and alert

  - name: refusal_rate_shift
    condition: refusal_rate deviates > 30% from 30d_median
    window: 1h
    severity: warning
    note: both directions are bad — over-refusing AND under-refusing

  - name: model_snapshot_mismatch
    condition: served_snapshot != manifest_snapshot
    severity: critical
    action: page_platform
    note: the provider changed the model under a floating alias

  - name: degraded_response_share
    condition: degraded_responses / total > 0.10
    window: 15m
    severity: warning

  # ── Agents ──────────────────────────────────────────────
  - name: partial_completion_spike
    condition: tasks_ending_partial / total > 0.05
    window: 1h
    severity: warning

  - name: tool_loop_detected
    condition: same_tool_same_args_repeated > 2 in one task
    severity: warning
    action: verify_ceiling_fired

  - name: agent_step_p99
    condition: p99_steps_per_task > 20
    severity: warning
    note: usually a tool failing or a retrieval recall drop
```

### 8.2 Dashboards

**Dashboard 1 — Reliability (on-call)**

| Panel | Content |
|---|---|
| Availability | Success rate with the SLO line, per provider |
| Breakers | State per `provider:model:operation`, colour-coded |
| Retry | Amplification ratio and fleet retry rate |
| Failover | Traffic share by provider |
| Timeouts | Rate and the current adaptive timeout value |
| Degradation | Current level, time in each level |
| Bulkheads | Occupancy per class |

**Dashboard 2 — Quality (the one nobody builds and everybody needs)**

| Panel | Content |
|---|---|
| Sampled quality | Score over time with release markers |
| Faithfulness | Trend with the 0.85 line |
| Hallucination rate | Trend |
| Refusal rate | Trend, both directions annotated |
| Quality by slice | Short / long / multi-hop / no-answer / per-tenant |
| Judge agreement | Sampled human labels vs the judge — catches judge drift |
| Output-token distribution | p50/p95 — shifts before quality scores |

**The judge-agreement panel is the subtle one:** *"A quality score can drift because the system changed or because the judge changed. Those need different responses, so I sample human labels monthly and track agreement with the judge. If agreement falls, the judge drifted and I fix the judge. If agreement holds and the score fell, the system drifted and I fix the system."*

---

## 9. Change management

| Change | Owner | Review | Notice |
|---|---|---|---|
| Retry policy | Platform | — | 1 day |
| Breaker thresholds | Platform | — | 1 day |
| Bulkhead sizes | Platform + capacity | — | 1 week |
| Timeout values | Platform | — | 1 day |
| Provider priority order | Platform | — | 1 day |
| Hedging delay | Platform + FinOps | FinOps | 1 day |
| **Degradation ladder order** | **Platform + product** | **Product** | **1 week** |
| Semantic validation thresholds | ML/eval owner | Eval owner | 1 week |
| Quality SLO thresholds | Eval owner + product | Product | 2 weeks |
| Provider addition/removal | Platform | Eng lead | 1 quarter |

**The ladder-order change needs product sign-off**, for the same reason as the capacity shedding ladder: reordering it changes who gets a degraded answer.

**The provider change is a quarter-long project**, not a config edit. A new provider means a new tokeniser, a new tool-call format, new rate limits, new refusal behaviour, and a cross-provider eval run. *"Adding a provider is not adding a URL. It is adding a system with different semantics, and the failover path will surface every difference at the worst possible moment unless you've already run 25% of traffic through it."*

---

## 10. Audit and compliance

| Record | Contents | Retention |
|---|---|---|
| Reliability incidents | Class, timeline, blast radius, MTTR, root cause | 3 years |
| Failover events | From provider, to provider, duration, cause | 3 years |
| Degradation events | Level, duration, requests affected, authority | 3 years |
| Breaker state transitions | Timestamps, trigger metric, threshold | 1 year |
| Quality SLO breaches | Metric, value, duration, action taken | 3 years |
| Semantic validation blocks | Stage, reason, count — **not content** | 1 year |
| Judge agreement samples | Human labels vs judge, per month | 2 years |

**The compliance-relevant point:** *"If a regulated decision was made on a degraded or unvalidated response, you need to be able to show what level the system was at. That is why degraded responses carry a marker and why degradation events are logged with an authority — 'the system was at REDUCED level between 14:02 and 14:31' is an auditable statement."*

---

## 11. Interview Q&A

**Q1. What's different about reliability for LLM systems versus normal services?**
Two things. The first is that the dependency is less reliable — LLM APIs have higher error rates, more variable latency, and harder rate limits than a typical internal service, so retry, circuit breaking, and failover matter more, not less. The second, and the more interesting one, is that there's a failure class with no classical analogue: the response can be perfectly well-formed, fluent, confident, and wrong. Everything I know about reliability engineering assumes a failure is visible — a 503, a timeout, a bad status code. A semantic failure has a 200 status. So I need a second layer that classical patterns don't give me: schema validation, citation validity, grounding checks, and a sampled judge. And I need a quality SLO with its own alert, because the failure mode is that every infrastructure metric stays green while the answers get worse.

**Q2. How do retries and circuit breakers differ?**
They solve different problems. Retry handles transient failure — it assumes the next attempt might succeed. The circuit breaker handles systemic failure — it assumes the dependency is unhealthy and that repeated attempts waste resources and actively slow recovery. They compose: a request fails, retries with backoff, and when the retries exhaust the breaker records a failure. After five consecutive failures the breaker opens and requests fail fast instead of each one walking the full retry ladder. Without the breaker, during an outage every request waits about seven seconds of backoff before failing, so your p99 becomes your retry ladder length and your concurrency slots are all occupied by requests that are going to fail anyway.

**Q3. What do you retry and what do you never retry?**
Retryable: rate limits, timeouts, 503s, connection errors. Non-retryable: auth errors, invalid requests, content policy violations, context length exceeded, and hard quota exhaustion — because those are deterministic and retrying makes them worse. Two special cases. First, a side-effecting call needs a deterministic idempotency key before it can be retried at all, because a timeout doesn't mean the operation failed, it means you don't know — and without a key, retrying a refund produces a duplicate refund. Second, and this is the AI-specific one, never retry a semantically wrong answer. At temperature 0 it will be wrong again identically; at higher temperature it will be differently wrong. Either way you've paid twice and you're no closer. Semantic failures get one of four responses: repair with a changed prompt, escalate to a stronger model, abstain, or fail honestly.

**Q4. Explain the circuit breaker configuration.**
Five consecutive failures trips it, a 30-second recovery timeout before probing, a maximum of 3 probe calls in the half-open state, and 2 successes to close. The key is that it's per-route, not per-provider — the key is `provider:model:operation`, because a provider can be healthy for chat completions and failing for embeddings, or rate-limited on the frontier model while the small model is fine. A single breaker per provider either trips too late or takes down routes that were working. And I tune the threshold by volume: three failures at low traffic where each failure is a meaningful sample, twenty plus a 50% error rate at high traffic where a brief blip shouldn't trip it.

**Q5. What's a bulkhead and why does it matter here?**
It's resource isolation by workload class, so one class can't exhaust the capacity of another. It matters more in AI than in most systems because the workloads are wildly heterogeneous — a batch re-embedding job and a user's question are the same API call and completely different resource profiles. I run separate bulkheads for critical, realtime, agents, and batch. The one people forget is agents: an agent run holds 5 to 40 concurrency slots for 30 to 90 seconds, so if agents share the realtime pool, one agent surge takes interactive RAG down. And queue-full has to be explicit — reject fast with a 429, don't queue unboundedly, because an unbounded queue turns a load problem into a latency problem and then into a memory problem.

**Q6. How do you set timeouts for an LLM call?**
Layered and adaptive. Layered means connection 5 seconds, read 30 seconds, total 60, and critically the inner timeouts must be shorter than the outer ones — a 12-second generation timeout inside a 3-second SLO isn't enforcing anything, it just happens that p95 is under it. Adaptive means the total is derived from the observed p99 with a 20% buffer, clamped between 10 and 120 seconds, recomputed from a rolling window of a thousand observations. LLM latency varies enormously with output length, so a fixed 30-second timeout is simultaneously too tight for a long synthesis and too loose for a classification. And per-class timeouts are separate, because a classification and a synthesis should not share a number.

**Q7. How does failover work across providers?**
A provider manager with a priority order, per-provider health state and circuit breakers, and a recheck after 30 seconds. On a retryable error it marks the provider unhealthy and moves to the next. The non-obvious design decision is that the secondary always takes 25% of steady traffic, not 0%. A failover provider you've never sent traffic to is one you've never tested, and you'll discover the tokeniser difference, the tool-call format difference, and their rate limit shape at 3 a.m. during an incident. So 25% steady keeps the path warm, and I run a cross-provider eval in CI so a prompt that only works on the primary is caught before the incident rather than during it. The caveat on the availability math is correlated failure — two providers in the same region is closer to 99.99% than the 99.9999% the multiplication suggests, because they share DNS and network paths.

**Q8. What is request hedging and when do you use it?**
It's sending a duplicate request to a second provider if the first hasn't responded within a threshold, and taking whichever returns first. It's for latency, not for failure — the primary hasn't failed, it's slow. With a 2-second hedge delay on a 2.7-second p95, about 12% of requests get hedged, it costs roughly $0.0014 per request extra, and it halves the p99. Below 1 second you're hedging a third of traffic for a marginal gain; above 4 seconds you're barely hedging anything. So 2 seconds, around the p95, hedges the tail only. And you have to cancel the pending request when one returns, or you pay for both.

**Q9. What's your graceful degradation ladder?**
Five levels. FULL is everything. REDUCED drops to a smaller model and skips reranking and verification. MINIMAL drops retrieval and tools entirely and uses the smallest model. CACHED serves only from the semantic cache and abstains on a miss. OFFLINE returns an honest error with an ETA. Two rules make it safe. First, safety is never a degradation level — the ladder can remove retrieval, reranking, verification, the frontier model, tools, caching, but it can never remove input or output guardrails, PII filtering, or the action gate. Capability degrades, safety doesn't. Second, never silently serve a worse answer — every degraded response carries a marker saying which level produced it, because at MINIMAL retrieval is off and the model may hallucinate, and a user who doesn't know that will trust it.

**Q10. How would you detect that answer quality dropped with no other signal?**
A sampled LLM-judge score as a first-class SLI, evaluated on about 1% of production traffic, with an alert at 3.8 out of 5 and a hard look at 3.5. Plus faithfulness, measured as whether the claims are entailed by the retrieved evidence, and the hallucination rate. Plus two things that move earlier than the scores do: the refusal rate in both directions, and the output-token distribution. And an output-token shift is worth calling out, because a provider changing a model's verbosity is a leading indicator that something underneath changed. The mistake is making quality a dashboard someone checks weekly rather than an SLI with a pager.

**Q11. What happens if the provider silently changes the model?**
That's the model-regression failure and it's nasty because you changed nothing. The defences are layered. Pin the snapshot in the release manifest — never a floating alias. Record the served snapshot per request and alert on any mismatch between served and manifest, which is critical severity because it means the provider mutated something under you. Run a frozen golden set against production daily, not just at release time, because release gates only catch changes you made and a provider change happens on their schedule. And keep a small held-out eval you never tune against, so drift in your own eval set doesn't hide drift in the model.

**Q12. How do agents fail differently?**
Partial failure. A task that fails at step 7 of 10 has already taken six side effects, and there's no classical pattern for that — a failed HTTP request either happened or didn't. So agents need durable checkpointing after every step so a crash resumes rather than restarts, idempotent side-effecting steps so a resume doesn't duplicate, compensating actions where a reversal is possible, and an explicit terminal state — completed, partial, or failed — never an ambiguous "done". The worst outcome isn't a failure, it's a task that reports success having completed six of ten steps. And separately, the tool-call loop needs ceilings inside the loop, not alerts after it, because an alert is a notification and only a counter is a control.

**Q13 (pushback). Isn't all this redundancy expensive? Is 99.9% worth it?**
The direct cost is about 4% of inference spend plus 25% for keeping a second provider warm, plus roughly 450 milliseconds of p95 for the semantic validation layer. Whether 99.9% is worth it depends on what a failure costs, and that's a product question, not an engineering one. For a support agent, an outage is lost productivity and a wrong answer is a bad refund — both matter, and I'd argue the wrong answer matters more because it's invisible. What I'd push back on is the idea that the semantic validation is optional. It's 410 milliseconds and seven hundredths of a cent per request, and without it the system confidently ships wrong answers with a 200 status code. If I have to cut the reliability budget, that's the last thing I give up, not the first.

**Q14. How do you test all of this?**
Chaos drills, and they have to be scheduled, because nobody does them voluntarily. I inject provider errors at a controlled rate and verify the breaker opens, the failover happens, and the latency SLO holds. I kill the primary provider entirely and verify the degradation ladder fires in the right order and that the safety checks are still running. I throttle to trigger bulkhead rejection and confirm the shedding order is right. I inject malformed JSON and confirm the repair path. I inject a semantically wrong answer — hardcode a bad response — and confirm the grounding check catches it. And I explicitly verify that after all of that, the guardrails and the action gate still fire. That last assertion is the one that matters most and the one most likely to be broken by a well-meaning optimisation.

**Q15. What's the single most important reliability metric?**
The sampled quality score. Availability and latency are necessary but they can both be perfect while the system produces confident wrong answers, and that's the failure that damages users and trust. Everything else on my dashboard is a leading indicator for that one number, or a guard against it going bad silently.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Two failure classes** | Infrastructure (visible) · **Semantic (invisible, 200 status)** |
| **Retry config** | max 3 · base 1 s · max delay 60 s · base 2 · **jitter ±50%** |
| **Retryable** | 429 · timeout · 503 · connection |
| **Never retry** | 401/403 · 400 · policy · context overflow · hard quota |
| **Never retry (AI)** | **A semantically wrong answer** |
| **Side-effect rule** | Deterministic **idempotency key**, or no retry at all |
| **Fleet retry cap** | 10% of traffic — above that, stop retrying globally |
| **Breaker config** | **5 failures · 30 s recovery · 3 half-open probes · 2 successes to close** |
| **Breaker key** | `provider:model:operation` — **not** per provider |
| **Breaker states** | CLOSED → OPEN → HALF_OPEN → CLOSED |
| **Bulkheads** | critical 10 · realtime 50 · agents 30 · batch 200 |
| **Queue behaviour** | Bounded — reject fast, never queue unboundedly |
| **Timeouts** | connect 5 s · read 30 s · total 60 s |
| **Adaptive timeout** | **p99 × 1.2**, clamped 10–120 s, 1,000-obs window |
| **Timeout rule** | Inner < outer; the sum must fit the SLO |
| **Failover providers** | primary 70% · **secondary 25% steady** · tertiary 5% |
| **Hedge delay** | **2.0 s** (~p95) · ~12% hedged · halves p99 |
| **Degradation levels** | FULL · REDUCED · MINIMAL · CACHED · OFFLINE |
| **Degradation rule 1** | **Safety is never a level** |
| **Degradation rule 2** | **Never silently serve a worse answer** |
| **Recovery** | One level per 5 min — never restore at once |
| **Validation stages** | schema → constraints → citations → grounding → refusal → judge |
| **Validation cost** | **~410 ms · $0.0007 per request** |
| **Sampled judge** | 1% of traffic, async |
| **Quality SLO** | ≥ 4.2/5.0 · alert < 3.8 |
| **Faithfulness SLO** | ≥ 0.85 · alert < 0.80 |
| **Hallucination SLO** | ≤ 0.03 · alert > 0.06 |
| **Schema-valid SLO** | ≥ 0.995 |
| **Agent partial-completion SLO** | ≤ 0.02 |
| **Agent defence** | Ceilings **inside** the loop, not alerts after |
| **Model-regression defence** | Pin snapshot · alert on served≠manifest · **daily production golden set** |
| **Availability math** | 2 providers ≈ **99.99%** (not 99.9999%) — correlated failure |
| **Reliability overhead** | ~4% of inference cost · ~450 ms p95 · 25% provider redundancy |
| **The rule that never breaks** | **Capability degrades; safety and honesty do not.** |

---

## 13. Cross-links

- [Capacity Planning and Autoscaling](05-capacity-planning-and-autoscaling.md) — saturation, load shedding, and the capacity side of degradation
- [Cost and FinOps](06-cost-and-finops.md) — retry storms as a cost event, and the cost of redundancy
- [DR, Failover and Backup](09-dr-failover-and-backup.md) — region loss, the failure this playbook does not cover
- [Deployment, Release and CI/CD](04-deployment-release-and-cicd.md) — rollback as the first incident action
- [Security, Guardrails and Compliance](08-security-guardrails-and-compliance.md) — the safety controls that never degrade
- [LLMOps for AI Systems](10-llmops-for-ai-systems.md) — reliability inside the operational lifecycle
- [Guardrails and Safety for AI Systems](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) — the four guardrail layers
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — the judge and the quality SLOs
- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md) — durable execution and partial failure
- [Availability Numbers and SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
