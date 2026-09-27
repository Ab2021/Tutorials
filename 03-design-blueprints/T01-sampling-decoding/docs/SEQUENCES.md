# Sequences: Sampling & Decoding

> `T01` · [HLD](../HLD.md) · [LLD](../LLD.md) · [Case study](../../../01-case-studies/T01-sampling-decoding.md)

End-to-end flows for the sampling policy plane. Each step names where it can fail; the failure
taxonomy is in [LLD §7](../LLD.md#7-error-handling).

---

## 1. Cold start — first request after a deploy

```mermaid
sequenceDiagram
    autonumber
    participant S as Surface
    participant R as Router
    participant P as Policy Resolver
    participant REG as Registry
    participant E as Engine

    S->>R: request(policy_name="support.chat", model_id="m-70b-r5")
    R->>P: resolve(surface, model_id)
    P->>REG: read surface->version pointer
    REG-->>P: "support.chat.v7"
    P->>REG: read v7 (immutable snapshot)
    REG-->>P: SamplingPolicy
    P-->>R: SamplingPolicy + policy_id
    R->>R: cache keyed by (surface, model_id, policy_id)
    R->>E: generate(prompt, params, policy_id)
    Note over E: first request also pays kernel<br/>warm-up and prefix-cache cold miss
    E-->>R: tokens + policy_id
    R-->>S: response + policy_id
```

**Failure points.** Registry unavailable → the resolver serves last-known-good and logs at WARN; the
request still succeeds. A *missing* surface policy is not a fallback — it returns 400, because a
surface that was never registered should not silently inherit another's decoding behaviour.

**Why the first request is special.** Prefix-cache state, compiled kernels and quantization
calibration all differ before warm-up, so the same prompt can produce a different completion. This
is why replay audits are re-run only *after* the warm-up window — auditing against a cold start
would produce false divergences.

---

## 2. Steady state — one decode step

```mermaid
sequenceDiagram
    autonumber
    participant E as Engine
    participant M as Mask
    participant T as Temperature
    participant TR as Truncation
    participant D as Draw
    participant X as Diagnostics

    E->>M: logits (vocab=128k)
    M->>M: zero forbidden indices, renormalise
    M->>T: masked probs
    T->>T: p^(1/T), renormalise
    T->>TR: reshaped probs
    TR->>TR: keep by rule, renormalise
    alt survivors >= 1
        TR->>D: candidate set
    else survivors == 0
        TR->>D: fallback: temperature-only
        TR-->>X: empty_set counter
    end
    D->>D: rng = H(request_id, seed, step)
    D-->>E: token_id
    D-->>X: entropy, survivors, retained_mass, ties
```

**Failure points.** The `survivors == 0` branch is the only in-band failure, and it is *expected* —
it is counted, alerted on by rate, and rescued. `MaskDisjointFromSupport` (the grammar permits
nothing with non-zero probability) is different and is raised: it means the grammar and the model
disagree, and no amount of sampling will fix it.

**Why the mask is first.** `run.py` §4 shows the degenerate case concretely: with 10 valid tokens all
in the tail, truncate-then-mask empties the candidate set while mask-then-truncate leaves 10
drawable candidates. Reordering the chain is a one-line change that silently breaks a fraction of
requests — which is why the test strategy asserts the ordering as a property rather than testing it
as a configuration.

---

## 3. Policy release — draft to active

```mermaid
sequenceDiagram
    autonumber
    participant A as Author
    participant G as Eval Gate
    participant REG as Registry
    participant C as Canary (5% one surface)
    participant M as Monitor

    A->>G: submit candidate policy (draft)
    G->>G: JSON validity, distinct-n, bigram overlap, judge fluency
    Note over G: 200 curated prompts, not 10,000 random
    alt eval failed
        G-->>A: rejected with per-metric deltas
    else eval passed
        G->>REG: write version (immutable, eval_run_id set)
        G->>C: point 5% of one surface at the new version
        C->>M: TTFT, ITL, output-length distribution, error rate
        alt 24h clean
            M->>REG: promote (surface -> new version)
        else regression
            M->>C: revert pointer to previous version
        end
    end
```

**Failure points.** The gate is the only writer to the registry, so a promotion that skips it is not
representable. A regression during canary reverts a *pointer*, not a deployment — rollback is
seconds, which is what makes the canary cheap enough to run on every policy change.

**The judge caveat is load-bearing.** The gate never judges with the same model family that
generated the text: "Claude will really like Claude, GPT will really like GPT" `[T]`. The 5th
percentile of outputs is sampled for human review, because a judge score used as ground truth is
how a change that lowers real quality raises the judged score.

---

## 4. Prefix-cache hit versus miss

```mermaid
sequenceDiagram
    autonumber
    participant S as Surface
    participant R as Router
    participant E as Engine

    S->>R: request(system_prompt_shared, policy_id)
    R->>R: look up prefix-cache locality
    alt cache hit on this pod
        R->>E: route to warm pod
        E-->>S: TTFT low
    else cache miss
        R->>E: route to any pod
        E->>E: recompute prefix
        E-->>S: TTFT high
    end
```

**Why this is in a *sampling* sequence document.** The coupling is indirect but real: a decoding
policy determines average output length, output length determines KV pressure, and KV pressure
determines which pods can accept the request. A policy that lengthens outputs moves the router's
placement decisions. This is the mechanism behind incident #4 in the case study's runbook — a
Copy Studio latency spike caused by six long candidates per brief occupying KV, not by anything in
the sampler.

---

## 5. Failure — empty candidate set

```mermaid
sequenceDiagram
    autonumber
    participant E as Engine
    participant TR as Truncation
    participant FB as Fallback
    participant X as Diagnostics
    participant AL as Alerting

    E->>TR: masked + reshaped probs
    TR->>TR: apply rule
    TR-->>FB: 0 survivors
    FB->>FB: re-run temperature-only (no truncation)
    FB-->>E: drawable token, fell_back=true
    FB->>X: sampler.empty_set +1, sampler.fallback +1
    X->>AL: rate > threshold for 5m?
    AL->>AL: page: check the grammar version first
```

**Diagnosis order matters.** The instinct is to loosen the policy. The correct first check is the
*grammar*: an empty set means the schema and the distribution have diverged, and that is usually a
schema-version change (or a token-healing heuristic that was disabled — see
[T03](../../T03-constrained-generation/HLD.md)), not a sampler problem. Loosening `top_p` would hide
the symptom and let the real defect ship.

---

## 6. Recovery — rollback

```mermaid
sequenceDiagram
    autonumber
    participant M as Monitor
    participant OP as Operator
    participant REG as Registry
    participant R as Router

    M->>OP: quality regression on support.chat
    OP->>REG: read surface->version history
    REG-->>OP: current v7, previous v6 (still readable)
    OP->>REG: move pointer to v6
    REG-->>R: next resolve returns v6
    Note over R: in-flight requests keep v7 in their<br/>response metadata -- version captured at resolve
    OP->>M: confirm metrics recover
```

**The subtlety.** In-flight requests must keep the `policy_id` they resolved with. The version is
captured at resolution and carried through, never re-read at completion — otherwise a rollback
mid-request would mislabel the output, and the audit trail would attribute a v7 completion to v6.
That is precisely the failure this design exists to prevent.

---

## 7. Scale-out

```mermaid
sequenceDiagram
    autonumber
    participant AU as Autoscaler
    participant ENG as Engine pool
    participant R as Router
    participant P as Policy Resolver

    AU->>ENG: add replicas (KV-utilisation signal, not CPU)
    ENG->>P: resolver already cached -- no re-resolve storm
    Note over P: the policy plane does not scale with tokens,<br/>only with the number of policies
    R->>R: rebalance; prefix locality degrades briefly
    R->>ENG: traffic spreads
```

**What scales and what does not.** The engine pool scales horizontally. The policy resolver scales
horizontally and trivially, because its work is O(1) per request and its state is immutable. The
*eval gate* does not scale with traffic at all — it scales with the number of releases, and at 10x
surfaces the hand-run 200-prompt eval stops fitting, which is the first thing that breaks in the
control plane rather than the data plane.

---

## 8. Scale-in

```mermaid
sequenceDiagram
    autonumber
    participant AU as Autoscaler
    participant R as Router
    participant ENG as Engine pool

    AU->>R: drain node
    R->>ENG: stop routing new requests
    ENG->>ENG: finish in-flight sequences
    Note over ENG: never kill mid-sequence for capacity --<br/>a partial generation is a failed request
    ENG-->>R: drained
    R-->>AU: node free
```

**Failure point.** Draining must respect `max_tokens`. A sequence with an unbounded generation budget
would hold a node indefinitely, which is why the case study's tunable order puts `max_tokens` first:
it is the only parameter that bounds worst-case occupancy.
