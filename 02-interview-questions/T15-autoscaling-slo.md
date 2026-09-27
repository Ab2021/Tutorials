# Interview Bank: Autoscaling and SLOs

> `T15` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T15-autoscaling-slo.md) · [Case study](../01-case-studies/T15-autoscaling-slo.md) · [Design blueprint](../03-design-blueprints/T15-autoscaling-slo/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** progressing from signal choice to contract-grade design

## How to use this bank

The questions are ordered so the bank reads as one interview: what the signal actually is, then the
SLO that the signal serves, then the arithmetic, then degradation, then open design on a fleet whose
GPU count cannot change. Ask them in order for a 40-minute loop, or sample the L5 block for a staff
screen.

Every number here is traceable to a transcript (`[T]`, speaker and talk named), a supporting repo
(`[R]`, path named), or is my own derivation (`[D]`, with assumptions shown). The topic has one
correction at its spine — **KV cache utilisation is the autoscaling signal, never CPU** — and one
contradiction in the corpus that is itself a question (see T15-Q9).

---

### Fundamentals — what you are actually scaling

#### T15-Q1 · Why does a CPU-based HPA never fire on a vLLM deployment?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** We run a standard Kubernetes HPA on CPU utilisation for our vLLM pods and it has never
scaled meaningfully. Someone says it's misconfigured. Is it?

**Model answer:** It is not misconfigured. It is measuring the wrong resource, and the corpus states
the correction directly: autoscaling for LLM serving means "Scaling based on **KV Cache utilization**
rather than CPU or standard memory usage" `[R]`
(`ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`).

The mechanism is that a vLLM replica with a saturated KV cache and a full batch is often at
*moderate* CPU. The work is on the GPU, and prefill kernels are launched asynchronously, so the host
process is doing tokenisation, sampling bookkeeping and request handling — not the arithmetic that
got harder. CPU utilisation therefore tracks *request-handling overhead*, and it does not rise
monotonically with load, because a larger batch is more GPU work per unit of host work.

The second half of the correction is that **GPU utilisation is also wrong**, for the opposite
reason: LLM inference saturates the device by design, so a healthy pod looks like an overloaded one
`[D]`. A utilisation trigger on GPU either fires constantly or, under a utilisation target written
for CPU-style workloads, never crosses its threshold.

The observable signature of this failure is *sustained latency growth under load with a flat replica
count* — the first row of the case study's failure-mode table. Nothing errors; the fleet simply does
not respond to demand.

**Signal:** Names CPU as the wrong resource *and* GPU utilisation as wrong for the opposite reason.
A candidate who only says "use GPU utilisation instead" has moved to a different wrong answer.

**Follow-ups:**
- *What should you scale on instead?* — KV utilisation and queue depth; T15-Q2.
- *Why doesn't the HPA look broken?* — it never errors, it just never fires.
- *Where does the corpus itself get this wrong?* — its own shipped manifest; T15-Q9.

**Red flags:** Proposes raising the CPU target or adding a memory metric, or believes the HPA is
merely mistuned.

---

#### T15-Q2 · What signal should an LLM autoscaler actually consume?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** If not CPU, then what? Give me the signal you would put on the scaler, and justify it.

**Model answer:** I would give a composite, but if forced to one signal it is **KV cache
utilisation**, because it is the direct expression of the resource that actually binds.

The reasoning is that the binding constraint on LLM serving is **KV memory, not compute** `[D]`.
Capacity math starts from `HBM_for_KV / (bytes_per_token × avg_context_len)`, and that ceiling is
what you run out of first. KV utilisation is a per-pod, cheap, direct measurement of how close a
replica is to eviction and preemption `[R]` (`04/06`), which is exactly the quantity that decides
whether the next request is served or recomputed.

The corpus's second example signal is **queue depth**: llm-d's contribution is described as
"saturation based autoscaling" `[T]` (Pravin, *Scaling Agentic AI Distributed Inference with llm-d*),
with the operator defining saturation rather than the framework doing it — "when the KV cache is 80%
full I declare that the cluster is saturated, or the number of active requests a particular cluster
is seeing on average is more than eight" `[T]`.

Note what that quote establishes: **80% and 8 are operator-defined examples, not framework
defaults.** A candidate who recites them as shipped defaults has misread the source.

My composite, in priority order: **queue depth drives scale-out** (it is causal about user harm),
**KV utilisation drives per-replica pressure actions** (it tells you *which* pod to relieve), and
**band-specific latency percentiles act as guardrails** rather than triggers.

**Signal:** Reaches the KV-is-the-binding-resource argument rather than naming a metric, and flags
the 80%/8 thresholds as operator examples.

**Follow-ups:**
- *When does KV utilisation stop discriminating?* — at 100%; T15-Q10.
- *When is queue depth meaningless?* — if the router sheds instead of queueing; T15-Q11.
- *What does each signal trigger?* — scale-out versus per-replica action; T15-Q10.

**Red flags:** Names a signal with no resource argument behind it, or quotes 80% and 8 as defaults.

---

#### T15-Q3 · Name the four scaling axes, in the order you would use them
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** "Add more replicas" is one way to absorb load. What are the others, and why does the
order matter?

**Model answer:** There are four scaling axes, and only the last one is a replica count:

1. **Batch parameters** — `max_num_batched_tokens`, `max_num_seqs`, chunked-prefill size. Time to
   effect: sub-second. Ceiling: the latency-versus-throughput frontier. Cost: free.
2. **Routing / work shaping** — send some traffic to a smaller model, raise the prefix cache hit
   rate. Sub-second. Ceiling: quality. Free.
3. **Admission / priority** — queue the batch band, protect interactive. Sub-second. Ceiling:
   user-visible queueing. Free.
4. **Replica count** — more pods. **15–20 s at best** `[R]` (`04/06`). Ceiling: the procured GPU
   count. Cost: CapEx and power.

The ordering matters because teams reach for axis 4 first — it is the one Kubernetes makes easy —
when axes 1–3 are faster, cheaper and reversible. In the case study's sovereign deployment, where
GPUs are procured annually and there is no spare pool, axes 1–3 are not optimisations; they are the
*only* real elasticity available. The case study's worked example is the proof: a 12 req/s nightly
batch shortfall against a fixed fleet is closed with re-parameterisation and small-model routing
**without procuring a fifth GPU** `[D]`.

Note the third axis is not a failure mode. Load shedding is a feature: a system that degrades
gracefully serves more goodput than one that accepts everything and misses every SLO `[T]` (the
corpus's flow-control framing, llm-d).

**Signal:** Lists all four with the correct latency-to-effect ordering, and volunteers that the last
one is the slowest and the most expensive rather than the default.

**Follow-ups:**
- *Which axis is free and instant?* — 1 and 3; 2 costs quality, not money.
- *Why is axis 4 last in a sovereign fleet?* — there are no spare GPUs; T15-Q27.
- *Which axis is a contract clause?* — axis 3; T15-Q21.

**Red flags:** Lists only replica count and vertical scaling, or treats admission control as an
outage rather than a design element.

---

#### T15-Q4 · What does the 15–20 second cold boot actually rule out?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** The corpus gives a cold-boot figure. What is it, and what does it forbid you from
building?

**Model answer:** The figure is "**Cold Booting**: Using **Un-quantized Base Images** and loading
weights from a high-speed Lustre/mount to reduce startup time from **minutes to 15-20 seconds**" `[R]`
(`ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`).

Two things about that number must be said precisely, because they are routinely conflated:

**It is an optimised floor, not a typical time.** Twenty seconds is what you get *after* you have
un-quantized base images and a high-speed mount. It does not include the model download, the
container pull, CUDA context initialisation, or weight load for a large model — those are the
"minutes" the guide says were reduced to reach twenty seconds. `[R]`

**It is the floor before anything is warm.** The replica is *up*, not *useful*: no prefix cache, no
compiled kernels for your shapes, no warmed allocator.

What it rules out is **reactive scaling as a primary strategy.** If demand can rise 2× in 30 seconds
— and the case study's step-change scenario does exactly that — then a policy whose fastest possible
response is 20 seconds cannot protect the SLO. The full readiness budget is the cheat sheet's:

```
T_ready ≈ T_schedule + T_image + T_weights + T_warm      [D]
```

where for a large model `T_weights + T_warm` dominates and the realistic figure is minutes `[D]`.
Any autoscaler whose `T_ready` exceeds the spike duration is decorative; that capacity must already
be resident.

**Signal:** Says "floor, not typical" and derives the design consequence — that the strategy must be
a warm floor plus prediction rather than reactive scale-out.

**Follow-ups:**
- *So what replaces reactive scaling?* — a warm floor sized to the ramp; T15-Q5.
- *Where does the budget go for a 70B model?* — weights and warmup; T15-Q15.
- *What is the agent-sandbox figure?* — seconds to 10–15 s; T15-Q25.

**Red flags:** Quotes 20 s as a planning assumption for *readiness*, or proposes scale-to-zero for an
interactive pool.

---

#### T15-Q5 · Why is the warm floor sized by the ramp and not by the average?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Your business-hours load is 55 req/s. Your engineer proposes running two replicas
because that covers the average with headroom. What is wrong with the plan?

**Model answer:** It sizes for the average and ignores the *rate of change*, which is the quantity
the cold-start floor makes binding.

The rule the case study states is: **the warm floor is sized by the fastest demand ramp you must
survive, not by average demand.** A replica that takes 15–20 s to become useful `[R]` cannot help you
with a 30-second step. So the question is not "what serves 55 req/s?" but "what serves the load
30 seconds from now, given that I cannot create capacity in 30 seconds?"

Run the arithmetic from the case study's model `[D]`, using its contract anchor of 32 req/s per H100
equivalent (see T15-Q16 for where that number comes from and why it is a contract, not a benchmark):

```
Replicas at saturation for 55 req/s  = 55 / 32  = 1.72  -> 2 replicas
Warm floor for a 2x ramp (110 req/s) = 110 / 32 = 3.44  -> 4 replicas
Headroom at business hours           = 4 x 32 = 128 req/s
Utilisation at business hours        = 55 / 128 = 43%
```

The counter-intuitive part — and the thing candidates miss — is the consequence: **43% is below the
55% fleet-utilisation target, and that is the point.** The warm floor deliberately trades average
utilisation for ramp survival. Those two requirements are in direct conflict, and the contract, not
the engineer, is what resolves them. Procurement must be told that the floor *is* the cost of the
SLO rather than being asked to hit a utilisation number that only a fleet sized to average demand
could hit.

**Signal:** Recognises that the floor is a ramp-survival artefact, and volunteers the utilisation
conflict rather than presenting 43% as a problem to be fixed.

**Follow-ups:**
- *What if the ramp is slower than the cold start?* — then reactive scaling is legitimate; T15-Q27.
- *How do you know the ramp distribution?* — measure it before setting the floor; T15-Q5's runbook.
- *Who owns the utilisation target?* — procurement, and it must be renegotiated; T15-Q19.

**Red flags:** Sizes to average with a safety factor, or treats low utilisation on a warm floor as
evidence of over-provisioning.

---

#### T15-Q6 · What is goodput, and why is throughput not enough?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** A colleague proposes scaling on tokens per second because "throughput is what we
sell." What is wrong with the objective?

**Model answer:** Throughput measures output produced; goodput measures output produced *usefully*.
The difference is the SLO, and scaling on the first maximises the wrong thing.

The cheat sheet's definition `[D]`:

```
goodput = |{r : TTFT(r) <= T_t  AND  ITL(r) <= T_i}| / |requests|
```

Every request that violates either bound is subtracted. That matters because **throughput and latency
trade against each other continuously** — the throughput/latency frontier is what continuous batching
exists to exploit `[R]` (`llm-inference-engineering-main/llm-inference-engineering-main/README.md`) — so a
fleet that runs at maximum batch size will show excellent tokens per second while a growing share of
requests miss their latency bound. It is producing tokens nobody can use.

Concretely: if you scale on throughput, the scaler has no incentive to stop growing the batch, and a
busy replica looks maximally efficient right up to the point where every user is unhappy. That is the
cheat sheet's failure row: *"Throughput looks great, users unhappy → optimising throughput, not
goodput"* `[D]`.

There is a second reason goodput is the right *autoscaling* objective: it is the only metric that
respects the SLO by construction. Throughput cannot be turned into a scaling trigger without
inventing a latency constraint somewhere else; goodput already contains one.

The honest caveat, which I would state: goodput needs an SLO definition **per band**, and it is
slower to react than queue depth, so it is a slow-loop scaling signal and the SLO metric rather than
the fast trigger.

**Signal:** Gives the formula with the SLO constraint inside it and explains that throughput scaling
actively rewards violating latency.

**Follow-ups:**
- *What does the throughput/latency frontier come from?* — continuous batching; `[R]` README.
- *Goodput vs attained rate?* — goodput is per-request quality; attained rate is the contract; T15-Q12.
- *Why per-band?* — interactive and batch have different bounds; T15-Q7.

**Red flags:** Says goodput is "throughput but better" with no SLO constraint, or proposes scaling on
tokens/s.

---

#### T15-Q7 · Why are TTFT and ITL two separate SLOs rather than one latency SLO?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Why does the cheat sheet insist on two latency SLOs instead of one "latency" number?

**Model answer:** Because they are driven by **different resources in different phases**, and a
single latency SLO hides which pool to scale.

```
TTFT_p95 <= T_t   (time to first token  -- prefill-bound)
ITL_p95  <= T_i   (inter-token latency  -- decode-bound)      [D]
```

TTFT is dominated by the prefill: how long the prompt takes to process before the first token
emerges. ITL is dominated by the decode loop: how fast each subsequent token is produced, which at
long context is a memory-bandwidth problem — decode reads the whole KV cache per token. These respond
to different interventions. Prefill responds to prefill pool capacity, chunked-prefill tuning and
sequence parallelism; decode responds to batch size, KV bandwidth, and quantisation.

The operational consequence is the cheat sheet's **two-tier scaling** rule: scale the decode pool on
ITL/goodput and the prefill pool on TTFT/queue depth `[D]`. A single "latency" SLO collapses those
two decisions into one number, and the failure is *scaling the wrong tier* — you add decode capacity
while the queue is prefill-bound, latency does not move, and the fleet grows for nothing.

Context for why this topic is hot right now: agentic traffic is **prefill-heavy**, with prefill
occupying roughly **98% of tokens** `[T]` (llm-d, *Scaling Agentic AI Distributed Inference with
llm-d*). That is because agents read code and documents and emit very little — output is largely tool
calls. If your workload has shifted agentic, your binding SLO has shifted toward TTFT, and a policy
that was correctly tuned for a decode-heavy chat mix is now scaling the wrong tier.

**Signal:** Ties each SLO to the phase and resource that drives it, and names the wrong-tier failure
as the consequence of merging them.

**Follow-ups:**
- *Which pool does an agentic workload stress?* — prefill; 98% of tokens `[T]`.
- *Where are the two pools physically separated?* — disaggregation; T13/T12.
- *Does ITL matter for batch?* — usually not; that is why SLOs are per band.

**Red flags:** Says "just use end-to-end latency," or believes TTFT and ITL scale together.

---

#### T15-Q8 · What is Little's Law doing in a capacity plan?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Show me how you turn a concurrency SLO into a number of GPUs. Where does Little's Law
enter?

**Model answer:** Little's Law is the bridge from a concurrency obligation to a throughput
requirement, and it is the step teams skip because concurrency *feels* like capacity already.

```
concurrency = throughput x latency                              [D]
```

The corpus gives a hard concurrency figure to anchor on: **~20,000 maximum concurrency on a tuned
1P1D (one prefill, one decode) pair** `[T]` (ROCm/WideEP, *Distributed Inference on ROCm with WideEP
on vLLM/llm-d*). The cheat sheet's worked conversion `[D]`:

```
20,000 concurrent sessions at ~10 s per session
  => throughput = 20,000 / 10 = 2,000 sessions/s
```

**2,000 sessions/s is the number you buy hardware for** — not 20,000, which is a stock, not a flow.
Conflating the two is the classic error: a candidate who says "we need to support 20,000 sessions so
we need N GPUs" has no idea what N is, because sessions per second and sessions are different units.

The second step closes it, and it is where the topic's spine reappears: the per-replica limit is not
FLOPs, it is KV:

```
max_concurrency = HBM_for_KV / (bytes_per_token x avg_context_len)        [D]
```

The cheat sheet's worked example: **40 GB of KV at 327 KB/token and 4k context gives ~30 sequences
per replica** `[D]` (see [T07](T07-kv-cache.md) for `bytes_per_token`). At 2,000 sessions/s with 10 s
sessions, that is **~67 replicas of that shape** `[D]`.

Then apply headroom: `replicas = ceil(peak_concurrency / max_concurrency_per_replica) x safety_factor`
with `safety_factor ≈ 1.3–1.5` to cover skew and scale-up lag `[D]`. And note *peak*, not mean —
sessions are long-lived and bursty `[T]`.

**Signal:** Converts the concurrency stock into a throughput flow before touching GPU count, and
uses the KV ceiling rather than FLOPs as the per-replica limit.

**Follow-ups:**
- *Why not size on FLOPs?* — KV binds first; the cheat sheet says so explicitly `[D]`.
- *Peak or mean concurrency?* — peak; sessions are bursty `[T]`.
- *What does the 1P1D pairing tell you?* — prefill and decode have separate ceilings; T07/T13.

**Red flags:** Uses concurrency directly as a GPU count, or sizes per-replica capacity from compute.

---

### The signals: what to scale on, and the corpus contradiction

#### T15-Q9 · The corpus ships a CPU-based HPA manifest. Defend it or condemn it.
**Difficulty:** L4 · **Depth expected:** 5–6 min

**Question:** The same supporting corpus contains both of these. Reconcile them.

**Model answer:** They cannot be reconciled — one is the rule and one is the anti-pattern, and
recognising that is the whole question.

**The rule**, from the optimisation chapter: "**Autoscaling**: Scaling based on **KV Cache
utilization** rather than CPU or standard memory usage" `[R]`
(`ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`).

**The contradiction**, from the infrastructure chapter of the *same* corpus: a Kubernetes
`HorizontalPodAutoscaler` for an `llm-service` Deployment with `minReplicas: 2`, `maxReplicas: 20`,
whose metric list is `type: Resource / name: cpu / type: Utilization / averageUtilization: 70`
followed by a `type: Pods` metric on `requests_per_second` at `averageValue: 100` `[R]`
(`.../11-infrastructure-and-mlops/01-llm-infrastructure.md`).

**Why the CPU half fails for LLM serving.** CPU utilisation on a GPU-serving pod tracks
tokenisation and request handling, not the load that matters (T15-Q1). A pod with a saturated KV
cache at moderate CPU sits happily below a 70% CPU target forever, so the manifest's first metric is
inert under exactly the condition it was written for.

**Why the RPS half is better but still insufficient.** `requests_per_second` at least measures
demand rather than a proxy, and it is closer to the contract's unit. But it **ignores request size**
— 1 req/s of 60k-token prompts is far heavier than 100 req/s of 200-token prompts `[D]` — which the
case study calls out as the bimodal-prompt failure. And a `target AverageValue: 100` is a magic
number with no derivation attached.

**The professional point.** If you copy the manifest, you have built the anti-pattern the same
corpus warns against. This is worth saying out loud in a design review, because **the wrong config is
the one that is easiest to find** `[R]`. A candidate who can only cite the rule has read the corpus;
a candidate who notices the corpus contradicts itself has *audited* it.

**Signal:** Names both paths, states that the manifest is the wrong-but-shipped pattern, and gives
the mechanism for *why* CPU fails rather than just asserting the rule.

**Follow-ups:**
- *Is the RPS metric salvageable?* — only for homogeneous short prompts; T15-Q13.
- *What would you replace the manifest with?* — KV utilisation plus queue depth; T15-Q10.
- *Why does this contradiction exist?* — chapters written by different authors for different
  workloads `[D]`.

**Red flags:** Defends the manifest because it is in the docs, or condemns it without explaining the
CPU mechanism.

---

#### T15-Q10 · KV utilisation saturates at 100%. What breaks, and what is the fix?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** We scaled on KV utilisation as advised. At 100% KV the metric stops telling us
anything, and we have also seen a single long prompt hold the metric high while the fleet is idle.
Diagnose both.

**Model answer:** Both are real limits of KV utilisation, and the case study lists them as the
signal's cons: it "saturates at 100% and stops discriminating", and it goes "high under a single long
prompt" `[D]`.

**Failure one — saturation.** KV utilisation is a *pressure* signal: it tells you how close a replica
is to eviction and preemption. Once the cache is full, the system is in a qualitatively different
regime — it is recomputing rather than retaining — and the metric is pinned at its ceiling
regardless of how much worse things get. Two fleets, one serving 1.2× its capacity and one serving 3×,
both read 100%. The signal has no gradient where you need it most.

**Failure two — one prompt holds the ceiling.** A single 60k-token request can fill a large fraction
of a replica's KV. KV utilisation reads near-maximum while the replica is doing very little useful
work per second. This is why the case study's framing is precise: **per-replica KV utilisation is a
pressure signal, not a load signal** `[D]`. It answers "is this pod about to evict?", not "is there
demand I am failing to serve?"

**The fix is the composite**, which is the case study's chosen design: **queue depth drives
scale-out** (it is the only signal causal about unmet demand), **KV utilisation drives per-replica
pressure actions** (which pod to relieve, or re-parameterise), and **band-specific latency
percentiles are guardrails, not triggers** `[D]`. The two signals answer different questions, which
is why the corpus's own example treats 80% KV and mean-active-requests > 8 as *alternative*
saturation declarations rather than one policy `[T]` (Pravin).

The honest caveat: the corpus does not publish a measured KV-utilisation-versus-goodput curve. I
would not assert where 80% sits on the knee; I would measure it.

**Signal:** Separates the pressure/load distinction, and reaches for the composite rather than
patching the one signal.

**Follow-ups:**
- *Why is queue depth the causal signal?* — it measures unmet demand directly; T15-Q11.
- *Why are latency percentiles guardrails?* — noisy; one slow request trips them.
- *Does the corpus give a goodput curve?* — no; do not assert one; T15-Q28.

**Red flags:** Proposes lowering the threshold, or claims KV utilisation is a load metric.

---

#### T15-Q11 · When does queue depth lie to you?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You switched the scaler to queue depth. Under what circumstances does that signal
become meaningless?

**Model answer:** Queue depth is the closest thing to a real load signal because it measures *unmet
demand* directly, but it has one structural precondition and one operational hazard.

**The precondition: a queue must exist.** Queue depth is only observable if there is a flow-control
layer that *holds* requests rather than rejecting them. The case study's own decision table says it
plainly — queue depth's exception is that it "breaks if the router sheds rather than queues" `[D]`.
llm-d's architecture makes the same dependency explicit: the router's flow-control component only
enqueues *after* saturation is detected, and until that point requests pass straight through `[T]`
(Pravin, llm-d). A router that returns 429 immediately produces a queue-depth metric that is
permanently near zero precisely when you are most overloaded.

**The consequence is severe**, and note the direction of the error: the signal goes *quiet* under
overload rather than saturating. That is worse than a saturated metric, because the scaler reads
"idle."

**The hazard: the queue hides which band is suffering.** Aggregate queue depth cannot distinguish
200 queued batch requests from 200 queued premium requests. Since the case study's overload ladder
deliberately queues the best-effort band first (T15-Q21), a single aggregate depth number will
trigger scale-out for traffic you had already decided to defer. Depth must therefore be measured
**per band**, or it will spend hardware on work you meant to delay.

**And the corollary the case study states as the revisit condition:** if the queue is removed in
favour of immediate shedding, the signal set collapses to KV plus latency `[D]` — you lose the only
metric that speaks about user harm.

**Signal:** Names the queue-must-exist precondition unprompted, notes that the failure mode is
*silence* rather than saturation, and requires per-band depth.

**Follow-ups:**
- *What replaces it if you shed instead of queue?* — KV plus band latency; T15-Q21.
- *Why does per-band matter?* — the ladder defers batch first; T15-Q21.
- *Is a 429 a success?* — not if the error taxonomy counts it; T15-Q12.

**Red flags:** Treats queue depth as unconditionally causal, or scales on an aggregate depth that
includes traffic already deferred.

---

#### T15-Q12 · A latency SLO can be met by shedding. Show me the mechanism.
**Difficulty:** L4 · **Depth expected:** 5 min

**Question:** Our TTFT p95 has been green for a quarter and customers are complaining. Explain how
both can be true, and tell me what SLO you would write instead.

**Model answer:** The mechanism is in the definition of a percentile: **latency percentiles are
computed over *served* requests.** Reject a request and it contributes nothing to the percentile. So
a fleet can meet every latency target it is given by shedding 30% of arrivals — the SLO is not
violated, it is *evaded*, and the fleet looks healthiest exactly when it is failing most users.

The second evasion is subtler and lives in the error taxonomy: a 429 is a "successful" response from
an instrumentation standpoint **unless you count it as an error** `[D]`. Availability SLOs written
as "99.9% success" therefore also fail to catch it — the case study marks availability as
"necessary, insufficient" for exactly this reason.

**What I would write instead** is the case study's third definition: an **attained-throughput SLO** —
"≥16 req/s sustained per deployment, measured hourly and weekly" `[T]` (Singh, *Scaling AI Inference
at NxtGen*), where 16 req/s on half an H100 or half an MI325X is a real customer commitment. The
decisive property is the denominator: it is computed over **demanded** requests, not served ones. It
cannot be met by shedding, by queueing past the window, or by degrading quality invisibly.

The latency and availability SLOs do not disappear; they become **guardrails** on the attained-rate
SLO. You may not attain the rate by violating TTFT, and you may not attain it by returning errors.

The operational test I would apply, and which I would put to a candidate: **can the customer
construct a dashboard that shows your attainment without trusting your telemetry?** If not, the SLO
is not written correctly `[D]`.

**Signal:** States the served-requests-percentile mechanism precisely, names the 429-in-success
evasion separately, and reaches for the demanded-request denominator.

**Follow-ups:**
- *Why does availability fail too?* — the 429 hides in "success"; T15-Q12's taxonomy.
- *What is the goodput relation?* — goodput is the internal efficiency metric, attained rate is the
  contract; T15-Q6.
- *What breaks if demand is not observable?* — no queue, no admission log, no denominator.

**Red flags:** Says "add more SLOs," or proposes tightening the latency percentile as the fix.

---

#### T15-Q13 · Requests per second as a scaling signal — when does it break?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** RPS is simple, it matches the contract's unit, and it is in the corpus's own manifest.
Give me its failure mode.

**Model answer:** RPS ignores **request size**, and that single omission destroys it for any workload
with a non-uniform prompt distribution.

The case study's formulation: RPS "ignores request size — 1 req/s of 60k-token prompts is heavier
than 100 req/s of 200-token prompts" `[D]`. The ratio there is a factor of 100 in RPS terms but the
opposite ordering in actual work. A fleet scaled on RPS will therefore under-provision exactly when
large requests arrive and over-provision when they leave.

The case study gives the concrete instance: **a single 60k-token prompt is heavier than 100 short
ones**, so "any per-request scaling signal is wrong for a bimodal prompt distribution. Scale on
*tokens* or on KV bytes, not on requests" `[D]`.

The corpus's manifest is a useful illustration of the trap in a second way: its RPS trigger is
`averageValue: 100` `[R]` (`11-infrastructure-and-mlops/01-llm-infrastructure.md`) — a round number
with no derivation. Even if RPS were the right signal, that threshold would be unjustified; it is a
placeholder that looks like a decision.

**When RPS is acceptable**, and I would say this rather than dismissing it: **homogeneous, short,
uniform workloads** — the case study's "when to use" column. If every request is a 200-token
classification, RPS is a fine proxy and is cheap to instrument. The signal is wrong for *your*
workload, not wrong universally.

**What I would scale on instead**, in order: tokens per second or KV bytes per second if the
distribution is wide, and queue depth if a flow-control layer exists. The general principle is to
scale on a signal that is *monotone in work*, and request count is only monotone in work when
requests are the same size.

**Signal:** Names request-size variance as the failure, gives the 60k-versus-100-short comparison,
and states the narrow regime where RPS is legitimate.

**Follow-ups:**
- *What is monotone in work here?* — tokens or KV bytes; T15-Q2.
- *Does the corpus's 100 threshold help?* — no derivation attached; T15-Q9.
- *How does this interact with agentic traffic?* — 98% of tokens are prefill `[T]`; T15-Q7.

**Red flags:** Calls RPS useless without naming the size-invariance assumption, or quotes 100
req/s as a default.

---

#### T15-Q14 · We scaled down and our costs went up. Explain.
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Replica count fell by a third last week and the GPU-hour bill fell, but compute cost
per request rose and p95 latency got worse. What happened?

**Model answer:** Scale-down evicted the **KV and prefix caches you had already paid to build**, and
the next request that would have hit them recomputed from scratch.

The case study states it as an edge case: "Removing a replica discards its prefix cache; the next
request that would have hit it recomputes. Scale-down must be prefix-aware, or it converts a memory
saving into a compute cost" `[D]`. The failure-mode table's row is the diagnostic tell: *"Cost falls
then compute rises → cache hit rate after scale-down"* `[D]`.

The mechanism is worth spelling out because it is not obvious from a replica-count graph. Prefix
caching means a request whose prompt shares a prefix with an earlier request skips the prefill for
that span. That saving is *resident in a specific replica's KV cache*. Removing the replica destroys
the saving, not just the capacity. If your traffic is multi-turn or template-heavy — which agentic
traffic is, since agents re-send growing conversation context every turn — a large fraction of your
prefill work is cache hits, and discarding them converts cheap reads into expensive recomputation.

There is a compound effect in agentic workloads specifically: **prefill is ~98% of tokens** `[T]`
(llm-d). So the work you are invalidating is the overwhelmingly dominant cost, and the recomputation
lands on the replicas that remain, raising their KV pressure — which, given the 28k-input cliff
`[T]` (ROCm/WideEP), can push them across the capacity line and cost more than the scale-down saved.

**What I would do.** Make scale-down **prefix-aware** — drain and retire replicas whose cache is
coldest, and avoid cycling the same replica. Add a cooldown long enough that a scale-down cannot be
immediately followed by a scale-up for the same traffic. And watch **cache hit rate after
scale-down events** as a first-class monitor, not just replica count and cost.

**Signal:** Names cache eviction as the cause, ties it to prefix reuse rather than generic caching,
and proposes prefix-aware scale-down rather than reversing the change.

**Follow-ups:**
- *Why is this worse for agentic traffic?* — multi-turn prefix reuse plus 98% prefill `[T]`.
- *What makes scale-down safe?* — draining and choosing the coldest replica; T15-Q15.
- *How does this relate to oscillation?* — the same 15–20 s floor causes both; T15-Q15.

**Red flags:** Attributes it to noisy neighbours or a bad batch config, or reverts the scale-down
without changing the policy.

---

#### T15-Q15 · The replica count is sawtoothing. Diagnose and fix.
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your replica count oscillates: up, over-shoot, down, repeat. It is costing you cold
starts and latency. Walk me through it.

**Model answer:** This is a **control-loop timing mismatch**, and the numbers in this corpus make it
almost inevitable with default settings.

The case study's edge case states it precisely: "With a 15–20 s cold start `[R]` and a 10 s metric
window, a step change produces scale-up, over-shoot, scale-down, and a repeat" `[D]`. The mechanism
is lag: the scaler adds capacity, but the new replicas do not relieve the metric for 15–20 seconds.
Meanwhile the metric window has already averaged in the *pre-relief* period, so the scaler adds more.
When the new replicas finally come up, the accumulated capacity overshoots demand by a wide margin,
the metric collapses, and the scaler removes it — returning to the original state.

Three fixes, in order `[D]`:

1. **Asymmetric windows — fast up, slow down.** Responding quickly to a genuine rise costs little
   (you were going to need it); responding quickly to a fall is what creates the oscillation, because
   the fall is usually the new capacity arriving rather than demand leaving. Kubernetes expresses
   this as `stabilizationWindowSeconds` per direction.

2. **A cooldown at least as long as the start time.** If `T_ready` is 20 s, a cooldown shorter than
   20 s guarantees the scaler will react to its own in-flight action. The case study's failure row
   says "cooldown ≥ start time" `[D]`.

3. **Hysteresis in the thresholds.** Separate scale-up and scale-down thresholds, far enough apart
   that noise cannot cross both.

There is a second-order cost specific to this topic that I would raise: each cycle is a **cold start
and a cache eviction** (T15-Q14). So oscillation is not merely a noisy metric — it repeatedly pays
the 15–20 s warmup and repeatedly discards prefix caches. The bill shows up as worse p95 and higher
compute cost, not as a replica-count graph.

**Signal:** Explains the lag mechanism — metric window shorter than the readiness time — rather than
just naming hysteresis, and connects the cycle to cache loss.

**Follow-ups:**
- *Why must the cooldown exceed `T_ready`?* — otherwise you react to your own action.
- *Why is slow-down right and slow-up wrong?* — the fall is usually your own capacity arriving.
- *What other cost does each cycle carry?* — cache eviction; T15-Q14.

**Red flags:** Says "add hysteresis" with no timing analysis, or proposes a longer window in both
directions, which makes step changes worse.

---

### Capacity arithmetic

#### T15-Q16 · Derive per-GPU capacity from the contract anchor
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You are given "this deployment is rated for 16 requests per second sustained" on half
an H100. Turn that into a per-GPU capacity number, and tell me what you are assuming.

**Model answer:** The anchor is a **contract**, not a benchmark, and the distinction governs how the
number may be used.

Singh's framing: "If I'm committing that I'm going to use half an NVIDIA H100 or half an AMD MI325X
to run maybe a Qwen 3.6 and I'm telling the customer, look, this deployment is rated for 16 requests
per second sustained. I need to make sure that it does that day in day out, otherwise I breach the
SLA" `[T]` (*Scaling AI Inference at NxtGen*). Note the ASR correction: "Quen 3.6/3.8" → **Qwen 3.x**.

The derivation `[D]`, from the case study:

```
Half H100 rated sustained = 16 req/s
Full H100 equivalent      = 32 req/s     (linear scaling assumed)
```

**Every assumption, stated**, because this is where a candidate is tested:

- **Linear scaling from half a GPU to a full one.** The case study flags this explicitly as an
  assumption, not a measurement. Doubling the slice does not reliably double throughput, because
  batch size and KV capacity do not scale linearly with SM count.
- **Model-specific.** It is stated for a Qwen-class model at some context length and token mix. The
  case study's caveat is blunt: "32 req/s per H100 is a *contracted* figure for a specific model,
  context length and token mix. It is not a benchmark for your model." `[D]`
- **It is a floor with penalty attached, not a peak.** "Sustained" means every hour, so the number
  includes the bad hours.
- **Half-GPU means fractional or MIG-style partitioning** `[T]` (Singh), with the KV partition
  contended (T15-Q23).

**What I would do with it:** use it to make the *shape* of the arithmetic concrete (T15-Q17's
4-replica result), then re-derive it by measuring my own deployment before committing procurement.
Quoting it as my capacity plan would be exactly the "no attribution" failure the bank warns about.

**Signal:** Derives the number and then lists the assumptions unprompted, particularly the linearity
assumption and the model-specificity caveat.

**Follow-ups:**
- *Why can't you scale half to full linearly?* — batch and KV do not scale with SMs; T15-Q20.
- *What is the contractual significance of "sustained"?* — measured over the contract's window;
  T15-Q24.
- *How would you re-derive it?* — measure your model at your context distribution; T15-Q17.

**Red flags:** Quotes 32 req/s as a benchmark, or applies it to a different model without re-measuring.

---

#### T15-Q17 · Size the interactive warm floor for a 2× ramp
**Difficulty:** L4 · **Depth expected:** 5 min

**Question:** Demand is 55 req/s at business hours and can double within 30 seconds. Cold start is
15–20 s. Show me the floor, and tell me what it costs.

**Model answer:** I would compute it in two steps and then state the conflict it creates.

**Step one — the ramp requirement.** Demand can reach 110 req/s faster than a replica can become
useful, so the floor must already serve 110. Using the 32 req/s anchor (T15-Q16, with its caveats):

```
Replicas at saturation for 55 req/s   = 55 / 32  = 1.72  -> 2 replicas   [D]
Warm floor for a 2x ramp to 110 req/s = 110 / 32 = 3.44  -> 4 replicas   [D]
Headroom at business hours            = 4 x 32   = 128 req/s
Utilisation at business hours         = 55 / 128 = 43%
```

**Step two — say the conflict out loud.** 43% is below the case study's 55% utilisation target, and
that is the *point*, not a defect. A fleet sized exactly to average demand cannot survive a step
change; a fleet sized for the ramp runs at low average utilisation. **Those two facts are in direct
conflict and the contract is what resolves them** `[D]`. The engineering answer is therefore not
"right-size to 55%" — it is "the floor is four replicas, and the utilisation target must be
renegotiated with procurement."

**What the floor buys, beyond capacity.** Four replicas also keep KV caches warm and spread across
more failure domains. The case study lists "meets ramp requirements; KV stays warm" as the warm
floor's pro `[D]`, and the cache point is load-bearing for agentic traffic where prefix reuse is a
large share of prefill work (T15-Q14).

**Where I would look for a cheaper floor rather than more GPUs.** If the model is small enough to
run on a fractional GPU, a micro-batching floor on a shared slice keeps a floor cheap `[T]` (Singh).
If the workload is session-shaped, session multiplexing moves idle cost from GPUs to storage
(T15-Q25). Both change *what idle costs* rather than how many replicas exist.

**Signal:** Produces the arithmetic, then volunteers the utilisation conflict as a contract
negotiation rather than quietly choosing one side of it.

**Follow-ups:**
- *What if the ramp is 90 seconds, not 30?* — reactive scaling becomes viable; the floor can shrink.
- *How do you find the real ramp distribution?* — measure arrivals before setting the floor.
- *What is the cheapest way to lower the floor?* — change what idle costs; T15-Q25.

**Red flags:** Sizes to 55 req/s, or sizes to 110 without noticing the utilisation consequence.

---

#### T15-Q18 · Close a 12 req/s nightly shortfall without buying a GPU
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** Nightly batch pushes demand to 140 req/s for 40 minutes. With four warm replicas you
are 12 req/s short. Procurement takes a year. What do you do?

**Model answer:** The arithmetic first, from the case study `[D]`:

```
Batch peak demand                     = 140 req/s for 40 min
Interactive at the time (off-hours)   ~= 10 req/s
Surplus capacity with 4 warm replicas  = 128 - 10 = 118 req/s
Deficit                               = 140 - 10 - 118 = 12 req/s
```

**12 req/s is small enough to absorb with axes 1 and 2 — batch re-parameterisation plus routing the
batch band to the small model — without procuring a fifth GPU.** The case study calls this "the
single most valuable output of the model: the correct answer to a 12 req/s shortfall in a fixed fleet
is not a GPU" `[D]`.

My plan, in the case study's ladder order `[D]`:

1. **Re-parameterise the batch replicas** (axis 1). Raise `max_num_batched_tokens` / `max_num_seqs`
   on the batch tier. This trades batch-band latency for throughput, which is free here because the
   batch band's SLO is a *4-hour window*, not a latency percentile.
2. **Route the batch band to the small model** (axis 2). Batch document processing very likely has a
   quality floor a smaller model meets. This is the largest single effect and costs quality on
   borderline items only.
3. **Raise prefix reuse** (axis 2). A nightly re-index job over a document corpus is the ideal
   prefix-cache workload — the same templates and headers recur.
4. **Reserve a time window** (axis 3) if 1–3 are insufficient, rather than relying on priority
   (T15-Q22).

**Why this is right rather than cheap.** The demand is off-hours, when the interactive SLO is
looser, and the batch SLO is a completion deadline rather than a latency bound. Both facts point the
same way: it is correct to degrade the batch band's per-request latency to fit the window. And the
alternative — two more GPUs — is the case study's 46.7 wasted GPU-hours/day (T15-Q19).

**What I would verify afterwards:** that the batch window still completes inside 4 hours, and that
interactive TTFT during the window did not breach. If axis 1 pushed interactive replicas into
tighter batches because they share a pool, I have moved the cost rather than removed it.

**Signal:** Reaches for axes 1–3 before axis 4 and justifies it from the *band's own SLO shape*
(deadline, not latency) rather than from cost alone.

**Follow-ups:**
- *What if the batch needed per-request latency?* — the ladder changes; steps 3–5 stop being free.
- *What does the fifth GPU actually cost?* — 46.7 GPU-hours/day; T15-Q19.
- *What else breaks first?* — the database or network at 256 req/s; T15-Q24.

**Red flags:** Buys the GPU, or closes the gap by degrading the interactive band off-hours.

---

#### T15-Q19 · What does a burst tier actually cost, and when is it worth it?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Your architect proposes two extra replicas that exist only for the nightly window.
Quantify the cost and tell me when you would approve it.

**Model answer:** The case study's arithmetic `[D]`, assuming the fifth and sixth replicas exist only
for the nightly window and cannot be reclaimed:

```
Idle cost = 2 GPUs x 24h - 2 GPUs x (40/60)h = 46.7 GPU-hours/day
Fraction of the day idle = 97.2%
```

So the tier runs idle **97.2% of the day**. Compare that with re-parameterising (T15-Q18): zero
additional GPU-hours, at the cost of some interactive TTFT during the batch window — which is
off-hours, where the SLO is looser.

The case study's conclusion is conditional, not absolute: **the burst tier is worth it only if the
batch window's latency requirement is hard *and* the interactive SLO is tight in the same period.
Off-hours, neither is true** `[D]`.

I would add the framing that makes this a design question rather than an arithmetic one. The
46.7 GPU-hours are not wasted if they buy something specific. Two things they can legitimately buy:

- **Insurance against a mis-forecast batch window.** If the nightly job occasionally runs 3× longer
  or arrives at business hours, resident burst capacity is the difference between a missed deadline
  and a served one. Price that against the contract's batch-band commitment.
- **A hard isolation boundary.** If the batch and interactive bands share a pool, re-parameterising
  the batch tier degrades the interactive tier. Separate resident capacity removes that coupling —
  which is the case study's 10× recommendation, where the batch tier becomes a distinct fleet `[D]`.

The distinction that matters: **the burst tier's cost is not "two GPUs," it is "two GPUs at 97.2%
idle."** Whether that is expensive depends entirely on what failure it prevents. My answer at 1×
Kestrel scale is reject; at 10× with per-agency SLOs, the isolation argument gets much stronger.

**Signal:** Reproduces the 46.7 GPU-hour arithmetic and makes the approval *conditional* on a stated
failure it prevents, rather than approving or rejecting on sticker price.

**Follow-ups:**
- *When does it become clearly right?* — when isolation, not capacity, is the goal; T15-Q28.
- *What is the cheaper version of the same insurance?* — reserved time window; T15-Q22.
- *How does this change at 10×?* — the batch tier separates entirely; T15-Q28.

**Red flags:** Approves it because "capacity is good," or rejects it without asking what failure it
prevents.

---

#### T15-Q20 · Derive the replica count from an SLO, end to end
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** Take the corpus's 20,000-concurrent-session figure. Turn it into a fleet size, and tell
me where the number could be wrong by an order of magnitude.

**Model answer:** Four steps, and I would flag the uncertainty at each.

**Step one — concurrency is a stock; convert it to a flow.** Little's Law: `concurrency = throughput
x latency` `[D]`. With ~10 s per session:

```
20,000 concurrent sessions / 10 s = 2,000 sessions/s                    [D]
```

2,000 sessions/s is what I buy hardware for. The 20,000 figure itself is `[T]` (ROCm/WideEP): the
**~20,000 maximum concurrency on a tuned 1P1D pair** — one prefill pod, one decode pod.

**Step two — the per-replica ceiling is KV, not FLOPs.** `[D]`

```
max_concurrency = HBM_for_KV / (bytes_per_token x avg_context_len)
```

The cheat sheet's worked instance: **40 GB of KV, 327 KB/token, 4k context ⇒ ~30 sequences per
replica** `[D]` (see [T07](T07-kv-cache.md)). Note how much of this is context length: at 32k instead
of 4k, the same 40 GB holds ~4 sequences. **The SLO fixes concurrency; the context distribution
fixes how many replicas that is.**

**Step three — divide, then add headroom.**

```
replicas = ceil(peak_concurrency / max_concurrency_per_replica) x safety_factor
2,000 sessions/s at 30 sessions/replica -> ~67 replicas;  x 1.3-1.5 safety  [D]
```

Peak, not mean — sessions are long-lived and bursty `[T]`.

**Step four — where this is wrong by 10×.** Three candidates, and I would name them honestly `[D]`:
(a) **the context distribution** — 4k versus 32k changes the per-replica count ~8×, and this is the
single largest lever; (b) **the 10 s session duration** — an agent session that runs 40 minutes
changes the flow by 240×, which is why the case study warns that long-lived agent sessions break
naive sizing and naive scale-down `[T]` (Hockin: agents are "idle 99.999% of the time"); (c) **the
1P1D pairing** — the 20,000 figure belongs to a *tuned disaggregated pair*, so applying it to a
colocated deployment is a category error.

And the honest limit: **the corpus does not publish a measured goodput curve or an autoscaling
convergence time** `[R]`, so I would not claim the fleet converges to this size within any particular
window.

**Signal:** Separates stock from flow before dividing, uses the KV ceiling rather than FLOPs, and
independently volunteers that the context distribution is the largest error term.

**Follow-ups:**
- *Why does the context distribution dominate?* — it is linear in `bytes_per_token x ctx`; T07.
- *What if sessions last 40 minutes?* — Little's Law inverts the fleet size; T15-Q25.
- *Is the 20,000 figure a colocated result?* — no; it is a tuned 1P1D pair `[T]`.

**Red flags:** Uses 20,000 as a replica count, or omits the context-length sensitivity entirely.

---

### Degradation, edge cases and failure modes

#### T15-Q21 · Walk the overload ladder. Where do real ladders break?
**Difficulty:** L4 · **Depth expected:** 5 min

**Question:** You are over capacity. Give me your degradation sequence, and tell me the most common
way this goes wrong in production.

**Model answer:** The case study's ladder, with the user-visible cost of each step `[D]`:

| Step | Action | User-visible cost | When it stops helping |
|---|---|---|---|
| 1 | Compress batch settings | Lower throughput per GPU | When the GPU is launch-bound |
| 2 | Raise the small-model routing threshold | Quality on borderline queries | When the tail is genuinely hard |
| 3 | Push best-effort band into the queue | Batch latency | When the batch window is at risk |
| 4 | Degrade best-effort to a smaller model | Visible quality change on batch | When the batch must be accurate |
| 5 | Shed best-effort with 429 + `Retry-After` | Explicit failure | Always helps, always costs trust |
| 6 | Shed premium traffic | Contract breach | Never an acceptable steady state |

**The most common real-world error is a ladder that jumps from step 1 to step 6 because nobody
defined the middle** `[D]`. Steps 2–5 are the whole design; they are the difference between a system
that degrades and one that fails. A ladder with only step 1 and step 6 is not a ladder.

Three things I would insist on beyond the table:

**Write the step 3/5 boundary into the contract.** The case study's chosen design puts the boundary
between queueing the batch and shedding it into the contract, "so that the degradation clause and
the implementation agree" `[D]`. Otherwise the first degraded request becomes a contract dispute.

**Test each step manually before automating it.** A degradation path exercised for the first time
during an incident is not a degradation path.

**Load shedding is a feature, not a failure.** A system that degrades gracefully serves more goodput
than one that accepts everything and misses every SLO `[T]` (the corpus's flow-control framing from
llm-d). Step 5 is a legitimate design choice, priced in trust.

The step that is easiest to get wrong in practice is 3: a permanently deprioritised class never runs
(T15-Q22).

**Signal:** Reproduces the ladder with the *why-it-stops-helping* column, and names the missing-
middle failure rather than reciting steps.

**Follow-ups:**
- *Which step needs a contract clause?* — the 3/5 boundary; T15-Q21.
- *Why does step 3 fail?* — the batch starves; T15-Q22.
- *Is a 429 counted in your SLO?* — only if the taxonomy says so; T15-Q12.

**Red flags:** Jumps from batch tuning to shedding, or presents shedding as an outage rather than a
designed step.

---

#### T15-Q22 · The nightly batch never gets a slot. Fix it.
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Interactive traffic is continuous, so the batch band queued at step 3 never drains. The
nightly deadline is being missed repeatedly. What is the actual fix?

**Model answer:** The fix is to stop relying on **priority** and reserve a **time** window, because
priority alone cannot schedule a class that is permanently outranked.

The case study's edge case: "Batch work deferred under step 3 may never get a window if interactive
traffic is continuous. Reserve a *time* window for batch rather than relying on priority alone" `[D]`.
The failure-mode row names the symptom and the detection: *"Batch starvation → nightly window
deadline missed repeatedly → batch completion time vs deadline"* `[D]`, with the business function as
the blast radius.

The mechanism is straightforward and worth stating: a strict priority scheme is a *work-conserving*
scheduler. As long as any higher-priority work exists, lower-priority work is never dispatched. If
higher-priority work is continuous — and interactive traffic is, by definition, always arriving —
then the batch queue's expected wait is unbounded, not merely long. Priority gives you *ordering*,
not *progress*.

**What I would implement**, in order `[D]`:

1. **A reserved time window.** Interactive demand is off-hours low (the case study's model uses
   ~10 req/s at night). Ring-fence capacity for the batch band in that window.
2. **A guaranteed floor share, not just a priority.** If the batch band holds a percentage of
   replicas it can always use, it makes progress regardless of interactive depth.
3. **A deadline-aware re-queue.** If the batch is projected to miss its 4-hour window, escalate it
   above the interactive band's *queued* (not admitted) traffic — step 4 of the ladder, which is
   available precisely because the batch band's SLO is a deadline rather than a latency bound.
4. **Monitor the deadline, not the queue depth.** The cheat sheet's monitoring guidance is to alert
   on attained rate against contract before latency; here the equivalent is to alert on projected
   completion time against the window.

The generalisable lesson is the case study's own framing: **a permanently deprioritised class never
runs**. Any degradation ladder that defers work needs a matching mechanism that guarantees the
deferred work eventually executes — otherwise step 3 is silent data loss with extra steps.

**Signal:** Explains that priority orders rather than schedules, and proposes a reserved window or a
guaranteed share rather than tuning the priority level.

**Follow-ups:**
- *Why is the wait unbounded rather than long?* — work-conserving scheduler with continuous
  higher-priority arrivals.
- *When does the batch escalate?* — deadline-aware, at step 4; T15-Q21.
- *What changes at 10×?* — the batch tier becomes a separate fleet; T15-Q28.

**Red flags:** Proposes raising the batch band's priority, which is what is already failing.

---

#### T15-Q23 · A multi-node replica is stuck Pending. What is missing?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** You scale out a model that needs 8 GPUs across 2 nodes. Some replicas never start —
pods sit Pending and you are paying for capacity that can never run. Diagnose, and say how confident
you are in the fix.

**Model answer:** The symptom is a **partial allocation**: a multi-node replica has some of its pods
scheduled and others not, and because the replica cannot run until all of them are placed, the
scheduled pods hold GPUs uselessly.

The correction is **gang scheduling** — all-or-nothing placement for the replica's pod group. The
cheat sheet lists it as a **hard requirement for multi-node replicas** — "without it, partial
allocations deadlock and you pay for capacity that can never run" — and the KEDA manifest in the
same sheet carries it as a required configuration, marked `[T]` in the cheat sheet's provenance.

**A provenance caveat I would state, because this bank holds to attribution discipline:** gang
scheduling is asserted as `[T]` in the T15 cheat sheet, but I could not locate the phrase or the
deadlock statement in the transcripts named in the T15 sources — I searched the llm-d, NxtGen and
Hockin transcripts for "gang", "deadlock", "co-schedule" and "Pending" and found nothing on point.
So I would present gang scheduling as **standard Kubernetes practice for multi-node workloads `[D]`**
and say so, rather than attributing to a speaker a claim I cannot locate. The mechanism itself is
sound and independent of provenance: with independent per-pod scheduling, a placement that is
feasible as a whole can be blocked by a partial placement that consumes the resource the rest of the
group needs.

**Why it is worse here than in a generic batch workload.** LLM replicas are large — the case study's
10× scenario speaks of forty-GPU floors — so a single replica can consume a large fraction of a node
or cluster. That makes partial-allocation waste expensive and makes fragmentation likely. It also
compounds with the KV constraint: a replica that never starts contributes zero KV capacity, so the
attained rate falls while the GPU-hours bill does not.

**The diagnosis I would run.** Compare *requested* versus *scheduled* pods per replica; check for
node fragmentation (enough GPUs free in aggregate, none in one place); check whether the scheduler
is placing pods individually. The fix is a pod group with all-or-nothing semantics; the confirming
measurement is that replica count and *ready* replica count converge.

**Signal:** Identifies partial allocation as the failure rather than generic "not enough GPUs," and
volunteers the provenance uncertainty instead of asserting a citation.

**Follow-ups:**
- *Why is aggregate free capacity not enough?* — fragmentation; a replica needs contiguous placement.
- *What does a stuck replica cost?* — GPU-hours with zero KV capacity.
- *How do you confirm the fix?* — ready-replica count converges to replica count.

**Red flags:** Says "add more nodes" or "raise the quota," or cites gang scheduling as a corpus
quote when it cannot be located there.

---

#### T15-Q24 · The fleet is healthy and the pipeline is failing. Why?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Attained rate per deployment is 100% of contract and latency is green. The end-to-end
document pipeline is still missing its 256 pages/second target. Explain, and tell me what you would
change about how the fleet is specified.

**Model answer:** Because **the fleet is one stage of a pipeline, and a pipeline runs at its slowest
stage.**

Singh's framing is the constraint that makes this a systems problem rather than a serving problem:
"your application server needs to handle that 256 requests per second, it would mean you need to
write into your databases at 256 requests per second and inference will need to happen at 256
requests per second. So your network bandwidth needs to support 256 requests per second" `[T]`
(*Scaling AI Inference at NxtGen*). And the constraint generalises: "You need to ensure that your
inference scales at the same rate as your applications, scales at the same rate as your databases"
`[T]`.

The case study's edge case states the consequence: "A fleet rated correctly and a database that
cannot sustain the write rate produces a pipeline that fails at the same point every time. **Test the
pipeline, not the fleet**" `[D]`.

**Why the inference SLO cannot detect it.** The attained-rate SLO is scoped *per deployment* — it
measures whether the fleet delivered its contracted rate. A downstream database bottleneck does not
reduce the fleet's attained rate; it backs up the queue *after* the fleet, or it forces the upstream
stages to throttle before requests ever arrive. Either way the fleet's own metric stays green while
the pipeline's throughput falls. This is the case study's "SLO measured over served requests only"
family of errors, one level up: correct metric, wrong scope.

**What I would change about the specification.** Three things `[D]`:

1. **Specify the pipeline, not the fleet.** A whole-pipeline SLO with a per-stage decomposition,
   so each stage's share of the 256 req/s is explicit and monitorable.
2. **Size each stage against peak, not average** — the case study's note that routing, gateway,
   storage and network are all in the rated path, and that the router tier can become the bottleneck
   at exactly the load where it matters most.
3. **Test end to end under load.** A per-stage benchmark suite cannot find a cross-stage bottleneck;
   only a pipeline test can.

And the honest limit: the corpus gives the 256 req/s target and names the stages, but publishes no
measured per-stage breakdown for this pipeline. I would not assert which stage is the bottleneck
without data.

**Signal:** Names the wrong-scope mechanism — the fleet's SLO is per-deployment and structurally
blind to downstream stages — rather than just saying "check the database."

**Follow-ups:**
- *Why doesn't a fleet-side queue-depth alert catch it?* — the queue may form downstream.
- *Which stage is usually guilty?* — often the database or fabric `[D]`; the corpus does not measure
  it here.
- *What is the correct test?* — end-to-end under load; T15-Q24.

**Red flags:** Says "add more GPUs," or blames the model, without scoping the SLO to the pipeline.

---

#### T15-Q25 · When is a warm floor of GPUs the wrong abstraction entirely?
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** Your agent workload holds sessions open for 40 minutes while the GPU sits idle between
turns. Someone proposes keeping the warm floor. Is the floor still the right model?

**Model answer:** For agent-shaped workloads, the floor is answering the wrong question — the cost
being minimised is not GPU count, it is **what idle costs**.

The framing that opens the door: agents are "idle 99.999% of the time" `[T]` (Hockin, *Is Kubernetes
Good for Agents?*). A 40-minute agent session spends almost all of it waiting on tool calls, network
I/O and model responses elsewhere. Holding a GPU-warm replica for that session means paying GPU-hours
for idle waiting.

The alternative is **session multiplexing**: the session's *state* is parked in storage and the
compute is released, so idle sessions cost storage rather than compute. The case study's decision
table lists it with the honest caveat — it "needs a snapshot/restore substrate, **still pre-production
grade**" `[T]` (Hockin) — and the revisit condition: if such a substrate becomes production-grade,
"the idle floor moves from GPUs-as-idle to sessions-as-suspended, and the cost model changes shape
entirely" `[D]`.

**The related pattern that is shipping.** The serverless agent pattern — a stateless agent loop, a
durable session store, and a sandbox execution tier — is reported by its own presenter as saving "a
lot on infrastructure costs, particularly for agents that are model-bound" `[T]` (Gosia Steinder).
Two attributions matter here: it is **the vendor's own result**, described as "encouraging" rather
than conclusive, and the saving is specifically claimed for *model-bound* agents.

**What I would do today, given the substrate is not production-grade** `[D]`:

1. **Keep the warm floor for the interactive band** — the 15–20 s cold start still binds, and the
   contract still has a sustained-rate clause.
2. **Track in-flight sessions, not QPS.** Long-running requests look idle to a QPS scaler, so
   scale-down must never target a replica holding live sessions `[T]` (Hockin).
3. **Cap session count per replica** so a single agent's 40-minute wait cannot pin capacity that
   the fleet needs.
4. **Price the agent band separately.** Agent sessions and short interactive requests have different
   idle profiles and should not share a floor's economics.

**Signal:** Reframes the question from "how many GPUs" to "what does idle cost," names session
multiplexing with its pre-production caveat, and does not over-claim the vendor result.

**Follow-ups:**
- *What must exist for multiplexing to work?* — snapshot/restore; not production-grade yet `[T]`.
- *Why does QPS-based scale-down fail here?* — in-flight sessions look idle; T15-Q14.
- *Is the serverless pattern proven?* — vendor's own results, model-bound agents only `[T]`.

**Red flags:** Asserts session multiplexing is available now, or scales down replicas with live
sessions because QPS is low.

---

#### T15-Q26 · One long prompt pins KV utilisation. What is the signal telling you?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** KV utilisation on one replica is pinned near its ceiling all afternoon, but aggregate
throughput is fine and the scaler has not fired. Is the signal broken, or is the system?

**Model answer:** Neither is broken; the signal is being read as the wrong kind of quantity. **KV
utilisation is a pressure signal, not a load signal** `[D]`.

The case study's edge case: "A single huge request can hold KV utilisation near its ceiling while
the fleet is under-utilised in throughput terms" `[D]`. Mechanically: a 60k-token prompt consumes a
large fraction of a replica's KV pool on its own. The metric correctly reports "this pod is close to
eviction," and it says nothing about how much demand the fleet is serving. Those are different
questions, and only the second one should drive scale-out.

**What the signal is legitimately telling you**, and what I would do about it `[D]`:

- **That replica will evict and preempt soon.** Preemption mid-stream is a distinct failure — a
  reclaimed request produces a truncated answer that looks like a model failure — so the pressure
  signal should trigger *routing and admission* action, not necessarily scale-out.
- **The context distribution has a long tail.** If one request can pin a replica, the p99 context
  length is what your per-replica ceiling must be computed against, not the mean.
- **That context caps may be the right control.** Capping per-tenant or per-request context is the
  cheap fix; the case study lists capping per-tenant context as the mitigation for the analogous
  fractional-GPU interference case `[D]`.

**Why this is a classic mis-scale.** If you scale out on per-replica KV utilisation alone, you add
capacity that the long prompt does not need (it is one request, and it will not be split across
replicas) and that the *other* traffic was not asking for. You have bought GPUs to hold a cache
entry. The composite design (T15-Q10) prevents this by routing scale-out to queue depth, which
measures demand, and reserving KV utilisation for per-replica action.

**The general rule I would state:** before scaling on any signal, ask whether it is monotone in
*demand*. KV utilisation is monotone in *pressure*, and pressure can come from a single request.

**Signal:** Separates pressure from load and proposes routing/admission action rather than
scale-out, plus the context-cap remedy.

**Follow-ups:**
- *What action does the pressure signal justify?* — routing away, or admission control; not
  necessarily scale-out.
- *What p99 does this imply for sizing?* — size the ceiling from the tail context, not the mean.
- *How does this interact with the 28k cliff?* — pressure at scale becomes recomputation; T15-Q20.

**Red flags:** Proposes scaling out on the pinned metric, or dismisses the signal as a false alarm.

---

### Open design

#### T15-Q27 · Design autoscaling for a fleet whose GPU count cannot change
**Difficulty:** L5 · **Depth expected:** 8–10 min

**Question:** Sovereign datacentre, annual CapEx procurement, no cloud region to burst into, a
contracted sustained rate per agency, and a nightly batch surge. Autoscaling is impossible. Design
the system.

**Model answer:** The reframe is the answer: **the autoscaler's job here is reallocation, not
acquisition.** Singh states the constraint that forces it — "for you to autoscale you need to have
available GPU capacity, which is again very expensive" `[T]` (*Scaling AI Inference at NxtGen*).

**Five things to build, in order.**

**1. Instrument before you scale.** Queue depth, KV utilisation, TTFT/ITL per band, demanded-versus-
served counts, and attained rate. The case study's runbook puts this first for a reason: a week of
this data decides every threshold that follows, and the corpus publishes **no** measured autoscaling
convergence time or goodput curve, so the thresholds cannot be looked up.

**2. Size the warm floor from the measured ramp, not from average demand** (T15-Q5) — and take the
utilisation consequence to procurement rather than hiding it. Then negotiate: the floor *is* the
price of the sustained-rate clause.

**3. Define the SLO as an attained rate over demanded requests** (T15-Q12), with latency percentiles
and an explicit 429-error-budget clause as guardrails. Make the SLO evaluator a real component that
computes attainment continuously, not a dashboard.

**4. Build the four-axis ladder before the autoscaler.** Batch re-parameterisation → prefix reuse →
model-choice routing → band admission → replica count `[D]`. In a fixed fleet, axes 1–3 are the only
genuine elasticity, and they are what closes the 12 req/s nightly gap without a GPU (T15-Q18).

**5. Add the autoscaler last, and only for reallocation between tiers.** In practice this means the
reclaimable batch tier is preemptible by interactive traffic — the one place where "scaling" is real.

**The degradation ladder is part of the design, not an incident response.** Write it down, put the
step-3/step-5 boundary in the contract, and test each step manually before automating (T15-Q21).

**What I would tell the candidate to name as the residual risks:** procurement forecast error is now
the dominant latency in the system (the case study's 10× observation), the warm floor will drift
upward if nobody reviews it quarterly, and a correctly-rated fleet can still miss the pipeline target
if another stage binds (T15-Q24).

**Signal:** Reframes autoscaling as reallocation, orders the work instrumentation → floor → SLO →
ladder → autoscaler, and treats the degradation ladder as a designed component rather than a
runbook artefact.

**Follow-ups:**
- *Where is the only true scaling?* — reclaimable-tier preemption; T15-Q18.
- *What do you negotiate with procurement?* — the utilisation target against ramp survival; T15-Q17.
- *What is the dominant latency at 10×?* — the procurement cycle; T15-Q28.

**Red flags:** Proposes a Kubernetes HPA as the answer, or designs the autoscaler before the SLO.

---

#### T15-Q28 · What changes at 10×, and what survives?
**Difficulty:** L5 · **Depth expected:** 6 min

**Question:** Your fleet grows tenfold. Which of your design decisions invert, and which are
architectural?

**Model answer:** The case study's 10× section is the answer, and its value is that it separates the
two categories explicitly.

**What inverts or becomes mandatory `[D]`:**

- **The routing tier stops being optional.** At ~550 req/s the fleet is well past the **~85 QPS
  crossover** where prefix-aware routing returns **~2× throughput** `[T]` (Singh — note this is a
  *vendor-affiliated* study: NxtGen works closely with NVIDIA and AMD, and the figure is from their
  own test, so I would mark it as a vendor claim and re-measure). At 10× it is mandatory on
  throughput grounds, not only cache grounds. At 1× it is *not yet* justified on throughput — the
  case study's break-even puts the crossover at roughly 10× business-hours load — and is justified
  only on prefix-reuse grounds.
- **The batch tier becomes a separate fleet.** Shared capacity at 10× means perpetual batch
  starvation; dedicated batch GPUs with their own relaxed SLO is the only stable answer.
- **SLOs become per-agency and differentiated.** A single attained-rate SLO across fourteen agencies
  "will be met for the aggregate and breached for the unlucky" `[D]`. Per-tenant isolation in the
  SLO evaluator mirrors the isolation the fleet already needs.
- **The warm floor becomes the cost line worth optimising.** At 1× it is four GPUs; at 10× it is
  forty, and "the difference between a 2× and a 1.5× ramp requirement is ten GPUs — the most
  expensive threshold in the system" `[D]`. That reframes ramp measurement as a financial
  instrument, not a monitoring detail.
- **Procurement lead time becomes the dominant latency.** Demand forecasting becomes an engineering
  discipline with a model and an error budget.
- **At 0.1× the autoscaler is pure overhead** and a hand-set replica count is strictly better. "Do
  not build the machinery you cannot yet justify." `[D]`

**What survives — the architectural claims `[D]`:** the four-axis ladder, KV utilisation as a
pressure signal (never a load signal), attained rate as the SLO, and the warm floor sized by ramp
rather than average. Plus the two spine constants: the **15–20 s cold-boot floor** `[R]` and the
**28k-input KV-recomputation cliff at concurrency 256** `[T]` (ROCm/WideEP), which is a capacity-
planning input at any scale.

**The test I would apply to any design decision here:** is this claim about *GPUs*, or about *how
work is admitted, shaped and measured*? The first kind inverts with scale; the second kind does not.

**Signal:** Separates the invariant architectural claims from the scale-dependent ones, and flags the
85 QPS figure as a vendor-affiliated measurement rather than a neutral benchmark.

**Follow-ups:**
- *Which claim is architectural?* — ramp-sized floor, KV as pressure, attained rate; T15-Q5/Q10/Q12.
- *Why is the 85 QPS figure weaker evidence?* — vendor-affiliated `[T]`; T15-Q16's discipline.
- *What is the most expensive threshold?* — the ramp assumption; it is worth ten GPUs at 10×.

**Red flags:** Says "everything scales linearly," or drops the ramp-sized warm floor as unnecessary
at volume.

---

## Whiteboard exercises

### Exercise 1 — Turn a contract clause into a fleet size
**Prompt.** "Agency contract: 'this deployment is rated for 16 requests per second sustained' on
half an H100. Your platform serves fourteen agencies, business-hours aggregate demand is 55 req/s,
and demand can double within 30 seconds. Cold start is 15–20 s. Show me the fleet, the SLO you would
report, and the number you would take to procurement."

**What the candidate must produce:** the contract-to-capacity derivation with its assumptions
stated, the warm-floor arithmetic, the utilisation conflict surfaced rather than hidden, and an SLO
whose denominator is demanded requests.

**Expected answer sketch:**

```
Contract anchor (Singh [T]; vendor-neutral, but a CONTRACT not a benchmark)
   half H100 rated   = 16 req/s sustained
   full H100 equiv   = 32 req/s          [D]  (linear scaling ASSUMED - say it)

Warm floor (sized by RAMP, not average)
   55 / 32  = 1.72 -> 2 replicas   (serves the average - WRONG answer)
   110 / 32 = 3.44 -> 4 replicas   (survives a 2x step in <20 s)   [D]
   headroom   = 4 x 32 = 128 req/s
   utilisation = 55 / 128 = 43%  <-- below the 55% target, and that is the POINT

Cold-start floor  = 15-20 s (optimised; minutes realistically)      [R] 04/06
   T_ready ~= T_schedule + T_image + T_weights + T_warm             [D]
   => reactive scaling cannot beat a 30 s ramp. Only the floor can.

SLO to report (NOT a latency percentile)
   attained_rate per deployment, >= 16 req/s, over DEMANDED requests, weekly  [T]
   guardrails: TTFT p95, ITL p95, explicit 429 error budget

To procurement
   the floor is the price of the clause; 43% utilisation is a consequence,
   not an over-provisioning error
```

**Grading rubric (full marks requires all four):**
- Derives 32 req/s from the contract and **states the linear-scaling assumption** rather than
  presenting it as measured.
- Sizes the floor from the ramp and volunteers the 43%-versus-55% conflict as a negotiation, not a
  defect.
- Writes the SLO with a **demanded**-request denominator and explains the shedding evasion.
- Names the 15–20 s cold-boot floor as the reason reactive scaling is ruled out.

---

### Exercise 2 — The scaler that never fired, and the one that fired too much
**Prompt.** "Two incidents. (a) A vLLM deployment with a CPU HPA has never scaled; latency has grown
for three weeks. (b) After switching to KV utilisation the replica count now oscillates every few
minutes and p95 is worse than before. Diagnose both and give me the metric set you would actually
ship."

**What the candidate must produce:** a mechanism for each failure, a discriminating check for each,
and a composite signal design with each signal's role stated.

**Expected answer sketch:**

```
(a) CPU HPA never fires
    mechanism: work is on the GPU; CPU tracks tokenisation + request handling
    CORRECTION: scale on KV Cache utilization, not CPU or memory   [R] 04/06
    but GPU util is ALSO wrong (saturated by design)               [D]
    check: replica count vs queue depth (flat count, rising queue)
    note : the corpus ships a CPU HPA manifest that is this anti-pattern [R] 11/01

(b) Oscillation
    mechanism: metric window (10 s) SHORTER than readiness (15-20 s)  [D]
      -> scale up -> overshoot -> metric collapses -> scale down -> repeat
    check: replica-count variance; cold starts per hour
    cost: every cycle also EVICTS prefix cache (cost up, hit rate down)  [D]
    fix : fast-up / slow-down asymmetric windows; cooldown >= T_ready
          hysteresis; do NOT widen both directions

Metric set to ship (composite)
    queue depth   -> drives SCALE-OUT          (per BAND, not aggregate)
    KV util       -> drives per-replica ACTION (pressure, not load)
    TTFT/ITL p95  -> GUARDRAILS only (noisy)
    attained rate -> the SLO metric (demanded-request denominator, weekly)
    goodput       -> internal efficiency, slow loop
    CPU           -> nothing. delete it.

Known limits, stated not hidden
    80% KV / mean-active > 8 are OPERATOR EXAMPLES, not defaults   [T] Pravin
    no measured goodput curve or convergence time in this corpus   [R]
```

**Grading rubric:**
- Gives the CPU mechanism (wrong resource, not misconfiguration) *and* notes GPU utilisation is
  wrong for the opposite reason — both halves.
- Explains oscillation as a **timing** mismatch with both numbers (10 s window, 15–20 s start), not
  as "noisy metrics."
- Connects each oscillation cycle to cache eviction, so the cost is explained and not just the noise.
- Assigns each signal a distinct **role** (scale-out / per-replica action / guardrail / SLO metric)
  and marks the 80% and 8 thresholds as operator examples.

---

### Exercise 3 — Defend the warm floor to procurement
**Prompt.** "Procurement says: 'You are running fourteen GPUs at 43% utilisation. Halve the floor and
we will still meet the average.' Argue your case, and tell me what evidence would change your mind."

**What the candidate must produce:** the ramp argument with numbers, the contract-clause argument,
the cheaper alternatives to more GPUs, and an explicit falsification condition.

**Expected answer sketch:**

```
Their claim
   "43% utilisation means we are over-provisioned"

Why it is wrong, in their language
   floor is sized by the RAMP, not the average                     [D]
   cold start = 15-20 s OPTIMISED, minutes realistically           [R] 04/06
   demand can 2x in 30 s
   => a replica that takes 20 s to become useful cannot help
      with a 30 s step. Only resident capacity can.

   halving the floor -> 2 replicas -> 64 req/s capacity
   vs a 110 req/s step change = 46 req/s of unmet demand,
   which is a BREACH of the sustained-rate clause, not a latency blip

What actually closes a gap WITHOUT a GPU (offer these first)     [D]
   1. batch re-parameterisation   (instant, free, reversible)
   2. prefix reuse                (multi-turn + template traffic)
   3. route batch band to smaller model
   4. band admission / deferral
   worked: nightly 12 req/s deficit closed at steps 1-2, no 5th GPU

Utilisation is the WRONG target here
   a fleet sized to average cannot survive a step change;
   a fleet sized to the ramp runs at low average utilisation.
   Both facts are true. The CONTRACT resolves them - renegotiate it.

What would change my mind
   measured ramp: if the 99th-pct 30 s ramp is actually 1.2x, the
   floor drops. Show me the arrival distribution over 90 days.
   I will re-derive the floor quarterly (drift: the floor becomes
   the peak if nobody reviews it).                              [D]
```

**Grading rubric:**
- Puts the **ramp** at the centre of the argument and pairs it with the cold-start number — not
  utilisation percentages alone.
- Converts the argument into contract language (a breach of the sustained-rate clause), because that
  is the currency procurement responds to.
- Offers the four no-GPU alternatives **before** defending the hardware, showing the floor is the
  residual and not the first resort.
- Commits to a **falsification condition** — a measured ramp distribution — and a review cadence for
  floor drift.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
  — Abhishek Singh (NxtGen). The 16 req/s contracted sustained rate on half an H100/MI325X; the
  256 req/s end-to-end pipeline across application, database, inference and network; "autoscaling
  when it comes to models is a little touchy… for you to autoscale you need to have available GPU
  capacity which is again very expensive"; KEDA signals for autoscaling; KV cache utilisation as a
  deployment hyperparameter with the ~90%-of-VRAM example; the ~85 QPS crossover showing double the
  throughput, ₹10 lakh → ₹5 lakh/month at ~200 users on two cards; fractional GPUs and shared VRAM;
  and "inference scales at the same rate as your applications, scales at the same rate as your
  databases."
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — Pravin. The workload-variant autoscaler described as "saturation based autoscaling"; the
  operator-defined saturation examples (KV cache 80% full; mean active requests > 8); flow control,
  priority bands and admission; prefill at ~98% of agentic tokens; and the least-attained-service
  fairness result cutting request latencies by up to 2× and sometimes 3×.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the ~20,000 maximum concurrency on a tuned 1P1D pair, and the 28k-input KV-recomputation cliff
  producing a throughput drop at concurrency 256.
- `refs/Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt`
  — agents "idle 99.999% of the time"; sandbox cold-start times ("seconds usually", 10–15 s with a
  browser/runtime); warm-pool behaviour; and the explicitly aspirational Agent Substrate wake target
  (low three-digit ms, 10⁴ activations/s, up to 200,000 nodes).
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — the serverless agent pattern (stateless agent loop, durable session store, sandbox execution
  tier) with the presenter's own claimed infrastructure-cost savings, specifically for model-bound
  agents.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md` `[R]`
  — the KV-cache-utilisation autoscaling rule ("rather than CPU or standard memory usage") and the
  15–20 s cold-boot figure with un-quantized base images and a high-speed Lustre mount.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md` `[R]`
  — the shipped `HorizontalPodAutoscaler` manifest whose metrics are CPU at 70% utilisation plus
  `requests_per_second` at `averageValue: 100`, which contradicts the guide's own autoscaling rule
  and is the subject of T15-Q9.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
  — continuous batching and the throughput/latency frontier underlying the goodput argument in
  T15-Q6.
- Companion cheat sheet (in this knowledge base, not `refs/`): [T07 — KV cache](../00-cheat-sheets/T07-kv-cache.md)
  for `bytes_per_token`, used in the KV-bound concurrency ceiling.

**Derived content in this bank (`[D]`):** the contract-to-capacity derivation and its linear-scaling
assumption (T15-Q16); the warm-floor arithmetic and the 43%-versus-55% utilisation conflict
(T15-Q17); the nightly 12 req/s deficit closure (T15-Q18); the 46.7 GPU-hours/day and 97.2% idle
burst-tier cost (T15-Q19); the Little's Law conversion of the 20,000-session figure to 2,000
sessions/s and the ~67-replica result (T15-Q20); the pressure-versus-load framing for KV utilisation
(T15-Q10, Q26); the queue-depth-is-silent-under-shedding mechanism (T15-Q11); the oscillation timing
analysis and the cooldown ≥ `T_ready` rule (T15-Q15); the cache-eviction chain from scale-down
(T15-Q14); and the ordering of the four scaling axes. The **four scaling axes**, the **overload
ladder** and the **`T_ready` budget** are the case study's own derivations and are re-attributed here
rather than claimed as mine.

**Attribution and ASR notes.** ASR corrections applied, following the case study's table: "VLM" →
**vLLM**; "LLMD"/"LMD" → **llm-d**; "kada" → **KEDA**; "ASLA" → **SLA**; "on-remise" → **on-premise**;
"Quen 3.6/3.8" → **Qwen 3.x**. One further correction I make here: the NxtGen transcript's "Saram"
at ~[15:33] is almost certainly **Sarvam** (Sarvam AI), rendered here as such. One provenance
discrepancy I could not resolve and therefore do not paper over: **gang scheduling is marked `[T]`
in the T15 cheat sheet, but the phrase and its deadlock claim do not appear in the llm-d, NxtGen or
Hockin transcripts**, which I searched for "gang", "deadlock", "co-schedule" and "Pending". I present
it in T15-Q23 as standard Kubernetes practice `[D]` and say so at the point of use. The ~85 QPS /
2× / ₹10 lakh → ₹5 lakh figures are attributed to Singh but flagged as a **vendor-affiliated study**
(NxtGen states it works closely with NVIDIA and AMD), and the serverless-agent cost savings are
flagged as the **vendor's own reported results**. The corpus measures **no** goodput curve, no
autoscaling convergence time and no warm-floor utilisation figure; none is asserted here.
