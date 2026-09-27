# Interview Bank: Routing, Gateways & Semantic Dispatch

> `T14` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T14-routing-gateways.md) · [Case study](../01-case-studies/T14-routing-gateways.md) · [Design blueprint](../03-design-blueprints/T14-routing-gateways/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** progressing from layer separation, through replica and model selection, to gateway reliability and open design

## How to use this bank

The questions are ordered so the bank reads as one interview: first the four routing layers and the
mistake each one prevents, then the two routing layers you operate (replica selection, then model
selection), then the gateway that carries fallback and governance, then cost leverage and diagnosis,
and finally open design on agentic traffic. Ask them in order for a 45-minute loop, or sample the
L4/L5 block for a senior screen.

Every number here is traceable to a transcript (`[T]`, speaker named), a supporting repo (`[R]`,
path named), or is my own derivation (`[D]`, assumptions shown). Two of the corpus's most quotable
figures — "cuts total spend by half" and the llm-d crossover — are claims made under stated
conditions, not laws; a candidate who repeats them without the conditions does not own them.

The single most common failure this bank is designed to detect is **collapsing the layers**: using
one router to answer "which model", "which replica" and "which expert" at once. Candidates who
separate them cleanly and then say which layer each number belongs to are the ones to hire.

---

### Fundamentals — the four layers, and the mistake each one prevents

#### T14-Q1 · Name the four routing layers and the question each answers
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** "Routing" is used to mean five different things in this building. Lay out the routing
layers, name what each one decides, and tell me which ones an operator actually controls.

**Model answer:** Four layers, and they answer four different questions `[T]`:

- **L0 — the gateway / control plane.** *Which provider, and what happens when it fails.* Unified
  API, auth, virtual keys and budgets, rate-limit handling, fallback chains. This is where
  "N providers × M concerns" glue in application code becomes a single policy-enforced choke point `[R]`.
- **L1 — qualitative routing.** *Which model.* Is this prompt easy or hard, code or prose, a
  knowledge lookup or a piece of reasoning? This is the vLLM semantic router's territory — signals,
  partition, difficulty score, algorithm, model decision `[T]`.
- **L2 — performance routing.** *Which replica.* Given a request already labelled with a model, which
  pod of that model holds the prefix and has capacity? This is llm-d's EPP.
- **L3 — in-engine routing.** *Which expert.* The MoE router inside the language model, per token.
  **Not operator-controlled** `[T]`.

The layering is the whole point, and the corpus states the boundary explicitly: llm-d "is a
performance-oriented router" and is "not concerned with the difference in the qualitative
performance of these models", while semantic routing "can perhaps help with picking the right model
for the right task" `[T]` Singh. And at the bottom: "MoE routing is not done at the level of llm-d —
it's more at the LM level" `[T]` Pravin.

So an operator controls L0, L1 and L2 — and each one consumes *different inputs*. L1 reads content.
L2 reads load and KV-cache state. L0 reads identity, quota and provider health. A single component
that tried to read all three would be a monolith with three failure modes.

**Signal:** Separates the layers by the *question* rather than by product name, and volunteers that
L3 is out of scope before being asked.

**Follow-ups:**
- *Which layer does the KV cache affect?* — L2 only; L1 is content-blind to it.
- *Why can't L1 and L2 be one component?* — different inputs, different failure cadence; see T14-Q4.
- *Where does the residency constraint live?* — L0 policy, as a filter; see T14-Q21.

**Red flags:** Lists four products rather than four decisions, or places model choice and replica
choice in the same box.

---

#### T14-Q2 · Expert routing is not yours. Explain the boundary.
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** A product manager asks you to "route easy prompts to cheaper experts to save money."
What do you tell them?

**Model answer:** I would say the request is architecturally incoherent as stated, and then redirect
it to where the goal is achievable.

The incoherence is precise: **MoE routing happens inside the language model, not at the serving
router** `[T]`. Expert selection is computed inside each MoE layer from the token representations
themselves, layer by layer, at inference time. llm-d's own guidance is explicit — "MoE routing is not
done at the level of llm-d. So it's more at the LM level" `[T]` Pravin. There is no interface to it,
which is why L3 is drawn on the architecture diagram purely to name the boundary.

It is worth explaining *why* no such interface exists, because the reason is the same one that makes
MoE work at all: the router's decision is a function of hidden states the gateway never sees. The
gateway has text; the expert choice is made on internal representations deep in the stack. Even with
a patch, you would be overriding a learned function with a heuristic.

The second half of the answer is that the idea also would not save money. Experts are not "cheaper"
or "more expensive" in a meaningful sense — every token routes through the same weights on the same
GPUs regardless of which expert is selected. The cost is in the dispatch and combine all-to-all and
in the GEMM shape, not in the expert's identity.

What the platform *can* do is exactly what L1 exists for: pick a **model tier** for an easy request.
That is a real and achievable cost lever — "route the easy majority of queries to a small model and
save the big one for the hard ones" `[T]`. The redirection, not the refusal, is the useful answer.

**Signal:** Names L3 as inside the model and outside operator control, and offers the L1 alternative
rather than stopping at "no."

**Follow-ups:**
- *Would a cheaper expert save anything?* — no; cost is the collective and the GEMM.
- *What is the achievable version?* — model-tier routing at L1; see T14-Q14.
- *Does the operator control parallelism inside the engine?* — yes, and it is a separate decision
  from routing `[T]`.

**Red flags:** Agrees to build expert steering, or refuses without naming the L1 lever that does
achieve the goal.

---

#### T14-Q3 · What does the gateway actually do, and what does it cost you?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Justify the L0 gateway. Give me the job list and the price.

**Model answer:** The gateway is a control plane between applications and models that exposes "one
consistent (almost always OpenAI-compatible) API and centralizing the cross-cutting concerns that
would otherwise be smeared across every service" `[R]`. Its jobs `[R]`:

- **Unified API** across providers, so client SDKs never change
- **Fallback chains** — an ordered list of provider/model alternates on a retryable failure
- **Load balancing** across keys, regions and providers
- **Retries** with exponential backoff, jitter, and honoured `Retry-After`
- **Rate-limit handling** — detecting 429s, cooling down throttled deployments, rerouting
- **Virtual keys with dollar budgets** and hard caps
- **Spend attribution** per key, team and model
- **Observability** — who called, what was tried, why it failed, what won
- **Caching** — exact-match and semantic
- **Guardrails / PII** filtering at the choke point

The mental model is the one to keep: it converts an `N providers × M concerns` glue problem into a
single policy-enforced choke point.

The price is threefold. First, **one extra network hop** — modest against 500–2,000 ms of model
latency, but real under load, and mitigated by co-locating the gateway with the app `[R]`. Second,
**you now own a critical-path component**: the guide's warning is that a gateway which centralises
everything but runs as one instance has simply *moved* your single point of failure. Third,
**governance config accumulates** — LiteLLM's YAML "strains at enterprise-governance scale" `[R]`,
which is a real operational ceiling, not a jab at one tool.

The honest counterweight: for a single provider and a prototype, this is overkill, and a thin
in-app abstraction is the right answer `[R]`. The gateway earns its place at the second team, not
the first.

**Signal:** Gives the job list as *cross-cutting concerns centralised*, and volunteers both the hop
and the SPOF before being asked about either.

**Follow-ups:**
- *When is it overkill?* — one provider, one team, a wrapper that fits in your head `[R]`.
- *What is the gateway's own SLO?* — the case study target is ≥99.9%, and non-load-bearing until
  proven.
- *Why must the state be externalised?* — counters and keys must survive a replica restart `[R]`.

**Red flags:** Describes the gateway as "a load balancer for LLMs," or adopts one without pricing
the HA burden it creates.

---

### Layer 2 — performance routing: which replica

#### T14-Q4 · What is the EPP, and what are filter, score and rank?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** llm-d's router is called the EPP. Decompose it. What are the three stages doing?

**Model answer:** The **EPP** is the **Endpoint Picker** — "a single deployment that sits between the
vLLM pods and attaches to the gateway" `[T]` Pravin. It hooks in through the Kubernetes Gateway API
Inference Extension, and the transcript renders its hook point as an "external proxy plugin" — which
is the Envoy external-processing (`ext_proc`) extension point, ASR-garbled in the source `[T]`.
Practically: the gateway (GKE's, Istio's, whatever you run) hands the EPP the candidate endpoints,
and the EPP returns the pick `[T]`.

Selection is a three-stage pipeline with **composable** filters and scorers `[T]`:

| Stage | Job | Named policies |
|---|---|---|
| **Filter** | Narrows the candidate set | prefill filter (prefill nodes only); prefix-cache-identity filter |
| **Score** | Ranks the survivors | token-load scorer; active-request scorer; prefix affinity |
| **Rank** | Picks one | "where to route based on the token load or the active request a particular instance is serving" |

The worked example from the talk is the clearest way to hold it: a **prefill** request goes
prefill-filter → prefix-cache-identity filter → token-load scorer; a **decode** request needs only
the active-request scorer, "because they pull the KV cache from the prefill" `[T]`. Same pipeline,
different policy composition per phase — which is why the composability is the design, not the
individual scorers.

Two things a strong candidate adds. First, **routing is only half the EPP's job** — the other half is
flow control, and they are separate concerns: routing is *placement*, flow control is *admission* `[T]`.
Second, L2 operates on a request that L1 has already labelled with a model; the EPP picks a replica
*of that model*, not a model. If it ever picks a different model, the layering has collapsed.

**Signal:** Names the three stages in order and gives the prefill/decode policy difference as the
worked example, rather than listing scorers abstractly.

**Follow-ups:**
- *Why does decode need fewer stages?* — the KV cache arrives from prefill; only load matters `[T]`.
- *What is the EPP's second responsibility?* — flow control and admission; see T14-Q7.
- *Where does the filter matter most?* — hard policy, not preference; see T14-Q21.

**Red flags:** Describes the EPP as a load balancer, or cannot say what distinguishes a filter from
a scorer.

---

#### T14-Q5 · Precise KV-event routing versus approximate hash routing
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Two ways to do prefix-aware routing: hash the prompt, or consume KV events. Explain the
difference and tell me which you would ship.

**Model answer:** Both attempt to answer "which replica already holds this prefix?", and they differ
in whether they *know* or *guess*.

**Approximate / hash routing** hashes the request (or its prefix) and maps the hash to an instance.
It needs no state from the engine, which is why it is easy. The corpus's judgement on it is blunt:
it "doesn't go there, which is mainly based on the hash, and that has all the consistency
problems" `[T]` Pravin.

The consistency problem is concrete `[D]`: a hash tells you where a prefix *would* be if the cache
were perfect and the replica set were fixed. Neither holds. Evictions happen constantly, and a
rolling deploy changes the replica set so every hash re-maps at once. The cheat sheet's failure
signature for it is exact: "prefixes route wrong after a restart" and "traffic piles onto one
replica."

**Precise / KV-event routing** is driven by per-request block events. Every vLLM pod emits "events on
when a KV cache is created and also evicted" `[T]` Pravin — "for every request we get the KV events"
`[T]` — and the EPP maintains a live view of where each block actually resides. It answers the real
question rather than a proxy for it. The contributor's report is that "for the precise we have
significant improvements," and the GLM 5.x deployment study on H100/H200 is cited as showing precise
KV-cache affinity beating approximate `[T]`.

I would ship precise, and I would say out loud what it costs: the router becomes a **stateful
consumer of an event stream**, and the failure is silent. If the event stream stops, L2 degrades into
a load balancer and nothing errors — cache hit rate just falls and cost rises. So precise routing is
only shippable with an alert on the event rate, and with a documented fallback (approximate plus
sticky sessions) decided in advance.

**Signal:** Frames it as know-versus-guess, names the hash consistency failure concretely, and
volunteers the silent-degradation risk of the precise path.

**Follow-ups:**
- *What breaks approximate routing?* — eviction and replica churn; hashes do not re-converge.
- *What breaks precise routing?* — a dead event stream; it degrades silently.
- *How would you detect that?* — alert on KV event rate versus request rate; see T14-Q26.

**Red flags:** Calls hashing "good enough" without naming the consistency failure, or ships precise
routing with no health signal on the event plane.

---

#### T14-Q6 · What does the KV event plane carry, and what happens when it dies?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Draw me the arrow nobody draws: the one from the engine back to the router. What
travels on it, what does the router do with it, and what is the blast radius when it stops?

**Model answer:** This is the load-bearing integration of the whole topic, and it is a cross-team
interface rather than a configuration flag.

**What travels.** Every vLLM pod emits, per request, events on **KV cache block creation and
eviction** `[T]` Pravin. The EPP maintains a distributed view — "a live view of where the cache is
located" `[T]` — and that view is what makes the prefix-cache-identity **filter** and the
prefix-affinity scorer meaningful. Without it, L2 has no idea where anything is.

**What the router does with it.** It uses the view to filter candidates to replicas that actually
hold the prefix, then scores those survivors on load. Note the ordering consequence: the event plane
feeds the *filter*, so a stale view does not merely mis-score — it removes the correct replica from
the candidate set entirely.

**Blast radius when it stops.** The cheat sheet's signature is precise: "cache hit rate falls; cost
rises, latency flat," detected by "KV event rate vs request rate," mitigated by alerting on event
rate = 0 `[D]`. That flat-latency, rising-cost shape is the tell — the service looks healthy on every
latency dashboard while quietly paying fresh-token prices for work it already did. This is exactly
the "cost up, latency flat, quality flat" incident in the runbook, and the first check is the model-
choice distribution for L1 drift and the KV event rate for L2.

**Two operational realities.** First, **restarts leave gaps**: the router's world view is rebuilt
from events, so a mid-stream restart means it should start pessimistic — route on load until the
stream refills — rather than trusting a half-built view. Second, this is a **cross-team dependency**:
the engine team owns emitting the events, the platform team owns consuming them, and the corpus is
explicit that cache-aware routing requires the engine to talk to the router. Budget the integration;
it is not a library import.

Worth noting the direction of travel: the roadmap adds **session metadata on KV blocks** and a
KV-cache retention API, so a router can eventually reason about sessions rather than blocks `[T]`.

**Signal:** Draws the arrow in the right direction (engine → router, not the reverse), places the
events in the *filter* rather than the scorer, and names flat-latency-rising-cost as the signature.

**Follow-ups:**
- *Which stage do the events feed?* — the prefix-cache-identity filter.
- *What is the symptom?* — hit rate down, cost up, latency flat; see T14-Q26.
- *What is the roadmap direction?* — session metadata on KV blocks; see T14-Q27.

**Red flags:** Treats the event plane as an implementation detail of the router, or cannot say what
the router does when the events stop.

---

#### T14-Q7 · Flow control, and the saturation signal that is yours to define
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** The EPP does placement. What else does it do, and how does it know when to stop
placing?

**Model answer:** It does **flow control**, and the corpus is emphatic that this is a separate
responsibility from routing: routing is placement, flow control is admission `[T]` Pravin. Once
placement has decided where a request *could* go, flow control decides whether to dispatch it at all.

**The mechanism.** The EPP keeps a queue. When saturation is detected it stops dispatching into the
fleet and admits from the queue according to a policy you choose `[T]`.

**The critical design point: the saturation threshold is the operator's, not a default.** "The flow
control mainly comes into the picture when your cluster is saturated, and the saturation mechanism
is decided by you" `[T]`. The corpus's two examples are "when the KV cache is 80% full I declare the
cluster saturated, or the number of active requests a particular cluster is seeing on average is
more than eight" `[T]` Pravin. Those are illustrations of the *kind* of rule, not settings — a
candidate who quotes 80% and 8 as values rather than as examples has missed the point, and the
cheat sheet flags exactly this: "they are examples, not law — pick from your own goodput curve."

**Why route on pressure rather than queue depth.** Queue depth tells you how much work is waiting;
KV pressure tells you how close you are to the cliff where the cache stops holding and work gets
recomputed. The cheat sheet's failure signature for the naive version is "good routing on average,
terrible P99 — no saturation signal." Pressure is the leading indicator; queue depth is the lagging
one.

**The implementation shape.** The talk's own decomposition is request handling → flow control (which
maintains the queue) → request scheduling (which picks the instance) → a data layer holding cache and
saturation state `[T]`. That is a real pipeline with a real state store, which is why the EPP is a
deployment rather than a filter.

**Signal:** Separates admission from placement unprompted, and treats 80%/8 as *examples of the
pattern* rather than as recommended defaults.

**Follow-ups:**
- *Why not just use queue depth?* — it is lagging; KV pressure is leading.
- *How do you choose your threshold?* — from measured goodput, not from a blog post.
- *What happens to queued requests?* — that is the queue policy; see T14-Q8.

**Red flags:** Recites 80% and 8 as the correct settings, or conflates flow control with load
balancing.

---

#### T14-Q8 · FCFS "adds nothing." What replaces it?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You have a saturation signal and a queue of admitted work. What is your queue policy,
and what is wrong with the obvious answer?

**Model answer:** The obvious answer is first-come-first-serve, and the corpus rejects it in one
line: FCFS "doesn't add anything extra to your policies because it's going to slow down everything"
`[T]` Pravin. What replaces it is **priority bands**.

**Why FCFS is empty.** It is not that FCFS is unfair — it is that FCFS is not a *policy*. It contains
no statement about what the business values, so under saturation it degrades everything uniformly.
An interactive customer waiting behind a batch job experiences the same queue as the batch job, and
the system has no mechanism to prefer either.

**What priority bands do.** The corpus's description: "a premium or a best-effort where, under such
saturation, only the premium traffic is dispatched and the rest are not dispatched until you have
capacity" `[T]`. The stated purpose is to make sure "customer-facing or interactive workloads don't
suffer while your batch processing workloads can wait and then retry later" `[T]`.

Note the important second-order effect: **the non-premium band is not dropped, it is deferred.** That
distinction matters because it determines what the batch tier's contract is — its requests will
eventually run, at unknown latency, which is a very different promise from a 429.

**What a strong answer adds.** First, the band assignment must happen **at the edge, not in the
router** — the failure signature is "batch traffic starves interactive," and the first check is
"validate the band assignment at the edge." A router that classifies bands itself is a second
classifier to keep calibrated. Second, the band scheme fails if everything is labelled premium, which
is a political failure as much as a technical one. Third, the case study's chosen ordering is
priority bands *first* — cheap and immediate — then elastic scale-out, with explicit 429 shedding
only for the non-interactive band.

**Signal:** Says FCFS is "not a policy" rather than "unfair," and names the defer-versus-drop
distinction for the batch band.

**Follow-ups:**
- *Where is the band assigned?* — at the edge; a mislabelled band is a policy failure.
- *What is the other overload lever?* — elastic scale-out, with the caveat that autoscaling needs
  available GPU capacity `[T]`.
- *Whom do you shed?* — only the batch band, and only once its SLA is negotiated.

**Red flags:** Ships FCFS and calls it a policy, or proposes dropping all non-premium traffic without
stating the contract change.

---

#### T14-Q9 · Least-attained service and the starvation hazard
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** You have three agent sessions running. One is enormous and two are short. Walk me
through what happens without a fairness mechanism, and what the mechanism is.

**Model answer:** Without one, the short sessions **starve**, and the corpus describes the hazard
concretely: "there is a small session and a [large] session that comes in and then takes all the
dispatch cycles. So the shorter sessions are starving now" `[T]` Pravin.

State the direction carefully, because it is easy to invert: **the big session is the monopoliser.**
It is not that short sessions crowd out long ones. A large agentic session generates a large amount
of in-flight work, and a scheduler with no fairness signal will keep feeding it — so the sessions
that would have completed quickly never get dispatched.

**The mechanism is least-attained service** — the corpus also calls it "agentic program-aware
fairness" `[T]`. Sessions are placed in **different queues**, and dispatch prioritises by **attained
service**: the session that has consumed the least so far goes next. That is the fairness signal, and
it is per-*session*, not per-request — which is the design decision that matters, because a
per-request scheduler cannot see that one session has already consumed thousands of requests.

**The measured effect.** The team reports request latencies reduced "by up to 2× or sometimes 3×"
with least-attained service `[T]` Pravin. Read the detail carefully, because it is the interesting
part: "the overall token throughput increased a bit but the request latencies came down by a lot" `[T]`.
That is exactly the shape you would predict — shortest-job-first disciplines improve completion time
without much changing aggregate throughput — and a candidate who reports a 3× throughput claim has
misread the source.

The corpus also marks the boundary of the two fairness strategies: least-attained service addresses
**compute** saturation; turn priority addresses **KV cache** saturation; integrating them is stated
work in progress `[T]`. A candidate who knows which saturation each one targets has the mechanism,
not the vocabulary.

**Signal:** Gets the direction right (the big session monopolises), identifies attained service as
the *per-session* signal, and reports 2–3× as **latency** rather than throughput.

**Follow-ups:**
- *Why per-session rather than per-request?* — only a session view reveals accumulated consumption.
- *Which saturation does it fix?* — compute; turn priority handles KV; see T14-Q10.
- *What does the fairness cost?* — strict FIFO guarantees, which some workloads genuinely need.

**Red flags:** Says short sessions starve long ones, reports the 2–3× as a throughput gain, or
describes fairness as a per-request property.

---

#### T14-Q10 · Turn priority — the opposite fairness strategy
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** The team shipped a second fairness strategy that inverts the intuition of the first.
Explain it, and explain why both can be right.

**Model answer:** The second strategy is **turn priority**, and its intuition is that a session deep
into its work is *cheap to finish*. The corpus: "at turn 100 most probably it is likely to finish
faster than the other ones because it's going for a longer time. So if you prioritize those agents,
those agents will finish and then evict their KV cache" `[T]` Pravin.

So where least-attained service protects the *short* session by prioritising the least-served,
turn priority protects the *long* session by prioritising the nearly-done — and the payoff mechanism
is not the session's own latency, it is the **KV cache it frees**. Finishing a long session evicts a
large KV footprint, which makes room for everything else. The reported effect is throughput
improvement specifically "when there is a KV cache saturation happening" `[T]`.

**Why both can be right — the resolution is which resource is saturated.** The corpus states it
directly: least-attained service addresses **compute** saturation, turn priority addresses **KV
cache** saturation, and the team is working on integrating them `[T]`. That is the whole answer:

| Saturated resource | Right strategy | Why |
|---|---|---|
| Compute / dispatch cycles | least-attained service | share the dispatch fairly across sessions |
| KV cache capacity | turn priority | finish and evict the biggest resident footprint |

The generalisable lesson, and the reason this is a good interview question: **"fairness" is not one
policy.** A scheduling discipline is fair or unfair *with respect to a constrained resource*. Choose
the discipline by naming the bottleneck first, and expect to need both in a real system, since a
fleet under load is usually near both limits in different phases.

Note the honest limit: the integration of the two is stated as in-progress work `[T]`, not as a
shipped feature you can turn on. A candidate should not claim a combined policy exists.

**Signal:** Gets the mechanism right (finish-and-evict, not session-priority-for-its-own-sake), and
resolves the apparent contradiction by naming which resource is saturated.

**Follow-ups:**
- *What frees KV fastest?* — completing a long session, which evicts its whole footprint.
- *Are the two strategies integrated today?* — no; stated as work in progress `[T]`.
- *How would you choose at runtime?* — instrument which resource is at its limit; see T14-Q11.

**Red flags:** Treats turn priority as "favour the big session," or claims a combined policy is
available.

---

#### T14-Q11 · Design the overload policy for a mixed fleet
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** You run one inference fleet serving three traffic classes: interactive customer chat,
internal agent sessions with long tool-call pauses, and nightly batch summarisation. The fleet
saturates most evenings. Design the admission and queueing policy end to end, and tell me what you
would instrument.

**Model answer:** I would structure this as: define saturation, define bands, define the discipline
per band, define degradation, then define the instrumentation that tells me whether any of it worked.

**1. Define saturation, from my own goodput curve.** Not from the corpus's examples. The corpus's
KV-80%-full and mean-active-requests-over-8 are explicitly illustrations of the *kind* of rule and
"the saturation mechanism is decided by you" `[T]`. So I sweep load at my real context distribution,
find where goodput per GPU stops rising, and set the threshold below it with hysteresis — the
"routing flap" failure signature is metrics too noisy or too slow, and hysteresis is the fix `[D]`.

**2. Three bands, assigned at the edge.** Interactive = premium. Nightly batch = best-effort. The
internal agent sessions are the interesting middle, and the tool-call pause is what makes them
interesting: a session waiting on a tool call is consuming **no compute** but holding **KV cache** `[T]`.
That suggests a third band whose policy is not queue position but **KV residency** — the corpus's
roadmap direction is exactly this, with a KV-cache retention API and session metadata letting the
router evict a paused session's cache to CPU and re-fetch on return. The reported effect of CPU KV
offload is ~5× TTFT improvement on the turn where a paused agent session comes back `[T]`.

**3. Discipline per band.** Under compute saturation, **least-attained service** across agent
sessions, because the starvation hazard is real and the measured payoff is 2–3× on request latency `[T]`.
Under KV saturation, **turn priority**, because finishing a nearly-done session evicts its footprint
`[T]`. Premium interactive traffic is dispatched first in both regimes; batch is **deferred, not
dropped** `[T]`.

**4. Degradation order.** Explicit, tested, and ordered: another replica → a smaller model → a cached
answer `[D]`. Degrading to a smaller model is legitimate here *because* the band already encodes the
quality contract; doing it silently for premium traffic would break trust.

**5. Instrumentation.** The routing decision distribution as the primary dashboard — which band,
which model, which replica, at what confidence, at what cost. Then: KV event rate versus request
rate (L2's health), cache hit rate per pool, routing overhead p50/p99 per layer, queue depth per
band, and 429s by band. For agentic traffic specifically, the corpus argues the headline metrics
change: request latency, **session completion time**, and KV cache hit rate per session matter more
than TTFT and ITL, which are interactivity metrics `[T]`.

**Signal:** Defines saturation from measurement rather than quoting the corpus's examples, identifies
the paused-agent KV-residency problem unprompted, and matches each discipline to the resource it
protects.

**Follow-ups:**
- *Why is the paused agent session special?* — no compute, but it holds KV; see T14-Q27.
- *What metric would you watch for agentic traffic?* — session completion time, not TTFT `[T]`.
- *What is the failure if you skip hysteresis?* — routing flap; requests oscillate between replicas.

**Red flags:** Adopts 80%/8 as defaults, uses one fairness discipline for both saturation regimes, or
cannot state the degradation order.

---

### Layer 1 — qualitative routing: which model

#### T14-Q12 · The semantic router's pipeline, and why it is not an if-else
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** A colleague says the semantic router is "a lookup table from prompt to model." Correct
them, and walk me through what actually happens to a request.

**Model answer:** The correction comes from the project's own maintainer, and it is the most
important thing to know about it: "some people think this is an if-else condition, like this prompt
is matching to the model description it will route, but no — there are multiple layers of pipeline"
`[T]`.

The pipeline `[T]`:

| Stage | Job |
|---|---|
| **Signal** | Detect properties of the prompt — "the prompt can be code, long prompt, and multiple signals" |
| **Partition** | Decide which of those signals are relevant given the routing configuration |
| **Score** | Produce a **difficulty score** |
| **Algorithm** | Apply a routing algorithm — the talk names confidence, rem, fusion and workflow loops, each published with its own before/after benchmark |
| **Model decision** | Commit to a model |

So it is a *classifier over content features*, not a matcher over model descriptions. The distinction
is not pedantic: a lookup table cannot express "this is a three-sentence question whose consequence
is large," which is the canonical hard case.

The project also inherits a constraint worth stating: the classifier "is ambert 32 model" — ASR for a
**ModernBERT-class encoder**, the exact name uncertain, which the maintainers "fine-tune on a regular
basis and publish on the Hugging Face repositories" `[T]`. You inherit a model you did not train on
your own traffic.

And the honest operational admission, which a weak candidate will not volunteer: routing quality is
configuration-driven, and "if you are using the default state in the config and you find there's a
gap between your X state and Y state then you need to update your config… so this is kind of manual
work you need to do at present" `[T]`. Budget for that tuning; it is not a set-and-forget component.

**Signal:** Describes a multi-stage pipeline with a difficulty score at its centre, and volunteers
either the inherited-classifier constraint or the manual-configuration admission.

**Follow-ups:**
- *What is the output of the score stage?* — a difficulty score, which is the L1 knob.
- *Who trained the classifier?* — the maintainers, on their data; see T14-Q17.
- *How does this sit relative to L2?* — it picks the model, then hands off; see T14-Q1.

**Red flags:** Describes an if-else or a description-matching lookup, or assumes the classifier is
trained on your traffic.

---

#### T14-Q13 · Why does a router expose 50–60 response headers?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** The semantic router emits "around 50 to 60 headers" per response `[T]`. Isn't that just
noise? What is the observability surface for, and where does it belong in your architecture?

**Model answer:** The headers are not noise; for a system whose entire job is a decision, **the
decision record *is* the product.** The maintainer's framing: the headers report "which model it is
being chosen, what is the confidence of this," and can be used "as a source of truth, like where did
your request go through, what were the consequences, what will happen if you choose this model or
not" `[T]`.

That gives three distinct uses, and it is worth separating them:

1. **Per-request debuggability.** A user complains the assistant got dumber. Without the decision
   record you are guessing; with it you can see the model, the confidence and the reason for *that*
   request.
2. **Aggregate monitoring.** The routing distribution — which model, at what confidence, for which
   tenant — is the primary dashboard for L1, because the failure mode is **silent**. The cheat
   sheet's signature: "cost up, latency flat, quality flat — classifier drifted toward the frontier
   model." Latency will not page you; the distribution will.
3. **Benchmark construction.** The maintainer makes this point explicitly — from that information
   "we can create a benchmark as well" `[T]`.

**Where it belongs.** Two caveats. First, headers are a **convenience transport**, not an audit
trail: they are per-response and easy to lose. The moment routing decisions become contractual — a
customer promised a model class or a region — the requirement escalates from "nice headers" to an
**immutable decision log** `[D]`. Second, at 50–60 headers the surface is large enough that nobody
reads it; the value comes from picking the three or four fields you alert on and exporting them as
telemetry, not from the header count. A candidate who proposes "log all 60" has described a
log-volume problem, not observability.

The case study treats this as a non-functional requirement in its own right: "every response carries
the chosen model, the confidence, and the reason."

**Signal:** Frames the headers as the decision record and names the silent-drift failure that only
the distribution catches — rather than praising observability in general.

**Follow-ups:**
- *Which metric catches classifier drift?* — the chosen-model distribution, not latency.
- *When are headers insufficient?* — when the decision is contractual; you need an immutable log.
- *What would you alert on out of 60 headers?* — model choice, confidence, and reason, per tenant.

**Red flags:** Calls it over-engineering, or proposes emitting all headers into the log pipeline
without saying which drive action.

---

#### T14-Q14 · Difficulty routing: the arithmetic and the honest ceiling
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** "A routing tier alone often cuts total spend by half" `[T]`. Do the arithmetic and tell
me whether that is a claim you would repeat to a CFO.

**Model answer:** I would repeat it only with its conditions attached, because the corpus's own case-
study arithmetic shows when it fails.

**The claim, as stated.** "A small, cheap classifier looks at each incoming query and decides whether
it is easy or hard. The easy majority go to a small, fast model. Only the genuinely hard queries
reach the expensive one… a routing tier alone often cuts total spend by half with no drop in quality
that users notice" `[T]`. The same talk calls it "the single highest leverage pattern most teams add"
and a separate transcript puts it in the same family: "routing easy questions to a small model often
beats any single low-level trick" `[T]`.

**Where the arithmetic supports it.** If nearly all traffic is cheap-qualified and the cheap model is
~20× cheaper, the saving is near-total on that traffic. The case-study scenario with 90% easy
Dispatch traffic gets an 85.5% saving on Dispatch.

**Where it breaks.** The Aperture Freight model is the counter-example, and the reason is the whole
lesson: Dispatch is **84% of token volume** but only **~21% of the priced mix**, because 300 Ops
sessions/day at 60k-token prompts on a frontier model dominate cost despite being 16% of tokens.
Blended saving is therefore `0.855 × 0.21 ≈ 18%` of total spend — real, but a long way from half `[D]`.

So the honest framing for a CFO: **difficulty routing's value is bounded by the cheap model's share
of the *priced* mix, not by the share of requests.** The talk's "half" presumes a traffic mix without
a heavy frontier-priced tail. Say that out loud before quoting the number.

**What corroborates the shape.** RouteLLM (UC Berkeley / LMSYS, ICLR 2025) is cited at ~95% of
frontier quality at 45–85% cost reduction `[R]` — note the range, and note the guide's own caveat
that this is "a benchmark-specific ceiling, not a guarantee," and that the escalation rate is the
live cost variable.

**Signal:** Quotes the claim with attribution, then bounds it using the priced-mix argument rather
than either repeating or dismissing it.

**Follow-ups:**
- *Why does 84% of tokens equal 21% of cost?* — the frontier-priced Ops tail dominates.
- *What is the sensitivity?* — the easy fraction and the price ratio; both move the answer.
- *Does that make routing worthless?* — no; on the right mix it is the largest single lever.

**Red flags:** Repeats "half the spend" unconditionally, or rejects the pattern because one mix
under-delivers.

---

#### T14-Q15 · The expensive-router anti-pattern
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** An engineer proposes using the frontier model to classify requests and route them,
because it is the most capable. What is wrong with that, and what is the arithmetic?

**Model answer:** It is the single most quotable design error in this topic, and the corpus's own
speaker admits to having built it: "I've even built solutions wherein we've used very expensive
mixture-of-expert models because they're so smart and so capable, but they are expensive to run… I
would still use them to do some form of routing. So I would use the same Qwen 3… to decide which is
the right model to address — but that's not the right thing to do, because if you solve that problem
efficiently you are going to address the query significantly faster and you're going to meet that
SLA" `[T]` Singh. Note the correction is on **both** cost and latency grounds.

**The arithmetic `[D]`.** If the router costs the same order as the model it routes to, the routing
decision is added latency and added cost on **every** request — including the ones routed to the
expensive model, which gain nothing. The saving must exceed the routing cost across the whole
population. A router costing 30% of a frontier call and diverting 50% of traffic to a model 20×
cheaper nets out only at high diversion rates; at low diversion rates it is a **pure loss**. Write
the condition down: `diverted_fraction × saving_per_diverted > router_cost`, where both sides are
per-request and the router cost is paid universally.

**The overhead ladder that makes it concrete** `[R]` — vendor/practitioner figures, order-of-
magnitude only: rule-based `< ~1 ms`; embedding/semantic `~5 ms`; ML classifier or LLM-as-router
`~50–100 ms`; against typical model latency of `500–2,000 ms`.

So a 5 ms encoder is a rounding error against a 900-token prompt whose prefill alone is tens of
milliseconds. A 90 ms LLM-as-router is **10–18% of the total budget**, justifiable only for the
ambiguous tail — which is exactly the guide's recommendation: a **two-stage hybrid** that handles
the confident majority semantically and escalates the ambiguous tail to an LLM-as-router `[R]`.

**Signal:** States the loss condition rather than "it's slower," and puts the 5 ms / 50–100 ms /
500–2,000 ms ladder against the decision.

**Follow-ups:**
- *When is an LLM router justified?* — the ambiguous tail only; see T14-Q16.
- *What does a 90 ms router cost you?* — 10–18% of the latency budget `[D]`.
- *What is the pattern the guide recommends?* — the two-stage hybrid.

**Red flags:** Proposes the frontier model as router, or evaluates the router's cost without
counting the requests it routes to the expensive model.

---

#### T14-Q16 · The cascade, and why the escalation rate is the whole game
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You deploy a cascade: cheap model first, escalate to the frontier model when
confidence is low. Six weeks later the bill is unchanged. What happened, and how do you fix it?

**Model answer:** The escalation rate became the live cost variable, and it drifted up. The guide
states the failure and the fix in one breath: cascades are "the highest-leverage cost play," but the
failure modes are "double-spend on escalated queries" and "noisy confidence" — and "a cascade that
escalates most traffic saves little" `[R]`.

**The mechanism.** A cascade pays for the cheap model on every request, then pays for the frontier
model *again* on every escalated request. So the cost is:

```
cost = cheap × N + frontier × (escalation_rate × N)     [D]
```

Above some escalation rate the cascade is strictly **worse than routing statically**, because static
routing would have paid for exactly one model per request. That crossover is the number to compute
before shipping, and it is a function of the price ratio: with a 20× price gap, escalating more than
roughly a third of traffic approaches the static-frontier cost from below, and beyond that the cheap
model is pure overhead `[D]` — assumptions: the cheap model is ~1/20 the price, the frontier call is
not cached, and no prompt caching applies.

**Why it drifts silently.** Two reasons. First, **confidence calibration** — if the cheap model is
systematically under-confident on your domain, it escalates a large fraction of easy traffic, and
nothing errors. Second, **distribution drift** — as traffic changes, the fraction falling below the
threshold changes with it. Neither shows up in latency.

**How I would fix it, in order.**
1. **Instrument the escalation rate as a first-class metric**, with the price ratio beside it, so
   the crossover is visible rather than discovered on an invoice.
2. **Check calibration on labelled traffic** — is the cheap model's confidence actually predictive
   of quality? RouteLLM's reported range (~95% of frontier quality at 45–85% cost reduction) is a
   benchmark-specific ceiling, not a target `[R]`.
3. **Raise the threshold or retrain** — but understand that raising the threshold trades quality,
   so it must be moved with evals attached, which is exactly the corpus's guidance for the L1
   threshold knob: "move it in small steps and re-run evals each time."
4. **Consider that the honest answer is a classifier, not a cascade** — if the escalation rate
   cannot be brought down, the confidence signal is the problem, and a properly trained classifier
   at the front is the fix.

**Signal:** Writes the cascade cost as cheap-plus-escalated-frontier and identifies the crossover
where it loses to static routing — rather than diagnosing "the cheap model is bad."

**Follow-ups:**
- *What is the threshold knob?* — the difficulty threshold; the single knob trading cost for quality.
- *What is noisy confidence?* — a signal uncorrelated with actual quality; the double-spend driver.
- *Cascade or classifier?* — cascade if escalation is genuinely rare; classifier otherwise.

**Red flags:** Blames the cheap model's quality, or proposes raising the threshold without an eval
loop to hold quality.

---

#### T14-Q17 · Classifier drift, per-tenant thresholds and the ambiguous tail
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** Your L1 classifier has been in production for a year. Two things are now true: the
model-choice distribution has drifted, and one large tenant is getting frontier answers for trivial
questions. Diagnose both, and tell me what changes at 10× traffic.

**Model answer:** These are two different failures with one shared root: **the classifier was trained
on someone else's distribution, and there is only one of it.**

**Failure one: time drift.** The maintainers fine-tune and republish the classifier on a regular
basis `[T]`, but that model was trained on **their** data. Your distribution drifts away from it, and
the corpus names the consequence: "Classifier drift — the semantic router's maintainers fine-tune
that model… but that model was trained on *their* data. Your distribution will drift away from it.
Re-fit on your traffic, or accept a decay you cannot see." The detection is the routing distribution,
not latency; the fix is either re-fitting on your own labelled traffic or tuning the config, and the
maintainer is honest that the latter is manual work today `[T]`.

**Failure two: one threshold for many tenants.** The case study states it plainly: "A classifier
trained on the aggregate mis-serves any tenant whose difficulty distribution differs. Per-tenant
thresholds, or the 4B-model tenant gets frontier answers forever." That is exactly the observed
symptom — the tenant whose definition of "hard" differs from the aggregate gets over-served. Note
that this is not a classifier accuracy problem; the classifier may be perfectly accurate *on
average*. It is a policy problem: **one threshold cannot serve two distributions.**

**What changes at 10×** `[D]`, from the case study's own framing:
- **Per-tenant routing replaces global routing.** Routing configuration becomes a per-tenant
  artifact, because a single threshold is guaranteed to be wrong for someone.
- **The routing tier becomes the most critical service you run** — its own SLO, its own on-call, its
  own capacity model.
- **Routing decisions become contractual.** Once a customer is promised a model class or a region,
  the router is enforcing an SLA, and the auditability requirement escalates from headers to an
  immutable decision log.
- **All remaining value and risk live in the tail**, which is where LLM-as-router earns its cost.

**The honest limit I would state:** this corpus contains **no measurement of classifier drift over
time** and no head-to-head of a semantic router against a naive baseline on traffic resembling this
scenario. Both of those are things I would have to measure myself, and I would say so rather than
citing a number.

**Signal:** Separates temporal drift from the single-threshold policy failure, and refuses to assert
a drift measurement the corpus does not contain.

**Follow-ups:**
- *What detects drift?* — the model-choice distribution per tenant, not aggregate latency.
- *Why is per-tenant thresholding needed?* — one threshold cannot serve two distributions.
- *What is the tail's role at 10×?* — it is where the remaining value and risk concentrate.

**Red flags:** Proposes retraining the vendor classifier as the only fix, or asserts a drift
magnitude the corpus does not measure.

---

### Layer 0 — the gateway, fallback and reliability

#### T14-Q18 · Build the fallback chain: what is retryable, and what is not
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** The frontier provider throttles you at peak. Design the fallback. What do you retry,
what do you never retry, and what does the chain look like?

**Model answer:** The context is empirical: rate-limit and capacity errors are "the single largest
production failure mode," with roughly 5% of requests failing and roughly 60% of those failures
capacity-driven `[R]` (guide, citing Datadog's 2026 State of AI Engineering). So this is not a
corner case.

**What is retryable** `[R]`: 429, 5xx, timeout, model-not-found. **What is never retryable**: 400-class
client errors — "never retry 400-class client errors, which just waste quota" `[R]`. Retrying a
malformed request across three providers burns three times the quota to produce the same error.

**The three rules for the retries you do make** `[R]`:
- **Exponential backoff** — wait longer each failure.
- **Jitter** — randomise the wait "so clients do not retry in lockstep and create a thundering herd
  that worsens the outage."
- **Honour `Retry-After`** — use `max(retry_after, computed_backoff)`. "Ignoring `Retry-After` is the
  most common backoff bug."

Plus **circuit breakers** that track endpoint health globally and open when the failure rate crosses
a threshold, so you stop hammering a dead provider on every request `[R]`.

**The chain shape.** The case study's chosen shape for the frontier-API path: an **ordered chain
across two external providers plus the self-hosted pool as the final hop**, with circuit breakers
and `Retry-After` honoured, and explicitly **not** `max_retries > 2` on the primary. The structural
insight is that the ordered cross-provider chain is the *fix*, not a patch: load balancing across
keys, regions and providers "multiplies your effective rate-limit headroom, which is the most direct
structural fix for capacity failures" `[R]`.

**Two caveats to state.** First, if the two providers are the same upstream, the chain is theatre.
Second, **failover on a streamed response is impossible after the first token** — decide the failover
boundary before streaming begins; after it, the only honest option is to terminate and let the client
retry.

**Signal:** Distinguishes retryable from non-retryable error classes precisely, and names
`Retry-After` handling and the pre-first-token failover boundary.

**Follow-ups:**
- *Why jitter specifically?* — to prevent a synchronised retry herd.
- *What is the structural fix, not the patch?* — multi-key/multi-region/multi-provider load balancing.
- *What breaks streaming failover?* — tokens already emitted; decide the boundary first.

**Red flags:** Retries everything, omits `Retry-After`, or claims a stream can be transparently
failed over mid-response.

---

#### T14-Q19 · Count your retry layers
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** You add retries at the gateway. During the next provider slowdown, load on the provider
doubles and the outage gets worse. What did you miss, and how do you fix it systematically?

**Model answer:** I missed that retries are already happening at several layers, and I added another
multiplier.

**The mechanism.** The guide's caution is that **"blind retries amplify outages"** by adding load
during a failure `[R]`. What makes it a *multiplicative* problem in a real stack is that retries are
rarely in one place. Count them `[D]`:

| Layer | Typical retry behaviour |
|---|---|
| Model SDK inside the app | framework default, often 2 attempts |
| Application-level wrapper | bespoke, usually undocumented |
| **The gateway** | ordered fallback chain across providers |
| The client / browser / mobile app | user-initiated or SDK-level |
| Upstream orchestrator or job runner | whole-task retry |

If three of those are active with two attempts each, one user action becomes up to eight provider
requests — and they arrive *during* the throttle window, which is precisely when the provider can
least afford them. Blind retries at the gateway are therefore not a reliability feature; they are a
load amplifier aimed at the failing component.

**The systematic fix, in order** `[D]`:
1. **Enumerate every retry layer explicitly** and write the count down. The case study's own framing
   is "count the retry layers explicitly and disable all but one." You cannot reason about a
   multiplier you have not counted.
2. **Keep exactly one layer authoritative** — the gateway, because it is the only layer that sees
   provider health globally and can apply a circuit breaker. Disable the SDK defaults and the
   bespoke wrapper.
3. **Make the surviving retries well-behaved**: backoff, jitter, `Retry-After`, circuit breaker `[R]`.
4. **Bound it** — cap attempts (the case study sets `max_retries` at 2 on the primary, explicitly
   not more), and bound total retry *time*, not just count.
5. **Make retry rate an alarm.** The failure signature is "load spike during a provider slowdown,"
   detected by "retry rate vs request rate" `[D]`. If retry rate is not on a dashboard, this failure
   is invisible until the provider tells you.

**Signal:** Goes straight to the multiplier across layers rather than "add backoff," and assigns
authority to the one layer that can see global health.

**Follow-ups:**
- *Which layer should own retries, and why?* — the gateway; it alone sees provider-wide health.
- *What bounds the damage?* — attempt cap plus a total-time budget plus a breaker.
- *What alarm catches it?* — retry rate against request rate.

**Red flags:** Adds backoff at the gateway and stops, without counting the layers beneath it.

---

#### T14-Q20 · The gateway is a new single point of failure
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You are introducing a gateway to improve reliability. Make the case that you might be
making it worse, and tell me how you prevent that.

**Model answer:** This is the strongest engineering signal in the topic, and the guide states the
failure mode in one line: **"A gateway that centralizes everything but runs as one instance has
simply *moved* your single point of failure"** `[R]`.

**Why the risk is easy to miss.** Before the gateway, provider failures were partial — one provider
down meant degraded service on one path. After it, *all* model traffic traverses one component, so
its availability becomes the floor for the whole product. You have traded a set of independent
partial failures for one total failure. The case study makes it a non-functional requirement with a
specific shape: "availability of the routing tier itself ≥ 99.9%, and **non-load-bearing until
proven**."

**How I prevent it** `[R]` `[D]`:
1. **Multiple stateless replicas behind a load balancer.** "≥3 replicas" is the case study's chosen
   posture.
2. **Externalise all shared state.** Rate-limit counters and usage into Redis, keys and spend into a
   database, so no replica is authoritative and any replica can die. A replica holding in-memory
   counters is a replica whose death resets your quotas.
3. **Health-check the replicas, and route around them** — the gateway's own load balancer must not
   be the same component as the gateway.
4. **Roll out in shadow mode first**, and keep it non-load-bearing until proven. The guide's rollout
   order is explicit: observe mode, non-critical workloads, then attribution before enforcement,
   then fallback chains, then caching and routing — and "make the gateway highly available *before*
   it becomes load-bearing" `[R]`.
5. **Have a bypass.** The strongest version of "non-load-bearing" is that a direct-to-provider path
   exists and has been tested. An untested fallback path is not a fallback; it is a second outage
   waiting for the first.

**The same argument generalises to L2.** The case study lists "Router (EPP) down" as a failure with
blast radius "entire fleet," mitigated by a redundant router and a gateway fallback route. Whatever
you centralise, you must replicate.

**Signal:** Names the move-the-SPOF failure unprompted, and prioritises externalised state and a
tested bypass over replica count alone.

**Follow-ups:**
- *What must be externalised, and why?* — counters and keys; any replica must be able to die.
- *When does it become load-bearing?* — after shadow mode and after HA is proven, not before.
- *Does the EPP have the same risk?* — yes; blast radius is the entire fleet.

**Red flags:** Proposes a single gateway instance, or claims HA is achieved by replicas while state
stays in-process.

---

#### T14-Q21 · Build versus buy, with a constraint that decides it
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** You need a gateway. Walk me through the choice — and explain how a non-technical
requirement can settle a technical comparison before it starts.

**Model answer:** I would structure this as: constraints first, then the tool comparison, then the
decision that the constraints already made.

**The constraint layer.** The case study's scenario has a contractual requirement that some freight
customers' traffic be processed in-country. That single line **disqualifies every managed option
outright** — OpenRouter, Portkey and Cloudflare AI Gateway all fail it, because "data transits a
third party and you inherit their availability as a hard dependency" `[R]`. The point a candidate
should make explicitly: **a hard constraint decides the question before any feature comparison
begins.** Comparing LiteLLM's routing strategies against OpenRouter's model breadth is wasted work
if residency has already removed half the field. This is the same discipline as "region as a filter,
never a score" — hard constraints are evaluated first and are not traded off.

**The remaining comparison** `[R]`:

| Option | Strong at | Weak at |
|---|---|---|
| Thin in-app abstraction | no new hop, no new SPOF | no cross-team budgets; glue gets copy-pasted; breaks at the second team |
| Self-host LiteLLM | data in perimeter; full control; virtual keys with dollar budgets; native OTel | "YAML config strains at enterprise-governance scale"; *you* must make it HA |
| Envoy AI Gateway | infra-grade priority-based fallback, retries, timeouts; Kubernetes-native | lower-level; you compose more policy |

**What I would choose and why.** Self-hosted LiteLLM, because residency disqualifies the managed
tier and the case-study scenario has platform-team bandwidth to own an HA deployment. The YAML-scaling
critique is real but is a *later* problem than residency; I would note it as the known growth limit
and plan the migration path rather than pre-emptively solving it.

**What I would say about the tool table itself.** Treat versions and exact feature claims as
point-in-time, and note that several comparison figures in this landscape originate from **vendor
marketing** `[R]`. A candidate who picks a product by reading its own comparison page has not
evaluated anything.

**Revisit condition.** If the residency requirement were lifted for some tenants, a managed gateway
for *those* tenants removes an HA burden — so the decision is per-tenant, not global, and I would
record the constraint that produced it.

**Signal:** Leads with the constraint rather than the feature matrix, and identifies which option
class is disqualified before comparing the rest.

**Follow-ups:**
- *Which managed gateway would you pick if residency were lifted?* — depends on whether you need a
  full control plane or just observability `[R]`.
- *What is LiteLLM's known scaling limit?* — YAML config at enterprise-governance scale.
- *Where does the residency rule live?* — as a filter, evaluated before any scoring.

**Red flags:** Starts with a feature comparison, or picks a managed gateway while a residency
constraint stands.

---

### Cost, leverage and the order of work

#### T14-Q22 · Why is routing the last big lever on the cost ladder?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** The corpus calls routing "the last big cost lever." Show me the ladder and explain what
that position means.

**Model answer:** The ladder, from the LLMOps cost talk, starts at naive 16-bit serving one request
at a time as **100 cost units** `[T]`:

```
naive 16-bit, one request at a time      100
+ continuous batching                   ~ 42
+ 4-bit quantization                    ~ 26
+ caching the stable system prompt      ~ 11   <-- the routing step
```

"That is roughly a tenth of the original bill and the model never changed" `[T]`. The final step is
the one routing owns, and the supporting number is that **cache reads can be roughly 10× cheaper than
fresh tokens** `[T]`.

**What "last lever" means, precisely.** It does not mean least important. It means **last in the
causal chain**: the earlier steps get you the *capability* to cache cheaply, and routing is what
determines whether that capability is actually *used*. Caching builds the memory; routing determines
the hit rate. A deployment with a perfect prefix cache and round-robin routing has a near-zero hit
rate — the cheat sheet's phrasing is that a naive load balancer's prefix cache hit rate is
"atrocious."

**The arithmetic** `[D]`, using the cheat sheet's own derivation: with `effective_cost = hit_rate ×
cached_cost + (1 − hit_rate) × fresh_cost` and cached cost at 1/10 of fresh, moving hit rate from
0.2 to 0.8 cuts per-token cost by roughly `0.6 × 0.9 = 54%` — which is the 26 → 11 step. Assumption:
the cached and fresh token prices differ by the ~10× the corpus reports, and the hit-rate move is
from 0.2 to 0.8.

**Why this framing matters operationally.** It tells you the order of work: caching is worthless
without routing, so the two must ship together, and buying more GPUs to compensate for a broken
router is paying twice for one problem. The cheat sheet's "when to use what" row is blunt: "You want
the cost ladder's final step — fix routing before buying more GPUs" `[T]`.

**Signal:** Reproduces the ladder with its numbers, and explains "last" as *last in the causal chain*
rather than least important.

**Follow-ups:**
- *What does caching give you that routing does not?* — capacity to serve cheaply; routing gives the
  hit rate.
- *What is the ~10× figure?* — cache reads versus fresh tokens.
- *What breaks the pair?* — a short cache TTL makes prefix routing pointless.

**Red flags:** Reads the ladder as a ranked list of independent optimisations, or treats routing as
an optimisation to do later.

---

#### T14-Q23 · Where "cuts spend by half" breaks
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** Two levers, both real: difficulty routing and prefix routing. Take the scenario —
40,000 Dispatch sessions/day at 900/120 tokens over 2.5 turns, 300 Ops Analyst sessions/day at
60,000/3,000 tokens — and tell me what each is worth and why they do not simply add.

**Model answer:** They act on different axes, so they **multiply**. Difficulty routing reduces the
**price per token**; prefix routing reduces the **number of tokens**. Working through the case
study's arithmetic `[D]`, all inputs attributed:

**Where the tokens are.**
```
Dispatch = 40,000 × 2.5 × (900 + 120) = 102.0M tokens/day
Ops      =    300 ×  1  × (60,000 + 3,000) = 18.9M tokens/day
Total    = 120.9M tokens/day
```
Dispatch is **84% of tokens** but, once the frontier-priced Ops prompts are priced, only about
**21% of the priced mix** — `102.0 / (102.0 + 18.9 × 20) ≈ 0.21`, assuming the ~20× price gap
between a 4B-class and a frontier model.

**Lever one — difficulty routing.** At 90% easy and 20× cheaper, Dispatch's cost index goes
`102.0 × (0.9/20 + 0.1) = 14.8`, an 85.5% saving *on Dispatch* — but only `0.855 × 0.21 ≈ 18%` of
total spend. This is the honest correction to "a routing tier alone often cuts total spend by half"
`[T]`: the claim presumes a mix without a heavy frontier-priced tail.

**Lever two — prefix routing.** 2.5 turns/session means 60% of the 100,000 daily turns are
continuations, currently recomputing because round-robin sends each turn to a different replica
(~0% hit rate today). At a 70% hit rate on those turns:
```
continuations          = 60,000
hitting cache          = 60,000 × 0.70 = 42,000
tokens moved to cache  = 42,000 × 900 = 37.8M
effective reduction    = 37.8M × (1 − 1/10) = 34.0M effective tokens
as a share of Dispatch prompt tokens = 34.0 / 81.0 ≈ 42%
```

**Why they multiply.** Lever one changes the price of the tokens that remain; lever two removes
tokens from the fresh-token pool entirely. Applying a lower price to a smaller pool compounds, which
is why the case study says the two "multiply rather than add" and why the ordering work is decided
on value per engineer-week rather than on theoretical size.

**Signal:** Runs both calculations with stated assumptions, and names the priced-mix correction that
bounds the headline claim.

**Follow-ups:**
- *Which lever touches which axis?* — price per token versus token count.
- *Why is Dispatch 84% of tokens but 21% of cost?* — the frontier-priced Ops tail.
- *Which would you do first?* — see T14-Q24.

**Red flags:** Adds the two savings percentages, or accepts "half the spend" without the mix
qualification.

---

#### T14-Q24 · Order the work: value per engineer-week
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You have one engineer-quarter and three candidate work items: L2 prefix routing, L1
difficulty routing, and L0 gateway with fallback. You also have a rate-limit incident, a cost review,
and a cache-miss review — three pressures, three different answers. Decide the order and justify it
to the people whose item is not first.

**Model answer:** I would order it **L2 → L1 → L0**, and the justification is value per engineer-week
rather than severity of the loudest complaint.

**The case study's own numbers** `[D]`, from its §8 Step 4:

| Work item | Effort | Annual saving (est.) | Value/week |
|---|---|---|---|
| **L2 — precise prefix routing** | 3 weeks | ~$95k from cache reuse | **~$32k/week** |
| **L1 — difficulty classifier** | 4 weeks | ~$74k from model substitution | ~$18k/week |
| **L0 — gateway + fallback** | 3 weeks | avoided outage + budget control | ~$10k/week (risk-adjusted) |

**Why L2 first, and this is the crux of the answer.** It is both the **largest** and the **cheapest**
win, and — the part I would emphasise — it was discovered as an *incident*, not a project. The
cache-miss review already predicts the benefit, so the business case is already argued. Nothing needs
to be trained, nothing needs a labelled dataset, and the deliverable is verifiable in a week: the KV
event rate equals the request rate, and the cache hit rate moves. L1 by contrast requires a
classifier, a threshold, shadow mode, and a labelled comparison over two weeks.

**Why L0 is third despite an outage.** Because the incident was a *throttling* event with a bounded
blast radius, and the risk-adjusted value is genuinely lower than either cost lever. The ordering is
not a claim that fallback is unimportant; it is a claim that three engineer-weeks spent on the
gateway return less than three spent on prefix routing. I would say explicitly that if the incident
had been a **hard outage** rather than throttling, L0 goes first — severity changes the ordering, and
the ordering should be re-derived rather than remembered.

**What I would do for the people whose item is not first.** Give each a **date and an exit
criterion**, not a promise. L0 gets a documented interim: an in-app fallback with backoff and jitter
is a few days of work and covers most of the incident's risk while the gateway waits. That is the
real answer to the political constraint — the ordering is justified by arithmetic, and the losers
get a cheap partial mitigation, not a queue position.

**And the sequencing rule that matters more than the order:** L1 ships in **shadow mode first** —
log the decision it would have made, act on nothing, compare against actual cost and quality for two
weeks. L0 likewise: observe, then attribution before enforcement, then fallback. Every one of these
levers is cheaper to validate in shadow than to debug in production.

**Signal:** Orders by value per engineer-week with the numbers, and justifies the *loss* to the
loudest stakeholder with a cheap interim mitigation rather than a promise.

**Follow-ups:**
- *What would flip the order?* — a hard outage rather than throttling puts L0 first.
- *Why is L2 cheaper than L1?* — no classifier, no labels, verifiable in a week.
- *How does each ship?* — shadow mode first; observe before enforce.

**Red flags:** Orders by loudest complaint, or treats the ordering as permanent rather than
re-derived when the pressures change.

---

### Diagnosis, failure modes and agentic traffic

#### T14-Q25 · "The assistant got dumber last Tuesday." Debug it.
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** A customer complains the assistant got dumber last Tuesday. You have no alerts. Walk me
through your debug order and tell me why the obvious first check is wrong.

**Model answer:** The obvious first check is "did the model change?" and it is wrong because **routing
failures present as quality complaints.** The case study states the principle and the order: "Model-
choice distribution first (silent mis-routing), then cache hit rate, then provider fallback
activation, then the model itself."

**Why the model is last.** The model is the only component in the chain that changes on a *planned*
schedule and is versioned and announced. Everything upstream of it changes silently: a classifier
threshold drifts, an event stream dies, a fallback activates, a router restarts. If the model were
the cause, you would already know — you deployed it.

**The order, with the check for each** `[D]`:

1. **Model-choice distribution (L1).** Did the fraction routed to the frontier model move? The
   failure signature is "cost up, latency flat, quality flat — classifier drifted toward the
   frontier model," and the first action is to check the distribution, **not** the latency. If the
   distribution shifted, quality should have *improved*, which makes this the check for a *cost*
   complaint. For a quality complaint, look for the opposite: cheap-model share rising.
2. **Cache hit rate (L2).** "Cache hit rate collapsed — KV events stopped, or replicas were
   replaced. First action: KV event rate versus request rate." A dead event plane routes by load
   alone, which can also mean the wrong prefix context reaches the wrong replica.
3. **Fallback activation.** "Provider throttling again — fallback chain missing a hop, or
   `Retry-After` ignored." Fallback providers differ in style, and a mid-session switch is
   user-visible — the case study's signature is "users see different answer style mid-session."
4. **Then the model itself.**

**What I would add that the list does not say.** First, **check the timing against deploys** — Tuesday
is a deploy day in most organisations, and the fastest discriminator is whether anything shipped.
Second, **check whether it is one tenant or all of them.** If it is one tenant, that is the
single-threshold failure (T14-Q17), not a system regression, and the debug order changes entirely.
Third, **the absence of alerts is itself the finding**: routing failures are silent by construction,
which is why the routing distribution must be a primary dashboard rather than a derived one.

**Signal:** Puts the model last and explains why — it is the only versioned, announced component —
and adds the tenant-scope discriminator.

**Follow-ups:**
- *Why is latency flat in the classic L1 drift case?* — the wrong model was still fast.
- *What if only one tenant is affected?* — per-tenant threshold problem; see T14-Q17.
- *What would have caught it earlier?* — the routing distribution as a primary dashboard.

**Red flags:** Starts by comparing model versions, or debugs quality without ever looking at the
routing distribution.

---

#### T14-Q26 · Cache hit rate collapsed. Differential diagnosis.
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Your prefix cache hit rate drops from 70% to near zero overnight. Latency dashboards
are green. Give me the differential and the discriminating check for each candidate.

**Model answer:** The green latency dashboards are the diagnostic clue, not a reassurance. This is a
**cost-shaped** failure, and the cheat sheet's signature for it is exact: "cache hit rate near zero
in Prometheus — naive LB ignoring prefix locality — first check: router affinity."

**The differential, ranked** `[D]`:

| Candidate | Mechanism | Discriminating check |
|---|---|---|
| **KV event stream stopped** | L2 loses its world view and degrades silently to a load balancer | KV event rate vs request rate; alert on event rate = 0 |
| **Replica set churned** | A rolling deploy reshuffled every hash under approximate routing; hashes do not re-converge | correlate the drop timestamp with the deploy; check router restart |
| **Router restart lost state** | The view is rebuilt from events; a mid-stream restart leaves gaps | router uptime vs the drop timestamp |
| **Cache TTL or eviction policy changed** | short TTL makes prefix routing pointless | config diff; TTL interacts with routing |
| **Routing policy reverted to round-robin** | a config rollback or a fallback route took over | which replica served the turn, per request |

**The two that matter most, and why.** The event stream is first because it is the failure that the
precise-routing design *creates*: the corpus is explicit that precise routing "breaks silently when
the event stream stops." It is the price of the accuracy you bought, and it must have an alarm — the
case study's detection column says exactly that.

Replica churn is second because it is the failure that approximate routing *creates*, and the two are
easy to confuse. The discriminator is that churn is deploy-correlated while an event outage is not —
and the fix differs: churn argues for precise routing, an event outage argues for fixing the event
plane. Recommending the wrong one makes it worse.

**The confirming measurement.** After any fix, re-check that hit rate actually returns. If hit rate
recovers but cost does not, the diagnosis was incomplete — the cost may also be moving for L1
reasons (T14-Q25), and the two levers are independent.

**What I would say about the green latency dashboards.** They are green *because* the failure is
benign-looking: the service still answers, just at fresh-token prices. This is why cache hit rate
belongs on the primary dashboard next to the routing distribution, and why the case study lists it
under "monitor" rather than "debug."

**Signal:** Puts the event-plane failure first, distinguishes it from replica churn by the deploy
correlation, and reads green latency as evidence *for* the hypothesis rather than against it.

**Follow-ups:**
- *Which candidate is deploy-correlated?* — replica churn under approximate hashing.
- *What is the fix if it is churn?* — precise KV-event routing; hashes do not re-converge.
- *What if hit rate recovers but cost does not?* — a second, independent cause at L1.

**Red flags:** Blames the engine or the model, or proposes adding replicas before checking the event
rate.

---

#### T14-Q27 · Agentic traffic: route sessions, not requests
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** Your traffic is now ~70% agentic, and these workloads look nothing like chat. Tell me
what is structurally different about them, and how it changes your routing and scheduling design.

**Model answer:** Four things change, and each has a design consequence.

**1. The token profile inverts.** Agentic workloads are "heavily skewed — prefill is occupying like
98% of the tokens," because the agent is reading code and documents, while output is small: "tool
call, a few user permissions and all that" `[T]` Pravin. Consequence: the workload is
**prefill-dominated**, and the corpus confirms it operationally — when comparing 2P2D against 3P1D
PD-disaggregation ratios, "having more prefill than the decode definitely helped" `[T]`. So the
routing decision is overwhelmingly about prefill capacity.

**2. The unit of work changes from request to session.** "A simple task… goes through multiple turns
— it takes multiple requests to the inference stack and then comes back… So the unit of work
transfers from being a single request to being a session" `[T]`. Consequence: a router that decides
per request cannot see the thing that matters. The corpus names the direction as **session-aware
routing** — "you're not just routing requests, you're routing sessions… which sessions are more
active, which sessions have a tool call coming back after a long time" `[T]`.

**3. The bottlenecks are decoupled in time.** An agent waiting on a tool call consumes **no compute**
but holds **KV cache** — potentially for minutes. Consequence: two fairness disciplines are needed,
because the saturated resource differs. Least-attained service addresses compute saturation; turn
priority addresses KV saturation `[T]` (see T14-Q9, Q10). And the corpus's mechanism for the paused
session is explicit: with CPU KV offloading, a session that pauses and returns "saved about 5× in
TTFT" because the cache did not have to be recomputed `[T]`.

**4. Sessions are bursty and fan out.** "You may invoke sub-agents which will parallelly do different
tasks at the same time and then converge onto one outcome" `[T]`. Consequence: concurrency is
spiky in a way round-robin absorbs badly, and the parallelism decision belongs to the operator
separately from routing — "for the parallelism I think it is up to the operator… it's based on your
throughput and interactivity requirement" `[T]`.

**What I would change in the design.** Three things:
- **Adopt session-aware routing**, and instrument per-session rather than per-request. The corpus
  argues the headline metrics change: request latency, **program/session completion time**, and KV
  cache hit rate per session — TTFT and ITL are interactivity metrics and matter less here `[T]`.
- **Add KV-residency policy for paused sessions.** The roadmap direction is a KV-cache retention API
  plus **session metadata on KV blocks**, so the router knows which session a block belongs to and
  can evict a paused session's cache and re-fetch on return `[T]`. Note this is stated as in-progress
  work, not a shipped feature.
- **Budget the enablement knobs** that the corpus names for this workload class: multi-token
  prediction (~2× throughput), CPU KV offloading (~5× TTFT on the returning turn), and PD-ratio
  tuning toward prefill `[T]`.

**The limits I would state.** The ~70% and ~98% figures are one speaker's framing of current traffic
`[T]`, not a benchmark; the session-routing roadmap items are not shipped; and the corpus contains
no measurement of a session-aware router against a request-level baseline. Do not assert those.

**Signal:** Names the prefill-dominance, the request→session unit change, and the paused-session
KV-residency problem — and separates shipped mechanisms from roadmap ones.

**Follow-ups:**
- *Why do the two fairness strategies both matter here?* — compute and KV saturate at different
  times within a session.
- *What metric replaces TTFT?* — session completion time `[T]`.
- *What is the roadmap mechanism?* — session metadata on KV blocks; not yet shipped.

**Red flags:** Treats agentic traffic as "chat with more turns," or asserts the roadmap items as
available capabilities.

---

#### T14-Q28 · Defend the layering to someone who wants one router
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** A staff engineer argues: "Three components is two too many. One router, one config, one
on-call. Why should I run a gateway *and* a semantic router *and* an EPP?" Make the case for the
layering — and tell me when they are right.

**Model answer:** I would make the case on three grounds, and then concede a real one.

**Ground one: the layers consume different inputs.** L1 reads *content* — code, length, difficulty.
L2 reads *load and KV-cache state* — queue depth, active requests, and per-block create/evict events.
L0 reads *identity and provider health* — keys, budgets, quotas, 429s. A merged component would have
to ingest all of it, and the inputs have incompatible cadences: KV events arrive per request per
pod; budgets change per billing cycle; tenant policy changes per contract. Merging them means the
fastest-moving input sets the release cadence for the slowest-moving policy.

**Ground two: the projects say so, and they built them that way.** The corpus is explicit on both
sides. From llm-d's side: it "is a performance-oriented router" and "is not concerned with the
difference in the qualitative performance of these models" — while semantic routing "can perhaps
help with picking the right model for the right task" `[T]` Singh. From the semantic router's side,
the lifecycle is stated as gateway → semantic router → llm-d → vLLM `[T]`, and the router is described
as sitting "before llm-d" and deciding which model to route to, with llm-d then handling efficiency `[T]`.
The separation is not a vendor accident; it is the design both projects converge on.

**Ground three: the failure modes are independent and individually survivable.** L1 can mis-route by
model and L2 still places correctly. The KV event plane can die and L1 is unaffected. If they were
one process, a KV-event backlog would stall model selection. Separate blast radii are worth
operational surface.

**And the concession — where the staff engineer is right.** If your fleet collapses to one replica
per model, L2 has nothing to do. If it collapses to one model, L1 has nothing to do. The case
study's revisit condition says exactly this. And for a prototype with a single provider, the guide's
answer is a **thin in-app abstraction** — a gateway is overkill and adds a hop plus an HA burden `[R]`.
The honest version of the argument is not "always three components"; it is "three components once
you have three *decisions*, and not before."

**What I would add as the closing point.** The strongest counter to "one config" is that each layer
has a different **change rate and owner**: the KV event contract is an engine-team interface, the
difficulty threshold is a product decision, and budgets are finance. One config would mean one
change-approval path for all three.

**Signal:** Argues from inputs, cadence and ownership rather than "best practice," and concedes the
collapse conditions without being pushed.

**Follow-ups:**
- *When does L2 have nothing to do?* — one replica per model.
- *What does a single-provider prototype need?* — a thin in-app abstraction `[R]`.
- *Which layer has the fastest change rate?* — L2, driven by per-request KV state.

**Red flags:** Argues "separation of concerns" with no reference to inputs or cadence, or concedes
the layering is unnecessary without naming the conditions under which it is.

---

## Whiteboard exercises

### Exercise 1 — Draw the four layers and place the arrows
**Prompt.** "Draw me the request path for a platform serving two model classes with multi-turn
traffic. Label each box by the *question* it answers, name the layer number, and draw the arrow
nobody draws. Then tell me what happens at each arrow when it stops working."

**What the candidate must produce:** the four boxes in order, labelled by question rather than
product; the KV-event arrow drawn in the correct **direction** (engine → router); and the
degradation statement for each arrow that can break.

**Expected answer sketch:**

```
  client
    |
    v
[L0] GATEWAY  -- "which provider, and what on failure?"
    |            auth, virtual keys, budgets, fallback chain
    v
[L1] QUALITATIVE -- "which MODEL?"
    |            signal -> partition -> score -> algorithm -> decision
    |            writes the decision into response headers (50-60 of them)
    v
[L2] PERFORMANCE -- "which REPLICA?"
    |            EPP:  filter -> score -> rank
    |            + flow control: queue, saturation, bands
    |                    ^
    |                    |  KV EVENTS: block created / evicted   <-- THE ARROW
    +-------> [ pool: small model ]  [ pool: frontier self-hosted ]  [ external API ]
                        |
                        v
                  [L3] MoE router  -- "which EXPERT?"
                       inside the LM. NOT operator-controlled.

Degradation:
  KV event arrow dies  -> L2 silently becomes a load balancer.
                          hit rate falls, COST rises, latency FLAT.
  L1 threshold drifts  -> wrong model chosen. cost/quality move, latency flat.
  L0 gateway dies      -> total outage. this is the SPOF you created.
```

**Grading rubric (full marks requires all four):**
- Draws four boxes labelled by *question*, not by product name, and places L3 outside the operator's
  control before being asked `[T]`.
- Draws the KV-event arrow from engine to router, and states that its failure is silent with flat
  latency and rising cost — the single most valuable sentence in this exercise.
- Names the L2 pipeline as filter → score → rank, and places flow control as a *separate*
  responsibility from routing `[T]`.
- Volunteers the gateway SPOF and the EPP's fleet-wide blast radius without prompting.

---

### Exercise 2 — Diagnose a silent cost regression
**Prompt.** "Six weeks ago your monthly bill rose 30%. Latency p50 and p99 are unchanged. Quality
complaints are unchanged. You added no new providers and changed no models. Here are three
dashboards: (a) chosen-model distribution, (b) prefix cache hit rate by pool, (c) KV event rate
versus request rate. Tell me which you look at first, what each would show, and what you do."

**What the candidate must produce:** a ranked diagnostic order with the signature each dashboard
would show, a fix per branch, and a statement of which two causes are *independent* and could be
present simultaneously.

**Expected answer sketch:**

```
Symptom: cost +30%, latency FLAT, quality FLAT
=> This is a routing-shaped failure. Both levers act on cost without
   touching latency. Model itself is unchanged (stated).

Ranked checks                     What it shows                Fix
1. KV event rate vs req rate  --> events = 0 => L2 blind     --> fix event plane;
   (c)                             degrades to load balancer      alert on rate=0
2. Prefix cache hit rate (b)  --> collapsed => locality lost --> precise routing;
                                                                  check replica churn
3. Chosen-model distribution  --> frontier share rising      --> classifier drifted;
   (a)                             (silent over-serving)         retrain or retune
                                                              threshold (shadow first)

INDEPENDENT: (1)/(2) act on token COUNT (cache reuse);
             (3) acts on PRICE PER TOKEN (model substitution).
             Both can be true at once -- check both, do not stop at the first.

Confirming measurement after any fix: hit rate AND cost must both move.
If hit rate recovers but cost does not, the other lever is also broken.
```

**Grading rubric:**
- Reads flat latency plus rising cost as evidence of a *routing* failure rather than an
  infrastructure one, and says why.
- Puts the KV event rate first or second, and names its failure as **silent degradation to a load
  balancer** rather than an error.
- Recognises that the cache levers and the model-choice lever are independent and multiplicative, so
  both must be checked rather than stopping at the first hit.
- Insists on a confirming measurement, and states what it would mean if the fix did not move the
  number.

---

### Exercise 3 — Design admission and fairness for a saturated agentic fleet
**Prompt.** "One fleet, three traffic classes: interactive customer chat, long agent sessions with
tool-call pauses, and nightly batch. It saturates every evening. Design the admission policy, the
queueing discipline, and the failure/fallback behaviour. Then tell me what you would instrument, and
what would change your design at 10× traffic."

**What the candidate must produce:** a saturation definition derived from measurement, a band
structure assigned at the edge, a discipline matched to the *saturated resource*, an ordered
degradation path, and the instrumentation that would falsify the design.

**Expected answer sketch:**

```
1. SATURATION (operator-defined, NOT the corpus defaults)
   sweep load at real context mix -> goodput/GPU knee -> threshold below it
   + hysteresis  (else: routing flap)                       [D]
   corpus's KV-80% / active>8 are EXAMPLES of the pattern    [T]

2. BANDS (assigned at the EDGE, not in the router)
   premium  : interactive chat        -> dispatched first
   standard : agent sessions          -> queued, fairness discipline
   best-effort: nightly batch         -> DEFERRED, not dropped  [T]

3. DISCIPLINE = f(saturated resource)                        [T]
   compute saturation -> least-attained service  (per SESSION)
                         fix: 2-3x request LATENCY, not throughput
   KV saturation      -> turn priority
                         mechanism: finish long session -> evicts its KV
   paused on tool call -> no compute, HOLDS KV -> offload to CPU
                         (reported ~5x TTFT on the returning turn)  [T]

4. DEGRADATION (ordered, tested)
   another replica -> smaller model -> cached answer        [D]
   streaming: failover boundary is BEFORE the first token

5. INSTRUMENT
   routing distribution (band, model, replica, confidence, cost)
   KV event rate vs request rate   <- L2's only health signal
   hit rate by pool | per-layer routing p50/p99 | queue depth/band
   agentic: SESSION completion time, not TTFT/ITL            [T]

AT 10x: per-tenant routing configs; routing tier becomes the most
       critical service you run; decisions become contractual [D]
```

**Grading rubric:**
- Derives the saturation threshold from a measured goodput curve and explicitly refuses to adopt the
  corpus's 80% / >8 as defaults — that refusal is the discriminator.
- Matches each fairness discipline to the resource it protects, and puts **least-attained service**
  as the compute-side signal with the starvation direction stated correctly (the big session
  monopolises; short sessions starve) `[T]`.
- Identifies the paused agent session as the case where compute and KV pressure diverge, and treats
  it as a KV-residency problem rather than a queueing one.
- States an ordered, tested degradation path and names at least one metric that would falsify the
  design — and, at 10×, says routing becomes contractual rather than staying an optimisation.

---

## Sources

Transcripts (`refs/`):
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — Pravin (IBM Research). The EPP as a single deployment between the pods and the gateway; the
  external-proxy hook; filter→score→rank with the prefill/decode policy difference; per-request KV
  create/evict events; precise versus approximate prefix routing and "all the consistency problems";
  operator-defined saturation (KV 80% full, mean active requests > 8); the FCFS critique; priority
  bands; least-attained service and turn priority with the compute-versus-KV-saturation split;
  the fairness-starvation hazard; "MoE routing is not done at the level of llm-d"; ~70% agentic
  traffic, prefill ≈98% of tokens; MTP ~2× and CPU KV offload ~5× TTFT; 2P2D versus 3P1D; the
  KV-cache retention API and session metadata on KV blocks; session completion time as the agentic
  headline metric.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Inside_vLLM_Semantic_Router.txt` — the
  signal→partition→score→algorithm→model-decision pipeline; "around 50 to 60 headers"; the "ambert
  32" classifier (ASR; a ModernBERT-class encoder, name uncertain) fine-tuned and republished; the
  92.66 vs 96.0 live-coding benchmark against GPT-5.5, self-reported and "a bit outdated"; the
  manual-configuration admission; enterprise guards (input filtering, semantic classification,
  policy routing); and the gateway → semantic router → llm-d → vLLM lifecycle.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
  — Abhishek Singh. The two-model-class example; "llm-d is not concerned with the difference in the
  qualitative performance of these models"; the 2025 regex-routing history; the expensive-MoE-as-
  router anti-pattern with Qwen 3.x; the ~2× throughput at ~85 QPS crossover and the ₹10 lakh →
  ₹5 lakh/month claim (vendor-affiliated, 200 users).
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the instruction that MoE expert routing is not influenced from llm-d; the vLLM/llm-d serving
  stack context for the L2/L3 boundary.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
  — the 100 → 42 → 26 → 11 cost ladder with the final routing step; cache reads "roughly 10 times
  cheaper than fresh tokens"; difficulty-based routing as "the single highest leverage pattern most
  teams add"; "a routing tier alone often cuts total spend by half"; "optimization without
  measurement is just guessing."

Supporting repositories (`refs/`):
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md`
  `[R]` — the gateway job table and the `N providers × M concerns` framing; the routing-strategy
  ladder with the <1 ms / ~5 ms / 50–100 ms / 500–2,000 ms overhead figures (flagged vendor and
  practitioner estimates); retry/backoff/jitter/`Retry-After`/circuit-breaker rules; the "moved your
  single point of failure" warning; the 2026 tool landscape with self-hosted versus managed
  tradeoffs; and the Datadog 2026 and RouteLLM figures cited through it.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`
  `[R]` — inference gateway components (auth and rate limiting, model router, context tracker and
  sticky sessions, output filter).
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]` — routing as a
  discipline distinct from serving.

**ASR corrections applied.** "LMD"/"LLMD" → *llm-d*; "VLM"/"BLM" → *vLLM*; "LightLLM" → *LiteLLM*;
"KSER" → *KServe*; "H00"/"H00s" → *H100*; "quen 3.8"/"Quen 3.8" → *Qwen 3.x*; "ambert 32" → a
ModernBERT-class encoder classifier (name uncertain, treated as approximate); "XRO / external proxy
plugin" → the Envoy external-processing (`ext_proc`) hook, which the Gateway API Inference Extension
uses; "EP that is the end point" → the EPP, Endpoint Picker. The llm-d talk says "GLM 5.2" where other
material in this corpus says GLM 5.1, and the speaker checks the model size on stage and is unsure;
the model version in that study is therefore reported here as **GLM 5.x** and not relied on. The
speaker's H200-1M / H100-250k context figures are given with an explicit "I need to check" and are
treated as unverified. One semantic-router figure — "40% of P50 routing latency we identify" — is
ambiguous in the transcript; its exact meaning is **not** relied on anywhere in this bank. No
precise configuration factorisation from any garbled passage is quoted.

**Derived content in this bank (`[D]`):** the hash-consistency failure walk-through in T14-Q5; the
event-plane blast-radius analysis and the "flat latency, rising cost" reading in T14-Q6 and Q26; the
queue-depth-versus-KV-pressure argument in T14-Q7; the cascade cost expression and its crossover
condition in T14-Q16; the retry-layer enumeration in T14-Q19; the gateway HA ordering in T14-Q20;
the `effective_cost` hit-rate arithmetic restated in T14-Q22; the priced-mix and both lever
calculations in T14-Q23 (case-study inputs, arithmetic shown); the debug order and its rationale in
T14-Q25; the instrumentation list and degradation ordering in T14-Q11 and Exercise 3; and the
per-tenant/contractual implications at 10× in T14-Q17 and T14-Q28. All are labelled inline where they
appear, with assumptions stated.
