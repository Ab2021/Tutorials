# Cross-Topic Scenarios: End-to-End Design Interviews

> **Transcript coverage:** primary · **Coverage:** T11–T19 · **Scenario count:** 14 · **Format:** situation → the question → topics in play → a sequenced decision path → the tradeoff being tested → red flags
> [T11 bank](T11-parallelism-moe.md) · [T12](T12-disaggregation-kv-transfer.md) · [T13](T13-serving-engines.md) · [T14](T14-routing-gateways.md) · [T15](T15-autoscaling-slo.md) · [T16](T16-agentic-inference.md) · [T17](T17-observability-evals.md) · [T18](T18-guardrails-security.md) · [T19](T19-finops-sovereignty.md)

## How to use this file

Each scenario is deliberately built so that **no single cheat sheet answers it**. The interview signal is
not whether the candidate knows the topics but whether they can decide *in what order* — what to
establish first, what that decision forecloses, and what must be measured before moving on.

Run one per session. Insist on the sequence: a candidate who lists every relevant technique without
saying what happens first has not answered the question. The `Topics in play` line is for the
interviewer's tracking, not to be read out; part of the exercise is whether the candidate finds the
relevant topics unprompted.

Where a scenario turns on a number, the number comes from the corpus and is attributed at the point of
use. Where an answer involves arithmetic of my own, it is marked `[D]` with its assumptions shown, in the
same convention as the topic banks.

---

## S1 · The cost mandate that must not regress quality

**Situation.** A platform team runs inference for an internal assistant: 40 requests/second at peak,
12-hour business-day concentration, a p95 TTFT SLO of 800 ms that they meet comfortably. The CFO has
issued a 50%-cost-reduction mandate for the next quarter with an explicit constraint: **no quality
regression**. The team's current setup is a single fp16 model on continuous batching behind a gateway with
round-robin routing. They have per-pod GPU metrics and nothing else.

**The question.** Give me your plan for the quarter, and tell me what you would refuse to do.

**Topics in play.** [T19](../00-cheat-sheets/T19-finops-sovereignty.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T13](../00-cheat-sheets/T13-serving-engines.md)

**Model answer — the decision path.**

**1. Establish what is measurable before optimising anything.** This is the first decision and it
forecloses the most. The mandate says "no quality regression," which is not a statement anyone can honour
without an eval suite that runs in CI — so the eval gate is not a nice-to-have, it is the *precondition*
for taking the quality-risky steps at all. Equally, per-request and per-tenant token accounting is what
makes a saving provable; the corpus is explicit that cost is a first-class observability signal rather
than an invoice reading `[T]`. Two weeks spent here buys the rest of the quarter.

**2. Apply the ladder in order, with the correct gate at each step.** Batching is already on, so the
`100 → 42` step is substantially banked and the team's real baseline is lower than they think — the first
arithmetic correction to make, because it sets the true target `[T]`. That leaves caching-plus-routing
(the 0.42 factor) and quantisation (0.62) `[T]`.

**3. Caching and routing before quantisation, and this is the load-bearing sequencing choice.**
Caching is where the cheap money is and it has *no quality dimension* — it changes what the model
processes, not what it produces. It also requires the round-robin routing to become cache-aware, because
a prefix cache is worthless if the request lands on a replica that does not hold it; the corpus's chosen
diagnostic is **prefix cache hit rate per pod** `[T]`. Measure that first. If hit rate is low, the finding
is that the application's harness rewrites its prefix every turn — an application change, not a platform
tuning knob.

**4. Quantisation last, gated, and measured on variance as well as quality.** Only now, because it is the
one step that can breach the mandate. And the eval must check both axes: the corpus notes quantisation
worsens the non-determinism you already have at temperature 0 `[T]`, so a quality pass with a variance
explosion is still a change to the team's debuggability and should be a recorded decision.

**5. Report cost per successful task, not cost per token.** This is the discipline that makes the mandate
honest. The corpus's Token Raj case is the reason: cost per 10k-token task fell from 30¢ in February to
16¢ in April, drafted lines fell from 630 to 91, files touched from 8.2 to 3.6, **and billing fell less
than the workload because thinking tokens rose** `[T]`. A token metric cannot detect that shape; a
successful-task metric can.

**What I would refuse to do:** apply all three levers simultaneously. With concurrent changes there is no
attribution, and when the eval eventually moves you will not know which step did it — which converts a
recoverable regression into an open-ended investigation. Also: reach for a smaller model. That is a
quality decision disguised as a cost decision, and it belongs in the eval conversation, not the
efficiency one.

**The tradeoff being tested.** Whether the candidate treats "no quality regression" as a constraint to be
verified or a phrase to be acknowledged. The right answer costs two weeks of measurement up front and the
wrong one saves those two weeks and risks the whole programme.

**Red flags.** Starts with quantisation because it is the visible technical win. Reports progress on the
token metric with no quality series alongside. Applies everything at once. Has no story for what happens
if the ladder exhausts before 50% — the correct answer is to report the frontier with the residual
attributed, not to degrade something nobody measured.

---

## S2 · The p99 incident during a traffic spike

**Situation.** It is 11:40 on a Monday. The assistant's traffic is 3× normal because a marketing campaign
launched. p50 latency is fine. **p99 TTFT has gone from 700 ms to 6 seconds** and the support queue is
filling. The dashboard shows GPU utilisation at 94% and nothing else red. The team's instinct is to add
replicas.

**The question.** You have ten minutes. What do you do, and what do you tell the incident channel?

**Topics in play.** [T15](../00-cheat-sheets/T15-autoscaling-slo.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T13](../00-cheat-sheets/T13-serving-engines.md)

**Model answer — the decision path.**

**1. Immediately: decide whether this is capacity or queueing, because the responses are opposite.**
p50 fine with p99 catastrophic is a *distribution* problem, not a uniform slowdown. Adding replicas helps
if the fleet is genuinely saturated; it makes things worse if the problem is that a subset of requests is
monopolising capacity, because new replicas join the same broken routing. Get the answer before acting:
what fraction of requests are slow, and are the same requests slow end-to-end?

**2. Look at the preemption and recomputation rate before the utilisation number.** The corpus documents
the specific mechanism: **a cliff at 28k inputs / concurrency 256 from KV recomputation** `[T]`. A server
at 94% utilisation with low preemption is healthy and busy; one at 94% with rising recomputation is
thrashing — requests are being evicted and re-prefilled rather than served. Those are the same
utilisation figure and completely different situations, and the fix for the second is not more GPUs.

**3. Check for a long-request tail.** A single very long context can consume the cache that shorter
requests need, and its effect is invisible in p50. The saturation thresholds the corpus names — **KV 80%,
active requests > 8** `[T]` — are headroom signals, and the reason a fleet at 94% with long requests
present breaches its tail is that the headroom those thresholds describe is gone.

**4. Check the routing and the cache together.** With spiky traffic, a load-balancing router spreads
requests across replicas and destroys prefix locality — every replica holds a partial cache and nothing
hits. The corpus's diagnostic is again **prefix cache hit rate per pod** `[T]`: if hit rate dropped when
traffic rose, the incident is a routing-induced cache collapse, and the response is routing policy, not
capacity.

**5. Then, and only then, scale.** Autoscaling on the right signal matters here. The corpus's guidance is
to scale on saturation signals rather than CPU, and the guide's own material is internally inconsistent on
this — one source specifies a CPU-based HPA manifest while another says scale on KV-cache utilisation
`[T]`. CPU on a GPU-bound workload is close to a lagging indicator with no causal relationship to the
binding constraint, so the KV signal is the one to use. Note also that scale-up on a KV-bound workload is
slow to take effect, because a new replica starts with a cold cache and must warm before it helps.

**6. Communicate the mechanism, not the symptom.** The incident channel gets: "p99 breach caused by KV
recomputation under a long-context tail with cache locality lost to load-balanced routing; mitigation is
routing policy change plus bounded concurrency for long requests; capacity increase is the second lever
and will lag by the cache warm-up."

**What I would fix afterwards:** the alerting. p99 should have paged before the support queue filled —
goodput, the fraction of requests meeting both TTFT and ITL targets, is the metric that would have caught
it `[T]`, because it moves when the tail degrades while the mean looks healthy.

**The tradeoff being tested.** Whether the candidate reaches for capacity reflexively. The reflex is the
most common wrong answer and it is expensive: it masks the real cause, costs money, and the incident
recurs at the next spike.

**Red flags.** Scales first. Reads utilisation as the answer rather than preemption rate. Cannot explain
why p50 is fine while p99 is not. Has no cache-locality story for a load-balanced router.

---

## S3 · The multi-tenant fairness complaint

**Situation.** A SaaS company serves 200 tenants from a shared inference fleet. Tenant A — the largest,
with a batch analytics workload — has started running many long sessions. Three smaller tenants have
complained that their interactive requests have become slow at unpredictable times, always coinciding with
Tenant A's batch runs. The platform owner's first instinct was to raise all limits, which "helped for a
week."

**The question.** Diagnose the fairness problem and propose a fix that does not just add capacity.

**Topics in play.** [T16](../00-cheat-sheets/T16-agentic-inference.md) · [T15](../00-cheat-sheets/T15-autoscaling-slo.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T18](../00-cheat-sheets/T18-guardrails-security.md)

**Model answer — the decision path.**

**1. Name the mechanism precisely, because the direction matters and teams get it backwards.** The corpus
documents a **least-attained-service** fairness problem in exactly this setting: **one large session
monopolises dispatch and starves short sessions** `[T]`. The victim is not the small tenant being
out-competed by volume; it is that a single long-running session, once admitted, holds scheduling
position and the short interactive requests queue behind it. That reframes the fix: the problem is not
Tenant A's *volume*, it is the granularity of the scheduling unit.

**2. Establish the measurement before the policy.** Per-tenant latency percentiles, not fleet-wide ones.
The fleet-wide p99 looked acceptable throughout, because the small tenants are a minority of requests —
which is exactly why this ran for weeks before anyone complained. This is the observability lesson from
S2 in a different guise: aggregate metrics hide exactly the populations most likely to be harmed.

**3. Fix the scheduling granularity first.** The corpus's own remedy is **session-level fairness using
least-attained service with turn priority** `[T]`, reported as delivering **2–3× latency improvement**
for the affected short sessions. The key property is that fairness is enforced *between sessions* rather
than between requests, so one session cannot accumulate dispatch share simply by issuing turns back to
back.

**4. Separate latency classes before considering per-tenant quotas.** Tenant A's batch work and the small
tenants' interactive work have different latency requirements — the corpus's agent material shows an agent
workload running **98% prefill / 2% decode** `[T]`, which is nothing like an interactive chat profile.
Serving them from one pool with one SLO wastes capacity on the class that does not need it and guarantees
the class that does need it will be starved at peak. Distinct pools with separate admission is usually
cheaper than any quota scheme.

**5. Only then consider limits, and prefer admission control over caps.** The "raise all limits" response
helped for a week because it moved the contention point rather than removing it. Bounded concurrency per
session — enforced structurally — is the control that actually prevents one session from monopolising
dispatch.

**6. The multi-tenancy question underneath.** Shared fleet means shared cache tiers and shared routing.
Before tightening anything, confirm the credential and cache scoping is per-tenant — the corpus's rule is
**per-tenant scopes, never a shared service account** `[D]`, and a content-keyed prefix cache that can
serve a hit across tenants is a leak as well as a fairness problem `[D]`. Fairness work that ignores
isolation can improve the latency numbers while leaving a worse problem in place.

**What I would tell the complainants:** not "we raised your limits" but "we changed how the scheduler
shares dispatch between sessions, and here is your p95 before and after."

**The tradeoff being tested.** Whether the candidate sees the problem as *capacity* (add more) or as
*allocation* (change the sharing rule). The instinct to buy the problem away is the expensive answer and
it does not fix the next tenant.

**Red flags.** Proposes per-tenant quotas as the first move. Cannot say whether the starvation direction
is large-session-vs-small or small-vs-large. Measures only fleet-wide latency. Ignores the cache and
credential isolation question.

---

## S4 · "We're being asked to serve a 1M-context model"

**Situation.** A product team wants to launch a document-analysis feature: upload a 1M-token corpus, ask
questions across it. They have quoted a competitor's pricing and want to know what capacity they need to
serve it in-house. Currently they run 16 GPUs across two 8-GPU nodes serving a 70B chat model, and they
believe the same fleet will work because "the model is the same size."

**The question.** What capacity do they actually need, and what will surprise them?

**Topics in play.** [T11](../00-cheat-sheets/T11-parallelism-moe.md) · [T12](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [T13](../00-cheat-sheets/T13-serving-engines.md) · [T19](../00-cheat-sheets/T19-finops-sovereignty.md)

**Model answer — the decision path.**

**1. Correct the premise first: the binding constraint is KV memory, not weights.** The model size is
unchanged, which is why the team concluded the fleet is unchanged — but a 1M-token context changes the
*state* per request by orders of magnitude while the weights are constant. This is the single most
important correction and it reframes everything downstream.

**2. Do the arithmetic, with assumptions stated.** The corpus's working formula is
`bytes ≈ tokens × top_k × hidden_dim × dtype_bytes × 2` for the MoE attention path `[D]`, and the general
shape for KV is `2 × layers × kv_heads × head_dim × dtype_bytes × tokens` per sequence `[D]`. Take a
representative 70B-class configuration — 80 layers, 8 KV heads, head_dim 128, fp16 `[D]`:

```
per token per sequence = 2 × 80 × 8 × 128 × 2 bytes ≈ 328 KB
at 1M tokens           ≈ 328 GB per sequence
```

That single number is the answer to the capacity question: **one concurrent 1M-token request needs more
KV memory than an entire 8-GPU node holds** at fp16 with 80–192 GB accelerators. The team's existing
fleet cannot hold *one* such request, let alone serve concurrency. Every input here is mine `[D]`; the
interviewer should substitute the real configuration and redo it, because the point is the method and the
shape of the answer, not the specific figure.

**3. Therefore the viable designs are a short list, and they are all about the state.** `[D]`

- **Reduce precision of the KV cache specifically.** 4-bit KV cuts the figure ~4× and is the standard
  first lever; it is also a quality decision requiring evaluation, so it belongs behind the same gate as
  weight quantisation (S1).
- **Disaggregate prefill and decode.** This is where the corpus's T12 material becomes load-bearing: a
  long-context request is overwhelmingly prefill, so separating prefill from decode lets each tier be
  sized and scaled for its own constraint, and KV has to be *transferred* between them `[T]`. That transfer
  is the new bottleneck, and it is a network design problem, not a GPU one.
- **Chunked prefill.** The corpus's own configurations pair long-context serving with chunked prefill so a
  single long prompt does not monopolise a scheduler step.
- **Retrieval instead of brute-force context.** The honest engineering answer, and the one the product team
  will resist: 1M-token contexts are the expensive way to answer questions over a corpus. Retrieval
  narrows the context and the cost falls by orders of magnitude — at the price of building retrieval and
  accepting that some questions will not be answerable from the retrieved subset.

**4. Then re-derive the parallelism, because the existing plan does not extend.** The corpus's own finding
is that naive parallelism loses to a tuned mix: a **naive single-host 8-way TP loses to a tuned
TP+PP+SP+EP combination across 16 GPUs per replica** `[T]`, and the guidance is that **TP stays inside the
node while EP and DP carry the wide dimension** — "TP stays 1" is *not* the rule `[T]`. For long context
specifically, sequence parallelism and the context-parallel variants matter because the attention
computation over a 1M-token sequence has to be split somewhere.

**5. State the honest cost, because the product team has a competitor's price.** Compute the cost per
request at the derived capacity and compare it to the quoted price. If the competitor is cheaper, the
likely reason is that they are using retrieval, a smaller model, or aggressive KV quantisation — and the
product decision is which of those tradeoffs to accept, not whether the fleet is big enough.

**What I would refuse to specify:** a GPU count. The number depends on the target concurrency, the
accepted KV precision, whether retrieval is in scope, and the latency SLO — and giving a count before
those are fixed would be a guess dressed as an answer. The deliverable is the memory arithmetic, the
design options with their consequences, and the questions that fix the count.

**The tradeoff being tested.** Whether the candidate identifies *state* rather than *weights* as the
capacity driver. A candidate who reasons about model size alone will produce a fleet that cannot serve a
single request, and will not know it until the load test.

**Red flags.** Sizes the fleet from parameter count. Treats the long context as a quality feature with no
memory consequence. Proposes the design without any memory arithmetic. Misses that prefill dominates and
therefore misses disaggregation as the natural architecture.

---

## S5 · The sovereignty constraint changes the topology

**Situation.** A European insurance company runs a claims-triage assistant. It uses a self-hosted
open-weights model in one EU region, tools that call core claims systems, a hosted judge model for
evaluation, traces in a third-party observability SaaS, and a hosted embedding API for retrieval. A
regulator has now said that claims data — including anything derived from it — must be processed and
stored within national borders. Legal wants an assessment in two weeks.

**The question.** What changes, and what does it cost?

**Topics in play.** [T19](../00-cheat-sheets/T19-finops-sovereignty.md) · [T18](../00-cheat-sheets/T18-guardrails-security.md) · [T12](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md)

**Model answer — the decision path.**

**1. Refuse the yes/no frame and produce a profile.** The corpus's own framing is that sovereignty is a
**system property**, not a flag, with dimensions of **control, choice, trust, economics and continuity** —
and the source itself notes "choice" is often folded into "control", so it is **four or five depending on
how strictly you split it** `[T]`. The useful output is a rating per dimension against the requirement,
because the requirement here attaches to *data flow*, which is not the dimension teams usually audit.

**2. Find the data flows, not the deployment location.** The team's instinct will be "we already deploy
in-region." But the corpus's central point is that **the inference layer itself is a sovereignty component
people forget** — "if inference leaves your control, can you really call that sovereign AI?" `[T]`. Here
three flows leave the boundary and none of them is the model:

- **The hosted judge model** sees claims text on the evaluation path. Evaluation is not "the production
  path," which is exactly why it gets missed, and it is nonetheless a processing flow.
- **The third-party observability platform** holds traces containing prompt text — claims data by any
  reasonable reading.
- **The embedding API** processes the retrieved corpus, which is claims data.

The supporting asymmetry is the corpus's: **"you train once… but inference is not like that"** `[T]`.
Inference is continuous and handles live data on every request, so it deserves more scrutiny than the
weights, not less.

**3. Fix in order of cost, and price the forfeiture explicitly.** `[D]`

- **Redact at the collector, before storage** `[T]` — the corpus's own remedy, and it is cheap. Traces can
  stay with the vendor if they carry no claims data; the cost is losing the prompt text you would want for
  debugging quality incidents.
- **Move the judge in-region.** The cost is GPU capacity for a second model and the operational burden of
  maintaining it. The alternative — evaluating on redacted samples — is cheaper and changes what you are
  measuring, which is a real and under-appreciated cost.
- **Replace or self-host the embedding API** if the retrieval corpus is in scope. This is the largest
  change, because it means re-baselining retrieval quality.
- **Constrain the KV cache to the region.** This is where the T12 material bites: if prefill and decode
  are disaggregated across a border, KV transfers cross it too, and a pooled memory tier is a data store
  `[T]`. Every one of those is a flow the requirement covers.

**4. Re-derive routing and capacity as constrained optimisation.** A cache-aware global router picks the
least-loaded replica; a router constrained to one jurisdiction cannot, so routing becomes
constraint-satisfying rather than cost-optimising (T14) `[D]`. The effective cache-hit rate falls, which
raises unit cost, and the capacity pool is smaller and less elastic, which may force an SLO renegotiation.

**5. Give Legal the number, not the assurance.** The organisation should see that the requirement has a
cost — the lost cache locality, the second GPU pool, the re-baselined retrieval — because an unpriced
constraint is one that gets quietly violated the first time there is budget pressure `[D]`. Separately,
establish whether this is genuinely mandatory: if it is, the cost is the cost; if it is an internal
preference, it is a cost decision and should be argued as one in the open.

**The tradeoff being tested.** Whether the candidate audits *data flows* or *deployment regions*. The
region is where the model is; the flows are where the data goes, and the judge and the trace store are the
two that always get missed because neither sits on the request path.

**Red flags.** Answers "yes, we're in-region." Audits the model and weights only. Claims compliance while
prompt text still leaves via the observability vendor. Presents the constraint as free.

---

## S6 · The agent's token bill grows faster than its usage

**Situation.** An agentic coding assistant launched eight months ago. Week-over-week active users are up
22%. The inference bill is up 340%. The team's dashboard shows cost per 1k tokens is down 18% over the
same period, which they have been citing as evidence that things are going well.

**The question.** What is happening, and what do you do about it?

**Topics in play.** [T16](../00-cheat-sheets/T16-agentic-inference.md) · [T19](../00-cheat-sheets/T19-finops-sovereignty.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T13](../00-cheat-sheets/T13-serving-engines.md)

**Model answer — the decision path.**

**1. Name the metric error first, because the team is describing it as success.** Cost per token falling
18% while spend rises 340% is not a contradiction — it is the corpus's Token Raj shape exactly: cost per
10k-token task fell from **30¢ to 16¢**, drafted lines fell from **630 to 91**, files touched from **8.2 to
3.6**, and yet **billing fell less than the workload because thinking tokens rose** `[T]`. Efficiency
genuinely improved; capability genuinely expanded; the bill genuinely rose. All three are true, and only
one of them is on their dashboard.

**2. Decompose the gap into the plausible causes and get the data to discriminate.**

```
bill = users × tasks/user × steps/task × tokens/step × price/token

usage (users)                up 22%
price/token                  down 18%
=> the remaining 3.6x must live in tasks/user, steps/task or tokens/step
```

`[D]`. That is the whole investigation, and it is arithmetic: with price down 18%, the multiplicative
product of the other three terms must be up roughly **4.4×** `[D]`. The candidate should be able to say
this in the first two minutes, because it tells the team which three metrics to go and plot.

**3. Check steps per task first — it is the agent-specific term and the earliest indicator.** The corpus
documents a **10–100× agentic compute multiplier** over a single chat turn and a failure signature where a
loop does not terminate `[T]`. A modest degradation in the agent's efficiency — a tool that got slower, a
prompt change that added a retry, a model upgrade that reasons more — multiplies across every step. Steps
per task plotted daily will show a change immediately, and it is the single highest-value series the team
does not have.

**4. Check tokens per step, and specifically the input side.** Agents resend accumulating context every
turn, so input tokens grow within a session while output stays small — the corpus's agent profile is **98%
prefill / 2% decode** `[T]`. The likely findings are context growth from a new retrieval step, larger tool
definitions, or history that is no longer being pruned. Note that this interacts with caching: every one
of those changes can also break the prefix, so the cost effect is larger than the token count suggests.

**5. Check the cache before concluding anything about the workload.** The corpus's diagnostic is **prefix
cache hit rate per pod** `[T]`. If a prompt change moved something volatile into the prefix, hit rate
falls and the same workload costs materially more — with no change in steps or tokens. This is the
cheapest thing to check and the most commonly the answer.

**6. Check retries and failures.** Failed tasks consume tokens and produce nothing. This is where cost per
*successful* task earns its place — a decline in success rate turns one billable task into two or three,
invisibly if you only watch cost per token.

**7. Then the fix, which is controls rather than optimisations.** Structural budgets on the agent loop —
step, token and wall-clock caps, enforced rather than alerted, because by the time an alert fires the
money is spent `[T]`. Session reuse, which the corpus reports gives **roughly 5× TTFT improvement on a
session's return** `[T]` and correspondingly avoids re-prefilling. And a per-task cost attribution with a
task ID propagated through the trace, so the next time this happens the diagnosis takes an hour rather
than a quarter `[D]`.

**8. Reframe what "good" means.** The right headline is cost per successful outcome, not cost per token
`[D]` — and if the decomposition shows the growth is driven by users doing more work per task, that is a
product success being reported as a cost failure.

**The tradeoff being tested.** Whether the candidate recognises the Token Raj shape and whether they
investigate *before* optimising. A team that reacts by squeezing the serving stack will make the tokens
cheaper and watch the bill keep rising, because the driver is not on the serving side.

**Red flags.** Accepts the cost-per-token narrative. Proposes quantisation or a smaller model as the fix
without decomposing. Has no steps-per-task metric. Cannot distinguish reinvestment from leakage.

---

## S7 · Doubling the GPUs halved the throughput per GPU

**Situation.** A team was serving a large MoE model on one 8-GPU node at a throughput they considered
acceptable. To meet demand they added a second node and configured the replica across all 16 GPUs using
the same tensor-parallel setting as before. Aggregate throughput went up, but **throughput per GPU fell
by about 35%** and inter-token latency got worse. They are asking whether 16-way TP has a bug.

**The question.** What happened, and what would you configure instead?

**Topics in play.** [T11](../00-cheat-sheets/T11-parallelism-moe.md) · [T12](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [T13](../00-cheat-sheets/T13-serving-engines.md) · [T19](../00-cheat-sheets/T19-finops-sovereignty.md)

**Model answer — the decision path.**

**1. Not a bug — the communication surface changed shape.** Going from intra-node to cross-node
parallelism moves collectives from a fast intra-node fabric onto the network, so every all-reduce in the
critical path now costs materially more. The corpus's own finding is that this is the classic mistake: a
**naive single-host 8-way TP loses to a tuned TP+PP+SP+EP mix across 16 GPUs per replica** `[T]`, and the
team has effectively scaled the *wrong* dimension.

**2. State the rule correctly, because the folk version is wrong.** The folk version is "TP stays 1."
The corpus's actual position is that **TP stays inside the node** — the exact statement is that TP stays 1
is *not* the rule, and the same team the corpus draws on runs **TP8 + 2P2D + EP8 + DP16** `[T]`. So TP
does not have to be 1; it has to fit within the node's interconnect. What carries the wide dimension is
**EP and DP**, not TP.

**3. Get the sizing arithmetic out, because the MoE structure is what makes the alternative available.**
`experts_per_GPU = total_experts / EP_degree` `[D]`. A model with 256 experts at EP8 gives 32 experts per
GPU; at EP32 across 4 nodes it gives 8 `[T]`. The MoE architecture is precisely what makes it possible to
shard the *experts* across many devices while keeping attention local — which is why expert parallelism
and data parallelism are the natural way to scale out, and TP is not.

**4. Recognise the cost structure of EP too.** Expert parallelism is not free: it introduces all-to-all
communication. The corpus's concrete example is **naive MoE with 2 all-to-all operations plus 6 kernels
reduced to 3 kernels** via a top-k permute → grouped GEMMs → unpermute → reduction/scale pipeline `[T]`.
So the recommendation is not "use EP everywhere" but "use EP with the kernel-level implementation that
keeps the all-to-all from dominating" — and that is an engine-capability question, which is why the corpus
flags **AMD-versus-NVIDIA WideEP enablement gaps and pending PRs** `[T]` as a real constraint on which
topology is available to you.

**5. Consider the wider dimension: prefill/decode disaggregation.** The team's symptom — throughput per
GPU down and ITL worse — often also indicates that prefill and decode are interfering. The corpus reports
the same team running **2P2D** in the reference configuration `[T]`, and its long-context material shows a
**cliff at 28k inputs / concurrency 256 from KV recomputation** `[T]`, which is what happens when prefill
traffic evicts decode state. Note that the corpus flags a **2P4D configuration as preliminary and not to
be trusted** `[T]` — so the candidate should propose 2P2D and not 2P4D, and should say why.

**6. Be honest that there is no universal winner.** The corpus's own line is that **there is no universal
winner** `[T]`, and a candidate who confidently prescribes one factorisation without asking about the
fabric, the engine version, the model's expert count and the workload mix is guessing. The right answer is
the search space plus the method for choosing within it.

**What to configure instead, as a starting point:** keep TP inside the node (2 or 8, per the model's
attention structure), add pipeline or sequence parallelism for the layers, enable **expert parallelism**
for the MoE dimension, and use **data parallelism** to scale replicas. Then measure — because the
headline result the corpus reports is that this tuned mix beats naive 8-way TP across the same 16 GPUs
`[T]`.

**The tradeoff being tested.** Whether the candidate knows that TP is a *within-node* technique and that
MoE architectures offer an orthogonal axis. The naive answer is "TP has a bug" or "add more nodes with the
same config."

**Red flags.** Prescribes tensor parallelism across nodes. Says "TP must always be 1." Proposes 2P4D
without noting the corpus marks it preliminary. Cannot produce the `experts_per_GPU` arithmetic. Assumes
EP is free.

---

## S8 · The model upgrade that passed evals and broke production

**Situation.** A team upgraded their serving model to a new version. The eval suite — 400 golden cases,
all passing — showed a 3% quality improvement. They rolled out to 100% over a weekend. By Wednesday,
support had a cluster of complaints about a specific class of question: multi-turn follow-ups where the
user refers back to something said several turns earlier. The eval suite has no multi-turn cases.

**The question.** What went wrong, and what is the process that would have caught it?

**Topics in play.** [T17](../00-cheat-sheets/T17-observability-evals.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T13](../00-cheat-sheets/T13-serving-engines.md) · [T15](../00-cheat-sheets/T15-autoscaling-slo.md)

**Model answer — the decision path.**

**1. The immediate mistake is the rollout, not the model.** A 100% cutover over a weekend on the strength
of an offline eval removed the organisation's ability to detect a problem before users did. The
corpus's framing is that evals are a gate in CI, but a gate before *merging* is not the same as a gate
before *shipping to all traffic*. The mechanism that would have caught this is a canary with online
quality signal — sampled judge scoring and user feedback on the changed slice — rather than an offline
suite alone.

**2. The deeper mistake is the eval set, and the corpus names this failure exactly.** The signature is
**"eval passes, users unhappy"**, attributed to the eval set being **too easy or off-distribution**, with
the fix being to **add hard cases and real traffic samples** `[T]`. A 400-case suite with no multi-turn
cases is not measuring the workload; it is measuring a proxy that happens to correlate on the easy
dimension. This is the correct diagnosis and it is where most of the answer should live.

**3. The third mistake is the missing feedback loop.** The complaint cluster took three days to surface
because there is no path from production traces to the eval set. The corpus's structure puts trajectory
and online evaluation as separate planes from offline eval precisely because production is where the
distribution lives `[D]`.

**4. The diagnosis method, if you were arriving on Wednesday without knowing the cause.** Decompose by
traffic slice rather than looking at the aggregate. Slice by turns-per-conversation, by request class, by
route — the last one matters because a partial rollout leaves two models serving simultaneously, so a
quality difference by replica is diagnostic. The corpus's routing material identifies per-replica
attribution as a trace attribute worth carrying `[T]`, and this is the case that justifies it.

**5. The process fix, in three parts.**

- **Eval set derived from production.** Sample real traffic into the suite on a schedule, weighted toward
  the hard slices. The corpus's guidance is **1–5% evaluation sampling** and **10% drift threshold** `[T]`,
  which gives the cadence. A suite built only from curated examples will always be easier than production.
- **Canary plus online signal.** Never 100% on offline evidence alone. Because outputs are not
  deterministic even at temperature 0 `[T]`, and quantisation worsens that `[T]`, the online signal has to
  be a sampled judge and user feedback rather than an equality check.
- **Model-version as a first-class dimension in the trace.** `gen_ai.request.model` and the replica are
  both attributes you want `[T]`, so a regression can be attributed to a version rather than to "the
  system."

**6. What to check before rolling back.** Whether the regression is quality or something else. A new model
version can change output *length*, which changes latency and cost; can change prompt-format sensitivity,
which interacts with the prefix cache; and can change tokenisation, which changes the effective context
budget. If the complaint were about latency or cost rather than correctness, the same decomposition
applies to a different metric.

**The tradeoff being tested.** Whether the candidate treats an offline eval pass as sufficient evidence
for a full rollout. The eval suite's coverage is the actual question, and the corpus provides the exact
failure signature for a suite that is too easy.

**Red flags.** Blames the model. Proposes adding multi-turn cases without addressing how cases get into
the suite in future. Rolls back without a diagnosis. Has no canary or online-quality mechanism.

---

## S9 · The agent that emailed the customer list

**Situation.** A support agent reads tickets, searches a knowledge base, and can email customers and issue
refunds up to £50. On Tuesday it sent an email containing a list of 40 customer records to an external
address. The triggering content was in a support ticket: a paragraph instructing the agent to forward the
list. The ticket was filed by an unauthenticated user through the public contact form. The agent's email
tool has its own allowlist of domains, and the external address was on it because the sales team had
added it months earlier.

**The question.** Contain it, then fix it so it cannot recur.

**Topics in play.** [T18](../00-cheat-sheets/T18-guardrails-security.md) · [T16](../00-cheat-sheets/T16-agentic-inference.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md)

**Model answer — the decision path.**

**1. Containment before analysis.** Revoke the email tool's credential and the domain allowlist entry;
identify the recipients of the disclosure; preserve the trace and the ticket so the audit record survives.
Do not rotate logs or clean up before the record is secured — the corpus's governance requirement is a
decision-level audit trail per tool invocation recording principal, scope, approval and outcome `[T]`, and
that is what incident response and any notification obligation depends on.

**2. Then name the attack correctly, because the label determines the fix.** This is **indirect prompt
injection**: the payload arrived through content the agent was *told to trust*, which is the corpus's
precise formulation and the reason it is the dominant agent-era attack `[T]`. It is not a jailbreak. The
attacker never had an account; they used a public form. That means user-side controls — rate-limiting the
user, reviewing their account, blocking their session — address nothing.

**3. Work the chain and find where it actually broke.** The corpus's chain is: untrusted source → enters
context as data → model treats it as instruction → emits a tool call → executes with the agent's
credentials `[T]`. In this incident the breaks are at the last two:

- **The tool-call rail did not stop it, and could not have.** The call was a permitted tool, a valid
  schema, and a permitted recipient. This is the T18 bank's structural gap: the rail cannot distinguish a
  valid call from an appropriate one, because appropriateness depends on intent and intent is not in the
  arguments. The candidate should say this plainly rather than proposing a better rail.
- **The credential scope did not bound it.** The agent's mail scope could reach the customer list and an
  external domain. That is the load-bearing failure: **blast radius = capability × reach ×
  reversibility** `[D]`, and here all three were large.

**4. The four rails, and which two were missing.** Input rail existed (the ticket came through the public
form, so it *was* the user input — a rail there is weak by construction). Retrieval rail: absent — nothing
tagged the ticket content as untrusted, and there is no instruction/data separation in the context to
carry that tag except in-band `[T]`. Tool-call rail: present but structurally insufficient. Output rail:
absent — nothing checked that the outgoing email contained a bulk customer list. The corpus's observation
is that teams implement the two chatbot rails and skip the middle two `[T]`; here the retrieval rail was
skipped and the tool-call rail was present but not backed by capability limits.

**5. Fixes, ordered by what they actually bound.**

- **Narrow the credential scope.** The agent's mail capability should be limited to the ticket it is
  working, and it should not be able to enumerate the customer list at all. This is the fix that would have
  prevented this incident, and it is the one the corpus points at: **prompt instructions do not reduce
  blast radius**, but credential scope does `[D]`. Re-derive the scope from the task: the agent needs to
  reply to *this* customer, not to send arbitrary mail to allowlisted domains.
- **Make the domain allowlist a per-task decision, not a standing global one.** A sales-team addition made
  months ago was still active on a support agent.
- **Add the retrieval rail.** Tag ticket content as untrusted in-band; strip obvious payload carriers.
  Both are mitigations rather than controls, and should be described as such.
- **Human approval for the irreversible or high-consequence actions.** Refunds are already bounded at £50;
  bulk outbound mail should be treated similarly — the corpus's guidance is approval for **irreversible**
  actions `[T]`, and an externally-sent email is irreversible.
- **Run the confused-deputy check.** Does the agent hold a permission a user does not? An agent that can
  enumerate the full customer list but whose users cannot is the escalation path by definition `[D]`.

**6. The honest residual.** A determined injection can still induce a permitted action to a permitted
recipient. The candidate should say this rather than promising it cannot recur — and should note that the
audit trail is what makes the next occurrence detectable in minutes rather than on a customer complaint.

**The tradeoff being tested.** Whether the candidate reaches for a guardrail model or for capability
limits. The guardrail answer — "we'd add an injection detector" — does not address the incident, because
the payload was a plausible paragraph and the call was valid. The capability answer does.

**Red flags.** Calls it a jailbreak and proposes user-side controls. Proposes a better prompt. Treats the
tool-call rail as the fix. Promises it cannot happen again. Rotates the logs before securing the record.

---

## S10 · The agent's TTFT budget is blown by the tool loop

**Situation.** An agent product has an SLO of p95 first-token-in-2-seconds. Measured against the inference
fleet, the serving tier meets it easily — p95 TTFT is 340 ms. But user-perceived time-to-first-useful-
output is p95 14 seconds, and the product team has been escalating to the platform team for weeks because
the dashboard is green.

**The question.** Where is the time going, and whose problem is it?

**Topics in play.** [T16](../00-cheat-sheets/T16-agentic-inference.md) · [T12](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T13](../00-cheat-sheets/T13-serving-engines.md)

**Model answer — the decision path.**

**1. Resolve the metric conflict first, because it is the actual source of the escalation.** "TTFT" means
two different things to the two teams. To the serving tier it is the time from request arrival to first
token — 340 ms, green. To the user it is the time from their message to the first *useful* output, which
for an agent is after the model has decided which tools to call, called them, and received results. The
corpus's latency decomposition is the tool that settles it:

```
end_to_end = queue_wait + prefill(TTFT) + N × ITL + Σ tool_time + retries
```

`[T]`. Every span should map onto one of those terms, and **if a span does not map to a term, the
instrumentation is missing something** `[T]`. Here the missing term is `Σ tool_time`, and it is almost
certainly most of the 14 seconds.

**2. Get the trace, not another dashboard.** The corpus's failure signature for this exact situation is
**"cannot explain a slow request — missing spans between gateway and engine"**, with the first check being
trace propagation `[T]`. The platform team has metrics; they need spans. A single end-to-end trace with
tool spans will end the argument in five minutes.

**3. Expect the answer to be tool latency, and quantify it.** A plausible shape `[D]`: three sequential
tool calls at 1.5–3 s each, plus 340 ms prefill and a few hundred ms of decode, gives 6–11 s — the right
order of magnitude. The corpus's T17 worked example shows the same pattern with **1.2 s prefill, 90 ms
decode, and over 1 s of tool time** `[T]`, so tool time dominating is the normal case, not the anomaly.

**4. Then the fixes, which are mostly not platform fixes.** `[D]`

- **Parallelise independent tool calls.** Three sequential calls become one round. This is usually the
  single largest win and it is an agent-harness change.
- **Stream intermediate progress.** The user-visible metric is time-to-first-useful-output, and an agent
  that reports "searching the knowledge base…" changes perceived latency without touching real latency.
  This is a product change and it is often the cheapest fix available.
- **Reduce round trips.** Fewer, coarser tools beat many fine-grained ones — which also improves prefix
  cacheability, because tool definitions are part of the prefix `[T]`.
- **Session reuse.** The corpus reports **roughly 5× TTFT improvement on a session's return** `[T]` from
  keeping sessions warm, which matters here because agents are described as **"idle 99.999% of the time"**
  `[T]` — the session sits idle between turns while the KV that would make the next turn fast is evicted.

**5. Then the platform-side items, correctly scoped.** Once tool time is addressed, there is a real
serving question: an agent workload is **98% prefill / 2% decode** `[T]`, so it is TTFT-bound rather than
ITL-bound, and the serving tier should be configured and scaled for that. Prefill/decode disaggregation
helps because prefill dominates and the two tiers have different bottlenecks `[T]`; KV has to be
transferred between them, which is the T12 design problem. But this is the *second* conversation.

**6. The SLO itself needs rewriting.** A per-request TTFT SLO is the wrong contract for an agent, because
one user turn generates many model calls. The right SLO is per user *turn*, decomposed into its terms, and
owned jointly — the platform team owns the prefill and decode terms, the application team owns tool time
and retries. Right now each team owns half a metric and neither owns the one the user experiences.

**The tradeoff being tested.** Whether the candidate separates serving latency from end-to-end latency and
whether they can assign ownership. The failure mode here is a platform team optimising a term that is
already 2% of the experience while the product team waits.

**Red flags.** Proposes scaling the fleet. Accepts the green dashboard. Cannot write the decomposition.
Treats the 14 seconds as a serving problem without a trace.

---

## S11 · The cache hit rate collapsed after the gateway migration

**Situation.** A team replaced their ad-hoc load balancer with a proper inference gateway. Throughput
improved, error rates fell, and operational visibility was much better. Two weeks later, finance reported
the inference bill was **up 38%** on flat traffic. The serving metrics are unchanged — same model, same
quantisation, same batch sizes. The only visible change is the gateway.

**The question.** Explain the increase and fix it without reverting the migration.

**Topics in play.** [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T19](../00-cheat-sheets/T19-finops-sovereignty.md) · [T13](../00-cheat-sheets/T13-serving-engines.md) · [T17](../00-cheat-sheets/T17-observability-evals.md)

**Model answer — the decision path.**

**1. The suspect is routing, and the corpus names the exact signal.** **Prefix cache hit rate per pod is
the signal that exposes bad routing** `[T]`. Before the migration, the load balancer probably had
sufficient session affinity — however incidentally — that repeated prefixes landed on the same replica.
A well-built gateway that balances for load spreads them across all replicas, and every replica holds a
partial cache that nothing hits.

**2. Quantify it before changing anything.** Prefix caching is a **0.42 factor** on the cost ladder, and
cache reads are **roughly 10× cheaper** than fresh processing `[T]`. Going from a high hit rate to near
zero means paying full prefill on every request, which is comfortably the right order of magnitude for a
38% bill increase `[D]`. The candidate should say the mechanism produces an effect of the observed size —
that is what makes the diagnosis credible rather than a guess.

**3. Fix it with cache-aware routing, not with affinity alone.** The temptation is to re-enable sticky
sessions. That works and it is the wrong fix, because session affinity and cache affinity are different
requirements: a session may legitimately move, and a prefix may be shared across many sessions. The
correct routing decision is "which replica most likely holds this prefix," which means the router needs to
know cache state — the corpus's own framing is that the router consumes per-request create *and* evict
events, with an offload tier and a retention API `[T]`. A router that only balances load cannot do this;
one that observes cache lifecycle can.

**4. Then check whether the harness changed too.** The migration often coincides with prompt-template
changes — and the corpus's reminder is that **tools are part of the prefix**, so adding or reordering a
tool invalidates the cache for every subsequent turn `[T]`. If the gateway rollout also introduced
per-request metadata into the prompt, the hit rate would have collapsed regardless of routing. Check the
prompt before blaming the router.

**5. Check what the gateway added to the request path.** Gateways do useful things that cost tokens:
injecting a system prompt, adding retrieved context, appending tool definitions, adding a routing
directive. Each of those is an input token increase, and if it sits *early* in the prefix it defeats
caching for the whole request. This is a common and quiet cause of a bill increase after a gateway
rollout `[D]`.

**6. Measure the fix with the right metric.** Prefix cache hit rate per pod, plotted over the migration
date, is the single chart that proves the diagnosis and proves the fix. If the candidate proposes any
other metric as the validation, they have not internalised the corpus's diagnostic.

**7. Note the governance angle.** The pre-migration load balancer was accidentally providing a
cost-relevant behaviour. That is worth recording — routing has cost consequences that are easy to lose
when a component is replaced for reasons of observability or reliability.

**The tradeoff being tested.** Whether the candidate knows that routing and caching are one mechanism
rather than two. A candidate who treats the gateway as cost-neutral will look for the increase in
quantisation, model version or batch size, and find nothing.

**Red flags.** Attributes the increase to the model or quantisation. Proposes reverting the migration.
Suggests sticky sessions without noting the difference between session and cache affinity. Cannot name
the diagnostic metric.

---

## S12 · The multi-region failover that will not meet its RTO

**Situation.** A regulated workload must survive the loss of a region. The stated requirement is an RTO
of 2 minutes and an RPO of zero. The serving design is prefill/decode disaggregated within a region, with
a KV offload tier and a prefix cache, and a global router that sends requests to the least-loaded healthy
region. There is no cross-region replication of cache state.

**The question.** Will this meet the requirement, and what has to change?

**Topics in play.** [T12](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [T15](../00-cheat-sheets/T15-autoscaling-slo.md) · [T14](../00-cheat-sheets/T14-routing-gateways.md) · [T19](../00-cheat-sheets/T19-finops-sovereignty.md)

**Model answer — the decision path.**

**1. Separate the two requirements, because only one of them is a serving problem.** RPO of zero is about
**state**, and the state here is the conversation and the KV that represents it. RTO of 2 minutes is about
**capacity being available**, which on a GPU fleet means pre-warmed capacity that is provisioned and
idle. The candidate should not answer them together.

**2. Test RPO first, because it fails immediately.** With no cross-region replication, losing a region
loses the in-flight sessions and any cache state. But the more interesting question is what "zero data
loss" means for an agent: the durable session log is the state of record (the corpus's agent design uses a
stateless loop with a durable session log `[T]`), and KV caches are derived state that can be recomputed.
So RPO zero may be satisfiable by replicating the *log* while letting the KV cache be rebuilt — at a
latency cost, not a data cost. That distinction is the whole answer, and a candidate who conflates the
cache with the record will over-engineer the solution.

**3. Test RTO second, and expect it to fail on warm-up.** 2 minutes is plausible for routing traffic to a
healthy region. It is not plausible for a *cold* region to absorb the load: GPU capacity is not instantly
provisioned, replicas start with cold caches, and the corpus's long-context material shows how quickly a
cold fleet degrades — the **cliff at 28k inputs / concurrency 256 from KV recomputation** `[T]` is what
happens when cache capacity is under pressure. So the RTO is achievable only with warm standby capacity
in the second region, which is a cost decision made in advance and cannot be made during the incident.

**4. The KV transfer question, which is where T12 becomes load-bearing.** Disaggregating prefill and
decode within a region already means KV is transferred between tiers over a fast fabric. Extending that
across regions means transferring KV over a WAN link — latency and bandwidth that are an order of magnitude
worse. Whether that is viable depends on the model's KV size per session, which brings in the memory
arithmetic: `2 × layers × kv_heads × head_dim × dtype_bytes × tokens` per sequence `[D]`. For a long
session this can be tens of gigabytes, and the transfer time at WAN bandwidth may exceed the RTO on its
own `[D]`. The honest conclusion is usually: **do not plan to move KV across regions; plan to recompute
it from the durable log.**

**5. The router's behaviour under failover, which is the part that gets skipped.** A global router that
routes to the least-loaded healthy region has to detect unhealthiness and shift load. The failure modes to
design for: **flapping**, where a region goes healthy and unhealthy repeatedly and load oscillates;
**thundering herd**, where all traffic moves at once and the surviving region's autoscaler cannot respond
fast enough because GPU capacity is not elastic on a 2-minute timescale; and **state mismatch**, where
sessions continue in the new region without their history. The autoscaling design (T15) has to account for
capacity that cannot be added quickly, which means the survivable design is over-provisioned standby
rather than reactive scaling.

**6. What I would tell them to change, in order.** `[D]`

- **Replicate the durable session log synchronously**, so RPO zero is actually met, and accept that KV is
  recomputed. This is the design decision that makes the rest tractable.
- **Hold warm standby capacity** sized to absorb the full load, accepting the cost — this is the price of
  a 2-minute RTO and there is no version of it that is cheap.
- **Make the router's failover policy explicit**: health criteria, hysteresis to prevent flapping, and a
  staged shift rather than an instant one to protect the surviving region.
- **Test it**, by actually killing a region during business hours. A failover design that has not been
  executed is a design, not a capability — and the first execution will discover the router's health check
  is wrong, which is what you want to learn in a test.
- **Re-examine the RTO with the business.** If the true cost of 10 minutes rather than 2 is small, the
  cost difference is large. This is an SLO conversation with a price attached, and it should be had before
  the standby fleet is bought.

**The tradeoff being tested.** Whether the candidate separates state from capacity, and whether they
recognise that KV is derived rather than authoritative. The instinct to replicate everything produces the
most expensive possible design.

**Red flags.** Treats RTO and RPO as one requirement. Plans to replicate KV across regions without doing
the bandwidth arithmetic. Assumes reactive autoscaling can serve as failover. Proposes instant traffic
shift with no hysteresis. Has no plan to test the failover.

---

## S13 · The reasoning upgrade that tripled the bill

**Situation.** A team replaced their model with a reasoning variant. Accuracy on the eval suite went from
71% to 88% — a genuine, verified improvement on hard cases. Two weeks later the unit economics look bad:
cost per task is up 2.8×, p95 latency is up 4×, and the fleet is at capacity during peaks in a way it was
not before. Product wants the accuracy and cannot have the cost. Engineering has been asked to "optimise
the model."

**The question.** What are the actual options, and which would you choose?

**Topics in play.** [T19](../00-cheat-sheets/T19-finops-sovereignty.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T13](../00-cheat-sheets/T13-serving-engines.md) · [T15](../00-cheat-sheets/T15-autoscaling-slo.md)

**Model answer — the decision path.**

**1. State clearly that "optimise the model" is not available.** You cannot quantise a reasoning model into
being cheap without changing the thing you upgraded for — and quantisation would require re-running the
eval that justified the upgrade in the first place `[T]`. The realistic levers are about *which requests
get the expensive model*, not about making the expensive model cheaper. Getting the framing right in the
first minute is most of the answer.

**2. Decompose where the extra cost actually is.** Reasoning models emit long chains of thinking tokens,
so the increase is on the **output** side, unlike most inference cost which is input-dominated. That has
two consequences: output tokens are typically priced higher than input tokens, and the p95 latency
increase is a direct consequence of generating more tokens, not of a serving regression. Both matter for
choosing the fix.

**3. Then apply the classic answer: routing.** Send easy requests to the cheap model and hard ones to the
reasoning model. This requires a **classifier or heuristic for difficulty**, and the corpus's routing
material covers the mechanism `[T]`. The design questions are: what is the accuracy on the easy slice, and
what fraction of traffic is genuinely hard? If hard cases are 20% of traffic and the cost multiplier is
2.8×, routing gives roughly a 1.36× blended increase instead of 2.8× `[D]` — the accuracy is preserved
where it matters and the cost is contained.

**4. The risk to watch: silent quality regression on the easy slice.** Routing is a quality decision
disguised as a cost decision. The eval must be re-run split by slice, because a blended number hides the
easy slice degrading. If accuracy was 71% → 88% overall and the routing policy sends everything the
classifier calls easy to the old model, the easy-slice accuracy is whatever it was before — which may be
fine, or may not.

**5. Consider budgeted reasoning instead of routing.** Many reasoning models allow a thinking budget.
Bounding it converts cost from an emergent property of the model into a tunable parameter, and it is
usually the cheapest change available because it requires no routing infrastructure. The tradeoff is
that you are truncating reasoning, so it must be evaluated the same way a quantisation would be.

**6. Reconsider the SLO, because reasoning models are not interactive.** A 4× p95 latency increase may be
acceptable for some surfaces and not others. The corpus's guidance is that different request classes can
hold different targets, and that agents already run at a very different latency profile from chat — the
agent profile is **98% prefill / 2% decode** `[T]`, and a reasoning model pushes in the other direction
with long decode. If the workload is mixed, a tiered SLO is cheaper than one fleet sized for the worst
case.

**7. Fix the capacity problem separately, because it may be transient.** The fleet at capacity during
peaks is partly a function of longer requests occupying slots for longer. Batching helps less when
sequences are long, because each occupies KV memory for the duration. The options are more capacity,
tighter admission control on the reasoning tier, or accepting a lower SLO for non-interactive requests
during peaks.

**8. What I would actually recommend:** reasoning budget first (cheapest, no infrastructure), then routing
if the easy slice is meaningful, and a slice-disaggregated eval to prove neither cost anything in quality.
And I would ask whether the *product* can express which requests deserve the expensive model — because the
best classifier is usually the user's own action, not a model.

**The tradeoff being tested.** Whether the candidate treats this as a serving optimisation or a
policy/allocation problem. There is no serving configuration that makes reasoning free; there is a routing
policy that spends it only where it pays.

**Red flags.** Proposes quantising the reasoning model. Suggests a smaller model without the slice-level
eval. Reports blended accuracy. Tries to make it interactive with the same latency SLO.

---

## S14 · The guardrail that broke the latency SLO

**Situation.** Following a security review, a team added four guardrails to their agent: an input
classifier, a retrieval-content scanner, a tool-call validator, and an output filter. Each is implemented
by a model call to a small hosted model. Security signed off. Two weeks later p95 latency has gone from
1.8 s to 4.6 s, the SLO is breached, and the product team is asking to remove them.

**The question.** Neither party is wrong. What do you do?

**Topics in play.** [T18](../00-cheat-sheets/T18-guardrails-security.md) · [T15](../00-cheat-sheets/T15-autoscaling-slo.md) · [T17](../00-cheat-sheets/T17-observability-evals.md) · [T13](../00-cheat-sheets/T13-serving-engines.md)

**Model answer — the decision path.**

**1. Establish that the guardrails are the right four and in the right positions, before optimising them.**
The corpus's model is that guardrails belong at **four positions — input, retrieval, tool-call, output** —
and that teams implement the two chatbot rails and skip the middle two `[T]`. Here all four are present
and the middle two are the ones with real consequences behind them, so the set is correct. Removing them
is not on the table, and the candidate should say so before proposing anything: this is a *cost* problem,
not a correctness problem.

**2. Decompose the added latency before defending or attacking anything.** 2.8 s of added latency across
four rails is 700 ms each, and that is implausible for small classifiers unless they are being called
serially in the critical path. The first question is not "which rail is unnecessary" but "which rail is on
the critical path." The input and tool-call rails are necessarily in the request path. The retrieval rail
runs after retrieval and before generation — also in the path, but it can be **parallelised with other
pre-generation work**. The output rail may be able to run **concurrently with streaming** rather than
blocking the response `[D]`.

**3. The largest structural win: don't run all four on all requests.** The corpus's own layering is that
deterministic checks come before model-based ones, and that a cheap classifier can gate an expensive one.
Concretely, most requests are uninteresting — a short question with no retrieval and no tool call. A
capability-based triage: the tool-call rail only runs if there is a tool call (which is often absent), the
retrieval rail only runs if there was retrieval, the output rail only if the output matches a
risk pattern. That removes whole rails from most requests without removing any control `[D]`.

**4. Then: batch the rails against the generation.** A rail implemented as a model call can be batched
with other work rather than serialised. The corpus's serving material on continuous batching applies
directly `[T]` — the same batching win that makes generation cheap makes classifier calls cheap, provided
they are not being issued one-at-a-time synchronously from the request handler.

**5. Then: choose the right implementation for each rail.** The corpus documents a **layered judge
architecture** — a distilled model inline, a frontier model on a 1–5% sample, and human gold — reported
with **~97% lower cost and ~10× lower P50** than running the frontier model inline, at **88–92% agreement**
`[T]`. The same shape applies to rails: a small distilled classifier inline is the correct implementation
for a rail that must run on every request. If the team is calling a hosted frontier model synchronously
four times per request, that is the bug.

**6. Then: reconsider whether model-based rails are needed where deterministic ones suffice.** Schema
validation, path containment, URL allowlisting, recipient allowlists and tool allowlists are not model
calls. They are exact, they are free, and they cannot be talked out of anything. The corpus's ordering
principle is that deterministic checks should be unconditional and model checks should sit in front of
them or behind them, never as the only control on a dangerous path `[D]`. Any of the four that *could* be
deterministic should be.

**7. Then: negotiate the SLO, with the price visible.** If the security requirement genuinely costs
latency, the honest move is to present the curve — latency versus rail coverage — and let the decision be
made by whoever owns both. The corpus's framing on the false-refusal tradeoff applies: measure the
false-positive rate, classify what is being blocked, and prefer narrowing the scope of a rail over
weakening it `[D]`.

**8. What I would refuse.** Removing a rail because it is slow, and weakening a rail globally under
schedule pressure. If a rail must go, the correct move is to replace it with a capability limit that
bounds the same consequence without classifying content — read-only credentials for the tools the tool-call
rail was guarding, for instance `[D]`.

**The tradeoff being tested.** Whether the candidate can hold both constraints and find the structural
answer, or picks a side. The wrong answers are "remove the guardrails" and "the SLO must yield"; the right
one is that four synchronous model calls in a request path is an implementation choice, not a requirement.

**Red flags.** Proposes removing rails. Defends the latency as necessary without decomposing it. Cannot
distinguish deterministic checks from model-based ones. Proposes weakening a rail globally rather than
narrowing its scope.

---

## Sources

Every scenario draws on at least three topic banks, each of which carries its own `## Sources` block
listing the `refs/...` paths behind it. The corpus paths that carry the load-bearing claims in this file
are listed here; consult the linked bank for the full provenance of each topic.

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
  — the cost ladder (100 → 42 → 26 → 11) with per-step factors, cache reads ~10× cheaper, the precision
  journey, and "always rerun your evals after quantizing." Behind S1, S4, S6, S11, S13, S14.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
  and `.../How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt` — the three planes, the
  latency decomposition, goodput, tail sampling, the "eval passes, users unhappy" failure signature, and
  the layered judge architecture. Behind S1, S2, S8, S10, S11, S13, S14.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` (llm-d session) — prefix cache hit rate per pod as the diagnostic for bad
  routing, the saturation thresholds (KV 80%, active requests > 8), KV to 90% of VRAM, and the offload
  tier with create/evict events. Behind S2, S6, S7, S11, S12.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` — the naive-8-way-TP finding, the tuned
  TP+PP+SP+EP mix across 16 GPUs per replica, `experts_per_GPU`, the naive-2-all-to-all-plus-6-kernels to
  3-kernel reduction, "there is no universal winner," the 2P4D preliminary flag, and the 28k-input /
  concurrency-256 KV-recomputation cliff. Behind S4, S7, S12.
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — the agent fairness result (one large session monopolising dispatch and starving short sessions,
  least-attained service with turn priority, the 2–3× improvement), session return giving ~5× TTFT
  improvement, agents idle 99.999% of the time, the durable session log, and the sovereignty dimensions.
  Behind S3, S5, S6, S10.
- `refs/Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt`
  and the Token Raj material — cost per 10k-token task 30¢ → 16¢, drafted lines 630 → 91, files touched
  8.2 → 3.6, and billing falling less than the workload because thinking tokens rose. Behind S1, S6.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Guardrails_Stop_Injection_Leaks_Hallucination_NeMo_Guardrails.txt`
  and `refs/ai-system-design-guide-main/ai-system-design-guide-main/12-security-and-access/01-llm-security.md`
  — the four rail positions, guardrails as checks at each boundary, the middle-two omission, redaction at
  the boundary, and the May 2026 escalation material. Behind S3, S5, S9, S14.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_10_Incorporating_Tools.txt`
  and `..._11_Agents_and_Multi-Agent_Communication.txt`, plus
  `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
  — tool incorporation, the agent loop, the 98% prefill / 2% decode profile, and non-determinism at
  temperature 0. Behind S4, S6, S8, S9, S10, S13.
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  and `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
  — agent identity as distinct from user identity, the sandbox as an execution tier, and the
  decision-level audit trail (principal, scope, approval, outcome). Behind S5, S9.

**Derived content in this file (`[D]`).** The following are my own constructions, not corpus claims, and
are marked `[D]` at the point of use: every capacity and cost arithmetic, including the KV-per-sequence
figure in S4 and the 4.4× decomposition in S6; the `blast_radius = capability × reach × reversibility`
reductions in S9; the three-part process fix in S8; the rail-parallelisation and triage proposals in S14;
the failover remediation ordering in S12; the routing-versus-stacking analysis in S11; and every red-flag
list and grading judgement. Corpus figures are attributed to their talks wherever they appear and are
reproduced as reported, not as verified benchmarks — in particular the ladder factors, the fairness
result, the ~5× session-return TTFT improvement and the Token Raj figures come from vendor or practitioner
talks rather than independent measurement. The sovereignty dimension enumeration is the speaker's own,
including the four-or-five ambiguity noted in S5.
