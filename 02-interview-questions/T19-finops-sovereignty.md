# Interview Bank: FinOps, Efficiency & Sovereignty

> `T19` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T19-finops-sovereignty.md) · [Case study](../01-case-studies/T19-finops-sovereignty.md) · [Design blueprint](../03-design-blueprints/T19-finops-sovereignty/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** the cost ladder, unit economics and build-vs-buy, sovereignty, then open design

## How to use this bank

Two halves that interviewers usually test separately, and the best candidates connect them: **the cost
ladder** (an ordered sequence of four cumulative levers) and **sovereignty** (a system property with
four or five dimensions, one of which is the inference layer itself).

The trap on the cost side is treating the ladder as a menu of independent discounts — Q1 and Q6 exist to
catch that. The trap on the sovereignty side is the flag framing — Q16 exists to catch that.

---

### The cost ladder

#### T19-Q1 · Walk me through the cost ladder
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Give me the levers that take an LLM workload from an unoptimised baseline to an efficient
one, and the order they go in.

**Model answer:** The corpus gives an **ordered, cumulative** ladder `[T]`:

| Step | Lever | Cost after | Factor applied |
|---|---|---|---|
| 0 | Baseline (unbatched, fp16, no cache, single model) | **100** | — |
| 1 | **Batching / continuous batching** | **42** | ×0.42 |
| 2 | **Quantisation** | **26** | ×0.62 |
| 3 | **Caching + routing** | **11** | ×0.42 |

So `100 → 42 → 26 → 11`, roughly a **9× reduction** `[T]`. Every figure is from the corpus's cost talk.

**The word that matters is cumulative.** These apply in sequence, and each factor is a multiplier on the
running total, not a discount off the original. That is why the order is not arbitrary and why you cannot
simply take "the best one": 0.42 × 0.62 × 0.42 ≈ 0.109, which is where 11 comes from `[D]`.

**Why this order is the natural one** `[D]`:

- **Batching first** because it is free throughput — it recovers the GPU idle time you are already paying
  for, and it requires no quality decision at all. There is no reason not to.
- **Quantisation second** because it is a *quality* decision with a mandatory verification step
  (T19-Q3), so it wants a working serving stack to measure against.
- **Caching and routing third** because they require knowing your traffic — which prefixes repeat, which
  requests are easy — and that knowledge comes from having batching and metrics in place first.

**The honest caveat I would add:** the ladder is a *modelled* sequence from a talk, not a benchmark of
your workload `[T]`. Your factors will differ. What generalises is the ordering logic and the habit of
measuring each step's effect separately — because if you apply all three at once and quality drops, you
will not know which one did it.

**Signal:** Recites the ladder with the cumulative ordering intact, and identifies batching as the
free move and quantisation as the one requiring verification.

**Follow-ups:**
- *Which step is free?* — batching; it requires no quality tradeoff.
- *Are the factors guaranteed?* — no; they are from the corpus's cost talk, not your workload `[T]`.
- *Why not apply everything at once?* — you lose attribution when quality moves.

**Red flags:** Treats the three levers as independent percentage discounts, or reverses the order
(quantising before batching).

---

#### T19-Q2 · Why is batching first?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Batching gives the single largest step on the ladder — 100 → 42. Why is it first, and what
does it cost you?

**Model answer:** Because it is the only lever that converts capacity you are already paying for into
throughput, and its cost is latency rather than quality.

**The mechanism.** Decode is memory-bandwidth-bound, not compute-bound: generating one token for one
sequence reads the whole model's weights and does a trivial amount of arithmetic with them. Every
additional sequence in the batch reuses those same weight reads `[D]`. So the marginal cost of a second
sequence in the batch is far below the first — which is exactly why the factor is 0.42 rather than
something near 1 `[T]`.

**Why first, specifically** `[D]`:

- **No quality decision.** Quantisation changes the numbers; caching changes what the model sees; routing
  changes which model serves you. Batching changes none of those — the same model produces the same
  outputs, just more of them at once. So it is the one step that never needs a quality gate.
- **It compounds with everything after it.** The quantization and caching factors apply to an
  already-batched baseline. Optimise in the other order and you are multiplying smaller numbers.
- **It makes the later levers measurable.** You need a saturated server before you can see a routing
  improvement; on an idle one every routing decision looks fine.

**What it costs** `[D]`: **latency, and specifically tail latency.** A request admitted into a larger
batch waits longer for its first token and interleaves its decode steps with more neighbours, so ITL
rises. The tradeoff is bounded by your SLO, which is why the corpus's serving material is organised around
TTFT and ITL rather than throughput alone `[T]`. Batching is free in *quality* but not in *experience*.

**The operational point:** because the cost is latency, the batching decision is an SLO decision, and it
belongs with whoever owns the SLO — not with whoever owns the GPU bill. A team that batches
unconditionally to hit a cost target will eventually breach a latency SLO and then over-correct.

**Signal:** Explains the memory-bandwidth mechanism rather than saying "batching is more efficient," and
names tail latency as the thing it costs.

**Follow-ups:**
- *Why is the marginal sequence cheap?* — the weight reads are shared across the batch `[D]`.
- *What does batching cost?* — TTFT and ITL, so it is an SLO decision `[T]`.
- *Why does it compound?* — the later factors multiply an already-batched baseline.

**Red flags:** Says batching is free with no latency cost, or cannot explain why one GPU serves many
sequences cheaply.

---

#### T19-Q3 · Quantisation and the mandatory eval
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Quantisation is step two. What does it buy, what does it cost, and what is the rule that
comes with it?

**Model answer:** Roughly a **×0.62 cost factor** `[T]`, and it comes with the corpus's most-repeated and
most-skipped rule: **"always rerun your evals after quantizing"** `[T]`.

**What it buys.** Smaller weights mean less memory traffic per token, and decode is
memory-bandwidth-bound — so precision reduction attacks the bottleneck directly `[D]`. It also shrinks
the KV cache footprint, which raises how many sequences fit and therefore how large a batch you can run.
Two compounding wins, which is why a single step yields 0.62.

**The precision journey** `[T]`: **16-bit → 8-bit → 4-bit.** Each step is a larger saving and a larger
quality risk, and the risk is not linear — 8-bit is usually close to lossless in practice, while 4-bit
requires real evaluation on your task `[D]`.

**What it costs — three things, not one** `[D]`:

1. **Quality on your task.** Not a theoretical concern; measurable, and task-dependent. A code model and
   a summarisation model degrade differently at the same precision.
2. **More non-determinism.** The corpus states that quantisation *worsens* the non-determinism you already
   have at temperature 0 `[T]`. This matters beyond quality: it breaks replay-based debugging and makes
   exact-match tests meaningless (T17-Q1).
3. **Model fingerprinting.** The quantisation scheme is inferable from outputs `[T]`. If your inference
   stack is confidential — a sovereignty concern (T19-Q19) — quantisation is a disclosure channel. That
   is a security consideration most teams never consider, and volunteering it is a strong signal.

**Why the rule exists.** Because quantisation is the step where a cost optimisation silently becomes a
quality regression, and it is the step most likely to be applied by a platform team who never see the
outputs. Evals are the only thing standing between the two. The corpus's remedy is to make it structural:
**eval-gated CI**, where a prompt, model, quantisation or config change runs the suite and blocks the
merge on regression `[T]`.

**What I would do concretely:** baseline the eval suite at fp16, quantise, rerun, and compare on *both*
quality and variance. If quality holds and variance grows, that is still a real change to your
debuggability and should be a recorded decision rather than a silent default.

**Signal:** Names all three costs — quality, non-determinism, fingerprinting — and treats the eval rule as
a CI gate rather than advice.

**Follow-ups:**
- *Where does it sit in the ladder and why?* — step 2, after the free batching win, because it needs a
  quality gate.
- *What does it do beyond cost?* — worsens non-determinism and fingerprints the model `[T]`.
- *When is it not worth it?* — when the eval regression is real and the cost saving is small at your
  volume; the gate decides, not the platform team.

**Red flags:** Applies quantisation without rerunning evals, or claims it is lossless without reference to
a measurement.

---

#### T19-Q4 · Caching, and what actually gets cached
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Step three is caching plus routing and it gives another 0.42. What is being cached, and why
is the saving so large?

**Model answer:** Specifically **KV cache for shared prompt prefixes** — and the saving is large because
of an asymmetry in how inference is billed in compute terms.

**The asymmetry** `[D]`: prefill is compute-bound and processes the whole prompt; decode is
memory-bandwidth-bound and emits one token per step. In most workloads the prompt dominates the arithmetic
— which is why a cached prefix, which skips that prefill entirely, is worth far more than it looks. The
corpus's cost talk gives the number: **cache reads are roughly 10× cheaper** than fresh processing `[T]`.

**What makes a prefix cacheable** `[D]`: a shared, *stable* prefix — the system prompt, the tool
definitions, the retrieved context block, the conversation history up to the last turn. The requirement is
stability: any change early in the prompt invalidates everything after it. That is why the corpus's agent
material puts such emphasis on **harness authors keeping the prefix stable**, including the point that
**tools are part of the prefix** — so adding a tool mid-session can invalidate the cache for every
subsequent turn `[T]`.

**Where the second half of the factor comes from: routing.** Caching only pays if the request reaches a
replica that holds the cache. Route blindly and the prefix cache is cold on the replica you land on, and
you pay full prefill plus the cost of a wasted cache write. So the router must be cache-aware — the
corpus's routing material is explicit that **prefix cache hit rate per pod is the signal that exposes bad
routing** `[T]` (T14). Caching and routing are one step on the ladder because they are one mechanism.

**The cost side** `[D]`: cache memory is GPU memory, so caching competes with batch capacity. The corpus's
serving guidance is to fill it aggressively — **KV to 90% of VRAM** `[T]` — on the reasoning that unused
KV memory is wasted capacity. But it is a tradeoff, not a free lunch: more cache means less room for
concurrent sequences, and the right point depends on your prefix-repetition rate.

**What I would measure:** prefix cache hit rate per pod, and the distribution of unique prefixes. If hit
rate is low, the problem is usually the harness rewriting its own prefix, not the serving configuration —
and that is a finding you take to the application team, not a tuning knob.

**Signal:** Identifies KV prefix caching specifically, connects it to routing as one mechanism, and points
at prefix stability as an application-team responsibility.

**Follow-ups:**
- *Why are cache reads ~10× cheaper?* — they skip prefill compute `[T]`.
- *What breaks caching?* — an unstable prefix; tools are part of it `[T]`.
- *How full should the KV cache be?* — the corpus says fill to 90% `[T]`, as a capacity tradeoff.

**Red flags:** Says "we cache responses" (a different and much smaller optimisation), or caches without
making routing cache-aware.

---

#### T19-Q5 · Why is the ladder ordered this way?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Justify the ordering of the four levers. Could you get the same result in a different order?

**Model answer:** You could get a similar arithmetic result — multiplication commutes — but you would not
get the same *engineering* result, and the ordering encodes three real constraints.

**Constraint one: the free move comes first** `[D]`. Batching is the only lever with no quality cost and
no application-team dependency. There is no scenario in which batching should wait for anything. Its
position first is not a preference; it is the absence of any reason to defer it.

**Constraint two: measurement dependencies** `[D]`. Quantisation needs a working, measurable serving
stack to prove it did not regress quality (T19-Q3). Caching needs to *see* traffic to know which prefixes
repeat — and it needs batching in place to make the memory tradeoff (cache vs concurrent sequences)
tractable, because on an unbatched server you have memory to spare and the tradeoff is invisible. So
step 3 genuinely requires steps 1 and 2 to have happened.

**Constraint three: organisational dependencies** `[D]`, and this is the one candidates miss. Each step
belongs to a different owner:

| Step | Owner | Currency |
|---|---|---|
| Batching | serving/platform | latency SLO |
| Quantisation | ML + eval owner | quality |
| Caching | **application team** (prefix stability) + platform (routing) | effort, harness discipline |
| Routing | gateway/platform | complexity |

The ordering is roughly **increasing organisational difficulty**. Batching is a config change. Quantisation
needs an eval gate. Caching requires the application team to change how they build prompts — which is
much harder than changing a server flag, and it is where most programmes stall.

**The practical consequence** `[D]`: if you are doing this in an organisation, work the ladder in order
for two reasons — because the arithmetic compounds that way, and because each step funds the credibility
for the next. A platform team that delivers 100 → 42 has the standing to ask the application team for
prefix discipline. One that starts by asking for it has neither a result nor a mandate.

**Signal:** Separates the arithmetic (order-insensitive) from the engineering constraints (order-
sensitive), and identifies the organisational escalation as the real ordering logic.

**Follow-ups:**
- *Which step needs an eval gate?* — quantisation; it is the quality-risk step.
- *Which step needs the application team?* — caching; prefix stability is not a server setting.
- *Why does order matter if multiplication commutes?* — measurement and credibility dependencies.

**Red flags:** Treats the order as arbitrary arithmetic, or starts with routing/caching because it sounds
most sophisticated.

---

#### T19-Q6 · Do you have to take all four steps?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** A team applies all three levers and reports a 9× saving on paper. Six weeks later the bill
is down 3×. What happened?

**Model answer:** Something in the gap between modelled factors and realised savings, and there are four
usual causes I would check in this order.

**One: the baseline was never real** `[D]`. The ladder's 100 is an *unoptimised* baseline — unbatched,
fp16, no cache. If the team was already running continuous batching (most vLLM deployments are by
default), their true baseline was already ~42 and the "9× on paper" was really ~3.8×. This is the most
common cause and it is a measurement error, not a technical one.

**Two: the factors are workload-dependent** `[D]`. The corpus's numbers come from one cost talk about one
workload `[T]`. Quantisation's benefit depends on batch size and sequence length; caching's benefit
depends entirely on **prefix repetition**, which varies enormously by application. A workload with unique
prompts per request gets almost nothing from a prefix cache — the factor is 1.0, not 0.42.

**Three: the saving was reinvested** `[D]`. This is the one that is not an error. The corpus's own
cautionary tale is exactly this shape: the **Token Raj study** shows cost per 10k-token task falling from
**30¢ in February to 16¢ in April**, with code lines drafted falling from **630 to 91**, files touched
from **8.2 to 3.6** — and yet **the billing fell less than the workload, because thinking tokens rose**
`[T]`. Efficiency gains get spent on more capable behaviour. That is often the right business outcome and
it looks like a failure if you only watch the bill.

**Four: leakage** `[D]`. Unbounded retries, cache-busting prompt construction, a replica that lost its
cache, a workload that shifted to longer contexts. The corpus's observability guidance is that you cannot
see any of these without **per-request and per-tenant token accounting** `[T]` (T17) — and that without
it, cost is a number you discover on an invoice, three weeks after the decision.

**What I would do:** recompute the baseline from real telemetry before accepting any ladder claim, then
attribute the gap. If it is cause three, that is a product conversation, not an engineering one — and the
right metric changes from cost per token to cost per *task* or per *outcome*.

**Signal:** Names the false-baseline cause first and the reinvestment cause as a legitimate outcome rather
than a failure, and requires per-request telemetry to attribute.

**Follow-ups:**
- *Most common cause?* — a baseline that was never unoptimised.
- *Which factor is most workload-dependent?* — caching; it needs prefix repetition.
- *What changes if it is cause three?* — the metric; cost per task, not cost per token `[T]`.

**Red flags:** Treats the factors as guaranteed, or reports a saving without per-request telemetry behind
it.

---

#### T19-Q7 · The KV cache memory tradeoff
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** The corpus says fill the KV cache to 90% of VRAM `[T]`. Justify that, and tell me when it is
wrong.

**Model answer:** The default reasoning is sound — unused KV memory is wasted capacity — but it is a
*default*, and there are three situations where 90% is the wrong target.

**Why the default is right** `[D]`: KV memory is the resource that determines how many sequences you can
hold, and holding more sequences is what makes batching possible. Memory you have not allocated is memory
that cannot hold a sequence, and on a decode-bound server the throughput ceiling is set by how many
concurrent sequences you can admit. So filling it is how you buy the batching win from T19-Q2.

**Three cases where it is wrong** `[D]`:

1. **Long-context traffic.** A single request with a very long context can need most of the cache to
   itself. A server at 90% utilisation with no free blocks either preempts or recomputes — and the corpus
   documents the cliff that produces: **the cliff at 28k inputs / concurrency 256 from KV recomputation**
   `[T]`. Fill aggressively and you have less headroom to absorb one long request, which converts a
   capacity problem into a latency incident.
2. **Mixed workloads with a small fast lane.** If you have a latency-sensitive class and a batch class
   sharing a server, filling the cache for the batch class starves the fast lane's admission. The corpus's
   saturation thresholds — **KV 80%, active requests > 8** `[T]` — imply you want a *headroom* signal, not
   a *maximum fill* target.
3. **When the cache is not paying.** If your prefix repetition rate is low (T19-Q6), the memory is
   occupied by cache that never hits, and it would be better spent on concurrent-sequence capacity. The
   metric to check is hit rate; if it is low, the problem is not the fill percentage.

**The reframing I would offer** `[D]`: "fill to 90%" is really two separate targets that got merged — how
much memory to *give* the KV cache, and how much headroom to *keep* for admission. The first should be
generous; the second depends on your context-length distribution and your SLO. And the signal that tells
you which you have wrong is the preemption/recomputation rate, not the utilisation number. High
utilisation with low preemption is healthy; high utilisation with rising preemption is the cliff arriving.

**Signal:** Accepts the default but names long-context headroom and workload mixing as the exceptions, and
points at preemption rate rather than utilisation as the real signal.

**Follow-ups:**
- *When does it hurt?* — long contexts, mixed latency classes, and low prefix repetition.
- *What is the real signal?* — preemption/recomputation rate, not the fill percentage `[T]`.
- *What does the 28k/256 cliff illustrate?* — KV recomputation converting capacity pressure into latency
  `[T]`.

**Red flags:** Treats 90% as an invariant to configure, or cannot describe what happens when the cache
fills.

---

#### T19-Q8 · Rack-scale and pooled memory
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** The corpus mentions a rack-scale four-server pooled-memory demonstration `[T]`. What problem
does that solve, and what is the FinOps read?

**Model answer:** It addresses the constraint that has bounded everything in this bank so far — KV memory
is per-GPU and per-node — and the FinOps read is that it changes which axis you scale on.

**The problem** `[D]`: every tradeoff in T19-Q4 and T19-Q7 is a memory tradeoff inside one accelerator.
Cache versus concurrent sequences. Long context versus batch size. And the corpus's own parallelism
material shows the same pressure from the other side: more GPUs gives more compute, but attention and KV
state are what make very long context and high concurrency expensive, and the corpus's headline
long-context story includes the **cliff at 28k inputs / concurrency 256 from KV recomputation** `[T]`.

**What pooling changes** `[D]`: if KV for a request can live in a pooled tier shared across four servers
rather than in one GPU's HBM, then the effective cache is much larger, and eviction stops being the
dominant failure mode. The corpus pairs this with the **offload tier and retention API** in the agent
serving material, where KV is explicitly created and evicted with an API rather than left to the
scheduler, and where **offloading produced roughly a 5× TTFT improvement on a session's return** `[T]`
(see the T16 bank). Same idea at two scales: **the cache you can keep is the latency you do not repay.**

**The FinOps read** `[D]`:

- **It shifts the lever from "buy more accelerators" to "use the memory you own."** If pooled memory
  absorbs the cache working set, you may serve the same traffic with fewer GPUs, or hold longer contexts
  on the same fleet. Both are direct cost changes.
- **It is a *structural* decision**, and the corpus's framing is that structural decisions are the
  irreversible ones — the structure-versus-efficiency distinction `[T]` (T19-Q13). Pooling is a
  deployment topology choice, not a tuning knob.
- **The accounting changes.** Pooled memory that is 80% idle across a fleet is a visible, attributable
  cost. Distributed cache tiers make utilisation a first-class metric alongside GPU utilisation.

**What I would be careful about.** The corpus presents this as a *demonstration*, not a production
capability with published economics `[T]`. I would treat it as directionally important and not cite it as
a number. The generalisable claim is the one I would defend: **memory capacity, not compute, is the
binding constraint for long-context and high-concurrency serving**, and every lever that relieves it shows
up as cost.

**Signal:** Connects pooled memory to the same constraint the cache levers attack, and marks it as a
structural rather than a tuning decision without overclaiming the demo.

**Follow-ups:**
- *What constraint does it relieve?* — per-accelerator KV capacity `[D]`.
- *Is it a tuning change?* — no; it is a topology decision, and structural decisions are the irreversible
  ones `[T]`.
- *Can you cite the economics?* — no; the corpus reports a demonstration, not published numbers `[T]`.

**Red flags:** Cites the demo as a production benchmark, or misses that it addresses the same memory
constraint as caching.

---

### Unit economics and the build-versus-buy decision

#### T19-Q9 · Cost per task, not cost per token
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** A team reports their cost per million tokens fell 40%. Is that good news?

**Model answer:** Unanswerable as stated, and the reason is the corpus's Token Raj lesson `[T]`.

**What the study shows** `[T]`: between February and April, cost per 10k-token task fell from **30¢ to
16¢** — a genuine ~47% improvement, matching the team's claim. And in the same period, **code lines
drafted fell from 630 to 91** and **files touched from 8.2 to 3.6** `[T]`. Read naively, the workload
shrank dramatically. But the corpus's point is the punchline: **the billing fell less than the workload**
— because **thinking tokens rose** `[T]`.

**What that means** `[D]`: the efficient model reasons more per task. Fewer tokens per unit of *output*
coincided with more tokens per *task*. The cost per token is genuinely better and the total spend is
genuinely higher, and **both are true simultaneously.** A cost-per-token metric cannot distinguish
"we got more efficient" from "we bought more capability at a better unit price" — and the second is
usually the desired outcome.

**So the question I would ask back:** what happened to cost per *task*, and to tasks per *outcome*?

```
cost_per_task   = tokens_per_task × price_per_token
cost_per_outcome = cost_per_task / success_rate
```

`[D]`. The second term is the one a business cares about, and it is the one nobody measures. A
cheaper-per-token model with a lower success rate can raise cost per outcome, because failed tasks still
consume tokens and humans still have to redo the work.

**The three metrics I would report together** `[D]`:

1. **Cost per token** — the lever metric; drives the ladder.
2. **Cost per task** — the workload metric; comparable across model changes.
3. **Cost per successful outcome** — the business metric; includes retries and failure.

**Why this matters operationally.** If the team is being asked to cut cost, and they optimise metric 1
while metric 3 rises, they will hit their target and hurt the business. That is exactly the shape of the
corpus's efficiency-versus-structure point: efficiency improvements that are real can still be
mis-measured `[T]`.

**Signal:** Immediately reframes to cost per task and per outcome, cites the Token Raj numbers as the
reason cost-per-token is insufficient, and names the failure-rate term.

**Follow-ups:**
- *What is the punchline of the study?* — billing fell less than workload because thinking tokens rose `[T]`.
- *Which metric would you report to a CFO?* — cost per successful outcome.
- *How do you get these numbers?* — per-request and per-tenant token accounting `[T]` (T17).

**Red flags:** Congratulates the team on the token metric, or treats falling workload size as the whole
story without noting the billing divergence.

---

#### T19-Q10 · Build versus buy
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** A team proposes self-hosting to save money. Walk me through the decision.

**Model answer:** I would frame it as a **structural** decision first and an economic one second, because
the corpus's distinction is that **structure versus efficiency** is the axis that matters — buy-versus-
build and sovereignty are **irreversible structural decisions**, while tuning is efficiency `[T]`.

**Why structure first** `[D]`: if self-hosting turns out to be more expensive, the reversal cost is not
"switch back" — it is the migration, the retraining, the contracts, the compliance posture you built
around it, and the team you hired. Decisions whose reversal cost approaches their adoption cost are
structural, and they should be decided on strategic grounds before the spreadsheet.

**The break-even, stated properly** `[T]`/`[D]`:

```
self_host_cost = GPU capex or reserved lease
               + datacentre / power / cooling
               + serving engineering headcount      <-- the term teams omit
               + ops, upgrades, model onboarding
               + idle capacity you pay for anyway   <-- including the tail

buy_cost       = tokens × price_per_token
               + integration and gateway work
```

The corpus's explicit warning is that **self-host break-even must include ops cost** `[T]`. The failure
mode is well known: teams compare GPU-hour price against token price, omit the three headcounts and the
idle tail, and conclude self-hosting is 5× cheaper. It is not; it is 5× cheaper at 100% utilisation, which
you will not have.

**The utilisation question is the whole analysis** `[D]`. Self-hosting is a fixed-cost business and buying
is a variable-cost one. Fixed costs reward utilisation; variable costs reward volatility. So:

- **High, stable, predictable load** favours self-host.
- **Spiky, growing, or uncertain load** favours buying — you pay for the spike and nothing else.
- **The hard case is high-but-spiky**, which needs reserved baseline plus burst, and that is a hybrid
  whose complexity is itself a cost.

**What I would actually recommend** `[D]`: run the analysis, then **defer the structural commitment until
you have the utilisation data**, because the data is cheap to get and the commitment is not. Concretely:
instrument per-request cost and utilisation first (T17), run a self-host pilot on a bounded workload, and
only then decide. The corpus's framing supports this — efficiency measurements are reversible and
inform structural choices; you should buy the information before you buy the cluster.

**And the sovereignty overlay** `[D]`: if the decision is being driven by a compliance requirement rather
than by cost, the break-even may be irrelevant — you self-host because you must, and the analysis becomes
"what is the cheapest compliant option" rather than "is self-hosting cheaper" (T19-Q16). Distinguishing
which conversation you are in is the first thing I would establish.

**Signal:** Separates structural from efficiency framing, names omitted ops cost as the classic error, and
makes utilisation the hinge of the analysis.

**Follow-ups:**
- *What does the break-even usually omit?* — ops headcount and the idle tail `[T]`.
- *What is the hinge?* — utilisation: fixed cost versus variable cost.
- *When is the economics moot?* — when compliance requires self-hosting; then it is a constrained
  optimisation.

**Red flags:** Compares GPU-hour price to token price directly, or recommends the structural commitment
before measuring utilisation.

---

#### T19-Q11 · Reserved, on-demand or spot
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** How would you structure the capacity purchase for an inference platform?

**Model answer:** As a portfolio matched to the load's *predictable* and *unpredictable* components, with
the serving layer told which is which.

**The shape** `[D]`:

- **Baseline load** → reserved or owned capacity. Cheapest per unit, and it should cover the load you are
  confident about — not the mean, but the level you rarely drop below.
- **Predictable peaks** → on-demand. Expensive per unit, no commitment.
- **Opportunistic / fault-tolerant work** → spot, if and only if your serving layer can handle
  interruption. Batch evaluation, offline scoring, and some agent workloads can. Interactive serving
  cannot — an interrupted request is a failed request.

**The serving-layer consequence, which is the part candidates miss** `[D]`: the capacity mix determines
what the autoscaler can do. Spot capacity implies preemption, preemption implies request loss unless the
router can retry elsewhere, and retries interact with your SLO and with prefix caching (a retry lands on a
cold replica). So the purchase decision is not purely a finance decision — it constrains the routing and
autoscaling design in T14 and T15, and it should be taken with those owners in the room.

**Where the corpus's material bites** `[T]`:

- **Saturation thresholds** — KV 80%, active requests > 8 `[T]` — are the signals that tell you whether
  you are paying for capacity you are not using, or using capacity you have not bought (T15).
- **The 100 → 42 → 26 → 11 ladder is a cheaper lever than any purchase decision** `[D]`. Before
  restructuring contracts, work the efficiency ladder; a 9× capacity reduction changes the purchase
  conversation entirely. The wrong order is to sign a three-year commitment and then discover batching.
- **Idle capacity is the cost of a bad SLO** `[D]`. If you are provisioning for a p99 you never actually
  need, the cheap fix may be an SLO conversation, not a capacity one.

**What I would measure:** utilisation over time *by capacity class*, preemption rate, and the ratio of
provisioned to used. Then I would re-derive the reserve level from the actual distribution rather than
from a peak observed once during an incident.

**Signal:** Matches capacity class to load predictability, and connects the purchase decision to
autoscaling and routing constraints rather than treating it as a finance exercise.

**Follow-ups:**
- *What can run on spot?* — fault-tolerant offline work only; not interactive serving.
- *What does spot imply downstream?* — preemption, retries and cold caches in the router.
- *What comes before renegotiating contracts?* — the efficiency ladder; it changes the requirement.

**Red flags:** Proposes spot for interactive serving, or buys capacity before optimising utilisation.

---

#### T19-Q12 · The cost of a bad SLO
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Product wants a p99 TTFT of 200 ms. What does that cost, and how do you discuss it?

**Model answer:** I would not quote a number — I would show that the cost curve is convex in the SLO, and
make the tradeoff explicit so the decision is made with its price attached.

**Why it is convex** `[D]`: provisioning for a percentile means holding capacity that is idle most of the
time, and the further out the percentile, the more capacity is idle for the same traffic. Moving p99 to
p99.9 does not cost 10% more; it costs whatever it takes to absorb the rare event, which on a shared fleet
means headroom sized for a peak that almost never happens. This is also why the corpus's guidance is to
alert on **goodput** — the fraction of requests meeting *both* the TTFT and ITL targets — rather than on a
mean `[T]`: goodput is the metric that makes the SLO's cost visible, because a mean can look healthy while
a meaningful fraction of users are not being served to target (T17).

**The conversation I would have** `[D]`:

1. **Convert the SLO into capacity.** What utilisation can we hold and still meet p99 at peak? That is
   the real requirement, and it is usually a much lower utilisation than the team expects.
2. **Price the SLO against the levers that are not capacity.** A 200 ms p99 is partly a *batching*
   decision (T19-Q2), partly a *caching* decision (T19-Q4), partly a *routing* decision (T14), and partly
   a *model size* decision. Buying the SLO with GPUs is the most expensive way to get it; getting it by
   keeping the prefix cacheable and the routing cache-aware is much cheaper.
3. **Offer a tiered SLO.** Different request classes can hold different targets — interactive chat at
   p99 200 ms, background agent turns at p99 5 s. Since the corpus's own agent material shows agents
   running **98% prefill / 2% decode** `[T]`, an agent workload's latency profile is nothing like chat's,
   and forcing one SLO on both wastes capacity on the one that does not need it.
4. **Show the percentile curve.** Cost versus p99 as a curve, not a point, so the decision-maker sees the
   inflection rather than a single price.

**The honest answer to "what does it cost":** it costs whatever the difference is between the utilisation
you can hold at that SLO and the utilisation you would hold at the next one down — and I would bring that
number as a curve with two or three alternatives, because the useful decision is rarely "yes or no" but
"which of these three."

**Signal:** Refuses to quote a point price, shows the convexity, and proposes meeting the SLO with
non-capacity levers first and a tiered target second.

**Follow-ups:**
- *Why is it convex?* — headroom sized for a rare peak is idle the rest of the time `[D]`.
- *What metric exposes the cost?* — goodput, not mean latency `[T]`.
- *Cheapest way to buy latency?* — cache and routing first; GPUs last.

**Red flags:** Quotes a hardware number for the SLO without the utilisation requirement, or accepts a
single global SLO for heterogeneous workload classes.

---

#### T19-Q13 · Structural versus efficiency
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** The corpus distinguishes structural decisions from efficiency decisions `[T]`. Why does that
distinction matter more than the cost numbers?

**Model answer:** Because it is a decision *ordering* rule, and getting the order wrong is how organisations
spend money they cannot recover.

**The distinction** `[T]`:

- **Efficiency decisions** — batching, quantisation, caching, routing, autoscaling policy. Reversible,
  measurable, cheap to try. Their natural home is continuous improvement, and the cost ladder is exactly
  this category.
- **Structural decisions** — self-host versus buy, deployment topology, sovereignty posture, which
  inference stack you build on. Hard or impossible to reverse, and their cost is dominated by the
  reversal rather than the adoption.

**Why the ordering matters** `[D]`: structural decisions **foreclose** efficiency options. Commit to a
cloud provider's managed inference and you cannot apply the serving-engine lever you wanted. Commit to a
sovereign deployment topology and the caching/routing architecture is constrained by where the data may
live. Commit to self-hosting and you have taken on a fixed-cost business whose economics depend on
utilisation you have not yet measured. In each case the efficiency ladder you *can* run afterwards is
smaller than the one you had before.

**The rule I would apply** `[D]`: **measure with efficiency levers before committing structurally.**
Instrument per-request cost and utilisation (T17), run the ladder, learn your actual load shape — and
*then* make the structural decision with data. Efficiency work buys information; structural work spends
it. Doing them in the other order is buying a fixed-cost commitment priced against a load profile you
guessed.

**The second half of the rule, which matters at staff level** `[D]`: some structural decisions are
correctly *not* economic. Sovereignty is the clear case (T19-Q16) — if a regulator requires the inference
layer to be under your control, the break-even is irrelevant. So the ordering rule is not "always optimise
first"; it is **"know which conversation you are in."** A cost-driven structural decision should wait for
efficiency data. A compliance-driven one should not, and pretending otherwise wastes everyone's time.
Telling those two apart at the start of a project is most of the value a senior engineer adds here.

**What it looks like when it goes wrong** `[D]`: a three-year commitment signed on projected volumes,
followed by a batching project that halves the requirement; or a service migrated to a managed platform
for cost reasons, followed by a discovery that the routing and caching architecture the team designed
cannot be implemented there. Both are the same error — a structural commitment made before the efficiency
work that would have informed it.

**Signal:** States the foreclosure direction (structural constrains efficiency, not vice versa) and adds
the compliance exception where economics is correctly subordinate.

**Follow-ups:**
- *Which direction is the constraint?* — structural forecloses efficiency options.
- *What does efficiency work buy?* — information for the structural decision.
- *When is the economics irrelevant?* — when compliance requires the structure `[T]`.

**Red flags:** Treats all decisions as tunable, or applies the cost analysis to a decision that is
actually compliance-driven.

---

#### T19-Q14 · Showback and chargeback
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Finance wants per-team cost attribution for the inference platform. How do you build it?

**Model answer:** From per-request telemetry at the gateway, attributed by principal — because that is the
only place the request still knows who it belongs to.

**The mechanism** `[D]`:

1. **Every request carries a principal and a tenant** at the gateway (T14). Attribution is a routing-layer
   responsibility, not a serving-layer one — by the time a request is in a vLLM scheduler, the tenant
   context is usually gone.
2. **Token accounting per request** — input, output, cached and fresh — recorded as span attributes and
   gateway metrics `[T]`. The corpus's cost material is emphatic that **cost is a first-class
   observability signal**, and the observability material gives the shape: per-request and per-tenant
   token telemetry is what turns the cost ladder into an operational metric rather than a blog post `[T]`.
3. **Price it with the real formula**, not a flat rate:

```
cost_task = (prompt_tokens − cached_tokens) × p_fresh
          + cached_tokens × p_cached            # ~10x cheaper [T]
          + completion_tokens × p_out
```

A flat per-token rate systematically over-charges teams with high cache-hit rates — precisely the teams
doing the right thing — and under-charges the ones whose prompts bust the cache.

4. **Allocate shared cost explicitly.** GPU idle time, the reserve, the platform team itself. An
   attribution model that only allocates marginal cost creates a tragedy of the commons: every team sees
   its own usage as cheap and the shared bill grows. The corpus's framing of the platform layer as a
   distinct tier `[T]` supports making the platform's own cost a visible line item.

**Showback versus chargeback** `[D]`: start with **showback** — visibility, no money moving. It changes
behaviour on its own in most organisations, because teams optimise what they can see, and it generates no
incentive to game the attribution. Move to chargeback only once the attribution model has survived a
quarter of arguments, because a wrong chargeback is much worse than a wrong dashboard: it creates
perverse incentives that take longer to unwind than the savings it produces.

**One specific thing to get right:** attribute **cached tokens separately**. If a team's harness stability
produces a 70% cache-hit rate, that should show up as a cost advantage for them (T19-Q4). Attribution that
hides it removes the incentive to keep prefixes stable, which is the application-team behaviour the whole
caching lever depends on.

**Signal:** Builds attribution at the gateway, prices with the cache-aware formula, and starts with
showback for the incentive reason.

**Follow-ups:**
- *Where must attribution happen?* — the gateway; the tenant context is gone by the scheduler.
- *Why separate cached tokens?* — otherwise you remove the incentive for prefix stability.
- *Showback or chargeback first?* — showback; a wrong chargeback creates perverse incentives.

**Red flags:** Allocates only marginal cost, or uses a flat per-token rate that ignores cache hits.

---

#### T19-Q15 · Where does the AI budget actually go?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** A team's inference bill tripled with flat user numbers. Where do you look first?

**Model answer:** In this order: retries, context growth, and agent loops — then the serving configuration.

**One: retries and failures** `[D]`. Failed requests consume tokens and produce nothing. An eval-gated
pipeline that started failing, a flaky tool call, a schema violation causing regeneration — each turns one
billable task into two or three. The metric is cost per *successful* task (T19-Q9), and it is the first
thing that moves when something upstream breaks. If the team is only watching cost per token, this is
invisible.

**Two: context growth** `[D]`. Cost scales with tokens *in*, and tokens-in grows with conversation
length, retrieved-context size and tool-result size. A change that added a retrieval step, enlarged the
tool definitions, or lengthened the system prompt raises the per-turn cost of every request — often
silently, because nobody re-measured the prompt. In an agent, this compounds: every turn re-sends the
accumulated context. The corpus's agent material puts it directly — an agent's cost profile is dominated
by the fact that **the context grows within a session** and the same prefix is charged for repeatedly
unless cached `[T]`.

**Three: agent loops** `[D]`. The corpus documents a **10–100× agentic compute multiplier** over a single
chat turn `[T]`, and the failure signature where a loop does not terminate and cost runs away. The
controls are step, token and wall-clock budgets enforced structurally, not advisory (T16-Q8). If the bill
tripled, check whether step counts per task tripled — that is a graph you can plot per day and it will
show the change immediately.

**Four: the serving configuration** `[D]`. Only after the above: did a cache get invalidated by a prompt
change (T19-Q4), did routing start missing the warm replica, did a quantisation get rolled back? The
corpus's own diagnostic is **prefix cache hit rate per pod** `[T]` — if it dropped, the problem is
upstream of the engine.

**What I would instrument to answer this in future** `[D]`: tokens per task, steps per task, cache-hit
rate, retry rate, and cost per successful task — five series that between them localise any cost change to
one of the four causes above within a day.

**Signal:** Orders the investigation from semantic causes (retries, context, loops) to infrastructure
causes, and names the metrics that localise each.

**Follow-ups:**
- *Which is most often the cause?* — context growth and retries, because neither is visible in a token
  price.
- *What is the agent-specific multiplier?* — 10–100× over a chat turn `[T]`.
- *What catches it?* — steps per task and cost per successful task, plotted per day.

**Red flags:** Starts by tuning the serving engine, or has no per-task metrics and proposes reading the
invoice.

---

### Sovereignty

#### T19-Q16 · What is sovereignty, and why is the flag framing wrong?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** A customer asks whether your AI deployment is "sovereign." How do you answer?

**Model answer:** By refusing the binary and describing it as a system property — which is the corpus's
own framing `[T]`.

**The dimensions** `[T]`: the corpus enumerates **control, choice, trust, economics and continuity**.
Note the honest ambiguity the speaker flags: "choice" is often folded into "control", so you can
legitimately describe it as **four or five dimensions depending on how strictly you split it** `[T]`. A
candidate who recites five confidently when the source itself flags the ambiguity is showing less care
than one who notes it.

- **Control** — who can change the model, the weights, the serving stack, the data flows. Can you turn it
  off? Can you refuse a change?
- **Choice** — whether you can move: to another provider, another model, another deployment topology, or
  on-premises, without rebuilding.
- **Trust** — verifiability. Can you inspect what is running, and can you demonstrate it to someone else?
- **Economics** — whether the cost structure is one you can bear and predict, and whether value accrues
  where the work happens.
- **Continuity** — whether the arrangement survives an acquisition, a contract expiry, an export control
  change, a vendor pivoting away from the product.

**Why the flag framing is wrong** `[T]`: sovereignty is not something a deployment *has* or *does not
have*. It is a **system property** with a degree in each dimension, and the right profile depends on the
requirement. A regulated bank and a startup asking about data residency have different answers and both
can be served well. The binary framing forces a yes/no where the honest answer is a profile.

**Why the framing matters practically** `[D]`: a binary answer invites a binary procurement question
("are they sovereign?"), which is unanswerable, and it hides the dimension where the customer is actually
exposed. The useful conversation maps the requirement to a dimension — "you need continuity and control;
here is where this deployment stands on each, and here is the migration path" — and that conversation is
both more honest and more likely to close.

**Signal:** Gives the dimensions rather than a flag, notes the four-or-five ambiguity as the source does,
and reframes the question as a profile against a requirement.

**Follow-ups:**
- *How many dimensions?* — four or five; the source flags "choice" as sometimes folded into "control" `[T]`.
- *Which is most often missing?* — continuity; it is about what happens later.
- *Why not just answer yes or no?* — it hides which dimension the customer is exposed on.

**Red flags:** Answers "yes, we're sovereign" or "no, we're not," or lists dimensions without noting the
framing point.

---

#### T19-Q17 · The inference layer is a sovereignty component
**Difficulty:** L3 · **Depth expected:** 3–4 min

**Question:** A team has self-hosted training and keeps its weights on-premises, but calls a hosted API for
inference. Are they sovereign?

**Model answer:** Not on the dimension that matters most in production, and this is the corpus's central
sovereignty point: **the inference layer itself is a sovereignty component that people forget** `[T]`.

**Why it is forgotten** `[D]`: sovereignty discussions cluster around *artefacts* — weights, training data,
checkpoints — because those are what you can point at and lock in a room. Inference is a *running
service* with a different property: it handles every live request, and it operates continuously. The
corpus's supporting observation is the asymmetry: **"you train once… but inference is not like that"** `[T]`.
Training is episodic; inference is the ongoing operation, and it is where the data actually flows.

**The consequence** `[D]`: a team that controls its weights but not its inference has sovereignty over a
snapshot and not over the operation. The prompt — which contains the customer data, the retrieved context,
the live request — leaves the boundary on every call. If the requirement is about the *data*, weight
custody does not satisfy it.

**Where it sits in the stack.** The corpus lays out the stack as **Linux → accelerators → models and
architectures → inference engines (vLLM) and distributed inference → application** `[T]`. The point is
that sovereignty is a property of the whole stack, and the inference-engine layer is the one people skip —
they will argue about accelerators and about weights and never ask who runs the vLLM deployment.

**So my answer to the team** `[D]`: on **control** and **trust**, the hosted inference call is a gap:
you cannot inspect what runs, and you cannot refuse a change. On **continuity**, the exposure is that the
API's terms, model or availability can change without your consent. Whether that matters depends on the
requirement — a workload with no sensitive prompt data may be fine — but it should be a *decision* taken
against a stated dimension, not an oversight. And the corpus's own framing supports the practical test:
**"if inference leaves your control, can you really call that sovereign AI?"** `[T]`

**Signal:** Names the inference layer specifically as the forgotten component, uses the train-once/
inference-continuous asymmetry, and locates the gap on named dimensions rather than declaring failure.

**Follow-ups:**
- *Why is inference more consequential than training?* — it is continuous and handles live data `[T]`.
- *Where does it sit in the stack?* — between models/architectures and the application `[T]`.
- *Which dimensions does it break?* — control and trust, and continuity if terms can change.

**Red flags:** Treats weight custody as sufficient, or cannot say what specifically is lost when inference
is hosted.

---

#### T19-Q18 · When compliance drives the topology
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** A regulator requires that customer data never leaves the country, and inference must be
performed there. Redesign the deployment. What breaks?

**Model answer:** The constraint propagates through every layer, and the interesting work is the
second-order breakage rather than the regional deployment itself.

**The direct change** `[D]`: a regional inference deployment — in-region accelerators, in-region
storage for KV and any cache, in-region model artefacts, in-region logging and trace storage. The
corpus's stack framing helps here: the requirement attaches to the *inference engine* layer as much as to
the model, so "we self-host the weights" is not sufficient (T19-Q17).

**What breaks, in the order I would expect** `[D]`:

1. **Observability.** Traces and logs are the most commonly overlooked data flow. A trace containing
   prompt text is customer data leaving the region. The corpus's own guidance — **redact at the boundary,
   before storage** `[T]` — becomes mandatory rather than best practice, and the tail-sampling policy may
   have to change because a sampled trace is still a data export (T17).
2. **Caching.** A prefix cache holding customer context cannot be replicated across regions, so you lose
   the ability to serve a global request from a warm replica anywhere. Effective cache-hit rate falls,
   which raises cost — this is the efficiency-forfeiture direction from T19-Q13.
3. **Routing.** A global router that picks the least-loaded replica can no longer do so; routing becomes
   constraint-satisfying rather than cost-optimising, and the corpus's routing material on cache-aware
   routing `[T]` now has to operate within a partition (T14).
4. **Model choice.** Frontier models served from a single region may simply be unavailable in-region. The
   fallback is a locally-served open-weights model, which changes quality, cost and the eval baseline —
   and every one of those needs re-measurement.
5. **Failure and capacity.** A single-region deployment has no cross-region failover for the constrained
   workload, and in-region capacity is finite. The autoscaling design (T15) has to be re-derived rather
   than copied, and the SLO may need to be renegotiated because the capacity pool is smaller and less
   elastic.
6. **Third-party services.** Any hosted dependency in the request path — a guardrail model, an eval judge,
   an embedding API — is now potentially a data export. The corpus's four-rail model makes this concrete:
   a rail implemented by a hosted model sends the content it judges to a third party (T18).

**How I would approach it** `[D]`: treat the constraint as a **partition boundary** and re-derive every
design that assumed a global pool. Then, crucially, **price the forfeiture explicitly** — the lost cache
hit rate, the smaller capacity pool, the fallback model's quality delta — because the organisation should
see that compliance has a cost and it is not zero. Presenting it as free is how the constraint gets
quietly violated later under cost pressure.

**The structural point** `[T]`: this is a structural decision in the corpus's sense — topology is not a
tuning knob, and the efficiency ladder available afterwards is smaller than before. That is fine if the
requirement is real; the error is not pricing it (T19-Q13).

**Signal:** Traces the constraint through observability, caching, routing and third-party calls rather
than stopping at "deploy in the region," and insists on pricing the forfeiture.

**Follow-ups:**
- *Most overlooked breakage?* — traces and logs; they carry prompt text `[T]`.
- *What happens to the cost ladder?* — the caching lever is constrained, so effective cost rises.
- *Why price the forfeiture?* — an unpriced constraint gets violated quietly under budget pressure.

**Red flags:** Treats it as a deployment-location question only, or claims the design meets the
requirement while traces and hosted rails still export data.

---

#### T19-Q19 · Quantisation, fingerprinting and disclosure
**Difficulty:** L5 · **Depth expected:** 4 min

**Question:** You quantise a model for cost. How does that interact with sovereignty?

**Model answer:** Through two channels — one obvious and one that surprises people — and both are about
what your deployment reveals or fails to control.

**The obvious channel: quality drift on your own eval.** Quantisation changes outputs (T19-Q3), and if the
model is a sovereignty-relevant asset — a fine-tuned model whose behaviour *is* part of what you control —
then a quantisation you cannot evaluate is a change to an asset you cannot inspect. The corpus's rule,
**always rerun evals after quantising** `[T]`, is doing double duty here: it is a quality gate and a
control check.

**The non-obvious channel: fingerprinting.** The corpus states that the quantisation scheme is inferable
from outputs — quantisation **fingerprints the model** `[T]`. So the *implementation choice* is observable
to anyone who can query the deployment, which means:

- If the deployment's configuration is confidential — a competitive or security-relevant fact — the choice
  of quantisation leaks it.
- A fingerprint lets an observer distinguish two deployments serving the same weights, which matters for
  attribution and for anyone trying to determine whether a hosted endpoint is the same system as the one
  they evaluated.
- Combined with the corpus's point about **model fingerprinting via quantisation in the inference
  material** `[T]`, it means "we quantised" is a fact you should assume is discoverable rather than
  internal.

**What I would do about it** `[D]`: treat quantisation as a **disclosure decision** as well as a cost
decision. That means recording *which* scheme was applied and when, so that an observed fingerprint can be
correlated with a version rather than guessed at; and — for a deployment where the configuration is
genuinely confidential — recognising that the standard 8-bit or 4-bit schemes advertise themselves. There
is no general fix; the point is to make the choice consciously rather than to discover it in a competitor's
analysis.

**The broader framing** `[D]`: cost work and sovereignty work are not separate tracks. Every efficiency
lever has a control, trust or disclosure consequence — caching has data-retention implications, routing
has data-residency implications, quantisation has disclosure implications — and the corpus's own structure
puts them in one topic for that reason. A cost optimisation reviewed only against the bill is a control
change reviewed by nobody.

**Signal:** Names fingerprinting as a disclosure channel rather than a quality issue, and frames
quantisation as a control decision reviewed alongside the cost one.

**Follow-ups:**
- *What is the fingerprinting risk?* — the scheme is inferable from outputs, so the configuration leaks `[T]`.
- *What is the mitigation?* — no general fix; treat it as a recorded disclosure decision.
- *Why do cost and sovereignty belong together?* — every efficiency lever has a control consequence.

**Red flags:** Treats quantisation as purely a cost/quality tradeoff with no disclosure dimension.

---

#### T19-Q20 · Efficiency gains that increase sovereignty risk
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** Give me a case where a cost optimisation made a deployment less sovereign.

**Model answer:** I would use caching, because it is the clearest case and the one teams are actively
encouraged to pursue.

**The case** `[D]`: prefix caching (T19-Q4) is one of the largest steps on the cost ladder and its
guidance is unambiguous — keep prefixes stable, make routing cache-aware, fill the KV cache. And caching
means **retaining customer-derived state** — the KV for a prefix that contains a system prompt, retrieved
documents, or conversation history — for longer, across more requests, in a tier that may be shared. Every
one of those is a data-retention decision made by a cost optimisation.

Concretely, the questions it raises `[D]`:

- **How long does a cached prefix live, and who else can hit it?** A prefix cache keyed only by content
  hash can serve a hit to a different tenant whose prefix happens to match — which is a cross-tenant leak
  if the prefix contains tenant-specific content.
- **Where does the cache tier live?** The corpus pairs caching with an **offload tier** where KV is
  explicitly created and evicted through an API `[T]`. If that tier spans a partition or a jurisdiction,
  the residency analysis from T19-Q18 now has to cover the cache.
- **What does eviction guarantee?** The corpus's agent material has the router consuming both **create and
  evict events** `[T]`, which is exactly the mechanism a retention policy needs — but only if someone has
  decided what the retention period should be. A cache with an unbounded lifetime is a data-retention
  decision made by default.

**The general shape** `[D]`: **efficiency levers buy money with state.** Batching buys compute with
latency. Caching buys compute with retained state. Routing buys utilisation with data locality. Each
purchase has a control consequence, and each is invisible if the optimisation is reviewed only against the
bill.

**What I would do** `[D]`: attach a **retention and scope decision to every caching change**, the same way
an eval gate attaches to every quantisation change. Concretely: what is the cache's lifetime, what keying
does it use, is it tenant-scoped, where does the tier live, and does eviction actually erase? Five
questions, answerable in a design review, and they convert a silent control change into a recorded one.

**The symmetric case worth mentioning:** routing for cost can also be a sovereignty gain — in-region
routing is a sovereignty requirement satisfied by a topology decision — which is why this is not
"optimisation is bad" but "optimisation is a control change."

**Signal:** Picks a concrete lever and traces the state-retention and residency consequences, then
proposes attaching a retention decision to caching changes as a standing practice.

**Follow-ups:**
- *What does caching buy and with what?* — compute, with retained customer-derived state.
- *What is the cross-tenant risk?* — a content-keyed prefix cache can serve a hit across tenants.
- *What is the control?* — a retention and scope decision attached to every caching change.

**Red flags:** Cannot produce a case, or treats caching as control-neutral because it is "just memory."

---

#### T19-Q21 · Sovereignty as a system property — worked example
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** Take a specific deployment — a European bank running an agentic customer-service product —
and assess its sovereignty profile.

**Model answer:** I would score it across the dimensions from T19-Q16 and be specific about where each
answer comes from, because the value of the exercise is the profile, not a verdict.

**The deployment as I would want to see it** `[D]`: an agent that reads customer records, drafts
responses, and can issue small credits; a self-hosted open-weights model in an EU region; tools calling
core banking APIs; a hosted judge model for evaluation; traces stored in a third-party observability
platform.

| Dimension | Assessment | Where it comes from |
|---|---|---|
| **Control** | **Strong on weights, weaker on the stack.** Self-hosted model means you can refuse a model change. But the serving engine, the observability platform and the judge are third-party, so much of the running system is not yours to change. | T19-Q17 |
| **Choice** | **Moderate.** Open weights give a migration path between accelerators and engines. The core-banking tool integrations and the eval harness built around a specific judge model are the switching costs. | `[D]` |
| **Trust** | **Mixed, and this is the weak one.** You cannot inspect the judge model, and evaluation results are only as verifiable as the judge. Traces in a third-party platform are the other gap — and they contain prompt text, which for a bank means customer data. | T18-Q25, T17-Q12 |
| **Economics** | **Moderate.** Self-hosting converts to fixed cost, so cost depends on utilisation you have to sustain. The hosted judge and observability are variable and predictable. | T19-Q10 |
| **Continuity** | **Moderate–strong.** Open weights and self-hosting are good for continuity. The agent framework, the judge, and the observability vendor are the exposures if any of them changes terms or is acquired. | `[D]` |

**The three findings I would raise** `[D]`:

1. **The eval judge is the sovereignty hole.** A hosted model judging customer-service quality sees the
   customer's data. That is a data flow the regulator's question is about, and it is easy to miss because
   evaluation is not "the production path." Either run the judge in-region or evaluate on redacted
   samples — with the caveat that redaction changes what you are measuring.
2. **Traces are the second hole.** Same mechanism — prompt text leaving the boundary into an observability
   vendor — and the corpus's remedy (redact at the collector, before storage `[T]`) applies directly.
3. **The agent's tool scope is a control question, not only a security one.** Whether the agent can issue
   credits without approval is simultaneously a blast-radius question (T18-Q15) and a control question in
   the sovereignty sense — the organisation's ability to refuse an action it did not sanction.

**What I would *not* say:** that the deployment is or is not sovereign. The bank's point is that the
useful output is the profile plus the two named holes, because those are actionable and the verdict is
not.

**Signal:** Produces a dimension-by-dimension profile with the source of each assessment, and identifies
the evaluation and tracing data flows as the sovereignty gaps rather than the model weights.

**Follow-ups:**
- *Which dimension is weakest?* — trust; the judge and the trace store are unverifiable third parties.
- *Why is the eval judge a sovereignty issue?* — it sees customer data on the evaluation path.
- *What is the deliverable?* — a profile with named gaps, not a verdict.

**Red flags:** Answers yes or no, or audits only the model and the weights.

---

#### T19-Q22 · When sovereignty and cost conflict
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** A sovereignty requirement costs 40% more than the non-compliant option. Product wants the
cheaper one. How do you handle it?

**Model answer:** I would establish which conversation we are in, then make the cost of the requirement
visible rather than arguing the requirement.

**Step one: is the requirement actually a requirement?** `[D]` The corpus's distinction helps: if a
regulator or a contract mandates it, this is not a cost decision at all and the 40% is the price of doing
business — arguing it wastes credibility (T19-Q13). If it is an internal preference, a customer's
aspiration, or a competitive positioning choice, then it *is* a cost decision and should be argued as one,
in the open, with the profile as the evidence.

That distinction is the first thing to establish, and it is usually ambiguous at the start — which is
itself the finding worth surfacing.

**Step two: reduce the requirement's cost using the efficiency ladder** `[D]`. A 40% gap is often
partly recoverable without touching the constraint. Working the ladder (T19-Q1) inside the compliant
topology narrows the gap — batching and quantisation are available regardless of topology; caching is
available within the region even if it cannot span regions (T19-Q18). This changes the argument from
"compliance costs 40%" to "compliance costs 15% after optimisation," which is a much easier conversation
and is also the honest number.

**Step three: price the alternative's risk** `[D]`. "The cheaper option" has a cost too: the probability
of a compliance finding, the contractual exposure, the migration cost if the requirement becomes
mandatory later. That is a structural reversal cost in the corpus's sense (T19-Q13), and it is usually
larger than a 40% running difference. I would put that number in front of the decision-maker — not to
win the argument, but because they are choosing between two options and only one of them has been priced.

**Step four: offer the middle** `[D]`. Partial compliance is frequently available: in-region inference
with redacted traces, a regional cache tier with a shorter retention, an in-region judge. The bank's
profile framing (T19-Q16) is designed for exactly this — because sovereignty is a degree, not a flag, a
partial answer is a legitimate design rather than a compromise.

**What I would not do:** claim the requirement is free, or claim the cheap option is unsafe. Both are
usually false, and the second destroys the credibility needed to argue the first.

**Signal:** Establishes whether the requirement is mandatory before arguing cost, then reduces the gap
with the efficiency ladder and prices the alternative's reversal risk.

**Follow-ups:**
- *First question?* — is it a real requirement, or an internal preference.
- *How do you narrow the gap?* — work the efficiency ladder inside the compliant topology.
- *Is partial compliance legitimate?* — yes; sovereignty is a degree, so a profile is a design.

**Red flags:** Argues the requirement is free, or treats the choice as binary when a partial profile
exists.

---

### Open design

#### T19-Q23 · Build the cost dashboard
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Design the cost and efficiency dashboard for an inference platform. What is on it?

**Model answer:** Five layers, each answering a different question — and the discipline is not to collapse
them, because a dashboard that shows only the bill cannot explain it.

**Layer 1 — the bill.** Total spend, by capacity class (reserved, on-demand, spot) and by model. This is
what finance sees and it is the only layer that is lagging: by the time it moves, the cause is weeks old.

**Layer 2 — unit economics** `[D]`. Cost per token, cost per task, and **cost per successful task**
(T19-Q9). The third is the business-facing one and the one that catches retries and failures.

**Layer 3 — the efficiency levers** `[T]`. The ladder, instrumented: batch size distribution, quantisation
in effect and its eval status, **prefix cache hit rate per pod** — the corpus's diagnostic for bad routing
`[T]` — and the offload tier's create/evict rates where that exists.

**Layer 4 — workload shape** `[D]`. Tokens in and out per request, steps per task, retries per task,
context length distribution. This is the layer that explains *why* layer 1 moved: context growth and loop
count are the two most common causes (T19-Q15) and neither is visible in a token price.

**Layer 5 — attribution** `[D]`. Cost by tenant and by team, with **cached tokens priced separately**
(T19-Q14) so that harness discipline shows up as an advantage rather than being averaged away.

**What I would deliberately put on the same screen** `[D]`: cost per successful task next to the eval
pass rate. The corpus's Token Raj story is precisely a case where cost per token improved while the
workload and billing diverged `[T]`, and the only way to see that shape early is to put the cost metric
and the quality metric side by side. A cost dashboard without a quality series adjacent to it is how a
team optimises itself into a regression.

**What I would leave off:** model-level cost comparisons in isolation. "Model A is cheaper per token than
model B" is not actionable unless it is paired with the outcome rate, because the cheaper model with a
lower success rate can cost more per outcome.

**Signal:** Builds the dashboard in layers with distinct questions, and insists on quality sitting beside
cost rather than on a separate dashboard.

**Follow-ups:**
- *Which layer is most often missing?* — workload shape; it is what explains the bill.
- *Why price cached tokens separately?* — to keep the incentive for prefix stability.
- *Why put quality on the cost dashboard?* — the Token Raj shape is invisible otherwise `[T]`.

**Red flags:** Designs a bill dashboard with no unit economics, or separates cost and quality entirely.

---

#### T19-Q24 · The 90-day cost programme
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** You are given a mandate to cut inference cost 50% in a quarter, with no quality regression
allowed. Plan it.

**Model answer:** I would sequence it by the ladder's order and by organisational difficulty (T19-Q5),
with a measurement gate before each step and an explicit statement of what I will not do.

**Weeks 1–2: establish the baseline and the gates.** `[D]` This is the step teams skip and it is the one
that determines whether the rest works. Two things are needed: real per-request and per-tenant token
accounting `[T]`, and an eval suite that can run in CI and detect a regression. Without the first I cannot
prove a saving; without the second I cannot take the quantisation step at all, because the mandate forbids
a quality regression and the only way to demonstrate compliance is to measure it.

**Weeks 2–4: batching.** `[D]` The largest single step and the only one with no quality cost (T19-Q2).
Verify continuous batching is on, tune the maximum batch size against the latency SLO, and check that
scheduling is not being defeated by an application pattern. Expected: the 100 → 42 step, or its equivalent
from wherever the actual baseline sits.

**Weeks 4–7: caching and routing.** `[D]` This is where the largest remaining win is and where the
application team becomes necessary — prefix stability is a harness property, not a server flag (T19-Q4).
Concretely: audit prompt construction, move anything volatile out of the prefix, keep tools stable, and
make the router cache-aware. Measure **prefix cache hit rate per pod** `[T]`; if it is low, the finding is
an application change, not a serving one. Expected: a large share of the 0.42 factor, realised only if
hit rate actually rises.

**Weeks 7–10: quantisation, gated.** `[D]` Applied only with the eval gate in place, and verified on both
quality and variance because quantisation worsens non-determinism `[T]`. This is deliberately last among
the technical steps despite being second on the ladder, because it is the only one that can breach the
"no quality regression" constraint, so it should be attempted once everything cheap and safe is done and
the team has headroom to evaluate properly.

**Weeks 10–13: the workload-level levers** `[D]`. Context management and routing policy at the application
level — smaller retrieved context, tighter tool definitions, prompt-length discipline, and model routing
that sends easy requests to a smaller model. These often exceed the serving-layer wins at this stage
because the serving layer is now efficient and the remaining cost is the shape of the work.

**What I would say explicitly at the start** `[D]`: I will report cost per successful task alongside cost
per token, because the mandate forbids a quality regression and the token metric cannot detect one
(T19-Q9). And if the levers are exhausted before 50%, I will report the frontier honestly with the
residual attributed — because the alternative is a team that hits the number by degrading something
nobody measured, which is exactly the Token Raj failure shape `[T]`.

**Signal:** Sequences by ladder order with measurement gates, holds quantisation until the eval gate
exists, and pre-commits to reporting cost per successful task rather than cost per token.

**Follow-ups:**
- *Why is quantisation not second, given the ladder?* — it is the only step that can breach the no-
  regression constraint, so it waits for the eval gate.
- *Where does the application team enter?* — caching; prefix stability is harness work `[T]`.
- *What if the ladder runs out?* — report the frontier with the residual attributed; do not degrade an
  unmeasured thing.

**Red flags:** Starts with quantisation because it is the visible technical win, or reports progress on the
token metric with no quality series.

---

#### T19-Q25 · Efficiency as a product feature
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Should efficiency work be invisible infrastructure, or a product-visible feature?

**Model answer:** Both, for different audiences — and getting the split wrong is why efficiency
programmes stall.

**Infrastructure-invisible is right for the mechanism** `[D]`. Nobody should have to know that continuous
batching exists. If a product team has to reason about the serving configuration, the platform has leaked.
The corpus's framing of a serving engine as an abstraction — vLLM sitting between the model and the
application in the stack `[T]` — is exactly this: the engine's job is to make the application not think
about it.

**Product-visible is right for the constraints the product controls** `[D]`, and this is the part teams
miss. Several of the largest efficiency levers are **application properties**, not server properties:

- **Prefix stability.** Cache hit rate is determined by how the harness builds prompts (T19-Q4). A product
  team that does not know this will bust the cache on every turn and then ask the platform team why costs
  are high.
- **Context discipline.** Retrieved-context size and tool-result size set the per-turn token count
  (T19-Q15).
- **Session reuse.** The corpus's agent material shows **roughly 5× TTFT improvement on a session's
  return** from keeping sessions warm `[T]` — a product behaviour with a large cost and latency effect.

So the split I would draw: **the platform hides the mechanism and surfaces the contract.** The contract is
a short list of properties the application must maintain — stable prefix, bounded context, session reuse,
tool list stability — expressed in product terms ("keep the prompt prefix stable so returning users are
cheap and fast") rather than in serving terms.

**Why this matters more than it sounds** `[D]`: when the contract is invisible, the failure mode is
blaming. The platform team sees a low cache-hit rate and concludes the application is badly built; the
application team sees a cost increase and concludes the platform is expensive. Both are describing the
same missing contract. Writing it down converts an argument into a checklist.

**And one thing I would make visible to end users** `[D]`: nothing about cost. Users should see effect —
faster responses from a warm session, for instance — not the internal economics.

**Signal:** Splits mechanism (hidden) from contract (surfaced), and names prefix stability and session
reuse as application-owned levers, not platform tuning.

**Follow-ups:**
- *What must the application own?* — prefix stability, context discipline, session reuse.
- *How do you express it?* — as a product contract, not serving configuration.
- *What breaks without it?* — blame flows in both directions for one missing contract.

**Red flags:** Treats all efficiency work as platform-owned, or exposes serving internals to product teams
instead of a contract.

---

#### T19-Q26 · FinOps for agents specifically
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Agent workloads break the standard FinOps approach. What is different and what do you do
about it?

**Model answer:** Four things differ, and each one defeats a standard cost-control mechanism.

**One: cost is per task, not per request, and the task has a variable length.** `[D]` A chat request has a
predictable token count. An agent task is a loop whose step count is decided at runtime and can vary by an
order of magnitude. The corpus documents a **10–100× agentic compute multiplier** over a chat turn `[T]`
and a failure signature where a loop does not terminate. Budgets therefore have to be enforced
*structurally* — step, token and wall-clock caps — rather than monitored, because by the time a monitoring
alert fires the money is spent (T16-Q8).

**Two: the cost is dominated by input, not output.** `[D]` Agents resend accumulating context every turn,
so the input side grows quadratically within a session while output stays small. A per-output-token
cost model is the wrong shape. The corpus's agent material puts the balance starkly — **98% prefill / 2%
decode** `[T]` — which means the thing to optimise is what goes *into* the context, not what comes out.
Concretely: cache the stable prefix, bound the retrieved context, and stop re-sending history that is not
needed.

**Three: the cost driver is a graph, not a stream.** `[D]` Standard FinOps attributes a cost to a request
or a tenant. Agent cost belongs to a *task* that spans many requests, possibly across services and
sub-agents. Attribution needs a task identifier propagated through the whole trace, or per-tenant
attribution fragments into unassignable per-call records. The corpus's observability material on trace
propagation (T17) is the mechanism.

**Four: efficiency gains get reinvested faster than in chat.** `[D]` The Token Raj shape — cost per task
falling while billing rises because thinking tokens rose `[T]` — is a *structural* property of agents,
because an agent's natural response to a cheaper token is to think longer. In chat, a cheaper token mostly
reduces the bill; in agents, it partly buys capability. So the metric has to be cost per *successful
outcome* from the start (T19-Q9), or the programme will report success while the bill grows.

**What I would build** `[D]`: task-level cost attribution with a propagated task ID; structural budgets
rather than alerts; per-task input/output token split with the input side as the optimisation target; and
cost per successful task as the headline, with the success rate from the eval and trajectory machinery
alongside it (T17). Plus one agent-specific metric: **steps per task**, which is the earliest indicator
that something has changed.

**Signal:** Identifies all four differences with the mechanism for each, and lands on task-level
attribution and structural budgets as the required infrastructure.

**Follow-ups:**
- *Why budgets not alerts?* — by the time an alert fires the spend has happened.
- *Where is the cost?* — the input side; 98% prefill / 2% decode `[T]`.
- *Why is the efficiency gain partly reinvested?* — cheaper tokens let the agent think longer `[T]`.

**Red flags:** Applies per-request attribution to agent workloads, or reports cost per token for an agent
product.

---

#### T19-Q27 · Design the efficiency architecture for a new product
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** Design the cost and efficiency architecture for a new agentic product from day one, before
you have any traffic. What do you build and what do you defer?

**Model answer:** I would build the measurement and the constraints now and defer the optimisations,
because the optimisations depend on traffic shape and the measurement does not.

**Build now, because retrofitting is expensive** `[D]`:

1. **Per-request token accounting with a propagated task ID** `[T]`. This is the one piece of
   infrastructure that must exist before traffic, because you cannot reconstruct history. Input, output,
   cached and fresh tokens, plus the task identifier spanning the whole agent trace (T17).
2. **An eval harness with a gate in CI** `[T]`. Before any model or quantisation choice, because it is
   the mechanism that makes the "no quality regression" constraint enforceable later (T19-Q3).
3. **Structural budgets on the agent loop** `[T]` — step, token and wall-clock caps. These are cheap to
   build and they are the only thing that stops a runaway loop before it spends (T16-Q8).
4. **A gateway that owns attribution and routing** `[D]`. Tenant context and cache-aware routing both
   live here, and adding it later means changing every client (T14).
5. **A stable prefix contract with the application team** `[D]`. Cheapest to establish at the start and
   hardest to retrofit, because by month six there are a dozen prompt templates that all need auditing
   (T19-Q25).

**Defer, because the traffic will tell you** `[D]`:

- **Quantisation.** It is a quality-risk decision and it needs the eval gate and a real workload to
  measure against. Do not make it before you can evaluate it.
- **Cache sizing and the KV fill target.** These depend on prefix repetition and context-length
  distribution, which you do not know yet (T19-Q7).
- **Capacity purchase structure.** Reserved versus on-demand depends on the load's predictability, which
  is observed, not designed (T19-Q11).
- **Self-host versus buy.** This is structural (T19-Q13) and it should be decided with utilisation data,
  not before (T19-Q10).

**The principle I would state** `[D]`: **measurement is cheap and reversible; optimisations are expensive
and sometimes irreversible.** Build the former eagerly and the latter late. The corpus's structure-versus-
efficiency framing is exactly this ordering rule (T19-Q13), and the specific trap is committing to a
structural choice — a capacity contract, a deployment topology, a model — before the efficiency data that
would have informed it exists.

**The one exception I would make to "defer":** if a compliance requirement attaches to the deployment
topology, that is structural and it must be decided now, because it forecloses everything else (T19-Q18).
So the rule is: defer optimisations; commit to requirements.

**Signal:** Separates measurement-and-constraints (build now, unretrofittable) from optimisations (defer,
traffic-dependent), and carves out compliance-driven structural choices as the exception.

**Follow-ups:**
- *What must exist before traffic?* — token accounting with task IDs, an eval gate, structural budgets.
- *Why defer quantisation specifically?* — it needs an eval gate and a real workload to measure.
- *What is the exception?* — a compliance-driven topology decision, which is structural and forecloses.

**Red flags:** Plans to add cost telemetry "once we have traffic," or commits to a capacity contract before
measuring utilisation.

---

#### T19-Q28 · What would you tell a CFO about inference cost?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You have fifteen minutes with a CFO who wants to know what inference will cost next year.
What do you say?

**Model answer:** Four things, in this order, and I would resist the urge to lead with the ladder.

**One: the number they want does not exist yet, and here is what would make it exist** `[D]`. Inference
cost is a function of workload shape, and workload shape is not yet determined — context lengths, session
lengths, step counts and retry rates all move it by multiples. What I can give them is a model with stated
assumptions and the *measurement plan* that will replace it with an observation within a quarter. A CFO
who has been given a number with hidden assumptions has been given a liability, not a forecast.

**Two: the efficiency ladder, as an ordered plan with the factors attributed** `[T]`. `100 → 42 → 26 →
11`, from batching, quantisation and caching-plus-routing, stated as the corpus's reported figures for a
representative workload rather than as a promise about ours `[T]`. The important part is the ordering
logic: the first step is free, the second needs an eval gate, the third needs the application team. That
tells them where the risk is and who has to move.

**Three: the metric that should be their headline is cost per successful outcome** `[D]`, not cost per
token — and I would give the Token Raj story to justify it `[T]`: cost per 10k-token task fell from 30¢
in February to 16¢ in April, drafted lines fell from 630 to 91, files touched from 8.2 to 3.6, **and the
billing fell less than the workload because thinking tokens rose**. That is a concrete case where the
efficiency improved and the spend did not fall proportionally, and it is the shape they should expect.
Better to hear it now than to explain it in a variance meeting.

**Four: which decisions are structural, and therefore need them now** `[T]`. Self-host versus buy, capacity
commitments and any compliance-driven topology choice are irreversible in the corpus's sense, and they
should be decided on strategy and requirements, not on a cost model built before the workload exists.
I would bring those forward as *their* decisions and everything else as mine.

**What I would refuse to do** `[D]`: give a single number for next year. It would be wrong, and — worse —
it would be quoted back at me in twelve months as the baseline I failed to hit. The deliverable from the
meeting is a model with visible assumptions, a measurement plan, and a clear list of which decisions are
theirs.

**Signal:** Refuses a point forecast, gives the ladder as an ordered plan with attributed factors, moves
the headline metric to cost per successful outcome, and separates structural decisions (the CFO's) from
efficiency ones (the team's).

**Follow-ups:**
- *Why refuse a single number?* — workload shape is undetermined; the assumption set would be hidden.
- *What is the right headline metric?* — cost per successful outcome `[D]`.
- *Which decisions are the CFO's?* — structural ones: capacity commitments, self-host, topology.

**Red flags:** Gives a confident point forecast, leads with the cost ladder as if the factors were
guaranteed, or reports cost per token as the business metric.

---

## Whiteboard exercises

### Exercise 1 — Cut 50% of inference cost in a quarter
**Prompt.** "You have one quarter, a mandate to halve inference cost, and a hard constraint that quality
must not regress. Draw the plan: what you do each week, what you measure before moving on, and what could
force you to stop."

**What the candidate must produce:** the ladder applied in order, the two gates that must exist before
step 2, and an explicit statement of the frontier if the target is not reached.

**Expected answer sketch:**

```
WEEK   ACTION                          GATE BEFORE MOVING ON
 1-2   BASELINE + GATES                <-- do not skip
       - per-request/per-tenant token accounting [T]
       - eval suite gated in CI [T]
       (no eval gate => quantisation is off the table,
        because the constraint is "no quality regression"
        and you cannot demonstrate it without measuring)

 2-4   BATCHING                        verify against the LATENCY SLO,
       (free move -- no quality cost)  not just the cost target
       100 -> 42                       <- the only step with no
                                          quality cost  [T]

 4-7   CACHING + ROUTING               prefix cache hit rate PER POD [T]
       (needs the APPLICATION team)    low hit rate => harness problem,
       -> 0.42 factor                     not a serving problem  [T]

 7-10  QUANTISATION (gated)            eval passed on QUALITY and on
       -> 0.62 factor                  VARIANCE (non-determinism worsens) [T]
                                       last because it is the only step that
                                       can breach the no-regression rule

10-13  WORKLOAD-LEVEL                   context length, steps per task
       context discipline, model routing, smaller context

REPORT ALONGSIDE THE COST LINE, ALWAYS
  cost per successful task  and  eval pass rate     [D]
  (Token Raj shape: cost per token improved, billing
   fell LESS than workload because thinking tokens rose [T])

STOP AND ESCALATE IF
  - the eval shows a real regression -> revert the step, do not ship it
  - hit rate will not rise without a product change -> that is a product
    decision, not a platform one
  - the ladder is exhausted before 50% -> report the frontier with the
    residual attributed; do not degrade an unmeasured thing   [D]
```

**Grading rubric:**
- Puts the measurement and eval gates in weeks 1–2 and says explicitly that quantisation is unavailable
  without them, because the constraint is no-regression and it is unverifiable otherwise.
- Applies the levers in ladder order with the correct per-step gate — latency SLO for batching, hit rate
  for caching, quality *and* variance for quantisation.
- Identifies caching as requiring the application team, and treats a low hit rate as a harness finding
  rather than a serving-configuration one.
- Commits to reporting cost per successful task with the eval rate alongside, and states what happens if
  the ladder exhausts before the target — the frontier reported honestly rather than an unmeasured
  degradation.

---

### Exercise 2 — Assess a sovereignty profile
**Prompt.** "A bank runs a customer-service agent: self-hosted open-weights model in-region, tools calling
core banking, a hosted judge model for evaluation, and traces in a third-party observability platform.
Assess the sovereignty profile and name the gaps."

**What the candidate must produce:** a dimension-by-dimension profile with the source of each assessment,
two named gaps, and no verdict.

**Expected answer sketch:**

```
DIMENSION      RATING        WHERE IT COMES FROM
Control        strong on     self-hosted weights = can refuse a model
               weights,      change; serving engine, judge and
               weaker on     observability are third party  [T]
               the stack
Choice         moderate      open weights = migration path; the banking
                             integrations and eval harness are the
                             switching costs                    [D]
Trust          WEAKEST       judge is unverifiable (see T18-Q25);
                             traces hold prompt text = customer data
Economics      moderate      self-host = fixed cost, depends on
                             sustained utilisation  [T]
Continuity     moderate/     open weights good; agent framework,
               strong        judge and observability vendor are the
                             exposures           [D]

THE TWO GAPS TO RAISE
  1. THE EVAL JUDGE -- a hosted model judging customer service
     SEES CUSTOMER DATA. It is a data flow on the evaluation
     path, which is why teams miss it.              [D]
     fix: in-region judge, or evaluate on redacted samples
          (and note redaction changes what you measure)
  2. TRACES -- prompt text leaves the boundary into a vendor.
     the corpus's own remedy: REDACT AT THE COLLECTOR,
     BEFORE STORAGE  [T]

DELIVERABLE IS A PROFILE, NOT A VERDICT  [T]
  sovereignty is a system property with a degree in each
  dimension, not a flag. The gaps are actionable; the
  verdict is not.
```

**Grading rubric:**
- Produces all five dimensions with a rating *and* the evidence behind each, and declines to give a
  yes/no verdict — noting the framing point that "choice" is sometimes folded into "control."
- Identifies the eval judge and the trace store as the gaps, and explains that they are missed because
  neither is on the "production path."
- Connects trust to the guardrail material — an unverifiable judge is also an unverifiable control
  (T18-Q25).
- Applies the corpus's own remedy for traces (redact at the collector, before storage) rather than
  proposing generic encryption.

---

### Exercise 3 — Decide on self-hosting
**Prompt.** "A team says self-hosting will cut their inference bill 5× based on GPU-hour pricing. The
workload is spiky: 3× weekday peaks, quiet weekends, growing 15% a month. Make the recommendation."

**What the candidate must produce:** the corrected break-even, the utilisation analysis, and a decision
that separates the economic question from the structural one.

**Expected answer sketch:**

```
WHAT THEIR 5x OMITS                            [T]
  - serving engineering headcount      <-- the corpus's explicit warning
  - datacentre / power / cooling         is that break-even must include
  - upgrades and model onboarding        OPS COST
  - IDLE CAPACITY THEY PAY FOR ANYWAY

CORRECTED SHAPE
  self_host = fixed (capex/lease + power + headcount + ops)
            + idle tail YOU own
  buy       = variable (tokens x price) -- you pay for the peak
              and nothing else

UTILISATION IS THE HINGE  [D]
  fixed cost rewards utilisation; variable cost rewards volatility
  spiky + 15%/month growth = the worst profile for a fixed-cost bet:
     - provision for the peak -> idle most of the week
     - provision for the mean -> SLO breach at every peak
     - reserved baseline + burst = hybrid, and the hybrid's
       COMPLEXITY is itself a cost

STRUCTURAL, NOT TUNED  [T]
  reversal cost ~ adoption cost. Migrating back, the contracts,
  the team you hired. Decide it on strategy, not a spreadsheet
  built before you know the utilisation.

RECOMMENDATION
  DEFER the commitment; BUY the information first  [D]
  - instrument per-request cost and utilisation       (T17)
  - run the efficiency ladder: 100 -> 42 -> 26 -> 11  [T]
    a 9x capacity reduction CHANGES THE REQUIREMENT entirely
  - pilot self-host on a bounded, PREDICTABLE slice
  - re-run the break-even with real numbers, then decide
  UNLESS a compliance requirement forces the topology -- then it is
  a constrained optimisation, not a cost decision  [T]
```

**Grading rubric:**
- Names the omitted terms, especially ops headcount and the idle tail, and cites the corpus's warning
  that self-host break-even must include ops cost.
- Identifies utilisation as the hinge and shows why spiky-plus-growing is the worst profile for a
  fixed-cost commitment.
- Sequences the efficiency ladder *before* the structural commitment, on the grounds that a 9× capacity
  reduction changes the requirement the commitment would be sized against.
- Distinguishes the economic question from the structural one, and carves out the compliance case where
  the break-even is irrelevant rather than pretending it applies.

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
  — the cost ladder (100 → 42 → 26 → 11) with its per-step factors, the batching/quantisation/caching
  ordering, cache reads ~10× cheaper, the precision journey 16 → 8 → 4 bit, and the rule "always rerun
  your evals after quantizing."
- `refs/Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt`
  — the sovereignty dimensions (control, choice, trust, economics, continuity) with the speaker's note
  that choice is often folded into control, sovereignty as a system property rather than a flag, the
  inference layer as a sovereignty component, the stack ordering, and "you train once … but inference is
  not like that."
- `refs/Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt`
  and the Token Raj material within the corpus — the February-to-April cost-per-task movement (30¢ → 16¢),
  drafted code lines (630 → 91), files touched (8.2 → 3.6), and the finding that billing fell less than
  the workload because thinking tokens rose.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — KV to 90% of VRAM, the saturation thresholds (KV 80%,
  active requests > 8), prefix cache hit rate per pod as the signal for bad routing, and the
  offload tier with create/evict events.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  (topic `T11` material, read for cross-reference) — the KV recomputation cliff at 28k inputs /
  concurrency 256 and the long-context behaviour that constrains cache fill.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/` — the serving-engine and routing
  reference material behind cache-aware routing, tiered SLOs and capacity structure.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_10_Incorporating_Tools.txt`
  and `..._11_Agents_and_Multi-Agent_Communication.txt` — the agent cost shape: context growth within a
  session, the 10–100× agentic compute multiplier, and the 98% prefill / 2% decode profile.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
  — cost as a first-class observability signal, per-request and per-tenant token accounting, and
  redaction at the collector before storage.
- `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
  — structural budgets, step/token/wall-clock caps, and the audit and attribution requirements that make
  chargeback possible.

**Derived content in this bank (`[D]`):** the cumulative-multiplication arithmetic behind the ladder's
11 and the ~9× figure; the owner-per-step table and the organisational-ordering argument in T19-Q5; the
four-cause decomposition of a missed saving in T19-Q6; the cost-per-task and cost-per-successful-outcome
formulas in T19-Q9; the corrected self-host break-even shape and the utilisation hinge in T19-Q10 and
Exercise 3; the convexity argument and tiered-SLO proposal in T19-Q12; the structural-foreclosure rule in
T19-Q13; the five-layer dashboard design in T19-Q23; the 90-day programme sequencing in T19-Q24 and
Exercise 1; the agent-specific FinOps analysis in T19-Q26; the build-now-versus-defer split in T19-Q27;
and the CFO framing in T19-Q28. Reported figures are attributed to their talks; the sovereignty dimension
enumeration is the speaker's, including the four-or-five ambiguity, which is noted wherever it appears.
