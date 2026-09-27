# Interview Bank: Continuous Batching, Scheduling & Fairness

> `T08` · **Transcript coverage:** partial · [Cheat sheet](../00-cheat-sheets/T08-batching-scheduling.md) · [Case study](../01-case-studies/T08-batching-scheduling.md) · [Design blueprint](../03-design-blueprints/T08-batching-scheduling/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer here is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a transcript statement with the speaker or talk named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here with every assumption shown. Where the corpus does not supply a figure, the answer says so rather than inventing one.

Questions are ordered to read as one interview: foundations, then mechanism, then the tradeoffs, then debugging, then design at scale. One warning carried from the case study, because it is the error this topic punishes hardest: **the starvation runs from the big session to the small ones.** A large session that takes every dispatch cycle starves the short sessions; the scheduler's job is to stop that, not to protect the large one.

---

### Foundations

#### T08-Q1 · Why does static batching waste most of a batch for LLMs?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Static batching works fine for image classification. Why is it the wrong shape for LLM serving?
**Model answer:** Because admission happens only at batch boundaries, and LLM response lengths are the least homogeneous workload there is. Static batching is the traditional ML pattern — "all requests must be the same size and start/end together" — and the corpus names the consequence: it is "inefficient for LLMs due to variable response lengths" `[R]`. The failure has a precise shape, and the corpus's worked example is the clearest statement of it: **"If one user asks for 500 tokens and another for 5 tokens, the GPU remains idle for the 5-token user for 495 cycles"** `[R]`. Idle here does not mean free — the slot is *held* by a sequence that finished at iteration 5 and cannot be replaced, because nothing new is admitted until the batch ends. Generalise: a batch of 32 requests with a p50 of 200 tokens and a p95 of 2,000 runs for 2,000 iterations; the median request finishes at iteration 200 and its slot is dead for the remaining 1,800, so the average slot is useful for 200 of 2,000 iterations — **10% occupancy, ~90% waste** `[D]` (assumptions: p50/p95 as given, batch runs to its longest member). The point to carry forward is that the loss scales with the *dispersion* of the length distribution, not with its mean. A homogeneous batch wastes nothing; production LLM traffic is never homogeneous. That arithmetic is what sits behind the corpus's throughput table — static 1x, continuous **4x–10x** `[R]`.

**Signal:** Gives the mechanism (admission only at batch boundaries) rather than "static batching is inefficient," and states that the penalty is a function of length dispersion.
**Follow-ups:**
- *How would you measure the waste?* — Slot-occupancy per iteration; the ratio of useful slot-iterations to total is the metric.
- *What does the 4x–10x depend on?* — The length distribution, which is why the corpus gives a range; at p95:p50 of 10:1 the arithmetic predicts more `[D]`.
**Red flags:** "Static batching waits for the slowest request" with no mechanism; believes batch size, not dispersion, drives the loss.

#### T08-Q2 · What does continuous batching actually change?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Define continuous batching. What is the one thing it moves, and what does it make possible?
**Model answer:** It moves the admission decision from the batch boundary to the iteration boundary. Continuous batching "(pioneered by **Orca** and **vLLM**) allows new requests to join the batch and finished requests to leave at the end of every individual token generation step" `[R]`. The corpus's comparison table gives the four consequences: join/leave is "**Any iteration**" rather than start/end; GPU utilisation is "High (always saturated)" rather than "Low (waiting for longest)"; throughput goes from 1x to **4x–10x**; latency becomes "Balanced" rather than "Highest for shortest" `[R]`. The mechanism that makes it affordable is the KV cache plus paged allocation — "Continuous batching allows the 5-token user's request to exit the GPU immediately after its last token, **freeing up VRAM and compute slots** for a new request from the queue" `[R]`. Two things to say that the table does not. First, **this is the floor, not the achievement**: continuous batching is table stakes in every engine in the corpus's May 2026 landscape, so its value is in what it makes possible — the fairness machinery only exists because admission happens per iteration. Second, it is a priced step on the cost ladder, not a free lunch: on the LLMOps talk's numbers, naive 16-bit serving one request at a time is 100 cost units and turning continuous batching on takes it to about 42 `[T]` (LLMOps cost talk).

**Signal:** Names the iteration boundary as the change, treats 4x–10x as a floor rather than an achievement, and connects it to what it enables.
**Follow-ups:**
- *Why a range and not a number?* — It is a function of the length distribution `[D]`; the corpus gives 4x–10x `[R]`.
- *What does it cost?* — Engine state and scheduler complexity, and it presupposes paged KV — Q3.
**Red flags:** Calls it "dynamic batching" with no iteration-level detail; does not connect the freed slot to the allocator.

#### T08-Q3 · Why does continuous batching need paged KV?
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** Continuous batching frees a slot when a request finishes. What has to be true for that freed slot to actually be reused?
**Model answer:** The allocator has to be able to hand it back as usable space. In a contiguous allocator a sequence's KV occupies one span sized to its maximum length, so when a 5-token request finishes at iteration 5 the span is released as a hole of the wrong shape — a new request whose maximum length exceeds the hole cannot use it. The engine then holds plenty of free bytes and nowhere to put the next sequence, and the symptom is a replica that refuses work while reporting free memory. Paged allocation fixes this: KV is allocated in fixed-size blocks with a per-sequence block table, so a sequence is a list of blocks and any freed block is immediately reusable by any sequence `[R]`. The scheduling consequence is the part that matters: **continuous batching's join/leave is only real if the allocator can reclaim.** Without paging, the scheduler makes its admission decision against a number that overstates available capacity, and from the outside that is indistinguishable from a scheduler that has stopped admitting — the failure signature is "GPU idle while requests queue." There is a second reason the two techniques ship together, and it is the fairness link: uniform blocks are what make a per-tenant KV *slot* share a countable, enforceable quantity. That is what makes tiered iteration-level scheduling a mechanism rather than an advisory (Q12).

**Signal:** Connects the allocator's shape to the scheduler's admission decision, rather than reciting "paged attention reduces fragmentation."
**Follow-ups:**
- *What is the fairness link?* — Uniform blocks make per-tenant slot shares countable; Q12.
- *What is the symptom of getting it wrong?* — Free memory, no admission, jobs queueing — Q23.
**Red flags:** Says paging "saves memory" without saying what the scheduler does with the reclaimed blocks.

#### T08-Q4 · Which metric does each mechanism actually move?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A platform lead asks what each scheduling technique buys. Give them the metrics, not the names.
**Model answer:** The metrics are TTFT, TPOT/ITL, throughput and goodput, and the corpus gives defaults of "< 200 ms / < 30 ms" for TTFT/TPOT — explicitly "defaults, not requirements" `[R]`. Now map. **Static batching** moves nothing useful. **Continuous batching** moves throughput — 1x to 4x–10x `[R]` — and by reclaiming slots stops the shortest requests being worst-served. **In-flight batching** is the only mechanism that moves *both phases at once*, because prefill uses idle compute while decode uses memory bandwidth `[R]`. **Chunked prefill** moves TPOT for everyone *except* the request being chunked, and deliberately worsens that request's own TTFT `[R]`. **Fairness scheduling** moves per-tenant tail latency with no throughput cost — the llm-d work reports request latencies down by up to 50% (2–3x) while "overall token throughput increased a bit but … the request latencies … came down by a lot" `[T]` (llm-d talk, Pravin). And the metric that should decide everything is **goodput**, `|{r : latency(r) ≤ SLO}| / |requests|`, because throughput rises monotonically with batch size while goodput does not. The named trap is over-batching: throughput fine, P99 terrible.

**Signal:** Separates TTFT from TPOT from goodput and assigns each mechanism to the metric it moves — including the one it makes worse.
**Follow-ups:**
- *Why goodput rather than throughput?* — It is the only one carrying the SLO; Q15.
- *Which mechanism improves both phases?* — In-flight fusion, and it is the only one — Q7.
**Red flags:** Uses "latency" for all four; optimises throughput and reports it as the SLO metric.

#### T08-Q5 · One fleet, two physical regimes. Explain that.
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Prefill and decode are often described as two workloads. What actually differs, and why does it matter to a scheduler?
**Model answer:** They are bound by different resources, and the fleet's traffic contains both. Prefill reads the whole prompt in one parallel pass, so it is **compute-bound** and it sets TTFT; decode writes one token at a time, so it is **bandwidth-bound** and it sets TPOT `[R]`. The case study's estate puts the two ends side by side: the catalogue enrichment job submits 5,000 short requests in a burst and wants throughput, while the merchandising copilot holds a 60k-token product history and wants its next token in under 50 ms. The classification jobs are effectively prefill-only — "Classification is a **Prefill-only** task; it processes the entire input and produces a single output in one parallel pass, making it compute-optimal" `[R]` — while the copilot is entirely decode-bound. This is "one fleet, two physical regimes," and the reason it is a scheduling fact rather than a curiosity is the asymmetry: decode at batch 1 runs at an arithmetic intensity roughly 150x below the hardware's ridge point ([T06](../01-case-studies/T06-inference-fundamentals.md)), so it leaves the compute units idle, and prefill is the mirror image. **A fleet that mixes the regimes in one iteration fills both resources; a fleet that serves them in separate phases leaves one idle at all times.** The corollary is the first thing the scheduler must know about a request — its class — before any fairness rule applies.

**Signal:** Explains why the two phases are *complementary*, not merely different, and derives the mixing opportunity from that.
**Follow-ups:**
- *Which phase does a long prompt hurt?* — Prefill: it delays TTFT, and unchunked it delays everyone's TPOT — Q8.
- *Which phase does batching help?* — Decode: bandwidth-bound, so a larger batch amortises the weight read.
**Red flags:** Treats prefill and decode as "the same thing at different speeds"; cannot say which resource each is bound by.

#### T08-Q6 · Where does a prefill-only workload belong in the schedule?
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** A classification service sends 5,000 requests with no expected output tokens. Do you put it in the general queue?
**Model answer:** No — it is a class of its own, and mixing it blindly is a named failure. Classification is "a **Prefill-only** task; it processes the entire input and produces a single output in one parallel pass, making it **compute-optimal**" `[R]`. Two consequences the scheduler has to own. First, it never produces decode rows, so a batch made largely of classification work has no bandwidth-bound work to hide behind — the fusion that makes in-flight batching cheap (Q7) has nothing to fuse with. Second, it is the *cheapest* request from a KV standpoint and the *most expensive* from a compute standpoint, exactly the inverse of a long streaming generation. So the failure of blind mixing is precise: the classification burst consumes the prefill budget that chunked prefill was protecting for interactive TTFT. Same resource, and the interactive class loses. In the case study's fleet this is why the three batch classification jobs are a distinct class whose own SLO is the fleet's — "Batch-class goodput | Maximise; no latency SLO | Its own SLO is the fleet's." The rule is therefore not "batch work is low priority" but **"prefill-only traffic is a distinct physical regime and gets its own share of the prefill budget,"** which is what the per-iteration cap on prefill chunks enforces (§6).

**Signal:** Identifies that a prefill-only class competes for the *prefill budget* specifically, not for general capacity.
**Follow-ups:**
- *What breaks if you mix them?* — The burst eats the interactive class's prefill share and TTFT collapses.
- *What is this class's metric?* — Throughput and finish time; TTFT and TPOT are meaningless for a single-output request.
**Red flags:** Treats classification as "just cheap requests"; gives it high priority "because it is fast."

---

### Mechanism

#### T08-Q7 · In-flight batching: what exactly is mixed, and why does it help both metrics?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Explain in-flight batching. Why is it the only technique in this topic that improves TTFT and TPOT together?
**Model answer:** It mixes the two phases inside one iteration. The corpus describes the older model and the fix: "Previously, serving engines processed a batch of 'Prefill' (heavy compute) OR a batch of 'Decode' (heavy memory). **In-Flight Batching** (TensorRT-LLM) allows mixing them: 1 request is in the Prefill phase. 15 requests are in the Decode phase. **Benefit**: The Prefill request utilizes the GPU's idle compute cores while the Decode requests utilize the memory bandwidth" `[R]`. Why it improves both: the two phases are bound by different resources, so they are not in competition. Decode at batch 1 sits at an arithmetic intensity roughly 150x below the hardware's ridge point ([T06](../01-case-studies/T06-inference-fundamentals.md)) — memory-bound, compute units idle. Prefill is compute-heavy and bandwidth-light. Fusing them means each phase uses what the other leaves idle, so the prefill's wall time is partly *hidden* rather than charged to the batch. Note the ratio in the corpus's example — one prefill to fifteen decodes — and treat it as a **tuning parameter, not a constant**. The mix is what the per-iteration plan decides, and getting it wrong in either direction has a named failure: too much prefill and the decode streams stall (Q8); too little and the prefill queue grows unbounded and TTFT collapses. That is why the plan is explicit about "which prefill chunks, which decode rows" per iteration (§3) — ambiguity here becomes a latency incident.

**Signal:** Derives the both-metrics property from the ridge-point asymmetry rather than asserting it, and treats the 1:15 ratio as a decision.
**Follow-ups:**
- *Is the 1:15 ratio fixed?* — No; it is the corpus's illustrative example `[R]` and a per-iteration decision.
- *What failure does this mechanism create?* — The stall — Q8. This technique is the cause of the problem the next question fixes.
**Red flags:** "Prefill and decode at the same time" with no resource argument; treats 1:15 as a configuration constant.

#### T08-Q8 · The stall: what is it, how big is it, and who pays for it?
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Define an LLM serving "stall." Then tell me the size of its blast radius.
**Model answer:** Definition: "Massive context prompts (1M+ tokens) can hang a batch for seconds during the Prefill phase, causing '**stalls**'" `[R]`. Mechanism: "A 'stall' occurs when a massive new request arrives and its Prefill phase (which is compute-hungry) takes **2-3 seconds** to complete. During this time, the GPU is so busy with the prefill that it **cannot generate tokens for existing users in the 'Decode' phase**, causing their **TPOT to spike**" `[R]`. Now size it. If 15 decode streams are in flight `[R]`, each is delayed by the *full* stall — the delay is not shared out, it is charged to every stream in parallel. Against the copilot's 50 ms P95 TPOT target, a 2.5 s stall is a **50x breach** `[D]` (assumptions: 2.5 s is the mid-point of the corpus's 2–3 s range; the 50 ms target is the case study's interactive P95 target). It is experienced as one frozen second by every interactive user on that replica. That is the organisational shape of the problem: **one request's arrival becomes every tenant's incident**, and the platform team is blamed for a breach caused by another team's job. Two things a strong answer adds. The stall is a *scheduling* failure, not an admission failure — the request was admitted legitimately, and the scheduler chose to put its whole prefill in one iteration. And the fix (chunked prefill) does not remove the cost; it redistributes it, which is the subject of the next two questions.

**Signal:** Gives the mechanism *and* the arithmetic of the blast radius, and volunteers that the affected parties are not the request that caused it.
**Follow-ups:**
- *Why can't it just be queued?* — It was admitted; the failure is how its prefill was composed into an iteration.
- *What bounds it?* — Chunk size — Q9 and Q10.
**Red flags:** "Some requests are slow"; does not know the delay is charged to every in-flight stream concurrently.

#### T08-Q9 · Chunked prefill: the mechanism and the arithmetic
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** How does chunked prefill fix the stall, and what does it cost? Show me the numbers.
**Model answer:** The mechanism: "Instead of prefilling 128k tokens at once, the engine breaks the prefill into smaller chunks (**e.g., 4k tokens each**) and interleaves them with the ongoing Decode steps of other users. This maintains a steady **TPOT** even when heavy requests arrive" `[R]`. The corpus's own timing: it "breaks that 3-second prefill into small **200ms chunks**, processing one chunk and then doing one round of decoding for everyone else, before returning to the next prefill chunk" `[R]`. The arithmetic `[D]`, assumptions stated: a 128k-token prompt at 4k tokens per chunk is `128,000 / 4,000 = 32` chunks; at ~200 ms per chunk that is `32 × 200 ms = 6.4 s` of prefill wall time, against the 2–3 s the corpus quotes unchunked. So the trade is two-sided and both sides are real: the large request's own TTFT worsens by roughly 2.6x, while the worst-case TPOT disruption to every co-tenant falls from 2–3 s to ~200 ms — about a **12.5x reduction in tail disruption for that 2.6x cost**. Chunked prefill deliberately makes the large request slower in order to bound everyone else's tail. Two operational notes. Chunking bounds only the *duration* of the disruption; a pathological arrival mix can still consume every iteration's prefill slots, so the fraction of each iteration allocated to prefill chunks must be capped separately (§6). And the chunk cannot shrink indefinitely — below some size the per-iteration dispatch overhead dominates, which is why the corpus's 4k example is a starting point rather than a law.

**Signal:** Produces the chunk arithmetic unprompted and frames the trade as a two-sided exchange rather than an improvement.
**Follow-ups:**
- *What if you chunk at 16k instead of 4k?* — Tail disruption rises to ~800 ms and the request's own TTFT falls to ~1.6 s; correct only if long prompts are rare `[D]` from the case study's sensitivity table, which holds per-chunk time constant.
- *What must be capped in addition to chunk size?* — The prefill share per iteration, or decode starves.
**Red flags:** Describes chunked prefill as "reducing latency"; cannot produce a chunk count from the numbers.

#### T08-Q10 · Your chunk size is a statement about your tenancy model. Defend that.
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Chunk size looks like a performance knob. Argue that it is actually an architecture decision.
**Model answer:** It is the exchange rate between one tenant's TTFT and every other tenant's TPOT, which makes it a statement about who shares the device. On the case study's numbers a 4k chunk buys a ~12.5x reduction in tail disruption for a ~2.6x increase in the large request's TTFT `[D]` (32 chunks × 200 ms = 6.4 s `[D]` against 2–3 s unchunked `[R]`). Whether that is a good trade depends entirely on who is on the other side of it. **On a shared fleet it is obviously correct.** With 15 concurrent decode streams `[R]`, one chunked prefill saves `15 × 2.3 s = 34.5 s` of aggregate disruption for `3.9 s` of added TTFT — roughly a **9x return, and it grows with the number of co-tenants** `[D]` (assumptions: 2.3 s saved per stream is the 2.5 s stall minus the 200 ms chunk; 3.9 s added is 6.4 s chunked minus 2.5 s unchunked; the case study's break-even computation). **On a dedicated replica it is obviously wrong**: there are no co-tenants, so the return is zero and the 3.9 s is a real cost paid for nothing. The direction of the dial is symmetric: 4k → 16k raises tail disruption to ~800 ms and cuts TTFT to ~1.6 s, which is correct only if long prompts are rare `[D]` (case study §8 sensitivity). So the honest framing is that you are not choosing a chunk size — you are declaring how many tenants share the device and how much of their tail you are willing to buy with one request's TTFT. That is why the runbook tunes it first: it is the TTFT-versus-TPOT dial and the cheapest thing to change.

**Signal:** Reframes a tuning knob as a policy statement, prices the return against the co-tenant count, and states the condition under which the answer inverts.
**Follow-ups:**
- *What would change your chunk size?* — Co-tenant count and the arrival distribution of long prompts.
- *When would you disaggregate instead?* — When long-context traffic makes the interference cost exceed a second hop's operational cost — Q27.
**Red flags:** Treats chunk size as a throughput knob; cannot say when the trade reverses; quotes the 12.5x or 9x as a measured benchmark rather than derived arithmetic.

#### T08-Q11 · Tiered iteration-level scheduling: what does it do that a queue cannot?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Noisy neighbours are handled by "tiered iteration-level scheduling." Unpack that — and say why a prioritised queue is not enough.
**Model answer:** A queue schedules *requests*; this schedules *iterations*, which lets it do three things a queue cannot. The corpus's statement: "Each tenant is assigned a 'share' of the total GPU cycles. In the continuous batching loop, the scheduler ensures that a single tenant doesn't occupy 100% of the KV cache slots. If Tenant A is overwhelming the system, the scheduler will prioritize 'Prefill' steps for Tenant B and C, or only process a subset of Tenant A's decode iterations per cycle" `[R]`. Extract three properties. **The enforcement point is the KV slot, not the request** — a tenant that cannot acquire blocks cannot grow its batch, however many requests it has queued, which is what makes the policy real rather than advisory. **The scheduler can prefer prefill for a starved tenant** — possible only because it composes each iteration; a queue would have to serve whole requests, so its only lever is ordering. **Fairness is enforced at two layers**: gateway token buckets for coarse rate limiting, engine admission for per-iteration control. Why a queue-based policy fails is the same reason FCFS fails — the llm-d team states plainly that first-come-first-served "doesn't add anything extra to your policies because … it's going to slow down everything" `[T]` (llm-d talk, Pravin). The cost of the tiered approach is honest: per-tenant accounting sits in the hot loop of every iteration, and shares are configuration with an owner and a review date.

**Signal:** Names the KV slot as the enforcement point and explains why the prefill-preference trick *requires* iteration-level composition.
**Follow-ups:**
- *What if shares are static and demand is not?* — They go stale; the revisit trigger is a tenant consistently exceeding its share off-peak.
- *Why not give each tenant a replica?* — It destroys the pooling that justified the purchase; it is an explicit non-goal.
**Red flags:** Describes fairness as request prioritisation; cannot say where it is enforced.

#### T08-Q12 · A tenant is inside its gateway quota and still consuming all the KV slots. Now what?
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** The gateway says the noisy tenant is under its limit. Co-tenants say their latency has collapsed. Which one is wrong?
**Model answer:** Neither — this is the designed-in gap in the fairness posture, not a tuning bug. The gateway's token bucket is coarse and edge-visible: it counts tenant requests and **cannot see the GPU's instantaneous state**. The engine's per-iteration admission can see the state and **cannot see tenant identity at the edge**. The corpus requires both: "This is enforced at the Gateway via **token-bucket rate limiting** and at the serving engine via **specific scheduling policies**" `[R]`. The failure when only the gateway layer exists is exactly this question: the requests arrive inside the budget, and each one, once admitted, is entitled to KV blocks until it finishes. **Quota on arrival is not quota on occupancy**, and occupancy is what the co-tenants experience. Fix and diagnostic are the same thing: enforce at the engine with **per-tenant KV slot caps**, and monitor **per-tenant KV slot occupancy against the cap** as the fairness metric. The arithmetic on the case study's fleet is simple and reassuring: 14 tenants with a 1/14 cap, so a single tenant can occupy at most ~7% of the fleet's slots `[D]` (assumption: equal shares across 14 tenants, §8). The residual question is whether 7% is enough for the burst tenant to finish its own work — and on these numbers it is, because 5,000 requests × 200 tokens = 1M output tokens is on the order of 16 GPU-seconds of work spread across the fleet `[D]` (assumption: ~2,000 tokens/s per GPU under continuous batching, 32 GPUs). The line to keep is the case study's: the batch class **"is not actually expensive; it is only disruptive when it is allowed to take the whole batch."**

**Signal:** Separates arrival-rate quota from occupancy quota and points at the KV slot as the surface that actually binds.
**Follow-ups:**
- *What is the metric?* — Per-tenant KV slot occupancy against the cap, not the gateway counter.
- *What if the 7% cap is too tight for the tenant to finish?* — Then shares must be demand-weighted rather than equal; the case study's 14→40 sensitivity is where that breaks.
**Red flags:** Proposes lowering the gateway quota; cannot name the two enforcement layers or which one binds.

#### T08-Q13 · What signal do you autoscale on, and what is wrong with the obvious ones?
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** You are asked to put an autoscaler on an inference fleet. Which metric?
**Model answer:** KV cache utilisation, and the corpus is unusually specific: "**Autoscaling**: Scaling based on **KV Cache utilization** rather than CPU or standard memory usage" `[R]`. Why each alternative fails deserves its own reason. **CPU utilisation** is meaningless for GPU inference and will essentially never trigger. **Standard memory utilisation** is misleading because a paged allocator's behaviour is not host-OS-like. **QPS** is a proxy for load, not capacity — a long-prompt request and a short one count the same, so it breaks under variable prompt length. **Compute utilisation** is worse than useless, and the reason is the case study's scaling trap: a replica at 100% KV utilisation and 30% compute utilisation will not be flagged by any conventional autoscaler, and it will be the one adding latency, because in a paged engine **the binding resource is KV blocks** — compute sits idle while requests queue for blocks. The signal must therefore lead latency rather than lag it, and KV occupancy does: it rises before the queue does. Two practical notes. The metric has to be exported **per replica**, or the autoscaler cannot act on the right one. And goodput — `|{r : latency(r) ≤ SLO}| / |requests|` — is the secondary signal once per-class SLO instrumentation exists, because it aligns scaling with the actual goal rather than with the resource.

**Signal:** Names the binding resource as the *reason*, and volunteers the 100%-KV / 30%-compute replica as the one conventional autoscalers miss.
**Follow-ups:**
- *What is the failure if you get it wrong?* — Latency degrades at "normal" utilisation with no scale-out — Q23.
- *Is the signal sufficient on its own?* — No; it is only useful if a new replica is ready in seconds — Q14.
**Red flags:** "Scale on GPU utilisation"; proposes a QPS threshold as the primary signal.

---

### Tradeoffs

#### T08-Q14 · Why is cold-boot time part of the autoscaling decision?
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** The autoscaler fires correctly. Why does that not settle the matter?
**Model answer:** Because an autoscaler that reacts correctly and arrives late has not helped. The corpus's number: using "**Un-quantized Base Images**" and loading weights from a high-speed Lustre/mount reduces startup time "from **minutes to 15-20 seconds**" `[R]`. The case study's framing is the one to reproduce: **autoscaling signal and cold-boot time are one design.** A KV-pressure autoscaler with a 15-second boot against a 30-second latency-degradation window is a working autoscaler; the same autoscaler against a 3-minute boot is a dashboard. Two mechanism notes that show understanding rather than recall. The reason *un-quantized* base images matter is that dequantizing at boot is work a pre-quantized image has already done, so it moves a fixed cost off the critical path of every scale-out. The reason a fast mount matters is that the weight load dominates at these model sizes. Both are cheap changes sitting on the critical path of every replica you will ever start. The operational consequence: cold-start time is a monitored metric in its own right (§10), because it bounds the effectiveness of everything built on top of it. And there is a second-order effect worth volunteering — with a slow boot, an operator's rational response is to over-provision a warm buffer pool, which is idle capacity the finance committee is already watching.

**Signal:** States the signal-and-boot-time coupling as *one* design decision and can say what each of the two speedups actually fixes.
**Follow-ups:**
- *What is the metric to watch?* — Time from scale decision to replica ready, end to end.
- *What if the boot cannot be made fast enough?* — Pre-warm a buffer pool and accept the idle-capacity cost explicitly.
**Red flags:** Treats cold start as an ops detail unrelated to autoscaling; proposes scaling earlier without noting the capacity cost.

#### T08-Q15 · Throughput is up and P99 is worse. Is the system healthy?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A tuning change raised tokens/second by 30% and p99 latency by 60%. Ship it?
**Model answer:** No — this is the classic over-batching own-goal, and the metric that says so is goodput. Throughput is monotonically increasing in batch size up to a point; goodput, `|{r : latency(r) ≤ SLO}| / |requests|`, is not. Past the knee you are converting latency into throughput no user benefits from, because the requests that gained throughput are the ones missing their SLO. The corpus's failure signature is exactly "Throughput fine, P99 terrible" with cause "over-batching" and the fix "goodput, not throughput." The knobs that trade this way are `--max-num-seqs`, which raises throughput and raises ITL, and the prefill share per iteration, which trades TTFT against TPOT stability. There is a second reason this question matters on a shared fleet: an over-batched replica is memory-saturated, so the requests it holds longer hold KV blocks longer, and co-tenants feel it as a capacity problem rather than as an over-batching problem. So the answer has two parts — the objective is goodput, and the tuning order reflects it: raise `--max-num-seqs` until ITL crosses the SLO, then stop, because the stop condition is an SLO and not a plateau. What not to say: "raise the batch size until throughput stops improving," which is the instruction that produces this failure.

**Signal:** Refuses the framing, names goodput, and knows that the batch-size knob trades throughput against ITL rather than raising both.
**Follow-ups:**
- *Which knob first?* — `--max-num-seqs` to raise throughput until ITL crosses the SLO; then chunked prefill for TTFT.
- *How does over-batching look to a co-tenant?* — KV exhaustion; it presents as a capacity incident.
**Red flags:** Reports throughput as the health metric; proposes raising batch size to fix P99.

#### T08-Q16 · A large session takes every dispatch cycle and two short sessions starve. Diagnose the fairness problem.
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Three agentic sessions share a replica. The scheduler is fair at the request level and users still complain. What is wrong, and what is the fix?
**Model answer:** Get the direction right first, because it is the thing this topic punishes hardest. The problem is that **one large session monopolises the scheduler and the short sessions starve** — *not* that short sessions slow the long one down. The llm-d account is explicit: "there is a small session and a large session that comes in and then takes all the dispatch cycles. So the shorter sessions are starving now" `[T]` (llm-d talk, Pravin). The reason request-level fairness cannot see it is that the unit of work has changed: "the unit of work transfers from being a single request to being a session" `[T]` — an agentic program is many requests across many turns, so per-request round-robin is perfectly fair to the requests and grossly unfair to the programs. The fix is **least-attained service** applied to sessions: compute `attained(s) = service_received(s) / demand(s)` and dispatch `argmin_s attained(s)` — the session furthest behind its fair share, not the oldest request. The second-order effect is the part worth volunteering, because it is counter-intuitive and it is the whole justification: "short sessions finish much faster and leaving the room for larger sessions to also finish much faster." The reported result is latencies down by up to 50% (2–3x) with "overall token throughput increased a bit but … the request latencies … came down by a lot" `[T]` — so session fairness here is not a throughput tax; it is close to free, and it speeds the monopolist up too, because a finished session's KV is released. The complementary strategy is **turn priority**: prioritise agents near completion so they finish and evict their KV, which is what helps when the constraint is KV saturation rather than compute. The speaker notes the two are effectively opposite mechanisms and that integrating them was active work `[T]`.

**Signal:** Gets the starvation direction right, applies fairness per session rather than per request, and knows the large session benefits too rather than being sacrificed.
**Follow-ups:**
- *Why session-level and not request-level?* — The unit of work is a multi-turn program; request-level fairness cannot see it.
- *What is the opposite strategy, and when do you use it?* — Turn priority, under KV saturation `[T]`.
- *What triggers the saturation regime?* — Operator-defined, e.g. KV 80% full or average active requests > 8 `[T]`.
**Red flags:** Says short sessions starve the long one (the direction is inverted); treats FCFS as a neutral default; assumes fairness must cost throughput.

#### T08-Q17 · Tensor or pipeline parallelism for a latency-critical tenant?
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** A model does not fit comfortably on one GPU. Do you reach for TP or PP, and why?
**Model answer:** Tensor parallelism, unless the model does not fit at all. The mechanism, from the corpus's own interview answer: "TP performs the matrix multiplications of a single layer across multiple GPUs simultaneously. This means the latency of that layer is reduced by the number of GPUs. PP, conversely, processes different layers sequentially… For a single user's request, PP adds the latency of all GPUs, whereas TP divides the latency across all GPUs" `[R]`. The table adds the operational facts: TP is "used for **90% of production serving** within a single node (8x GPUs)" and "requires NVLink"; PP is "used only for massive models spanning multiple nodes" and its cost is lower utilisation from **bubble time** `[R]`. So the ordering for a shared fleet is **replicate first, then TP, then PP**: replication adds throughput with no per-request penalty, TP reduces per-request latency at a communication cost, and PP adds per-request latency outright, so it should never be chosen for a latency-sensitive tenant unless the model cannot be hosted any other way. The scale reference that forces the multi-node conversation at all: "**Llama 4 405B requires ~800GB VRAM**" `[R]`. Two consequences worth stating. TP's latency gain evaporates without fast interconnect, so NVLink is a *procurement* requirement for a latency SLO, not an implementation detail. And PP is not a "slower but cheaper" option — its bubble makes its throughput worse too.

**Signal:** Gives the layer-level mechanism rather than the label, and can state the ordering and the condition for each.
**Follow-ups:**
- *What forces PP anyway?* — A model that does not fit a node; ~800 GB for a 405B `[R]`.
- *What does TP require?* — NVLink-class interconnect within the node `[R]`.
**Red flags:** "PP is for big models, TP is for small ones"; assumes TP always wins regardless of interconnect.

#### T08-Q18 · Why is a Layer 4 load balancer wrong for LLM traffic?
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** You are putting a load balancer in front of a replicaset. Does the layer matter?
**Model answer:** Yes, and the reason is that the connection is not the unit of work. "LLMs are almost always served via **Server-Sent Events (SSE)** or **WebSockets**" `[R]`, which are long-lived, so a Layer 4 balancer makes a connection-level decision once and pins that client to that replica for the whole session. The consequence is that a busy replica keeps receiving connections while an idle one receives none, because the balancer has no event on which to re-balance. The fix the corpus names: "Standard load balancers (**Layer 4**) struggle with long-lived AI connections. **The Fix**: Use **Layer 7 Load Balancers** (Envoy/Istio) that understand the '**End of Sequence**' token and can **re-balance traffic between user turns** rather than just at the connection level" `[R]`. Two distinct problems are named and it is worth separating them: the **stickiness** problem (a long connection pins a client) and the **semantic** problem (knowing where a turn ends requires an application-layer concept — the end-of-sequence token). Only L7 addresses both. The third piece, which the case study folds into the same section, is that the gateway's "**Context Tracker**" deliberately routes a user's prompt cache to "the **same GPU node** (**Sticky sessions**)" `[R]` — the *opposite* of load balancing. A fleet cannot have perfect stickiness and perfect balance at once; it chooses per class (Q19).

**Signal:** Separates the stickiness problem from the semantic one, and volunteers that L7 is also the enabler of prefix-sticky routing.
**Follow-ups:**
- *What does "between turns" buy?* — It is the only safe point to move a session; mid-turn would break the stream.
- *How does this interact with prefix caching?* — Directly and adversarially — Q19.
**Red flags:** "L4 is too slow"; no awareness of SSE/WebSockets or the end-of-sequence token.

#### T08-Q19 · Prefix caching wants stickiness and load balancing wants spread. How do you choose?
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Every request has a long shared prefix. Your balancer spreads load perfectly and cost went up. Explain and fix.
**Model answer:** The conflict is structural, not accidental: prefix caching's entire value is that a request's computed prefix is still resident on a node, and round-robin routing destroys that. The llm-d account of the naive balancer: "you start going to the old pods and you start recomputing again. So which wastes a lot of GPU cycles and basically reduces your … token throughput and also increases your TTFT" `[T]` (llm-d talk). The chosen policy is **prefix-sticky routing with L7 rebalancing at turn boundaries**, plus a **hot-prefix escape hatch**: after a deviation threshold, route to a second node and accept the cache miss. Its cost is real — uneven load, and it requires the gateway's Context Tracker `[R]`. Three refinements worth volunteering. First, **precise beats approximate**: the llm-d work found precise KV-cache routing, driven by create/evict events from the engine, "perform much better than the approximate," because under agentic load KV eviction is continuous and a hash-based approximation goes stale `[T]`. Second, the alternatives are not free either — consistent hashing on prefix is deterministic and survives node changes but is imbalanced under skew; always-least-loaded gives the best balance with zero locality; sticky-by-session is simple but equally imbalanced and longer-lived. Third, **a cache miss is slow, not wrong**, so the failure is graceful: recompute. That is what makes the escape hatch safe to pull rather than a correctness risk. The revisit trigger is quantitative — when replica load skew exceeds the point where imbalance costs more than the cache misses it saves.

**Signal:** Frames it as a class-scoped choice with a named escape hatch, and knows a miss is graceful rather than an error.
**Follow-ups:**
- *Which metric decides?* — Cache hit rate, which routing changes silently affect; §10 monitors it for exactly this reason.
- *What does the gateway need to route on prefix?* — The Context Tracker, plus engine KV events for the precise variant `[T]`.
**Red flags:** Picks one policy globally; claims stickiness is free; treats a cache miss as a correctness bug.

#### T08-Q20 · One house engine for everything, or engine per workload?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You run chat, a JSON function-calling backend, a multimodal surface and an MoE model. One engine or several?
**Model answer:** Engine per workload, and the corpus states it as a decision rather than a preference: "the right answer is **engine-per-workload** rather than a single house engine" `[R]`. The May 2026 landscape that drives the split `[R]`: **vLLM v0.18.2+** for public mixed-traffic chat where the patching cadence matters most; **SGLang v0.4.3+** on the **text-only** path for JSON function-calling, on a reported **~29%** throughput advantage from async constrained decoding; **TensorRT-LLM** for a single latency-critical model where peak NVIDIA throughput is worth the operational cost; and disaggregated prefill for long-CoT reasoning at low concurrency. For MoE, either vLLM v0.18+ or SGLang v0.4.3+ with a MoE scheduler. Two caveats the corpus attaches, both adopted as hard rules: the ~29% figure is **vendor-published** (April 2026, text-only, structured output only) and the version numbers and security claims are the supporting repo's assertions as of May 2026 — **both must be verified against the projects' own advisory feeds and release notes before any procurement or upgrade decision**, which is what the corpus itself instructs. And the operational posture: "Always be on a patched version"; run a canary on a **second engine** at 1–5% of traffic; "Treat the engine as part of the deployment manifest… Pin all four." The cost of the split is honest, and the case study says so: running two engines with a canary is a posture; running four across workload classes is a product with its own build, test and upgrade pipeline (§11). So the answer is scoped — a house default plus one specialised engine on a canary, and more engines only when a workload's category demands it.

**Signal:** Gives the mapping *and* the verification posture for the figures behind it, and prices the operational complexity of the split.
**Follow-ups:**
- *What is the anti-pattern?* — Four engines before the pipeline to upgrade them exists.
- *What changes the answer?* — A workload's category shifting, or an engine's security posture changing.
**Red flags:** Quotes the ~29% as an established benchmark; picks an engine on throughput alone with no patching story.

---

### Debugging

#### T08-Q21 · Every tenant's p99 TPOT rises at once. Walk the diagnosis.
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** p95 TPOT on one replica goes from 42 ms to 910 ms for every tenant on it, then recovers. Debug it.
**Model answer:** The simultaneity localises the fault before any data is collected. All tenants on the replica moved together, so the cause is on the replica, not in any tenant's configuration — and it is not a fairness failure, because a fairness failure is tenant-scoped by definition (Q22). Ranked hypotheses, cheapest first. **(1) A long prefill, unchunked or under-chunked.** The instrument is TPOT by class correlated with prompt-length arrival times; if the spike lines up with the arrival of a long-prompt request, the mechanism is the stall — "the GPU… **cannot generate tokens for existing users in the 'Decode' phase**" `[R]`. **(2) The prefill share consumed every iteration.** Chunking bounds the *duration* of the disruption, not the share of the iteration, so a pathological arrival mix can still starve decode; check the prefill:decode mix and prefill chunk occupancy per iteration. **(3) An engine config regression.** Continuous batching is the 4x–10x baseline and a config regression silently removes it, so verify it is on after every upgrade. **(4) A cold MoE expert set**, if the model is expert-routed — adding load reducing throughput is that failure's signature (Q25); a non-monotonic throughput-versus-batch-size curve confirms it. Discriminate on the KV number: a prefill block shows KV high while decode waits, whereas a capacity wall shows KV pinned and admission rejections climbing. The action for (1) and (2) is the same and it is the cheapest dial in the runbook — reduce the chunk size, then cap the prefill share per iteration. For (3) it is a rollback; for (4) it is routing-aware scheduling, not a latency fix.

**Signal:** Uses the simultaneity to localise before hypothesising, and puts the prefill budget ahead of more exotic causes.
**Follow-ups:**
- *What is the decisive measurement?* — TPOT by class against prompt-length arrival times.
- *Why is "reduce chunk size" first?* — It is the TTFT-versus-TPOT dial and the cheapest thing to change (§10).
**Red flags:** Blames one tenant; proposes adding capacity; does not check whether chunked prefill is actually on.

#### T08-Q22 · One tenant's SLA breaches while everyone else is fine. Diagnose.
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Search breaches its 400 ms TTFT contract. The other thirteen tenants are healthy. What is happening?
**Model answer:** The blast radius is one tenant, so the cause is tenant-scoped: this is a fairness failure, not a stall — the stall is fleet-wide by construction because it charges its delay to every in-flight stream. The instrument is **per-tenant KV slot occupancy**, and the question is whether one tenant is holding more than its share. The likely finding is that a batch job's burst is occupying slots the interactive tenant needs, and the reason it could is the two-layer gap: a gateway token bucket limits arrivals and cannot see GPU state, so a tenant under quota can still consume every slot it acquires (Q12). The fix is to **enforce at the engine** with per-tenant KV slot caps — a tenant that cannot acquire blocks cannot grow its batch, whatever it has queued. The confirmation is §10's fairness metric: per-tenant KV slot occupancy against the cap. Two secondary hypotheses deserve naming so they are eliminated rather than assumed away. **A routing change** may have sent that tenant's traffic to a replica that is full; cache hit rate falls after a routing change and so does that tenant's TTFT (Q24). **A prefix-cache miss** on its hot prefix would show as recomputation cost — slow, not wrong. And if the cap is already in place and still breached, run the diagnostic the case study names: compare gateway counters against engine KV occupancy. A mismatch means the gateway layer is measuring arrivals rather than occupancy, which is the whole bug.

**Signal:** Uses the blast radius to eliminate the stall hypothesis immediately, then goes straight to per-tenant slot occupancy.
**Follow-ups:**
- *What if occupancy is within the cap?* — The tenant's own demand grew; shares are stale and need demand-weighting.
- *What is the permanent fix?* — Engine-level caps, with gateway buckets as a complement only `[R]`.
**Red flags:** Reaches for chunked prefill for a single-tenant symptom; proposes per-tenant replicas as the first move.

#### T08-Q23 · Latency is degrading, compute utilisation is flat, and the autoscaler has not fired.
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** Users are complaining, the dashboard looks calm, and no new replica has appeared. What is broken?
**Model answer:** The autoscaler is on the wrong signal, and this is the case study's named scaling trap rather than an anomaly. A replica at 100% KV utilisation and 30% compute utilisation will not be flagged by any conventional autoscaler, and it will be the one adding latency — because in a paged engine the binding resource is KV blocks, and GPU compute sits idle while requests queue for blocks. So a compute-based or host-memory-based signal is both lagging and misleading: it fires, if at all, long after latency has degraded, and in exactly the regime where scaling is needed it may never fire. Two things to verify before changing a threshold. First, that the KV utilisation metric is actually **exported per replica** — the corpus's chosen signal "breaks if the metric is not exported per replica" — because an autoscaler reading a fleet-level average will not see one saturated node. Second, whether the deployment is hitting an admission or queueing limit before memory: queue depth is a useful cross-check because it distinguishes "busy" from "blocked on KV." The fix is to switch the signal to KV cache utilisation and to assert in CI that it is KV and not CPU — §10's deploy step 3 exists precisely because this regression is easy to reintroduce. There is also a second-order failure worth naming: the case study's revisit trigger, where a fleet scales up while latency is fine, means the threshold was set for throughput rather than for the latency SLO.

**Signal:** Names the trap explicitly and checks per-replica metric export before touching any threshold.
**Follow-ups:**
- *Why is compute a lagging signal here?* — It stays low while the binding resource is exhausted; the queue is on memory, not on compute.
- *What else must be true for the fix to work?* — Fast cold boot, or the scale-out arrives late — Q14.
**Red flags:** Raises the CPU threshold; concludes more capacity is needed without checking the signal.

#### T08-Q24 · Cost rose after a routing change and nothing else changed.
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** A balancer migration went out on Tuesday. Wednesday's GPU bill is up materially. What happened?
**Model answer:** The balancer stopped being prefix-sticky, so requests miss their prefix cache and pay to recompute prompts that were previously free. The case study lists this as failure mode and incident number four with the same diagnosis, and the tell is that it is invisible on a latency-only dashboard for a while: **cache hit rate falls**, and the cost appears as extra prefill work, so it manifests as throughput loss and TTFT degradation before it manifests as an SLO breach. The magnitude is worth connecting to the cost ladder — prompt caching is a distinct rung, and on the LLMOps talk's numbers the ladder runs 100 → 42 → 26 → 11 for naive serving, continuous batching, 4-bit quantization and a cached stable system prompt respectively, so losing a cache is of the same order as losing an entire optimization step `[T]` (LLMOps cost talk). The diagnostic sequence: compare cache hit rate before and after; confirm the Context Tracker's sticky-session path is still in the request path `[R]`; confirm the L7 balancer is still rebalancing at end-of-sequence rather than per connection (Q18). The remediation is to restore stickiness — or, and this is the honest second option, to accept a re-baselined cost with a documented reason if the balancing was introduced deliberately to fix skew. What you must not do is leave it undiagnosed, because a routing change that silently changes cost is also a routing change that silently changes TTFT for the tenants behind the hot prefix.

**Signal:** Goes to cache hit rate immediately and knows the cost signature of a lost cache is increased prefill work, not increased decode work.
**Follow-ups:**
- *Why does cost appear before latency?* — Recomputation is prefill work; it degrades throughput and TTFT first.
- *What is a legitimate reason to accept it?* — Hot-prefix imbalance that costs more than the misses save — §5.5's revisit trigger.
**Red flags:** Attributes cost to traffic growth; does not know what a Context Tracker does.

#### T08-Q25 · Adding requests to the batch made throughput go down. Explain.
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** You scaled a batch up and tokens/second fell. Nothing else changed. What is going on?
**Model answer:** You are almost certainly serving an expert-routed MoE model, and monotonicity has stopped holding. The corpus names the mechanism: "adding requests to the batch can **decrease** throughput if it forces a **colder set of experts** to be active. Optimal batch size depends on the **distribution of routing patterns** in the batch, not just batch count" `[R]`. Three related facts make it concrete. **Expert weight residency**: "a **400B-parameter MoE with 17B active per token** wastes most of its VRAM keeping unused experts hot. The engine has to be aware of expert-to-token routing and either pin hot experts or stream cold ones." **Per-token routing latency**: "the router decision happens **per token** and adds a measurable cost" — so it is on the critical path, not amortised once per request. **Pipeline-aware scheduling**: the best engines "schedule new requests into batches that **share expert activations** with the in-flight batch." The consequence is a rule change for this whole topic: everywhere else bigger batches are better; here **batch composition matters as much as batch size.** A request routing to experts the current batch is not using forces weights to be fetched or a different expert set activated, and the added request is a net negative. The corpus's summary line is the one to remember: "**MoE serving is no longer 'vLLM with bigger weights.'** It is a different scheduling problem." So the response is not to cap the batch blindly — it is to schedule by routing-pattern overlap and re-tune from scratch, because all the existing batch-size tuning was done under an assumption that no longer holds.

**Signal:** Names the non-monotonicity as a *composition* effect rather than a size effect, and does not propose adding capacity.
**Follow-ups:**
- *What is the fix?* — Routing-aware scheduling that groups requests sharing expert activations.
- *Does this apply to dense models?* — No; monotonicity holds and larger batches are better.
**Red flags:** Concludes the fleet is under-provisioned; raises the batch cap; has not heard of expert residency.

---

### Scale and design

#### T08-Q26 · Design the scheduling layer for this fleet. Where do you start?
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Fourteen tenants, one fleet, per-tenant SLAs, and no budget for more GPUs. Design the scheduling layer end to end.
**Model answer:** Start from the constraint: the SLA is per-tenant and the hardware is not. Four decisions in dependency order — and the case study's central claim is that they are **one decision, not four**.

**1. Batching (the floor).** Continuous batching with in-flight fusion and 4k chunked prefill. Continuous batching is not a tuning option; it is the baseline every engine in the landscape provides, and its 4x–10x `[R]` is the precondition for everything else. In-flight fusion is the only mechanism that improves both TTFT and TPOT `[R]`. Chunked prefill is what makes fusing safe on a shared fleet — trading ~2.6x TTFT on a 128k request for a ~12.5x reduction in tail disruption `[D]`.

**2. Fairness (the mechanism).** Tiered iteration-level shares enforced at the **KV slot**, with gateway token buckets as a complement rather than the primary control `[R]`. The KV slot is the enforcement point because a tenant that cannot acquire blocks cannot grow its batch; on 14 tenants a 1/14 cap bounds any single tenant at ~7% of fleet slots `[D]`.

**3. Autoscaling (the response).** KV cache utilisation as the signal, exported per replica, with goodput secondary once per-class SLO instrumentation exists `[R]` — and cold-boot time treated as part of the same design, because a 15-second boot against a 30-second degradation window works and a 3-minute boot does not.

**4. Routing (the constraint on all of it).** Prefix-sticky with L7 rebalancing between turns, plus a hot-prefix escape hatch `[R]`.

Then name what you would **not** do: no dedicated replica per tenant in Phase 1 (it destroys the pooling that justified the purchase), no autoscaling on CPU or host memory, no prefill/decode split without chunking, no single engine across every workload — and the manifest rule: a model is not "Llama 4 Maverick," it is "**Llama 4 Maverick on vLLM v0.18.3 with this batch config on this hardware**." Pin all four `[R]`.

**Signal:** Presents the four as one coupled decision with a dependency order, and states the non-goals as firmly as the choices.
**Follow-ups:**
- *Which would you change first under pressure?* — Chunk size: it is the TTFT-versus-TPOT dial and the cheapest to move (§10).
- *What breaks the whole design?* — A tenant with a contractual isolation requirement; that is the one case where a dedicated replica is correct.
**Red flags:** Answers with a component list and no ordering; proposes per-tenant replicas; omits the routing constraint entirely.

#### T08-Q27 · When does disaggregating prefill and decode pay for itself?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Disaggregated prefill/decode is often described as a config flag. When is it the right call?
**Model answer:** When the interference cost of sharing a device exceeds the operational cost of separating the phases. The mechanism is that it removes the stall **by construction**: if prefill runs on different hardware, a long prefill cannot block another tenant's decode, so the whole chunking trade (Q10) stops being a trade. The corpus treats it as "primarily for **very long context workloads**" `[R]`, and it is explicit about the cost: "A second hop for the KV transfer; operational complexity." That second hop is not a detail — the KV cache must move from the prefill node to the decode node, which is why disaggregated deployments are where peer-to-peer KV transfer and cache-tiering work appears. The practitioner account is that the replica ratio is workload-determined rather than a constant: "when we tried 2P 2D versus 3P 1D kind of ratios we saw that … some perform better than the other. It depends on the workload" — and for agentic traffic, which runs prefill-heavy — roughly **98%** of tokens are prefill `[T]` (llm-d talk, Pravin) — "having more prefill than the decode definitely helped" `[T]` (Banghua Zhu; the ratio is that speaker's own measurement across H100/H200 deployments, i.e. self-reported). The same work pairs disaggregation with a KV cache hierarchy that moves blocks from HBM down to DRAM and external storage `[T]`, which is what makes a paused agent session's cache cheaper to retain than to recompute. The decision rule on the case study's fleet: **do not disaggregate at 14 tenants**; it becomes the default at 10x, where the interference cost exceeds the operational cost (§11). Two conditions to check before adopting: context length must actually be long, or the benefit is zero and the cost is all that remains; and the engine's disaggregated path must be patched, which for SGLang is one of the paths reported unpatched (Q29).

**Signal:** States the condition (interference cost versus operational cost) rather than the technology, and knows the second hop is the price.
**Follow-ups:**
- *What is the ratio question?* — Prefill:decode replica ratios are workload-determined; measure which phase is the bottleneck first `[T]`.
- *What makes it safe operationally?* — A KV cache hierarchy and peer-to-peer transfer, so a stale cache is cheap to fetch rather than recompute `[T]`.
**Red flags:** Recommends disaggregation universally; treats it as a config flag with no second hop.

#### T08-Q28 · A security advisory names an engine version in your manifest. What do you do?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** An advisory lands affecting an inference engine. Walk me through the response.
**Model answer:** Determine which **code path** you use first — that is the whole question, and it is why the manifest pins what it pins. The corpus's reporting is specific: a high-severity **multimodal RCE** in vLLM affecting versions before **v0.18.2**, and **unpatched RCEs in SGLang's multimodal and disaggregated-prefill code paths**, with the text-only path reported safe. **These are specific and consequential claims that must be verified against the projects' own advisory feeds before acting** — which is what the corpus itself instructs: "**Watch the security advisory feeds**, not just the release notes." So the sequence: check your path against the advisory's scope; if your traffic is text-only on SGLang, you are in the reported-safe path and the decision is a monitored exception rather than an emergency; if it is multimodal or disaggregated-prefill, you upgrade or isolate the path by moving that traffic to the alternate engine. The reason you can do the latter at all is that the operational posture already runs a canary on a **second engine** at 1–5% of traffic `[R]` — during a security event that canary is your evacuation route, not a testing nicety. The manifest rule is what makes the diagnosis possible in the first place: "A model is not 'Llama 4 Maverick'; it is '**Llama 4 Maverick on vLLM v0.18.3 with this batch config on this hardware**.' Pin all four" `[R]`. Without the engine version in the manifest you cannot answer the only question the advisory poses. And the posture resolves the apparent conflict with version pinning: patched versions are mandatory, so pinning means pinning a *patched* version, not pinning forever.

**Signal:** Asks which code path before which version, and uses the second-engine canary as an evacuation route rather than a testing nicety.
**Follow-ups:**
- *What is the general rule?* — Patched-version-or-nothing, with the manifest pinning all four attributes.
- *How do you stop this recurring?* — Advisory-feed monitoring as a monitored surface (§10), not release-note reading.
**Red flags:** Quotes the CVE as established fact without the verification caveat; upgrades everything blindly; cannot say which path the fleet uses.

#### T08-Q29 · The engine landscape: what would you deploy in May 2026, and what would you verify first?
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Pick the engines for a mixed estate, and tell me which of your own numbers you do not trust.
**Model answer:** The mapping is straightforward; the verification posture is the answer. The corpus's landscape: **vLLM v0.18.2+** as the house default — easiest to operate, best security cadence, multimodal path patched; **SGLang v0.4.3+** on the **text-only** path for JSON function-calling and structured output, on a reported **~29%** throughput advantage from async constrained decoding; **TensorRT-LLM** for a single latency-critical model where peak NVIDIA throughput justifies the operational cost; disaggregated prefill for long-CoT reasoning at low concurrency. For MoE, either vLLM v0.18+ or SGLang v0.4.3+ with a MoE scheduler. What I would verify before acting, in order:

1. **The security claims.** The vLLM multimodal RCE below v0.18.2 and the unpatched SGLang RCEs in the multimodal and disaggregated-prefill paths are specific, consequential, and stated as of May 2026 — check the projects' advisory feeds, which is what the corpus instructs.
2. **The ~29% figure.** It is **vendor-published** (April 2026), on the text-only path, for structured output specifically. It is not a general SGLang-versus-vLLM claim and must not be repeated as one.
3. **The version numbers.** They are the supporting repo's assertions as of May 2026, and version numbers move faster than documents.

A practitioner data point worth adding with the same caveat: the SGLang-side GLM 5.2 work reports "over 2.2x improvement" and "up to 500 tokens per second per user" at 1M context on H100/H200 `[T]` (Banghua Zhu) — that is the serving team's own measurement of their own stack and is therefore **self-reported, not independently confirmed**. Then the posture: one house engine plus one specialised engine on a canary at 1–5% of traffic, patched versions only, and the manifest pinning model, engine version, batch config and hardware together. And the escalation to warn against: four engines across workload classes before the build-test-upgrade pipeline exists to maintain them (§11).

**Signal:** Separates the mapping (stable) from the figures (verification-required) and never presents a vendor number as a benchmark.
**Follow-ups:**
- *When does the mapping change?* — A workload's category shifts, or an engine's security posture changes.
- *What is the cheapest insurance?* — The second-engine canary; it is both a migration path and an evacuation route.
**Red flags:** Presents the ~29% as an established result; recommends an engine with no patching story; misses that the SGLang caveat is path-specific.

#### T08-Q30 · Traffic grows 10x. What breaks, what holds, and what inverts?
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Your fleet goes from 14 tenants to 140. Walk me through the design, not the hardware.
**Model answer:** Four changes, one inversion, and a list of survivors.

**What breaks first: fairness as configuration.** At 14 tenants with slack, static equal shares work. At 140 with none, shares must be demand-adaptive and the scheduler becomes a genuine multi-tenant resource allocator with all the machinery that implies — accounting, admission, preemption policy, and a way to re-allocate without a config deploy. The case study's sensitivity entry points at the mechanism: at 40 tenants, per-tenant share falls to 2.5% and "the batch class may no longer finish within its window," which is exactly when equal division stops being defensible `[D]`.

**What becomes the default: disaggregation.** The stall exists because prefill and decode share a device. At 10x the interference cost exceeds the operational cost of separating them, and the corpus already flags disaggregated prefill as the long-context path `[R]`.

**What becomes a product rather than a posture:** the engine-per-workload split. Running two engines with a canary `[R]` is a posture; running four across workload classes is a product with its own build, test and upgrade pipeline.

**What becomes capacity planning rather than autoscaling:** at 10x, responding to KV pressure is not enough, because cold starts and share reallocation are not instantaneous — the fleet needs predicted demand.

**What inverts:** "maximise batch size" stops being universally true. It fails for MoE models, where composition matters more than size and throughput is non-monotonic in batch count `[R]`, and it fails for latency-sensitive tenants, for whom batch size is a liability rather than a benefit.

**What survives, because it is structural:** iteration-level scheduling, chunked prefill, KV-utilisation autoscaling, prefix stickiness, and the prefill:decode fusion. Note *why* these survive: they are properties of how the loop is composed, not of how much hardware is attached — which is also why the tuning parameters do not survive alongside them.

**Signal:** Separates structural survivors from tuning parameters, and names a genuine inversion rather than "everything scales."
**Follow-ups:**
- *What is the first sign the shares have gone stale?* — A tenant consistently exceeding its share off-peak; that is the §5.2 revisit trigger.
- *Does the cold-boot work still matter at 10x?* — More, not less: it bounds response time when share reallocation cannot act instantly.
**Red flags:** "Add more GPUs"; assumes the fairness design scales unchanged; cannot name anything that inverts.

---

## Whiteboard exercises

### Exercise 1 — Compose one iteration
**Prompt.** One replica is running 15 in-flight decode streams. At iteration N, three things arrive together: (a) a 128k-token prompt, (b) 400 classification requests with no expected output tokens, and (c) a 5,000-request burst from the batch-class tenant, 200 tokens each. You control chunk size, the per-iteration prefill share, and per-tenant KV slot caps. Produce the per-iteration plan and defend every number in it.

**What to produce.** The iteration composition — how many prefill chunks, how many decode rows, and whose work each is — the chunk size with its arithmetic, the prefill share cap, the tenant cap, and the metrics you would watch to know you were wrong.

**Expected whiteboard.**

```
Iteration N (per-iteration plan — one plan per iteration, not one policy)
  decode rows    : 15 in-flight streams, kept in the batch  (bandwidth-bound, cheap to keep)
  prefill chunks : 1 chunk of CHUNK_SIZE, from ONE request only
  classification : its OWN prefill budget, capped — never the interactive share
  batch class    : admitted only up to its KV slot cap

CHUNK_SIZE = 4k tokens                                   [R]
  128k / 4k  = 32 chunks
  x 200 ms   = 6.4 s prefill for that one request        [D]
  worst-case TPOT delay for the other 14 streams = ~200 ms
                (vs 2-3 s unchunked)                     [R]

PREFILL SHARE CAP: strictly < 100% of the iteration's token budget
  reason: chunking bounds the DURATION of a stall, not the SHARE of the iteration;
          a pathological arrival mix can still consume every iteration's prefill slots

TENANT CAP: 1/14 of fleet KV slots -> max ~7% per tenant  [D]

Watch: TPOT p95 by class | prefill:decode mix per iteration | prefill chunk occupancy
       per-tenant KV slot occupancy vs cap | admission-rejection rate | cache hit rate
```

**Grading rubric.**
- Composes *one iteration explicitly* — names the decode rows, the chunk count, and the class boundaries — rather than describing a policy.
- Justifies the 4k chunk with the 32-chunk / 200 ms arithmetic and states the exchange rate (the long request's TTFT up ~2.6x for ~12.5x less tail disruption).
- Caps the prefill share *separately* from the chunk size, and explains that chunking bounds duration while the cap bounds the share of the iteration.
- Gives the batch-class tenant a KV slot cap rather than unlimited admission on the grounds that its work is cheap, and gives the prefill-only class its own prefill budget.

### Exercise 2 — Diagnose a fleet-wide TPOT regression
**Prompt.** Over roughly 30 minutes, p95 TPOT on one replica rises from 42 ms to 910 ms for *every* tenant on it, then recovers. Compute utilisation never exceeded 34%; KV utilisation sat at 71%. No deploy is recorded. You have TPOT by class, prompt-length arrival logs, per-tenant KV occupancy, the prefill:decode mix per iteration, and the engine config. Produce the diagnosis, the discriminating measurement, and the fix.

**What to produce.** A ranked hypothesis list, the measurement that confirms or kills each, the read on the utilisation numbers, and the fix with its rollback.

**Expected whiteboard.**

```
Blast radius: FLEET-WIDE and SIMULTANEOUS -> cause is on the replica, not in a tenant
  => eliminates: fairness failure (tenant-scoped by definition)
  => eliminates: autoscaler-signal failure (no scale-out event is in question here)

Ranked hypotheses                            confirming measurement
 1. unchunked / under-chunked long prefill   TPOT spike aligned to a prompt-length arrival
 2. prefill share consumed every iteration   prefill:decode mix per iteration ~ 100% prefill
 3. continuous batching regressed in config  config diff; throughput vs the 4x-10x baseline
 4. cold MoE expert set                      throughput vs batch-size curve is NON-monotonic

Discriminator on the utilisation pair (compute 34%, KV 71%):
     prefill-blocked decode  -> KV rises while decode waits; admission still healthy
     capacity wall           -> KV pinned, admission rejections climbing
  observed = consistent with (1) or (2); NOT with (4) [non-monotonic] and NOT with a wall

Fix:      reduce chunk size  ->  then cap the prefill share per iteration
          cost: the long request's OWN TTFT rises. That is the deliberate trade, not a bug.
Rollback: revert the engine config if the config-diff check implicates (3)
```

**Grading rubric.**
- Uses the fleet-wide simultaneity to eliminate tenant-scoped causes *before* hypothesising.
- Identifies the prefill budget as the contested resource, and separates "chunking bounds duration" from "the share cap bounds occupancy."
- Reads the compute/KV pair as evidence rather than noise — 34% compute against 71% KV is consistent with a prefill-blocked decode, not with a capacity wall.
- Fixes on the cheapest dial first (chunk size), names the cost it imposes on the large request, and includes a config-regression rollback.

### Exercise 3 — Make fairness bind, and prove it
**Prompt.** Fourteen tenants share one fleet; four are interactive with a contractual 400 ms TTFT. One tenant submits 5,000-request bursts of 200-token outputs and has just caused a breach. Design the fairness policy, show the arithmetic that proves the burst tenant cannot starve search, and state the direction of the starvation you are preventing. Then say what would make you change the shares.

**What to produce.** The two enforcement layers with the binding one marked, the cap arithmetic, the direction statement, and a quantitative revisit trigger.

**Expected whiteboard.**

```
DIRECTION — get this right, it is the error this topic punishes:
    one BIG session/burst takes all the dispatch cycles
      -> the SHORT / interactive sessions starve
    NOT the reverse. The scheduler's job is to stop the monopolist,
    not to protect it from the small tenants.

Two layers, only one of which binds:
  gateway : token bucket — coarse, tenant-visible, BLIND to GPU state
  engine  : per-tenant KV SLOT CAP — binds, because a tenant that cannot
            acquire blocks cannot grow its batch                  [R]

Arithmetic [D], assumptions shown:
  14 tenants, equal share   -> 1/14 ~ 7% of fleet KV slots, max, per tenant
  burst work: 5,000 x 200 tokens = 1,000,000 output tokens
  at ~2,000 tok/s per GPU x 32 GPUs -> ~16 GPU-seconds of work, spread over the fleet
  => the burst is NOT expensive; it is only disruptive when it is allowed
     to take the whole batch
  cost to the burst tenant    : it finishes later, within its own share
  benefit to the other 13     : worst case is losing their OWN share, never 100%

CHANGE THE SHARES WHEN: a tenant consistently exceeds its share at off-peak
  => that is the signal the shares are static where they should be demand-adaptive
     (the case study's 14 -> 40 sensitivity is where equal division breaks outright)
```

**Grading rubric.**
- States the starvation direction correctly — one big session monopolises the scheduler and the short/interactive sessions starve — and does not invert it.
- Puts enforcement at the **engine's KV slot cap**, with the gateway bucket named as a complement that cannot see GPU state.
- Shows the cap arithmetic (1/14 ≈ 7%) and the burst's real cost, and concludes the batch class "is not actually expensive; it is only disruptive when it is allowed to take the whole batch."
- Gives a quantitative revisit trigger for the shares (demand-adaptive rather than static) rather than "review periodically."

## Sources

- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/04-batching-strategies.md` — the static-versus-dynamic framing, the continuous-batching comparison table with the join/leave, utilisation, 1x-versus-**4x–10x** and latency rows, in-flight batching with the 1-prefill/15-decode example, chunked prefill with the **4k** chunk and **200 ms** figures, the stall definition with the **2–3 s** prefill and the TPOT spike, and the longest-tail worked example (**500** tokens versus **5** tokens, **495** idle cycles).
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md` — tiered iteration-level scheduling with the per-tenant share of KV cache slots, the gateway-token-bucket plus engine-policy split, **KV cache utilization** as the autoscaling signal, cold boot from un-quantized base images at **15–20 s**, tensor versus pipeline parallelism with the **90%-of-production** guidance and the NVLink requirement and the **~800 GB** 405B figure, SSE/WebSocket serving and the Layer 4 versus Layer 7 problem with the end-of-sequence token, the inference gateway's Context Tracker and sticky sessions, the May 2026 engine landscape with version numbers and the **~29%** SGLang figure and the vLLM/SGLang security advisories, MoE-aware serving with expert residency and the **non-monotonic batching profile**, the engine-per-workload decision table, and the patched-version / second-engine-canary / pin-all-four posture.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — the prefill/decode bottleneck split, the TTFT/TPOT/throughput/latency metric table with the **< 200 ms / < 30 ms** targets, and the "prefill-only classification is **compute-optimal**" framing.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/05-paged-attention.md` — paged allocation and the block table as the precondition for reclaiming the slots continuous batching frees, and the source of the uniform-block property that makes per-tenant slot caps countable.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt` — the fixed per-step dispatch cost ("each step here essentially has a fixed cost … we have to dispatch to our GPU and then we need to wait for the GPU to finish processing") and the GPU under-saturation that motivate batching. **Lecture 1 lives in the `_2` directory; verified by directory listing.**
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — the **session-level starvation direction** (a large session takes all the dispatch cycles and the shorter sessions starve), agentic-program-aware fairness via least-attained service with latencies down up to **50% / 2–3x** and the small throughput gain, FCFS "doesn't add anything extra," the **KV-80%-full** and **average-active-requests > 8** saturation thresholds, the **~98%** prefill share of agentic tokens, turn priority and its relationship to least-attained service, the naive-load-balancer recomputation cost, and precise-versus-approximate KV cache routing.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the **100 → 42 → 26 → 11** cost ladder with continuous batching as the 100→42 rung and a cached system prompt as the 26→11 rung, and the prefill-compute / decode-bandwidth framing that explains why batching helps decode.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — prefill/decode disaggregation with workload-dependent replica ratios (2P2D versus 3P1D), the KV cache hierarchy from HBM to DRAM to external storage, and the **over-2.2x / up-to-500-tokens-per-second-per-user** figures, cited only as the serving team's own self-reported measurement.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` — topic inventory confirming coverage of continuous batching, in-flight batching, chunked prefill, prefill-decode disaggregation and token streaming. **This file is a table of contents and contains no figures.**
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
