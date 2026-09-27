# Interview Bank: KV Cache, Paged Attention & Tiering

> `T07` · **Transcript coverage:** partial · [Cheat sheet](../00-cheat-sheets/T07-kv-cache.md) · [Case study](../01-case-studies/T07-kv-cache.md) · [Design blueprint](../03-design-blueprints/T07-kv-cache/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a transcript statement with the speaker or talk named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here with its assumptions shown. Provider pricing figures and the RAD-O capability are **vendor claims** and are flagged as such where they appear; where the corpus supplies no figure, the answer says so rather than inventing one.

Questions are ordered to read as one interview: foundations, then mechanism, then tiering and eviction, then the economics of caching, then the sharing gate, then debugging, then design at scale. The case study's five decision tables — cost reduction, allocation, sharing scope, storage tier, eviction policy, and what gets cached in an agent loop — are probed directly at Q3, Q7, Q13, Q19, Q20 and Q11.

---

### Foundations

#### T07-Q1 · Why the cache exists, and why it is always sound
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Forget the frameworks for a second. Why does a KV cache exist at all, and why is reusing an earlier token's keys and values guaranteed to be correct rather than an approximation?
**Model answer:** Both halves come from causal masking. Because "the inputs at position three cannot depend on the word in position 4," a later token cannot change an earlier token's hidden representation — "the reason why we don't have to update the embedding of the previous words is because we're using masked attention" `[T]` (CMU lecture 1). The chain rule requires the same thing: "otherwise the chain rule of probability doesn't hold here" `[T]`. So the key and value vectors for a position are a **pure function of that position's prefix**, and caching them is exact — not an approximation of a recomputation, but the same numbers. That invariance is the load-bearing fact for this whole topic: prefix sharing, copy-on-write, offload to a slower tier and eviction are all consequences of it, and none of them introduces error. It also fixes what a cache *hit* even means: two requests hit the same cached block only if their prefixes are identical, which is why the matching rule is exact and positional. The one thing the cache does change is the cost profile — prefill computes the whole prompt's K and V in one parallel pass, and decode reuses them, so each decode step reads the cache rather than recomputing it.
**Signal:** Derives caching from the mask rather than from "so we don't recompute," and states that a hit is exact rather than approximate.
**Follow-ups:**
- *What could break the invariance?* — Nothing in the architecture; it is what a causal mask is for. Speculative and parallel decoding still respect it.
- *What does it imply about cache matching?* — Exact and positional; the mechanism behind every prefix failure in Q23.
**Red flags:** Calls the cache "an optimisation that might change outputs"; confuses the cache with the weights; cannot say why earlier representations are stable.

#### T07-Q2 · The arithmetic: 0.3125 MB per token
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Give me the formula for KV cache size and take me from a model shape to a number I can plan against.
**Model answer:** `bytes/token = 2 (K and V) × layers × KV heads × head_dim × dtype bytes`. For an 80-layer, 8-KV-head, 128-head-dim model in BF16 `[R]`: `2 × 8 × 128 × 2 = 4,096 bytes` per layer per token; `4,096 × 80 = 327,680 bytes ≈ 0.3125 MB` per token `[D]`. Multiply by context: `0.3125 MB × 128,000 = 40 GB`, which is the corpus's quoted **~42 GB per user at 128k** `[R]`. The operational form of that unit is what matters: **every 1,000 tokens of context costs about 305 MB**. A 32k-token repository context is 10 GB. A 200k-token one is 62.5 GB — more than the model's FP8 weights, which at one byte per parameter for a 70B model is 70 GB `[D]`. The framing that should stick is the corpus's own: "The KV Cache is the most significant memory consumer in long-context AI systems. Managing this cache effectively is the difference between a system that scales to 2M tokens and one that crashes at 10k" `[R]`. And the division that follows is the one that sizes your deployment: `max_concurrency ≈ HBM_for_KV / (bytes_per_token × avg_context_len)` `[D]`.
**Signal:** States the formula, derives per-token, and immediately converts to both a per-session figure and the concurrency division.
**Follow-ups:**
- *Where does the 8 come from?* — GQA KV heads, a per-model constant; Llama 3.1 has 8 across the series `[T]`.
- *What halves it?* — FP8 KV — but validate quality at the p99 context first (Q19).
**Red flags:** Quotes 42 GB with no formula; treats context length as a feature rather than a budget line.

#### T07-Q3 · GQA, MQA, and the shape of the reduction curve
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Walk me through the attention architectures that reduce cache size, and tell me which one you would pick and why.
**Model answer:** Three rows `[R]`. MHA is 1:1 — 1x reduction, no quality loss, baseline. **GQA** is 8:1 — **8x reduction at under 0.2% quality loss**. **MQA** shares one KV across all heads — **64x to 128x reduction at 2-3% quality loss**. The corpus's mechanism note is the useful one: GQA "allows the model to attend to the same KV 'memory' from multiple 'reasoning' heads, drastically reducing the memory bandwidth needed during the Decode phase" `[R]`. Two things follow that candidates usually miss. First, the 8x is *already inside* the 42 GB figure — without GQA the same 128k session would need roughly `41.25 × 8 = 330 GB` per user `[D]`, more than four times the largest single accelerator before weights are loaded. GQA is therefore not an optimisation you chose at deployment; it is the precondition for serveable long context, and if the selected model is MHA then long context is probably off the table entirely. Second, the reduction-versus-quality curve is **strongly non-linear**: 8x costs almost nothing, while 8-16x more reduction costs an order of magnitude more quality. That non-linearity is why GQA is a model-selection gate and MQA is not a deployment knob. The KV head count is a per-model constant — "in grouped query attention they always have eight uh over all of them in the llama 3.1 series" `[T]` (CMU lecture 1) — so the formula in Q2 depends on a choice made at training time.
**Signal:** Knows the 8x is baked into the 42 GB, so an MHA model would be ~330 GB per user, and treats GQA as a precondition rather than a tuning choice.
**Follow-ups:**
- *Can you add GQA after training?* — No; it is an architectural choice. If memory binds on an MHA model, quantization (Q19) is the remaining lever.
- *When would MQA be right?* — Only when memory is the hard binding constraint and the task tolerates 2-3% — not a code-review agent.
**Red flags:** Calls MQA a free win; cannot say where the 8:1 ratio comes from; treats the three rows as interchangeable.

#### T07-Q4 · Contiguous allocation: name both failures
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** What was wrong with how KV memory was allocated before paging, and be specific about the failure modes.
**Model answer:** Two distinct failures, and conflating them is the usual mistake `[R]`. **Internal fragmentation**: a request pre-allocates for `max_sequence_length` — say 8,192 tokens — so if the user generates 10 tokens, "99.9% of that reserved block is wasted." The waste is created by the *reservation*, not by the usage. **External fragmentation**: "Memory is broken into gaps too small for a new 'large block,' even if total free memory is high." Because a contiguous buffer must be one unfragmented region, a long-running server accumulates holes, and the next request fails even though the aggregate free bytes are ample. Together these produce the corpus's **~60-80% waste** figure, or equivalently a memory efficiency of about 60% `[R]`. The nuance that matters for an agent platform is that the two failure modes scale differently with context length. A chatbot reserving 8k and using 200 tokens wastes 97% of its reservation; an agent session reserving 200k and using 150k wastes only 25% — *but* the reservation is what caps concurrency at roughly one such session per device, and the contiguous requirement means the allocator needs a free region of that exact size. Neither failure is fixed by adding memory. They are fixed by not reserving, which is Q6.
**Signal:** Separates internal from external fragmentation and ties the waste to the reservation rather than to usage — the distinction that predicts why long context breaks first.
**Follow-ups:**
- *Why does a request reserve max length at all?* — Because a contiguous buffer must be sized before the output length is known.
- *What replaces it?* — Paged blocks plus a block table; Q6.
**Red flags:** Conflates the two failure modes; says "it wastes memory" with no mechanism; believes external fragmentation is a memory-capacity problem.

#### T07-Q5 · Prefill, decode, and why the cache is read-bound
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** How does the cache change the cost structure of the two serving phases, and why should that matter to me?
**Model answer:** Prefill processes the whole prompt in one parallel pass and writes a K and V for every position; it is compute-bound and it sets TTFT. Decode runs one token at a time, and its arithmetic intensity is roughly one multiply-add per weight read, so it is **memory-bandwidth bound** — the cache is on the critical path of every single step. This is the mechanism behind the corpus's statement that GQA drastically reduces "the memory bandwidth needed during the Decode phase" `[R]`, and it is also why PagedAttention's block-table indirection is a real cost rather than free bookkeeping: decode has bandwidth to spare relative to compute, but the indirection adds address work to a loop that is latency-sensitive. Two consequences follow. First, a cache *hit* is a prefill shortcut, so the warm-prefix TTFT target in the case study — ≤300 ms — is a different regime from a cold prefill `[R]`. Second, because decode reads the whole cache each step, cache size and decode latency are linked: a session holding 41.25 GB of KV is not just occupying memory, it is extending every subsequent step.
**Signal:** Distinguishes compute-bound prefill from bandwidth-bound decode and uses it to explain both the GQA benefit and why the block table costs something.
**Follow-ups:**
- *Why does the cache make disaggregation necessary?* — Because KV is the state that must move between the two phases ([T12](../01-case-studies/T12-disaggregation-kv-transfer.md)).
- *What does a cache miss cost in real terms?* — Q15 does the TFLOP arithmetic.
**Red flags:** Treats prefill and decode as the same workload; thinks decode is compute-bound; calls the block table free.

---

### Mechanism

#### T07-Q6 · PagedAttention in three steps
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Explain PagedAttention. What is actually happening, and what does fixing fragmentation unlock beyond the memory number?
**Model answer:** Three steps `[R]`. **Tokens to blocks**: break the cache into fixed-size blocks, "e.g., 16 tokens per block." **Logical versus physical**: "The model thinks it's attending to a contiguous sequence (Logical memory), but the blocks are scattered throughout VRAM (Physical memory)." **The lookup table**: "A **Block Table** maps logic indices to physical addresses." The measured result is that "Memory waste drops from **~60-80%** down to **less than 4%**," with memory efficiency rising "from 60% to 96%+" `[R]`. Two consequences beyond the waste figure, and these are the ones a strong answer volunteers. First, **blocks are allocated lazily as generation proceeds**, so allocation tracks usage rather than reservation — which is the property that turns a reservation-shaped cost into a growth-shaped one. Second, and more important, per-block addressing makes two capabilities expressible that are impossible with one buffer per session: **sharing** (two sequences can point at the same physical block — Q8) and **tiering** (an individual block can be moved or evicted independently — Q12). The case study's framing is that without the block table "there is one contiguous buffer per session and none of these distinctions can be expressed." The honest cost: attention kernels must be written to gather from a block table, so this is a kernel change, not a bookkeeping change, and every attention read pays an indirection.
**Signal:** Gives all three steps and names sharing and tiering as the capabilities paging unlocks, not just the waste number.
**Follow-ups:**
- *Why is it a kernel change?* — The gather has to happen inside attention; Q5 explains why the decode loop is latency-sensitive.
- *What is the block size tradeoff?* — Q7.
**Red flags:** "PagedAttention saves memory" with no mechanism; has never heard of the block table; claims the memory saving is the main benefit.

#### T07-Q7 · Block size: the third effect nobody names
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** You have to pick a block size for a production deployment. What are the forces, and how would you actually choose?
**Model answer:** Three forces pull in different directions, and the third is the one people forget `[D]`. **Larger blocks** mean a smaller block table (fewer entries per sequence), and longer contiguous reads inside the attention kernel — better coalescing. **Smaller blocks** mean less internal fragmentation in the final partially-filled block and a smaller gather granularity. The approximate expected internal waste is on the order of `(block_size − 1) / (2 × seq_len)` — at 2,048 tokens and a 16-token block that is about 0.4% `[D]`, which is why the corpus's illustrative 16 is a sane default rather than a magic number `[R]`. The third force, and the reason this is an L4 question: **block size is the sharing granularity**. Two sequences can share a physical block only if they share the *entire* block's token prefix, so a larger block delays the divergence point and makes copy-on-write coarser. In a workload whose entire saving comes from prefix sharing (Q8), that effect can dominate the fragmentation maths. The engineering rule is therefore: pick a starting block size, then **measure waste directly against the <4% target** `[R]` rather than deriving it, and retune when it drifts. The case study's revisit condition is explicit — measured waste above 4%, or a visible block-table overhead, both indicate the block size is wrong, and "measure, do not guess."
**Signal:** Names sharing granularity as the third effect and refuses to derive waste analytically when it can be measured against a target.
**Follow-ups:**
- *What does halving the block size do?* — Table overhead up, waste down, sharing finer; the case study lists this as a measured, not computed, tradeoff.
- *Would you use the same block size for a chat product and an agent?* — No — the context distribution differs, which is why the case study tunes it to the observed distribution.
**Red flags:** "16 is standard, use 16"; only considers fragmentation; changes block size without a waste metric.

#### T07-Q8 · Copy-on-write prefix sharing
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A hundred users of your product start from the same 5,000-token system prompt. What does the cache look like, and what discipline keeps it correct?
**Model answer:** The corpus's worked scenario `[R]`: 100 users share that 5,000-token prefix. **Traditional**: store the prefix's KV 100 times — **500,000 tokens** resident in VRAM. **PagedAttention**: store it **once** and have all 100 block tables point at the same physical blocks. **Copy-on-write**: when a user generates a unique token, "a new block is created just for them, while the shared blocks remain unchanged." Run the memory `[D]` at 0.3125 MB per token: unshared is `500,000 × 0.3125 MB = 156 GB`, which does not fit on a node; shared is `5,000 × 0.3125 MB = 1.56 GB` plus per-session tails. That is a **100x reduction**, and it is the difference between "does not fit" and "fits trivially." At the case study's 40 concurrent sessions the same comparison is 62.5 GB unshared against 1.56 GB — a 40x saving that grows linearly with concurrency. The discipline that keeps it correct: shared blocks are **immutable and reference-counted**, and divergence costs **one block, not a copy of the prefix**. The failure mode is a write to a shared block, and that is a correctness bug, not a performance one — another session reads the mutated block and produces a wrong answer. Note the strategic consequence the case study draws: because this saving is nearly free technically, the only reason not to take it is that sharing scope is a data-isolation decision (Q20), which is why it is escalated upward rather than settled in engineering.
**Signal:** Gives both the mechanism and the immutability/refcount discipline, and sizes the saving in GB rather than only in tokens.
**Follow-ups:**
- *What enforces immutability?* — Refcount > 1 means allocate-on-write; a write to a refcounted block is the bug to test for.
- *Does this apply outside the system prompt?* — Yes — repository context across sessions on the same repo is the largest practical case, and it is gated on legal (Q20).
**Red flags:** Describes sharing without a write path; cannot size the saving; assumes shared blocks may be written if the prefix "is the same."

#### T07-Q9 · The block manager and Paged Swap
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Who actually owns the blocks at runtime, and what happens when VRAM runs out?
**Model answer:** A **block manager** owns the physical block pool and does three jobs: it allocates a block when a sequence needs one it does not hold, it tracks **refcounts** so a shared block is not freed while another sequence still references it, and it reclaims blocks when a sequence ends. The eviction path is explicit in the corpus: "If VRAM is full, the manager can 'swap' inactive KV blocks to CPU RAM and bring them back when needed (**Paged Swap**)" `[R]`. The important structural point is *why* this is possible at all: paging made memory addressable **per block**, so eviction is a per-block operation rather than a whole-sequence one. That granularity is exactly what makes a tier system expressible (Q12); with contiguous per-session buffers there is nothing meaningful to move. Two operational consequences worth stating. First, eviction is driven by *inactivity* under Paged Swap, so a parked session's blocks are the natural victims — and the case study's rule is to evict parked sessions to the lower tier **on a timer, not only under pressure**, because idle sessions holding VRAM cause a concurrency collapse before any pressure signal fires. Second, the swapper is a correctness surface, not just a capacity mechanism: a block restored from the wrong tier, or restored against a different model revision, returns state that produces a plausible wrong answer. Block identity must be its prefix and version, not its position.
**Signal:** Names allocation, refcounting and reclamation as the manager's three jobs, and says why per-block addressing is what makes swap possible.
**Follow-ups:**
- *What does a swap cost on restore?* — A recall latency that lands in TTFT; Q14 prices it.
- *Why evict on a timer?* — Q25 works the resume-latency failure.
**Red flags:** Treats swap as free capacity; no mention of refcounts; assumes a restored block is trustworthy.

#### T07-Q10 · Concurrency: 4 → 20-30, and where paging stops helping
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** How much concurrency does paging actually buy, and does that number hold at long context?
**Model answer:** The corpus's headline comparison is "In traditional serving, we might only fit **4 requests** because we have to 'reserve' max-length blocks; with PagedAttention, we can fit **20-30 requests**" `[R]` — same VRAM, short contexts. That is a 5-7.5x win, and it comes entirely from removing the reservation rather than from compressing anything. The case study makes the regime boundary explicit with its own arithmetic `[D]`, applying the waste figures to a 40 GB KV budget:

| Allocation | Usable KV | Sessions at 8k context (2.56 GB) | Sessions at 132k (41.25 GB) |
|---|---|---|---|
| Contiguous, ~70% waste | 12 GB | **4** | **0** |
| Paged, <4% waste | 38.4 GB | **15** | **0** |

Two lessons, and they are different lessons. At 8k context paging is a **2.5-3.75x concurrency win**, in the same range as the quoted 4 → 20-30. At 132k tokens per session **paging does not save you at all** — a single session exceeds the budget either way, so admitted context is the only remaining lever, and that is a product decision rather than a serving-engine one. Conflating these two regimes is the most common planning error in this topic: the fix for the short-context case is allocation, and the fix for the long-context case is deciding not to admit that much context. Never quote the 20-30 figure as a general throughput claim.
**Signal:** Separates the regime where paging is the fix from the regime where it is irrelevant, using the two-column arithmetic rather than one number.
**Follow-ups:**
- *What is the lever at 132k?* — Admitted context; Q28 orders all the levers by leverage.
- *Why does the 8k row say 15 and the corpus say 20-30?* — Different KV budgets and average context lengths; the ratio is the portable part.
**Red flags:** Quotes 20-30 requests as a universal throughput multiplier; believes paging rescues long-context concurrency.

#### T07-Q11 · What actually gets cached in an agent loop
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Your agent runs hundreds of steps. Which parts of the prompt do you cache, and what is the cost of each choice?
**Model answer:** The case study's decision table has five levels `[R]`, and the ordering is by ambition. **Nothing** — you pay a full prefill every step, which for a 500-step trajectory is the dominant cost of the product; never. **System prompt + tool schema only** — the largest hit rate and the smallest risk, and it matches what many services cache, so it is the floor. **+ accumulated observations** — this caches the bulk of the trajectory's growth, and it is the case study's chosen level. **+ actions and thoughts** — maximises the cached prefix `[T]`, but adds surface for a prefix mismatch. **Continuous summarisation** — destroys the cache; see Q17. The structural fact underneath the choice is that the **trajectory is the bulk**: lecture 11 reports agent trajectories of "up to as much as like a hundred steps… maybe 50 tool calls, 100 actions" spanning "over uh hundreds of thousands of tokens," and the same lecture reports using agents "for up to 2,000 steps… like tens of millions of tokens" `[T]`. A policy that caches only the system prompt therefore leaves the expensive part uncached; the prompt is cheap and the history is expensive. Two caveats to volunteer. The cache **scope** is service-dependent, so verify rather than assume it (Q16). And the observations cache is fragile if the representation layer is regenerated each step — "You reprocess it every time" `[T]` — because a regenerated representation changes the prefix and the hit is lost.
**Signal:** Chooses a level with a reason rooted in where the tokens actually are, and flags the regeneration hazard unprompted.
**Follow-ups:**
- *When are actions and thoughts safe to cache?* — Only with strict prefix discipline; any client that re-renders history differently between turns breaks it.
- *How does this interact with the admitted-context budget?* — The trajectory grows until it hits the budget, which is when condensation at a boundary enters; Q17 and Q28.
**Red flags:** Caches only the system prompt for a 500-step agent; assumes the provider caches the whole prompt; ignores that history is the bulk.

---

### Tiering and eviction

#### T07-Q12 · The three tiers, and what sets the boundary
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Describe the tiered storage model for KV blocks, and tell me how you would decide where the tier boundary goes.
**Model answer:** The corpus names the design directly: "Frameworks like **SGLang** use a tiered system: `Most Recent (VRAM) -> Frequent (HBM) -> Occasional (SSD)`" `[R]`, with Paged Swap moving blocks down on eviction and back up on recall. The tradeoff is stated bluntly — VRAM is "instant access, strictly limited size," while disk is "slower access, nearly unlimited" `[R]` — so this is a latency-for-capacity exchange, not a free win. The design rule the case study lands on, and the one worth saying out loud, is that **the tier boundary is set from the resume SLO, not from capacity**. The SLO it is set against is the ≤300 ms session TTFT after a warm prefix `[R]`. That inverts the naive design: you do not ask "how much VRAM do I have and what spills"; you ask "what recall latency can a resumed session tolerate," and you size the recent tier to hold every session that must answer inside that budget. The revisit condition follows from the same logic — if measured resume TTFT breaches, **the fix is a larger recent tier, not a faster disk**, because the tier below it was chosen for cost and will never be fast enough. Two configuration notes: the deployment knob that sets how much of VRAM the cache may occupy is `--gpu-memory-utilization`, run at 0.90 in the corpus's own example `[T]` (NextGen talk); and the eviction timer is what actually decides which sessions stay recent, because sessions leave the top tier on inactivity as much as on pressure.
**Signal:** Sets the boundary from an SLO rather than from capacity, and refuses the faster-disk reflex for a breach.
**Follow-ups:**
- *What happens if you set the boundary from capacity?* — Resume spikes; Q25 diagnoses it.
- *What drives a session out of VRAM?* — An inactivity timer as much as memory pressure.
**Red flags:** "Put as much as fits in VRAM and spill the rest"; treats the tier boundary as a capacity decision.

#### T07-Q13 · Eviction: why LRU is the wrong default here
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You have to choose an eviction policy. Why is the obvious one wrong for this workload, and what would you do instead?
**Model answer:** LRU evicts by recency, and it is catastrophic when entry sizes are wildly unequal — which is exactly this workload. Picture the two classes: a 5,000-token shared system prefix or a 32k-token repository prefix is a **large entry with enormous reuse value**; a session's unique trajectory tail is **many small entries with low reuse value**. Under pressure LRU can evict the large shared prefix to make room for a handful of small tails, and the price of that decision is not a one-off: every session that referenced the prefix now pays a full recompute, which is a cost multiplier. The case study's phrasing is the one to remember — it "evicts a large shared prefix to free a small unique tail — a catastrophic trade" `[R]`. The chosen policy is **cost-aware: evict by `recompute cost × reuse probability`**, with the immutable prefix pinned. The property that makes it safe is that its failure mode is **graceful**: if the reuse estimate is wrong, you pay a recompute, not a wrong answer — unlike a sharing bug, the blast radius is cost and latency. Adjacent options and when they apply: LFU with a decay keeps hot prefixes but adapts slowly and disadvantages new hot entries; "never evict until forced" is correct only for the immutable shared prefix; "pin forever" guarantees hits but grows unboundedly, so it applies to the system prompt and tool schema only. Revisit signal: a repository-context hit rate below target means the **reuse model is wrong**, not that the tier is too small.
**Signal:** States the failure as "evicts a large shared prefix to free a small unique tail" and justifies cost-aware eviction on its graceful-failure property.
**Follow-ups:**
- *Where is LRU still fine?* — Small entries of similar size; the policy is workload-conditional, not universally wrong.
- *What does "pin the immutable prefix" risk?* — Unbounded growth if the prefix set is not actually bounded; hence the pin applies to two objects, not a class.
**Red flags:** "LRU is standard practice"; no awareness that unequal entry sizes invalidate it; cannot name a failure mode for the chosen policy.

#### T07-Q14 · Idle sessions, recall latency and the TTFT budget
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** An agent session pauses for an hour and resumes. What are your options, and how do you price them?
**Model answer:** Three options, and this is the common case rather than an edge case for an agent product. **Hold the VRAM**: the session stays instant, but you are holding ~10-40 GB per idle session, and idle sessions holding VRAM cause a concurrency collapse — it is a listed failure mode with the mitigation "evict parked sessions to the lower tier on a timer, not only under pressure." **Discard and recompute**: no storage cost, but recomputing a long context is a full prefill, which the case study calls the most expensive option per resume. **Spill and recall**: pay a restore cost that lands **in TTFT**, which is the metric carrying the ≤300 ms warm-prefix target `[R]`. On the size of that restore cost, the corpus's number comes from the operational talks rather than this case study: CPU KV offload is reported to **save about 5x in TTFT** when an agentic session comes back after a pause, by avoiding the recompute `[T]` (llm-d talk). Note the framing carefully — that is a claim about offload *versus recompute*, not a universal recall latency, and the case study itself defers the figure to the sibling case studies. Two design consequences: the tier design is where the product's latency profile is decided, because a recalled session pays before its first token; and the metric that catches a tiering failure is the **resume-latency histogram**, not aggregate TTFT, because a tiering problem is invisible in the average.
**Signal:** Prices all three options and puts the recall cost in TTFT rather than in storage, with the resume histogram named as the instrument.
**Follow-ups:**
- *What does a recompute cost instead?* — Q15 does the TFLOP arithmetic on a prefix miss.
- *Why is the resume histogram the right instrument?* — A boundary set from capacity shows up only on the resume path; Q25.
**Red flags:** Treats offload as free capacity; assumes pressure-driven eviction is enough; measures aggregate TTFT.

---

### Economics of caching

#### T07-Q15 · The break-even, derived
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Someone tells you cached input is 90% cheaper, so long context is now cheap. Is that right?
**Model answer:** The discount is real and the conclusion is conditional — on a cache **hit**. Caching has a write premium as well as a read discount, so it only pays above a reuse count. The corpus's guidance: "If your cached prefix is reused more than **1.1x to 1.5x**, it is cheaper to use caching than raw tokens. Anthropic charges a **25% premium on cache writes**, so for short prefixes the break-even is higher (**3-5x reuse**)" `[R]`. Derive it `[D]`, with base input price normalised to 1 and the prefix reused `N` times:

```
Without caching:  N × 1.00
With caching:     1.25  (write premium, paid once)
                + N × 0.10  (reads at a 90% discount)
Break-even:       1.25 + 0.10N < N   →   1.25 < 0.90N   →   N > 1.39
```

So **~1.4 reuses**, consistent with the quoted 1.1-1.5x range. The short-prefix case is not a contradiction but a different regime: there the write premium applies to a prefix that is a small fraction of the request while the uncached suffix is paid at full price every time, which is where the quoted 3-5x comes from. Applied to the case study's shape: a 5,000-token prefix reused once per session is reused hundreds of times a day, so the 1.39 break-even is cleared by orders of magnitude — caching the prefix is not a marginal decision. The CFO-facing consequence is the important one: **the discount is a hit-rate multiplier, not a flat price cut**. At a 95% hit rate a 90% discount is transformative; at 20% it is noise. That is why hit rate, not the discount, is the monitored metric. And a flag for any cost model: every provider price in this corpus is a **vendor list price** `[R]` and must be re-verified before it enters a model.
**Signal:** Derives the inequality rather than quoting the range, and reframes the discount as a hit-rate multiplier.
**Follow-ups:**
- *What does a miss cost in GPU time?* — Recomputing a 5,000-token prefix is `2 × 70e9 × 5,000 = 700 TFLOP`; at an assumed 400 effective TFLOPS that is ~1.75 s, and 40 sessions missing once an hour is ~2% of a GPU permanently `[D]`.
- *Which provider prices does the corpus carry?* — All four are vendor list prices; Anthropic 90% discount with a 25% write premium, OpenAI ~50%, Google $0.20/1M reads with a separate storage fee, DeepSeek $0.003625/M and $0.0028/M — all `[R]`, all requiring verification.
**Red flags:** Quotes "90% cheaper" as a flat price cut; cannot derive the break-even; builds a cost model on unverified vendor list prices.

#### T07-Q16 · The cache-scope caveat
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Prompt caching is described as the single most important inference optimisation. Where does the assumption behind that break?
**Model answer:** The optimisation is real and the corpus ranks it first — "**number one, which is really, really important**… it is **prompt caching or KV caching**" `[T]` (CMU lecture 11). The mechanism as taught: on the first call you "calculate the representations for **all of them**. But the next time… **you've already calculated the auto regressive representations for these**. And so then you just need to **feed in the next observation and action**," which "**save[s] all the compute**" `[T]`. The assumption that breaks is **scope**: "a lot of services will **only cache the things that you had in your prompt**. So some of them will only cache the **system message and the observation**. But in reality you can also **cache the action** as well if you're clever about it" `[T]`. If your cost model assumes the whole trajectory is cached and the service caches only the system message and observations, your spend is higher than modelled with no behavioural change to explain it — a silent forecast miss. The second half of the caveat is regeneration: asked whether the accessibility tree is re-generated every turn, the answer is "As far as I know, the answer is yes. You reprocess it every time" `[T]`. A regenerated representation changes the prefix, so the cache that looked as though it covered the observation may not be hitting at all. The discipline that follows is simple and non-negotiable: **verify the boundary, do not assume it** — instrument the hit rate per prefix class so scope is a measurement rather than a belief (Q23).
**Signal:** Treats the cache boundary as something to verify rather than assume, and connects re-generation to hit loss.
**Follow-ups:**
- *How do you verify it?* — Per-prefix-class hit rate: system prompt, repository, trajectory.
- *What is the consequence of getting it wrong?* — Costs above forecast with a healthy-looking aggregate; Q26.
**Red flags:** Assumes the provider caches whatever you send; has no per-class hit-rate metric.

#### T07-Q17 · Condensation versus caching
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Context condensation is sold as a cost reduction. Explain why it can raise your bill, and how you would design around it.
**Model answer:** Because two mechanisms in the same loop pull in opposite directions, and the conflict is documented rather than inferred. **Caching** skips the re-prefill of an unchanged prefix. **Condensation** takes earlier steps, feeds them to a model and summarises them, letting the system "remove um you know **half of the context** while still keeping most of the relevant information. It's not perfect. uh sometimes you lose information that would be useful later," with a measured **2x or more cost reduction while maintaining performance on SWE-bench** `[T]` (CMU lecture 11). The conflict is stated on the slide: "**prompt caching is less effective**," and the reason is structural — "**You lose one of your inputs** when you're doing prompt caching" `[T]`. Summarising earlier steps rewrites the prefix, and a rewritten prefix is a **cache miss for every subsequent step**. So continuous summarisation pays the summarisation cost *and* loses the caching benefit, which is exactly the listed failure "cost rises despite summarising." The resolution is architectural, not a tuning parameter: **condense at a cache boundary**. Choose a summarisation point, summarise once, and treat the summary as a new immutable prefix that is cached from then on; never rewrite the prefix on every step. Two placement rules earn the marks: the boundary belongs at a **step boundary where the observation is complete**, never mid-tool-call, or the summary omits the call's result; and the loop must condense because the trajectory is otherwise unbounded — the 2,000-step case is the design case, not the exception. Validate by measuring the hit rate before and after introducing the boundary.
**Signal:** States the conflict as structural (a prefix rewrite) and gives the summarise-once-then-freeze resolution rather than a sampling or tuning knob.
**Follow-ups:**
- *Where exactly does the boundary go?* — At a step boundary with a complete observation; a mid-tool-call boundary loses the result.
- *What does the case study say to re-measure?* — Hit rate before and after the boundary, per prefix class.
**Red flags:** Proposes summarising every *k* steps; treats condensation as a pure win; does not connect it to cache invalidation.

#### T07-Q18 · Caching versus RAG
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Why not just retrieve the relevant chunks instead of holding tens of thousands of tokens in a cache?
**Model answer:** The corpus poses this objection and answers it on three axes `[R]`. **Recall**: "Context caching gives 100% recall (the whole doc is in the window), whereas RAG depends on retrieval accuracy." **Coherence**: "The model can see cross-references across the whole document." **Economics**: "At 50k tokens, the cost of a cached input is often lower than the complexity of maintaining a vector database and retrieval pipeline." The honest position — and the one that separates a strong answer — is that these are **complements, not alternatives**: retrieval decides **what enters** the context, caching decides **what stays cheap**. The reason to say that rather than pick a side is arithmetic: a 200k-token cached context at 0.3125 MB per token is **62.5 GB of VRAM for one session** `[D]`, a hard hardware constraint that RAG simply does not have. For a code-review agent the coherence axis is the strongest argument, because cross-file reasoning is the product — but the cache size is the binding constraint, so a repository-scale context is a retrieval problem while a stable, heavily-reused prefix is a caching problem. One caution on citing the corpus here: its economics claim is framed for "medium-sized documents," not arbitrary repository sizes, so do not extend the 50k figure to a 200k repository as though it scaled linearly.
**Signal:** Refuses the either/or, names which axis decides which part of the context, and declines to over-extend the 50k economics claim.
**Follow-ups:**
- *What would you measure to decide?* — Retrieval precision and prefix hit rate on the same eval set; the two numbers answer different questions.
- *What does 62.5 GB imply for a 200k repository?* — You cannot cache it whole; it must be admitted selectively or retrieved.
**Red flags:** "RAG makes long context unnecessary"; quotes the 50k economics as though it held at 200k; treats the two as mutually exclusive.

#### T07-Q19 · What is real in KV compression
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Beyond paging, what levers exist for shrinking the cache, and which of them would you actually deploy?
**Model answer:** Four levers with very different evidence behind them `[R]`. **KV quantization** — halves or quarters the largest memory consumer, independent of model architecture, but it "adds error to attention, not just weights; the error compounds with context length." So it must be validated at the **p99** context, not the mean, and the case study makes it the last lever in the tuning order precisely because it is the only one that can change output quality. **Token eviction** — drops the least-attended tokens for a direct saving, but loses information irrecoverably and is "hard to predict what mattered"; it breaks on tasks requiring exact recall of an early detail, which is a real risk for code review, so the case study restricts it to research and cache-pressure relief. **Low-rank compression** — shrinks the stored representation, approximation quality varies, needs per-model validation; deferred. **RAD-O (Retrieval Augmented Decoding)** — "**compresses** the KV cache of long documents into 'Latent tokens'": rather than storing full KV vectors for 1M tokens it stores a representation "that is **10x smaller**," which "Enables **2M+ token contexts on hardware that previously only supported 200k**." Take RAD-O seriously as a direction and treat it as **a vendor-class claim with no attached benchmark** `[R]` — in the table for completeness, "watch, do not adopt." The deployment answer: GQA-bearing model first, KV quantization as the first lever if memory binds, and if the eval shows quantization costs more quality than the memory it saves, **reduce admitted context instead** — which is a product decision and should be escalated as one rather than absorbed by engineering.
**Signal:** Separates the four levers by evidence quality, refuses to adopt the vendor claim, and names the escalation path when quantization fails its eval.
**Follow-ups:**
- *Why validate quantization at p99 context?* — The error compounds with length, so the mean context understates it.
- *What is the decision if quantization fails?* — Admitted context, escalated as a product decision.
**Red flags:** Adopts RAD-O on its vendor description; deploys KV quantization without an eval gate; proposes token eviction for exact-recall workloads.

---

### The sharing gate

#### T07-Q20 · What sharing scope is permitted
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Your product's biggest available saving is sharing KV blocks between requests. Walk me through the decision, and tell me who owns it.
**Model answer:** This is the highest-leverage decision in the topic, and it is not an engineering decision. The options, with the failure mode of each `[R]`:

| Option | Saving | Failure when it breaks |
|---|---|---|
| **Per-session only (Phase 1 chosen)** | None — forfeits the 100x | Safe by construction |
| Share the immutable infrastructure prefix within a tenant | Most of the system-prompt saving | Breaks if the "immutable" prefix ever embeds tenant data — the classification must be enforced, not asserted |
| Share repository context within a tenant, across sessions | The largest practical saving | "A cache-hit bug becomes a cross-session data leak"; and it breaks if two branches share a hash prefix and diverge |
| Share across tenants | Maximum efficiency | A cross-tenant leak — and the KV tensors "are not human-readable but *are* invertible in principle" |
| Cross-tenant with cryptographic isolation | Efficiency with a provable boundary | Unproven at this scale; encrypting blocks and decrypting on read costs |

Chosen: per-session caching plus a shared immutable infrastructure prefix, explicitly classified as containing no customer data. **Why it is a legal question**: a shared prefix is shared *tensor memory*, and whether two tenants may share a KV block is a data-isolation question, not a performance one. **Why it must be decided deliberately**: the engineering change is small — the block table already supports it — "which is precisely why the decision must be made deliberately rather than by default." Left to drift, the default will be sharing. And the value of a yes grows with scale: at 40 sessions the saving is 40x, at 400 it is 400x, so whatever legal decides is worth re-asking at 10x.
**Signal:** Separates the five options by their failure mode, argues the legal framing from "shared tensor memory," and notes the default-drift risk that makes the decision urgent.
**Follow-ups:**
- *How would you prove isolation?* — An invariant test in CI on every release plus a canary with synthetic tenants, which is the case study's mitigation.
- *What reopens the question at 10x?* — The value of a yes grew and the cost of a no grew faster; the answer is not permanent.
**Red flags:** Treats this as a performance decision; shares by hash equality; cannot name the failure mode of the option it recommends.

#### T07-Q21 · Hashing is not authorisation
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Your prefix cache keys on a hash of the prompt. Two tenants' prompts hash to the same value. What have you actually built?
**Model answer:** A silent cross-tenant leak. A prefix hash answers "are these the same bytes," which is a **lookup** question; it cannot answer "may these two principals share memory," which is an **authorisation** question. When the two are conflated, a hash collision is not a cache inefficiency — it is one customer seeing another's context, and the case study rates the blast radius as "**catastrophic — contract-ending**." The mitigation is stated as a rule: "Sharing scope by **tenant identity**, never by hash" `[R]`, enforced with an invariant test on every release and a canary using synthetic tenants, and the recovery procedure is containment — "**disable all sharing immediately**, then investigate. Recovery is not the goal; containment is." Three things a strong answer adds. First, the sharing key must carry the tenant **as well as** the model version and the precision, because omitting any of the three produces a wrong answer rather than a slow one `[D]`. Second, "the tensors aren't human-readable anyway" is not a defence — the corpus notes they "are **invertible in principle**" `[R]`. Third, the adjacent case is a business question, not a technical one: the same repository read by two different customers is per-tenant scope by default, and any exception needs a contractual basis rather than a technical argument.
**Signal:** Reframes the bug from a lookup collision to an authorisation gap, and names containment as the incident response.
**Follow-ups:**
- *What goes in the sharing key?* — Tenant, model version, precision — the last two are the cache-correctness half (Q22).
- *What is the CI test?* — An invariant test plus a synthetic-tenant canary, on every release.
**Red flags:** Relies on hash collision probability as a safeguard; treats the leak as a performance bug; proposes "encrypt the hash" as the fix.

#### T07-Q22 · The cache correctness contract
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Write the contract that makes the cache correct. What must be true at all times?
**Model answer:** Four clauses, because a wrong cache hit is a **wrong answer**, not a slow one. **One — key composition.** The cache key must include the **model version**, the **precision**, and the **tenant**. A weight update invalidates every block, and BF16 versus FP8 KV is a different cache; omitting any of the three produces a wrong answer rather than a slow one. **Two — prefix exactness.** Prefix matching is exact and positional: "A timestamp, a request ID or a re-ordered tool schema at the front of the prompt invalidates everything after it. Keep the prefix byte-identical and put variable data last." **Three — immutability.** Shared blocks are read-only and reference-counted; copy-on-write is what protects divergence `[R]`, and a write to a shared block is a correctness bug rather than a performance one. **Four — boundary placement.** Condensation boundaries fall at step boundaries where the observation is complete, never mid-tool-call, or the frozen prefix contains a truncated result. The operational clauses that follow: version the immutable prefix and treat it as read-only; **purge the cache on deploy**; add a **prefix-change canary**, because a prefix mutation "silently converts a cheap deployment into an expensive one"; and monitor prefix-mutation events per prefix class. The framing to close on is the case study's own: the cache is not a correct-by-default optimisation — its correctness is a property of the prefix-matching rule, and each clause above is a way that rule can be violated.
**Signal:** Enumerates key composition, prefix exactness, immutability and boundary placement as one contract rather than as four unrelated tips.
**Follow-ups:**
- *What does an unversioned key do on a model upgrade?* — Serves yesterday's KV against today's weights; Q27 works the failure.
- *Why does a separator change matter?* — It is a prefix mutation, so it invalidates a whole population; Q23 diagnoses it.
**Red flags:** Keys on a request hash alone; treats the cache as transparent to the model; no purge-on-deploy step.

---

### Debugging

#### T07-Q23 · Hit-rate collapse
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Cost triples overnight. Traffic is flat, the hit rate has fallen from 96% to 71%, and no deploy is recorded. Diagnose it.
**Model answer:** Start from the mechanism: prefix matching is **exact and positional**, so any change early in the prompt invalidates everything after it. Ranked suspects, cheapest to check first. **A reordered tool schema** — JSON key order is stable in a dict but not always in serialisation, and this is the classic silent invalidator. **A volatile field injected at the front** — a timestamp, request ID, session ID or trace ID; the case study's rule is to keep the prefix byte-identical and put variable data last. **A separator or whitespace change** in a template. **A regenerated representation layer** — lecture 11: "You reprocess it every time" `[T]` — which changes the observation prefix on every step. **A client that re-renders history differently between turns**, which destabilises the trajectory prefix. **A prompt-template version shipped without a canary** — a plausible "no deploy" cause, because it can be a content or config change rather than a code release. The instrument is the **per-prefix-class hit rate** (system prompt, repository, trajectory) split out, because an aggregate number tells you a collapse happened and the split tells you which class broke. The action is to diff the prefix against the last known-good, then add a prefix-change canary so the next mutation alerts instead of costing money. Note the trap: the reflex fixes are capacity or a sampler change, and both leave the prefix broken. A related but distinct signature is worth naming — a naive load balancer sending follow-up turns to random replicas recomputes the prefix on every turn and produces a cache hit rate the corpus describes as "atrocious" in Prometheus `[T]` (NextGen talk); that is a routing bug, not a prefix bug, and the fix is prefix-aware routing.
**Signal:** Goes straight to prefix exactness, uses the per-class hit rate as the instrument, and separates the prefix bug from the routing bug.
**Follow-ups:**
- *Why is the aggregate hit rate insufficient?* — It cannot tell you which prefix class broke, and the classes have different owners.
- *What is the durable fix?* — A prefix-change canary plus per-class monitoring; Q22's contract.
**Red flags:** Blames capacity, the provider or the model; does not know prefix matching is positional; has no per-class metric.

#### T07-Q24 · OOM at low utilisation
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Requests start failing with out-of-memory errors while aggregate VRAM use looks unremarkable. Walk me through it.
**Model answer:** Out-of-memory with VRAM that looks mostly free is a **fragmentation or mismatch** signature, and there are three things to check in order. **One — is paging actually on everywhere?** A single contiguous path — an engine's non-paged fallback, a component that still reserves max-length, a different engine version on one replica — reintroduces external fragmentation, so total free memory is high while no contiguous region is large enough for the next reservation. **Two — is the block size mismatched to the workload?** Waste above the 4% target means the allocator is holding space it cannot reuse; the case study's revisit rule is explicit that measured waste over 4%, or a visible block-table overhead, both indicate the block size is wrong. **Three — is it actually one long session?** Aggregate utilisation hides a single session, and "a few long sessions exhaust VRAM" is its own listed failure with the mitigation "per-session cap with explicit rejection." So the metrics to pull are **memory per session p50 and p99**, not aggregate. Two differentials worth naming, because they look similar in aggregate and have different fixes: **throughput collapsing at long context with recomputation on every request** is an eviction problem, not a fragmentation one — the corpus records a configuration where "when we tested 28k there is a KV cache recomputation happening every single time and due to that like we had a drop in performance" `[T]` (ROCm/WideEP talk, vLLM Bengaluru meetup) — and that is fixed by eviction policy and tiering, not by the allocator. And **prefill/decode imbalance** shows as GPUs idle while requests queue, which is a different problem again.
**Signal:** Diagnoses from per-session p50/p99 rather than aggregate utilisation, and separates fragmentation from recomputation from phase imbalance.
**Follow-ups:**
- *What is the fix for a single long session?* — A per-session cap plus condensation at a boundary; never silent truncation.
- *How do you confirm paging is on?* — Assert it and measure waste directly; do not assume it from the feature flag.
**Red flags:** "Add memory" or "get a bigger GPU"; reads only aggregate utilisation; has no per-session memory metric.

#### T07-Q25 · Resume latency spike
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** TTFT is normal on first turns and spikes on session resume. What is happening, and what do you fix?
**Model answer:** The session's blocks were evicted from the recent tier during the pause, so the resume pays a tier recall — or a recompute if the block is gone. Three checks in order. **First, the eviction trigger.** If sessions are evicted only under memory pressure, this is the named failure: the rule is to "evict parked sessions to the lower tier on a timer, not only under pressure," because a parked session's blocks are the cheapest thing to free and pressure-driven eviction fires too late. **Second, the tier boundary.** It is set from the resume SLO, so a boundary set from capacity instead — "fill VRAM and spill the rest" — produces exactly this symptom by construction. **Third, the instrument.** Measure **resume TTFT as its own histogram**; a tiering failure is invisible in aggregate TTFT because it affects one path only. The fix is a **larger recent tier or a shorter eviction timer**, and the case study is explicit that the fix is *not* a faster disk — the tier below is chosen for cost and will not close an SLO gap. If the recall path itself is the problem rather than the boundary, the corpus's operational figure for the offload path is that CPU KV offload is reported to **save about 5x in TTFT** when an agentic session returns after a pause `[T]` (llm-d talk) — so compare that against the recompute cost before concluding offload is insufficient. Note the deployment knob that sets how much of VRAM the cache may hold: `--gpu-memory-utilization`, run at 0.90 in the corpus's example `[T]` (NextGen talk).
**Signal:** Separates eviction timing from tier size from recall speed, and refuses the faster-disk reflex for an SLO breach.
**Follow-ups:**
- *Why is resume latency the right SLI?* — It is the only path where a tiering decision is observable.
- *What does a recompute cost instead?* — Q15's miss arithmetic: ~1.75 s of GPU time for a 5,000-token prefix `[D]`.
**Red flags:** "Cache misses happen"; proposes faster storage or more VRAM without checking the eviction timer; measures only aggregate TTFT.

#### T07-Q26 · Cost above forecast with a normal hit rate
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Spend is 40% above forecast and the hit rate is exactly where you predicted. Where is the money going?
**Model answer:** Into the **writes**. If hits are normal, the extra cost is prefixes you are writing and then barely reading, each paying the 25% write premium `[R]` for too few reads. The break-even makes the sign obvious: `1.25 + 0.10N < N → N > 1.39` `[D]`, so a prefix written and read once costs about 35% *more* than not caching it, and one written and read twice is roughly break-even. The instrument is therefore not the hit rate but **writes per distinct prefix** — a distribution nobody instruments by default. Three causes to look for. **Prefix fragmentation**: a small variable element early in the prompt — a per-request user blob, a session ID, an ordering difference — makes every request a distinct prefix, so the shared classes keep hitting and the aggregate hit rate looks healthy while a long tail of one-shot prefixes accumulates write charges. **A minimum cacheable prefix length set too low**, so short prefixes are written and never re-read. **Condensation boundaries firing too often**, each creating a new prefix that is written once. The fixes are the corresponding knobs: raise the minimum prefix length worth caching, stabilise the front of the prompt so prefixes are genuinely shared, and move condensation to fewer, deliberate boundaries (Q17). The general lesson, and the one to say: the read discount is a **hit-rate** multiplier while the write premium is a **prefix-count** multiplier — monitoring only one of them is how a cache becomes expensive while looking healthy.
**Signal:** Knows the write premium is charged per distinct prefix, and instruments writes-per-prefix rather than hit rate.
**Follow-ups:**
- *Why does a normal hit rate hide this?* — The hit rate is a ratio over reads; writes are a separate, unmonitored stream.
- *How does condensation create writes?* — Each boundary mints a new prefix that must be written; Q17.
**Red flags:** "Raise the hit rate" when the hit rate is already on target; does not know a write premium exists; never looks at writes.

#### T07-Q27 · Wrong answers from old KV
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** After a deploy, a surface starts producing output that is subtly wrong but fluent, and latency and hit-rate metrics look perfect. What is your hypothesis?
**Model answer:** Stale KV served against new weights — the one cache failure that produces a **plausible wrong answer** rather than a cost or latency problem, and the reason it is invisible in exactly the metrics you would reach for. The cache is working perfectly; it is serving yesterday's model. The mitigation on the write side is a clause of the correctness contract: the cache key must include the **model version and precision**, so a weight update invalidates everything `[R]`, and the operational step is to **purge the cache on deploy**. Detection needs two things a cache dashboard does not have: the model version recorded *in* the cache key, and a **post-deploy eval** on a fixed prompt set, because nothing in latency or hit rate will move. Three related cases worth separating, because the fixes differ. **Stale weights** — an unversioned key across a revision: purge and version the key. **Stale tiers** — a block restored from the wrong tier or against a re-rendered prefix: the tier is a correctness surface, so a block's identity must be its prefix and version, not its position. **Stale corpora** — the "static" corpus behind a persistent disk cache changed: the case study's rule is that this option breaks exactly when the static corpus moves, and stale cache is a wrong answer, which is why a per-customer repository rules that option out. The general discipline: every cache layer carries a version in its key and gets purged on deploy. Note the contrast with Q26 and Q23 — those are cost failures with healthy correctness; this is a correctness failure with healthy cost metrics, which is why it needs a different instrument.
**Signal:** Names staleness as a correctness bug invisible in cache metrics, and separates stale weights from stale tiers from stale corpora.
**Follow-ups:**
- *What does the post-deploy eval need?* — A fixed prompt set run against the new revision, plus the version field in the key.
- *Why is the disk-cache-of-a-corpus option a poor fit here?* — The per-customer repository is not static, and invalidation is not automatic.
**Red flags:** Only inspects hit rate and latency; no model version in the cache key; treats the tier as storage only.

---

### Scale and design

#### T07-Q28 · Sizing a node: the division that decides everything
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Whiteboard it. You have one node, a 70B-class model, and a 40 GB KV budget after weights. How many concurrent sessions does it serve, and what are your levers?
**Model answer:** Do the division before anything else. Take a session with a 32k-token repository context and a 100k-token trajectory — 132,000 tokens — at 0.3125 MB per token: `132,000 × 0.3125 MB = 41.25 GB` per session `[D]`. **One session already exceeds a 40 GB KV budget.** Forty concurrent sessions would need **1,650 GB** of KV alone. Without GQA the same session would need `41.25 × 8 = 330 GB` — more than four times the largest single accelerator, before weights — which is why the 8x at under 0.2% `[R]` is a precondition rather than an optimisation (Q3). The forced conclusion, and the sentence to say out loud: **concurrency is set by the KV cache, not by compute. At 128k-class contexts a single node serves a handful of sessions, not dozens.** Everything else in the design follows from that number. The levers, in order of leverage `[D]`: **admitted context** (132k → 64k halves session memory to 20 GB and roughly doubles concurrency — the cheapest lever available and a *product* decision); **KV quantization to 8-bit** (halves it again, validate at p99 context first); **sharing the repository context within a tenant** (removes up to 32k tokens per session of marginal cost, a ~24% reduction at this shape); **condensing the trajectory at boundaries** (bounds growth but costs hit rate at every boundary); and **paging**, which is table stakes but does not rescue you at this size (Q10). One boundary condition to state: the model's 128k maximum context is a property of the model `[T]`, not a capacity plan, which is why the case study admits context by policy rather than by capability.
**Signal:** Performs the division, states the forced conclusion as a sentence, and orders the levers with admitted context first because it is a product decision.
**Follow-ups:**
- *Why is admitted context a product decision?* — It changes what the product can do, so it is a business tradeoff rather than a config value.
- *What changes at 100 concurrent sessions?* — The shared-prefix saving becomes 100x and per-session context becomes the dominant term; Q30.
**Red flags:** Sizes on weights first and treats KV as an afterthought; quotes 42 GB but cannot divide it by a context; proposes paging as the long-context fix.

#### T07-Q29 · Design the cache architecture end to end
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Design the cache architecture for a long-horizon agent platform running many concurrent sessions. Give me the decisions in dependency order and tell me what you would tune first.
**Model answer:** Eight decisions, in the order they constrain each other.
**1. Three lifetimes, three policies.** Split the prompt into the immutable prefix (system prompt + tool schema), the semi-stable context (repository), and the volatile trajectory. Each has a different sharing scope, eviction priority and legal status; the block table is what makes them independently addressable — without it there is one buffer per session and none of these distinctions can be expressed.
**2. Allocation.** Paged blocks with a block size tuned to the observed context distribution; waste from 60-80% to under 4% `[R]`.
**3. Sharing scope.** Per-session plus a shared immutable infrastructure prefix classified as containing no customer data; repository sharing deferred to a legal ruling; cross-tenant sharing forbidden without a written basis (Q20).
**4. Prefix discipline.** Byte-identical immutable prefix, variable content last, exact positional matching, versioned.
**5. Tiering.** VRAM → host memory → SSD, with the tier boundary set from the resume SLO and eviction on an inactivity timer as well as on pressure.
**6. Eviction.** Cost-aware by `recompute cost × reuse probability`, with the immutable prefix pinned (Q13).
**7. Caching scope.** System prompt + tool schema + observations; condensation only at explicit cache boundaries (Q11, Q17).
**8. Correctness.** Key includes model version, precision and tenant; shared blocks immutable and refcounted; purge on deploy (Q22).
**Tune order:** admitted context → prefix stability → sharing scope → eviction policy → **KV quantization last**, because it is the only step that can change output quality. **Monitor:** hit rate split by prefix class, waste against the <4% target, recomputation rate, sessions per node, memory per session p50/p99, resume TTFT, eviction and recall counts by tier, and prefix-mutation events.
**Signal:** Presents the decisions in dependency order, justifies the tune order by the quality-risk argument, and closes with a monitoring set rather than an architecture diagram alone.
**Follow-ups:**
- *What is the single most important metric?* — Cache hit rate split by prefix class.
- *What is the first thing you tune?* — Admitted context, and it needs the earliest product conversation.
**Red flags:** Jumps to tiering before prefix discipline; tunes quantization first; no per-prefix-class instrumentation.

#### T07-Q30 · 10x
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** The estate grows 10x in sessions. What breaks, what inverts, and what survives unchanged?
**Model answer:** Five shifts. **The cache stops being an optimisation and becomes the architecture** — at 10x sessions per-session memory binds every node, and the tier design becomes the thing that decides the product's cost curve. **Shared-prefix scope is the whole ballgame**, and the saving is linear in concurrency: at 40 sessions it is 40x, at 400 it is 400x — so whatever legal decided in Phase 1 is worth re-asking, because "the value of a yes grows with scale while the cost of a no grows faster." **Condensation becomes mandatory, and it fights caching** — the 2x-or-more reduction `[T]` only materialises if summaries land at cache boundaries; continuous summarisation pays twice (Q17). **Tiering becomes a storage system** — with 10x sessions and 24-hour resumability the SSD tier is measured in terabytes and needs the discipline of any storage tier: durability, eviction, consistency and a recall-time SLO. **The cache stops being single-node property**, which is where routing enters: the corpus's operational report is that "precise KV cache routing perform much better than the approximate," where approximate means you "hash a particular request and then know which instance it goes to," because with saturated KV and constant eviction "maintaining KV events show that precise KV cache affinity helps a lot" `[T]` (llm-d talk) — an event-based affinity beats hashing, and a naive load balancer makes the hit rate "atrocious" regardless of how good the cache is `[T]` (NextGen talk). **What inverts:** "does the model fit on the GPU" stops being the sizing question; at 10x it is "how many sessions fit," answered by `context × 0.3125 MB`, a number with nothing to do with the parameter count. **What survives:** the block table, copy-on-write, the exact-prefix rule and the break-even arithmetic — those are structural.
**Signal:** Correctly moves the cache from a single-node optimisation to a routed, storage-backed system, and separates what inverts from what is structural.
**Follow-ups:**
- *Why does hashing lose to events at this scale?* — Eviction churns block state, so an approximate hash goes stale; per-block create/evict events do not.
- *What stays true at any scale?* — The block table, COW, exact-prefix matching and the break-even arithmetic.
**Red flags:** "Just add GPUs"; assumes the cache stays a single-node concern; no view on routing or on the legal question reopening.

---

## Whiteboard exercises

### Exercise 1 — Size the node and defend the number
**Prompt.** A 70B-class model on one node: 80 layers, 8 KV heads, head dim 128, BF16 KV. After weights and activations you have **40 GB** for KV. Sessions hold a 32k-token repository context and a trajectory that grows toward 100k tokens. Produce the per-session memory, the sessions per node, and the ordered list of levers you would pull. You have 20 minutes and no benchmark data.

**What to produce.** The per-token derivation with units, the per-session multiplication, the concurrency division, a table showing contiguous versus paged at two context lengths, and the levers ranked by leverage with the owner of each named.

**Expected whiteboard.**

```
Step 1 — per token
  2 (K,V) x 8 kv_heads x 128 head_dim x 2 bytes (BF16) = 4,096 B / layer / token
  4,096 x 80 layers                                    = 327,680 B
                                                       ~ 0.3125 MB / token   [D]

Step 2 — per session
  32k repo + 100k trajectory = 132,000 tokens
  132,000 x 0.3125 MB = 41.25 GB per session          [D]
  >>> ONE session exceeds the 40 GB KV budget.

Step 3 — concurrency  (max_concurrency ~ KV_budget / (bytes_per_token x ctx))
  +---------------+-----------+------------------+------------------+
  | allocation    | usable KV | @8k  (2.56 GB)   | @132k (41.25 GB) |
  +---------------+-----------+------------------+------------------+
  | contiguous 70%|   12 GB   |   4              |   0              |
  | paged    <4%  |  38.4 GB  |  15              |   0              |
  +---------------+-----------+------------------+------------------+
  >>> paging is the fix at 8k. At 132k the ONLY lever is admitted context.

Step 4 — levers, by leverage
  1. admitted context 132k -> 64k   : 41.25 GB -> 20 GB, ~2x concurrency  [product decision]
  2. KV quantization to 8-bit       : halves again   [validate at p99 context]
  3. share repo context in-tenant   : -32k tokens/session, ~24%           [gated on legal]
  4. condense trajectory at boundary: bounds growth, costs hit rate/boundary
  5. paging                         : table stakes, does not rescue 132k
```

**Grading rubric.**
- Derives 0.3125 MB per token from the formula with units shown, rather than quoting the 42 GB figure or a per-token number from memory.
- Performs the division and states the forced conclusion out loud: **concurrency is set by the KV cache, not by compute**, and one 132k session exceeds the budget.
- Names admitted context as the first lever and labels it a **product** decision — not a config value — and puts KV quantization last on the quality-risk argument.
- Distinguishes the two regimes in the table: paging is the fix at 8k and irrelevant at 132k.

### Exercise 2 — Diagnose a hit-rate collapse
**Prompt.** `Quarry Review` cost per repository review rises 2.9x overnight. Request volume, model revision and output lengths are flat. Cache hit rate fell from 96% to 71%, and no production deploy is recorded. You have the serving logs, the cache's per-prefix-class metrics, and the prompt template repository. Produce the diagnosis, the fix and the rollback.

**What to produce.** A ranked hypothesis list with a decisive test per hypothesis and an explicit direction-of-evidence argument; the prefix layout as you believe it should be; the fix; and the rollback.

**Expected whiteboard.**

```
Hypotheses (ranked by likelihood x cheapness to test)
  1. Tool schema reordered (serialisation order) ..... test: byte-diff the prefix vs last known-good
  2. Volatile field injected early (ts / req id / sess) test: grep the template head for volatile tokens
  3. Separator / whitespace change in template ........ test: template repo diff, not deploy log
  4. Representation layer regenerated per step ........ test: prefix hash stability across steps
  5. Client re-renders history differently ............ test: prefix diff across turns 1..N

Direction of evidence:
  hit rate 96% -> 71%  (a POPULATION of prefixes broke, not a per-request failure)
  cost 2.9x, volume FLAT, output length FLAT
        -> consistent with (1)-(3): one class of prefix invalidated for everyone
        -> NOT consistent with capacity, the provider, or the model (volume flat)

Per-class split is the decisive instrument:
  system prompt  96% -> 96%   (unchanged)
  repository     96% -> 96%   (unchanged)
  trajectory     96% -> 41%   <-- the broken class

Prefix layout (target):
  [ IMMUTABLE system prompt | tool schema | repo context ]  <- byte-identical, versioned
  [ trajectory: observations | actions ]                    <- append-only, volatile LAST

Fix:      restore the prefix -> verify per-class hit rate -> land a prefix-change canary
Rollback: re-pin the prompt template version; the cache needs no purge (content-addressed)
```

**Grading rubric.**
- Reads the *direction* of the evidence jointly: volume flat with a hit-rate collapse eliminates capacity, the model and the provider, and points at a prefix mutation affecting a whole population.
- Names the **per-prefix-class hit rate** as the decisive instrument rather than the aggregate, and identifies the trajectory class as the broken one.
- States the mechanism — exact positional prefix matching, so a change early invalidates everything after it — rather than treating it as a generic "cache miss."
- Prescribes a **prefix-change canary** as the durable fix, not just restoring the prefix, and gives a rollback that does not require a cache purge.

### Exercise 3 — Decide sharing scope with legal in the room
**Prompt.** You operate 40 concurrent sessions per node. Legal will approve an immutable infrastructure prefix shared within a tenant and is willing to consider more. Engineering wants cross-tenant sharing because it would collapse the memory bill. Produce the sharing decision, the enforcement mechanism, the isolation test, and the one thing you will not do.

**What to produce.** A decision table with a saving estimate and a failure mode per row; the block-table sketch showing how scope is enforced; the CI test; and the explicit refusal with its reason.

**Expected whiteboard.**

| Option | Saving at 40 sessions | Failure when it breaks | Decision |
|---|---|---|---|
| Per-session only | 0 | None — safe by construction | Baseline |
| + immutable infra prefix, in-tenant | 62.5 GB -> 1.56 GB on the prefix (40x) | "immutable" prefix embeds tenant data — classification must be **enforced** | **Ship** |
| + repository context, in-tenant, cross-session | Removes up to 32k tokens/session (~24%) | A cache-hit bug becomes a cross-session leak; branch hashes diverge | Phase 2, pending legal |
| + cross-tenant | Linear in concurrency | Cross-tenant leak; KV is "invertible in principle" | **Refuse** without a written basis |
| + cross-tenant, cryptographic isolation | Same | Unproven at this scale; encryption cost per read | Research only |

```
Block table — scope enforcement is a field, not a hash
  tenant:T1  session:s7  -> [blk 4][blk 9][blk 12]   refcount 3  SHARED, IMMUTABLE
  tenant:T1  session:s8  -> [blk 4][blk 9][blk 31]   blk 31 private (COW on divergence)
  tenant:T2  session:s1  -> [blk 55][blk 56]         <- never resolves to T1's blocks
                                    ^ lookup key = (tenant, prefix_hash, model_ver, precision)
                                      hash answers "same bytes"; tenant answers "may share"

CI test:   invariant assertion  no block is referenced by two tenants, every release
           canary with synthetic tenants T1/T2 -> assert zero shared block ids
Incident:  on any invariant failure -> DISABLE ALL SHARING, then investigate
           (containment, not recovery)

The thing I will not do: share by hash equality. Hashing is a lookup, not an authorisation.
```

**Grading rubric.**
- Sizes the in-tenant immutable-prefix saving in GB at the stated concurrency (62.5 GB -> 1.56 GB, a 40x saving) rather than asserting "it saves memory."
- Puts **tenant identity in the lookup key** alongside the prefix hash, the model version and the precision — and states the distinction between a lookup question and an authorisation question.
- Specifies a CI invariant test plus a synthetic-tenant canary, and an incident response that is containment (disable sharing) rather than recovery.
- Refuses cross-tenant sharing explicitly, on the "invertible in principle" ground rather than on a claim that the tensors are unreadable, and flags that the engineering change is small — which is why the decision must be deliberate.

## Sources

- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/02-kv-cache-and-context-caching.md` — the 42 GB per-user calculation with its full expression, the GQA/MQA reduction-versus-quality table (8x at <0.2%; 64-128x at 2-3%), the `Most Recent (VRAM) -> Frequent (HBM) -> Occasional (SSD)` tiering model attributed to SGLang, the four providers' prompt-caching prices with the 25% write premium and the 1.1-1.5x and 3-5x break-even guidance, RAD-O and its 10x latent-token compression claim, and the caching-versus-RAG recall/coherence/economics argument. **All four pricing rows are vendor list prices and require re-verification before use; RAD-O is a vendor-class claim with no attached benchmark.**
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/05-paged-attention.md` — internal and external fragmentation with the 99.9% reservation-waste example, the three-step blocks / logical-versus-physical / block-table mechanism with the 16-token block, the 60-80% to under 4% waste reduction and the 60% to 96%+ efficiency gain, the block manager with allocation and Paged Swap eviction, the 100-users-by-5,000-token copy-on-write example, and the 4-requests-to-20-30-requests concurrency comparison.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — the prefill/decode bottleneck split and the TTFT and TPOT definitions that frame the memory and bandwidth argument in Q5 and Q14.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` — prompt caching named as the highest-priority technique and its mechanism, the cache-scope caveat about which parts of the prompt services actually cache, the accessibility-tree re-processing answer, agent trajectory lengths from ~100 steps to 2,000 steps with their token counts, context condensation with the 2x-or-more cost reduction on SWE-bench, and the explicit statement that condensation makes prompt caching less effective. (This lecture is in the **no-suffix** CMU directory — verified by listing.)
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt` — masked attention as the reason prior embeddings need not be recomputed with the position-3/position-4 formulation, the chain-rule constraint, the 8 GQA KV heads across the Llama 3.1 series, and the 128k maximum context as a model property. (This lecture is in the **`_2`** CMU directory — verified by listing.)
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — CPU KV offload saving "about 5x in TTF" when an agentic session returns after a pause, and the precise-versus-approximate KV cache routing result with the argument from per-block create/evict events, used in Q14, Q25 and Q30.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt` — the naive-load-balancer failure where the prefix is recomputed three times in three conversations and the exported hit rate is "atrocious," and the 90%-of-VRAM KV utilisation configuration, used in Q12, Q23 and Q30.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` — the configuration in which KV cache recomputation happened on every request at 28k input tokens, used in Q24 as the differential against a fragmentation diagnosis.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` — topic inventory confirming corpus coverage of KV cache compression (quantization, eviction, cross-head sharing, low-rank), PagedAttention, prompt caching, RadixAttention and prefill-decode disaggregation. **This file is a table of contents and contains no figures** — it is cited for coverage only and supplies none of the numbers in this bank.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
- `01-case-studies/T07-kv-cache.md` — the authoritative source for every decision table, the §8 arithmetic (41.25 GB per session, 1,650 GB for forty sessions, the 40x and 100x sharing savings, the 1.39 break-even and the 70 s/hour recompute cost) and the §9 numbers table, including the note that engine-advisory and CVE-class claims live in the sibling serving-topics rather than here.
