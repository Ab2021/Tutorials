# Interview Bank: Parallelism, MoE & WideEP

> `T11` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T11-parallelism-moe.md) · [Case study](../01-case-studies/T11-parallelism-moe.md) · [Design blueprint](../03-design-blueprints/T11-parallelism-moe/HLD.md)
> **Questions:** 28 (8 × L3, 13 × L4, 7 × L5) · **Format:** progressing from mechanism to cluster design

## How to use this bank

The questions are ordered so the bank reads as one interview: fundamentals first, then sizing
arithmetic, then configuration and failure diagnosis, then open design. Ask them in order for a
35-minute loop, or sample the L4/L5 block for a senior screen.

Every number here is traceable to a transcript (`[T]`, speaker named), a supporting repo (`[R]`,
path named), or is my own derivation (`[D]`). If a candidate quotes a figure, ask where it came from —
a candidate who cannot attribute a number does not own it.

---

### Fundamentals — what each parallelism actually splits

#### T11-Q1 · What does tensor parallelism split, and what does it cost you?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** A colleague says "just use tensor parallel 8, it's the standard." Explain what TP
actually splits and what it charges you for the privilege.

**Model answer:** Tensor parallelism shards each layer's weight matrices across devices, so every
device holds a slice of every layer and they cooperate on the same forward pass. It is the standard
answer for a dense model that does not fit on one GPU, and it is the reason a single model can span
8 GPUs inside a node.

The cost is communication, and it is the most expensive kind: an **all-reduce on every layer**, not
once per forward pass. That is why TP is bounded by the interconnect. Over NVLink inside a node it is
affordable; the moment TP spans nodes, every layer pays a cross-node all-reduce and scaling falls off
a cliff. The practical rule the corpus supports is **keep TP intra-node** `[D]`.

The second cost is shape. TP slices matrices into thinner and thinner GEMMs as the degree rises,
which is arithmetic-inefficiency territory. That is a large part of why the headline configuration
result in the corpus is a *rejection* of naive 8-way TP: for DeepSeek prefill on B200 in a
disaggregated prefill pool, a tuned mix across **16 GPUs per model replica** beats naive single-host
8-way TP on both TTFT and throughput per GPU `[T]` Kwon. The mechanism is not that TP is bad — it is
that TP was doing a job (spanning the model) that pipeline, sequence and expert parallelism do better,
and its communication bill was being paid for nothing.

**Signal:** The candidate volunteers the all-reduce-per-layer frequency unprompted, and locates TP's
limit at the *node boundary* rather than at a GPU count.

**Follow-ups:**
- *So where does TP stop?* — the intra-node rule and the NVLink/PCIe distinction.
- *What is the alternative to TP for spanning a model?* — pipeline parallelism and its bubbles.
- *Why does 8-way TP specifically lose in the corpus's prefill result?* — see T11-Q13.

**Red flags:** Says "TP is for large models" without naming what it communicates, or treats 8 as a
magic number rather than an interconnect-dependent limit.

---

#### T11-Q2 · Pipeline versus tensor parallelism — when does PP win?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** When would you reach for pipeline parallelism instead of tensor parallelism?

**Model answer:** PP splits the model by **layers into stages**, so device 0 holds layers 1–N and
device 1 holds N+1–2N. Communication is point-to-point between adjacent stages rather than an
all-reduce on every layer, which is why it is the natural way to span nodes — it is far cheaper per
boundary crossing than TP.

What it costs is **pipeline bubbles**: while stage 1 is working, stage 0 has nothing useful to do
unless you feed it another microbatch. The corpus gives the arithmetic — bubble fraction is
approximately `(PP_degree − 1) / (microbatches + PP_degree − 1)` `[D]` — which says the bubble
amortises only as microbatches grow. So PP is a high-concurrency technique: it pays off when you have
enough in-flight requests to keep every stage busy, and it is actively harmful at low load, where the
bubble dominates and you have effectively idled most of your hardware `[D]`.

There is a second, less obvious use that the corpus highlights. In the tuned prefill configuration,
PP was used not to span an unspannable model but to **parallelise chunks of a long prefill sequence**
`[T]` Kwon. That is a latency play for TTFT, not a memory play — worth naming, because it is the
non-obvious half of why that configuration wins.

**Signal:** Names the bubble formula or the amortisation intuition, and distinguishes the two uses —
spanning a model versus chunking a long prefill.

**Follow-ups:**
- *Write the bubble fraction.* — the formula and what it implies about concurrency.
- *Does PP help TTFT or throughput?* — both, via different mechanisms.
- *What happens to PP at low concurrency?* — the bubble dominates; it is the wrong tool.

**Red flags:** Calls PP "cheaper communication" with no mention of idle stages, or asserts PP is
strictly worse than TP.

---

#### T11-Q3 · Why does an MoE model need expert parallelism at all?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** DeepSeek V3 has 256 experts and GLM 5.1 has 256 experts across 78 layers `[T]`. Why
doesn't data parallelism handle this the way it handles a dense model?

**Model answer:** Because MoE breaks the assumption that makes DP work.

Data parallelism replicates the *whole model* on every device and splits the batch. That works when
the model fits on one device. An MoE model's total parameter count is far larger than its **active**
parameters per token — you have to store all 256 experts, but any given token only routes to a few
(GL.M 5.1 is top-k **8** `[T]`). So DP alone fails: you cannot fit the weights.

The naive fallback — tensor-parallelise the whole thing — is also wrong, because TP shards the
matrices you are *actually computing on*, and you are only computing on a handful of experts per
token. You would pay TP's all-reduce bill across weights that are idle.

Expert parallelism matches the sharding to the sparsity: put **different experts on different GPUs**,
and route tokens to the device that owns the expert they selected. EP8 on a 256-expert model gives
**32 experts per GPU**; EP32 across four nodes of 8 gives **8 experts per GPU** `[T]` ROCm/WideEP.
That is the whole trade in one line — higher EP frees VRAM per GPU (which you can spend on KV cache)
at the cost of more cross-node all-to-all traffic.

**Signal:** Frames it as a total-versus-active-parameter mismatch, not as "MoE is big so you shard
it." Names the EP arithmetic.

**Follow-ups:**
- *Why not just TP the MoE too?* — TP splits what you compute; you are not computing on most experts.
- *What does raising EP cost you?* — all-to-all volume; see T11-Q8 and T11-Q10.
- *What does the freed VRAM buy?* — KV cache, which is what makes long-context concurrency work.

**Red flags:** Says "MoE is large so you need more GPUs" without the active/total distinction, or
believes DP is impossible rather than insufficient.

---

#### T11-Q4 · What is WideEP, and what does each half of it handle?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** "WideEP" appears in the title of the ROCm talk. Decompose it. What is doing the work?

**Model answer:** WideEP is a division of labour: **data parallel attention + expert parallel MoE**
`[T]` ROCm/WideEP. The "wide" refers to the fact that these two mechanisms carry the dimension that
would otherwise be taken by tensor parallelism — so the configuration scales out across many GPUs
without paying TP's per-layer all-reduce across the fabric.

Attention is replicated by DP, which is embarrassingly parallel — no communication beyond what DP
already costs. The MoE layers are sharded by EP, which does pay all-to-all, but only for the experts a
token actually routes to. So you get a large effective deployment where the expensive collective
happens only where the sparsity justifies it.

The configuration that expresses this is deliberately counter-intuitive: `--tensor-parallel-size 1`
alongside `--data-parallel-size 16 --enable-expert-parallel` `[T]`. Anyone reflexively reaching for TP
on an MoE model is paying for communication that DP and EP would have carried more cheaply.

One important nuance: **that "TP = 1" is stated for the WideEP attention case and is easy to
over-generalise.** The same ROCm team's own target configuration runs **TP8 intra-node + 2P2D + EP8
("shallow EP") + DP16** `[T]`. The durable rule is *TP stays inside the node; EP and DP carry the wide
dimension* — not "TP is always 1."

**Signal:** States the split as DP-attention / EP-MoE, and — at L4 and above — volunteers the
intra-node-TP nuance without being prompted.

**Follow-ups:**
- *Why is DP fine for attention but not for experts?* — attention weights fit per replica; expert
  weights do not.
- *What are "shallow" and "wide" EP?* — EP8 intra-node versus EP32 across four nodes.
- *Correct the sentence "TP is always 1 on MoE."* — see T11-Q15.

**Red flags:** Describes WideEP as a vLLM flag rather than a decomposition, or repeats "TP = 1" as a
universal law.

---

#### T11-Q5 · What does sequence parallelism overlap, and why does it matter for prefill?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Where does sequence parallelism fit, and what specifically is it hiding?

**Model answer:** Sequence parallelism splits along the **sequence dimension** — the tokens of a long
prompt — and its function in a modern serving stack is to **overlap communication with compute** `[T]`.

The reason it matters is that prefill is the phase where a long prompt is a single large computational
block, and it is the phase that determines TTFT. If you shard that block across devices, the devices
must exchange activations, and if the exchange sits on the critical path you have converted a compute
problem into a communication stall. SP arranges the work so the communication happens while useful
arithmetic is still running on the device — the all-gather/reduce-scatter is scheduled underneath the
compute rather than gating it.

That is why SP appears specifically in the long-prefill, TTFT-bound recipe. In the corpus's tuned
prefill configuration, sequence parallelism is one of the **four named mechanisms** that let a
16-GPU-per-replica deployment beat naive 8-way TP, alongside pipeline parallelism (chunking the
prefill), expert parallelism (better GEMM shapes), and tensor parallelism staying at 2 rather than 8
`[T]` Kwon.

Note the honest caveat: the exact factorisation in that source is ASR-garbled. The four mechanisms and
the 16-GPU figure are reliable; a precise product of parallel degrees is not `[T]`.

**Signal:** Says "overlap communication with compute" and connects SP to the prefill/TTFT path rather
than listing it as one of seven equivalent options.

**Follow-ups:**
- *Why does prefill specifically need this?* — one large block; the exchange gates TTFT.
- *How does SP differ from context parallel?* — SP shards the sequence for the layer computation; CP
  shards it for attention; see T11-Q6.
- *What does "overlap" require of the implementation?* — chunked scheduling; the communication must be
  issued before its result is needed.

**Red flags:** Cannot distinguish SP from TP, or claims SP reduces memory rather than hiding latency.

---

#### T11-Q6 · Context parallel versus decode context parallel
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Two of vLLM's seven parallelism types both have "context" in the name. What separates
context parallel from decode context parallel?

**Model answer:** They both shard along the context dimension, but they do it at different layers of
the stack, and that difference determines which phase each one helps.

**Context parallel (CP)** splits the context across devices at the **attention level** — each device
holds a slice of the sequence and the attention computation is partitioned accordingly. It is a way to
make a very long sequence's attention tractable at all.

**Decode context parallel (DCP)** shards the context **around the KV cache** — it is a KV-level
partition `[T]`. That distinction matters because decode is the phase where the KV cache, not the
computation, is the binding constraint: for each generated token you read the whole cache, and at long
context that read dominates. Sharding the cache across devices lets those reads proceed in parallel.

So the practical pairing is: **CP for the very long sequence, DCP for long-context decode.** In the
cheat-sheet's own "when to use what" formulation, long-context decode points at DCP, and very long
context on a single sequence points at CP plus DCP together.

The reason this is an L4 question rather than a vocabulary test is that it forces a candidate to say
*which resource is binding*. If a candidate says "they both help long context," press them: they help
different phases and against different bottlenecks, and choosing wrong means paying communication for
no relief on the actual constraint.

**Signal:** Locates the difference at the attention level versus the KV level, and ties each to the
phase it helps.

**Follow-ups:**
- *Which one do you reach for when decode is the slow phase at 128k context?* — DCP.
- *Can you run both?* — yes, for very long context on a single sequence.
- *What does DCP cost?* — KV-level communication per step.

**Red flags:** Treats DCP as an alias for CP, or says "context parallel is for long context" and stops
there.

---

#### T11-Q7 · How many parallelism types does vLLM expose, and why does breadth matter?
**Difficulty:** L3 · **Depth expected:** 90 s

**Question:** Name vLLM's parallelism types. Why does having seven of them matter operationally?

**Model answer:** Seven: **tensor, pipeline, data, expert, sequence, context, and decode-context
(DCP)** `[T]` Kwon.

The breadth matters for two reasons. First, the corpus's central claim is that **"there's no universal
winner"** — the right plan depends on the model architecture, the cluster setup, and the workload
shape `[T]` Kwon. A serving stack that offers only TP and DP cannot express the configurations that
win on MoE or on long prefill.

Second, and more subtly for an operator: the combination space is large enough that configurations
interact in ways you cannot predict from first principles. The corpus's own 2P4D result was flagged by
the speaker as **preliminary — not to be trusted** `[T]`. That is a statement about the field's
maturity: when the expert who built the engine is unsure of a configuration, the operational
consequence is that **you benchmark your parallel plan before committing to it** rather than reasoning
your way to it.

**Signal:** Lists all seven accurately and connects breadth to the no-universal-winner principle
rather than treating the list as trivia.

**Follow-ups:**
- *Which single one would you drop last?* — DP; it is the outer layer everything else sits in.
- *Which are MoE-specific?* — EP, and its pairing with DP in WideEP.
- *Why can't you predict configuration interactions?* — benchmarking discipline; see T11-Q27.

**Red flags:** Recites four or five and invents the rest, or claims one configuration dominates
regardless of workload.

---

### Sizing and the sizing arithmetic

#### T11-Q8 · Sizing EP: 256 experts at EP8 versus EP32
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** You are serving a 256-expert MoE. Walk me through what changes between EP8 and EP32, and
tell me which you would choose for a long-context workload.

**Model answer:** The arithmetic itself is trivial — `experts_per_GPU = total_experts / EP_degree`.
GLM 5.1's 256 experts at **EP8 gives 32 experts per GPU**; at **EP32 across four nodes of eight GPUs,
8 experts per GPU** `[T]` ROCm/WideEP.

What that buys is VRAM. On an MI300 at **192 GB per GPU** or an MI355 at **288 GB per GPU** `[T]`, the
difference between 32 and 8 expert sets per device is a very large amount of memory, and the first
place that memory goes is the **KV cache**. For a long-context workload that is decisive: KV capacity
is what determines how many concurrent long sequences you can hold, and the corpus documents exactly
this failure — a throughput collapse at concurrency 256 driven by KV recomputation at **28k inputs**
`[T]` ROCm/WideEP. More KV headroom moves that cliff.

The cost is entirely in the network. Every MoE layer performs a **dispatch all-to-all and a combine
all-to-all**, so raising EP degree raises the collective's span — and EP32 crosses node boundaries that
EP8 does not. Past some EP degree, adding GPUs makes things worse because all-to-all dominates `[D]`.

So for long context I would go wide on EP only if the fabric can carry it, and I would **measure the
crossover**. The honest answer is that EP degree is a function of the interconnect, not of the expert
count.

**Signal:** Reaches the KV-capacity argument unprompted — that EP's real benefit for long context is
KV headroom, not compute — and names the network as the countervailing cost.

**Follow-ups:**
- *What is "shallow" versus "wide" EP here?* — EP8 intra-node versus EP32 across four nodes.
- *How would you find the crossover empirically?* — sweep EP degree against the same NIAH matrix.
- *What breaks first when you go too wide?* — the all-to-all fabric; see T11-Q18.

**Red flags:** Answers only the division and stops, or claims wide EP is free VRAM with no
communication consequence.

---

#### T11-Q9 · Choosing an EP degree for a cluster you did not design
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You inherit a cluster — N nodes, 8 GPUs each, a given fabric. You must serve a
256-expert MoE with long context. There is no benchmark budget for a full sweep. How do you choose?

**Model answer:** I would decide in this order, and I would be explicit that the first three steps are
arithmetic and the last is the only one that needs a measurement.

**Step one: what must fit.** Compute total weight bytes for the model, divide by device HBM (192 GB on
MI300, 288 GB on MI355 `[T]`), and find the minimum EP degree that leaves acceptable room for KV. That
is a floor, not a choice.

**Step two: where the node boundary falls.** `[D]` The jump from EP8 to EP16 or EP32 crosses from
intra-node to inter-node collective. Since every MoE layer pays a dispatch and a combine all-to-all,
the fabric topology — not the GPU count — sets the practical ceiling. The corpus's own team runs
**TP8 + EP8 + DP16 intra-node as "shallow EP"** and treats EP32 across four nodes as the wide case
`[T]`. That is a strong hint that EP8 intra-node is the safe default and widening is the experiment.

**Step three: what the workload actually needs.** Long context means KV is the binding constraint, so
EP degree is really a KV-headroom decision (see T11-Q8). Size to the concurrency you must hold at your
target context length, not to a round number.

**Step four, the only measurement:** run the **NIAH matrix** — 10 needles × concurrency × 3 shapes, 72
configurations, with a ≥7-needle pass threshold `[T]` — on two or three candidate EP degrees. That is
cheap, bounded, and it tests correctness under the exact condition you care about.

I would then commit to the smallest EP that meets the KV requirement without crossing the node
boundary, and keep a documented fallback.

**Signal:** Separates arithmetic constraints from the measurement, and refuses to answer the fabric
question from first principles — it is the one thing that must be measured.

**Follow-ups:**
- *Why EP8 as the default?* — it is the largest degree that stays intra-node on an 8-GPU node.
- *What is your fallback if the fabric saturates?* — reduce EP, spend the VRAM on KV discipline
  instead.
- *What would change your answer at 10× the concurrency?* — KV capacity becomes everything; see
  T11-Q28.

**Red flags:** Picks a degree because "32 is a nice number," or proposes a full sweep with no budget
for it, or ignores the node boundary entirely.

---

#### T11-Q10 · Derive the all-to-all volume
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Estimate the communication volume of one MoE layer's expert routing. What does the
answer tell you about what to optimise?

**Model answer:** Start from what moves. Each token is dispatched to the `top_k` experts selected for
it, and the results are combined back. So per layer `[D]`:

```
bytes ≈ tokens × top_k × hidden_dim × dtype_bytes × 2
```

The `top_k` factor is the routing fan-out — GLM 5.1 is top-k **8** `[T]` — and the `× 2` accounts for
both directions: the **dispatch** all-to-all that sends tokens to their experts and the **combine**
all-to-all that brings the results back `[T]`.

Two conclusions follow.

First, the volume scales with **tokens**, which means it scales with batch size and sequence length.
Prefill sends a whole prompt's worth of tokens through every MoE layer at once; decode sends one token
per sequence per step but must do it *every* step. That asymmetry is why prefill and decode have
genuinely different optimal parallel plans — and it is the reason prefill/decode disaggregation exists
at all.

Second, this is why the corpus states that **"distributed inference is becoming more of a
communication and memory problem, not a computation problem"** `[T]`. When a linear layer's cost is
smaller than the cost of moving its inputs and outputs, the fabric is your bottleneck.

For optimisation, the levers are: reduce `top_k` (model-dependent, usually not yours to change),
reduce `dtype_bytes` (quantisation), keep the collective intra-node, and — biggest of all — **fuse so
you launch fewer collectives**. The corpus's own kernel work took a naive MoE layer from **2 all-to-all
+ 6 kernels down to 3 kernels** `[T]`, with the 6-stage fused path covering top-k permute, grouped
GEMMs, unpermute and reduction/scale.

**Signal:** Writes the formula with both the `top_k` fan-out and the factor of two for dispatch plus
combine, and draws the prefill-versus-decode asymmetry out of it.

**Follow-ups:**
- *Why does decode hurt if it sends fewer tokens per step?* — it pays per step, every step.
- *What is the single biggest lever on the volume?* — dtype and collective count, not the topology.
- *How does this connect to disaggregation?* — different phases, different optimal plans; T12.

**Red flags:** Omits the combine direction, or concludes only "MoE is chatty" without extracting a
design consequence.

---

#### T11-Q11 · Lay out a 671B MoE across 64 GPUs
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** You have 64 GPUs across 8 nodes of 8, NVLink inside each node. Serve a 671B-parameter
MoE with 256 experts, top-k 8, for a long-context, latency-sensitive workload. Give me your plan and
tell me what you would need to measure before committing.

**Model answer:** I would structure this as constraints first, then a candidate plan, then the
measurements that could falsify it.

**Constraints.** 671B parameters is far beyond one device's HBM, so replication is off the table and
the weights must be sharded. With 256 experts, the natural axis is EP: EP8 gives 32 experts per GPU,
EP32 gives 8 `[T]`. 64 GPUs with 8 per node means EP8 is exactly one node — the largest EP degree that
never crosses a node boundary, which is the property I most want, because **every MoE layer does a
dispatch and a combine all-to-all** `[T]` and I want those intra-node.

**Candidate plan.** Eight replicas of EP8 — the outer layer is **DP** (replicas, embarrassingly
parallel), and within each replica the attention is DP'd while the MoE is EP'd. That is the WideEP
decomposition `[T]`. If 32 experts per GPU does not leave enough VRAM for the KV cache the long-context
workload demands, I widen to EP16 or EP32 across nodes and accept inter-node all-to-all — but I treat
that as a decision with a price, not a default.

**The prefill question.** If TTFT is the binding SLO, I would consider a disaggregated prefill pool and
add the mechanisms the corpus credits: **pipeline parallelism to chunk the long prefill, sequence
parallelism to overlap communication with compute, expert parallelism for GEMM shapes, and TP held
low** — the corpus's tuned configuration does this across 16 GPUs per replica and beats naive 8-way TP
`[T]` Kwon. I would not quote an exact factorisation; the source is ASR-garbled and only the four
mechanisms and the 16-GPU figure are reliable.

**What I would measure before committing.** The **NIAH matrix** — 10 needles × concurrency × 3 shapes,
72 configurations, ≥7-needle pass threshold `[T]` — at my target context length, plus TTFT and
throughput per GPU against the naive baseline. And I would hold the plan loosely: the field's own
experts flag preliminary configurations as not to be trusted `[T]`.

**Signal:** Leads with the node-boundary constraint rather than the GPU count, proposes DP+EP as the
outer structure, and volunteers both NIAH and a falsification criterion.

**Follow-ups:**
- *Why 8 replicas rather than one wide replica?* — DP is free; wide EP costs fabric.
- *When would you spend the node boundary?* — only when KV headroom demands it.
- *What SLO would push you to disaggregate?* — TTFT-bound long prefill; see T12.

**Red flags:** Picks TP because "671B is big," treats EP32 as strictly better than EP8, or commits
without naming a measurement.

---

#### T11-Q12 · Why is distributed inference a communication problem now?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** A transcript line says distributed inference "is becoming more of a communication and
memory problem, not a computation problem" `[T]`. Justify that claim.

**Model answer:** Because the arithmetic has become cheap relative to moving its inputs and outputs.

Three forces drive it. First, **MoE makes the weights sparse but the traffic dense**: you store 256
experts, compute on a handful per token, and pay a dispatch and combine all-to-all at every MoE layer
to do it `[T]`. The computation per token went down; the cross-device traffic did not.

Second, **attention is memory-bound at long context**. Decode reads the entire KV cache for every
generated token, which is why the corpus's cliff appears not as a compute stall but as a throughput
collapse at **28k inputs** where KV is recomputed every time `[T]`. Nothing about the FLOPs changed;
the memory behaviour did.

Third, **accelerator ratios have moved faster than interconnect ratios** `[D]`. Compute and HBM
bandwidth have scaled aggressively across generations; the fabric between nodes has not kept pace. So
the fraction of wall-clock time spent waiting on a collective rises even when nothing about your model
changed.

The practical consequences are exactly the ones the corpus's configurations express: keep TP
**intra-node** so the per-layer all-reduce stays on NVLink; use **DP + EP** so the wide dimension is
carried by mechanisms whose collectives are sparser; **fuse kernels** so you issue fewer launches (2
all-to-all + 6 kernels → 3 `[T]`); and **disaggregate** prefill from decode because the two phases have
different bottleneck shapes.

**Signal:** Gives at least two independent mechanisms — MoE traffic density and attention's
memory-bound decode — rather than restating the quote.

**Follow-ups:**
- *Which of those is specific to MoE?* — the all-to-all density; the KV point applies to dense models.
- *What does this imply for hardware selection?* — fabric and HBM capacity dominate the spec.
- *Where does it show up first in production?* — scaling past a node boundary; see T11-Q20.

**Red flags:** Agrees with the quote and paraphrases it, offering no mechanism.

---

### Configuration and tuning

#### T11-Q13 · A candidate proposes 8-way TP for DeepSeek prefill. Respond.
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** An engineer proposes deploying DeepSeek prefill on B200 with naive single-host 8-way
tensor parallelism. What does the corpus evidence say, and what do you propose instead?

**Model answer:** The corpus evidence says that is measurably the wrong choice. For DeepSeek prefill on
B200 in a disaggregated (prefill-only) pool, a naive single-host **8-way tensor parallel** deployment is
**worse** than a tuned mix deployed across **16 GPUs per model replica** — with much lower TTFT and much
higher throughput per GPU `[T]` Kwon.

The 8-way TP plan is not stupid; it is the default that emerges when you think of "sharding a model"
as one problem. Its flaw is that it pays an **all-reduce at every layer** to shard things that other
mechanisms shard more cheaply, and it does so on a phase — prefill — where a long prompt is a single
large block and latency is what you are being judged on.

The tuned alternative uses **four mechanisms** `[T]`:
- **Pipeline parallelism** to parallelise chunks of the long prefill sequence;
- **Sequence parallelism** to overlap communication with compute;
- **Expert parallelism** for better GEMM shapes in the MoE layers;
- **Tensor parallelism held at 2**, not 8.

Important epistemic caveat I would state out loud: **the exact factorisation in the source is
ASR-garbled.** The speaker's phrasing runs "two-way tensor parallel … two-way pipeline parallel …
tensor parallel plus sequence parallelism … across 16 GPUs." The reliable claims are the four
mechanisms and the 16-GPU figure; the load-bearing point is the **contrast with 8-way TP**, not the
precise product of degrees. I would not quote a factorisation I could not defend.

So my proposal: adopt the mixed plan as the hypothesis, and benchmark against the 8-way baseline on my
own hardware and workload shape, because — per the same corpus — **"there's no universal winner"** `[T]`.

**Signal:** States the contrast correctly, names at least three of the four mechanisms, and volunteers
the ASR caveat rather than reciting a factorisation.

**Follow-ups:**
- *Which mechanism is doing the latency work?* — PP chunking the prefill and SP overlapping comms.
- *Why does TP stay at 2?* — enough to fit, small enough to avoid the all-reduce bill.
- *Why is this a prefill-specific result?* — decode has a different bottleneck shape; T12.

**Red flags:** Accepts the 8-way plan because it is conventional, or quotes a precise factorisation
the source does not reliably support.

---

#### T11-Q14 · When does pipeline parallelism actually earn its keep?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** PP adds bubbles. Under what conditions is it nevertheless the right call?

**Model answer:** PP earns its keep in three distinct situations, and they are worth separating
because the justification differs each time.

**One: the model does not fit and TP's bill is too high.** PP's communication is point-to-point
between adjacent stages, which is far cheaper than an all-reduce per layer. If you must span nodes,
PP is the mechanism whose cost grows most slowly with span.

**Two: you are latency-bound on long prefill and want to chunk the sequence.** This is the least
obvious use and the one the corpus's tuned configuration exploits — pipeline parallelism
"parallelises chunks of the long prefill sequence" `[T]` Kwon. Here PP is not a memory workaround; it
is a latency technique.

**Three: concurrency is high enough to amortise the bubble.** The bubble fraction is approximately
`(PP_degree − 1) / (microbatches + PP_degree − 1)` `[D]`, so bubbles shrink as microbatches grow —
which means PP is fundamentally a **high-concurrency technique**. At low load, `microbatches` is small,
the fraction is large, and you have idled most of your hardware to no purpose `[D]`.

That third point is the one I would press a candidate on, because it is where PP goes wrong in
practice: teams adopt it at low traffic, see worse latency, and conclude PP is bad — when in fact they
did not have the concurrency PP requires. The failure is a mismatched assumption, not a bad mechanism.

**Signal:** Names the high-concurrency precondition without prompting, and offers the prefill-chunking
use in addition to the memory use.

**Follow-ups:**
- *Write the bubble fraction and interpret it.* — the formula above.
- *What happens if you deploy PP at 5 concurrent requests?* — the bubble dominates.
- *How does PP interact with disaggregation?* — it is a natural fit in a prefill pool with deep
  queues; T12.

**Red flags:** Calls PP simply "worse than TP," or recommends it for a low-traffic deployment.

---

#### T11-Q15 · Challenge the statement "TP stays at 1 on MoE"
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** The ROCm talk's configuration sets `--tensor-parallel-size 1`. A candidate concludes
"TP is always 1 on MoE models." Correct them.

**Model answer:** The statement over-generalises a case-specific setting, and the same talk's own
evidence refutes it.

`--tensor-parallel-size 1` belongs to the **WideEP attention case**: DP covers attention, EP covers
MoE, and TP is set to 1 because adding it would pay a per-layer all-reduce for a dimension DP already
handles for free `[T]` ROCm/WideEP. Within that configuration, TP = 1 is correct and deliberate.

But the **same ROCm team's target configuration** runs **TP8 + 2P2D + EP8 + DP16** `[T]` — TP is very
much in use, at degree 8. What distinguishes it is not whether TP is used but **where it stays**: TP8
is **intra-node**, so its all-reduce stays on NVLink rather than crossing the fabric. They call the
EP8 arrangement *shallow* EP and EP32 across four nodes the *wide* case `[T]`.

So the durable rule is: **TP stays inside the node; EP and DP carry the wide dimension.** The
inference a candidate should be able to make is *why*: TP's communication frequency (every layer)
makes its cost scale with span, so it is bounded by the interconnect's reach, not by a preference.

This matters practically because a team that believes "TP is always 1" will under-provision a
configuration that legitimately wants intra-node TP, and a team that believes "TP 8 is standard" will
pay a cross-node all-reduce on every layer. Both errors come from treating a heuristic as a law.

**Signal:** Names the specific configuration TP=1 belongs to, cites the TP8+EP8+DP16 counter-example,
and derives the intra-node rule rather than asserting it.

**Follow-ups:**
- *What is "shallow" versus "wide" EP?* — EP8 intra-node versus EP32 across four nodes.
- *Why is TP bounded by the node?* — all-reduce per layer; cost scales with span.
- *What would make you use intra-node TP on an MoE?* — fitting within the node when the weight
  footprint demands it.

**Red flags:** Defends "TP is always 1," or cannot name a case where TP is used on an MoE model.

---

#### T11-Q16 · Why does kernel fusion matter in the MoE layer?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** A naive MoE layer is "2 all-to-all + 6 kernels," fused down to 3 `[T]`. What is actually
being saved?

**Model answer:** Launch overhead and intermediate memory traffic.

The naive path decomposes the MoE layer into six kernel launches: the corpus describes the six-stage
fused MoE as covering **top-k permute → grouped GEMMs → unpermute → reduction/scale** `[T]`. Each
launch has fixed cost, and each boundary between kernels forces its intermediate tensor out to memory
and back. At MoE layer count — 78 layers in GLM 5.1 — those fixed costs and round trips are paid
dozens of times per forward pass.

Fusing collapses the six into a single fused MoE kernel, so the whole sequence becomes **dispatch
all-to-all, combine all-to-all, fused MoE — three kernels** `[T]`. The two all-to-all collectives
cannot be fused away because they are genuine cross-device communication; what is left is one
computational kernel instead of six.

The reason this belongs on a whiteboard rather than in a trivia round is the diagnostic value: if a
deployment is kernel-launch-bound, the signature is that GPU utilisation looks low while throughput is
also low — no single kernel is slow, there are just too many of them. The first check is whether the
fused path is actually in use `[T]`.

**Signal:** Names memory round trips and launch overhead rather than "fusing is faster," and knows the
two all-to-all collectives survive fusion.

**Follow-ups:**
- *Why can't the all-to-all be fused away?* — it is real inter-device communication.
- *What is the symptom of an unfused MoE path?* — low utilisation, poor throughput, no slow kernel.
- *Why does this matter more for MoE than dense?* — more layers and more collectives per layer.

**Red flags:** Says fusion removes the collectives, or treats 6→3 as a compression of data volume.

---

#### T11-Q17 · What does the "no universal winner" principle mean operationally?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** The corpus states there is no universal winner in parallelism choice `[T]`. What does
that mean for how a team should work?

**Model answer:** It is a claim about method, not just about configurations.

The full statement is that parallelism must be chosen per **model architecture, cluster setup, and
workload shape** `[T]` Kwon — three axes, all of which vary between deployments. The same model on a
different fabric can want a different plan; the same fabric serving a different request mix can want a
different plan again.

Operationally that implies four practices.

**One: treat a parallel plan as a hypothesis, not a default.** Every plan — including the ones in this
corpus — is a result measured somewhere, and should be re-measured in your environment.

**Two: budget for the measurement.** If you cannot sweep, use the bounded proxies: the NIAH matrix for
correctness under context, TTFT and throughput-per-GPU against a naive baseline, and a small EP-degree
sweep.

**Three: hold conclusions loosely, including your own.** The corpus's own 2P4D result was explicitly
flagged by the speaker as **preliminary — not to be trusted** `[T]`. When the person who built the
engine declines to stand behind a configuration, the correct professional response is to treat
configuration claims as provisional rather than to cite them as findings.

**Four: keep the reasoning, not just the answer.** A team should be able to say *why* its plan is what
it is — which constraint bound which choice — because when the model or the fleet changes they will
need to re-derive it rather than look it up.

The anti-pattern is a runbook of magic flags copied from a conference talk with no record of the
conditions under which they were measured.

**Signal:** Converts the principle into practice — hypothesis framing, measurement budget, provisional
claims — rather than agreeing with it.

**Follow-ups:**
- *What would you measure first if you had one hour?* — NIAH plus a TTFT/throughput baseline.
- *How do you record a configuration decision?* — with the constraint it satisfied and the conditions.
- *Does that mean published configurations are useless?* — no; they are priors, not conclusions.

**Red flags:** Treats the quote as a platitude, or proposes copying a published configuration directly
into production.

---

#### T11-Q18 · The all-to-all fabric is saturated. Diagnose.
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your MoE deployment's inter-node links are saturated and adding GPUs makes throughput
worse, not better. Walk me through the diagnosis.

**Model answer:** This is the characteristic failure of scaling past a collective's boundary, and the
first thing to establish is that it *is* that failure rather than something masquerading as it.

**Step one: confirm it is the collective.** Check the fabric counters — if inter-node links are near
saturation while intra-node NVLink is not, the traffic is crossing a boundary it would rather not
cross. Every MoE layer does a **dispatch and a combine all-to-all** `[T]`, so the volume scales with
the number of MoE layers, which is why this shows up on deep MoE models.

**Step two: find which degree crossed the boundary.** In order of likelihood `[D]`: **EP degree** (the
all-to-all's span), then **TP** (all-reduce every layer — if TP ever spans nodes, this is your answer
immediately), then **PP** (point-to-point, much less likely). The corpus's own arrangement is the
template: **TP8 + EP8 + DP16 intra-node**, with EP32 across four nodes as the deliberately "wide"
case `[T]`.

**Step three: reduce the span, then re-spend.** Drop EP to the largest degree that stays intra-node —
on 8-GPU nodes that is EP8. You lose expert-count headroom, which is VRAM, which for long context is
KV capacity: so pair the reduction with KV discipline rather than pretending there is no cost.

**Step four: reduce the volume rather than the span.** Quantise to cut `dtype_bytes` in
`tokens × top_k × hidden_dim × dtype_bytes × 2` `[D]`. Verify the **fused MoE kernel is in use**, since
the unfused path issues more collectives and launches `[T]`. And check whether the workload can be
disaggregated so prefill and decode stop sharing a fabric at cross purposes.

**Step five: verify against a counter-metric.** After reducing EP, re-run NIAH and confirm throughput
per GPU actually improved. If it did not, the diagnosis was wrong.

**Signal:** Orders the suspects correctly, proposes reducing span before adding hardware, and insists
on a confirming measurement.

**Follow-ups:**
- *Which degree is most likely guilty?* — EP; then TP if it spans nodes.
- *What is the cost of fixing it?* — KV headroom; see T11-Q8.
- *How do you know the fix worked?* — throughput per GPU, not aggregate throughput.

**Red flags:** Recommends adding bandwidth or GPUs without reducing span, or cannot say which
collective is at fault.

---

### Long context, failure modes and regression testing

#### T11-Q19 · Throughput collapses at ~28k context. What is happening?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your service runs fine, then throughput drops off a cliff — and the cliff is at
concurrency 256, with inputs around 28k tokens. Diagnose it.

**Model answer:** This is the corpus's documented cliff, and the mechanism is **KV recomputation** —
not compute, and not a bug.

At **28k inputs**, the KV cache for the in-flight sequences exceeds what the deployment can hold, so
KV is being **recomputed every time** rather than retained `[T]` ROCm/WideEP. The visible signature is
a throughput drop at concurrency 256 — the concurrency at which the working set crosses the capacity
line. Below it, everything is cached and the deployment looks healthy; above it, work is being redone
and throughput per unit of hardware falls.

The reason it presents as a throughput collapse rather than a memory error is that the system is
degrading gracefully: it keeps serving, it just pays for the same prefill repeatedly.

The fix space, in order of leverage `[D]`:

1. **Give KV more room.** Raise EP degree to shrink per-GPU expert footprint (EP32 gives 8 experts/GPU
   versus EP8's 32 `[T]`), if the fabric can carry the wider all-to-all. This is the direct trade from
   T11-Q8.
2. **Increase KV efficiency.** Quantised KV and better paging buy capacity without new hardware.
3. **Admit less work.** Bound concurrency below the cliff and queue, rather than accepting unbounded
   concurrency and recomputing. This is a deliberate SLO choice: predictable latency over maximum
   admitted load.
4. **Disaggregate.** Separate prefill from decode so the two phases stop competing for the same
   resource; see T12.

The critical point for an interview: the cliff is at a **specific context length and concurrency**,
and the number that matters is the pair, not either alone. A candidate who says "it's KV" without
connecting it to the capacity line has the word but not the model.

**Signal:** Names KV recomputation, gives the 28k/concurrency-256 pairing, and offers at least three
mitigations with their costs.

**Follow-ups:**
- *Why is it graceful rather than fatal?* — it recomputes instead of erroring.
- *Which fix would you try first?* — measure KV occupancy, then either capacity or admission control.
- *How do you find the cliff for your own workload?* — sweep concurrency at fixed context and watch
  throughput per GPU.

**Red flags:** Blames the fabric, the model, or a memory leak, or says "add GPUs" without explaining
what the GPUs would be spent on.

---

#### T11-Q20 · Tensor parallelism stops scaling past 8. Why?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Your TP degree goes from 8 to 16 and throughput barely moves, or regresses. Explain.

**Model answer:** Because 8 is typically where TP crosses the node boundary, and TP's cost structure
makes that crossing unusually expensive.

TP shards every layer's matrices, so it requires an **all-reduce on every layer** — not once per
forward pass, not once per step, but at every layer of the model `[T]`. Eight GPUs is the common
NVLink domain size, so a TP degree of 8 fits inside one node and its all-reduce travels over NVLink.
Push to 16 and the collective spans nodes, and now every layer pays a cross-node round trip.

Two things compound it. The first is simply that the fabric is slower and higher-latency than NVLink.
The second is latency *sensitivity*: an all-reduce is a synchronisation point, so its latency sits
directly on the critical path rather than overlapping with compute the way, say, a well-scheduled
sequence-parallel exchange can. That is why TP behaves qualitatively worse past the boundary rather
than merely degrading smoothly.

The fix is not to tune TP; it is to **change which mechanism carries the span** `[T]`:
- Let **DP** carry the outer dimension — replicas need no communication at all.
- Let **EP** carry the MoE sharding, where the collective is sparse and only covers routed experts.
- Use **PP** if you must cross nodes, because its point-to-point cost grows far more slowly with span.

And the direct evidence that this is the right instinct: the corpus's winning prefill configuration
holds **TP at 2** while deploying across 16 GPUs per replica, precisely so the all-reduce stays cheap
`[T]` Kwon.

**Signal:** Connects the GPU count to the node boundary rather than treating 8 as magic, and
identifies the all-reduce's per-layer frequency as the reason.

**Follow-ups:**
- *Why not just use a faster fabric?* — it helps, but the frequency stays per-layer.
- *What mechanism should carry the span?* — DP and EP; PP if crossing is unavoidable.
- *What does the corpus hold TP at in its tuned config?* — 2, across 16 GPUs.

**Red flags:** Attributes it to "diminishing returns," or proposes tuning TP degree rather than
replacing it.

---

#### T11-Q21 · Experts are unevenly loaded. What do you check?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Your MoE deployment shows some GPUs consistently busier than others. Walk through what
you would look at.

**Model answer:** Uneven expert load is a routing-balance problem, so the first artefact I want is an
**expert load histogram** — how many tokens each expert received over a representative window.

What it will show is one of three things, and they need different responses `[D]`:

**Load skew from the model's own routing.** Some experts are genuinely more popular, and with top-k
routing the imbalance is real rather than a bug. The mitigation is architectural — a higher EP degree
spreads experts more thinly across devices so that a hot expert's traffic is shared, at the cost of
all-to-all span (T11-Q8).

**Load skew from the input distribution.** The router's choices depend on what is being served, so a
workload shift — a new document source, a new tenant — changes which experts are hot. The signal is
that imbalance appears at a specific time rather than being stable.

**Load skew from a configuration asymmetry.** With EP across nodes, a device holding more hot experts
than its peers becomes a straggler in the all-to-all, and since the collective synchronises, the
slowest participant sets the pace. This is where uneven load converts directly into lost throughput.

I would also verify the fabric, because a device that *appears* overloaded may simply have worse links.
And I would check whether the imbalance correlates with throughput loss before optimising it — balanced
load is a means, not the goal.

One thing I would not do is try to influence routing from the serving layer: **MoE routing happens
inside the language model, not at the router** `[T]`. It is not a knob the serving stack owns.

**Signal:** Reaches for the histogram first, distinguishes stable skew from time-varying skew, and
knows routing is not a serving-layer lever.

**Follow-ups:**
- *Which fix is architectural?* — raising EP degree, if the fabric allows.
- *Why does one slow expert hurt everyone?* — the all-to-all synchronises; the straggler sets the pace.
- *Could you fix it at the router?* — no; routing is inside the LM `[T]`.

**Red flags:** Proposes changing routing policy from the serving layer, or treats any imbalance as a
bug to be fixed rather than a property to be managed.

---

#### T11-Q22 · MoE is much slower on AMD than NVIDIA. What is the honest answer?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** A team reports that their MoE workloads run materially slower on AMD than on NVIDIA with
equivalent hardware. What is going on, and what do you tell them?

**Model answer:** The technically honest answer is that this is an **enablement gap, not a hardware
verdict**, and the corpus says so explicitly.

The ROCm talk is titled around WideEP on vLLM and llm-d precisely because enabling MoE parallelism
efficiently on AMD is active work: the failure-signature table lists "MoE slow on AMD vs NVIDIA" with
the check being "WideEP enablement gaps / pending PRs" `[T]` ROCm/WideEP. Features land as pull
requests, and until one lands the path is simply not taken.

That framing matters because the naive reading — "AMD is slower at MoE" — is both wrong and expensive:
it forecloses an accelerator choice for reasons that may be transient. It also inverts the causality:
the corpus's own team reports the AMD path producing strong results, including **~20,000 max
concurrency on a 1P1D pair** and a **72-configuration NIAH matrix passing** `[T]` — so the capability
exists; it is the enablement timing that varies.

What I would tell the team concretely:
- **Check the upstream status** of the relevant WideEP work rather than benchmarking once and
  concluding. The check is named in the corpus `[T]`.
- **Confirm the fused path is in use**, since an unfused MoE is slower on any vendor.
- **Re-baseline after each release**, and record the software version with the result.

The generalisable lesson, and the reason this is an interview question at all: **a performance result
is a property of a stack, not of silicon.** A benchmark that does not name its software versions is
not a benchmark. That same principle is why the corpus's bring-up observation exists — new hardware
requires re-taking the whole stack from the ground up, and coding agents make that easier without
removing it `[T]`.

**Signal:** Frames it as enablement timing with a named check, cites AMD-side results to show the
capability is real, and refuses to make a vendor claim from a single measurement.

**Follow-ups:**
- *What would you do before recommending the platform switch?* — check upstream status; re-baseline
  per release.
- *What must accompany any benchmark you publish?* — software versions, config, workload shape.
- *How does this relate to bring-up effort?* — the stack is re-taken, not ported `[T]`.

**Red flags:** Concludes a vendor is slower on one run, or attributes it to hardware without checking
the enablement path.

---

#### T11-Q23 · Regression-testing a long-context configuration
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You change your parallel plan. How do you know you have not broken long-context
correctness?

**Model answer:** I would use the pattern the ROCm team already published, because it is bounded,
cheap and directly targeted: the **NIAH matrix**.

Concretely: **10 needles × concurrency × 3 shapes = 72 configurations**, all of which they report
passing `[T]` ROCm/WideEP. The structure is the valuable part:

- **10 needles** means the test is not pass/fail on finding one thing; it measures how much of a long
  context is still recoverable. The team's working threshold is a **≥7-needle pass** `[T]`, which gives
  a graded signal rather than a binary.
- **Concurrency is a dimension**, not a constant. This is the crucial design choice for serving work,
  because the documented KV cliff is a *concurrency-driven* phenomenon — throughput collapses at
  concurrency 256 with 28k inputs `[T]`. A correctness test run at concurrency 1 would never surface
  it.
- **3 shapes** vary the placement of the needles, so you are not testing one positional pattern.

The reason this belongs in the same suite as the performance test is that a parallelism change can
preserve latency and throughp ut while silently degrading retrieval from long context — for instance
if a context-parallel split mishandles attention at the boundaries. Performance and correctness have
to move together.

I would also record the configuration the result belongs to: NIAH passing on EP8 tells you nothing
about EP32 unless you say so.

**Signal:** Names the 72-configuration matrix, the ≥7-needle threshold, and — this is the
discriminator — explains *why* concurrency is one of the three axes.

**Follow-ups:**
- *Why include concurrency in a correctness test?* — the cliff is concurrency-driven.
- *What is the pass threshold and why graded?* — ≥7 of 10 needles; graded beats binary.
- *What else runs alongside?* — TTFT and throughput per GPU against the naive baseline.

**Red flags:** Proposes running a needle test at concurrency 1, or treats correctness and performance
as separate workstreams.

---

#### T11-Q24 · Should the serving layer influence expert routing?
**Difficulty:** L5 · **Depth expected:** 3 min

**Question:** A product team asks you to "route easier prompts to cheaper experts" from the gateway.
What do you tell them?

**Model answer:** I would tell them the request is architecturally incoherent as stated, and then
redirect it to where the same goal is achievable.

The incoherence is precise: **MoE routing happens inside the language model, not at the serving-router
layer** `[T]`. Expert selection is computed inside each MoE layer from the token representations
themselves, layer by layer, at inference time. There is no serving-layer interface to it, and the
corpus's own guidance is explicit: do not try to influence expert choice from llm-d `[T]`.

It is worth explaining *why* no such interface exists, because the reason is the same one that makes
MoE work: the router's decision is a function of hidden states you do not have at the gateway. The
gateway sees text; the expert choice is made on internal representations deep in the stack. Even if you
could patch it, you would be overriding a learned function with a heuristic.

What the gateway *can* legitimately do is exactly what T14 describes — decide **which model or which
replica** serves the request. That is the L0/L1/L2 routing decision, and the four-layer separation
matters here: MoE expert selection is the **L3** layer, and it is not operator-controlled.

So the productive redirection is: if the goal is cost reduction for easy traffic, that is a real and
achievable goal — but the lever is model-tier routing at the gateway, not expert steering. And I would
note the honest limit of the original idea: experts are not "cheaper" or "more expensive" in a
meaningful sense, since every token routes through the same weights on the same GPUs regardless of
which expert is chosen. The cost is in the all-to-all and the GEMMs, not in the expert identity.

**Signal:** Identifies the L3 layer as inside the model and out of operator control, and offers the
correct alternative lever rather than just refusing.

**Follow-ups:**
- *Where are the four routing layers separated?* — L0 gateway, L1 model, L2 replica, L3 expert; T14.
- *What is the achievable version of the request?* — difficulty-based model routing at the gateway.
- *Would a cheaper expert even save money?* — no; the cost is the collective and the GEMM.

**Red flags:** Says "yes, we can route to cheaper experts," or refuses without offering the
achievable alternative.

---

#### T11-Q25 · Benchmark methodology for a parallel-plan decision
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** You must recommend a parallel configuration to your team. How do you make sure your
benchmark is not misleading?

**Model answer:** Five rules, drawn from the failures the corpus itself exhibits.

**One: name the software stack.** A performance result is a property of the whole stack — engine
version, kernels, driver. The corpus's AMD-versus-NVIDIA question (T11-Q22) is unresolved precisely
because enablement lands as pending PRs `[T]`, so the same hardware gives different answers at
different times. Version-stamp every result.

**Two: report throughput *per GPU*, not aggregate.** Aggregate throughput always improves when you add
hardware, so it cannot distinguish a good plan from a wasteful one. The corpus's headline result is
specifically that the tuned plan gives "much higher throughput per GPU" `[T]` — that is the metric that
carries the argument.

**Three: fix the workload shape and state it.** Because parallelism must be chosen per model
architecture, cluster setup and workload shape `[T]`, a benchmark without a stated shape is not
reusable. Say the context length, the concurrency, and the prefill/decode mix.

**Four: prefer a counter-metric that can falsify you.** TTFT alone can be gamed by doing less work;
throughput alone ignores latency. Use both, plus NIAH for correctness (T11-Q23). If the plan is better,
it should be better on more than one axis.

**Five: flag results you do not trust — including your own.** The corpus's own 2P4D number was labelled
by the speaker as **preliminary, not to be trusted** `[T]`. That is the standard to hold yourself to:
if you are not confident, say so in the artefact rather than letting a number acquire authority it has
not earned.

**Signal:** Leads with version-stamping and per-GPU normalisation — the two errors that most often
invalidate published benchmarks — and holds their own result to the same scepticism.

**Follow-ups:**
- *Why per-GPU rather than aggregate?* — adding hardware always raises the aggregate.
- *What makes a result reusable?* — a stated workload shape and stack version.
- *What do you do with a result you half-trust?* — publish it with the caveat, not without it.

**Red flags:** Reports aggregate throughput, omits versions, or presents a preliminary number as
established.

---

#### T11-Q26 · What breaks first as MoE models keep growing?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Expert counts are climbing — Kimi K3 at 896 experts, DeepSeek V3 at 256, GLM 5.1 at 256
across 78 layers `[T]`. What is the first thing to break, and what do you do about it?

**Model answer:** I would argue the network breaks before the compute does, and I would defend that by
following the scaling through.

**The mechanism.** `experts_per_GPU = total_experts / EP_degree` `[T]`. Holding the device footprint
constant as expert count rises means EP degree must rise proportionally — Kimi K3's 896 experts need
roughly 3.5× the EP degree of a 256-expert model to keep the same experts-per-GPU. And since every MoE
layer performs a **dispatch and a combine all-to-all** `[T]`, raising EP degree raises the collective's
span and its cost. The model gets sparser in FLOPs and denser in traffic — the exact trend behind the
corpus's claim that distributed inference is becoming a communication and memory problem `[T]`.

**The second thing to break is memory.** Storing more experts means more weight bytes per device, which
competes directly with KV cache — and the corpus's documented cliff at **28k inputs** with KV
recomputation `[T]` is what losing that competition looks like.

**What I would do about it, in order `[D]`:**

1. **Keep the collective intra-node.** On 8-GPU nodes the ceiling is EP8; beyond that, the topology
   dictates the design. This is the single most protective constraint.
2. **Compress aggressively.** Quantisation cuts `dtype_bytes` in the all-to-all volume formula
   directly, and cuts weight footprint, which buys KV room. It attacks both bottlenecks at once.
3. **Fuse relentlessly** — 2 all-to-all + 6 kernels → 3 `[T]` — because at scale you pay per launch
   per layer and layer counts are growing too.
4. **Disaggregate**, so prefill and decode stop fighting over one fabric at cross purposes; the two
   phases have genuinely different optimal plans.
5. **Consider that the answer may be architectural rather than a tuning knob** — if the collective
   span is the binding constraint, the honest response is to size clusters by fabric topology and
   accept a lower experts-per-GPU ratio.

**Signal:** Derives the network conclusion from the EP arithmetic rather than asserting it, and names
memory as the second failure — then ties remedies to the two bottlenecks separately.

**Follow-ups:**
- *What does 896 experts do to your EP degree?* — roughly 3.5× a 256-expert model at fixed
  experts-per-GPU.
- *Which single lever attacks both bottlenecks?* — quantisation: dtype bytes and weight footprint.
- *When is the answer "buy different hardware"?* — when fabric topology, not GPU count, is binding.

**Red flags:** Says "just add GPUs," or discusses only compute scaling while ignoring the collective
and KV.

---

### Open design

#### T11-Q27 · Design the parallelism plan for a greenfield deployment
**Difficulty:** L5 · **Depth expected:** 8–10 min

**Question:** You are standing up serving for a new MoE model on hardware you have not used before. No
existing configuration applies. Design the process from zero to a committed plan.

**Model answer:** I would run this as four stages, and the discipline is that each stage's output is
what the next stage is allowed to assume.

**Stage 1 — constraints, by arithmetic.** Compute total weight bytes and divide by device HBM (192 GB
on MI300, 288 GB on MI355 `[T]`) to get the minimum sharding that fits. Then find the node boundary —
8 GPUs on a standard node — because it is the boundary that decides which mechanisms are cheap.
Output: a feasible region, not a configuration.

**Stage 2 — a hypothesis from priors.** Start from the corpus's decomposition: **DP for attention, EP
for MoE, TP small and intra-node** `[T]`. Pick the largest EP degree that stays inside a node, since
every MoE layer pays a dispatch and combine all-to-all. Estimate KV capacity from the remaining VRAM
and predict the context/concurrency at which you expect trouble — the corpus's template is a cliff at
**28k inputs** driven by KV recomputation `[T]`. Output: a defensible first configuration *and a
prediction of how it fails*.

**Stage 3 — measure, with bounded tests.** Run the **NIAH matrix** (10 needles × concurrency × 3
shapes = 72 configs, ≥7-needle threshold `[T]`) for correctness, and TTFT plus throughput-per-GPU
against a naive baseline. Sweep EP degree across two or three values only, since the fabric, not the
GPU count, is the variable. Output: a decision, with the falsifier stated.

**Stage 4 — commit with the reasoning attached.** Record the configuration, the constraint it
satisfied, the stack version, and the workload shape it was measured on. Hold it as provisional.

**Two things I would expect and plan for.** First, **bring-up on new hardware is a from-scratch
effort** — "the whole stack must be re-taken from the ground up," and coding agents make it easier
without removing it `[T]`. Second, **the plan will need revisiting** as the model or workload changes;
the corpus's own preliminary 2P4D result being flagged not-to-be-trusted `[T]` is the field's honest
state, and my documentation should be falsifiable rather than authoritative.

**Signal:** Structures as constraints → hypothesis → measurement → commit-with-reasoning, predicts the
failure mode before measuring, and budgets for bring-up effort rather than assuming portability.

**Follow-ups:**
- *What is your falsifier?* — NIAH plus per-GPU throughput against baseline; if neither improves, the
  hypothesis is wrong.
- *Why sweep EP and not TP?* — EP's span crosses the boundary; TP's cost is dominated by frequency.
- *What makes the committed plan safe to change later?* — recorded constraints; see T11-Q17.

**Red flags:** Proposes copying a published configuration, has no falsifier, or assumes the plan will
still hold when the model changes.

---

#### T11-Q28 · Revisit the plan when the workload changes
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** Six months on, your deployment is healthy, but the workload has shifted from short
requests to a mix with a heavy long-context prefill component. What breaks, and how do you re-plan?

**Model answer:** The plan does not become wrong; it becomes **optimal for a workload you no longer
have**. That distinction is the whole answer, and it follows directly from the corpus's three axes —
parallelism must be chosen per model architecture, cluster setup, **and workload shape** `[T]`.

What actually changes, mechanically:

**Prefill and decode stop sharing a plan.** Long prefill is a single large computational block where
TTFT dominates; decode is memory-bound on KV. A configuration tuned for a short-request mix is
tuned for the decode-heavy case. The corpus's tuned prefill recipe — **PP to chunk the long prefill,
SP to overlap communication with compute, EP for GEMM shapes, TP held at 2, across 16 GPUs per
replica** `[T]` — is a different plan from what the old mix wanted.

**The KV cliff moves into range.** Longer inputs mean the working set crosses the capacity line sooner;
the corpus's cliff at **28k inputs** with recomputation at concurrency 256 `[T]` is the failure to
watch for, and a long-context mix walks you toward it.

**So I would re-plan in this order.** First, re-state the workload shape explicitly — context
distribution, concurrency, prefill/decode ratio — because the old one is baked into the current plan.
Second, re-run the measurement suite on the new shape: the **NIAH matrix** (72 configurations,
≥7-needle threshold `[T]`) and per-GPU throughput. Third, decide whether the answer is a different
configuration or a **different topology** — and the most likely topology answer is **disaggregation**,
separating prefill and decode into pools with their own plans, because that is exactly the situation
disaggregation exists for.

The generalisable point I would want a candidate to reach: the trigger for re-planning is a change in
the **workload shape**, not a failure, and the professional habit is to have recorded the shape the
current plan was measured on so you can detect the divergence early.

**Signal:** Names the workload-shape axis as the reason the old plan is obsolete, predicts the KV
cliff, and reaches for disaggregation as the likely topology change.

**Follow-ups:**
- *Which phase's plan is now wrong?* — prefill; the tuned prefill recipe is the template.
- *What is the likely topology change?* — disaggregation into prefill and decode pools; see T12.
- *How would you have caught this earlier?* — by recording the measured workload shape with the plan.

**Red flags:** Re-tunes flags without revisiting the shape, or treats a healthy deployment as evidence
the plan is still right.

---

## Whiteboard exercises

### Exercise 1 — Size an MoE deployment from a specification
**Prompt.** "You have 32 GPUs across 4 nodes of 8 with NVLink inside each node, and 192 GB HBM per GPU.
Serve a 256-expert, top-k 8 MoE for a workload with a 32k-token p95 context and 128 concurrent
sessions, with TTFT under 2 seconds. Lay out the parallelism plan and show your arithmetic."

**What the candidate must produce:** the sharding arithmetic, an EP-degree choice defended against the
node boundary, a KV-capacity estimate, and an explicit statement of which assumption they would
measure first.

**Expected answer sketch:**

```
Constraints
  node boundary        = 8 GPUs (NVLink inside)          [T]
  EP8  -> 256/8 = 32 experts/GPU
  EP16 -> 256/16 = 16 experts/GPU   (crosses node boundary)
  EP32 -> 256/32 =  8 experts/GPU   (2 node hops)
  HBM 192 GB/GPU                                        [T]

Candidate plan (start narrow, widen only if KV demands)
  DP = 4 replicas        (embarrassingly parallel, no comms)
  EP = 8 per replica     (largest degree inside one node)
  TP = 1..2 intra-node   (do NOT let TP span the boundary)
  MoE layer comms: dispatch A2A + combine A2A per layer

KV estimate
  weights/replica = total_params/8 + overhead -> subtract from 192 GB
  remainder -> KV pool -> max_concurrent x 32k tokens
  predict the cliff: corpus template = 28k inputs @ conc 256   [T]

First measurement
  NIAH 10 needles x concurrency x 3 shapes = 72 configs, >=7 pass [T]
  + TTFT and throughput-per-GPU vs naive 8-way-TP baseline
```

**Grading rubric (full marks requires all four):**
- Places EP8 deliberately because it is the largest intra-node degree, and says so — not because 8 is
  round.
- Shows `experts_per_GPU = total_experts / EP_degree` explicitly and computes at least two candidates.
- Produces a KV estimate that predicts *where* the deployment will break, citing the 28k-input cliff
  as the template.
- Names a measurement that could falsify the plan, and states it before being asked.

---

### Exercise 2 — Diagnose a throughput cliff
**Prompt.** "Your MoE service was healthy at 100 concurrent sessions with 16k contexts. Since a
marketing push, traffic is at 300 concurrent sessions with 30k contexts. Throughput per GPU has fallen
by roughly half and GPU utilisation looks low. The fabric counters are normal. Diagnose and fix."

**What the candidate must produce:** a ranked differential diagnosis, a discriminating check for each
candidate, and a fix with its cost stated.

**Expected answer sketch:**

```
Symptom: throughput/GPU down, utilisation LOW, fabric NORMAL

Ranked candidates                      Discriminating check
1. KV recomputation (the cliff)   -->  KV cache occupancy vs capacity;
   [T] 28k inputs / conc 256           is prefill being redone per request?
2. Unfused MoE path               -->  kernel trace: 6 kernels vs 3 per MoE layer
   [T] 2 A2A + 6 kernels -> 3          is the fused kernel in use?
3. Admission above the knee       -->  queue depth vs admitted concurrency
4. Expert load skew               -->  expert load histogram; straggler in A2A

Fix, with cost
  Give KV room:  raise EP (EP8 -> EP16/EP32)  COST: cross-node A2A   [T]
  Or bound admission below the cliff          COST: queueing latency
  Or quantise KV                              COST: accuracy regression risk
  Then RE-MEASURE NIAH + per-GPU throughput to confirm the diagnosis
```

**Grading rubric:**
- Puts KV recomputation first and cites the 28k/256 pairing as the template rather than as a fact about
  this deployment.
- Notices that fabric counters being normal is evidence *against* the all-to-all hypothesis — a
  discriminator, not a detail.
- Explains why low utilisation co-occurs with low throughput (work is being repeated, not stalled).
- States the cost of each fix, and insists on a confirming measurement afterwards.

---

### Exercise 3 — Defend a parallel plan to a sceptical reviewer
**Prompt.** "A staff engineer says: 'You've chosen TP=2, PP=2, SP, EP and DP across 16 GPUs per
replica for prefill. That's over-engineered. Naive 8-way TP on one host is simpler and we already know
it works.' Argue your case, and tell me what would change your mind."

**What the candidate must produce:** the evidence, the mechanism behind each choice, an honest
statement of what is uncertain, and a falsification condition.

**Expected answer sketch:**

```
The claim under test
  For DeepSeek prefill on B200 in a disaggregated prefill pool:
  naive single-host 8-way TP  <  tuned mix across 16 GPUs/replica
  (much lower TTFT, much higher throughput per GPU)          [T] Kwon

WHY each mechanism earns its place                          [T]
  PP  -> parallelises CHUNKS of the long prefill sequence
  SP  -> overlaps communication with compute
  EP  -> better GEMM shapes in the MoE layers
  TP  -> held at 2, NOT 8: the per-layer all-reduce is the cost

What I will NOT claim
  the exact factorisation -- the source is ASR-garbled.
  Reliable: the 4 mechanisms + the 16-GPU figure.
  Load-bearing: the CONTRAST with 8-way TP.                 [T]

What would change my mind
  my own benchmark: per-GPU throughput and TTFT, same shape,
  vs an 8-way-TP baseline. There is "no universal winner"    [T]
  -- including for my own configuration.
```

**Grading rubric:**
- States the contrast result precisely and attributes it to the speaker and talk.
- Gives a mechanism for at least three of the four mechanisms, not just a list.
- Volunteers the ASR caveat and refuses to quote an unsupported factorisation — this is the
  highest-value discriminator in the exercise.
- Commits to a falsification condition on their own plan, consistent with the no-universal-winner
  principle.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the seven parallelism types; Kimi K3 at 896 experts, DeepSeek V3 and GLM 5.1 at 256 experts with
  GLM 5.1's 78 layers, top-k 8 and MLA; EP8/EP32 expert-per-GPU arithmetic; MI300 at 192 GB and MI355
  at 288 GB; the 2-all-to-all-plus-6-kernels to 3-kernel fusion; the six-stage fused MoE; the
  72-configuration NIAH matrix with a ≥7-needle threshold; ~20,000 max concurrency on a 1P1D pair; the
  28k-input KV-recomputation cliff at concurrency 256; TP8 + 2P2D + EP8 + DP16 with the shallow/wide EP
  distinction; and the AMD-versus-NVIDIA enablement-gap signature.
- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
  — the seven parallelism types, the tuned 16-GPU-per-replica prefill configuration against naive
  8-way TP with its four mechanisms and the ASR caveat, and "there's no universal winner" with the
  three axes of model architecture, cluster setup and workload shape.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — MoE routing happening inside the language model and the instruction not to influence expert choice
  from llm-d, and the preliminary 2P4D result flagged as not to be trusted.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
  — the distributed-inference-as-communication-and-memory framing, kernel fusion, and the bring-up
  effort for new hardware.
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md` `[R]` —
  the communication-volume and bubble formulas, and the intra-node TP / wide EP topology guidance.

**Derived content in this bank (`[D]`):** the bubble-fraction and all-to-all volume formulas; the
step ordering in the diagnosis questions (T11-Q18, Q19, Q28); the ranking of suspects in T11-Q18; the
"compute has scaled faster than interconnect" reasoning in T11-Q12; and the four-stage planning process
in T11-Q27. All are labelled inline where they appear.
