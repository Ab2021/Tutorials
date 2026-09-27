# Interview Bank: Prefill/Decode Disaggregation & KV Transfer

> `T12` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T12-disaggregation-kv-transfer.md) · [Case study](../01-case-studies/T12-disaggregation-kv-transfer.md) · [Design blueprint](../03-design-blueprints/T12-disaggregation-kv-transfer/HLD.md)
> **Questions:** 30 (9 × L3, 14 × L4, 7 × L5) · **Format:** progressing from why the split exists, through the transfer plane and the retention lifecycle, to sizing, failure diagnosis and open design

## How to use this bank

The questions are ordered so the bank reads as one interview: the two hardware profiles first, then the
transfer plane, then what the router has to know, then ratio arithmetic, then the session lifecycle,
then failure diagnosis and design. Ask them in order for a 45-minute loop, or sample the L4/L5 block
for a senior/staff screen.

Every number here is traceable to a transcript (`[T]`, speaker and talk named), a supporting repo
(`[R]`, path named), or is my own derivation (`[D]`, assumptions shown inline). Vendor claims are
marked as vendor claims. If a candidate quotes a figure, ask where it came from — a candidate who
cannot attribute a number does not own it, and in this topic that discipline is not academic: the
corpus's headline topology result was flagged preliminary by the speaker who produced it.

**ASR caution running through this bank.** Several proper nouns in the transcripts are garbled, and the
corrections are load-bearing. `NIXL` appears as "NVIDIA and Excel" and as "P2P K"; `MoRI-O` appears as
"Mori" and "MOI IO"; `MI355X` as "M355"; `SGLang` as "Ashlan"; `LiteLLM` as "LightLLM"; "harness" as
"honey". I use the corrected names throughout and flag the correction where the original matters. Where
a source garbles a precise configuration factorisation, I say the figure is reliable and the
factorisation is not, rather than quoting one.

---

### Fundamentals — the two hardware profiles, and what actually gets split

#### T12-Q1 · Why does prefill/decode disaggregation exist at all?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** A colleague says "disaggregation is just the current fashion, it's the same engine split
in two." Explain what problem it actually solves.

**Model answer:** It exists because prefill and decode are two different workloads at the hardware
level, not two modes of one workload. Peter DeSantis states it most cleanly: under an autoregressive
transformer there are two phases — prefill/encoding, which is "extremely compute-intensive", and
auto-regressive token generation, which is "extremely memory-bandwidth [bound], because each subsequent
token requires us to access every model weight" `[T]` DeSantis, *Constraint Driven Innovation*. His
conclusion is the one to quote: "the profile of those two workloads is radically different if you look
at it at a hardware level" `[T]`.

That mismatch is why colocation is structurally awkward. Prefill wants large batches of prompt tokens
driven at compute peak; decode wants large *sequence* batches and is bound by bytes of weights read per
token at HBM bandwidth. No single batch shape is optimal for both, and the optimum moves with the
traffic mix. Colocated, each phase starves the other.

The workload drift makes it acute and measurable. Roughly **~70% of inference traffic is now agentic**
`[T]` Pravin, *Scaling Agentic AI with llm-d*, and within that traffic **prefill is ~98% of the tokens**
`[T]` Pravin, because an agent reads code and documents while emitting comparatively few output tokens.
So the bottleneck moved to prefill while decode capacity sits idle — the case study's own scenario has
TTFT p95 going 400 ms → 4 s while ITL barely moves.

Disaggregation gives each phase its own pool, independently scaled, at the cost of moving the KV cache
between them. And because the mismatch is silicon-level rather than software-level, the split outlives
any particular chip.

**Signal:** Names compute-bound versus memory-bandwidth-bound *and* the independent-scaling payoff — a
weak answer says "it's faster" without saying which phase was being starved by which.

**Follow-ups:**
- *What are the two phases at hardware level?* — DeSantis's framing and the SRAM-chip argument.
- *What does the split cost you?* — a KV transfer on every request plus a second failure domain; T12-Q11.
- *Why not just make prefill faster on the same silicon?* — the coupling, not the speed, is the problem;
  see T12-Q17 and the whiteboard exercises.

**Red flags:** Describes disaggregation as a vLLM feature rather than an architectural response to a
hardware asymmetry, or claims it is unconditionally faster.

---

#### T12-Q2 · Walk me through one request in a disaggregated deployment
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Trace a single agentic request from arrival to first token, naming what decides each hop.

**Model answer:** Six steps, and the last two are the ones people omit.

1. **The router picks a prefill pod.** llm-d's EPP filters to prefill instances, then scores on
   prefix-cache identity and token load `[T]` Pravin.
2. **Prefill computes attention over the full prompt and materialises the KV blocks.** In agentic
   traffic this is where ~98% of the tokens are spent `[T]` Pravin.
3. **The prefill worker writes the KV blocks into the transfer engine.** On NVIDIA that is NIXL; on AMD
   it is MoRI-O, which supports both write and read modes `[T]` Chaitanya, *ROCm with WideEP*.
4. **Decode reads the blocks and begins generating.** It is scored on **active requests**, not on prefix
   cache — "the decode requests just need active request scorer because they don't pull the KV cache
   from the prefill" `[T]` Pravin. That asymmetry in scorers is a real design detail, not trivia.
5. **If the session pauses** — a tool call, a human wait — the KV blocks are evicted to CPU DRAM and
   later to an external/pooled tier, tagged with session metadata so the router knows which blocks
   belong to which session `[T]` Pravin.
6. **On resume the blocks are restored rather than recomputed.** CPU offload is reported to save about
   **~5× in TTFT** when an agentic session comes back after a pause `[T]` Pravin.

Steps 5–6 are where the economics live. Disaggregation *without* a retention tier is a latency
optimisation; disaggregation *with* one is what makes multi-turn agentic sessions affordable at all.

**Signal:** Volunteers the retention step without prompting, and distinguishes the prefill-side scorer
from the decode-side one.

**Follow-ups:**
- *Why does decode use a different scorer?* — it isn't pulling from the prefix cache; active-request
  load is the useful signal.
- *What happens if step 5 is missing?* — every resume re-prefills; see T12-Q20 and Q22.
- *What happens if step 3's agent dies?* — decode waits, then recomputes; see T12-Q27.

**Red flags:** Stops at "prefill sends KV to decode," or believes decode routes on prefix-cache
affinity the way prefill does.

---

#### T12-Q3 · What is NIXL, and where does it sit in the stack?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** "We'll move the KV over NIXL." What is NIXL, and what layer of the system is it?

**Model answer:** NIXL is the **transfer library** — not a topology. The corpus's repo describes it as
"a transport layer for moving inference state across memory and network backends" `[R]`
`refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md`. In the llm-d
deployment it is the thing the KV cache is "transferred between the pre-fill and the decode nodes
using" `[T]` Pravin, *Scaling Agentic AI with llm-d* — where the ASR renders NIXL as "NVIDIA and Excel",
and the same talk separately describes a "P2P K" that "transfers the KV cache between one node to the
other node" `[T]`. Both are corrected to NIXL; the correction is worth stating out loud because a
candidate who repeats the ASR string has clearly not read the source.

Where it sits: **behind the engine's KV connector abstraction**, alongside MoRI-O on AMD and
third-party stores like Mooncake `[T]` Kwon, *vLLM: Building Open and Efficient Inference for Agents*.
That placement is the point — it means you can change transports without redesigning the pools, and it
means a heterogeneous fleet is reachable through one interface (T12-Q4).

One sourcing note I would state if asked where this came from: `TOPICS.md` attributes NIXL and
KV-transfer material to `11-infrastructure-and-mlops/01-llm-infrastructure.md`. That file contains no
NIXL and no KV-transfer content — it covers deployment options, scaling, cost and the accelerator
landscape. The reliable NIXL sources are the repo above `[R]` and the llm-d transcript `[T]`, not that
path.

**Signal:** Says "transport library, not a topology," places it behind the connector abstraction, and
notes the ASR correction rather than repeating it.

**Follow-ups:**
- *What's the AMD equivalent?* — MoRI-O, with write and read modes; T12-Q9.
- *What is the connector abstraction for?* — T12-Q4.
- *What does NIXL run over?* — the fabric ranking; T12-Q8.

**Red flags:** Calls NIXL a connector, a cache, or a topology; cites the mis-attributed
`01-llm-infrastructure.md` path as the NIXL source.

---

#### T12-Q4 · What does the KV connector abstraction actually buy you?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Why did vLLM introduce a connector abstraction instead of just calling NIXL directly?

**Model answer:** Because the movement of KV is not one movement. Kwon's framing: "in the prefill
disaggregation case, the movement of KV cache is pretty dynamic and complex because it needs to move
between prefill instance to decode instance, prefill instance to the distributed KV storage pool like
Mooncake" `[T]` Kwon. That is a genuinely multi-directional problem, and it also has to cover the
idle-cache case: the connector "allows to leave this idle KV cache, store this idle KV cache to
external memory like CPU memory or disk, and bring it back when it's needed" `[T]` Kwon.

So the abstraction gives you three things:

1. **Transport independence.** "We did a lot of efforts to design this abstraction and make sure it is
   well working with third party libraries like Mooncake and also well working with other KV transfer
   mechanism like prefill disaggregation" `[T]` Kwon. NIXL-class P2P transfer and a third-party store
   sit behind one interface.
2. **A heterogeneous fleet.** The case study's fleet is NVIDIA-first with an AMD half; the same
   interface reaches NIXL today and MoRI-O later.
3. **One place to implement retention.** The CPU/DRAM/external offload path is the same connector
   machinery as the PD transfer `[D]`.

The cost is real and worth naming: a lowest-common-denominator API, with per-transport quirks leaking
through — and it is explicitly *not* a compatibility guarantee. Engine version skew between pools
produces silently corrupted output, not a crash (T12-Q29).

**Signal:** Names the multiple movement directions as the motivation, and recognises that the retention
path and the PD path share the same machinery.

**Follow-ups:**
- *What leaks through the abstraction?* — transport-specific features it cannot express.
- *Is it a compatibility guarantee across engine versions?* — no; T12-Q29.
- *Which transport would you pick first on an NVIDIA fleet?* — NIXL; case study chooses the abstraction
  over NIXL so the AMD half stays reachable.

**Red flags:** Treats the connector as a synonym for NIXL, or describes it as an optimisation rather
than an interface.

---

### The evolution ladder and the third phase

#### T12-Q5 · "Disaggregation" is not one thing. Where does it start?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** A team says they have "gone disaggregated". What are the possible meanings, and does the
distinction matter operationally?

**Model answer:** It matters a great deal, because the four stages have very different cost and risk
profiles. Zhu gives the ladder directly `[T]` Zhu, *SGLang and Miles*:

```
collocate → prefill/decode split → prefill/decode disaggregation → EPD
```

The **first** step is a *logical* split on shared hardware; the **third** is a physical one across
nodes; the **fourth** adds a third phase for vision encoders. So disaggregation is a continuum, not a
binary.

The middle rung is the one candidates skip. The single-node **4 GPUs prefill + 4 GPUs decode**
configuration is a legitimate production deployment — "if you want you could do a node. So four GPUs
run let's say prefill and four GPUs run decode" `[T]` Singh, *NxtGen*. It buys phase isolation, which is
exactly the contention problem T12-Q1 describes, *without* a network hop on the request path and
without a second node to operate.

Why that ordering is the professional one: the most common migration failure is jumping to the physical
split, paying a network hop on every request, before establishing that the workload is asymmetric enough
and the load high enough to amortise it. The continuity also has a measured price point — the llm-d
study found ~2× throughput above roughly **85 QPS**, with below that the split adding cost without
buying utilisation `[T]` Singh (vendor-affiliated; T12-Q18).

**Signal:** Names the ladder and treats the in-node 4+4 split as a legitimate stage rather than a
compromise, then connects the stages to the crossover.

**Follow-ups:**
- *What does the in-node split not give you?* — independent scaling across nodes and true failure
  isolation.
- *What is the price of moving to stage three?* — a transfer on every request plus a second failure
  domain; T12-Q11.
- *What triggers stage four?* — multimodal traffic; T12-Q6.

**Red flags:** Treats disaggregation as a binary, or recommends a cross-node split at low QPS.

---

#### T12-Q6 · EPD — when does the encoder get its own pool?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** You are adding image inputs to a text-only agent. Does that change your pool topology?

**Model answer:** Probably yes, and the reason is that the encoder is a third *workload profile*, not
just a third stage. Zhu's ladder ends at EPD: "now for vision language model, there are also like EPD,
where you also disaggregate encoder in this case" `[T]` Zhu.

The justification is the same physics as PD, applied to a third phase: image and video encoding is
neither the compute-bound large matmul of prefill nor the bandwidth-bound token loop of decode, so it
contends with both for the wrong reason. The operational problem is worse than the contention, though —
if the encoder is *not* disaggregated, it lands on the prefill pool "and its cost is invisible in the
text-token metrics" `[T]`/`[D]`. Your prefill pool looks mysteriously slow and your dashboards cannot
show you why, because the metric you are looking at counts text tokens.

There is a second-order point worth volunteering. DeSantis argues that phase-specialised silicon —
SRAM-heavy decode chips that trade compute transistors for memory — will earn a place, while warning
those chips are poor at prefill *and* poor at moving long context: "if you're running agentic AI, you
probably need really, really long context windows, and it's hard to move that context in and out of an
SRAM chip" `[T]`. So hardware specialisation *raises* the stakes on the transfer plane rather than
lowering them. Disaggregation is not a way to avoid moving KV; it is a commitment to it.

Sequencing caution: the case study schedules EPD as phase 2 and states why — "adding a third
disaggregation axis before the first is stable is how migrations fail."

**Signal:** Identifies the encoder as a third resource profile and names the metrics-invisibility
failure, rather than just reciting EPD as a three-letter acronym.

**Follow-ups:**
- *Why not just give prefill more capacity?* — an encoder's cost is not visible in text metrics, so you
  cannot size it from the data you have.
- *Does specialised silicon reduce transfer pressure?* — no; DeSantis's long-context point.
- *What's the sequencing rule?* — stabilise the first axis before adding the second.

**Red flags:** Says "EPD is just PD with images," or does not know what the third stage is for.

---

### The transfer plane — moving KV between two independently scaled pools

#### T12-Q7 · Derive the KV transfer volume for one request
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You are about to disaggregate. Before you design anything, tell me how many bytes cross
the wire for a typical request, and show me the arithmetic.

**Model answer:** The formula, from the cheat sheet `[D]`:

```
transfer_bytes = bytes_per_token × prompt_len
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

The factor of 2 is for K **and** V; `n_kv_heads` rather than `n_heads` because grouped-query attention
is what makes the number survivable.

Worked example, with the assumption stated: for **Llama-3.1-70B at fp16 the figure is ~327 KB/token**,
so a **28k-token prompt moves ≈ 9 GB** before the first decode token is produced `[D]`. Arithmetic:
327 KB × 28,000 = 9.156 × 10⁹ B ≈ 9.16 GB. The assumption is that 327 KB/token is a property of that
model's attention configuration at fp16 — layers, KV heads, head dim — not of the hardware or the
engine.

The honest boundary: **the corpus does not state a KV transfer byte figure.** The case study says so
explicitly — "Not measured in this corpus: KV transfer bytes per token, transfer latency in
microseconds, and any head-to-head of NIXL versus MoRI-O." So the right answer gives the *method* as
the deliverable, offers the worked example clearly labelled as a derivation, and refuses to quote a
measured number that does not exist.

Why it is the first thing to compute: it decides whether PD disaggregation is viable on your fabric at
all `[D]`. `transfer_time = transfer_bytes / fabric_BW`, and if that is a large fraction of the prefill
it replaces, the split cannot pay — which is also why the same deployment is fine at 8k context and
broken at 28k (T12-Q13, Q27).

**Signal:** Writes the formula including `n_kv_heads` and the factor of 2, states the derivation
assumption out loud, and volunteers that the byte figure is not measured in this corpus.

**Follow-ups:**
- *What halves it?* — GQA/MQA (fewer KV heads) and quantised KV.
- *What changes with MLA?* — compressed latent cache; a different bytes/token entirely.
- *Why does 28k keep appearing in this topic?* — it is the corpus's documented capacity cliff.

**Red flags:** Quotes a KV byte figure as if it were measured, forgets the factor of 2, or treats
transfer volume as independent of prompt length.

---

#### T12-Q8 · Rank the transports, and justify the ranking
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** You have a choice of how the KV crosses between pools. Rank your options.

**Model answer:** **Pooled memory < RDMA << TCP/IP** `[T]` Jongryool Kim, *Disaggregated LLM Serving with
Shared Memory KV Cache at Rack Scale*: "Instead of using TCP/IP we generally use RDMA. This is fast, but
this pooled-memory-based data sharing is faster than RDMA" `[T]`.

Two caveats to say in the same breath, because they are what separates a candidate who read the talk
from one who read a slide:

- **This is a vendor claim.** Kim's talk is SK hynix describing SK hynix hardware, and he described the
  comparison numbers as "very initial performance numbers," against a baseline of Mooncake and a DRAM
  cache `[T]`. Treat the ordering as a vendor claim, not a benchmark.
- **The ranking is a statement about the denominator.** Everything reduces to
  `transfer_time = transfer_bytes / fabric_BW` `[D]`, so "pooled memory is fastest" is really "pooled
  memory gives the largest usable bandwidth and the shortest path." Pooled memory also removes GPU-side
  PCIe and network contention from store-and-reuse — a second, independent benefit `[T]` Kim.

Operationally: on NVLink/NVSwitch the break-even is easy; over TCP/IP it is usually a loss; over pooled
memory it is the cheapest `[T]`/`[D]`. The cheat sheet's rule is blunt and correct — if your fabric
bandwidth is unknown, compute `KV_bytes / BW` before designing anything, and if you have TCP/IP only,
probably do not split.

**Signal:** Gives the ranking *with* the vendor-claim caveat and connects it to the bandwidth
denominator rather than treating it as a league table.

**Follow-ups:**
- *What's the second benefit of pooled memory besides latency?* — contention removal, plus OOM
  resilience; T12-Q25.
- *What do you do on a TCP/IP-only fabric?* — measure first; probably do not split.
- *What does NIXL run over?* — RDMA-class transports; it is the library, not the medium.

**Red flags:** Presents the pooled-memory advantage as an established benchmark, or ranks transports
without reference to bandwidth.

---

#### T12-Q9 · Write mode or read mode — who blocks when the other side is slow?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** MoRI-O supports both a write mode and a read mode for KV transfer. Why would you care
which one you use?

**Model answer:** Because the choice decides **which pool absorbs backpressure** — and one of your two
pools is the scarce resource. MoRI-O is "a KV cache transfer engine … a GPU to GPU, GPU-initiated KV
transfer" with "both support mode, like write mode as well as read mode" `[T]` Chaitanya, *ROCm with
WideEP*.

The semantics:

| Mode | Who initiates | Who blocks | Failure surfaces |
|---|---|---|---|
| **Write** (prefill pushes) | Prefill | Prefill waits on a slow decode | At the sender |
| **Read** (decode pulls) | Decode | Decode tracks availability itself | At the receiver |

Write mode's appeal is control: prefill owns the timing and gets a natural completion signal. Its cost
is that under decode-side backpressure **prefill stalls** — and prefill is the scarce resource in a
98%-prefill agentic workload. Read mode's appeal is that the backpressure lands on the correct side:
decode controls its own admission. Its cost is more state on the decode side, and the risk of
over-committing reads and thrashing.

So the rule is: **default to read when prefill is the bottleneck**, which is precisely the agentic case
(T12-Q1). The case study's chosen policy is to expose both and default to read, revisiting if
measurement shows one path never wins on this fabric.

Why this is not trivia: it changes which pool's queue depth you alert on and where a request's P99 tail
forms. The cheat sheet's failure signature "long tail on P99 → one slow transfer blocks the request" is
a direct consequence of getting this wrong, and the mitigation is async transfer with a
timeout-and-fallback-to-recompute.

**Signal:** Frames it as a backpressure-placement decision tied to which pool is scarce, not as a
performance preference.

**Follow-ups:**
- *Does NIXL offer both?* — the corpus does not say; this is not measured here.
- *Which pool's queue do you alert on if you choose write?* — decode's, because that is where the stall
  will surface as prefill blocking.
- *What mitigates the P99 tail either way?* — async transfer plus a recompute fallback.

**Red flags:** Treats the two modes as interchangeable, or picks one without naming which pool is
scarce.

---

#### T12-Q10 · Design the transfer plane for a fabric you did not choose
**Difficulty:** L5 · **Depth expected:** 6–8 min

**Question:** You have inherited a two-node deployment on a fabric you did not select, with no option to
buy NICs. Design the KV transfer plane end to end, and tell me what would make you abandon it.

**Model answer:** Constraints first, then mechanism, then the failure path.

**Step 1 — the volume.** `bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes`, times
the p95 prompt length `[D]`. Use p95, not mean: transfer cost is a tail phenomenon and the mean prompt
hides the requests that hurt.

**Step 2 — the arithmetic that decides everything.** `transfer_time = transfer_bytes / usable_BW` `[D]`,
compared against the prefill it replaces:
`prefill_time ≈ 2 × N_params × prompt_tokens / (FLOPs_peak × MFU)` `[D]`. Add queueing. If `transfer_time`
is a meaningful fraction of `prefill_time`, no design downstream of here saves you.

**Step 3 — transport and isolation.** Rank the transport (pooled memory < RDMA << TCP/IP) and then check
the thing the ranking does not tell you: **the KV plane shares the east-west fabric with MoE all-to-all**
under WideEP, and they contend directly. Without separate NICs, the remaining lever is QoS classes — or
reducing EP degree, which costs KV headroom.

**Step 4 — direction.** Read mode if prefill is the bottleneck (T12-Q9).

**Step 5 — the degraded path, declared before the incident.** Async transfer with a timeout, falling
back to recompute, so one slow transfer cannot stall a request indefinitely; and if the transfer agent
is down, fail the pool back to colocated mode rather than to a stall. Health-check the agent as a
first-class dependency — the case study lists "KV transfer agent down" with a blast radius of *every
disaggregated request*.

**Step 6 — compatibility.** Pin both pools to one engine version. A KV-layout change between vN and vN+1
produces silently corrupted output, not a crash (T12-Q29).

**What would make me abandon it:** `transfer_time` approaching `prefill_time` on measured traffic; a
short-prompt share high enough that recompute beats transfer for most requests; or goodput below the
colocated baseline after two single-change iterations.

**Signal:** Does the arithmetic before the design, names the fabric-sharing problem, and declares the
degraded mode without being asked.

**Follow-ups:**
- *What if there is genuinely no separate NIC?* — QoS classes, then reduce EP; T12-Q28.
- *What's the fallback when the agent dies?* — fail back to colocated; T12-Q27.
- *How do you test the degraded path?* — inject transfer failure in a load test; if you have never run
  it, you do not have it.

**Red flags:** Designs the happy path only, or picks a transport without computing the volume first.

---

#### T12-Q11 · Write the break-even condition for the split
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Give me the condition under which disaggregation beats colocation, term by term. Then tell
me the four ways it loses.

**Model answer:** From the cheat sheet `[D]`:

```
split_wins ⟺ (prefill_gain + decode_gain) > transfer_time + queueing + failure_risk
```

Unpacked:

- **`prefill_gain` + `decode_gain`** — phase-pure batches let each pool run its own optimal batch
  shape, and each pool scales independently. This is the whole upside.
- **`transfer_time`** — `transfer_bytes / fabric_BW`, and it is on the critical path for every request
  (T12-Q7).
- **`queueing`** — the split adds a scheduling hop and at least one more decision. At low load this term
  dominates because there is nothing to amortise it against.
- **`failure_risk`** — two pools are two failure domains, and a prefill-pool outage now fails
  decode-capable requests too. This term is the one candidates forget; it is not a performance term at
  all.

**The four loss conditions:**

1. **Short prompts.** For a ~200-token prompt, transfer latency can exceed recomputation. The case study
   calls this "the single most common cause of 'disaggregation made us slower'", and the mitigation is
   to route short prompts to a collocated replica rather than forcing every request down the
   disaggregated path.
2. **Low QPS / chat-shaped traffic.** Two half-idle pools are worse than one busy pool; the split is a
   high-utilisation technique `[D]`, with the measured crossover around ~85 QPS `[T]` Singh.
3. **TCP/IP-only fabric.** Usually a loss; compute `KV_bytes/BW` first `[D]`.
4. **Shared saturated fabric.** MoE all-to-all and KV transfer contend directly under WideEP and both
   degrade together (T12-Q28).

And the epistemic caveat that belongs in the answer: the corpus's own best-topology evidence (2P4D) was
flagged preliminary by the speaker `[T]` Chaitanya, so the empirical side of this ledger is thinner than
the mechanism side. Which is exactly why the case study's runbook starts with a colocated baseline.

**Signal:** Writes all four terms including `failure_risk` and `queueing`, names short prompts as the
classic loss case, and volunteers the preliminary-evidence caveat.

**Follow-ups:**
- *Which term is cheapest to shrink?* — transfer_time, via quantised KV or a model with fewer KV heads.
- *What does queueing actually cost at low QPS?* — the hop is a fixed latency with no amortisation;
  T12-Q18.
- *How do you establish the colocated baseline?* — pin the engine version, run the real workload, record
  per-phase p95s and KV hit rate per session.

**Red flags:** Gives only `transfer_time` on the cost side, or asserts the split is a win above some GPU
count rather than above a load.

---

### What the router knows — KV events and cache-aware routing

#### T12-Q12 · What does the router consume to know where KV lives?
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** The router has to decide which pod serves a request. What does it actually know, and where
does that knowledge come from?

**Model answer:** It consumes the engine's **KV events — per request, both create and evict**. That
"both" is the load-bearing part of the answer.

> "whenever a VM pod creates a KV cache or evicts a KV cache the LLMD router gets to know this and it
> uses that to know … which particular instance a particular host has to be routed to" `[T]` Pravin.
> And, confirming the granularity: "for every request we get the KV events" `[T]`. And again on
> eviction: "when a KV cache is evicted, we maintain — so we get events on when a KV cache is created
> and also evicted" `[T]`.

The mechanism, in the router's own terms: it "builds a live view of where the cache is located" with the
distributed KV cache mechanism, using KV-event metrics; then it **filters** (by load, by prefix cache);
then it **ranks** (by token load or active requests) `[T]` Pravin. That is the producer → filter → rank
pipeline.

Two details worth drawing out:

- **Why eviction events matter as much as create events.** A router that only observed creation would
  hold a view that goes stale in the optimistic direction — it would keep sending requests to an
  instance whose blocks are gone. Eviction events are what make the view correct rather than merely
  populated. Eviction here is a normal, evented part of the cache lifecycle, not an anomaly.
- **The two phases score differently.** Prefill uses a prefix-cache identity filter plus a token-load
  scorer; decode "just need[s] active request scorer because they don't pull the KV cache from the
  prefill" `[T]` Pravin.

**Signal:** Says create **and** evict, explains that the pair is what keeps the cache-location view
correct, and distinguishes the prefill and decode scorers.

**Follow-ups:**
- *What breaks if you only consume creation events?* — stale optimistic routing; requests hit instances
  that no longer hold the prefix.
- *Why doesn't decode need the prefix-cache filter?* — it is not looking for a cache to reuse; it is
  looking for spare attention budget.
- *How precise must this be?* — T12-Q14.

**Red flags:** Mentions only creation events, or characterises eviction as chaotic or as a failure mode.

---

#### T12-Q13 · Eviction: normal lifecycle, or something going wrong?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Your KV cache is evicting blocks steadily under production load. Is that an incident?

**Model answer:** No. Eviction is what a bounded cache *does*, and in this stack it is a first-class,
instrumented part of the lifecycle: the engine emits per-request create and evict events and the router
consumes both `[T]` Pravin (T12-Q12).

The operator's framing makes the same point. The saturation pattern the corpus describes is declarative
— "when the KV cache is 80% full I declare the cluster is saturated" `[T]` Pravin — and saturation is
described alongside "there's KV eviction happening and maintaining KV events" `[T]`. Eviction is the
mechanism that keeps a full cache usable, and there is an API whose whole purpose is to *steer* it (see
the retention API, T12-Q21): "if you know that it's going to take a longer time you can evict the KV
cache and put it to the CPU and then come back to it when it comes back" `[T]` Pravin. Deliberate,
evented, steerable.

What genuinely is an incident is **thrash**, which is a different thing: repeated evict/restore without
useful progress. The case study lists it as a failure signature — "repeated evict/restore; throughput
oscillates," detected by KV utilisation crossing the saturation threshold repeatedly `[T]`/`[D]` — and
the fix is to push to a lower tier rather than letting the top tier thrash.

So the distinction to draw is: eviction at a rate proportional to churn is healthy; eviction oscillating
around a threshold on a period is a configuration problem.

**Signal:** Distinguishes evented eviction from thrash, knows the retention API steers eviction, and does
not use catastrophising language about it.

**Follow-ups:**
- *What is thrash, and how do you detect it?* — oscillation on a period; alert on threshold crossings.
- *What steers eviction?* — the retention API with session metadata; T12-Q21.
- *What happens to blocks whose metadata is lost?* — orphans; a slow leak, needing a reaper.

**Red flags:** Calls eviction rampant, chaotic, or a bug; or cannot distinguish eviction from thrash.

---

#### T12-Q14 · Precise versus approximate prefix routing
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** The router can locate a request's prefix cache precisely or approximately. Which, and why
does the gap matter?

**Model answer:** Precisely, and the corpus has a measured comparison. There are two mechanisms: the
**precise** path, which uses the KV-event stream to maintain an accurate view of where blocks live, and
the **approximate** path, which "hashes a particular request and then knows which instance it goes to"
`[T]` Pravin — cheap and stateless, but which "has all the consistency problems" `[T]`.

The reported result: on a GLM-5.2 agentic deployment on H100s and H200s, "**precise KV cache routing
perform[ed] much better than the approximate**" `[T]` Pravin.

The mechanism for why the gap is large *specifically here* is the part that makes this an L4 question.
Approximate routing implicitly assumes the cache a request would hit is still resident. Under churn that
assumption breaks: the hash-stable answer and the actual location diverge, and the router sends the
request to a pod that no longer holds the prefix, which then re-prefills. The corpus attributes the gap
directly to this: "because of all these agentic workloads where the KV cache is saturated there's KV
eviction happening and maintaining KV events … precise KV cache affinity helps a lot" `[T]`.

So the generalisable statement is: **approximate routing degrades exactly when eviction is frequent —
which is exactly when routing matters most.** In a low-churn, low-load deployment the two paths are
nearly equivalent and approximate is cheaper; at saturation the approximate path silently converts cache
hits into full prefills, which is the most expensive possible failure for a prefill-bound workload.

**Signal:** Ties the gap to eviction frequency rather than to "precise is newer," and can say when
approximate would be fine.

**Follow-ups:**
- *What infrastructure does precise require?* — KV-event consumption; T12-Q12.
- *When is approximate acceptable?* — low churn, low load, or as a bootstrap before events are wired.
- *How would you measure the gap on your traffic?* — split by routing path and compare prefill tokens
  per request and KV hit rate per session.

**Red flags:** Says "precise is better" with no mechanism, or does not connect it to eviction rate.

---

#### T12-Q15 · P2P GPU→GPU KV sharing — what problem does it solve?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** An instance holds the KV your request needs, but it is busy. What are your options?

**Model answer:** Three, and P2P fetching is the middle one. The corpus states the decision space
directly:

> "peer-to-peer is when your KV cache is located in one instance and there is low availability of load
> in other instance. So instead of doing the — instead of you, you can either choose to queue in one
> instance or to do the recompute in another instance, but the middle ground there is to go to the other
> instance and fetch the KV cache from the previous instance, and that helps save a lot for the
> recomputation" `[T]` Pravin.

So:

1. **Queue** on the busy instance — wait. Preserves the cache, costs latency, propagates the queue.
2. **Recompute** on the idle instance — no transfer, but you pay a full prefill, which in an agentic
   workload is ~98% of the token cost (T12-Q1).
3. **Fetch** the cache over the fabric — pay a transfer, avoid the recompute.

The cheat sheet lists P2P GPU↔GPU KV as "nothing; peers share / fabric bandwidth / middle ground: avoid
recompute without a pool" `[T]`/`[D]`. Note the last clause: P2P is what you reach for when you do *not*
have a shared pooled-memory tier — it is the same NIXL-class transport as the PD path, used laterally
between peers rather than between phases `[D]`.

Two consequences worth drawing out. First, the decision is the **router's**, and it can only make it
because it consumes KV events — this is a direct application of T12-Q12, not a separate feature.
Second, P2P competes for the same east-west bandwidth as the PD transfer and the MoE all-to-all, so it
is the right answer only when the fabric has headroom; under contention, option 1 or 2 may be genuinely
better than option 3.

**Signal:** Names the three-way choice and identifies fetching as a fabric-bandwidth purchase that beats
recompute only when the fabric is not the bottleneck.

**Follow-ups:**
- *Who decides to fetch?* — the router, using its cache-location view.
- *When does fetching lose?* — when the fabric is the constraint; then queue or recompute.
- *How does this differ from a shared pool?* — no shared address space; you move bytes rather than
  sharing them; T12-Q25.

**Red flags:** Treats P2P as free, or does not recognise it as competing with the same fabric as PD
transfer.

---

### Sizing the pools — the prefill:decode ratio

#### T12-Q16 · Derive the prefill:decode ratio from the token mix
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Two people on your team propose opposite ratios. Both cite measurements. Reconcile them.

**Model answer:** Start from the rule: `P:D ratio ≈ (aggregate prefill token rate) / (aggregate decode
token rate)` `[T]`, with the corpus's own instruction attached — "**measure your own ratio**; it is
workload-dependent" `[T]`.

Now apply it to the two data points the corpus actually gives, with the arithmetic shown:

**(a) Pravin's agentic coding workload.** ~70% of inference traffic is agentic `[T]`, and within agentic
traffic prefill is ~98% of the tokens `[T]`. So the token-share ratio is
`98 : 2 ≈ 49 : 1` prefill:decode `[D]`. More prefill is what helps — which is what the speaker reports
from comparing 2P2D against 3P1D: "having more prefill than the decode definitely helped" `[T]`.

**(b) Chaitanya's 28k-in / 1k-out long-input shape.** The configuration carries "around 7.34 million
tokens and decode itself takes around 5 million tokens" `[T]` — so `2.34 : 5 ≈ 1 : 2.1` `[D]`. Here
*decode* dominates, and 2P4D performed best (flagged preliminary; T12-Q19).

**They are not contradictory.** (a) is a **token-share** statement: a coding agent reads a repository
and emits a few tokens, so prefill tokens vastly outnumber decode tokens. (b) is a **step-count**
statement: many concurrent long sequences, each generating 1k output tokens, so the aggregate number of
decode *steps* across those sequences dominates even though each step emits one token per sequence. Same
technique, opposite ratio, because the shapes differ.

Two operational conclusions. First, the ratio is a property of the **workload class**, not of the model
or the hardware — so a single global ratio is the classic first-production mistake, and the mature
answer is per-workload pools with independent autoscaling. Second, the ratio is not a set-and-forget
constant: alert on sustained utilisation asymmetry between pools (the case study's threshold is
sustained >30%) and re-derive.

**Signal:** Computes both ratios from the corpus figures and explains the divergence as token-share
versus step-count — not as one speaker being wrong.

**Follow-ups:**
- *What do you instrument to get your own ratio?* — token counters split by phase, per workload class.
- *What does the ratio do to pool floors?* — a 49:1 ratio still needs ≥2 decode GPUs; T12-Q17.
- *What does the ratio do at 10×?* — one global ratio becomes indefensible; classification becomes a
  prerequisite.

**Red flags:** Picks one ratio and dismisses the other as noise, or derives the ratio from model size
rather than from measured traffic.

---

#### T12-Q17 · Eight GPUs, a 98/2 token mix — split them
**Difficulty:** L5 · **Depth expected:** 5–6 min

**Question:** You have one 8-GPU node, 192 GB HBM per GPU. Telemetry says the traffic is ~98% prefill
tokens. Split the node, and defend the number.

**Model answer:** I would derive it, not recall it.

**Step 1 — the naive ratio.** 98/2 by tokens says the node's work is overwhelmingly prefill, so a
symmetric 4P4D misallocates roughly half the node. Carrying the ratio through gives ≈ **7P1D** `[D]`.
That is the case study's own derivation, and it is the right starting point.

**Step 2 — the two constraints that floor it.** The 7P1D answer is not deployable, for two reasons:

- **Availability.** A one-GPU decode pool has no failure tolerance. Losing it loses all decode capacity.
- **Burst headroom.** 98/2 is a *sustained average*, not a per-second constant. Tool-call returns arrive
  in bursts, and the decode pool must absorb them without queueing.

So the practical floor is **2 decode GPUs**, giving **6P2D** `[D]`.

**Step 3 — say why 4P4D is not the answer.** Chaitanya's CI runs 1P1D and 2P2D, but those are
explicitly a coverage policy — the shapes worth testing — not an optimum `[T]`. The case study's
conclusion is the same as mine: on an 8-GPU node with a genuine 98/2 mix the split should be 6P2D, and
the symmetric split is a *coverage* choice rather than an optimal one.

**Step 4 — what the 2 decode GPUs must survive.** Decode is memory-bandwidth bound, so the constraint on
that pool is KV capacity for in-flight sequences. This is where the retention tier does real work: if
paused sessions' KV is offloaded to DRAM rather than pinned in HBM, the decode pool's effective capacity
is larger than its HBM suggests (T12-Q20).

**Step 5 — the risk in my own answer.** If the mix re-flattens toward chat, 6P2D starves decode, and I
have over-provisioned the wrong pool. That is exactly what the sustained utilisation-asymmetry alert
exists to catch, and it is why I would record the mix the split was chosen for.

**Signal:** Refuses the symmetric split, shows the derivation to 7P1D, then names the two constraints
that floor it at 2 decode GPUs — and states the risk in their own answer.

**Follow-ups:**
- *What floors the decode pool?* — availability plus burst, not the token ratio.
- *How would you validate the split?* — colocated baseline first, then the identical suite on 6P2D;
  whiteboard exercise 1.
- *What changes at 10× concurrency?* — per-workload pools; a single ratio stops being defensible.

**Red flags:** Recommends 4P4D because it is symmetric and familiar, or recommends 7P1D with no
availability floor.

---

#### T12-Q18 · The ~85 QPS crossover — what exactly was measured?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Someone puts "disaggregation gives 2× throughput" on a slide, citing llm-d. Interrogate
that.

**Model answer:** The claim as reported: "at about 85 queries per second, you're seeing double the
throughput" `[T]` Singh, *NxtGen*. The speaker's own framing is that the size of the effect is
surprising — "it's not supposed to be such a drastic improvement, but it is" `[T]`.

What it is: a **measured crossover on a multi-accelerator study** — MI325X, H100 SXM and Intel Gaudi 3
appear in that study `[T]` — run by a cloud provider that sells inference capacity. So it is a
vendor-affiliated result, and the case study marks it as such. Cite it as a prior, not as a law.

**Why a crossover must exist** (this is the part that makes the number credible rather than arbitrary):
below it, the split adds a hop, a scheduling decision and a second failure domain without buying
utilisation — two half-idle pools beating one busy pool is not a thing. Above it, phase-pure batches and
independent scaling dominate. This is the same "at low load the split loses" rule from T12-Q11, made
quantitative.

**Sanity-check the magnitude** with the speaker's own contract, and show the assumption: an enterprise
SLA of **16 req/s on half an H100** `[T]` Singh. So one H100-equivalent ≈ 32 req/s, and
`85 / 32 ≈ 2.66 H100-equivalents` of raw capacity `[D]`. That is under one 8-GPU node — consistent with
the crossover being reachable on modest hardware, which is what makes the migration schedulable rather
than aspirational.

**What it does not license:** "disaggregation doubles throughput" as a general statement. The double is
at a specific QPS, for a specific model class, on specific accelerators, in a vendor study. Your
crossover must be measured.

**Signal:** Quotes the number with the affiliation caveat, explains why a crossover must exist, and
refuses to generalise the 2×.

**Follow-ups:**
- *What is the unit of the 16 req/s contract?* — half an H100, for that model class; it is the speaker's
  SLA framing.
- *What would move your crossover?* — prompt-length distribution, fabric, model, mix.
- *What do you do below the crossover?* — stay colocated; T12-Q24.

**Red flags:** States "2× throughput" as a universal property, or cannot say what the 85 QPS is a
crossover *of*.

---

#### T12-Q19 · A 2P4D result and a 3P1D result. Reconcile them.
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** One talk says more prefill helped; another says the best topology was decode-heavy. Which
is right?

**Model answer:** Both, and the reconciliation is mechanical — but the *epistemic* half of the answer
matters as much as the arithmetic half.

**The two claims.** Pravin, on a GLM-5.2 agentic deployment over H100/H200: comparing 2P2D against 3P1D
ratios, "having more prefill than the decode definitely helped" `[T]`. Chaitanya, on a 64-GPU GLM-5.1
sweep over MI300 with full PD disaggregation, on a 28k-in/1k-out shape: "for 2P4D we have the best
performance" `[T]`.

**The reconciliation.** Look at what each workload *is*:

- Pravin's is **prefill-heavy in token share** — ~98% of tokens are prefill (T12-Q1, Q16), with few and
  short outputs. More prefill capacity is therefore the correct answer.
- Chaitanya's is **decode-heavy in step count** — "around 7.34 million tokens and decode itself takes
  around 5 million tokens" `[T]`, i.e. ~68% decode `[D]`. Many concurrent long sequences, each
  generating ~1k tokens, so aggregate decode steps dominate. More decode capacity is therefore the
  correct answer.

Same technique, opposite ratio, because the shapes differ. A ratio is a property of a workload class,
not of the technique.

**The epistemic half, which is where this becomes an L5 question.** Weight the evidence before you
choose between them:

- The 2P4D result was flagged by its own author — "these are all preliminary results, these are like
  very initial results, so don't look into the performance here" `[T]` Chaitanya. It is not a finding.
- The CI topologies (1P1D, 2P2D) are a coverage policy, not an optimum `[T]`.
- Pravin's comparison is an operational observation on a production-shaped deployment, not a controlled
  sweep.

So the strongest claim actually available from this corpus is narrow: *more prefill helped for
prefill-heavy agentic traffic.* A candidate who treats 2P4D as the answer has mistaken a preliminary
number for a result; a candidate who treats the two as contradictory has missed the token arithmetic.

**Signal:** Explains the divergence with token share versus step count, and voluntarily discounts the
preliminary result — unprompted.

**Follow-ups:**
- *Which claim is stronger?* — Pravin's, and say why: it is not self-flagged as preliminary.
- *What would you benchmark instead of arguing?* — both ratios on your own mix; T12-Q16.
- *What does "preliminary" oblige you to do?* — say so when you cite it; the case study's interview
  guidance is to volunteer this before being asked.

**Red flags:** Picks one result as correct, or cites 2P4D without the preliminary caveat.

---

### Retention — the KV lifecycle across a session

#### T12-Q20 · What happens to the KV when a session pauses?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** An agent makes a tool call and the user walks away for ten minutes. What happens to its KV
cache?

**Model answer:** The wrong answer is "nothing" — holding it in HBM blocks capacity for every other
session, and at ~70% agentic traffic `[T]` Pravin the number of simultaneously-paused sessions crosses
HBM capacity immediately. So it is tiered:

```
HBM (active)  →  DRAM (on pause)  →  external / pooled tier (long tail)
```

with **session metadata attached to the blocks** so the router knows which blocks belong to which
session `[T]` Pravin. The rationale in the speaker's words: agentic workloads "sometimes take a pause and
come back after some time. So at that point of time you want to evict the KV cache and offload to the
CPU and then do not do the recompute again" `[T]`.

Why it pays: "that saved about **5× in TTFT** when the agentic session comes back after a pause" `[T]`
Pravin (T12-Q22 interrogates this figure). Kwon states the same property from the engine side — in
multi-turn agent sessions "we never recompute the tokens in the previous turns," as long as storage
allows `[T]`. And the tier ladder is the same design in SGLang's HiCache: systems that "move the KV
cache down from HBM to DRAM and even to your external storage" `[T]` Zhu.

**The failure mode to name unprompted:** a pause longer than the retention TTL. The session resumes, the
blocks are gone, and the client sees a TTFT spike indistinguishable from a cold start. The correct
observability response is to alert on a session-resume-recompute counter rather than on TTFT alone,
because TTFT alone cannot tell a cache miss from a slow request.

**Signal:** Reaches for the tier ladder without prompting, attaches session metadata to the blocks, and
names the TTL-expiry observability gap.

**Follow-ups:**
- *Which tier change is cheapest?* — the TTL and the tier thresholds; the runbook tunes them first.
- *What happens when DRAM also fills?* — spill to the external tier; the long tail stays cached.
- *How do you detect a resume that recomputed?* — the session-resume-recompute counter.

**Red flags:** Says KV stays in HBM, or says it is simply dropped and recomputed.

---

#### T12-Q21 · The retention API and session metadata
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** What is the KV retention API, and why does session metadata matter more than the API
itself?

**Model answer:** The API is the control surface that turns eviction from a *reaction* into a
*decision*. The corpus describes it precisely:

> "the KV cache retention API, which I think there's a PR from Nvidia on that, looking at providing APIs
> in the vLLM layer to orchestrate the movement of KV cache, and also it adds a session metadata to the
> KV cache so that we know which session a particular blocks of KV cache belong to" `[T]` Pravin.

Note the sourcing — it is described as **a PR**, not shipped functionality. Say that if you cite it.

Two capabilities, and the second is the more important one:

1. **Orchestration.** Move this session's KV now, deliberately, rather than only under memory pressure.
   The trigger is knowledge the scheduler has and the cache does not: "if you know that it's going to
   take a longer time you can evict the KV cache and put it to the CPU and then come back to it when it
   comes back … all these things can help in freeing the KV cache for more active sessions" `[T]`.
2. **Attribution.** Session metadata is what makes the router's decisions *session-aware* rather than
   request-aware: "you're not just routing requests, you're routing for sessions. So you need to decide
   which sessions are more active, which sessions have a tool call which is coming back after a long
   time, and then route them accordingly" `[T]`. It also depends on the engine being able to tell
   whether a given KV cache needs evicting at all — "this needs more plumbing with the vLLM" `[T]`.

**The failure mode without attribution:** if the session metadata is lost — a restart, an eviction of the
metadata store — the blocks become **orphaned**. They consume memory but cannot be attributed to any
session, so nothing will ever reclaim them by normal means. The case study is precise about the shape:
"a slow leak, not a crash," fixed with a reaper on a timer plus a metadata checksum.

So the honest ranking: the API lets you schedule; the metadata is what makes the schedule correct. An
implementation with the API and no attribution is worse than none, because it can evict the wrong thing.

**Signal:** Names orchestration *and* attribution, ranks attribution as the load-bearing half, and names
the orphaned-block failure mode.

**Follow-ups:**
- *What's the shipped status?* — a PR, per the speaker; treat as not-yet-GA.
- *What does orphaned metadata look like in production?* — memory creeping with no traffic growth; check
  block accounting, allocated versus attributed.
- *How does retention interact with the pool ratio?* — offloaded KV frees HBM, which changes the
  effective decode capacity; T12-Q17.

**Red flags:** Describes the retention API as a cache-tuning flag, or does not mention session metadata
at all.

---

#### T12-Q22 · "CPU KV offload saved ~5× TTFT." Interrogate that claim.
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** You want to put this number in a business case. What is it actually claiming, and what
would you have to verify first?

**Model answer:** The claim as stated: "CPU KV cache offloading — because these agentic workloads
sometimes take a pause and come back after some time. So at that point of time you want to evict the KV
cache and offload to the CPU and then do not do the recompute again. So that saved about 5× in TTFT when
the agentic session comes back after a pause" `[T]` Pravin.

Four things to establish before it goes in a business case:

1. **What is the baseline?** It is **recomputation**, not colocated serving. So "5×" is the ratio between
   restoring blocks from host DRAM and re-prefilling the same tokens. That is a different quantity from
   "disaggregation is 5× faster," and conflating the two is how a business case becomes a credibility
   problem.
2. **What is the workload state?** A paused agentic session *returning*. This is resume latency, not
   steady-state TTFT and not ITL. The whole benefit accrues to the multi-turn pattern.
3. **What is the provenance?** A single speaker's operational observation on a GLM-5.2 deployment over
   H100s and H200s `[T]` Pravin — the case study records it as "speaker's observation," not a controlled
   benchmark, and the case study's benchmark table gives it no conditions beyond "on session return
   after a pause."
4. **What does it not license?** Anything about transfer latency in microseconds, or a NIXL-versus-MoRI-O
   comparison. The case study is explicit: "Not measured in this corpus: KV transfer bytes per token,
   transfer latency in microseconds, and any head-to-head of NIXL versus MoRI-O. Do not assert these."

**Why the claim still matters despite all that.** Because the mechanism is unambiguous and the
amplification is large. The case study's arithmetic, with assumptions shown: a session that pauses 30
times per lifecycle and prefills 8,000 tokens per turn costs `30 × 8,000 = 240,000` prefill tokens
without retention, and `8,000` with it — a **96.7% reduction** `[D]`. Even if the 5× figure is the only
measured benefit and the hit rate is imperfect, the token arithmetic is overwhelming. That is why "KV
cache hit rate per session" `[T]` is a first-class agentic SLI.

**Signal:** Separates "5× versus recompute" from "5× versus colocated," bounds the provenance, and
refuses to extend it to unmeasured quantities — while still arguing the mechanism is decisive.

**Follow-ups:**
- *What's the right baseline for a business case?* — the colocated deployment you are migrating from.
- *What would you instrument to verify it?* — resume latency split by hit/miss, plus a
  session-resume-recompute counter.
- *Show the amplification arithmetic.* — 240k → 8k tokens per session; state the 30-pauses/8k-turns
  assumptions.

**Red flags:** Repeats "5× faster" without naming the baseline, or uses it to justify the *split* rather
than the *retention tier*.

---

#### T12-Q23 · Offload tier or PD split — which do you ship first?
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You have budget for one change this quarter. Both are on the table. Which, and why?

**Model answer:** **Retention first**, and the reasoning is about risk and diagnosability, not about
which is more exciting. The case study's runbook is explicit: "Enable the KV retention tier *before*
disaggregating. Retention is the higher-value change and it is lower-risk; do it first so that a
disaggregation regression is isolated."

Four reasons, in order of weight:

1. **They solve different problems.** Retention fixes *repeated work* — the same session being
   pre-filled thirty times. The split fixes *phase interference* — prefill and decode stealing each
   other's cycles. If your pain is the former, the split does nothing for it. The case study's own
   arithmetic makes the former enormous: 240,000 prefill tokens per session down to 8,000, a 96.7%
   reduction `[D]` (assumptions: 30 pauses, 8k tokens per turn).
2. **Retention is one pool.** It does not add a failure domain, does not put a network hop on the request
   path, and does not require a second deployment to operate. The split does all three — that is the
   `failure_risk` term in the break-even (T12-Q11).
3. **It is lower risk on hardware you already own**, which matters because the case study's political
   constraint is "prove it on the existing node first."
4. **It is diagnostically isolating.** If TTFT improves from retention and then degrades when you split,
   you know which change did what. Do them in the other order and a regression is ambiguous — and an
   ambiguous regression in a migration is how the migration gets cancelled.

**The honest counter-argument, which I would state before being asked:** if the traffic is genuinely
prefill-bottlenecked at high QPS, retention does not remove the coupling, and the split's independent
scaling is the only thing that does. So the correct framing is sequencing rather than ranking — and the
cheat sheet's "when to use what" says the same: multi-turn sessions with idle gaps → offload tier first
(cheaper); long prompts + high concurrency with both TTFT and ITL SLO-bound → PD disaggregation.

**Signal:** Sequences by risk and diagnosability rather than by expected gain, and can state what each
change does *not* fix.

**Follow-ups:**
- *What does retention not fix?* — phase contention.
- *What does the split not fix?* — repeated prefill across turns.
- *When would you invert the order?* — a steady-state prefill-bound service with high QPS and short
  sessions; retention has nothing to reuse.

**Red flags:** Recommends the split because it is more architecturally interesting, or treats the two as
alternatives rather than sequenced complements.

---

#### T12-Q24 · When does disaggregation lose outright?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** Give me the conditions under which you would not disaggregate, even though it is the
fashionable answer.

**Model answer:** Four, plus the structural cost to name regardless.

1. **Short prompts.** For a ~200-token prompt, transfer latency can exceed recomputation. The case study
   calls this "the single most common cause of 'disaggregation made us slower'," and the mitigation is
   operational rather than architectural: route short prompts to a collocated replica. Do not force
   every request down the disaggregated path just because the path exists.
2. **Low QPS, chat-shaped traffic.** Two half-idle pools are worse than one busy pool; the split is a
   high-utilisation technique `[D]`. The measured crossover is around **~85 QPS** `[T]` Singh
   (vendor-affiliated; T12-Q18).
3. **Fabric mismatch.** Over TCP/IP the transfer is usually a loss `[T]`/`[D]`. Compute
   `KV_bytes / fabric_BW` before designing anything — if you do not know your fabric bandwidth, you do
   not know whether the split can pay.
4. **Shared saturated fabric.** MoE all-to-all and KV transfer contend directly under WideEP, and both
   degrade together (T12-Q28). Disaggregating onto a fabric that is already the bottleneck moves the
   bottleneck, it does not remove it.

**And the cost even when it wins**, which candidates routinely omit: it doubles the failure domains and
adds a hop. A prefill-pool outage now fails decode-capable requests too, so you must plan the degradation
path up front — fall back to colocated, or to KV recomputation — and you must have tested that path.
The case study's chosen configuration for the in-node stage is exactly this discipline: prove it on the
existing node, then move out.

**Signal:** Names at least three loss conditions plus the "route short prompts away" mitigation, and
names the failure-domain cost rather than treating it as free.

**Follow-ups:**
- *What do you do with the short-prompt tail?* — a collocated replica, routed by prompt length.
- *What is your degraded mode?* — colocated fallback or recompute; declare it, test it.
- *How do you know you're below the crossover?* — measure goodput against a colocated baseline.

**Red flags:** Treats the split as universally correct, or cannot name a single condition where it
loses.

---

### Rack-scale pooling and the hardware axis

#### T12-Q25 · Pooling mode versus sharing mode
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** A rack-scale pooled-memory box offers two modes. Why does the distinction matter, and
which would you ask for?

**Model answer:** It matters because one mode is a capacity extension and the other is a coherence
commitment. Kim's framing:

| Mode | Semantics | What you get |
|---|---|---|
| **Memory pooling** | Each node dynamically allocates additional memory, but "that memory region is isolated between node" `[T]` | More room. No sharing. |
| **Sharing mode** | "Multiple nodes can see the same memory address space. So each node can access the same data" `[T]` | True KV sharing — the same blocks usable by several servers |

Topology context: the demo is four servers with a pooled memory box mid-rack — "a really disaggregated
memory pool … multiple servers can use this memory pool at the same time" `[T]` Kim.

**Which I would ask for depends on what I am buying.**

- If the problem is **capacity and OOM** — prefill and decode competing for HBM, or a deployment where
  KV occupancy is the binding constraint — pooling mode gets you the benefit with no coherence risk. It
  buys the OOM-resilience property: "if there is out of memory at the decoding side and prefill side …
  we can continuously do the prefill because we already uploaded that KV cache to the pool memory side"
  `[T]`. That is architecturally significant on its own — it decouples prefill's ability to make progress
  from decode's ability to consume (T12-Q30).
- If the problem is **cross-server prefix reuse** — several servers legitimately serving the same prefix
  — sharing mode is the only mode that gives it, and the case study lists "sharing-mode coherence risk"
  as a first-class caveat. Coherence is not a footnote; it is the price.

There is also a reuse claim that is sharper than "caching": "the old KV cache can be stored in the
memory pool, so we can reuse it for the next request **without any additional storing operation**" `[T]`
Kim. Reuse *without re-store* is a different and stronger property than reuse-with-transfer, and it is
worth separating when you write the business case.

**Signal:** Separates capacity-extension from address-space-sharing, names coherence as sharing mode's
price, and recognises reuse-without-re-store as a distinct property.

**Follow-ups:**
- *Which mode do you need for cross-server prefix reuse?* — sharing mode.
- *What's the failure mode of sharing mode?* — coherence; the case study flags it explicitly.
- *When is pooled memory pointless?* — cross-rack or cross-datacentre topology, or when your prompts are
  short enough that recompute beats any transfer.

**Red flags:** Treats the two modes as performance tiers of one thing, or does not mention coherence.

---

#### T12-Q26 · Separate the pooled-memory vendor claims from the architecture
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** The rack-scale talk makes three claims. Sort them by how much weight they can carry in a
procurement decision.

**Model answer:** Three claims, and they are not epistemically equal.

**Claim 1 — "faster than RDMA."** "Instead of using TCP/IP we generally use RDMA. This is fast, but this
pooled-memory-based data sharing is faster than RDMA" `[T]` Kim. Status: **vendor claim.** The speaker is
SK hynix describing SK hynix hardware; the comparison baseline is Mooncake and a DRAM cache; and he
described the numbers as "very initial performance numbers" `[T]`. It is entirely plausible — the
transfer path is shorter and there is no network round trip — but the corpus contains no independent
head-to-head of pooled memory against a NIXL/RDMA KV transfer. This claim cannot carry a procurement
decision on its own.

**Claim 2 — OOM resilience.** "If there is out of memory at the decoding side and prefill side … we can
continuously do the prefill because we already uploaded that KV cache to the pool memory side" `[T]`.
Status: **architectural.** This is the most interesting statement in the talk and it is not a benchmark
at all. It says the pool decouples prefill's ability to make progress from decode's ability to consume —
which is a structural property of where the KV lives, and it gets *stronger* with scale, not weaker. At
10× the difference between a backlog and an outage is whether prefill can keep working when decode is
behind.

**Claim 3 — contention removal.** Storing and reusing KV creates contention on GPU-side PCIe bandwidth
and on the network; routing the traffic through the pool removes it `[T]`. Status: **architectural**, and
it is the second independent reason to consider the pool — independent of latency, and complementary to
the plane-isolation argument in T12-Q28.

So the honest sorting is: one vendor claim on immature numbers, two architectural arguments. The cheat
sheet's "when to use what" is consistent with that sorting — pooled memory appears for "you control the
datacentre and the network is the bottleneck," which is an architectural condition, not a benchmark
result.

What I would do before buying: measure my own `KV_bytes/BW` and my own store/reuse volume, then ask for
a benchmark on *my* prompt distribution against *my* current transport. And I would ask about the failure
path — the case study lists "pooled memory unreachable" as a failure mode whose mitigation is keeping a
DRAM tier as an intermediate so the pool is not the only fallback.

**Signal:** Sorts the three by epistemic status, identifies OOM resilience as the architectural gem, and
proposes a measurement rather than accepting the latency claim.

**Follow-ups:**
- *Which claim would you build a business case on?* — claim 2, with claim 3 as support.
- *What's the failure path if the pool is unreachable?* — a DRAM tier as intermediate; disable pooling.
- *How does this argument change at 10×?* — both architectural claims get stronger; the latency claim
  still needs measuring.

**Red flags:** Accepts "faster than RDMA" as established, or cannot distinguish a vendor benchmark from a
structural property.

---

### Failure modes, operations and the operational envelope

#### T12-Q27 · TTFT got worse after you disaggregated. Diagnose.
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Two weeks after moving from colocated to a split deployment, TTFT p95 is worse than it was.
Walk me through the diagnosis.

**Model answer:** The cheat sheet names this signature directly: "Split made TTFT *worse* → transfer
latency exceeds the prefill saving → first check fabric bandwidth vs `KV_bytes/BW`."

**Ordered diagnosis** `[D]`:

1. **Do the arithmetic first.** Compute `KV bytes` for the p95 prompt from
   `2 × n_layers × n_kv_heads × head_dim × dtype_bytes × prompt_len`, divide by *achieved* (not
   theoretical) fabric bandwidth, and compare to `prefill_time`. If `transfer_time` is a meaningful
   fraction of the prefill it replaces, you have your answer and no amount of config tuning will fix it.
   Doing this first prevents a week of tuning the wrong layer.
2. **Check the prompt-length distribution.** If a substantial share of traffic is short prompts, the
   transfer can exceed recomputation for those requests, and you are paying a hop for nothing. The
   mitigation is routing, not transport: send short prompts to a collocated replica (T12-Q24).
3. **Check whether the transfer is overlapped or serialised.** If the decode pool blocks waiting for the
   complete KV before it can begin, you have converted a compute problem into a serialisation problem.
   Async transfer with a timeout is the fix; a full-block wait is the anti-pattern.
4. **Check the fabric for contention.** Is MoE all-to-all sharing the east-west links? Under WideEP it
   is, and both phases degrading together is the discriminator (T12-Q28).
5. **Check queueing.** The split adds a scheduling hop. Compare goodput, not just TTFT.

**Fixes with costs:** route short prompts away (operational complexity); move to a faster transport —
pooled memory < RDMA << TCP/IP (hardware); switch transfer direction to read so backpressure lands on
decode (more decode-side state); reduce KV volume with quantised KV (accuracy risk); or **revert to
colocated**. Reverting is a legitimate engineering outcome, not a failure — and the case study's runbook
is built so that it is available, because you recorded a colocated baseline before you moved.

**The discipline point:** you should already have the baseline and a counter-metric. TTFT alone can
improve while the system gets worse; goodput plus per-phase p95s plus KV hit rate per session is the
minimum set.

**Signal:** Does the bytes/bandwidth arithmetic before touching configuration, treats revert as a
legitimate outcome, and asks for a counter-metric.

**Follow-ups:**
- *What's the counter-metric?* — goodput and throughput per GPU against the colocated baseline.
- *What if TTFT improved but goodput fell?* — you moved cost, not removed it; compare per-GPU, not
  aggregate.
- *How do you isolate transfer from queueing?* — measure transfer time directly, and compare a
  single-request path against the loaded path.

**Red flags:** Starts tuning the engine, blames the model, or has no colocated baseline to compare
against.

---

#### T12-Q28 · KV and EP all-to-all are sharing the fabric
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** Both TTFT and ITL have degraded together at peak, and the east-west links are near
saturation. What is happening, and what do you change?

**Model answer:** Two planes on one wire. Under WideEP every MoE layer performs a **dispatch and a
combine all-to-all** (T11), and the PD path adds a per-request KV transfer on top. They contend directly
for the same fabric.

**The discriminator that identifies this:** both phases degrade together, with no single culprit. That is
different from a ratio problem (one pool saturated, one idle) and different from a volume problem (TTFT
degrades, ITL holds). If aggregate link utilisation is your only metric, you cannot tell which plane is
responsible — which is why the detection instruments are **per-plane bandwidth counters and NIC queue
depth**, not aggregate utilisation.

**Mitigations, in order** `[D]`:

1. **Separate the planes.** QoS classes, or separate NICs. This is the structural fix, and the case
   study's 10× section is blunt about the timing: "the first thing that breaks is not the engine — it is
   the NIC. Plan separate planes before you need them."
2. **Reduce EP degree** to shrink the all-to-all span — accepting the KV-headroom cost, because fewer
   experts per GPU means more weight bytes per GPU and less room for cache (T11-Q8).
3. **Reduce KV volume** — quantised KV cuts `dtype_bytes` in the volume formula directly.
4. **Re-derive the ratio** so one pool is not over-sending relative to its work.
5. **Consider pooled memory**, which removes GPU-side PCIe and network contention from store-and-reuse
   entirely `[T]` Kim — a second reason to consider it beyond latency (T12-Q26).

**And re-measure per-GPU throughput**, not aggregate, after each single change. If you change two things
and throughput improves, you have learned nothing about which one worked.

**Signal:** Uses "both phases degrade together" as the discriminator for a shared-fabric problem, and
proposes plane separation before degree reduction.

**Follow-ups:**
- *Why reduce EP rather than increase it?* — the all-to-all span is the cost; widening makes this worse.
- *What does reducing EP cost you?* — KV headroom; you are trading cache capacity for fabric.
- *How do you attribute traffic to a plane?* — per-plane counters; aggregate utilisation cannot do it.

**Red flags:** Recommends more bandwidth without reducing span or separating planes, or cannot say which
plane is at fault.

---

#### T12-Q29 · Engine version skew between the pools
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Prefill is on engine vN; someone rolls decode to vN+1. What do you expect to see?

**Model answer:** **Silently corrupted output** — not a crash, and not an error rate. If the KV layout
changed between the versions, decode reads blocks it is misinterpreting, and the tokens it produces are
wrong in a way nothing in the system flags.

That is why the case study's incident table pairs the symptom with the check rather than with an alert:
symptom **"accuracy drift, no latency change"**, likely cause **engine version skew between pools**,
first action **verify both pools report the same engine version**. The control is to pin both pools to
one engine version and deploy them as a **single deployable unit**.

Why this deserves a question rather than a footnote: two of this topic's strongest architectural
arguments push in the opposite direction. The connector abstraction exists so transports are swappable
(T12-Q4), and the split exists so pools scale independently (T12-Q1). Both encourage treating the two
halves as independently deployable — and they are, right up until a KV layout change makes them
incompatible. Version pinning is the explicit counterweight: the abstraction is an interface for
transports, not a compatibility guarantee across engine versions.

A related skew worth mentioning in the same breath, from the case study's edge cases: a VLM whose encoder
is not disaggregated lands on the prefill pool and its cost is invisible in text-token metrics. That is a
different kind of skew — the metrics do not describe the work being done — and it is what EPD exists to
fix (T12-Q6).

**Signal:** Knows the failure is silent corruption rather than a crash, and names version verification as
the check rather than an error-rate alert.

**Follow-ups:**
- *How do you canary a KV-layout change?* — both pools in lockstep; the layout change is a breaking
  change, not a rolling one.
- *What else does independent scaling break?* — admission, health checks and the degraded mode all
  become two-sided; T12-Q30.
- *How does EPD relate?* — a third pool means a third version to pin.

**Red flags:** Expects an error or a crash, or proposes rolling the pools independently.

---

#### T12-Q30 · Prefill full, decode idle — the inverted signature
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** You are paged: the prefill pool's queues are full and the decode pool is completely
idle. Diagnose it, and tell me why this is a design failure rather than an incident.

**Model answer:** **The decode pool is unhealthy and admission has not noticed.** Requests enter prefill,
produce KV, and stall because there is nowhere healthy to hand off to — so prefill's queue grows while
decode sits empty. This is the exact *inverse* of the usual overload signature, where prefill is the
bottleneck and decode looks fine, and that inversion is the diagnostic: a healthy system under load
shows pressure in both pools, not one full and one empty.

The case study's entry: failure "decode pool failure with healthy prefill", symptom **"prefill full,
decode empty — the inverted signature"**, detection **per-pool active-request counts**, blast radius
**all new sessions**, mitigation **cross-pool admission check**, recovery **re-route to collocated
replica; restart decode**. In-flight decodes are fine — this only affects new sessions, which is why it
can be live for a while before anyone notices.

**Why it is a design failure rather than an incident.** An admission controller that checks only its
*local* pool will happily admit into the healthy pool and manufacture the backlog. The fix is structural:
admission must check the **downstream** pool too. Every design decision in this topic that treats the
two pools as one system — cross-pool admission, the two-sided health check, the declared fallback — is
what prevents this class of failure. The incident is the symptom; the missing cross-pool check is the
cause.

**Two related asymmetric cases worth naming:**

- **Recovery is the same shape as a transfer-agent failure.** When the KV transfer agent is down, the
  case study's recovery is "fail the pool back to collocated mode; drain and restart." Same medicine:
  route around the broken half rather than into it.
- **Decode pool cold start.** A newly scaled decode pod has no prefix cache and no in-flight requests, so
  the active-request scorer rates it perfectly and it absorbs a flood. Warm a decode pod before adding it
  to the pool — the asymmetry in how the two phases are scored (T12-Q12) creates an asymmetry in how they
  fail when new.

**Signal:** Recognises the inverse signature immediately, names the downstream admission check as the
*design* fix, and connects it to decode cold-start scoring without prompting.

**Follow-ups:**
- *How do you warm a decode pod?* — drive traffic to it before it joins the serving pool.
- *What must admission know about the downstream pool?* — its liveness and its queue depth, not just the
  local one.
- *What's the degraded mode?* — re-route to a collocated replica; the migration must always keep that
  path available.

**Red flags:** Diagnoses it as a prefill problem (the queue is there, but prefill is the victim), or
proposes scaling the prefill pool up.

---

## Whiteboard exercises

### Exercise 1 — Size the split on a single node
**Prompt.** "You run serving for a coding agent on one 8-GPU node, 192 GB HBM per GPU, 2 TB of host DRAM.
Telemetry says ~70% of traffic is agentic and prefill is ~98% of tokens. TTFT p95 is 4 s during
repository-indexing bursts; ITL has barely moved. The CFO will fund a second node if you can prove the
current one is misallocated. Lay out the split, the retention plan, and the first measurement you would
take. Show your arithmetic."

**What the candidate must produce:** the ratio derived from the token mix, a pool layout defended against
availability and burst, the retention tier placed *before* the split, the fabric arithmetic that decides
whether the split is viable at all, and a colocated baseline to measure against.

**Expected answer sketch:**

```
Workload
  agentic share of traffic        ~70%                 [T] Pravin
  prefill share of tokens         ~98%                 [T] Pravin
  TTFT p95 4 s, ITL flat  ->  prefill-bound, not decode-bound

Ratio derivation
  token share      98:2  ~= 49:1 prefill:decode        [D]
  but a 1-GPU decode pool has no availability + no burst headroom
  floor: 2 decode GPUs                                 [D]
  8-GPU node    ->  6P2D,  NOT 4P4D
  (4P4D is a COVERAGE shape, not an optimum)           [T] Chaitanya (CI runs 1P1D/2P2D)

Retention tier -- BEFORE the split
  HBM (active) -> DRAM (pause) -> external (long tail)
  session metadata attached to blocks                  [T] Pravin (retention API, a PR)
  expected: ~5x TTFT on resume vs recompute            [T] Pravin
  arithmetic: 30 pauses x 8k tokens = 240k -> 8k = 96.7% fewer prefill tokens  [D]

Fabric check BEFORE committing
  bytes/token = 2 x n_layers x n_kv_heads x head_dim x dtype_bytes   [D]
  transfer_time = (bytes/token x p95_prompt) / achieved_BW           [D]
  if transfer_time ~ prefill_time  ->  do not split

First measurement
  colocated baseline: TTFT/ITL/session-completion p95, KV hit rate per session
  then the IDENTICAL suite on 6P2D
  watch the crossover: ~2x throughput at ~85 QPS (vendor-affiliated)  [T] Singh
```

**Grading rubric (full marks requires all four):**
- Derives 6P2D (or defends another number with the availability and burst constraints stated explicitly)
  rather than defaulting to the symmetric 4P4D.
- Places the retention tier before the split and gives a risk or diagnosability reason, not just a
  cost reason.
- Does the KV bytes/bandwidth arithmetic before committing, and states the assumption behind
  bytes/token rather than quoting a byte figure as measured.
- Names a colocated baseline and a per-phase counter-metric, and does not offer TTFT alone.

---

### Exercise 2 — Diagnose a degraded migration
**Prompt.** "Two weeks after moving from colocated to a 6P2D split across two nodes, throughput per GPU
has fallen, TTFT p95 is worse, and ITL is now also worse at peak. East-west links are near saturation.
The fleet runs a 256-expert MoE with WideEP, EP8 intra-node. Diagnose it, and tell me what you change,
in what order, and what you re-measure."

**What the candidate must produce:** a ranked differential with a discriminating check for each entry, a
fix with its cost stated, a single-change sequencing discipline, and a confirmation metric.

**Expected answer sketch:**

```
Symptom set
  throughput/GPU DOWN, TTFT DOWN, ITL DOWN
  east-west near saturation

Discriminator: BOTH phases degrade together  ->  shared fabric, not one slow pool
  (ratio problem = one pool saturated, one idle)
  (volume problem = TTFT degrades, ITL holds)

Ranked candidates                          Discriminating check
1. KV transfer + EP all-to-all share one   per-plane BW counters;
   plane                                    NIC queue depth  -> QoS / separate NICs
2. Transfer on the critical path            is transfer overlapped or does decode
   (decode blocks on full KV)               block on the whole KV?  -> async + timeout
3. Ratio wrong for the new mix              rolling utilisation asymmetry > 30%
4. Transfer direction wrong for the         which pool's queue grows?
   bottleneck                               -> read mode when prefill is scarce
5. KV volume too large at p95 prompt        bytes/token x p95_prompt / achieved_BW
                                             -> quantise KV

Order of operations
  isolate planes -> re-measure -> then ratio -> then direction -> then volume
  ONE change at a time; two changes teach you nothing
  counter-metric: throughput PER GPU, not aggregate
```

**Grading rubric:**
- Uses "both phases degrade together" as the discriminator for a shared-fabric diagnosis rather than
  guessing at the most familiar cause.
- Gives a *different* discriminating check for the fabric, ratio, direction and volume hypotheses; the
  checks must be able to distinguish them.
- Proposes plane separation (QoS/NICs) ahead of reducing EP degree, and states the cost of reducing EP
  (KV headroom).
- Commits to one change at a time and a per-GPU counter-metric, and refuses to accept aggregate
  throughput as evidence.

---

### Exercise 3 — Defend the migration to a sceptical head of infrastructure
**Prompt.** "Your head of infrastructure says: 'We tried a distributed project before and it halved
throughput. Show me why disaggregation wins here, on the hardware I already own, with numbers — and tell
me what would make you abandon it.' Argue your case."

**What the candidate must produce:** the mechanism (not a benchmark number) as the lead argument, the
numbers with attribution and their epistemic weight, a risk-ordered sequencing plan, a declared degraded
mode, and an explicit falsifier.

**Expected answer sketch:**

```
The claim under test
  prefill is compute-intensive; decode is memory-bandwidth bound   [T] DeSantis
  "radically different ... at a hardware level"                    [T]
  agentic traffic ~70% of requests, ~98% of tokens are prefill     [T] Pravin
  -> the coupling is the problem, not prefill's speed
  a 2x faster prefill on shared silicon still steals SMs from in-flight decodes

Evidence I will cite, with its weight
  ~5x TTFT on session return from CPU KV offload   [T] Pravin       (vs RECOMPUTE, not vs colocated)
  ~2x throughput at ~85 QPS                        [T] Singh        (vendor-affiliated study)
  2P4D best on the 28k/1k shape                    [T] Chaitanya    PRELIMINARY - author says ignore it
  4 GPU prefill + 4 GPU decode inside ONE node     [T] Singh        (the proving-ground step)
  1P1D / 2P2D                                      [T] Chaitanya    (CI coverage, not an optimum)
  NOT MEASURED IN THIS CORPUS: transfer latency in us; NIXL vs MoRI-O head-to-head

Sequencing -- risk-ordered, and each step isolates one thing
  1. colocated baseline, engine version pinned, numbers recorded
  2. retention tier   (one pool; no new failure domain; higher value; isolates retention)
  3. 6P2D IN-NODE split
  4. cross-node ONLY past the measured crossover (~85 QPS)

What would make me abandon it (falsifier, in measured quantities)
  transfer_time approaching prefill_time on MY fabric, p95 prompt
  short-prompt share high enough that recompute beats transfer for most requests
  goodput below the colocated baseline after two single-change iterations

Degraded mode, declared before the incident
  transfer agent down -> fail the pool back to colocated, not to a stall
  and test that path, or it does not exist
```

**Grading rubric:**
- Leads with the hardware-profile mechanism and the coupling argument, not with a throughput number —
  the number is supporting evidence, not the case.
- Attributes every figure, and volunteers the preliminary and vendor-attributed caveats *unprompted*.
  This is the single highest-value discriminator in the exercise.
- Sequences by risk (baseline → retention → in-node split → cross-node) and says what each step isolates.
- States a falsifier in measurable terms and declares a degraded mode; accepts that reverting is a
  legitimate outcome.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — Pravin (IBM Research). KV create **and** evict events consumed per request by the llm-d/EPP router
  and the live cache-location view they build; prefill filter + prefix-cache scorer versus the decode
  active-request scorer; ~70% agentic traffic and ~98% prefill token share; CPU KV offload saving ~5×
  TTFT on session return; the retention API (a PR) with session metadata; precise versus approximate KV
  routing and the eviction link; P2P KV fetching as the middle ground between queueing and recompute;
  session-aware routing; the 80%-full saturation declaration; MTP and the GLM-5.2 / H100-H200 deployment;
  the 2P2D-versus-3P1D comparison; and MoE routing happening inside the LM, not at the router. ASR
  corrections: "NVIDIA and Excel" → NIXL; "P2P K" → NIXL / P2P KV transfer.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — Chaitanya Sri Krishna. MoRI and MoRI-O as the AMD KV transfer engine with **write and read modes**;
  the 64-GPU GLM-5.1 sweep (78 layers, 256 experts, top-k 8, sparse MLA) from EP4 to EP32 on MI300;
  the 28k-in/1k-out shape with 7.34 M total and ~5 M decode tokens; the drop at concurrency 256 and
  "KV cache recomputation happening every single time"; the ~20k max-concurrency figure for 1P1D; 2P4D
  flagged preliminary; 1P1D and 2P2D as CI coverage; and "distributed inference is becoming more of a
  communication and memory problem … not really related to computation anymore." ASR corrections:
  "Mori" / "MOI IO" → MoRI / MoRI-O; "M355" → MI355X; "YDP" → WideEP.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
  — Abhishek Singh. The single-node "four GPUs run prefill and four GPUs run decode" split; the ~2×
  throughput crossover at ~85 QPS and the "not supposed to be such a drastic improvement" framing; the
  16 req/s on half an H100 enterprise contract; the KV-utilisation knob and the ~90% configuration; and
  the dumb-load-balancer / KV-affinity argument for inference-aware routing.
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt`
  — rack-scale pooled memory over four servers with a mid-rack pool; **memory pooling versus sharing
  mode** and the per-node isolation versus shared-address-space distinction; "faster than RDMA" and the
  TCP/IP → RDMA → pooled-memory ranking; OOM resilience and prefill continuing when decode is behind;
  PCIe/network contention removal; reuse without an additional storing operation; and the comparison
  against Mooncake and a DRAM cache with "very initial performance numbers." Vendor claims; attributed as
  such throughout.
- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
  — the KV connector abstraction and why the movement of KV is "pretty dynamic and complex" across
  prefill→decode and prefill→distributed-store directions; interop with Mooncake; the idle-KV offload to
  CPU memory or disk; and "we never recompute the tokens in the previous turns."
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
  — the collocate → prefill/decode split → prefill/decode disaggregation → **EPD** evolution ladder, and
  the HBM → DRAM → external storage KV tiering. ASR correction: "Ashlan" → SGLang.
- `refs/Agentic_AI_Infra_transcripts/Peter_DeSantis_-_Constraint_Driven_Innovation_A_Look_at_the_AI_Systems_Problem.txt`
  — prefill/encoding as "extremely compute-intensive" versus autoregressive generation as "extremely
  memory-bandwidth [bound]"; "the profile of those two workloads is radically different if you look at it
  at a hardware level"; specialised SRAM-heavy decode silicon and its weakness at long context.
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md` `[R]` — NIXL as
  "a transport layer for moving inference state across memory and network backends", in the prefill-and-
  decode-disaggregation reading list alongside DistServe, Splitwise, Mooncake and Dynamo.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]` — the prefill/decode
  asymmetry and the Prefill-Decode Disaggregation material (advantages, disadvantages, where it is
  overkill, co-located versus disaggregated).
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md` `[R]`
  — gateway responsibilities (context tracker / sticky sessions) and the GPU scaling table, as the
  routing-side counterpart to the transfer plane.

**Note on a corpus mis-attribution:** `TOPICS.md` lists `ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md`
as the `[R]` source for "NIXL and KV transfer". That file contains no NIXL and no KV-transfer content —
it covers deployment options, scaling, cost, monitoring and the accelerator landscape. It is therefore
**not** cited as a NIXL source anywhere in this bank. The reliable NIXL material is the gpu-perf README
above `[R]` plus the llm-d transcript `[T]`, with the ASR correction to "NVIDIA and Excel" noted.

**Derived content in this bank (`[D]`):** the KV transfer volume formula and the Llama-3.1-70B worked
example (~327 KB/token, ~9 GB at 28k); the break-even condition's term-by-term unpacking; the
prefill:decode ratio arithmetic for both corpus data points (49:1 and 1:2.1) and the token-share
versus step-count reconciliation; the 7P1D → 6P2D derivation and the two-GPU decode floor; the
2.66-H100-equivalents sanity check on the 85 QPS crossover; the step ordering in the diagnostic
questions (T12-Q27, Q28, Q30); the ordered mitigation list in T12-Q28; the write-versus-read
backpressure table in T12-Q9; the claim-status sorting in T12-Q26; and the three whiteboard exercise
sketches. All are labelled inline where they appear.

**Not measured in this corpus, and therefore not asserted anywhere above:** KV transfer bytes per token
as a measured quantity, transfer latency in microseconds, any head-to-head of NIXL against MoRI-O, and
any independent (non-vendor) comparison of pooled memory against an RDMA transport.
