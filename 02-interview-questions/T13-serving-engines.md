# Interview Bank: Serving Engines & Orchestration

> `T13` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T13-serving-engines.md) · [Case study](../01-case-studies/T13-serving-engines.md) · [Design blueprint](../03-design-blueprints/T13-serving-engines/HLD.md)
> **Questions:** 30 (9 × L3, 14 × L4, 7 × L5) · **Format:** progressing from the engine landscape, through engine internals and orchestration, to the agent substrate and cross-layer design

## How to use this bank

The questions are ordered so the bank reads as one interview: what each engine bet on, then what
lives inside an engine, then the orchestrator above it, then the agent substrate below it, then the
economics and the design. Ask them in order for a 45-minute loop, or sample the L4/L5 block for a
senior screen.

This topic has an unusual property: the *engine* is the least consequential choice in it. A strong
candidate says so early and spends the rest of the interview on the boundaries between layers. A
weak one argues about benchmark tables.

Every number here is traceable to a transcript (`[T]`, speaker named), a supporting repo (`[R]`,
path named), or is my own derivation (`[D]`, assumptions shown). Vendor and press claims are marked
as such and never presented as measured results. The corpus contains **no** head-to-head vLLM vs
SGLang vs TensorRT-LLM comparison on identical hardware and models, and no cost-per-session figure
for an agent platform; if a candidate asserts either, ask where it came from.

---

### The landscape — what each engine bet on

#### T13-Q1 · Name the engines, and the one bet each made
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Give me the serving-engine landscape in one pass. For each engine, what is the single
idea it is built around, and who is it for?

**Model answer:** Eight names, and each one is a different bet:

- **vLLM** — paged attention plus continuous batching, behind a plugin core. The bet is **breadth**:
  most models, most hardware, most features `[T]` Kwon. Its distinguishing virtue is operational —
  Prasad's framing is that vLLM leads "not about the performance… ease of use. If you look at the
  other inference engine, they are very difficult to configure" `[T]` Prasad.
- **SGLang** — **RadixAttention**, a radix tree over KV prefixes, extended by HiCache tiering. The
  bet is **prefix reuse**, which is why it wins on multi-turn and few-shot `[R]`
  (`llm-inference-engineering-main/README.md`).
- **TensorRT-LLM** — **ahead-of-time engine build**. The bet is **peak performance on a fixed
  NVIDIA configuration** `[R]` (same).
- **llm-d** — not an engine at all: a **distributed serving stack** above the engine, turning
  "many inference servers into one" `[T]` Prasad.
- **KServe** — Kubernetes-native model lifecycle; complements llm-d rather than competing `[T]`
  Pravin.
- **Ray Serve** — general Python distributed serving; you build the LLM optimisations.
- **TGI** — HuggingFace-native; less momentum than vLLM/SGLang.
- **llama.cpp / Ollama** — CPU and edge via GGUF; laptops and single-user, not datacentre
  throughput `[T]` Prasad.

The durable point is that these are not ranked; they occupy different niches, and the layering
(engine under distributed stack under gateway) is what lets you swap any one of them.

**Signal:** Groups the engines by *bet* rather than reciting names, and volunteers that llm-d is
not an engine before being corrected.

**Follow-ups:**
- *Which two are genuinely substitutable for each other?* — vLLM and SGLang; T13-Q15.
- *Why is llm-d in the same list as vLLM but a different layer?* — T13-Q6.
- *Where is Ollama correctly deployed?* — T13-Q5.

**Red flags:** Ranks the engines by a benchmark table without naming a workload, or treats llm-d as
an engine.

---

#### T13-Q2 · Why does vLLM lead on ease of use rather than raw performance?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** vLLM is the default choice for most teams. Is it because it is the fastest? Explain
what it actually wins on.

**Model answer:** No — and the corpus is explicit that this is not the axis. Prasad's framing is
that vLLM leads "not about the performance… ease of use. If you look at the other inference engine,
they are very difficult to configure" `[T]` Prasad.

Three things constitute that ease of use, and they are structural rather than cosmetic.

**One: the entry points are a single command.** Online serving is `vllm serve` — one command to an
OpenAI-compatible *and* Anthropic-compatible endpoint, and "any agent framework that speaks the
OpenAI or Anthropic API can work out of the box" `[T]` Kwon. That is an integration-cost decision,
not a performance one.

**Two: onboarding is a repository, not a research project.** Prasad points at the recipes
repository: "if you want to run some model, this is the place where you can go and get the
parameters" `[T]`.

**Three: the hardware matrix is broad** — more than 10 backends behind a plugin structure `[T]`
Kwon — so the engine is rarely the reason a model cannot be served.

The counterpoint a strong candidate adds: **the V2 engine is now the default for single-replica
serving** `[T]` Singh, which means the "one replica, maximum throughput" case is no longer something
you hand-tune. The performance argument for picking a different engine has narrowed to specific
workloads (T13-Q13), not to the general case.

**Signal:** Names ease of use as the axis and gives two or more structural reasons, rather than
saying "it's popular."

**Follow-ups:**
- *What are the two entry points?* — T13-Q7.
- *Where does the performance argument still bite?* — structured output; T13-Q13.
- *What is the cost of that breadth?* — many knobs, throughput-tuned defaults.

**Red flags:** Claims vLLM is fastest, or cites GitHub stars as the reason.

---

#### T13-Q3 · RadixAttention versus a prefix cache
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** SGLang's core idea is RadixAttention. What is it, and how is it different from just
caching the system prompt?

**Model answer:** RadixAttention stores prefixes in a **radix tree keyed by token sequence**. A new
request walks the tree and reuses the **longest matching prefix** `[R]`
(`llm-inference-engineering-main/README.md`).

The difference from a flat prefix cache is generality, and it is the whole point. A system-prompt
cache matches one exact string. A radix tree matches **any common prefix** — so two different
conversations that happen to share a document, a few-shot exemplar, or a set of tool definitions
share the KV for the shared part even though neither request is identical to the other.

That is why the corpus attributes SGLang's strength on **multi-turn and few-shot** workloads to this
mechanism `[R]`. In a multi-turn agent session, each turn extends the previous prefix; a radix tree
makes turn *n* cost the delta rather than the whole conversation. In few-shot, many requests share
the exemplar block and diverge only in the query.

**HiCache** is the extension: it pushes the same radix tree across **GPU → CPU → storage**, so a
prefix evicted from HBM is recovered from DRAM rather than recomputed `[R]`. That is the
engine-level version of KV tiering, and it composes with — but does not replace — a separate KV store
(T12).

The honest limit: a radix tree only pays when prefixes actually repeat. For single-turn, unique-prompt
traffic, the tree is bookkeeping with no hits.

**Signal:** Says "any common prefix, not exact match" and gives a concrete workload where that
matters; adds HiCache as the tiering extension.

**Follow-ups:**
- *What workload would show no benefit?* — unique single-turn prompts.
- *How does HiCache relate to a KV store?* — composes with it; different tiers; T12.
- *Why does this make SGLang a multi-turn engine?* — prefix shape matches agent sessions.

**Red flags:** Describes it as an exact-match prompt cache, or claims it helps every workload.

---

#### T13-Q4 · What does "ahead-of-time engine build" actually cost you?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** TensorRT-LLM makes you compile an engine before you serve. Explain what that buys and
what it charges, and tell me when you would accept the charge.

**Model answer:** The model is "prepare the model ahead of time instead of figuring things out on
the fly" — a build step that produces an engine with kernel fusion, custom attention kernels, a
paged KV cache, in-flight batching, CUDA graphs and speculative decoding baked in `[R]`
(`llm-inference-engineering-main/README.md`).

What it buys is the best achievable kernels for that exact configuration. The guide's claim is the
highest peak tokens/sec per dollar on H200/B200/B300 for hand-tuned models `[R]` (`04/06`) — a
vendor-and-guide claim, not an independently measured result, and I would say so.

What it charges is operational, and it is threefold `[D]`:

1. **A build per configuration.** The engine is bound to a model, a GPU, a parallelism setting, and
   often a batch shape. Every new model needs a multi-hour, model-and-GPU-specific compilation.
2. **Tight version pinning.** Upgrading is a rebuild, not a redeploy.
3. **No exit without a re-platform.** There is no path off CUDA without a full re-platform `[R]`
   (`04/06`).

Prasad's summary is the same observation from a user's seat — "the other inference engine, they are
very difficult to configure" `[T]`.

So I would accept the charge under exactly three conditions together: one or two flagship models that
will not rotate for a year or two, a committed NVIDIA fleet, and a need for every last token/sec. It
**breaks when the model portfolio rotates faster than the build pipeline** can absorb — which is the
normal state for a team tracking open-weight releases.

**Signal:** Names the build-per-configuration cost as the primary charge rather than just "it's
NVIDIA-only," and states the three conditions jointly.

**Follow-ups:**
- *Is the engine a drop-in after a shape change?* — no; config-bound; see the failure table.
- *What is the exit cost?* — a full re-platform off CUDA.
- *Which corpus workload would justify it?* — one flagship model, fixed fleet; T13-Q14.

**Red flags:** Calls it simply "faster," or claims you can swap models without a rebuild.

---

#### T13-Q5 · Where do Ollama and llama.cpp belong, and where do they not?
**Difficulty:** L3 · **Depth expected:** 90 s

**Question:** A team proposes Ollama for a production multi-tenant inference service because "it was
easy to get working locally." Respond.

**Model answer:** Ollama and llama.cpp occupy a real and legitimate niche — CPU and edge inference
over GGUF quantised weights: developer laptops, single-user local use, air-gapped or private
deployments where one user is the whole load.

They are the wrong tool here, and the corpus gives the precise reason. Prasad: **"Ollama is meant
for desktops… where your batch size is equal to zero or one"** `[T]` Prasad. There is no continuous
batching at scale. Batching is the mechanism that makes datacentre inference economically coherent —
it is what amortises weight reads across concurrent requests — and a batch-size-one engine gives it
up entirely.

So the failure is not gradual. It **breaks immediately under concurrent production load**: the first
few simultaneous requests do not degrade throughput, they destroy it, because the engine was designed
for a queue of one.

The productive answer is to distinguish the two requirements that got conflated. "Easy to run
locally" is a *developer experience* requirement and is genuinely met. "Serve many concurrent
tenants" is a *throughput* requirement and is not. The right response is to let developers keep
Ollama on their laptops and serve production on vLLM — the OpenAI-compatible API surface means the
application code does not change between the two.

**Signal:** Names batch size as the mechanism rather than saying "it's not production-grade," and
separates the dev-experience requirement from the throughput requirement.

**Follow-ups:**
- *What specifically does batching buy?* — amortising weight reads across requests.
- *Why is the API-compatibility point load-bearing?* — it makes the dev/prod split free.
- *What about edge deployment at scale?* — still batching-limited; one user per device.

**Red flags:** Says "Ollama is bad," or proposes it for any multi-tenant service.

---

#### T13-Q6 · Separate the three layers, and say what state each owns
**Difficulty:** L4 · **Depth expected:** 4 min

**Question:** You are whiteboarding a serving stack. Draw the layers and tell me what state each one
owns. Then tell me where the design fails.

**Model answer:** Three layers, three owners, and the ownership question is the useful part.

**The engine** (vLLM/SGLang/TensorRT-LLM) owns GPU memory, the KV cache, batching, and the model's
execution. It owns state **within a request and within a replica**.

**The orchestrator** (llm-d) owns placement, admission, and the cluster-level view of KV. It owns
state **about replicas** — which blocks exist where, which pods are saturated `[T]` Pravin.

**The gateway** owns auth, quotas and semantic routing — and that is a separate concern with its own
topic (`T14`). It owns state **about tenants and policy**.

**KServe** sits alongside as the model lifecycle and deployment layer, and Pravin is explicit that
llm-d does not replace it: "KServe is more for deploying the models… llm-d works with KServe. So it's
not something that replaces KServe. So Tesla in fact uses KServe with llm-d" `[T]`.

**The substrate** owns session state — the sandbox, its filesystem, its identity.

Two framings worth saying out loud. Prasad's analogy: "If you compare vLLM with Docker, they are the
same… But when you want to scale, from one system to ten system to a hundred system, that time you
will require the orchestration" `[T]`. And the design failure: **no layer owns the session
end-to-end.** The engine drops it between turns, the router never had it, and the substrate only
exists if you built one. That gap is where the money leaks (T13-Q30).

**Signal:** Assigns state ownership before naming any product, and volunteers that the session spans
no single layer.

**Follow-ups:**
- *Which layer does model selection belong to?* — neither; llm-d is a performance router `[T]`.
- *What breaks if you couple engine and orchestrator?* — you cannot change either; T13-Q18.
- *Where do you draw the gateway boundary?* — T14.

**Red flags:** Draws boxes labelled with product names and no state ownership, or puts model selection
inside llm-d.

---

### Inside the engine

#### T13-Q7 · vLLM has two entry points. Which one when?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** vLLM can be used two ways. Name them, say what each is for, and explain why the
distinction matters operationally.

**Model answer:** Two entry points `[T]` Kwon.

**The `LLM` class (Python)** — offline batch inference. "You give it a Hugging Face model name and
you call the `generate`" and vLLM handles model loading, optimisation, scheduling and memory
management under the hood `[T]`. This is the path for bulk jobs: scoring a corpus, generating a
synthetic dataset, offline evaluation.

**`vllm serve`** — online serving. "A single command" that gives you an OpenAI-compatible endpoint,
and vLLM "also support[s] Anthropic APIs too — any agent framework that speaks the OpenAI or
Anthropic API can work out of the box" `[T]` Kwon.

Why the distinction matters: they have different SLOs and therefore different tuning. Offline batch
cares about throughput per GPU and can tolerate a queue; online serving cares about TTFT and ITL
distributions per request. The engine exposes different levers in each mode, and a flag tuned for one
is usually wrong for the other.

A second reason, specific to this corpus: the API layer "is pretty much similar to the few years
ago… it actually didn't really change that much for agents. What has changed a lot for agent is
everything underneath it" `[T]` Kwon. The stable API surface is exactly what makes the two entry
points interchangeable from the client's perspective.

ASR note: the transcript renders "vLLM" as "VLM" and the class as "the EDLM class"; both are
corrected here.

**Signal:** Names both entry points by function (offline batch / online serving) and connects the
distinction to differing SLOs.

**Follow-ups:**
- *Which one is the agent-facing path?* — `vllm serve`; OpenAI/Anthropic compatibility.
- *Why does the stable API surface matter?* — it lets the layer underneath change freely.
- *How does the V2 engine change this?* — default for single-replica serving `[T]` Singh.

**Red flags:** Knows only `vllm serve`, or treats the two as interchangeable for tuning purposes.

---

#### T13-Q8 · Why does vLLM support more than 10 hardware backends?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** vLLM supports more than 10 hardware backends `[T]`. What architectural property makes
that tractable, and what does that property not buy you?

**Model answer:** The property is **a plugin structure over one shared core** `[T]` Kwon. A new
accelerator implements a **bounded interface** rather than forking the engine. Everything above that
interface — the API layer, scheduling, the KV abstractions — is shared, so a backend author is
responsible for kernels and collectives, not for re-implementing a serving engine.

That is the architectural achievement the cheat sheet calls out explicitly: it is the reason more
than 10 backends are supportable, and it is worth copying if you build internal serving `[T]`/`[D]`.

What it does **not** buy is performance parity, and this is where a weak candidate over-reads the
number. A backend that exists is not a backend that is optimised. The corpus's own warning is that
**bringing up new hardware means re-taking the whole stack** — kernels, attention, quantization,
collectives, MoE — and to budget **months, not weeks** `[T]` Kwon. The plugin boundary makes the
engine *loadable* on new silicon; it does not make it *fast* there.

So the correct reading of ">10 backends" is a statement about **portability**, not about
performance. It supports the heterogeneity strategy in T11 (the API layer is standardised, the
hardware underneath is not) — a fleet can be multi-vendor without being multi-API.

The practical consequence: when you choose an accelerator, the question is not "does vLLM support
it" but "how mature is the backend," and maturity is measured in merged work, not in the existence of
a plugin (T13-Q9).

**Signal:** Says "bounded interface over a shared core" and immediately qualifies it — portability,
not parity.

**Follow-ups:**
- *What does a backend author actually implement?* — kernels and collectives, not the engine.
- *How does that interact with the multi-vendor requirement?* — the API stays standard.
- *What is the cost of bring-up?* — T13-Q9.

**Red flags:** Reads ">10 backends" as "performance is equivalent everywhere," or believes a plugin
means no porting work.

---

#### T13-Q9 · What does bringing up a new accelerator actually cost?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Your procurement team wants to add a second accelerator vendor. What does the
engineering bring-up involve, and how would you cost it into the decision?

**Model answer:** The corpus is unusually blunt: **"the whole stack must be re-taken from the ground
up"** — kernels, attention, quantization, collectives, MoE, all of it — and the budget is **months,
not weeks** `[T]` Kwon. Coding agents make that easier without removing it.

I would decompose the cost into four lines `[D]`:

1. **Kernel coverage.** The attention variants and quantisation formats your models actually use must
   exist and be correct on the new silicon, not merely compile.
2. **Collectives.** The parallelism plan from T11 depends on all-reduce, all-to-all and
   point-to-point working at the topology's shape. This is where MoE workloads fail first — the
   corpus's own failure signature is "MoE slow on AMD vs NVIDIA," checked by looking at WideEP
   enablement gaps and pending PRs `[T]` ROCm/WideEP.
3. **Verification.** You must re-establish correctness, not just speed: the long-context regression
   suite, output comparisons against the incumbent, and the accuracy effect of any new quantisation.
4. **Operational surface.** Monitoring, autoscaling signals, and driver/firmware lifecycle for a
   second vendor.

The decision rule I would give procurement: **a second vendor costs months of engineering before it
saves anything**, so it must be justified by a constraint that a single vendor cannot meet —
sovereignty, supply, or price at scale — rather than by a benchmark. And the benchmark itself must
name its stack, because a performance result is a property of a stack, not of silicon `[D]`.

The good news is the architecture is already shaped for this: the plugin core (T13-Q8) means the
bring-up is a backend effort, not an engine fork.

**Signal:** Quotes the "whole stack from the ground up, months not weeks" framing and decomposes it
into named workstreams rather than treating bring-up as a port.

**Follow-ups:**
- *Which workload fails first on new silicon?* — MoE; the WideEP enablement gap `[T]`.
- *What must accompany the benchmark you use to justify it?* — stack versions and workload shape.
- *What makes it cheaper than it used to be?* — coding agents, without removing the work `[T]`.

**Red flags:** Calls it a "port," or budgets weeks, or justifies the decision on a single benchmark.

---

#### T13-Q10 · What does the KV connector abstract, and what does it let you swap?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** vLLM exposes a "KV connector." What is it abstracting, and what design freedom does
that abstraction buy?

**Model answer:** The KV connector is a **transport-agnostic abstraction over KV movement** — the
corpus names NIXL, MoRI-O, and stores like Mooncake as the things behind it `[T]` Kwon. It is the
seam between the engine and the KV-transfer layer that T12 covers.

What it abstracts is *how bytes get from one place to another*: peer-to-peer over RDMA, a shared
memory pool, or an external store. What stays constant above the seam is the engine's own logic —
which blocks exist, when they are evicted, what a prefix hit means.

That separation is what lets you change the transfer mechanism without changing the engine. It
matters for three concrete decisions:

**Disaggregation.** Moving KV between a prefill pool and a decode pool is a transfer problem, and the
connector is the interface you configure rather than code against (T12).

**Prefix reuse across replicas.** If the orchestrator routes by KV locality (T13-Q17), the transport
underneath must be able to actually move or share that KV.

**Vendor portability.** The connector is one of the places a new accelerator's bring-up lands
(T13-Q9), because the transport primitives differ per vendor.

The honest limit, and the thing a strong candidate volunteers: **the connector abstracts transport,
not layout.** Two engine versions with different KV layouts produce events and blocks that are not
comparable, which is exactly the version-skew failure in T13-Q18. A transport-agnostic abstraction
does not make the data above it version-agnostic.

**Signal:** Separates what is abstracted (transport) from what is not (layout, block semantics), and
connects it to disaggregation.

**Follow-ups:**
- *Which layer consumes these events?* — llm-d's data layer; T13-Q20.
- *What does the abstraction not solve?* — layout compatibility across versions; T13-Q18.
- *Why does it matter for a multi-vendor fleet?* — transports are vendor-specific; T13-Q9.

**Red flags:** Describes it as a cache, or claims it makes KV portable across engine versions.

---

#### T13-Q11 · SGLang's machinery beyond RadixAttention
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** RadixAttention is SGLang's headline idea. Name the other pieces of its machinery and
say what each one is for.

**Model answer:** Four, per Zhu's talk `[T]` Zhu, and they are not variations on one theme — each
attacks a different resource.

**HiCache** — "enable people to move the KV cache down from HBM to DRAM and even to your external
storage." This is the engine-level version of KV tiering, the same three-level logic as T07 and
T12 §4.4, implemented inside the engine rather than as an external store. It buys effective KV
capacity at the cost of a slower tier.

**HiSparse** — a sparse-attention optimisation that processes "the full KV with a hot buffer" to cut
memory and raise throughput. The bet is that most of a long context does not need the same treatment
as the hot part.

**Spec V2 / overlap scheduler** — scheduling designs that **overlap** to raise throughput. This is a
latency-hiding play, not a memory play: the point is to keep the device busy while something else is
in flight.

**Chunk pipeline parallelism** — chunking a long prompt and processing the chunks in parallel, "plus
a second chunking dimension across GPUs." Both are TTFT plays on long prefill, and the second is the
same intuition as the prefill-chunking use of pipeline parallelism in T11.

The framing I would offer: **RadixAttention is the bet; these are the extensions that keep the bet
paying as context grows.** HiCache answers "the KV does not fit," HiSparse answers "the KV is too
expensive to read," the overlap scheduler answers "the device is idle," and chunk pipelining answers
"the prompt is too long to prefill serially."

**Signal:** Assigns each mechanism to a distinct bottleneck rather than listing four features, and
notes that two of them are TTFT plays.

**Follow-ups:**
- *Which one is a capacity play?* — HiCache; it is tiering.
- *Which mirrors a mechanism from T11?* — chunk pipeline parallelism ↔ PP for prefill chunking.
- *Does HiCache replace a KV store?* — composes with it, does not replace it `[R]`.

**Red flags:** Treats these as synonyms for prefix caching, or claims HiSparse is a quantisation.

---

#### T13-Q12 · Read the "2.2× on GLM-5.2" claim properly
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** SGLang's team reports "over 2.2× improvement and up to 500 tokens per second per user"
on GLM-5.2 `[T]`. What does that number actually license you to conclude?

**Model answer:** Almost nothing on its own, and the reason is a specification detail the corpus
makes explicit: it was **measured against SGLang's own day-zero baseline** `[T]` Zhu. It is a
**version-over-version** improvement — SGLang got 2.2× better than SGLang — not a head-to-head
against vLLM or TensorRT-LLM. It is a **vendor claim**, and it should be labelled as one.

So it licenses exactly one conclusion: the engine improved substantially on that model. It does not
license "SGLang is 2.2× faster than vLLM," and a candidate who draws that conclusion has misread a
self-comparison as a competitive one.

Three questions I would ask of *any* engine number, and they are the substance of this answer `[D]`:

1. **Against what baseline?** Self-baseline, incumbent version, or a competing engine on the same
   hardware? The corpus's 2.2× is the first; the ~29% structured-output result (T13-Q13) is a
   competitive one, and correspondingly harder to obtain.
2. **At what workload shape?** Prompt lengths, concurrency, output lengths, and the prefill/decode
   mix. A throughput figure without a shape is not reusable.
3. **Who measured it, and on what stack?** Vendor-published, vendor-affiliated, or independent — plus
   the versions. A performance result is a property of a stack, not of silicon `[D]`.

And the honest boundary: **this corpus contains no head-to-head vLLM/SGLang/TensorRT-LLM comparison
on identical hardware and models.** The case study says so explicitly. If a candidate asserts one,
they are citing something outside this corpus.

**Signal:** Identifies it as a self-baseline version-over-version claim and refuses the competitive
reading; supplies a general rubric for reading engine numbers.

**Follow-ups:**
- *What would have made it a competitive claim?* — the same model, hardware and shape across engines.
- *Which corpus figure is competitive?* — the ~29% structured-output advantage; T13-Q13.
- *Does that make the 2.2× useless?* — no; it is a real improvement rate, correctly scoped.

**Red flags:** Repeats 2.2× as a reason to switch engines, or cannot say what the baseline was.

---

### Choosing and running an engine

#### T13-Q13 · The ~29% structured-output result — what does it justify?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** SGLang v0.4.3 is reported at **~29% throughput advantage over vLLM on structured-output
and function-calling workloads** `[R]`. Is that enough to justify running two engines?

**Model answer:** It is the strongest published reason to run two engines, and it is also a
vendor-published benchmark — both halves matter.

**What it is.** Attributed to async constrained decoding, on a specific workload class
(structured output / function calling), at version v0.4.3 versus the contemporaneous vLLM `[R]`
(`04/06`, citing an April 2026 SGLang blog). It is a *competitive* comparison, which already makes it
more useful than the self-baseline 2.2× claim (T13-Q12) — but it is still published by the vendor
whose engine wins.

**What it justifies.** Not "switch to SGLang." It justifies a **second pool for one workload class**.
The guide's posture is explicit: "production traffic on vLLM, 1–5% canary on SGLang or TensorRT-LLM,
alert on quality or latency divergence" `[R]` (`04/06`). The case study adopts exactly that — vLLM as
the default, SGLang as a canary for the structured-output path — and names the revisit trigger:
**when structured output exceeds ~30% of traffic, it earns its own pool.**

**Why two engines is the right answer rather than one.** The 29% is workload-specific. Paying a
second engine's costs (a second support matrix, a second CVE feed, a second pinning discipline) for
a general 5% would be bad economics; paying them for a 29% win on a workload class you can *route to*
is defensible.

**Why the canary framing matters.** A 1–5% canary is how you convert a vendor claim into your own
measurement — and it gives you the quality-divergence alarm that the same guide asks for.

**Signal:** Separates "vendor-published but competitive" from "measured," and lands on a second pool
for a workload class rather than an engine migration.

**Follow-ups:**
- *What is the revisit trigger?* — structured output above ~30% of traffic.
- *What is the cost of the second engine?* — support matrix, CVE feed, pinning; T13-Q15.
- *How is this different from the 2.2× claim?* — competitive baseline versus self-baseline; T13-Q12.

**Red flags:** Migrates the whole fleet on one vendor benchmark, or dismisses it because it is
vendor-published.

---

#### T13-Q14 · When is TensorRT-LLM the right call?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** Give me the conditions under which you would put TensorRT-LLM in production, and the
condition under which you would take it back out.

**Model answer:** In: **one or two flagship models, a committed NVIDIA fleet for roughly two years,
and a need for every last token per second** `[D]` — the case study's formulation, and it follows
directly from the build cost.

The justification is peak performance. The guide's claim is the highest peak tokens/sec per dollar on
H200/B200/B300 for hand-tuned models `[R]` (`04/06`) — I would mark that as a vendor-and-guide claim
rather than an independent measurement, and note that the corpus has no head-to-head to confirm it.

Out: **when the model portfolio rotates faster than the build pipeline** can absorb it. That is the
exception row in the case study's decision table, and it is the normal state for a team tracking
open-weight releases. A multi-hour, model-and-GPU-specific compilation per configuration is fine at
two models a year and untenable at two a month.

Three secondary conditions worth naming `[D]`:

- **Configuration stability.** The engine is bound to model, GPU, parallelism and often batch shape,
  so a workload whose shape moves (a new context length, a new concurrency regime) is a rebuild.
- **The exit is a re-platform.** There is no path off CUDA without one `[R]`, so this is a two-year
  commitment, not a quarterly experiment.
- **The team must own a build pipeline.** The build step *is* the product (T13-Q4); if nobody owns
  it, it becomes an outage on upgrade day.

And the migration path in and out: run it as a 1–5% canary beside vLLM, exactly as T13-Q13 describes,
so that entering and leaving are both routing changes rather than projects.

**Signal:** States both the in-conditions and the out-condition, and frames the commitment as
two-year because of the CUDA lock-in.

**Follow-ups:**
- *What has to exist before you adopt it?* — an owned build pipeline.
- *How do you migrate in?* — 1–5% canary; T13-Q13.
- *What marks the claim as unverified?* — no head-to-head in this corpus; T13-Q12.

**Red flags:** Recommends it for a rotating open-weight portfolio, or ignores the build pipeline
ownership question.

---

#### T13-Q15 · One engine, or two?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** You have to decide whether your platform runs one engine or two. Argue both sides and
give me a decision rule that survives a workload change.

**Model answer:** I would argue it as a **workload-class** decision, not a platform-philosophy one,
because that is the only framing that gives a rule instead of a preference.

**The case for one.** A single engine is one support matrix, one CVE feed, one version-pinning
discipline, one set of KV semantics, and one thing to upgrade. Cross-engine correctness disappears as
a category. Given that the engine is the least consequential layer (T13-Q30), consolidating it is
consistent with where the value actually is.

**The case for two.** The measured reason is specific: a ~29% throughput advantage on
structured-output and function-calling workloads `[R]` (`04/06`). That is not a general speedup — it
is a *class* speedup, and the architecture can route to it. So the second engine earns its place only
if (a) the workload class exists at material volume, (b) the routing can actually direct traffic to it
(T14), and (c) you have the canary discipline to detect quality divergence.

**My decision rule.** Run one engine by default. Add a second only for a **named workload class**
whose measured gain exceeds the cost of the second support matrix, introduced as a **1–5% canary**
with alerting on quality and latency divergence `[R]` — and give the second engine an expiry: if the
class does not grow past roughly 30% of traffic (T13-Q13) *and* the primary engine closes the gap,
retire it.

**Why the rule survives a workload change.** It is stated in terms of a workload class and a measured
gain, so a shift in traffic re-opens the question automatically rather than silently invalidating a
decision. That is the same discipline T13-Q21 applies to llm-d: the trigger is a measured position,
not a date.

**At 10×**, the case study's prediction applies: consolidation *within* tiers and diversification
*across* them. Two engines across three workload classes is manageable; ten across fifty is not
(T13-Q29).

**Signal:** Frames it as workload-class + measured gain + canary + expiry, rather than "two engines is
more flexible" or "one engine is simpler."

**Follow-ups:**
- *What makes the second engine's gain specific rather than general?* — T13-Q13.
- *What is the expiry condition?* — the class fails to grow and the primary closes the gap.
- *How does this change at 10×?* — consolidation within tiers; T13-Q29.

**Red flags:** Argues from flexibility with no measurement, or runs two engines with no canary and no
expiry.

---

#### T13-Q16 · A CVE lands in a code path you use
**Difficulty:** L5 · **Depth expected:** 4 min

**Question:** An advisory lands: your engine has an unpatched vulnerability in the multimodal and
disaggregated-prefill paths. Text-only is unaffected. Walk me through the next 72 hours.

**Model answer:** The corpus contains this exact case, and the recorded real-world response is the
part worth knowing: **deployments moved multimodal traffic back to vLLM and kept SGLang for text-only
function calling** `[R]` (`04/06`, as of May 2026; vLLM ≥ v0.18.2 is required for multimodal on the
vLLM side).

The generalisable lesson: **the mitigation is to move traffic, not merely to pin.** Pinning protects
you only if the vulnerable path is unreachable, and an advisory is not a promise that nobody has
found the path yet. So the capability you need *before* the incident is a live migration path.

Concretely, in order `[D]`:

1. **Establish reachability.** Which of my pools serve multimodal or disaggregated prefill? If none,
   the incident is a scheduling problem, not an outage.
2. **Route away.** Move the affected traffic to a second engine or a patched pool using the routing
   layer, not a redeploy. This is why the engine-per-workload split (T13-Q15) is a resilience
   decision as much as a performance one.
3. **Rotate credentials and identity** for anything on the affected path, and treat the sandbox
   boundary as potentially crossed if the path is agent-facing (T13-Q24).
4. **Patch on the advisory cadence, not the release cadence.** The platform's requirement is ≤1 week
   from upstream CVE to patched in prod `[R]` — advisory feeds, not release notes.
5. **Move traffic back deliberately**, with the canary alerting on quality and latency divergence.

The architectural preconditions are what I would actually grade: *can you route by path, not just by
model?* and *do you have enough headroom elsewhere to absorb the moved traffic?* Without both, "move
the traffic" is advice you cannot execute.

**Signal:** Leads with "move traffic, not pin" and names the recorded precedent; lists the
preconditions (path-level routing, headroom) rather than only the steps.

**Follow-ups:**
- *Why is pinning insufficient?* — it protects only while the path is unreachable.
- *What routing capability does this require?* — path- or route-level, not per-model; T14.
- *What is the patch cadence requirement?* — ≤1 week, advisory-driven `[R]`.

**Red flags:** Pins and waits, or proposes a full-fleet redeploy as the mitigation, or has no
second path.

---

#### T13-Q17 · Version pinning — what exactly is the tuple?
**Difficulty:** L4 · **Depth expected:** 3 min

**Question:** "Pin your versions" is advice everyone repeats. Pin *what*, exactly, and what breaks if
you do it per-model instead of per-deployment?

**Model answer:** The tuple the guide gives is: **a model is "Llama-X on vLLM vN with config C on
hardware H"** `[R]` (`04/06`). Four elements — model revision, engine version, configuration, and
hardware — and the reason all four belong in one string is that any of them changing invalidates the
others' measured behaviour.

The default policy: **pin the full tuple per deployment, patch on a schedule**, driven by advisory
feeds on a ≤1-week cadence `[R]`.

Three strategies and their failure modes `[R]`:

- **Track upstream latest.** Always patched, but constant churn, and it **breaks during a CVE week
  when three patches land at once** — the exact week you least want a surprise regression.
- **Pin per deployment, patch on a schedule.** Reproducible; the default. Its cost is that you are
  exposed between patches, which is a deliberate accepted risk.
- **Pin per model, allow divergence.** Each model optimised independently — but **cross-model bugs
  are hidden and you maintain two support matrices**. The guide allows it only with per-model owners.

Why per-model divergence is the trap: the pinning strategy is what makes a configuration
*reproducible*, and reproducibility is the input to every other discipline here. If two models on one
pool carry different engine versions, then (a) their KV layouts may differ (T13-Q18), (b) an incident
can no longer be diagnosed by version, and (c) the canary comparison in T13-Q13 loses its baseline.

**Revisit if** the engine ships a stable LTS line with backported security fixes, which changes the
cadence arithmetic entirely `[D]`.

**Signal:** Names all four elements of the tuple and explains why per-model divergence breaks
reproducibility rather than just calling it untidy.

**Follow-ups:**
- *What drives the patch cadence?* — advisory feeds, ≤1 week `[R]`.
- *Why is per-model pinning risky?* — cross-model bugs hidden; two support matrices.
- *What changes the calculus?* — an LTS line with backports.

**Red flags:** Pins only the engine version, or pins per model without naming the cost.

---

#### T13-Q18 · Engine version skew across replicas
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Two replicas in the same pool are running different engine versions. The service looks
healthy. What is actually broken?

**Model answer:** The routing is broken, silently, and it is broken in the way that is hardest to
notice: it still works.

The mechanism: **the router scores on KV events** — vLLM pods emit KV create and evict events and the
llm-d data layer maintains "what all cache is available in the vLLM, which pods are saturated" `[T]`
Pravin. If two replicas have **different KV layouts**, their events are not comparable, so the
router's prefix-locality decisions become approximate. The visible symptom is exactly the failure
llm-d exists to prevent: **prefix cache hit rate falling**, with the cost consequence that follows.

The fix is structural, not a tuning knob: **pin the engine as one deployable unit across the pool.**
The pool is the unit of version homogeneity; a mixed pool is not a pool, it is two pools sharing a
router.

Three second-order points a strong candidate adds `[D]`:

- **Detect it.** Compare KV event rate against request rate; a divergence is the earliest signal
  (T13-Q22). Output-hash comparison across replicas catches the *correctness* half, which is the
  model-registry-drift failure: two "same" models at different revisions produce different KV layouts
  and different outputs `[T]`/`[D]`, and the symptom is silent correctness, not latency.
- **Roll forward, not sideways.** The safe upgrade is a canary pool at the new version with the old
  pool pinned, compared on quality and latency `[R]` — not two versions interleaved behind one router.
- **The registry is the source of truth for revisions**, which is KServe's role in the llm-d split
  (T13-Q23), not the router's.

**Signal:** Says the router degrades silently and names the KV-event comparability as the mechanism,
rather than answering "inconsistent behaviour."

**Follow-ups:**
- *What is the first metric that moves?* — prefix cache hit rate / KV event rate; T13-Q22.
- *How do you upgrade safely instead?* — canary pool, not interleaved versions.
- *Who owns revision truth?* — the registry; KServe; T13-Q23.

**Red flags:** Says "it's fine as long as both work," or proposes fixing it in the router.

---

### Orchestration — llm-d, KServe and the layer above the engine

#### T13-Q19 · What does llm-d add over a Kubernetes Service?
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** You have vLLM replicas behind a plain Kubernetes Service. Someone proposes llm-d. What
specifically does llm-d add that the Service cannot do?

**Model answer:** llm-d adds **KV awareness**, and the harm its absence causes is specific and
measurable.

A Kubernetes Service distributes by connection or round robin with **no knowledge of the KV cache**.
The failure that produces is the one Singh draws out in detail: a **three-turn conversation whose
prefix is recomputed three times**, with the cache hit rate "atrocious" in Prometheus `[T]` Singh.
Nothing is broken — the service is healthy, the requests succeed — but you are paying for the same
prefill three times, once per turn.

llm-d is "a native Kubernetes stack for distributed LLM inference," built on "the Gateway API
inference extension of Kubernetes," in CNCF, "in collaboration with Google, CoreWeave, Nvidia, Red
Hat" `[T]` Pravin. Prasad's one-liner: it **"turns many inference servers into one"** `[T]`.

Two boundaries worth stating, because both are commonly got wrong:

- **It is a performance router, not a model selector.** "llm-d is not concerned with the difference in
  the qualitative performance of these models — it's a performance-oriented router" `[T]` Singh.
  Qualitative model choice belongs to the semantic router in front of it (T14).
- **It is not an engine.** It sits above vLLM/SGLang and works with them; it does not replace KServe
  either (T13-Q23).

So the honest answer to "is a Service enough" is: **yes for a single replica or a stateless
short-prompt workload, no the moment multi-turn sessions dominate** — which, for an agent platform,
is always `[D]`.

**Signal:** Names KV awareness as the delta and the three-turn recomputation as the specific harm,
rather than saying "it's a smarter load balancer."

**Follow-ups:**
- *When is a Service genuinely enough?* — one replica per model, or stateless short prompts.
- *Where does model selection happen then?* — not here; T14.
- *What does llm-d cost you?* — another control plane and a shared router; T13-Q21/Q22.

**Red flags:** Describes llm-d as an engine or a load balancer, or claims it selects models.

---

#### T13-Q20 · Decompose llm-d's architecture
**Difficulty:** L3 · **Depth expected:** 2–3 min

**Question:** Draw llm-d. What are its components and what does each one do?

**Model answer:** Six pieces `[T]` Pravin:

- **Inference pool** — the set of vLLM/SGLang pods being served from.
- **Router / EPP (Endpoint Picker)** — "a single deployment that sits between the vLLM pods and…
  attaches to the gateway." This is the decision-maker: filter → score → rank.
- **Gateway API Inference Extension** — the hook into whatever gateway you already run (GKE, Istio,
  Envoy), so you do not replace your ingress.
- **KV events** — vLLM pods emit KV create and evict events, and "whenever a vLLM pod creates a KV
  cache or evicts a KV cache the llm-d router gets to know this." This is the feedback loop that
  makes prefix-aware routing possible, and it is **per-block events, not hashing** `[T]`/`[R]`.
- **Data layer** — "maintains all the states that… what all cache is available in the vLLM, which pods
  are saturated and all that."
- **Workload variant autoscaler** — "saturation based autoscaling."

Two properties that matter for adoption and are easy to miss:

1. **You can start without GPUs.** "You can even… start with inference simulator as well because you
   don't need a vLLM instance running on the GPU or CPU" `[T]` Pravin. That makes llm-d evaluable
   before you commit hardware.
2. **Flow control with priority bands** is part of the picture — premium versus best-effort traffic,
   with operator-defined saturation thresholds `[T]` (T13-Q23).

The critical architectural note for grading: the EPP is described as **"a single deployment"** `[T]`,
which makes it a fleet-wide single point of failure. That is the hook for T13-Q22.

**Signal:** Names the EPP, KV events and the data layer, and identifies the EPP as a single
deployment — volunteering the availability consequence.

**Follow-ups:**
- *Which component makes prefix-aware routing possible?* — KV events plus the data layer.
- *What is the single-deployment risk?* — T13-Q22.
- *How do you evaluate it before buying GPUs?* — the inference simulator.

**Red flags:** Lists components without the KV-event feedback loop, or misses that the EPP is one
deployment.

---

#### T13-Q21 · When does llm-d pay for itself?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** llm-d is another control plane to run and a shared dependency in the request path. At
what point does it earn that? And how would you decide for *your* fleet rather than a published
number?

**Model answer:** There is a published crossover and there is a decision procedure, and I would keep
them separate because the published number is not mine to rely on.

**The published number.** Singh reports a **~2× throughput effect at ~85 QPS** that halves a
card-months bill — for 200 users, ₹10 lakh → ₹5 lakh per month `[T]` Singh. Two caveats I would state
out loud: it is a **vendor-affiliated study**, and it is a single operating point. The guide's
framing, which I would adopt, is that the break-even is **immediate once the crossover is crossed and
negative below it** — below it you are paying for a control plane that has nothing to place.

**Why the shape of that answer is right.** llm-d's benefit scales with *prefix reuse opportunities
per unit time*, which scales with session concurrency and turn count. Its cost is roughly fixed: one
router deployment plus the engineering to operate it. So benefit-minus-cost is a curve that crosses
zero once, and the job is to find where, in your workload.

**The decision procedure for your own fleet `[D]`:**

1. **Measure the baseline first.** Prefix cache hit rate per session, TTFT and throughput per GPU
   behind a plain Service, at your real concurrency. If hit rate is already high, the Service is not
   costing you anything and llm-d has no lever.
2. **Find your QPS at the point where multi-turn sessions dominate.** The three-turn recomputation
   failure (T13-Q19) is the harm; estimate how often it happens.
3. **Convert to card-months, not percentages.** The published claim is expressed as a bill reduction,
   which is the right unit — a percentage of a small number is a small number.
4. **Compare against the operating cost** of the router deployment plus the engineering, and confirm
   the router is not itself the bottleneck (T13-Q22).

**And the case study's own sensitivity answer:** at 1,000 sessions, use a plain Deployment and a
Service; at 10,000, the sandbox idle tax dominates and llm-d is a tuning question; at 100,000,
**llm-d becomes mandatory** `[D]`/`[T]`. Which is to say: the crossover is real, but in an agent-heavy
platform it is usually crossed, because agent sessions are multi-turn by construction.

**Signal:** Separates the published number from the decision procedure, identifies the fixed-cost /
scaling-benefit shape, and insists on measuring the baseline hit rate first.

**Follow-ups:**
- *What is the baseline metric you would measure?* — prefix cache hit rate per session.
- *Why is "card-months" the right unit?* — it is a bill, not a ratio.
- *When does it become mandatory?* — at 10× scale; T13-Q29.

**Red flags:** Quotes the 2×/85 QPS figure as a law, or adopts llm-d without measuring the baseline
hit rate.

---

#### T13-Q22 · The EPP is a single deployment. Design its failure.
**Difficulty:** L5 · **Depth expected:** 4 min

**Question:** llm-d's router is "a single deployment" `[T]`. It is now in the request path for your
entire fleet. What happens when it dies, and what do you build?

**Model answer:** It is a fleet-wide single point of failure, and the honest answer to "what happens"
is: **nothing good, unless you built the fallback before you needed it.**

The failure is not a subtle degradation. Every new request flows through the EPP for placement, so its
loss is either total failure (if the gateway has no other route) or an immediate, unplanned regression
to naive routing (if it does). The blast radius is the entire fleet, and the detection signal is
router health and request success rate — not latency, which will look *better* under a naive fallback
because the router's work has stopped.

What I would build `[D]`:

1. **Redundant router deployment**, with the gateway able to fail over between instances. This is the
   minimum, and it converts a single point of failure into a failover event.
2. **An explicit gateway-level fallback route** that degrades to naive routing — round robin across
   the pool. Slower, poorer cache locality, but *serving*. The design question to answer on the
   whiteboard is "what does the gateway do when the EPP is gone," and "it errors" is not an
   engineering answer.
3. **Alert on the fallback firing**, not just on router liveness. A fallback that silently carries
   production traffic for a week is a cost incident (T13-Q30).
4. **Decide the KV-event behaviour explicitly.** If KV events stop arriving, the router's decisions
   become approximate without failing — the failure table calls this out as "prefix routing silently
   degrades," detected by comparing KV event rate to request rate. An EPP with zero KV events is a
   load balancer wearing a costume `[D]`.

**At 10×**, per the case study: the router stops being a deployment and becomes a **tier** — sharded,
replicated, with regional failover (T13-Q29). The design I would write today should be able to become
that without a rewrite.

**Signal:** Names the fleet-wide blast radius, distinguishes fail-stop from silent degradation, and
specifies a *degraded but serving* fallback.

**Follow-ups:**
- *What is the first thing to alert on?* — fallback activation, and KV event rate going to zero.
- *Why is the silent degradation the worse failure?* — it serves, at higher cost, unnoticed.
- *What does this look like at 10×?* — a sharded tier; T13-Q29.

**Red flags:** Assumes the gateway handles it, or proposes a fallback that errors instead of
degrading.

---

#### T13-Q23 · KServe and llm-d — who owns what?
**Difficulty:** L3 · **Depth expected:** 2 min

**Question:** Your platform team asks whether llm-d replaces KServe. Answer them.

**Model answer:** No, and the corpus is explicit. Pravin: **"KServe is more for deploying the models…
llm-d works with KServe. So it's not something that replaces KServe. So Tesla in fact uses KServe
with llm-d"** `[T]` Pravin.

The split is by concern:

- **KServe owns model lifecycle** — deploying models, revision management, and accounting. It answers
  "which model version is running here, and who put it there."
- **llm-d owns placement and admission** — which replica serves a request, how the pool is scaled,
  how flow control is applied. It answers "where does this request go right now."

They are complementary because they answer different questions about the same pods, and the case
study's decision table records it that way: KServe is "always present; complements llm-d, does not
compete with it" `[T]`.

Why the distinction is load-bearing rather than bookkeeping: **the model registry, not the router,
must be the source of truth for model revisions** `[D]`. The router's job is performance, and it has
no opinion about whether two replicas serving "the same" model name are actually the same revision —
that is the model-registry-drift failure with silent-correctness symptoms (T13-Q18).

**At 10×, KServe's role grows.** Once the model registry has to be authoritative across many pools,
the lifecycle layer stops being plumbing and becomes the source of truth (T13-Q29).

**Signal:** States complement-not-compete with the ownership split, and identifies the registry as the
revision source of truth.

**Follow-ups:**
- *Who is the source of truth for revisions?* — the registry; KServe.
- *What goes wrong if the router guesses?* — silent correctness drift; T13-Q18.
- *What does KServe's role look like at 10×?* — it grows; T13-Q29.

**Red flags:** Says llm-d replaces KServe, or cannot say which layer owns revisions.

---

### The agent substrate and the session lifecycle

#### T13-Q24 · Why do agents break Kubernetes' assumptions?
**Difficulty:** L3 · **Depth expected:** 3 min

**Question:** People run web services on Kubernetes every day. What is different about agent
workloads that makes standard Kubernetes a poor fit?

**Model answer:** Four properties, per Hockin `[T]` Hockin, and each breaks a different assumption.

- **Bursty.** Agents work in short spurts and then sit idle for "minutes, hours, even days, weeks."
  Kubernetes' scheduling model assumes a workload that either runs or does not; a workload that is
  *alive but not computing* is one it has no cheap representation for, so you hold resources for
  nothing.
- **Untrusted.** The agent runs code it wrote. It must run in a sandbox — OCI, gVisor, or microVMs via
  runtime classes — and Kubernetes supports the isolation mechanisms but does not choose them for you.
- **Single-tenant.** "You can't share a sandbox. That would kind of defeat the purpose" `[T]` — which
  leads directly to the cost: "we miss out on a ton of opportunities for optimizations, specifically
  the amortization of overheads." **This is the property that costs the most**, because it removes the
  multiplexing that makes Kubernetes efficient in the first place.
- **Human-in-the-loop.** Agents wait on people at approval checkpoints, which makes the workload
  **highly sensitive to perceived latency** — and a 30-minute approval wait is a normal event, not an
  outlier.

The synthesis: Kubernetes was built for workloads that are trusted, multiplexable, and continuously
busy. Agent workloads are none of those three, and they are the *opposite* of all three at once. That
is why the two state-of-the-art patterns are both unsatisfying (T13-Q25) and why the third path exists.

**Signal:** Names all four and identifies single-tenancy as the expensive one, because it removes
amortisation.

**Follow-ups:**
- *Which property drives the cost?* — single-tenancy; it removes multiplexing.
- *What are today's patterns?* — T13-Q25.
- *Why does human-in-the-loop matter to a serving engineer?* — it sets the idle-timer budget; T13-Q28.

**Red flags:** Says "agents are stateful" and stops, or misses the untrusted/sandbox requirement.

---

#### T13-Q25 · The two patterns, and why both are unsatisfying
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Walk me through the two state-of-the-art ways of running agent sandboxes today, and tell
me what is wrong with each.

**Model answer:** Both are real and both are unsatisfying, for different reasons `[T]` Hockin.

**Pattern one: a sandbox pod per agent.** Correct isolation, native Kubernetes, simple to reason
about. Its problem is startup and its consequence is waste. Pod startup is "seconds usually, but
that's not fast enough," and agents that load a browser or a runtime environment take **10–15 seconds
or more**. Since you cannot start one per turn at that latency, operators keep pods alive after a turn
— waiting "tens of seconds at the low end to tens of minutes at the high end" — **burning resources
nobody else can use.** Given the single-tenancy requirement (T13-Q24), those resources cannot be
lent out. That is the CFO's complaint, mechanised.

**Pattern two: a DIY mega-workload.** "A giant workload per machine" with a bespoke control plane and
a manager per node. Hockin's assessment is fair to it: "it can be very efficient" — but "you end up
reinventing large parts of Kubernetes," and it is "difficult to operate for smaller companies,
especially startups." It works; it costs a platform org.

**The shape of the gap.** Pattern one pays in idle resources; pattern two pays in engineering. There is
no option that pays in neither, which is the honest framing — and it is why the interesting question
is not "which is better" but "can we make the idle state cheap."

**Pattern three, and the premise.** Agent Substrate is built on "we don't want idle resources" — and
it is explicitly **"still pre-production grade"** `[T]`. Its target state is a workable release by the
end of the fall; a production design today cannot depend on it (T13-Q26), so the pragmatic answer is
pattern one plus a pilot.

**Signal:** Gives the cost of each pattern in its own currency (idle resources vs engineering) and
volunteers that the substrate is pre-production.

**Follow-ups:**
- *What is the warm-pool tax, exactly?* — cold-start latency converted into idle cost; T13-Q27.
- *Why can't you share the idle pod?* — single-tenancy; T13-Q24.
- *What is the honest production answer today?* — pattern one with a tuned idle timer, plus a pilot.

**Red flags:** Presents the substrate as production-ready, or says a warm pool is simply "the right
answer" without its cost.

---

#### T13-Q26 · Decompose Agent Substrate, and walk a session through it
**Difficulty:** L4 · **Depth expected:** 4–5 min

**Question:** Explain Agent Substrate's model. Name its components and then walk one agent session
from creation to overnight sleep.

**Model answer:** Six terms `[T]` Hockin:

| Term | Meaning |
|---|---|
| **Actor** | A stand-in for an agent, sandbox, or anything that acts like one |
| **Actor template** | "If an actor is a cookie, the actor template is the cookie cutter" — includes sandbox technology choice; many templates per substrate |
| **Worker** | Usually a pod; consumes resources; runs actors **serially** |
| **Per-node manager** | Manages all workers on a node; needed for networking |
| **Enlightened proxy** | Triggers wake-ups on receipt of traffic |
| **Golden snapshot** | The prepared template image the controller takes and stores, for fast resume |

**The walk-through, which is the part that shows understanding.** The controller sees an actor template
and does prep work: it spins the template up, takes the golden snapshot, and stores it. A higher-order
system then creates an actor from that template — and **"creating an actor doesn't run the actor."**
The user speaks first, so the prompt is routed to the enlightened proxy, which **wakes the actor**,
assigns it to a worker by message through the router, and talks to the worker on a **private
connection**; being new, the actor wakes **from the golden snapshot**. "The cool part is this happens
in a couple hundred milliseconds" — and he notes they are still working on that optimisation. When the
turn ends the actor is probably idle, so they **pause**, take the snapshot, shuffle the data off, and
the worker becomes unassigned and available to another actor. When no more messages arrive, the data
goes to cloud storage and it sleeps for the night.

Two things to say out loud: the vocabulary is **aspirational in places** — he warns "some of this is
aspirational, we're still working on it" — and the project is **pre-production grade**. Note also the
ASR caveat: the per-node manager and the proxy are rendered "eight-let" and "eight-net," and the
proper nouns are uncertain, so I use the descriptive terms.

**Signal:** Walks the lifecycle in order, states that creating an actor does not run it, and pairs the
wake latency claim with the "still working on it" caveat.

**Follow-ups:**
- *What makes the resume fast?* — the golden snapshot, taken before any actor exists.
- *Why must a worker run actors serially?* — it is the resource consumer; single-tenancy.
- *What is genuinely unbuilt?* — he says so himself; treat the scale numbers as targets.

**Red flags:** Describes it as a container runtime, omits the golden snapshot, or presents the wake
latency as measured production behaviour.

---

#### T13-Q27 · A 10–15 second cold start. What are your options?
**Difficulty:** L4 · **Depth expected:** 3–4 min

**Question:** Some of your agents load a browser and take 10–15 seconds to start. You cannot make a
user wait for that on every turn. What are your choices?

**Model answer:** There are **three**, not four, and each pays in a different currency `[T]` Hockin.

1. **Keep it warm.** Pay idle cost: hold the pod so the next turn is fast. Correct when wake latency
   dominates perceived quality — an interactive human waiting — and the idle window is short.
2. **Snapshot and resume.** Pay capture complexity: suspend the actor, store its state, and wake it
   from a prepared image rather than booting. This is Agent Substrate's mechanism, and it is the only
   option that makes the idle state cheap. Its current cost is maturity: **pre-production grade** `[T]`.
3. **Accept the latency.** Pay user experience, or hide it behind asynchronous work — the agent starts
   working on something the user has not asked to watch yet.

The reason "there is no fourth option today" is worth saying: the corpus's own framing is that agents
which load a heavyweight runtime **cannot be started per turn**, and the three above are the whole
space.

**The sizing rule, which is the discriminator.** Hockin's instruction is explicit: **"you have to do
some math, and it's different for everybody"** `[T]` — set the idle timer from **measured** startup
cost, not from a default. A timer set to a default either wastes money (too long) or produces cold
starts (too short). The input is the distribution of human think-time, and the target is to sit above
its p95.

**How much it is worth.** Against a 12-second cold start, a ~200 ms resume is roughly **60× faster to
first response**; against a 2-second lightweight start it is **10×** `[D]` (case-study arithmetic from
Hockin's figures). And the cost changes *shape*: a suspended actor's state lives in cloud storage
rather than a held pod, so the idle tax collapses to storage cost.

**Signal:** Gives exactly three options with their currencies, and states that the idle timer is
derived from measured startup cost rather than configured.

**Follow-ups:**
- *Where does the idle timer number come from?* — measured startup cost and think-time p95.
- *What does resume change about the cost?* — idle cost becomes storage cost.
- *Which option is production-ready today?* — one, with a substrate pilot; T13-Q25.

**Red flags:** Picks a warm pool without naming its cost, or sets the idle timer to a default.

---

#### T13-Q28 · Where does session state live?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** Your agent sessions span two human approval checkpoints, with waits of up to 30 minutes.
Where does session state live, and what breaks if you get it wrong?

**Model answer:** Four candidate homes, and each has a different lifetime — which is the actual
question `[T]`/`[D]`:

| Home | Survives | Breaks when |
|---|---|---|
| In the sandbox only | Nothing much | Every pod death — never for long sessions |
| In the KV cache tier | Reuses prefill; **~5× better TTFT on return** `[T]` Pravin | Engine upgrade or version change |
| External durable session store | Everything; the substrate's model | Nothing structural; costs duplication and restore latency |
| In the client | Zero server state | Client must replay; insecure |

**My design:** a **durable external session store for agent state**, a **KV tier for model state**, and
— this is the load-bearing part — an **explicit acknowledgement that the two have different
lifetimes.** An engine upgrade invalidates the KV tier and does not invalidate the durable store. If
you treat them as one thing, an engine patch becomes a mass session failure.

**Why the human-in-the-loop case forces it.** A 30-minute approval wait will **exceed most idle
timers**, producing teardown and a cold restart at exactly the moment the human presses approve — the
worst possible moment for a 10–15 second wait (T13-Q27). So the session must be **durable across the
wait**, which is a substrate or external-store problem and **not an engine problem.** No amount of
engine tuning fixes it.

Two consequences worth drawing out:

- It changes what "resume" means. Resuming from a durable store is *state recovery*; resuming from a
  golden snapshot (T13-Q26) is *environment recovery*. A full design usually needs both.
- It makes the KV tier an optimisation rather than a source of truth. That is the right posture: the
  ~5× TTFT benefit on return is worth having, but the system must be correct without it.

**Signal:** Separates the lifetimes explicitly, places the durable store as the source of truth, and
names the 30-minute wait exceeding idle timers as the forcing condition.

**Follow-ups:**
- *What invalidates the KV tier but not the store?* — an engine or model version change; T13-Q17.
- *How is this different from resume-from-snapshot?* — state recovery vs environment recovery.
- *Could you collapse the two tiers?* — only if KV became portable across versions; it does not.

**Red flags:** Puts long sessions in the sandbox, or treats the KV cache as durable, or misses the
approval-wait timeout.

---

#### T13-Q29 · What changes at 10×?
**Difficulty:** L5 · **Depth expected:** 4–5 min

**Question:** Your platform is healthy at 10,000 concurrent agent sessions. What breaks when it
becomes 100,000, and what should you have designed differently?

**Model answer:** Four shifts, and the last one is the one teams miss `[D]`/`[T]`.

**One: the router becomes a tier, not a deployment.** At 1× a single EPP deployment is acceptable with
a fallback (T13-Q22). At 10× it is a fleet-wide single point of failure and must become a **sharded,
replicated service with regional failover**. The design you write today should be able to become that
without a rewrite — which means keeping the router stateless where possible and the KV-event loop
explicit.

**Two: the sandbox layer becomes the platform.** At 10× the idle tax is the entire budget (T13-Q30),
so suspend/resume is no longer an optimisation — it is the product. The case study's sensitivity table
puts the substrate as the dominant term from 10× onward, and the industry is visibly heading that way.
If your session state lives in the sandbox, this is the point at which you cannot get there.

**Three: engine homogeneity returns, but at the pool level.** Ten engines across fifty pools is
unmanageable; two engines across three workload classes is not. Expect **consolidation within tiers and
diversification across them** — the same shape as the parallelism result in T11.

**Four: the engine version becomes a fleet-wide object.** A CVE at 10× means a coordinated rollout
across hundreds of replicas within a week. That is a **CI/CD problem, not a serving one** `[R]`
(`11-infrastructure-and-mlops/02-cicd.md`), and it is why the ≤1-week patch cadence (T13-Q17) has to be
engineered rather than promised.

**What survives unchanged:** the three-layer split (engine / orchestrator / substrate), the KV-event
feedback loop, per-workload engine choice, and pinned tuples. Those are architectural, not
scale-dependent — which is the answer to "what should you have designed differently": nothing about
the structure, only about the *sizing* of the router and the *ownership* of session state.

**And the honest ceiling:** Hockin's scale numbers — 10⁴ activations/s, 200,000 nodes, billions of
sessions — are described by him as **aspirational**, and the substrate is pre-production `[T]`. Treat
them as the direction of travel, not as a demonstrated operating point.

**Signal:** Names all four shifts, separates what scales from what survives, and marks the substrate
scale targets as aspirational.

**Follow-ups:**
- *Which shift is structural rather than budgetary?* — the router becoming a tier.
- *What survives at 10×?* — the three-layer split and the KV-event loop.
- *What is the CVE rollout problem?* — CI/CD, not serving `[R]`; T13-Q16.

**Red flags:** Answers only "more replicas," or treats Hockin's node targets as measured capability.

---

### Economics and cross-layer design

#### T13-Q30 · The 97% idle tax against the 0.16% engine saving
**Difficulty:** L5 · **Depth expected:** 5 min

**Question:** The CFO asks why the agent fleet costs more than the model serving does. Answer with
arithmetic, and tell me what it implies for where your team spends its engineering time.

**Model answer:** The answer is an ordering problem, and it comes out of two pieces of arithmetic.
All of the following is my derivation from attributed inputs; the case study labels it the same way,
and the prices are illustrative `[D]`.

**Step one: the idle tax.** Take 10,000 concurrent sessions, a 20-second median *compute* time, and a
12-minute (720 s) median *wall* time:

```
Fraction of session time computing = 20 / 720 ≈ 2.8%
Hold 10,000 pods at $0.10/hr     = $1,000/hr
Useful compute bought            = 2.8% → $28/hr
Idle tax                         = $972/hr ≈ 97.2% of spend
```

That is Hockin's "each of those agents are idle 99.999% of the time" `[T]` expressed as a bill. Note
that even a generous 30-second keepalive leaves the overwhelming majority of spend idle.

**Step two: the engine.** Suppose SGLang's ~29% structured-output advantage `[R]` applies to 20% of
token volume `[D]`:

```
Engine-level saving  = 0.29 × 0.20       = 5.8% of compute cost
As a fraction of total spend = 5.8% × 2.8% ≈ 0.16%
```

**So the ordering is: sandbox lifecycle first, engine last.** Optimising the engine choice in an
agent-heavy platform moves roughly **a sixth of one percent** of total spend. Optimising the sandbox
lifecycle moves the other 97%.

**The counter-case, which a strong candidate volunteers.** In a *non-agentic*, high-QPS inference
product, compute is close to 100% of spend and that 5.8% is real money. So the engine decision is
still worth making well — it is just not the *first* decision. The mistake is applying the agent-heavy
ordering to a non-agentic product, or vice versa.

**What it implies for engineering time:** the highest-leverage knob is the **sandbox idle timer**,
derived from measured startup cost (T13-Q27), followed by saturation thresholds, warm-pool size, and
priority bands — with engine batch parameters *last*, because they only trade TTFT against throughput
inside one pool.

And the boundary of this analysis: **the corpus contains no cost-per-session figure for an agent
platform**, and no head-to-head engine benchmark. Every number above is derived, with the assumptions
shown.

**Signal:** Produces both pieces of arithmetic, states the ordering, and volunteers the non-agentic
counter-case rather than over-claiming.

**Follow-ups:**
- *Which knob has the most leverage?* — the sandbox idle timer.
- *When is the engine saving material?* — high-QPS non-agentic inference; T13-Q13.
- *What is not measured here?* — cost per session; no head-to-head benchmark in this corpus.

**Red flags:** Optimises the engine first, or presents a derived percentage as a measured result.

---

## Whiteboard exercises

### Exercise 1 — Choose a serving stack from a specification
**Prompt.** "You are the platform lead at a 40-person company running a document-research agent on
Kubernetes. Traffic is multi-turn and prefix-heavy — median 12 minutes wall, under 20 seconds of
compute, two human approval checkpoints per session. You serve three models. Your team is three
engineers. Choose the engine, the orchestrator and the sandbox strategy, and tell me what would make
you change each one."

**What the candidate must produce:** the three layers named *with the state each owns* before any
product is named, a choice per layer with its cost stated, and an explicit revisit trigger per choice.

**Expected answer sketch:**

```
LAYER            OWNS                        CHOICE              COST OF THE CHOICE
engine           KV, batching, GPU memory    vLLM (pinned)       many knobs; thrpt-tuned defaults
orchestrator     placement, admission,       llm-d               +1 control plane; EPP is 1 deployment
                 cluster KV view
substrate        session state, sandbox      pod-per-agent +     idle tax ~97% of spend
                                             idle timer + pilot  (T13-Q30 arithmetic)
lifecycle        model revisions             KServe              complements llm-d [T] Pravin

ROUTING: prefix-aware, because a Service recomputes a 3-turn
prefix 3x and the hit rate is "atrocious" in Prometheus   [T] Singh

SECOND ENGINE: only for a named workload class --
structured output / function calling, ~29% [R] -- as a 1-5% canary

REVISIT TRIGGERS
  engine      -> structured output > ~30% of traffic (own pool)   [D]
  llm-d       -> fleet collapses to one replica per model
  sandbox     -> Agent Substrate leaves pre-production [T]
  version     -> LTS line with backported security fixes
```

**Grading rubric (full marks requires all four):**
- Names state ownership at each layer *before* naming a product — the case study's own whiteboard
  order.
- Chooses vLLM on ease-of-use/breadth grounds and can say why that is the right axis (T13-Q2), not
  because it is fastest.
- States the cost of each choice, including the warm pool's idle tax — a choice with no stated cost is
  not a choice.
- Gives a revisit trigger per layer, and marks Agent Substrate pre-production.

---

### Exercise 2 — Design the session lifecycle and cost it
**Prompt.** "Your agents load a browser and take 12 seconds to start. Human approval waits run up to
30 minutes. Sessions are 10,000 concurrent with a 20-second median compute time. Design the session
lifecycle, set the idle timer, and show me the arithmetic that justifies it."

**What the candidate must produce:** a lifecycle diagram with the state transitions named, a numeric
idle-timer derivation, the idle-tax arithmetic, and a statement of what the design cannot do today.

**Expected answer sketch:**

```
LIFECYCLE (Agent Substrate terms [T] Hockin)
  create actor ---> NOT RUNNING  (creating an actor doesn't run it)
       |
   inbound msg --> enlightened proxy wakes it
       |            assign to worker (private connection)
       |            resume from GOLDEN SNAPSHOT  ~couple hundred ms [T]
       v
    RUNNING ---- turn ends ----> PAUSE: snapshot, shuffle off, worker unassigned
       ^                              |
       |                              v
       +---- next msg ------------- SUSPENDED (state in cloud storage)

IDLE TAX  [D] from Hockin's figures, prices illustrative
  compute fraction = 20 s / 720 s                     ~ 2.8%
  10,000 pods x $0.10/hr = $1,000/hr
  useful = 2.8% -> $28/hr ;  idle = $972/hr           ~ 97.2%

IDLE TIMER
  derive from MEASURED startup cost + think-time p95
  "you have to do some math, and it's different for everybody" [T]
  NOT a default. Target: above p95 of the human gap.

WHAT IT CANNOT DO TODAY
  Agent Substrate is "still pre-production grade" [T]
  -> pilot in parallel; warm pool is the honest answer now
```

**Grading rubric:**
- Puts the idle timer on measured startup cost and think-time p95, and refuses to name a default —
  this is Hockin's own instruction.
- Notices that a 30-minute approval wait **exceeds the idle timer**, so the session must be durable
  across the wait (T13-Q28) — the discriminator in this exercise.
- Produces the idle-tax arithmetic and states that resume changes the *shape* of the cost (held pod →
  storage), not just its size.
- Marks the substrate pre-production and proposes a pilot rather than depending on it.

---

### Exercise 3 — Diagnose a cross-layer failure
**Prompt.** "Your agent platform is healthy. Overnight, prefix cache hit rate falls by 80%, cost per
completed session rises 40%, and p50 latency is unchanged. No deploys. Walk me through the diagnosis,
layer by layer, and tell me what you check first."

**What the candidate must produce:** a ranked differential *organised by layer*, a discriminating
check for each candidate, and a fix that names which layer owns it.

**Expected answer sketch:**

```
Symptom: hit rate DOWN 80% | cost UP 40% | p50 latency FLAT | no deploys

KEY DISCRIMINATOR: latency flat + cost up => work is being REDONE, not stalled.
                  a saturated engine would show latency, not just cost.

LAYER        CANDIDATE                  DISCRIMINATING CHECK
orchestrator KV events stopped      --> KV event rate vs request rate = 0?
             (EPP degrades silently)    an EPP with zero KV events is a
                                        load balancer in a costume  [D]
orchestrator a replica was replaced --> engine version per replica;
             (version skew)             KV layouts not comparable [T]
lifecycle    registry drift          --> output-hash comparison across
             (silent correctness)       replicas; registry is the truth
substrate    warm pool grew OR       --> pool occupancy; idle-timer firings
             idle timer drifted         cost up with latency flat = likely here
engine       cache disabled/shrunk   --> config diff vs pinned tuple;
             by an unnoticed change     --enable-prefix-caching, cache size

FIX BY OWNER
  orchestrator -> restart metrics path; re-pin engine as ONE unit per pool
  lifecycle    -> redeploy the drifted replica from the registry
  substrate    -> reset the idle timer from measured startup cost (T13-Q27)

ALERT ORDER (what you should have had): idle fraction > target AND
  KV event rate = 0, BEFORE alerting on latency   [D]
```

**Grading rubric:**
- Reads "latency flat, cost up" as evidence of repeated work rather than contention — the
  discriminator that eliminates the engine as the first suspect.
- Checks **KV event rate first**, because a zero event rate means the orchestrator has silently
  become a load balancer — the case study's own "first thing to instrument."
- Separates the *cost* cause (substrate/idle) from the *hit-rate* cause (orchestrator), since one
  symptom pair can have two independent causes.
- Assigns each fix to a layer and names the alert that would have caught it earlier.

---

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
  — the two entry points (the `LLM` class for offline batch and `vllm serve` for online serving),
  OpenAI *and* Anthropic API compatibility and "work out of the box" for agent frameworks, more than
  10 hardware backends behind a plugin structure, the whole-stack re-take for new-hardware bring-up
  (months, not weeks), dynamic memory partitioning, and the KV connector's transport-agnostic
  abstraction over NIXL, MoRI-O and Mooncake. ASR renders "vLLM" as "VLM" and the class as "the EDLM
  class"; both corrected here.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Opening_Note_vLLM_Inference_Meetup_Bengaluru_September_19_2026.txt`
  — Prasad Mukhedkar: vLLM-as-Docker and llm-d-as-orchestration, "not about the performance… ease of
  use," Ollama's "batch size is equal to zero or one" scope, the recipes repository, ~90K GitHub
  stars, and the 1,300+ meetup registrations on day two. ASR renders "Ollama" as "Olama"; corrected.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — Pravin (IBM Research): llm-d as a native Kubernetes stack on the Gateway API inference extension,
  CNCF and the collaborator list, the inference pool / router-EPP / data layer / workload variant
  autoscaler decomposition, KV events arriving per block, KServe complementarity with the Tesla
  example, the inference simulator, and the EPP described as "a single deployment."
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
  — Abhishek Singh: the vLLM V2 engine as the single-replica default, the three-turn prefix
  recomputation behind a plain load balancer with the cache hit rate "atrocious" in Prometheus,
  llm-d as a performance-oriented router rather than a model selector, and the ~2× throughput effect
  at ~85 QPS with the ₹10 lakh → ₹5 lakh/month card-months framing (vendor-affiliated study).
- `refs/Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt`
  — the four agent properties (bursty, untrusted, single-tenant with "you can't share a sandbox,"
  human-in-the-loop); the two state-of-the-art patterns and their costs; the Agent Substrate
  vocabulary (actor, actor template as cookie cutter, worker, per-node manager, enlightened proxy,
  golden snapshot), "creating an actor doesn't run the actor," pause/snapshot/shuffle-off, "a couple
  hundred milliseconds," "each of those agents are idle 99.999% of the time," the 10–15 s
  browser/runtime cold start, the tens-of-seconds-to-tens-of-minutes keepalive range, "you have to do
  some math, and it's different for everybody," 200,000 nodes, and the explicit statements that the
  scale numbers are aspirational and the project is pre-production grade. ASR renders the per-node
  manager as "eight-let" and the proxy as "eight-net"; the proper nouns are uncertain, so the
  descriptive terms are used throughout.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
  — Banghua Zhu: SGLang positioned for agentic workloads with broad hardware support; HiCache,
  HiSparse, Spec V2 / overlap scheduler, and chunk pipeline parallelism; and the "over 2.2×
  improvement and up to 500 tokens per second per user" on GLM-5.2, measured against its own day-zero
  baseline — a vendor claim, version-over-version. ASR renders "SGLang" as "Ashlan"/"SLR"; corrected.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`
  `[R]` — the May-2026 engine landscape, engine-per-workload as the posture, SGLang v0.4.3's ~29%
  structured-output advantage (vendor-published, citing an April 2026 SGLang blog) and its
  async-constrained-decoding attribution, the unpatched multimodal and disaggregated-prefill RCEs
  with vLLM ≥ v0.18.2 as the multimodal baseline, TensorRT-LLM's build step and CUDA lock-in, the
  1–5% canary posture, and the "Llama-X on vLLM vN with config C on hardware H" pinning tuple.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]` — how vLLM
  (PagedAttention, continuous batching), SGLang (RadixAttention as a radix tree over KV prefixes;
  HiCache across GPU → CPU → storage), TensorRT-LLM (ahead-of-time build with kernel fusion, custom
  attention kernels, paged KV, in-flight batching, CUDA graphs, speculative decoding) and GGUF work.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md`
  `[R]` — the self-hosting options table (vLLM / TGI / TensorRT-LLM / Ollama / llama.cpp) and the
  API-versus-self-host decision framework.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/02-cicd.md`
  `[R]` — deployment and rollout practice for engine upgrades, and the reason a fleet-wide CVE
  response is a CI/CD problem rather than a serving one.

**Derived content in this bank (`[D]`):** the four-line decomposition of accelerator bring-up cost
(T13-Q9); the three-question rubric for reading an engine benchmark (T13-Q12); the
three-conditions-jointly adoption test for TensorRT-LLM (T13-Q14); the workload-class + measured-gain
+ canary + expiry decision rule for one-versus-two engines (T13-Q15); the 72-hour CVE response
ordering (T13-Q16); the layer-by-layer differential structure and the "latency flat + cost up means
work is redone" discriminator in Whiteboard Exercise 3; the router-redundancy-plus-degraded-fallback
design (T13-Q22); and the re-statement of the case study's idle-tax and engine-saving arithmetic
(T13-Q30, ~97.2% and ~0.16%), where the inputs are attributed — 10,000 sessions, 20 s compute, 720 s
wall, $0.10/pod-hour, 29% on 20% of token volume — and every step is shown. All are labelled inline
where they appear. No benchmark in this bank is invented: figures are quoted as the speaker or
repository reported them, with vendor and press claims marked as such, and the corpus's two explicit
gaps — no head-to-head engine comparison on identical hardware, and no cost-per-session figure for an
agent platform — are stated where they bound an answer.
