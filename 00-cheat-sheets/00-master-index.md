# Master Index — Every Number Worth Remembering

> **Transcript coverage:** primary · **Purpose:** one card holding the corpus's quantitative anchors
> Companion: [TOPICS.md](../TOPICS.md) · [README.md](../README.md)

Every figure here is **as reported by the speaker** and attributed. None are independently
verified. Where a figure is a vendor claim, it says so. Where arithmetic is mine, it is labelled.
See [`../README.md#on-numbers`](../README.md#on-numbers).

---

## 0. Topic index

Nineteen topics in four layers. Each links to its own one-page cheat sheet; the sections below cut
across all of them by theme. `[T]`/`[R]`/`[D]` provenance is marked per line in each sheet.

**Layer A — Decoding Algorithms** (CMU-dominant)

| ID | Topic | The one number to remember |
|---|---|---|
| [`T01`](T01-sampling-decoding.md) | Sampling & decoding | temperature 0 is **still non-deterministic** `[T]` |
| [`T02`](T02-search-decoding.md) | Search & beam decoding | search error vs model error must both be measured `[T]` |
| [`T03`](T03-constrained-generation.md) | Constrained generation | regular → FSA, context-free → pushdown, Turing → inexpressible `[T]` |
| [`T04`](T04-test-time-compute.md) | Test-time compute | more samples buy accuracy only up to a point `[T]` |
| [`T05`](T05-verifiers-best-of-n.md) | Verifiers & best-of-N | the `log n − (n−1)/n` KL bound; size `n` to your **batch** (`n=32` is the lecturer's example) `[T]` |

**Layer B — Engine Internals**

| ID | Topic | The one number to remember |
|---|---|---|
| [`T06`](T06-inference-fundamentals.md) | Inference fundamentals | prefill is **compute**-bound; decode is **bandwidth**-bound `[T]` |
| [`T07`](T07-kv-cache.md) | KV cache & tiering | paged waste **<4%** vs **60–80%** contiguous `[R]` |
| [`T08`](T08-batching-scheduling.md) | Batching & scheduling | continuous batching: **100 → 42** `[T]` |
| [`T09`](T09-speculative-decoding.md) | Speculative decoding | MTP ≈ **2×**; exact, not approximate `[T]` |
| [`T10`](T10-quantization.md) | Quantization | **42 → 26**; always rerun evals `[T]` |

**Layer C — Distributed & Serving**

| ID | Topic | The one number to remember |
|---|---|---|
| [`T11`](T11-parallelism-moe.md) | Parallelism & MoE | WideEP: DP attention + EP MoE; **TP stays inside the node** `[T]` |
| [`T12`](T12-disaggregation-kv-transfer.md) | PD disaggregation | pooled memory **< RDMA << TCP/IP** `[T]` |
| [`T13`](T13-serving-engines.md) | Serving engines | **>10** hardware backends behind one plugin core `[T]` |
| [`T14`](T14-routing-gateways.md) | Routing & gateways | caching + routing: **26 → 11** `[T]` |
| [`T15`](T15-autoscaling-slo.md) | Capacity & SLOs | KV **80%** / **>8** active requests as saturation gates `[T]` |

**Layer D — Operations, Agents & Governance**

| ID | Topic | The one number to remember |
|---|---|---|
| [`T16`](T16-agentic-inference.md) | Agentic inference | **98%** of agent tokens are prefill `[T]` |
| [`T17`](T17-observability-evals.md) | Observability & evals | three planes: metrics, traces, quality `[D]` |
| [`T18`](T18-guardrails-security.md) | Guardrails & security | four rail positions; injection arrives as **data** `[T]` |
| [`T19`](T19-finops-sovereignty.md) | FinOps & sovereignty | **100 → 42 → 26 → 11**; the inference layer is itself a sovereignty component `[T]` |

---

## 1. The cost ladder — the corpus's signature result

The single most useful number set in the whole corpus. Same model, four stacked optimizations
`[T]` *LLMOps, Cut LLM Cost/Latency*:

| Configuration | Relative cost | Cumulative saving |
|---|---|---|
| Naive, 16-bit, single request | **100** | — |
| + continuous batching | **42** | 2.4× |
| + 4-bit quantization (AWQ/GPTQ) | **26** | 3.8× |
| + cached stable system prompt | **11** | **~9×** |

> **The model never changed.** ~10× came entirely from the serving layer.

**Corollary:** cache reads are **~10× cheaper** than fresh tokens `[T]`. And on routing, the
verbatim rule is: **"Prompt caching and difficulty-based routing usually move the bill more than
switching models does."** `[T]` LLMOps cost talk.

**The anti-pattern:** the instinct to downgrade to a smaller model. "Savings live in serving."
Exhaust free wins — prompt caching, difficulty routing — before touching the model `[T]`.

---

## 2. Latency and throughput

| Quantity | Value | Source |
|---|---|---|
| TTFT | set by **prefill** (compute-bound) | `[T]` LLMOps cost talk |
| ITL / TPOT | set by **decode** (memory-bandwidth-bound) | `[T]` LLMOps cost talk |
| One traced request | **1.2 s total** = retrieval 90 ms + generation >1 s | `[T]` LLMOps observability |
| Interactive chat SLA example | **16 req/s sustained** on half an H100 / half an MI325X | `[T]` NextGen (speaker's customer SLA) |
| Pipeline planning example | **256 pages/sec** end-to-end (app + DB + inference + network) | `[T]` NextGen |
| llm-d crossover | **~2× throughput at ~85 QPS** | `[T]` NextGen (vendor measurement) |
| KV offload on session return | **~5× TTFT improvement** | `[T]` llm-d talk |
| Max concurrency, 1P1D pair | **~20,000** | `[T]` ROCm/WideEP talk |
| Codex session length | median **~10 min**, mean **~20 min** | `[T]` Tworek, long-horizon agents |

**Decode is largely solved; the hard problem is producing the first token cheaply** `[T]` NextGen.

---

## 3. Agentic traffic shape — why agent serving is a different problem

| Quantity | Value | Source |
|---|---|---|
| Share of inference traffic that is agentic | **~70%** | `[T]` llm-d talk |
| Share of tokens that are **prefill** in agentic workloads | **~98%** | `[T]` llm-d talk |
| Agent conversation length | up to **~100 steps**, ~50 tool calls, hundreds of thousands of tokens | `[T]` CMU agents lecture |
| Heavy personal usage | **2,000 steps**, tens of millions of tokens | `[T]` CMU agents lecture |
| Context window, frontier long-context | **1M tokens** on H200, ~250k on H100 | `[T]` llm-d talk |
| Agents idle fraction | **99.999%** of the time | `[T]` Hockin, Kubernetes for agents |

> Agentic workloads invert the classic optimisation target: the prefill dominates, and sessions
> return after long idle gaps, which makes KV cache retention — not raw decode speed — the lever.

---

## 4. Model and hardware specs cited

| Item | Spec | Source |
|---|---|---|
| Llama 3.1 | 405B dense · layers 32/80/126 · hidden 4096/8192/16384 · **always 8 GQA KV heads** · 128k context | `[T]` CMU lecture 1 |
| DeepSeek V3 | 256 experts | `[T]` ROCm/WideEP talk |
| Kimi K3 | 896 experts | `[T]` ROCm/WideEP talk |
| GLM 5.1 | 78 layers · 256 experts · top-k 8 · sparse attention + MLA | `[T]` ROCm/WideEP talk |
| AMD MI300 | **192 GB** HBM/GPU | `[T]` ROCm/WideEP talk |
| AMD MI355 | **288 GB** HBM/GPU | `[T]` ROCm/WideEP talk |
| AMD MI325X | **256 GB** VRAM | `[T]` NextGen |
| TPU v8 | split first time into **8T (training) / 8I (inference)** | `[T]` Tiwary, Google |
| TPU training pod | **121 FP4 exaflops** (~3× jump); memory bandwidth 2×/4× jumps | `[T]` Tiwary |
| TPU inference pod | **11.6 exaflops** (~10× jump) | `[T]` Tiwary |
| Chip design lead time | **2–3 years** | `[T]` DeSantis, AWS |
| Google token volume | **3.2 quadrillion tokens/month** | `[T]` Tiwary |
| vLLM scale | ~90K GitHub stars, 1,300+ meetup registrations in 2 days | `[T]` vLLM Opening Note |

**Parallelism types in vLLM: 7** — tensor, pipeline, data, expert, sequence, context, and
decode-context parallelism (DCP) `[T]` Kwon.

---

## 5. Quality, evaluation and correctness

| Quantity | Value | Source |
|---|---|---|
| Retrieval funnel | dense **61%** → +keyword+RRF **78%** → +reranker **89%** recall | `[T]` LLMOps RAGOps |
| Funnel config | each retriever returns **20** candidates → RRF → reranker trims to final **4** | `[T]` LLMOps RAGOps |
| Eval movement | **3.6/5 → 4.2/5** after adding a reranker | `[T]` LLMOps evals |
| Prompt regression | **98% → 71%** valid JSON from one added sentence | `[T]` LLMOps prompt management |
| Judge bias sources | position · length · self-preference (Jung et al. 2023) | `[T]` LLMOps evals |
| Curated eval set | **200 curated beats 10,000 random** | `[T]` LLMOps evals |
| Critic reranking (SWE-bench) | **20% → 32%** at **~16× inference cost** | `[T]` CMU agents lecture |
| Context condensation | **~2× cost reduction** at maintained SWE-bench | `[T]` CMU agents lecture |
| Best-of-N sample count | **n = 32** — chosen because it fits one batch | `[T]` CMU reward models |
| Best-of-N KL bound | `log n − (n−1)/n` | `[T]` CMU reward models |
| Temperature 0.2 diversity | only **~20 unique outputs** from 100 draws | `[T]` CMU reward models |
| R1 reasoning gain | AIME **~15% → >70%**; thinking 800 → thousands of tokens | `[T]` CMU reasoning models |
| Distillation | **~100k trajectories → 73% SWE-bench Verified** | `[T]` Gao, Microsoft Research |
| Harness efficiency | same score with **half the tokens** | `[T]` Cohen, NVIDIA |
| Klarna | **~80%** resolution-time cut, then rehired humans in 2025 | `[T]` LLMOps multi-agent |

---

## 6. Cost and business

| Quantity | Value | Source |
|---|---|---|
| Token purchasing parity | **30¢ → 16¢** per 10k-token artifact, Feb → Apr | `[T]` Token Raj |
| Code lines drafted | **630 → 91** for the same task | `[T]` Token Raj |
| Files touched | **8.2 → 3.6** | `[T]` Token Raj |
| llm-d cost translation | **₹10 lakh → ₹5 lakh/month** for 200 users | `[T]` NextGen |
| Subscription pressure | $20 → 10× → 20×, quotas exhausted in **3–4 days** | `[T]` Token Raj |
| Auto-research experiment cost | **$30/experiment**, driven to **10× cheaper** | `[T]` Chuan Li |
| API field-trimming win | `get_experiment` **50× cheaper** after stripping fields | `[T]` Chuan Li |
| Government sovereign controls | **~700 controls** | `[T]` NextGen |

**The structural incentive** `[T]` Token Raj: *vendor revenue scales only as tokens are burned.*
Token spend does **not** correlate with business outcomes (lines of code, PRs, merged MRs) — so
budget on outcomes, not on token burn.

---

## 7. Scaling and training economics

| Quantity | Value | Source |
|---|---|---|
| Long-horizon RL | a **12-hour** trajectory ⇒ **2 gradient steps/day ≈ 14/week ≈ 60/month** | `[T]` Tworek |
| Frontier agent horizons | readouts at **~12 h and 16 h** | `[T]` Tworek |
| Agentic compute vs non-agentic | **10–100×** more inference compute | `[T]` Tiwary |
| Synthetic data share (projected) | **99.9%** of future training data | `[T]` Chen, Microsoft |
| Environments at scale | must run **hundreds of thousands** of agents simultaneously | `[T]` Gao |
| Agent fan-out | one agent can spawn **100,000 sub-agents** ad hoc | `[T]` Panel |
| Reference fan-out | Claude ~**300** references vs OpenAI ~**30** (serial) | `[T]` CMU agents lecture |

**The compounding problem** `[T]` Tworek: cost grows as *n* while reward information grows as
*1/n*, so the learning signal decays **quadratically** in horizon length. Linear would be
affordable; quadratic is "brutal".

---

## 8. Decision quick-reference

| If you are… | Do this | Not this |
|---|---|---|
| Choosing a serving engine | vLLM — leads on ease of use | TensorRT-LLM, unless you need the last % and can pay in config complexity `[T]` |
| Serving one user on a laptop | Ollama (batch size 0/1) | vLLM — "not meant for my laptop" `[T]` |
| Serving many users on GPUs | vLLM + llm-d | a naive round-robin load balancer `[T]` |
| Routing follow-up turns | prefix-cache-aware routing via per-block KV events | hash-based approximate routing `[T]` |
| Routing across model tiers | semantic router (qualitative model choice) | an expensive MoE used as the router itself `[T]` |
| Placing a request on a replica | llm-d / EPP (performance routing) | semantic router — it picks the *model*, not the *pod* `[T]` |
| Cutting cost | continuous batching → quantization → prompt caching → difficulty routing | downgrading the model `[T]` |
| Adding an agent loop | only if the path is genuinely unknown | if the steps are fixed — use a pipeline `[T]` |
| Adding a second agent | only for distinct skills or strict isolation | "the honest default is one capable agent with good tools" `[T]` |
| Adding a protocol (MCP/A2A) | when you cross a team or vendor boundary | for a single app calling a single model `[T]` |
| Adding memory | when an agent must remember a person across days | a bigger context window — *"it is not memory. It is just a larger disk."* `[T]` |
| Quantizing | rerun your evals afterwards | assume accuracy held `[T]` |

---

## 9. The rules the corpus states explicitly

Every entry below is a **verbatim** quote, checked word-for-word against the transcript named.
Paraphrases are not quoted. (This section was rebuilt during verification: two of the original ten
entries could not be traced to any transcript and were removed rather than reworded.)

1. **"Optimization without measurement is just guessing."** `[T]` LLMOps cost talk
2. **"there's no universal winner or there's a universal solution"** — on parallelism: it depends on
   model architecture, cluster setup and workload shape `[T]` Kwon
3. **"Model is going to eat a lot of thing."** — the model absorbs more of the harness over time, so
   optimise the harness around the model rather than the model. *(Quoted as transcribed, including the
   singular "thing"; the surrounding passage also renders "harnesses" as "honeys" — the ASR is poor
   here, but the sentence is unambiguous.)* `[T]` Chen
4. **"you are able to do full two gradient steps in a day of your … reinforcement learning agent.
   That means roughly 14 gradient steps a week."** — the sample-efficiency wall for long-horizon
   agents `[T]` Tworek
5. **"A bigger context window feels like the answer, but it is not memory. It is just a larger
   disk."** `[T]` LLMOps memory
6. **"The honest default is one capable agent with a good set of tools."** — split into multiple
   agents only for genuinely distinct skills `[T]` LLMOps multi-agent
7. **"rails are not a wall. They are a filter that determined attackers learn to slip past"** — so
   red team continuously `[T]` LLMOps guardrails
8. **"On their own, traces are logs nobody opens."** — tracing pays off only when wired to an
   alert→eval feedback loop `[T]` LLMOps observability
9. **Do not put indirection between the model and the metal.** The case for owning the whole stack is
   that AI systems are not ordinary large-org software — optimising them requires the network, the
   chip and the instruction set. *(A paraphrase of the argument, not a quote — the original entry here
   carried quotation marks around a sentence no transcript contains.)* `[T]` DeSantis
10. **"distributed inference is becoming more of communication and like memory problem and a systems
    problem but it's not really related to computation anymore"** `[T]` ROCm/WideEP talk

---

## 10. Formulas worth memorising

| Formula | Meaning |
|---|---|
| `TTFT ≈ f(prompt_tokens, prefill_throughput)` | first token is a prefill problem |
| `ITL ≈ f(model_bytes / memory_bandwidth)` | each subsequent token is a bandwidth problem |
| `Total latency ≈ TTFT + ITL × output_tokens` | the latency budget decomposition |
| `KL_bound(best-of-N) = log n − (n−1)/n` | upper bound on KL(best-of-N outputs \|\| target preference), loose in edge cases |
| `Acceptance = min(1, p_target/p_draft)` | speculative decoding correctness condition |
| `Cost/1k requests = (input_tok × in_price + output_tok × out_price) × 1000` | unit economics |
| `Break-even(hosted vs self-host) = fixed_GPU_cost / (hosted_price − marginal_self_cost)` | build-vs-buy |
| `Goodput = requests meeting SLO / total requests` | the metric that actually matters |

---

## Sources

Corpus files quoted above (all under `refs/`):
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Production_RAG_RAGOps_Hybrid_Search_Rerankers_HNSW.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Prompt_Management_as_Code_Versioning_Injection_DSPy.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Multi-Agent_Systems_with_LangGraph_The_Klarna_Uber_Lessons.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Guardrails_Stop_Injection_Leaks_Hallucination_NeMo_Guardrails.txt`
- `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts_2/Agent_Memory_Explained_MemGPT_Mem0_Zep_Beyond_the_Context_Window.txt`
- `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
- `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt`
- `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Opening_Note_vLLM_Inference_Meetup_Bengaluru_September_19_2026.txt`
- `Agentic_AI_Infra_transcripts_2/Jerry_Tworek_-_Opportunities_and_Challenges_for_Long_Horizon_Agents.txt`
- `Agentic_AI_Infra_transcripts_2/Jianfeng_Gao_-_Agentic_Modeling_via_Internalizing_Agent_Harnesses.txt`
- `Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt`
- `Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt`
- `Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt`
- `Agentic_AI_Infra_transcripts_2/Panel_Agentic_AI_Infrastructure_Platform.txt`
- `Agentic_AI_Infra_transcripts/Jonathan_Cohen_-_Accelerated_Computing_for_Agentic_AI.txt`
- `Agentic_AI_Infra_transcripts/Peter_DeSantis_-_Constraint_Driven_Innovation_A_Look_at_the_AI_Systems_Problem.txt`
- `Agentic_AI_Infra_transcripts/Chuan_Li_-_A_Lab_Notebook_for_Agents.txt`
- `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
- `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt`
- `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_9_Reasoning_Models.txt`
- `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt`
