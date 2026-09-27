# Topic Taxonomy & Source Map

The master index. Every artifact in this knowledge base is keyed by a topic ID (`T01`–`T19`), so
`T07` resolves to a case study, an interview bank, a cheat sheet, and a design blueprint that all
describe the same system.

- **Provenance legend and reading guide** → [`README.md`](README.md)
- **Build state** → [`PROGRESS.md`](PROGRESS.md)

---

## How the corpus is shaped

The 44 transcripts split into **two halves that barely overlap**:

- **The algorithmic half** — 12 CMU lectures on decoding, search, reasoning and agents. This is an
  academic *inference-algorithms* course. Grepping its ~131k words for serving vocabulary yields
  about 30 hits, and all but three are false positives ("explicitly", "expert", "Title:").
- **The infrastructure half** — 7 vLLM-meetup talks, 14 Agentic-AI-Infra talks, 12 LLMOps talks.
  These contain almost none of the algorithms.

They meet at only a few nodes: **KV cache**, **speculative decoding**, **latency metrics**, and
**cost**. Those four are the bridges, and they are marked below.

### Holes in the transcripts

Several topics you would expect to be central are **absent or name-dropped only**. They are filled
from the supporting repos, and every such claim is marked `[R]` rather than `[T]`:

| Topic | Transcript status | Where the material actually is |
|---|---|---|
| Paged attention | Name-dropped once (vLLM Opening Note), never explained | `llm-inference-engineering` README; guide `04-inference-optimization/05-paged-attention.md` |
| Continuous batching | Named once (llm-d talk) as something that "works at small scale"; mechanism never explained | `llm-inference-engineering`; guide `04-inference-optimization/04-batching-strategies.md` |
| NIXL | ASR-garbled ("NVIDIA and Excel", "P2P K") | named only in `gpu-perf-engineering-resources-main/README.md` `[R]`; reconstructed in `T12` from the llm-d transcript plus `[D]` |
| Pipeline parallelism | Never named; only TP/DP/EP appear | vLLM talk names it as one of 7 parallelism types; guide `04/06-serving-infrastructure.md` |
| Goodput | Word never appears; concept discussed via latency/throughput/KV-hit-rate tradeoffs | guide `04-inference-optimization/01-inference-fundamentals.md` |
| TensorRT-LLM | One mention, no detail | `llm-inference-engineering`; gpu-perf README §4 |
| SGLang | ASR-garbled ("Ashlan", "SLM", "HLANG") | `llm-inference-engineering`; the SGLang case-study talk |
| Ray, Dynamo | Absent | guide `04/06-serving-infrastructure.md` |

### ASR garbling warning

Several proper nouns survive transcription phonetically mangled. **Treat proper nouns in the
Banghua Zhu, Jianfeng Gao and Saurabh Tiwary talks as approximate.** Known corrections:

| Garbled | Actual |
|---|---|
| "Ashlan", "SLM", "HLANG", "ACL" | SGLang |
| "Mouse" | Miles (RL framework, forked from Slime) |
| "LightLLM" | LiteLLM |
| "honey", "honeys" | harness, harnesses |
| "O of tools" | OAuth for tools |
| "NVIDIA and Excel", "P2P K" | NIXL |
| "ambert 32" | a ModernBERT/DeBERTa-class encoder |
| "our chart", "EvoLab", "GFlowIA" | approximate project names |
| "VIO/Imagine" | Veo / Imagen |
| "jeves" | (a model name, unresolved) |

---

## Layer A — Decoding Algorithms

*The CMU half. Concerned with what token comes next and how the search over sequences is run.*

### `T01-sampling-decoding` — Sampling & Decoding Strategies
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_..._2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` | primary | Entropy, cross-entropy, perplexity, KL, calibration, typicality, EOS absorption; temperature; HF `generation_config` defaults (`top_k=50`, `top_p=0.95`); Llama recommended `temperature=0.6`/`top_p=0.9`; GPU non-determinism at temperature 0 |
| `CMU_..._2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` | primary | Ancestral sampling, temperature, top-k, top-p/nucleus, epsilon sampling, locally typical sampling, mirostat, eta/ADA, entropy-based truncation |
| `CMU_..._2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt` | substantive | Diversity metrics (unique-word ratio, bigram overlap); why sampling beats beam on open-ended generation |

### `T02-search-decoding` — Beam, A* and Best-First Search
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_..._2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt` | primary | Beam mechanics, `length_penalty` alpha, diverse beam (Hamming/cumulative/n-gram), stochastic beam + Gumbel-max, curse & blessing of beam search, likelihood trap, search vs model errors, BLEU/ROUGE |
| `CMU_.../CMU_LLM_Inference_5_A_and_Best_First_Search.txt` | primary | Weighted FSAs, semirings, priority-queue search, Dijkstra-style optimal search, A* `f=g+h`, admissibility (and why A* fails when memorized text has probability ≈1), hypothesis recombination, future-cost heuristics, best-first beam search (10x speedup, identical results) |

### `T03-constrained-generation` — Constrained & Structured Generation
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_.../CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt` | primary | Syntactic vs semantic constraints, JSON-as-FSA, logit masking, token healing heuristics, theory-of-computation hierarchy, FUDGE (top-200 + future discriminator), contrastive decoding (2x compute), adversarial decoding, reward-augmented decoding |
| `CMU_.../CMU_LLM_Inference_5_A_and_Best_First_Search.txt` | substantive | Semirings and FSA formulation underpinning constrained decoding |

### `T04-test-time-compute` — Chain-of-Thought, Self-Correction, Reasoning Models
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_.../CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt` | primary | Adaptive computation time (Graves 2016), latent-variable formulation, emergent CoT, three learning routes, mid-training, self-consistency, adaptive self-consistency (Dirichlet/Beta, alpha=3, 0.95 threshold), CoT faithfulness, complexity-based prompting |
| `CMU_.../CMU_LLM_Inference_8_Self-Refine_and_Self-Correction_Methods.txt` | primary | Self-Refine, Self-Debugging, Reflection, edit vectors (512-dim), the negative result (LLMs cannot self-correct reasoning), "hard-to-do is hard-to-check" |
| `CMU_.../CMU_LLM_Inference_9_Reasoning_Models.txt` | primary | STaR, DeepSeek R1 (470B, AIME 15%→70%+), GRPO mechanics (ε≈0.1, temperature 1 for on-policy), length control (S1, LCPO, exceed rate), four cognitive behaviors, RL vs SFT generalization mechanism |
| `CMU_.../CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` | passing | Thinking tool as long-CoT retrofit |

### `T05-verifiers-best-of-n` — Reward Models, Verifiers, Best-of-N
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_.../CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt` | primary | Rejection sampling formalism, KL bound `log n − (n−1)/n`, `n=32` batch-fit rationale, Bradley-Terry, generative reward models (intransitive, order-sensitive, template-sensitive), verbosity bias, preference-data provenance, RewardBench v2, post-hoc content filters |
| `CMU_.../CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` | substantive | Critic reranking 20%→32% on SWE-bench at ~16x inference cost |
| `CMU_..._2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt` | substantive | Best-of-N / reranking / MBR preview |

---

## Layer B — Engine Internals

*Where algorithms meet hardware. **KV cache and speculative decoding are the bridges to Layer C.***

### `T06-inference-fundamentals` — Prefill/Decode, Roofline, Latency Metrics
**Transcript coverage: primary** · **bridge topic**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_..._2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt` | primary | Transformer internals for inference (masked MHA, GQA, SwiGLU, MoE), Llama 3.1 table (405B, layers 32/80/126, hidden 4096/8192/16384, 8 GQA KV heads, 128k context), FLOPs accounting, quadratic attention vs linear layers, hardware landscape, compute rental |
| `CMU_.../CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt` | substantive | The H100 / Llama-3-8B / 100-token FLOPs workout; "encoding is faster than decoding" |
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | TTFT and ITL guidance for interactive chat vs request/session completion time for agentic work; KV hit rate per session; pod saturation and queueing signals |
| `vLLM_.../Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt` | substantive | 16 req/s sustained SLA example; 256 req/s pipeline arithmetic; "decode is a solved problem, producing the first token cheaply is the hard part" |
| `LLMOps_.../Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | substantive | Prefill = compute-bound (sets TTFT), decode = memory-bandwidth-bound; cost per 1k requests, P95, cache hit rate |

### `T07-kv-cache` — KV Cache, Paged Attention, Tiering
**Transcript coverage: partial** · **bridge topic — the single most cross-cutting subject in the corpus**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | KV events per block create/evict (emitted **per request**), offload to CPU saving ~5x TTFT on session return, P2P sharing as the middle ground, retention API with session metadata (an NVIDIA PR at time of speaking), saturation thresholds (e.g. 80% full) |
| `Agentic_AI_Infra_.../Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt` | primary | Rack-scale pooled memory ("Niagara"), memory-pooling vs sharing mode, store/load vs RDMA vs TCP/IP latency, OOM resilience, reused KV without recompute |
| `vLLM_.../Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` | substantive | KV recomputation at 28k inputs destroying throughput; KVA store; PD KV transfer over RDMA |
| `Agentic_AI_Infra_..._2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt` | substantive | Dynamic partitioning over one shared GPU pool with per-attention-type allocators; hybrid-attention (full vs linear/KDA) memory behaviour; KV connector, Mooncake interop |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` | substantive | HiCache HBM→DRAM→external-storage tiering; hybrid prefix cache |
| `CMU_.../CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` | passing | Prompt caching for agent loops; condensation reduces cache effectiveness |
| `llm-inference-engineering-main/README.md` `[R]` | primary | KV cache mechanics and paged attention (PagedAttention), block tables — the transcript gap-filler |
| guide `04-inference-optimization/02,05` `[R]` | primary | KV cache + context caching; paged attention |

### `T08-batching-scheduling` — Continuous Batching, Chunked Prefill, Fairness
**Transcript coverage: partial**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | Flow control and priority bands (premium vs best-effort), least-attained service cutting latencies 2–3x, turn-priority favouring agents near completion so their KV is evicted, saturation thresholds, "FCFS queueing adds nothing" |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | Overlap scheduler, Spec V2, chunked pipeline parallelism, fully async RL rollouts |
| `LLMOps_.../Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | substantive | Continuous batching as the second rung of the cost ladder; PagedAttention named |
| `vLLM_.../Opening_Note_....txt` | passing | Paged attention as prior community topic |
| `llm-inference-engineering-main/README.md` `[R]` | primary | Continuous batching mechanics — the transcript gap-filler |
| guide `04-inference-optimization/04-batching-strategies.md` `[R]` | primary | Batching strategies |

### `T09-speculative-decoding` — Draft Models, MTP, Medusa/EAGLE
**Transcript coverage: partial** · **bridge topic**

| Source | Depth | What it gives |
|---|---|---|
| `CMU_..._2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` | passing | **Framing only — not a treatment.** It does not develop the mechanism, the acceptance mathematics, or any speedup figure. (Previously mis-rated "primary, explained in depth".) |
| `CMU_..._2/CMU_LLM_Inference_1_..._Introduction.txt` | passing | Draft models as a course roadmap bullet |
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | **primary** | The corpus's only production figure: MTP giving ~2x throughput |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | EAGLE, MTP, DeepSeek MTP, Spec V2 |
| `llm-inference-engineering-main/README.md` `[R]` | primary | Medusa, EAGLE, n-gram speculation — the transcript gap-filler |

### `T10-quantization` — INT8/FP8/INT4, AWQ/GPTQ, Accuracy Cliffs
**Transcript coverage: partial**

| Source | Depth | What it gives |
|---|---|---|
| `LLMOps_.../Cut_LLM_Cost_Latency_..._vLLM.txt` | primary | 16-bit→4-bit as the third rung of the cost ladder; AWQ and GPTQ; **"always rerun your evals after quantizing"** (verbatim). **A ~5-minute overview with no quality figures at all** — the 100→42→26→11 ladder is illustrative units, not a measured bill. |
| `CMU_..._2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` | passing | **One student exchange at [53:04], not a treatment** (previously mis-rated "substantive"): quantization worsens temperature-0 non-determinism via rounding error, multi-GPU compounds it, a student floats quantization-fingerprinting |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | Low-precision training, 8-bit and 4-bit, FP4 native rollout, quantization-aware training |
| `vLLM_.../Distributed_Inference_on_ROCm_..._WideEP.txt` | passing | Quantization as an AITER op class |
| guide `03-training-and-adaptation/07-quantization-deep-dive.md` `[R]` | primary | Quantization methods and accuracy tradeoffs |

---

## Layer C — Distributed & Serving Infrastructure

### `T11-parallelism-moe` — TP/PP/EP/DP/CP, MoE, WideEP, ROCm
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` | primary | WideEP (DP attention + EP MoE), EP sizing arithmetic (256 experts/EP8 = 32 per GPU; EP32 = 4 nodes × 8 GPUs = 8 per GPU), MI300 192 GB vs MI355 288 GB HBM, fused MoE 6 kernels→3, AITER, Mori/MoRIO RDMA, NIAH 10-needle matrix (72 configs), max-concurrency ~20k on 1P1D, 28k-input KV recomputation cliff; WideEP keeps **TP inside the node** (the team's own target is TP8 intra-node alongside EP8 + DP16) |
| `Agentic_AI_Infra_..._2/Woosuk_Kwon_-_vLLM_..._for_Agents.txt` | primary | 7 parallelism types (tensor, pipeline, data, expert, sequence, context, decode-context); the B200 DeepSeek prefill worked case — naive single-host 8-way TP loses to a TP+PP+SP+EP mix across 16 GPUs per replica (exact factorisation ASR-garbled; TP is 2) |
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | substantive | TP/DP/WideEP with high-speed RDMA all-to-all for frontier models that cannot fit a node; "MoE routing is NOT done at the llm-d layer, it is at the LM level" |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | DP/TP/CP/PP/EP + DP-attention, DCP, PCB |

### `T12-disaggregation-kv-transfer` — PD/EPD Disaggregation, NIXL, Pooled Memory
**Transcript coverage: primary** · **bridge topic**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | Prefill ≈98% of tokens in agentic workloads; PD ratio comparison 2P2D vs 3P1D; routing to choose prefill vs decode and transfer KV; production claims (Tesla ~3x, AWS/Oracle/Google "Vortex", Capital One) |
| `vLLM_.../Distributed_Inference_on_ROCm_..._WideEP.txt` | primary | Full PD over Mori all-to-all; 2P4D best (flagged preliminary); CI topologies 1P1D and 2P2D; "distributed inference is becoming a communication and memory problem, not a computation problem" |
| `Agentic_AI_Infra_..._2/Jongryool_Kim_-_..._Shared_Memory_KV_Cache_at_Rack_Sc.txt` | primary | Physical rack-scale disaggregation; pooled memory vs sharing mode; comparison against Mooncake and a DRAM baseline |
| `Agentic_AI_Infra_.../Peter_DeSantis_-_Constraint_Driven_Innovation...txt` | substantive | Prefill (compute-bound) vs decode (memory-bandwidth-bound) as *hardware profiles* justifying different silicon |
| `vLLM_.../Scaling_AI_Inference_at_NxtGen...txt` | substantive | Single node split 4 GPUs prefill + 4 decode; disaggregation "adopted by every large organization" |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | Evolution collocate → P/D disaggregation → EPD (encoder disaggregation for VLMs) |
| `gpu-perf-engineering-resources-main/README.md` `[R]` | passing | NIXL named as "a transport layer for moving inference state across memory and network backends" — **the only repo occurrence of the name anywhere in the corpus**. The guide's `11-infrastructure-and-mlops/01-llm-infrastructure.md` was previously listed here as the NIXL gap-filler; it contains no NIXL or KV-transfer content and that attribution was wrong. |

### `T13-serving-engines` — vLLM, SGLang, TensorRT-LLM, llm-d, Kubernetes
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | llm-d architecture: Gateway API Inference Extension, router/EPP, inference pool, filter→score→rank pipeline, Kubernetes-native, KServe complementarity, runs on MacBooks and the inference simulator without a GPU, workload variant autoscaler, SIG agentic inference roadmap |
| `vLLM_.../Opening_Note_....txt` | primary | vLLM vs llm-d division of labour ("Docker for inference" vs "turns many inference servers into one"); vLLM vs TensorRT-LLM — vLLM leads on ease of use, not raw performance; Ollama vs vLLM; vLLM recipes repo; ~90K stars |
| `Agentic_AI_Infra_..._2/Woosuk_Kwon_-_vLLM_..._for_Agents.txt` | primary | vLLM internals: two entry points (`LLM` offline class vs `vllm serve`), OpenAI- *and* Anthropic-compatible APIs, >10 hardware backends with a plugin structure, dynamic memory partitioning, hybrid attention |
| `Agentic_AI_Infra_..._2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents...txt` | primary | Why K8s breaks on agents (bursty, untrusted, single-tenant, human-in-the-loop); Agent Substrate: actor / actor template / worker / "eight-let" / "eight-net", golden snapshots, suspend-resume, wake in low-hundred-ms, targets 10^5 nodes and 10^4 activations/sec |
| `Agentic_AI_Infra_..._2/Banghua_Zhu_-_..._Case_St.txt` | substantive | SGLang ecosystem, HiCache, HiSparse, deployment evolution |
| `vLLM_.../Scaling_AI_Inference_at_NxtGen...txt` | substantive | vLLM V2 engine as the new single-replica default; the missing inference reference architecture |

### `T14-routing-gateways` — Semantic Router, Performance Routing, Gateways
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Inside_vLLM_Semantic_Router.txt` | primary | Pipeline signal→partition→difficulty score→algorithm→model decision; algorithm families (confidence/rem/fusion/workflow loops); fine-tuned encoder classifier; ~50–60 HTTP response headers as observability source of truth; 40% P50 routing-latency figure; enterprise guards; the 92.66 vs 96.0 benchmark |
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | Precise prefix-cache-aware routing (per-block KV events) beating approximate hash routing; the naive-load-balancer anti-pattern; router filter/score chain examples; performance routing vs qualitative model choice |
| `vLLM_.../Scaling_AI_Inference_at_NxtGen...txt` | primary | Rule-based/regex routing in 2025 and why using an expensive MoE as the router itself is a mistake; two model classes on one box; semantic router picks the model, llm-d places the request |
| `LLMOps_.../Cut_LLM_Cost_Latency_..._vLLM.txt` | primary | Difficulty-based routing tier as "the single highest-leverage pattern"; LiteLLM and OpenRouter |
| `Agentic_AI_Infra_.../Jonathan_Cohen_-_Accelerated_Computing_for_Agentic_AI.txt` | passing | Switchyard routing algorithms in the NVIDIA Agent Toolkit |
| `Agentic_AI_Infra_..._2/Panel_Agentic_AI_Infrastructure_Platform.txt` | passing | Routing to smooth global spikes; voice-vs-text heterogeneity |
| guide `11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md` `[R]` | primary | Gateway patterns, fallback, LiteLLM |

### `T15-autoscaling-slo` — Capacity Planning, Autoscaling, SLO/SLA
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `vLLM_.../Scaling_AI_Inference_at_NxtGen...txt` | primary | 256 req/s worked pipeline; enterprise SLA at 16 req/s sustained on half an H100 / half an MI325X; ~2x throughput at the ~85 QPS llm-d crossover; ₹10 lakh→₹5 lakh/month for 200 users; KV utilisation as a deployment knob; "autoscaling models is touchy because you need available GPU capacity" |
| `vLLM_.../Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | primary | Saturation-based workload-variant autoscaler; the 0→1 KEDA problem; operator-defined saturation thresholds; metrics to use per workload class |
| `Agentic_AI_Infra_..._2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents...txt` | substantive | Autoscaling reframed as session multiplexing: agents idle 99.999% of the time; idle-keepalive tuning; warm pools |
| `Agentic_AI_Infra_..._3/Gosia_Steinder_-_Beyond_Harnesses...txt` | substantive | Serverless agent pattern: stateless agent loop + durable session log + sandbox tier |

---

## Layer D — Operations, Agents & Governance

### `T16-agentic-inference` — ReAct, Tools, MCP/A2A, Multi-Agent, Long-Horizon, Memory
**Transcript coverage: primary** — the largest topic in the corpus

| Source | Depth | What it gives |
|---|---|---|
| `CMU_.../CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` | primary | ReAct, CodeAct, planning tools, thinking tool, environment representation (accessibility tree, set-of-marks), context condensation (~2x cost cut at maintained SWE-bench), critic reranking (20%→32% at ~16x cost), multi-agent motivation critique (specialization is weak, parallelization is strong), Claude fans out ~300 references vs OpenAI ~30 serial, context-passing failures |
| `CMU_.../CMU_LLM_Inference_10_Incorporating_Tools.txt` | primary | Tool taxonomy, tool-call tags, WebGPT, Toolformer, function calling as a standard, MCP (17,000+ servers), PAL, sandboxing comparison (Python wrapper vs Docker vs WASM vs E2B/Daytona vs smolagents allow-block lists), RL for tool use, tool creation at test time |
| `LLMOps_.../How_AI_Agents_Actually_Work_ReAct_Tools_Reflexion.txt` | primary | ReAct (Yao 2022) and Reflexion (Shin 2023) mechanics; three production guards; plan-and-execute vs ReAct vs tree search; the "do you actually need the loop?" rule; cost per completed task |
| `LLMOps_..._2/Agent_Memory_Explained_MemGPT_Mem0_Zep_Beyond_the_Context_Window.txt` | primary | Two-tier memory, read/write halves, episodic/semantic/procedural, Mem0 upsert-not-append, Zep temporal knowledge graph, expiry, zero cross-user leakage as a stop-the-line rule |
| `LLMOps_..._2/MCP_vs_A2A_How_AI_Agents_Connect_and_How_to_Govern_Them.txt` | primary | MCP vertical vs A2A horizontal, agent cards, stdio vs HTTP+SSE, four governance pillars, adoption thresholds, trajectory eval |
| `LLMOps_.../Multi-Agent_Systems_with_LangGraph_The_Klarna_Uber_Lessons.txt` | primary | LangGraph mechanics (conditional edges, `interrupt`, checkpointer), topologies, Klarna 80% resolution-time cut then rehiring humans, Uber code-migration agents, "the honest default is one capable agent" |
| `Agentic_AI_Infra_..._2/Jerry_Tworek_-_Opportunities_and_Challenges_for_Long_Horizon_Agents.txt` | primary | 10-min median Codex sessions, 12–16 h frontier horizons, quadratic decay of learning signal, 14 gradient steps/week, value functions, continual learning |
| `Agentic_AI_Infra_..._2/Jianfeng_Gao_-_Agentic_Modeling_via_Internalizing_Agent_Harnesses.txt` | substantive | Harness layers, internalization, environment service at 10^5-agent scale, EvoLab, ~100k trajectories → 73% SWE-bench Verified |
| `Agentic_AI_Infra_..._2/Ankit_Sobti_-_From_Agent_Demos_to_Production...txt` | substantive | Determinism vs model complexity, compounding cost and confusion, hallucination layer, authorization layer, the autonomy curve |

### `T17-observability-evals` — OTel Tracing, LLM-as-Judge, RAGAS, Trajectory Eval
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `LLMOps_.../LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt` | primary | Trace/span model, span internals, OpenTelemetry GenAI semantic attributes, the alert→eval feedback loop, sampling policy (all errors, sampled successes), redaction at the boundary, the 1.2 s / 90 ms span example, "on their own, traces are logs nobody opens" |
| `LLMOps_.../How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt` | primary | Three-part mechanism, rubric+score+why, Jung 2023 biases (position, length, self-preference), pointwise vs pairwise/Elo, 200-curated-beats-10,000-random, the 3.6→4.2 worked example, 5th-percentile inspection, RAGAS/DeepEval/PromptFoo |
| `LLMOps_.../Production_RAG_RAGOps_Hybrid_Search_Rerankers_HNSW.txt` | substantive | The 61%→78%→89% recall funnel; RRF; top-20→4; RAGAS to prove each change |
| `LLMOps_.../Prompt_Management_as_Code_Versioning_Injection_DSPy.txt` | substantive | The 98%→71% JSON-validity regression caught by a nightly eval; prompt version IDs pinned per release |
| `LLMOps_..._2/MCP_vs_A2A_..._Govern_Them.txt` | substantive | Trajectory eval pass rate vs final-answer accuracy |
| `Agentic_AI_Infra_..._2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery...txt` | substantive | Tracing, simulation, evaluation, agent registry and gateway |
| `ai_evals_*.md` (guide root) `[R]` | primary | Eval methodology deep-dives |

### `T18-guardrails-security` — Rails, Injection, Sandboxing, Identity, Governance
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `LLMOps_.../LLM_Guardrails_Stop_Injection_Leaks_Hallucination_NeMo_Guardrails.txt` | primary | Four rail positions (input, output, retrieval, tool call), direct vs indirect injection, delimiting/spotlighting, tool-call gating (allow-list, strict schema, human approval, dry-run, audit log), NeMo Guardrails + Colang, Guardrails AI, Presidio, Llama Guard, false-refusal rate |
| `Agentic_AI_Infra_..._3/Gosia_Steinder_-_Beyond_Harnesses...txt` | primary | Structural vs semantic challenge taxonomy, zero trust, blast radius, no instruction/data separation, agents cannot report their own status, interception layer across three agent classes, identity→delegation→policy→intent-based access, Rossoctl |
| `Agentic_AI_Infra_..._2/Ankit_Sobti_-_..._Production...txt` | substantive | Hallucination layer ("don't recommend unless you know the source"), authorization layer scoping content to job role |
| `CMU_.../CMU_LLM_Inference_10_Incorporating_Tools.txt` | substantive | Sandboxing options and their escape hatches; prompt-injection exfiltration |
| `vLLM_.../The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | substantive | Kata containers for agents; confidential computing; network isolation first |
| `Agentic_AI_Infra_..._2/Saurabh_Tiwary_-_..._Full_Stack...txt` | substantive | Agent identity as distinct from user identity; agent registry and gateway |

### `T19-finops-sovereignty` — Token Economics, Cost Ladders, Sovereign AI
**Transcript coverage: primary**

| Source | Depth | What it gives |
|---|---|---|
| `LLMOps_.../Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | primary | The 100→42→26→11 cost ladder; ~10x cheaper cache reads; difficulty routing roughly halving spend; buy-vs-build break-even; "optimization without measurement is just guessing" |
| `vLLM_.../The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | primary | Token purchasing parity (30¢→16¢ Feb→Apr; 630→91 lines; 8.2→3.6 files); token spend not correlating with business outcomes; the vendor revenue incentive; the three-way sovereignty trust problem; TEEs and attestation |
| `vLLM_.../Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt` | primary | The four sovereignty dimensions — control, trust, economics, continuity; the train-once-infer-constantly asymmetry |
| `vLLM_.../Scaling_AI_Inference_at_NxtGen...txt` | substantive | Sovereign cloud stack layers; the ~2x llm-d crossover translated into ₹ lakh/month |
| `Agentic_AI_Infra_..._2/Panel_Agentic_AI_Infrastructure_Platform.txt` | substantive | Where margin accrues (hardware + vertical integration); customization ladder (context engineering → LoRA → full FT); hosting decided by data containment policy |
| guide `11-infrastructure-and-mlops/04-finops-and-token-economics.md` `[R]` | primary | FinOps methodology, attribution, unit economics |

---

## Cross-cutting themes

Eight themes surface in three or more transcripts. Each is threaded through the topic that owns
it, and flagged here so they are not lost between folders.

1. **KV cache as the central cost/latency lever** — hardware pool (Jongryool Kim), engine memory
   management (Kwon, Zhu), serving economics (LLMOps cost talk), chip architecture (DeSantis).
   *Owned by `T07`, threaded through `T08`, `T12`, `T19`.*
2. **Prefill/decode asymmetry** — the strongest bridge between the two corpus halves. *Owned by
   `T06`, threaded through `T12`, `T11`.*
3. **Harnesses are transitional; platforms are the destination** — Gao ("internalize the harness"),
   Chen ("the model will eat it"), Steinder ("beyond harnesses"), Tworek ("whatever we can
   backprop through wins"). *Owned by `T16`.*
4. **Determinism as the wrapping discipline** — Cohen (LLM + computer science), Chuan Li
   (restricted templates), Sobti (deterministic outcomes are better), Steinder (rule-based control).
   *Owned by `T16`, threaded through `T18`.*
5. **Trajectory-level, not final-answer, evaluation** — MCP vs A2A, Steinder, LLM-as-judge,
   multi-agent (human intervention rate). *Owned by `T17`.*
6. **Agent state across time** — Hockin (snapshot/suspend), memory talk (long-term store), Steinder
   (durable session log), Kwon (KV connector for idle sessions), Kim (pooled KV reuse). *Owned by
   `T16`, threaded through `T07`, `T15`.*
7. **"Don't over-engineer; match tooling to the task"** — appears in nearly every LLMOps episode as
   an explicit decision rule, and in the conference talks as "pick one constraint and go deep."
   *Owned by `T16`, reflected in every decision table.*
8. **Heterogeneity is the defining property** — GPU + CPU + TPU + Trainium + AMD, memory tiers,
   storage layers, protocols. *Owned by `T11`, threaded through `T13`.*

## Quantified anchors

The corpus's signature rhetorical device is the worked cost ladder. These are the highest-value
numbers and each is attributed to a named speaker. They are reused across the four families, so
they are collected here for consistency.

| Anchor | Value | Attribution |
|---|---|---|
| Cost ladder, same model | 100 → 42 → 26 → 11 units (naive → continuous batching → 4-bit → cached prompt) | LLMOps cost talk |
| Cache read vs fresh token | ~10x cheaper | LLMOps cost talk |
| Retrieval funnel | 61% → 78% → 89% recall (dense → +keyword+RRF → +reranker) | LLMOps RAGOps talk |
| Eval score movement | 3.6/5 → 4.2/5 after adding a reranker | LLMOps evals talk |
| Prompt regression | 98% → 71% valid JSON from one added sentence | LLMOps prompt talk |
| Agentic traffic share | ~70% of inference traffic; prefill ≈98% of tokens | llm-d talk |
| MTP throughput | ~2x | llm-d talk |
| CPU KV offload | ~5x TTFT improvement on session return | llm-d talk |
| llm-d crossover | ~2x throughput at ~85 QPS | NextGen talk |
| Enterprise SLA | 16 req/s sustained on half an H100 / half an MI325X | NextGen talk |
| Critic reranking | 20% → 32% on SWE-bench at ~16x inference | CMU agents lecture |
| Context condensation | ~2x cost reduction at maintained SWE-bench | CMU agents lecture |
| Speculative decoding | draft-generates / target-verifies | CMU probability lecture |
| Best-of-N | `n=32` because it fits one batch; KL bound `log n − (n−1)/n` | CMU reward-models lecture |
| R1 reasoning | AIME ~15% → >70%; thinking 800 → thousands of tokens | CMU reasoning lecture |
| Long-horizon RL | 12 h trajectory ⇒ ~14 gradient steps/week | Agentic Infra long-horizon talk |
| Harness efficiency | Same score with half the tokens | Agentic Infra accelerated-computing talk |
| Distillation | ~100k trajectories → 73% SWE-bench Verified | Agentic Infra harness talk |
| Token parity | 30¢ → 16¢ per 10k-token artifact, Feb → Apr | Token Raj talk |
| Klarna | ~80% resolution-time cut, then rehired humans in 2025 | LLMOps multi-agent talk |
