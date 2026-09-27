# Cheat Sheet: Serving Engines & Orchestration

> `T13` · **Transcript coverage:** primary · [Case study](../01-case-studies/T13-serving-engines.md) · [Blueprint](../03-design-blueprints/T13-serving-engines/HLD.md) · [Interview bank](../02-interview-questions/T13-serving-engines.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Hardware backends behind vLLM | **more than 10** | `[T]` Kwon |
| Architecture that makes that tractable | **a plugin structure over one shared core** | `[T]` Kwon |
| New-hardware bring-up cost | the **whole stack re-taken from the ground up** — months, not weeks | `[T]` Kwon |
| KV memory in naive serving | ~**20–40%** of GPU memory | `[R]` guide |
| Paged attention waste | **<4%** (vs ~60–80% contiguous) | `[R]` guide |
| llm-d's three pieces | **gateway + EPP (endpoint picker) + scheduler** | `[T]` llm-d |
| SGLang's core idea | **RadixAttention** — a radix tree over KV prefixes | `[R]` |
| HiCache tiers | **GPU → CPU → storage**, three levels | `[R]` SGLang |
| The Kubernetes ask for agents | **DRA + gang scheduling + checkpointing** | `[T]` Tim Hockin |

---

## The one-table summary

| Engine | Core bet | Best at | Watch out for |
|---|---|---|---|
| **vLLM** | paged attention + continuous batching, plugin core | breadth — most models, most hardware, most features | many knobs; defaults are throughput-tuned |
| **SGLang** | **RadixAttention** prefix reuse + HiCache tiering | prefix-heavy / multi-turn; structured output (compressed FSM) | smaller hardware matrix than vLLM |
| **TensorRT-LLM** | ahead-of-time compiled engines | peak perf on a *fixed* NVIDIA config | recompile per model/GPU/batch shape; narrow flexibility |
| **llm-d** | a **distributed serving stack** above the engine | multi-node scaling, prefix-aware routing, PD | it is a scheduler/gateway layer, not an engine |
| **KServe** | Kubernetes-native model serving | standardised deploy/rollout/scale on K8s | abstraction cost; engine features lag |
| **Ray Serve** | general Python distributed serving | composing LLM + non-LLM steps | not LLM-specific; you build the optimisations |
| **TGI** | HuggingFace-native serving | HF ecosystem integration | less momentum than vLLM/SGLang |
| **llama.cpp / Ollama** | CPU/edge, GGUF | laptops, edge, private single-user | not a datacentre throughput engine |

**The layering that matters** `[T]`: **engine** (vLLM/SGLang — executes the model) underneath a
**distributed stack** (llm-d — decides which replica, routes by prefix, splits phases) underneath a
**gateway** (auth, quotas, semantic routing — see [T14](T14-routing-gateways.md)). Keeping these
three layers distinct is what lets you swap any one of them.

---

## Formulas / architecture arithmetic

**Why vLLM's plugin structure matters** `[T]`: one shared core plus per-backend plugins means a new
accelerator implements a bounded interface rather than forking the engine. The corpus warns that
even so, **bringing up new hardware means re-taking the whole stack** — kernels, attention,
quantization, collectives, MoE, all of it. Budget months, not weeks.

**RadixAttention** `[R]`: prefixes are stored in a **radix tree** keyed by token sequence; a new
request walks the tree and reuses the longest matching prefix. This is strictly more general than a
flat prefix cache because it shares across *any* common prefix, not just an exact-match system
prompt. It is why SGLang wins on multi-turn and few-shot workloads.

**HiCache** `[R]`: extends the radix tree across **GPU → CPU → storage**, so a prefix evicted from
HBM is recovered from DRAM rather than recomputed. Same tiering logic as [T07](T07-kv-cache.md),
implemented in the engine.

**Compiled engines trade flexibility for peak** `[D]`: TensorRT-LLM builds an engine for a specific
model + GPU + parallelism + (often) batch shape. You get the best achievable kernels, and you pay
with a build step per configuration and a painful mismatch when traffic shape changes.

**Kubernetes for agents** `[T]` Tim Hockin — the three primitives inference workloads need and that
standard K8s lacked: **Dynamic Resource Allocation** (attach a specific GPU/topology, not "a GPU"),
**gang scheduling** (all-or-nothing for a multi-node replica — otherwise you deadlock holding half a
job), and **checkpoint/restore** (long-horizon agents must survive eviction). **Agent Substrate**
is the framing for running agents as first-class K8s workloads.

---

## Configuration

```bash
# vLLM — the throughput-tuned baseline
vllm serve <model> --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --enable-prefix-caching --enable-chunked-prefill

# SGLang — RadixAttention is on by default; enable the CPU tier
python -m sglang.launch_server --model <model> --enable-hierarchical-cache

# TensorRT-LLM — build then serve (two steps, config-specific)
trtllm-build --checkpoint_dir <ckpt> --output_dir <engine> --gemm_plugin auto
trtllm-serve <engine>
```

```yaml
# llm-d — the distributed stack is declared, not hand-rolled (Helm values, abbreviated)
gateway:
  class: inference-gateway
routing:
  epp:                                  # endpoint picker
    prefixCacheAware: true
    kvEvents: true                      # per-block KV events, not hashing [T]
```

**Choose the engine by the workload, not by benchmark tables** `[D]`: prefix-heavy multi-turn ⇒
SGLang; broadest model/hardware support ⇒ vLLM; fixed config and maximum NVIDIA throughput ⇒
TensorRT-LLM; multi-node ⇒ add llm-d regardless of engine.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Prefix cache hit rate near zero across replicas | routing ignores prefix locality | the gateway — see [T14](T14-routing-gateways.md) |
| TensorRT engine fails after a model change | engine built for a different config | rebuild; engines are config-bound `[D]` |
| Multi-node replica half-scheduled forever | no gang scheduling | K8s scheduler config `[T]` |
| Agent job evicted mid-run | no checkpoint/restore | add it; long-horizon agents need it `[T]` |
| GPU attached is the wrong topology | no DRA-style selection | you need device-level claims `[T]` |
| Engine upgrade breaks the serving stack | version skew between engine and stack layer | pin versions per layer `[D]` |
| Throughput good, no prefix reuse | engine cache off, or too small | `--enable-prefix-caching`, cache size |
| Serving works in dev, not in prod | dev used one replica; prod needs routing + gateway | build all three layers `[D]` |

---

## Gotchas

- **Keep the three layers separate.** Engine ≠ distributed stack ≠ gateway. Coupling them is how
  teams end up unable to change any one of them `[T]`/`[D]`.
- **vLLM's breadth is an architectural achievement, not an accident.** A plugin structure over a
  shared core is the reason >10 hardware backends are supportable — copy that structure when you
  build internal serving `[T]`.
- **Compiled engines are config-bound.** The flexibility/peak tradeoff is real; a TensorRT engine
  is not a drop-in for a shape change `[D]`.
- **SGLang's RadixAttention is more general than a system-prompt cache** — it matches *any* common
  prefix. That is why it does well on multi-turn and few-shot `[R]`.
- **HiCache is the engine-level version of KV tiering** and composes with, but does not replace,
  a separate KV store `[R]`.
- **Gang scheduling is not optional for multi-node replicas.** Without it, partial allocation
  deadlocks and you burn capacity on jobs that can never start `[T]`.
- **Agents need checkpoint/restore.** A long-horizon agent that is preempted without a checkpoint
  loses all its work — this is a Kubernetes-level requirement, not an app-level nicety `[T]`.
- **Benchmark tables are not transferable.** Every engine's numbers are for a specific
  model/hardware/load. Measure your own `[D]`.
- **The gateway is where the money and the policy live** — do not bury it as an afterthought; see
  [T14](T14-routing-gateways.md) and [T19](T19-finops-sovereignty.md).

---

## When to use what

| Situation | Choose |
|---|---|
| Broadest model/hardware support, fastest to adopt | **vLLM** |
| Prefix-heavy, multi-turn, structured output | **SGLang** (RadixAttention + HiCache) |
| Fixed NVIDIA config, squeeze peak throughput | **TensorRT-LLM** |
| Multi-node, prefix-aware routing, PD disaggregation | engine **+ llm-d** |
| Enterprise K8s standardisation | **KServe** |
| Mixed LLM + Python pipeline | **Ray Serve** |
| Laptop / edge / private single-user | **llama.cpp / Ollama** |
| New accelerator bring-up | expect a from-scratch port `[T]` |

---

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md` `[R]`
