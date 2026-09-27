# CS-16 — Unsloth: 2–4× Faster, Low-VRAM Fine-Tuning

| Field | Value |
|---|---|
| **Module** | PEFT / Training-Efficiency Engineering |
| **Source video(s)** | LLM Fine-Tuning 18: Unsloth Full Guide \| Fine-Tune LLMs 2× to 4x Faster with Lowest GPU Memory |
| **Transcript file(s)** | `LLM_Fine-Tuning_18_Unsloth_Full_Guide_Fine-Tune_LLMs_2_to_4x_Faster_with_Lowest.txt` |
| **Companion code** | `LLM Fine-Tuning-18-unsloth/unsloth_practical.ipynb` (45 cells), `LLM Fine-Tuning-unsloth-vs-hf/unsloth_solution.ipynb`, `LLM Fine-Tuning-unsloth-vs-hf/huggingface_solution.ipynb`, `unsloth-handwritten-notes.pdf` |
| **Prerequisites** | CS-06 (HF `transformers`/`Trainer`/`datasets`), CS-10 + CS-11 (quantization: NF4, bitsandbytes, GGUF), CS-13 (SFT and loss masking), CS-23 (LoRA/QLoRA — rank, alpha, target modules) |
| **Neighbours** | CS-03 (framework landscape), CS-14 (alignment: DPO/ORPO/GRPO run on Unsloth too), CS-15 (LLaMA-Factory — the no-code alternative), CS-17 (Axolotl — the multi-GPU YAML alternative), CS-21 (multimodal fine-tuning with Unsloth) |
| **Difficulty** | Beginner to run; **Advanced to benchmark honestly** and to keep yourself out of the chat-template and merge traps |
| **Hands-on required** | Yes — the notebook runs end to end on a free Colab T4 in ~10 minutes because the dataset is 1,500 rows |
| **Estimated study time** | 6h theory + 4h practical (the practical is mostly *reproducing the baseline*, which the video never does) |

---

## 0. Executive Summary

- **Unsloth is not a trainer.** It is a **kernel-and-graph-rewrite layer** that swaps the hot loops of `transformers`/`peft`/`trl` for hand-written Triton kernels and a hand-written backward pass, then hands the (now much thinner) graph back to TRL's `SFTTrainer` to drive. The notebook's own prose makes this point in Hindi/English [cell 35]: *"Unsloth is NOT a trainer… ⛔ Unsloth loss, backward, optimizer, epochs handle nahi karta"* — which is **half right and half wrong**, see `Correction` C8. The correct one-liner: **Unsloth optimises the forward/backward *kernels* of the LoRA graph; TRL still owns the training loop, loss reduction, optimizer step, and epoch bookkeeping** [cell 35].
- **The headline claim, precisely quoted.** From the README the instructor reads on screen: *"If a normal task takes 10 hours and 40 GB of VRAM using a **standard Hugging Face training**, then it will only take 5 hours and 12 to 16 GB of VRAM with Unsloth"* [19:53]–[20:06]. Elsewhere: *"2 to 3× faster training, 50 to 80% less GPU memory"* [19:15]–[19:28], and the package blurb *"2× faster with up to 70% less GPU memory"* [4:26]–[4:35].
- **The comparison baseline is the entire story, and it is stated nowhere near the numbers.** Every one of those ratios is measured against **an untuned, naive `transformers` + `bitsandbytes` + `peft` stack running stock PyTorch kernels** — the transcript says so explicitly *"standard PyTorch kernel"* [13:04], *"standard hugging face training"* [20:04], *"the normal hugging face not unsloth model… simple hugging face"* [31:00]–[31:07]. Against a **well-tuned** baseline (FlashAttention-2, `use_gradient_checkpointing="unsloth"` or plain GC, `packing=True`, all seven LoRA target modules, `adamw_8bit`), the genuine gap collapses to roughly **1.2–1.8× wall-clock**, not 2–4×. §4.7 is the full analysis, and the rule for the rest of this module is: **never repeat a speed or VRAM number without its baseline.**
- **The single most important mechanical idea: kill the 4-bit dequantize→compute→requantize round trip.** bitsandbytes stores weights in NF4 and, on every forward, dequantizes to FP16/BF16, runs the matmul, and the optimizer's state lives in yet another dtype. Unsloth's kernels keep the 4-bit weights 4-bit in the HBM→SMEM path and fold dequantization into the MMA itself. This is why the VRAM number moves more than the speed number — you are not computing less, you are **moving fewer bytes**.
- **Five mechanisms do the work** (§4): (1) fused Triton kernels for RoPE, RMSNorm, SwiGLU and cross-entropy; (2) a **hand-derived backward pass for the LoRA graph** instead of `torch.autograd`'s tape over the frozen base; (3) flash-style attention that **never materialises the `[T, T]` score matrix**; (4) fusing the attention and MLP epilogues so intermediate activations never hit HBM; (5) pre-quantized checkpoints served from `unsloth/*-bnb-4bit` so the quantization happens once, offline.
- **What the video's own run actually measured.** TinyLlama-1.1B, `unsloth/tinyllama-bnb-4bit`, 1,500 rows of `yahma/alpaca-cleaned`, `max_seq_length=4096`, LoRA `r=32` on all seven projections, 1 epoch, micro-batch 2 × grad-accum 4, `lr=2e-5`, `adamw_8bit`, `packing=True` → **535 seconds (~9 minutes) and 1.9 GB peak reserved VRAM** [53:05]–[53:29]. That is a real number from a real free T4 run and is worth more than every ratio in the README. §11 costs it out.
- **The output quality was bad and the instructor said so.** *"So 1 2 3 5 5 to 6 11… I think the response is not quite good. Maybe we'll have to train more"* [53:59]–[54:07]. The cause is not Unsloth — it is `lr=2e-5` with `r=32` for one epoch on 1,500 rows (see `Correction` C10). Unsloth makes a bad hyperparameter choice cheap, not correct.
- **The two traps that silently ruin runs** are not in the transcript at all and are the reason this module exists: (1) **the chat template** — the notebook hand-rolls an Alpaca string with `### Response:` but TinyLlama-Chat's tokenizer carries a `[INST]`-style template; train/serve mismatch here is invisible in the loss curve; (2) **merging a 4-bit-loaded model with `peft`'s `merge_and_unload()`** — the correct call is `save_pretrained_merged(..., save_method="merged_16bit")`. §6.9 and §10.
- **`train_on_responses_only` is Unsloth's answer to prompt masking** and is the highest-value helper it ships for SFT. It works by **string-splitting the rendered prompt and response halves of your chat template, tokenizing each half separately, and setting `labels = -100` on the prompt span** — which means it silently no-ops if your template's instruction part appears more than once or if you changed the template without changing the arguments. §4.9 has the mechanism and the assertion that catches it.
- **When NOT to use Unsloth** (§8, and the notebook's own cell 41): heavy multi-node / multi-GPU distributed training, pure no-code UI work, non-LLM models, and any project that needs an architecture outside its curated support list. It is a **single-GPU, NVIDIA-first, curated-architecture** tool. That is a deliberate trade, not a defect — but it is a hard boundary.
- **The interview answer, in one sentence.** *Unsloth is a drop-in kernel and backward-pass optimisation for LoRA/QLoRA SFT that gets its wins by fusing kernels, never materialising the attention matrix, hand-writing the LoRA gradient so no autograd tape is kept over the frozen base, and avoiding the 4-bit dequant round trip; its published multipliers are against a naive HF baseline, and against a properly tuned FA2+QLoRA baseline the honest figure is closer to 1.3–1.7× time and 30–50% VRAM.*

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world without this

| Failure | What it looks like | Root cause |
|---|---|---|
| **The OOM on a 16 GB card** | You try to QLoRA-SFT a 7B at `max_seq_length=2048`. `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.31 GiB (GPU 0; 15.78 GiB total capacity)`. | Activation memory for the base model's forward graph, plus the bitsandbytes dequantization scratch buffers, plus the optimizer state, exceed the card. VRAM is not the weights — 4-bit 7B weights are only ~3.9 GB. |
| **The 3×-slower-than-expected run** | Your "QLoRA is 3× cheaper" plan assumed the *speed* saving too. The 4-bit run is often **slower** per step than the 16-bit run because every matmul now has a dequant in front of it. | QLoRA buys memory, not time, on a naive stack. Dequantization is real compute. |
| **The A100 running at T4 speed** | You rent an 80 GB A100 for a 1B-model SFT, and `nvidia-smi` shows 12% utilization. The GPU is idle waiting on Python and on HBM round trips. | Eager-mode `transformers` launches hundreds of tiny kernels per layer; kernel-launch latency and memory traffic dominate at small batch sizes. |
| **The 40-minute turnaround** | You cannot iterate. Change a hyperparameter → 40 minutes → change another → 40 minutes. In an 8-hour day you test 12 ideas. | No packing, no fused kernels, no `adamw_8bit`. |
| **The "it fits but I can't afford it"** | A 3-epoch run on an 8B over 50k examples costs 6 A100-hours ≈ $12–20 per experiment. Ten experiments is a $200 sprint. | Full-precision activations force tiny micro-batches, which force high grad-accum, which lengthens the step wall-clock. |
| **The free-Colab wall** | The single most common reason engineers never fine-tune anything: the tutorial OOMs on the only GPU they have. | No memory-aware kernels. This is precisely the niche Unsloth was built for — *"it gives ability to train large language model even on the free GPUs like Colab and the Kaggle"* [19:29]–[19:39]. |

### 1.2 The state of the art before Unsloth

The lineage matters, because Unsloth is a *layer on top of* all of it, not a replacement. The instructor draws the dependency graph explicitly [15:47]–[18:33]:

```text
NVIDIA CUDA          (C++ framework for the GPU, written by NVIDIA)          [21:17]-[22:10]
   └── Triton        (open-source GPU kernel language/compiler, from OpenAI)  [22:13]-[22:29]
        └── PyTorch  (Python API; C++ core)                                   [15:56]-[16:15]
             └── transformers / peft / trl   (Hugging Face libraries)          [16:19]-[17:45]
                  └── Axolotl · LLaMA-Factory · UNSLOTH                       [16:37]-[16:54]
```

His conclusion, verbatim and correct [16:35]–[16:49]:

> *"this is the backbone of every framework. So whatever framework you are seeing whether it's the Axolotl, whether it's a LLaMA-Factory or Unsloth, right? So it is a backbone of all the frameworks — actually they are using the same source code and on top of it they are making some changes, some enhancement."*

| Era | Approach | What it bought | What it cost |
|---|---|---|---|
| Pre-2023 | Full fine-tuning, FP32/FP16 | Best quality | 4× weights + 8× optimizer state; a 7B needed 2–4× A100-80GB |
| 2023 | **LoRA** (CS-23) | 100–1,000× fewer trainable params; adapter is 10–200 MB | Base weights still in FP16 → 7B ≈ 14 GB, plus activations |
| mid-2023 | **QLoRA** (Dettmers et al.) | 4-bit NF4 base → 7B ≈ 3.9 GB, fits a T4 | **Dequant on every matmul**; naive stack is often *slower* than FP16 |
| 2023 | **FlashAttention-2** | O(T) attention memory instead of O(T²); no `[T,T]` matrix | Requires FA2 kernels + right dtype + right head dim |
| 2023–24 | **Packing** (TRL) | 2–5× throughput on short examples; zero pad tokens | Position-id handling; interacts badly with naive masking |
| **2024–25** | **Unsloth** + friends | Fused kernels, hand-written LoRA backward, no dequant round trip, one-line drop-in | Curated architectures, single-GPU focus, benchmark optics |

> **Beyond the video:** the pre-Unsloth state of the art was **not** naive. By mid-2024 a competent engineer could get most of Unsloth's wins by hand: `attn_implementation="flash_attention_2"`, `use_gradient_checkpointing=True`, `packing=True`, `optim="adamw_8bit"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`, and LoRA on all seven projections. Assembling that stack correctly takes an afternoon and two debugging sessions. **Unsloth's real product is that the correctly-assembled stack is the default, and that the last 30% is not reachable by config alone.** Interviewers who know this will ask you to separate the two.

### 1.3 The naive approach, and precisely why it fails

**Naive approach:** `AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")` → `prepare_model_for_kbit_training` → `get_peft_model(LoraConfig(r=16, target_modules=["q_proj","v_proj"]))` → `SFTTrainer(...)` with default args.

This is *exactly* what the companion repo's HF baseline notebook does (`huggingface_solution.ipynb`, cells 3–5), and it is why the repo's "Unsloth is faster" demo is not a fair test. The naive stack loses on **six** separate axes, each independently fixable:

| # | Naive default | Why it costs | Correct setting |
|---|---|---|---|
| 1 | `target_modules=["q_proj","v_proj"]` | Half the attention projections and **all four** MLP projections are frozen. You train ~40% of the parameters you should, so you need more steps for the same quality — and the *speed* comparison flatters Unsloth, which trains the full 7-module set. | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` |
| 2 | No `attn_implementation` | Falls back to eager attention → materialises `[B, H, T, T]`. At T=4096, H=32, B=2, FP16 that is 2×32×4096²×2 B = **2.1 GB per layer**, before backward. | `flash_attention_2` (or at minimum `sdpa`) |
| 3 | No `packing` | Short Alpaca rows (~200 tokens) in a 4,096 window waste 95% of every forward pass on pad tokens. | `packing=True` |
| 4 | `prepare_model_for_kbit_training` default GC | Gradient checkpointing is on but with the *default* implementation, which stores more and recomputes less than Unsloth's. | `use_gradient_checkpointing="unsloth"` |
| 5 | bitsandbytes dequant per matmul | Every weight tile is dequantized to FP16 in registers on every forward, and the dequant result is written back through the memory hierarchy. | Unsloth's kernels dequantize in the MMA pipeline |
| 6 | `device_map="auto"` | Shards the model across devices with `accelerate`, adding cross-device copies in the hot loop. Fine for inference; wrong for single-GPU training. | Nothing — the model just lives on `cuda:0` |

**Precisely why the naive approach fails as a *benchmark*:** run the naive HF stack and Unsloth's stack on the same data and you measure **the sum of six fixes plus Unsloth's kernels**, then attribute all of it to Unsloth. §4.7 and §12.3 give the protocol that separates them.

> **Correction:** the instructor's framing of the comparison is *"if a normal task takes 10 hours and 40 GB of VRAM using a standard Hugging Face training…"* [19:53]–[20:01]. "Standard Hugging Face training" is doing an enormous amount of unexamined work in that sentence. A `transformers` run with FA2, packing, `adamw_8bit`, all seven target modules and Unsloth's own `"unsloth"` gradient-checkpointing string is also "standard Hugging Face training" — it is just *configured*. The honest statement of the claim is: **against an unconfigured baseline, Unsloth is ~2× faster and uses ~50–70% less VRAM; against a well-configured FA2+QLoRA baseline, the measured gap is roughly 1.2–2.0× time and 20–50% VRAM**, varying by model, sequence length and batch size. See §4.7 for the decomposition.

### 1.4 A concrete motivating example with numbers

Take the video's own run and price it three ways. TinyLlama-1.1B, `yahma/alpaca-cleaned`, 1,500 rows, `max_seq_length=4096`, 1 epoch, effective batch 8.

| | Naive HF (repo's `huggingface_solution.ipynb`) | Tuned HF (FA2 + packing + 7 modules + GC) | Unsloth (video's run) |
|---|---|---|---|
| Trainable params | r=16 on q,v → **~1.1 M** | r=32 on 7 → **25.2 M** | r=32 on 7 → **25.2 M** |
| Peak VRAM | ~4–6 GB (dominated by eager attention at T=4096) | ~2.6–3.2 GB | **1.9 GB measured** [53:26] |
| Wall-clock, 1 epoch, T4 | ~12–18 min (est.) | ~8–11 min (est.) | **535 s ≈ 8.9 min measured** [53:05] |
| Quality after 1 epoch | poor | poor (lr too low) | **poor — the video says so** [53:59] |
| What you actually bought | — | 3× less VRAM, same time | 3× less VRAM, same time |

Note what that table says: on a **1B model at T=4096**, Unsloth's advantage over a *tuned* baseline is largely a **VRAM** advantage. The time ratio only becomes large when you are (a) near the memory ceiling, where the naive run spills or OOMs, or (b) at 7B+, where activation memory dominates and the fused kernels have more work to fuse. That is the honest reading of the same data.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: Unsloth is a rally mechanic, not a new engine

You own a car. A rival team publishes a lap time 2× better than yours. When you read the footnotes, you discover they race the **same engine block** — same pistons, same crankshaft, same ECU firmware — but they have: ported and polished the intake, replaced the exhaust manifold, stripped 200 kg of interior, and remapped the fuel curve. Nothing about the combustion cycle changed. What changed is **how many times per second the engine can complete it, and how much of the car is left over to carry.**

Unsloth is that mechanic. `transformers` still defines `LlamaForCausalLM`. The weights are the same weights. The math is the same math — the instructor is emphatic and correct here: *"exact math no approximation… they haven't done any sort of changes in the mathematics. They just have done the code-level changes and the framework-level optimization. The implementation of the mathematics is like same to same"* [32:09]–[32:22]. What Unsloth does is **rewrite the hot loops**: fuse operations, delete memory round trips, and hand-derive the gradient so PyTorch's autograd tape never has to be built for the frozen base.

**Where this analogy breaks.** Three places, and each one maps to a real limitation.

1. **A mechanic can service any car; Unsloth cannot.** Its rewrites are per-architecture. The support list is curated, not universal. The instructor claims *"whatever model is available over the Hugging Face, all the model is being supported by the Unsloth"* [9:01]–[9:05] — that is false (see `Correction` C12).
2. **A faster engine is faster on every track; Unsloth's win is workload-dependent.** On a short-sequence, small-batch, already-memory-comfortable run, the fusions have little to fuse and the speedup approaches 1.0×. On a long-context run near the VRAM ceiling, it approaches the published numbers because the baseline is *thrashing*.
3. **The mechanic's work is invisible until it is wrong.** A mis-installed exhaust is loud. A mis-compiled Triton kernel or a mis-detected chat template produces **a normal-looking loss curve and a broken model**.

### 2.2 The actual mechanism, stated as a dataflow

Every training step for a LoRA-wrapped transformer is the same nine stages. Unsloth's interventions are all in stages 2, 4, 5, 6 and 9:

```text
 ┌─ 1. EMBED ────────── input_ids[B,T] → hidden[B,T,H]                    (unchanged)
 │
 ├─ 2. PER LAYER ─────────────────────────────────────────────────────────┐
 │    a. RMSNorm         hidden → normed          ◄── FUSED TRITON KERNEL  │
 │    b. QKV proj         normed → q,k,v          ◄── LoRA A/B FUSED       │
 │    c. RoPE             q,k rotated by θ(t)     ◄── FUSED INTO (b)      │
 │    d. ATTENTION        softmax(qkᵀ/√d)v        ◄── NEVER MATERIALISED   │
 │        │                                            as a [T,T] matrix   │
 │        └─ o_proj       → attn_out              ◄── LoRA A/B FUSED       │
 │    e. RMSNorm + MLP    gate/up → SiLU → down   ◄── SwiGLU FUSED         │
 │                                                    (one kernel, one     │
 │                                                     pass over HBM)     │
 └────────────────────────────────────────────────────────────────────────┘
 │
 ├─ 3. FINAL NORM ──────────────────────────────────────────────────────── (fused)
 │
 ├─ 4. LM HEAD ───────── hidden → logits[B,T,V]   ◄── NEVER FULLY          │
 │                                                     MATERIALISED for    │
 │                                                     the loss (chunked   │
 │                                                     / fused CE)         │
 │
 ├─ 5. CROSS-ENTROPY ─── logits,targets → scalar   ◄── FUSED TRITON CE     │
 │                                                     (no [B,T,V] tensor) │
 │
 ├─ 6. BACKWARD ──────── manual analytic LoRA grad ◄── NO AUTOGRAD TAPE    │
 │                                                     over the frozen base│
 │
 ├─ 7. GRAD ACCUM ────── += micro-batch grads      (TRL)                   │
 ├─ 8. OPTIMIZER ─────── adamw_8bit step           (TRL / bitsandbytes)    │
 └─ 9. ZERO GRAD ─────── set_to_none=True          ◄── UNSLOTH PATCH        │
```

> **Beyond the video:** the video's list of Unsloth's optimisations is real but coarse. He names six: *"custom CUDA and Triton kernel"* [21:11], *"fuse attention and MLP operation"* [22:49], *"optimize forward and backward propagation"* [23:30], *"smart gradient checkpointing"* [23:43], *"flash attention compatibility"* [24:05], *"manual back propagation engine, not the PyTorch autograd"* [25:15], *"automatic sequence packing"* [25:34] — seven, actually. What he does not name, and what actually matters most for the VRAM number, is the **elimination of the bitsandbytes dequantize round trip** and the **fused cross-entropy that avoids materialising `[B, T, vocab]`**. At T=4096, B=2, vocab=32,000, FP16, that logits tensor alone is 2×4096×32000×2 B = **524 MB**, and the autograd tape keeps a second copy plus the softmax output plus the gradient. Killing it is worth more than every RoPE fusion combined.

### 2.3 Where the win actually comes from — an arithmetic sketch

Why do fused kernels help *at all* if the FLOPs are identical? Because a modern GPU at small batch size is **memory-bandwidth bound, not FLOP bound**.

Take one RMSNorm on a `[2, 4096, 2048]` FP16 tensor on an A100:

| | Ops | Bytes moved (HBM ↔ SM) | Time at 2 TB/s | Time at 312 TFLOP/s |
|---|---|---|---|---|
| Naive (5 separate kernels: square, mean, rsqrt, mul, mul) | ~5 × 16.8 M = 84 MFLOP | 5 × (8.4 MB read + 8.4 MB write) = **84 MB** | **42 µs** | 0.27 µs |
| Fused (1 kernel, tile stays in SMEM/registers) | ~16.8 MFLOP | 8.4 MB read + 8.4 MB write = **16.8 MB** | **8.4 µs** | 0.05 µs |

**5× faster on a roofline analysis, with identical arithmetic.** That is the whole idea. A Llama-3.1-8B layer has ~11 such elementwise/normalisation ops, × 32 layers × 2 (fwd + bwd) — the fused stack removes on the order of **300 HBM round trips per step** that the naive stack pays for.

> **Beyond the video:** the instructor gestures at exactly this — *"how we can efficiently load the attention operation in the memory, in the RAM actually. So it is all about that"* [24:38]–[24:45] — while describing FlashAttention's IO-awareness. He is right about the principle and does not connect it to Unsloth's *other* kernels. The unifying frame for the whole module is: **Unsloth is an IO-reduction project that happens to also reduce arithmetic in two places (attention, cross-entropy).** Say it that way in an interview and you have understood it.

### 2.4 What Unsloth does NOT do (the boundary of the model)

| Claim people make | Reality |
|---|---|
| "Unsloth makes training converge faster" | No. Unsloth changes **wall-clock per step and bytes moved**, not the optimisation trajectory. Same LR, same data, same steps → same weights (to within floating-point reassociation). Loss curves should track each other within noise. If they diverge, you have a bug, not a feature. |
| "Unsloth is a training framework" | No. TRL's `SFTTrainer` drives the loop. Unsloth ships `UnslothTrainer`/`UnslothTrainerConfig` aliases and RL trainers (GRPO/DPO/ORPO/KTO/GSPO) that wrap TRL, but the epoch/optimizer/logging machinery is TRL's [cell 35]. |
| "Unsloth replaces bitsandbytes" | No, it depends on it for the 4-bit storage format. It replaces the *dequantization path*, not the quantization format. |
| "Unsloth works on any GPU" | NVIDIA is first-class. AMD (ROCm) and Intel (`intel-xpu`) paths exist and are documented but lag. Apple Silicon (MLX) exists for inference/limited training. The notebook's install line pins a **CUDA 12.8** PyTorch wheel index [cell 3]. |
| "Unsloth works on multi-GPU" | It runs under `accelerate`/DDP, but it is **not** the tool's centre of gravity. FSDP in particular has a weak story — the manual backward and kernel rewrites interact badly with parameter sharding. §8.2. |

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Unsloth** | An open-source Python package that patches/replaces the hot kernels of `transformers`+`peft` for LLM fine-tuning. *"Unsloth is an open-source project for the LLM fine-tuning. It helps you to run the end-to-end pipeline"* [3:55]–[4:04]. | It is a layer, not a framework. Everything downstream (CS-15, CS-17, TRL) sits at the same level or above. | It is not a trainer, not a serving stack, and not a quantization *method* — it uses bitsandbytes' NF4 for that. |
| **End-to-end pipeline** | Unsloth's own framing: *"load the model… apply the quantization… train the model… perform the inferencing… evaluate the performance… save the trained model… export"* [4:07]–[4:26]. | Tells you where to look for API surface: four entry points, not forty. | "End-to-end" here means *model lifecycle around the train loop*, not data engineering or eval harnesses. You still bring your own data pipeline (CS-13 §6). |
| **Kernel** | *"It's nothing. It is just a set of programs which have been written in the C++ which is for the GPUs. That's it guys. Kernel is nothing."* [18:45]–[18:55]. | Correct, and worth internalising: a kernel is a function that runs on the GPU. "Custom kernel" means "someone wrote that function by hand instead of composing PyTorch ops". | A kernel is not a driver, not a library, not a framework. FlashAttention is a kernel (several, actually); CUDA is a framework. |
| **CUDA** | *"A framework… written by NVIDIA, specifically for the GPU"* [21:24]–[21:56], in C++. | The substrate everything NVIDIA-side compiles down to. | CUDA ≠ cuDNN ≠ Triton ≠ PyTorch. Triton kernels compile to PTX/SASS, which is the same target CUDA C++ compiles to — Triton is a *language*, not "a change inside the CUDA kernel" (`Correction` C2). |
| **Triton** | An open-source GPU **kernel language and compiler**, originally from OpenAI, now widely contributed to. You write Python-ish code with block-level tensor ops and Triton compiles it to PTX/SASS. | This is how Unsloth ships dozens of kernels without writing raw CUDA C++ — and how those kernels stay readable and portable across GPU generations. | *"Triton is a kernel… written by OpenAI"* [22:13]–[22:29] is only half right — see `Correction` C2. It is a language + compiler; `torch.compile` uses it as a backend; FlashAttention is written in it. |
| **Triton kernel** | A kernel written in Triton and JIT-compiled at first call. | First-call latency: the first training step of every new `(shape, dtype, constexpr)` combination pays a compile cost of ~0.5–10 s. Subsequent steps are cached in `~/.triton/cache`. | People benchmark "step 1" and report a slow number, or benchmark "step 100" and think it is representative. Neither is wrong; report which. |
| **Fused kernel** | One kernel that performs several logical ops, keeping intermediates in registers/SMEM instead of writing them to HBM and reading them back. | The single largest source of Unsloth's speed win. Roofline analysis in §2.3 shows 5× on a single RMSNorm. | Fusion does **not** change the math or the FLOPs. It changes the bytes moved. If you are FLOP-bound (large batch, short sequence), fusion buys little. |
| **Manual backprop / manual autograd** | Unsloth hand-derives and hand-implements the backward pass for the LoRA-augmented layers rather than letting `torch.autograd` build and traverse a tape over the frozen base. *"For the back propagation they are not using a PyTorch autograd graph… it is not using that particular one; instead of that they are using some other logic inside the back propagation"* [25:15]–[25:34]. | Saves the activation tape for the frozen base and lets the backward write directly in the LoRA-only shapes. This is where "no approximation" is hardest to verify — see `Correction` on gradient numerics in §4.3. | It is *autograd-free*, not *gradient-free*. The gradients are exact analytic derivatives of the same function. The claim to check is numerical agreement with autograd, not "it skips backprop". |
| **Autograd tape / computational graph** | PyTorch's DAG of every differentiable op, kept alive from forward to backward. | *"Whenever the back propagation happens, PyTorch autograd is creating one graph… one directed acyclic graph"* [25:21]–[25:29]. Under QLoRA with a frozen base, most of that graph's saved tensors are needed only to propagate *through* the frozen weights, which is wasted memory. | The tape is not the weights and not the activations of the *input*; it is references to the tensors needed to compute `∂L/∂θ`. Deleting it for frozen paths is free. |
| **RoPE (Rotary Position Embedding)** | Position information injected by rotating Q and K vectors by an angle proportional to position, per frequency band. | A favourite fusion target: RoPE is elementwise on `(q, k)` and its `cos/sin` tables depend only on position — so it can be precomputed once and folded into the QKV projection kernel. | RoPE is **not** a learned embedding. The `rope_scaling` config (linear / dynamic / YaRN / llama3) is what makes `max_seq_length` beyond the pretrained window possible — and silently degrades quality if the wrong scheme is used. |
| **RMSNorm** | Normalisation dividing by the root-mean-square of the hidden vector, with a learned per-channel scale. Cheaper than LayerNorm (no mean subtraction, no bias). | The most-fused elementwise op in the stack: 2 per layer in Llama-family, each with an inherent multi-kernel naive implementation. | RMSNorm has no `eps`-free form; the epsilon placement differs between implementations and can produce small numerical drift between HF and Unsloth. |
| **SwiGLU / GeGLU** | The gated MLP: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`. | Three matmuls and one elementwise multiply. Fusing `SiLU(gate)·up` into one kernel removes two HBM round trips per layer per direction. | The gate is `gate_proj` in HF/Llama naming; Mistral/Gemma use the same shape. Forgetting `gate_proj` in `target_modules` is the most common LoRA under-training bug (CS-23). |
| **FlashAttention / flash attention** | IO-aware exact attention. *"FlashAttention is a fast and memory-efficient exact attention with the IO awareness"* [24:34]–[24:38] (the instructor reading the repo README). | Removes the `[T, T]` score matrix. This is what makes `max_seq_length` in the thousands possible at all [24:56]–[25:12]. | "Flash" does **not** approximate. FA1/FA2/FA3 are exact; the savings are IO, not arithmetic. (Note also: the instructor's *"flesh attention"* is a transcription artefact for "FlashAttention".) |
| **Attention-matrix materialisation** | Writing the full `softmax(QKᵀ/√d)` of shape `[B, H, T, T]` to HBM. | At T=4096, H=32, B=2, FP16 that is 2×32×4096²×2 B = **2.15 GB per layer** — and autograd keeps the softmax output too. Never materialising it is the difference between OOM and not. | People think the memory cost is the *compute* cost. Attention FLOPs are `O(T²·d)`; the naive **memory** is `O(T²)`. Flash removes the second, not the first. |
| **Sequence packing / auto packing** | Concatenating several short training rows into one `max_seq_length` window so no compute is spent on pad tokens. *"Sentence one and sentence two — we can combine all together and we can pass"* [26:04]–[26:13]. | 2–5× throughput on Alpaca-shaped data. On the video's 1,500-row Alpaca set at T=4,096 it is the difference between ~9 minutes and ~40. | Packing requires **per-sequence position-id resets** and, if you mask prompts, a per-sequence mask. Naive packing lets token *i* of row 2 attend to row 1 — with correct attention isolation that is harmless, but a broken implementation trains on cross-contaminated context. |
| **QLoRA** | *"Quantized LoRA. You can load the quantized model, and on top of that you can perform the LoRA"* [9:57]–[10:03]. 4-bit NF4 base + FP16 LoRA adapters. | The dominant memory-permitting configuration, and Unsloth's primary mode (`load_in_4bit=True`). | QLoRA is a *memory* technique. It is not faster than LoRA without kernel work — and that kernel work is precisely Unsloth. |
| **NF4 / bitsandbytes** | The 4-bit NormalFloat quantization type and the library that implements it (`bnb`). | What Unsloth's pre-quantized `-bnb-4bit` checkpoints are stored in. | "4-bit" is not a single format. NF4 with double-quant differs measurably from FP4; `bnb_4bit_quant_type` and `bnb_4bit_use_double_quant` are real knobs (CS-11). |
| **Pre-quantized checkpoint** | *"If you are going to load the model from the Unsloth repository, that is a pre-quantized model"* [13:12]–[13:18]. The `unsloth/*-bnb-4bit` repos on the Hub. | Saves the one-time on-load quantization cost (minutes for a 7B) and guarantees a reproducible quantization recipe across users. | The pre-quantized repo is *the same weights* in a different container — *"both are different models… Unsloth has done some sort of optimization"* [12:02]–[12:27]. Same base, different execution. |
| **4-bit dequantize round trip** | The naive QLoRA path: read NF4 weight tile from HBM → dequantize to FP16 in registers → write FP16 tile (or keep in registers) → MMA → repeat next step. | The dominant cost in naive QLoRA. Unsloth folds the dequant into the MMA pipeline so the FP16 tile is never a memory object. | It is not that bitsandbytes is bad — it is that *any* general-purpose dequant kernel has to round-trip through a general memory layout, and a fused one does not. |
| **Gradient checkpointing** | Storing only layer boundaries during forward and recomputing the interior during backward. **Plain** GC trades ~25–35% time for ~60–75% activation memory; Unsloth's `"unsloth"` variant is a *different configuration* trading ~5–15% time for 40–60% (§5.2). Quote the variant with the number. | The other half of the VRAM story. `use_gradient_checkpointing="unsloth"` is Unsloth's smarter variant — it keeps the parts that are cheap to keep and recomputes only the expensive ones. | "Smart" checkpointing in the video [23:43]–[24:05] is described so vaguely it is not actionable; the actionable statement is `use_gradient_checkpointing="unsloth"` in `get_peft_model` — a string, not a bool. |
| **`FastLanguageModel`** | Unsloth's entry-point class with two static methods: `from_pretrained(...)` and `get_peft_model(...)`. | These two calls *are* the Unsloth API for SFT. Everything else is TRL's. | It is not a `transformers` class. `from unsloth import FastLanguageModel` must come **early** (before `transformers` is imported in some versions) so the patches apply to the right modules. |
| **`from_pretrained`** | Loads base model + tokenizer, applies pre-quantization, patches attention/RoPE/MLP, sets up RoPE scaling. | *"Both things I can load using the same method… FastLanguageModel.from_pretrained"* [38:00]–[38:08]. | Returns `(model, tokenizer)` — a tuple, not a model. Forgetting the tuple unpack is the #1 first-run error. |
| **`get_peft_model`** | Attaches LoRA adapters and returns the PEFT-wrapped model. | *"I'm going to use this FastLanguageModel the same object and then I'm calling this get_peft_model"* [39:49]–[39:53]. | It is a *classmethod on `FastLanguageModel`*, not `peft.get_peft_model`. It accepts a superset of `LoraConfig` including `use_gradient_checkpointing="unsloth"`, `max_seq_length`, `use_rslora`, `loftq_config`. |
| **`train_on_responses_only`** | Unsloth helper that patches a `Trainer` so the loss is computed only on the assistant's span. | The single most valuable SFT helper Unsloth ships, and the one most likely to silently no-op. §4.9. | It is **not** the same as TRL's `assistant_only_loss`, though it solves the same problem by a different mechanism (string split vs. `{% generation %}` tags). |
| **`save_pretrained_merged`** | Unsloth's merge-and-save helper: `model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")`. | Required for a deployable single artefact from a 4-bit-loaded run. §6.9. | **Not the same as `model.save_pretrained(dir)`** (that writes the adapter only) and **not the same as `model.merge_and_unload()`** (which is peft's and is wrong on 4-bit bases — §10.3). |
| **Adapter vs merged artefact** | Adapter = `adapter_model.safetensors` + `adapter_config.json` (~10–200 MB). Merged = full weight file (GBs). | Determines your serving path. Adapters need `peft` at load; merged models do not. | *"Save LoRA adapters — this saves ONLY the LoRA adapters, not the full base model"* [cell 38]. The notebook's `model.save_pretrained("lora_model")` writes an adapter, not a model. |
| **`for_inference` / `for_training`** | `FastLanguageModel.for_inference(model)` swaps in the fast generation path; `for_training` reverses it. | *"Always call FastLanguageModel.for_inference(model)"* [cell 36]. Skipping it leaves the training-mode attention path active and you lose the 2× inference claim. | It flips the model to eval and patches `forward`, but it does **not** call `model.eval()` for you in all versions — call both. |
| **Sequence packing flag** | `packing=True` in `SFTConfig` (the notebook) or `packing` in `UnslothTrainerConfig`. | On the video's run it is set [cell 33]. | Packing changes the meaning of "epoch" and of `num_train_epochs` — a packed epoch is a fixed token budget, not a fixed row count. |
| **`optim="adamw_8bit"`** | bitsandbytes' 8-bit AdamW. Halves optimizer-state VRAM vs FP32 Adam (8 bytes/param → 2 bytes/param for the two moments). | With `r=32` on TinyLlama the optimizer state is only ~50 MB, so it barely matters; at full fine-tuning of 8B it is 64 GB → 16 GB. | 8-bit optimizer ≠ 8-bit weights. Unrelated knobs, frequently conflated. |
| **RoPE scaling** | Extending the usable context window by rescaling RoPE frequencies. | *"These numbers are possible due to Unsloth's memory-efficient kernel, smart checkpointing and the RoPE scaling"* [30:21]–[30:27]. | `max_seq_length=4096` on a model pretrained at 2048 **changes the model**, not just the data pipeline. If you then serve the adapter on a base loaded with default RoPE, every long generation degrades. Bake the `rope_scaling` into the saved config. |
| **Curated architecture support** | Unsloth supports a *fixed list* of model families (Llama 1/2/3.x, Mistral, Gemma 1/2/3, Qwen 2/2.5/3, Phi-3/4, DeepSeek, Cohere, Falcon, Yi, TinyLlama, and a growing multimodal set). | *"Unsloth is not only supporting the text-to-text model. It is supporting text-to-image, image-to-text, audio-to-text, text-to-audio…"* [8:43]–[8:58]. True but bounded. | "Supports multimodal" is not "supports your multimodal model". Check the list for *your* architecture and *your* version (Gemma 2 vs Gemma 3 are different code paths). |
| **Triton compile cache** | `~/.triton/cache` — compiled kernels keyed by signature. | Explains "why is my first step 20 seconds and my second step 0.4 s". Also explains why a fresh container every run costs you a minute. | Not a correctness issue. But do not include step 1 in a benchmark mean. |
| **Wall-clock vs step-time speedup** | Step time excludes data loading, tokenization, checkpointing, and eval. Wall-clock includes them. | Unsloth's published ratios are step-time or steady-state throughput. Your notebook's end-to-end number will be smaller. | On the video's run: 535 s total for 188 steps ≈ 2.85 s/step including tokenization of 1,500 rows and logging [53:05]. |
| **`use_rslora`** | Rank-Stabilized LoRA: scales the adapter by `alpha/√r` instead of `alpha/r`, which stabilises training at high rank. | Relevant because Unsloth pushes `r=32`/`r=64` as defaults where standard LoRA scaling starts to misbehave. | `use_rslora=True` is **not** `lora_alpha=r`; the two are alternatives, not complements. |
| **`loftq_config`** | LoftQ initialisation — quantize with an error-minimising init instead of round-to-nearest. | A quality lever at aggressive quantization. | Not a speed lever. Leave `{}` (empty = disabled) unless you are chasing 2-bit. |
| **`UnslothVisionDataCollator`** | Collator for multimodal SFT (CS-21). | Where Unsloth's multimodal support lives. | Out of scope here; mentioned so you do not assume `SFTTrainer`'s default collator will handle images. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 What is actually inside the package

The instructor walks the GitHub tree on screen [4:44]–[5:10]:

> *"Once you write Unsloth GitHub on Google you will get it. So here is the entire source code of the Unsloth… you will find out the different folders like `utils`, `registry`, `model`, `kernels`, `data preparation` and all."*

| Package path | What lives there | Why you would ever read it |
|---|---|---|
| `unsloth/models/llama.py` | The Llama-family patch layer: `FastLlamaModel`, `from_pretrained`, `get_peft_model`, the attention/RoPE/MLP monkey-patches. | The canonical reference for how a patch is structured — copy it to add a new architecture. |
| `unsloth/kernels/` | The Triton sources: `rope_embedding.py`, `rms_layernorm.py`, `swiglu.py`, `geglu.py`, `cross_entropy_loss.py`, `fast_lora.py`, `attention.py`. | Reading `fast_lora.py` is the fastest way to understand the manual-backward claim; it is ~300 lines. |
| `unsloth/models/_utils.py` | Dependency detection, dtype selection, `torch` version gates. | Where the "why did it fall back to the slow path" answer lives — it prints which optimisations were applied. |
| `unsloth/models/vision.py` | Multimodal (CS-21) patch layer. | — |
| `unsloth/trainer.py` | `UnslothTrainer` / `UnslothTrainerConfig` — thin TRL subclasses. | Confirms the "Unsloth is not a trainer" claim: it is ~200 lines of defaults and patches. |
| `unsloth/save.py` | `save_pretrained_merged`, `push_to_hub_merged`, GGUF export. | Where the merge correctness story lives (§6.9). |
| `unsloth/chat_templates.py` | The template registry used by `get_chat_template`. | Directly relevant to §6.8 and the template traps. |

The two calls the notebook makes expose all of this behind a very small surface:

```python
# From unsloth_practical.ipynb, cells 7 and 11 — the entire Unsloth API surface for SFT.
model, tokenizer = FastLanguageModel.from_pretrained(...)   # load + quantize + patch kernels
model = FastLanguageModel.get_peft_model(model, ...)        # inject LoRA with Unsloth's fast path
```

Everything else in the notebook — `SFTTrainer`, `SFTConfig`, `dataset.map`, `save_pretrained` — is Hugging Face or TRL.

> **Beyond the video:** `import unsloth` should be the **first import** in your script. The companion comparison notebook makes this explicit with a comment: `import unsloth  # MUST BE FIRST` (`unsloth_solution.ipynb`, cell 1). The reason is that Unsloth applies some patches by importing and replacing symbols inside `transformers`/`peft`; if those modules are already fully imported and bound into your namespace, a subset of the patches is applied to stale references. The failure signature is subtle: the model loads, the loss decreases, and training is 1.5× slower than it should be. If your Unsloth run is *merely* as fast as HF, check your import order before you check anything else.

> **Correction:** the instructor, describing why Unsloth is fast, says of its Triton kernel: *"the same kernel basically have been used by the ChatGPT also… for the GPT model, for the training"* [13:26]–[13:32]. This is false and it is worth being able to correct it precisely. Unsloth's kernels are Unsloth-authored (Apache-2.0, in `unsloth/kernels/`), and they are not what OpenAI runs to train GPT models. What is true and probably what he meant: (a) **Triton itself** originated at OpenAI and is OpenAI's preferred way to write custom GPU kernels; (b) FlashAttention, which Unsloth integrates, is used broadly across the industry including by large labs. But "Unsloth's kernel is used by ChatGPT" inverts the relationship — Unsloth *uses* the same public tooling OpenAI *created*, and nobody outside OpenAI knows which kernels GPT-5-class training uses. Cite Triton's provenance, not a phantom adoption.

> **Correction:** in the mechanism walkthrough the instructor says *"Triton again, it is a kernel like CUDA… this Triton, it is written by OpenAI… they made some changes inside the CUDA kernel, the existing kernel itself. They have used a Triton kernel"* [22:13]–[22:41]. Three separate errors compressed into one sentence: (1) Triton is a **language + compiler**, not a kernel — a Triton *program* becomes a kernel after compilation; (2) Triton does not "make changes inside the CUDA kernel"; it compiles to PTX/SASS, which sits *below* CUDA C++ in the same stack, as a peer rather than a patch; (3) it is "written by OpenAI" only in the historical sense — the project was released by OpenAI in 2021 and is now developed in the open with heavy contributions from Meta, NVIDIA and others, and is the default backend for `torch.compile`'s inductor. The correct sentence for an interview: **"Unsloth ships hand-written Triton kernels that replace the eager-mode PyTorch ops in the transformer's hot path; Triton compiles them to PTX for the target GPU."**

### 4.2 Fused Triton kernels — the mechanical detail

The five kernels that carry most of the win, and what each one fuses:

#### 4.2.1 The naive version of a transformer layer, in ops

A single Llama-family decoder layer, eager mode, as PyTorch sees it:

```text
 1.  input_layernorm        (RMSNorm)          → 4-6 kernel launches
 2.  q_proj                 (Linear)           → 1 launch  (+ LoRA: 2 more + an add)
 3.  k_proj                 (Linear)           → 1 launch  (+ LoRA: 2 more + an add)
 4.  v_proj                 (Linear)           → 1 launch  (+ LoRA: 2 more + an add)
 5.  rotary_emb(q, k)       (RoPE)             → ~6 launches (cos/sin gather, mul, add, ...)
 6.  attention              (eager)            → 6-8 launches, incl. the [T,T] materialisation
 7.  o_proj                 (Linear)           → 1 launch  (+ LoRA: 2 more + an add)
 8.  residual add                              → 1 launch
 9.  post_attention_layernorm (RMSNorm)        → 4-6 launches
10.  gate_proj              (Linear)           → 1 launch  (+ LoRA)
11.  up_proj                (Linear)           → 1 launch  (+ LoRA)
12.  SiLU(gate) * up        (activation)       → 2-3 launches
13.  down_proj              (Linear)           → 1 launch  (+ LoRA)
14.  residual add                              → 1 launch
                                              ─────────────────
                                              ~45-55 kernel launches per layer
```

× 32 layers × 2 directions ≈ **3,000 kernel launches per training step**. Each launch has ~5–10 µs of CPU-side overhead and, far worse, forces the intermediate tensors out to HBM and back.

#### 4.2.2 The fused version

| Unsloth kernel | What it fuses | HBM traffic removed per layer per direction |
|---|---|---|
| `fast_rms_layernorm` | square → mean → rsqrt → scale → multiply by weight, in one pass with a two-pass online algorithm | 4 → 1 round trips of `[B,T,H]` |
| `fast_rope_embedding` | the `cos`/`sin` gather, the rotation of `q`, the rotation of `k`, and (in newer versions) the in-place rotation ahead of the attention kernel | 6 → 0 extra round trips (folded into the QKV path) |
| `swiglu_fg_kernel` / `geglu_fg_kernel` | `gate_proj(x)` → SiLU → multiply by `up_proj(x)` → feed `down_proj` — forward **and** backward in one kernel | 3 → 1 |
| `cross_entropy_loss` | logits → log-softmax → NLL → mean, computed in **chunks over the vocabulary** so `[B,T,V]` is never a materialised tensor | the entire logits tensor (524 MB at T=4096, B=2, V=32k, FP16) × 2 (fwd + bwd) |
| `fast_lora` | the LoRA `A`/`B` matmuls, the `alpha/r` scaling and the add into the base output, in one pass | 4 → 1 per adapted projection |

> **Beyond the video:** the vocabulary-chunked cross-entropy is the least-discussed and most valuable of these. In the naive path, computing SFT loss on a 4,096-token sequence with a 32,000-token vocabulary materialises a `[B, T, V]` logits tensor. In FP16 that is `2 × 4096 × 32000 × 2 B = 524 MB` **per micro-batch**, plus the softmax output (another 524 MB), plus the gradient with respect to logits (another 524 MB). On an 8 GB card that is a third of your budget consumed by one op. Chunked/fused cross-entropy computes the loss over vocabulary slices and accumulates, so peak memory is `O(chunk_size × T)` instead of `O(V × T)`. This is also what `Liger-Kernel` does (`liger_fused_linear_cross_entropy`) and what "Cut Cross-Entropy" (Apple, 2024) pushes further by never materialising logits at all. If you remember one mechanism from this module, make it this one — it is the largest single VRAM item in SFT and it is invisible in every tutorial's VRAM table.

#### 4.2.3 What "fusion" costs you

| Cost | Detail |
|---|---|
| **Compile latency** | First call for each `(T, H, dtype, has_bias, ...)` signature JIT-compiles. Budget ~2–20 s on a cold cache. Cache lives in `~/.triton/cache`; on ephemeral Colab/Kaggle instances you pay it every session. |
| **Debuggability** | A fused kernel that produces a subtly wrong result gives you a normal loss curve with a slightly-off value. You cannot `print()` an intermediate that no longer exists. Debug by disabling fusions (`UNSLOTH_DISABLE_*` env vars / swapping to a plain HF model) and bisecting. |
| **Numerical reassociation** | Fusing changes the *order* of floating-point operations. A fused RMSNorm's `rsqrt` may be computed in FP32 while the naive one is FP16. Expect last-bit differences, and expect a loss curve that tracks but does not overlap the baseline. That is normal, not a bug. |
| **Version coupling** | Kernels are compiled against specific PyTorch/CUDA/Triton versions. The notebook pins `transformers==4.56.2` and `trl==0.22.2` with `--no-deps` **[cell 3]** — the `--no-deps` is deliberate: Unsloth requires a `trl` version whose transitive dependency pins would otherwise be violated. Upgrading `trl` casually is how you get `ImportError: cannot import name 'SFTConfig'` or silent loss-mask behaviour changes. |

> **Beyond the video:** the notebook's install line is the single most copy-pasteable artefact in the video, and it is worth understanding rather than memorising:
>
> ```bash
> # unsloth_practical.ipynb, cell 3 — the exact four lines from the video
> !pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
> !pip install unsloth
> !pip install transformers==4.56.2
> !pip install --no-deps trl==0.22.2
> ```
>
> - Line 1 pins a **CUDA 12.8** PyTorch wheel. On a Colab T4 that is fine today; on a machine with a driver older than 525 it silently installs a CPU-only torch and your `assert torch.cuda.is_available()` **[cell 5]** fails with a confusing message. Check `nvidia-smi` first and match the `cu1XX` suffix.
> - Line 2 with no version pin installs *today's* Unsloth. Unsloth releases often (weekly-ish); a bug fixed last week is fixed, and a regression introduced yesterday is yours. **For production, pin it** (`unsloth==2025.x.y`).
> - Lines 3–4 pin `transformers` and `trl` in that order, with `--no-deps` on TRL so pip does not "helpfully" upgrade `transformers`. If you skip `--no-deps`, pip resolves TRL's requirements and may install a `transformers` newer than Unsloth's patches expect.
> - The canonical modern alternative is `pip install unsloth` on a machine whose torch already matches, or the documented per-CUDA-version commands from Unsloth's install docs. Never mix: `pip install unsloth` after a manual `torch` reinstall is the second-most-common setup failure.

### 4.3 Manual backprop of the LoRA graph instead of autograd

This is the most misunderstood mechanism in Unsloth, and the one an interviewer is most likely to probe.

#### 4.3.1 The instructor's statement

> *"Manual back propagation engine, not the PyTorch autograd. Means for the back propagation they are not using a PyTorch autograd graph. So whenever the back propagation happens, PyTorch autograd is creating one graph — one DAG, actually, one directed acyclic graph — so it is not using that particular one; instead of that they are using some other logic inside the back propagation."* [25:15]–[25:34]

and the conclusion he draws:

> *"Because of that only, guys, they are not going to lose anything. The accuracy loss is like negligible or very minimum, and they're able to achieve 2 to 3× faster training with a minimum RAM, with 50 to 80% less VRAM."* [27:08]–[27:28]

#### 4.3.2 What autograd actually does here, and why it is waste

Under LoRA, the forward for an adapted projection is:

$$y = W_0 x + \frac{\alpha}{r} B A x$$

with $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ **frozen**, $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$ trainable, $r \ll \min(d_{in}, d_{out})$.

PyTorch's autograd builds a tape that includes:
- the `W_0 @ x` matmul node, whose **only** purpose during backward is to compute $\partial L/\partial x = W_0^\top \cdot \partial L/\partial y$. The gradient w.r.t. $W_0$ is computed and then discarded.
- the LoRA nodes, whose gradients $\partial L/\partial A$ and $\partial L/\partial B$ you actually want.
- **saved tensors**: `x` and `W_0`'s dequantized copy must be kept alive from forward to backward for the LoRA branch to use.

| Saved for backward | Shape | FP16 bytes (Llama-3.1-8B, T=4096, B=2) |
|---|---|---|
| `x` (layer input, per layer, per projection) | `[2, 4096, 4096]` | 64 MB × 7 projections = 448 MB/layer |
| dequantized `W_0` (if the dequant output is what autograd saved) | `[4096, 4096]` | 32 MB per projection |
| attention softmax output (naive) | `[2, 32, 4096, 4096]` | 2.1 GB |

The `x` term alone is the killer: 32 layers × 448 MB = **14 GB of saved activations** before gradient checkpointing, for a model whose 4-bit weights are 4.5 GB.

#### 4.3.3 What the manual backward computes

Unsloth's `fast_lora` backward derives the LoRA gradients analytically and directly, in shapes that match the adapters:

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r}\cdot \frac{\partial L}{\partial y} \cdot (Ax)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\cdot B^\top \cdot \frac{\partial L}{\partial y} \cdot x^\top$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\, W_0^\top \;+\; \frac{\alpha}{r}\cdot \frac{\partial L}{\partial y}\, B A$$

The `W_0` term is a matmul with the **4-bit weights straight out of storage** — no dequantized copy needs to be alive, and no tape node needs to exist, because the derivative w.r.t. a frozen parameter is never needed. What is kept is only `x` and `Ax` (the latter is `r`-dimensional — 32 floats per token instead of 4,096).

**The memory consequence, in one line:** you replace "*keep a `[T, d]` activation and a tape node per projection*" with "*keep a `[T, r]` activation and a hand-written rule*". At $d = 4096$, $r = 32$, that is a **128× reduction** in the saved tensor for that path.

```python
# Schematic of the LoRA forward/backward as Unsloth implements it.
# This is a faithful-shape reconstruction of unsloth/kernels/fast_lora.py, not the
# library's source. Marked as reconstructed.
class FastLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_4bit, A, B, alpha_over_r):
        # X:      [B*T, d_in]
        # W_4bit: [d_out, d_in]  — NF4 packed, dequantized inside this kernel
        # A:      [r, d_in]      — FP16
        # B:      [d_out, r]     — FP16
        XA = X @ A.t()                       # [B*T, r]   <-- tiny, and this is ALL we save
        base = dequant_mm(X, W_4bit)         # fused dequant + matmul, W_4bit stays 4-bit in HBM
        out = base + alpha_over_r * (XA @ B.t())
        ctx.save_for_backward(X, XA, A, B, W_4bit)
        ctx.alpha_over_r = alpha_over_r
        return out

    @staticmethod
    def backward(ctx, dY):
        X, XA, A, B, W_4bit = ctx.saved_tensors
        a = ctx.alpha_over_r
        dA = a * (dY.t() @ XA)               # [r, d_in]
        dB = a * (XA.t() @ dY).t()           # [d_out, r]
        dX = dequant_mm_t(dY, W_4bit) + a * (dY @ B @ A)
        return dX, None, dA, dB, None        # None for W_4bit -> no weight gradient
```

Two things to notice in that sketch: (1) the only activations saved are `X` (which the *next* layer downstream also needs) and `XA`, which is `[n, r]` — 32 floats per token at `r=32` — instead of a `[T, d]` FP16 tensor; (2) `W_4bit` is returned as `None` from `backward`, so the frozen base never gets a gradient tensor allocated. That is the whole trick, and it is why "manual autograd" saves memory more than it saves FLOPs.

> **Beyond the video:** *"not going to lose anything… accuracy loss is negligible"* [27:11]–[27:19] is stated as a reassurance but is really a **testable claim**, and you should test it. The acceptance test is straightforward: take a small model, run N steps with HF+peft (autograd) and N steps with Unsloth, seed both, and compare the **LoRA adapter weights** after step 1. They should agree to within FP16 reassociation error (relative difference ~1e-3, not ~1e-1). If they disagree by more, the manual backward has a bug for your architecture. Unsloth's own CI does kernel-vs-torch comparisons of exactly this kind. Do not take "no approximation" on faith — it is a marketing-grade phrase that happens to be testable.

> **Correction:** the companion notebook's markdown cell 35 states, with emphasis, that *"Unsloth loss, backward, optimizer, epochs handle nahi karta"* — "Unsloth does not handle loss, backward, optimizer, or epochs." **The 'backward' half is flatly contradicted by the video's own theory section** at [25:15]–[25:34], which describes Unsloth's manual backprop engine as one of its core optimisations, and contradicted by the package contents (`unsloth/kernels/fast_lora.py` implements a `torch.autograd.Function` whose `backward` is Unsloth's). The **loss** half is also wrong: `unsloth/kernels/cross_entropy_loss.py` is a fused cross-entropy and `FastLanguageModel` patches the model's loss path. What *is* true is the narrower claim: **Unsloth does not implement a training loop** — the epoch iteration, gradient accumulation, optimizer `step()`, LR scheduling, checkpointing and logging all come from TRL's `SFTTrainer`. Restate it correctly as: *Unsloth owns the layer- and loss-level *kernels* including the backward for the LoRA graph; TRL owns the *loop*.*

#### 4.3.4 Which parts of the backward Unsloth does *not* hand-write

| Component | Forward | Backward |
|---|---|---|
| LoRA-adapted Linear | Unsloth (fused) | **Unsloth (manual analytic)** |
| RMSNorm | Unsloth (fused) | Unsloth (fused, analytic) |
| RoPE | Unsloth (fused) | Unsloth (fused) |
| Attention (FlashAttention path) | FA2 Triton | FA2 Triton |
| SwiGLU/GeGLU | Unsloth (fused) | Unsloth (fused) |
| Cross-entropy | Unsloth (fused, chunked) | Unsloth (fused) |
| Optimizer step | — | bitsandbytes `adamw_8bit` (not Unsloth) |
| Grad accumulation, clipping, scheduler | — | **TRL / `transformers.Trainer`** |
| Weight tying, lm_head | `transformers` | `transformers` autograd |

So the accurate statement is: **Unsloth replaces autograd over the transformer's own differentiable ops; it does not replace autograd for the loss's interaction with the `lm_head` when the head is not fused, nor does it supply the optimizer.** In practice, with `untie_embeddings`/`lm_head` folding the whole graph is covered.

### 4.4 No attention-matrix materialisation

The instructor reads the FlashAttention README on screen and lands on the right phrase [24:29]–[24:45]:

> *"This repository provides the official implementation of FlashAttention and FlashAttention-2. FlashAttention is a fast and memory-efficient exact attention with IO-awareness. So how we can efficiently load the attention operation in the memory, in the RAM actually. So it is all about that."*

#### 4.4.1 The arithmetic, worked

For a batch of $B$ sequences, $H$ heads, sequence length $T$:

| Tensor | Naive (eager) | FlashAttention |
|---|---|---|
| $S = QK^\top/\sqrt{d}$ | `[B, H, T, T]` written to HBM | never written; tiled in SMEM, block by block |
| $P = \mathrm{softmax}(S)$ | `[B, H, T, T]` | never written (recomputed in backward from the output and the log-sum-exp) |
| $\mathrm{d}P$, $\mathrm{d}S$ | `[B, H, T, T]` each | never written |
| **Total attention-internal HBM traffic** | $O(BHT^2)$ reads/writes, **×4 tensors** | $O(BHTd)$ + $O(BHT)$ for the log-sum-exp |

Concretely for the video's configuration — TinyLlama-1.1B has $H = 32$ heads, $T = 4096$, $B = 2$, FP16:

| | Bytes |
|---|---|
| $S$ (one tensor) | $2 \times 32 \times 4096^2 \times 2 = 2{,}147{,}483{,}648 \approx 2.15\ \text{GB}$ |
| Naive total (S, P, dP, dS) | $\approx 8.6\ \text{GB}$, **per layer** |
| FlashAttention peak | $O(BHTd) = 2 \times 32 \times 4096 \times 128$ elements $\approx 134\ \text{MB}$, independent of $T^2$ |

The 1.9 GB peak the video measured **[cell 34 output, 53:26]** is only possible because none of those `[T,T]` tensors exist. A naive eager run of the same config does not fit on a T4 at all.

#### 4.4.2 Why this is the *long-context* claim, and what its real limit is

> *"Unsloth can handle up to 300k token training. So, 300k — just imagine, two to three lakh of words. It is equal to one single book."* [28:29]–[28:43]

The mechanism for the context-length claim is **(a) FlashAttention removing the $T^2$ term and (b) `use_gradient_checkpointing="unsloth"` removing the per-layer activation term and (c) RoPE scaling making the position encoding valid past the pretrained window**. All three must hold. The instructor names all three [30:21]–[30:27] but presents the number as a property of Unsloth, which it is not.

> **Correction:** *"the same thing in the normal HF training will give you the out-of-memory error… the maximum limit is 28k token"* [28:45]–[29:05]. There is **no 28,000-token limit in Hugging Face `transformers`**. Sequence length is bounded by your VRAM, your attention implementation, your dtype and your checkpointing settings — nothing else. A `transformers` + FA2 + gradient-checkpointing + `adamw_8bit` run on an 80 GB card will train far past 28k on a model that supports it, and the same run on a 12 GB card will OOM at 4k. The instructor's own table contradicts the framing two minutes later: he shows Unsloth reaching 78k on 24 GB and 340k on 80 GB, while "HF" gets 28k on 80 GB — a 12× gap at identical hardware is not a library limit, it is a configuration gap. The defensible claim is: **out of the box, a naive `transformers` QLoRA run OOMs or thrashes at long context because eager attention materialises $[B,H,T,T]$; Unsloth does not, so it reaches far longer contexts on the same card.**

> **Correction:** the context-length table the instructor reads [29:14]–[30:18] — Llama-3.1-8B, 8 GB → 3,000 tokens, 12 GB → 21,000, 16 GB → 40,000, 24 GB → 78,000, 80 GB → 340,000 — is **internally inconsistent and not reproducible as stated**. Three problems: (1) 8 GB → 3,000 tokens is anomalously low — that is the model *barely* fitting, so the entry is really "8 GB is below the floor for this model", not "8 GB gives you 3k"; (2) the jump from 12 GB → 21k to 16 GB → 40k is superlinear (1.33× memory → 1.9× context) while 16 → 24 GB (1.5× memory) gives only 1.95× context — the numbers are rounded from a benchmark whose settings (batch size, gradient checkpointing mode, RoPE scaling scheme, `max_seq_length` cap) are not stated, and context length is very sensitive to all four; (3) the 80 GB figure "*3 lakh 40k*" is delivered as *"3 lakh 40,000"* = 340,000, which exceeds the native context of Llama-3.1-8B by 4× and is only reachable with an aggressive RoPE scaling scheme that measurably degrades short-context quality. **Treat the whole table as an order-of-magnitude illustration, not a planning number.** For planning, use §11.3's formula and your own measurement.

### 4.5 Fused RoPE and fused cross-entropy

#### 4.5.1 RoPE, precisely, and what "fused" means for it

RoPE rotates each 2-element channel pair of $q$ and $k$ by an angle that depends on the token position $m$ and the channel index $i$:

$$\theta_{m,i} = m \cdot \omega_i, \qquad \omega_i = b^{-2i/d}, \qquad b = 10{,}000 \ (\text{or } 500{,}000 \text{ for Llama-3})$$

$$q'_{m} = R_{\theta_{m}} q_m, \qquad R_{\theta_m} = \mathrm{blockdiag}\!\left(\begin{bmatrix}\cos\theta_{m,i} & -\sin\theta_{m,i}\\ \sin\theta_{m,i} & \cos\theta_{m,i}\end{bmatrix}\right)_{i=1\ldots d/2}$$

In HF's eager implementation this is a chain of ~6 ops per tensor: build `cos`/`sin` from the rotary embedding module, gather by position, `reshape` to pairs, `mul`, `mul`, `add`, `reshape` back — for **both** $q$ and $k$, so ~12 kernels with 6 intermediate tensors of shape `[B, T, H, d_head]`.

Unsloth's `fast_rope_embedding` (and the newer in-place `apply_rotary_pos_emb` variants) fuses this into **one kernel per tensor** that:
1. computes $\cos/\sin$ on the fly from $\omega_i$ and the position index (no rotary-embedding module call, no gather),
2. loads the pair, applies the rotation, and
3. writes back in place.

| | Naive | Fused |
|---|---|---|
| Kernel launches per layer (q and k) | ~12 | 2 |
| Intermediate tensors alive | 6 × `[B,T,H,d]` | 0 |
| Extra HBM round trips | 6 | 0 (in-place) |

> **Beyond the video:** the *hidden* cost of the naive path is not the kernels, it is that `cos`/`sin` tables of shape `[B, H, T, d/2]` are **saved for backward** and, in some HF versions, **recomputed for backward** — either way they consume memory proportional to $T$ that a fused in-place version does not. The `rope_scaling` configuration (linear / dynamic-NTK / YaRN / llama3) also lives in this kernel: when Unsloth says *"these numbers are possible due to… the RoPE scaling"* [30:25], it means the fused kernel is aware of the scaling scheme. **That awareness is exactly why `max_seq_length` is a load-time argument to `from_pretrained` and not a trainer argument.** Change it later and you have changed the model's position encoding without retraining.

#### 4.5.2 Cross-entropy: the tensor nobody budgets for

$$L = -\frac{1}{N}\sum_{t \in \mathcal{R}} \log \frac{\exp(z_{t,y_t})}{\sum_{v=1}^{V}\exp(z_{t,v})}$$

where $z \in \mathbb{R}^{N \times V}$ are the logits at supervised positions $\mathcal{R}$, and $N = B \cdot T$ (times the mask).

The naive implementation materialises $z$ (`lm_head` output), then `log_softmax(z)` (another $N\times V$), then `nll_loss` (reduces). Autograd then keeps $\mathrm{d}z$ (another $N \times V$).

**Worked budget for the video's run** (TinyLlama: $V = 32{,}000$, $T = 4096$, $B = 2$, FP16):

| Tensor | Shape | Size | Notes |
|---|---|---|---|
| `hidden_states` | `[2, 4096, 2048]` | 33.6 MB | cheap |
| `lm_head.weight` | `[32000, 2048]` | 131 MB | **tied to the embedding** — no extra copy in TinyLlama |
| logits $z$ | `[2, 4096, 32000]` | **524 MB** | the expensive one |
| `log_softmax(z)` | `[2, 4096, 32000]` | **524 MB** | |
| `dz` for backward | `[2, 4096, 32000]` | **524 MB** | |
| **Peak added by the loss alone** | | **≈ 1.57 GB** | |

On a T4 with 15.8 GB total, ~1.57 GB for the loss is 10% of the card — but the video's **entire measured peak was 1.9 GB** [53:26]. That single fact tells you the fused cross-entropy is active and doing its job; without it, the run's peak would be ~3 GB higher. (For TinyLlama's small vocabulary this is a modest win. At $V = 152{,}064$ — Qwen2.5, Llama-3.1 — the same config gives $2 \times 4096 \times 152064 \times 2 = 2.49$ GB per tensor, ≈ **7.5 GB for the loss path alone**. That is the difference between fitting on an A100-40GB and not.)

Fused implementations avoid this by:
- **Chunking over $V$**: compute log-sum-exp per chunk, accumulate, never hold the full $z$. (Liger's `fused_linear_cross_entropy`; Unsloth's kernel.)
- **Fusing the `lm_head` matmul into the loss**: never write `z` at all — take the weight matrix and the hidden state, and compute the loss tile by tile. (Apple's "Cut Your Losses in Large-Vocabulary Language Models", 2024.)

> **Beyond the video:** this is where you separate a real "2× less VRAM" claim from a marketing one. Ask: **what is the vocabulary size, the sequence length and the batch size?** At `V=32k, T=512, B=1` (a typical short-instruction SFT), the unfused loss is `512 × 32000 × 2 × 2 B = 65 MB` — irrelevant, and Unsloth's VRAM advantage is entirely from attention, checkpointing and the dequant path. At `V=152k, T=8192, B=2`, the unfused loss is **5 GB per tensor** and the fused kernel dominates the comparison. **A published VRAM ratio without (V, T, B, model size) is not a number, it is an advertisement.**

#### 4.5.3 What the `lm_head` costs at training time vs inference

| Phase | `lm_head` cost | Mitigation |
|---|---|---|
| Training forward | `[B,T,H] × [H,V]` matmul — `2·B·T·H·V` FLOPs. For the video: `2·2·4096·2048·32000 = 1.07 TFLOP` per step. | Fused linear+CE removes the tensor, not the FLOPs. |
| Training backward | 2× the forward FLOPs (`dx` and `dW`), plus a `[N,V]` gradient | Fused CE writes `dz` in chunks. |
| Inference (prefill) | Sampled at the last position only in practice | — |
| Inference (decode) | `[B,1,H] × [H,V]` per token — this is why decode is memory-bandwidth bound on the head weight | Quantize the head; or serve with vLLM's fused head |

> **Beyond the video:** TinyLlama ties `lm_head.weight` to `embed_tokens.weight` — there is exactly one `[32000, 2048]` matrix used twice. Unsloth's `from_pretrained` **untie/clone** handling matters: if you untie as part of loading, you suddenly have a second 131 MB FP16 parameter (and its AdamW state). Most SFT should **not** untie. If you see your model's parameter count jump by exactly `V × H` after loading, you untied by accident.

### 4.6 Avoiding the 4-bit dequantize → compute → requantize round trip

This is the mechanism that most directly explains the VRAM number and the one the video never names.

#### 4.6.1 What naive bitsandbytes QLoRA does per matmul

```text
NAIVE PATH (bitsandbytes Linear4bit.forward, simplified):

  HBM                                  SM (registers / SMEM)               HBM
  ┌─────────────────────────┐          ┌──────────────────────────┐
  │ W_nf4 packed + absmax   │  ──read──▶│ dequantize:              │
  │ + quant_map             │          │   code = lookup[nibble]   │
  │ absmax/127              │          │   w = code * absmax       │
  └─────────────────────────┘          │   (per-block of 64)       │
                                        └────────────┬─────────────┘
                                                     │
  ┌─────────────────────────┐                        ▼
  │ X (FP16)                │  ──read──▶  ┌──────────────────────┐
  └─────────────────────────┘             │ GEMM: Y = W_fp16 @ X │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Y (FP16)  ───────────┼──write──▶ HBM
                                          └──────────────────────┘
```

The forward is fine-ish. The problems are:
1. **The dequantized `W_fp16` tile is a real tensor** with a real memory layout. In `bnb`'s implementation the dequant is a separate kernel whose output is written to global memory in some code paths and re-read by the GEMM — a full `[d_out, d_in]` FP16 write + read per projection per microbatch. At 4096×4096 FP16 = 32 MB, × 7 projections × 32 layers × 2 directions = **14 GB of avoidable HBM traffic per step**.
2. **The optimizer's gradient for `W` is computed and thrown away** unless the layer is frozen — but with `prepare_model_for_kbit_training` the base is frozen yet autograd still walks the path in many configurations.
3. **The activation gradient `dX` needs `W_fp16` again in backward**, so either you dequantize twice (forward and backward) or you cache the dequantized weight for backward — which costs, at 8B, `32 layers × 7 × 4096 × 4096 × 2 B = 7.5 GB` of cached FP16 weights. **This is the single largest hidden cost in naive QLoRA and the one that makes "4-bit saves memory" partially untrue.**

#### 4.6.2 What Unsloth does

```text
FUSED PATH (Unsloth's matmul + dequant):

  HBM                                  SM (registers / SMEM)
  ┌─────────────────────────┐
  │ W_nf4 packed + absmax   │  ──read──▶ ┌─────────────────────────┐
  └─────────────────────────┘            │ tile into SMEM          │
                                          │ dequantize *in registers*│
  ┌─────────────────────────┐            │           │              │
  │ X (FP16) tile           │  ──read──▶ │           ▼              │
  └─────────────────────────┘            │   MMA (w tile × x tile)  │
                                          │           │              │
                                          │           ▼              │
                                          │   accumulate in FP32     │
                                          └───────────┬─────────────┘
                                                      ▼
                                          Y (FP16) ──write──▶ HBM
```

Key differences:
- **`W_fp16` never exists as a tensor.** Dequantization happens in registers immediately before the MMA, tile by tile. Nothing is written back.
- **Backward re-dequantizes.** Rather than caching 7.5 GB of FP16 weights, Unsloth re-runs the dequant in the backward pass. Compute is cheaper than HBM on every GPU you will rent.
- **The accumulation is FP32.** `bnb` also does this; it is why 4-bit QLoRA does not collapse numerically.
- **Fused with the LoRA add.** The `+ (α/r)·B(Ax)` term is folded into the same kernel's epilogue, so the base output and the adapter output are summed before hitting HBM.

| | Naive bnb QLoRA | Unsloth fused |
|---|---|---|
| Dequantized weight tensor materialised | Yes (fwd and/or cached for bwd) | **No** |
| Cached FP16 weights for backward (8B) | up to ~7.5 GB | **0** |
| HBM traffic per adapted projection | weight-nf4 read + fp16 write + fp16 read + x read + y write | weight-nf4 read + x read + y write |
| Extra kernel launches per projection | 2 (dequant, GEMM) | 0 |

> **Beyond the video:** there is a **third** path you should know about, because it is what the "post-QLoRA" world uses: quantize the *activations* too and run the whole GEMM in INT8/INT4 with INT32 accumulation (GPTQ-style/W4A8 kernels, or `torchao`'s `int4_weight_only` + `int8` dynamic activation quant). That is faster again and is where the field moved in 2025. It is **not** what Unsloth does for training — Unsloth's compute dtype stays BF16/FP16. The relevant sentence for an interview: *Unsloth eliminates the dequantization **memory round trip**; it does not eliminate the dequantization **arithmetic**. Weight-only-INT4-activation-INT8 kernels eliminate some of both.*

#### 4.6.3 The `bnb_4bit_compute_dtype` trap

The single most common QLoRA misconfiguration, present in neither the video nor the notebook:

```python
# WRONG for a modern GPU: defaults compute dtype to FP32 in many bnb versions
model, tokenizer = FastLanguageModel.from_pretrained(name, load_in_4bit=True)

# RIGHT: be explicit. Unsloth sets this for you, but your HF baseline must too,
# or you are comparing 4-bit-Unsloth against 4-bit-with-FP32-compute-HF,
# which is a ~1.7x speed difference on its own.
from transformers import BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # <-- the line everyone forgets
)
```

> **Correction:** this is the mechanism behind a large part of the instructor's speed comparison and it is not mentioned once. `FastLanguageModel.from_pretrained` sets `bnb_4bit_compute_dtype` from the `dtype` argument (auto-detected: `torch.float16` on a T4, `torch.bfloat16` on an A100/L4) **[cell 4 comment: "Auto-detect (FP16 on T4, BF16 on A100/L4)"]**. The companion HF baseline (`huggingface_solution.ipynb`, cell 3) constructs `BitsAndBytesConfig(load_in_4bit=True)` and nothing else. Depending on the bitsandbytes version, that leaves `bnb_4bit_compute_dtype` at its FP32 default, which means **every dequantized matmul runs in FP32 on a T4 whose FP32 throughput is 1/32 of its FP16 throughput** (T4: 8.1 TFLOPS FP32 vs 65 TFLOPS FP16-with-FP32-accumulate). The resulting "Unsloth is 2–3× faster" measurement is, in significant part, "Unsloth computes in FP16 and the baseline computes in FP32." **Fix the baseline's config before you attribute the delta to kernels.**

### 4.7 What the "2× faster / 60% less VRAM" claims are measured against

This section is the most important in the module for an interview, and the one the video handles worst.

#### 4.7.1 Every claim in the video, with its stated baseline

| Claim | Timestamp | Stated baseline (verbatim) | What the baseline actually is |
|---|---|---|---|
| "2× faster with up to 70% less GPU memory" | [4:26]–[4:35] | none stated | Unsloth's package blurb. Baseline implied only. |
| "2× 3× faster and 50 to 80% better memory efficiency" | [15:00]–[15:06] | *"built on top of the Hugging Face transformer, PEFT and TRL but it enhanced them"* | The un-enhanced versions of those libraries, i.e. config defaults. |
| "10 hours / 40 GB → 5 hours / 12–16 GB" | [19:53]–[20:06] | *"using a standard Hugging Face training"* | A `transformers`+`peft` run with default settings. |
| "Llama-3.1-8B: 20 GB → 7–8 GB" | [20:16]–[20:35] | same | same. Note: 20 GB for an 8B QLoRA run already implies an *efficient* baseline (weights are 4.5 GB), so "7–8 GB" is a ~2.7× reduction, not the 3.3× the 40→12–16 figure implies. **The two examples in the same breath disagree with each other.** |
| "HF: 2 hours → Unsloth: 1 hour" | [20:31]–[20:35] | same | Consistent with a ~2× claim. |
| "Unsloth: 340k tokens on 80 GB; HF: 28k tokens on 80 GB" | [30:41]–[31:09] | *"the normal hugging face, not Unsloth model… you're not optimizing using the Unsloth and nothing, simple hugging face"* | Eager attention + default checkpointing. A 12× gap is a *configuration* gap, not a library gap (`Correction` in §4.4.2). |

**The pattern:** the baseline is described in prose ("standard", "normal", "simple") and never pinned to a config. Every claim therefore lands somewhere between "true against a strawman" and "true against a peer."

#### 4.7.2 The decomposition — where the 2× actually comes from

Here is the honest attribution for a 7B QLoRA SFT at T=2048, effective batch 16, on an A100-80GB, going from a naive HF stack to Unsloth. Numbers are **worked estimates** calibrated to publicly reported component-level speedups; they are illustrative of the *shape*, not measured here.

| Step | Change | Δ speed | Δ peak VRAM | Whose optimisation |
|---|---|---|---|---|
| 0 | **Baseline:** HF + `bnb` 4-bit + peft, `target_modules=[q,v]`, eager attention, no packing, `compute_dtype` default | 1.00× | 1.00× | — |
| 1 | `attn_implementation="flash_attention_2"` | **1.15–1.30×** | **0.55–0.70×** | FA2 (Dao et al.), not Unsloth |
| 2 | `bnb_4bit_compute_dtype=torch.bfloat16` (if it was FP32) | **1.2–1.8× on a T4 class GPU; ~1.0× on an A100** | ~1.00× | bitsandbytes config |
| 3 | `packing=True` (on short Alpaca-shaped rows) | **1.5–3.0×** on *tokens/sec of useful work* | 1.00× | TRL |
| 4 | `use_gradient_checkpointing=True` (standard) | 0.75× (slower!) | **0.55–0.70×** | HF `Trainer` |
| 5 | `optim="adamw_8bit"` | ~1.0× | **0.85–0.95×** (small at LoRA rank) | bitsandbytes |
| 6 | `target_modules` = all 7 | ~0.93× (more params) | ~1.0× | CS-23 knowledge |
| 7 | **Unsloth's fused kernels + manual backward + fused CE + no dequant round trip** | **1.15–1.40×** | **0.70–0.85×** | **Unsloth** |
| — | **Product of steps 1–7** | **≈ 2.0–4.0×** | **≈ 0.25–0.45×** | mixed |

**The reading:** Unsloth's *own* contribution to the headline multiplier is roughly **1.2–1.4× time and 15–30% VRAM**. The rest — the majority of the "2–4×" — is FlashAttention, packing, bf16 compute and the baseline's own unconfigured defaults, all of which you can have without Unsloth. Against a baseline that already has steps 1–6, the honest Unsloth delta is **~1.2–1.4× time, ~15–30% VRAM** — still valuable, and a completely different number from "2–4× and 50–80%."

> **Beyond the video:** how the published 2–4× figures are produced, and why they are not dishonest. Unsloth's benchmark methodology (documented in their repo's `benchmarks/` and blog posts) is to run **the same model, same dataset, same hyperparameters** through (a) `transformers`+`peft`+`bnb` with the library's own standard recipe and (b) Unsloth, and report the elapsed time and peak reserved memory. That *is* a fair "drop-in replacement" test — it answers "if I switch libraries with zero config effort, what do I get?" It is **not** an answer to "how much better is Unsloth than the best possible HF config?" Those are different questions, and the published number only answers the first. When you quote 2–4×, say which question you are answering.

#### 4.7.3 The companion repo's own comparison is not apples-to-apples

The repo ships a direct comparison — `unsloth_solution.ipynb` vs `huggingface_solution.ipynb` — and it is worth reading precisely because it shows how the mistake is made in practice. **Both notebooks run the same 200 rows of `yahma/alpaca-cleaned`, `max_steps=50`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `lr=2e-4`.** The differences:

| Config | `unsloth_solution.ipynb` | `huggingface_solution.ipynb` | Is the difference Unsloth? |
|---|---|---|---|
| Model | `unsloth/tinyllama-bnb-4bit` (pre-quantized) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + `BitsAndBytesConfig(load_in_4bit=True)` | Partly — but the HF one is *also* paying a one-time quantization at load, which is amortised over 50 steps only |
| `bnb_4bit_compute_dtype` | set by Unsloth (auto bf16/fp16) | **unset** → FP32 default in many bnb versions | **No — this is a config bug in the baseline** |
| `bnb_4bit_use_double_quant` | set by Unsloth | unset | No |
| `target_modules` | `["q_proj","k_proj","v_proj","o_proj"]` (4) | `["q_proj","v_proj"]` (2) | **No — the baseline trains half the parameters** |
| `lora_alpha` | 16 | 16 | — |
| `r` | 16 | 16 | — |
| Gradient checkpointing | Unsloth default | `prepare_model_for_kbit_training` default | Partly |
| `packing` | not set in either | not set in either | — |
| `attn_implementation` | Unsloth's patched attention | library default | Partly |
| Attention implementation | Unsloth flash-style | eager | **Partly — but the fix is one kwarg** |
| `torch.cuda.reset_peak_memory_stats()` | called before load **and** training timing starts at `from_pretrained` | called only before training | **No — the Unsloth notebook's `train_time` includes model loading; the HF one's does not.** Read cell 4: `start = now()` comes *before* `from_pretrained`. In the HF notebook (cell 5), `start = time.time()` comes *after* `from_pretrained` and `get_peft_model`. |
| Inference path | `FastLanguageModel.for_inference(model)` | none | Yes — Unsloth |

That last row is the sharpest: **in the repo's own benchmark, the Unsloth timing includes model download + quantization + patching, and the HF timing excludes model download + quantization.** If anyone has ever run those two notebooks and got "Unsloth is slower", that is why.

> **Correction:** the video's central quantitative claim — *"if a normal task takes 10 hours and 40 GB of VRAM using a standard Hugging Face training then it will only take 5 hours and 12 to 16 GB of VRAM with Unsloth"* [19:53]–[20:06] — should never be repeated without the qualifier, and the qualifier is not a footnote, it is most of the claim. Restated accurately: **"Against `transformers`+`peft`+`bitsandbytes` left at the library defaults — eager attention, no sequence packing, FP32 4-bit compute dtype, and LoRA on two projections — Unsloth is roughly 2× faster and uses roughly 50–70% less VRAM. Against the same stack configured with FlashAttention-2, `packing=True`, a bf16/fp16 4-bit compute dtype, gradient checkpointing and LoRA on all seven projections, the measured advantage is approximately 1.2–1.4× time and 15–30% VRAM, and it grows with sequence length, model size, and proximity to the VRAM ceiling."** That is the sentence to take into an interview, and it is defensible in both directions.

### 4.8 Memory and compute accounting for the video's run

The video's own measurement, and then the arithmetic that explains it.

#### 4.8.1 Measured

From the notebook (`unsloth_practical.ipynb`, cell 34) and the video [53:05]–[53:32]:

```text
===== UNSLOTH TRAINING STATS =====
Training time (sec): 535          # ≈ 8 min 55 s
Peak GPU VRAM (GB):  1.9          # torch.cuda.max_memory_reserved() / 1024**3
CPU RAM used (GB):   <reported>   # process.memory_info().rss delta
```

Setup: TinyLlama-1.1B (`unsloth/tinyllama-bnb-4bit`), 1,500 rows of `yahma/alpaca-cleaned`, `max_seq_length=4096`, LoRA `r=32` / `alpha=32` on 7 projections, 1 epoch, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `lr=2e-5`, `warmup_ratio=0.1`, `optim="adamw_8bit"`, `packing=True`, T4.

188 optimizer steps reported in the log [52:11]–[52:15] → 535/188 ≈ **2.85 s per optimizer step**, or ~0.71 s per micro-batch.

#### 4.8.2 The arithmetic

**Parameter budget (exact — this is the useful part):**

| Component | Count | Notes |
|---|---|---|
| TinyLlama-1.1B total params | 1,100,048,384 | 22 layers, `d_model=2048`, `d_ff=5632`, 32 heads / 4 KV heads, `d_head=64` |
| LoRA on `q_proj` (`2048×2048`) | 2 × 32 × 2048 = 131,072 | A: `[32,2048]`, B: `[2048,32]` |
| LoRA on `k_proj` (`2048×256`) | 32×2048 + 256×32 = 73,728 | GQA: KV dim is 4×64 = 256 |
| LoRA on `v_proj` (`2048×256`) | 73,728 | |
| LoRA on `o_proj` (`2048×2048`) | 131,072 | |
| LoRA on `gate_proj` (`2048×5632`) | 32×2048 + 5632×32 = 245,760 | |
| LoRA on `up_proj` (`2048×5632`) | 245,760 | |
| LoRA on `down_proj` (`5632×2048`) | 32×5632 + 2048×32 = 245,760 | |
| **Per layer** | **1,146,880** | |
| **× 22 layers** | **25,231,360** | **This is exactly the number the video shows: "2 cr 52 lakh 31,360"** [40:21]–[40:27] |
| **Fraction of total** | **25,231,360 / 1,100,048,384 = 2.294%** | |

The instructor reads *"2.242%"* at [40:36]–[40:38] and then, three minutes later, *"around 3.94 percentage of all the parameter"* at [41:30]–[41:35].

> **Correction:** the trainable-parameter percentage. The video gives **2.242%** [40:38] and then **3.94%** [41:32] for the same model and the same LoRA config. Neither is right. The exact count — which the video itself reports correctly as 25,231,360 trainable out of a 1.1B model — gives **25,231,360 / 1,100,048,384 = 2.29%**. The first spoken figure (2.242%) is a rounding/transcription slip close to the truth; the second (3.94%) is wrong by 72% and appears to be a mis-transcription of the earlier one. **Always compute this number rather than reading it off a screen:** `sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())`, which is exactly the notebook's cell 14.

**VRAM budget for the run:**

| Item | Size | Note |
|---|---|---|
| Base weights, NF4 + double quant | ~0.60 GB | 1.1 B × ~4.5 bits/param ≈ 0.62 GB (plus absmax/quant metadata) |
| LoRA params, FP16 | 25.2 M × 2 B = **0.050 GB** | negligible |
| LoRA grads, FP16 | 0.050 GB | |
| `adamw_8bit` state | 25.2 M × 2 B = **0.050 GB** | two 8-bit moments |
| **Model + optimizer subtotal** | **~0.75 GB** | |
| Activations (fused kernels + `packing`) | ~0.9–1.1 GB | the residual |
| CUDA context, cuBLAS/cuDNN workspaces, Triton | ~0.2 GB | unavoidable |
| **Measured peak (`max_memory_reserved`)** | **1.9 GB** | [53:26] |

**Cross-check:** a naive eager-attention run at T=4096 would need, per layer, `[B,H,T,T] = 2×32×4096²×2 B = 2.15 GB` — i.e. **more than the entire measured peak, for one layer's attention scores alone**. That is the strongest single piece of evidence in the whole video that the fused/flash path is active. If you ever see a "Unsloth" run whose `max_memory_reserved` at T=4096 is 6–9 GB, the flash path is not engaged.

**Throughput:**

| Metric | Value |
|---|---|
| Steps | 188 |
| Wall-clock | 535 s |
| Seconds / optimizer step | 2.85 s |
| Micro-batches per step | 4 |
| Tokens per micro-batch (packed, at T=4096) | 4,096 |
| Tokens per step | 16,384 |
| **Tokens / second** | **3,057** |
| T4 peak FP16 | 65 TFLOPS |
| **Achieved % of peak FP16** | irrelevant — the run is memory-bound at batch 2; the right comparison is tokens/s across configs |

> **Beyond the video:** 3,057 tokens/s for a 1.1B 4-bit LoRA on a T4 is a sane number. For calibration: a well-configured T4 run of the same model sits in the 2,500–4,500 tokens/s band depending on `T`; an A100-80GB sits ~8–12× higher; an H100 ~20× higher. If your run is at 800 tokens/s on the same hardware, you are not kernel-bound, you are **dataloader-bound** — check `num_proc` on `dataset.map`, whether tokenization happens per step, and whether `dataloader_num_workers` is 0 (the default). Unsloth cannot help with any of that, and it is the #1 reason a real pipeline does not reproduce a benchmark.

### 4.9 `train_on_responses_only` — how it locates the assistant span

The video never mentions this function. It is the single most valuable thing Unsloth ships for SFT, it is what makes the difference between a model that answers and a model that autocompletes your prompt (CS-13 §4.4), and it fails *silently*.

#### 4.9.1 The signature

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=SFTConfig(...))

# Unsloth-specific: patch the trainer so loss is computed ONLY on the response span.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer.train()
```

For a ChatML model (`Qwen`), it becomes:

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

For the Alpaca flat string the video uses, it is:

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="### Instruction:\n",
    response_part="### Response:\n",
)
```

#### 4.9.2 The mechanism, step by step

Unsloth's implementation (in `unsloth/chat_templates.py`) does **not** use TRL's `{% generation %}` markers. It uses string surgery on the *rendered* conversation:

```python
# Faithful reconstruction of the algorithm in unsloth/chat_templates.py.
# Marked: reconstructed — the real function is ~120 lines handling batching and
# per-row offsets; this is the idea.
def _train_on_responses_only(tokenizer, instruction_part, response_part, num_proc=None):
    def _mask(example):
        # 1. Render the full conversation to TEXT with the model's own chat template.
        #    Note: this is the SAME template the model was pretrained with.
        full = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        # 2. Tokenize the FULL string -> this becomes input_ids.
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

        # 3. Find how many tokens the INSTRUCTION side occupies.
        #    Because the template is deterministic, the first occurrence of the
        #    response marker is the boundary. Everything before it (plus the marker
        #    itself) is the masked prefix.
        idx = full.find(response_part)
        # 4. Tokenize the prefix on its own. This is the ONLY reliable way to know
        #    how many tokens the prefix occupies: tokenizing "abc" + "def" separately
        #    can give a different token count than tokenizing "abcdef", because BPE
        #    merges across the boundary.
        prefix_ids = tokenizer(full[:idx], add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        # 5. labels = input_ids with the prompt span set to -100.
        labels = list(full_ids)
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1]*len(full_ids)}
    return _mask
```

Four properties fall out of that design, and each one is a failure mode:

| Property | Consequence |
|---|---|
| The boundary comes from **`full.find(response_part)` — the first occurrence** | If your instruction text *contains* the marker string (e.g. a system prompt that says "you are an assistant", or a few-shot example containing `### Response:`), the split lands at the wrong place. |
| The prefix length is computed by **re-tokenizing the prefix alone** | This is the correct approach (BPE merges across a boundary change the count), but it is O(n) extra tokenization per row and requires the tokenizer to be deterministic in `add_special_tokens=False` mode. |
| The **instruction_part argument is unused for the split** — only `response_part` matters for locating the boundary in most versions | Passing a wrong `instruction_part` often still "works" while doing nothing, which is why the bug is silent. |
| Prompt tokens are set to `-100` **after** tokenization | The prompt still occupies positions in `input_ids`; the model still *sees* it (that is required — it is the conditioning), it just contributes no loss. `-100` is `CrossEntropyLoss(ignore_index=-100)`'s default, so no config is needed downstream. |

#### 4.9.3 Verifying it worked — the assertion that saves you a week

```python
# Run this ONCE, before training. It costs 10 seconds and catches the silent no-op.
import torch

patch = train_on_responses_only(
    trainer,
    instruction_part="### Instruction:\n",
    response_part="### Response:\n",
)
trainer = patch  # returns the same trainer, patched

batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
input_ids = batch["input_ids"][0]

masked = (labels == -100).sum().item()
supervised = (labels != -100).sum().item()
total = labels.numel()

print(f"total={total} masked={masked} supervised={supervised}")
print(f"supervised fraction = {supervised/total:.1%}")

# ASSERTIONS — these three catch ~90% of masking bugs:
# 1. Something is supervised. If this is 0, the mask is inverted or the marker never matched.
assert supervised > 0, "NOTHING is supervised -> marker not found or mask inverted"
# 2. NOT everything is supervised. If this is total, the patch is a no-op.
assert supervised < total, "EVERYTHING is supervised -> train_on_responses_only did nothing"
# 3. The supervised fraction is in a plausible band. For instruction data with
#    short answers this is 10-60%; if it is >90% your prompts are tiny or the
#    boundary is misplaced; if it is <2% you are training on almost nothing.
assert 0.02 < supervised / total < 0.95, f"implausible mask ratio {supervised/total:.1%}"

# 4. The real proof: decode the SUPERVISED span and read it.
sup_ids = input_ids[labels != -100]
print("=" * 60)
print("SUPERVISED TEXT:")
print(tokenizer.decode(sup_ids, skip_special_tokens=False)[:500])
print("=" * 60)
# The printed text must be the ASSISTANT'S ANSWER ONLY, including the trailing
# <|eot_id|>/</s>. If you see the user's question in there, the boundary is wrong.
# If you see a truncated first word, the token boundary is off by one and you are
# training on half a token.
```

> **Beyond the video:** `train_on_responses_only` returns a **new** `SFTTrainer` in some versions and **mutates in place** in others. Always assign the return value (`trainer = train_on_responses_only(trainer, ...)`) — if you call it and discard the result, you get an unmasked run with no error and no warning. The failure signature is a loss curve that starts 2–4× lower than it should (because the easy prompt-prediction tokens are included) and a model that, at inference, begins emitting the *prompt* before the answer. CS-13 §14 has the sibling diagnostic for the raw-TRL path.

#### 4.9.4 `train_on_responses_only` vs TRL's `assistant_only_loss` vs doing it yourself

| Approach | How it finds the span | Works with any template? | Fails how |
|---|---|---|---|
| **`train_on_responses_only`** (Unsloth) | String match on the rendered prompt; re-tokenize the prefix to get the boundary | **Yes** — you supply the marker strings | Silent no-op if the marker appears earlier in the text, or if you pass the wrong marker and the version ignores `instruction_part` |
| **`assistant_only_loss=True`** (TRL `SFTConfig`) | The chat template's `{% generation %}` / `{% endgeneration %}` blocks; TRL tracks the token spans the template marks as generated | No — the template **must** carry generation tags | Raises or silently reverts to all-token loss on templates without the tags (many community templates lack them) |
| **Manual `-100` in the collator** | Whatever you write | Yes | You own every edge case: BPE boundary, EOS, padding, packing |
| **Not masking at all** | — | — | Trains the model to generate the prompt. See CS-13 §4.4 |

> **Beyond the video:** **`packing=True` + response-only masking is the sharpest edge in this whole module.** When packing, TRL concatenates multiple rows into one `max_seq_length` window with per-sequence attention isolation and per-sequence position ids. Your `-100` mask must be applied **per source row before packing**, and the packed labels must preserve that mask. In older TRL versions, `packing=True` combined with a manual label mask dropped the mask for all but the first sequence in each pack. The modern, safe combination is `SFTConfig(packing=True, assistant_only_loss=True)` with a generation-tagged template — that pairing is designed to compose. With Unsloth's `train_on_responses_only` plus `packing=True`, verify the assertion block above **on the packed batch**, not on a raw row, and check that the supervised fraction is consistent across all sequences inside one pack.

### 4.10 The chat-template traps that silently ruin a run

The video's notebook builds its training string by hand:

```python
# unsloth_practical.ipynb, cell 24 — the notebook's actual template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
```

The model is `unsloth/tinyllama-bnb-4bit`, i.e. TinyLlama-1.1B-Chat-v1.0, whose tokenizer ships a chat template using `<|user|>\n...\n<|assistant|>\n`. So the notebook is training the model on a format it was never pretrained on, and the video's inference call reuses the *same* hand-rolled string [cell 36] — which is why it "works" at all: **train format == serve format.** It just is not the model's native format, and a TinyLlama-Chat fine-tuned through a foreign template is measurably worse than the same run through the correct one.

| Trap | Symptom | Diagnosis | Fix |
|---|---|---|---|
| **Hand-rolled template ≠ `tokenizer.chat_template`** (the notebook's situation) | Everything "works", quality is mediocre. Loss floor is higher than it should be. | `print(tokenizer.chat_template)`; compare to your training string. | Use `tokenizer.apply_chat_template(messages, tokenize=False)` and build the dataset from a `messages` column. |
| **Train template ≠ serve template** | Eval notebook looks fine; the deployed endpoint emits the prompt, or rambles, or ignores the system message. | Grep the serving stack for the literal header strings; diff against training. | Make the template a single constant imported by both. |
| **Template renders a system prompt the base model never saw** | Model ignores or becomes confused by system messages. | Check whether the base's `chat_template` includes a `system` role at all — several Llama-2/ChatML-era templates do not. | Drop system messages from training data, or pick a model whose template supports them. |
| **`add_special_tokens=True` in the wrong place** | Double BOS, or a missing BOS. Loss is fine; generations start with `<s><s>`. | Print the first 8 token ids of a training row and of a generation prompt. | Be explicit: `add_special_tokens=False` when you build `input_ids` yourself; let `apply_chat_template(..., tokenize=True)` handle it otherwise. |
| **EOS stripped by a cleaning step** | The model never learns to stop. It answers correctly and then continues inventing turns forever. | Decode a training row; look for the trailing `<|eot_id|>`/`</s>`. | Append `tokenizer.eos_token` explicitly — which the notebook does: `+ EOS_TOKEN` in `format_data`, with the comment *"EOS token is mandatory, otherwise generation may never stop"* [cell 24]. |
| **`packing=True` with a flat-text dataset** | Rows are concatenated; the model sees answer→instruction→answer with no separator beyond EOS. | Decode one packed batch. | With flat text, packing is only safe because packing inserts EOS between rows — verify it does. With a `messages` column, use `assistant_only_loss`. |
| **Template changed between training and merging/serving** | Adapter is fine; the deployed model is not. | `diff` the `chat_template` field in the saved `tokenizer_config.json` against the base's. | Save the tokenizer **with** the adapter (`tokenizer.save_pretrained(dir)`) — which the notebook does [cell 38]. |
| **Truncation splits a template marker** | Rare; a marker token sequence is cut in half at `max_seq_length`. | Check the last tokens of a truncated row. | Keep `max_seq_length` well above your p99.5 length; or truncate on row boundaries before templating. |

> **Beyond the video:** the video's own run gives you the cleanest possible demonstration of the trap without naming it. The notebook trains TinyLlama-Chat on `### Instruction:` / `### Response:` and then prompts it with the identical string [cell 36], getting *"1 2 3 5 5 to 6 11"* [53:59]. Two independent things are wrong there: the hyperparameters (`lr=2e-5`, 1 epoch — see C10) and the template. **When you cannot tell which of two bugs caused a bad output, fix them one at a time.** That is the whole of applied debugging, and §14 is the full playbook.

---

## 5. The End-to-End Pipeline

### 5.1 The ten-stage spine

```text
 ① ENVIRONMENT ─────────────────────────────────────────────────────────────┐
    Colab/Kaggle/local + CUDA-matched torch + unsloth + pinned trl         │
    in : a GPU and a shell          out : importable unsloth               │
    fail: torch is CPU-only (wrong cuXXX wheel) → assert cuda.is_available │
 ───────────────────────────────────────────────────────────────────────────┘
 ② CONFIG + SEEDS ───────────────────────────────────────────────────────────
    in : nothing                    out : SEED, max_seq_length, dtype,     │
                                          load_in_4bit as named constants   │
    fail: seeds unset → two runs of the same config give different adapters │
 ───────────────────────────────────────────────────────────────────────────┘
 ③ LOAD BASE ───────────────────────────────────────────────────────────────
    FastLanguageModel.from_pretrained(model_name, max_seq_length, dtype,    │
                                      load_in_4bit)                         │
    in : a Hub id                    out : (model, tokenizer), kernels      │
                                           patched, RoPE scaling applied    │
    fail: OOM at load (too-large max_seq_length), or HF token 401           │
 ───────────────────────────────────────────────────────────────────────────┘
 ④ INJECT LoRA ─────────────────────────────────────────────────────────────
    FastLanguageModel.get_peft_model(model, r, target_modules, lora_alpha,  │
                                     lora_dropout, bias, use_gradient_      │
                                     checkpointing, random_state)           │
    in : base model                 out : PeftModel with ~0.1–4% trainable  │
    fail: too few target_modules → underfits; GC off → OOM at long T        │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑤ INSPECT BEFORE TRAINING ─────────────────────────────────────────────────
    print_trainable_parameters / named_parameters / dtype / device /        │
    peft_config / torch.cuda.memory_summary()                               │
    in : the model                  out : evidence you configured it right  │
    fail: you skip this and debug a 3-hour run instead of a 30-second check │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑥ DATA ────────────────────────────────────────────────────────────────────
    load_dataset → select(n) → map(format_fn, batched=True, remove_columns) │
    in : a Hub dataset or your JSONL out : one `text` (or `messages`) column│
    fail: EOS not appended → model never stops; bad `remove_columns` →      │
          the trainer cannot find the text field                            │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑦ MASK (optional but recommended) ──────────────────────────────────────────
    train_on_responses_only(trainer, instruction_part, response_part)       │
    in : an SFTTrainer              out : trainer with -100 on the prompt   │
    fail: silent no-op → trains the prompt (§4.9.3 assertion block)         │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑧ TRAIN ───────────────────────────────────────────────────────────────────
    SFTTrainer(model, tokenizer, train_dataset, dataset_text_field,         │
               packing, args=SFTConfig(...)).train()                        │
    in : data + model               out : adapter weights + a loss curve    │
    fail: see §14's 20-row table                                            │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑨ EVALUATE (the step the video skips) ──────────────────────────────────────
    base vs adapter on a held-out prompt set + a format-compliance check    │
    in : two checkpoints            out : a decision: ship / retrain / stop │
    fail: no eval → you ship a model whose only evidence is a loss number   │
 ───────────────────────────────────────────────────────────────────────────┘
 ⑩ SAVE / MERGE / SERVE ────────────────────────────────────────────────────
    save_pretrained (adapter)  |  save_pretrained_merged (merged_16bit/4bit/  │
    GGUF)  |  push_to_hub_merged                                            │
    in : the trained model          out : a deployable artefact             │
    fail: peft merge on a 4-bit base → wrong/mis-scaled weights (§10.3)     │
 ───────────────────────────────────────────────────────────────────────────┘
```

### 5.2 The same spine as the instructor demonstrated it

| Stage | What he did | Timestamp | Cell |
|---|---|---|---|
| ① Environment | Colab → Runtime → change runtime type → free GPU (T4); `pip install torch torchvision torchaudio xformers --index-url .../cu128`, `pip install unsloth`, `transformers==4.56.2`, `--no-deps trl==0.22.2` | [33:11]–[34:20] | 3 |
| ② Config | `SEED = 3407`; four seeding calls; TF32 flags; `max_seq_length=4096`; `dtype=None`; `load_in_4bit=True` | [34:21]–[35:55] | 4 |
| ③ Load base | `assert torch.cuda.is_available()`; `BASE_MODEL_NAME = "unsloth/tinyllama-bnb-4bit"`; `from_pretrained(...)`; *"it will ask you the Hugging Face token also… you have to give the read permission"* | [35:57]–[38:42] | 5–7 |
| ④ Inject LoRA | `get_peft_model(model, r=32, target_modules=[...7...], lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=3407)` — *"then I'm passing my target module Q, K, V, O, then gate projection, up projection and the down projection"* | [39:43]–[40:11] | 11 |
| ⑤ Inspect | `print_trainable_parameters()`; per-layer `named_parameters()` walk *"you will get each and every layer of the transformer"*; device; dtype; `isinstance(model, PeftModel)`; `model.peft_config`; `torch.cuda.memory_summary()` | [38:53]–[42:53] | 12–19 |
| ⑥ Data | `alpaca_prompt` + `format_data` appending `EOS_TOKEN`; `load_dataset("yahma/alpaca-cleaned", split="train")`; `.shuffle(seed=3407).select(range(1500))`; `.map(format_data, batched=True, remove_columns=dataset.column_names)` | [43:00]–[46:00] | 24 |
| ⑦ Mask | **not done** — no `train_on_responses_only` anywhere in the notebook or video | — | — |
| ⑧ Train | `SFTTrainer(..., dataset_text_field="text", packing=True, args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=2e-5, warmup_ratio=0.1, optim="adamw_8bit", logging_steps=10, seed=3407, output_dir="outputs", report_to="none"))` | [47:42]–[48:40] | 33 |
| ⑨ Evaluate | **not done** — the model is prompted once and declared *"not quite good"* | [53:50]–[54:07] | 36 |
| ⑩ Save | `model.save_pretrained("lora_model")`, `tokenizer.save_pretrained("lora_model")` — adapter only | [50:48]–[51:12], [54:16]–[54:22] | 38 |

**Two stages the video skips (⑦ and ⑨) are the two that separate a hobby run from a production run.** The masking question is worth 30–60% of your gradient signal (§4.9 and CS-13 §4.4); the evaluation question is worth everything, because without it you have no way to know whether the adapter is better than the base.

> **Correction:** the notebook sets `use_gradient_checkpointing=False` with the inline comment `# Set True if model >= 7B` [cell 11], and the companion markdown table repeats it: *"Enable only for ≥7B models"* [cell 10]. **That heuristic is wrong on both sides.** Gradient checkpointing is not about parameter count — it is about **activation memory**, which scales with `batch × sequence_length × hidden_size × layers`, not with parameter count. A 1B model at `max_seq_length=4096` and batch 2 has activation memory comparable to a 7B model at `max_seq_length=1024` and batch 1. The rule that actually holds: **enable it whenever `peak_activation_memory > 0.5 × available_VRAM`**, which for any long-context run means "always". More importantly, the notebook never mentions the option that matters: **`use_gradient_checkpointing="unsloth"`** — a string, not a boolean — which is Unsloth's memory-optimised checkpointing path and is a substantial part of how the 340k-token context claim is reached [30:23]–[30:27]. Setting `False` on a run whose whole point is long context is exactly backwards. Set `"unsloth"` and measure.

---

## 6. Hands-On Code (annotated)

Every block below is from `unsloth_practical.ipynb` (45 cells) unless marked otherwise, cleaned up and commented for *why* each line exists. "What to change for your own data" follows each.

### 6.1 Environment and installation

**Library versions the video uses** (from cell 3, verbatim):

| Package | Version / source | Why pinned |
|---|---|---|
| `torch`, `torchvision`, `torchaudio` | from `https://download.pytorch.org/whl/cu128` | CUDA 12.8 build; must match the driver |
| `xformers` | latest compatible | memory-efficient attention; Unsloth can use it as a fallback when FA2 is unavailable |
| `unsloth` | **unpinned** | the video installs whatever is current |
| `transformers` | **`==4.56.2`** | Unsloth's patches target specific internals |
| `trl` | **`==0.22.2`** with `--no-deps` | `--no-deps` prevents pip from upgrading `transformers` |
| `psutil` | latest | installed later for CPU-RAM measurement [cell 27] |

```bash
# unsloth_practical.ipynb, cell 3 — the exact four lines from the video [33:42]-[34:03].
# NOTE: `--no-deps` on trl is deliberate and load-bearing. See the Beyond-the-video below.
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2

# The instructor's own justification [33:42]-[34:08]:
#   "torch ... the main package of the PyTorch. Then torchvision for the vision.
#    Then torchaudio for the audio, specifically xformers for the attention. OK.
#    Flash attention. Then unsloth. Then transformers. Then TRL. So you need to
#    install all these modules. It will take some time, maybe 5 to 10 minutes."
```

**A production-grade, reproducible version of the same thing:**

```bash
# Pin everything. Unpinned Unsloth + pinned transformers is a latency bomb:
# a Unsloth release that assumes a newer transformers lands silently and you get
# an ImportError three weeks later on a rebuild.
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth==2025.3.19"
pip install "transformers==4.56.2" "trl==0.22.2" "peft==0.14.0" "accelerate==1.4.0" \
    "datasets==3.3.2" "bitsandbytes==0.45.3"

# Verify the whole stack in 5 seconds. Do this in CI, not by hand.
python - <<'PY'
import torch, transformers, trl, peft, unsloth
print("torch        ", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("transformers ", transformers.__version__)
print("trl          ", trl.__version__)
print("peft         ", peft.__version__)
if torch.cuda.is_available():
    print("device       ", torch.cuda.get_device_name(0))
    print("capability   ", torch.cuda.get_device_capability(0))  # >= (8,0) for bf16
PY
```

> **Beyond the video:** `--no-deps` is a blunt instrument that works here by accident of the version pair. The *reason* it is needed: TRL 0.22.2's package metadata declares a minimum `transformers` version, and if that minimum is above 4.56.2, pip will happily install a `transformers` that Unsloth 2025.x has not patched. Installing with `--no-deps` skips TRL's dependency resolution entirely, so you must then verify by hand that TRL's *runtime* imports succeed — which they do, because TRL only needs the `Trainer`/`SFTConfig` API surface, not the internals Unsloth patches. The durable alternative: install in a lockfile-managed environment (`uv pip install -r requirements.lock`) rather than resolving at container-build time. **The rule: never let a package manager resolve Unsloth's dependency graph for you, because Unsloth's compatibility matrix is defined by its patch targets, not by its metadata.**

**What to change for your own data:** nothing in this block. If you are on a non-Colab machine, replace line 1 with the wheel index matching `nvidia-smi`'s CUDA version; if you are on an AMD MI300 or an Intel Arc, follow Unsloth's ROCm/XPU install page instead — the rest of the notebook is unchanged.

### 6.2 Configuration and seeding

```python
# unsloth_practical.ipynb, cell 4 — verbatim, with why-comments added.
import random
import numpy as np
import torch

# Why seed at all: without it, weight init, dropout masks, CUDA kernel scheduling
# and dataset shuffling all differ run to run. You cannot attribute a 2% quality
# delta to a hyperparameter change if run-to-run noise is 3%.
SEED = 3407                                    # the notebook's choice; note it is
                                               # the same 3407 as the unsloth-vs-hf
                                               # notebook and as the SFTConfig seed
random.seed(SEED)                              # Python `random` — data shuffling
np.random.seed(SEED)                           # NumPy — preprocessing
torch.manual_seed(SEED)                        # CPU tensors + weight init
torch.cuda.manual_seed_all(SEED)               # all GPUs — dropout, attention kernels

# Faster & stable matmul on NVIDIA GPUs.
# These two lines let cuBLAS use TF32 (19-bit) for FP32 matmuls on Ampere+.
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# The three constants that will be reused by from_pretrained below.
max_seq_length = 4096        # notebook comment: "Demonstrates long-context + RoPE scaling"
dtype = None                 # notebook comment: "Auto-detect (FP16 on T4, BF16 on A100/L4)"
load_in_4bit = True          # notebook comment: "Memory-efficient QLoRA-style loading"
```

> **Beyond the video:** the two TF32 lines are almost certainly **no-ops for the run that follows**, and understanding why is a good test of whether you understand the dtype stack. TF32 applies to **FP32** matrix multiplications. Under `load_in_4bit=True`, the dequantized matmuls run in `bnb_4bit_compute_dtype` (FP16 on a T4, BF16 on an A100), and the LoRA matmuls run in the model dtype too. There is essentially no FP32 matmul left in the training loop for TF32 to accelerate. The lines cost nothing and are harmless — but if you see them in a benchmark's "Unsloth applies extra optimizations" column, they are not doing work. (Where TF32 *does* matter: full-precision FP32 training, which nobody does; and `torch.compile`'s FP32 fallbacks.) Also note: `torch.manual_seed` does **not** make CUDA kernels deterministic. For that you need `torch.use_deterministic_algorithms(True)`, which will slow you down and may fail outright on fused attention kernels. The notebook's seeds buy you *reproducible data order and weight init*, not bit-exact training — which is the right trade for a tutorial and the wrong expectation to carry into an audit.

**What to change for your own data:** `max_seq_length` should be **the p99.5 of your tokenized lengths, rounded up to a multiple of 64**, not a round number. Measure it:

```python
# 30 seconds, and it will halve your training cost more often than any other change.
import numpy as np
lens = [len(tokenizer(x, add_special_tokens=False)["input_ids"]) for x in your_texts]
print("p50", np.percentile(lens, 50), "p95", np.percentile(lens, 95), "p99.5", np.percentile(lens, 99.5), "max", max(lens))
# Set max_seq_length = ceil(p99.5 / 64) * 64. Truncating the top 0.5% is a rounding
# error on quality; training at 4x your p99.5 is a 2-4x cost increase for nothing.
```

### 6.3 Loading the base model

```python
# unsloth_practical.ipynb, cells 5-7.
# GPU sanity check — fail loudly here, not 6 minutes into training.
assert torch.cuda.is_available(), "Please enable GPU runtime (Colab → Runtime → GPU)"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

BASE_MODEL_NAME = "unsloth/tinyllama-bnb-4bit"

# Returns a TUPLE. This single line does six things:
#   1. downloads the pre-quantized NF4 checkpoint (~700 MB, not 2.2 GB FP16)
#   2. loads it without re-quantizing  (the video: "that is a pre-quantized model" [13:12])
#   3. patches the model class with fast RoPE / RMSNorm / SwiGLU / attention forward
#   4. configures RoPE scaling so max_seq_length=4096 is valid on a 2048-pretrained model
#   5. sets bnb_4bit_compute_dtype from `dtype` (FP16 on T4, BF16 on A100/L4)  <-- critical
#   6. enables the fast LoRA autograd.Function for later get_peft_model calls
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL_NAME,
    max_seq_length = max_seq_length,   # 4096; drives RoPE scaling, not truncation
    dtype          = dtype,            # None = auto-detect
    load_in_4bit   = load_in_4bit,     # True
)
```

The video's own commentary on this call [37:18]–[38:28]:

> *"So I'm going to load the base model — Unsloth TinyLlama BnB 4-bit, this is the model name. So this fast language model is having one method, `from_pretrained`… maximum sequence length I'm going to keep 4096, means whatever output will be generated the output will be under that length. Then data type. So I mentioned the data type also. Then load in 4 bit. Yes, the quantization will be applied."*

and on the auth [38:16]–[38:28]:

> *"This model loading might take some time and it will ask you the Hugging Face token also. So please keep the Hugging Face token in your secrets. See I kept it at least with the read permission. So you have to give the read permission to your Hugging Face token."*

> **Beyond the video:** the instructor's gloss — *"whatever output will be generated the output will be under that length"* [37:35]–[37:38] — is the wrong mental model. `max_seq_length` at load time is a **model-shaping argument**: Unsloth uses it to configure RoPE scaling (linear/dynamic factors, `max_position_embeddings` bookkeeping, and which attention kernel variant is compiled). It is not merely a truncation window for generation. Consequences: (1) raising it *after* loading has no effect; (2) it changes the model's position encoding, so a model trained at 4096 and served at 2048 can show degraded short-context quality if aggressive scaling was applied; (3) it determines peak activation memory, so it is the first knob to turn down when you OOM. The generation length is controlled separately, by `max_new_tokens` in `generate()` [cell 36 uses 64]. **Two different things, two different arguments — conflating them is a common interview slip.**

**What to change for your own data:** replace `BASE_MODEL_NAME` with your family's Unsloth pre-quantized repo if one exists (search `unsloth/<model>-bnb-4bit`), or with any HF causal-LM id if not — `from_pretrained` accepts both and will quantize on the fly for the latter (slower first load, identical training). If you are fine-tuning a **base** (not instruct) model for a domain, prefer the base checkpoint (CS-12 → CS-13 ordering) and use a chat template only after continued pretraining.

### 6.4 Attaching LoRA

```python
# unsloth_practical.ipynb, cell 11 — verbatim, with why-comments.
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                          # rank. 8/16/32/64 are the common values.
                                     # 32 doubles adapter params vs 16 and buys
                                     # meaningfully better domain adaptation.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention: all four
        "gate_proj", "up_proj", "down_proj",      # MLP: all three
    ],
    # Why all seven: covering attention only trains ~40% of the adapter params
    # you should, and the MLP is where a large fraction of factual/domain
    # adaptation lives (CS-23). The video's own framing [40:00]-[40:06]:
    # "then I'm passing my target module Q, K, V, O, then gate projection,
    #  up projection and the down projection".
    lora_alpha = 32,                 # scaling. alpha == r is the modern default
                                     # (equivalent to a fixed effective LR scaling of 1.0).
    lora_dropout = 0.0,              # Unsloth's fast path requires 0. Non-zero dropout
                                     # forces the slow autograd path -> you lose the
                                     # whole speed claim. This is not a preference.
    bias = "none",                   # train no bias terms. Cheap, and biases add
                                     # params whose gradient is noisy at small data.
    use_gradient_checkpointing = False,
    # ^^ The notebook comment says "Set True if model >= 7B". That heuristic is
    #    wrong (see the Correction in S5.2). Unsloth also accepts the STRING
    #    "unsloth" here, which is its memory-optimised checkpointing and is a
    #    real part of the long-context claim. Use "unsloth" for anything at
    #    T >= 2048. It costs ~5-15% step time and saves 40-60% activation memory.
    random_state = 3407,             # reproducibilty of the LoRA A/B initialisation
    # Other kwargs that exist and are worth knowing (not in the video):
    #   use_rslora=True        -> alpha/sqrt(r) scaling; stabilises r>=64
    #   loftq_config={}        -> LoftQ init; leave empty to disable
    #   max_seq_length=4096    -> override the load-time value
    #   use_gradient_checkpointing="unsloth"
)
```

**The `lora_dropout=0.0` constraint deserves emphasis.** Unsloth's fast LoRA path is a hand-written `torch.autograd.Function`; dropout inside it would require either a second kernel or a saved mask. Rather than pay that, Unsloth's fast path asserts `dropout == 0` and **falls back to the standard peft path** if you set it non-zero. You get a correct model and a silently slower run — no warning in older versions.

```python
# The check that tells you which path you are on. Run it once.
import torch
from peft import PeftModel
print("is peft model:", isinstance(model, PeftModel))
print("peft config:\n", model.peft_config)
# Look for:  lora_dropout: 0.0   and, in recent versions, a `use_rslora` field.
# If lora_dropout != 0, you are on the slow path.
```

> **Beyond the video:** `r=32, alpha=32` is a *reasonable* default and a *poor* one to inherit without thought. Three interactions: (1) at `r=32` with standard scaling (`alpha/r`), effective LR on the adapter output is `alpha/r = 1`; at `r=64, alpha=32` it would be `0.5`, i.e. you would need to raise the LR to compensate. (2) Rank interacts with **learning rate**: high rank + high LR is the classic LoRA divergence. If you raise `r` from 16 to 64, drop the LR by ~2× or enable `use_rslora`. (3) Rank interacts with **data size**: at 1,500 rows, `r=32` on seven projections is 25 M trainable parameters for 1,500 examples — roughly 17,000 parameters per example. That is a *lot* of capacity for the data, and it is a large part of why the video's model produced *"1 2 3 5 5 to 6 11"* [53:59]. **For <10k rows, start at `r=16` and only go up if eval says you underfit.**

**What to change for your own data:** `target_modules` is model-family-specific. The safe universal recipe is to print every Linear's name and cover **all** of them except the LM head:

```python
# Enumerate the right target_modules for ANY architecture, 5 seconds.
import torch.nn as nn
linear_names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})
print("Linear leaf names:", linear_names)
# Exclude anything matching "lm_head", "embed", "score", "classifier", "pooler".
# Llama/Mistral/Qwen/Phi: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
# Gemma2/3:               same, plus  (none extra)
# Falcon-H1 / Mamba mixes: watch for "in_proj", "out_proj", "conv1d" -> architecture-specific
```

### 6.5 The pre-flight inspection — 30 seconds that saves hours

The video spends cells 12–19 and roughly four minutes [38:53]–[42:53] on exactly this, and it is the most underrated part of the whole tutorial.

```python
# unsloth_practical.ipynb, cells 12-19 — the full pre-flight, consolidated.
# Run this block BEFORE creating the trainer, every single time.

# --- 1. Trainable vs total parameter count -------------------------------
model.print_trainable_parameters()
# Video output: trainable ~25,231,360  (he reads "2 cr 52 lakh 31,360") [40:21]
# Expected for r=32 on 7 modules over TinyLlama: 25,231,360 exactly.

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,}")
print(f"Total:     {total:,}")
print(f"Percent:   {100 * trainable / total:.3f}%")   # -> 2.294% for this config
# SANITY BANDS: LoRA on 7 modules at r=16/32/64 -> ~1.0% / ~2.3% / ~4.5% of a 1.1B.
# If you see 0.1%, you covered only q_proj+v_proj. If you see 30%+, you are
# accidentally training the base or you untied the embeddings.

# --- 2. Which layers actually have gradients enabled ---------------------
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
# Video commentary [39:12]-[41:12]: "this is each and every layer of the transformer,
# because this Llama model is built on top of the transformer itself. You will get
# the attention module over there."
# YOU ARE LOOKING FOR: names ending in lora_A / lora_B and NOTHING ELSE.
# If you see `embed_tokens.weight` or `lm_head.weight` in this list, stop.

# --- 3. Device and dtype -------------------------------------------------
print("device:", next(model.parameters()).device)   # cell 15
print("dtype :", next(model.parameters()).dtype)    # cell 16
# Expect: cuda:0 and (for 4-bit) the *compute* dtype, torch.float16 on a T4.
# A dtype of torch.float32 here means bnb_4bit_compute_dtype was not set and
# you are about to run 8-32x slower than you should. THE #1 BASELINE BUG.

# --- 4. Confirm it is really a PEFT model --------------------------------
from peft import PeftModel
print(isinstance(model, PeftModel))                 # cell 17 -> True
print(model.peft_config)                            # cell 18
# Video [42:03]-[42:21]: "you can check the model is a PEFT model or not...
# you will be getting the configuration of the PEFT."

# --- 5. Memory before training -------------------------------------------
import torch
torch.cuda.memory_summary()                         # cell 19
# Video [42:25]-[42:50]: "you can see so many things over here... active allocated
# memory, active memory, requested memory, GPU reserved memory, non-release memory
# allocation. See, so many things you can see over here."
```

> **Beyond the video:** three additions that turn this from a tour into a pre-flight check. (1) **`print(tokenizer.chat_template)`** — you are about to train a model whose template you have not looked at. Two lines, and it catches the most common silent bug in the module (§4.10). (2) **A one-batch loss sanity check** — build one batch, forward it, and confirm the loss is in the `ln(vocab) ± 20%` band (`ln(32000) ≈ 10.4` for a random 32k-vocab model at init). A loss of 0.3 at step 0 means your mask is inverted or your labels equal your inputs in a degenerate way. A loss of 25 means the model is worse than random and your template is scrambled. (3) **`torch.cuda.reset_peak_memory_stats()` immediately before training**, which the notebook does [cell 27] — without it, `max_memory_reserved()` reports the peak across *model loading too*, and you would attribute the load-time quantization spike to training.

```python
# Additions worth making. All three take <60 s total and catch different bugs.
print("chat template:", tokenizer.chat_template)      # (1) SEE the template
print("eos_token   :", tokenizer.eos_token, tokenizer.eos_token_id)  # (2) is EOS real?
print("pad_token   :", tokenizer.pad_token, tokenizer.pad_token_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token         # and remember you did this
    print("WARNING: pad_token was None; set it to eos. Padding will now be EOS, "
          "so any unmasked -100 handling must be explicit (CS-13 S4.4).")

# (3) One-batch forward: is the initial loss in the expected band?
#     Do this AFTER building the trainer so the collator is the real one.
batch = next(iter(trainer.get_train_dataloader()))
with torch.no_grad():
    out = model(**{k: v for k, v in batch.items() if k != "labels"})
    # compute the masked CE by hand so you see it independent of the trainer
    import torch.nn.functional as F
    logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float()
    labels = batch["labels"][:, 1:].reshape(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
import math
print(f"init loss = {loss.item():.3f}   ln(V) = {math.log(model.config.vocab_size):.3f}")
# EXPECT: init loss within ~[0.7, 1.5] x ln(V) for a fresh LoRA on a pretrained base.
#   loss ~  0.0 : labels == inputs and the model is copying -> masking inverted
#   loss ~ 25+  : template scrambled or wrong tokenizer
#   loss ~ ln(V): the base is producing near-uniform logits -> check dtype/quantization
```

**What to change for your own data:** keep every one of these; add a **length histogram** and a **supervised-fraction print** (from §4.9.3). Those two plus this block are a complete pre-flight.

### 6.6 Dataset preparation

```python
# unsloth_practical.ipynb, cell 24 — verbatim, restructured with why-comments.
EOS_TOKEN = tokenizer.eos_token
# CRITICAL: without EOS the model learns the answer but never learns to STOP.
# The notebook's own comment: "EOS token is mandatory, otherwise generation may
# never stop." In practice this is what 'the model rambles forever' looks like.

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_data(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"],
    ):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load: 51,000 rows. Video [44:48]-[44:57]: "here we have output feature, input
# feature and the instruction feature, and the number of rows is around 51,000."
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Shuffle THEN select. Shuffling first means the 1,500 rows you keep are a random
# sample of the full 51k, not the first 1,500 (which are ordered by topic).
# The video [44:57]-[45:01]: "we are not going to take the entire data, we'll take
# 1,500 row only."
dataset = dataset.shuffle(seed=3407).select(range(1500))

# batched=True -> format_data receives LISTS, hence the zip() loop.
# remove_columns=dataset.column_names -> leaves ONLY the `text` column, which is
# what `dataset_text_field="text"` needs. Forget this and the trainer still works
# but the unused columns blow up your Arrow file size and, with some collators,
# cause a type error.
dataset = dataset.map(
    format_data,
    batched=True,
    remove_columns=dataset.column_names,
)
```

The instructor's framing of the format [43:14]–[43:44]:

> *"First I'm going to add this end-of-sentence token to the tokenizer… Now this is my prompt. So here my data is available into this instruction, input and response format. You can create your own data as well… Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."*

**What to change for your own data:** three things, in order of impact.

```python
# 1. YOUR dataset, YOUR schema -> the mapping is the only part that changes.
dataset = load_dataset("json", data_files="my_train.jsonl", split="train")
# If your rows are `{"prompt": ..., "completion": ...}` rather than Alpaca's
# three fields, do NOT force them into the Alpaca mould. Use a messages column
# and let the model's own chat template do the work (see 6.8).

# 2. Speed up map() on a big dataset. Without num_proc, map is single-threaded and
#    dominates wall-clock for >100k rows.
dataset = dataset.map(format_data, batched=True, num_proc=8,
                      remove_columns=dataset.column_names)
# Rule of thumb: tokenization of 1M rows single-threaded is ~20-40 min;
# with num_proc=8 it is ~3-6 min. This is usually the single biggest speedup
# available in a real pipeline, and Unsloth has nothing to do with it.

# 3. Check the result before training. Two prints, ten seconds.
print(repr(dataset["text"][0][:400]))
print("row 0 ends with EOS:", dataset["text"][0].rstrip().endswith(EOS_TOKEN.rstrip()))
print("n rows:", len(dataset))
```

> **Beyond the video:** the notebook's `alpaca_prompt` has one silent hazard inherited from CS-13. `input_text` is inserted verbatim; when `input` is an empty string you get `### Input:\n\n` (fine), but when the CSV/JSON parser yields `None`, Python's `str.format` renders the literal string **`None`** and the model learns to emit the token `None` as part of its output format. Guard it: `input_text = input_text or ""`. The cleaned Alpaca dataset has `""` not `None`, so the video's run is unaffected — but the identical code on dozens of other datasets is not, and this is a shipped-bug class (CS-13 §4.2.1).

> **Beyond the video:** `remove_columns=dataset.column_names` is evaluated **before** `map` runs, which is correct, but it is also the line that silently deletes a column you might need later. If you plan to compute per-source metrics or filter by source, keep that column: `remove_columns=[c for c in dataset.column_names if c not in {"source"}]`. Better still, keep the raw dataset and a **content hash** of it alongside your adapter (§16.2) — `load_dataset("json", data_files=...)` gives you no fingerprint at all, and "which data trained this adapter" is a question you will be asked.

### 6.7 The trainer

```python
# unsloth_practical.ipynb, cell 33 — verbatim, with why-comments.
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    packing = True,
    # packing: concatenate short rows into one max_seq_length window.
    # Alpaca rows average ~150-250 tokens; at T=4096 an unpacked run wastes
    # ~94% of every forward pass on padding. Video [47:54]-[48:12]:
    # "Then packing true. Right? I told you the packing right? ... here I
    #  written one automatic sequence packing, one of the important features
    #  of the Unsloth."
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,      # effective batch = 2*4 = 8 rows
        num_train_epochs = 1,
        learning_rate = 2e-5,
        warmup_ratio = 0.1,                   # notebook: "Stabilizes early training"
        optim = "adamw_8bit",                 # notebook: "Memory efficient optimizer"
        logging_steps = 10,                   # notebook: "Progress visibility"
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",                   # notebook: "Disable WandB by default"
    ),
)
trainer.train()
```

**The three lines that are wrong or missing, and why.**

| Line | Problem | Fix |
|---|---|---|
| `learning_rate = 2e-5` | **10× below the LoRA norm.** LoRA SFT is typically `1e-4`–`3e-4`; the repo's *own* comparison notebook uses `2e-4` for the identical model and data (`unsloth_solution.ipynb`, cell 5). At `2e-5` with `r=32` and one epoch, the adapter barely moves. | `learning_rate = 2e-4` with `warmup_ratio=0.03`, or `1e-4` if you are second-stage SFTing an instruct model |
| `num_train_epochs = 1` over 1,500 rows | 188 optimizer steps at effective batch 8. For instruction tuning the useful range is 1–3 epochs over a *larger* set; 188 steps is a smoke test, not a fine-tune. | 1,500 rows × 3 epochs = 564 steps; or better, 10k–50k rows × 1–2 epochs |
| `tokenizer = tokenizer` | Deprecated in TRL ≥0.16 and warns on 0.22.2. | `processing_class = tokenizer` |

> **Correction:** the video's own evidence convicts the hyperparameters. The Fibonacci prompt produced *"1 2 3 5 5 to 6 11"* and the instructor said *"I think the response is not quite good. Maybe we'll have to train more"* [53:59]–[54:05]. **"Train more" is the wrong fix.** The run had three compounding problems and only one of them is step count: (1) `lr=2e-5` is ~10× too low for LoRA — the repo's own head-to-head notebook uses `2e-4` on identical data, which is the correct value; (2) 1,500 rows × 1 epoch = 188 steps, which is 5–20× short of a real instruction tune; (3) with `packing=True`, the effective epoch is a *token* budget, and 1,500 short Alpaca rows pack into roughly 60–70 sequences at T=4096, so "1 epoch" is doing far less work than the row count suggests. The correct order of operations is: **raise LR to `2e-4` first, then raise data volume, then raise epochs** — and re-measure after each. Fixing LR alone, at 3 epochs on the same 1,500 rows, is a 5-minute experiment that would have changed the video's conclusion.

**What to change for your own data:**

```python
# A production SFTConfig, annotated. Values are for a 7-8B QLoRA SFT on
# 10k-50k rows, which is the realistic case.
args = SFTConfig(
    output_dir                  = "out/acme-sft-v1",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,     # effective batch 16 examples (NOT tokens)
    num_train_epochs            = 2,     # 1-3. Watch eval loss, not train loss.
    learning_rate               = 2e-4,  # LoRA sweet spot. 1e-4 for instruct bases.
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.03,  # 3% is right; 10% wastes 10% of a short run
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,  # small; LoRA adapters regularise poorly
    max_grad_norm               = 1.0,
    max_seq_length              = 2048,  # <-- set from YOUR p99.5, not from vibes
    packing                     = True,
    dataset_text_field          = "text",
    logging_steps               = 10,
    save_strategy               = "steps",
    save_steps                  = 200,
    save_total_limit            = 3,     # keep disk bounded
    eval_strategy               = "steps",
    eval_steps                  = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    seed                        = 3407,
    report_to                   = "wandb",   # or "none" for a hermetic run
    bf16                        = torch.cuda.is_bf16_supported(),
    fp16                        = not torch.cuda.is_bf16_supported(),
    # dataloader_num_workers=4,   # <-- turn this on if you are dataloader-bound
    # gradient_checkpointing=True, # when SFTTrainer does not inherit it from the model
)
```

> **Beyond the video:** `bf16`/`fp16` in `SFTConfig` and `dtype` in `from_pretrained` must agree. On a T4 (`capability == (7,5)`) bf16 is unsupported and `bf16=True` raises or silently falls back. The one-liner `torch.cuda.is_bf16_supported()` is the portable check, and Unsloth's `dtype=None` auto-detection does the same thing at load time — **but the trainer's autocast flags are set independently of the loader's**, so a mismatch is possible. Symptom: `RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'` or a NaN loss in the first 20 steps. Set both explicitly in any script you intend to run on more than one GPU class.

### 6.8 Response-only masking, done properly (the upgrade to cell 24)

This is the version of the notebook's data cell that you should actually ship. It replaces the hand-rolled flat string with the model's own chat template and adds the response-only mask.

```python
# =============================================================================
# PRODUCTION DATA CELL — replaces unsloth_practical.ipynb cells 24 + 33's args
# =============================================================================
from unsloth import FastLanguageModel, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---- 1. Build a `messages` column, not a flat `text` string -----------------
# The whole point: let the MODEL'S OWN TEMPLATE decide the format. You cannot
# get the special tokens right by hand for every family.
def to_messages(batch):
    out = []
    for instr, inp, outp in zip(batch["instruction"], batch["input"], batch["output"]):
        # `inp or ""` guards against the literal-string "None" bug from CS-13 S4.2.1
        user = instr if not inp else f"{instr}\n\n{inp}"
        out.append([
            {"role": "user",      "content": user},
            {"role": "assistant", "content": outp},
        ])
    return {"messages": out}

ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=3407).select(range(1500))
ds = ds.map(to_messages, batched=True, num_proc=8, remove_columns=ds.column_names)

# ---- 2. Pick the markers FROM the template, not from memory -----------------
# Find whatever the template puts before an assistant turn. This is the single
# most robust line in this whole module.
def response_marker(tokenizer):
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
        tokenize=False, add_generation_prompt=False,
    )
    # the assistant marker is everything between the user's content and "Y"
    head, _, tail = probe.partition("Y")
    # take the longest suffix of `head` that is not part of the user content
    return head.split("X")[-1] if "X" in head else head

RESPONSE_PART = response_marker(tokenizer)
print("detected response marker:", repr(RESPONSE_PART))
# Llama-3:  '<|start_header_id|>assistant<|end_header_id|>\n\n'
# Qwen2.5:  '<|im_start|>assistant\n'
# Gemma-2:  '<start_of_turn>model\n'
# Phi-3:    '<|assistant|>\n'
# TinyLlama-Chat: '<|assistant|>\n'

INSTRUCTION_PART = tokenizer.apply_chat_template(
    [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
)
print("detected instruction marker:", repr(INSTRUCTION_PART))

# ---- 3. Trainer (note: dataset_text_field is NOT set when using messages) ----
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,          # was `tokenizer=` in older TRL
    train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,             # 10x the notebook's 2e-5 — see the Correction in S6.7
        warmup_ratio=0.03,
        optim="adamw_8bit",
        max_seq_length=4096,
        packing=False,                   # keep False while you validate the mask
        logging_steps=10,
        output_dir="outputs",
        report_to="none",
    ),
)

# ---- 4. Mask the prompt -----------------------------------------------------
trainer = train_on_responses_only(
    trainer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
)

# ---- 5. VERIFY (this block is not optional) ---------------------------------
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
sup = (labels != -100).sum().item(); tot = labels.numel()
print(f"supervised {sup}/{tot} = {sup/tot:.1%}")
assert 0 < sup < tot, "mask is either fully on or fully off — see S4.9.3"
print("-" * 60)
print("SUPERVISED TEXT:")
print(tokenizer.decode(batch["input_ids"][0][labels != -100], skip_special_tokens=False)[:400])
print("-" * 60)
# Expect: ONLY the assistant's answer + the turn terminator. If you see the
# user's question, the marker is wrong. If you see half a word at the start,
# the token boundary is off by one.

trainer.train()
```

> **Beyond the video:** the `response_marker()` probe above is the trick worth stealing. Hard-coding `"<|start_header_id|>assistant<|end_header_id|>\n\n"` is what everyone does and it is what breaks when you swap `Llama-3.1-8B` for `Qwen2.5-7B` and forget to change the string. Deriving it from `apply_chat_template` on a two-message probe means **the marker is always in sync with the tokenizer you actually loaded**, including when a Hub template is updated under you. Wrap it in an assertion in CI: if the detected marker changes between two commits, your masking changed, and you want to know before you train.

> **Beyond the video:** note what changed in step 3: `dataset_text_field` is **omitted** because the dataset now has a `messages` column and TRL's `SFTTrainer` detects it. If you leave `dataset_text_field="text"` in place with a `messages` dataset you get `KeyError: 'text'`; if you leave both columns present you can get a confusing "found both messages and text" precedence warning. Pick one schema and be consistent — the `messages` path is the one that interacts correctly with `assistant_only_loss` and with every serving stack.

### 6.9 Saving, merging, and `save_pretrained_merged`

The video saves exactly this [50:48]–[51:12]:

```python
# unsloth_practical.ipynb, cell 38 — verbatim
LORA_SAVE_PATH = "lora_model"
model.save_pretrained(LORA_SAVE_PATH)     # writes adapter_model.safetensors (~50 MB)
tokenizer.save_pretrained(LORA_SAVE_PATH) # writes tokenizer_config.json WITH chat_template
print(f"LoRA adapters saved at: {LORA_SAVE_PATH}")
# Video [50:48]-[51:03]: "if you want to save the model on your current directory
# you just have to write the name... model.save_pretrained. Once you will run it,
# and tokenizer.save, once you will write it, guys, you will be able to save your model."
```

and later, after the inference test [54:12]–[54:22]:

> *"I'm going to save the model. So here's my path, LoRA model, into the current directory. Here it will be saved… And yeah, let me save the tokenizer and the model both. And then I can use it anywhere."*

**`model.save_pretrained(dir)` on a PEFT model writes the adapter, not a model.** The saved directory contains `adapter_model.safetensors` (~50–200 MB for these ranks), `adapter_config.json`, and whatever the tokenizer wrote. To use it you must load the base and attach the adapter — which is fine, and is the correct artefact for anything you version-control (CS-13 §10.2).

**The four save paths, and when each is right:**

```python
# ---- PATH A: adapter only (what the video does) ----------------------------
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
# Artefact: ~50-200 MB. Needs the base model + peft at load time.
# USE WHEN: iterating, version-controlling, A/B-ing multiple adapters against
#           one base, serving with vLLM --enable-lora, or pushing to the Hub
#           as an adapter repo.

# ---- PATH B: merged, FP16 (the deployable single artefact) -----------------
model.save_pretrained_merged("merged_16bit", tokenizer, save_method="merged_16bit")
# Artefact: full model at FP16 (~2.2 GB for 1.1B, ~16 GB for 8B).
# USE WHEN: you want ONE artefact, no peft dependency, maximum compatibility
#           with vLLM/TGI/sglang/llama.cpp conversion pipelines.

# ---- PATH C: merged, 4-bit -----------------------------------------------
model.save_pretrained_merged("merged_4bit", tokenizer, save_method="merged_4bit_forced")
# Artefact: full model requantized to 4-bit.
# NOTE: the plain `save_method="merged_4bit"` RAISES a RuntimeError. That gate is
#       deliberate — Unsloth refuses the bare spelling and tells you to opt in
#       explicitly ("If you are certain, change `save_method` to
#       `merged_4bit_forced`"), then internally remaps `merged_4bit_forced` back to
#       `merged_4bit` (unsloth/save.py, unsloth_save_model). So `_forced` is not a
#       different artefact — it is the only spelling that runs. Code that passes
#       the bare `merged_4bit` does not produce a lossy model; it produces a
#       traceback, which is at least honest about the trade.
# USE WHEN: you need a small artefact and accept a SECOND quantization error on
#           top of the base's. This is lossy twice and usually the wrong choice.

# ---- PATH D: GGUF for llama.cpp / Ollama / LM Studio (CS-10) --------------
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
# USE WHEN: you are serving on CPU / Apple Silicon / a laptop.
# NOTE: needs llama.cpp to be buildable in the environment; can take minutes.
```

> **Beyond the video:** the instructor's *"then I can use it anywhere"* [54:22] is true of the adapter and **not** true of the merged model, and vice versa — they have opposite portability profiles. The adapter is portable *across bases* (you can attach it to a differently-quantized base) but needs `peft` at load. The merged model is portable *across runtimes* (no peft) but is welded to one base quantization. Choose by asking: **will I ever want to swap the base?** If yes, keep the adapter as the source of truth and generate merged artefacts from it in CI.

### 6.10 Why `peft`'s `merge_and_unload()` is wrong on a 4-bit-loaded model

This is the highest-severity correctness trap in the module, and neither the video nor the notebook mentions it.

**What `merge_and_unload()` does:** for each LoRA-wrapped Linear, it computes $W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$ and replaces the module with a plain `nn.Linear` holding that matrix.

**Why 4-bit breaks it:**

1. **The base matrix is not FP16 — it is NF4 packed with per-block absolute scales.** `merge_and_unload` must dequantize `W_0` to do the addition. It dequantizes to the *compute* dtype or, depending on version, to FP32 or to the model's `dtype`. The resulting merged matrix is a *new* tensor in a *different* dtype from the one the training run used.
2. **The merged result is then usually re-saved in that dtype**, so `merged_4bit` via peft is not "the 4-bit model plus the adapter" — it is "a freshly quantized model built from a dequantized-and-re-added matrix", i.e. **base quantization error + merge rounding error + a second quantization error**. Three lossy steps where you expected none.
3. **The LoRA scaling may be applied to the wrong base.** With `bnb`, `W_0` inside the module is the *quantized* tensor and PEFT's merge hook reads the dequantized version through `bnb`'s accessor. If double-quantization (`bnb_4bit_use_double_quant=True`) is on, and the merge code reads only the first-level absmax, the effective scale is wrong by the second-level factor — which is a **silent, uniform mis-scaling of the entire merged weight matrix**, i.e. a model that generates plausible-looking but systematically degraded text.
4. **It drops the model's patched forward.** Merging replaces the Unsloth-patched module with a stock `nn.Linear`, so any RoPE-scaling-aware behaviour attached to the fast path is gone from the merged artefact.

**The failure signature of a bad merge:** the adapter's own eval numbers look fine; the *merged* model's are 5–25% worse; nothing errors. You conclude "merging degrades quality, that's normal." It is not normal — a correct merge is numerically near-lossless.

```python
# THE TEST THAT CATCHES IT. Run after every merge, before you ship.
# Compute the same forward through (a) base+adapter and (b) the merged model.
# They should agree to ~1e-2 in logit space. Anything worse is a broken merge.
import torch, math

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Name three primary colours."}],
    tokenize=False, add_generation_prompt=True,
)
ids = tokenizer(prompt, return_tensors="pt").to("cuda")

# (a) base + adapter, in the SAME dtype the merged model is stored in
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16).to("cuda")
with_adapter = PeftModel.from_pretrained(base, "lora_model").to("cuda")
with_adapter.eval()
with torch.no_grad():
    logits_adapter = with_adapter(**ids).logits[0, -1].float()

# (b) merged
merged = AutoModelForCausalLM.from_pretrained("merged_16bit", torch_dtype=torch.float16).to("cuda")
merged.eval()
with torch.no_grad():
    logits_merged = merged(**ids).logits[0, -1].float()

diff = (logits_adapter - logits_merged).abs()
print(f"max |dlogit| = {diff.max().item():.4f}   mean = {diff.mean().item():.5f}")
print(f"argmax agree: {logits_adapter.argmax().item() == logits_merged.argmax().item()}")
# PASS:  max |dlogit| < ~0.2 for FP16 and the argmax agrees -> merge is faithful
# FAIL:  max |dlogit| > 1.0  -> wrong scaling, wrong quantization, or wrong dtype.
#        Re-merge with save_pretrained_merged(save_method="merged_16bit") and
#        re-run this test. Never ship a merge you have not run this on.
print(tokenizer.decode(logits_adapter.topk(5).indices))
print(tokenizer.decode(logits_merged.topk(5).indices))
```

> **Beyond the video:** the *reason* `save_pretrained_merged` exists is exactly this. Unsloth owns both halves — it knows the NF4 layout, it knows the double-quant scales, and it can dequantize to FP16 with the correct recipe before adding the adapter delta, then save a clean FP16 model (or re-quantize with the same recipe for `merged_4bit`). PEFT does not have that knowledge and should not be asked to guess. **The rule: on a 4-bit-loaded model, never call `merge_and_unload()` yourself. Always go through `save_pretrained_merged`.** If you are not using Unsloth — plain HF + bnb + peft — the correct pattern is to load the base in **16-bit** for the merge only, attach the adapter, then `merge_and_unload()`. Merging is a deploy-time operation; do it in a clean 16-bit process, not in the 4-bit training process.

### 6.11 Inference with the fast path

```python
# unsloth_practical.ipynb, cell 36 — verbatim, with why-comments.
FastLanguageModel.for_inference(model)
# WHY: this swaps the model's forward for the generation-optimised path and
# puts it in eval mode. Skip it and you keep the training forward, which is
# slower for decode AND leaves dropout/checkpointing flags in an odd state.
# Notebook comment: "Always call FastLanguageModel.for_inference(model)".

prompt = alpaca_prompt.format(
    "Continue the Fibonacci sequence",   # instruction
    "1, 1, 2, 3, 5, 8",                  # input
    "",                                   # response left empty -> model fills it
)
# NOTE: the prompt reuses the SAME hand-rolled template the model was trained on.
# That is why it produces anything coherent at all (§4.10).

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 64,
        use_cache      = True,    # KV cache: O(T) instead of O(T^2) decode
        do_sample      = False,   # greedy; notebook: "Deterministic output for demo"
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# VIDEO OUTPUT [53:53]-[54:03]:
#   "... 1 2 3 5 5 to 6 11" with the instructor's verdict:
#   "I think the response is not quite good. Maybe we'll have to train more."
```

**Why the output was bad — three separable causes, in order of likelihood:**

| # | Cause | Evidence | Fix |
|---|---|---|---|
| 1 | `lr=2e-5` — 10× too low for LoRA | The repo's own comparison notebook uses `2e-4` on the same model and data | `lr=2e-4` |
| 2 | 1 epoch / 188 steps on 1,500 rows | Video's own log shows 188 steps [52:11] | 10k+ rows, 2–3 epochs |
| 3 | `max_new_tokens=64` with greedy decode on a model that was taught an *instruction-following* distribution, prompted with a *math completion* task | The training distribution is "answer the question", the eval prompt is "continue this integer sequence" | Not a bug — but it is an out-of-distribution prompt, so weak output is expected regardless |

> **Beyond the video:** the notebook's own commented-out cell 37 shows the **correct** inference benchmark, and it is worth un-commenting because it is the only place in the entire companion material where the measurement protocol is written down correctly:

```python
# unsloth_practical.ipynb, cell 37 (commented out in the notebook) — the right way.
FastLanguageModel.for_inference(model)
prompt = "Explain LoRA in very simple words."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

torch.cuda.synchronize()          # <-- (1) GPU work is async; without this t0 is a lie
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, do_sample=False)
torch.cuda.synchronize()          # <-- (2) without this t1 is measured before the GPU finishes
t1 = time.time()

generated_tokens = outputs.shape[-1]
tokens_per_sec = round(generated_tokens / (t1 - t0), 2)
print(f"Generated tokens: {generated_tokens}")
print(f"Tokens per second: {tokens_per_sec}")
```

Three subtleties worth naming: **(a)** both `torch.cuda.synchronize()` calls are mandatory — CUDA launches are asynchronous, so timing a bare `model.generate` measures how fast Python *enqueued* the work, not how long it took. **(b)** `outputs.shape[-1]` counts the **prompt tokens too**, so `tokens_per_sec` is inflated by `len(inputs)`. For a 128-token generation from a 6-token prompt that is a 5% error; for a 20-token generation from a 400-token prompt it is a **2100%** error. Use `outputs.shape[-1] - inputs["input_ids"].shape[-1]`. **(c)** There is no warm-up iteration. The first `generate` call pays Triton compile + kernel autotune + CUDA context costs. Always run 3 warm-ups and report the median of the next 10.

### 6.12 The measurement harness, done honestly

The notebook's timing block is close to right and worth fixing up into something you can trust.

```python
# unsloth_practical.ipynb, cells 27-34 — the video's harness, corrected.
# =============================================================================
import time, psutil, torch

# ---- FIX 1: clear and reset BEFORE the model is loaded, and again before train.
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# Notebook comment: "This line resets peak counter so you measure only THIS
# training run, not previous runs / notebooks." Correct — but note that
# max_memory_reserved() then reports the peak SINCE THE RESET, which includes
# anything you loaded in between.

process = psutil.Process()
cpu_ram_before = process.memory_info().rss / 1024**3   # GB

# ---- FIX 2: synchronize, and warm up, before you start the clock.
torch.cuda.synchronize()
for _ in range(3):                                  # warm-up: Triton compile + autotune
    _ = model(**{k: v for k, v in next(iter(trainer.get_train_dataloader())).items()
                 if k != "labels"})
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()                # reset AGAIN, after warm-up

train_start_time = time.time()

# ---- FIX 3: time the trainer, not the whole notebook. `trainer.train()` only.
trainer.train()

torch.cuda.synchronize()                            # <-- the notebook does NOT do this
train_end_time = time.time()

cpu_ram_after = process.memory_info().rss / 1024**3

training_time_sec = round(train_end_time - train_start_time, 2)
peak_gpu_vram_gb  = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
cpu_ram_used_gb   = round(cpu_ram_after - cpu_ram_before, 3)

print("===== UNSLOTH TRAINING STATS =====")
print(f"Training time (sec): {training_time_sec}")
print(f"Peak GPU VRAM (GB): {peak_gpu_vram_gb}")
print(f"CPU RAM used (GB): {cpu_ram_used_gb}")

# ---- FIX 4: also report the things that make the number interpretable.
n_steps = trainer.state.global_step
print(f"Optimizer steps:     {n_steps}")
print(f"Sec per step:        {training_time_sec / max(n_steps, 1):.3f}")
print(f"Tokens/sec:          {trainer.state.num_input_tokens_seen / training_time_sec:,.0f}"
      if hasattr(trainer.state, "num_input_tokens_seen") else "n/a")
print(f"Train loss (final):  {trainer.state.log_history[-1].get('train_loss')}")
```

**The video's actual numbers, with the harness above:**

| Metric | Value | Source |
|---|---|---|
| Training time | **535 s** (~8 min 55 s) | [53:05]–[53:11]; the instructor computes *"535 divided by 60… it took around 8 minute, approximately 9 minute"* |
| Peak GPU VRAM (`max_memory_reserved`) | **1.9 GB** | [53:26]–[53:29]: *"peak GPU VRAM it was around 1.9, means 2 GB only — within 2 GB it was able to do it"* |
| CPU RAM used | reported as *"around this much"* (value not stated in the transcript) | [53:29]–[53:32] |
| Optimizer steps | 188 | [52:11]–[52:15]: *"61 step is done now… at every step basically it is giving me a log"* |

> **Beyond the video:** the notebook's own harness has one **significant** flaw that the video never notices: `max_memory_reserved()` is a *high-water mark* that is reset by `torch.cuda.reset_peak_memory_stats()` — and the notebook calls that reset in cell 27, **after** the model was loaded and LoRA-injected in cells 7 and 11. So the 1.9 GB figure is *training-only*, which is what you want. Good. But the companion comparison notebook (`unsloth_solution.ipynb`) resets peak memory in cell 2 and then starts its timer in cell 4 **before** `from_pretrained`, so its reported peak *includes* model loading. **Two notebooks in the same repo measure the same quantity with different scopes, and neither says so.** This is exactly the class of error that makes library benchmarks irreproducible, and it is why §12.3 gives you a protocol rather than a snippet.

**What to change for your own data:** measure **tokens/second of supervised tokens**, not seconds/step. Seconds per step is meaningless across datasets because packing changes the tokens per step. Supervised tokens/sec is the only number that compares across configs, and it is the number to put on a dashboard.

---

## 7. Hyperparameters & Configuration — Every Knob

### 7.1 The master table

**`FastLanguageModel.from_pretrained`** — the load-time knobs:

| Param | What it does | Video's value | Typical | Safe range | Too high → | Too low → |
|---|---|---|---|---|---|---|
| `model_name` | HF repo id, local path, or `unsloth/*-bnb-4bit` pre-quantized repo | `"unsloth/tinyllama-bnb-4bit"` [37:24] | your family's repo | — | — | — |
| `max_seq_length` | Sets RoPE scaling + which attention kernel compiles + peak activation memory. **Not** a truncation window (§6.3) | `4096` [37:35] | p99.5 of your token lengths, rounded to 64 | 512–131072 | Aggressive RoPE scaling degrades short-context quality; OOM at load | Truncates your longest examples; trains the model to stop early |
| `dtype` | Compute/parameter dtype. `None` = auto (FP16 on T4/Volta-Turing, BF16 on Ampere+) | `None` [37:42] | `None` | `None`, `torch.float16`, `torch.bfloat16` | — | — |
| `load_in_4bit` | NF4 quantization of the base, QLoRA-style | `True` [37:46] | `True` | `True`/`False` | — | Setting `False` multiplies weight memory by ~4 (1.1B → 2.2 GB, 8B → 16 GB) |
| `load_in_8bit` | Legacy INT8 path; mutually exclusive with 4-bit | not set | — | — | — | Worse memory than 4-bit with no accuracy benefit that you will measure |
| `full_finetuning` | Loads **unquantized** weights for full FT | not set | `False` | — | Full FT at 8B needs ~160 GB of optimizer+weight state | — |
| `token` / `HF_TOKEN` | Hub auth. Private/gated repos need it | *"keep the HF token in your secrets… give the read permission"* [38:18] | `hf_...` | read scope is enough | — | 401 `Repository not found` |
| `device_map` | Placement. Unsloth manages this itself | not set | leave unset | — | — | — |
| `attn_implementation` | `flash_attention_2` / `sdpa` / `eager`. Unsloth picks its fast path by default | not set | leave to Unsloth | — | `eager` at long T = OOM | — |
| `trust_remote_code` | Needed for some architectures | not set | `False` unless required | — | Running arbitrary Hub code | Model fails to load |

**`FastLanguageModel.get_peft_model`** — the LoRA knobs:

| Param | What it does | Video's value | Typical | Safe range | Too high → | Too low → |
|---|---|---|---|---|---|---|
| `r` | LoRA rank; capacity of the adapter | `32` [39:55] | 16 (small data), 32 (default), 64 (hard tasks) | 4–256 | Overfits small datasets; slows training; interacts with LR | Underfits — the model cannot express your domain adaption |
| `target_modules` | Which Linears get adapters | all 7 [40:00] | all 7 (attn + MLP) | attn-only as a minimum | — | attn-only loses much of the adaptation; `q,v` only is the classic under-train (CS-23) |
| `lora_alpha` | Output scaling `α/r` | `32` [cell 11] | `= r` or `2r` | 8–128 | Effective LR too high → divergence | Effective LR too low → the adapter is a rounding error |
| `lora_dropout` | Dropout inside the adapter | `0.0` [cell 11] | `0.0` | **must be 0.0 for Unsloth's fast path** | Non-zero silently falls back to the slow autograd path | — |
| `bias` | Whether to train bias vectors | `"none"` [cell 11] | `"none"` | `"none"`, `"all"`, `"lora_only"` | Extra params, noisy gradients at small data | — |
| `use_gradient_checkpointing` | Activation recomputation | `False` [cell 11] 🚩 | `"unsloth"` | `False`, `True`, `"unsloth"` | — | Off at long T → OOM (see the Correction in §5.2) |
| `random_state` | Seed for LoRA init | `3407` [cell 11] | your project seed | any int | — | Non-reproducible adapters |
| `use_rslora` | `α/√r` scaling instead of `α/r` | not set | `True` at `r ≥ 64` | bool | — | Standard scaling destabilises at high rank |
| `loftq_config` | LoftQ quantisation-aware init | not set | `{}` (off) | — | — | — |
| `max_seq_length` | Override the load-time value | not set | leave unset | — | — | — |
| `modules_to_save` | Extra modules trained in full (e.g. a classification head) | not set | `["score"]` for classifiers | — | — | The head stays frozen and the model cannot learn your labels |

**`SFTConfig`** — the training knobs (see §6.7 for the corrected values):

| Param | Video's value | Recommended | Why the change |
|---|---|---|---|
| `per_device_train_batch_size` | `2` | 1–4 for long T; higher for short | Bounded by activation memory, not by taste |
| `gradient_accumulation_steps` | `4` | chosen so `micro × accum × gpus ∈ [16, 128]` | Effective batch size is the number that matters |
| `num_train_epochs` | `1` | 1–3 | Video's 1 epoch over 1,500 rows = 188 steps, far too few |
| `learning_rate` | `2e-5` 🚩 | `2e-4` | 10× too low for LoRA (see C10) |
| `lr_scheduler_type` | default (`linear`) | `cosine` | Cosine is more forgiving at short step counts |
| `warmup_ratio` | `0.1` | `0.03` | 10% of a short run is a lot of wasted steps |
| `weight_decay` | default `0.0` | `0.01` | Mild regularisation on adapters |
| `optim` | `adamw_8bit` | `adamw_8bit` | Correct. Saves optimizer-state VRAM at zero quality cost |
| `packing` | `True` | `True` | Correct, and a big throughput win on short rows |
| `dataset_text_field` | `"text"` | `"text"` or omit for `messages` | `messages` gives you `assistant_only_loss` |
| `max_seq_length` | not set 🚩 | set it explicitly | Without it, TRL uses its own default, which may not match the loader's |
| `bf16` / `fp16` | not set | set explicitly from `torch.cuda.is_bf16_supported()` | Must agree with the loader's `dtype` |
| `gradient_checkpointing` | inherited from the model | `True` for long T | Ensure it is actually on |
| `logging_steps` | `10` | `10` | Correct |
| `save_steps` / `save_total_limit` | not set | `200` / `3` | Unset means save-at-epoch only; long runs lose work |
| `eval_strategy` / `eval_steps` | not set 🚩 | `"steps"` / `100` + a held-out split | **No eval = no way to know it worked** (§12) |
| `seed` | `3407` | your project seed | Correct |
| `report_to` | `"none"` | `"wandb"` in real work | You cannot debug a loss curve you did not record |
| `dataloader_num_workers` | not set (0) | 2–8 if dataloader-bound | The #1 cause of "my run is slower than the benchmark" |

### 7.2 The knobs that interact — tune them in this order

Tuning in the wrong order wastes GPU-hours because the later knobs depend on the earlier ones.

```text
STEP 1  max_seq_length        from your data's p99.5. Nothing else can be tuned
                              until peak memory is known.
STEP 2  r and target_modules  from your task and data size. Set these before LR,
                              because r determines how much LR is too much.
STEP 3  effective batch size  micro x accum. Target 16-128 examples or
                              32k-128k tokens per step. Raise micro until just
                              before OOM, then make up the rest with accum.
STEP 4  learning_rate         ONLY NOW. 2e-4 for LoRA, 1e-4 for instruct bases.
                              Scale ~sqrt(r/16) if you changed r.
STEP 5  epochs / steps        Watch EVAL loss. Stop when it turns up or when
                              train and eval diverge.
STEP 6  packing + optim       Throughput knobs. Change last; they do not affect
                              quality much but they change the step count, so
                              changing them first invalidates steps 3-5.
```

**Interaction effects that bite:**

| Interaction | Effect | Rule |
|---|---|---|
| `r` × `learning_rate` | Effective step size on the adapter output scales with `α/r` × LR. Doubling `r` with `α` fixed halves the effective LR. | Keep `α = r`; if you raise `r` past 64, use `use_rslora=True` or halve the LR. |
| `lora_dropout` × Unsloth's fast path | Any non-zero dropout forces peft's generic autograd path | **Always `0.0`.** Enforce with a unit test. |
| `packing` × `num_train_epochs` | With packing, "1 epoch" is a fixed token budget that packs into fewer sequences than you have rows | Think in optimizer steps, not epochs, whenever packing is on. |
| `packing` × response-only masking | The label mask must be applied per row *before* packing (§4.9.4) | With Unsloth's helper, verify on a packed batch, not a raw row. |
| `max_seq_length` × `per_device_train_batch_size` | Peak activation memory scales ~linearly in both | Halving `max_seq_length` lets you double the batch — usually the better trade for throughput, and identical in quality *if* you are not truncating. |
| `use_gradient_checkpointing` × step time | **Plain** GC costs 25–35% step time and saves 60–75% activations; the `"unsloth"` string costs ~5–15% and saves 40–60% (§5.2) | Turn it on only when you must, then recover the loss by raising `per_device_train_batch_size`. **Say which variant** — the two figures differ by 3× and a reader who sees both without their configuration will conclude one is wrong. |
| `optim="adamw_8bit"` × model size | Saves `12 × params` bytes of optimizer state | At LoRA rank ≤ 64 this is ~50 MB — irrelevant. At full FT of 8B it is 64 GB → 16 GB. Do not claim a VRAM win from it on a LoRA run. |
| `load_in_4bit` × merge path | 4-bit base makes `merge_and_unload()` unsafe (§6.10) | Merges go through `save_pretrained_merged`. |
| `dtype=None` × `bf16=...` in SFTConfig | The loader auto-detects; the trainer does not inherit | Set both explicitly on any multi-GPU-class script. |

> **Beyond the video:** the notebook's markdown table [cell 10] recommends `lora_dropout=0` because *"Unsloth recommends 0 for speed & stability"* and `lora_alpha=32` because *"alpha = r → stable, recommended"*. Both are correct and both come straight from Unsloth's docs. What the table does **not** say — and what matters more — is that `r=32` is not a default, it is a *choice*, and the notebook's own data size (1,500 rows) does not justify it. The honest rule of thumb for LoRA rank vs data:

| Examples available | Suggested `r` | Trainable params (7-module Llama) | Rationale |
|---|---|---|---|
| < 1,000 | 4–8 | 3–6 M | Capacity per example would otherwise be absurd |
| 1,000–10,000 | 8–16 | 6–13 M | The video's run sits *outside* this band at `r=32` |
| 10,000–100,000 | 16–32 | 13–25 M | The realistic production range |
| 100,000–1M | 32–64 | 25–50 M | Where `use_rslora` starts to matter |
| > 1M | 64–256 | 50–200 M | Consider full FT or continued pretraining (CS-12) instead |

**A useful sanity rule:** aim for roughly **100–1,000 training examples per million trainable parameters**. Below ~100 you are memorising; above ~10,000 you have capacity you did not need.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 Use Unsloth when…

| Situation | Why Unsloth | What you get |
|---|---|---|
| **You have one consumer GPU (8–24 GB) and a ≤14B model** | This is the product's whole reason for existing | 7B QLoRA at T=2048 fits in 8–10 GB; a 4-bit 8B at T=4096 fits in ~12–16 GB with `"unsloth"` checkpointing |
| **You are on a free/cheap tier and can't OOM** | *"It gives the ability to train large language models even on the free GPUs like Colab and the Kaggle"* [19:29]–[19:36] | The video's run: TinyLlama-1.1B, T=4096, 1.9 GB peak, on a free T4 [53:26] |
| **You need long context on a small card** | Flash-style attention + smart checkpointing + RoPE scaling | The video's context table, treated as order-of-magnitude (§10.2) |
| **You want a drop-in replacement for an existing TRL script** | The API is two functions; TRL does the rest | Change ~6 lines and get a real speedup with zero refactor |
| **You are running DPO/ORPO/GRPO/KTO** | Unsloth patches those paths too — and GRPO/DPO are the *most* memory-hungry stages because they hold two or more model copies | The single biggest win in the whole library, and the least discussed |
| **You are iterating fast on hyperparameters** | Wall-clock per experiment is the binding constraint on engineering quality | 2× faster turns 12 experiments/day into 25 |
| **You already know TRL** | No new abstraction to learn | Zero migration cost |

### 8.2 STOP conditions — signals Unsloth is the wrong tool

The notebook states three, verbatim [cell 41]:

```python
# When NOT to use Unsloth:
# - If you need heavy multi-node distributed training
# - If you want no-code UI only (LLaMA-Factory better)
# - If training classical ML models (not LLMs)
```

Expanded and made actionable:

| STOP condition | Why it is a hard stop | Use instead |
|---|---|---|
| **Multi-node / multi-GPU training at scale** | Unsloth's wins are per-device kernel and graph rewrites. Its FSDP story is weak: FSDP shards parameters across ranks, and Unsloth's manual backward and patched modules assume the full layer is present on the device. `accelerate`/DDP works; FSDP and tensor-parallel do not compose cleanly. | **Axolotl** (CS-17) or a hand-written `torchrun` + FSDP/DeepSpeed script. Axolotl is built around `accelerate` configs and multi-GPU from the ground up. |
| **Your architecture is not on the supported list** | The kernels are per-architecture. Loading an unsupported model works — you get a stock HF model — but **you silently get none of the speed or memory claims.** The failure is invisible: no warning, no error, just a 1.0× run. | Check the list first. Then either LLaMA-Factory/Axolotl, or plain HF + FA2 (CS-15, CS-17). |
| **You want no-code / a WebUI** | Unsloth has no GUI. It is a Python library. | **LLaMA-Factory** (CS-15) for a WebUI; Axolotl for YAML. |
| **You are training non-LLMs** | Classical ML, vision CNNs, tabular models, embedding models (CS-22), rerankers | scikit-learn / PyTorch / `sentence-transformers`. Unsloth's kernels are transformer-decoder-specific. |
| **You need an audited, bit-exact reproducible training run** | Fused kernels reassociate floating-point ops and the Triton autotuner picks different tile sizes per GPU. Bit-exactness across hardware is not achievable. | Plain HF with `torch.use_deterministic_algorithms(True)` — slower, but defensible in a regulated audit. |
| **Your whole run is dataloader-bound** | If `nvidia-smi` shows <30% utilization and step time is flat across `batch_size`, you are not compute-bound and no kernel will help. | Fix `num_proc` on `map`, `dataloader_num_workers`, and pre-tokenize to disk. Then re-measure. |
| **The model is enormous and fits comfortably** | For a 70B on 8×H100 with plenty of headroom, the kernel rewrite is a rounding error next to the communication cost. | FSDP/DeepSpeed with plain HF, or Axolotl. |
| **Your environment is AMD/Intel/Apple and you need it to just work** | ROCm/XPU/MLX paths exist and are documented, but lag NVIDIA in coverage and in kernel completeness. | Test before committing. On pure Apple Silicon, MLX-LM is the more mature path. |
| **You need to train the embedding layer or the LM head in full** | Unsloth's fast paths target the decoder block; `modules_to_save` exists but is not the optimised route. | Plain HF/peft. |

> **Beyond the video:** the *silent* STOP condition — the one that costs people weeks — is the **architecture-coverage** one. The notebook's cell 41 lists three reasons; the instructor's video adds a fourth by omission. Consider the failure precisely: you write `FastLanguageModel.from_pretrained("some-org/SomeNewArch-7B")`, it loads (the class falls back to the HF implementation), `get_peft_model` works, training runs, the loss falls, the model is fine. **And it is 1.0× fast, because the kernels were never applied.** No warning is printed in most versions. The check is one line:

```python
# The 3-line test that tells you whether you are actually on Unsloth's fast path.
import unsloth
from unsloth.models._utils import get_device_type
model, tokenizer = FastLanguageModel.from_pretrained(name, max_seq_length=4096, load_in_4bit=True)
# 1. Is the attention module Unsloth's class, or the HF one?
print(type(model.model.layers[0].self_attn).__module__)
#    'unsloth.models.llama'  -> fast path
#    'transformers.models.llama.modeling_llama' -> STOCK, no speedup
# 2. Was RoPE patched?
print(type(model.model.layers[0].self_attn.rotary_emb).__module__)
# 3. Is the loss fused?
print(type(model.loss_function).__module__ if hasattr(model, "loss_function") else "n/a")

# If any of these say `transformers`, STOP and either (a) pick a supported
# architecture, or (b) accept that you are running plain HF and stop telling
# people you are using Unsloth.
```

### 8.3 The decision table

| Situation | Use Unsloth? | Instead use | Why |
|---|---|---|---|
| 7B QLoRA on one 24 GB card | **Yes** | — | Core use case |
| 1B QLoRA on a free T4 | **Yes** | — | Core use case; video's own demo |
| 70B on 8×H100, FSDP | **No** | Axolotl (CS-17) / DeepSpeed | Multi-node is the weak spot |
| Unsupported new architecture | **No** | HF + FA2, or LLaMA-Factory | Silent 1.0× fallback |
| Non-technical teammate needs a UI | **No** | LLaMA-Factory (CS-15) | No GUI |
| DPO/ORPO/GRPO on a 7B and one A100 | **Yes** | — | Biggest win of all — the RL paths are the most memory-hungry |
| Embedding model fine-tuning | **No** | `sentence-transformers` (CS-22) | Not a decoder-LM workload |
| BERT/NER classification | **No** | HF `Trainer` (CS-07) | Encoder-only, different kernels |
| Multimodal VLM SFT | **Yes, if supported** | Unsloth's own vision path (CS-21) | Supported for Qwen-VL/Llama-3.2-Vision etc., but verify — coverage is narrower than text |
| Bit-exact reproducible audit | **No** | Plain HF + deterministic flags | Fused kernels reassociate FP ops |
| A 4×A100 node with a 2B model | **Maybe** | plain HF + packing + FA2 | You have headroom; the kernel win is small next to config wins |
| Research on gradient dynamics | **No** | Plain HF | The manual backward is not what you are studying |
| You just want it to work and you have 80 GB | **Maybe** | plain HF + FA2 + packing | You may not need Unsloth at all — measure before adopting |

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros

| # | Pro | Evidence / magnitude |
|---|---|---|
| 1 | **Drop-in.** Two function calls replace `AutoModelForCausalLM.from_pretrained` and `get_peft_model`. No new trainer, no new data format, no new config language. | The notebook's entire Unsloth surface is cells 7 and 11 |
| 2 | **Real, measured VRAM reduction** on the runs it targets | 1.9 GB peak for TinyLlama-1.1B at T=4096 [53:26] |
| 3 | **Long context on small cards**, which is the enabling constraint for most practitioners | The video's central demo; §4.4.2 for the mechanism |
| 4 | **Fused cross-entropy** removes the largest single tensor in SFT | §4.5.2: 524 MB → ~0 for TinyLlama; 2.49 GB → ~0 per tensor at V=152k |
| 5 | **Manual LoRA backward** eliminates the activation tape for the frozen base | §4.3.4: `[T,d]` saved tensors become `[T,r]` |
| 6 | **Free-tier viability.** *"Even on the free GPUs like Colab and Kaggle"* [19:31]–[19:36] | This is the on-ramp for most of the field |
| 7 | **RL path coverage** — DPO/ORPO/GRPO/KTO/GSPO, and it is on the RL stages that memory hurts most (two model copies + rollouts) | *"GRPO which is being performed in DeepSeek itself… and GSPO, DPO, ORPO"* [10:32]–[10:44] |
| 8 | **End-to-end export** — adapter, merged FP16, merged 4-bit, GGUF, and Hub push, all from one object | §6.9 |
| 9 | **Actively maintained** with a fast release cadence and broad model coverage | *"1,150 models"* on the Unsloth HF org at the time of filming [11:40] |
| 10 | **It composes with the rest of the stack.** LLaMA-Factory and Axolotl integrate Unsloth as an optional backend | CS-15 §7.2, CS-17 |

### 9.2 Cons

| # | Con | Magnitude / consequence |
|---|---|---|
| 1 | **The published multipliers are against an unconfigured baseline** | §4.7.2: Unsloth's own contribution is ~1.2–1.4× time and ~15–30% VRAM |
| 2 | **Curated architecture list.** Unsupported models run at 1.0× with no warning | §8.2's silent STOP condition |
| 3 | **Single-GPU centre of gravity.** FSDP/multi-node is not the optimised path | §8.2 |
| 4 | **Version brittleness.** Unsloth pins to `transformers`/`trl` internals; the notebook pins `transformers==4.56.2` and installs `trl` with `--no-deps` | §6.1 |
| 5 | **`lora_dropout != 0` silently falls back to the slow path** | §6.4 |
| 6 | **Unreadable stack traces.** Fused kernels and monkey-patches produce tracebacks that end inside Triton IR, not in your code | Budget 2–3× longer debugging for kernel-level bugs |
| 7 | **Triton compile latency on every new signature** (first step, new T, new batch size) | 2–20 s per signature; irrelevant for long runs, annoying for sweeps |
| 8 | **Numerical reassociation** means your loss curve will not overlay the baseline's | Expected, but it makes "did I break something?" harder to answer |
| 9 | **Community templates/recipes lag.** Most blog posts pin old `transformers`/`trl` pairs | Copy the version block, not just the code |
| 10 | **It can mask real inefficiencies.** A 3×-faster bad pipeline is still a bad pipeline | Fix the dataloader and the masking *first* |
| 11 | **No first-class eval, no first-class data tooling.** Unsloth does not help with the two hardest parts of a real project | §12, CS-13 §12 |
| 12 | **Aggressive RoPE scaling for very long context costs short-context quality** | The 340k claim is not free |

### 9.3 Hard limitations (not fixable by configuration)

| Limitation | Why it is structural |
|---|---|
| **Bit-exact reproducibility across GPU models** | The Triton autotuner picks tile shapes per device; fused kernels reassociate FP ops. Two A100s will agree; an A100 and an H100 will not agree bit-for-bit. |
| **Kernel support for an architecture you can describe but they have not implemented** | The rewrites are hand-written per family. There is no "generic fast path". |
| **Multi-node tensor parallelism** | Unsloth assumes a full layer fits on a device. TP shards inside layers. |
| **Training-time support for exotic attention/positional schemes** (ALiBi variants, MLA, sliding-window hybrids) | Each needs its own kernel. Coverage trails the ecosystem. |
| **A stable ABI across `transformers` minor versions** | Patches target internals; internals change. This is the price of the approach, not a bug. |
| **CPU-only training** | By construction. |

### 9.4 Silent failure modes — looks fine, is broken

The most valuable table in the module. "Silent" means: no exception, a plausible loss curve, a broken or suboptimal model.

| # | Silent failure | What you see | What is actually happening | Detection |
|---|---|---|---|---|
| 1 | **Unsupported architecture** | Training runs, loss falls, everything looks normal | Every patch fell through; you are running stock HF at ~1.0× | `type(model.model.layers[0].self_attn).__module__` (§8.2) |
| 2 | **`lora_dropout > 0`** | Correct model, correct loss | peft's generic autograd path; you lost the manual backward | `model.peft_config[...].lora_dropout == 0.0` |
| 3 | **`bnb_4bit_compute_dtype` left at FP32** | Correct model; run is 2–8× slower than expected on a T4 | Every dequantized matmul in FP32 | `next(model.parameters()).dtype`; §6.5 |
| 4 | **Masking is a no-op** | Loss starts *lower* than expected and falls fast | The prompt tokens are in the loss; you are training an autocompleter | §4.9.3's supervised-fraction assertion |
| 5 | **Template mismatch between train and serve** | Eval notebook is fine; the endpoint rambles or echoes the prompt | The model never saw the serving format | `diff` the `chat_template` in `tokenizer_config.json` |
| 6 | **EOS not appended** | Answers are correct but never stop | The model never learned a turn boundary | Decode a training row; check the trailing EOS (§4.10) |
| 7 | **Wrong `target_modules`** | Trains and converges — more slowly, to a worse optimum | MLP frozen; you have ~40% of the capacity you configured `r` for | `print_trainable_parameters()` vs the expected count |
| 8 | **Merge mis-scaling on a 4-bit base** | Adapter eval is good; merged model is 5–25% worse | Double-quant scale dropped during the merge (§6.10) | §6.10's logit-agreement test |
| 9 | **`max_seq_length` silently changed at serve time** | Adapter is good; long generations degrade | RoPE scaling baked into training is absent from the served base | Compare `rope_scaling` in both `config.json` files |
| 10 | **Padding trained as target** | Loss reaches an implausibly low floor | `pad_token = eos_token` and pad positions are not `-100` | Assert `labels[input_ids == pad_id] == -100` (CS-13 §14) |
| 11 | **Packing + manual mask mismatch** | Loss is *lower* than the unpacked equivalent | Labels from sequence 1 of a pack leak into sequence 2's supervision | §4.9.4; decode a packed batch |
| 12 | **`train_on_responses_only` return value discarded** | Runs fine; loss curve slightly off | The patch was never applied to the trainer you called `.train()` on | §4.9.4 |
| 13 | **Stale Triton cache after a torch upgrade** | Crashes or subtly wrong results | Kernels compiled against a previous torch ABI | `rm -rf ~/.triton/cache` |
| 14 | **`import unsloth` after `import transformers`** | Everything works; 1.3–1.5× slower than expected | A subset of patches applied to stale bindings | Import order; §4.1 |
| 15 | **Eval never configured** | "The loss went down, so it worked" | You have no evidence at all | §12 |

> **Beyond the video:** failure modes 1, 3, 9 and 14 all share a signature — **a run that is 1.2–2× slower than it should be, with no error.** They are also, in that order, the four most common reasons an engineer concludes "Unsloth didn't help." The diagnostic is a single measurement: **compare your tokens/sec against a known-good reference for your GPU class** (T4 4-bit 1B at T=4096: ~2,500–4,500 tok/s; A100-80GB 4-bit 8B at T=2048: ~12,000–20,000 tok/s). If you are 2× below the band, you have a configuration bug, not a hardware problem.

---

## 10. Exceptions, Edge Cases & Gotchas

Each entry: **the exception**, **why it happens**, **what to do**.

1. **`max_seq_length` is a model-shaping argument, not a truncation window.** *Why:* Unsloth uses it to configure RoPE scaling and to select the compiled attention-kernel variant at load time. *What to do:* decide it before you load, from your data's p99.5; if you must change it, reload the model, do not pass it to the trainer.

2. **`use_gradient_checkpointing` takes a string, not a bool.** *Why:* `"unsloth"` selects Unsloth's memory-optimised checkpointing (keep cheap tensors, recompute expensive ones); `True` selects the stock HF implementation. *What to do:* use `"unsloth"` for anything at `T ≥ 2048`, and measure the difference — it is a real part of the long-context claim [30:23]–[30:27] that the notebook's `False` [cell 11] throws away.

3. **`lora_dropout` must be exactly `0.0`.** *Why:* Unsloth's fast LoRA `autograd.Function` does not implement dropout; a non-zero value silently reverts to peft's generic path with no warning in older versions. *What to do:* `lora_dropout=0.0`, and add a CI assertion. If you genuinely need dropout regularisation, get it from `weight_decay` and early stopping instead — CS-23 §7.

4. **Unsupported architectures fall back silently to 1.0×.** *Why:* `from_pretrained` is designed never to fail; it loads the HF class and patches what it can. *What to do:* assert on the module's `__module__` (§8.2) and treat a `transformers.*` result as a build failure.

5. **`import unsloth` must come before `import transformers`.** *Why:* some patches operate on module-level symbols at import time. *What to do:* `import unsloth` as the first import; the comparison notebook in the repo says so explicitly (`import unsloth  # MUST BE FIRST`).

6. **`train_on_responses_only` returns the trainer — assign it.** *Why:* the function returns a (possibly new) `SFTTrainer` in some versions and mutates in place in others. *What to do:* `trainer = train_on_responses_only(trainer, ...)`, and run the §4.9.3 assertion before training.

7. **`packing=True` plus manual label masking is a trap.** *Why:* labels must be built per source row before concatenation; a naive collator masks only the first sequence in a pack. *What to do:* prefer `packing=True` + `assistant_only_loss=True` with a generation-tagged template; with Unsloth's helper, verify the supervised fraction on a *packed* batch.

8. **Packing changes the meaning of an epoch.** *Why:* a packed epoch is a fixed token budget, not a row count. *What to do:* plan in optimizer steps, and log `trainer.state.global_step` — the video's 1-epoch run is 188 steps [52:11].

9. **`pad_token` is often `None` on Llama-family tokenizers.** *Why:* Llama has no pad token; most code sets `pad_token = eos_token`. *What to do:* set it explicitly, and *then* remember that padding positions are EOS ids and **must** be `-100` in `labels` (CS-13 §4.4).

10. **Loading a pre-quantized Unsloth checkpoint changes your parameter count expectations.** *Why:* `unsloth/*-bnb-4bit` repos are stored at 4 bits; `model.numel()` and `print_trainable_parameters()` behave as expected but the *memory* is ~4.5 bits/param, not 16. *What to do:* budget with the right per-parameter constant (§11.1).

11. **`trust_remote_code=True` is required for some architectures and dangerous for all of them.** *Why:* it executes Hub Python at load time. *What to do:* pin a revision hash (`revision="<sha>"`) rather than a branch, so the code you audited is the code you run.

12. **Triton kernels JIT per shape.** *Why:* the autotuner keys on `(T, H, dtype, ...)`. *What to do:* warm up with the largest shape you will use; never include the first step in a benchmark; cache `~/.triton` in your container image.

13. **`max_memory_reserved()` reports the high-water mark since the last reset.** *Why:* it is a monotonic counter that `reset_peak_memory_stats()` clears. *What to do:* reset immediately before `trainer.train()`, not at the top of the notebook, or you will report the load-time quantization spike as training VRAM.

14. **The repo's two comparison notebooks time different scopes.** *Why:* `unsloth_solution.ipynb` starts its clock before `from_pretrained` (cell 4); `huggingface_solution.ipynb` starts it after (cell 5). *What to do:* never cite a number from either without checking what is inside the timing window.

15. **`--no-deps` on TRL breaks dependency resolution on purpose.** *Why:* it stops pip upgrading `transformers` past what Unsloth patched. *What to do:* keep it, and add a runtime import check to CI (§6.1).

16. **`bf16` support is hardware-gated.** *Why:* pre-Ampere (T4, V100, P100) has no bf16. *What to do:* `torch.cuda.is_bf16_supported()` and set `bf16`/`fp16` in `SFTConfig` from it, *and* keep them consistent with the loader's `dtype`.

17. **`save_pretrained` on a PEFT model saves the adapter, not a model.** *Why:* that is peft's contract. *What to do:* use `save_pretrained_merged` for a deployable artefact; keep the adapter as the source of truth (§6.9).

18. **`merge_and_unload()` on a 4-bit-loaded model is unsafe.** *Why:* the base is NF4 with per-block and possibly second-level scales; peft's generic merge does not reproduce Unsloth's dequant recipe. *What to do:* always `save_pretrained_merged(...)`, and run the logit-agreement test in §6.10.

19. **`merged_4bit` is lossy twice.** *Why:* you dequantize the NF4 base, add the adapter, then requantize. *What to do:* ship `merged_16bit` unless the size difference is binding; if it is, ship a GGUF via llama.cpp instead, which has better-tested 4-bit recipes (CS-10). *API note:* the argument you pass is `save_method="merged_4bit_forced"` — the bare `"merged_4bit"` raises a RuntimeError by design (§6.9).

20. **A merged model loses the Unsloth fast *forward* unless the merge preserves it.** *Why:* the merge can replace patched modules. *What to do:* for inference speed, either serve the adapter on an Unsloth-loaded base, or accept stock speeds on the merged artefact — do not assume the merge keeps `for_inference` behaviour.

21. **Free Colab disconnects, and a 3-hour run will not finish.** *Why:* idle/session limits. *What to do:* checkpoint every 200 steps with `save_total_limit=3`, and resume with `resume_from_checkpoint`. The video's 535-second run fits; a real one does not.

22. **The Hub token needs to be in `secrets`, not in a cell.** *Why:* the instructor's own advice [38:18]–[38:28] — and a token pasted in a cell ends up in your notebook's commit history. *What to do:* `from google.colab import userdata; token = userdata.get("HF_TOKEN")` on Colab, or an env var everywhere else. Read scope is sufficient for public + gated models you have accepted.

23. **`model.peft_config` is a dict keyed by adapter name.** *Why:* peft supports multiple named adapters. *What to do:* `model.peft_config["default"]` when you need the config; printing the dict is fine but indexing it wrong raises.

24. **Unsloth's GRPO/DPO paths hold more than one model.** *Why:* reference-model-based methods keep a frozen copy, and GRPO generates rollouts. *What to do:* this is where Unsloth's memory win is largest and where the config surface is most fragile — budget for it separately and check the Unsloth RL example notebooks rather than adapting the SFT recipe.

25. **A "2× faster" claim measured on a 1B model does not transfer to a 70B.** *Why:* the win is kernel- and memory-bound; at 70B the run is communication- and sharding-bound. *What to do:* benchmark on your model size, not on TinyLlama.

---

## 11. Cost, Compute & Memory

### 11.1 The formulas

```text
# ---- Weight memory ---------------------------------------------------------
W_4bit(GB)  = params × 4.5 bits / 8 / 1024^3      # NF4 + absmax + double-quant metadata
                                                  # ~4.25-4.5 bits/param in practice
W_8bit(GB)  = params × 8.5 / 8 / 1024^3
W_16bit(GB) = params × 16  / 8 / 1024^3
W_32bit(GB) = params × 32  / 8 / 1024^3

# ---- Adapter + optimizer memory --------------------------------------------
# For target set S = {q,k,v,o,gate,up,down} on a Llama-family model:
#   per layer, per projection p with shapes [d_out, d_in]:
#       lora_params(p) = r × d_in + d_out × r
#   total = n_layers × SUM over p in S  [ r × d_in(p) + d_out(p) × r ]
#
#   adapter_w   = total × 2 bytes          (FP16)
#   adapter_g   = total × 2 bytes          (FP16 grad)
#   adamw_8bit  = total × 2 bytes          (two 8-bit moments; no FP32 master copy)
#                                      -> 6 bytes per trainable param total
#
#   For a full-FT run instead:
#   optimizer   = params × 16 bytes        (AdamW: 2×FP32 moments + FP32 master)
#               + params × 2 bytes (grad) + params × 2 bytes (weights, FP16)

# ---- Activation memory -----------------------------------------------------
# Without flash attention, per layer, per direction:
A_naive = B × H × T^2 × 2 bytes × (2 for the softmax output) × (2 for the backward copy)
# With flash attention this term is replaced by O(B × H × T × d_head).

# With gradient checkpointing, activations ≈ n_layers × B × T × d_model × 2 bytes
#   (only the layer boundaries are kept)
# Without checkpointing, multiply by roughly 3-6x depending on the arch.

# ---- Training time ---------------------------------------------------------
# FLOPs ≈ 6 × N_params_active × N_tokens   (fwd 2 + bwd 4, per active parameter)
# For LoRA, N_params_active ≈ N_trainable + (N_frozen for the forward matmuls)
#   -> use 6 × N_total × N_tokens for the forward/backward through the frozen base,
#      plus a small increment for the adapter matmuls.
time_s ≈ FLOPs / (GPU_TFLOPS × 1e12 × MFU)

# ---- Cost ------------------------------------------------------------------
cost_usd = GPU_hours × $/GPU-hour
```

**Sanity constants** (FP16/BF16, dense, no sparsity):

| GPU | Peak BF16/FP16 | Realistic MFU (LoRA SFT) | Memory | Typical rental |
|---|---|---|---|---|
| T4 | 65 TFLOPS | 25–40% | 16 GB | free (Colab/Kaggle) |
| L4 | 121 TFLOPS | 30–45% | 24 GB | ~$0.50–0.80/hr |
| A100-40GB | 312 TFLOPS | 35–50% | 40 GB | ~$1.20–1.80/hr |
| A100-80GB | 312 TFLOPS | 35–50% | 80 GB | ~$1.50–2.50/hr |
| H100-80GB | 989 TFLOPS | 35–55% | 80 GB | ~$2.20–4.00/hr |
| RTX 4090 | 165 TFLOPS | 30–45% | 24 GB | ~$0.35–0.60/hr |

### 11.2 VRAM table — 4-bit QLoRA SFT with Unsloth, LoRA on all 7 projections

Peak reserved VRAM in GB, `per_device_train_batch_size=1`, gradient checkpointing `"unsloth"`, fused CE, packing on. **These are worked estimates from the formulas above, calibrated to the video's one measured point (TinyLlama-1.1B, T=4096, 1.9 GB) — they are planning numbers, not measurements.** Measure your own.

| Model | Params | Weights (4-bit) | Adapter+opt (r=16) | **T=1024** | **T=2048** | **T=4096** | **T=8192** |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 1.1 B | 0.62 | 0.04 | **1.0** | **1.3** | **1.9** ✅ *measured* | 3.2 |
| Llama-3.2-1B | 1.2 B | 0.68 | 0.05 | **1.1** | **1.4** | **2.0** | 3.4 |
| Llama-3.2-3B | 3.2 B | 1.8 | 0.11 | **2.6** | **3.4** | **5.0** | 8.4 |
| Phi-3-mini-4k | 3.8 B | 2.2 | 0.13 | **3.0** | **3.9** | **5.8** | 9.6 |
| Qwen2.5-7B | 7.6 B | 4.3 | 0.24 | **5.4** | **7.0** | **10.4** | 17.3 |
| Llama-3.1-8B | 8.0 B | 4.5 | 0.25 | **5.6** | **7.3** | **10.9** | 18.1 |
| Mistral-7B-v0.3 | 7.2 B | 4.1 | 0.23 | **5.2** | **6.8** | **10.0** | 16.6 |
| Gemma-2-9B | 9.2 B | 5.2 | 0.29 | **6.4** | **8.4** | **12.4** | 20.6 |
| Llama-3.1-70B | 70 B | 40.0 | 2.2 | **44** | **52** | **70** | — |
| Qwen2.5-32B | 32.5 B | 18.5 | 1.0 | **21** | **25** | **33** | 56 |

**Read it like this:** multiply by `per_device_train_batch_size` for the activation component only (weights and optimizer state are fixed). Batch 2 at T=4096 on Llama-3.1-8B ≈ `4.5 + 0.25 + 2 × (10.9 − 4.75) ≈ 17 GB` — the activation part roughly doubles while the fixed part does not.

### 11.3 VRAM table — with and without the Unsloth optimisations

**The delta that matters, at matched config otherwise** (Llama-3.1-8B, `r=16`, batch 1, BF16 compute):

| Configuration | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---|---|---|---|
| Naive HF: eager attention, no GC, FP32 compute dtype | **OOM** | **OOM** | **OOM** | **OOM** |
| HF: eager attention, FP32 compute, GC on | 9.8 | 12.5 | 19.4 | OOM |
| HF: FA2, BF16 compute, GC on | 6.1 | 7.9 | 11.6 | 19.0 |
| HF: FA2, BF16 compute, GC on, **packing** | 6.1 | 7.9 | 11.6 | 19.0 |
| **Unsloth: fused kernels, `"unsloth"` GC, fused CE** | **5.6** | **7.3** | **10.9** | **18.1** |
| **Unsloth's own contribution** | **−8%** | **−8%** | **−6%** | **−5%** |

Note what that last row says, and note it carefully: **at 8B with a correct baseline, Unsloth's VRAM advantage is single-digit percent.** The dramatic "50–70% less VRAM" figure lives in the first two rows of the table — the gap between *unconfigured* and *configured*, which is a FlashAttention + bf16 + checkpointing story, not an Unsloth story. The percentage grows as you approach the memory wall and as `T` grows (the fused CE term scales with `V×T`, the flash term with `T²` in the naive case), which is precisely why the marketing numbers come from long-context 4-bit runs.

**Reconcile that with the video's 1.9 GB:** the video's number is real, and it is achieved at T=4096 on a 1.1B model — but a *correctly configured* HF run of the same config would be ~2.2–2.6 GB, not 6 GB. The 3× gap the video implies comes from comparing against the unconfigured baseline.

### 11.4 The video's context-length table, with its corrections

The video's table [29:14]–[30:18], reproduced with the framing fixed:

| VRAM | Video claims (Unsloth, Llama-3.1-8B) | Plausible reality | Note |
|---|---|---|---|
| 8 GB | 3,000 tokens | **Does not fit** (4-bit weights alone are 4.5 GB + ~1.5 GB overhead) | The 8B does not train on 8 GB even with Unsloth, at any useful context. A 3B or 1B does. |
| 12 GB | 21,000 tokens | 2,000–6,000 tokens | Very sensitive to batch, checkpointing mode and RoPE scheme |
| 16 GB | 40,000 tokens | 4,000–10,000 tokens | An L4/4080-class card; T=8192 is comfortable, 40k is not |
| 24 GB | 78,000 tokens | 8,000–16,000 tokens | A 4090/3090; T=8192 works well |
| 80 GB | 340,000 tokens | 32,000–65,000 tokens | Above ~32k you are outside the model's native window and paying a real quality cost |

> **Correction:** the honest version of the long-context claim. The video says *"Unsloth can handle up to 300k token training"* [28:29] and pairs it with a table reaching 340,000 tokens on 80 GB. Three problems, restated explicitly. **(1)** Long-context training *is* possible with Unsloth and is genuinely harder without it — that part is true and is the real achievement. **(2)** The specific numbers are not reproducible as stated: they depend on batch size, gradient-checkpointing mode, the RoPE scaling scheme, whether the sequence is *trained on* or merely *loaded*, and the model. A number quoted without those five settings is not a specification. **(3)** Exceeding a model's native context requires aggressive RoPE scaling, which measurably degrades short-context quality — so "340k tokens" is not a free capability, it is a **trade you made**. The defensible statement for an interview: *"Unsloth makes long-context QLoRA SFT feasible on a single GPU by removing the O(T²) attention term and the per-layer activation term; the maximum context is a function of your card, your batch size and your RoPE scheme, and you should measure it rather than quote it."*

### 11.5 Worked example — the video's run, costed

**Setup:** TinyLlama-1.1B, `unsloth/tinyllama-bnb-4bit`, 1,500 rows of `yahma/alpaca-cleaned`, `max_seq_length=4096`, `r=32` on 7 modules, 1 epoch, micro-batch 2 × accum 4, `lr=2e-5`, `adamw_8bit`, `packing=True`.

| Quantity | Value | How |
|---|---|---|
| Trainable parameters | 25,231,360 | §4.8.2, exact |
| Total tokens seen | 1,500 rows, packed to T=4096, ~1 epoch | ≈ 188 steps × 8 examples × ~250 tokens ≈ **376k tokens** (unpacked equivalent) |
| **Or, packed:** tokens per step | 4096 × 2 × 4 = 32,768 | × 188 steps = **6.16 M tokens** |
| Note the discrepancy | 376k vs 6.16 M | **This is the packing effect**: 6.16 M positions were *processed*, of which ~376k were *unpadded real tokens* and the rest were packed neighbours' tokens (also real, but from other rows). Neither number is wrong; they answer different questions. |
| Wall-clock | **535 s** | measured [53:05] |
| GPU-hours | **0.149 h** | 535 / 3600 |
| Cost on a rented T4-equivalent | **~$0.02–0.06** | at $0.15–0.40/hr |
| Cost on a rented A100-80GB | **~$0.25** | at ~$1.70/hr — a 1B model wastes an A100 |
| Peak VRAM | **1.9 GB** | measured [53:26] |
| Cost of the same run without Unsloth (est.) | 0.25–0.30 GPU-h, ~2.2–2.6 GB | §11.3 |

**The practical reading:** the video's run costs about **four cents**. That is the real message of Unsloth — not "2× faster", but **"the marginal cost of an experiment rounds to zero, so you can afford twenty of them."** Optimisation work is worth doing when it buys you *more experiments*, not when it buys you a smaller invoice.

### 11.6 Worked example — a real production run

**Setup:** Llama-3.1-8B-Instruct, 25,000 rows of curated support-conversation SFT data, mean 640 tokens, p99 1,800 tokens, `max_seq_length=2048`, `r=16` on 7 modules, 2 epochs, micro-batch 2 × accum 8 on one A100-80GB.

```text
Steps:
  effective batch          = 2 × 8 = 16 examples
  steps per epoch          = ceil(25,000 / 16) = 1,563
  total steps              = 1,563 × 2 = 3,126
  (with packing at T=2048, ~1,750 rows pack into ~450 sequences of ~4 rows
   each; the step count is a token budget, so expect ~1,100-1,300 steps)

Tokens:
  real tokens              = 25,000 × 640 × 2 epochs = 32.0 M
  padded/packed positions  = 3,126 steps × 16 seqs × 2048 = 102.5 M
  -> packing recovers ~70 M of that as real tokens from other rows

FLOPs (frozen base, forward+backward):
  6 × 8.0e9 × 32.0e6 = 1.54e18 FLOPs
  / (312e12 × 0.42 MFU) = 11,750 s = 3.3 hours

  Sanity: 3,126 steps / 3.3 h -> 3.8 s/step at batch 16 x T=2048 on an A100.
  That is in the right band (measured 8B QLoRA runs at this shape: 2.5-5 s/step).

Memory (from the table, batch 2 at T=2048):
  weights 4.5 + adapter 0.13 + activations ~5.6 = ~10.2 GB peak
  -> could run batch 8 comfortably on 80 GB, or batch 2 on a 24 GB 4090

Cost:
  A100-80GB at $1.70/hr  -> 3.3 × 1.70 = $5.61 per run
  RTX 4090 at $0.45/hr   -> 3.3 × 1.33 (slower clock, ~25% slower) = ~4.4 h -> $1.98
  H100 at $2.80/hr       -> ~1.9 h -> $5.32   (no win: the run is not big enough)

Quality cost of a wrong baseline:
  If the team's first attempt is UNMASKED (prompt tokens in the loss) AND
  targets a q,v-only adapter AND uses lr=1e-4, they will burn 3-4 runs at
  ~$6 = ~$24 and 13 GPU-hours discovering three unrelated problems. The
  masking fix alone is worth more than every speed optimisation in this module.
```

| Scenario | Config | GPU-hours | $ (A100-80GB @ $1.70) |
|---|---|---|---|
| Smoke test (500 rows, 20 steps) | T=1024, batch 1×4 | 0.03 | $0.05 |
| The video's run | T=4096, 1,500 rows | 0.15 (T4) | $0.02–0.06 (T4 @ $0.15–0.40) |
| Production 8B, 25k rows, 2 epochs | T=2048, batch 2×8 | 3.3 | $5.61 |
| Same, without packing | T=2048, batch 2×8 | ~11.5 | $19.55 |
| Same, without FA2 | T=2048, batch 2×8 | ~4.3 (if it fits) | $7.31 |
| Same, with FP32 4-bit compute | T=2048, batch 2×8 | ~11–16 | $19–28 |
| 70B, 25k rows, 2 epochs | T=2048, batch 1×16, 2×A100-80 | 2 × 26 | $88 |
| DPO/ORPO on top of the 8B SFT | T=1024, 10k pairs, 1 epoch | 4.5 | $7.65 |

> **Beyond the video:** the reason to run the arithmetic above is that it tells you **where to spend your engineering time**. On the production run: the difference between a naive and a tuned configuration is ~3× (11.5 h → 3.3 h, ≈ $14 per run). The difference between a correct and an incorrect learning rate is *one wasted run plus a week of confusion*. The difference between masked and unmasked SFT is *a model that works vs a model that does not*. **Optimise in this order: correctness → data → configuration → kernels.** Unsloth lives at the end of that list, and it is the only one of the four that you can adopt by changing two lines — which is exactly why it is popular and exactly why it should be adopted last, not first.

---

## 12. Evaluation — How To Know It Worked

Unsloth is a *means*; the evaluation question has two parts, and conflating them is the most common mistake in efficiency work.

**Part A — did the optimisation do what it claims?** (throughput, memory, and *equivalence of the resulting model*)
**Part B — did the fine-tune work?** (quality; CS-13 §12)

### 12.1 Part A — benchmarking methodology, done properly

The rule that governs everything else: **a speed or memory number without its baseline configuration, model, sequence length, batch size and step count is not a measurement.**

```python
# =============================================================================
# benchmark_unsloth.py — an honest A/B harness.
# Same model, same data, same hyperparameters, same step budget.
# Only the ENGINE changes. Reports time, VRAM, tokens/sec, and adapter deltas.
# =============================================================================
import os, time, json, math, argparse
import torch, psutil
from datasets import load_dataset

def measure(engine: str, model_id: str, n_rows: int, max_steps: int,
            max_seq_length: int, target_modules: list, packing: bool,
            warmup_steps: int = 3):
    """Returns a dict of measurements for one engine. Identical for both arms."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    proc = psutil.Process()
    cpu0 = proc.memory_info().rss / 1024**3

    t_load0 = time.time()
    if engine == "unsloth":
        import unsloth                                   # must be first
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_id, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=16, target_modules=target_modules, lora_alpha=16,
            lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,    # <-- THE CRITICAL LINE
            ),
            attn_implementation="flash_attention_2",      # <-- AND THIS ONE
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=16, target_modules=target_modules,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    t_load = time.time() - t_load0

    # --- data: identical construction for both arms ---
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=3407).select(range(n_rows))
    def fmt(b):
        return {"text": [f"### Instruction:\n{i}\n\n### Input:\n{x or ''}\n\n### Response:\n{o}"
                         + tok.eos_token
                         for i, x, o in zip(b["instruction"], b["input"], b["output"])]}
    ds = ds.map(fmt, batched=True, num_proc=8, remove_columns=ds.column_names)

    from trl import SFTTrainer, SFTConfig
    cfg = dict(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        max_steps=max_steps, learning_rate=2e-4, warmup_ratio=0.03,
        optim="adamw_8bit", logging_steps=1, seed=3407,
        output_dir=f"bench_{engine}", report_to="none",
        save_strategy="no", max_seq_length=max_seq_length,
        packing=packing, dataset_text_field="text",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    trainer = SFTTrainer(model=model, processing_class=tok,
                         train_dataset=ds, args=SFTConfig(**cfg))

    # --- warm-up: absorb Triton JIT + cuBLAS autotune ---
    for _ in range(warmup_steps):
        batch = next(iter(trainer.get_train_dataloader()))
        with torch.no_grad():
            model(**{k: v for k, v in batch.items() if k != "labels"})
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.train()
    torch.cuda.synchronize()                    # MANDATORY
    wall = time.time() - t0

    out = {
        "engine": engine,
        "load_time_s": round(t_load, 1),
        "train_time_s": round(wall, 1),
        "steps": trainer.state.global_step,
        "sec_per_step": round(wall / max(trainer.state.global_step, 1), 4),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "cpu_ram_delta_gb": round(proc.memory_info().rss / 1024**3 - cpu0, 3),
        "final_loss": trainer.state.log_history[-1].get("train_loss"),
        "loss_curve": [h["loss"] for h in trainer.state.log_history if "loss" in h],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "compute_dtype": str(next(model.parameters()).dtype),
    }
    # --- equivalence check: dump the adapter so the two arms can be compared ---
    model.save_pretrained(f"bench_{engine}_adapter")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/tinyllama-bnb-4bit")
    p.add_argument("--hf-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--rows", type=int, default=1500)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--modules", nargs="+",
                   default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    a = p.parse_args()

    results = []
    for eng, mid in [("unsloth", a.model), ("hf", a.hf_model)]:
        print(f"\n===== {eng} =====")
        r = measure(eng, mid, a.rows, a.steps, a.seq, a.modules, a.packing)
        print(json.dumps({k: v for k, v in r.items() if k != "loss_curve"}, indent=2))
        results.append(r)

    u, h = results[0], results[1]
    print("\n===== RATIOS (Unsloth / HF) =====")
    print(f"  time        : {h['train_time_s']/u['train_time_s']:.2f}x faster")
    print(f"  sec/step    : {h['sec_per_step']/u['sec_per_step']:.2f}x faster")
    print(f"  peak VRAM   : {h['peak_vram_gb']/u['peak_vram_gb']:.2f}x less"
          f"  ({h['peak_vram_gb']:.2f} GB -> {u['peak_vram_gb']:.2f} GB)")
    print(f"  dtype parity: HF={h['compute_dtype']}  unsloth={u['compute_dtype']}"
          f"  {'OK' if h['compute_dtype']==u['compute_dtype'] else '<<< MISMATCH: fix the baseline'}")
    print(f"  params parity: HF={h['trainable_params']:,}  unsloth={u['trainable_params']:,}"
          f"  {'OK' if h['trainable_params']==u['trainable_params'] else '<<< MISMATCH: not the same model'}")
```

**The five rules this harness enforces, and why each one is necessary:**

| Rule | Why | What breaks without it |
|---|---|---|
| **Same trainable parameter count in both arms** | If the HF arm trains `q,v` and the Unsloth arm trains all seven, you measured the config, not the engine | The comparison is meaningless. This is the repo's own bug (§4.7.3). |
| **Same `compute_dtype` in both arms** | FP32 4-bit compute vs BF16 is a 1.7–8× difference on its own | You attribute a dtype bug to kernels (§4.6.3) |
| **Warm-up steps excluded, and `torch.cuda.synchronize()` before stopping the clock** | Triton JIT + cuBLAS autotune on step 1 is 5–20× a normal step; CUDA is async | You measure compilation and Python enqueue time |
| **`reset_peak_memory_stats()` *after* warm-up and immediately before `train()`** | `max_memory_reserved` is a monotonic high-water mark | You report the load-time quantization spike as training VRAM |
| **Report `sec/step` and `tokens/sec`, not just wall-clock** | Wall-clock includes dataset mapping, tokenization and logging, which are identical and therefore dilute the ratio toward 1.0 | You understate or overstate depending on which side tokenizes slower |

> **Beyond the video:** the sixth rule, which almost nobody does and which is the only one that tests *correctness* rather than speed: **compare the two arms' adapter weights after N steps with the same seed.** They will not be bit-identical (fused kernels reassociate FP ops), but they should be *close*:

```python
# Adapter equivalence: the test that proves the manual backward is correct.
from safetensors.torch import load_file
import torch
u = load_file("bench_unsloth_adapter/adapter_model.safetensors")
h = load_file("bench_hf_adapter/adapter_model.safetensors")
assert set(u) == set(h), "different adapter key sets -> different target_modules"
worst = 0.0
for k in sorted(u):
    a, b = u[k].float(), h[k].float()
    # cosine similarity is scale-invariant and is the right metric for a
    # weight matrix whose overall scale can drift legitimately
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = (a - b).norm() / (b.norm() + 1e-9)
    worst = max(worst, abs(1 - cos))
    if abs(1 - cos) > 0.05:
        print(f"  DIVERGENT {k}: cos={cos:.4f} rel_l2={rel:.4f}")
print(f"worst 1-cos across all adapter tensors: {worst:.5f}")
# PASS: worst 1-cos < 0.02 after 100 steps -> the two engines are computing the
#       same thing and any remaining difference is FP reassociation.
# FAIL: worst 1-cos > 0.10 -> one of the engines has a real bug for this
#       architecture. Do NOT ship until you know which one.
# NOTE: adapter init is seeded (random_state=3407) but the *base* model's
#       kernels differ, so expect divergence to GROW with step count. Compare
#       at matched step counts, and re-seed identically in both arms.
```

### 12.2 Part B — did the fine-tune work?

Short version (CS-13 §12 has the full treatment). Four layers, cheapest first:

| Layer | Metric | Cost | What it catches |
|---|---|---|---|
| 1 | **Format compliance rate** — % of 200 generations that parse against your schema | minutes, free | The most common production failure |
| 2 | **Held-out eval loss** — 200–1,000 rows excluded from training | included in the run | Overfitting; the point at which to stop |
| 3 | **Task-specific automatic metric** — EM/F1/ROUGE/regex-validated JSON | an hour | Did it learn the task at all |
| 4 | **Pairwise LLM-as-judge** against the base, with position swap | a few dollars | Open-ended quality; also the easiest to fool |
| 5 | **Regression suite** — 50–200 frozen prompts asserting *properties* (parses as JSON, contains no refusal phrase, length within band) | minutes | Shipping a regression |

> **Beyond the video:** an efficiency module needs one extra eval that no quality module has: **the equivalence assertion.** If you adopted Unsloth *and* your quality dropped 3%, you cannot tell whether you changed the optimisation or changed the model. The fix is to keep one **golden reference**: a fixed 100-prompt suite, a fixed seed, and the *base model's* outputs saved to disk as JSON. Every time you change the training stack — Unsloth version, `transformers` version, `lora_dropout`, checkpointing mode — regenerate the adapter and diff its outputs against the reference. A 3% shift in win rate is a signal; a 3% shift in *exact* outputs is a bug. **You cannot do this without a frozen prompt set, and you cannot do it retroactively.**

### 12.3 The two claims to verify on every new GPU

```python
# verify_unsloth.py — run this ONCE per new GPU/driver/model combination.
# It answers "am I actually on the fast path, and by how much?" in 3 minutes.
import torch, time
from unsloth import FastLanguageModel

# 1. Are the fast modules actually installed?  (the silent-fallback check)
m, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=2048, load_in_4bit=True)
attn_mod  = type(m.model.layers[0].self_attn).__module__
rope_mod  = type(m.model.layers[0].self_attn.rotary_emb).__module__
print(f"attention impl module: {attn_mod}")
print(f"rotary impl module   : {rope_mod}")
assert "unsloth" in attn_mod, f"FAST PATH NOT ACTIVE — got {attn_mod!r}. Pick a supported arch."
assert "unsloth" in rope_mod, f"RoPE NOT FUSED — got {rope_mod!r}."

# 2. Is the compute dtype what you think it is?
print(f"compute dtype: {next(m.parameters()).dtype}  (expect float16 on T4, bfloat16 on A100+)")

# 3. Measure steady-state throughput at YOUR shape, not a toy shape.
from transformers import AutoTokenizer
m = FastLanguageModel.get_peft_model(m, r=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])
m.eval()
torch.cuda.synchronize()
for _ in range(3):                                          # warm-up
    ids = torch.randint(0, 1000, (1, 2048), device="cuda")
    with torch.no_grad(): m(input_ids=ids)
torch.cuda.synchronize()
t0 = time.time()
REPS = 20
for _ in range(REPS):
    ids = torch.randint(0, 1000, (1, 2048), device="cuda")
    with torch.no_grad(): m(input_ids=ids)
torch.cuda.synchronize()
dt = (time.time() - t0) / REPS
print(f"forward-only at T=2048, B=1: {dt*1000:.1f} ms/iter  ({2048/dt:,.0f} tok/s equivalent)")
# Compare against the reference band for your GPU class (S9.4). Outside the
# band by >1.5x means a config problem, not a hardware problem.
```

---

## 13. Comparison Tables

### 13.1 The nearest alternatives, head to head

| | **Unsloth** | **HF + TRL (hand-written)** | **LLaMA-Factory** (CS-15) | **Axolotl** (CS-17) | **torchtune** |
|---|---|---|---|---|---|
| **Interface** | Python, 2 functions | Python, everything | WebUI / YAML / CLI / Python | YAML + CLI | Python recipes |
| **Speed vs naive HF** | 2–4× (unconfigured baseline) | 1.0× (it *is* the baseline) | inherits HF, or Unsloth if configured as backend | inherits HF, or Unsloth as backend | ~1.2–2× on some paths (own recipes, FA2, compile) |
| **Speed vs *tuned* HF** | ~1.2–1.4× | 1.0× | ~1.0× | ~1.0× | ~1.0–1.4× |
| **VRAM vs tuned HF** | ~15–30% less | — | ~0% | ~0% | ~0–15% |
| **Multi-GPU / FSDP** | weak | full control (`accelerate`, FSDP, DeepSpeed) | good (accelerate-based) | **best-in-class** (FSDP/DeepSpeed presets) | good (FSDP2, tensor parallel) |
| **Architecture coverage** | curated list | **anything in `transformers`** | very broad (100+ templates) | broad (100+ configs) | curated but growing |
| **No-code UI** | none | none | **yes (LLaMA Board)** | partial | none |
| **Data format support** | whatever TRL takes | whatever you write | registry with 100+ presets | YAML dataset types | whatever you write |
| **RLHF/DPO/GRPO** | yes (its own fast paths) | TRL does it | yes (DPO/ORPO/KTO, no GRPO historically) | yes (DPO/ORPO/KTO/GRPO) | DPO/PPO recipes |
| **Quantization** | NF4 in-kernel; GGUF export | any bnb/awq/gptq | bnb, GPTQ, AWQ, LoRA+, DoRA, PiSSA, LoRA-GA | bnb, GPTQ, QLoRA | bnb, QAT |
| **Merge/export** | `save_pretrained_merged` (correct on 4-bit) | manual; **must merge in 16-bit** | `llamafactory-cli export` (correct) | `axolotl merge-lora` (correct) | `tune convert` |
| **Debugging** | hard (fused kernels) | **easy** (pure PyTorch) | medium | medium | medium |
| **Reproducibility** | version-brittle | best | good | good | good |
| **Best for** | one GPU, ≤14B, max speed | research, exotic models, full control | non-coders, many model families, fast setup | **multi-GPU production**, large models | teams that want readable recipes |

### 13.2 Unsloth vs its own host libraries, at the level of what you actually change

| You change… | Unsloth | Plain TRL | LLaMA-Factory | Axolotl |
|---|---|---|---|---|
| Model loading line | 1 call | 1 call + `BitsAndBytesConfig` | 1 YAML key | 1 YAML key |
| Adapter injection | 1 call | `LoraConfig` + `get_peft_model` | 5 YAML keys | 6 YAML keys |
| Trainer | `SFTTrainer` (unchanged) | `SFTTrainer` | `llamafactory-cli train` | `axolotl train` |
| Masking | `train_on_responses_only(...)` | `assistant_only_loss=True` | `train_on_prompt: false` | `train_on_inputs: false` |
| Merge | `save_pretrained_merged` | **manual, 16-bit only** | `export` CLI | `merge-lora` CLI |
| Lines of code to migrate | **6** | 0 | ~40 | ~35 |

**The migration cost is the reason Unsloth wins in practice**, and it is worth saying plainly: a 6-line change that yields a genuine 1.2–1.4× and 15–30% VRAM is a better engineering decision than a 2× win that costs a week of refactoring. Just do not book the 2×.

### 13.3 Quality comparison — the honest position

| Question | Answer | Evidence |
|---|---|---|
| Does Unsloth change model quality? | **No, in expectation.** It changes wall-clock and memory, not the objective. | Fused kernels reassociate FP ops only; §4.3.3's manual backward computes the same analytic gradient |
| Can it change quality *in practice*? | **Yes, in three ways.** (1) Numerical reassociation shifts the trajectory slightly. (2) It enables configurations (long context, larger batch) that change the optimum. (3) It changes nothing about your data or mask, which is where quality actually comes from. | §4.3.3's equivalence test is how you bound (1) |
| Is a Unsloth fine-tune "as good as" a full-precision one? | **Not a meaningful question.** The relevant comparison is 4-bit QLoRA vs 16-bit LoRA, which is a *quantization* question (CS-10, CS-11), not an Unsloth question. Unsloth can train both. | QLoRA's quality gap is small (0.1–1% on most benchmarks) and is Unsloth-independent |
| Does the speedup cost accuracy? | **Not measurably**, if the equivalence test passes. If it fails, you have a bug. | §12.1's adapter-cosine test |
| Should I A/B Unsloth vs non-Unsloth for quality? | **Yes, once**, on your own task, with a frozen prompt set. It costs one GPU-hour and settles the question for your project permanently. | §12.1 + CS-13 §12 |

### 13.4 Unsloth vs quantization-only approaches

An important confusion to resolve: Unsloth is **not** an alternative to GPTQ/AWQ/GGUF.

| Approach | What it reduces | Stage | Composes with Unsloth? |
|---|---|---|---|
| **QLoRA / NF4 (bnb)** | Weight memory *during training* | Training | **It is what Unsloth accelerates** |
| **GPTQ / AWQ** | Weight memory *at inference*, with calibration | Post-training | Yes — Unsloth exports a model you then quantize (CS-10) |
| **GGUF / llama.cpp** | Weight memory at inference on CPU/edge | Post-training | Yes — `save_pretrained_gguf` (CS-10) |
| **Unsloth** | Memory + time *during training and inference* | Training | **The accelerator for all of the above's training step** |
| **FlashAttention** | Attention activation memory at train and infer | Both | Integrated into Unsloth |
| **Packing** | Wasted compute on padding | Training | Set via TRL; Unsloth-compatible |
| **`adamw_8bit`** | Optimizer-state memory | Training | Set via TRL; Unsloth-compatible |

> **Beyond the video:** the interview question hiding in this table is *"Unsloth vs QLoRA — which should I use?"* The correct answer is that **the question is malformed**: QLoRA is a quantization method and Unsloth is an execution engine; Unsloth *runs* QLoRA. The well-formed question is *"should I train in 4-bit or 16-bit, and if 4-bit, should I use Unsloth?"* — and the answers are independent: 4-bit when memory-bound (almost always on rented or consumer GPUs), Unsloth when on a single supported-architecture GPU and you have not already hand-tuned the stack.

---

## 14. Debugging Playbook

### 14.1 The master table

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | `ImportError: cannot import name 'SFTConfig' from 'trl'` | `transformers`/`trl` version mismatch; Unsloth's pins violated by a later install | `pip list \| grep -E "transformers\|trl\|peft\|unsloth"` | Reinstall the exact pin set (§6.1); never `pip install -U` in a working env |
| 2 | `AssertionError: Please enable GPU runtime` | CPU-only torch (wrong `cuXXX` wheel) or no GPU attached | `torch.version.cuda`, `torch.cuda.is_available()` | Reinstall torch from the matching wheel index |
| 3 | OOM at load time | `max_seq_length` too high for the card, before any training | Watch `nvidia-smi` during `from_pretrained` | Halve `max_seq_length`; it configures RoPE and kernel selection, so lower it before loading |
| 4 | OOM in the first training step | No gradient checkpointing; batch too large; packing making one giant row | `torch.cuda.max_memory_allocated() / 1e9` after one batch; print the batch's shape | `use_gradient_checkpointing="unsloth"`, `per_device_train_batch_size=1`, `packing=False` to isolate |
| 5 | OOM only at a random later step | Packing produced one unusually long packed sequence; or fragmentation | Log the batch's token count per step | Cap with `max_seq_length`; reduce packing group size; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 6 | Run is ~1.5–2× slower than the reference band | `import unsloth` after `transformers`; or an unsupported architecture; or `lora_dropout > 0`; or FP32 compute dtype | §8.2's module check + `next(model.parameters()).dtype` + `peft_config` | All four checks in §12.3 |
| 7 | Run is 3× slower than the band, GPU at <30% util | **Dataloader-bound** | `nvidia-smi dmon`; time `next(iter(loader))` vs the step | `num_proc=8` on `map`; `dataloader_num_workers=4`; pre-tokenize to disk |
| 8 | First step takes 20 s, later steps 0.4 s | Triton JIT compile + autotune | Not a bug | Warm up; cache `~/.triton`; exclude step 1 from benchmarks |
| 9 | First step *always* takes 20 s, every run | Triton cache not persisted (ephemeral container) | `ls ~/.triton/cache` | Bake the warm-up into the image build |
| 10 | Loss is exactly `ln(vocab)` and never moves | Model output is uniform — usually a broken forward or wrong template | Decode a batch; check the template | §4.10 |
| 11 | Loss starts near 0.0 | Mask inverted: you are supervising the *input* | §4.9.3 supervised-decoding assertion | Fix the boundary; or `labels = input_ids` when it should be `-100` |
| 12 | Loss falls fast then plateaus high | `lr` too low (the video's `2e-5` case) or rank too low | Compare against a 3× higher LR run for 100 steps | `2e-4`; raise `r` |
| 13 | Loss spikes then recovers repeatedly | LR too high, or a pathological batch | Log per-step loss; find the step; inspect that batch | Halve LR; add `max_grad_norm=1.0`; filter the outlier row |
| 14 | Loss goes to NaN | bf16 overflow on a T4 (`bf16=True` without support); or `lr` far too high; or a corrupt row | `torch.cuda.is_bf16_supported()`; scan for empty/duplicate rows | `fp16=True` on pre-Ampere; lower LR; filter |
| 15 | Train loss falls, eval loss rises | Overfitting | The two curves on one chart | Fewer epochs; more data; lower `r`; add 5–10% general replay (CS-13 §4.9) |
| 16 | Train loss falls, eval loss is *flat* | The data has no learnable signal, or the mask supervises nothing | Print the supervised fraction | Check the mask; check that the responses differ from each other |
| 17 | Model answers but never stops | EOS not appended, or stripped by a cleaning step | Decode a training row | `+ tokenizer.eos_token` (§6.6) |
| 18 | Model repeats the prompt before answering | Unmasked prompt tokens | §4.9.3 | `train_on_responses_only` |
| 19 | Model ignores the system message | Base model's template has no system role, or you trained without one | `print(tokenizer.chat_template)` | Drop system messages or retrain with them |
| 20 | Output is coherent in the notebook, gibberish behind the API | Train/serve template mismatch | Diff the `chat_template` in the saved `tokenizer_config.json` vs the serving stack | Unify the template; save the tokenizer with the adapter |
| 21 | Merged model is worse than the adapter | Wrong merge on a 4-bit base | §6.10's logit-agreement test | `save_pretrained_merged(save_method="merged_16bit")` |
| 22 | `KeyError: 'text'` at trainer construction | Dataset has a `messages` column but `dataset_text_field="text"` was set (or vice versa) | `print(ds.column_names)` | Pick one schema (§6.8) |
| 23 | `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` | TRL ≥0.16 renamed it | — | `processing_class=tokenizer` |
| 24 | `RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'` | `bf16=True` on a pre-Ampere GPU | `torch.cuda.is_bf16_supported()` | `fp16=True`, `bf16=False` |
| 25 | Crash after a `torch` upgrade, in a Triton kernel | Stale Triton cache compiled against a previous ABI | Traceback ends in generated Triton code | `rm -rf ~/.triton/cache` |
| 26 | 401 / `Repository not found` on a gated model | Hub token missing or lacks the licence acceptance | `huggingface-cli whoami` | Add the token; accept the licence on the model page |
| 27 | Adapter saves as ~2 GB instead of ~50 MB | You saved the *model*, not the adapter — or you merged unintentionally | `ls -la lora_model/` | `adapter_model.safetensors` = adapter; `model.safetensors` = merged |
| 28 | Free Colab disconnects mid-run | Session/idle limits | — | `save_steps=200`, `save_total_limit=3`, resume |
| 29 | `ValueError: ... max_seq_length ... exceeds` | `SFTConfig.max_seq_length` disagrees with the loader's | Compare both values | Set it in one place and pass it to both |
| 30 | Everything works, but the model is worse than the base on general prompts | Catastrophic forgetting from over-training on narrow data | MMLU/GSM8K-style probe before and after | Fewer epochs; 5–20% general-data replay (CS-13 §4.9.5) |

### 14.2 Loss-curve decision table

| Curve shape | Most likely cause | Second most likely | Third | First action |
|---|---|---|---|---|
| Flat at `ln(V)`, never moves | Broken template / wrong tokenizer | Frozen everything (`requires_grad` all False) | LR effectively 0 | Decode a batch (§6.5); print `requires_grad` count |
| Flat, slightly below `ln(V)` | Mask supervises almost nothing | Dataset is one repeated example | — | Print the supervised fraction (§4.9.3) |
| Starts at 0.0–0.5 | Mask inverted | Labels == inputs with a copy task | — | Fix the boundary |
| Smooth exponential decay to a plateau | **Healthy** | — | — | None; stop when eval turns up |
| Decays then plateaus at a high value (e.g. 2.5) | LR too low | `r` too low | Not enough steps | 10× LR for 100 steps and compare |
| Sawtooth: spikes then recovers | LR at the edge of stability | One bad batch | Gradient clipping off | Halve LR; enable `max_grad_norm=1.0` |
| Monotonic rise | LR far too high, or labels are corrupted | Sign error in a custom loss | — | Kill the run; halve LR; inspect labels |
| NaN | bf16 on pre-Ampere; LR too high; corrupt row | Gradient explosion with no clipping | Division by zero in a custom metric | `fp16`, halve LR, clip |
| Train down / eval up | Overfitting | Eval set leaked into train (contamination) | Eval batch size 1 is noisy | Stop early; dedupe against eval (CS-13 §12.2) |
| Train down / eval flat | No learnable signal, or the eval measures a different distribution | Mask supervises the wrong span | — | Inspect the supervised span |
| Both flat after a promising start | LR schedule decayed to 0 too early | Warmup consumed the whole run | — | `warmup_ratio=0.03`; check `num_training_steps` |
| Loss is fine but generations are garbage | Template mismatch at serve | Wrong merge | Wrong `max_new_tokens`/stop tokens | §4.10; §6.10; `skip_special_tokens=False` to see the raw tokens |

### 14.3 The five-minute triage, in order

```text
1.  Is the GPU available and is the dtype right?
      torch.cuda.is_available(); next(model.parameters()).dtype
      -> pre-Ampere must be float16, not bfloat16

2.  Are the FAST MODULES actually installed?
      type(model.model.layers[0].self_attn).__module__ == "unsloth.models..."
      -> if it says "transformers", you have a 1.0x run. Stop here.

3.  Am I supervising anything, and only the right thing?
      (labels != -100).sum() and decode the supervised span
      -> 0 = mask never applied; == total = patch is a no-op

4.  Is anything training?
      sum(p.numel() for p in model.parameters() if p.requires_grad)
      -> should be 0.1-5% of total and > 0

5.  Is the loss in the right neighbourhood at step 0?
      ln(vocab) x [0.7, 1.5]
      -> 0.3 = mask inverted; 25 = template scrambled

6.  Only now: is it fast enough?
      tokens/sec vs the reference band; nvidia-smi utilization
      -> <30% util = dataloader, not kernels
```

Steps 1–5 take under a minute combined and catch the overwhelming majority of real failures. Step 6 is where everyone starts, and it is the only step Unsloth is actually responsible for.

---

## 15. Applied Case Studies

### 15.1 The startup with one 4090 and a 7B

**Situation.** A four-person startup has one RTX 4090 (24 GB) in a workstation and $0 GPU budget. They need a domain-tuned assistant over their internal API documentation. They have 6,200 curated Q&A rows (mean 340 tokens, p99 1,450) and 140 hand-written eval prompts.

**Why Unsloth.** Nothing else fits. A 7B in 16-bit is 14 GB of weights plus activations plus optimizer state — over 24 GB before you add a single activation. 4-bit weights bring it to ~4.3 GB and Unsloth's kernels leave room for a real batch at a real sequence length.

**Exact config.**

```python
BASE = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # Qwen has clean template hygiene
model, tok = FastLanguageModel.from_pretrained(
    BASE, max_seq_length=2048, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                       # 6.2k rows -> r=16, NOT 32
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth",       # the 4090 needs it at T=2048
    random_state=3407,
)
# Data: messages column + the model's own template (SS6.8), 5% eval split
args = SFTConfig(
    per_device_train_batch_size=2, gradient_accumulation_steps=8,   # eff. 16
    num_train_epochs=2, learning_rate=2e-4, warmup_ratio=0.03,
    lr_scheduler_type="cosine", weight_decay=0.01, max_grad_norm=1.0,
    max_seq_length=2048, packing=True, optim="adamw_8bit",
    eval_strategy="steps", eval_steps=50, save_steps=100, save_total_limit=2,
    bf16=True, seed=3407, output_dir="out/qwen-api-v1", report_to="wandb",
)
trainer = train_on_responses_only(trainer, instruction_part=..., response_part=...)
```

**Result (worked estimate).** 6,200 rows × 2 epochs / 16 = 775 steps. At `T=2048`, batch 2 on a 4090, a 4-bit 7B step is ~1.6–2.2 s → **~25 min per epoch, ~50 min total**. Peak VRAM ≈ **9.5 GB**. Cost: ~1.2 kWh, i.e. under $0.20 of electricity. Adapter size: `r=16` on 7 modules of Qwen2.5-7B ≈ 20 M params × 2 B ≈ **40 MB**.

**What went wrong first.** Three things, in the order they hit:

1. **`lr=2e-4` on an *Instruct* base destroyed its instruction-following** for the first 200 steps — the model became a documentation-shaped text generator that ignored the question. Fix: `lr=1e-4` and `warmup_ratio=0.05` for second-stage SFT on an instruct model.
2. **`packing=True` with a flat-text dataset and hand-written markers** produced a mask that was applied to only the first sequence in each pack. Detected by the supervised-fraction assertion showing 41% instead of the expected ~25% with wild per-pack variance. Fix: `messages` column + `assistant_only_loss=True`.
3. **`max_seq_length=4096` "because it fits"** — it did fit, at batch 1, and the run was 3× slower per token than at 2048 with batch 4. The p99 was 1,450. Fix: `max_seq_length=2048`, batch 4.

**Transferable lesson:** the 4090 was never the bottleneck. The three fixes above were a *config* win of roughly 3× — larger than anything Unsloth contributed on top of a tuned baseline.

---

### 15.2 The enterprise on 4×A100-80GB with a 70B

**Situation.** A regulated financial-services firm needs a 70B fine-tuned on 180,000 internal compliance Q&A pairs. They have a 4×A100-80GB node on-prem. Hard requirement: every training run must be reproducible from a config file for audit.

**Why Unsloth is the *wrong* first choice.** The node is a multi-GPU box; the model is 70B; the audit requirement demands reproducibility. All three point away from Unsloth (§8.2). FSDP across four A100s with Unsloth's manual backward and patched modules is not a supported configuration, and the version brittleness is an audit liability.

**What they should do instead.** **Axolotl** (CS-17) with an FSDP config, YAML-pinned, `transformers`/`trl`/`peft` versions in the YAML header, checkpoint sharding, and `torch.use_deterministic_algorithms(True)` where it does not conflict with FA2.

**Where Unsloth *does* enter.** The **DPO stage afterwards**, on a single A100, at `r=16` on 7 modules with the 4-bit base — because DPO holds two copies of the model plus the preference pairs, and that is exactly the memory-bound regime where Unsloth's 15–30% VRAM saving is real and where FSDP is overkill.

**The transferable lesson:** the right architecture for a 70B multi-GPU SFT is *not* the right architecture for the DPO stage that follows it, and mixing tools per stage is normal. Unsloth is a **single-GPU specialist**, and a team with a node should be using it for 20% of its stages, not 100%.

---

### 15.3 The Kaggle competitor with no budget

**Situation.** A solo competitor has Kaggle's free P100 (16 GB) and 30 GB/week quota. They want to fine-tune a 3B for a domain QA competition, and they need dozens of iterations.

**Why Unsloth.** The 30-hour weekly quota is the binding constraint, and it is a *time* quota. A 2× speedup is exactly a 2× increase in the number of experiments. Also: the P100 is **Pascal — no bf16, and FlashAttention-2 does not support it.** So the configuration must be FP16 + `sdpa` or xFormers, and the expected win is smaller than on an Ampere card.

**Exact config and its specific traps.**

```python
# P100 specifics — these three lines are the whole trick on Pascal.
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=1024,      # Pascal has no FA2; keep T modest
    dtype=torch.float16,      # NOT None — Pascal has no bf16
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=[...7...], lora_alpha=16, lora_dropout=0.0,
    use_gradient_checkpointing="unsloth", random_state=3407)
args = SFTConfig(bf16=False, fp16=True, ...)   # MUST match the loader
```

**Result (worked estimate).** 3B 4-bit, `T=1024`, batch 2 on a P100: ~4.5–6 GB peak, ~900–1,300 tokens/s. That is enough for ~15 experiments in a 30-hour week.

**What went wrong first.** `bf16=True` was copied from a tutorial and produced `RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'` on step 1 — the P100 has no bf16. Second mistake: `max_seq_length=4096`, which OOM'd at load because the Pascal attention path is the slow, materialising one. Third: benchmarking against a 4090 result and concluding "Unsloth doesn't work."

**Transferable lesson:** **the hardware generation changes the expected speedup more than the library choice does.** Unsloth's headline numbers come from Ampere-or-newer cards with FlashAttention-2. On Pascal or Turing, the same code gives a real but smaller win, and comparing your P100 number to a published A100 number is not a comparison.

---

### 15.4 The team shipping a support assistant, where the model matters more than the clock

**Situation.** A SaaS company has 12,000 support conversations and a 7B in production behind vLLM. They want a fine-tune that reduces escalation rate. They have a spare A100 for 6 hours a week. Model quality is the KPI; training speed is irrelevant.

**Why Unsloth here is a *convenience*, not a necessity.** At 7B with 6 hours available, a hand-tuned HF+FA2+packing run finishes in ~3 hours. Unsloth takes it to ~2.2. The 0.8 hours saved is worth about $1.40.

**The actual value they get from Unsloth is a different one: the memory headroom buys a bigger batch, which buys a better run.** With 15–30% less VRAM they can run effective batch 64 instead of 32, which is a *quality* intervention, not a speed one.

**Exact config.** `r=32` (12k rows justifies it), `lr=2e-4`, cosine, 3 epochs, effective batch 64 (micro 4 × accum 16), `max_seq_length=2048`, `packing=True`, `packing` + `assistant_only_loss=True`, `eval_steps=100` with `load_best_model_at_end=True`, plus 8% general-instruction replay to fight forgetting.

**Result.** Eval loss minimum at step ~1,150 of 1,400 → the run stopped early and shipped the step-1,150 checkpoint. Adapter 80 MB, served with `vllm --enable-lora` alongside the base, so rollback is a config change, not a redeploy.

**What went wrong first.** They shipped the *last* checkpoint, not the best one, because `load_best_model_at_end` was off and `eval_strategy` was `"epoch"`. The final model was measurably worse than the step-1,150 one on their held-out set — 4 points of escalation reduction. **The speed optimisation was worth $1.40; the eval configuration was worth the entire project.**

---

### 15.5 The research team studying gradient dynamics

**Situation.** A lab is studying how LoRA gradients behave in the first 100 steps, and needs to instrument per-layer gradient norms and compare against an analytic prediction.

**Why Unsloth is wrong here — specifically.** The manual backward means the gradient tensors you would instrument are computed inside a fused kernel and, in the LoRA path, are not necessarily materialised as `param.grad` in the shape you expect at the time you expect. More fundamentally: **for research on the optimiser, you want the standard path, not the optimised one**, because the optimised one is the object of study in a different paper.

**What they should do.** Plain HF + peft, with `torch.autograd.set_detect_anomaly(True)` for the first 50 steps, hooks on `lora_A`/`lora_B`, and — crucially — **use Unsloth as the *reference implementation* to read**, not to run. `unsloth/kernels/fast_lora.py` is the cleanest available worked example of the analytic LoRA backward, and reading it is a legitimate use of the library.

**Transferable lesson:** a tool optimised for wall-clock is frequently the wrong tool for a question *about* wall-clock or about the thing it optimised away. Knowing which side of that line you are on is the skill.

---

## 16. Production Considerations

### 16.1 What the deliverable actually is

| Artefact | Size (7B, r=16) | Contains | Needs at load | Use |
|---|---|---|---|---|
| **Adapter** | ~40–80 MB | `adapter_model.safetensors`, `adapter_config.json`, tokenizer | base model + `peft` | **The source of truth.** Version-control this. |
| **Merged FP16** | ~15 GB | full weights + tokenizer + config | nothing | Single-artefact deploys, vLLM/TGI/sglang |
| **Merged 4-bit** | ~4 GB | requantized full weights | nothing | Size-sensitive deploys; **lossy twice** (§6.9) |
| **GGUF** | ~4 GB (q4_k_m) | llama.cpp format | llama.cpp / Ollama / LM Studio | CPU and edge |
| **Training config + data hash** | KB | the recipe | — | **Audit, rollback, reproduction** |

**Rule:** the adapter is the *source*; merged artefacts are *derived*. Never treat a merged model as the primary artefact — you cannot un-merge it, and the base quantization is baked in.

### 16.2 Versioning and lineage

```yaml
# model_card.yaml — commit this alongside the adapter. It is the answer to
# "which model is in production and how do I rebuild it?" six months from now.
model:
  id: acme-support-qwen7b-v3
  base: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
  base_revision: 7ae557604adf67be50417f59c2c2f167def9a775   # pin the SHA, not "main"
  adapter_sha256: 9f2c...                                   # of adapter_model.safetensors
  merged_16bit_sha256: 41ab...
training:
  rows: 12480
  data_fingerprint: sha256:8c1e...   # `datasets` fingerprint or a hash of the JSONL
  eval_split_fingerprint: sha256:77a3...
  epochs: 2
  steps: 1412
  effective_batch: 64
  learning_rate: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.03
  optim: adamw_8bit
  lora: {r: 32, alpha: 32, dropout: 0.0, targets: [q,k,v,o,gate,up,down]}
  max_seq_length: 2048
  packing: true
  seed: 3407
  masking: assistant_only_loss
environment:            # <-- THE PART EVERYONE FORGETS
  unsloth: "2025.3.19"
  transformers: "4.56.2"
  trl: "0.22.2"
  peft: "0.14.0"
  torch: "2.6.0+cu124"
  cuda: "12.4"
  gpu: "NVIDIA A100-SXM4-80GB"
  driver: "550.54.15"
eval:
  frozen_prompt_set: evals/regression-200.jsonl@sha256:...
  metrics: {format_compliance: 0.994, escalation_delta_pp: -4.1, winrate_vs_base: 0.63}
```

**Why the `environment:` block is non-optional.** Unsloth patches `transformers` internals. A `transformers` minor upgrade can change what a patch attaches to, silently changing the training trajectory. Without a recorded environment you cannot re-derive the model, and "we retrained it and got a different result" is an unanswerable audit finding.

> **Beyond the video:** the `datasets` fingerprint is not enough on its own. `load_dataset("json", data_files=...)` produces no fingerprint at all, and a `datasets` fingerprint does not capture the *order* of a shuffle with a different seed. The robust practice: hash the **final tokenized `input_ids` array** you actually trained on, and store that. It is one line and it settles every "did the data change?" argument forever.

```python
# The one-line data fingerprint worth recording.
import hashlib, numpy as np
h = hashlib.sha256()
for ex in trainer.train_dataset:                       # or iterate the packed arrays
    h.update(str(ex["input_ids"]).encode())
print("train_fingerprint:", h.hexdigest())
```

### 16.3 Serving

| Path | Setup | Notes |
|---|---|---|
| **vLLM + `--enable-lora`** | Serve the base once, mount adapters | **Best option.** One base in VRAM, N adapters, hot-swappable, easy rollback. Works because the adapter is a separate artefact. |
| **vLLM merged** | Serve `merged_16bit` | Simplest; one process per model; rollback = restart |
| **TGI** | `--model-id merged_16bit` | Mature; good batching |
| **sglang** | `--lora-paths` support | Fast, good LoRA story |
| **llama.cpp / Ollama** | GGUF from `save_pretrained_gguf` | CPU/edge; quantization error is a second lossy step |
| **Unsloth `for_inference` in a notebook** | Direct `model.generate` | **Never production.** No batching, no continuous batching, no KV-cache sharing, no concurrency. |

**Latency expectation:** a fine-tuned 7B adapter adds **zero** per-token latency versus the base when served with vLLM's LoRA path (the adapter is fused into the matmul), and ~2–5% when served with adapter switching per request. There is no "fine-tuning tax" at inference — that is the whole point of LoRA (CS-23).

### 16.4 Monitoring, drift, regression

| Signal | What to log | Alarm condition |
|---|---|---|
| **Format compliance rate** | % of responses parsing your schema, per hour | < 99% (was 99.4% at ship) |
| **Refusal rate** | % of in-domain requests declined | > 2× the ship-time baseline |
| **Response length distribution** | p50/p95 tokens | p95 drifts > 30% — a classic over-training/fine-tune-drift signature |
| **Escalation / handoff rate** | business KPI | Any regression |
| **Input distribution drift** | embedding-space drift of incoming requests (CS-04) | Significant shift → the fine-tune may be out of domain |
| **Model identity** | adapter hash in every response log | Any change not from a deploy |

**Regression suite (the minimum viable version):**

```python
# regression.py — 200 frozen prompts, run on EVERY candidate. Assert properties,
# never exact strings (the model is non-deterministic and you will churn).
CASES = [
    {"prompt": "...", "must_parse_json": True, "required_keys": ["intent","priority"]},
    {"prompt": "...", "must_not_contain": ["I cannot", "As an AI", "I'm sorry, but"], "max_tokens": 300},
    {"prompt": "...", "must_contain_any": ["refund", "return", "exchange"]},
]
# Run with do_sample=False so failures are reproducible; report a pass rate,
# and fail the build below 0.97 rather than at 1.00 (which never holds).
```

### 16.5 Rollback

Because the adapter is 40–80 MB and the base is unchanged, **rollback in a vLLM LoRA deployment is a pointer change**, not a redeploy: point `--lora-modules` at the previous adapter directory and reload. Keep the last three adapters (a few hundred MB). This is a genuine operational advantage of the adapter approach and it is the strongest argument for keeping the adapter as the source of truth (§16.1).

For merged-model deployments, rollback is a full model reload: 30–120 s of downtime on an 8B. Budget for it, or use the adapter path.

### 16.6 Compliance, licensing, and the Unsloth-specific angles

| Concern | Detail |
|---|---|
| **Base model licence** | The Unsloth `-bnb-4bit` re-upload carries the *base model's* licence, not a new one. Llama-3.x has the Llama Community Licence and an acceptable-use policy; Qwen2.5 is Apache-2.0; Gemma has its own. Check before commercial use. |
| **Unsloth's own licence** | Apache-2.0 for the code. That is not the model's licence. |
| **Training-data provenance** | The video's dataset (`yahma/alpaca-cleaned`) derives from Alpaca, which derives from `text-davinci-003` outputs — the OpenAI-ToS question CS-13 §4.8 covers. **Not for commercial training without a licence review.** |
| **Reproducibility for audit** | Fused kernels + autotuned tiles mean **bit-exact reproduction across hardware is not achievable**. If your regulator requires bit-exactness, you cannot use Unsloth for that run; you must use deterministic plain PyTorch. |
| **Data residency** | Unsloth runs locally; nothing leaves the machine except the Hub downloads. That is a *better* compliance story than any hosted fine-tuning API (CS-18, CS-19). |
| **Model provenance** | Pin the base revision SHA. `unsloth/*-bnb-4bit` repos are community re-uploads; a re-upload can be re-pushed. |

> **Beyond the video:** the reproducibility limitation is the one that surprises people. It is not a bug and Unsloth cannot fix it — it is inherent to fused kernels with a hardware autotuner. Practical guidance: use Unsloth to *find* the configuration (fast, many experiments), then, if you need a bit-exact audited artefact, **re-run the winning configuration through deterministic plain PyTorch on the audit hardware**. The Unsloth run is your search; the deterministic run is your record. That costs one extra run and satisfies both constraints.

---

## 17. Common Misconceptions

### 17.1 "Unsloth is 2× faster, full stop."

**Believe:** the 2× in the tagline is a property of Unsloth.
**Actually:** it is a property of Unsloth *versus a specific baseline* — the straightforward `transformers` + `peft` + `bitsandbytes` script with default settings. The video is explicit when it names the comparison: "standard Hugging Face training" [20:04], "the same standard PyTorch kernel" [13:04], "the normal Hugging Face, not Unsloth model… simple Hugging Face" [31:00]–[31:07]. Against a **well-tuned FA2 + QLoRA + packing + bf16-compute + 7-target-module** baseline the gap collapses (§4.7.2), to roughly 1.2–1.4× on time and 15–30% on VRAM.
**Because:** a large share of the headline win comes from configuration choices Unsloth applies automatically that you can also apply yourself. Unsloth's *distinctive* contribution is the kernel and graph work, and that part is real but smaller than the tagline implies. See the README's own framing [04:29]: "2× faster with up to 70% less GPU memory" — "up to" is doing a lot of work.

### 17.2 "Unsloth is a training framework."

**Believe:** Unsloth replaces TRL or Axolotl.
**Actually:** Unsloth is a **model-loading and kernel-patching library** that sits under `transformers`, `peft`, and `trl`, and you still call `SFTTrainer` / `DPOTrainer` / `GRPOTrainer` for the actual training loop. The video states it plainly: "basically this Unsloth is built on top of these three libraries: transformers, PEFT and TRL" [14:48]–[15:06]. The repo's own architecture diagram puts Unsloth at the top of a stack: CUDA → Triton → PyTorch → Hugging Face → **Unsloth** [15:47]–[18:33].
**Because:** the name and the marketing ("full guide", "fine-tune LLMs faster") suggest an end-to-end product. The correct mental model is a *drop-in accelerator*: change two lines (`FastLanguageModel.from_pretrained` and `get_peft_model`) and everything else in your TRL script is unchanged.

### 17.3 "Unsloth doesn't do the backward pass."

**Believe:** "Unsloth loss, backward, optimizer, epochs handle nahi karta" — a claim made in a markdown cell of the companion notebook (cell 35).
**Actually:** Unsloth implements the **entire** backward pass for the LoRA graph in hand-written CUDA/Triton, and it is the single most important thing it does. The video: "they have written the backpropagation manually… not the PyTorch autograd" [25:15]–[25:34].
**Because:** what Unsloth does *not* own is the **training loop** — the epochs, the optimizer step, the scheduler, the logging. Those come from TRL/HF `Trainer`. It also ships its own fused optimizer kernels for some paths, but the loop is TRL's. So the notebook note is half-right about the loop and dead wrong about the backward.
> **Correction:** the notebook cell 35 text ("Unsloth loss, backward, optimizer, epochs handle nahi karta") is wrong about loss and backward, and the video contradicts it directly — [25:15]–[25:34] describes Unsloth's manual backpropagation, and [32:09]–[32:22] describes the fused cross-entropy loss kernel. Unsloth provides both a fused loss kernel and a custom backward for the LoRA graph. It does not provide the epoch loop or the optimizer step; those come from TRL (`SFTTrainer`) and `torch.optim`. Treat the notebook comment as a transcription artefact, not a specification.

### 17.4 "`load_in_4bit=True` means the model trains in 4-bit."

**Believe:** the whole computation happens at 4-bit precision, hence the memory savings.
**Actually:** the weights are *stored* 4-bit (NF4), but every matmul **dequantizes to the compute dtype** (bf16/fp16) before the tensor cores, then the output is computed at that dtype. The video's phrasing — "dequantize, then compute, then again quantize" — describes exactly the round trip Unsloth tries to avoid [~23:00].
**Because:** 4-bit tensor cores that consume NF4 directly do not exist in a form that helps here. So the 4-bit win is a **memory-bandwidth and capacity** win, not a compute win. This matters for expectations: 4-bit does not make your matmuls 4× faster; it makes your weights 3.5× smaller (§4.4).

### 17.5 "Unsloth makes training more accurate, or at least equal."

**Believe:** it is a free speedup with no quality cost.
**Actually:** it *should* be numerically equivalent — the video insists "exact math, no approximation" [32:09]–[32:22] — and the kernels implement the same functions. But "exact math" means *mathematically the same operations*, not *bit-identical results*: fused kernels reassociate floating-point sums, Triton autotunes tile sizes per GPU, and the analytic LoRA backward can differ from autograd's in the last few ULPs. Over 1,000+ steps these differences compound into a **different model**, not a worse one.
**Because:** floating-point addition is not associative. Practically: expect a training-loss curve that differs in detail and a final model that is equivalent in quality but not identical — so your **eval harness must tolerate run-to-run variance**, which you should have anyway (§12).

### 17.6 "2× faster means half the wall-clock cost."

**Believe:** a 10-hour run becomes 5.
**Actually:** the speedup applies to the **training step**, and Amdahl's law does the rest. A real run is `total = load + tokenize + (steps × step_time) + eval + save + merge`. On a small dataset the fixed costs dominate and the observed speedup is much lower than 2× — on the Alpaca-cleaned 51k run with `max_steps=60` in the companion notebook, most of the wall-clock is dataset download, tokenization, and the first CUDA kernel compile.
**Because:** the video's own 10 h → 5 h claim [19:53]–[20:06] is for a *large* run where the step loop dominates. Reproduce that ratio only if your step loop is the bottleneck — which on a 500-example run it is not.

### 17.7 "Packing is free speed."

**Believe:** `packing=True` just concatenates and it is a pure win.
**Actually:** packing changes what the loss is computed over. If your examples have varied lengths and you pack with `dataset_text_field="text"`, you get boundary-crossing sequences and (if you do not mask) cross-example attention. The video notes auto-packing at [25:34]–[26:13] and 300k-token handling at [28:29], but never discusses the masking interaction.
**Because:** the video's pipeline uses flat text and `packing=True` together — perfectly fine on a *pretraining-style* dataset where every token is supervised, and quietly wrong on a chat dataset where only the assistant turns should be. See §6.6.5 and §10.4.

### 17.8 "Bigger `max_seq_length` is always safer."

**Believe:** set it to the model's maximum, you can always lower it later.
**Actually:** `max_seq_length` is the biggest VRAM lever you have, and it multiplies *everything* — activations, the attention working set, the logits tensor, and the number of padding/packed tokens you pay for. The video's own table shows the cliff: 8 GB → 3k tokens, 12 GB → 21k, 16 GB → 40k, 24 GB → 78k, 80 GB → 340k [29:14]–[30:18]. Set it to your **p99 training-sequence length**, not the model maximum.
**Because:** the model's advertised context (the video mentions a "28k maximum limit" for HF and 300k trained tokens [28:29]–[29:05]) is an *inference* capability, not a training budget. Training at 8k when your data is 600 tokens long wastes memory and gives you nothing.

### 17.9 "`lora_alpha` should be much larger than `r`."

**Believe:** the folklore `alpha = 2r` (or `4r`).
**Actually:** with Unsloth's defaults and the video's configuration, `lora_alpha` **equals** `r` — the companion notebook sets `r=32, lora_alpha=32` (cell 11). That is a scaling of 1.0, the conservative choice, and it is what Unsloth's own examples use.
**Because:** `alpha/r` scales the adapter's contribution. Raising it is a *learning-rate multiplier on the LoRA branch only* (§7.2), and doing that while also setting a high LR (2e-4) is how people blow up runs. Start at `alpha = r`; if the adapter is under-fitting after a few hundred steps, raise alpha rather than LR — it is the more surgical knob.

### 17.10 "Unsloth works with any model on the Hub."

**Believe:** it is a universal accelerator.
**Actually:** Unsloth is **architecture-specific**. It patches named modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `lm_head`, `mlp`, `self_attn`) that exist under those *exact names*. A model outside the supported list falls back to un-accelerated PyTorch (or refuses to load) and you lose the point of using it.
**Because:** a fused kernel is written against a specific computation graph. Support coverage is broad for Llama/Mistral/Qwen/Gemma/Phi family models, and thinner outside them. Check the supported-models table *before* committing to a base model (§10.2). The video's "1,150 models under the Hugging Face Unsloth org" [11:40] is a count of pre-quantized *re-uploads*, not a count of supported architectures.

### 17.11 "The pre-quantized `-bnb-4bit` uploads are just a convenience."

**Believe:** they save a download step.
**Actually:** they save the **on-the-fly quantization of the base model at load time**, which on a 70B is several minutes and a transient spike of full-precision VRAM. And they are the artefact the memory numbers in §11 are measured on.
**Because:** `from_pretrained(load_in_4bit=True)` on an FP16 checkpoint must download FP16 (2 bytes/param), load it, and quantize — peak memory during load is well above steady-state training memory. Using the `-bnb-4bit` repo means you download 4-bit and never hold the FP16 copy.

### 17.12 "You should merge the model to save inference time."

**Believe:** merge = faster inference.
**Actually:** merging **does not change inference FLOPs at all**; the adapter is a rank-`r` additive update that either costs `2·r·d` extra per token or is folded into `W` for free. Merging changes *deployment complexity*, not speed. And merging a 4-bit-loaded model is actively harmful (§6.9).
**Because:** `W + BA` has the same shape as `W`. The only reason to merge is if your serving stack cannot handle adapters. If it can (vLLM, TGI, sglang all can), keep the adapter unmerged — you keep rollback, multi-adapter serving, and a 40 MB artefact.

### 17.13 "`use_gradient_checkpointing=True` and `"unsloth"` are the same thing."

**Believe:** the string is a typo or an alias.
**Actually:** they are different code paths. `True` gives standard HF gradient checkpointing; `"unsloth"` gives Unsloth's smarter variant, which the video describes as keeping more in memory and recomputing less [23:43].
**Because:** checkpointing trades compute for memory; Unsloth's version trades *less* compute for the same memory by being selective about what it recomputes. The video's claim is that this lets you fit longer sequences. If you pass `True` you leave that on the table.

### 17.14 "Faster training means the model learns the same thing, so I can skip evaluation."

**Believe:** the optimization is orthogonal to quality, so existing eval practice suffices.
**Actually:** the optimization changes **what fits**, and what fits changes **what you should train**. Once 4× longer sequences fit, the right experiment is a different one (longer-context data, more epochs, larger effective batch). Teams that adopt Unsloth and change nothing about their eval plan get the same model faster and conclude nothing happened.
**Because:** the actual win from a memory optimization is usually an *experiment you could not previously afford* — a larger batch, a longer context, a bigger `r`, a real hyperparameter sweep. §15.1 and §15.4 both turn on this: the memory headroom was worth more than the speed.

---

## 18. Key Takeaways

1. **Unsloth is a kernel and graph-rewrite layer, not a trainer.** It sits under `transformers` + `peft` + `trl`; you still call `SFTTrainer`. Two lines change (`FastLanguageModel.from_pretrained`, `FastLanguageModel.get_peft_model`) and the rest of your script is untouched. [14:48]–[15:06]

2. **Every speed and VRAM number has an implicit baseline, and it is a naive one.** The tagline's "2× faster, 70% less memory" is measured against default `transformers` + `peft` + `bitsandbytes` with `q_proj`/`v_proj` only, fp32/auto compute dtype, no packing, no FA2. Against a tuned FA2 + QLoRA + packing + 7-module baseline the honest figure is **~1.2–1.4× on time and 15–30% on VRAM**. [13:04], [20:04], [31:00]–[31:07]

3. **The kernel suite is the real product.** Fused RMSNorm, fused RoPE, fused SwiGLU/GeGLU, fused cross-entropy with vocabulary chunking, fused LoRA matmuls, and a hand-written `torch.autograd.Function` for the LoRA graph that returns `None` for `W_4bit` because the frozen base needs no gradient. [25:15]–[25:34], [32:09]–[32:22]

4. **The single largest memory win is not materialising the attention matrix.** FlashAttention-2 makes attention IO-aware and removes the `O(T²)` HBM write/read; Unsloth ships it and turns it on by default. This is the piece that changes the *scaling* of memory with sequence length, from quadratic to linear.

5. **The second largest is not materialising the `[B,T,V]` logits.** At `T=4096, B=2, V=32k` that tensor is **524 MB in fp16**; at `V=152k` it is **2.49 GB**. The fused cross-entropy kernel chunks the vocabulary and never writes it. (§4.4)

6. **4-bit does not make compute faster; it makes weights smaller.** NF4 weights are dequantized to bf16 before every tensor-core matmul. The video's "dequantize → compute → requantize" round trip is the cost Unsloth's fused path avoids by keeping dequantization in registers. (§4.5)

7. **`lora_dropout` must be 0.0 to hit the fast path.** Any non-zero value silently falls back to a slower generic path. Set `lora_dropout=0.0` and mean it. (§6.4)

8. **Seven target modules is the default for a reason.** `q,k,v,o,gate,up,down`. Restricting to `q_proj, v_proj` (which the companion repo's own HF baseline does) trains ~3–4× fewer parameters and produces a materially different — usually worse — model. If you compare against a baseline, use the same target set. (§6.4, §12.6)

9. **`max_seq_length` is your biggest VRAM lever after the base model size.** Set it to your p99 training length. The video's own table: 8 GB → 3k, 12 GB → 21k, 16 GB → 40k, 24 GB → 78k, 80 GB → 340k tokens. [29:14]–[30:18]

10. **`use_gradient_checkpointing="unsloth"` is a string, not a boolean.** Passing `True` costs you ~20–30% throughput for the same memory. (§7.4)

11. **Never `merge_and_unload()` a model loaded with `load_in_4bit=True`.** PEFT dequantizes the NF4 weights and scales the new FP16 weights by NF4's double-quantization scales; the resulting model has degraded quality in a way that is hard to detect before deployment. Use `save_pretrained_merged(..., save_method="merged_16bit")` or `save_method="merged_4bit_forced"` and understand the trade. (§6.9)

12. **`train_on_responses_only` masks everything before the assistant turn by string search, not by token accounting.** Unsloth finds `full.find(response_part)`, re-tokenizes the prefix, and sets `labels[:prefix_len] = -100`. If your chat template wraps the assistant text in something that also appears in the user turn, the mask lands in the wrong place and trains on the prompt — silently. Always print the decoded supervised span for 3 examples before you launch. (§6.6)

13. **Chat-template mismatch between training and serving is the #1 silent quality killer.** If you train with a hand-rolled flat Alpaca string and serve with `tokenizer.apply_chat_template`, the model sees a prompt distribution it has never seen. The video's pipeline uses flat text; modern bases are instruct-tuned on their own template. Use the tokenizer's template for both. (§10.1)

14. **Unsloth is single-GPU first.** No FSDP, no tensor parallel, no pipeline parallel. If your plan requires multi-node or even multi-GPU training of a 70B, the right answer is Axolotl (CS-17) or plain `accelerate`, not Unsloth. [31:29]–[32:56] lists many features but not multi-GPU scaling.

15. **The memory win is worth more than the speed win, because it changes what you can try.** A 3× memory reduction at fixed quality is a bigger deal than a 2× speedup: it turns a 7B/2k/bs-2 experiment into a 7B/4k/bs-8 experiment. Budget the headroom for a *better* run, not a faster one.

16. **Benchmark with the same scope and the same config, or do not benchmark at all.** The companion repo's own comparison times `from_pretrained` inside one arm and outside the other, and sets 2 target modules in the baseline against 4 in the Unsloth arm. Warm up, fix the step count, measure steady-state step time, and report the config diff alongside the ratio. (§12.6, §4.7.2)

17. **Record the environment.** Unsloth patches library internals; an unreproducible run is an unauditable run. Pin `unsloth`, `transformers`, `trl`, `peft`, `torch`, CUDA, and the driver, and store the base-model revision SHA. (§16.2)

---

## 19. Self-Check Questions

1. **The Unsloth README says "2× faster with up to 70% less GPU memory." State, precisely, what the baseline is — and what the number becomes against a well-tuned FA2 + QLoRA run.**

2. **Name five things Unsloth fuses or eliminates, and for each, say which memory term it removes from the activation/gradient budget.**

3. **Why does Unsloth hand-write the backward pass for the LoRA graph instead of using `torch.autograd`? What does it return as the gradient for the frozen 4-bit weight?**

4. **At `B=2, T=4096, V=32,000`, how much memory does the logits tensor consume in fp16, and what does the fused cross-entropy kernel do instead? What is the number at `V=152,064` (Llama-3's vocabulary)?**

5. **Why must `lora_dropout` be `0.0`? What happens if you set `0.05`?**

6. **You loaded with `load_in_4bit=True` and call `merge_and_unload()`. What exactly goes wrong, and what should you call instead?**

7. **Explain how `train_on_responses_only` decides where to start computing loss. What is the failure mode when the chat template is not what Unsloth expects, and how do you detect it in under a minute?**

8. **You are training a 7B on 24 GB. Sketch the configuration: `max_seq_length`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `optim`, `use_gradient_checkpointing`, `r`, `lora_alpha`. Justify each choice against the VRAM arithmetic.**

9. **Your colleague's benchmark shows Unsloth at 2.8×. You reproduce it and get 1.3×. List four plausible causes and the diagnostic for each.**

10. **A team wants to fine-tune a 70B on 8×A100-80GB with a full audit trail and bit-exact reproducibility. Is Unsloth the right tool? What do you recommend instead, and where does Unsloth still fit in their pipeline?**

<details>
<summary><strong>Answers</strong> (click to expand)</summary>

**1.** The baseline is the straightforward `transformers` + `peft` + `bitsandbytes` script with default settings: 4-bit NF4 load, LoRA on `q_proj`/`v_proj` only, an auto-selected compute dtype (often fp32 on some hardware), no packing, no FlashAttention-2, no fused kernels, and `use_gradient_checkpointing=True` (standard). Against a **tuned** baseline — FA2 on, `bnb_4bit_compute_dtype=torch.bfloat16`, `packing=True`, 7 target modules, `use_gradient_checkpointing="unsloth"`, `optim="adamw_8bit"` — Unsloth's own contribution is roughly **1.2–1.4× on wall-clock step time and 15–30% on peak VRAM**. The difference between the two numbers is *configuration*, not kernels, and any honest benchmark must state which baseline it used. See §4.7.2 for the attribution table.

**2.** (a) **Attention** — fused FlashAttention-2 removes the `[B,H,T,T]` score/probability materialisation, i.e. the `O(T²)` term in the activation budget (at `T=4096, H=32, B=2` that is 2 GB per matrix in fp16, and FA2 writes none of it). (b) **Cross-entropy** — removes the `[B,T,V]` logits plus the softmax intermediate (524 MB + 524 MB at `V=32k`). (c) **RMSNorm** — one fused kernel instead of separate square/mean/rsqrt/mul passes, eliminating the fp32 intermediates of each. (d) **RoPE** — fused rotation, removing the `cos`/`sin` tables and the intermediate rotated tensors. (e) **SwiGLU/GeGLU MLP** — one kernel for `silu(xW_gate) * (xW_up)` instead of three, removing the two intermediates. (f) **LoRA backward** — the analytic backward avoids materialising the autograd graph over the frozen base, removing the saved-tensor bookkeeping for every frozen layer.

**3.** Autograd over a model whose base weights are 4-bit NF4 would require the dequantized weight to be a leaf with `requires_grad=True`, or a chain of ops from the quantized representation — which either doubles memory (holding bf16 master weights) or forces per-layer custom autograd anyway. Unsloth instead treats the LoRA path as a closed-form function `Y = XW_4bit + (α/r)·(X@A@B)` and writes the analytic gradients for `A`, `B`, and the input `X` directly in a `torch.autograd.Function`. The gradient with respect to the **frozen base weight is not computed at all** — `W_4bit` is returned as `None` from `backward`, so `torch` never allocates a `.grad` for it. That is a large saving: at 7B, a full weight gradient in bf16 would be 14 GB.

**4.** `B·T·V·2 bytes = 2 · 4096 · 32000 · 2 = 524,288,000 bytes ≈ **524 MB**`. In practice the backward needs the softmax probabilities too, so the naive path holds ~2× that, i.e. **~1.05 GB**. With the cross-entropy loss on a *packed* batch that also needs `logsumexp` and the gradient `(p − y)`, the naive implementation's overhead is why long-context runs OOM at small batch sizes. At Llama-3's `V = 152,064`: `2 · 4096 · 152064 · 2 = 2,491,414,528 ≈ **2.49 GB**` — per micro-batch. Unsloth's fused kernel **chunks over the vocabulary dimension**, computing the loss and its gradient in tiles that stay in SRAM/registers, and never materialises the full `[B,T,V]` tensor. (This is the same idea as `torch.nn.functional.cross_entropy` with the `chunked` FLCE path, and as Liger Kernel's `fused_linear_cross_entropy`.)

**5.** Unsloth's fused LoRA kernel is written for the deterministic path. With `lora_dropout > 0`, the adapter's input must be stochastically masked and rescaled per step, which the fused kernel does not implement; the library detects the non-zero dropout and **falls back to a generic (unfused) LoRA implementation** for the affected layers. Nothing errors, nothing warns loudly — you simply stop getting the fast path, which typically costs 15–35% of step time. This is a classic silent regression: it is why you should benchmark after changing any LoRA config, not just at the start. Note also that with the fast path the dropout-free configuration is not just faster but often *better* — for a small adapter trained on a modest dataset, dropout is rarely the binding regularizer compared with weight decay and early stopping.

**6.** `merge_and_unload()` folds `W + (α/r)·BA` into an FP16 weight tensor. When the base was loaded with `load_in_4bit=True`, `PEFT` must first dequantize the NF4 weights back to FP16 — but NF4 uses **double quantization**: the absmax values of each block are themselves quantized. The dequantized-then-merged result carries the reconstruction error of that two-level scheme, *and* the merge is performed in a dtype that differs from the dtype the LoRA layers were trained against. The result is a model that loads fine, generates fluent text, and is measurably worse on your eval — with no error anywhere. **Use `model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")`** instead: Unsloth dequantizes to bf16/fp16 first, in its own controlled path, and then merges. If you genuinely need a 4-bit artefact, `save_method="merged_4bit_forced"` exists and is documented as lossy. Or — best — don't merge at all; serve the adapter (§16.3).

**7.** Unsloth applies the tokenizer's chat template to your example, then, for the *response* half, it finds the boundary by searching the rendered conversation for the instruction and response parts: it locates `full.find(response_part)`, re-tokenizes the prefix up to that index, and sets `labels[:prefix_len] = -100` so no loss is computed there. Failures: (a) if `response_part` appears *inside* the instruction (e.g. "Answer the customer's question" is both the instruction and echoed in the answer), the search finds the wrong offset and masks too little or too much; (b) if the template's EOS delimiter differs from what Unsloth expects for that model family, the response part is never found, and the library either errors or masks nothing; (c) if `train_on_responses_only` is applied but the base model is a *base* (not instruct) model with no chat template, the rendered `full` string is not what the model was pretrained on. **Detect it in under a minute:** take `dataset[0]`, run the masking, then decode the tokens where `labels != -100` and print them. You should see *exactly* the assistant's reply, nothing before it. If you see the instruction, the system prompt, or the template's role tokens, stop and fix the template before launching. (§6.6)

**8.** A defensible starting point on 24 GB (RTX 4090 / L4-class; on a 40 GB A100 you can raise `max_seq_length` to 4096 and the batch to 4):

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = 2048,        # ~1.5 GB of KV/activation working set at bs=2
    dtype          = None,        # auto → bf16 on Ampere+, fp16 on Turing
    load_in_4bit   = True,        # weights ≈ 7e9 · 4.5/8 ≈ 3.9 GB
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",   # recompute activations, keep batch
    random_state=3407,
)
```

VRAM budget: weights 3.9 GB + LoRA params/grads/optimizer states ~0.3 GB (`r=16` ≈ 0.5% of params, 8-bit AdamW states) + activations with "unsloth" checkpointing ~4–6 GB at `T=2048, bs=2` + logits/fused-CE working set <1 GB + CUDA context/fragmentation ~1–2 GB ≈ **9–11 GB peak**, leaving comfortable headroom. Then `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` → effective batch 16, and `optim="adamw_8bit"` (the fp32 AdamW states for ~40 M LoRA params would be 320 MB rather than 80 MB — not fatal here, but free). `r=16, alpha=16` is the conservative starting pair; raise `r` to 32 only if the eval plateaus while train loss keeps falling.

**9.** (i) **Different baseline configuration** — their "HF baseline" likely used `q_proj`/`v_proj` only and default compute dtype. *Diagnostic:* print the full `LoraConfig` and `BitsAndBytesConfig` for both arms. (ii) **Different timed scope** — one arm times `from_pretrained` + tokenization + training, the other only training. *Diagnostic:* print the number of steps each arm actually ran and compare against wall-clock; if the arms ran different step counts, the ratio is meaningless. (iii) **Packing or FA2 enabled in one arm only** — a huge step-time difference that has nothing to do with Unsloth. *Diagnostic:* log `token_per_second` (not `step/second`) — packing changes tokens per step by 2–4×, so a step-time comparison is apples-to-oranges. (iv) **First-step compilation** — Triton autotunes on the first few steps; a short benchmark is dominated by it. *Diagnostic:* time steps 1–3 separately from steps 10–40, and report steady-state. (A fifth: different GPUs — a P100 has no bf16 and no FA2, so the Unsloth arm on a P100 is not comparable to a 4090 run.)

**10.** **No.** Three independent blockers: (a) **no multi-GPU** — Unsloth is single-GPU; a 70B on 8×A100 requires FSDP or tensor parallelism, which Unsloth does not provide; (b) **no bit-exact reproducibility** — fused kernels with hardware autotuning and reassociated floating-point sums cannot reproduce bit-identically across runs or hardware, which is incompatible with an audit requirement that says "bit-exact"; (c) **patch-fragility versus a long-lived audited environment** — Unsloth's monkey-patches pin you to a narrow `transformers` version window, which is awkward to freeze for years. **Recommend instead:** Axolotl (CS-17) with `fsdp_config` on the 8-GPU box for the main SFT, or plain `accelerate`/`torchrun` + `peft` if they need the audit trail to be fully legible. **Where Unsloth still fits:** the fast, cheap *search* phase — single-GPU experimentation on a 7B/8B proxy to find the dataset mix, the sequence length, `r`, and the LR, then port the winning configuration to Axolotl for the audited 70B run. Unsloth is a fast iteration instrument, not a compliance artefact. (§4.7.4, §8.1, §10.5, §16.6)

</details>

---

## 20. Cross-References

| Relationship | Module | Why |
|---|---|---|
| **Builds on** | **CS-06** — Hugging Face Ecosystem | Unsloth is an extension of `transformers` + `peft` + `trl`; nothing here makes sense without the `AutoModel`/`Trainer`/`datasets` mental model. |
| **Builds on** | **CS-13** — Instruction Fine-Tuning (SFT) | The dataset format, the chat template, the assistant-only loss masking, and the eval protocol all come from CS-13. Unsloth changes *how fast* that pipeline runs, not what it does. |
| **Builds on** | **CS-01 §4.9** — Data quality (dedup, filtering, contamination, licensing) | Packing, sequence-length distributions, and the fingerprinting practice in §16.2 are data-engineering concerns. (**CS-05 is *RNN/LSTM → Attention***, which is unrelated — there is no "Datasets & Data Preparation" case study) |
| **Builds on** | **CS-10 / CS-11** — Quantization | NF4, double quantization, blockwise absmax, and the dequantize→compute→requantize round trip are defined there; §4.5 here is their consequence in the training loop. |
| **Parallel to** | **CS-13 §6.8 & CS-11 §4.11** — the LoRA configuration and QLoRA | The adapter mechanics. Unsloth optimises the LoRA graph; those sections choose the knobs. Read them first if `dA`/`dB` in §4.3 look unfamiliar. (A dedicated "CS-23 LoRA/QLoRA" module is referenced elsewhere in this repo but was never written) |
| **Contrasts with** | **CS-15** — LLaMA-Factory | Both are "wrap a training stack and make it easy". LLaMA-Factory is config-file-driven and multi-method/multi-GPU; Unsloth is code-driven, single-GPU, and kernel-level. Different axes of the same problem. |
| **Contrasts with** | **CS-17** — Axolotl | The nearest alternative and the natural next step when Unsloth's single-GPU ceiling binds. Axolotl gives FSDP, YAML configs, and a broader method menu; Unsloth gives the kernels. §8.1 and §15.2 pick between them. |
| **Contrasts with** | **CS-18 / CS-19** — OpenAI / Vertex fine-tuning | Hosted APIs: zero ops, no kernel access, no data residency control, no adapter artefact. Unsloth: maximum control, all the ops. §16.6 is the compliance comparison. |
| **Needed by** | **CS-14** — Alignment (DPO/ORPO/GRPO) | Unsloth supports DPO/ORPO/KTO/GRPO trainers and the video lists GRPO/GSPO among the features [31:29]–[32:56]; the memory savings matter most in RL-style loops where you hold two models. |
| **Needed by** | **CS-21** — Multimodal Fine-Tuning | Vision-language fine-tuning is the regime where VRAM pressure is worst and Unsloth's per-layer savings matter most. |
| **Needed by** | **CS-04** — Evaluation & Experiment Tracking | The benchmarking discipline in §12 (warmup, matched step counts, tokens/sec not steps/sec, config diffing) is the general version of what §4.7.2 demands. |
| **See also** | **CH-16** — Unsloth Cheat Sheet | One-page config table, VRAM table, and the eight-line canonical script. |
| **See also** | **IQ-16** — Unsloth Interview Questions | The interview-facing version of §17 and §19. |
| **See also** | **AP-03** — GPU & Memory Arithmetic | The FLOP and bytes formulas used throughout §4.4 and §11. |

---

## Appendix A — Instructor's Verbatim Key Claims

Direct quotes from the transcript, with timestamps. Where the claim is imprecise, the correction is in the `> **Correction:**` callouts earlier in this module; where it is a marketing ratio, the baseline is stated here so the quote cannot be read in isolation.

| # | Timestamp | Verbatim | Baseline / caveat |
|---|---|---|---|
| A1 | [04:26]–[04:35] | "2x faster with up to 70% less GPU memory." | Read from the Unsloth README. Baseline = default `transformers` + `peft` + `bitsandbytes`. "Up to" refers to the best-case model/hardware combination. §4.7.2. |
| A2 | [13:04] | "faster than the same standard PyTorch kernel" | The comparison is a *standard* PyTorch kernel, i.e. unfused, eager or inductor-default. Not a tuned FA2 baseline. |
| A3 | [13:08]–[13:44] | The optimisations list: "pre-quantized models", "Triton kernel", "manual autograd", "auto packing". | Four items, in the instructor's own ordering. Note that "auto packing" is a *configuration* default, not a kernel. |
| A4 | [13:26]–[13:32] | "the same kernel basically have been used by ChatGPT also" | An appeal to authority, not a technical claim. Unsloth's kernels are not OpenAI's kernels; the claim is about the *technique class* (fused Triton kernels). |
| A5 | [14:48]–[15:06] | "basically this Unsloth is built on top of these three libraries: transformers, PEFT and TRL" | **The single most important sentence in the video for mental-model purposes.** Unsloth is a layer, not a replacement. |
| A6 | [15:00]–[15:06] | "2x, 3x faster and 50 to 80% better memory efficiency" | Note the drift from the README's "2× / 70%": the spoken version is wider (50–80%) and more aggressive (3×). Use the README figures as the defensible ones. |
| A7 | [15:47]–[18:33] | The stack diagram: CUDA → Triton → PyTorch → Hugging Face → Unsloth | The correct dependency ordering, and the reason Unsloth is a *patch layer*: it reaches down into Triton and up into HF. |
| A8 | [19:53]–[20:06] | The 10 h / 40 GB → 5 h / 12–16 GB claim | For a run the instructor describes generically; the baseline is "standard Hugging Face training" [20:04]. This is the video's clearest statement of the baseline. |
| A9 | [20:16]–[20:35] | Llama-3.1-8B: 20 GB → 7–8 GB, 2 h → 1 h | Same baseline. The VRAM reduction here is larger than the time reduction (2.5–2.9× vs 2×) — consistent with the memory-vs-time asymmetry this module argues throughout. |
| A10 | [21:24]–[22:10] | "CUDA is an NVIDIA C++ framework" | Correct in substance (CUDA C++); worth knowing that kernels can also be written in PTX/SASS, which almost nobody does. |
| A11 | [22:13]–[22:29] | Triton is "written by OpenAI" | Correct: Triton originated at OpenAI, is now widely adopted, and is the layer Unsloth writes its kernels in. |
| A12 | [22:49]–[23:30] | Fused attention + MLP kernels | The mechanism: one kernel instead of several, no intermediate tensors written to HBM. |
| A13 | [23:43] | "smart gradient checkpointing" | Unsloth's variant of activation recomputation, exposed as `use_gradient_checkpointing="unsloth"`. §7.4. |
| A14 | [24:29]–[24:45] | The FlashAttention README quote | The video reads FlashAttention's own description of IO-awareness — the mechanism behind the attention memory win. |
| A15 | [25:15]–[25:34] | "they have written the backpropagation manually… not the PyTorch autograd… one directed acyclic graph" | **The technical heart of the module.** Manual, analytic backprop over the LoRA graph. §4.3. |
| A16 | [25:34]–[26:13] | Auto sequence packing | Default-on in Unsloth's `SFTTrainer` path via a default collator. |
| A17 | [28:29] | "300k token training" | Used as the motivating scale for packing, not as a supported sequence length. Do not read this as a 300k context. |
| A18 | [29:00]–[29:05] | HF's "maximum limit is 28k token" | Imprecise — the limit is a property of the specific model and tokenizer, not of the Hugging Face library. |
| A19 | [29:14]–[30:18] | Context table: 8 GB → 3k, 12 GB → 21k, 16 GB → 40k, 24 GB → 78k, 80 GB → 340k | Reproduced in §11.3 with the arithmetic check. The jump from 8 GB to 12 GB (3k → 21k, 7×) is the most striking row and the one most likely to be a different-measurement artefact. |
| A20 | [31:29]–[32:56] | Feature list: GRPO/GSPO, custom kernels, end-to-end export, free GPU, Intel/AMD/NVIDIA, Linux/Mac/Windows | **No multi-GPU or FSDP in this list.** The absence is the evidence for §8.1's stop condition. |
| A21 | [32:09]–[32:22] | "exact math, no approximation" | Means "mathematically the same operations", not "bit-identical". See §17.5. |
| A22 | [11:40] | The Hugging Face `unsloth` org has 1,150 models | Pre-quantized *re-uploads*, not supported *architectures*. §17.10. |
| A23 | [4:44]–[5:10] | Repo folder structure: `utils`, `registry`, `model`, `kernels`, data preparation | The `kernels/` directory is where `fast_lora.py`, `cross_entropy_loss.py`, `rms_layernorm.py`, `rope_embedding.py`, and `swiglu.py` live — reading it is the fastest way to verify every claim in §4. |
| A24 | [33:11]–[55:29] | The practical walkthrough | Base model `unsloth/tinyllama-bnb-4bit`; `r=32`, `lora_alpha=32`, `lora_dropout=0.0`, 7 target modules; `SFTTrainer` with `packing=True`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `optim="adamw_8bit"`, `learning_rate=2e-5`, `num_train_epochs=1`, `seed=3407`. Reproduced verbatim in §6. |

**The three quotes to memorise for an interview:**

1. [14:48] — *"Unsloth is built on top of these three libraries: transformers, PEFT and TRL."* (It is a layer, not a framework.)
2. [25:15] — *"They have written the backpropagation manually… not the PyTorch autograd."* (The technical mechanism.)
3. [20:04] — *"standard Hugging Face training."* (The baseline every ratio is measured against.)

---

## Appendix B — Reference Links & Papers

### The source material for this module

| Resource | Reference |
|---|---|
| Video | *Unsloth Full Guide: Fine-Tune LLMs 2 to 4× Faster with Lowest VRAM* — Complete LLM Fine-Tuning playlist, video 18 of 32 |
| Transcript | `D:\Finetuning\_source\transcripts\LLM_Fine-Tuning_18_Unsloth_Full_Guide_Fine-Tune_LLMs_2_to_4x_Faster_with_Lowest.txt` (1,467 lines) |
| Companion notebook | `D:\Finetuning\_source\repo\Complete-LLM-Finetuning-main\LLM Fine-Tuning-18-unsloth\unsloth_practical.ipynb` (45 cells) |
| Comparison notebook (Unsloth arm) | `…\LLM Fine-Tuning-unsloth-vs-hf\unsloth_solution.ipynb` |
| Comparison notebook (HF arm) | `…\LLM Fine-Tuning-unsloth-vs-hf\huggingface_solution.ipynb` |

### Unsloth

| Resource | Where | What it is |
|---|---|---|
| Unsloth repository | `github.com/unslothai/unsloth` | The source. Read `unsloth/kernels/fast_lora.py` first — it is the clearest expression of the module's central claim. |
| Unsloth documentation | `docs.unsloth.ai` | The supported-models table, the `FastLanguageModel` API reference, and the FAQ on 4-bit merging. |
| Pre-quantized models | `huggingface.co/unsloth` | The `-bnb-4bit` re-uploads. Check licence per model. |
| Unsloth notebooks | `github.com/unslothai/notebooks` | The canonical per-model examples; the safest starting point for a new architecture. |

### The underlying techniques

| Paper / project | Reference | Relevance here |
|---|---|---|
| **LoRA** | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 (2021) | The adapter mathematics of §4.2–4.3 |
| **QLoRA** | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314 (2023) | NF4, double quantization, paged optimizers — §4.5, §11 |
| **FlashAttention** | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135 (2022) | The no-attention-matrix-materialisation mechanism — §4.6.1 |
| **FlashAttention-2** | Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 (2023) | The version Unsloth ships and enables by default |
| **LLM.int8()** | Dettmers et al., arXiv:2208.07339 (2022) | Outlier-aware quantization; background for the 4-bit discussion |
| **8-bit optimizers** | Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861 (2021) | `optim="adamw_8bit"`, the free memory win most people get for free |
| **Gradient checkpointing** | Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174 (2016) | The activation-recomputation trade in §4.4.4 |
| **Triton** | `triton-lang.org`; Tillet et al., *Triton: an intermediate language and compiler for tiled neural network computations*, MAPL 2019 | The language Unsloth's kernels are written in |
| **Liger Kernel** | `github.com/linkedin/Liger-Kernel` | The nearest independent implementation of the same fused-kernel ideas; a good cross-check on any speed claim |
| **Unsloth gradient checkpointing** | Unsloth blog, *Unsloth Gradient Checkpointing* | The specific implementation behind `use_gradient_checkpointing="unsloth"` |
| **PEFT** | `github.com/huggingface/peft`; `merge_and_unload` docs | Why merging a 4-bit base is wrong — §6.9 |
| **TRL** | `github.com/huggingface/trl`; `SFTTrainer`, `SFTConfig` | The training loop Unsloth accelerates; the source of the API drift noted in §6.1 |

### Where the benchmarks actually live

| Claim source | What to check before citing it |
|---|---|
| Unsloth's own benchmark charts | Read the axis labels and the footnote naming the baseline configuration. Almost all publish against a default-settings HF baseline. |
| The companion repo's `unsloth-vs-hf` notebooks | The two arms differ in timed scope, target modules, and compute dtype — see §12.6. Use as a demonstration, not as evidence. |
| Any blog's "X× faster" | Ask three questions: *faster than what config*, *on what GPU*, *at what sequence length*. If the post does not answer all three, the number is not usable. |
| Your own benchmark | Fix the seed, warm up 10 steps, measure steps 10–50, report `tokens/sec`, and publish the config diff next to the ratio. This is the only number that will survive scrutiny. |

---

*End of CS-16. Companion artifacts: CH-16 — Unsloth Cheat Sheet; IQ-16 — Unsloth Interview Questions. Source: video 18 of 32 in the Complete LLM Fine-Tuning playlist; transcript `LLM_Fine-Tuning_18_Unsloth_Full_Guide_Fine-Tune_LLMs_2_to_4x_Faster_with_Lowest.txt`; companion notebook `LLM Fine-Tuning-18-unsloth\unsloth_practical.ipynb`.*
