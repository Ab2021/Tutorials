# 11. Inference Optimization — Making Models Fast in Production

## Table of Contents
- [Why Inference Optimization?](#why-inference-optimization)
- [KV Caching Deep Dive](#kv-caching-deep-dive)
- [Speculative Decoding](#speculative-decoding)
- [Quantization for Inference](#quantization-for-inference)
- [Continuous Batching](#continuous-batching)
- [vLLM: Paged Attention](#vllm-paged-attention)
- [Text Generation Inference (TGI)](#text-generation-inference-tgi)
- [GGUF and llama.cpp](#gguf-and-llamacpp)
- [Latency vs Throughput](#latency-vs-throughput)
- [Practical: Deploying Our Model](#practical-deploying-our-model)

---

## Why Inference Optimization?

```
Training: Happens ONCE (hours). Cost amortized.
Inference: Happens MILLIONS of times. Cost is per-request.

Example:
  Training: 4 hours on 1 GPU = ~$4
  Inference: 1000 requests/day × 365 days × $0.001/request = $365/year

Inference cost dominates! Optimization here saves real money.

Key metrics:
  - Latency: Time to generate one response (user wait time)
  - Throughput: Requests processed per second (server capacity)
  - Memory: GPU VRAM needed to serve the model
  - Cost: $/token or $/request
```

---

## KV Caching Deep Dive

### The Redundancy Problem in Autoregressive Generation

```
Generating "The cat sat on the mat":

Step 1: Input "The"          → compute K,V for ALL layers for "The"
Step 2: Input "The cat"      → recompute K,V for "The" AND "cat"
Step 3: Input "The cat sat"  → recompute K,V for ALL three tokens
...

Without caching:
  Step 1: 1 token processed
  Step 2: 2 tokens processed  (redundant: "The" recomputed)
  Step 3: 3 tokens processed  (redundant: "The", "cat" recomputed)
  Step N: N tokens processed  (redundant: N-1 tokens recomputed!)
  
  Total work: 1 + 2 + 3 + ... + N = N(N+1)/2 = O(N²)

With KV cache:
  Step 1: Compute K₁, V₁, cache them
  Step 2: Compute K₂, V₂ only → attend to cached K₁₂, V₁₂
  Step 3: Compute K₃, V₃ only → attend to cached K₁₂₃, V₁₂₃
  
  Total work: N (each step only processes 1 new token)
  Speedup: O(N) vs O(N²) → N/2× faster!
```

### Cache Implementation

```python
def generate_with_kv_cache(model, input_ids, max_length):
    past_key_values = None  # Will hold the KV cache
    
    for step in range(max_length):
        # Only process NEW token(s), pass cached KV
        outputs = model(
            input_ids=input_ids[:, -1:] if past_key_values else input_ids,
            past_key_values=past_key_values,  # Pass cache
            use_cache=True,                     # Enable caching
        )
        
        # Update cache
        past_key_values = outputs.past_key_values
        
        # Get next token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return input_ids
```

---

## Speculative Decoding

### The Idea: Draft and Verify

```
Problem: Large models are slow but accurate.
         Small models are fast but less accurate.

Solution: Use a SMALL model to draft multiple tokens quickly,
          then verify the draft with the LARGE model in ONE pass.

Algorithm:
  1. Small model (Gemma-2B) drafts 4 tokens: "The cat sat on"
  2. Large model (Gemma-7B) processes all 4 in ONE forward pass
  3. Large model accepts tokens it agrees with, rejects others

Example:
  Draft: "The cat sat on"
  Large model verification:
    "The" → agree ✅
    "cat" → agree ✅
    "sat" → agree ✅
    "on"  → disagree ❌ (large model prefers "upon")
    
  Accept first 3, regenerate from "sat":
  Output: "The cat sat upon"

Speedup: If draft acceptance rate = 75%,
  and drafting 4 tokens = same cost as 1 large model step:
  Effective speedup = 4 × 0.75 = 3× faster!
```

---

## Quantization for Inference

### Post-Training Quantization (PTQ)

```
Take a trained model and COMPRESS it for faster inference:

fp16 (2 bytes) → int8 (1 byte) → int4 (0.5 bytes)

Methods:
  GPTQ: Group-wise quantization, popular for 4-bit
  AWQ:  Activation-aware weight quantization (protects important weights)
  GGUF: Universal format for llama.cpp (CPU/mixed inference)

Quality comparison (Gemma-2B):
  fp16:  MMLU = 45.2%  Speed: 1×    Memory: 4 GB
  int8:  MMLU = 44.9%  Speed: 1.5×  Memory: 2 GB
  GPTQ4: MMLU = 43.7%  Speed: 2×    Memory: 1 GB
  AWQ4:  MMLU = 44.1%  Speed: 2×    Memory: 1 GB
```

### QLoRA's 4-bit is Training-Time Quantization

```
QLoRA (our training):
  - Quantize base model to 4-bit for TRAINING
  - LoRA adapters stay in fp16
  - Dequantize on-the-fly during forward pass

For deployment, you have choices:
  1. Keep 4-bit model + merged LoRA → fastest, lowest memory
  2. Merge LoRA → dequantize to fp16 → re-quantize with GPTQ → best quality
  3. Export to GGUF → deploy on llama.cpp → runs on CPU!
```

---

## Continuous Batching

### The Static Batching Problem

```
Static batching (naive):
  Request 1: "Hello world" → 50 tokens generated
  Request 2: "Hi"          → 5 tokens generated
  Request 3: "Tell me about AI..." → 200 tokens generated
  
  Batch together: Wait for ALL to finish (200 tokens = slowest)
  Request 1 finishes at step 50 → WASTED 150 steps of GPU time
  Request 2 finishes at step 5  → WASTED 195 steps!
```

### Continuous Batching (vLLM, TGI)

```
As soon as a request finishes, IMMEDIATELY start a new one:

Time →
GPU:  [Req1][Req2][Req3][----][----][----][----]
      [Req4][----][----][----][----][----][----]  ← fill gaps!
      [Req5][Req6][----][----][----][----][----]  ← more filling!

No GPU idle time. Throughput increases 2-10× over static batching.
```

---

## vLLM: Paged Attention

### The KV Cache Memory Problem

```
Standard KV cache: Pre-allocate memory for max_seq_length for EACH request.
  Max length = 2048, batch = 32:
  KV cache = 32 × 2048 × (K + V per layer) = potentially GIGABYTES

But most requests don't reach max_seq_length!
  Request 1: 50 tokens (2048 pre-allocated → 97.6% wasted)
  Request 2: 120 tokens (2048 pre-allocated → 94.1% wasted)
```

### Paged Attention: Virtual Memory for KV Cache

```
Inspired by OS virtual memory:
  - Divide KV cache into PAGES (blocks of e.g. 16 tokens)
  - Allocate pages ON DEMAND as tokens are generated
  - Pages can be non-contiguous in physical GPU memory

Request 1 (50 tokens): 4 pages allocated (not 128!)
Request 2 (120 tokens): 8 pages allocated (not 128!)

Memory savings: 50-90% KV cache reduction!
  → Fit MORE requests simultaneously → higher throughput
```

### vLLM Usage

```python
from vllm import LLM, SamplingParams

# Load model with vLLM engine
llm = LLM(model="path/to/merged_model", quantization="awq")

# Generate
prompts = ["Review: 'Great phone!' Rating: 5"]
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(prompts, params)

# vLLM handles:
# ✅ Paged attention (memory efficient)
# ✅ Continuous batching (high throughput)
# ✅ Optimized CUDA kernels
# ✅ Tensor parallelism (multi-GPU)
```

---

## Text Generation Inference (TGI)

### HuggingFace's Production Server

```
TGI = HuggingFace's optimized inference server

Features:
  ✅ Continuous batching
  ✅ Flash Attention
  ✅ Quantization (GPTQ, AWQ, bitsandbytes)
  ✅ Token streaming (send tokens as they're generated)
  ✅ Multi-GPU with tensor parallelism
  ✅ LoRA adapter support (serve multiple adapters!)
  ✅ gRPC and REST API

Launch:
  docker run --gpus all \
    -v /path/to/model:/model \
    ghcr.io/huggingface/text-generation-inference \
    --model-id /model
```

---

## GGUF and llama.cpp

### Running on CPU

```
llama.cpp: C++ inference engine that runs on CPU (and GPU).
GGUF: The file format for llama.cpp models.

Why CPU inference?
  ✅ No GPU needed — runs on any computer
  ✅ Ideal for edge deployment (laptops, phones)
  ✅ 4-bit quantized models are small (1-2 GB for Gemma-2B)
  ❌ Slower than GPU (but fast enough for many use cases)

Convert our model:
  1. Merge LoRA adapters: model.merge_and_unload()
  2. Save in fp16: model.save_pretrained("merged_model")
  3. Convert to GGUF: python convert.py merged_model --outtype q4_K_M
  4. Run: ./main -m model.gguf -p "Your prompt here"

GGUF quantization types:
  Q4_0:  4-bit (fastest, lowest quality)
  Q4_K_M: 4-bit with key weight protection (recommended)
  Q5_K_M: 5-bit (better quality, more memory)
  Q8_0:  8-bit (near fp16 quality, 2× memory of Q4)
```

---

## Latency vs Throughput

```
LATENCY = Time for ONE request to complete
  User-facing metric: "How long does the user wait?"
  
THROUGHPUT = Requests processed per second
  Server-side metric: "How many users can we serve?"

You can optimize for one at the expense of the other:

High throughput, higher latency:
  Batch many requests together → GPU is efficient → more requests/sec
  But each request waits for the whole batch → individual latency increases

Low latency, lower throughput:
  Process requests immediately → GPU may be underutilized → fewer requests/sec
  But each request starts immediately → individual latency is minimal

Continuous batching: Gets the best of both!
  New requests join the batch immediately → low latency
  GPU stays busy → high throughput
```

---

## Practical: Deploying Our Model

```
OPTION 1: Simple Python Script (Development)
  python inference.py --model_dir ./outputs/run_xxx/final_model --interactive
  
  ✅ Easy, no extra dependencies
  ❌ Single request at a time
  ❌ Not production-ready

OPTION 2: vLLM Server (Production, GPU)
  1. Merge LoRA: python -c "model.merge_and_unload(); model.save_pretrained('merged')"
  2. Serve: python -m vllm.entrypoints.openai.api_server --model ./merged
  3. API: curl localhost:8000/v1/completions -d '{"prompt": "...", "max_tokens": 256}'
  
  ✅ High throughput, low latency
  ✅ OpenAI-compatible API
  ❌ Requires GPU

OPTION 3: GGUF + llama.cpp (CPU / Edge)
  1. Convert to GGUF
  2. Run locally on any machine
  
  ✅ No GPU needed
  ✅ Small file size (1-2 GB)
  ❌ Slower generation
```
