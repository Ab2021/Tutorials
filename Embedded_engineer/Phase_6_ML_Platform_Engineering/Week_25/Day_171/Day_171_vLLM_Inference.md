# Day 171: The Memory Manager: vLLM & PagedAttention
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** Standard Hugging Face inference wastes 60% of GPU memory due to memory fragmentation in the KV Cache. **vLLM** with **PagedAttention** treats GPU memory like Virtual RAM (Pages), enabling massive throughput improvements.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the KV Cache fragmentation problem in standard Transformer attention.
2.  **Implement** a high-throughput inference server using the `vllm` library.
3.  **Demonstrate** Continuous Batching (Iteration-level scheduling) vs Static Batching.
4.  **Benchmark** performance: Tokens/Sec vs Latency.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with GPU (Linux) or Colab.

### Software Environment
- `pip install vllm`. (Note: vLLM requires CUDA).

---

## 📖 Theoretical Foundation

### 1. The Paging Analogy
*   **Operating Systems:** Don't allocate contiguous RAM for a process. They allocate 4KB "Pages" mapped to varying physical addresses.
*   **PagedAttention:** Don't allocate contiguous VRAM for a User's KV Cache (which grows). Allocate "Block Tables".
*   **Benefit:** Zero internal fragmentation. Memory can be filled to 99% utilization.

### 2. Continuous Batching
*   **Static Batching:** Wait for 4 users. Run them together. User A generates 100 tokens. User B generates 1 token. GPU waits for User A to finish before returning User B. (Head-of-Line Blocking).
*   **Continuous Batching:** User B finishes after 1 step. Eject User B. Insert User E immediately into the batch.
*   **Result:** 20x higher throughput.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Running a vLLM Server

Compatible with OpenAI API.

```bash
# 1. Launch Server (Llama 2 7B)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 4096
```

### 👨‍💻 Core Implementation: Python SDK Usage

For direct integration without HTTP.

#### 📁 `src/vllm_inference.py`
```python
from vllm import LLM, SamplingParams

# 1. Initialize Engine
# PagedAttention is enabled by default
llm = LLM(model="facebook/opt-125m") 

# 2. Define Prompts (Batch)
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum mechanics in one sentence:",
]

# 3. Sampling Params
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# 4. Generate
# This uses Continuous Batching under the hood
outputs = llm.generate(prompts, sampling_params)

# 5. Print
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
```

### 👨‍💻 Core Implementation: Benchmarking Script

Measure Throughput.

#### 📁 `src/benchmark_vllm.py`
```python
import time
import asyncio
import aiohttp
import numpy as np

async def send_request(session, prompt):
    start = time.time()
    async with session.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "prompt": prompt,
            "max_tokens": 100,
        }
    ) as response:
        await response.read()
        return time.time() - start

async def benchmark(concurrency):
    prompts = ["Tell me a joke"] * 100 # Total requests
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_global = time.time()
        
        # Semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def bound_req(p):
            async with sem:
                return await send_request(session, p)
                
        for p in prompts:
            tasks.append(bound_req(p))
            
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_global
        
    print(f"Concurrency: {concurrency}")
    print(f"Throughput: {len(prompts) / total_time:.2f} req/s")
    print(f"Avg Latency: {np.mean(latencies):.2f} s")

# Run
# asyncio.run(benchmark(concurrency=10))
# asyncio.run(benchmark(concurrency=50))
```

---

## 🔬 Lab Exercise: "Context Overflow"

### Task
Observe PagedAttention swapping.
1.  Configure vLLM with limited GPU memory (`--gpu-memory-utilization 0.4`).
2.  Send a request with extremely long context (near limit).
3.  Send a second concurrent request.
4.  **Observation:** vLLM may **Preempt** the second request or swap blocks to CPU RAM if GPU VRAM is full.
5.  **Log:** Look for `Swapped out X blocks`.
6.  **Contrast:** In Hugging Face standard pipeline, this would just Crash (OOM).

---

## 📖 Advanced Theory: Speculative Decoding
vLLM is getting faster with Speculative Decoding.
*   **Idea:** Run a tiny "Draft Model" (125M params) to guess the next 5 tokens.
*   **Verify:** Run the Big Model (70B) once to verify the 5 tokens in parallel.
*   **Result:** If Draft is accurate, you get 5 tokens for the cost of 1 Big Model run.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Utilization:** Standard HF transformers pipeline typically achieves < 20% GPU Compute utilization during generation. vLLM pushes this to > 80%.
2.  **API Compatibility:** vLLM implements the OpenAI API standard (`/v1/completions`). This means you can swap out `openai.api_key` for your local URL and existing apps just work.
3.  **Hardware Support:** vLLM is heavily optimized for NVIDIA. AMD ROCm support is improving but laggy.

### API Summary
```python
LLM(model="...").generate(prompts, params)
```

---

**Day 171 Complete** ✅

*Next: Day 172 - LoRA & PEFT - Fine-tuning Giants.*
