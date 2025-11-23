# Day 44: Model Serving & Optimization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. vLLM Complete Production Setup

```python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uvicorn

# Initialize async vLLM engine
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-13b-chat-hf",
    tensor_parallel_size=2,  # 2 GPUs
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.95,
    enable_prefix_caching=True,  # KV cache reuse
    disable_log_stats=False
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    if request.stream:
        return StreamingResponse(
            stream_results(request.prompt, sampling_params),
            media_type="text/event-stream"
        )
    else:
        results = await engine.generate(
            request.prompt,
            sampling_params,
            request_id=f"req_{id(request)}"
        )
        
        output = results.outputs[0]
        return {
            "text": output.text,
            "tokens": len(output.token_ids),
            "finish_reason": output.finish_reason
        }

async def stream_results(prompt: str, sampling_params: SamplingParams):
    """Stream tokens as they're generated."""
    request_id = f"stream_{id(prompt)}"
    
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            yield f"data: {output.outputs[0].text}\n\n"

@app.get("/metrics")
async def metrics():
    """Expose vLLM metrics."""
    stats = await engine.get_model_stats()
    return {
        "num_requests_running": stats.num_requests_running,
        "num_requests_waiting": stats.num_requests_waiting,
        "gpu_cache_usage": stats.gpu_cache_usage_sys,
        "num_preempted": stats.num_preempted
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

### 2. TGI Deployment with Docker

```dockerfile
# Dockerfile for TGI
FROM ghcr.io/huggingface/text-generation-inference:latest

ENV MODEL_ID="meta-llama/Llama-2-13b-chat-hf"
ENV NUM_SHARD=2
ENV QUANTIZE="bitsandbytes-nf4"
ENV MAX_BATCH_PREFILL_TOKENS=4096
ENV MAX_TOTAL_TOKENS=8192
ENV MAX_INPUT_LENGTH=4000

CMD text-generation-launcher \
    --model-id $MODEL_ID \
    --num-shard $NUM_SHARD \
    --quantize $QUANTIZE \
    --max-batch-prefill-tokens $MAX_BATCH_PREFILL_TOKENS \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --max-input-length $MAX_INPUT_LENGTH \
    --port 8080
```

```python
# Client for TGI
import requests

class TGIClient:
    def __init__(self, url="http://localhost:8080"):
        self.url = url
    
    def generate(self, prompt: str, max_tokens: int = 512):
        response = requests.post(
            f"{self.url}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
        )
        return response.json()["generated_text"]
    
    def generate_stream(self, prompt: str, max_tokens: int = 512):
        response = requests.post(
            f"{self.url}/generate_stream",
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens}
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)["token"]["text"]
```

### 3. Speculative Decoding Implementation

```python
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, k=5):
        self.draft_model = draft_model  # Small model (7B)
        self.target_model = target_model  # Large model (70B)
        self.k = k  # Number of draft tokens
    
    def generate(self, prompt: str, max_tokens: int = 512):
        tokens = tokenize(prompt)
        
        while len(tokens) < max_tokens:
            # Draft: Generate k tokens with small model
            draft_tokens = self.draft_model.generate(
                tokens,
                max_new_tokens=self.k,
                do_sample=True
            )
            
            # Verify: Check all k tokens with large model in parallel
            logits = self.target_model.forward(
                torch.cat([tokens, draft_tokens])
            )
            
            # Accept tokens while probabilities match
            accepted = 0
            for i in range(self.k):
                draft_prob = draft_model.get_prob(draft_tokens[i])
                target_prob = target_model.get_prob_from_logits(logits[len(tokens) + i])
                
                if random.random() < min(1, target_prob / draft_prob):
                    tokens.append(draft_tokens[i])
                    accepted += 1
                else:
                    break
            
            # If all rejected, sample from target model
            if accepted == 0:
                next_token = target_model.sample(logits[len(tokens)])
                tokens.append(next_token)
        
        return tokens
```

### 4. Benchmark Suite

```python
import time
import numpy as np
from typing import List, Dict

class LLMBenchmark:
    def __init__(self, model_url: str):
        self.model_url = model_url
    
    def benchmark_latency(
        self,
        prompts: List[str],
        num_runs: int = 100
    ) -> Dict:
        """Measure latency metrics."""
        ttfts = []  # Time to first token
        total_latencies = []
        tokens_per_second = []
        
        for prompt in prompts[:num_runs]:
            start = time.time()
            first_token_time = None
            total_tokens = 0
            
            for token in self.generate_stream(prompt):
                if first_token_time is None:
                    first_token_time = time.time() - start
                total_tokens += 1
            
            total_time = time.time() - start
            
            ttfts.append(first_token_time)
            total_latencies.append(total_time)
            tokens_per_second.append(total_tokens / total_time)
        
        return {
            "ttft_p50": np.percentile(ttfts, 50),
            "ttft_p95": np.percentile(ttfts, 95),
            "ttft_p99": np.percentile(ttfts, 99),
            "total_latency_p50": np.percentile(total_latencies, 50),
            "total_latency_p95": np.percentile(total_latencies, 95),
            "tokens_per_second_avg": np.mean(tokens_per_second)
        }
    
    def benchmark_throughput(
        self,
        prompts: List[str],
        duration_seconds: int = 60
    ) -> Dict:
        """Measure throughput under load."""
        start_time = time.time()
        completed_requests = 0
        total_tokens = 0
        
        while time.time() - start_time < duration_seconds:
            prompt = prompts[completed_requests % len(prompts)]
            output = self.generate(prompt)
            completed_requests += 1
            total_tokens += len(output.split())
        
        elapsed = time.time() - start_time
        
        return {
            "requests_per_second": completed_requests / elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "total_requests": completed_requests
        }
```

### 5. Multi-GPU Tensor Parallelism

```python
# Tensor parallelism splits model layers across GPUs
import torch
import torch.distributed as dist

class TensorParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # Each GPU gets a slice of the weight matrix
        self.out_features_per_gpu = out_features // world_size
        
        self.weight = torch.nn.Parameter(
            torch.randn(self.out_features_per_gpu, in_features)
        )
    
    def forward(self, x):
        # Each GPU computes its slice
        local_output = torch.matmul(x, self.weight.t())
        
        # All-gather to combine results
        output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        # Concatenate along output dimension
        output = torch.cat(output_list, dim=-1)
        
        return output
```

### 6. KV Cache Sharing (Prefix Caching)

```python
class PrefixCacheManager:
    def __init__(self):
        self.cache = {}  # prefix_hash -> kv_cache
    
    def get_cached_kv(self, prefix: str):
        """Get cached KV for common prefix."""
        prefix_hash = hash(prefix)
        return self.cache.get(prefix_hash)
    
    def store_kv(self, prefix: str, kv_cache):
        """Store KV cache for prefix."""
        prefix_hash = hash(prefix)
        self.cache[prefix_hash] = kv_cache
    
    def generate_with_cache(self, prompt: str, model):
        """Generate using cached KV for common prefix."""
        # Find longest matching prefix
        best_prefix = ""
        for cached_prefix in self.cache.keys():
            if prompt.startswith(cached_prefix) and len(cached_prefix) > len(best_prefix):
                best_prefix = cached_prefix
        
        if best_prefix:
            # Reuse cached KV
            kv_cache = self.get_cached_kv(best_prefix)
            remaining_prompt = prompt[len(best_prefix):]
            
            # Only process remaining tokens
            output = model.generate(remaining_prompt, past_kv=kv_cache)
        else:
            # No cache hit, process full prompt
            output = model.generate(prompt)
        
        return output
```
