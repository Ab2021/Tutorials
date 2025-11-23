# Day 43: Production LLM Deployment
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. vLLM PagedAttention Implementation

**Problem:** KV cache memory is fragmented and wasted.

**Traditional Approach:**
```python
# Pre-allocate max sequence length
kv_cache = torch.zeros(batch_size, max_seq_len, num_heads, head_dim)
# Wastes memory for shorter sequences
```

**PagedAttention (vLLM):**
```python
# Allocate in pages (blocks)
block_size = 16  # tokens per block
num_blocks = (max_seq_len + block_size - 1) // block_size

# Only allocate blocks as needed
kv_cache_blocks = []
for i in range(num_blocks_needed):
    kv_cache_blocks.append(allocate_block())

# Can share blocks across sequences (prefix caching)
```

**Benefits:**
- **Memory Efficiency:** 2-4x reduction in memory usage.
- **Throughput:** 24x higher than naive implementation.
- **Sharing:** Reuse KV cache for common prefixes.

### 2. Complete vLLM Deployment

```python
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # Use 2 GPUs
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.9
)

# FastAPI app
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    
    return {
        "text": outputs[0].outputs[0].text,
        "tokens_generated": len(outputs[0].outputs[0].token_ids)
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Continuous Batching

**Static Batching:**
```python
# Wait for batch to fill
batch = []
while len(batch) < batch_size:
    batch.append(wait_for_request())

# Process entire batch
outputs = model.generate(batch)
```
**Problem:** Late requests wait for entire batch to complete.

**Continuous Batching (vLLM):**
```python
# Process requests as they arrive
active_requests = []

while True:
    # Add new requests
    if new_request_available():
        active_requests.append(get_request())
    
    # Remove completed requests
    active_requests = [r for r in active_requests if not r.is_complete()]
    
    # Generate next token for all active requests
    if active_requests:
        next_tokens = model.generate_next_token(active_requests)
        for req, token in zip(active_requests, next_tokens):
            req.append_token(token)
```

**Benefits:**
- **Lower Latency:** Requests don't wait for batch to fill.
- **Higher Throughput:** GPU is always busy.

### 4. Quantization with GPTQ

```python
from transformers import AutoModelForCausalLM, GPTQConfig

# Quantize model to 4-bit
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer="meta-llama/Llama-2-7b-hf"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Model is now 4x smaller
# 7B model: 28 GB → 7 GB
```

### 5. Monitoring with Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
tokens_generated = Counter('llm_tokens_generated_total', 'Total tokens generated')
active_requests = Gauge('llm_active_requests', 'Active requests')
gpu_memory = Gauge('llm_gpu_memory_used_bytes', 'GPU memory used')

@app.post("/generate")
async def generate(request: GenerateRequest):
    request_count.inc()
    active_requests.inc()
    
    start_time = time.time()
    
    try:
        outputs = llm.generate([request.prompt], sampling_params)
        
        latency = time.time() - start_time
        request_latency.observe(latency)
        
        num_tokens = len(outputs[0].outputs[0].token_ids)
        tokens_generated.inc(num_tokens)
        
        return {"text": outputs[0].outputs[0].text}
    finally:
        active_requests.dec()

# Start Prometheus metrics server
start_http_server(9090)
```

### 6. Autoscaling with Kubernetes

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_active_requests
      target:
        type: AverageValue
        averageValue: "10"
```

### 7. Cost Optimization

```python
class CostOptimizedRouter:
    def __init__(self):
        self.small_model = LLM("gpt-3.5-turbo")  # $0.001/1K tokens
        self.large_model = LLM("gpt-4")  # $0.03/1K tokens
    
    def route(self, prompt: str) -> str:
        # Classify query complexity
        complexity = self.classify_complexity(prompt)
        
        if complexity == "simple":
            return self.small_model.generate(prompt)
        else:
            return self.large_model.generate(prompt)
    
    def classify_complexity(self, prompt: str) -> str:
        # Simple heuristic
        if len(prompt) < 100 and "?" in prompt:
            return "simple"
        return "complex"
```

### 8. Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedLLM:
    def __init__(self, llm):
        self.llm = llm
        self.cache = {}
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Create cache key
        cache_key = hashlib.md5(
            f"{prompt}:{kwargs}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate
        output = self.llm.generate(prompt, **kwargs)
        
        # Cache result
        self.cache[cache_key] = output
        
        return output
```
