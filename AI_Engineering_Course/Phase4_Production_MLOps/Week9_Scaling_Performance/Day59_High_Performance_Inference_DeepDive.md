# Day 59: High-Performance Inference Serving
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Continuous Batching Scheduler Implementation

```python
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    arrival_time: float
    generated_tokens: List[str] = field(default_factory=list)
    finished: bool = False

class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size=4, max_tokens_per_batch=2048):
        self.waiting_queue = queue.Queue()
        self.running_batch: List[Request] = []
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.lock = threading.Lock()
    
    def add_request(self, request: Request):
        """Add new request to queue."""
        self.waiting_queue.put(request)
    
    def schedule(self):
        """Schedule requests for next iteration."""
        with self.lock:
            # 1. Remove finished requests
            self.running_batch = [req for req in self.running_batch if not req.finished]
            
            # 2. Add new requests if capacity allows
            while not self.waiting_queue.empty():
                # Check slot capacity
                if len(self.running_batch) >= self.max_batch_size:
                    break
                
                # Check token capacity (simplified)
                current_tokens = sum(len(r.generated_tokens) + len(r.prompt) for r in self.running_batch)
                next_req = self.waiting_queue.queue[0] # Peek
                if current_tokens + len(next_req.prompt) > self.max_tokens_per_batch:
                    break
                
                # Add to batch
                req = self.waiting_queue.get()
                self.running_batch.append(req)
            
            return self.running_batch

    def step(self):
        """Simulate one decoding step."""
        batch = self.schedule()
        
        if not batch:
            time.sleep(0.01)
            return
        
        # Simulate forward pass (in reality, this calls the model)
        print(f"Processing batch size: {len(batch)}")
        for req in batch:
            # Generate dummy token
            req.generated_tokens.append(" token")
            
            # Check finish condition
            if len(req.generated_tokens) >= req.max_tokens:
                req.finished = True
                print(f"Request {req.request_id} finished.")

# Usage Simulation
scheduler = ContinuousBatchingScheduler()

# Add requests arriving at different times
scheduler.add_request(Request("req1", "Hello", 10, time.time()))
scheduler.add_request(Request("req2", "World", 5, time.time()))

# Simulate loop
for _ in range(15):
    scheduler.step()
    if _ == 2:
        print("New request arriving...")
        scheduler.add_request(Request("req3", "Late", 5, time.time()))
```

### 2. PagedAttention Memory Manager (Simulation)

```python
class BlockTable:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.physical_blocks = [] # List of actual memory blocks
        self.free_blocks = set()
        self.mapping = {} # request_id -> [physical_block_indices]
    
    def allocate(self, request_id, num_tokens):
        """Allocate blocks for a new request."""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        
        allocated_indices = []
        for _ in range(num_blocks):
            if self.free_blocks:
                idx = self.free_blocks.pop()
            else:
                idx = len(self.physical_blocks)
                self.physical_blocks.append(f"Block_{idx}")
            allocated_indices.append(idx)
        
        self.mapping[request_id] = allocated_indices
        return allocated_indices
    
    def append_token(self, request_id):
        """Handle memory when a new token is generated."""
        # Check if last block is full
        # In real implementation, we track usage per block
        # Here we simplify: assume we might need a new block
        pass 

    def free(self, request_id):
        """Free blocks associated with request."""
        if request_id in self.mapping:
            indices = self.mapping[request_id]
            for idx in indices:
                self.free_blocks.add(idx)
            del self.mapping[request_id]
            print(f"Freed blocks for {request_id}: {indices}")

# Usage
mem_manager = BlockTable(block_size=4)
req1_blocks = mem_manager.allocate("req1", 10) # Needs 3 blocks (4+4+2)
print(f"Req1 blocks: {req1_blocks}")

req2_blocks = mem_manager.allocate("req2", 5)  # Needs 2 blocks (4+1)
print(f"Req2 blocks: {req2_blocks}")

mem_manager.free("req1")
req3_blocks = mem_manager.allocate("req3", 8)  # Needs 2 blocks, reuses freed
print(f"Req3 blocks: {req3_blocks}")
```

### 3. Speculative Decoding Implementation

```python
import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, temp=1.0):
        self.draft_model = draft_model
        self.target_model = target_model
        self.temp = temp

    def generate(self, input_ids, gamma=4):
        """
        gamma: number of speculative tokens
        """
        # 1. Draft Generation (Autoregressive)
        draft_tokens = []
        curr_input = input_ids
        
        for _ in range(gamma):
            with torch.no_grad():
                logits = self.draft_model(curr_input)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                curr_input = torch.cat([curr_input, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1) # [1, gamma]
        
        # 2. Target Verification (Parallel)
        # Input to target: original + draft tokens
        # We need logits for all positions
        with torch.no_grad():
            full_input = torch.cat([input_ids, draft_tokens], dim=1)
            target_logits = self.target_model(full_input)
            
        # 3. Acceptance Loop
        # Compare draft tokens with target predictions
        # Target predicts t_{i+1} given t_0...t_i
        
        n_accepted = 0
        final_tokens = []
        
        for i in range(gamma):
            # Target probability for token at i
            # Logits index: input_len + i - 1 (prediction for next)
            target_probs = F.softmax(target_logits[:, input_ids.shape[1] + i - 1, :] / self.temp, dim=-1)
            draft_token_id = draft_tokens[0, i]
            
            # Rejection Sampling (simplified: greedy match)
            target_token_id = torch.argmax(target_probs)
            
            if draft_token_id == target_token_id:
                final_tokens.append(draft_token_id)
                n_accepted += 1
            else:
                # Reject and stop
                final_tokens.append(target_token_id) # Use target's correction
                break
                
        # If all accepted, generate one more from target
        if n_accepted == gamma:
            last_logits = target_logits[:, -1, :]
            extra_token = torch.argmax(last_logits)
            final_tokens.append(extra_token)
            
        return torch.tensor(final_tokens)

# Note: This is a simplified logic. 
# Real implementation handles KV cache updates for rejected tokens.
```

### 4. Benchmarking Script (TTFT & TPOT)

```python
import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/generate"

def benchmark_request(prompt, max_tokens):
    start_time = time.time()
    
    # Streaming request to measure TTFT
    response = requests.post(API_URL, json={
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }, stream=True)
    
    ttft = 0
    first_token_received = False
    token_count = 0
    
    for chunk in response.iter_content(chunk_size=None):
        if not first_token_received:
            ttft = time.time() - start_time
            first_token_received = True
        token_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    tpot = (total_time - ttft) / (token_count - 1) if token_count > 1 else 0
    
    return {
        "ttft": ttft,
        "tpot": tpot,
        "total_time": total_time,
        "tokens": token_count
    }

def run_benchmark(concurrency=10, num_requests=100):
    prompts = ["Hello world"] * num_requests
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda p: benchmark_request(p, 50), prompts))
    
    # Aggregate
    avg_ttft = np.mean([r['ttft'] for r in results])
    p95_ttft = np.percentile([r['ttft'] for r in results], 95)
    avg_tpot = np.mean([r['tpot'] for r in results])
    throughput = sum([r['tokens'] for r in results]) / sum([r['total_time'] for r in results])
    
    print(f"Avg TTFT: {avg_ttft*1000:.2f} ms")
    print(f"P95 TTFT: {p95_ttft*1000:.2f} ms")
    print(f"Avg TPOT: {avg_tpot*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/s")

# run_benchmark()
```

### 5. Tensor Parallelism Logic (Layer-wise)

```python
# Conceptual logic for Row/Column Parallel Linear Layer
class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, parallel_mode='column'):
        super().__init__()
        self.mode = parallel_mode
        world_size = dist.get_world_size()
        
        if mode == 'column':
            # Split output features
            self.weight = nn.Parameter(torch.randn(out_features // world_size, in_features))
        else:
            # Split input features
            self.weight = nn.Parameter(torch.randn(out_features, in_features // world_size))
            
    def forward(self, x):
        if self.mode == 'column':
            # Output is partial (split across GPUs)
            return F.linear(x, self.weight)
        else:
            # Input is partial (split across GPUs)
            partial_out = F.linear(x, self.weight)
            # All-Reduce to sum up results from all GPUs
            dist.all_reduce(partial_out)
            return partial_out
```

### 6. vLLM Engine Initialization (Example)

```python
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1, # Number of GPUs
    gpu_memory_utilization=0.90, # Reserve 90% of GPU memory
    max_num_batched_tokens=4096
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
