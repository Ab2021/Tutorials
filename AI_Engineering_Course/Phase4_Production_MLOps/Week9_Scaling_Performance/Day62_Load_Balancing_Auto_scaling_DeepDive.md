# Day 62: Load Balancing & Auto-scaling
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Load Balancer Simulation

Comparing Round Robin vs Least Connections vs Peak EWMA.

```python
import random
import heapq
import numpy as np

class Replica:
    def __init__(self, id, speed_factor=1.0):
        self.id = id
        self.active_requests = 0
        self.speed_factor = speed_factor # Higher is slower (e.g., thermal throttling)
        self.latencies = []
        
    def process(self, duration):
        """Simulate processing time."""
        actual_duration = duration * self.speed_factor
        self.latencies.append(actual_duration)
        return actual_duration

class LoadBalancer:
    def __init__(self, replicas):
        self.replicas = replicas
        self.rr_index = 0
        
    def round_robin(self):
        replica = self.replicas[self.rr_index]
        self.rr_index = (self.rr_index + 1) % len(self.replicas)
        return replica
    
    def least_connections(self):
        # Find replica with min active_requests
        return min(self.replicas, key=lambda r: r.active_requests)
    
    def peak_ewma(self):
        # Simplified: choose replica with lowest recent average latency
        # In reality, we track EWMA of latency
        return min(self.replicas, key=lambda r: np.mean(r.latencies[-10:]) if r.latencies else 0)

# Simulation
replicas = [Replica(0, 1.0), Replica(1, 1.0), Replica(2, 2.0)] # Replica 2 is slow
lb = LoadBalancer(replicas)

# Generate traffic
requests = [random.uniform(0.1, 1.0) for _ in range(100)]

# Test Least Connections
print("Testing Least Connections...")
total_latency = 0
for req_duration in requests:
    replica = lb.least_connections()
    replica.active_requests += 1
    
    # Simulate async completion (simplified)
    latency = replica.process(req_duration)
    total_latency += latency
    
    replica.active_requests -= 1 # Assume it finishes immediately for this sim step

print(f"Total Latency: {total_latency:.2f}")
```

### 2. Auto-scaler Logic (PID Controller)

Using a PID controller for smooth scaling (preventing oscillation).

```python
class AutoScaler:
    def __init__(self, target_concurrency=10, min_replicas=1, max_replicas=10):
        self.target = target_concurrency
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        
        # PID State
        self.kp = 1.0 # Proportional gain
        self.ki = 0.1 # Integral gain
        self.kd = 0.5 # Derivative gain
        self.integral = 0
        self.prev_error = 0
        
    def update(self, current_concurrency):
        """
        Calculate desired replicas based on concurrency metric.
        """
        # Error: How far are we from target per replica?
        # We want current_concurrency / replicas = target
        # So desired_replicas = current_concurrency / target
        
        # Let's use error in terms of total capacity
        current_load_per_replica = current_concurrency / self.current_replicas
        error = current_load_per_replica - self.target
        
        # PID Calculation
        self.integral += error
        derivative = error - self.prev_error
        
        adjustment = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Update replicas (soft update)
        new_replicas = self.current_replicas + adjustment
        
        # Clamp
        new_replicas = max(self.min_replicas, min(self.max_replicas, round(new_replicas)))
        
        self.prev_error = error
        self.current_replicas = new_replicas
        
        return int(new_replicas)

# Simulation
scaler = AutoScaler(target_concurrency=10)
traffic_pattern = [5, 20, 50, 100, 120, 80, 40, 10] # Concurrency over time

print("\nAuto-scaling Simulation:")
for load in traffic_pattern:
    replicas = scaler.update(load)
    print(f"Load: {load}, Replicas: {replicas}, Load/Replica: {load/replicas:.1f}")
```

### 3. Priority Queue Implementation

Handling VIP users vs Free users.

```python
import queue
import time
from dataclasses import dataclass, field

@dataclass(order=True)
class PriorityRequest:
    priority: int # Lower is higher priority
    arrival_time: float
    request_id: str = field(compare=False)

class RequestScheduler:
    def __init__(self):
        self.queue = queue.PriorityQueue()
        
    def add_request(self, req_id, user_tier):
        priority = 0 if user_tier == 'vip' else 10
        req = PriorityRequest(priority, time.time(), req_id)
        self.queue.put(req)
        
    def get_next_request(self):
        if self.queue.empty():
            return None
        return self.queue.get()

# Usage
scheduler = RequestScheduler()
scheduler.add_request("free_user_1", "free")
scheduler.add_request("vip_user_1", "vip")
scheduler.add_request("free_user_2", "free")

print("\nProcessing Order:")
while not scheduler.queue.empty():
    req = scheduler.get_next_request()
    print(f"Processing: {req.request_id} (Priority: {req.priority})")
```

### 4. Multi-Model Serving (LoRA Swapping)

Conceptual implementation of adapter swapping.

```python
class LoRAServer:
    def __init__(self, base_model_path):
        self.base_model = self.load_base_model(base_model_path)
        self.adapters = {} # id -> weights
        self.active_adapter = None
        
    def load_base_model(self, path):
        print(f"Loading base model from {path} (140GB)...")
        return "BaseModel"
    
    def register_adapter(self, adapter_id, path):
        print(f"Loading adapter {adapter_id} from {path} (100MB)...")
        self.adapters[adapter_id] = f"Weights_{adapter_id}"
        
    def process(self, request, adapter_id):
        # Swap if needed
        if self.active_adapter != adapter_id:
            self._swap_adapter(adapter_id)
            
        # Inference
        print(f"Running inference with {adapter_id}")
        
    def _swap_adapter(self, adapter_id):
        # In reality: copy weights from CPU/RAM to GPU
        # LoRA weights are small, so this is fast (~10-50ms)
        print(f"Swapping GPU weights: {self.active_adapter} -> {adapter_id}")
        self.active_adapter = adapter_id

# Usage
server = LoRAServer("llama-3-70b")
server.register_adapter("finance", "lora/finance")
server.register_adapter("coding", "lora/coding")

server.process("What is P/E ratio?", "finance")
server.process("Write a Python script", "coding")
```

### 5. Distributed Tracing (OpenTelemetry)

How to instrument a request.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def handle_request(request):
    with tracer.start_as_current_span("gateway_handle") as span:
        span.set_attribute("request.id", request.id)
        span.set_attribute("model", request.model)
        
        # 1. Auth
        with tracer.start_as_current_span("auth_check"):
            check_auth(request)
            
        # 2. Router
        with tracer.start_as_current_span("router_select"):
            replica = select_replica(request)
            
        # 3. Inference (Remote Call)
        with tracer.start_as_current_span("model_inference"):
            response = call_replica(replica, request)
            
        return response
```

### 6. Rate Limiter (Token Bucket)

```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate # tokens per second
        self.last_refill = time.time()
        
    def _refill(self):
        now = time.time()
        delta = now - self.last_refill
        new_tokens = delta * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
        
    def consume(self, tokens):
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Usage
limiter = TokenBucket(capacity=10, refill_rate=1) # 1 req/sec burst 10
for i in range(15):
    if limiter.consume(1):
        print(f"Request {i}: Allowed")
    else:
        print(f"Request {i}: Throttled")
```
