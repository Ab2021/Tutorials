# Day 70: Cost Management & FinOps
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Model Cascade Router Implementation

Routing requests based on complexity/confidence.

```python
import random

class ModelCascade:
    def __init__(self):
        self.cheap_model = MockModel("Llama-3-8B", cost=0.0002)
        self.expensive_model = MockModel("GPT-4", cost=0.03)
        
    def process_request(self, prompt):
        # 1. Try Cheap Model
        response, confidence = self.cheap_model.generate(prompt)
        
        # 2. Check Confidence
        if confidence > 0.8:
            print(f"Served by {self.cheap_model.name} (Conf: {confidence:.2f})")
            return response
            
        # 3. Fallback to Expensive Model
        print(f"Fallback to {self.expensive_model.name} (Conf: {confidence:.2f})")
        response, _ = self.expensive_model.generate(prompt)
        return response

class MockModel:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost
        
    def generate(self, prompt):
        # Simulate confidence based on prompt length (heuristic)
        # Longer prompts might be harder? Or random.
        confidence = random.random()
        return "Response", confidence

# Usage
cascade = ModelCascade()
for _ in range(5):
    cascade.process_request("Hello")
```

### 2. Token Budget Manager

Rate limiting based on *Cost* rather than *Requests*.

```python
class CostLimiter:
    def __init__(self, daily_budget_usd):
        self.daily_budget = daily_budget_usd
        self.current_spend = 0.0
        
    def check_budget(self, estimated_cost):
        if self.current_spend + estimated_cost > self.daily_budget:
            raise Exception("Daily Budget Exceeded")
        return True
        
    def add_spend(self, cost):
        self.current_spend += cost
        print(f"Current Spend: ${self.current_spend:.4f} / ${self.daily_budget}")

    def estimate_cost(self, model, input_tokens, output_tokens):
        # Pricing table (per 1k tokens)
        pricing = {
            "gpt-4": {"in": 0.03, "out": 0.06},
            "gpt-3.5": {"in": 0.0005, "out": 0.0015}
        }
        
        p = pricing.get(model, {"in": 0, "out": 0})
        cost = (input_tokens / 1000 * p['in']) + (output_tokens / 1000 * p['out'])
        return cost

# Usage
limiter = CostLimiter(daily_budget_usd=1.0)
try:
    cost = limiter.estimate_cost("gpt-4", 1000, 500)
    if limiter.check_budget(cost):
        # Run Model...
        limiter.add_spend(cost)
except Exception as e:
    print(e)
```

### 3. Spot Instance Interruption Handler

Simulating graceful shutdown on Spot termination warning.

```python
import signal
import time
import sys

class SpotWorker:
    def __init__(self):
        self.running = True
        # Register signal handler for SIGTERM (Spot interruption warning)
        signal.signal(signal.SIGTERM, self.handle_interruption)
        
    def handle_interruption(self, signum, frame):
        print("\nReceived Spot Interruption Warning! Checkpointing...")
        self.checkpoint()
        self.running = False
        sys.exit(0)
        
    def checkpoint(self):
        print("Saving state to S3...")
        time.sleep(1) # Simulate upload
        print("Checkpoint saved.")
        
    def run(self):
        print("Worker started. Processing jobs...")
        while self.running:
            print(".", end="", flush=True)
            time.sleep(0.5)
            # Simulate work

# Usage
# Run this, then send SIGTERM (kill -15 pid) to simulate spot interruption
# worker = SpotWorker()
# worker.run()
```

### 4. Prompt Compression (LLMlingua Concept)

Removing non-essential tokens.

```python
def simple_compress(prompt, ratio=0.5):
    """
    Naive compression: Remove stop words or low-entropy words.
    Real LLMlingua uses a small model to calculate perplexity of each token.
    """
    words = prompt.split()
    n_keep = int(len(words) * ratio)
    
    # Keep first few words (context) and important keywords
    # This is just a dummy implementation
    compressed = " ".join(words[:n_keep]) + "..."
    return compressed

prompt = "I want you to act as a very helpful assistant that knows everything about history."
print(f"Original: {prompt}")
print(f"Compressed: {simple_compress(prompt)}")
```

### 5. Cost Estimator Script

Calculating ROI of Fine-tuning vs Prompting.

```python
def calculate_roi(
    req_per_day,
    prompt_tokens,
    gen_tokens,
    base_model_cost_in, # per 1k
    base_model_cost_out,
    ft_model_cost_in,
    ft_model_cost_out,
    ft_training_cost
):
    daily_tokens_in = req_per_day * prompt_tokens
    daily_tokens_out = req_per_day * gen_tokens
    
    daily_cost_base = (daily_tokens_in/1000 * base_model_cost_in) + \
                      (daily_tokens_out/1000 * base_model_cost_out)
                      
    daily_cost_ft = (daily_tokens_in/1000 * ft_model_cost_in) + \
                    (daily_tokens_out/1000 * ft_model_cost_out)
                    
    savings_per_day = daily_cost_base - daily_cost_ft
    
    if savings_per_day <= 0:
        return "Fine-tuning is more expensive per request."
        
    break_even_days = ft_training_cost / savings_per_day
    
    return {
        "daily_cost_base": daily_cost_base,
        "daily_cost_ft": daily_cost_ft,
        "savings_per_day": savings_per_day,
        "break_even_days": break_even_days
    }

# Example: GPT-4 vs Fine-tuned Llama-3
res = calculate_roi(
    req_per_day=10000,
    prompt_tokens=1000,
    gen_tokens=200,
    base_model_cost_in=0.03, base_model_cost_out=0.06, # GPT-4
    ft_model_cost_in=0.0002, ft_model_cost_out=0.0002, # Llama-3 Self-hosted
    ft_training_cost=500 # One-time training cost
)
print(res)
```
