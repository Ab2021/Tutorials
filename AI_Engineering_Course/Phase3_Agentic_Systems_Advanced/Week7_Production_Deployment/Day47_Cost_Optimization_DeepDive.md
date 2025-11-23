# Day 47: Cost Optimization Strategies
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Intelligent Model Routing

```python
from transformers import pipeline
import openai

class CostOptimizedRouter:
    def __init__(self):
        # Complexity classifier (small, fast model)
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.models = {
            "simple": {"name": "gpt-3.5-turbo", "cost_per_1k": 0.002},
            "medium": {"name": "self-hosted-13b", "cost_per_1k": 0.0005},
            "complex": {"name": "gpt-4", "cost_per_1k": 0.06}
        }
    
    def classify_complexity(self, prompt: str) -> str:
        """Classify query complexity."""
        # Heuristics
        word_count = len(prompt.split())
        has_reasoning = any(word in prompt.lower() for word in ["explain", "analyze", "compare"])
        has_creation = any(word in prompt.lower() for word in ["write", "create", "generate"])
        
        if word_count < 20 and not has_reasoning:
            return "simple"
        elif has_creation or word_count > 100:
            return "complex"
        else:
            return "medium"
    
    def generate(self, prompt: str):
        """Generate with cost-optimized model selection."""
        complexity = self.classify_complexity(prompt)
        model_config = self.models[complexity]
        
        # Log routing decision
        print(f"Routing to {model_config['name']} (complexity: {complexity})")
        
        # Generate
        if "gpt" in model_config["name"]:
            response = openai.ChatCompletion.create(
                model=model_config["name"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            # Self-hosted model
            return self.self_hosted_generate(prompt)
```

### 2. Multi-Level Caching System

```python
import hashlib
import redis
from sentence_transformers import SentenceTransformer

class MultiLevelCache:
    def __init__(self):
        # L1: In-memory cache (exact match)
        self.l1_cache = {}
        
        # L2: Redis cache (exact match, persistent)
        self.l2_cache = redis.Redis(host='localhost', port=6379, db=0)
        
        # L3: Semantic cache (similar queries)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_cache = {}  # embedding -> response
        self.embeddings = []
        self.responses = []
    
    def get(self, prompt: str):
        """Get cached response (check all levels)."""
        # L1: Exact match (fastest)
        if prompt in self.l1_cache:
            print("L1 cache hit")
            return self.l1_cache[prompt]
        
        # L2: Redis exact match
        cached = self.l2_cache.get(prompt)
        if cached:
            print("L2 cache hit")
            response = cached.decode('utf-8')
            self.l1_cache[prompt] = response  # Promote to L1
            return response
        
        # L3: Semantic match
        semantic_result = self._semantic_search(prompt)
        if semantic_result:
            print("L3 cache hit (semantic)")
            return semantic_result
        
        return None
    
    def put(self, prompt: str, response: str):
        """Store in all cache levels."""
        # L1: In-memory
        self.l1_cache[prompt] = response
        
        # L2: Redis (persistent)
        self.l2_cache.setex(prompt, 86400, response)  # 24 hour TTL
        
        # L3: Semantic cache
        embedding = self.embedder.encode(prompt)
        self.embeddings.append(embedding)
        self.responses.append(response)
    
    def _semantic_search(self, prompt: str, threshold=0.95):
        """Search for semantically similar cached query."""
        if not self.embeddings:
            return None
        
        query_emb = self.embedder.encode(prompt)
        similarities = np.dot(self.embeddings, query_emb)
        
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= threshold:
            return self.responses[max_idx]
        
        return None
```

### 3. Prompt Compression

```python
class PromptCompressor:
    def __init__(self):
        self.abbreviations = {
            "information": "info",
            "please": "",
            "could you": "",
            "I would like": "",
            "detailed": "",
            "comprehensive": ""
        }
    
    def compress(self, prompt: str) -> str:
        """Compress prompt to reduce tokens."""
        compressed = prompt
        
        # Remove filler words
        for full, abbrev in self.abbreviations.items():
            compressed = compressed.replace(full, abbrev)
        
        # Remove extra whitespace
        compressed = " ".join(compressed.split())
        
        # Remove politeness
        compressed = compressed.replace("Please ", "")
        compressed = compressed.replace("Thank you", "")
        
        return compressed
    
    def compress_system_prompt(self, system_prompt: str) -> str:
        """Aggressively compress system prompt."""
        # Extract key instructions only
        lines = system_prompt.split(".")
        key_lines = [l for l in lines if any(word in l.lower() for word in ["must", "should", "always", "never"])]
        
        return ". ".join(key_lines)

# Example
compressor = PromptCompressor()
original = "Please provide a detailed and comprehensive answer to: What is AI?"
compressed = compressor.compress(original)
# "What is AI?"
# Savings: 70% tokens
```

### 4. Batch Processing for Cost Efficiency

```python
class BatchProcessor:
    def __init__(self, model, batch_size=10, max_wait_time=5):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = []
        self.results = {}
    
    async def process(self, prompt: str, request_id: str):
        """Add to batch queue and wait for result."""
        self.queue.append({"prompt": prompt, "id": request_id})
        
        # Wait for batch to fill or timeout
        start_time = time.time()
        while len(self.queue) < self.batch_size and time.time() - start_time < self.max_wait_time:
            await asyncio.sleep(0.1)
        
        # Process batch
        if self.queue:
            await self._process_batch()
        
        # Return result
        return self.results.get(request_id)
    
    async def _process_batch(self):
        """Process entire batch in single request."""
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        # Create batch prompt
        batch_prompt = "\n\n".join([
            f"Query {i+1}: {item['prompt']}"
            for i, item in enumerate(batch)
        ])
        
        # Single API call for entire batch
        response = await self.model.generate(batch_prompt)
        
        # Parse responses
        responses = response.split("\n\n")
        for item, resp in zip(batch, responses):
            self.results[item['id']] = resp
```

### 5. Cost Tracking and Budgeting

```python
from prometheus_client import Counter, Gauge

class CostTracker:
    def __init__(self):
        self.cost_total = Counter('llm_cost_total_dollars', 'Total cost', ['model', 'user'])
        self.daily_cost = Gauge('llm_daily_cost_dollars', 'Daily cost')
        self.budget_remaining = Gauge('llm_budget_remaining_dollars', 'Budget remaining')
        
        self.daily_budget = 1000  # $1000/day
        self.monthly_budget = 20000  # $20k/month
        self.user_limits = {}  # user_id -> limit
        
        self.daily_spend = 0
        self.monthly_spend = 0
        self.user_spend = {}
    
    def record_cost(self, user_id: str, model: str, input_tokens: int, output_tokens: int):
        """Record cost and check budgets."""
        # Calculate cost
        pricing = self.get_pricing(model)
        cost = (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']
        
        # Update metrics
        self.cost_total.labels(model=model, user=user_id).inc(cost)
        self.daily_spend += cost
        self.monthly_spend += cost
        self.user_spend[user_id] = self.user_spend.get(user_id, 0) + cost
        
        self.daily_cost.set(self.daily_spend)
        self.budget_remaining.set(self.daily_budget - self.daily_spend)
        
        # Check budgets
        self._check_budgets(user_id, cost)
    
    def _check_budgets(self, user_id: str, cost: float):
        """Check if budgets are exceeded."""
        # Daily budget
        if self.daily_spend > self.daily_budget:
            raise BudgetExceededError(f"Daily budget exceeded: ${self.daily_spend:.2f}")
        
        # User limit
        user_limit = self.user_limits.get(user_id, 100)
        if self.user_spend.get(user_id, 0) > user_limit:
            raise BudgetExceededError(f"User {user_id} exceeded limit: ${self.user_spend[user_id]:.2f}")
    
    def get_pricing(self, model: str):
        """Get pricing for model."""
        pricing_table = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "self-hosted-7b": {"input": 0.0001, "output": 0.0001}
        }
        return pricing_table.get(model, {"input": 0, "output": 0})
```

### 6. Model Distillation for Cost Reduction

```python
# Generate training data with expensive model
def generate_distillation_dataset(prompts, teacher_model="gpt-4"):
    dataset = []
    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model=teacher_model,
            messages=[{"role": "user", "content": prompt}]
        )
        dataset.append({
            "prompt": prompt,
            "response": response.choices[0].message.content
        })
    return dataset

# Fine-tune cheaper model on this data
from transformers import Trainer, TrainingArguments

def distill_model(dataset, student_model="gpt-3.5-turbo"):
    # Fine-tune student on teacher's outputs
    training_args = TrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=8
    )
    
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train()
    
    # Now use distilled model (30x cheaper, 80-90% quality)
```
