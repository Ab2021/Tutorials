# Day 68: A/B Testing & Online Evaluation
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Traffic Router Implementation (Hashing)

Deterministic routing based on User ID.

```python
import hashlib

class ABRouter:
    def __init__(self, salt="experiment_v1"):
        self.salt = salt
        
    def get_variant(self, user_id):
        """
        Returns 'A' (Control) or 'B' (Treatment).
        """
        # Create hash of UserID + Salt
        hash_input = f"{user_id}{self.salt}".encode('utf-8')
        hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
        
        # Modulo 100 to get bucket (0-99)
        bucket = hash_val % 100
        
        # 50/50 Split
        if bucket < 50:
            return 'A'
        else:
            return 'B'

# Usage
router = ABRouter()
print(f"User 123 -> {router.get_variant('123')}")
print(f"User 456 -> {router.get_variant('456')}")
# Deterministic: User 123 will always get the same variant for this salt
```

### 2. Thompson Sampling (Multi-Armed Bandit)

Adaptive routing logic.

```python
import numpy as np

class ThompsonSampler:
    def __init__(self, n_arms=2):
        # Alpha = Successes + 1, Beta = Failures + 1
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        
    def select_arm(self):
        """Sample from Beta distribution for each arm."""
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """
        arm: index (0 or 1)
        reward: 1 (success) or 0 (failure)
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Simulation
bandit = ThompsonSampler()
true_conversion_rates = [0.10, 0.15] # Arm 1 is better

print("Running Bandit...")
for i in range(1000):
    arm = bandit.select_arm()
    # Simulate user interaction
    reward = 1 if np.random.random() < true_conversion_rates[arm] else 0
    bandit.update(arm, reward)

print(f"Estimated Rates: {bandit.alpha / (bandit.alpha + bandit.beta)}")
# Should converge to [0.10, 0.15] and select Arm 1 more often
```

### 3. Online Metric Logger (Implicit Feedback)

Calculating "Acceptance Rate" for code completion.

```python
class CopilotLogger:
    def __init__(self):
        self.logs = []
        
    def log_suggestion(self, request_id, suggestion, user_id, model_version):
        self.logs.append({
            "type": "suggestion",
            "id": request_id,
            "text": suggestion,
            "user": user_id,
            "model": model_version,
            "timestamp": time.time()
        })
        
    def log_acceptance(self, request_id, accepted_char_count):
        """
        accepted_char_count: How many chars of the suggestion did the user keep?
        """
        # Find the suggestion
        suggestion = next((x for x in self.logs if x['id'] == request_id), None)
        if suggestion:
            total_len = len(suggestion['text'])
            rate = accepted_char_count / total_len if total_len > 0 else 0
            print(f"Request {request_id}: Acceptance Rate {rate:.2f}")
            
            # Log metric
            # mlflow.log_metric("acceptance_rate", rate)

# Usage
logger = CopilotLogger()
logger.log_suggestion("req1", "def hello():", "u1", "v1")
logger.log_acceptance("req1", 12) # User kept "def hello():" -> 100%
```

### 4. LLM-as-a-Judge (Online Sampling)

Using GPT-4 to evaluate production samples.

```python
import openai

def evaluate_pair(prompt, response_a, response_b):
    eval_prompt = f"""
    User Query: {prompt}
    
    Response A: {response_a}
    
    Response B: {response_b}
    
    Which response is better? Reply with 'A' or 'B'.
    """
    
    # Call Judge
    # completion = openai.ChatCompletion.create(...)
    # return completion.choices[0].message.content
    return "B" # Mock

# Sampling Loop
def run_online_eval(production_logs, sample_size=100):
    samples = random.sample(production_logs, sample_size)
    wins_a = 0
    wins_b = 0
    
    for log in samples:
        winner = evaluate_pair(log['prompt'], log['response_a'], log['response_b'])
        if winner == 'A': wins_a += 1
        else: wins_b += 1
        
    print(f"Win Rate - A: {wins_a}%, B: {wins_b}%")
```

### 5. Interleaving Logic (Ranking)

Merging two lists for unbiased evaluation.

```python
def interleave_results(list_a, list_b):
    """
    Team Draft Interleaving.
    """
    result = []
    source_map = {} # doc_id -> 'A' or 'B'
    
    idx_a, idx_b = 0, 0
    
    while idx_a < len(list_a) and idx_b < len(list_b):
        # Pick from A
        if list_a[idx_a] not in result:
            result.append(list_a[idx_a])
            source_map[list_a[idx_a]] = 'A'
        idx_a += 1
        
        # Pick from B
        if list_b[idx_b] not in result:
            result.append(list_b[idx_b])
            source_map[list_b[idx_b]] = 'B'
        idx_b += 1
        
    return result, source_map

# Usage
res_a = ["doc1", "doc2", "doc3"]
res_b = ["doc2", "doc4", "doc5"]
merged, sources = interleave_results(res_a, res_b)
print("Merged:", merged)
print("Sources:", sources)
# If user clicks 'doc4', Model B gets a point.
```
