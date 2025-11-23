# Lab 2: Speculative Decoding

## Objective
Speed up inference by 2x using a Draft Model.
Draft Model (Small) guesses tokens. Target Model (Large) verifies them.

## 1. The Logic (`speculative.py`)

```python
import torch
import time

# Mocking the models for demonstration
# In reality, load 'opt-125m' (Draft) and 'opt-1.3b' (Target)

def draft_model_predict(context):
    # Fast, low quality
    time.sleep(0.01) 
    return "cat"

def target_model_verify(context, candidate):
    # Slow, high quality
    time.sleep(0.1)
    return True if candidate == "cat" else False

def standard_generation(n_tokens):
    start = time.time()
    for _ in range(n_tokens):
        target_model_verify("", "") # Simulate full forward pass
    return time.time() - start

def speculative_generation(n_tokens):
    start = time.time()
    # Draft generates 5 tokens (0.05s)
    # Target verifies 5 tokens in 1 parallel pass (0.1s)
    # Total = 0.15s for 5 tokens.
    # Standard = 0.5s for 5 tokens.
    
    # Simulation
    steps = n_tokens // 5
    for _ in range(steps):
        time.sleep(0.01 * 5) # Draft
        time.sleep(0.1)      # Verify
    return time.time() - start

print(f"Standard: {standard_generation(10):.2f}s")
print(f"Speculative: {speculative_generation(10):.2f}s")
```

## 2. Real Implementation
Use vLLM's built-in speculative decoding:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --speculative-model facebook/opt-125m \
    --num-speculative-tokens 5
```

## 3. Submission
Submit the speedup ratio (e.g., "1.5x faster").
