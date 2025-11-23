# Day 61: Caching & Speculative Decoding
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Semantic Cache Implementation (Redis + Embeddings)

A robust caching layer can save 30-50% of LLM calls in production.

```python
import redis
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, redis_host='localhost', threshold=0.9):
        self.redis = redis.Redis(host=redis_host, port=6379, db=0)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.vector_dim = 384
        
    def _get_embedding(self, text):
        return self.model.encode(text)
    
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get(self, prompt):
        """Retrieve cached response if semantic match found."""
        query_vec = self._get_embedding(prompt)
        
        # In production, use Redis Vector Search (RediSearch)
        # Here, we iterate keys for demonstration (slow for large DB)
        keys = self.redis.keys("cache:*")
        
        best_sim = -1
        best_response = None
        
        for key in keys:
            data = json.loads(self.redis.get(key))
            cached_vec = np.array(data['embedding'])
            sim = self._cosine_similarity(query_vec, cached_vec)
            
            if sim > best_sim:
                best_sim = sim
                best_response = data['response']
        
        if best_sim >= self.threshold:
            print(f"Cache Hit! Similarity: {best_sim:.4f}")
            return best_response
        
        print("Cache Miss.")
        return None
    
    def set(self, prompt, response):
        """Store prompt and response."""
        vec = self._get_embedding(prompt)
        key = f"cache:{hash(prompt)}"
        data = {
            'prompt': prompt,
            'response': response,
            'embedding': vec.tolist()
        }
        self.redis.set(key, json.dumps(data))

# Usage
cache = SemanticCache()
# cache.set("What is the capital of France?", "Paris.")
# print(cache.get("Tell me the capital of France")) # Should hit
```

### 2. Speculative Decoding Verification Logic

Implementing the acceptance logic (Rejection Sampling / Greedy).

```python
import torch
import torch.nn.functional as F

def speculative_verify(
    target_model, 
    input_ids, 
    draft_tokens, 
    draft_probs=None, 
    temperature=1.0
):
    """
    input_ids: [batch, seq_len]
    draft_tokens: [batch, K]
    """
    K = draft_tokens.shape[1]
    
    # 1. Run Target Model on [input, draft]
    # We need logits for the positions corresponding to draft tokens
    full_input = torch.cat([input_ids, draft_tokens], dim=1)
    
    with torch.no_grad():
        target_logits = target_model(full_input).logits
        
    # Target logits for the draft positions
    # We predict t_{i} using t_{0...i-1}
    # Logits index starts at seq_len - 1
    start_idx = input_ids.shape[1] - 1
    relevant_logits = target_logits[:, start_idx : start_idx + K, :]
    
    accepted_tokens = []
    
    for i in range(K):
        draft_token = draft_tokens[0, i]
        logits = relevant_logits[0, i, :]
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Greedy Verification
        target_token = torch.argmax(probs)
        
        if target_token == draft_token:
            accepted_tokens.append(target_token)
        else:
            # Rejection
            # We take the target's token as the correction
            accepted_tokens.append(target_token)
            break
            
    # If all accepted, we get one bonus token from the last position
    if len(accepted_tokens) == K:
        last_logits = target_logits[:, -1, :]
        bonus_token = torch.argmax(last_logits)
        accepted_tokens.append(bonus_token)
        
    return torch.tensor(accepted_tokens).unsqueeze(0)

```

### 3. Prompt Lookup Decoding (N-gram Matcher)

A simple draft strategy that requires no model.

```python
def prompt_lookup_draft(input_ids, num_draft=3):
    """
    Look for the last n-gram in the input and copy subsequent tokens.
    """
    input_list = input_ids[0].tolist()
    L = len(input_list)
    
    # Try 2-gram match
    if L < 2: return []
    
    last_2gram = input_list[-2:]
    
    # Search backwards
    matches = []
    for i in range(L - 3, -1, -1):
        if input_list[i:i+2] == last_2gram:
            # Found match, suggest next tokens
            start = i + 2
            end = min(start + num_draft, L) # Wait, we want new tokens?
            # Actually, we want to copy what followed this n-gram previously
            # But usually we copy from context.
            # Let's assume we copy up to num_draft tokens
            available = input_list[start : start + num_draft]
            if available:
                return torch.tensor([available])
            
    return torch.tensor([[]]) # No match
```

### 4. KV Cache Eviction (StreamingLLM Logic)

Simulating the "Attention Sink" retention policy.

```python
class StreamingKVCache:
    def __init__(self, max_size=100, sink_size=4):
        self.max_size = max_size
        self.sink_size = sink_size
        self.cache = [] # List of (K, V) tuples
        
    def update(self, new_kv):
        """
        new_kv: (K, V) for new token
        """
        self.cache.append(new_kv)
        
        if len(self.cache) > self.max_size:
            self.evict()
            
    def evict(self):
        # Policy: Keep [0:sink_size] (Attention Sinks)
        # Keep [-(max_size - sink_size):] (Rolling Window)
        # Evict the middle
        
        num_to_keep_rolling = self.max_size - self.sink_size
        
        sinks = self.cache[:self.sink_size]
        rolling = self.cache[-num_to_keep_rolling:]
        
        self.cache = sinks + rolling
        # In real implementation, we manipulate tensors indices
        
    def get_cache(self):
        return self.cache

# Why this works:
# The first few tokens (sinks) accumulate massive attention scores.
# If evicted, the model collapses (perplexity explosion).
# Keeping them stabilizes the distribution, even if they are semantically irrelevant.
```

### 5. Radix Tree for Prefix Caching (Conceptual)

```python
class RadixNode:
    def __init__(self):
        self.children = {} # token_id -> RadixNode
        self.kv_block_id = None # Points to GPU memory block
        self.ref_count = 0

class PrefixCache:
    def __init__(self):
        self.root = RadixNode()
        
    def insert(self, token_ids, block_id):
        node = self.root
        for token in token_ids:
            if token not in node.children:
                node.children[token] = RadixNode()
            node = node.children[token]
        
        node.kv_block_id = block_id
        node.ref_count += 1
        
    def match_prefix(self, token_ids):
        """Find longest matching prefix."""
        node = self.root
        matched_tokens = []
        last_valid_block = None
        
        for token in token_ids:
            if token in node.children:
                node = node.children[token]
                matched_tokens.append(token)
                if node.kv_block_id is not None:
                    last_valid_block = node.kv_block_id
            else:
                break
                
        return matched_tokens, last_valid_block

# Usage in vLLM:
# When a request comes, traverse the tree.
# If match found, initialize the request's page table with the cached block IDs.
# Skip computation for matched tokens.
```

### 6. Benchmarking Speculative Decoding

To measure the speedup:

```python
def benchmark_speculative(target, draft, prompt):
    # 1. Standard Generation
    start = time.time()
    out_std = target.generate(prompt, max_new_tokens=50)
    time_std = time.time() - start
    
    # 2. Speculative Generation
    start = time.time()
    out_spec = speculative_generate(target, draft, prompt, max_new_tokens=50)
    time_spec = time.time() - start
    
    speedup = time_std / time_spec
    print(f"Standard: {time_std:.2f}s, Speculative: {time_spec:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
```
