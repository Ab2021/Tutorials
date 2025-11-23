# Day 45: Inference Optimization Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. GPTQ Quantization Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Load model with GPTQ quantization
quantization_config = GPTQConfig(
    bits=4,  # 4-bit quantization
    dataset="c4",  # Calibration dataset
    tokenizer="meta-llama/Llama-2-7b-hf",
    group_size=128,  # Quantization group size
    desc_act=True  # Use activation order
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Model is now 4x smaller: 28 GB → 7 GB
```

**Manual GPTQ Implementation:**
```python
def gptq_quantize_layer(weight, inputs, bits=4):
    """Quantize a single layer using GPTQ algorithm."""
    # Compute Hessian (second-order information)
    H = torch.matmul(inputs.T, inputs) / inputs.shape[0]
    H_inv = torch.inverse(H + 1e-6 * torch.eye(H.shape[0]))
    
    # Quantize weights column by column
    quantized_weight = torch.zeros_like(weight)
    error = torch.zeros_like(weight)
    
    for col in range(weight.shape[1]):
        # Quantize column
        w_col = weight[:, col] + error[:, col]
        q_col = quantize_to_bits(w_col, bits)
        quantized_weight[:, col] = q_col
        
        # Compute error
        err = w_col - q_col
        
        # Propagate error to remaining columns
        error[:, col+1:] -= torch.outer(err, H_inv[col, col+1:]) / H_inv[col, col]
    
    return quantized_weight

def quantize_to_bits(tensor, bits):
    """Quantize tensor to specified bits."""
    n_levels = 2 ** bits
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (n_levels - 1)
    
    quantized = torch.round((tensor - min_val) / scale)
    dequantized = quantized * scale + min_val
    
    return dequantized
```

### 2. Flash Attention Implementation

```python
import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, block_size=64):
    """
    Flash Attention: Memory-efficient attention.
    
    Standard attention: O(N²) memory
    Flash attention: O(N) memory
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Output and normalization accumulators
    O = torch.zeros_like(Q)
    l = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)
    m = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=Q.device)
    
    # Process in blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for i in range(num_blocks):
        # Query block
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        Q_block = Q[:, :, q_start:q_end, :]
        
        for j in range(num_blocks):
            # Key/Value block
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, seq_len)
            K_block = K[:, :, kv_start:kv_end, :]
            V_block = V[:, :, kv_start:kv_end, :]
            
            # Compute attention scores for this block
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # Online softmax (numerically stable)
            m_new = torch.maximum(m[:, :, q_start:q_end, :], scores.max(dim=-1, keepdim=True)[0])
            
            # Update normalization
            l_new = torch.exp(m[:, :, q_start:q_end, :] - m_new) * l[:, :, q_start:q_end, :] + \
                    torch.sum(torch.exp(scores - m_new), dim=-1, keepdim=True)
            
            # Update output
            O[:, :, q_start:q_end, :] = \
                (O[:, :, q_start:q_end, :] * torch.exp(m[:, :, q_start:q_end, :] - m_new) * l[:, :, q_start:q_end, :] + \
                 torch.matmul(torch.exp(scores - m_new), V_block)) / l_new
            
            # Update accumulators
            m[:, :, q_start:q_end, :] = m_new
            l[:, :, q_start:q_end, :] = l_new
    
    return O
```

### 3. Prefix Caching System

```python
import hashlib
from typing import Dict, Tuple

class PrefixCache:
    def __init__(self, max_cache_size=1000):
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get_cache_key(self, prefix: str) -> str:
        """Generate cache key from prefix."""
        return hashlib.md5(prefix.encode()).hexdigest()
    
    def get(self, prefix: str):
        """Get cached KV for prefix."""
        key = self.get_cache_key(prefix)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, prefix: str, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Store KV cache for prefix."""
        if len(self.cache) >= self.max_cache_size:
            # Evict least frequently used
            lfu_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lfu_key]
            del self.access_count[lfu_key]
        
        key = self.get_cache_key(prefix)
        self.cache[key] = (k_cache, v_cache)
        self.access_count[key] = 1
    
    def generate_with_cache(self, model, prompt: str, system_prompt: str = ""):
        """Generate using cached KV for system prompt."""
        # Check if system prompt is cached
        cached_kv = self.get(system_prompt) if system_prompt else None
        
        if cached_kv:
            k_cache, v_cache = cached_kv
            # Only process user prompt
            output = model.generate(
                prompt,
                past_key_values=(k_cache, v_cache)
            )
        else:
            # Process full prompt
            full_prompt = system_prompt + prompt if system_prompt else prompt
            output, k_cache, v_cache = model.generate_with_kv(full_prompt)
            
            # Cache system prompt KV
            if system_prompt:
                # Extract KV for system prompt only
                sys_len = len(model.tokenizer.encode(system_prompt))
                sys_k = k_cache[:, :, :sys_len, :]
                sys_v = v_cache[:, :, :sys_len, :]
                self.put(system_prompt, sys_k, sys_v)
        
        return output
```

### 4. Semantic Caching

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = {}  # embedding -> response
        self.embeddings = []
        self.responses = []
        self.threshold = similarity_threshold
    
    def get(self, query: str):
        """Get cached response if semantically similar query exists."""
        if not self.embeddings:
            return None
        
        # Embed query
        query_emb = self.embedder.encode(query)
        
        # Find most similar cached query
        similarities = np.dot(self.embeddings, query_emb)
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        # Return cached response if above threshold
        if max_sim >= self.threshold:
            return self.responses[max_sim_idx]
        
        return None
    
    def put(self, query: str, response: str):
        """Cache query-response pair."""
        query_emb = self.embedder.encode(query)
        self.embeddings.append(query_emb)
        self.responses.append(response)
```

### 5. KV Cache Quantization

```python
def quantize_kv_cache(k_cache, v_cache, bits=8):
    """Quantize KV cache to reduce memory."""
    def quantize_tensor(tensor, bits):
        n_levels = 2 ** bits - 1
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / n_levels
        zero_point = min_val
        
        # Quantize
        quantized = torch.round((tensor - zero_point) / scale).clamp(0, n_levels)
        
        return quantized.to(torch.uint8), scale, zero_point
    
    # Quantize K and V separately
    k_quant, k_scale, k_zero = quantize_tensor(k_cache, bits)
    v_quant, v_scale, v_zero = quantize_tensor(v_cache, bits)
    
    return (k_quant, k_scale, k_zero), (v_quant, v_scale, v_zero)

def dequantize_kv_cache(k_quant_data, v_quant_data):
    """Dequantize KV cache."""
    k_quant, k_scale, k_zero = k_quant_data
    v_quant, v_scale, v_zero = v_quant_data
    
    k_cache = k_quant.float() * k_scale + k_zero
    v_cache = v_quant.float() * v_scale + v_zero
    
    return k_cache, v_cache
```

### 6. Multi-Query Attention

```python
class MultiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q has multiple heads, K/V are shared
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, self.head_dim)  # Single head
        self.v_proj = torch.nn.Linear(hidden_size, self.head_dim)  # Single head
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q (multi-head)
        Q = self.q_proj(hidden_states)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        
        # Project K, V (single head, then broadcast)
        K = self.k_proj(hidden_states)  # (batch, seq, head_dim)
        V = self.v_proj(hidden_states)
        K = K.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # Broadcast to all heads
        V = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output
```
