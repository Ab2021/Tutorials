# 🤖 LLM & RAG Coding — Part 4: Advanced LLM Topics
> **Difficulty:** Hard | **Focus:** LoRA, Quantization, RLHF, Agentic Patterns, Flash Attention
> **Flipkart Relevance:** 🔥🔥🔥 — LLM fine-tuning, inference optimization, production agents

---

## 🔴 PROBLEM 1: LoRA — Low-Rank Adaptation from Scratch

### Theory
Fine-tuning all parameters of a 70B LLM is prohibitively expensive. LoRA adds small **low-rank trainable matrices** to frozen pre-trained weights.

**Original weight update:** $\Delta W \in \mathbb{R}^{d \times k}$ — d×k parameters to learn.

**LoRA decomposition:**
$$\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

**Modified forward pass:**
$$h = W_0 x + \frac{\alpha}{r} B A x$$

- W₀: frozen pre-trained weights
- B, A: trainable LoRA adapters
- r: rank (4, 8, 16 typical)
- α: scaling factor (usually equal to r)

**Parameter reduction:**
- Full fine-tuning: d×k parameters
- LoRA: r×(d+k) parameters
- For d=k=4096, r=8: 134M → 65K (2000× reduction!)

### Things to Focus On
- ✅ Only B and A are trained — W₀ stays frozen
- ✅ B initialized to zeros; A initialized with Gaussian → initial output = 0 (stable start)
- ✅ At inference: merge W_merged = W₀ + (α/r) × B × A → no inference latency!
- ✅ QLoRA: quantize W₀ to 4-bit → even smaller GPU footprint
- ✅ Applied to: Q, K, V projection matrices in attention (most impactful)

### Implementation
```python
import numpy as np
from typing import List

class LoRALinear:
    """
    Linear layer with LoRA adaptation.
    
    Forward: h = W₀x + (α/r) * B * A * x
    Training: only A, B trained; W₀ frozen
    """
    
    def __init__(self, d_in: int, d_out: int, r: int = 8, alpha: int = None,
                 pretrained_weights: np.ndarray = None):
        """
        d_in:  input dimension
        d_out: output dimension
        r:     LoRA rank (number of low-rank dimensions)
        alpha: LoRA scaling (default = r)
        """
        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r
        
        # === FROZEN pre-trained weights ===
        if pretrained_weights is not None:
            self.W0 = pretrained_weights.copy()
        else:
            # Xavier initialization for base weights
            std = np.sqrt(2 / (d_in + d_out))
            self.W0 = np.random.randn(d_in, d_out) * std
        
        # === TRAINABLE LoRA adapters ===
        # A: initialized with Gaussian (random)
        self.A = np.random.randn(d_in, r) * 0.02   # (d_in, r)
        # B: initialized to ZERO → initial ΔW = 0 → no change from pretrained
        self.B = np.zeros((r, d_out))               # (r, d_out)
        
        # Bias (always trainable if present)
        self.bias = np.zeros(d_out)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with LoRA.
        x: (..., d_in)
        Returns: (..., d_out)
        """
        # Base model (frozen)
        base_out = x @ self.W0  # (..., d_out)
        
        # LoRA adaptation
        lora_out = (x @ self.A) @ self.B  # (..., d_out)  via low-rank path
        
        return base_out + self.scaling * lora_out + self.bias
    
    def merge_weights(self) -> np.ndarray:
        """
        Merge LoRA weights into W0 for deployment.
        Result: W_merged = W0 + (alpha/r) * B * A^T
        This removes LoRA overhead at inference time!
        """
        # LoRA update matrix
        lora_update = self.scaling * (self.A @ self.B)  # (d_in, d_out)
        return self.W0 + lora_update
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters (only A and B)."""
        return self.A.size + self.B.size
    
    def get_total_params(self) -> int:
        return self.W0.size + self.A.size + self.B.size

class LoRAModel:
    """
    Replace linear layers in a transformer with LoRA-wrapped versions.
    """
    
    def __init__(self, d_model: int, n_heads: int, r: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Replace Q, K, V, O projections with LoRA
        self.W_Q = LoRALinear(d_model, d_model, r=r)
        self.W_K = LoRALinear(d_model, d_model, r=r)
        self.W_V = LoRALinear(d_model, d_model, r=r)
        self.W_O = LoRALinear(d_model, d_model, r=r)
    
    def trainable_params(self) -> int:
        """Only LoRA A and B matrices are trained."""
        return sum([
            self.W_Q.get_trainable_params(),
            self.W_K.get_trainable_params(),
            self.W_V.get_trainable_params(),
            self.W_O.get_trainable_params(),
        ])
    
    def total_params(self) -> int:
        return sum([
            self.W_Q.get_total_params(),
            self.W_K.get_total_params(),
            self.W_V.get_total_params(),
            self.W_O.get_total_params(),
        ])

# Test
d_model, r = 1024, 8
W_pretrained = np.random.randn(d_model, d_model) * 0.02

lora_layer = LoRALinear(d_model, d_model, r=r, pretrained_weights=W_pretrained)
x = np.random.randn(4, 20, d_model)  # batch=4, seq=20

out = lora_layer.forward(x)
print(f"Input:  {x.shape}")   # (4, 20, 1024)
print(f"Output: {out.shape}") # (4, 20, 1024)
print(f"Trainable params: {lora_layer.get_trainable_params():,}")  # 2*1024*8 = 16,384
print(f"Total params:     {lora_layer.get_total_params():,}")       # 1024*1024 + 16384 ~1M
print(f"Overhead: {lora_layer.get_trainable_params()/lora_layer.get_total_params()*100:.1f}%")

# After training: merge for zero-overhead inference
W_merged = lora_layer.merge_weights()
print(f"Merged weight shape: {W_merged.shape}")  # Same as W_pretrained ✓
```

---

## 🔴 PROBLEM 2: Quantization — INT8 Post-Training

### Theory
Quantization reduces model weight precision to save memory and speed up inference.

**FP32 → INT8 linear quantization:**
$$W_{int8} = \text{round}\left(\frac{W_{fp32}}{\text{scale}}\right), \quad \text{scale} = \frac{\max(|W|)}{127}$$

**Dequantization (for inference):**
$$W_{fp32} \approx W_{int8} \times \text{scale}$$

**Per-tensor vs Per-channel:**
- Per-tensor: one scale for entire weight matrix (less accurate)
- Per-channel: one scale per output neuron (more accurate, standard)

**Memory reduction:** FP32 (4 bytes) → INT8 (1 byte) = 4× compression.
Speed: INT8 matmul is 2-4× faster on modern GPUs/CPUs.

### Implementation
```python
def quantize_int8(W: np.ndarray, per_channel: bool = True) -> tuple:
    """
    Post-training quantization: FP32 → INT8.
    
    W: (out_dim, in_dim) weight matrix
    per_channel: if True, compute scale per output row
    
    Returns: (W_int8, scale) where W_int8 is uint8 and scale restores FP32
    """
    if per_channel:
        # One scale per output channel (row of W)
        max_vals = np.abs(W).max(axis=1, keepdims=True)  # (out_dim, 1)
        scale = max_vals / 127.0  # Map [-max, max] → [-127, 127]
    else:
        # Global scale
        max_val = np.abs(W).max()
        scale = max_val / 127.0
    
    # Quantize: round to nearest integer
    W_int8 = np.round(W / (scale + 1e-8)).astype(np.int8)
    
    # Clamp to [-128, 127] (int8 range)
    W_int8 = np.clip(W_int8, -128, 127)
    
    return W_int8, scale

def dequantize_int8(W_int8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Restore FP32 approximation from INT8."""
    return W_int8.astype(np.float32) * scale

def quantization_error(W_original: np.ndarray, 
                         W_quantized: np.ndarray) -> dict:
    """Measure quantization error."""
    diff = W_original - W_quantized
    return {
        'max_error': float(np.abs(diff).max()),
        'mean_error': float(np.abs(diff).mean()),
        'relative_error': float(np.abs(diff).mean() / (np.abs(W_original).mean() + 1e-8)),
        'memory_savings': f"{(1 - 1/4) * 100:.0f}% (4x compression)",
    }

# Test
np.random.seed(42)
W = np.random.randn(512, 512).astype(np.float32) * 0.02  # Typical transformer weight scale

W_int8, scale = quantize_int8(W, per_channel=True)
W_restored = dequantize_int8(W_int8, scale)

metrics = quantization_error(W, W_restored)
print("INT8 Post-Training Quantization:")
print(f"  Max error:      {metrics['max_error']:.6f}")
print(f"  Mean error:     {metrics['mean_error']:.6f}")
print(f"  Relative error: {metrics['relative_error']:.4%}")
print(f"  Memory: FP32 = {W.nbytes/1024:.1f}KB → INT8 = {W_int8.nbytes/1024:.1f}KB")
print(f"  Compression: {W.nbytes / W_int8.nbytes}x")
```

---

## 🔴 PROBLEM 3: Reward Model & RLHF (Conceptual Implementation)

### Theory
RLHF (Reinforcement Learning from Human Feedback) consists of 3 phases:

1. **Supervised Fine-Tuning (SFT):** Train on human-written demonstrations
2. **Reward Model (RM):** Train on human preference pairs (preferred vs rejected)
3. **PPO/GRPO:** Optimize policy using RM as reward signal

**Reward Model training:**
Given (prompt, response_good, response_bad), train RM to predict:
$$P(\text{good} > \text{bad}) = \sigma(r_\theta(\text{good}) - r_\theta(\text{bad}))$$

**Loss (Bradley-Terry preference model):**
$$\mathcal{L} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

where $y_w$ = winning/preferred response, $y_l$ = losing/rejected response.

### Implementation
```python
def reward_model_loss(r_preferred: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Train reward model on preference pairs.
    
    r_preferred: (batch,) reward scores for preferred responses
    r_rejected:  (batch,) reward scores for rejected responses
    
    Loss = -mean(log(σ(r_preferred - r_rejected)))
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # Margin between preferred and rejected
    margin = r_preferred - r_rejected
    
    # Loss: maximize probability that preferred > rejected
    loss = -np.mean(np.log(sigmoid(margin) + 1e-8))
    
    # Accuracy: fraction where preferred correctly ranked higher
    accuracy = np.mean(r_preferred > r_rejected)
    
    return loss, accuracy

def rlhf_ppo_loss(log_probs_new: np.ndarray, log_probs_old: np.ndarray,
                   rewards: np.ndarray, advantages: np.ndarray,
                   epsilon: float = 0.2) -> dict:
    """
    PPO (Proximal Policy Optimization) loss for RLHF.
    
    Clipped objective prevents too-large policy updates.
    
    log_probs_new: log π_θ(a|s) from current policy
    log_probs_old: log π_θ_old(a|s) from reference policy (KL anchor)
    rewards:       rm(response) − β*KL(policy || ref_policy)
    advantages:    R - V(s) normalized
    epsilon:       clipping range (0.1-0.3 typical)
    """
    # Policy ratio: how much current policy has changed from old
    ratio = np.exp(log_probs_new - log_probs_old)
    
    # PPO clipped objective
    # Unclipped: ratio * advantage
    # Clipped: clip(ratio, 1-eps, 1+eps) * advantage
    # Take min → conservative update
    policy_loss_unclipped = ratio * advantages
    policy_loss_clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -np.mean(np.minimum(policy_loss_unclipped, policy_loss_clipped))
    
    # Value function loss (predict expected reward)
    # (simplification — full PPO also trains value head)
    
    # KL penalty term for safety
    kl = np.mean(log_probs_old - log_probs_new)   # Approx KL(old || new)
    
    return {
        'policy_loss': float(policy_loss),
        'mean_ratio': float(ratio.mean()),
        'clipped_fraction': float((np.abs(ratio - 1) > epsilon).mean()),
        'approx_kl': float(kl),
    }

# Test reward model loss
np.random.seed(42)
batch_size = 32
r_preferred = np.random.randn(batch_size) + 1  # Generally higher
r_rejected  = np.random.randn(batch_size) - 1  # Generally lower

loss, acc = reward_model_loss(r_preferred, r_rejected)
print(f"Reward model loss: {loss:.4f}")
print(f"Reward model accuracy: {acc:.4f}")  # Should be ~1.0 (clear separation)

# Test PPO loss
log_probs_new = np.random.randn(16) * 0.1
log_probs_old = np.random.randn(16) * 0.1
rewards = np.random.randn(16) + 0.5
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

ppo = rlhf_ppo_loss(log_probs_new, log_probs_old, rewards, advantages)
print(f"\nPPO metrics: {ppo}")
```

---

## 🔴 PROBLEM 4: Agentic Tool Use Framework

### Theory
An **agent** is a loop: Observe → Think → Act → Observe → ...

**ReAct (Reason + Act):** The model produces thought traces interleaved with tool calls:
```
Thought: I need to find the price of iPhone 15 on Flipkart.
Action: search_products(query="iPhone 15", category="smartphones")
Observation: [{"name": "iPhone 15", "price": 79999, "rating": 4.5}]
Thought: Found it. The price is ₹79,999.
Answer: The iPhone 15 is priced at ₹79,999 on Flipkart.
```

### Implementation
```python
from typing import List, Dict, Callable, Any

class Tool:
    """A tool that an agent can call."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ReActAgent:
    """
    ReAct (Reason + Act) Agent implementation.
    
    Loop:
    1. Build prompt with context + tools
    2. Get model response (thought + action or answer)
    3. Execute tool, add observation to context
    4. Repeat until "Answer:" found or max_steps
    """
    
    def __init__(self, llm, tools: List[Tool], max_steps: int = 5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
    
    def _build_system_prompt(self) -> str:
        tool_descriptions = '\n'.join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        return f"""You are a helpful assistant with access to the following tools:

{tool_descriptions}

Use this format:
Thought: <your reasoning>
Action: <tool_name>(param1=value1, param2=value2)
Observation: <tool result will be inserted here>

When you have enough information:
Answer: <your final answer>

Begin!"""
    
    def _parse_action(self, response: str) -> tuple:
        """
        Parse model output to extract tool name and arguments.
        Returns: (tool_name, kwargs) or (None, None) if answer found
        """
        import re
        
        if 'Answer:' in response:
            answer = response.split('Answer:')[-1].strip()
            return None, answer
        
        # Try to parse Action: tool_name(args)
        action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response)
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            
            # Simple key=value parsing
            kwargs = {}
            for pair in args_str.split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    kwargs[k.strip()] = v.strip().strip('"\'')
            
            return tool_name, kwargs
        
        return None, None
    
    def run(self, query: str) -> Dict:
        """Run the agent loop."""
        messages = [self._build_system_prompt(), f"Question: {query}"]
        trajectory = []
        
        for step in range(self.max_steps):
            prompt = '\n'.join(messages)
            response = self.llm.generate(prompt)
            messages.append(response)
            trajectory.append({'step': step, 'response': response})
            
            tool_name, args_or_answer = self._parse_action(response)
            
            if tool_name is None:
                # Got final answer
                return {
                    'answer': args_or_answer,
                    'steps': step + 1,
                    'trajectory': trajectory,
                }
            
            if tool_name in self.tools:
                # Execute tool
                try:
                    tool_result = self.tools[tool_name](**args_or_answer)
                    observation = f"Observation: {tool_result}"
                except Exception as e:
                    observation = f"Observation: Error - {str(e)}"
                
                messages.append(observation)
                trajectory.append({'observation': str(tool_result)})
            else:
                messages.append(f"Observation: Unknown tool '{tool_name}'")
        
        return {'answer': "Max steps reached.", 'steps': self.max_steps, 'trajectory': trajectory}


# Define some tools for Flipkart e-commerce agent
def search_products(query: str, category: str = None) -> List[Dict]:
    """Mock product search."""
    products = {
        'iphone': [{'name': 'iPhone 15', 'price': 79999, 'rating': 4.5, 'in_stock': True}],
        'laptop': [{'name': 'Dell XPS 15', 'price': 129999, 'rating': 4.3, 'in_stock': True}],
    }
    for key, results in products.items():
        if key in query.lower():
            return results
    return [{'name': 'No products found', 'price': 0, 'rating': 0}]

def get_delivery_date(product_name: str, pincode: str) -> str:
    """Mock delivery date lookup."""
    return f"Expected delivery for {product_name} to {pincode}: 2-3 business days"

def check_discount(product_name: str) -> str:
    """Mock discount checker."""
    discounts = {'iPhone 15': '10% off with HDFC card', 'Dell XPS 15': '5% cashback with Flipkart Pay Later'}
    return discounts.get(product_name, 'No active discounts')

# Demo
tools = [
    Tool("search_products", "Search for products on Flipkart. Args: query, category (optional)", search_products),
    Tool("get_delivery_date", "Get delivery date for a product. Args: product_name, pincode", get_delivery_date),
    Tool("check_discount", "Check available discounts. Args: product_name", check_discount),
]

class MockAgentLLM:
    """Mock LLM that follows ReAct format."""
    def generate(self, prompt: str) -> str:
        if "Question:" in prompt and "Observation:" not in prompt:
            return """Thought: I need to find the iPhone 15 on Flipkart.
Action: search_products(query="iPhone 15", category="smartphones")"""
        elif "Observation:" in prompt:
            return """Thought: Found the product. Price is ₹79,999.
Answer: The iPhone 15 is priced at ₹79,999 on Flipkart with a rating of 4.5/5."""
        return "Answer: I couldn't find that information."

agent = ReActAgent(MockAgentLLM(), tools, max_steps=5)
result = agent.run("What is the price of iPhone 15?")
print(f"Answer: {result['answer']}")
print(f"Steps taken: {result['steps']}")
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"LoRA rank r=8 vs r=64 — what's the tradeoff?"** → r=8: fewer params, faster, less expressive (fine for style/tone). r=64: more expressive, can learn task-specific knowledge better, but heavier. Start with 8; increase if underfitting.
2. **"My quantized model has 5% worse accuracy. What do you do?"** → (a) Switch per-tensor to per-channel quantization. (b) Do quantization-aware training (QAT) instead of post-training. (c) Use FP8 instead of INT8 (better precision, newer hardware supports it). (d) Quantize only FFN layers; keep attention in FP16.
3. **"RLHF vs DPO — what's the key difference?"** → RLHF: train explicit reward model → use RL (PPO) to optimize policy. DPO: no explicit RM; directly optimize policy on preference pairs via a clever reparameterization. DPO is simpler, more stable, no RL loop needed. Both use same preference data.
4. **"What are the failure modes of ReAct agents?"** → (a) Tool hallucination (calling non-existent tools). (b) Infinite loops (tool returns no progress). (c) Context window overflow (long trajectories). (d) Error propagation (wrong tool output → wrong reasoning). Mitigations: constrained output format, max_steps, error handling, planning before acting (Plan-and-Execute).
