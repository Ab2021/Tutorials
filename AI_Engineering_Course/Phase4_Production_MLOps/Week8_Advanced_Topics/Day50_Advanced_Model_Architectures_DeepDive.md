# Day 50: Advanced Model Architectures
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Mixture of Experts (MoE) Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=8, num_experts_per_token=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, hidden_size)  # (batch * seq_len, hidden_size)
        
        # Router logits
        router_logits = self.router(x_flat)  # (batch * seq_len, num_experts)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        
        # Softmax over selected experts
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each token
        for i in range(batch_size * seq_len):
            token_output = torch.zeros(hidden_size, device=x.device)
            
            for j in range(self.num_experts_per_token):
                expert_idx = top_k_indices[i, j]
                gate_value = top_k_gates[i, j]
                
                # Run expert
                expert_output = self.experts[expert_idx](x_flat[i:i+1])
                token_output += gate_value * expert_output.squeeze(0)
            
            output[i] = token_output
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output

# Usage
moe_layer = MoELayer(hidden_size=768, num_experts=8, num_experts_per_token=2)
x = torch.randn(2, 10, 768)  # (batch=2, seq_len=10, hidden=768)
output = moe_layer(x)
```

### 2. Sparse Attention (Longformer)

```python
class LongformerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, global_attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Local attention (sliding window)
        output = self._sliding_window_attention(Q, K, V)
        
        # Global attention (for special tokens)
        if global_attention_mask is not None:
            output = self._add_global_attention(output, Q, K, V, global_attention_mask)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_size)
        output = self.o_proj(output)
        
        return output
    
    def _sliding_window_attention(self, Q, K, V):
        """Local attention within sliding window."""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        output = torch.zeros_like(Q)
        
        for i in range(seq_len):
            # Define window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Attention within window
            q_i = Q[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)
            k_window = K[:, :, start:end, :]  # (batch, heads, window, head_dim)
            v_window = V[:, :, start:end, :]
            
            # Compute attention
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            output[:, :, i:i+1, :] = torch.matmul(attn, v_window)
        
        return output
    
    def _add_global_attention(self, output, Q, K, V, global_mask):
        """Add global attention for special tokens."""
        # Global tokens attend to all positions
        global_indices = torch.where(global_mask)[0]
        
        for idx in global_indices:
            q_global = Q[:, :, idx:idx+1, :]
            scores = torch.matmul(q_global, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            output[:, :, idx:idx+1, :] = torch.matmul(attn, V)
        
        return output
```

### 3. Model Merging (Task Arithmetic)

```python
def merge_models(base_model, finetuned_models, alphas):
    """
    Merge multiple fine-tuned models using task arithmetic.
    
    merged = base + α1(model1 - base) + α2(model2 - base) + ...
    """
    merged_state_dict = {}
    base_state_dict = base_model.state_dict()
    
    for key in base_state_dict.keys():
        # Start with base weights
        merged_weights = base_state_dict[key].clone()
        
        # Add task vectors
        for model, alpha in zip(finetuned_models, alphas):
            model_weights = model.state_dict()[key]
            task_vector = model_weights - base_state_dict[key]
            merged_weights += alpha * task_vector
        
        merged_state_dict[key] = merged_weights
    
    # Load merged weights into base model
    merged_model = copy.deepcopy(base_model)
    merged_model.load_state_dict(merged_state_dict)
    
    return merged_model

# Usage
base = load_model("llama-2-7b")
math_model = load_model("llama-2-7b-math")
code_model = load_model("llama-2-7b-code")

merged = merge_models(
    base,
    [math_model, code_model],
    alphas=[0.5, 0.5]
)
```

### 4. Retrieval-Enhanced Transformer (RETRO)

```python
class RETROLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, retrieval_db):
        super().__init__()
        self.retrieval_db = retrieval_db
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x):
        # x: (seq_len, batch, hidden_size)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieval_db.retrieve(x)
        # retrieved_docs: (num_docs, batch, hidden_size)
        
        # Self-attention
        x_self = self.self_attention(x, x, x)[0]
        x = x + x_self
        
        # Cross-attention with retrieved docs
        x_cross = self.cross_attention(x, retrieved_docs, retrieved_docs)[0]
        x = x + x_cross
        
        # FFN
        x_ffn = self.ffn(x)
        x = x + x_ffn
        
        return x
```

### 5. Continual Learning with EWC

```python
class EWC:
    def __init__(self, model, dataloader, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_params = {}
        
        # Compute Fisher Information Matrix
        self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader):
        """Compute Fisher Information for each parameter."""
        self.model.eval()
        
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
            self.optimal_params[name] = param.clone().detach()
        
        for inputs, targets in dataloader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.pow(2)
        
        # Average over dataset
        for name in self.fisher_information:
            self.fisher_information[name] /= len(dataloader)
    
    def penalty(self):
        """Compute EWC penalty."""
        loss = 0
        for name, param in self.model.named_parameters():
            fisher = self.fisher_information[name]
            optimal = self.optimal_params[name]
            loss += (fisher * (param - optimal).pow(2)).sum()
        
        return self.lambda_ewc * loss

# Usage
model = YourModel()
old_task_dataloader = ...

ewc = EWC(model, old_task_dataloader)

# Train on new task
for inputs, targets in new_task_dataloader:
    outputs = model(inputs)
    task_loss = F.cross_entropy(outputs, targets)
    ewc_loss = ewc.penalty()
    
    total_loss = task_loss + ewc_loss
    total_loss.backward()
    optimizer.step()
```

### 6. State Space Model (Mamba-like)

```python
class StateSpaceLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # State update: h_t = A * h_{t-1} + B * x_t
            h = torch.matmul(h, self.A.T) + torch.matmul(x[:, t], self.B.T)
            
            # Output: y_t = C * h_t + D * x_t
            y = torch.matmul(h, self.C.T) + x[:, t] * self.D
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)
```
