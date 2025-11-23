# Lab 1: Build a GPT from Scratch

## Objective
Implement the Transformer architecture (Decoder-only) from scratch in PyTorch.
We will replicate the architecture of GPT-2.
This is the "Rite of Passage" for every AI Engineer.

## 1. The Architecture (`gpt.py`)

We need to implement:
1.  **Self-Attention Head**
2.  **Multi-Head Attention**
3.  **Feed-Forward Network**
4.  **Block (Transformer Layer)**
5.  **GPT Model**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_head = d_model // n_head
        self.n_head = n_head
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Causal Mask (Lower Triangular)
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels
        
        # Calculate Query, Key, Value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for Multi-Head: (B, T, n_head, d_head) -> (B, n_head, T, d_head)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # Attention Score: (Q @ K.T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply Mask (Fill future positions with -inf)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        
        # Aggregate Values: Att @ V
        y = att @ v # (B, n_head, T, d_head)
        
        # Reassemble Heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        # Residual Connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_head, max_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model) # Final LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Positional Encoding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        # Transformer Blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Logits
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten for CrossEntropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx[:, -self.max_len:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            
            # Apply Softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
```

## 2. Training on Shakespeare (`train.py`)

We will train this on the "Tiny Shakespeare" dataset.

```python
import torch
from gpt import GPT

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 64 # Context Length
MAX_ITERS = 1000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 128
N_HEAD = 4
N_LAYER = 4

# 1. Load Data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# 2. Initialize Model
model = GPT(vocab_size, EMBED_DIM, N_HEAD, N_LAYER, BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 3. Training Loop
print(f"Training on {DEVICE}...")
for iter in range(MAX_ITERS):
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# 4. Generate
print("\nGenerating Text:")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
```

## 3. Running the Lab

1.  Download `input.txt` (Tiny Shakespeare).
2.  Run `python train.py`.
3.  Observe the loss going down and the generated text becoming more coherent (from random characters to Shakespearean gibberish).

## 4. Challenge
*   **Scale Up:** Increase `EMBED_DIM` to 384 and `N_LAYER` to 6. Does the loss drop faster?
*   **Top-K Sampling:** Modify the `generate` function to only sample from the top K tokens to reduce randomness.

## 5. Submission
Submit a sample of the generated text.
