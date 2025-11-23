# Lab 5: Efficient Attention (Tiling)

## Objective
Attention is $O(N^2)$. This is slow.
**FlashAttention** optimizes this by tiling the computation to keep data in SRAM (fast cache).
We will implement a simplified tiling loop to understand the concept.

## 1. Naive Attention
```python
S = Q @ K.T
P = softmax(S)
O = P @ V
```
This materializes `S` and `P` (N x N matrices) in HBM (GPU Memory), which is slow.

## 2. Tiled Attention (`tiled.py`)

```python
import torch

def tiled_attention(Q, K, V, block_size=64):
    N, d = Q.shape
    O = torch.zeros(N, d)
    
    # Split into blocks
    for i in range(0, N, block_size):
        # Load block of Q into SRAM (simulated)
        Qi = Q[i:i+block_size]
        
        for j in range(0, N, block_size):
            # Load block of K, V into SRAM
            Kj = K[j:j+block_size]
            Vj = V[j:j+block_size]
            
            # Compute partial attention
            S_ij = Qi @ Kj.T
            P_ij = torch.softmax(S_ij, dim=-1) # Simplified (Real FlashAttn uses online softmax)
            
            O[i:i+block_size] += P_ij @ Vj
            
    return O
```

## 3. Analysis
Real FlashAttention is much more complex (Online Softmax, Backward pass recomputation).
But the core idea is **Block-wise processing** to avoid large matrix writes.

## 4. Submission
Explain why Tiling reduces memory bandwidth usage.
