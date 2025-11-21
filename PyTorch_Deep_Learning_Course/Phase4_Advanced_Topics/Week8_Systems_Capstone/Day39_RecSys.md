# Day 39: Recommender Systems - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Collaborative Filtering, NCF, and Two-Tower Models

## 1. Theoretical Foundation: The Matrix Completion Problem

Users $U$, Items $I$. Interaction Matrix $Y$ (Sparse).
Goal: Predict missing entries.
**Collaborative Filtering**: Users who liked X also liked Y.

## 2. Neural Collaborative Filtering (NCF)

Generalizing Matrix Factorization (MF) with Neural Networks.
*   **MF**: $y_{ui} = p_u \cdot q_i$ (Dot Product).
*   **NCF**: $y_{ui} = \text{MLP}(\text{Concat}(p_u, q_i))$.
*   Learns non-linear interactions.

## 3. Two-Tower Architecture (Deep Retrieval)

Standard for industrial scale (YouTube, Netflix).
*   **User Tower**: Encodes user features (History, Demographics) $\to$ Embedding $u$.
*   **Item Tower**: Encodes item features (ID, Description, Image) $\to$ Embedding $v$.
*   **Similarity**: $s = \cos(u, v)$.
*   **Inference**: Pre-compute all item embeddings. Use ANN (FAISS) to find nearest items to user embedding $u$.

## 4. Implementation: Two-Tower Model

```python
import torch
import torch.nn as nn

class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, user_id, item_id):
        u = self.user_embed(user_id)
        i = self.item_embed(item_id)
        
        u_vec = self.user_tower(u)
        i_vec = self.item_tower(i)
        
        # Dot Product
        return (u_vec * i_vec).sum(dim=1)
```

## 5. Ranking (DeepFM / DCN)

After Retrieval (getting top 100 items), we need precise Ranking.
**DeepFM**: Combines FM (Factorization Machines) for low-order interactions and DNN for high-order interactions.
**DCN (Deep & Cross Network)**: Explicit feature crossing.
