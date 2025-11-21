# Day 32: Graph Neural Networks - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: GCN, GAT, and Message Passing

## 1. Theoretical Foundation: Non-Euclidean Data

Images (Grid) and Text (Sequence) have fixed structure.
**Graphs** (Social Networks, Molecules, Knowledge Graphs) have irregular structure.
*   Nodes $V$, Edges $E$.
*   Adjacency Matrix $A$.

**The Challenge**:
*   No fixed ordering of neighbors (Permutation Invariance).
*   Variable number of neighbors.

## 2. Message Passing Neural Networks (MPNN)

The core paradigm of GNNs.
For every node $v$:
1.  **Message**: Collect information from neighbors $N(v)$.
    $$ m_v = \text{Aggregate}(\{ h_u | u \in N(v) \}) $$
2.  **Update**: Update node state.
    $$ h_v' = \text{Update}(h_v, m_v) $$

## 3. Architectures

### GCN (Graph Convolutional Network)
*   Simple average of neighbors.
*   $h_v' = \sigma(W \cdot \sum \frac{h_u}{\sqrt{deg(v)deg(u)}})$
*   Spectral interpretation (Laplacian smoothing).

### GAT (Graph Attention Network)
*   Weighted sum of neighbors using Attention.
*   $\alpha_{vu} = \text{Attention}(h_v, h_u)$.
*   Allows focusing on important neighbors.

## 4. Implementation: PyTorch Geometric (PyG)

PyG is the standard library for GNNs.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. Define Graph
# Edges: [2, Num Edges] (COO format)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float) # Node features

data = Data(x=x, edge_index=edge_index)

# 2. Define Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2) # 2 Classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
```

## 5. Tasks

*   **Node Classification**: Predict label of a node (Fraud detection).
*   **Link Prediction**: Predict if edge exists (Recommender Systems).
*   **Graph Classification**: Predict label of whole graph (Molecule toxicity).
