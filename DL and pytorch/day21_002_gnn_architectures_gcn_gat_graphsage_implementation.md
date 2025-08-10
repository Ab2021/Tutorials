# Day 21.2: GNN Architectures (GCN, GAT, GraphSAGE) - A Practical Guide

## Introduction: The Evolution of Message Passing

The simple Graph Convolutional Network (GCN) provided a powerful and efficient baseline for learning on graphs. However, its message-passing scheme is fixed and somewhat limited: it always aggregates neighbor information by taking a simple, degree-normalized average. 

Subsequent research has focused on developing more sophisticated and flexible aggregation and update mechanisms. This has led to a zoo of GNN architectures, each with its own strengths.

This guide provides a practical overview of three of the most foundational and influential GNN architectures: **GCN**, **GAT**, and **GraphSAGE**. We will implement them using the PyTorch Geometric (PyG) library, the standard tool for practical GNN development.

**Today's Learning Objectives:**

1.  **Review the GCN:** Solidify the understanding of the GCN as a simple neighborhood averaging scheme.
2.  **Learn Graph Attention Networks (GAT):** Understand how GAT uses self-attention to learn the importance of different neighbors, allowing for a weighted aggregation.
3.  **Explore GraphSAGE:** Learn about this framework that generalizes the aggregation step, allowing for different aggregator functions like Mean, Max, or even an LSTM.
4.  **Implement a Full GNN Model in PyG:** Build a complete node classification model using PyG's built-in layers and train it on a standard citation network dataset (Cora).

---

## Part 1: The Cora Dataset - A Classic GNN Benchmark

To compare these architectures, we need a dataset. The **Cora** dataset is a standard benchmark for node classification.

*   **Nodes:** 2,708 scientific publications.
*   **Edges:** 10,556 citation links between them.
*   **Node Features:** A 1,433-dimensional binary vector for each paper, indicating the presence or absence of words from a fixed dictionary.
*   **Task:** To classify each paper into one of seven subjects (e.g., "Neural Networks," "Rule Learning").

We will use PyTorch Geometric to easily load and inspect this dataset.

```python
import torch
from torch_geometric.datasets import Planetoid

print("--- Part 1: The Cora Dataset ---")

# Load the dataset
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0] # Get the first and only graph object

# Print information about the graph
print(f"Dataset: {dataset.name}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features per node: {data.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"\nGraph object:\n{data}")

# The `data` object also contains train/val/test masks, which are boolean tensors
# indicating which nodes to use for each phase.
print(f"\nNumber of training nodes: {data.train_mask.sum()}")
print(f"Number of validation nodes: {data.val_mask.sum()}")
print(f"Number of test nodes: {data.test_mask.sum()}")
```

---

## Part 2: The Architectures in PyTorch Geometric

PyG makes implementing and swapping different GNN layers incredibly easy.

### 2.1. Graph Convolutional Network (GCN)

*   **Recap:** Aggregates neighbor features by taking a normalized mean. It's a simple, efficient, and strong baseline.
*   **PyG Layer:** `torch_geometric.nn.GCNConv`

### 2.2. Graph Attention Network (GAT)

*   **The Idea:** Why should every neighbor be treated equally? A GAT layer learns the relative importance of different neighbors using **self-attention**.
*   **How it Works:**
    1.  It first applies a linear transformation to the node features.
    2.  For each node, it computes an **attention score** for every one of its neighbors. This score determines how important that neighbor's message is.
    3.  The scores are normalized using a softmax function.
    4.  The final node update is a **weighted sum** of the neighbors' transformed features, where the weights are the learned attention scores.
    5.  Like the Transformer, it can also use **multi-head attention** to stabilize learning.
*   **PyG Layer:** `torch_geometric.nn.GATConv`

### 2.3. GraphSAGE (Sample and Aggregate)

*   **The Idea:** To create a general, inductive framework for GNNs. The key idea is to separate the neighborhood sampling from the aggregation step.
*   **How it Works:** GraphSAGE is a two-step process:
    1.  **Sample:** For each node, sample a fixed number of its neighbors (instead of using all of them). This makes the model scalable to massive graphs.
    2.  **Aggregate:** Apply a general, permutation-invariant **aggregator function** to the sampled neighbors' features. The paper explored several aggregators:
        *   **Mean Aggregator:** Takes the element-wise mean (this is equivalent to the GCN convolution).
        *   **Pool Aggregator:** Takes the element-wise max or mean of the neighbors' features after passing them through a linear layer.
        *   **LSTM Aggregator:** Applies an LSTM to a random permutation of the neighbors' features.
*   **PyG Layer:** `torch_geometric.nn.SAGEConv`

---

## Part 3: Building and Training a GNN for Node Classification

Let's build a simple two-layer GNN using PyG and train it on the Cora dataset. The beauty of PyG is that we can easily swap out the convolutional layers to compare the different architectures.

### 3.1. The GNN Model

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

print("\n--- Part 3: Building and Training a GNN ---")

class GNN(nn.Module):
    def __init__(self, model_type='GCN', hidden_channels=16, dropout=0.5):
        super(GNN, self).__init__()
        self.dropout = dropout

        # --- We can choose the layer type based on the argument ---
        if model_type == 'GCN':
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        elif model_type == 'GAT':
            # For GAT, we can specify the number of attention heads
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels, heads=8)
            self.conv2 = GATConv(hidden_channels * 8, dataset.num_classes, heads=1)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, dataset.num_classes)
        else:
            raise ValueError("Model type not supported")

    def forward(self, x, edge_index):
        # x shape: [num_nodes, num_node_features]
        # edge_index shape: [2, num_edges]
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # The output is raw logits for each node
        # Shape: [num_nodes, num_classes]
        return x

# --- Instantiate a GCN model ---
model = GNN(model_type='GCN')
print("GCN Model Architecture:")
print(model)
```

### 3.2. The Training Loop

```python
# --- Setup for Training ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(model_type='GCN', hidden_channels=16).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    # The model takes the full graph data as input
    out = model(data.x, data.edge_index)
    # We compute the loss only on the training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Check accuracy on train, val, and test sets
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs

# --- Run the Training ---
print("\n--- Training a GCN on Cora ---")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
```

**To Compare Architectures:**
You could simply change the `model_type` when instantiating the model (e.g., `model = GNN(model_type='GAT').to(device)`) and re-run the same training loop to compare the final test accuracy.

## Conclusion

While GCN provides a strong and simple baseline, more advanced architectures offer greater flexibility and power by improving the core message-passing and aggregation steps.

**Key Architectural Differences:**

*   **GCN (Graph Convolutional Network):** The simplest model. It performs a weighted average of neighbor features, where the weights are fixed and based on node degrees. It is a **spectral** method.

*   **GAT (Graph Attention Network):** It improves on GCN by introducing an **attention mechanism**. Instead of a simple average, it computes a weighted average where the weights are learned based on the features of the connected nodes. This allows the model to assign more importance to more relevant neighbors.

*   **GraphSAGE (Sample and Aggregate):** It provides a general framework for **spatial** graph convolutions. Its key innovations are to **sample** a fixed-size neighborhood for each node (making it scalable) and to use a general, learnable **aggregator function** (like Mean, Max-Pool, or even an LSTM) to combine the neighbor information.

For most applications, starting with a GCN or GraphSAGE is a strong baseline. If the task might benefit from nodes paying different levels of attention to their neighbors (e.g., in a social network or a knowledge graph), a GAT is an excellent choice. Thanks to libraries like PyTorch Geometric, experimenting with these different powerful architectures is straightforward.

## Self-Assessment Questions

1.  **GCN Limitation:** What is the main limitation of the GCN's aggregation scheme?
2.  **GAT:** How does a GAT decide how much importance to give to each neighbor?
3.  **GraphSAGE:** What are the two main conceptual steps in the GraphSAGE algorithm?
4.  **Inductive Learning:** GraphSAGE was designed to be "inductive." What does this mean? (Hint: Can it generalize to entirely new nodes or graphs it hasn't seen during training?)
5.  **PyG `Conv` Layers:** What are the two main arguments that every GNN `Conv` layer in PyTorch Geometric requires in its `forward` method?

