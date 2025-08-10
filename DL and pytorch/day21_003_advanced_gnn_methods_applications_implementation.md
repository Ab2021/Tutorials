# Day 21.3: Advanced GNN Methods & Applications - A Practical Guide

## Introduction: The Expanding World of Graph Learning

The foundational GNN architectures (GCN, GAT, GraphSAGE) opened the door to applying deep learning to graph data. Building on these, the field has rapidly expanded to tackle more complex graph structures, larger-scale problems, and a diverse range of applications beyond simple node classification.

This guide provides a high-level, practical overview of several advanced GNN methods and highlights the exciting real-world problems they are solving.

**Today's Learning Objectives:**

1.  **Understand Graph-Level Tasks:** Learn how GNNs can be adapted for tasks like graph classification and regression using global pooling.
2.  **Explore Heterogeneous Graphs:** Grasp the concept of graphs with multiple types of nodes and edges and see how specialized GNNs handle them.
3.  **Learn about GNNs for Link Prediction:** Understand how GNNs can be used to predict missing edges or recommend new connections in a graph.
4.  **Discover Dynamic and Temporal Graphs:** Get a high-level view of how GNNs are combined with recurrent models to handle graphs that change over time.
5.  **Survey the Broad Applications of GNNs:** See how graph learning is being applied in fields like drug discovery, recommendation systems, and traffic forecasting.

---

## Part 1: Graph-Level Tasks - Graph Classification

**The Task:** So far, we have focused on node-level tasks (classifying each node). But what if we want to classify an **entire graph**? 
*   *Example:* Given a dataset of molecules (each represented as a graph), classify each molecule as toxic or non-toxic.

**The Architecture:** To get a single representation for the entire graph, we need to aggregate the information from all the individual node embeddings. This is achieved with a **global pooling** layer.

1.  First, we process the graph through several GNN layers to get rich, contextualized node embeddings `H`.
2.  Then, we apply a **global pooling** operation to the set of all node embeddings to produce a single graph-level embedding vector `h_G`.
    *   **Global Mean Pooling:** `h_G = mean(H_i)` for all nodes `i`.
    *   **Global Max Pooling:** `h_G = max(H_i)` for all nodes `i`.
    *   **Global Add Pooling:** `h_G = sum(H_i)` for all nodes `i`.
3.  This single graph embedding `h_G` is then passed to a standard MLP classifier to make the final prediction.

### 1.1. Implementation Sketch with PyG

PyTorch Geometric provides easy-to-use global pooling layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

print("--- Part 1: Graph Classification ---")

# --- 1. Load a Graph Classification Dataset ---
# The MUTAG dataset is a collection of small graphs, each representing a chemical compound.
# The task is to classify each compound as mutagenic or not.
dataset = TUDataset(root='./data/TUDataset', name='MUTAG')

# Create a DataLoader that yields entire graphs.
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- 2. The Graph Classification Model ---
class GraphClassifier(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        # 2. Global Pooling
        # The `batch` vector tells the pooling layer which nodes belong to which graph.
        h_graph = global_mean_pool(h, batch)

        # 3. Classify
        return self.classifier(h_graph)

# --- 3. Dummy Usage ---
model = GraphClassifier(dataset.num_node_features, dataset.num_classes)

# Get one batch (a mini-batch of graph objects)
first_batch = next(iter(loader))

logits = model(first_batch.x, first_batch.edge_index, first_batch.batch)

print(f"Model created for graph classification.")
print(f"Input batch contains {first_batch.num_graphs} graphs.")
print(f"Output logits shape: {logits.shape}") # (num_graphs, num_classes)
```

---

## Part 2: Link Prediction

**The Task:** Given a graph, predict whether an edge is likely to exist between two nodes that are not currently connected. This is a fundamental task for recommender systems.

*   *Example:* In a social network, predict which users might become friends. In an e-commerce network of users and products, recommend new products to a user.

**The Architecture:**
1.  Process the graph through several GNN layers to get final node embeddings for all nodes.
2.  To predict if an edge exists between node `i` and node `j`, take their final embeddings, `h_i` and `h_j`.
3.  Combine these two vectors to produce a single score. A simple and effective way is to take their **dot product**: `score = h_i^T * h_j`.
4.  Pass this score through a sigmoid function to get a probability of the edge existing.

**The Training Process:**
*   **Positive Examples:** The real edges that exist in the graph.
*   **Negative Examples:** To train the model, we need negative examples (edges that don't exist). We create these by **randomly sampling** pairs of nodes that are not connected.
*   The model is then trained as a binary classifier to distinguish between positive and negative edges.

---

## Part 3: Heterogeneous Graphs

**The Problem:** Many real-world graphs are **heterogeneous**: they have different types of nodes and different types of edges.
*   *Example:* An e-commerce graph with `(user, 'buys', product)`, `(user, 'is_friends_with', user)`, and `(product, 'is_similar_to', product)`.

**The Solution: Relational GNNs (R-GCNs) and HeteroGNNs**
Specialized GNNs have been developed to handle this complexity.

*   **How they work (High-Level):** Instead of having one set of weights `W` for all message passing, they learn a **different set of weights for each edge type**. When a message is passed from node `i` to node `j` along an edge of type `r`, a specific weight matrix `W_r` is used.
*   **PyG Support:** PyTorch Geometric has excellent support for heterogeneous graphs through its `HeteroConv` layer and `HeteroData` object, which allows you to define different node and edge types explicitly.

---

## Part 4: Dynamic and Temporal Graphs

**The Problem:** Many graphs are not static; they evolve over time. New nodes and edges can appear or disappear.
*   *Example:* A social network, a financial transaction network, a traffic network.

**The Solution: Spatiotemporal GNNs (STGNNs)**
This class of models combines GNNs with recurrent neural networks (like GRUs or LSTMs) to learn from both the spatial graph structure and the temporal dynamics.

*   **How it works (High-Level):**
    1.  The input is a sequence of graph snapshots at different time steps: `[G_1, G_2, ..., G_T]`.
    2.  At each time step `t`, a **GNN** is used to perform message passing on the graph snapshot `G_t`, capturing the spatial relationships.
    3.  The output of the GNN for each node is then fed into a **recurrent cell** (like a GRU) that is maintained for that node. The GRU updates its hidden state based on the new spatial information.
    4.  This allows the model to learn how node representations evolve over time.

**Application:** Traffic forecasting is a classic application. The road network is a graph, and the traffic speeds on the roads are features that change over time. An STGNN can predict future traffic speeds across the entire network.

## Conclusion: The Broad Applicability of Graph Learning

Graph Neural Networks are a powerful and flexible tool that can be applied to a surprisingly diverse range of problems. By moving beyond simple node classification, advanced GNN methods are providing state-of-the-art solutions in many scientific and industrial domains.

**Key Application Areas:**

*   **Drug Discovery & Cheminformatics:** Predicting molecular properties, finding new drug candidates, and modeling protein-protein interactions. (Graph Classification)
*   **Recommender Systems:** Recommending products, friends, or content by predicting missing links in a user-item or social graph. (Link Prediction)
*   **Knowledge Graphs & NLP:** Enhancing language models by incorporating knowledge from large-scale knowledge graphs. (Heterogeneous GNNs)
*   **Traffic & Supply Chain Forecasting:** Modeling complex, dynamic network systems to predict future states. (Temporal GNNs)
*   **Computer Vision:** Scene graph generation, which describes the objects in an image and their relationships.

As more data is understood in terms of its underlying relationships, the importance and application of GNNs will only continue to grow.

## Self-Assessment Questions

1.  **Graph Classification:** How does a GNN produce a single vector representation for an entire graph?
2.  **Link Prediction:** How do you typically frame a link prediction task for training? What are the positive and negative samples?
3.  **Heterogeneous Graphs:** What is the main architectural difference in a GNN designed for a heterogeneous graph compared to a standard GCN?
4.  **Temporal Graphs:** What two types of neural network architectures are typically combined to create a spatiotemporal GNN?
5.  **Applications:** Name two distinct real-world applications of GNNs and the type of GNN task (e.g., node classification, link prediction) they correspond to.

