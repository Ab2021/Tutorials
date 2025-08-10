# Day 21.1: Graph Neural Networks & Mathematical Foundations - A Practical Guide

## Introduction: Beyond Grids and Sequences

So far, we have worked with highly structured data: sequences (1D grids) and images (2D grids). But what about data that is inherently irregular and relational? 

*   **Social Networks:** A network of users and their friendships.
*   **Molecules:** A graph of atoms (nodes) and chemical bonds (edges).
*   **Citation Networks:** A graph of academic papers (nodes) and their citations (edges).
*   **Knowledge Graphs:** A network of entities and their relationships.

This type of data is best represented as a **graph**. A graph consists of a set of **nodes** (or vertices) and a set of **edges** (or links) that connect them. Standard deep learning models like CNNs and RNNs cannot be applied directly to graphs because graphs lack a fixed structure: they are permutation-invariant (the order of nodes doesn't matter) and can have a variable number of neighbors for each node.

**Graph Neural Networks (GNNs)** are a specialized class of neural networks designed to work directly with graph-structured data. The core idea of a GNN is to learn a vector representation (an embedding) for each node by aggregating information from its local neighborhood.

This guide will introduce the mathematical foundations of GNNs and show how graph data is represented for processing.

**Today's Learning Objectives:**

1.  **Understand Graph-Structured Data:** Learn the core terminology of graphs: nodes, edges, adjacency matrix, and feature matrix.
2.  **Grasp the Core GNN Idea: Message Passing:** Understand the high-level concept of a node updating its representation by aggregating messages from its neighbors.
3.  **Represent Graphs Numerically:** Learn how to use the **Adjacency Matrix (A)** and the **Node Feature Matrix (X)** to represent a graph in a way a computer can understand.
4.  **Implement the Fundamental GNN Equation:** See the mathematical formula for a simple Graph Convolutional Network (GCN) and understand its connection to the message passing idea.
5.  **Use a Library for Graph Data:** See how a specialized library like `PyTorch Geometric` simplifies the process of creating and handling graph data objects.

---

## Part 1: Representing Graphs

To work with a graph in a neural network, we need to represent its structure and features numerically. We use two main matrices:

1.  **The Node Feature Matrix (X):**
    *   **What it is:** A matrix where each row corresponds to a node in the graph, and each column corresponds to a feature of that node.
    *   **Shape:** `(N, F)`, where `N` is the number of nodes and `F` is the number of input features per node.
    *   **Example:** In a social network, features might include a user's age, gender, or the word embeddings of their profile description.

2.  **The Adjacency Matrix (A):**
    *   **What it is:** A square matrix that describes the connection structure of the graph.
    *   **Shape:** `(N, N)`.
    *   **Values:** `A[i, j] = 1` if there is an edge from node `i` to node `j`, and `0` otherwise. For an undirected graph, this matrix is symmetric.

---

## Part 2: The Core GNN Idea - Message Passing

The fundamental operation in a GNN is an iterative process of **message passing** and **aggregation**. A GNN layer updates the embedding for each node based on the embeddings of its neighbors.

For a single node `i`, one layer of a GNN performs the following steps:

1.  **Message Passing:** Each neighbor `j` of node `i` sends its current feature vector `h_j` (its "message") to node `i`.
2.  **Aggregation:** Node `i` aggregates all the messages it received from its neighbors. This aggregation function must be **permutation-invariant**, meaning it doesn't matter in what order the messages arrive. Common choices are `sum`, `mean`, or `max`.
3.  **Update:** Node `i` updates its own old feature vector `h_i` using the aggregated message and a neural network layer (e.g., a linear layer with an activation function).

By stacking multiple GNN layers, a node's representation can incorporate information from nodes that are further and further away (its 2-hop neighbors, 3-hop neighbors, etc.).

![GNN Message Passing](https://i.imgur.com/xG5h3XF.png)

---

## Part 3: The Graph Convolutional Network (GCN)

The Graph Convolutional Network (GCN) is a specific and highly influential type of GNN that provides an efficient way to implement this message passing idea using matrix operations.

### 3.1. The GCN Layer Formula

The update rule for a single GCN layer is surprisingly simple and elegant:

`H_new = f( D_hat^(-0.5) * A_hat * D_hat^(-0.5) * H_old * W )`

Let's break this down:

*   `H_old`: The matrix of node features from the previous layer (or the initial `X`). Shape: `(N, F_in)`.
*   `W`: A learnable weight matrix for the layer. Shape: `(F_in, F_out)`.
*   `H_old * W`: This is a standard linear transformation of the node features.

Now for the graph structure part:
*   `A_hat = A + I`: This is the adjacency matrix `A` with self-loops added (via the identity matrix `I`). This is important so that a node includes its **own features** from the previous layer when it updates itself.
*   `D_hat`: This is the **degree matrix** of `A_hat`. It's a diagonal matrix where `D_hat[i, i]` is the number of neighbors of node `i` (including itself).
*   `D_hat^(-0.5)`: The inverse square root of the degree matrix.

**The Key Operation: `A_hat * H_old`**
*   Think about what this matrix multiplication does. The `i`-th row of the result is the sum of the feature vectors of all the neighbors of node `i`. This is exactly the **message passing and aggregation (sum)** step!

**The Normalization: `D_hat^(-0.5) * ... * D_hat^(-0.5)`**
*   Multiplying by the adjacency matrix sums up the neighbor features, but it has a problem: high-degree nodes will have feature vectors with much larger magnitudes, which can lead to instability. 
*   Multiplying by the inverse degree matrix terms normalizes the feature vectors, essentially taking the **average** of the neighbor messages. This keeps the scale of the feature vectors consistent.

*   `f`: A non-linear activation function, like `ReLU`.

### 3.2. Implementing the GCN Formula

Let's see this in code.

```python
import torch
import torch.nn.functional as F

print("--- Part 3: The GCN Formula ---")

# --- 1. Define a sample graph ---
# 4 nodes, 3 features per node
# Adjacency Matrix (A)
#   0 -- 1
#   |    |
#   2 -- 3
A = torch.tensor([
    [0, 1, 1, 0], # Node 0 is connected to 1, 2
    [1, 0, 0, 1], # Node 1 is connected to 0, 3
    [1, 0, 0, 1], # Node 2 is connected to 0, 3
    [0, 1, 1, 0]  # Node 3 is connected to 1, 2
], dtype=torch.float32)

# Node Feature Matrix (X or H_old)
X = torch.randn(4, 3) # (N, F_in)

# Learnable Weight Matrix (W)
W = torch.randn(3, 5) # (F_in, F_out)

# --- 2. Implement the GCN layer propagation ---

# a) Add self-loops
I = torch.eye(A.size(0))
A_hat = A + I

# b) Calculate the Degree Matrix D_hat
# The degree of a node is the sum of its row in the adjacency matrix
D_hat = torch.diag(A_hat.sum(dim=1))

# c) Calculate the inverse square root of D_hat
# Add a small epsilon for numerical stability
D_hat_inv_sqrt = torch.pow(D_hat.diagonal(), -0.5).diag()

# d) The normalization matrix
norm_matrix = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

# e) The GCN propagation
H_new_linear = norm_matrix @ X @ W

# f) Apply activation function
H_new = F.relu(H_new_linear)

print(f"Original Feature Matrix shape (H_old): {X.shape}")
print(f"Weight Matrix shape (W): {W.shape}")
print(f"New Feature Matrix shape (H_new): {H_new.shape}")
```

---

## Part 4: Using PyTorch Geometric

Implementing GNNs from scratch using dense adjacency matrices is inefficient for large, sparse graphs. Specialized libraries like **PyTorch Geometric (PyG)** and **Deep Graph Library (DGL)** are the standard tools for GNN research and application.

They provide:
*   An efficient representation for sparse graphs.
*   A large collection of common graph datasets.
*   Optimized implementations of many different GNN layers (`GCNConv`, `GATConv`, etc.).

### 4.1. A PyG Data Object

PyG represents a graph as a single `Data` object.

```python
from torch_geometric.data import Data

print("\n--- Part 4: PyTorch Geometric ---")

# --- Representing our graph in PyG ---

# Instead of a dense adjacency matrix, PyG uses a sparse format
# called an edge index list.
# Shape: (2, num_edges)
# Row 1 is the source node of each edge, Row 2 is the target node.
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 3], # Source nodes
    [1, 2, 0, 3, 0, 3, 1, 2]  # Target nodes
], dtype=torch.long)

# The node feature matrix is the same
x_features = torch.randn(4, 3) # 4 nodes, 3 features

# Create the PyG Data object
graph_data = Data(x=x_features, edge_index=edge_index)

print("A PyTorch Geometric Data object:")
print(graph_data)
```

This `Data` object can then be passed directly to PyG's built-in GNN layers, which handle the complex message passing and aggregation internally in a highly optimized way.

## Conclusion

Graph Neural Networks provide a powerful framework for applying deep learning to relational, non-Euclidean data. The core idea is **message passing**, where nodes iteratively update their vector representations by aggregating information from their neighbors.

**Key Takeaways:**

1.  **Graphs are Everywhere:** Many important datasets have an underlying graph structure that can be exploited.
2.  **GNNs Learn Node Embeddings:** The goal of a GNN is to learn a low-dimensional vector embedding for each node that incorporates both its own features and the structure of its local neighborhood.
3.  **The GCN Formula:** The simple matrix formula `D_hat^(-0.5) * A_hat * D_hat^(-0.5) * H * W` provides an efficient way to implement a message passing step where nodes aggregate the normalized features of their neighbors.
4.  **Use Specialized Libraries:** For any practical application, use a library like PyTorch Geometric or DGL. They provide efficient data structures for sparse graphs and optimized implementations of common GNN layers.

With this foundational understanding, you are now ready to explore the specific architectures and applications of GNNs.

## Self-Assessment Questions

1.  **Adjacency Matrix:** If a graph is undirected, what special property will its adjacency matrix have?
2.  **Message Passing:** What are the two main steps a node performs in a single GNN layer?
3.  **GCN Formula:** In the GCN formula, what is the purpose of adding the identity matrix `I` to the adjacency matrix `A`?
4.  **Normalization:** Why is it important to normalize by the node degrees (the `D_hat` terms)?
5.  **PyG `edge_index`:** How does the `edge_index` format represent the graph's structure?

