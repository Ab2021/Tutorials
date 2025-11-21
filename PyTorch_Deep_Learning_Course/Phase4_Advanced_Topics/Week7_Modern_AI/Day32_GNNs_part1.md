# Day 32: GNNs - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Oversmoothing, GraphSAGE, and Heterogeneous Graphs

## 1. The Oversmoothing Problem

In CNNs, deeper is better (ResNet-152).
In GCNs, deeper is **worse**.
*   Repeated averaging (Laplacian smoothing) makes all node features converge to the same value.
*   Nodes become indistinguishable.
*   Solution: Skip connections, DropEdge, or shallow networks (2-3 layers).

## 2. Inductive vs Transductive

*   **Transductive (GCN)**:
    *   Train on a specific graph.
    *   Cannot handle new nodes without retraining.
    *   The graph structure is fixed during training.
*   **Inductive (GraphSAGE)**:
    *   Learn a function to aggregate neighbors.
    *   Can generalize to unseen nodes/graphs.
    *   Samples neighbors (fixed size) instead of using full adjacency.

## 3. GraphSAGE (Sample and Aggregate)

Scalable GNN.
Instead of full $A$, sample $k$ neighbors.
$$ h_v = \sigma(W \cdot \text{Concat}(h_v, \text{Mean}(\{h_u\}))) $$
*   Allows mini-batch training on massive graphs (Pinterest).

## 4. Heterogeneous Graphs

Standard GNN assumes one type of node/edge.
**Heterogeneous**:
*   Nodes: Users, Movies, Directors.
*   Edges: (User, Watch, Movie), (Movie, DirectedBy, Director).
*   **Relation-GCN (R-GCN)**: Different weight matrices for different edge types.

## 5. Link Prediction (Autoencoder)

How to predict missing edges?
1.  Encoder (GNN): Get node embeddings $z_u, z_v$.
2.  Decoder (Dot Product): Score $= z_u^T z_v$.
3.  Loss: Maximize score for real edges, minimize for negative samples.
