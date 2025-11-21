# Day 32: Graph Neural Networks - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Graph Theory, Message Passing, and Applications

### 1. Why can't we use standard CNNs on Graphs?
**Answer:**
*   CNNs assume a fixed grid structure (Euclidean).
*   Graphs have irregular structure (variable neighbors, no order).
*   Convolution requires a fixed kernel size, which doesn't apply to graphs.

### 2. What is "Permutation Invariance" in GNNs?
**Answer:**
*   The output of the GNN should not depend on the order in which we list the nodes in the adjacency matrix.
*   If we re-index nodes (1 becomes 5, 5 becomes 1), the result should be the same (permuted).
*   Sum/Mean/Max aggregators are permutation invariant.

### 3. Explain the "Message Passing" paradigm.
**Answer:**
*   Nodes send messages to their neighbors.
*   A node aggregates received messages.
*   The node updates its own state based on the aggregated message and its previous state.

### 4. What is "Oversmoothing"?
**Answer:**
*   As GNN layers increase, node representations converge to the same value (stationary distribution of the random walk).
*   Information from the whole graph is washed out.
*   Limits GNN depth to 2-4 layers usually.

### 5. What is the difference between GCN and GAT?
**Answer:**
*   **GCN**: Fixed weights for neighbors (based on degree). Isotropic.
*   **GAT**: Learned weights (Attention) for neighbors. Anisotropic. Can prioritize important connections.

### 6. What is "GraphSAGE"?
**Answer:**
*   Inductive framework.
*   Samples a fixed number of neighbors (e.g., 10) instead of using all neighbors.
*   Aggregates them (Mean/LSTM/Pool).
*   Allows training on massive graphs by mini-batching.

### 7. What is "Weisfeiler-Lehman (WL) Test"?
**Answer:**
*   A classical algorithm to test graph isomorphism (are two graphs the same?).
*   Standard MPNNs are at most as powerful as the 1-WL test.
*   They cannot distinguish certain structures (e.g., triangles vs other cycles) if node degrees are identical.

### 8. How do you handle "Edge Features"?
**Answer:**
*   In standard GCN, edges are just binary (exist or not).
*   To use edge features (e.g., distance, bond type), we can include them in the message function: $m_{uv} = f(h_u, h_v, e_{uv})$.

### 9. What is a "Heterogeneous Graph"?
**Answer:**
*   A graph with multiple types of nodes and edges.
*   Example: E-commerce (User, Item, Shop).
*   Requires specialized GNNs (HeteroConv, R-GCN) that have specific parameters for each relation type.

### 10. What is "Global Pooling" (Readout)?
**Answer:**
*   Aggregating all node embeddings to get a single graph embedding.
*   Used for Graph Classification (e.g., is this molecule toxic?).
*   Sum, Mean, or Max pooling.

### 11. What is "Spectral Convolution"?
**Answer:**
*   Defining convolution in the Fourier domain using the Graph Laplacian eigenvectors.
*   GCN is a first-order approximation of Spectral Convolution (ChebNet).

### 12. Why is "Adjacency Matrix" sparse?
**Answer:**
*   Most real-world graphs are sparse (avg degree << N).
*   Storing full $N \times N$ matrix is $O(N^2)$ memory (impossible for 1M nodes).
*   We use Sparse Matrices (COO/CSR) or Edge Lists ($2 \times E$).

### 13. What is "Self-Loop" in GCN?
**Answer:**
*   Adding an edge from a node to itself.
*   Ensures the node's own features are included in the update step (otherwise it only sees neighbors).
*   $\tilde{A} = A + I$.

### 14. How do you train GNNs on massive graphs (1B nodes)?
**Answer:**
*   **Neighbor Sampling** (GraphSAGE).
*   **ClusterGCN**: Partition graph into clusters, train on clusters.
*   **Simplified GCN (SGC)**: Precompute powers of $A$, remove non-linearities.

### 15. What is "Link Prediction"?
**Answer:**
*   Predicting if an edge should exist.
*   Self-supervised task.
*   Model outputs similarity score for pairs.
*   Used for Friend Recommendation.

### 16. What is "PyTorch Geometric"?
**Answer:**
*   Library built on PyTorch for GNNs.
*   Handles sparse operations, scatter/gather, and standard datasets.

### 17. What is "Graph Isomorphism Network" (GIN)?
**Answer:**
*   A GNN architecture designed to be as powerful as the WL test.
*   Uses Sum aggregator and MLP for update.
*   Theoretically most expressive MPNN.

### 18. Can GNNs handle dynamic graphs?
**Answer:**
*   Yes. **Temporal Graph Networks (TGN)**.
*   Combine GNN (space) with RNN (time).

### 19. What is "Label Propagation"?
**Answer:**
*   Classical algorithm. Propagate labels to neighbors.
*   GCN can be seen as a learnable, feature-aware version of Label Propagation.

### 20. Applications of GNNs?
**Answer:**
*   Drug Discovery (Molecules).
*   Traffic Prediction (Road networks).
*   Recommender Systems (User-Item bipartite graphs).
*   Chip Design (Circuit graphs).
