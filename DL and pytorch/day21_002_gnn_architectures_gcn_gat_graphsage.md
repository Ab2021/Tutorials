# Day 21.2: GNN Architectures - GCN, GAT, and GraphSAGE Deep Dive

## Overview

Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE represent three fundamental paradigms in graph neural network architecture design, each addressing different aspects of the core challenges in graph-based learning through distinct mathematical frameworks and computational approaches that have shaped the development of modern graph neural networks. Understanding the theoretical foundations, architectural innovations, and practical implementations of these seminal approaches provides essential knowledge for designing effective graph-based machine learning systems and appreciating the evolution of graph neural network architectures from spectral methods through attention mechanisms to scalable inductive learning frameworks. This comprehensive exploration examines the mathematical derivations underlying each architecture, their computational complexity and scalability properties, the theoretical guarantees and limitations of each approach, and the practical considerations for implementing and optimizing these architectures across diverse graph learning tasks from node classification and link prediction to graph-level property prediction and generation.

## Graph Convolutional Networks (GCN)

### Theoretical Foundation and Mathematical Derivation

**Spectral Graph Theory Foundation**:
GCN emerged from the desire to extend convolutional neural networks to non-Euclidean graph-structured data. The theoretical foundation builds on spectral graph theory, where convolution is defined through the graph Fourier transform.

**Graph Fourier Transform**:
For a signal $\mathbf{x} \in \mathbb{R}^n$ on graph vertices, the Graph Fourier Transform is:
$$\hat{\mathbf{x}} = \mathbf{U}^T \mathbf{x}$$

where $\mathbf{U}$ contains the eigenvectors of the normalized graph Laplacian:
$$\mathcal{L} = \mathbf{I}_n - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$

**Spectral Convolution**:
Convolution in the spectral domain is element-wise multiplication:
$$\mathbf{g} \star \mathbf{x} = \mathbf{U} \left( (\mathbf{U}^T \mathbf{g}) \odot (\mathbf{U}^T \mathbf{x}) \right)$$

where $\mathbf{g}$ is a filter and $\odot$ denotes element-wise multiplication.

**Parameterized Filters**:
A learnable filter can be parameterized as:
$$g_{\boldsymbol{\theta}} = \text{diag}(\boldsymbol{\theta})$$

where $\boldsymbol{\theta} = [\theta_1, \theta_2, \ldots, \theta_n]$ are learnable parameters.

**Computational Challenges**:
Direct spectral convolution has several limitations:
1. **Computational Complexity**: $O(n^3)$ for eigendecomposition
2. **Non-localized Filters**: Filters are not spatially localized
3. **Basis Dependence**: Filters depend on graph structure (not transferable)

### Chebyshev Polynomial Approximation

**Polynomial Approximation Framework**:
To address computational challenges, GCN uses Chebyshev polynomial approximation:
$$g_{\boldsymbol{\theta}}(\boldsymbol{\Lambda}) \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\boldsymbol{\Lambda}})$$

where $T_k$ are Chebyshev polynomials and $\tilde{\boldsymbol{\Lambda}} = \frac{2\boldsymbol{\Lambda}}{\lambda_{\max}} - \mathbf{I}$.

**Chebyshev Polynomials**:
Defined recursively:
$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$$

**Localized Convolution**:
The approximated convolution becomes:
$$\mathbf{g}_{\boldsymbol{\theta}} \star \mathbf{x} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\mathcal{L}}) \mathbf{x}$$

where $\tilde{\mathcal{L}} = \frac{2\mathcal{L}}{\lambda_{\max}} - \mathbf{I}$.

**Key Properties**:
- **Localized**: K-hop neighborhood influence
- **Efficient**: $O(K|\mathcal{E}|)$ complexity (linear in edges)
- **Transferable**: No dependence on specific eigenvectors

### Simplified GCN Layer

**First-Order Approximation**:
Kipf and Welling simplified the Chebyshev approach by:
1. Setting $K = 1$ (first-order approximation)
2. Assuming $\lambda_{\max} \approx 2$
3. Setting $\theta_0 = -\theta_1 = \theta$

This yields:
$$\mathbf{g}_{\boldsymbol{\theta}} \star \mathbf{x} \approx \theta (\mathbf{I} + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}) \mathbf{x}$$

**Renormalization Trick**:
To prevent exploding/vanishing gradients:
$$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$
$$\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$$

**Final GCN Layer Formulation**:
$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

where:
- $\mathbf{H}^{(l)} \in \mathbb{R}^{n \times d_l}$: Node features at layer $l$
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$: Learnable weight matrix
- $\sigma$: Non-linear activation function

### Mathematical Properties and Analysis

**Symmetry and Normalization**:
The normalized adjacency matrix $\mathbf{S} = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$ has important properties:
- **Symmetric**: $\mathbf{S} = \mathbf{S}^T$
- **Bounded Eigenvalues**: $\lambda(\mathbf{S}) \in [-1, 1]$
- **Smooth Operation**: Promotes similarity between connected nodes

**Message Passing Interpretation**:
GCN can be interpreted as a message passing algorithm:
$$\mathbf{m}_{ij}^{(l+1)} = \frac{1}{\sqrt{d_i d_j}} \mathbf{h}_j^{(l)} \mathbf{W}^{(l)}$$
$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \mathbf{m}_{ij}^{(l+1)}\right)$$

**Theoretical Guarantees**:
- **Universal Approximation**: Multi-layer GCNs can approximate any permutation-invariant function on graphs
- **Localization**: K-layer GCN has receptive field of K-hop neighborhood
- **Stability**: Bounded propagation due to normalized operations

### Over-smoothing Analysis

**Mathematical Formulation**:
After $L$ GCN layers:
$$\mathbf{H}^{(L)} = \mathbf{S}^L \mathbf{H}^{(0)} \prod_{l=0}^{L-1} \mathbf{W}^{(l)}$$

**Spectral Analysis**:
If $\mathbf{S}$ has dominant eigenvalue with uniform eigenvector, node representations converge:
$$\lim_{L \to \infty} \mathbf{H}^{(L)} \propto \mathbf{1} \mathbf{v}^T$$

**Mitigation Strategies**:
1. **Residual Connections**: $\mathbf{H}^{(l+1)} = \mathbf{H}^{(l)} + \mathbf{S} \mathbf{H}^{(l)} \mathbf{W}^{(l)}$
2. **Jumping Knowledge**: Combine representations from all layers
3. **DropEdge**: Randomly remove edges during training

## Graph Attention Networks (GAT)

### Attention Mechanism Foundation

**Motivation**: 
GAT addresses limitations of GCN by introducing attention mechanisms that allow nodes to selectively attend to their neighbors, providing:
- **Adaptive Aggregation**: Different importance weights for neighbors
- **Interpretability**: Attention weights show which neighbors are important
- **Inductive Capability**: Can generalize to unseen graph structures

**Self-Attention for Graphs**:
For node $i$ with neighbors $\mathcal{N}(i)$, compute attention coefficients:
$$e_{ij} = a(\mathbf{W} \mathbf{h}_i, \mathbf{W} \mathbf{h}_j)$$

where $a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$ is the attention mechanism.

### Single-Head Attention Mechanism

**Attention Function Design**:
GAT uses a single-layer feedforward network:
$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \| \mathbf{W} \mathbf{h}_j]\right)$$

where:
- $\mathbf{W} \in \mathbb{R}^{F' \times F}$: Learnable linear transformation
- $\mathbf{a} \in \mathbb{R}^{2F'}$: Learnable attention parameters
- $\|$: Concatenation operation

**Normalized Attention Coefficients**:
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

**Feature Update**:
$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

### Multi-Head Attention

**Motivation**: 
Multi-head attention enables the model to attend to different aspects of neighbor information simultaneously.

**Multi-Head Computation**:
$$\mathbf{h}_i^{(k)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)$$

**Concatenation** (for intermediate layers):
$$\mathbf{h}_i' = \|_{k=1}^{K} \mathbf{h}_i^{(k)}$$

**Averaging** (for output layer):
$$\mathbf{h}_i' = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)$$

### Theoretical Analysis of GAT

**Attention as Graph Filtering**:
GAT can be viewed as applying a learnable, adaptive filter to the graph:
$$\mathbf{H}' = \sigma\left(\mathbf{A}_{\text{att}} \mathbf{H} \mathbf{W}\right)$$

where $\mathbf{A}_{\text{att}}$ has entries $\alpha_{ij}$.

**Expressiveness**:
GAT is strictly more expressive than GCN because:
1. **Adaptive Weights**: Can assign different importance to different neighbors
2. **Non-uniform Aggregation**: Not limited to degree-normalized aggregation
3. **Multiple Attention Heads**: Can capture different types of relationships

**Inductive Learning Capability**:
GAT can generalize to unseen nodes and graphs because:
- Attention computation only depends on node features
- No dependence on global graph properties
- Can handle variable neighborhood sizes

### Computational Complexity

**Time Complexity**:
For graph with $n$ nodes, $m$ edges, and $K$ attention heads:
- **Attention Computation**: $O(K \cdot m \cdot F')$
- **Feature Transformation**: $O(n \cdot F \cdot F')$
- **Total**: $O(K \cdot m \cdot F' + n \cdot F \cdot F')$

**Space Complexity**:
- **Attention Coefficients**: $O(m)$ (can be computed on-the-fly)
- **Parameters**: $O(K \cdot F \cdot F' + K \cdot 2F')$

### Attention Interpretability

**Attention Weight Analysis**:
The learned attention weights $\alpha_{ij}$ provide insights into:
- **Structural Importance**: Which graph connections are most relevant
- **Feature Relevance**: How different node features influence attention
- **Task-Specific Patterns**: Attention patterns vary across different tasks

**Visualization Techniques**:
1. **Edge Weight Visualization**: Color edges by attention weights
2. **Attention Distribution Plots**: Histogram of attention values
3. **Attention Entropy**: Measure of attention concentration

**Mathematical Properties of Attention**:
- **Permutation Invariance**: $\sum_{j \in \mathcal{N}(i)} \alpha_{ij} = 1$
- **Sparsity**: Often exhibits sparse attention patterns
- **Asymmetry**: $\alpha_{ij} \neq \alpha_{ji}$ in general

## GraphSAGE (Sample and Aggregate)

### Inductive Learning Framework

**Transductive vs Inductive Learning**:
- **Transductive** (GCN): Fixed graph structure, cannot generalize to new nodes
- **Inductive** (GraphSAGE): Learn node embeddings for unseen nodes and graphs

**Core Innovation**:
GraphSAGE learns a function to generate embeddings based on node's local neighborhood structure and features, rather than maintaining embeddings for specific nodes.

**General Framework**:
For each node $v$ at layer $k$:
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k\left(\left\{\mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v)\right\}\right)$$
$$\mathbf{h}_v^{(k)} = \sigma\left(\mathbf{W}^{(k)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(k-1)}, \mathbf{h}_{\mathcal{N}(v)}^{(k)}\right)\right)$$

### Aggregation Functions

**Mean Aggregator**:
$$\text{AGGREGATE}_k^{\text{mean}} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$$

**Properties**:
- **Permutation Invariant**: Order of neighbors doesn't matter
- **Differentiable**: Supports end-to-end training
- **Simple**: Computationally efficient

**LSTM Aggregator**:
$$\text{AGGREGATE}_k^{\text{LSTM}} = \text{LSTM}\left(\text{RANDOM-PERMUTATION}\left(\left\{\mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v)\right\}\right)\right)$$

**Properties**:
- **Higher Capacity**: Can model complex neighbor interactions
- **Sequential Processing**: May capture ordering information (when desired)
- **Computational Cost**: More expensive than mean aggregator

**Pooling Aggregator**:
$$\text{AGGREGATE}_k^{\text{pool}} = \max\left(\left\{\sigma\left(\mathbf{W}_{\text{pool}} \mathbf{h}_{u}^{(k-1)} + \mathbf{b}\right), \forall u \in \mathcal{N}(v)\right\}\right)$$

**Properties**:
- **Max Operation**: Captures most salient neighbor features
- **Learnable Transformation**: $\mathbf{W}_{\text{pool}}$ and $\mathbf{b}$ are learnable
- **Non-linear**: Can model complex neighbor relationships

### Neighborhood Sampling

**Motivation**:
For nodes with large neighborhoods, aggregating over all neighbors is computationally expensive and may lead to over-smoothing.

**Sampling Strategy**:
For each layer $k$, sample $S_k$ neighbors uniformly at random:
$$\hat{\mathcal{N}}^k(v) = \text{SAMPLE}(\mathcal{N}(v), S_k)$$

**Benefits**:
1. **Computational Efficiency**: Fixed computational cost per node
2. **Regularization**: Acts as implicit regularization
3. **Scalability**: Enables training on large graphs

**Theoretical Analysis**:
Under certain conditions, the sampled aggregation converges to the full aggregation:
$$\mathbb{E}\left[\text{AGGREGATE}(\{\mathbf{h}_u : u \in \hat{\mathcal{N}}(v)\})\right] \approx \text{AGGREGATE}(\{\mathbf{h}_u : u \in \mathcal{N}(v)\})$$

### Training and Optimization

**Unsupervised Loss**:
Encourage nearby nodes to have similar representations:
$$J_G(\mathbf{z}_u) = -\log\left(\sigma(\mathbf{z}_u^T \mathbf{z}_v)\right) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log\left(\sigma(-\mathbf{z}_u^T \mathbf{z}_{v_n})\right)$$

where:
- $v$ is a node co-occurring with $u$ on random walk
- $v_n$ are negative samples
- $P_n(v)$ is negative sampling distribution

**Supervised Loss**:
Standard cross-entropy for node classification:
$$J_G = -\sum_{v \in \mathcal{V}_{\text{labeled}}} \sum_{c=1}^{C} y_{v,c} \log\left(\sigma(\mathbf{W}^{(out)} \mathbf{z}_v)_c\right)$$

**Mini-batch Training**:
1. **Sample target nodes**: $\mathcal{B} \subset \mathcal{V}$
2. **Sample neighborhoods**: For each layer and each node in current batch
3. **Compute forward pass**: Through sampled subgraph
4. **Compute gradients**: Only for parameters affecting current batch

### Theoretical Properties

**Universal Approximation**:
Under appropriate conditions, GraphSAGE can approximate any function defined on the graph.

**Generalization Bound**:
For inductive learning, GraphSAGE has generalization bounds that depend on:
- **Sample Complexity**: Number of training nodes
- **Neighborhood Diversity**: Variety in sampled neighborhoods  
- **Aggregator Properties**: Theoretical properties of aggregation function

**Inductive Capability**:
Key advantages for inductive learning:
1. **Feature-based**: Relies only on node features, not node identity
2. **Local**: Decision based on local neighborhood structure
3. **Generalizable**: Can handle nodes not seen during training

## Comparative Analysis

### Computational Complexity Comparison

**GCN**:
- **Time**: $O(|\mathcal{E}| \cdot F \cdot F')$ per layer
- **Space**: $O(|\mathcal{V}| \cdot F' + |\mathcal{E}|)$
- **Scalability**: Limited by full graph operations

**GAT**:
- **Time**: $O(|\mathcal{E}| \cdot F \cdot F' \cdot K)$ per layer
- **Space**: $O(|\mathcal{V}| \cdot F' \cdot K + |\mathcal{E}| \cdot K)$
- **Scalability**: Attention computation can be expensive

**GraphSAGE**:
- **Time**: $O(|\mathcal{V}| \cdot S^L \cdot F \cdot F')$ per layer (with sampling)
- **Space**: $O(|\mathcal{V}| \cdot F')$
- **Scalability**: Best for large graphs due to sampling

### Expressive Power Comparison

**GCN**:
- **Limitations**: Fixed aggregation weights based on graph structure
- **Strengths**: Spectral foundation provides theoretical guarantees
- **Best for**: Homophilous graphs with smooth node features

**GAT**:
- **Limitations**: Quadratic complexity in neighborhood size
- **Strengths**: Adaptive attention, interpretable, handles heterophily
- **Best for**: Graphs requiring selective neighbor attention

**GraphSAGE**:
- **Limitations**: Sampling may lose information
- **Strengths**: Inductive learning, scalable, flexible aggregators
- **Best for**: Large-scale graphs, inductive tasks

### Practical Considerations

**When to Use GCN**:
- Small to medium graphs ($< 10^5$ nodes)
- Homophilous graphs (connected nodes have similar labels)
- Need for spectral interpretation
- Transductive learning scenarios

**When to Use GAT**:
- Need for interpretable attention weights  
- Heterophilous graphs (connected nodes may have different labels)
- Multi-relational graphs
- When different neighbors have different importance

**When to Use GraphSAGE**:
- Large graphs ($> 10^5$ nodes)
- Inductive learning (new nodes at test time)
- Need for scalability
- When sampling is acceptable approximation

## Implementation Considerations

### Memory Optimization

**Gradient Checkpointing**:
Trade computation for memory by recomputing intermediate activations:
```python
def checkpoint_layer(layer, input):
    return torch.utils.checkpoint.checkpoint(layer, input)
```

**Sparse Operations**:
Utilize sparse matrix operations for adjacency matrices:
- COO (Coordinate) format for construction
- CSR (Compressed Sparse Row) for matrix operations
- Memory savings: $O(|\mathcal{E}|)$ instead of $O(|\mathcal{V}|^2)$

### Training Stability

**Gradient Clipping**:
Prevent exploding gradients in deep GNNs:
$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\
\frac{\tau \mathbf{g}}{\|\mathbf{g}\|} & \text{otherwise}
\end{cases}$$

**Layer Normalization**:
Normalize features at each layer:
$$\mathbf{h}_i^{(l)} = \frac{\mathbf{h}_i^{(l)} - \boldsymbol{\mu}^{(l)}}{\boldsymbol{\sigma}^{(l)}}$$

**Residual Connections**:
Add skip connections to mitigate over-smoothing:
$$\mathbf{H}^{(l+1)} = \mathbf{H}^{(l)} + \text{GNN-Layer}(\mathbf{H}^{(l)})$$

## Key Questions for Review

### Theoretical Understanding
1. **Spectral vs Spatial**: How do spectral methods (GCN) differ from spatial methods (GAT, GraphSAGE) in their theoretical foundations?

2. **Over-smoothing**: Why do GCNs suffer from over-smoothing, and how do GAT and GraphSAGE address this issue?

3. **Expressiveness**: What makes GAT more expressive than GCN, and when does this additional expressiveness matter?

### Architectural Design
4. **Attention Mechanisms**: How does the attention mechanism in GAT enable better handling of heterophilous graphs?

5. **Aggregation Functions**: What are the trade-offs between different aggregation functions in GraphSAGE?

6. **Sampling Strategies**: How does neighborhood sampling in GraphSAGE affect model performance and scalability?

### Practical Applications
7. **Transductive vs Inductive**: When should you choose transductive (GCN) versus inductive (GraphSAGE) learning approaches?

8. **Scalability**: How do the computational complexities of these methods affect their applicability to large-scale graphs?

9. **Interpretability**: How can attention weights in GAT be used for graph analysis and interpretation?

### Implementation Details
10. **Normalization**: What role does the renormalization trick play in GCN, and why is it necessary?

11. **Multi-head Attention**: How does multi-head attention in GAT compare to multi-head attention in Transformers?

12. **Neighborhood Definition**: How does the definition of neighborhood affect the performance of each architecture?

### Advanced Topics
13. **Theoretical Guarantees**: What theoretical guarantees exist for the approximation capabilities of each architecture?

14. **Generalization**: How do these architectures generalize to different graph types and domains?

15. **Optimization Landscapes**: How do the optimization landscapes differ between these architectures?

## Conclusion

Graph Convolutional Networks, Graph Attention Networks, and GraphSAGE represent three foundational paradigms in graph neural network architecture design, each addressing different fundamental challenges in graph-based learning through distinct mathematical frameworks that have shaped the development of the entire field of geometric deep learning. These architectures demonstrate the evolution from spectral methods that leverage mathematical properties of graph Laplacians, through attention mechanisms that enable adaptive neighbor selection, to inductive frameworks that enable scalable learning on large and dynamic graph structures.

**Complementary Strengths**: Each architecture addresses specific limitations of graph-based learning - GCN provides a solid spectral foundation with efficient computation, GAT introduces adaptive attention for heterogeneous graphs, and GraphSAGE enables inductive learning for large-scale applications, with their combined insights forming the foundation for modern graph neural network design.

**Theoretical Foundations**: The mathematical principles underlying these architectures - from spectral graph theory and Chebyshev approximations in GCN, through attention mechanisms and multi-head architectures in GAT, to sampling theory and inductive learning in GraphSAGE - provide the theoretical framework for understanding when and why each approach works effectively.

**Practical Impact**: These architectures have enabled graph neural networks to address real-world problems across diverse domains, from social network analysis and recommendation systems to molecular property prediction and knowledge graph reasoning, demonstrating the practical value of principled architectural design in geometric deep learning.

**Future Directions**: Understanding these foundational architectures provides the knowledge base for developing novel graph neural network architectures that combine their complementary strengths while addressing their individual limitations, pointing toward more expressive, scalable, and theoretically grounded approaches to learning on graph-structured data.

The insights gained from studying GCN, GAT, and GraphSAGE are essential for anyone working with graph neural networks, providing both the theoretical understanding necessary for principled architecture design and the practical knowledge required for effective implementation and optimization in real-world applications. These architectures continue to serve as building blocks for more sophisticated approaches while providing the fundamental concepts that define the field of graph neural networks.