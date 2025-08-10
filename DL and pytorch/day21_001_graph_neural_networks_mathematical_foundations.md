# Day 21.1: Graph Neural Networks - Mathematical Foundations and Theory

## Overview

Graph Neural Networks (GNNs) represent a fundamental paradigm shift in deep learning that extends neural architectures to non-Euclidean graph-structured data, enabling the modeling of complex relational patterns and dependencies that exist in molecular structures, social networks, knowledge graphs, and countless other domains where traditional convolutional and recurrent approaches fail to capture the inherent structural relationships. Understanding the mathematical foundations of graph theory, the theoretical principles underlying message passing frameworks, the spectral and spatial approaches to graph convolutions, and the theoretical guarantees regarding expressiveness and generalization provides essential knowledge for developing effective graph-based machine learning systems. This comprehensive exploration examines the mathematical theory of graphs and their representations, the fundamental principles of graph neural architectures, the theoretical analysis of GNN expressiveness and limitations, and the foundational concepts that enable learning on irregular, non-grid-like data structures through principled mathematical frameworks that bridge discrete mathematics, linear algebra, and deep learning theory.

## Graph Theory Fundamentals

### Mathematical Definition of Graphs

**Graph Structure**:
A graph $G = (V, E)$ consists of:
- **Vertices (Nodes)**: $V = \{v_1, v_2, \ldots, v_n\}$ with $|V| = n$
- **Edges**: $E \subseteq V \times V$ with $|E| = m$

**Types of Graphs**:
- **Undirected**: $(v_i, v_j) = (v_j, v_i)$
- **Directed**: $(v_i, v_j) \neq (v_j, v_i)$ in general
- **Weighted**: $w: E \rightarrow \mathbb{R}$ assigns weights to edges
- **Attributed**: Nodes and/or edges have feature vectors

**Adjacency Matrix**:
$$A_{ij} = \begin{cases}
w_{ij} & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

**Properties**:
- **Symmetric**: $A = A^T$ for undirected graphs
- **Sparse**: $\|A\|_0 \ll n^2$ in most real networks
- **Non-negative**: $A_{ij} \geq 0$ for positive edge weights

### Graph Laplacian Theory

**Degree Matrix**:
$$D_{ii} = \sum_{j=1}^{n} A_{ij} = \deg(v_i)$$

**Graph Laplacian**:
$$L = D - A$$

**Normalized Laplacians**:
- **Symmetric**: $L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$
- **Random Walk**: $L_{rw} = D^{-1} L = I - D^{-1} A$

**Properties of Graph Laplacian**:
1. **Positive Semidefinite**: $\mathbf{x}^T L \mathbf{x} \geq 0$ for all $\mathbf{x}$
2. **Quadratic Form**: $\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij} (x_i - x_j)^2$
3. **Spectrum**: $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
4. **Multiplicity of zero eigenvalue equals number of connected components**

**Spectral Properties**:
$$L \mathbf{u}_k = \lambda_k \mathbf{u}_k$$

where $\{\mathbf{u}_k\}_{k=1}^n$ form orthonormal eigenvector basis.

### Graph Signal Processing

**Graph Signals**:
A signal on graph $G$ is a function $f: V \rightarrow \mathbb{R}^d$:
$$\mathbf{f} = [f(v_1), f(v_2), \ldots, f(v_n)]^T \in \mathbb{R}^{n \times d}$$

**Graph Fourier Transform**:
$$\hat{f}(\lambda_k) = \langle \mathbf{f}, \mathbf{u}_k \rangle = \sum_{i=1}^{n} f(v_i) u_k(v_i)$$

**Inverse Transform**:
$$f(v_i) = \sum_{k=1}^{n} \hat{f}(\lambda_k) u_k(v_i)$$

**Frequency Interpretation**:
- **Low frequencies**: $\lambda_k$ small → smooth signals
- **High frequencies**: $\lambda_k$ large → oscillatory signals

**Graph Convolution in Spectral Domain**:
$$(\mathbf{g} * \mathbf{f})_G = \mathbf{U} \left( (\mathbf{U}^T \mathbf{g}) \odot (\mathbf{U}^T \mathbf{f}) \right)$$

where $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n]$.

## Message Passing Framework

### General Message Passing Paradigm

**Message Passing Neural Network (MPNN)**:
At layer $\ell$, for each node $v_i$:

**Message Computation**:
$$\mathbf{m}_{ij}^{(\ell+1)} = M^{(\ell)} \left( \mathbf{h}_i^{(\ell)}, \mathbf{h}_j^{(\ell)}, \mathbf{e}_{ij} \right)$$

**Message Aggregation**:
$$\mathbf{a}_i^{(\ell+1)} = \text{AGG}^{(\ell)} \left( \left\{ \mathbf{m}_{ij}^{(\ell+1)} : j \in \mathcal{N}(i) \right\} \right)$$

**Node Update**:
$$\mathbf{h}_i^{(\ell+1)} = U^{(\ell)} \left( \mathbf{h}_i^{(\ell)}, \mathbf{a}_i^{(\ell+1)} \right)$$

**Functions**:
- $M^{(\ell)}$: Message function (learnable)
- $\text{AGG}^{(\ell)}$: Aggregation function (typically permutation invariant)
- $U^{(\ell)}$: Update function (learnable)

### Theoretical Properties

**Permutation Equivariance**:
For permutation matrix $P$:
$$\text{GNN}(P\mathbf{H}, PAP^T) = P \cdot \text{GNN}(\mathbf{H}, A)$$

**Proof Sketch**:
Message passing operations respect node ordering invariance through:
1. **Symmetric aggregation**: $\text{AGG}$ is permutation invariant
2. **Local updates**: Only depend on node and neighbor features
3. **Adjacency transformation**: $PAP^T$ preserves graph structure

**Translation to Expressive Power**:
GNNs can only distinguish graphs that have different multisets of node features after message passing iterations.

### Aggregation Functions

**Common Choices**:

**Sum Aggregation**:
$$\text{AGG}_{\text{sum}}(\{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_k\}) = \sum_{i=1}^{k} \mathbf{m}_i$$

**Mean Aggregation**:
$$\text{AGG}_{\text{mean}}(\{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_k\}) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{m}_i$$

**Max Aggregation**:
$$\text{AGG}_{\text{max}}(\{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_k\}) = \max_{i=1}^{k} \mathbf{m}_i$$

**Attention Aggregation**:
$$\text{AGG}_{\text{att}}(\{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_k\}) = \sum_{i=1}^{k} \alpha_i \mathbf{m}_i$$

where $\alpha_i = \text{softmax}(f(\mathbf{m}_i))$.

**Theoretical Analysis**:
- **Sum/Mean**: Preserve multiset cardinality information
- **Max**: Focus on most significant features
- **Attention**: Adaptive importance weighting

## Spectral Graph Neural Networks

### Spectral Convolution Theory

**Motivation**: Extend CNN convolution to graphs via spectral domain.

**Spectral Filter**:
$$g_\theta = \text{diag}(\theta) = \text{diag}(\theta_1, \theta_2, \ldots, \theta_n)$$

**Spectral Graph Convolution**:
$$\mathbf{y} = g_\theta * \mathbf{x} = \mathbf{U} g_\theta \mathbf{U}^T \mathbf{x}$$

**Problems**:
1. **Computational Complexity**: $O(n^3)$ eigendecomposition
2. **Localization**: Filter affects entire graph
3. **Transferability**: Eigenvectors depend on specific graph structure

### Chebyshev Spectral CNNs

**Chebyshev Polynomial Approximation**:
$$g_\theta(\lambda) \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\lambda})$$

where $T_k$ are Chebyshev polynomials and $\tilde{\lambda} = \frac{2\lambda}{\lambda_{\max}} - 1$.

**Recursive Definition**:
$$T_0(x) = 1, \quad T_1(x) = x$$
$$T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$

**Chebyshev Convolution**:
$$\mathbf{y} = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) \mathbf{x}$$

where $\tilde{L} = \frac{2L}{\lambda_{\max}} - I$.

**Advantages**:
1. **Efficiency**: $O(Km)$ complexity (linear in edges)
2. **Localization**: K-hop neighborhood
3. **Stability**: Well-conditioned numerical properties

### Graph Convolutional Networks (GCN)

**Simplified Spectral Approach**:
First-order Chebyshev approximation with $\lambda_{\max} \approx 2$:
$$g_\theta * \mathbf{x} \approx \theta_0 \mathbf{x} + \theta_1 (L - I) \mathbf{x} = \theta_0 \mathbf{x} - \theta_1 D^{-1/2} A D^{-1/2} \mathbf{x}$$

**Further Simplification**:
Set $\theta_0 = -\theta_1 = \theta$:
$$g_\theta * \mathbf{x} \approx \theta (I + D^{-1/2} A D^{-1/2}) \mathbf{x}$$

**Renormalization Trick**:
To prevent exploding/vanishing gradients:
$$\tilde{A} = A + I, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

**GCN Layer**:
$$\mathbf{H}^{(\ell+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

**Matrix Form Analysis**:
$$\mathbf{S} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

Properties of $\mathbf{S}$:
- **Symmetric**: $\mathbf{S} = \mathbf{S}^T$
- **Normalized**: Eigenvalues in $[-1, 1]$
- **Smooth**: Promotes similarity between connected nodes

## Spatial Graph Neural Networks

### Neighborhood Aggregation

**Spatial Perspective**:
Instead of spectral transforms, directly operate on graph topology.

**GraphSAGE Framework**:
$$\mathbf{h}_v^{(\ell+1)} = \sigma \left( \mathbf{W}^{(\ell)} \cdot \text{CONCAT} \left( \mathbf{h}_v^{(\ell)}, \text{AGG} \left( \left\{ \mathbf{h}_u^{(\ell)} : u \in \mathcal{N}(v) \right\} \right) \right) \right)$$

**Aggregation Variants**:

**Mean Aggregator**:
$$\text{AGG}_{\text{mean}} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(\ell)}$$

**LSTM Aggregator**:
$$\text{AGG}_{\text{LSTM}} = \text{LSTM} \left( \text{RANDOM-PERMUTATION} \left( \left\{ \mathbf{h}_u^{(\ell)} : u \in \mathcal{N}(v) \right\} \right) \right)$$

**Pooling Aggregator**:
$$\text{AGG}_{\text{pool}} = \max \left( \left\{ \sigma(\mathbf{W}_{\text{pool}} \mathbf{h}_u^{(\ell)} + \mathbf{b}) : u \in \mathcal{N}(v) \right\} \right)$$

### Graph Attention Networks (GAT)

**Attention Mechanism**:
Compute attention coefficients:
$$e_{ij} = \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_i \| \mathbf{W} \mathbf{h}_j] \right)$$

**Normalized Attention**:
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

**Feature Update**:
$$\mathbf{h}_i' = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right)$$

**Multi-Head Attention**:
$$\mathbf{h}_i' = \|_{k=1}^{K} \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j \right)$$

**Final Layer** (average instead of concatenate):
$$\mathbf{h}_i' = \sigma \left( \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j \right)$$

**Theoretical Properties**:
1. **Self-attention**: Can attend to node's own features
2. **Neighborhood Focus**: Attention weights sum to 1 over neighborhood
3. **Inductive Capability**: Can generalize to unseen graphs

## Expressiveness and Theoretical Limitations

### Weisfeiler-Lehman Test

**Graph Isomorphism Testing**:
The $k$-dimensional Weisfeiler-Lehman ($k$-WL) test is a combinatorial algorithm for graph isomorphism.

**1-WL Algorithm**:
1. **Initialization**: Assign initial colors $c^{(0)}(v)$ to nodes
2. **Iteration**: For $t = 1, 2, \ldots$:
   $$c^{(t)}(v) = \text{HASH} \left( c^{(t-1)}(v), \{\{c^{(t-1)}(u) : u \in \mathcal{N}(v)\}\} \right)$$
3. **Termination**: When colors stabilize

**Connection to GNNs**:
**Theorem**: The expressive power of standard GNNs is at most as powerful as the 1-WL test.

**Proof Sketch**:
1. Both update node representations based on neighborhood multisets
2. Both use permutation-invariant aggregation
3. 1-WL can simulate any GNN computation
4. GNNs cannot distinguish graphs that 1-WL cannot distinguish

### Limitations of Standard GNNs

**Regular Structures**:
Standard GNNs cannot distinguish:
- Complete graphs of same size
- Regular graphs with same degree
- Certain geometric graphs

**Example - Cycle Graphs**:
All nodes in cycle $C_n$ have identical 1-WL colors after any number of iterations.

**Over-smoothing Problem**:
As depth increases, node representations become increasingly similar:
$$\lim_{\ell \to \infty} \mathbf{h}_i^{(\ell)} = \mathbf{h}_j^{(\ell)} \quad \forall i, j$$

**Mathematical Analysis**:
For GCN with $\mathbf{S} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$:
$$\mathbf{H}^{(\ell)} = \mathbf{S}^\ell \mathbf{H}^{(0)} \prod_{k=0}^{\ell-1} \mathbf{W}^{(k)}$$

If $\mathbf{S}$ has dominant eigenvalue with uniform eigenvector, representations converge to constant.

### Higher-Order Extensions

**$k$-GNNs**:
Extend message passing to $k$-tuples of nodes:
$$\mathbf{m}^{(\ell+1)}(v_1, \ldots, v_k) = M^{(\ell)} \left( \mathbf{h}^{(\ell)}(v_1, \ldots, v_k), \{\mathbf{h}^{(\ell)}(u_1, \ldots, u_k)\} \right)$$

**Complexity**: Exponential in $k$ - $O(n^k)$ representations.

**Higher-Order Graph Neural Networks**:
Use higher-order graph structures (simplicial complexes, hypergraphs) to capture $k$-way interactions.

**Graph Networks (GN)**:
Most general framework with:
- Node updates: $\mathbf{v}_i' = \phi^v(\mathbf{v}_i, \rho^{v \leftarrow e}(\{\mathbf{e}_{ki} : k \in \mathcal{N}(i)\}), \mathbf{u})$
- Edge updates: $\mathbf{e}_{ij}' = \phi^e(\mathbf{e}_{ij}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{u})$
- Global updates: $\mathbf{u}' = \phi^u(\rho^{u \leftarrow v}(\{\mathbf{v}_i'\}), \rho^{u \leftarrow e}(\{\mathbf{e}_{ij}'\}), \mathbf{u})$

## Universal Approximation Properties

### Graph Function Approximation

**Graph Functions**:
Functions $f: \mathcal{G} \rightarrow \mathbb{R}^d$ that map graphs to vectors.

**Permutation Invariant Functions**:
$$f(\sigma(G)) = f(G) \quad \forall \text{ permutation } \sigma$$

**Deep Sets Theorem**:
Any permutation invariant function can be written as:
$$f(\{x_1, \ldots, x_n\}) = \rho \left( \sum_{i=1}^{n} \phi(x_i) \right)$$

for suitable functions $\phi$ and $\rho$.

**Extension to Graphs**:
**Theorem**: Any permutation invariant graph function can be approximated by:
$$f(G) = \rho \left( \sum_{v \in V} \phi(\mathbf{h}_v^{(L)}) \right)$$

where $\mathbf{h}_v^{(L)}$ is the final node representation after $L$ layers.

### Approximation Bounds

**VC Dimension Analysis**:
The VC dimension of GNNs grows polynomially with:
- Number of parameters
- Graph size
- Network depth

**Sample Complexity**:
For $\epsilon$-approximation with probability $1-\delta$:
$$N = O\left( \frac{d \log(1/\epsilon) + \log(1/\delta)}{\epsilon^2} \right)$$

where $d$ is VC dimension.

**Generalization Bounds**:
$$\mathbb{E}[L(\theta)] \leq \hat{L}(\theta) + O\left( \sqrt{\frac{d \log N + \log(1/\delta)}{N}} \right)$$

## Inductive vs Transductive Learning

### Problem Formulations

**Transductive Learning**:
- **Goal**: Learn on fixed graph $G = (V, E)$
- **Examples**: Node classification on citation networks
- **Challenge**: Cannot generalize to new nodes/graphs

**Inductive Learning**:
- **Goal**: Learn generalizable representations
- **Examples**: Graph classification, molecular property prediction
- **Challenge**: Need to generalize across different graph structures

### Theoretical Differences

**Parameter Sharing**:
- **Transductive**: Can have node-specific parameters
- **Inductive**: Must share parameters across nodes/graphs

**Generalization**:
**Theorem**: Inductive GNNs have better generalization bounds when:
$$\text{Complexity}(\text{parameter space}) \ll \text{Complexity}(\text{node-specific space})$$

**Sample Efficiency**:
Inductive learning requires more diverse training graphs but generalizes better to unseen structures.

## Graph Neural ODE

### Continuous-Depth GNNs

**Motivation**: Replace discrete layers with continuous depth evolution.

**Graph Neural ODE**:
$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), A; \theta)$$

**Initial Condition**:
$$\mathbf{h}(0) = \mathbf{H}^{(0)}$$

**Output**:
$$\mathbf{h}(T) = \mathbf{h}(0) + \int_0^T f(\mathbf{h}(t), A; \theta) dt$$

**Advantages**:
1. **Adaptive depth**: Automatically determine optimal depth
2. **Memory efficiency**: Constant memory during backpropagation
3. **Theoretical analysis**: Tools from dynamical systems

### Stability Analysis

**Equilibrium Points**:
$$\mathbf{h}^* : f(\mathbf{h}^*, A; \theta) = 0$$

**Lyapunov Stability**:
Equilibrium $\mathbf{h}^*$ is stable if Jacobian $\nabla_{\mathbf{h}} f(\mathbf{h}^*, A; \theta)$ has negative real eigenvalues.

**Over-smoothing Prevention**:
Design $f$ such that:
$$\frac{d}{dt} \|\mathbf{h}_i(t) - \mathbf{h}_j(t)\|^2 \not\to -\infty$$

## Geometric Deep Learning Principles

### Symmetry and Invariance

**Group Actions**:
Let $G$ be a group acting on domain $\Omega$. For $g \in G$ and $x \in \Omega$:

**Invariance**:
$$f(g \cdot x) = f(x)$$

**Equivariance**:
$$f(g \cdot x) = \rho(g) \cdot f(x)$$

where $\rho$ is a representation of $G$.

**Applications to Graphs**:
- **Node permutation**: $S_n$ symmetry group
- **Graph isomorphism**: Automorphism group
- **Spatial transformations**: Euclidean group for geometric graphs

### Universality of Geometric Deep Learning

**Fundamental Theorem**: Any equivariant linear layer between feature spaces can be characterized by the symmetries of the domain.

**Graph Case**:
The most general permutation equivariant linear layer is:
$$L(\mathbf{X}) = \mathbf{X} \mathbf{A} + \mathbf{P} \mathbf{X} \mathbf{B}$$

where $\mathbf{P}$ is permutation matrix representation of adjacency.

This recovers GCN, GraphSAGE, and other architectures as special cases.

## Key Questions for Review

### Mathematical Foundations
1. **Graph Laplacian**: What are the key properties of graph Laplacians and how do they relate to graph structure?

2. **Spectral Theory**: How does the spectrum of the graph Laplacian encode structural information?

3. **Graph Signals**: What is the interpretation of frequency in the context of graph signals?

### Message Passing Framework
4. **Expressiveness**: What determines the expressive power of message passing neural networks?

5. **Aggregation Functions**: How do different aggregation functions affect the theoretical properties of GNNs?

6. **Permutation Equivariance**: Why is permutation equivariance essential for GNNs?

### Spectral vs Spatial Approaches
7. **Localization**: How do spectral and spatial approaches differ in their notion of locality?

8. **Computational Complexity**: What are the computational trade-offs between spectral and spatial methods?

9. **Transferability**: Which approach generalizes better across different graph structures?

### Theoretical Limitations
10. **Weisfeiler-Lehman**: What is the connection between GNN expressiveness and the WL test?

11. **Over-smoothing**: What causes over-smoothing and how can it be mitigated theoretically?

12. **Regular Graphs**: Why do standard GNNs struggle with highly regular graph structures?

### Advanced Extensions
13. **Higher-Order**: How do higher-order GNNs increase expressive power and at what cost?

14. **Continuous Models**: What advantages do Graph Neural ODEs provide over discrete models?

15. **Geometric Principles**: How do geometric deep learning principles unify different GNN architectures?

## Conclusion

Graph Neural Networks represent a fundamental extension of deep learning to non-Euclidean domains, providing principled mathematical frameworks for learning on graph-structured data through spectral graph theory, message passing paradigms, and geometric deep learning principles that enable the modeling of complex relational patterns across diverse applications from molecular chemistry to social network analysis. This comprehensive exploration has established:

**Mathematical Rigor**: Deep understanding of graph theory, spectral analysis, and message passing frameworks provides the theoretical foundation for designing and analyzing graph neural architectures with principled approaches to handling non-Euclidean data structures.

**Theoretical Foundations**: Systematic analysis of expressiveness limitations, universal approximation properties, and connections to graph isomorphism testing reveals both the power and fundamental constraints of graph neural networks.

**Architectural Principles**: Coverage of spectral and spatial approaches demonstrates how different mathematical perspectives lead to complementary architectural designs with distinct computational and theoretical trade-offs.

**Symmetry and Invariance**: Understanding of permutation equivariance and geometric deep learning principles provides the mathematical framework for designing architectures that respect the inherent symmetries of graph-structured data.

**Advanced Extensions**: Analysis of higher-order methods, continuous models, and geometric principles shows how theoretical insights drive the development of more powerful and principled graph neural architectures.

**Fundamental Limitations**: Recognition of over-smoothing, expressiveness bounds, and structural blind spots provides crucial understanding of when and why standard approaches may fail.

Graph Neural Networks and their mathematical foundations are crucial for modern machine learning because:
- **Non-Euclidean Data**: Enable learning on irregular, graph-structured data that appears throughout science and technology
- **Relational Reasoning**: Provide frameworks for modeling complex relationships and dependencies between entities
- **Theoretical Understanding**: Offer principled approaches with mathematical guarantees and analysis tools
- **Universal Applicability**: Apply to diverse domains from chemistry and biology to social networks and knowledge graphs
- **Symmetry Preservation**: Respect fundamental symmetries that are essential for generalization and interpretability

The theoretical principles, mathematical frameworks, and architectural insights covered provide essential knowledge for understanding modern graph-based machine learning, developing effective GNN systems, and contributing to advances in geometric deep learning and relational AI. Understanding these foundations is crucial for working with graph neural networks and developing applications that require learning from complex structured data.