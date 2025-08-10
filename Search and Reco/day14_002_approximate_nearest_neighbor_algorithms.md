# Day 14.2: Approximate Nearest Neighbor Algorithms and Indexing Techniques

## Learning Objectives
By the end of this session, students will be able to:
- Understand the theoretical foundations of approximate nearest neighbor (ANN) algorithms
- Analyze different ANN algorithm families and their trade-offs
- Evaluate locality-sensitive hashing and its variants for similarity search
- Design hierarchical and graph-based indexing structures for vector search
- Understand product quantization and other compression techniques
- Apply ANN algorithms to large-scale search and recommendation scenarios

## 1. Theoretical Foundations of ANN

### 1.1 The Curse of Dimensionality

**High-Dimensional Space Challenges**

**Distance Concentration**
In high-dimensional spaces, distances between points tend to become similar:
- **Concentration Phenomenon**: All pairwise distances converge to similar values
- **Loss of Discrimination**: Difficult to distinguish between near and far neighbors
- **Mathematical Analysis**: For Gaussian distributions, distance ratios converge to 1
- **Practical Implications**: Traditional distance-based methods lose effectiveness

**Volume and Density Issues**
- **Exponential Volume Growth**: Volume grows exponentially with dimension
- **Sparse Data**: Data becomes increasingly sparse in high dimensions
- **Empty Space Problem**: Most of the space is empty, data concentrated on boundaries
- **Nearest Neighbor Paradox**: In very high dimensions, all points are approximately equidistant

**Computational Complexity**
- **Brute Force Scaling**: O(nd) complexity becomes prohibitive
- **Index Structure Breakdown**: Traditional spatial indexes (R-trees, kd-trees) fail
- **Search Space Explosion**: Exponential growth in search space size
- **Memory Requirements**: Storage requirements grow exponentially

### 1.2 ANN Problem Formulation

**Approximate Nearest Neighbor Definition**

**Problem Statement**
Given a set S of n points in d-dimensional space and query point q:
- **Exact NN**: Find point p ∈ S that minimizes distance d(q, p)
- **Approximate NN**: Find point p such that d(q, p) ≤ (1 + ε) × d(q, p*)
- **ε-Approximation**: Allow solutions within (1 + ε) factor of optimal
- **Probabilistic Guarantees**: Succeed with probability at least 1 - δ

**Quality Metrics**
- **Approximation Ratio**: How close to optimal is the returned result?
- **Recall@k**: Fraction of true k-nearest neighbors found
- **Precision**: Accuracy of returned approximate neighbors
- **Success Probability**: Probability of finding good approximation

**Efficiency Metrics**
- **Query Time**: Time to answer single query
- **Preprocessing Time**: Time to build index structure
- **Space Complexity**: Memory required for index
- **Update Complexity**: Cost of adding/removing points

### 1.3 Fundamental Trade-offs

**The ANN Trade-off Space**

**Quality vs Speed**
- **Higher Accuracy**: More computation required for better results
- **Faster Queries**: Must accept lower quality results
- **Parameter Tuning**: Adjust parameters to balance quality and speed
- **Application Dependent**: Different applications have different requirements

**Space vs Time**
- **Larger Indexes**: More memory can enable faster queries
- **Compression**: Reduce memory at cost of accuracy or speed
- **Preprocessing vs Query Time**: More preprocessing can reduce query time
- **Memory Hierarchy**: Consider cache effects and memory access patterns

**Preprocessing vs Query Performance**
- **Index Construction**: More complex preprocessing for better query performance
- **Online vs Offline**: Trade-off between online flexibility and offline optimization
- **Dynamic Updates**: Cost of maintaining index as data changes
- **Batch vs Incremental**: Different strategies for different update patterns

## 2. Locality-Sensitive Hashing (LSH)

### 2.1 LSH Fundamentals

**Core LSH Concepts**

**Hash Function Families**
A family H of hash functions is (r, cr, p₁, p₂)-sensitive if:
- **Near Points**: If d(p, q) ≤ r, then Pr[h(p) = h(q)] ≥ p₁
- **Far Points**: If d(p, q) ≥ cr, then Pr[h(p) = h(q)] ≤ p₂
- **Gap Assumption**: We assume cr > r (there's a gap between near and far)
- **Probability Gap**: p₁ > p₂ (higher collision probability for similar items)

**LSH Amplification**
- **AND Construction**: Use multiple hash functions: g(p) = (h₁(p), h₂(p), ..., hₖ(p))
- **OR Construction**: Use multiple hash tables with different random hash functions
- **Parameter Selection**: Choose k (AND) and L (OR) to optimize performance
- **Theoretical Analysis**: Success probability = 1 - (1 - p₁ᵏ)ᴸ for near neighbors

**Query Processing**
1. **Hash Query Point**: Compute hash values for query point
2. **Retrieve Candidates**: Get all points in same buckets
3. **Filter Candidates**: Apply exact distance computation to candidates
4. **Return Results**: Return closest candidates found

### 2.2 LSH Families for Different Metrics

**Hamming Distance LSH**

**Random Projection**
For binary vectors and Hamming distance:
- **Hash Function**: h(x) = x[i] (select random coordinate)
- **Collision Probability**: Pr[h(x) = h(y)] = 1 - d_H(x,y)/d
- **Simple Implementation**: Very easy to implement and compute
- **Applications**: Text similarity, duplicate detection

**Euclidean Distance LSH**

**Random Projection (Johnson-Lindenstrauss)**
For vectors in Euclidean space:
- **Hash Function**: h(x) = ⌊(a·x + b)/w⌋
- **Random Vector**: a drawn from d-dimensional Gaussian
- **Collision Probability**: Function of distance and bin width w
- **Parameter Tuning**: Choose w to balance collision rates

**Cosine Similarity LSH**

**Sign Random Projection**
For cosine similarity (angle between vectors):
- **Hash Function**: h(x) = sign(a·x)
- **Random Hyperplane**: a drawn from d-dimensional Gaussian
- **Collision Probability**: Pr[h(x) = h(y)] = 1 - θ(x,y)/π
- **Geometric Interpretation**: Hyperplanes divide space into regions

**Jaccard Similarity LSH**

**MinHash**
For set similarity using Jaccard coefficient:
- **Hash Function**: h(S) = min{π(x) : x ∈ S}
- **Random Permutation**: π is random permutation of universe
- **Collision Probability**: Pr[h(A) = h(B)] = |A ∩ B|/|A ∪ B|
- **Implementation**: Use random hash functions instead of permutations

### 2.3 Advanced LSH Techniques

**Multi-Probe LSH**

**Intelligent Probing**
Instead of using many hash tables, probe multiple buckets:
- **Perturbation Sequence**: Generate sequence of hash perturbations
- **Query-Specific**: Adapt probing strategy to specific query
- **Reduced Memory**: Fewer hash tables needed
- **Controlled Recall**: Systematically increase recall by probing more buckets

**LSH Forest**

**Prefix Trees**
Organize LSH into prefix tree structure:
- **Variable-Length Prefixes**: Use different prefix lengths for different densities
- **Adaptive Strategy**: Adapt prefix length based on local density
- **Dynamic Adjustment**: Adjust parameters based on query distribution
- **Memory Efficiency**: More efficient memory usage than standard LSH

**Learning to Hash**

**Data-Dependent Hashing**
Learn hash functions from data:
- **Supervised Learning**: Use labeled similarity data to learn hash functions
- **Unsupervised Learning**: Use data distribution to optimize hash functions
- **Deep Hashing**: Use deep neural networks to learn hash functions
- **End-to-End Training**: Train hash functions and downstream tasks jointly

## 3. Tree-Based Methods

### 3.1 Space-Partitioning Trees

**KD-Trees and Variants**

**Classical KD-Trees**
- **Recursive Partitioning**: Split space along alternating dimensions
- **Median Splitting**: Split at median to balance tree
- **Query Algorithm**: Backtrack when necessary to find all neighbors
- **Curse of Dimensionality**: Performance degrades rapidly with dimension

**Randomized KD-Trees**
- **Random Dimension Selection**: Randomly choose splitting dimension
- **Multiple Trees**: Use forest of randomized trees
- **Improved Performance**: Better performance in high dimensions
- **FLANN Implementation**: Fast Library for Approximate Nearest Neighbors

**Hierarchical K-Means Trees**

**Clustering-Based Partitioning**
- **K-Means Splitting**: Use k-means to partition data at each node
- **Hierarchical Structure**: Recursively apply k-means clustering
- **Multiple Centroids**: Each internal node has k centroids
- **Best-Bin-First Search**: Prioritize promising branches during search

### 3.2 Metric Trees

**Ball Trees**

**Hypersphere Organization**
- **Ball Construction**: Each node represents a ball (center + radius)
- **Nested Structure**: Child balls contained within parent balls
- **Distance-Based**: Works with any metric, not just Euclidean
- **Triangle Inequality**: Use triangle inequality to prune search

**M-Trees**

**Dynamic Metric Trees**
- **Balanced Structure**: Maintain balanced tree structure
- **Distance Computation**: Store distances to reduce computation
- **Split Strategies**: Various strategies for node splitting
- **General Metrics**: Support arbitrary distance functions

**VP-Trees (Vantage Point Trees)**

**Distance-Based Partitioning**
- **Vantage Point Selection**: Choose vantage point for each node
- **Distance Threshold**: Partition by distance to vantage point
- **Recursive Structure**: Recursively apply to subsets
- **Metric Space**: Works in any metric space

### 3.3 Hybrid Approaches

**Multi-Index Methods**

**Composite Indexing**
- **Multiple Indexes**: Use different indexing methods simultaneously
- **Result Fusion**: Combine results from multiple indexes
- **Complementary Strengths**: Different methods work well for different queries
- **Adaptive Selection**: Choose best index for each query

**Hierarchical Navigable Small World (HNSW)**

**Graph-Tree Hybrid**
- **Layered Structure**: Multiple layers with different connection densities
- **Navigable Small World**: Each layer forms navigable small world graph
- **Greedy Search**: Greedy search from top layer to bottom
- **Performance**: Excellent empirical performance across many datasets

## 4. Graph-Based Methods

### 4.1 Navigable Small World Networks

**Small World Properties**

**Network Characteristics**
- **Short Path Lengths**: Logarithmic diameter in network size
- **High Clustering**: Nodes form tight local clusters
- **Navigability**: Greedy routing can find short paths
- **Scale-Free**: Often exhibit scale-free degree distributions

**NSW Construction**
- **Incremental Construction**: Add nodes one by one to graph
- **Bidirectional Links**: Maintain bidirectional connections
- **Connection Strategy**: Connect to closest nodes in current graph
- **Dynamic Properties**: Graph properties emerge during construction

**Query Processing**
- **Greedy Search**: Start from random node, greedily move to closer neighbors
- **Multiple Restarts**: Use multiple random starting points
- **Beam Search**: Maintain beam of best candidates during search
- **Local Optima**: Handle local optima through restarts or beam search

### 4.2 Hierarchical NSW (HNSW)

**Multi-Layer Architecture**

**Layer Construction**
- **Exponential Decay**: Probability of node appearing in layer l: e^(-l/ln(2))
- **Layer 0**: All nodes present in bottom layer
- **Higher Layers**: Fewer nodes, sparser connections
- **Skip List Inspiration**: Similar to probabilistic skip lists

**Search Algorithm**
1. **Entry Point**: Start from top layer entry point
2. **Layer Search**: Greedily search within each layer
3. **Layer Descent**: Move to next layer when no improvement
4. **Final Layer**: Perform detailed search in bottom layer

**Construction Algorithm**
1. **Layer Assignment**: Randomly assign new node to layers
2. **Connection Phase**: Connect to M closest neighbors in each layer
3. **Pruning Phase**: Prune connections to maintain degree bounds
4. **Update Phase**: Update connections of existing neighbors

**Parameter Optimization**
- **M Parameter**: Number of connections per node
- **efConstruction**: Size of dynamic candidate list during construction
- **efSearch**: Size of dynamic candidate list during search
- **ml Parameter**: Level multiplier for layer assignment

### 4.3 Other Graph-Based Methods

**K-Nearest Neighbor Graphs**

**Exact KNN Graphs**
- **Construction**: Build graph where each node connects to k nearest neighbors
- **Search Process**: Navigate graph by following edges to closer nodes
- **Local Search**: Use local search techniques to find approximate neighbors
- **Graph Quality**: Quality depends on accuracy of initial KNN graph

**Approximate KNN Graphs**
- **NN-Descent**: Iterative algorithm to approximate KNN graph
- **Random Initialization**: Start with random graph structure
- **Local Updates**: Iteratively improve graph quality
- **Convergence**: Algorithm converges to high-quality approximate KNN graph

**Proximity Graphs**

**Relative Neighborhood Graph (RNG)**
- **Definition**: Include edge (u,v) if no point w is closer to both u and v
- **Geometric Properties**: Subset of Delaunay triangulation
- **Sparse Structure**: Typically has O(n) edges
- **Navigation**: Can navigate using geometric properties

**Gabriel Graph**
- **Definition**: Include edge (u,v) if circle with uv as diameter contains no other points
- **Relationship**: Subset of Delaunay triangulation, superset of MST
- **Construction**: Can be constructed efficiently
- **Applications**: Used in geographic routing and mesh generation

## 5. Quantization-Based Methods

### 5.1 Vector Quantization

**Product Quantization (PQ)**

**Subspace Decomposition**
- **Dimension Splitting**: Split d-dimensional vectors into m subvectors
- **Independent Quantization**: Quantize each subspace independently
- **Codebook Learning**: Learn k centroids for each subspace
- **Compact Representation**: Represent vectors using m × log₂(k) bits

**Distance Computation**
- **Lookup Tables**: Precompute distances between query subvectors and centroids
- **Additive Property**: d²(x,y) ≈ Σᵢ d²(x̃ᵢ, ỹᵢ) for PQ-encoded vectors
- **Fast Computation**: Distance computation reduces to table lookups
- **Memory Efficiency**: Significant memory reduction with small accuracy loss

**Optimized Product Quantization (OPQ)**
- **Rotation Learning**: Learn optimal rotation before quantization
- **Balanced Subspaces**: Ensure subspaces have similar importance
- **Joint Optimization**: Jointly optimize rotation and quantization
- **Improved Accuracy**: Better approximation quality than standard PQ

### 5.2 Scalar Quantization and Compression

**Scalar Quantization**

**Uniform Quantization**
- **Linear Quantization**: Map continuous values to discrete levels
- **Quantization Step**: Uniform spacing between quantization levels
- **Simple Implementation**: Easy to implement and compute
- **Reconstruction Error**: Uniform distribution of quantization error

**Non-Uniform Quantization**
- **Optimal Quantization**: Choose quantization levels to minimize distortion
- **Lloyd-Max Algorithm**: Iterative algorithm for optimal quantization
- **Data-Dependent**: Adapt quantization to data distribution
- **Improved Performance**: Better rate-distortion trade-off

**Binary Embeddings**

**Binary Hashing**
- **Sign Quantization**: Quantize each dimension to {-1, +1}
- **Extreme Compression**: Single bit per dimension
- **Hamming Distance**: Use Hamming distance for similarity
- **Hardware Efficiency**: Very fast computation using XOR operations

**Learning Binary Codes**
- **Supervised Hashing**: Learn binary codes from similarity labels
- **Deep Binary Networks**: Use deep networks to learn binary representations
- **Balanced Codes**: Ensure balanced distribution of 0s and 1s
- **Preservation**: Preserve similarity structure in binary space

### 5.3 Hybrid Quantization Approaches

**Composite Quantization**

**Multiple Codebooks**
- **Additive Quantization**: x ≈ c₁ + c₂ + ... + cₘ
- **Dictionary Learning**: Learn multiple dictionaries jointly
- **Sparse Coding**: Use sparse combinations of codewords
- **Flexible Approximation**: More flexible than product quantization

**Residual Quantization**

**Hierarchical Refinement**
- **Coarse Quantization**: First level of quantization
- **Residual Encoding**: Encode quantization residuals
- **Multiple Levels**: Apply recursively for finer approximation
- **Adaptive Precision**: Use different precision levels for different regions

## 6. Performance Analysis and Optimization

### 6.1 Theoretical Analysis

**Complexity Analysis**

**Time Complexity**
- **Preprocessing**: O(n log n) to O(n²) depending on method
- **Query Time**: O(log n) to O(√n) for different approaches
- **Space Complexity**: O(n) to O(n²) for index storage
- **Update Complexity**: Cost of maintaining index with insertions/deletions

**Approximation Quality**
- **Approximation Ratio**: Expected ratio of returned distance to optimal
- **Concentration Bounds**: Probability bounds on approximation quality
- **Recall Analysis**: Expected fraction of true neighbors found
- **Failure Probability**: Probability of not finding good approximation

**Parameter Selection**

**LSH Parameters**
- **Hash Functions**: Number of hash functions per table (k)
- **Hash Tables**: Number of hash tables (L)
- **Optimization**: Choose k and L to minimize total query cost
- **Theoretical Guidelines**: Use approximation theory for parameter selection

**Graph Parameters**
- **Connectivity**: Degree of connectivity in graph construction
- **Search Parameters**: Beam width, number of entry points
- **Construction vs Query**: Balance construction cost with query performance
- **Empirical Tuning**: Often requires empirical tuning for specific datasets

### 6.2 Implementation Considerations

**Memory Management**

**Cache Efficiency**
- **Data Layout**: Organize data for cache-friendly access patterns
- **Prefetching**: Use prefetching to reduce memory latency
- **Memory Hierarchy**: Consider different levels of memory hierarchy
- **SIMD Instructions**: Use SIMD for parallel distance computations

**Parallelization**

**Query Parallelization**
- **Independent Queries**: Process multiple queries in parallel
- **Shared Index**: Multiple threads share same index structure
- **Load Balancing**: Balance query load across threads
- **Lock-Free Structures**: Use lock-free data structures when possible

**Construction Parallelization**
- **Parallel Construction**: Build index using multiple threads
- **Data Partitioning**: Partition data across threads
- **Synchronization**: Coordinate updates to shared structures
- **Scalability**: Ensure algorithm scales with number of threads

### 6.3 Evaluation Methodologies

**Benchmark Datasets**

**Standard Benchmarks**
- **SIFT**: 128-dimensional SIFT descriptors
- **GIST**: 960-dimensional scene descriptors
- **GloVe**: Word embeddings of various dimensions
- **Deep Learning Features**: Features from pre-trained neural networks

**Evaluation Metrics**

**Quality Metrics**
- **Recall@k**: Fraction of true k-NN found in returned results
- **Precision**: Fraction of returned results that are true neighbors
- **Mean Average Precision (MAP)**: Average precision across queries
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranked retrieval quality

**Efficiency Metrics**
- **Queries Per Second (QPS)**: Throughput measure
- **Average Query Time**: Mean time per query
- **Index Build Time**: Time to construct index
- **Memory Usage**: RAM required for index

**Trade-off Analysis**
- **Pareto Curves**: Plot quality vs efficiency trade-offs  
- **Parameter Sweeps**: Analyze performance across parameter ranges
- **Scalability Studies**: Performance as dataset size increases
- **Dimensionality Effects**: Performance across different dimensions

## 7. Study Questions

### Beginner Level
1. What is the curse of dimensionality and how does it affect nearest neighbor search?
2. How does locality-sensitive hashing work and what are its main advantages?
3. What are the key differences between tree-based and graph-based ANN methods?
4. How does product quantization achieve compression while preserving similarity?
5. What are the main trade-offs in approximate nearest neighbor algorithms?

### Intermediate Level
1. Compare LSH, HNSW, and product quantization approaches in terms of accuracy, speed, and memory usage for different types of data.
2. Design an ANN system for a specific application (e.g., image search, recommendation system) and justify your choice of algorithm and parameters.
3. How would you handle dynamic datasets where vectors are frequently added or removed?
4. Analyze the theoretical guarantees provided by different ANN algorithms and their practical implications.
5. Design an evaluation framework for comparing ANN algorithms across different datasets and query types.

### Advanced Level
1. Develop a theoretical framework for understanding the fundamental limits of approximate nearest neighbor search in high-dimensional spaces.
2. Design a novel hybrid ANN algorithm that combines multiple approaches to achieve better performance than existing methods.
3. Create a comprehensive analysis of the memory hierarchy effects in ANN algorithms and propose optimizations.
4. Develop adaptive ANN algorithms that automatically adjust their parameters based on query characteristics and data distribution.
5. Design a distributed ANN system that can scale to billions of vectors while maintaining sub-second query latency.

## 8. Implementation Guidelines and Best Practices

### 8.1 Algorithm Selection Guidelines

**Dataset Characteristics**
- **Size**: Small datasets may benefit from simple methods, large datasets need scalable approaches
- **Dimensionality**: High-dimensional data requires specialized techniques
- **Data Distribution**: Clustered vs uniform data affects algorithm choice
- **Update Frequency**: Static vs dynamic datasets have different requirements

**Performance Requirements**
- **Latency vs Throughput**: Different algorithms optimize for different metrics
- **Memory Constraints**: Available memory affects choice of compression techniques
- **Accuracy Requirements**: Some applications can tolerate lower accuracy for speed
- **Scalability Needs**: Consider future growth in data size and query load

### 8.2 Production Deployment Considerations

**System Integration**
- **API Design**: Design clean APIs for index construction and querying
- **Monitoring**: Monitor query performance, accuracy, and resource usage
- **A/B Testing**: Test algorithm changes with real user traffic
- **Fallback Strategies**: Implement fallbacks for algorithm failures

**Optimization Techniques**
- **Parameter Tuning**: Systematic approaches to parameter optimization
- **Model Selection**: Techniques for choosing between different algorithms
- **Hardware Acceleration**: Use GPUs or specialized hardware when available
- **Distributed Systems**: Scale across multiple machines for very large datasets

This comprehensive coverage of approximate nearest neighbor algorithms provides the theoretical foundation and practical guidance needed to implement efficient similarity search systems for modern applications requiring fast, accurate retrieval from large-scale vector databases.