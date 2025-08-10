# Day 14.1: Vector Space Models for Language - Distributional Semantics and Mathematical Foundations

## Overview

Vector space models for language represent a fundamental paradigm in computational linguistics and natural language processing that transforms discrete linguistic symbols into continuous mathematical representations, enabling computational systems to capture, manipulate, and reason about semantic relationships between words, phrases, and larger linguistic structures. These models are grounded in the distributional hypothesis, which posits that linguistic items with similar distributions across contexts tend to have similar meanings, providing a mathematical foundation for automatically learning semantic representations from large text corpora. The transition from symbolic to distributed representations has revolutionized natural language processing by enabling neural networks to process linguistic information effectively, leading to breakthrough applications in machine translation, sentiment analysis, question answering, and numerous other language understanding tasks that require sophisticated semantic reasoning capabilities.

## Mathematical Foundations of Distributional Semantics

### The Distributional Hypothesis

**Harris Distributional Hypothesis (1954)**
"Linguistic elements that occur in similar contexts tend to have similar meanings"

**Mathematical Formulation**
For words $w_i$ and $w_j$, semantic similarity is proportional to distributional similarity:
$$\text{semantic\_sim}(w_i, w_j) \propto \text{distributional\_sim}(\mathbf{c}_i, \mathbf{c}_j)$$

where $\mathbf{c}_i$ and $\mathbf{c}_j$ are context vectors for words $w_i$ and $w_j$.

**Context Definition**
Context can be defined in multiple ways:
- **Window-based**: Words within fixed window size $k$
- **Syntactic**: Syntactically related words (subject, object, modifier)
- **Document-based**: Words co-occurring in same document
- **Sentence-based**: Words co-occurring in same sentence

**Firth's Contextual Principle (1957)**
"You shall know a word by the company it keeps"

This principle provides the theoretical foundation for all distributional semantic models.

### Vector Space Model Theory

**Vector Space Construction**
Given vocabulary $V = \{w_1, w_2, ..., w_n\}$ and context features $C = \{c_1, c_2, ..., c_m\}$:

**Co-occurrence Matrix**: $\mathbf{M} \in \mathbb{R}^{|V| \times |C|}$
$$\mathbf{M}_{i,j} = \text{count}(w_i, c_j)$$

**Word Representation**: Each word $w_i$ is represented as vector $\mathbf{v}_i \in \mathbb{R}^{|C|}$

**Geometric Properties**
- **Euclidean Distance**: $d(\mathbf{v}_i, \mathbf{v}_j) = \|\mathbf{v}_i - \mathbf{v}_j\|_2$
- **Cosine Similarity**: $\cos(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$
- **Manhattan Distance**: $d(\mathbf{v}_i, \mathbf{v}_j) = \|\mathbf{v}_i - \mathbf{v}_j\|_1$

**Dimensionality Considerations**
High-dimensional spaces exhibit counterintuitive properties:
- **Curse of Dimensionality**: Distance becomes less meaningful in high dimensions
- **Sparsity**: Most entries in co-occurrence matrix are zero
- **Hub Problem**: Some dimensions dominate similarity calculations

### Information-Theoretic Foundations

**Mutual Information**
Measure of association between word and context:
$$I(w; c) = \log \frac{P(w, c)}{P(w)P(c)}$$

**Pointwise Mutual Information (PMI)**
$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)P(c)} = \log \frac{\text{count}(w, c) \cdot N}{\text{count}(w) \cdot \text{count}(c)}$$

where $N$ is total number of word-context pairs.

**Positive PMI (PPMI)**
Address negative values in PMI:
$$\text{PPMI}(w, c) = \max(0, \text{PMI}(w, c))$$

**Properties of PMI**:
- Measures strength of association
- Handles frequency biases better than raw counts
- Provides theoretical justification for weighting schemes

**Shifted PMI**
$$\text{SPMI}(w, c) = \max(0, \text{PMI}(w, c) - \log k)$$

where $k$ is smoothing parameter that shifts the distribution.

## Traditional Vector Space Models

### Term-Document Matrix Model

**Boolean Model**
$$\mathbf{M}_{i,j} = \begin{cases}
1 & \text{if term } i \text{ occurs in document } j \\
0 & \text{otherwise}
\end{cases}$$

**Term Frequency (TF)**
$$\text{tf}_{i,j} = \frac{\text{count}(t_i, d_j)}{\sum_k \text{count}(t_k, d_j)}$$

**Inverse Document Frequency (IDF)**
$$\text{idf}_i = \log \frac{|D|}{|\{d : t_i \in d\}|}$$

**TF-IDF Weighting**
$$\text{tf-idf}_{i,j} = \text{tf}_{i,j} \times \text{idf}_i$$

**Rationale**: 
- High weight for frequent terms in specific documents (high TF)
- Low weight for terms occurring in many documents (low IDF)

### Latent Semantic Analysis (LSA)

**Singular Value Decomposition**
Decompose term-document matrix $\mathbf{M}$:
$$\mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{|V| \times k}$: Left singular vectors (term concepts)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{k \times k}$: Singular values (concept strengths)
- $\mathbf{V} \in \mathbb{R}^{|D| \times k}$: Right singular vectors (document concepts)

**Dimensionality Reduction**
Retain top $k$ singular values:
$$\mathbf{M}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

**Low-Rank Approximation Benefits**:
- Noise reduction
- Capture latent semantic relationships
- Handle synonymy and polysemy
- Computational efficiency

**Semantic Similarity**
$$\text{sim}(w_i, w_j) = \cos(\mathbf{u}_i, \mathbf{u}_j)$$

where $\mathbf{u}_i$ and $\mathbf{u}_j$ are rows of $\mathbf{U}_k$.

**Latent Semantic Space Properties**
- **Synonymy Resolution**: Similar words have similar representations
- **Polysemy Handling**: Multiple senses captured in single vector
- **Transitivity**: $\text{sim}(A, C)$ can be high even if $A$ and $C$ don't co-occur

### Hyperspace Analogue to Language (HAL)

**Construction Method**
1. Move sliding window of size $L$ through corpus
2. For each word pair $(w_i, w_j)$ within window:
   $$\mathbf{M}_{i,j} += \frac{L - d + 1}{L}$$
   where $d$ is distance between words

**Weighted Co-occurrence**
Closer words receive higher weights, capturing positional information.

**Normalization**
Row normalize to obtain probability distributions:
$$\mathbf{P}_{i,j} = \frac{\mathbf{M}_{i,j}}{\sum_k \mathbf{M}_{i,k}}$$

**Symmetric HAL**
$$\mathbf{M}_{\text{sym}} = \mathbf{M} + \mathbf{M}^T$$

Combines both directions of co-occurrence information.

### Random Indexing

**Motivation**
Address computational and memory limitations of traditional methods.

**Random Vectors**
Assign each context element a sparse, high-dimensional random vector:
$$\mathbf{r}_j \in \{-1, 0, 1\}^d$$

where most elements are 0, few are ±1.

**Incremental Construction**
For each occurrence of word $w_i$ in context $c_j$:
$$\mathbf{v}_i += \mathbf{r}_j$$

**Johnson-Lindenstrauss Lemma**
Random projection preserves distances with high probability:
$$\mathbb{P}[(1-\epsilon)\|\mathbf{u} - \mathbf{v}\|^2 \leq \|\mathbf{f}(\mathbf{u}) - \mathbf{f}(\mathbf{v})\|^2 \leq (1+\epsilon)\|\mathbf{u} - \mathbf{v}\|^2] \geq 1 - \delta$$

**Advantages**:
- Scalable to large corpora
- Online learning capability
- Memory efficient
- Theoretically grounded

## Context Representation Methods

### Window-Based Contexts

**Fixed Window Size**
Context of word $w_t$ includes words within window $[-k, +k]$:
$$\text{Context}(w_t) = \{w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}\}$$

**Distance Weighting**
Assign higher weights to closer words:
$$\text{weight}(w_{t+i}) = \frac{1}{|i|} \text{ or } \frac{k - |i| + 1}{k}$$

**Asymmetric Windows**
Different left and right context sizes:
$$\text{Context}(w_t) = \{w_{t-l}, ..., w_{t-1}, w_{t+1}, ..., w_{t+r}\}$$

**Dynamic Windows**
Vary window size based on sentence boundaries or punctuation.

### Syntactic Contexts

**Dependency Relations**
Use syntactic dependencies as contexts:
- $(w, \text{nsubj}, \text{verb})$: $w$ is subject of verb
- $(w, \text{dobj}, \text{verb})$: $w$ is object of verb
- $(w, \text{amod}, \text{noun})$: $w$ modifies noun

**Grammatical Relations Matrix**
$$\mathbf{M}_{w,r,w'} = \text{count}(w \xrightarrow{r} w')$$

**Advantages**:
- Linguistically motivated
- Captures precise semantic relationships
- Less sensitive to word order variations

**Challenges**:
- Requires syntactic parsing
- Parsing errors propagate
- Sparsity issues

### Document-Level Contexts

**Document Co-occurrence**
Words co-occurring in same document:
$$\mathbf{M}_{i,j} = \text{number of documents containing both } w_i \text{ and } w_j$$

**Topic-Based Contexts**
Use topic models (LDA) to define contexts:
$$\mathbf{M}_{w,t} = P(w | \text{topic}_t)$$

**Paragraph Vectors**
Learn distributed representations of documents alongside words.

## Matrix Factorization Methods

### Non-negative Matrix Factorization (NMF)

**Problem Formulation**
Factor non-negative matrix $\mathbf{M}$ into non-negative factors:
$$\mathbf{M} \approx \mathbf{W} \mathbf{H}$$

where $\mathbf{W} \in \mathbb{R}_+^{m \times k}$, $\mathbf{H} \in \mathbb{R}_+^{k \times n}$

**Optimization Objective**
$$\min_{\mathbf{W}, \mathbf{H}} \|\mathbf{M} - \mathbf{W}\mathbf{H}\|_F^2 \quad \text{s.t. } \mathbf{W}, \mathbf{H} \geq 0$$

**Multiplicative Updates**
$$\mathbf{W}_{ik} \leftarrow \mathbf{W}_{ik} \frac{(\mathbf{M}\mathbf{H}^T)_{ik}}{(\mathbf{W}\mathbf{H}\mathbf{H}^T)_{ik}}$$
$$\mathbf{H}_{kj} \leftarrow \mathbf{H}_{kj} \frac{(\mathbf{W}^T\mathbf{M})_{kj}}{(\mathbf{W}^T\mathbf{W}\mathbf{H})_{kj}}$$

**Sparse NMF**
Add sparsity constraints:
$$\min_{\mathbf{W}, \mathbf{H}} \|\mathbf{M} - \mathbf{W}\mathbf{H}\|_F^2 + \lambda(\|\mathbf{W}\|_1 + \|\mathbf{H}\|_1)$$

### Probabilistic Matrix Factorization

**Probabilistic LSA (PLSA)**
$$P(w, d) = \sum_{z} P(w | z) P(z | d) P(d)$$

where $z$ represents latent topics.

**Expectation-Maximization Algorithm**
E-step:
$$P(z | w, d) = \frac{P(w | z) P(z | d)}{\sum_{z'} P(w | z') P(z' | d)}$$

M-step:
$$P(w | z) = \frac{\sum_d n(w, d) P(z | w, d)}{\sum_{w'} \sum_d n(w', d) P(z | w', d)}$$

**Latent Dirichlet Allocation (LDA)**
Full Bayesian treatment of PLSA:
- Document-topic distributions: $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$
- Topic-word distributions: $\boldsymbol{\phi}_k \sim \text{Dir}(\boldsymbol{\beta})$

**Gibbs Sampling for LDA**
$$P(z_i = k | \mathbf{z}_{-i}, \mathbf{w}) \propto \frac{n_{k,w_i}^{(-i)} + \beta}{n_k^{(-i)} + W\beta} \cdot \frac{n_{d_i,k}^{(-i)} + \alpha}{n_{d_i}^{(-i)} + K\alpha}$$

## Evaluation Methods for Vector Spaces

### Intrinsic Evaluation

**Word Similarity Tasks**
Compare model similarities with human judgments:
$$\text{correlation} = \text{corr}(\text{human\_sim}, \text{model\_sim})$$

**Standard Datasets**:
- **WordSim-353**: 353 word pairs with similarity ratings
- **SimLex-999**: Focus on similarity vs relatedness
- **MEN**: Large-scale dataset with 3000 word pairs

**Word Analogy Tasks**
Test semantic relationships: $a : b :: c : ?$
$$\mathbf{d} = \arg\max_{w \in V} \cos(\mathbf{w}, \mathbf{b} - \mathbf{a} + \mathbf{c})$$

**Categories**:
- **Semantic**: country-capital, family relationships
- **Syntactic**: past tense, plurals, comparatives

**Concept Categorization**
Cluster words into semantic categories:
- **Purity**: $\text{Purity} = \frac{1}{N} \sum_i \max_j |C_i \cap L_j|$
- **V-measure**: Harmonic mean of homogeneity and completeness

### Extrinsic Evaluation

**Downstream Task Performance**
Evaluate embeddings on specific applications:
- **Text Classification**: Document categorization
- **Named Entity Recognition**: Entity identification
- **Sentiment Analysis**: Opinion mining
- **Machine Translation**: Translation quality

**Embedding as Features**
Use vectors as input features to supervised models:
$$\mathbf{f}(\text{sentence}) = \text{aggregate}(\{\mathbf{v}_{w_i} : w_i \in \text{sentence}\})$$

**Aggregation Methods**:
- **Mean**: $\frac{1}{|S|} \sum_{w \in S} \mathbf{v}_w$
- **Weighted Mean**: $\sum_{w \in S} \text{weight}(w) \mathbf{v}_w$
- **Max Pooling**: $\max_{w \in S} \mathbf{v}_w$

### Statistical Significance Testing

**Bootstrap Confidence Intervals**
Resample evaluation data to estimate confidence intervals:
1. Sample $n$ pairs with replacement
2. Compute correlation
3. Repeat $B$ times
4. Calculate percentile-based confidence interval

**Permutation Tests**
Test null hypothesis of no correlation:
1. Randomly permute one set of scores
2. Compute correlation with permuted scores
3. Repeat many times to build null distribution
4. Compare observed correlation to null distribution

## Limitations and Challenges

### Sparsity Problems

**Zipf's Law**
Word frequency follows power law:
$$f(r) = \frac{C}{r^\alpha}$$

where $f(r)$ is frequency of $r$-th most frequent word.

**Consequences**:
- Most word pairs never co-occur
- Rare words have unreliable statistics
- Zeros dominate co-occurrence matrices

**Mitigation Strategies**:
- **Smoothing**: Add small constants to counts
- **Dimensionality Reduction**: Reduce noise through SVD
- **Subsampling**: Downsample frequent words

### Polysemy and Word Sense Disambiguation

**Single Vector per Word**
Traditional models assign one vector per word type, averaging over all senses.

**Problems**:
- "Bank" (financial institution) vs "bank" (river side)
- Context-dependent meanings lost
- Semantic drift in embeddings

**Multi-Prototype Models**
Learn multiple vectors per word:
$$\mathbf{v}_{w,s} = \text{vector for word } w \text{ in sense } s$$

**Sense Clustering**
Cluster contexts to identify different senses:
1. Collect all contexts for word $w$
2. Cluster contexts using k-means
3. Learn separate embedding for each cluster

### Frequency Biases

**High-Frequency Word Dominance**
Common words (the, and, of) dominate similarity calculations.

**Solutions**:
- **Subsampling**: Randomly discard frequent words
$$P(\text{discard } w) = 1 - \sqrt{\frac{t}{f(w)}}$$

- **Weighting Schemes**: Downweight frequent words
$$\text{weight}(w) = \frac{1}{f(w)^\alpha}$$

### Geometric Properties

**Hubness Problem**
Some points become nearest neighbors to many others.

**Triangle Inequality Violations**
High-dimensional spaces often violate triangle inequality.

**Concentration of Measure**
In high dimensions, distances become similar:
$$\lim_{d \to \infty} \frac{\text{Var}[\|\mathbf{x} - \mathbf{y}\|]}{\mathbb{E}[\|\mathbf{x} - \mathbf{y}\|]^2} = 0$$

## Key Questions for Review

### Theoretical Foundations
1. **Distributional Hypothesis**: How does the distributional hypothesis provide theoretical justification for vector space models?

2. **Context Definition**: What are the trade-offs between different context definitions (window-based, syntactic, document-based)?

3. **Information Theory**: How do PMI and related measures address frequency biases in co-occurrence statistics?

### Model Architectures
4. **Matrix Factorization**: When should SVD, NMF, or probabilistic methods be preferred for dimensionality reduction?

5. **Random Indexing**: What are the theoretical guarantees and practical advantages of random indexing approaches?

6. **LSA vs PLSA**: How do deterministic and probabilistic matrix factorization methods differ in their assumptions and outputs?

### Evaluation Methods
7. **Intrinsic vs Extrinsic**: What are the relative merits of intrinsic word similarity tasks versus downstream task evaluation?

8. **Analogy Tasks**: What linguistic phenomena do word analogy tasks actually capture, and what are their limitations?

9. **Statistical Significance**: How should statistical significance be properly assessed in embedding evaluation?

### Practical Challenges
10. **Sparsity Handling**: What methods are most effective for addressing sparsity in co-occurrence matrices?

11. **Polysemy**: How can vector space models be extended to handle multiple word senses effectively?

12. **Scalability**: What are the computational and memory trade-offs in different vector space model approaches?

## Conclusion

Vector space models for language provide the mathematical foundation for representing linguistic meaning in computational systems, enabling machines to capture, manipulate, and reason about semantic relationships through geometric operations in high-dimensional spaces. This comprehensive exploration has established:

**Theoretical Framework**: Deep understanding of the distributional hypothesis and information-theoretic foundations demonstrates how mathematical principles can be used to automatically learn semantic representations from text corpora, providing the conceptual basis for modern neural language models.

**Mathematical Foundations**: Systematic coverage of co-occurrence matrices, matrix factorization methods, and dimensionality reduction techniques reveals the mathematical structures underlying distributed semantic representations and their geometric properties in vector spaces.

**Classical Methods**: Comprehensive analysis of traditional approaches including LSA, HAL, and random indexing shows how early vector space models established key principles and techniques that continue to influence modern embedding methods and neural language models.

**Context Representation**: Understanding of different context definition strategies demonstrates how linguistic theory informs the design of computational models and affects the types of semantic relationships captured in vector representations.

**Evaluation Methodologies**: Integration of intrinsic and extrinsic evaluation approaches provides frameworks for assessing the quality of semantic representations and understanding their strengths and limitations across different applications.

**Challenges and Limitations**: Analysis of sparsity, polysemy, frequency biases, and geometric properties reveals the fundamental challenges in representing meaning through vectors and motivates advanced techniques for addressing these limitations.

Vector space models for language are crucial for NLP because:
- **Foundation for Modern NLP**: Establish the theoretical and practical foundations for neural language models and embeddings
- **Semantic Computation**: Enable computational systems to perform semantic operations through geometric manipulations
- **Scalable Learning**: Provide methods for automatically learning semantic representations from large text corpora
- **Cross-Linguistic Applications**: Support multilingual and cross-lingual natural language processing applications
- **Interpretable Representations**: Offer geometric interpretations of semantic relationships that can be analyzed and understood

The theoretical frameworks and mathematical techniques covered provide essential knowledge for understanding how meaning can be represented and computed in vector spaces. Understanding these principles is fundamental for developing modern neural language models, embeddings, and other distributed semantic representations that power contemporary natural language processing systems and applications.