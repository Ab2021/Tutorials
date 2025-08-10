# Day 14.2: Word2Vec and Skip-gram Models - Neural Word Embeddings and Efficient Training

## Overview

Word2Vec represents a revolutionary approach to learning distributed word representations that fundamentally transformed natural language processing by introducing efficient neural methods for capturing semantic and syntactic relationships between words through dense vector embeddings. Developed by Mikolov et al. at Google, Word2Vec comprises two primary architectures - Skip-gram and Continuous Bag-of-Words (CBOW) - that learn word representations by predicting words from their contexts or contexts from words, respectively. The Skip-gram model, in particular, has proven exceptionally effective at learning high-quality word embeddings that capture remarkable semantic properties, including the famous linear relationships that enable analogical reasoning through vector arithmetic. This comprehensive exploration examines the mathematical foundations of Word2Vec architectures, the optimization techniques that make training efficient on large corpora, advanced methods for handling rare words and negative sampling, and the theoretical principles underlying the emergence of semantic structure in learned embeddings.

## Mathematical Foundations of Word2Vec

### Skip-gram Architecture

**Core Idea**
Given a target word, predict surrounding context words within a fixed window size.

**Objective Function**
For a sequence of words $w_1, w_2, ..., w_T$, maximize the average log probability:
$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

where $c$ is the context window size.

**Conditional Probability**
Using softmax to define conditional probability:
$$p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{u}_{w_I})}{\sum_{w=1}^{W} \exp(\mathbf{v}_w^T \mathbf{u}_{w_I})}$$

where:
- $\mathbf{u}_{w_I}$ is input vector representation of word $w_I$
- $\mathbf{v}_{w_O}$ is output vector representation of word $w_O$
- $W$ is vocabulary size

**Two Embedding Matrices**
- **Input embeddings**: $\mathbf{U} \in \mathbb{R}^{d \times |V|}$ where $d$ is embedding dimension
- **Output embeddings**: $\mathbf{V} \in \mathbb{R}^{d \times |V|}$

**Forward Pass**
1. Look up input embedding: $\mathbf{u}_{w_I} = \mathbf{U}[:, w_I]$
2. Compute scores: $\mathbf{s} = \mathbf{V}^T \mathbf{u}_{w_I}$
3. Apply softmax: $\mathbf{p} = \text{softmax}(\mathbf{s})$
4. Extract probability: $p(w_O | w_I) = \mathbf{p}[w_O]$

### CBOW (Continuous Bag-of-Words) Architecture

**Core Idea**
Predict center word from surrounding context words.

**Context Representation**
Average context word vectors:
$$\mathbf{h} = \frac{1}{2c} \sum_{j=-c, j \neq 0}^{c} \mathbf{u}_{w_{t+j}}$$

**Prediction**
$$p(w_t | \text{Context}(w_t)) = \frac{\exp(\mathbf{v}_{w_t}^T \mathbf{h})}{\sum_{w=1}^{W} \exp(\mathbf{v}_w^T \mathbf{h})}$$

**Skip-gram vs CBOW Comparison**
| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Prediction | Context from word | Word from context |
| Training data | More (multiple contexts per word) | Less (one prediction per context) |
| Rare words | Better representation | Worse representation |
| Frequent words | Less emphasis | More emphasis |
| Training speed | Slower | Faster |

## Optimization Challenges and Solutions

### Computational Complexity Problem

**Softmax Bottleneck**
Computing softmax requires summing over entire vocabulary:
$$p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{u}_{w_I})}{\sum_{w=1}^{W} \exp(\mathbf{v}_w^T \mathbf{u}_{w_I})}$$

**Complexity**: $O(|V| \times d)$ per training example
**Problem**: Prohibitive for large vocabularies (millions of words)

### Hierarchical Softmax

**Binary Tree Construction**
Construct binary tree where leaves are vocabulary words:
- Each word $w$ has unique path from root: $n_1, n_2, ..., n_{L(w)}$
- Each internal node has learnable vector: $\mathbf{v}'_{n}$

**Probability Computation**
$$p(w | w_I) = \prod_{j=1}^{L(w)-1} \sigma([[n_{j+1} = \text{ch}(n_j)]] \cdot \mathbf{v}_{n_j}'^T \mathbf{u}_{w_I})$$

where:
- $[[\cdot]]$ is Iverson bracket (1 if true, 0 if false)  
- $\text{ch}(n_j)$ is left child of node $n_j$
- $\sigma(x) = \frac{1}{1 + \exp(-x)}$ is sigmoid function

**Huffman Tree Optimization**
Assign shorter codes to frequent words:
- Frequent words: shorter paths, fewer computations
- Rare words: longer paths, more computations
- Expected complexity: $O(\log |V|)$ instead of $O(|V|)$

**Training Procedure**
For each training pair $(w_I, w_O)$:
1. Find path from root to leaf $w_O$
2. For each node $n_j$ on path:
   - Compute $f = \sigma(\mathbf{v}_{n_j}'^T \mathbf{u}_{w_I})$
   - Update: $\mathbf{v}_{n_j}' \leftarrow \mathbf{v}_{n_j}' + \eta (t_j - f) \mathbf{u}_{w_I}$
   - Update: $\mathbf{u}_{w_I} \leftarrow \mathbf{u}_{w_I} + \eta (t_j - f) \mathbf{v}_{n_j}'$

where $t_j = 1$ if going left, $t_j = 0$ if going right.

### Negative Sampling

**Core Idea**
Instead of normalizing over entire vocabulary, distinguish target word from noise words.

**Objective Function**
$$\mathcal{L} = \log \sigma(\mathbf{v}_{w_O}^T \mathbf{u}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{u}_{w_I})]$$

where:
- $k$ is number of negative samples
- $P_n(w)$ is noise distribution for sampling negative words

**Noise Distribution**
Empirically, unigram distribution raised to 3/4 power works best:
$$P_n(w) = \frac{f(w)^{3/4}}{\sum_{w'} f(w')^{3/4}}$$

**Rationale**: 
- Raw frequency: $P(w) = f(w)/\sum f(w')$ gives too much weight to frequent words
- Uniform: $P(w) = 1/|V|$ doesn't account for frequency at all
- $3/4$ power: Compromise between uniform and unigram

**Sampling Algorithm**
```python
def sample_negative(word_freqs, k):
    # Create cumulative distribution
    freqs_34 = [f**0.75 for f in word_freqs]
    total = sum(freqs_34)
    cumulative = [sum(freqs_34[:i+1])/total for i in range(len(freqs_34))]
    
    # Sample k words
    negatives = []
    for _ in range(k):
        r = random.random()
        idx = binary_search(cumulative, r)
        negatives.append(idx)
    return negatives
```

**Training Update**
For positive pair $(w_I, w_O)$ and negative samples $\{w_{i}^{neg}\}$:

1. **Positive update**:
   $$\mathbf{v}_{w_O} \leftarrow \mathbf{v}_{w_O} + \eta (1 - \sigma(\mathbf{v}_{w_O}^T \mathbf{u}_{w_I})) \mathbf{u}_{w_I}$$

2. **Negative updates**:
   $$\mathbf{v}_{w_i^{neg}} \leftarrow \mathbf{v}_{w_i^{neg}} - \eta \sigma(\mathbf{v}_{w_i^{neg}}^T \mathbf{u}_{w_I}) \mathbf{u}_{w_I}$$

3. **Input update**:
   $$\mathbf{u}_{w_I} \leftarrow \mathbf{u}_{w_I} + \eta \left[(1 - \sigma(\mathbf{v}_{w_O}^T \mathbf{u}_{w_I})) \mathbf{v}_{w_O} - \sum_{i=1}^{k} \sigma(\mathbf{v}_{w_i^{neg}}^T \mathbf{u}_{w_I}) \mathbf{v}_{w_i^{neg}}\right]$$

## Advanced Training Techniques

### Subword Information with FastText

**Motivation**
- Handle out-of-vocabulary words
- Capture morphological information
- Improve representations for rare words

**Character n-gram Representation**
Represent each word as bag of character n-grams:
$$\text{word} = \langle \text{word} \rangle$$
$$\text{n-grams}(\text{"word"}) = \{\langle\text{wo}, \text{wor}, \text{ord}, \text{rd}\rangle\}$$

**Embedding Computation**
$$\mathbf{u}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

where $\mathcal{G}_w$ is set of n-grams for word $w$ and $\mathbf{z}_g$ is embedding for n-gram $g$.

**Benefits**:
- **Morphological awareness**: Related words (run, running, runs) have similar embeddings
- **OOV handling**: Can compute embeddings for unseen words
- **Language flexibility**: Works well for morphologically rich languages

**Training Complexity**
Each word now involves $|\mathcal{G}_w|$ embeddings instead of 1.

### Dynamic Context Windows

**Fixed vs Dynamic Windows**
- **Fixed**: Always use window size $c$
- **Dynamic**: Randomly sample window size $\tilde{c} \in [1, c]$ for each word

**Sampling Procedure**
```python
def dynamic_window(max_window):
    return random.randint(1, max_window)
```

**Benefits**:
- **Closer words get more weight**: Smaller windows emphasize immediate context
- **Diverse training examples**: Same word pair trained with different distances
- **Better rare word representations**: More focused contexts

### Subsampling Frequent Words

**Problem**
Frequent words (the, of, and) provide little information but dominate training.

**Subsampling Probability**
$$P(\text{discard } w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

where:
- $f(w_i)$ is frequency of word $w_i$
- $t$ is threshold parameter (typically $10^{-5}$ to $10^{-4}$)

**Effect**:
- **Frequent words**: High discard probability
- **Rare words**: Low discard probability
- **Balanced training**: More attention to informative word pairs

**Mathematical Justification**
Subsampling preserves relative frequencies while reducing dominance of frequent words:
$$\frac{f'(w_1)}{f'(w_2)} \approx \frac{f(w_1)}{f(w_2)} \text{ for rare words}$$

## Theoretical Analysis and Properties

### Matrix Factorization Perspective

**Implicit Matrix Factorization**
Word2Vec with negative sampling implicitly factorizes matrix of PMI values:
$$\mathbf{M}_{ij} = \text{PMI}(w_i, w_j) - \log k$$

where $k$ is number of negative samples.

**Proof Sketch**
At convergence, Skip-gram objective equivalent to:
$$\mathbf{u}_{w_I}^T \mathbf{v}_{w_O} = \log P(w_O | w_I) + \text{const}$$

Given training distribution and negative sampling:
$$\log P(w_O | w_I) = \text{PMI}(w_I, w_O) - \log k + \text{const}$$

**Connection to Classical Methods**
Word2Vec bridges neural and count-based methods:
- **Input**: Raw co-occurrence counts (implicit)
- **Process**: Neural training with efficient approximations
- **Output**: Dense factorization of PMI matrix

### Embedding Space Properties

**Linear Relationships**
Famous example: $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$

**Mathematical Explanation**
If semantic relationships are additive:
$$\mathbf{v}_{a:b} = \mathbf{v}_b - \mathbf{v}_a$$

Then: $\mathbf{v}_{king:queen} \approx \mathbf{v}_{man:woman}$

**Empirical Analysis**
- **Syntactic analogies**: Verb tenses, plurals, comparatives
- **Semantic analogies**: Country-capital, family relationships
- **Limitations**: Not all relationships are linear

### Geometric Properties

**Cosine Similarity Interpretation**
$$\cos(\mathbf{u}_i, \mathbf{u}_j) = \frac{\mathbf{u}_i^T \mathbf{u}_j}{\|\mathbf{u}_i\| \|\mathbf{u}_j\|}$$

High cosine similarity indicates words appearing in similar contexts.

**Vector Length**
Vector magnitude often correlates with word frequency:
$$\|\mathbf{u}_w\| \approx f(w)^\alpha$$

**Anisotropy Problem**
Embeddings occupy narrow cone in vector space rather than full space:
- **Cause**: Optimization dynamics and frequency effects
- **Solution**: Post-processing techniques (centering, PCA)

## Implementation Optimizations

### Efficient Data Structures

**Huffman Tree Implementation**
```python
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self.code = ""

def build_huffman_tree(word_freqs):
    heap = [HuffmanNode(w, f) for w, f in word_freqs.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
    
    return heap[0]
```

**Negative Sampling Table**
Pre-compute sampling table for efficient negative sampling:
```python
def create_sampling_table(word_freqs, table_size=1e8):
    freqs_pow = [f**0.75 for f in word_freqs]
    total_pow = sum(freqs_pow)
    
    table = []
    cumulative = 0
    word_idx = 0
    
    for i in range(int(table_size)):
        table.append(word_idx)
        if i / table_size > cumulative:
            word_idx += 1
            cumulative += freqs_pow[word_idx] / total_pow
    
    return table
```

### Memory Optimization

**Vocabulary Encoding**
- Use integer IDs instead of strings
- Create word-to-ID and ID-to-word mappings
- Store only necessary information

**Batch Processing**
Process multiple training examples together:
```python
def batch_skipgram(words, window_size, batch_size):
    batch = []
    for i, word in enumerate(words):
        for j in range(max(0, i-window_size), 
                      min(len(words), i+window_size+1)):
            if i != j:
                batch.append((word, words[j]))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
```

### Parallel Training

**Hogwild Training**
Asynchronous SGD without locks:
- Multiple threads update shared parameters
- Sparse updates minimize conflicts
- Empirically stable for Word2Vec

**Thread Safety**
```python
import threading

class ThreadSafeWord2Vec:
    def __init__(self, vocab_size, embed_size):
        self.embeddings = np.random.randn(vocab_size, embed_size)
        self.lock = threading.Lock()
    
    def update_embedding(self, word_id, gradient):
        # Atomic update without explicit locking
        # Relies on GIL and sparse updates
        self.embeddings[word_id] += gradient
```

## Evaluation and Analysis

### Intrinsic Evaluation

**Word Similarity Tasks**
Compute correlation with human judgments:
```python
def evaluate_similarity(embeddings, similarity_dataset):
    model_sims = []
    human_sims = []
    
    for word1, word2, human_score in similarity_dataset:
        if word1 in embeddings and word2 in embeddings:
            v1 = embeddings[word1]
            v2 = embeddings[word2]
            model_score = cosine_similarity(v1, v2)
            
            model_sims.append(model_score)
            human_sims.append(human_score)
    
    return pearson_correlation(model_sims, human_sims)
```

**Word Analogy Tasks**
Solve analogies using vector arithmetic:
```python
def solve_analogy(embeddings, a, b, c, top_k=1):
    """Solve a:b :: c:? analogy"""
    if not all(w in embeddings for w in [a, b, c]):
        return None
    
    target = embeddings[b] - embeddings[a] + embeddings[c]
    
    # Find most similar word (excluding input words)
    similarities = {}
    for word, vec in embeddings.items():
        if word not in [a, b, c]:
            sim = cosine_similarity(target, vec)
            similarities[word] = sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

### Probing Tasks

**Syntactic Information**
Test if embeddings capture syntactic properties:
- **POS Tagging**: Use embeddings as features
- **Syntactic Parsing**: Evaluate on parsing tasks
- **Grammatical Relations**: Test subject-verb agreement

**Semantic Information**
Evaluate semantic understanding:
- **Semantic Role Labeling**: Identify semantic roles
- **Word Sense Disambiguation**: Distinguish word senses
- **Semantic Similarity**: Compare with human judgments

### Visualization and Analysis

**t-SNE Visualization**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, perplexity=30):
    # Select subset of embeddings
    word_vecs = np.array([embeddings[w] for w in words if w in embeddings])
    word_labels = [w for w in words if w in embeddings]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    word_vecs_2d = tsne.fit_transform(word_vecs)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1])
    
    for i, word in enumerate(word_labels):
        plt.annotate(word, (word_vecs_2d[i, 0], word_vecs_2d[i, 1]))
    
    plt.title('Word Embeddings Visualization')
    plt.show()
```

**Nearest Neighbors Analysis**
```python
def find_nearest_neighbors(embeddings, word, k=10):
    if word not in embeddings:
        return []
    
    target_vec = embeddings[word]
    similarities = {}
    
    for w, vec in embeddings.items():
        if w != word:
            sim = cosine_similarity(target_vec, vec)
            similarities[w] = sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
```

## Extensions and Variants

### Doc2Vec (Paragraph Vector)

**Architecture**
Extend Word2Vec to learn document representations:
- **PV-DM**: Distributed Memory version (like CBOW)
- **PV-DBOW**: Distributed Bag of Words version (like Skip-gram)

**PV-DM Formulation**
$$\mathcal{L} = \sum_{d \in D} \sum_{w \in d} \log p(w | \text{context}(w), \mathbf{d})$$

where $\mathbf{d}$ is document vector.

**Applications**:
- Document classification
- Information retrieval
- Sentiment analysis

### GloVe (Global Vectors)

**Motivation**
Combine benefits of matrix factorization and neural methods.

**Objective Function**
$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

where:
- $X_{ij}$ is co-occurrence count
- $f(x)$ is weighting function
- $w_i, \tilde{w}_j$ are word vectors

**Weighting Function**
$$f(x) = \begin{cases}
(x/x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}$$

### Sense2Vec

**Multiple Senses per Word**
Learn separate embeddings for different word senses:
- Use POS tags: "bank_NOUN" vs "bank_VERB"
- Use entity types: "Washington_PERSON" vs "Washington_GPE"

**Training**
Treat different senses as separate vocabulary items.

## Key Questions for Review

### Architecture and Training
1. **Skip-gram vs CBOW**: When should you choose Skip-gram over CBOW, and what are the computational trade-offs?

2. **Negative Sampling**: Why is the 3/4 power of unigram distribution optimal for negative sampling?

3. **Hierarchical Softmax**: How does the tree structure in hierarchical softmax affect the quality of learned embeddings?

### Optimization Techniques
4. **Subsampling**: How does subsampling frequent words improve embedding quality and training efficiency?

5. **Dynamic Windows**: What are the benefits of using dynamic context windows during training?

6. **FastText**: How do subword embeddings in FastText handle morphologically rich languages better than Word2Vec?

### Theoretical Understanding
7. **Matrix Factorization**: What is the relationship between Word2Vec and traditional matrix factorization methods?

8. **Linear Relationships**: Why do word embeddings exhibit linear algebraic structure for analogical reasoning?

9. **Geometric Properties**: What causes anisotropy in embedding spaces, and how can it be addressed?

### Evaluation and Analysis
10. **Intrinsic vs Extrinsic**: How do intrinsic evaluation tasks relate to downstream performance?

11. **Bias Detection**: How can social and cultural biases be detected and measured in word embeddings?

12. **Cross-Lingual Evaluation**: How should word embeddings be evaluated across different languages?

## Conclusion

Word2Vec and Skip-gram models represent a paradigm shift in natural language processing by introducing efficient neural methods for learning high-quality word embeddings that capture rich semantic and syntactic relationships. This comprehensive exploration has established:

**Architectural Innovation**: Deep understanding of Skip-gram and CBOW architectures demonstrates how neural networks can be designed to learn distributed word representations efficiently from large text corpora while capturing meaningful linguistic relationships.

**Optimization Breakthroughs**: Systematic analysis of hierarchical softmax, negative sampling, and other training optimizations reveals how computational challenges in neural language modeling can be addressed through clever approximations and sampling strategies.

**Theoretical Foundations**: Understanding of the matrix factorization perspective and geometric properties provides mathematical insights into why neural word embeddings exhibit linear algebraic structure and enable analogical reasoning through vector arithmetic.

**Training Techniques**: Comprehensive coverage of subsampling, dynamic windows, subword modeling, and parallel training demonstrates advanced techniques for improving embedding quality and training efficiency on large-scale datasets.

**Evaluation Methodologies**: Integration of intrinsic and extrinsic evaluation approaches provides frameworks for assessing embedding quality across different dimensions and understanding the relationship between geometric properties and downstream task performance.

**Extensions and Variants**: Analysis of FastText, Doc2Vec, GloVe, and other extensions shows how the core Word2Vec principles can be adapted and extended for different applications and linguistic phenomena.

Word2Vec and Skip-gram models are crucial for modern NLP because:
- **Foundation for Neural NLP**: Established the paradigm of learning distributed representations that underlies modern neural language models
- **Semantic Computation**: Enabled computational systems to perform semantic reasoning through geometric operations on word vectors
- **Scalable Learning**: Provided efficient methods for learning from large text corpora that scale to web-scale datasets
- **Transfer Learning**: Created general-purpose representations that transfer effectively across different NLP tasks and domains
- **Analogical Reasoning**: Demonstrated that neural networks can capture abstract relationships that enable logical reasoning through vector arithmetic

The mathematical frameworks, optimization techniques, and theoretical insights covered provide essential knowledge for understanding how neural word embeddings work and how to implement them effectively. Understanding these principles is fundamental for developing modern neural language models, contextual embeddings, and other advanced natural language processing systems that rely on distributed semantic representations.