# Day 14.3: Advanced Embedding Techniques - GloVe, FastText, and Contextual Models

## Overview

Advanced embedding techniques represent the evolution of word representation learning beyond basic Word2Vec models, incorporating sophisticated mathematical frameworks and architectural innovations that address fundamental limitations of earlier approaches while enabling more nuanced understanding of linguistic meaning and structure. These techniques include GloVe (Global Vectors) which combines global matrix factorization with local context windows, FastText which captures morphological information through subword representations, ELMo which introduces contextualized embeddings that vary based on surrounding words, and various specialized approaches for handling polysemy, rare words, multilingual scenarios, and domain-specific applications. The advancement from static to contextual embeddings has fundamentally transformed natural language processing by enabling models to capture nuanced meanings that depend on specific contexts, leading to breakthrough performance improvements across diverse language understanding tasks.

## GloVe: Global Vectors for Word Representation

### Mathematical Foundation

**Co-occurrence Statistics**
Define co-occurrence matrix $X$ where $X_{ij}$ represents number of times word $j$ appears in context of word $i$:
$$X_{ij} = \sum_{k=1}^{C} \frac{1}{d(i,j,k)}$$

where $d(i,j,k)$ is distance between words $i$ and $j$ in context $k$.

**Global Statistics Utilization**
Unlike Word2Vec which uses local context windows, GloVe leverages global co-occurrence statistics from entire corpus.

**Ratios of Co-occurrence Probabilities**
Key insight: ratios reveal semantic relationships better than raw probabilities:
$$P_{ik} = P(w_k | w_i) = \frac{X_{ik}}{X_i}$$

where $X_i = \sum_j X_{ij}$.

**Example Analysis**:
For words related to "ice" and "steam":
- $\frac{P(solid | ice)}{P(solid | steam)} = \frac{8.9 \times 10^{-5}}{2.2 \times 10^{-6}} = 40.5$ (large)
- $\frac{P(fashion | ice)}{P(fashion | steam)} = \frac{9.3 \times 10^{-6}}{9.2 \times 10^{-6}} = 1.01$ (near 1)

Ratios discriminate relevant from irrelevant words effectively.

### GloVe Model Architecture

**Objective Function Derivation**
Start with ratio of probabilities:
$$F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}$$

**Requirements for Function $F$**:
1. **Vector space structure**: $F$ should depend on differences of word vectors
2. **Symmetry**: $F(\mathbf{w}_i - \mathbf{w}_j, \mathbf{w}_k) = \frac{F(\mathbf{w}_i, \mathbf{w}_k)}{F(\mathbf{w}_j, \mathbf{w}_k)}$
3. **Linearity**: $F(\mathbf{w}_i, \mathbf{w}_k) = \exp(\mathbf{w}_i^T \mathbf{w}_k)$

**Final Objective**
$$J = \sum_{i,j=1}^{V} f(X_{ij}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

where:
- $\mathbf{w}_i, \tilde{\mathbf{w}}_j$ are word and context vectors
- $b_i, \tilde{b}_j$ are bias terms
- $f(X_{ij})$ is weighting function

**Weighting Function**
$$f(x) = \begin{cases}
(x/x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}$$

**Properties**:
- $f(0) = 0$ (continuity at origin)
- Non-decreasing to avoid overweighting rare pairs
- Saturates for frequent pairs to avoid overweighting common words

**Typical Parameters**: $x_{\max} = 100$, $\alpha = 0.75$

### Training Procedure

**Optimization Algorithm**
Use AdaGrad for adaptive learning rates:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla J(\theta_t)$$

where $G_t$ accumulates squared gradients.

**Initialization Strategy**
- Word vectors: Random uniform $[-0.5, 0.5]$
- Biases: Zero initialization
- Context vectors: Independent random initialization

**Convergence Analysis**
GloVe typically converges faster than Word2Vec because:
- Direct optimization of global objective
- No sampling-based approximations
- Stable gradients from matrix factorization

**Final Word Representations**
Average word and context vectors:
$$\mathbf{v}_w^{\text{final}} = \frac{\mathbf{w}_w + \tilde{\mathbf{w}}_w}{2}$$

**Rationale**: Both sets capture complementary information about word meaning.

### Computational Complexity

**Co-occurrence Matrix Construction**
- **Time**: $O(C)$ where $C$ is corpus size
- **Space**: $O(|V|^2)$ worst case, typically much smaller due to sparsity

**Training Complexity**
- **Per iteration**: $O(\text{nnz}(X) \times d)$ where $\text{nnz}(X)$ is number of non-zero entries
- **Total**: Depends on convergence rate and matrix sparsity

**Memory Optimization**
- Store only non-zero entries
- Use sparse matrix representations
- Batch processing for large vocabularies

## FastText: Subword Information

### Character N-gram Representation

**Motivation**
Word-level embeddings face limitations:
- Out-of-vocabulary (OOV) words
- Morphologically rich languages
- Rare word representations
- Compositionality at character level

**Character N-gram Extraction**
For word $w$ with characters $c_1, c_2, ..., c_l$:
$$\mathcal{G}_w = \{<c_1c_2...c_n>, <c_2c_3...c_{n+1}>, ..., <c_{l-n+1}...c_l>\}$$

**Boundary Markers**
Add special characters to mark word boundaries:
$$\text{"where"} \rightarrow \text{"<where>"}$$

**N-gram Examples**:
- 3-grams of "where": {<wh, whe, her, ere, re>}
- 4-grams of "where": {<whe, wher, here, ere>}

### FastText Architecture

**Word Representation**
$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

where $\mathbf{z}_g$ is embedding of character n-gram $g$.

**Skip-gram with Subwords**
$$s(w_c, w_t) = \sum_{g \in \mathcal{G}_{w_c}} \mathbf{z}_g^T \mathbf{v}_{w_t}$$

**Training Objective**
$$\mathcal{L} = -\sum_{(w_c,w_t) \in \mathcal{D}} \log \sigma(s(w_c, w_t)) - \sum_{(w_c,n) \in \mathcal{N}} \log \sigma(-s(w_c, n))$$

where $\mathcal{D}$ is positive pairs and $\mathcal{N}$ is negative samples.

**Gradient Updates**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_g} = \sum_{w: g \in \mathcal{G}_w} \frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}$$

Each n-gram embedding receives gradients from all words containing it.

### Subword Tokenization Strategies

**Byte Pair Encoding (BPE)**
Iterative merging algorithm:
1. Initialize vocabulary with characters
2. Find most frequent adjacent pair
3. Merge pair into single token
4. Repeat until desired vocabulary size

**Algorithm**:
```
vocab = set(characters)
while |vocab| < target_size:
    pairs = count_pairs(corpus)
    best_pair = max(pairs, key=pairs.get)
    corpus = merge_vocab(corpus, best_pair)
    vocab.add(''.join(best_pair))
```

**SentencePiece**
Language-independent tokenization:
- Treats whitespace as regular character
- No pre-tokenization required
- Unigram language model for segmentation

**Unigram Language Model**:
$$P(X) = \prod_{i=1}^{|X|} P(x_i)$$

Segment to maximize likelihood under unigram model.

**WordPiece**
Similar to BPE but uses likelihood-based merging:
$$\text{score}(pair) = \frac{\text{freq}(pair)}{\text{freq}(first) \times \text{freq}(second)}$$

### Morphological Analysis

**Handling Inflectional Morphology**
Character n-grams capture morphological regularities:
- "running", "runner", "runs" share "run"
- Suffix patterns: "-ing", "-er", "-s"
- Prefix patterns: "un-", "re-", "pre-"

**Cross-lingual Benefits**
Subword information helps related languages:
- Shared character patterns
- Cognates: "information" (EN) ↔ "información" (ES)
- Morphological similarities

**Rare Word Handling**
Even unseen words can be represented:
$$\mathbf{v}_{\text{unseen}} = \sum_{g \in \mathcal{G}_{\text{unseen}}} \mathbf{z}_g$$

Quality depends on n-gram overlap with training vocabulary.

### FastText Extensions

**Language Identification**
Use character n-grams for language detection:
$$P(\text{lang} | \text{text}) = \text{softmax}(W \cdot \text{FastText}(\text{text}))$$

**Text Classification**
Average word embeddings for document representation:
$$\mathbf{d} = \frac{1}{|N|} \sum_{n=1}^{N} \mathbf{v}_{w_n}$$

**Hierarchical Softmax**
Use Huffman tree for efficient training:
- Frequent words: shorter paths
- Rare words: longer paths
- $O(\log |V|)$ complexity per prediction

## Contextualized Embeddings

### Motivation for Context-Dependent Representations

**Polysemy Problem**
Same word, different meanings:
- "bank": financial institution vs river bank
- "play": theatrical performance vs sports activity
- "python": snake vs programming language

**Static Embedding Limitations**
Word2Vec/GloVe assign single vector per word:
$$\mathbf{v}_{\text{bank}} = \text{constant vector}$$

Averages over all contexts and senses.

**Context-Dependent Solution**
$$\mathbf{v}_{w,c} = f(w, \text{context})$$

Vector changes based on surrounding words.

### ELMo: Embeddings from Language Models

**Bidirectional Language Model**
**Forward LM**: $P(t_1, t_2, ..., t_N) = \prod_{k=1}^{N} P(t_k | t_1, ..., t_{k-1})$
**Backward LM**: $P(t_1, t_2, ..., t_N) = \prod_{k=1}^{N} P(t_k | t_{k+1}, ..., t_N)$

**BiLM Objective**
$$\sum_{k=1}^{N} (\log P(t_k | t_1, ..., t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{\text{LSTM}}, \Theta_s) + \log P(t_k | t_{k+1}, ..., t_N; \Theta_x, \overleftarrow{\Theta}_{\text{LSTM}}, \Theta_s))$$

**Architecture**
1. **Character CNN**: Maps tokens to representations
2. **Bidirectional LSTM**: $L$ layers in each direction
3. **Softmax**: Predict next/previous word

**Character-level Input**
$$\mathbf{x}_k^{\text{LM}} = \text{CharCNN}(\text{token}_k)$$

Benefits:
- Handles OOV words
- Captures morphology
- Robust to tokenization

**ELMo Representation**
$$\text{ELMo}_k^{\text{task}} = E(R_k; \Theta^{\text{task}}) = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \mathbf{h}_{k,j}^{LM}$$

where:
- $R_k = \{\mathbf{x}_k^{LM}, \overrightarrow{\mathbf{h}}_{k,j}^{LM}, \overleftarrow{\mathbf{h}}_{k,j}^{LM} | j = 1, ..., L\}$
- $s_j^{\text{task}}$ are task-specific softmax weights
- $\gamma^{\text{task}}$ is task-specific scaling factor

**Layer-wise Analysis**
- **Layer 0**: Token representation (syntax-heavy)
- **Layer 1**: Local syntax (POS, NER)  
- **Layer 2**: Global syntax + semantics
- **Higher layers**: More semantic information

### Training Strategies

**Pre-training Phase**
Train bidirectional language model on large corpus:
- Dataset: 1 Billion Word Benchmark
- Architecture: 2-layer BiLSTM with 4096 units
- Training time: Several days on multiple GPUs

**Fine-tuning Phase**
1. Freeze pre-trained BiLM parameters
2. Learn task-specific weighting parameters $s_j^{\text{task}}, \gamma^{\text{task}}$
3. Add ELMo representations to task-specific model

**Integration Methods**
**Input Concatenation**:
$$[\mathbf{x}_k; \text{ELMo}_k^{\text{task}}]$$

**Hidden State Concatenation**:
$$[\mathbf{h}_k; \text{ELMo}_k^{\text{task}}]$$

**Weighted Sum**:
$$\alpha \mathbf{h}_k + (1-\alpha) \text{ELMo}_k^{\text{task}}$$

## Specialized Embedding Techniques

### Sense Embeddings

**Multi-Prototype Models**
Learn multiple vectors per word:
$$\mathbf{V}_w = \{\mathbf{v}_{w,1}, \mathbf{v}_{w,2}, ..., \mathbf{v}_{w,k}\}$$

**Context Clustering**
1. Collect all contexts for word $w$
2. Cluster contexts using k-means
3. Learn separate embedding for each cluster

**Cluster Assignment**
$$c^* = \arg\max_c P(c | \text{context})$$
$$\mathbf{v}_w = \mathbf{v}_{w,c^*}$$

**AdaGram Model**
Nonparametric approach to sense discovery:
- Automatic determination of sense number
- Hierarchical Dirichlet processes
- Sense probability estimation

### Cross-lingual Embeddings

**Bilingual Dictionary Approach**
Learn transformation between embedding spaces:
$$\min_W \|WX - Y\|_F^2$$

where $X$ and $Y$ are embeddings of translation pairs.

**Canonical Correlation Analysis (CCA)**
Find correlated subspaces:
$$\max_{u,v} \text{corr}(Xu, Yv)$$

**Adversarial Training**
Discriminator distinguishes language embeddings:
$$\min_\theta \max_\phi \mathbb{E}_{x \sim P_{\text{src}}} [\log D_\phi(f_\theta(x))] + \mathbb{E}_{y \sim P_{\text{tgt}}} [\log(1 - D_\phi(y))]$$

**MUSE (Multilingual Unsupervised and Supervised Embeddings)**
Adversarial training + refinement:
1. Adversarial training for rough alignment
2. Refinement with Procrustes analysis
3. Cross-domain similarity local scaling (CSLS)

### Domain-Specific Embeddings

**Domain Adaptation**
Fine-tune general embeddings on domain-specific corpora:
$$\mathcal{L}_{\text{domain}} = \mathcal{L}_{\text{general}} + \lambda \mathcal{L}_{\text{specific}}$$

**Multi-Domain Embeddings**
Learn shared and domain-specific components:
$$\mathbf{v}_{w,d} = \mathbf{v}_w^{\text{shared}} + \mathbf{v}_w^{d}$$

**Technical Domain Examples**:
- **Medical**: BERT fine-tuned on PubMed (BioBERT)
- **Legal**: Embeddings on legal documents
- **Scientific**: SciBERT on scientific papers

## Evaluation and Comparison

### Intrinsic Evaluation

**Word Similarity Benchmarks**
- **WordSim-353**: General word similarity
- **SimLex-999**: True similarity vs relatedness
- **MEN**: Large-scale similarity dataset
- **SimVerb-3500**: Verb similarity

**Word Analogy Tasks**
$$\mathbf{d} = \arg\max_{w \in V} \cos(\mathbf{w}, \mathbf{b} - \mathbf{a} + \mathbf{c})$$

**Categories**:
- Semantic: countries, capitals, family
- Syntactic: verb tenses, plurals
- Morphological: comparative, superlative

**Multilingual Evaluation**
- Cross-lingual word similarity
- Bilingual dictionary induction
- Cross-lingual document classification

### Extrinsic Evaluation

**Downstream Tasks**
- **NER**: Named Entity Recognition
- **POS**: Part-of-Speech tagging  
- **Sentiment**: Sentiment classification
- **QA**: Question answering
- **MT**: Machine translation

**Contextualized vs Static Comparison**
| Task | Word2Vec | GloVe | ELMo | Performance Gap |
|------|----------|--------|------|----------------|
| NER | 84.3 | 84.6 | 92.2 | +7.6 |
| Sentiment | 87.2 | 87.8 | 91.0 | +3.2 |
| SQuAD | 76.9 | 77.1 | 85.8 | +8.7 |

**Probing Tasks**
Test linguistic knowledge in embeddings:
- **Surface**: Sentence length, word content
- **Syntactic**: Tree depth, POS tags
- **Semantic**: Semantic roles, coreference

### Computational Analysis

**Training Time Comparison**
| Model | Corpus Size | Training Time | Hardware |
|-------|-------------|---------------|----------|
| Word2Vec | 100B tokens | 1 day | 8 CPUs |
| GloVe | 6B tokens | 85 min | 1 GPU |
| FastText | 16B tokens | 3 hours | 20 CPUs |
| ELMo | 1B tokens | 2 weeks | 8 GPUs |

**Memory Requirements**
- **Storage**: Vocabulary size × embedding dimension
- **Training**: Depends on architecture complexity
- **Inference**: Context-dependent models require more memory

**Inference Speed**
- **Static**: O(1) lookup
- **Contextualized**: O(sequence length × model depth)

## Recent Advances and Future Directions

### Transformer-based Embeddings

**BERT Preview**
Bidirectional encoder representations:
- Masked language modeling
- Next sentence prediction
- Transformer architecture

**Advantages over ELMo**:
- True bidirectionality
- Self-attention mechanisms
- Better parallelization

### Dynamic Embeddings

**Time-aware Embeddings**
$$\mathbf{v}_{w,t} = f(w, \text{time})$$

Capture semantic change over time:
- Historical word meaning evolution
- Trending topic analysis
- Temporal document analysis

**Meta-embeddings**
Combine multiple embedding types:
$$\mathbf{v}_w^{\text{meta}} = \alpha_1 \mathbf{v}_w^{\text{Word2Vec}} + \alpha_2 \mathbf{v}_w^{\text{GloVe}} + \alpha_3 \mathbf{v}_w^{\text{FastText}}$$

### Multimodal Embeddings

**Vision-Language Models**
Joint learning of visual and textual representations:
- Image-caption pairs
- Visual question answering
- Cross-modal retrieval

**Audio-Text Embeddings**
Speech and text alignment:
- Speech recognition
- Audio captioning
- Cross-modal search

## Key Questions for Review

### Model Comparison
1. **GloVe vs Word2Vec**: What are the fundamental differences in how GloVe and Word2Vec learn word representations?

2. **Subword Benefits**: When are subword-based models like FastText most beneficial compared to word-level embeddings?

3. **Contextualization**: How do contextualized embeddings like ELMo address limitations of static embeddings?

### Technical Understanding
4. **Matrix Factorization**: How does GloVe's approach to matrix factorization differ from traditional techniques like SVD?

5. **Character N-grams**: What are the trade-offs between different n-gram sizes in FastText?

6. **Layer Combination**: How should different layers be weighted in contextualized embeddings like ELMo?

### Evaluation and Analysis
7. **Intrinsic vs Extrinsic**: How do performance improvements on intrinsic tasks correlate with downstream task performance?

8. **Cross-lingual Quality**: What factors determine the success of cross-lingual embedding alignment?

9. **Domain Adaptation**: When is domain-specific training necessary versus general-purpose embeddings?

### Practical Considerations
10. **Computational Trade-offs**: How do you balance embedding quality against computational requirements?

11. **OOV Handling**: What strategies work best for handling out-of-vocabulary words in different applications?

12. **Polysemy Resolution**: How effectively do different approaches handle polysemous words in practice?

## Conclusion

Advanced embedding techniques represent significant evolution in word representation learning, progressing from static context-independent vectors to sophisticated context-aware models that capture nuanced linguistic phenomena and support diverse natural language processing applications. This comprehensive exploration has established:

**Methodological Innovation**: Understanding of GloVe's global statistical approach, FastText's subword modeling, and ELMo's contextualization demonstrates how different mathematical frameworks and architectural choices address specific limitations of earlier embedding methods.

**Technical Sophistication**: Deep analysis of matrix factorization, character-level processing, and bidirectional language modeling reveals the mathematical and computational techniques that enable more sophisticated semantic representation learning.

**Linguistic Coverage**: Systematic treatment of polysemy handling, morphological analysis, cross-lingual modeling, and domain specialization shows how advanced techniques address real-world linguistic complexity and diversity.

**Evaluation Frameworks**: Comprehensive coverage of intrinsic and extrinsic evaluation methods provides tools for assessing embedding quality across different dimensions and understanding the strengths and limitations of various approaches.

**Practical Applications**: Integration of computational considerations, training strategies, and deployment techniques demonstrates how advanced embeddings can be effectively implemented and utilized in production systems.

**Research Trajectory**: Analysis of the progression from Word2Vec through ELMo illustrates the evolution toward more sophisticated contextual understanding that culminates in modern transformer-based models.

Advanced embedding techniques are crucial for modern NLP because:
- **Context Sensitivity**: Enable meaning representations that adapt to specific linguistic contexts and usage patterns
- **Morphological Awareness**: Capture subword structure and morphological relationships critical for many languages
- **Cross-lingual Capability**: Support multilingual and cross-lingual natural language processing applications
- **Domain Adaptability**: Allow specialization for specific domains while maintaining general linguistic knowledge
- **Foundation for Progress**: Establish principles and techniques that inform modern contextualized language models

The theoretical frameworks and practical techniques covered provide essential knowledge for understanding how word embeddings have evolved to address increasingly sophisticated requirements in natural language understanding. Understanding these principles is fundamental for working with modern language models and developing embedding-based solutions for complex linguistic tasks across diverse domains and applications.