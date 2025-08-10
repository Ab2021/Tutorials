# Day 17.1: GPT Architecture and Autoregressive Training - Generative Language Modeling

## Overview

GPT (Generative Pre-trained Transformer) represents a revolutionary approach to language modeling that leverages the transformer decoder architecture for autoregressive text generation, establishing the foundation for modern large-scale generative language models through innovative unsupervised pretraining on massive text corpora followed by task-specific fine-tuning. Unlike BERT's bidirectional approach designed primarily for understanding tasks, GPT's unidirectional architecture enables natural language generation while maintaining the powerful attention mechanisms and scalability benefits of transformer architectures. The mathematical principles underlying autoregressive modeling, the architectural design choices that enable effective generation, the training methodologies that produce coherent and contextually appropriate text, and the theoretical foundations that explain GPT's remarkable few-shot learning capabilities provide crucial insights into how large language models can be trained to generate human-like text across diverse domains and applications.

## Historical Context and Motivation

### Evolution of Language Modeling

**Traditional N-gram Models**
Classical language models estimate probability using local context:
$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-n+1}, ..., w_{i-1})$$

**Limitations**:
- **Short context**: Limited to fixed n-gram windows
- **Data sparsity**: Many n-grams unseen in training data
- **No semantic understanding**: Purely statistical associations

**Neural Language Models**
RNN-based models introduced distributed representations:
$$P(w_i | w_1, ..., w_{i-1}) = \text{softmax}(W \mathbf{h}_i + \mathbf{b})$$

where $\mathbf{h}_i = \text{RNN}(\mathbf{h}_{i-1}, \mathbf{e}_{w_{i-1}})$.

**Improvements**:
- **Distributed representations**: Dense vector embeddings
- **Longer context**: RNN hidden state maintains history
- **Generalization**: Better handling of unseen word combinations

**Limitations**:
- **Sequential processing**: Cannot parallelize training
- **Vanishing gradients**: Difficulty capturing long-range dependencies
- **Limited context**: Fixed-size hidden state bottleneck

### The Transformer Revolution

**Attention-Based Modeling**
Transformers enabled parallel processing and unlimited context:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Key Advantages**:
- **Parallelization**: All positions processed simultaneously
- **Long-range dependencies**: Direct connections between distant positions
- **Scalability**: Efficient training on large datasets

**GPT's Innovation**
Combine transformer architecture with autoregressive training:
- **Decoder-only**: Focus on generation rather than encoding
- **Causal masking**: Maintain autoregressive property
- **Unsupervised pretraining**: Learn from raw text without labels
- **Transfer learning**: Fine-tune for specific tasks

## GPT Architecture Deep Dive

### Decoder-Only Transformer

**Architectural Choice**
GPT uses only the decoder part of the original transformer:
- **No encoder**: Direct generation from input context
- **Causal attention**: Prevent future information leakage
- **Autoregressive**: Generate one token at a time

**Mathematical Framework**
For input sequence $\mathbf{x}_{1:i-1}$, predict next token:
$$P(x_i | \mathbf{x}_{1:i-1}) = \text{softmax}(W_e \mathbf{h}_i^{(L)} + \mathbf{b}_e)$$

where $\mathbf{h}_i^{(L)}$ is the final layer representation at position $i$.

**Layer Structure**
Each GPT layer consists of:
1. **Masked self-attention**: $\mathbf{A}^{(l)} = \text{MaskedAttention}(\mathbf{H}^{(l-1)})$
2. **Position-wise feed-forward**: $\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$
3. **Residual connections**: $\mathbf{H}^{(l)} = \mathbf{H}^{(l-1)} + \text{Sublayer}(\mathbf{H}^{(l-1)})$
4. **Layer normalization**: Applied before each sublayer (pre-norm)

### Causal Self-Attention

**Masking Mechanism**
Prevent attention to future positions:
$$\mathbf{A}_{i,j} = \begin{cases}
\text{softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}}\right) & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

**Implementation**
Apply mask before softmax:
$$\text{scores} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$$
$$\text{masked\_scores}_{i,j} = \begin{cases}
\text{scores}_{i,j} & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}$$
$$\mathbf{A} = \text{softmax}(\text{masked\_scores})$$

**Causal Attention Pattern**
The resulting attention matrix is lower triangular:
$$\mathbf{A} = \begin{bmatrix}
\alpha_{1,1} & 0 & 0 & 0 \\
\alpha_{2,1} & \alpha_{2,2} & 0 & 0 \\
\alpha_{3,1} & \alpha_{3,2} & \alpha_{3,3} & 0 \\
\alpha_{4,1} & \alpha_{4,2} & \alpha_{4,3} & \alpha_{4,4}
\end{bmatrix}$$

**Information Flow**
Each position can only attend to previous positions:
$$\mathbf{h}_i^{(l)} = \text{LayerNorm}\left(\mathbf{h}_i^{(l-1)} + \sum_{j=1}^{i} \alpha_{i,j}^{(l)} W^V \mathbf{h}_j^{(l-1)}\right)$$

### Position Encoding

**Learned Positional Embeddings**
GPT uses learned position embeddings (unlike sinusoidal in original transformer):
$$\mathbf{h}_0 = \mathbf{W}_e[x_i] + \mathbf{W}_p[i]$$

where:
- $\mathbf{W}_e \in \mathbb{R}^{|V| \times d}$: Token embedding matrix
- $\mathbf{W}_p \in \mathbb{R}^{n_{ctx} \times d}$: Position embedding matrix
- $n_{ctx}$: Maximum context length

**Advantages of Learned Embeddings**:
- **Task-specific**: Optimized for specific applications
- **Flexible**: Can learn complex positional relationships
- **Interpretable**: Position embeddings can be analyzed

**Limitations**:
- **Fixed length**: Cannot handle sequences longer than training
- **Less generalizable**: May not transfer well to different domains

### Model Configurations

**GPT-1 (Original)**
- **Layers**: 12
- **Hidden size**: 768
- **Attention heads**: 12
- **Context length**: 512
- **Parameters**: 117M
- **FFN size**: 3072 (4× hidden size)

**Key Design Decisions**
- **Head dimension**: $d_k = d_v = 64$ (768/12)
- **Activation**: GELU instead of ReLU
- **Layer norm**: Pre-normalization (before attention/FFN)
- **Dropout**: Applied throughout model (0.1)

**Vocabulary and Tokenization**
- **BPE tokenization**: Byte Pair Encoding
- **Vocabulary size**: 40,000 tokens
- **Special tokens**: `<|endoftext|>` for sequence boundaries

## Autoregressive Training Methodology

### Language Modeling Objective

**Maximum Likelihood Estimation**
Train to maximize probability of observed sequences:
$$\mathcal{L} = \sum_{i=1}^{n} \log P(x_i | x_1, ..., x_{i-1}; \boldsymbol{\theta})$$

**Per-token Loss**
Cross-entropy loss for each position:
$$\mathcal{L}_i = -\log P(x_i | x_1, ..., x_{i-1})$$

**Batch Training**
For batch of sequences:
$$\mathcal{L}_{\text{batch}} = \frac{1}{|B|} \sum_{s \in B} \sum_{i=1}^{|s|} \mathcal{L}_{s,i}$$

**Teacher Forcing**
During training, use ground truth previous tokens:
- **Input**: `[START] The cat sat on`
- **Target**: `The cat sat on [END]`
- **Loss**: Computed on positions 1-5

### Training Data Preparation

**Text Corpus**
GPT-1 trained on BookCorpus dataset:
- **Size**: ~5GB of text
- **Content**: Over 11,000 books
- **Domains**: Fiction, non-fiction, various genres
- **Preprocessing**: Tokenization and sequence formation

**Sequence Construction**
Create training examples from continuous text:
```python
def create_training_sequences(text, seq_length, tokenizer):
    tokens = tokenizer.encode(text)
    sequences = []
    
    for i in range(0, len(tokens) - seq_length, seq_length):
        sequence = tokens[i:i + seq_length + 1]  # +1 for target
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        sequences.append((input_seq, target_seq))
    
    return sequences
```

**Data Augmentation**
Techniques to improve training:
- **Sequence packing**: Combine short sequences to utilize full context
- **Random truncation**: Vary sequence lengths during training
- **Document shuffling**: Mix content from different sources

### Optimization Strategy

**Learning Rate Schedule**
Linear warmup followed by cosine decay:
$$\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
\eta_{\max} \cdot \frac{1 + \cos(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}})}{2} & \text{if } t > T_{\text{warmup}}
\end{cases}$$

**Training Configuration**
- **Optimizer**: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **Learning rate**: $2.5 \times 10^{-4}$
- **Batch size**: 64 sequences
- **Gradient clipping**: Max norm = 1.0
- **Weight decay**: 0.01

**Training Stability**
Techniques for stable training:
- **Gradient accumulation**: Handle large effective batch sizes
- **Mixed precision**: FP16 for memory efficiency
- **Gradient checkpointing**: Trade computation for memory

### Perplexity and Evaluation

**Perplexity Metric**
Measure of model uncertainty:
$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_1, ..., x_{i-1})\right)$$

**Lower perplexity indicates better language modeling performance.**

**Information-Theoretic Interpretation**
Perplexity relates to entropy:
$$H = -\sum_x P(x) \log P(x)$$
$$\text{PPL} = 2^H$$

**Cross-Entropy Loss Relationship**:
$$\text{Cross-Entropy} = \log(\text{PPL})$$

## Unsupervised Pretraining Strategy

### Self-Supervised Learning Framework

**Learning from Raw Text**
GPT learns language patterns without explicit labels:
- **Input**: Raw text sequences
- **Supervision signal**: Next token prediction
- **No human annotation**: Fully unsupervised
- **Scalable**: Can use any text corpus

**Emergent Capabilities**
Pretraining develops various language skills:
- **Syntax**: Grammar and sentence structure
- **Semantics**: Word meanings and relationships
- **World knowledge**: Facts and common sense
- **Reasoning**: Basic logical inference

### Transfer Learning Paradigm

**Two-Stage Training**
1. **Pretraining**: Unsupervised language modeling
2. **Fine-tuning**: Supervised task-specific training

**Mathematical Formulation**
**Pretraining objective**:
$$\mathcal{L}_1 = \sum_{i} \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)$$

**Fine-tuning objective**:
$$\mathcal{L}_2 = \sum_{i} \log P(y_i | x_1^i, ..., x_m^i)$$

where $\{x_1, ..., x_m\}$ is input sequence and $y$ is label.

**Combined Training**:
$$\mathcal{L}_3 = \mathcal{L}_2 + \lambda \mathcal{L}_1$$

Adding language modeling as auxiliary task during fine-tuning.

### Task Adaptation Framework

**Unified Input Representation**
Convert all tasks to sequence-to-sequence format:

**Classification**:
```
Input: [START] text [DELIM] [CLASSIFY]
Output: label
```

**Textual Entailment**:
```
Input: [START] premise [DELIM] hypothesis [DELIM] [CLASSIFY] 
Output: entailment/contradiction/neutral
```

**Question Answering**:
```
Input: [START] context [DELIM] question [DELIM] [CLASSIFY]
Output: answer
```

**Similarity**:
```
Input: [START] text1 [DELIM] text2 [DELIM] [CLASSIFY]
Output: similar/different
```

### Fine-tuning Methodology

**Task-Specific Layers**
Add minimal task-specific components:
$$P(y | x_{1:m}) = \text{softmax}(W_y \mathbf{h}_m^{(L)} + \mathbf{b}_y)$$

**Learning Rate Strategy**
Different learning rates for different components:
- **Pretrained layers**: Small learning rate (1e-5)
- **Task-specific layers**: Larger learning rate (1e-4)
- **Gradual unfreezing**: Progressively unfreeze layers

**Regularization Techniques**
Prevent catastrophic forgetting:
- **Lower learning rates**: Preserve pretrained knowledge
- **Early stopping**: Prevent overfitting
- **Dropout**: Additional regularization during fine-tuning
- **Auxiliary loss**: Include language modeling loss

## Generation Capabilities

### Autoregressive Generation

**Sampling Process**
Generate text token by token:
```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_length):
        # Get model predictions
        with torch.no_grad():
            outputs = model(torch.tensor([tokens]))
            logits = outputs.logits[0, -1, :]  # Last position
        
        # Apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)
        
        # Check for end token
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(tokens)
```

**Temperature Sampling**
Control randomness in generation:
$$P_{\text{temp}}(x_i) = \frac{\exp(\frac{z_i}{T})}{\sum_j \exp(\frac{z_j}{T})}$$

**Temperature effects**:
- **T → 0**: Deterministic (argmax)
- **T = 1**: Normal sampling
- **T > 1**: More random/creative
- **T → ∞**: Uniform sampling

### Top-k and Top-p Sampling

**Top-k Sampling**
Sample only from k most likely tokens:
```python
def top_k_sampling(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[next_token_idx]
```

**Top-p (Nucleus) Sampling**
Sample from tokens with cumulative probability p:
```python
def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff index
    cutoff = torch.where(cumulative_probs > p)[0]
    if len(cutoff) > 0:
        last_idx = cutoff[0].item()
        sorted_logits[last_idx:] = -float('inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    return sorted_indices[next_token_idx]
```

### Controllable Generation

**Prompt Engineering**
Guide generation with carefully crafted prompts:
```python
prompts = {
    'story': "Once upon a time, in a land far away,",
    'technical': "To implement this algorithm, we need to",
    'conversation': "Human: How are you?\nAI:",
    'creative': "The most unusual thing about this painting is"
}
```

**Conditional Generation**
Generate text conditioned on specific attributes:
- **Style**: Formal vs informal language
- **Domain**: Technical vs creative writing
- **Sentiment**: Positive vs negative tone
- **Length**: Short vs long responses

## Theoretical Analysis

### Information-Theoretic Perspective

**Entropy and Surprisal**
Language models estimate information content:
$$H(X) = -\sum_x P(x) \log P(x)$$

**Surprisal** of token $x_i$:
$$I(x_i) = -\log P(x_i | x_{1:i-1})$$

**Relationship to Loss**:
Cross-entropy loss equals average surprisal.

**Compression Viewpoint**
Language models as compression algorithms:
- **Good model**: Assigns high probability to actual text
- **Compression ratio**: Related to perplexity
- **Optimal compression**: Achieves entropy lower bound

### Scaling Properties

**Parameter Scaling**
Performance improves with model size:
$$\text{Loss} \propto N^{-\alpha}$$

where $N$ is number of parameters and $\alpha \approx 0.076$.

**Data Scaling**
More training data improves performance:
$$\text{Loss} \propto D^{-\beta}$$

where $D$ is dataset size and $\beta \approx 0.095$.

**Compute Scaling**
Training compute affects final performance:
$$\text{Loss} \propto C^{-\gamma}$$

where $C$ is compute budget and $\gamma \approx 0.050$.

### Expressiveness Analysis

**Universal Approximation**
Transformers can approximate any sequence-to-sequence function:
- **Theoretical result**: With sufficient parameters and data
- **Practical limitation**: Computational constraints
- **Empirical observation**: Larger models more expressive

**Context Length Effects**
Longer context enables better modeling:
$$P(x_i | x_{1:i-1}) \text{ vs } P(x_i | x_{i-k:i-1})$$

**Memory and Computation Trade-off**:
- **Quadratic attention**: $O(n^2)$ complexity
- **Linear alternatives**: Approximate attention patterns
- **Hierarchical approaches**: Multi-scale processing

## Comparative Analysis with BERT

### Architectural Differences

| Aspect | GPT | BERT |
|--------|-----|------|
| **Architecture** | Decoder-only | Encoder-only |
| **Attention** | Causal (masked) | Bidirectional |
| **Training** | Next token prediction | Masked token prediction |
| **Primary use** | Generation | Understanding |
| **Context** | Left-to-right | Full context |

### Training Objectives

**GPT Autoregressive**:
$$\mathcal{L}_{\text{GPT}} = -\sum_{i=1}^{n} \log P(x_i | x_{1:i-1})$$

**BERT Masked LM**:
$$\mathcal{L}_{\text{BERT}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\mathcal{M}^c})$$

### Representation Differences

**GPT Representations**:
- **Causal**: Each position sees only previous context
- **Generation-focused**: Optimized for next token prediction
- **Temporal**: Strong temporal/sequential bias

**BERT Representations**:
- **Bidirectional**: Full context at every position
- **Understanding-focused**: Optimized for masked prediction
- **Symmetric**: Less temporal bias

### Task Suitability

**GPT Better For**:
- Text generation
- Language modeling
- Creative writing
- Dialogue systems
- Few-shot learning (with sufficient scale)

**BERT Better For**:
- Text classification
- Named entity recognition
- Question answering (extractive)
- Natural language inference
- Tasks requiring full context understanding

## Key Questions for Review

### Architecture Design
1. **Decoder-Only Choice**: Why does GPT use only the decoder part of the transformer architecture?

2. **Causal Masking**: How does causal attention enable autoregressive generation while preventing information leakage?

3. **Position Encoding**: What are the trade-offs between learned and sinusoidal position encodings in GPT?

### Training Methodology
4. **Autoregressive Training**: How does next-token prediction lead to coherent long-form text generation?

5. **Teacher Forcing**: What are the benefits and potential issues with teacher forcing during training?

6. **Perplexity Evaluation**: How does perplexity relate to generation quality and what are its limitations?

### Transfer Learning
7. **Pretraining Benefits**: What linguistic knowledge does unsupervised pretraining provide for downstream tasks?

8. **Task Adaptation**: How should fine-tuning be approached differently for generation vs classification tasks?

9. **Few-Shot Learning**: What properties of autoregressive training enable few-shot learning capabilities?

### Generation Quality
10. **Sampling Strategies**: How do different sampling methods (greedy, top-k, top-p) affect generation quality?

11. **Controllable Generation**: What techniques can guide GPT to generate text with specific properties?

12. **Length and Coherence**: How can long-form coherent generation be achieved with autoregressive models?

### Scaling and Performance
13. **Scaling Laws**: How do model performance and capabilities change with increasing scale?

14. **Context Length**: What are the computational and modeling trade-offs of longer context windows?

15. **Efficiency**: How can autoregressive generation be made more efficient for practical applications?

## Conclusion

GPT's architecture and autoregressive training methodology established a new paradigm for generative language modeling that demonstrates how transformer decoders can be trained on vast amounts of text to develop sophisticated language generation capabilities and emergent few-shot learning abilities. This comprehensive exploration has established:

**Architectural Innovation**: Deep understanding of the decoder-only transformer design, causal attention mechanisms, and position encoding strategies demonstrates how architectural choices enable effective autoregressive generation while maintaining computational efficiency and scalability.

**Training Methodology**: Systematic analysis of next-token prediction, teacher forcing, and optimization strategies reveals how simple autoregressive objectives can lead to complex language understanding and generation capabilities through large-scale training.

**Transfer Learning Framework**: Understanding of unsupervised pretraining followed by supervised fine-tuning shows how general language models can be adapted to diverse downstream tasks while leveraging learned linguistic knowledge.

**Generation Capabilities**: Comprehensive coverage of sampling strategies, controllable generation, and text quality demonstrates the practical applications and limitations of autoregressive language models for various generation tasks.

**Theoretical Foundations**: Analysis of information-theoretic principles, scaling laws, and expressiveness provides insights into why autoregressive training is effective and how performance scales with model and data size.

**Comparative Analysis**: Understanding of differences with bidirectional models like BERT reveals the complementary strengths and appropriate use cases for different transformer architectures.

GPT's architecture and training approach are crucial for modern NLP because:
- **Generation Excellence**: Established the foundation for high-quality text generation systems
- **Scalability**: Demonstrated how simple objectives scale to create increasingly capable models
- **Transfer Learning**: Showed how unsupervised pretraining benefits diverse downstream applications
- **Few-Shot Learning**: Revealed emergent capabilities that arise from large-scale autoregressive training
- **Foundation for Progress**: Created the template for modern large language models like GPT-2, GPT-3, and beyond

The principles and techniques covered provide essential knowledge for understanding modern generative language models, implementing autoregressive training systems, and developing applications that leverage the unique capabilities of decoder-based transformer architectures. Understanding these foundations is crucial for working with current generation models and contributing to the ongoing advancement of language generation technology.