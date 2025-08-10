# Day 7.1: Vision Transformers - Architecture and Mathematical Foundations

## Overview
Vision Transformers (ViTs) represent a paradigmatic shift in computer vision, demonstrating that the transformer architecture, originally designed for natural language processing, can achieve state-of-the-art performance on visual tasks without the inductive biases inherent in convolutional neural networks. This transformation has fundamentally challenged the dominance of CNNs and opened new research directions in visual understanding. The mathematical foundations of Vision Transformers encompass attention mechanisms, positional encodings, multi-head self-attention, and the adaptation of sequence modeling to visual data through patch-based representations.

## Mathematical Foundations of Attention Mechanisms

### Self-Attention Fundamentals

**Core Attention Mechanism**
The fundamental attention operation computes a weighted average of values based on the similarity between queries and keys:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix  
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix
- $n$: Sequence length (number of patches)
- $d_k$: Key/Query dimension
- $d_v$: Value dimension

**Scaled Dot-Product Attention Analysis**
The scaling factor $\frac{1}{\sqrt{d_k}}$ is crucial for gradient stability:

$$\text{Var}\left[\frac{q^T k}{\sqrt{d_k}}\right] = \frac{\text{Var}[q^T k]}{d_k} = \frac{d_k \sigma^2}{d_k} = \sigma^2$$

Without scaling, the variance grows with $d_k$, pushing softmax into saturation regions with vanishing gradients.

**Attention Matrix Properties**
The attention matrix $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ satisfies:
- **Row-stochastic**: $\sum_{j=1}^{n} A_{ij} = 1$ for all $i$
- **Non-negative**: $A_{ij} \geq 0$ for all $i, j$
- **Permutation equivariant**: $A(P \cdot X) = P \cdot A(X) \cdot P^T$ for permutation matrix $P$

### Multi-Head Attention Architecture

**Mathematical Formulation**
Multi-head attention applies attention in parallel across different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each attention head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameter Matrices**:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$: Query projection for head $i$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$: Key projection for head $i$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$: Value projection for head $i$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$: Output projection

**Computational Complexity Analysis**
For sequence length $n$ and model dimension $d$:
- **Self-attention**: $O(n^2d + nd^2)$
- **Feed-forward**: $O(nd^2)$
- **Total per layer**: $O(n^2d + nd^2)$

**Memory Complexity**:
- **Attention matrix storage**: $O(hn^2)$ for $h$ heads
- **Activations**: $O(nd)$

### Attention Pattern Analysis

**Information Flow in Self-Attention**
The attention mechanism creates a fully connected graph where information can flow between any pair of positions:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j$$

**Rank Analysis**
The attention matrix typically has low effective rank:
$$\text{rank}(A) \ll n$$

This indicates that attention focuses on a small number of relevant positions.

**Spectral Properties**
The attention matrix eigenvalues $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$ with $\lambda_1 = 1$ reveal attention concentration patterns.

## Vision Transformer Architecture

### Patch Embedding and Tokenization

**Image to Sequence Transformation**
An image $I \in \mathbb{R}^{H \times W \times C}$ is divided into non-overlapping patches:

$$x_p = \text{flatten}(I[i:i+P, j:j+P, :])$$

Where $(i, j) = (pP, qP)$ for patch indices $(p, q)$ and patch size $P \times P$.

**Linear Embedding**
Patches are projected to model dimension:
$$z_0 = [x_{\text{cls}}; E x_{p_1}; E x_{p_2}; ...; E x_{p_N}] + E_{pos}$$

Where:
- $E \in \mathbb{R}^{d \times P^2C}$: Patch embedding matrix
- $x_{\text{cls}}$: Learnable class token
- $E_{pos} \in \mathbb{R}^{(N+1) \times d}$: Position embeddings
- $N = \frac{HW}{P^2}$: Number of patches

**Mathematical Analysis of Patch Size**
The choice of patch size $P$ creates a trade-off:
- **Small patches**: Higher resolution, more tokens, quadratic attention cost
- **Large patches**: Lower resolution, fewer tokens, loss of fine-grained detail

**Effective receptive field**: Each patch initially "sees" only $P \times P$ pixels, but attention enables global interaction.

### Positional Encoding Strategies

**1D Learned Positional Encodings**
Standard ViT uses learnable position embeddings:
$$E_{pos}[i] \in \mathbb{R}^d \text{ for position } i \in \{0, 1, ..., N\}$$

**2D Positional Encodings**
For explicit spatial structure:
$$E_{pos}[i, j] = E_{pos}^{row}[i] + E_{pos}^{col}[j]$$

**Sinusoidal Positional Encodings**
Adapted from NLP transformers:
$$\text{PE}(pos, 2k) = \sin(pos / 10000^{2k/d})$$
$$\text{PE}(pos, 2k+1) = \cos(pos / 10000^{2k/d})$$

**Relative Position Encodings**
Encode relative spatial relationships:
$$A_{ij} = \text{softmax}\left(\frac{(x_i W^Q)(x_j W^K)^T + R_{ij}}{\sqrt{d_k}}\right)$$

Where $R_{ij}$ depends on spatial offset $(i-j)$.

**Rotary Position Embedding (RoPE)**
Multiplicative position encoding that preserves relative positions:
$$q_m = R_{\Theta, m} W^Q x_m, \quad k_n = R_{\Theta, n} W^K x_n$$

Where $R_{\Theta, m}$ is rotation matrix at position $m$.

### Transformer Block Architecture

**Layer Structure**
Each transformer block consists of:
1. **Multi-head self-attention (MSA)** with residual connection and layer norm
2. **Feed-forward network (FFN)** with residual connection and layer norm

**Mathematical Formulation**:
$$z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$$

**Layer Normalization**
$$\text{LN}(x) = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$
- $\gamma, \beta \in \mathbb{R}^d$: Learnable parameters

**Feed-Forward Network**
$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

**GELU Activation**:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$$

**Approximation**: $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$

### Classification Head and Output

**Class Token Processing**
The class token $z_L^{(0)}$ from the final layer is used for classification:
$$y = \text{LN}(z_L^{(0)}) W_{head}$$

Where $W_{head} \in \mathbb{R}^{d \times K}$ for $K$ classes.

**Global Average Pooling Alternative**
$$y = \text{GAP}(z_L^{(1:N)}) W_{head} = \frac{1}{N} \sum_{i=1}^{N} z_L^{(i)} W_{head}$$

## ViT Variants and Improvements

### Hierarchical Vision Transformers

**Swin Transformer Architecture**
Introduces hierarchical representation through:
1. **Patch Merging**: Reduces spatial resolution while increasing channels
2. **Shifted Window Attention**: Limits attention to local windows

**Window-based Multi-Head Self-Attention (W-MSA)**:
For window size $M \times M$:
$$\text{W-MSA}(z) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where attention is computed only within each window.

**Shifted Window Multi-Head Self-Attention (SW-MSA)**:
$$\text{SW-MSA}(z) = \text{W-MSA}(\text{CyclicShift}(z))$$

**Computational Complexity Reduction**:
- **Standard MSA**: $O(n^2d)$ where $n = H \times W$
- **Window MSA**: $O(M^2 \cdot \frac{n}{M^2} \cdot d) = O(nd)$ for fixed window size

**Patch Merging Operation**:
$$\text{PatchMerge}(z) = \text{Linear}(\text{Concat}(z_{0::2,0::2}, z_{1::2,0::2}, z_{0::2,1::2}, z_{1::2,1::2}))$$

### Pyramid Vision Transformer (PVT)

**Multi-Scale Feature Extraction**
PVT introduces spatial reduction attention:
$$\text{SRA}(Q, K, V) = \text{Attention}(Q, \text{Reshape}(K'), \text{Reshape}(V'))$$

Where $K', V' = \text{Conv2D}_{R \times R}(K, V)$ with reduction ratio $R$.

**Progressive Shrinking Strategy**:
- **Stage 1**: $\frac{H}{4} \times \frac{W}{4}$, $C_1$ channels
- **Stage 2**: $\frac{H}{8} \times \frac{W}{8}$, $C_2$ channels  
- **Stage 3**: $\frac{H}{16} \times \frac{W}{16}$, $C_3$ channels
- **Stage 4**: $\frac{H}{32} \times \frac{W}{32}$, $C_4$ channels

### Data-Efficient Image Transformers (DeiT)

**Knowledge Distillation for Vision Transformers**
$$\mathcal{L}_{DeiT} = (1-\alpha)\mathcal{L}_{CE}(y, y_{hard}) + \alpha \mathcal{L}_{KL}(y, y_{teacher})$$

**Distillation Token**
Additional learnable token alongside class token:
$$z_0 = [x_{cls}; x_{dist}; x_{p_1}^E; ...; x_{p_N}^E] + E_{pos}$$

**Hard and Soft Distillation**:
- **Hard**: Use teacher's predicted class as ground truth
- **Soft**: Use teacher's probability distribution

### ConvNeXt and Hybrid Architectures

**ConvNeXt Design Principles**
Modernizing CNNs with transformer-inspired designs:
1. **Patchify stem**: $4 \times 4$ convolution with stride 4
2. **ResNeXt-style blocks**: Depthwise convolutions
3. **Inverted bottleneck**: Expand-then-reduce pattern
4. **Layer normalization**: Replace batch normalization
5. **GELU activation**: Replace ReLU

**ConvNeXt Block**:
$$\begin{align}
y &= \text{DWConv}_{7 \times 7}(x) \\
y &= \text{LN}(y) \\
y &= \text{Conv}_{1 \times 1}(y) \cdot 4 \\
y &= \text{GELU}(y) \\
y &= \text{Conv}_{1 \times 1}(y) \\
y &= x + y
\end{align}$$

**Hybrid CNN-Transformer Models**
Combine convolutional feature extraction with transformer processing:
$$f_{hybrid}(x) = \text{Transformer}(\text{CNN}_{backbone}(x))$$

## Advanced Attention Mechanisms

### Linear Attention Approximations

**Performer Architecture**
Approximate attention using random feature maps:
$$\text{Attention}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)\phi(K)^T \mathbf{1}}$$

Where $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^r$ with $r \ll d$.

**Random Fourier Features**:
$$\phi(x) = \sqrt{\frac{2}{r}}[\cos(\omega_1^T x + b_1), ..., \cos(\omega_r^T x + b_r)]$$

**Computational Complexity**: Reduces from $O(n^2d)$ to $O(nrd)$.

### Sparse Attention Patterns

**Local Attention Windows**
Limit attention to local neighborhoods:
$$A_{ij} = \begin{cases}
\text{attention}(q_i, k_j) & \text{if } |i-j| \leq w \\
0 & \text{otherwise}
\end{cases}$$

**Dilated Attention**
Attention with skip connections:
$$A_{ij} \neq 0 \text{ only if } j \in \{i-dw, i-d(w-1), ..., i, ..., i+dw\}$$

**Block-Sparse Attention**
Attention constrained to predefined blocks:
$$A = \text{BlockDiag}(A_1, A_2, ..., A_B)$$

### Cross-Attention Mechanisms

**Encoder-Decoder Cross-Attention**
$$\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$

**Deformable Attention**
Learnable spatial offsets in attention:
$$A_{ij} = \text{attention}(q_i, k_{j + \Delta p_{ij}})$$

Where $\Delta p_{ij}$ are learned spatial offsets.

## Training Dynamics and Optimization

### ViT Training Challenges

**Optimization Landscape Analysis**
Vision transformers exhibit different optimization properties compared to CNNs:
- **Loss surface**: More non-convex with sharper minima
- **Gradient flow**: Different patterns due to attention mechanism
- **Learning dynamics**: Slower initial convergence, better final performance

**Data Efficiency Challenges**
ViTs require large datasets for effective training:
$$\text{Performance} \propto \log(\text{Dataset Size})$$

**Inductive Bias Comparison**:
- **CNNs**: Translation equivariance, locality, spatial hierarchy
- **ViTs**: Permutation equivariance, global receptive field, learned spatial structure

### Regularization and Data Augmentation

**DropPath (Stochastic Depth)**
Randomly drop entire residual paths during training:
$$x_{l+1} = x_l + \text{Bernoulli}(p) \cdot F(x_l)$$

**CutMix for Vision Transformers**
Combine images and labels:
$$x = \lambda x_A + (1-\lambda) x_B$$
$$y = \lambda y_A + (1-\lambda) y_B$$

**Mixup in Patch Space**
Apply mixup to individual patches rather than entire images.

**Token Dropping**
Randomly drop patch tokens during training:
$$z'_l = \text{RandomDrop}(z_l, p_{drop})$$

### Advanced Training Techniques

**LayerScale**
Scale residual connections with learnable parameters:
$$x_{l+1} = x_l + \gamma_l \cdot F(x_l)$$

Where $\gamma_l$ is initialized to small values.

**Gradient Clipping**
Essential for ViT training stability:
$$g_{clipped} = g \cdot \min\left(1, \frac{\text{clip\_norm}}{||g||_2}\right)$$

**Warmup Learning Rate Schedule**
$$\eta(t) = \begin{cases}
\eta_{base} \frac{t}{T_{warmup}} & \text{if } t \leq T_{warmup} \\
\eta_{base} \cos\left(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi\right) & \text{if } t > T_{warmup}
\end{cases}$$

## Theoretical Analysis and Properties

### Expressivity and Universal Approximation

**Universal Approximation for Vision Transformers**
ViTs with sufficient depth and width can approximate any continuous function on bounded domains:

**Theorem**: For any continuous function $f: [0,1]^{H \times W} \rightarrow \mathbb{R}^K$ and $\epsilon > 0$, there exists a ViT $\mathcal{F}$ such that:
$$\sup_{x \in [0,1]^{H \times W}} ||f(x) - \mathcal{F}(x)|| < \epsilon$$

**Attention as Matrix Factorization**
Self-attention can be viewed as low-rank matrix factorization:
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) = \text{softmax}(XW^QW^{K^T}X^T)$$

**Rank Analysis**:
$$\text{rank}(A) \leq \min(n, d)$$

### Generalization Theory

**PAC-Bayesian Bounds for ViTs**
Generalization error bounded by:
$$\mathbb{E}[L(\mathcal{F})] \leq \hat{L}(\mathcal{F}) + \sqrt{\frac{KL(\mathcal{Q}||\mathcal{P}) + \log(2n/\delta)}{2(n-1)}}$$

Where:
- $\mathcal{Q}$: Posterior distribution over models
- $\mathcal{P}$: Prior distribution
- $KL$: Kullback-Leibler divergence

**Rademacher Complexity**
For ViT function class $\mathcal{F}$:
$$\mathfrak{R}(\mathcal{F}) = \mathbb{E}_{\sigma} \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i f(x_i)$$

**Attention Entropy and Generalization**
Higher attention entropy correlates with better generalization:
$$H(A_i) = -\sum_{j=1}^{n} A_{ij} \log A_{ij}$$

### Interpretability Analysis

**Attention Map Visualization**
Average attention weights across heads and layers:
$$\bar{A}_{ij} = \frac{1}{L \cdot h} \sum_{l=1}^{L} \sum_{h=1}^{h} A_{ij}^{(l,h)}$$

**Attention Distance Analysis**
Average attention distance measures local vs global focus:
$$d_{att} = \sum_{i,j} A_{ij} \cdot ||p_i - p_j||_2$$

**Attention Rollout**
Compute attention flow through the network:
$$A_{roll}^{(l)} = A^{(l)} \cdot A_{roll}^{(l-1)}$$

**CLS Token Attention**
Analyze how class token attends to patches:
$$A_{cls}^{(l)} = A^{(l)}[0, 1:N]$$

## Model Scaling and Efficiency

### ViT Scaling Laws

**Performance vs Model Size**
Empirical scaling relationship:
$$\text{Performance} = \alpha - \beta \cdot \text{Model Size}^{-\gamma}$$

**Compute-Optimal Scaling**
Optimal allocation between model size and training data:
$$C = 6NLd^2 + 12Ld^2n$$

Where:
- $N$: Training tokens
- $L$: Number of layers  
- $d$: Model dimension
- $n$: Sequence length

### Efficient ViT Architectures

**Mobile-ViT**
Combines CNNs and transformers for mobile deployment:
$$y = \text{Conv}(\text{Transformer}(\text{Conv}(x)))$$

**EfficientFormer**
Dimension-consistent pure transformer:
- **MetaFormer block**: Abstract transformer structure
- **Pool former**: Replace attention with pooling
- **ConvFormer**: Replace attention with convolution

**FastViT**  
Efficient training and inference:
- **Structural re-parameterization**: Convert multi-branch to single-branch
- **Token mixing**: Efficient alternatives to self-attention
- **Large kernel convolutions**: Replace some attention layers

### Quantization and Pruning

**Post-Training Quantization**
$$W_{quant} = \text{Round}\left(\frac{W}{\Delta}\right) \cdot \Delta$$

Where $\Delta = \frac{W_{max} - W_{min}}{2^b - 1}$ for $b$-bit quantization.

**Attention-Aware Pruning**
Prune attention heads with low attention entropy:
$$\text{Importance}(h) = H(A^{(h)}) \cdot \text{Gradient}(h)$$

**Knowledge Distillation for Compression**
$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, y_{gt}) + (1-\alpha) \mathcal{L}_{KL}(\sigma(z_s/T), \sigma(z_t/T))$$

## Applications and Downstream Tasks

### Object Detection with ViTs

**DETR (Detection Transformer)**
End-to-end object detection with transformers:
$$\mathcal{L}_{DETR} = \sum_{i=1}^{N} [\alpha \mathcal{L}_{class}(c_i, \hat{c}_{\sigma(i)}) + \mathbf{1}_{\{c_{\sigma(i)} \neq \emptyset\}} \mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})]$$

**Vision Transformer Backbone**
Replace CNN backbone with ViT:
$$\text{Features} = \text{ViT}(x) \rightarrow \text{Detection Head}$$

### Semantic Segmentation

**Segmentation Transformer (SETR)**
Use ViT backbone with decoder:
$$\text{Segmentation} = \text{Decoder}(\text{ViT}_{encoder}(x))$$

**Per-Pixel Classification**
$$p(y_i = c) = \text{softmax}(\text{Linear}(z_i^{(L)}))$$

### Vision-Language Tasks

**CLIP Architecture**
Contrastive learning between vision and text:
$$\mathcal{L}_{CLIP} = -\frac{1}{2}\mathbb{E}[\log \frac{\exp(\text{sim}(v, t_+)/\tau)}{\sum_{t'} \exp(\text{sim}(v, t')/\tau)}]$$

**Multi-Modal Transformers**
Cross-attention between vision and language tokens:
$$z_{VL} = \text{CrossAttn}(z_V, z_L, z_L)$$

## Key Questions for Review

### Architecture and Design
1. **Patch Embedding**: How does the choice of patch size affect the balance between computational efficiency and representational capacity?

2. **Positional Encoding**: What are the trade-offs between different positional encoding strategies in vision transformers?

3. **Attention Patterns**: How do attention patterns in ViTs differ from CNNs in terms of receptive field development?

### Training and Optimization
4. **Data Efficiency**: Why do Vision Transformers require larger datasets compared to CNNs, and how can this be addressed?

5. **Optimization Dynamics**: What are the key differences in training dynamics between ViTs and CNNs?

6. **Regularization**: Which regularization techniques are most effective for Vision Transformers?

### Theoretical Understanding
7. **Inductive Biases**: How do the different inductive biases of ViTs vs CNNs affect their learning and generalization?

8. **Universal Approximation**: What are the theoretical guarantees for Vision Transformers as universal function approximators?

9. **Attention Analysis**: How can attention mechanisms be analyzed to understand what ViTs learn?

### Practical Applications
10. **Model Selection**: When should one choose ViTs over CNNs for different computer vision tasks?

11. **Hybrid Architectures**: What are the benefits of combining convolutional and attention mechanisms?

12. **Efficiency Trade-offs**: How do different ViT variants balance accuracy and computational efficiency?

## Conclusion

Vision Transformers represent a fundamental paradigm shift in computer vision, demonstrating that attention mechanisms can effectively process visual information without the spatial inductive biases of convolutional neural networks. This comprehensive exploration has established:

**Mathematical Foundations**: Deep understanding of self-attention mechanisms, multi-head attention, and their adaptation to visual data provides the theoretical framework for understanding how transformers process images through patch-based sequence modeling.

**Architectural Innovations**: Systematic coverage of ViT variants including hierarchical approaches (Swin), efficient designs (DeiT), and hybrid architectures demonstrates the evolution and adaptation of transformer architectures for visual tasks.

**Training Dynamics**: Understanding of optimization challenges, data efficiency requirements, and specialized training techniques enables effective deployment of ViTs in practical applications.

**Theoretical Analysis**: Exploration of expressivity, generalization bounds, and interpretability provides insight into why and when Vision Transformers excel compared to traditional convolutional approaches.

**Scaling and Efficiency**: Coverage of model scaling laws, efficient architectures, and compression techniques addresses practical deployment considerations for Vision Transformers.

**Application Domains**: Integration with object detection, segmentation, and multi-modal tasks demonstrates the versatility and broad applicability of transformer architectures in computer vision.

Vision Transformers have fundamentally transformed computer vision by:
- **Challenging Convolutional Dominance**: Showing that attention can match or exceed CNN performance
- **Global Receptive Fields**: Enabling global information processing from the first layer
- **Transfer Learning**: Achieving excellent transfer performance across diverse visual tasks
- **Multi-Modal Integration**: Facilitating seamless integration with language and other modalities
- **Architectural Innovation**: Inspiring new hybrid designs combining the best of both approaches

As the field continues to evolve, Vision Transformers remain at the forefront of computer vision research, with ongoing developments in efficiency, theoretical understanding, and novel applications continuing to expand their impact and practical utility.