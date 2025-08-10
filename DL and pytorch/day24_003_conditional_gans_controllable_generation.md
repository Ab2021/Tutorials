# Day 24.3: Conditional GANs and Controllable Generation - Mathematical Foundations of Guided Synthesis

## Overview

Conditional Generative Adversarial Networks represent a fundamental advancement in generative modeling that enables precise control over the generation process through explicit conditioning mechanisms, transforming GANs from systems that produce random samples from learned distributions to sophisticated tools capable of generating specific content according to user-defined constraints, attributes, or input conditions. Understanding the mathematical foundations of conditional generation, from basic class-conditional models and attribute-guided synthesis to advanced controllable generation techniques and disentangled representation learning, reveals how conditioning information can be integrated into adversarial training frameworks to achieve unprecedented levels of control over generated content while maintaining high sample quality. This comprehensive exploration examines the theoretical principles underlying various conditioning strategies, the architectural innovations that enable effective condition integration, the mathematical analysis of controllability and disentanglement, and the advanced techniques for achieving fine-grained control over generation processes across diverse applications from image-to-image translation and style transfer to text-to-image synthesis and interactive content creation.

## Conditional GAN Framework

### Mathematical Formulation of Conditioning

**Conditional Generation Objective**:
Extend the standard GAN framework to include conditioning information:
$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x}|\mathbf{c})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log(1 - D(G(\mathbf{z}|\mathbf{c})))]$$

**Conditional Distributions**:
- **Data Distribution**: $p_{\text{data}}(\mathbf{x}|\mathbf{c})$
- **Generator Distribution**: $p_g(\mathbf{x}|\mathbf{c})$
- **Prior Distribution**: $p(\mathbf{z})$ (typically unconditional)

**Joint vs Conditional Modeling**:
$$p(\mathbf{x}, \mathbf{c}) = p(\mathbf{x}|\mathbf{c})p(\mathbf{c})$$

**Conditional Jensen-Shannon Divergence**:
$$\text{JS}(p_{\text{data}}(\mathbf{x}|\mathbf{c}), p_g(\mathbf{x}|\mathbf{c})) = \frac{1}{2}\text{KL}\left(p_{\text{data}}(\mathbf{x}|\mathbf{c}) \left|\left| \frac{p_{\text{data}}(\mathbf{x}|\mathbf{c}) + p_g(\mathbf{x}|\mathbf{c})}{2}\right.\right) + \frac{1}{2}\text{KL}\left(p_g(\mathbf{x}|\mathbf{c}) \left|\left| \frac{p_{\text{data}}(\mathbf{x}|\mathbf{c}) + p_g(\mathbf{x}|\mathbf{c})}{2}\right.\right)$$

**Conditional Mutual Information**:
$$I(\mathbf{X}; \mathbf{C}) = \mathbb{E}[\log \frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x})p(\mathbf{c})}] = H(\mathbf{X}) - H(\mathbf{X}|\mathbf{C})$$

### Types of Conditioning

**1. Class-Conditional Generation**:
$$\mathbf{c} \in \{1, 2, \ldots, K\}$$ (discrete class labels)
$$G(\mathbf{z}, c) \rightarrow \mathbf{x} \in \text{Class } c$$

**2. Attribute-Conditional Generation**:
$$\mathbf{c} \in \{0, 1\}^A$$ (binary attribute vector)
$$G(\mathbf{z}, \mathbf{c}) \rightarrow \mathbf{x} \text{ with attributes } \mathbf{c}$$

**3. Continuous Conditioning**:
$$\mathbf{c} \in \mathbb{R}^d$$ (continuous control parameters)
$$G(\mathbf{z}, \mathbf{c}) \rightarrow \mathbf{x}$$

**4. Structured Conditioning**:
$$\mathbf{c} = \{\text{layout}, \text{segmentation map}, \text{edge map}, \ldots\}$$

**5. Text Conditioning**:
$$\mathbf{c} = \text{Embed}(\text{``A red car driving on a mountain road''})$$

### Conditioning Integration Strategies

**Concatenation-Based Conditioning**:
$$\mathbf{z}' = [\mathbf{z}; \mathbf{c}]$$
$$G(\mathbf{z}') = G([\mathbf{z}; \mathbf{c}])$$

**Multiplicative Conditioning**:
$$\mathbf{h}' = \mathbf{h} \odot f(\mathbf{c})$$

where $f(\mathbf{c})$ learns condition-specific gating.

**Additive Conditioning**:
$$\mathbf{h}' = \mathbf{h} + g(\mathbf{c})$$

**Normalization-Based Conditioning**:
$$\text{ConditionalBN}(\mathbf{x}, \mathbf{c}) = \gamma(\mathbf{c}) \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta(\mathbf{c})$$

**Attention-Based Conditioning**:
$$\alpha_i = \frac{\exp(\mathbf{c}^T \mathbf{h}_i)}{\sum_j \exp(\mathbf{c}^T \mathbf{h}_j)}$$
$$\mathbf{h}_{\text{attended}} = \sum_i \alpha_i \mathbf{h}_i$$

## Class-Conditional GANs

### AC-GAN (Auxiliary Classifier GAN)

**Architecture**:
Discriminator performs both real/fake classification and class prediction:
$$D(\mathbf{x}) \rightarrow [P(\text{real/fake}), P(\text{class}|\mathbf{x})]$$

**Loss Function**:
$$\mathcal{L}_D = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{cls}}$$
$$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{cls}}$$

**Adversarial Loss**:
$$\mathcal{L}_{\text{adv}} = \mathbb{E}[\log D_{\text{adv}}(\mathbf{x})] + \mathbb{E}[\log(1 - D_{\text{adv}}(G(\mathbf{z}, \mathbf{c})))]$$

**Classification Loss**:
$$\mathcal{L}_{\text{cls}} = \mathbb{E}[\log D_{\text{cls}}(c|\mathbf{x})] + \mathbb{E}[\log D_{\text{cls}}(c|G(\mathbf{z}, \mathbf{c}))]$$

**Mathematical Analysis**:
AC-GAN optimizes:
$$\max_D \mathcal{L}_D = \max_D \mathbb{E}[\log D_{\text{adv}}(\mathbf{x}) + \log D_{\text{cls}}(c|\mathbf{x})] + \mathbb{E}[\log(1-D_{\text{adv}}(G(\mathbf{z},\mathbf{c}))) + \log D_{\text{cls}}(c|G(\mathbf{z},\mathbf{c}))]$$

**Information-Theoretic Perspective**:
AC-GAN maximizes mutual information between generated images and class labels:
$$\max I(G(\mathbf{z}, \mathbf{c}); \mathbf{c})$$

### Projection Discriminator

**Mathematical Framework**:
Instead of concatenating condition with features, use inner product:
$$D(\mathbf{x}, \mathbf{c}) = \mathbf{w}^T \phi(\mathbf{x}) + \mathbf{v}_{\mathbf{c}}^T \phi(\mathbf{x})$$

**Embedding-Based Projection**:
$$\mathbf{v}_{\mathbf{c}} = \mathbf{E} \mathbf{c}$$

where $\mathbf{E} \in \mathbb{R}^{d \times k}$ is learned embedding matrix.

**Advantages**:
- More parameter efficient than concatenation
- Better gradient flow to generator
- Improved sample quality and diversity

**Gradient Analysis**:
$$\frac{\partial D(\mathbf{x}, \mathbf{c})}{\partial \mathbf{x}} = \mathbf{w}^T \frac{\partial \phi(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{v}_{\mathbf{c}}^T \frac{\partial \phi(\mathbf{x})}{\partial \mathbf{x}}$$

The condition-specific term $\mathbf{v}_{\mathbf{c}}$ provides direct gradient signal.

### SAGAN (Self-Attention GAN)

**Self-Attention Mechanism for Class Conditioning**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Class-Conditional Attention**:
Condition attention computation on class information:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q + \mathbf{c}\mathbf{W}_{Q,c}$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K + \mathbf{c}\mathbf{W}_{K,c}$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V$$

**Spectral Normalization**:
$$\mathbf{W}_{\text{SN}} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}$$

where $\sigma(\mathbf{W})$ is spectral norm.

**Two Time-Scale Update Rule**:
$$\alpha_G = 0.0001, \quad \alpha_D = 0.0004$$

## Attribute-Conditional Generation

### AttGAN Architecture

**Attribute Encoder-Decoder Framework**:
$$\mathbf{z}_{\text{att}} = E_{\text{att}}(\mathbf{x})$$
$$\mathbf{x}' = G(\mathbf{z}_{\text{att}}, \mathbf{a}')$$

where $\mathbf{a}'$ is target attribute vector.

**Attribute Classification Loss**:
$$\mathcal{L}_{\text{att}} = \sum_{k=1}^{K} \mathbb{E}[\text{BCE}(C_k(\mathbf{x}), a_k)]$$

**Reconstruction Loss**:
$$\mathcal{L}_{\text{rec}} = \mathbb{E}[||\mathbf{x} - G(E(\mathbf{x}), \mathbf{a})||_1]$$

**Attribute Consistency Loss**:
$$\mathcal{L}_{\text{consistency}} = \mathbb{E}[||\mathbf{a}' - C(G(E(\mathbf{x}), \mathbf{a}'))||_2^2]$$

**Total Objective**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}} + \lambda_{\text{att}} \mathcal{L}_{\text{att}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{consistency}}$$

### StarGAN: Multi-Domain Transfer

**Unified Architecture**:
Single generator for multiple domains:
$$G(\mathbf{x}, \mathbf{c}) \rightarrow \mathbf{x}'$$

where $\mathbf{c}$ encodes target domain information.

**Domain Classification**:
$$D_{\text{cls}}(\mathbf{x}) \rightarrow p(\text{domain}|\mathbf{x})$$

**Cycle Consistency**:
$$\mathcal{L}_{\text{cyc}} = \mathbb{E}[||\mathbf{x} - G(G(\mathbf{x}, \mathbf{c}'), \mathbf{c})||_1]$$

where $\mathbf{c}$ and $\mathbf{c}'$ are original and target domains.

**Complete Loss Function**:
$$\mathcal{L}_D = -\mathbb{E}[\log D_{\text{src}}(\mathbf{x})] - \mathbb{E}[\log(1-D_{\text{src}}(G(\mathbf{x},\mathbf{c})))] + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}^r$$

$$\mathcal{L}_G = -\mathbb{E}[\log D_{\text{src}}(G(\mathbf{x},\mathbf{c}))] + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}^f + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}$$

**Domain Embedding**:
$$\mathbf{c} = \text{Embed}(\text{domain\_id})$$

### STGAN: Fine-Grained Control

**Selective Transfer Units (STUs)**:
$$\text{STU}(\mathbf{f}, \mathbf{a}) = \mathbf{f} + \mathbf{a} \odot \Delta\mathbf{f}$$

where $\Delta\mathbf{f}$ is learned attribute-specific feature modification.

**Attribute Selection Matrix**:
$$\mathbf{M} = \text{sigmoid}(\mathbf{W}_M \mathbf{a})$$

**Selective Feature Update**:
$$\mathbf{f}' = \mathbf{f} + \mathbf{M} \odot \tanh(\mathbf{W}_{\Delta} \mathbf{f})$$

**Difference Attention**:
Focus on regions most relevant to attribute changes:
$$\alpha_{i,j} = \frac{\exp(\mathbf{a}^T \mathbf{f}_{i,j})}{\sum_{m,n} \exp(\mathbf{a}^T \mathbf{f}_{m,n})}$$

## Text-to-Image Generation

### AttnGAN Architecture

**Attention-Driven Generation**:
$$\mathbf{c}_i = \sum_{j=1}^{T} \alpha_{i,j} \mathbf{e}_j$$

where $\alpha_{i,j}$ is attention weight and $\mathbf{e}_j$ are word embeddings.

**Word-Level Attention**:
$$\alpha_{i,j} = \frac{\exp(\mathbf{h}_i^T \mathbf{e}_j)}{\sum_{k=1}^{T} \exp(\mathbf{h}_i^T \mathbf{e}_k)}$$

**Multi-Stage Generation**:
$$\mathbf{I}_0 = G_0(\mathbf{z}, \mathbf{c})$$
$$\mathbf{I}_1 = G_1(\mathbf{I}_0, \mathbf{c}_1)$$
$$\mathbf{I}_2 = G_2(\mathbf{I}_1, \mathbf{c}_2)$$

**DAMSM Loss (Deep Attentional Multimodal Similarity Model)**:
$$\mathcal{L}_{\text{DAMSM}} = \mathcal{L}_{\text{word}} + \mathcal{L}_{\text{sentence}}$$

**Word-Level Loss**:
$$\mathcal{L}_{\text{word}} = -\sum_{i=1}^{N} \log \frac{\exp(\gamma \mathbf{e}_i^T \mathbf{v}_i)}{\sum_{j=1}^{N} \exp(\gamma \mathbf{e}_i^T \mathbf{v}_j)}$$

**Sentence-Level Loss**:
$$\mathcal{L}_{\text{sentence}} = -\sum_{i=1}^{N} \log \frac{\exp(\gamma \mathbf{s}_i^T \mathbf{u}_i)}{\sum_{j=1}^{N} \exp(\gamma \mathbf{s}_i^T \mathbf{u}_j)}$$

### StackGAN Architecture

**Stage I: Low-Resolution Generation**:
$$\mathbf{I}_0 = G_0(\mathbf{z}, \phi(\mathbf{t}))$$

where $\phi(\mathbf{t})$ is text embedding.

**Stage II: High-Resolution Refinement**:
$$\mathbf{I}_1 = G_1(\mathbf{I}_0, \phi(\mathbf{t}), \mathbf{z}_1)$$

**Conditioning Augmentation**:
$$\hat{\mathbf{c}} \sim \mathcal{N}(\mu(\phi(\mathbf{t})), \Sigma(\phi(\mathbf{t})))$$

**KL Regularization**:
$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(\mathcal{N}(\mu(\phi(\mathbf{t})), \Sigma(\phi(\mathbf{t}))) || \mathcal{N}(0, \mathbf{I}))$$

**Benefits**:
- Prevents overfitting to training text
- Adds stochasticity to conditioning
- Improves sample diversity

## Image-to-Image Translation

### Pix2Pix Framework

**Conditional GAN for Paired Data**:
$$G: \{\mathbf{x}, \mathbf{z}\} \rightarrow \mathbf{y}$$
$$D: \{\mathbf{x}, \mathbf{y}\} \rightarrow [0, 1]$$

**U-Net Generator**:
Encoder-decoder with skip connections:
$$\mathbf{E}_i = \text{Encoder}_i(\mathbf{E}_{i-1})$$
$$\mathbf{D}_i = \text{Decoder}_i(\text{Cat}(\mathbf{D}_{i-1}, \mathbf{E}_{L-i}))$$

**PatchGAN Discriminator**:
$$D(\mathbf{x}, \mathbf{y}) = \text{Average of patch-wise predictions}$$

**L1 Reconstruction Loss**:
$$\mathcal{L}_{L1} = \mathbb{E}[||\mathbf{y} - G(\mathbf{x}, \mathbf{z})||_1]$$

**Combined Objective**:
$$\mathcal{L}_{pix2pix} = \mathcal{L}_{GAN} + \lambda \mathcal{L}_{L1}$$

**Mathematical Analysis**:
The L1 term encourages structural similarity:
$$\mathcal{L}_{L1} = \sum_{i,j,c} |y_{i,j,c} - G(\mathbf{x})_{i,j,c}|$$

### CycleGAN: Unpaired Translation

**Cycle Consistency**:
$$\mathcal{L}_{\text{cyc}} = \mathbb{E}[||F(G(\mathbf{x})) - \mathbf{x}||_1] + \mathbb{E}[||G(F(\mathbf{y})) - \mathbf{y}||_1]$$

**Forward and Backward Mappings**:
$$G: X \rightarrow Y$$
$$F: Y \rightarrow X$$

**Dual Discriminators**:
$$D_X: \text{discriminate between } \mathbf{x} \text{ and } F(\mathbf{y})$$
$$D_Y: \text{discriminate between } \mathbf{y} \text{ and } G(\mathbf{x})$$

**Complete Objective**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}$$

**Identity Loss**:
$$\mathcal{L}_{\text{identity}} = \mathbb{E}[||G(\mathbf{y}) - \mathbf{y}||_1] + \mathbb{E}[||F(\mathbf{x}) - \mathbf{x}||_1]$$

**Mathematical Properties**:
Cycle consistency implies approximate bijection:
$$G \circ F \approx \text{Id}_Y, \quad F \circ G \approx \text{Id}_X$$

### UNIT (Unsupervised Image-to-Image Translation)

**Shared Latent Space Assumption**:
$$\mathbf{z}_{X_1} = E_{X_1}(\mathbf{x}_1) \approx \mathbf{z}_{X_2} = E_{X_2}(\mathbf{x}_2)$$

**VAE-GAN Architecture**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}}^{X_1} + \mathcal{L}_{\text{GAN}}^{X_2} + \lambda_0 \mathcal{L}_{\text{VAE}}^{X_1} + \lambda_0 \mathcal{L}_{\text{VAE}}^{X_2} + \lambda_1 \mathcal{L}_{\text{CC}}^{X_1} + \lambda_1 \mathcal{L}_{\text{CC}}^{X_2}$$

**VAE Loss**:
$$\mathcal{L}_{\text{VAE}} = D_{\text{KL}}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})) + \lambda_2 \mathbb{E}[||\mathbf{x} - G(E(\mathbf{x}))||_1]$$

**Cycle Consistency Loss**:
$$\mathcal{L}_{\text{CC}} = \mathbb{E}[||\mathbf{x}_1 - G_1(E_2(G_2(E_1(\mathbf{x}_1))))||_1]$$

## Disentangled Representation Learning

### β-VAE and Disentanglement

**β-VAE Objective**:
$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p(\mathbf{x}|\mathbf{z})] - \beta D_{\text{KL}}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))$$

**Disentanglement Pressure**:
Higher $\beta$ encourages:
- Independent latent factors
- Sparse representations
- Interpretable dimensions

**Information Bottleneck Interpretation**:
$$\min I(\mathbf{X}; \mathbf{Z}) - \beta I(\mathbf{Z}; \mathbf{Y})$$

### InfoGAN Framework

**Mutual Information Maximization**:
$$\min_G \max_D V(D,G) = V(D,G) - \lambda I(\mathbf{c}; G(\mathbf{z}, \mathbf{c}))$$

**Variational Lower Bound**:
$$I(\mathbf{c}; G(\mathbf{z}, \mathbf{c})) \geq \mathbb{E}[\log Q(\mathbf{c}|G(\mathbf{z}, \mathbf{c}))] + H(\mathbf{c})$$

**Auxiliary Network Q**:
$$Q(\mathbf{c}|\mathbf{x}) = \text{softmax}(\text{FC}(D_{\text{features}}(\mathbf{x})))$$

**Information-Theoretic Loss**:
$$\mathcal{L}_I = \mathbb{E}[\log Q(\mathbf{c}|G(\mathbf{z}, \mathbf{c}))]$$

**Code Types**:
- **Categorical**: $c_1 \sim \text{Cat}(K=10, p=0.1)$
- **Continuous**: $c_2, c_3 \sim \text{Uniform}(-1, 1)$

**Semantic Meaning Discovery**:
Different latent codes learn interpretable factors:
- Categorical: digit identity (MNIST)
- Continuous: rotation, width, stroke style

### ControllableGAN

**Semantic-Aware Generation**:
$$\mathbf{z}_{\text{semantic}} = \mathbf{z}_{\text{noise}} + \sum_{i} \alpha_i \mathbf{d}_i$$

where $\mathbf{d}_i$ are semantic directions and $\alpha_i$ are control parameters.

**Direction Discovery**:
$$\mathbf{d}_{\text{attribute}} = \mathbb{E}[\mathbf{z}|\text{attribute}=1] - \mathbb{E}[\mathbf{z}|\text{attribute}=0]$$

**Orthogonalization**:
$$\mathbf{d}_i' = \mathbf{d}_i - \sum_{j<i} \frac{\mathbf{d}_i^T \mathbf{d}_j'}{||\mathbf{d}_j'||^2} \mathbf{d}_j'$$

**Controllability Metrics**:
$$\text{Controllability} = \frac{\text{Attribute Change}}{\text{Overall Change}}$$

## Advanced Conditioning Techniques

### Feature-wise Linear Modulation (FiLM)

**Conditional Normalization**:
$$\text{FiLM}(\mathbf{x}_i, \gamma_i, \beta_i) = \gamma_i \mathbf{x}_i + \beta_i$$

**Condition-Dependent Parameters**:
$$\gamma_i, \beta_i = f_i(\mathbf{c})$$

where $f_i$ is learned condition-to-parameter mapping.

**Layer-Specific Modulation**:
$$\mathbf{h}^{(l)} = \text{FiLM}(\text{Conv}^{(l)}(\mathbf{h}^{(l-1)}), \gamma^{(l)}, \beta^{(l)})$$

**Channel-wise vs Spatial Modulation**:
- **Channel-wise**: $\gamma, \beta \in \mathbb{R}^C$
- **Spatial**: $\gamma, \beta \in \mathbb{R}^{H \times W \times C}$

### Cross-Modal Attention

**Multi-Modal Feature Fusion**:
$$\mathbf{f}_{\text{fused}} = \mathbf{f}_{\text{visual}} + \text{Attention}(\mathbf{f}_{\text{visual}}, \mathbf{f}_{\text{text}})$$

**Cross-Attention Mechanism**:
$$\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}_{\text{vis}} \mathbf{K}_{\text{text}}^T}{\sqrt{d}}\right) \mathbf{V}_{\text{text}}$$

**Co-Attention Networks**:
Parallel attention in both modalities:
$$\mathbf{A}_{\text{vis}} = \text{softmax}(\mathbf{F}_{\text{vis}} \mathbf{F}_{\text{text}}^T)$$
$$\mathbf{A}_{\text{text}} = \text{softmax}(\mathbf{F}_{\text{text}} \mathbf{F}_{\text{vis}}^T)$$

### Hierarchical Conditioning

**Multi-Scale Condition Injection**:
$$\mathbf{h}^{(l)} = \text{Generator}^{(l)}(\mathbf{h}^{(l-1)}, \mathbf{c}^{(l)})$$

**Condition Hierarchy**:
- **Global**: Scene-level attributes
- **Object**: Object-specific properties  
- **Local**: Fine-grained details

**Progressive Conditioning**:
$$\mathbf{c}^{(l)} = \text{Refine}(\mathbf{c}^{(l-1)}, \text{resolution}_l)$$

**Multi-Resolution Consistency**:
$$\mathcal{L}_{\text{consistency}} = \sum_l ||\text{Downsample}(\mathbf{x}) - G^{(l)}(\mathbf{z}, \mathbf{c}^{(l)})||_2^2$$

## Evaluation of Conditional Generation

### Conditional Inception Score

**Class-Conditional IS**:
$$\text{IS}_{\text{conditional}} = \mathbb{E}_c[\text{IS}(G(\mathbf{z}|c))]$$

**Attribute-Conditional IS**:
$$\text{IS}_{\text{attr}} = \exp(\mathbb{E}[\text{KL}(p(y|\mathbf{x}, \mathbf{a}) || p(y|\mathbf{a}))])$$

### Conditional FID

**Class-Specific FID**:
$$\text{FID}_c = ||\boldsymbol{\mu}_{r,c} - \boldsymbol{\mu}_{g,c}||_2^2 + \text{Tr}(\boldsymbol{\Sigma}_{r,c} + \boldsymbol{\Sigma}_{g,c} - 2(\boldsymbol{\Sigma}_{r,c}\boldsymbol{\Sigma}_{g,c})^{1/2})$$

**Multi-Class FID**:
$$\text{mFID} = \frac{1}{K} \sum_{c=1}^{K} \text{FID}_c$$

### Disentanglement Metrics

**Mutual Information Gap (MIG)**:
$$\text{MIG} = \frac{1}{K} \sum_{j=1}^{K} \frac{I(\mathbf{z}_j; \mathbf{v}_{\pi(j)}) - \max_{k \neq \pi(j)} I(\mathbf{z}_j; \mathbf{v}_k)}{H(\mathbf{v}_{\pi(j)}))}$$

**β-VAE Metric**:
$$\text{β-VAE score} = \mathbb{E}[V(\hat{\mathbf{v}}_k, \mathbf{v}_k)]$$

where $V$ measures vote classification accuracy.

**SAP Score (Separated Attribute Predictability)**:
$$\text{SAP} = \frac{1}{K} \sum_{j=1}^{K} \max_i \text{score}_{i,j} - \max_{i' \neq i} \text{score}_{i',j}$$

### Controllability Assessment

**Attribute Manipulation Accuracy**:
$$\text{Accuracy} = \frac{\text{Correct Attribute Changes}}{\text{Total Manipulations}}$$

**Semantic Consistency**:
$$\text{Consistency} = \text{IoU}(\text{Unchanged Regions})$$

**Interpolation Smoothness**:
$$\text{Smoothness} = \frac{1}{T-1} \sum_{t=1}^{T-1} \text{LPIPS}(\mathbf{x}_t, \mathbf{x}_{t+1})$$

## Key Questions for Review

### Theoretical Foundations
1. **Conditioning Theory**: How does conditional GAN training relate to conditional probability estimation and what theoretical guarantees exist?

2. **Information Theory**: What role does mutual information play in controllable generation and disentanglement?

3. **Disentanglement**: What mathematical conditions are necessary and sufficient for achieving disentangled representations?

### Architectural Design
4. **Conditioning Integration**: What are the trade-offs between different methods of integrating conditioning information (concatenation, FiLM, attention)?

5. **Multi-Modal Fusion**: How can cross-modal attention mechanisms effectively combine information from different modalities?

6. **Hierarchical Control**: What architectural principles enable hierarchical and multi-scale controllable generation?

### Training and Optimization
7. **Multi-Task Learning**: How should different loss components be balanced in conditional GAN training?

8. **Cycle Consistency**: What theoretical justification exists for cycle consistency constraints, and when are they necessary?

9. **Information Bottleneck**: How does the information bottleneck principle guide design choices in controllable generation?

### Evaluation and Analysis
10. **Conditional Metrics**: How do evaluation metrics need to be adapted for conditional generation tasks?

11. **Disentanglement Assessment**: What metrics best capture the quality of disentangled representations?

12. **Controllability Measurement**: How can the precision and scope of controllable generation be quantitatively assessed?

### Applications and Practical Considerations
13. **Text-to-Image**: What are the key challenges in aligning textual and visual representations for controllable generation?

14. **Image Translation**: When is paired vs unpaired training more appropriate for image-to-image translation tasks?

15. **Real-World Deployment**: What practical considerations affect the deployment of controllable generation systems?

## Conclusion

Conditional GANs and controllable generation represent a fundamental advancement in generative modeling that transforms the paradigm from random sample generation to precise, user-directed content creation through sophisticated mathematical frameworks that integrate conditioning information, disentangled representation learning, and controllable synthesis mechanisms. The evolution from simple class conditioning to sophisticated multi-modal, hierarchical, and fine-grained control demonstrates how theoretical insights from information theory, representation learning, and optimization can be translated into practical systems that enable unprecedented levels of creative control over generated content.

**Mathematical Sophistication**: The theoretical foundations underlying conditional generation, from mutual information maximization and cycle consistency to disentanglement principles and information bottleneck theory, provide the rigorous mathematical framework necessary for understanding how conditioning information can be effectively integrated into adversarial training while maintaining generation quality and enabling controllable synthesis.

**Architectural Innovation**: The development of sophisticated conditioning mechanisms including FiLM, cross-modal attention, hierarchical control structures, and style-based manipulation demonstrates how architectural innovations can enable increasingly precise and intuitive control over generation processes while maintaining the stability and quality of adversarial training.

**Multi-Modal Integration**: The advancement of text-to-image generation, cross-modal translation, and multi-attribute control illustrates how conditional GANs can effectively bridge different modalities and enable complex, multi-faceted control over generated content through learned associations between diverse types of conditioning information and visual output.

**Evaluation Science**: The development of specialized evaluation metrics for conditional generation, disentanglement assessment, and controllability measurement provides the quantitative tools necessary for rigorous assessment of generative systems while revealing the subtle relationships between different aspects of conditional generation quality and control precision.

**Practical Impact**: These advances have enabled breakthrough applications in creative industries, interactive content creation, data augmentation, and artistic tools, demonstrating how fundamental research in controllable generation translates to systems that augment human creativity and enable novel forms of human-AI collaboration in creative processes.

Understanding conditional GANs and controllable generation provides essential knowledge for researchers and practitioners working in generative AI, computer vision, and human-computer interaction, offering both the theoretical insights necessary for developing next-generation controllable systems and the practical understanding required for deploying sophisticated generative tools in real-world applications. The principles and techniques established in this field continue to drive innovation in AI-assisted creativity and remain highly relevant for emerging challenges in controllable, interpretable, and user-directed artificial intelligence systems.