# Day 9.2: Regularization Techniques Deep Dive - Mathematical Theory and Analysis

## Overview
Regularization techniques represent one of the most crucial and mathematically sophisticated approaches to controlling model complexity and preventing overfitting in deep learning. These methods work by constraining the optimization process, adding penalties to the loss function, or modifying the training procedure to encourage solutions that generalize better to unseen data. The theoretical foundations of regularization draw from diverse mathematical fields including functional analysis, probability theory, information theory, and optimization theory. This comprehensive exploration examines the mathematical principles underlying various regularization techniques, their theoretical properties, interactions with optimization algorithms, and their role in the implicit biases of deep learning systems.

## Mathematical Foundations of Regularization

### Regularization as Constrained Optimization

**Lagrangian Formulation**
Regularization can be viewed as solving a constrained optimization problem:
$$\min_{f} \mathcal{L}(f) \quad \text{subject to} \quad \Omega(f) \leq t$$

Using Lagrange multipliers, this becomes:
$$\min_{f} \mathcal{L}(f) + \lambda \Omega(f)$$

Where $\lambda$ is the Lagrange multiplier (regularization parameter).

**Duality Relationship**
For every value of constraint $t$, there exists a corresponding $\lambda$ such that the solutions are equivalent. This establishes the duality between constrained and penalized formulations.

**KKT Conditions**
The Karush-Kuhn-Tucker conditions for the regularized problem:
$$\nabla \mathcal{L}(f^*) + \lambda \nabla \Omega(f^*) = 0$$
$$\lambda \geq 0, \quad \Omega(f^*) - t \leq 0, \quad \lambda(\Omega(f^*) - t) = 0$$

### Bayesian Perspective on Regularization

**Maximum A Posteriori (MAP) Estimation**
Regularization corresponds to MAP estimation with appropriate priors:
$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} P(D|\theta)P(\theta)$$

Taking negative logarithm:
$$\hat{\theta}_{MAP} = \arg\min_{\theta} -\log P(D|\theta) - \log P(\theta)$$

The second term $-\log P(\theta)$ acts as regularization.

**Prior-Regularizer Correspondences**:
- **Gaussian Prior**: $P(\theta) \propto \exp(-\frac{1}{2\sigma^2}\|\theta\|_2^2) \Rightarrow$ L2 regularization
- **Laplace Prior**: $P(\theta) \propto \exp(-\frac{1}{\tau}|\theta|) \Rightarrow$ L1 regularization
- **Student-t Prior**: Heavy-tailed distributions $\Rightarrow$ Robust regularization

**Hierarchical Bayesian Models**
Automatic relevance determination through hierarchical priors:
$$\theta_i \sim \mathcal{N}(0, \alpha_i^{-1}), \quad \alpha_i \sim \text{Gamma}(a, b)$$

This leads to adaptive regularization where irrelevant parameters are automatically pruned.

### Information-Theoretic Foundations

**Minimum Description Length (MDL)**
Regularization can be interpreted through coding theory:
$$\text{Code Length} = \text{Data Encoding} + \text{Model Encoding}$$
$$\mathcal{L}_{total} = -\log P(D|\theta) - \log P(\theta)$$

**Mutual Information Regularization**
Limit mutual information between parameters and training data:
$$\mathcal{L}_{MI} = \mathcal{L}_{emp} + \beta I(\theta; S)$$

Where $I(\theta; S)$ is mutual information between parameters and training set.

**Rate-Distortion Theory**
Optimal trade-off between compression (regularization) and reconstruction error:
$$R(D) = \min_{P(\hat{X}|X): \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

## Weight Regularization Techniques

### L2 Regularization (Ridge Regression)

**Mathematical Formulation**
$$\mathcal{L}_{L2} = \mathcal{L}_{emp} + \lambda \sum_{i} w_i^2 = \mathcal{L}_{emp} + \lambda \|\mathbf{w}\|_2^2$$

**Gradient Update Rule**:
$$w_{t+1} = w_t - \eta \frac{\partial \mathcal{L}_{emp}}{\partial w_t} - \eta \lambda w_t$$
$$w_{t+1} = (1 - \eta\lambda) w_t - \eta \frac{\partial \mathcal{L}_{emp}}{\partial w_t}$$

The term $(1 - \eta\lambda)$ causes exponential decay of weights.

**Closed-Form Solution (Linear Case)**
For linear models, L2 regularization has analytical solution:
$$\mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

**Spectral Analysis**
The effect of L2 regularization on the spectrum of $X^T X$:
- Eigenvalues: $\tilde{\lambda}_i = \lambda_i + \lambda$
- Condition number: $\kappa = \frac{\lambda_{max} + \lambda}{\lambda_{min} + \lambda}$ (improved)

**Shrinkage Factor**
Each eigenvalue is shrunk by factor:
$$s_i = \frac{\lambda_i}{\lambda_i + \lambda}$$

Smaller eigenvalues (noise) are shrunk more than larger ones (signal).

**Effective Degrees of Freedom**
$$df_{eff} = \text{tr}(X(X^T X + \lambda I)^{-1}X^T) = \sum_{i=1}^p \frac{\lambda_i}{\lambda_i + \lambda}$$

### L1 Regularization (Lasso)

**Mathematical Formulation**
$$\mathcal{L}_{L1} = \mathcal{L}_{emp} + \lambda \sum_{i} |w_i| = \mathcal{L}_{emp} + \lambda \|\mathbf{w}\|_1$$

**Subgradient**
L1 norm is not differentiable at zero, requiring subgradient:
$$\partial |w| = \begin{cases}
\{1\} & \text{if } w > 0 \\
[-1, 1] & \text{if } w = 0 \\
\{-1\} & \text{if } w < 0
\end{cases}$$

**Soft Thresholding**
The proximal operator for L1 regularization:
$$\text{prox}_{\lambda|\cdot|}(w) = \text{sign}(w) \max(|w| - \lambda, 0)$$

**Update Rule**:
$$w_{t+1} = \text{sign}(w_t - \eta \frac{\partial \mathcal{L}_{emp}}{\partial w_t}) \max(|w_t - \eta \frac{\partial \mathcal{L}_{emp}}{\partial w_t}| - \eta\lambda, 0)$$

**Sparsity Properties**
L1 regularization promotes sparsity through geometric constraints:
- L1 ball has sharp corners at axes
- Objective level sets likely intersect at sparse points
- Automatic feature selection occurs

**Theoretical Guarantees**
Under restricted isometry property (RIP), Lasso recovers true sparse solution:
$$\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2 \leq \frac{4\lambda}{\phi} \sqrt{s}$$

Where $s$ is sparsity level and $\phi$ is restricted eigenvalue.

### Elastic Net Regularization

**Mathematical Formulation**
Combines L1 and L2 penalties:
$$\mathcal{L}_{EN} = \mathcal{L}_{emp} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

**Alternative Parameterization**:
$$\mathcal{L}_{EN} = \mathcal{L}_{emp} + \lambda[(1-\alpha)\|\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_1]$$

Where $\alpha \in [0,1]$ controls the mixing ratio.

**Properties**
- **Sparsity**: Like L1, induces sparsity
- **Grouping Effect**: Like L2, selects correlated features together
- **Stability**: More stable than pure L1 for correlated features

**Proximal Operator**
$$\text{prox}_{\lambda EN}(w) = \frac{\text{sign}(w)\max(|w| - \lambda\alpha, 0)}{1 + \lambda(1-\alpha)}$$

### Advanced Weight Regularization

**Group Regularization**
For structured sparsity across groups:
$$\Omega_{group}(\mathbf{w}) = \sum_{g \in G} \sqrt{|g|} \|\mathbf{w}_g\|_2$$

Where $G$ is collection of groups and $|g|$ is group size.

**Fused Regularization**
Encourages smoothness in ordered parameters:
$$\Omega_{fused}(\mathbf{w}) = \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \sum_{i=1}^{p-1} |w_{i+1} - w_i|$$

**Nuclear Norm Regularization**
For matrix parameters, promotes low-rank solutions:
$$\Omega_{nuclear}(W) = \|W\|_* = \sum_{i} \sigma_i(W)$$

Where $\sigma_i$ are singular values of $W$.

**Spectral Regularization**
Constrains spectral properties:
$$\Omega_{spectral}(W) = \rho(W) = \max_i |\lambda_i(W)|$$

Where $\rho(W)$ is spectral radius.

## Dropout and Stochastic Regularization

### Standard Dropout Theory

**Mathematical Model**
During training, multiply each activation by Bernoulli random variable:
$$\tilde{h}_i = \begin{cases}
h_i / p & \text{with probability } p \\
0 & \text{with probability } 1-p
\end{cases}$$

**Expectation and Variance**
$$\mathbb{E}[\tilde{h}_i] = h_i, \quad \text{Var}[\tilde{h}_i] = h_i^2 \frac{1-p}{p}$$

**Noise Injection Perspective**
Dropout can be viewed as adding multiplicative noise:
$$\tilde{h}_i = h_i \cdot \epsilon_i, \quad \epsilon_i \sim \text{Bernoulli}(p)/p$$

**Regularization Effect**
Dropout implicitly regularizes by penalizing co-adaptation:
$$\mathcal{L}_{dropout} \approx \mathcal{L}_{emp} + \lambda \sum_{i,j} w_{ij}^2 \sigma_i^2$$

Where $\sigma_i^2$ is variance of input to unit $i$.

### Advanced Dropout Variants

**DropConnect**
Randomly drop connections instead of activations:
$$y = f((M \circ W)x + b)$$

Where $M$ is binary mask and $\circ$ is element-wise product.

**Spatial Dropout**
For convolutional layers, drop entire feature maps:
$$\tilde{F}_{:,:,k} = \begin{cases}
F_{:,:,k} / p & \text{with probability } p \\
0 & \text{with probability } 1-p
\end{cases}$$

**Variational Dropout**
Use learned dropout rates:
$$\log \alpha_i = \theta_i + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Gaussian Dropout**
Replace binary masks with Gaussian noise:
$$\tilde{h}_i = h_i \cdot (1 + \mathcal{N}(0, \alpha))$$

### Theoretical Analysis of Dropout

**Approximate Inference Interpretation**
Dropout approximates inference in Bayesian neural networks:
$$q(\mathbf{w}) = \prod_{i} q(w_i), \quad q(w_i) = p \delta(w_i - \theta_i) + (1-p) \delta(w_i)$$

**Monte Carlo Integration**
Dropout training performs approximate Bayesian inference:
$$\mathbb{E}_{q(\mathbf{w})}[f(\mathbf{w}, \mathbf{x})] \approx \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{w}_t, \mathbf{x})$$

Where $\mathbf{w}_t$ are sampled weights with dropout.

**Generalization Bound**
PAC-Bayesian bound for dropout:
$$R(f) \leq \hat{R}(f) + \sqrt{\frac{KL(q(\mathbf{w})||p(\mathbf{w})) + \log(2\sqrt{n}/\delta)}{2n}}$$

**Information-Theoretic Analysis**
Dropout limits mutual information between parameters and data:
$$I(W; S) \leq H(W) - H(W|M, S)$$

Where $M$ are dropout masks.

### Batch Effects and Normalization

**Batch Dropout Interaction**
Dropout affects batch statistics in batch normalization:
$$\mu_B = \frac{1}{|B|} \sum_{i \in B} \tilde{x}_i, \quad \sigma_B^2 = \frac{1}{|B|} \sum_{i \in B} (\tilde{x}_i - \mu_B)^2$$

**Variance Scaling**
Need to adjust batch norm variance estimates:
$$\hat{\sigma}_B^2 = \frac{p}{p + (1-p)} \sigma_B^2$$

## Normalization as Regularization

### Batch Normalization Theory

**Mathematical Formulation**
$$\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

Where:
- $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$: Batch mean
- $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$: Batch variance
- $\gamma, \beta$: Learnable parameters

**Regularization Effects**
1. **Noise Injection**: Stochastic batch statistics add noise
2. **Constraint**: Forces activations to have specific statistics
3. **Smoothing**: Reduces sensitivity to parameter changes

**Internal Covariate Shift Hypothesis**
Original motivation (though disputed):
$$\Delta \mu_l = \mathbb{E}[\mu_l^{(t+1)} - \mu_l^{(t)}], \quad \Delta \sigma_l = \mathbb{E}[\sigma_l^{(t+1)} - \sigma_l^{(t)}]$$

BN aims to reduce these shifts.

**Loss Surface Smoothing**
BN makes loss landscape smoother:
$$\|\nabla^2 \mathcal{L}\| \text{ decreases with BN}$$

**Gradient Flow Analysis**
BN affects gradient norms and directions:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial w}$$

The Jacobian $\frac{\partial \hat{x}}{\partial w}$ is modified by normalization.

### Layer Normalization

**Mathematical Formulation**
Normalize across features for each example:
$$\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

**Advantages over Batch Normalization**
- **Batch Independence**: No dependence on batch statistics
- **RNN Compatibility**: Works well with sequential models
- **Inference Consistency**: Same normalization at train and test

**Theoretical Properties**
- **Scale Invariance**: $\text{LN}(\alpha x) = \text{LN}(x)$ for $\alpha > 0$
- **Translation Invariance**: $\text{LN}(x + \beta \mathbf{1}) = \text{LN}(x)$

### Instance and Group Normalization

**Instance Normalization**
Normalize each channel separately:
$$\text{IN}(x_{n,c}) = \gamma_c \frac{x_{n,c} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_c$$

**Group Normalization**
Normalize across groups of channels:
$$\text{GN}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta$$

**Comparative Analysis**
- **Batch Norm**: Good for large batches, poor for small batches
- **Layer Norm**: Batch-independent, good for RNNs
- **Instance Norm**: Good for style transfer, removes style information
- **Group Norm**: Compromise between layer and instance norm

### Spectral Normalization

**Mathematical Foundation**
Constrain spectral norm (largest singular value) of weight matrices:
$$\text{SpectralNorm}(W) = \frac{W}{\sigma(W)}$$

Where $\sigma(W)$ is the spectral norm.

**Power Iteration Method**
Efficient computation of spectral norm:
$$\mathbf{u}^{(t+1)} = \frac{W^T \mathbf{v}^{(t)}}{\|W^T \mathbf{v}^{(t)}\|}, \quad \mathbf{v}^{(t+1)} = \frac{W \mathbf{u}^{(t+1)}}{\|W \mathbf{u}^{(t+1)}\|}$$

**Lipschitz Control**
Spectral normalization controls Lipschitz constant:
$$\|f(x_1) - f(x_2)\| \leq L \|x_1 - x_2\|$$

Where $L$ is bounded by product of spectral norms.

**Applications**
- **GAN Training**: Stabilizes discriminator training
- **Robust Training**: Improves adversarial robustness
- **Optimization**: Better conditioning of optimization landscape

## Advanced Regularization Techniques

### Cutout and Random Erasing

**Cutout**
Randomly mask rectangular regions during training:
$$\tilde{x}_{i,j} = \begin{cases}
0 & \text{if } (i,j) \in \text{masked region} \\
x_{i,j} & \text{otherwise}
\end{cases}$$

**Random Erasing**
More sophisticated masking with different fill strategies:
- **Random Values**: Fill with random noise
- **Mean Values**: Fill with dataset mean
- **Mode-specific**: Different strategies for different data types

**Regularization Mechanism**
Forces model to use multiple features for decisions:
$$P(\text{correct} | \text{partial input}) > P(\text{correct} | \text{single feature})$$

### Mixup and CutMix

**Mixup**
Linear interpolation of inputs and labels:
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**Theoretical Analysis**
Mixup encourages linear behavior between training samples:
$$f(\lambda x_1 + (1-\lambda) x_2) \approx \lambda f(x_1) + (1-\lambda) f(x_2)$$

**Vicinal Risk Minimization**
Mixup implements vicinal risk minimization:
$$R_{vic}(f) = \int \int L(f(x), y) P(x, y | \tilde{x}, \tilde{y}) P(\tilde{x}, \tilde{y}) dx dy d\tilde{x} d\tilde{y}$$

**CutMix**
Spatial mixing instead of linear interpolation:
$$\tilde{x} = M \odot x_A + (1-M) \odot x_B$$
$$\tilde{y} = \lambda y_A + (1-\lambda) y_B$$

Where $M$ is binary mask and $\lambda$ is proportional to masked area.

### Label Smoothing

**Mathematical Formulation**
Replace hard labels with soft distributions:
$$\tilde{y}_k = \begin{cases}
1 - \alpha + \frac{\alpha}{K} & \text{if } k = y \\
\frac{\alpha}{K} & \text{otherwise}
\end{cases}$$

**Cross-Entropy with Label Smoothing**
$$\mathcal{L}_{LS} = -(1-\alpha) \log p_y - \alpha \sum_{k \neq y} \frac{1}{K-1} \log p_k$$

**Regularization Effect**
- **Prevents Overconfidence**: Limits maximum predicted probability
- **Calibration**: Improves probability calibration
- **Generalization**: Reduces overfitting to training labels

**Information-Theoretic Interpretation**
Label smoothing adds entropy regularization:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} - \lambda H(p)$$

Where $H(p) = -\sum_k p_k \log p_k$ is entropy of predictions.

### Consistency Regularization

**Mathematical Framework**
Enforce consistent predictions under data transformations:
$$\mathcal{L}_{consistency} = \mathbb{E}_{x,T} [d(f(x), f(T(x)))]$$

Where $T$ is a transformation and $d$ is distance measure.

**Mean Teacher**
Maintain exponential moving average model:
$$\theta_{teacher}^{(t)} = \alpha \theta_{teacher}^{(t-1)} + (1-\alpha) \theta_{student}^{(t)}$$

**Consistency Loss**:
$$\mathcal{L}_{MT} = \mathbb{E}[||f_{student}(x + \epsilon_1) - f_{teacher}(x + \epsilon_2)||^2]$$

**Virtual Adversarial Training (VAT)**
Use adversarial perturbations for consistency:
$$r_{vadv} = \arg\max_{r: \|r\| \leq \epsilon} KL(p(y|x) || p(y|x+r))$$

**VAT Loss**:
$$\mathcal{L}_{VAT} = KL(p(y|x) || p(y|x + r_{vadv}))$$

## Regularization in Deep Learning Context

### Architecture-Specific Regularization

**Convolutional Networks**
- **Spatial Dropout**: Drop entire feature maps
- **Cutout**: Mask image regions
- **Data Augmentation**: Geometric and photometric transforms

**Recurrent Networks**
- **Variational Dropout**: Consistent masks across time
- **Recurrent Dropout**: Different patterns for different connections
- **Gradient Clipping**: Prevent exploding gradients

**Transformer Networks**
- **Attention Dropout**: Regularize attention weights
- **DropPath**: Randomly skip transformer layers
- **Layer Drop**: Stochastic depth in transformers

### Multi-Task Regularization

**Shared Representation Learning**
$$\mathcal{L}_{total} = \sum_{i=1}^{T} \lambda_i \mathcal{L}_i + \Omega(\theta_{shared})$$

**Cross-Task Regularization**
$$\Omega_{cross} = \sum_{i \neq j} \|f_i(\theta) - f_j(\theta)\|^2$$

**Task-Specific vs Shared Parameters**
Balance between task-specific adaptation and shared regularization.

### Implicit Regularization in SGD

**Edge of Stability**
SGD operates at edge of stability region:
$$\eta \lambda_{max}(H) \approx 2$$

Where $H$ is Hessian of loss function.

**Noise Injection**
Mini-batch gradient noise provides implicit regularization:
$$\nabla \mathcal{L}_{batch} = \nabla \mathcal{L} + \epsilon$$

**Flat Minima Bias**
SGD biases toward flat minima with better generalization properties.

**Learning Rate Effects**
Different learning rates provide different regularization strengths:
- **Large LR**: Strong implicit regularization
- **Small LR**: Weak implicit regularization

## Key Questions for Review

### Mathematical Foundations
1. **Bayesian Interpretation**: How do different regularization techniques correspond to different prior distributions in Bayesian inference?

2. **Optimization Perspective**: What is the relationship between constrained optimization and penalized optimization formulations of regularization?

3. **Information Theory**: How can regularization be understood through information-theoretic principles like MDL and rate-distortion theory?

### Weight Regularization
4. **L1 vs L2**: What are the geometric and analytical differences between L1 and L2 regularization, and when should each be used?

5. **Elastic Net**: How does elastic net combine the benefits of L1 and L2 regularization, and what are its theoretical guarantees?

6. **Spectral Properties**: How do different regularization techniques affect the spectral properties of the optimization landscape?

### Dropout and Stochastic Methods
7. **Dropout Theory**: What is the theoretical justification for dropout as a regularization technique, and how does it relate to Bayesian inference?

8. **Noise Injection**: How do different forms of noise injection (additive, multiplicative, structured) provide regularization benefits?

9. **Batch Effects**: How does dropout interact with batch normalization, and what adjustments are needed?

### Normalization Techniques
10. **Regularization Mechanism**: How do normalization techniques like batch normalization provide implicit regularization?

11. **Trade-offs**: What are the trade-offs between different normalization schemes (batch, layer, instance, group) in terms of regularization effects?

12. **Spectral Control**: How does spectral normalization control model behavior, and when is it most beneficial?

## Conclusion

Regularization techniques represent a sophisticated and mathematically rich approach to controlling model complexity and improving generalization in deep learning. This comprehensive exploration has established:

**Mathematical Foundations**: Deep understanding of regularization through multiple theoretical lenses - optimization theory, Bayesian inference, information theory, and statistical learning theory - provides a complete picture of how and why regularization works.

**Weight Regularization**: Systematic analysis of L1, L2, and elastic net regularization reveals their different geometric properties, optimization characteristics, and application domains, with particular emphasis on sparsity induction and feature selection.

**Stochastic Regularization**: Comprehensive treatment of dropout and its variants demonstrates how stochastic training procedures provide regularization through noise injection and ensemble averaging effects.

**Normalization Methods**: Understanding of batch normalization, layer normalization, and spectral normalization shows how constraining activation and weight statistics provides implicit regularization and optimization benefits.

**Advanced Techniques**: Coverage of modern regularization approaches like mixup, cutout, label smoothing, and consistency regularization reveals how domain-specific knowledge can be incorporated into regularization strategies.

**Deep Learning Integration**: Analysis of how regularization interacts with modern architectures and optimization procedures, including implicit regularization effects of SGD and architecture-specific considerations.

Regularization techniques are essential for practical deep learning because:
- **Generalization**: Enable models to perform well on unseen data
- **Optimization**: Improve training dynamics and convergence properties  
- **Robustness**: Increase model stability and reduce sensitivity to perturbations
- **Interpretability**: Some techniques provide insights into model behavior and feature importance
- **Efficiency**: Allow training of larger models without overfitting

The theoretical frameworks and practical techniques covered provide the foundation for designing effective regularization strategies that are appropriate for specific architectures, datasets, and application domains. Understanding these principles is crucial for developing robust and generalizable deep learning systems.