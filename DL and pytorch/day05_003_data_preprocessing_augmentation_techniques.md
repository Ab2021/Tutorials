# Day 5.3: Data Preprocessing and Augmentation Techniques

## Overview
Data preprocessing and augmentation form the critical foundation of successful machine learning pipelines, transforming raw data into optimized formats that enhance model training, improve generalization, and address dataset limitations. These techniques encompass mathematical transformations, statistical normalization methods, and intelligent data generation strategies that can significantly impact model performance. This comprehensive exploration examines the theoretical foundations, mathematical principles, and practical implementations of preprocessing and augmentation techniques across diverse data modalities.

## Theoretical Foundations of Data Preprocessing

### Mathematical Framework for Data Transformation

**Feature Space Transformation Theory**
Data preprocessing can be formalized as mappings between feature spaces:
$$T: \mathcal{X} \rightarrow \mathcal{X}'$$

Where:
- $\mathcal{X}$: Original feature space
- $\mathcal{X}'$: Transformed feature space
- $T$: Transformation function

**Properties of Good Transformations**:
- **Invertibility**: $T^{-1}$ exists when information preservation is required
- **Stability**: Small changes in input produce small changes in output
- **Computational Efficiency**: $T$ can be computed efficiently for large datasets
- **Statistical Properties**: $T$ improves statistical properties relevant to learning

**Composition of Transformations**:
$$T_{composite} = T_n \circ T_{n-1} \circ \ldots \circ T_2 \circ T_1$$

The order of composition matters for non-commutative transformations.

**Information Theory Perspective**:
Preprocessing should maximize relevant information while minimizing noise:
$$I(X'; Y) \geq I(X; Y)$$

Where $X'$ is transformed data, $X$ is original data, and $Y$ is target variable.

### Statistical Foundations

**Central Limit Theorem Applications**:
Many preprocessing techniques assume normality or rely on CLT:
- **Standardization**: Transforms data to approximate standard normal distribution
- **Whitening**: Removes correlation and normalizes variance
- **Quantile Transformation**: Maps to uniform or normal distributions

**Bias-Variance Decomposition in Preprocessing**:
$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Preprocessing can affect both bias and variance:
- **Smoothing operations**: Reduce variance but may increase bias
- **Sharpening operations**: May reduce bias but increase variance

**Robustness Theory**:
Robust preprocessing methods minimize impact of outliers:
- **Breakdown Point**: Fraction of contaminated data a method can handle
- **Influence Function**: How much a single outlier affects the result
- **M-estimators**: Robust statistical estimators used in preprocessing

## Fundamental Preprocessing Techniques

### Normalization and Standardization

**Z-Score Standardization**:
$$z = \frac{x - \mu}{\sigma}$$

**Mathematical Properties**:
- **Mean**: $E[Z] = 0$
- **Variance**: $\text{Var}(Z) = 1$
- **Linear Transformation**: Preserves linear relationships
- **Outlier Sensitivity**: Affected by extreme values

**Implementation Considerations**:
```python
class StandardScaler:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Compute mean and standard deviation from training data"""
        self.mean = torch.mean(X, dim=0, keepdim=True)
        self.std = torch.std(X, dim=0, keepdim=True, unbiased=False)
        # Add epsilon for numerical stability
        self.std = torch.clamp(self.std, min=self.epsilon)
        return self
    
    def transform(self, X):
        """Apply standardization"""
        if self.mean is None or self.std is None:
            raise ValueError("Must fit scaler before transform")
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X_scaled):
        """Reverse the standardization"""
        return X_scaled * self.std + self.mean
```

**Min-Max Normalization**:
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Properties**:
- **Range**: $[0, 1]$ by default, can be scaled to $[a, b]$
- **Preservation**: Maintains relative distances within original range
- **Outlier Impact**: Sensitive to outliers in determining range

**Robust Scaling**:
Uses median and interquartile range instead of mean and standard deviation:
$$x_{robust} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

Where $\text{IQR}(x) = Q_3(x) - Q_1(x)$.

**Max Absolute Scaling**:
$$x_{scaled} = \frac{x}{\max(|x|)}$$

Preserves sparsity and doesn't shift the data.

### Quantile Transformations

**Uniform Quantile Transformation**:
Maps data to uniform distribution using empirical cumulative distribution function:
$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{X_i \leq x}$$

Transformed value: $U = F_n(X)$ where $U \sim \text{Uniform}(0,1)$

**Normal Quantile Transformation** (Rank-based Inverse Normal):
$$Y = \Phi^{-1}\left(\frac{\text{rank}(X) - 0.5}{n}\right)$$

Where $\Phi^{-1}$ is the inverse standard normal CDF.

**Power Transformations**:

**Box-Cox Transformation**:
$$y(\lambda) = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

**Yeo-Johnson Transformation** (handles negative values):
$$y(\lambda) = \begin{cases}
\frac{(x+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \geq 0 \\
\log(x+1) & \text{if } \lambda = 0, x \geq 0 \\
-\frac{(-x+1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, x < 0 \\
-\log(-x+1) & \text{if } \lambda = 2, x < 0
\end{cases}$$

### Dimensionality Reduction and Feature Extraction

**Principal Component Analysis (PCA)**:
Finds orthogonal directions of maximum variance:

**Mathematical Formulation**:
Given data matrix $X \in \mathbb{R}^{n \times d}$, compute covariance matrix:
$$C = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

Eigendecomposition: $C = V\Lambda V^T$

Principal components: columns of $V$ corresponding to largest eigenvalues in $\Lambda$

**Dimensionality Reduction**:
$$X_{reduced} = (X - \bar{X})V_k$$

Where $V_k$ contains first $k$ eigenvectors.

**Variance Explained**:
$$\text{Variance Explained} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**Independent Component Analysis (ICA)**:
Assumes data is linear mixture of independent sources:
$$X = AS$$

Where:
- $A$: Mixing matrix
- $S$: Source signals (independent components)

Goal: Find unmixing matrix $W$ such that $Y = WX \approx S$

**Objective**: Maximize non-Gaussianity of components using measures like:
- **Kurtosis**: $\text{kurt}(Y) = E[Y^4] - 3(E[Y^2])^2$
- **Negentropy**: $J(Y) = H(Y_{Gauss}) - H(Y)$

**Linear Discriminant Analysis (LDA)**:
Finds projection that maximizes class separability:

**Objective Function**:
$$J(W) = \frac{W^T S_B W}{W^T S_W W}$$

Where:
- $S_B$: Between-class scatter matrix
- $S_W$: Within-class scatter matrix

**Between-class Scatter**:
$$S_B = \sum_{i=1}^{c} n_i (\mu_i - \mu)(\mu_i - \mu)^T$$

**Within-class Scatter**:
$$S_W = \sum_{i=1}^{c} \sum_{x \in X_i} (x - \mu_i)(x - \mu_i)^T$$

## Advanced Preprocessing Techniques

### Outlier Detection and Handling

**Statistical Methods**:

**Z-Score Method**:
$$z = \frac{x - \mu}{\sigma}$$
Outliers: $|z| > \text{threshold}$ (typically 2.5 or 3)

**Interquartile Range (IQR) Method**:
- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$
- Outliers: Values beyond fences

**Modified Z-Score** (uses median):
$$M_i = \frac{0.6745(x_i - \text{median})}{\text{MAD}}$$

Where MAD is Median Absolute Deviation:
$$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$

**Isolation Forest**:
Uses random partitioning to isolate outliers:
- Outliers are easier to isolate (require fewer splits)
- Anomaly score based on average path length in isolation trees

**Local Outlier Factor (LOF)**:
Measures local density deviation:
$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}}{|N_k(A)|}$$

Where $\text{lrd}_k$ is local reachability density.

**One-Class SVM**:
Learns decision boundary around normal data:
- Maps data to high-dimensional space
- Finds hyperplane separating data from origin
- Points far from hyperplane are outliers

### Missing Data Imputation

**Statistical Imputation Methods**:

**Mean/Median/Mode Imputation**:
$$x_{missing} = \begin{cases}
\bar{x} & \text{for continuous variables (mean)} \\
\text{median}(x) & \text{for skewed distributions} \\
\text{mode}(x) & \text{for categorical variables}
\end{cases}$$

**Forward/Backward Fill** (for time series):
- Forward fill: Use last observed value
- Backward fill: Use next observed value
- Linear interpolation: $x_t = x_{t-1} + \frac{t-t_{-1}}{t_{+1}-t_{-1}}(x_{t+1} - x_{t-1})$

**Advanced Imputation Methods**:

**K-Nearest Neighbors (KNN) Imputation**:
$$\hat{x}_i = \frac{\sum_{j \in \text{KNN}(i)} w_{ij} x_j}{\sum_{j \in \text{KNN}(i)} w_{ij}}$$

Where $w_{ij} = \frac{1}{d(x_i, x_j) + \epsilon}$ is distance-based weight.

**Multiple Imputation by Chained Equations (MICE)**:
Iterative process:
1. For each variable with missing values, fit regression model using other variables
2. Predict missing values using fitted model
3. Repeat until convergence

**Matrix Factorization Imputation**:
Assume low-rank structure: $X \approx UV^T$
Minimize: $\sum_{(i,j) \in \Omega} (X_{ij} - (UV^T)_{ij})^2 + \lambda(||U||_F^2 + ||V||_F^2)$

Where $\Omega$ is set of observed entries.

### Feature Engineering and Selection

**Polynomial Features**:
Generate interaction terms and higher-order features:
$$\phi(x_1, x_2) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]$$

For degree $d$ and $n$ features, number of polynomial features:
$$\binom{n + d}{d} = \frac{(n + d)!}{n! \cdot d!}$$

**Binning/Discretization**:

**Equal-width Binning**:
$$\text{bin}(x) = \left\lfloor \frac{x - x_{min}}{(x_{max} - x_{min})/k} \right\rfloor$$

**Equal-frequency Binning**:
Create bins with approximately equal number of samples.

**Optimal Binning** (using entropy):
Minimize information loss: $H(Y|X_{binned})$

**Feature Selection Methods**:

**Filter Methods**:
- **Correlation**: Remove highly correlated features
- **Mutual Information**: $I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$
- **Chi-square Test**: For categorical variables
- **ANOVA F-test**: For continuous variables

**Wrapper Methods**:
- **Forward Selection**: Start with empty set, add features
- **Backward Elimination**: Start with all features, remove features
- **Recursive Feature Elimination**: Iteratively remove least important features

**Embedded Methods**:
- **L1 Regularization** (Lasso): $\min_w \frac{1}{2}||Xw - y||_2^2 + \alpha||w||_1$
- **Tree-based Feature Importance**: Use feature importance from random forests
- **Elastic Net**: Combines L1 and L2: $\alpha \rho ||w||_1 + \alpha(1-\rho)||w||_2^2$

## Data Augmentation Theory and Techniques

### Mathematical Framework of Data Augmentation

**Augmentation as Distribution Expansion**:
Given training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ from distribution $P(x,y)$, augmentation creates:
$$\mathcal{D}_{aug} = \{(T_j(x_i), y_i)\}_{i=1,j=1}^{n,m}$$

Where $T_j$ are augmentation transformations.

**Invariance and Equivariance**:
- **Invariance**: $f(T(x)) = f(x)$ (label unchanged)
- **Equivariance**: $f(T(x)) = T'(f(x))$ (label transforms accordingly)

**Data Augmentation Objectives**:
1. **Increase effective dataset size**: $|\mathcal{D}_{aug}| > |\mathcal{D}|$
2. **Improve generalization**: Reduce overfitting
3. **Enforce prior knowledge**: Encode known invariances
4. **Balance class distribution**: Address class imbalance

**Theoretical Guarantees**:
Under certain conditions, augmentation provably improves generalization bounds:
$$\text{Test Error} \leq \text{Training Error} + O\left(\sqrt{\frac{\log|\mathcal{H}|}{n_{eff}}}\right)$$

Where $n_{eff}$ is effective sample size after augmentation.

### Image Augmentation Techniques

**Geometric Transformations**:

**Rotation**:
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Rotated coordinates: $(x', y') = R(\theta)(x, y)$

**Translation**:
$$T(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

**Scaling**:
$$S(s_x, s_y) = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Shearing**:
$$H(h_x, h_y) = \begin{bmatrix} 1 & h_x & 0 \\ h_y & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Affine Transformations** (combination of above):
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**Photometric Transformations**:

**Brightness Adjustment**:
$$I'(x,y) = I(x,y) + \beta$$

**Contrast Adjustment**:
$$I'(x,y) = \alpha \cdot I(x,y)$$

**Gamma Correction**:
$$I'(x,y) = I(x,y)^\gamma$$

**Histogram Equalization**:
$$I'(x,y) = \text{round}\left(\frac{L-1}{n \cdot m} \sum_{k=0}^{I(x,y)} h(k)\right)$$

Where $h(k)$ is histogram of intensity $k$.

**Color Space Transformations**:

**Hue, Saturation, Value (HSV) Adjustments**:
- Convert RGB to HSV
- Modify H, S, V channels independently
- Convert back to RGB

**Channel Shuffling**:
Randomly permute color channels: $(R, G, B) \rightarrow (G, B, R)$, etc.

**Advanced Augmentation Methods**:

**Mixup**:
Create synthetic samples by linear interpolation:
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**CutMix**:
Combine regions from different images:
$$\tilde{x} = M \odot x_A + (1-M) \odot x_B$$
$$\tilde{y} = \lambda y_A + (1-\lambda) y_B$$

Where $M$ is binary mask and $\lambda = \frac{\text{Area of mask}}{H \times W}$.

**CutOut/Random Erasing**:
Randomly mask rectangular regions:
$$I'(x,y) = \begin{cases}
0 & \text{if } (x,y) \in \text{masked region} \\
I(x,y) & \text{otherwise}
\end{cases}$$

**AutoAugment**:
Use reinforcement learning to find optimal augmentation policies:
- Search space: combinations of transformations and their magnitudes
- Reward: validation accuracy on target task
- Policy: sequence of transformations with probabilities

### Text Data Augmentation

**Lexical Augmentation**:

**Synonym Replacement**:
Replace words with synonyms from WordNet or word embeddings:
$$w_i \rightarrow \text{synonym}(w_i) \text{ with probability } p$$

**Random Insertion**:
Insert random synonyms at random positions:
$$S = w_1, w_2, \ldots, w_n \rightarrow w_1, \ldots, w_k, \text{synonym}(w_j), w_{k+1}, \ldots, w_n$$

**Random Swap**:
Swap positions of two random words:
$$w_i \leftrightarrow w_j \text{ for random } i, j$$

**Random Deletion**:
Delete words with probability $p$:
$$w_i \rightarrow \emptyset \text{ with probability } p$$

**Syntactic Augmentation**:

**Dependency Tree Manipulation**:
- Parse sentence into dependency tree
- Apply transformations preserving grammatical structure
- Generate new sentences from modified trees

**Template-based Generation**:
- Extract syntactic templates
- Fill templates with different entities/words
- Maintain semantic relationships

**Semantic Augmentation**:

**Back Translation**:
1. Translate text to intermediate language
2. Translate back to original language
3. Results in paraphrases with same meaning

**Contextual Word Replacement**:
Use pre-trained language models (BERT, GPT) to suggest replacements:
$$P(w_i | w_1, \ldots, w_{i-1}, w_{i+1}, \ldots, w_n)$$

**Paraphrasing Models**:
Train sequence-to-sequence models to generate paraphrases:
$$P(\text{paraphrase} | \text{original sentence})$$

### Time Series Augmentation

**Temporal Transformations**:

**Time Warping**:
Non-linear time axis transformation:
$$t' = f(t) \text{ where } f \text{ is monotonic}$$

Common warping functions:
- **Linear scaling**: $f(t) = at + b$
- **Polynomial**: $f(t) = at^2 + bt + c$
- **Gaussian warping**: Local time distortions

**Window Slicing**:
Extract random subsequences:
$$X_{sub} = X[i:i+w] \text{ for random } i$$

**Jittering**:
Add random noise:
$$X'(t) = X(t) + \mathcal{N}(0, \sigma^2)$$

**Magnitude Transformations**:

**Scaling**:
$$X'(t) = \alpha \cdot X(t)$$

**Magnitude Warping**:
Smooth random variations in magnitude:
$$X'(t) = X(t) \cdot (1 + \mathcal{G}(t))$$

Where $\mathcal{G}(t)$ is smooth random function.

**Frequency Domain Augmentation**:

**Spectral Analysis**:
Apply augmentations in frequency domain:
1. Compute FFT: $\mathcal{F}\{X(t)\} = X(f)$
2. Modify spectrum: $X'(f) = T(X(f))$
3. Inverse FFT: $X'(t) = \mathcal{F}^{-1}\{X'(f)\}$

**Frequency Masking**:
Zero out random frequency bands:
$$X'(f) = \begin{cases}
0 & \text{if } f \in \text{masked bands} \\
X(f) & \text{otherwise}
\end{cases}$$

### Advanced Augmentation Strategies

**Adversarial Training and Augmentation**:

**Fast Gradient Sign Method (FGSM)**:
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y; \theta))$$

**Projected Gradient Descent (PGD)**:
Iterative refinement of adversarial examples:
$$x_{adv}^{(t+1)} = \Pi_{||x-x_{adv}||_\infty \leq \epsilon} (x_{adv}^{(t)} + \alpha \cdot \text{sign}(\nabla_x L(x_{adv}^{(t)}, y; \theta)))$$

**Generative Augmentation**:

**Variational Autoencoders (VAE)**:
Learn latent representation and generate new samples:
$$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$$
$$p(x|z) = \text{decoder}(z)$$

New samples: $x_{new} = \text{decoder}(z_{sampled})$ where $z_{sampled} \sim p(z)$

**Generative Adversarial Networks (GANs)**:
Generate realistic synthetic samples:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Meta-Learning for Augmentation**:

**AutoAugment Family**:
- **AutoAugment**: RL-based policy search
- **RandAugment**: Simplified uniform sampling
- **TrivialAugment**: Single transformation per sample

**Population Based Augmentation (PBA)**:
Evolve augmentation policies during training:
- Population of augmentation policies
- Evolutionary selection based on validation performance
- Adaptation throughout training process

## Preprocessing Pipeline Design and Implementation

### Pipeline Architecture Patterns

**Scikit-learn Style Pipeline**:
```python
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    """Base class for all preprocessing transforms"""
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit transform parameters to training data"""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Apply transformation to data"""
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

class Pipeline:
    """Sequential application of transforms"""
    
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = dict(steps)
    
    def fit(self, X, y=None):
        """Fit all transforms in sequence"""
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.fit_transform(Xt, y)
        
        # Fit final step
        self.steps[-1][1].fit(Xt, y)
        return self
    
    def transform(self, X):
        """Apply all transforms in sequence"""
        Xt = X
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
        return Xt
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
```

**PyTorch Transform Composition**:
```python
class Compose:
    """Compose several transforms together"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string

# Usage example
transform_pipeline = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], 
             std=[0.229, 0.224, 0.225])
])
```

**Functional vs. Object-Oriented Transforms**:

**Functional Style**:
```python
def normalize_image(image, mean, std):
    """Functional normalization"""
    return (image - mean) / std

def random_crop(image, size, padding=None):
    """Functional random crop"""
    if padding:
        image = F.pad(image, padding)
    
    h, w = image.shape[-2:]
    th, tw = size
    
    i = torch.randint(0, h - th + 1, (1,)).item()
    j = torch.randint(0, w - tw + 1, (1,)).item()
    
    return image[..., i:i+th, j:j+tw]
```

**Object-Oriented Style**:
```python
class RandomCrop:
    """Random crop transform"""
    
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding
    
    def __call__(self, image):
        return random_crop(image, self.size, self.padding)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
```

### Advanced Pipeline Features

**Conditional Transforms**:
```python
class ConditionalTransform:
    """Apply transform based on condition"""
    
    def __init__(self, condition_fn, transform, else_transform=None):
        self.condition_fn = condition_fn
        self.transform = transform
        self.else_transform = else_transform or (lambda x: x)
    
    def __call__(self, data):
        if self.condition_fn(data):
            return self.transform(data)
        else:
            return self.else_transform(data)

# Example: Apply strong augmentation to training data only
conditional_augment = ConditionalTransform(
    condition_fn=lambda data: data.get('is_training', False),
    transform=StrongAugmentation(),
    else_transform=WeakAugmentation()
)
```

**Probabilistic Transforms**:
```python
class RandomApply:
    """Apply transform with given probability"""
    
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
    
    def __call__(self, data):
        if torch.rand(1) < self.p:
            return self.transform(data)
        return data

class RandomChoice:
    """Randomly choose one transform from list"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        transform = random.choice(self.transforms)
        return transform(data)
```

**Parametric Transforms with Scheduling**:
```python
class ScheduledTransform:
    """Transform with parameters that change over time"""
    
    def __init__(self, transform_class, param_schedule):
        self.transform_class = transform_class
        self.param_schedule = param_schedule
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __call__(self, data):
        params = self.param_schedule(self.epoch)
        transform = self.transform_class(**params)
        return transform(data)

# Example: Gradually increase augmentation strength
def augmentation_schedule(epoch):
    # Increase strength over first 100 epochs
    strength = min(1.0, epoch / 100.0)
    return {
        'rotation_range': strength * 30,  # Up to 30 degrees
        'brightness_range': strength * 0.3,  # Up to ±30%
        'noise_std': strength * 0.1  # Up to 10% noise
    }
```

## Domain-Specific Preprocessing Strategies

### Computer Vision Preprocessing

**Image Format Handling**:
```python
class ImageLoader:
    """Robust image loading with format handling"""
    
    def __init__(self, backend='PIL', color_mode='RGB'):
        self.backend = backend
        self.color_mode = color_mode
    
    def __call__(self, path):
        try:
            if self.backend == 'PIL':
                from PIL import Image
                image = Image.open(path).convert(self.color_mode)
                return torch.from_numpy(np.array(image)).permute(2, 0, 1)
            
            elif self.backend == 'cv2':
                import cv2
                image = cv2.imread(path)
                if self.color_mode == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(image).permute(2, 0, 1)
            
        except Exception as e:
            # Fallback to dummy image
            print(f"Failed to load {path}: {e}")
            return torch.zeros(3, 224, 224)
```

**Multi-scale Processing**:
```python
class MultiScaleResize:
    """Resize to multiple scales for multi-scale training"""
    
    def __init__(self, scales=[224, 256, 288, 320]):
        self.scales = scales
    
    def __call__(self, image):
        scale = random.choice(self.scales)
        return F.interpolate(image.unsqueeze(0), size=(scale, scale), 
                           mode='bilinear', align_corners=False).squeeze(0)
```

**Aspect Ratio Preservation**:
```python
class AspectRatioResize:
    """Resize while preserving aspect ratio"""
    
    def __init__(self, max_size=512, pad_value=0):
        self.max_size = max_size
        self.pad_value = pad_value
    
    def __call__(self, image):
        c, h, w = image.shape
        max_dim = max(h, w)
        
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), 
                                mode='bilinear', align_corners=False).squeeze(0)
        else:
            new_h, new_w = h, w
        
        # Pad to square
        pad_h = (self.max_size - new_h) // 2
        pad_w = (self.max_size - new_w) // 2
        
        padding = [pad_w, pad_w, pad_h, pad_h]
        return F.pad(image, padding, value=self.pad_value)
```

### Natural Language Processing Preprocessing

**Comprehensive Text Preprocessing**:
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self, 
                 lowercase=True,
                 remove_punct=True,
                 remove_stopwords=True,
                 remove_numbers=False,
                 stemming=False,
                 lemmatization=True,
                 min_token_length=2):
        
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_token_length = min_token_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (keep spaces and basic punctuation)
        if self.remove_punct:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize_and_process(self, text):
        """Tokenize and apply word-level processing"""
        tokens = word_tokenize(text)
        
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Convert to lowercase
            if self.lowercase:
                token = token.lower()
            
            # Remove stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming or lemmatization
            if self.stemming:
                token = self.stemmer.stem(token)
            elif self.lemmatization:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def __call__(self, text):
        """Apply full preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_process(cleaned)
        return ' '.join(tokens)
```

**Sequence Preprocessing**:
```python
class SequencePreprocessor:
    """Preprocessing for sequence data"""
    
    def __init__(self, 
                 vocab=None,
                 max_length=512,
                 padding_token='<PAD>',
                 unknown_token='<UNK>',
                 start_token='<START>',
                 end_token='<END>'):
        
        self.vocab = vocab or {}
        self.max_length = max_length
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Special token indices
        self.special_tokens = {
            padding_token: 0,
            unknown_token: 1,
            start_token: 2,
            end_token: 3
        }
        
        # Merge with vocabulary
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def text_to_sequence(self, text, add_special_tokens=True):
        """Convert text to sequence of token indices"""
        tokens = text.split()
        
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
        
        # Convert to indices
        sequence = [self.vocab.get(token, self.vocab[self.unknown_token]) 
                   for token in tokens]
        
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length-1] + [self.vocab[self.end_token]]
        
        # Pad if too short
        while len(sequence) < self.max_length:
            sequence.append(self.vocab[self.padding_token])
        
        return torch.tensor(sequence, dtype=torch.long)
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        tokens = [self.inverse_vocab.get(idx.item(), self.unknown_token) 
                 for idx in sequence]
        
        # Remove padding and special tokens
        filtered_tokens = []
        for token in tokens:
            if token == self.padding_token:
                break
            if token not in [self.start_token, self.end_token]:
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
```

### Time Series Preprocessing

**Temporal Feature Engineering**:
```python
class TimeSeriesFeatureExtractor:
    """Extract temporal features from time series data"""
    
    def __init__(self, include_datetime_features=True,
                 include_lag_features=True,
                 include_rolling_features=True,
                 lag_periods=[1, 7, 30],
                 rolling_windows=[7, 30, 90]):
        
        self.include_datetime_features = include_datetime_features
        self.include_lag_features = include_lag_features
        self.include_rolling_features = include_rolling_features
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
    
    def extract_datetime_features(self, df, datetime_col='timestamp'):
        """Extract features from datetime column"""
        dt = pd.to_datetime(df[datetime_col])
        
        features = {
            'year': dt.dt.year,
            'month': dt.dt.month,
            'day': dt.dt.day,
            'dayofweek': dt.dt.dayofweek,
            'hour': dt.dt.hour,
            'minute': dt.dt.minute,
            'quarter': dt.dt.quarter,
            'is_weekend': dt.dt.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': dt.dt.is_month_start.astype(int),
            'is_month_end': dt.dt.is_month_end.astype(int),
        }
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_lag_features(self, series, target_col):
        """Extract lagged values"""
        features = {}
        for lag in self.lag_periods:
            features[f'{target_col}_lag_{lag}'] = series.shift(lag)
        
        return pd.DataFrame(features, index=series.index)
    
    def extract_rolling_features(self, series, target_col):
        """Extract rolling window statistics"""
        features = {}
        
        for window in self.rolling_windows:
            rolling = series.rolling(window=window, min_periods=1)
            features.update({
                f'{target_col}_rolling_mean_{window}': rolling.mean(),
                f'{target_col}_rolling_std_{window}': rolling.std(),
                f'{target_col}_rolling_min_{window}': rolling.min(),
                f'{target_col}_rolling_max_{window}': rolling.max(),
                f'{target_col}_rolling_median_{window}': rolling.median(),
            })
        
        return pd.DataFrame(features, index=series.index)
    
    def __call__(self, df, target_col, datetime_col='timestamp'):
        """Extract all temporal features"""
        feature_dfs = [df]
        
        if self.include_datetime_features:
            dt_features = self.extract_datetime_features(df, datetime_col)
            feature_dfs.append(dt_features)
        
        if self.include_lag_features:
            lag_features = self.extract_lag_features(df[target_col], target_col)
            feature_dfs.append(lag_features)
        
        if self.include_rolling_features:
            rolling_features = self.extract_rolling_features(df[target_col], target_col)
            feature_dfs.append(rolling_features)
        
        return pd.concat(feature_dfs, axis=1)
```

## Key Questions for Review

### Theoretical Foundations
1. **Transformation Theory**: How do mathematical transformations affect the statistical properties of data, and when is information preserved vs. lost?

2. **Normalization Methods**: What are the theoretical differences between various normalization techniques, and when is each appropriate?

3. **Augmentation Mathematics**: How can data augmentation be formalized mathematically, and what are the theoretical guarantees for improved generalization?

### Preprocessing Techniques
4. **Outlier Detection**: What are the assumptions and limitations of different outlier detection methods, and how do they affect downstream learning?

5. **Missing Data**: How do different imputation methods bias the data distribution, and when is each method theoretically justified?

6. **Feature Selection**: What are the theoretical foundations of filter, wrapper, and embedded feature selection methods?

### Augmentation Strategies
7. **Invariance vs. Equivariance**: How do different augmentation techniques enforce invariance or equivariance properties in learned models?

8. **Domain-Specific Augmentation**: What principles guide the design of augmentation techniques for different data modalities (images, text, time series)?

9. **Advanced Augmentation**: How do generative models and meta-learning approaches improve upon traditional augmentation techniques?

### Pipeline Design
10. **Pipeline Composition**: What are the principles of good preprocessing pipeline design, and how do they ensure reproducibility and efficiency?

11. **Conditional Processing**: How can preprocessing pipelines adapt to different data characteristics or training phases?

12. **Performance Optimization**: What strategies optimize preprocessing pipelines for computational efficiency and memory usage?

## Advanced Topics and Future Directions

### Automated Preprocessing and AutoML

**Neural Architecture Search for Preprocessing**:
```python
class PreprocessingNAS:
    """Neural Architecture Search for preprocessing pipelines"""
    
    def __init__(self, search_space, performance_metric='accuracy'):
        self.search_space = search_space
        self.performance_metric = performance_metric
        self.controller = self._build_controller()
    
    def _build_controller(self):
        """Build RNN controller for architecture generation"""
        return torch.nn.LSTM(input_size=100, hidden_size=256, num_layers=2)
    
    def generate_pipeline(self):
        """Generate preprocessing pipeline using controller"""
        # Sample preprocessing operations
        operations = []
        for step in range(self.max_pipeline_length):
            op_logits = self.controller(...)
            op_idx = torch.multinomial(F.softmax(op_logits, dim=-1), 1)
            operations.append(self.search_space[op_idx])
        
        return PreprocessingPipeline(operations)
    
    def evaluate_pipeline(self, pipeline, dataset):
        """Evaluate pipeline performance"""
        # Apply pipeline to dataset
        processed_data = pipeline.transform(dataset)
        
        # Train simple model and measure performance
        model = SimpleClassifier()
        performance = train_and_evaluate(model, processed_data)
        
        return performance[self.performance_metric]
```

**Learned Preprocessing Functions**:
```python
class LearnedPreprocessor(torch.nn.Module):
    """Learnable preprocessing using neural networks"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.preprocessor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """Apply learned preprocessing"""
        return self.preprocessor(x)
    
    def fit(self, X, y, model):
        """Train preprocessor end-to-end with main model"""
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(model.parameters())
        )
        
        for epoch in range(num_epochs):
            # Apply preprocessing
            X_processed = self(X)
            
            # Forward through main model
            predictions = model(X_processed)
            loss = F.cross_entropy(predictions, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Differential Privacy in Preprocessing

**Private Data Augmentation**:
```python
class PrivateAugmentor:
    """Data augmentation with differential privacy guarantees"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = self._calculate_sensitivity()
    
    def _calculate_sensitivity(self):
        """Calculate global sensitivity of augmentation operations"""
        # Maximum change in output due to single record change
        return 1.0  # Depends on specific augmentation
    
    def private_augment(self, data):
        """Apply augmentation with noise for privacy"""
        # Apply base augmentation
        augmented = self.base_augmentation(data)
        
        # Add calibrated noise
        noise_scale = self.sensitivity / self.epsilon
        noise = torch.normal(0, noise_scale, size=augmented.shape)
        
        return augmented + noise
```

### Continual Learning and Adaptive Preprocessing

**Adaptive Preprocessing for Distribution Shift**:
```python
class AdaptivePreprocessor:
    """Preprocessing that adapts to distribution shifts"""
    
    def __init__(self, base_preprocessor, adaptation_rate=0.01):
        self.base_preprocessor = base_preprocessor
        self.adaptation_rate = adaptation_rate
        self.reference_stats = {}
        self.current_stats = {}
    
    def detect_shift(self, new_batch):
        """Detect distribution shift using statistical tests"""
        # Compute statistics for new batch
        batch_stats = self._compute_stats(new_batch)
        
        # Compare with reference statistics
        shift_detected = self._statistical_test(batch_stats, self.reference_stats)
        
        return shift_detected
    
    def adapt(self, new_batch):
        """Adapt preprocessing based on new data"""
        if self.detect_shift(new_batch):
            # Update preprocessing parameters
            self._update_parameters(new_batch)
            
    def _update_parameters(self, new_batch):
        """Update preprocessing parameters using exponential moving average"""
        new_stats = self._compute_stats(new_batch)
        
        for key in self.reference_stats:
            self.reference_stats[key] = (
                (1 - self.adaptation_rate) * self.reference_stats[key] + 
                self.adaptation_rate * new_stats[key]
            )
```

## Conclusion

Data preprocessing and augmentation represent fundamental components of the machine learning pipeline, serving as the crucial interface between raw data and learning algorithms. This comprehensive exploration has established:

**Theoretical Foundations**: Deep understanding of mathematical transformations, statistical principles, and information-theoretic perspectives provides the basis for principled preprocessing and augmentation decisions that preserve or enhance relevant information while reducing noise and bias.

**Core Techniques**: Comprehensive coverage of normalization, standardization, outlier detection, missing data imputation, and feature engineering provides practitioners with a rich toolkit for handling diverse data quality and distribution challenges.

**Advanced Methods**: Understanding of modern augmentation techniques, including adversarial methods, generative approaches, and meta-learning strategies, enables practitioners to leverage state-of-the-art techniques for improving model performance and generalization.

**Domain Specialization**: Specialized preprocessing and augmentation strategies for computer vision, natural language processing, and time series data address the unique characteristics and challenges of different data modalities.

**Pipeline Engineering**: Systematic approaches to designing, implementing, and optimizing preprocessing pipelines ensure reproducible, efficient, and maintainable data processing workflows that scale from research to production.

**Future Directions**: Awareness of emerging trends in automated preprocessing, differential privacy, and adaptive systems provides insight into the evolution of data preprocessing technology and its integration with modern machine learning paradigms.

The design and implementation of effective preprocessing and augmentation strategies require careful consideration of data characteristics, task requirements, computational constraints, and theoretical principles. As datasets become increasingly diverse and complex, and as machine learning models become more sophisticated, the importance of well-designed preprocessing and augmentation pipelines continues to grow.

The techniques and principles covered in this module provide the foundation for building preprocessing systems that not only improve model performance but also ensure data quality, fairness, and privacy while maintaining computational efficiency and scalability across diverse deployment environments.