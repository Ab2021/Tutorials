# Day 1.4: Problem Formulation and Data Types

## Course: Comprehensive Deep Learning with PyTorch - 45-Day Masterclass
### Day 1, Part 4: Problem Taxonomy and Data Handling Strategies

---

## Overview

Problem formulation is the critical first step in any machine learning project. The way we frame a problem determines not only which algorithms and architectures we can apply, but also how we measure success and what data we need to collect. This module provides a comprehensive framework for understanding different types of learning problems and data modalities, enabling you to make informed decisions about problem setup and solution approaches.

## Learning Objectives

By the end of this module, you will:
- Master the taxonomy of machine learning problem types
- Understand different learning paradigms and their applications
- Analyze structured vs unstructured data handling strategies
- Design appropriate problem formulations for real-world challenges
- Navigate the nuances of data collection and preparation strategies

---

## 1. Learning Paradigm Taxonomy

### 1.1 Supervised Learning

#### Definition and Theoretical Foundation

**Mathematical Framework:**
Supervised learning operates on the fundamental assumption that we can learn a mapping function f: X → Y from input-output pairs (xᵢ, yᵢ) drawn from some unknown joint distribution P(X,Y).

**Formal Definition:**
Given a training dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where xᵢ ∈ X (input space) and yᵢ ∈ Y (output space), the goal is to learn a function f that minimizes the expected risk:

R(f) = E_{(x,y)~P(X,Y)}[L(f(x), y)]

Where L is a loss function measuring the discrepancy between predictions and true values.

#### Classification Problems

**Binary Classification:**
The simplest supervised learning problem where Y = {0, 1} or Y = {-1, +1}.

**Mathematical Formulation:**
- **Decision boundary:** Hyperplane separating classes
- **Probabilistic interpretation:** P(Y=1|X=x) 
- **Loss functions:** Cross-entropy, hinge loss, logistic loss

**Examples and Applications:**
- **Medical diagnosis:** Disease present/absent
- **Email filtering:** Spam/not spam
- **Credit approval:** Approve/deny
- **Quality control:** Pass/fail

**Technical Considerations:**
- **Class imbalance:** When one class is much rarer than others
- **Decision threshold optimization:** Choosing optimal probability cutoff
- **ROC curve analysis:** Trade-offs between true positive and false positive rates
- **Calibration:** Ensuring predicted probabilities reflect true likelihood

**Multi-class Classification:**
Extension to problems where Y = {1, 2, ..., K} for K > 2.

**Approaches:**
1. **One-vs-Rest (OvR):** Train K binary classifiers
2. **One-vs-One (OvO):** Train K(K-1)/2 pairwise classifiers
3. **Direct multi-class:** Algorithms that naturally handle multiple classes

**Loss Functions:**
- **Categorical cross-entropy:** -Σᵢ yᵢ log(ŷᵢ)
- **Sparse categorical cross-entropy:** For integer-encoded labels
- **Focal loss:** Addresses class imbalance by down-weighting easy examples

**Multi-label Classification:**
Each instance can belong to multiple classes simultaneously.

**Mathematical Difference:**
- **Multi-class:** Σᵢ yᵢ = 1 (mutually exclusive)
- **Multi-label:** yᵢ ∈ {0,1} for each label i (not mutually exclusive)

**Applications:**
- **Document tagging:** News articles with multiple topics
- **Image annotation:** Objects present in an image
- **Genomics:** Multiple gene functions
- **Movie genres:** Films belonging to multiple genres

**Evaluation Metrics:**
- **Instance-based:** Hamming loss, subset accuracy, Jaccard similarity
- **Label-based:** Micro/macro averaged precision, recall, F1-score
- **Ranking-based:** Coverage, ranking loss, average precision

#### Regression Problems

**Theoretical Foundation:**
Regression problems involve predicting continuous numerical values where Y ⊆ ℝ.

**Linear Regression:**
The fundamental regression model assumes linear relationship:
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

**Assumptions:**
1. **Linearity:** Relationship between X and Y is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of errors
4. **Normality:** Errors are normally distributed

**Non-linear Regression:**
When relationships are inherently non-linear:
- **Polynomial regression:** y = β₀ + β₁x + β₂x² + ... + βₚxᵖ
- **Exponential models:** y = ae^(bx)
- **Logistic growth:** y = L/(1 + ae^(-bx))

**Loss Functions:**
- **Mean Squared Error (MSE):** (1/n)Σᵢ(yᵢ - ŷᵢ)²
- **Mean Absolute Error (MAE):** (1/n)Σᵢ|yᵢ - ŷᵢ|
- **Huber loss:** Combines MSE and MAE for robustness
- **Quantile loss:** For predicting specific quantiles

**Applications:**
- **Price prediction:** Real estate, stock prices
- **Demand forecasting:** Inventory management
- **Scientific modeling:** Physics, chemistry simulations
- **Engineering:** Control systems, optimization

**Advanced Topics:**
- **Heteroscedasticity:** Non-constant variance handling
- **Multicollinearity:** Correlated predictor variables
- **Outlier detection:** Robust regression techniques
- **Confidence intervals:** Uncertainty quantification

#### Time Series and Sequential Prediction

**Time Series Characteristics:**
- **Trend:** Long-term increase or decrease
- **Seasonality:** Regular patterns repeating over time
- **Cyclical patterns:** Irregular long-term fluctuations
- **Stationarity:** Statistical properties don't change over time

**Traditional Approaches:**
- **ARIMA:** AutoRegressive Integrated Moving Average
- **Exponential smoothing:** Weighted averages with exponential decay
- **State space models:** Kalman filters and variants
- **Spectral analysis:** Frequency domain methods

**Deep Learning Approaches:**
- **RNNs/LSTMs:** Handle variable-length sequences
- **Convolutional networks:** Capture local temporal patterns
- **Transformer models:** Attention-based sequence modeling
- **Neural ODEs:** Continuous-time dynamics learning

**Challenges:**
- **Non-stationarity:** Changing statistical properties
- **Missing data:** Irregular sampling or gaps
- **Multiple seasonalities:** Daily, weekly, yearly patterns
- **Concept drift:** Underlying relationships change over time

### 1.2 Unsupervised Learning

#### Philosophical Foundation

**Core Principle:**
Unsupervised learning seeks to discover hidden structure in data without explicit target variables. The goal is to learn about the underlying probability distribution P(X) or find meaningful patterns within the data.

**Information-Theoretic Perspective:**
Many unsupervised methods can be understood through information theory:
- **Maximize information:** Preserve maximum information about the data
- **Minimize redundancy:** Remove correlated or redundant features
- **Find efficient representations:** Compress data while preserving important structure

#### Clustering

**Theoretical Foundation:**
Clustering partitions data into groups such that:
1. **Intra-cluster similarity:** Points within clusters are similar
2. **Inter-cluster dissimilarity:** Points in different clusters are dissimilar

**Distance Metrics:**
- **Euclidean distance:** √(Σᵢ(xᵢ - yᵢ)²)
- **Manhattan distance:** Σᵢ|xᵢ - yᵢ|
- **Cosine similarity:** (x·y)/(||x||||y||)
- **Mahalanobis distance:** √((x-y)ᵀΣ⁻¹(x-y))

**K-Means Clustering:**
**Algorithm:**
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Mathematical Formulation:**
Minimize: J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

**Limitations:**
- Assumes spherical clusters
- Requires pre-specifying K
- Sensitive to initialization
- Poor performance with varying cluster sizes

**Hierarchical Clustering:**
**Agglomerative approach:**
1. Start with each point as its own cluster
2. Repeatedly merge closest clusters
3. Continue until single cluster remains
4. Cut dendrogram at desired level

**Linkage criteria:**
- **Single linkage:** Minimum distance between clusters
- **Complete linkage:** Maximum distance between clusters
- **Average linkage:** Average distance between all pairs
- **Ward linkage:** Minimizes within-cluster variance

**Density-Based Clustering (DBSCAN):**
**Core concepts:**
- **Core points:** Have at least minPts neighbors within ε radius
- **Border points:** Within ε of a core point but not core themselves
- **Noise points:** Neither core nor border points

**Advantages:**
- Discovers clusters of arbitrary shape
- Automatically determines number of clusters
- Robust to outliers
- No need to specify cluster centers

**Applications:**
- **Market segmentation:** Customer grouping for targeted marketing
- **Gene expression analysis:** Finding co-expressed gene groups
- **Image segmentation:** Grouping pixels with similar properties
- **Social network analysis:** Community detection

#### Dimensionality Reduction

**Curse of Dimensionality:**
High-dimensional data suffers from:
- **Sparsity:** Data becomes sparse in high dimensions
- **Distance concentration:** All points become equidistant
- **Computational complexity:** Algorithms scale poorly
- **Visualization challenges:** Cannot directly visualize high-dimensional data

**Principal Component Analysis (PCA):**
**Mathematical Foundation:**
Find orthogonal directions of maximum variance in the data.

**Algorithm:**
1. Center the data: X̃ = X - μ
2. Compute covariance matrix: C = (1/n)X̃ᵀX̃
3. Find eigenvalues and eigenvectors of C
4. Select top k eigenvectors (principal components)
5. Project data: Y = X̃W where W contains principal components

**Properties:**
- **Variance preservation:** First k components capture maximum variance
- **Dimensionality reduction:** Reduce from d to k dimensions
- **Data reconstruction:** Can approximate original data
- **Linear transformation:** Components are linear combinations of original features

**Limitations:**
- **Linearity assumption:** Only captures linear relationships
- **Interpretability:** Components may not have clear meaning
- **Outlier sensitivity:** Outliers can dominate components
- **Global method:** Single global transformation for entire dataset

**Non-linear Dimensionality Reduction:**

**t-SNE (t-distributed Stochastic Neighbor Embedding):**
**Concept:** Preserve local neighborhood structure in lower dimensions
**Algorithm:**
1. Compute pairwise similarities in high-dimensional space
2. Initialize random embedding in low-dimensional space
3. Compute pairwise similarities in embedding space
4. Minimize KL divergence between similarity distributions

**Advantages:**
- Excellent for visualization
- Preserves local structure
- Reveals non-linear manifolds

**Limitations:**
- Computationally expensive
- Hyperparameter sensitive
- May distort global structure

**UMAP (Uniform Manifold Approximation and Projection):**
**Theoretical foundation:** Based on topological data analysis and manifold learning
**Advantages over t-SNE:**
- Better preservation of global structure
- Faster computation
- More stable results
- Better scalability

**Autoencoders:**
Neural network-based dimensionality reduction that learns non-linear mappings
**Architecture:** Encoder maps to lower dimension, decoder reconstructs original
**Loss function:** Reconstruction error (MSE for continuous, cross-entropy for discrete)
**Variants:** Variational autoencoders, denoising autoencoders, sparse autoencoders

#### Density Estimation

**Goal:** Estimate the probability density function P(X) from observed data

**Parametric Approaches:**
Assume specific functional form for the density:
- **Gaussian distribution:** P(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))
- **Gaussian Mixture Models:** P(x) = Σᵢ πᵢ N(x; μᵢ, Σᵢ)
- **Exponential family:** Large class including normal, exponential, Poisson

**Non-parametric Approaches:**
Make minimal assumptions about density form:
- **Kernel Density Estimation:** P(x) = (1/nh)Σᵢ K((x-xᵢ)/h)
- **Histogram methods:** Divide space into bins, count observations
- **k-nearest neighbors:** Density proportional to k/volume

**Applications:**
- **Anomaly detection:** Identify points in low-density regions
- **Data generation:** Sample from learned density
- **Feature learning:** Discover important data characteristics
- **Outlier detection:** Points far from typical density

### 1.3 Semi-Supervised Learning

#### Theoretical Motivation

**Problem Setting:**
Given a small amount of labeled data DL = {(xᵢ, yᵢ)} and a large amount of unlabeled data DU = {xⱼ}, learn a function that performs better than using labeled data alone.

**Fundamental Assumptions:**
1. **Smoothness assumption:** Points close in input space should have similar outputs
2. **Cluster assumption:** Decision boundaries should lie in low-density regions
3. **Manifold assumption:** Data lies on a lower-dimensional manifold

#### Approaches and Algorithms

**Self-Training:**
1. Train initial model on labeled data
2. Predict labels for unlabeled data
3. Add most confident predictions to training set
4. Retrain model and repeat

**Advantages:**
- Simple to implement
- Works with any base classifier
- Natural confidence-based selection

**Disadvantages:**
- Error propagation
- May reinforce initial biases
- No theoretical guarantees

**Co-Training:**
Requires two views of the data (different feature sets):
1. Train separate classifiers on each view
2. Each classifier labels data for the other
3. Add most confident predictions to training set
4. Retrain both classifiers

**Requirements:**
- Features can be split into two views
- Each view is sufficient for learning
- Views are conditionally independent given the label

**Graph-Based Methods:**
**Label Propagation:**
1. Construct graph with data points as nodes
2. Connect similar points with weighted edges
3. Propagate labels through graph structure
4. Minimize energy function that encourages smoothness

**Mathematical Formulation:**
Minimize: Σᵢⱼ wᵢⱼ(yᵢ - yⱼ)² + λΣᵢ(yᵢ - ŷᵢ)²

Where wᵢⱼ are edge weights and ŷᵢ are predicted labels.

**Generative Models:**
Use Expectation-Maximization with mixture models:
1. **E-step:** Estimate posterior probabilities for unlabeled data
2. **M-step:** Update model parameters using both labeled and unlabeled data
3. Repeat until convergence

**Modern Deep Learning Approaches:**
- **Consistency regularization:** Encourage consistent predictions on augmented versions
- **Pseudo-labeling:** Use model predictions as targets for unlabeled data
- **MixMatch:** Combines consistency regularization with traditional techniques
- **FixMatch:** Uses weak and strong augmentation for pseudo-labeling

### 1.4 Self-Supervised Learning

#### Paradigm Shift

**Definition:**
Self-supervised learning creates supervisory signals from the data itself, without human annotation. It learns representations by solving pretext tasks designed to capture useful structure in the data.

**Key Insight:**
Use the inherent structure in data (temporal, spatial, semantic relationships) to create learning signals.

#### Computer Vision Applications

**Pretext Tasks:**
1. **Rotation Prediction:** Predict rotation angle applied to image
2. **Jigsaw Puzzles:** Reconstruct shuffled image patches
3. **Colorization:** Predict color channels from grayscale
4. **Inpainting:** Fill in masked regions of images
5. **Contrastive Learning:** Distinguish between similar and dissimilar images

**Contrastive Learning (SimCLR, MoCo):**
**Core idea:** Learn representations where similar images are close and dissimilar images are far apart.

**Algorithm:**
1. Apply data augmentation to create positive pairs
2. Sample negative examples from different images
3. Train network to maximize similarity of positive pairs
4. Minimize similarity of negative pairs

**Loss function (InfoNCE):**
L = -log(exp(sim(zᵢ, z⁺)/τ) / Σⱼexp(sim(zᵢ, zⱼ)/τ))

Where sim is similarity function, z⁺ is positive example, τ is temperature.

#### Natural Language Processing Applications

**Masked Language Modeling (BERT):**
**Task:** Predict masked tokens in a sentence
**Example:** "The cat [MASK] on the mat" → predict "sat"
**Learning:** Bidirectional context understanding

**Autoregressive Language Modeling (GPT):**
**Task:** Predict next token given previous tokens
**Example:** "The cat sat" → predict "on"
**Learning:** Sequential dependencies and generation

**Next Sentence Prediction:**
**Task:** Determine if two sentences are consecutive in original text
**Learning:** Sentence-level relationships and discourse understanding

**Advanced Techniques:**
- **ELECTRA:** Detect replaced tokens instead of predicting masked tokens
- **RoBERTa:** Optimized training of BERT-style models
- **T5:** Text-to-text transfer transformer treating all tasks as generation

#### Time Series and Sequential Data

**Pretext Tasks:**
1. **Temporal Ordering:** Predict correct order of shuffled segments
2. **Future Prediction:** Predict future values from past observations
3. **Reconstruction:** Recover original signal from corrupted version
4. **Transformation Prediction:** Identify applied transformations

**Applications:**
- **Speech recognition:** Learn phonetic representations
- **Music analysis:** Capture musical structure and patterns
- **Physiological signals:** ECG, EEG pattern recognition
- **Financial time series:** Market pattern recognition

---

## 2. Problem Types by Output Structure

### 2.1 Classification Taxonomies

#### Standard Classification

**Binary Classification:**
**Problem characteristics:**
- Two mutually exclusive classes
- Decision boundary divides feature space
- Probability interpretation: P(class = 1 | features)

**Evaluation metrics:**
- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **Precision:** TP / (TP + FP)
- **Recall (Sensitivity):** TP / (TP + FN)
- **Specificity:** TN / (TN + FP)
- **F1-score:** 2 × (Precision × Recall) / (Precision + Recall)

**Multi-class Classification:**
**Approaches:**
1. **Direct methods:** Algorithms naturally handling multiple classes
2. **Decomposition methods:** Reduce to binary classification problems

**Softmax activation:**
P(yᵢ = k | x) = exp(zₖ) / Σⱼexp(zⱼ)

**Challenges:**
- **Class imbalance:** Unequal class frequencies
- **Hierarchical classes:** Natural class hierarchies
- **Overlapping classes:** Ambiguous boundaries

#### Structured Classification

**Multi-label Classification:**
Each instance can belong to multiple classes simultaneously.

**Problem formulation:**
- **Binary relevance:** Independent binary classifier for each label
- **Label powerset:** Treat each unique label combination as separate class
- **Classifier chains:** Model label dependencies explicitly

**Evaluation challenges:**
- **Exact match:** Predicted label set must exactly match true labels
- **Hamming loss:** Fraction of incorrectly predicted labels
- **Subset accuracy:** Fraction of instances with perfectly predicted label sets

**Hierarchical Classification:**
Labels organized in tree or DAG structure.

**Constraints:**
- If instance belongs to class, it must belong to all ancestor classes
- Predictions must respect hierarchy structure

**Applications:**
- **Document classification:** Topic hierarchies
- **Gene function prediction:** Gene Ontology
- **E-commerce:** Product category trees

### 2.2 Regression Variations

#### Standard Regression

**Linear Regression Assumptions:**
1. **Linearity:** E[Y|X] is linear in X
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Var(ε|X) is constant
4. **Normality:** ε ~ N(0, σ²)

**When assumptions violated:**
- **Non-linearity:** Use polynomial features, splines, or neural networks
- **Heteroscedasticity:** Use robust standard errors or weighted least squares
- **Non-normality:** Transform variables or use robust methods
- **Dependence:** Use time series methods or clustered standard errors

#### Multi-output Regression

**Problem formulation:**
Predict multiple continuous targets simultaneously:
Y ∈ ℝᵈ where d > 1

**Approaches:**
1. **Independent regression:** Separate model for each output
2. **Multi-task learning:** Shared representations with task-specific outputs
3. **Chained regression:** Use previous outputs as features for subsequent predictions

**Applications:**
- **Weather forecasting:** Temperature, humidity, pressure
- **Financial modeling:** Multiple asset price prediction
- **Robotics:** Joint angle prediction for multiple joints

#### Quantile Regression

**Motivation:**
Instead of predicting mean, predict specific quantiles of conditional distribution.

**Loss function (quantile loss):**
L_τ(y, ŷ) = (y - ŷ)(τ - I(y < ŷ))

Where τ ∈ [0,1] is the desired quantile.

**Applications:**
- **Risk management:** Value-at-Risk estimation
- **Inventory planning:** Demand forecasting with uncertainty
- **Medical diagnosis:** Reference ranges for biomarkers

#### Survival Analysis

**Problem setting:**
Predict time until event occurs, handling censored observations.

**Hazard function:**
h(t) = lim_{dt→0} P(t ≤ T < t + dt | T ≥ t) / dt

**Cox Proportional Hazards Model:**
h(t|x) = h₀(t)exp(βᵀx)

**Applications:**
- **Medical research:** Time to disease recurrence
- **Engineering:** Equipment failure prediction
- **Marketing:** Customer churn analysis

---

## 3. Data Types and Handling Strategies

### 3.1 Structured Data

#### Tabular Data Characteristics

**Row-column structure:**
- Rows represent observations/instances
- Columns represent features/variables
- Fixed schema with defined data types

**Feature types:**
- **Numerical:** Continuous or discrete numeric values
- **Categorical:** Finite set of discrete values
- **Ordinal:** Categorical with natural ordering
- **Binary:** True/false, yes/no, 1/0 values

#### Numerical Data Handling

**Continuous Variables:**
**Preprocessing steps:**
1. **Missing value imputation:** Mean, median, mode, or model-based
2. **Outlier detection:** Statistical tests, isolation forest, local outlier factor
3. **Scaling:** Standardization (z-score), normalization (min-max), robust scaling
4. **Transformation:** Log, square root, Box-Cox to achieve normality

**Distribution considerations:**
- **Normal distribution:** Many algorithms assume normality
- **Skewed distributions:** May need transformation
- **Heavy tails:** Robust methods may be needed
- **Multimodal distributions:** May indicate mixed populations

**Discrete Variables:**
**Count data:**
- **Poisson distribution:** For rare events
- **Negative binomial:** For overdispersed counts
- **Zero-inflated models:** When excess zeros present

**Ordinal data:**
- **Natural ordering:** Education level, satisfaction ratings
- **Encoding strategies:** Ordinal encoding preserving order
- **Distance metrics:** Appropriate measures respecting ordinality

#### Categorical Data Handling

**Encoding strategies:**

**One-hot encoding:**
Transform categorical variable with k categories into k binary variables.
**Advantages:**
- No ordinal assumptions
- Compatible with linear models
- Clear interpretation

**Disadvantages:**
- High dimensionality for many categories
- Sparse representations
- Curse of dimensionality

**Label encoding:**
Assign integer values to categories.
**When appropriate:**
- Ordinal variables with natural ordering
- Tree-based algorithms (can handle arbitrary encodings)
- Deep learning with embedding layers

**Target encoding:**
Replace categories with target variable statistics (mean, median).
**Advantages:**
- Captures category-target relationship
- Reduces dimensionality
- Works well with high-cardinality features

**Risks:**
- Overfitting to training data
- Requires cross-validation for proper implementation
- May leak information about target

**Hash encoding:**
Use hash functions to map categories to fixed number of buckets.
**Benefits:**
- Handles unknown categories in test data
- Fixed dimensionality regardless of category count
- Memory efficient

**High-cardinality categorical features:**
**Challenges:**
- Many unique categories (thousands or millions)
- Sparse one-hot representations
- Limited training examples per category

**Solutions:**
- **Frequency-based grouping:** Combine rare categories
- **Embedding layers:** Learn dense representations
- **Feature hashing:** Reduce dimensionality through hashing
- **Entity embeddings:** Deep learning approach for categorical features

#### Missing Data Strategies

**Types of missing data:**

**Missing Completely at Random (MCAR):**
Missing mechanism independent of observed and unobserved data.
**Example:** Survey responses lost due to computer malfunction

**Missing at Random (MAR):**
Missing mechanism depends on observed data but not unobserved data.
**Example:** Income not reported, but missingness depends on age

**Missing Not at Random (MNAR):**
Missing mechanism depends on unobserved data.
**Example:** High-income individuals refuse to report income

**Imputation strategies:**

**Simple imputation:**
- **Mean/median/mode:** Replace with central tendency
- **Forward/backward fill:** Use previous/next valid value
- **Zero/constant fill:** Replace with fixed value

**Advanced imputation:**
- **KNN imputation:** Use similar instances for imputation
- **Regression imputation:** Predict missing values using other features
- **Multiple imputation:** Generate multiple plausible values
- **Deep learning:** Autoencoders, GANs for missing value generation

**Handling missingness:**
- **Indicator variables:** Add binary flag for missingness
- **Special category:** Treat missing as separate category
- **Model-based:** Incorporate missingness into probabilistic model

### 3.2 Unstructured Data

#### Text Data

**Preprocessing pipeline:**

**Tokenization:**
Split text into meaningful units (words, subwords, characters).
**Challenges:**
- **Language-specific rules:** Different tokenization for different languages
- **Ambiguity:** Contractions, hyphenated words, punctuation
- **Subword tokenization:** BPE, WordPiece, SentencePiece for rare words

**Normalization:**
- **Lowercasing:** Convert to uniform case
- **Punctuation removal:** Remove or standardize punctuation
- **Accent removal:** Normalize accented characters
- **Spelling correction:** Fix common typos and misspellings

**Stop word removal:**
Remove common words with little semantic meaning.
**Considerations:**
- **Language-dependent:** Different stop words for different languages
- **Context-dependent:** "Not" might be important for sentiment analysis
- **Domain-specific:** Technical terms might be stop words in general but important in domain

**Stemming and lemmatization:**
- **Stemming:** Reduce words to root form (running → run)
- **Lemmatization:** Reduce to canonical form considering part of speech
- **Language models:** Modern approaches use contextual embeddings

**Feature extraction:**

**Bag-of-words (BoW):**
Represent document as vector of word counts.
**Properties:**
- **Order ignored:** No positional information
- **Sparse representations:** Most features are zero
- **High dimensionality:** Vocabulary size determines dimensions

**TF-IDF (Term Frequency-Inverse Document Frequency):**
Weight terms by frequency and rarity.
TF-IDF(t,d) = TF(t,d) × log(N/DF(t))

**Advantages:**
- **Rare word emphasis:** Important rare words get higher weights
- **Common word de-emphasis:** Frequent but uninformative words get lower weights
- **Interpretable:** Clear meaning for feature weights

**N-grams:**
Capture sequences of n consecutive words.
**Benefits:**
- **Context preservation:** Some word order information retained
- **Phrase capture:** Common phrases treated as single units
- **Negation handling:** "not good" vs "good" distinction

**Word embeddings:**
Dense vector representations capturing semantic similarity.
- **Word2Vec:** Skip-gram and CBOW approaches
- **GloVe:** Global vector representations
- **FastText:** Subword information incorporation
- **Contextual embeddings:** BERT, GPT, ELMo with context dependence

#### Image Data

**Image representation:**
Images as multi-dimensional arrays:
- **Grayscale:** Height × Width
- **Color (RGB):** Height × Width × 3 channels
- **Multi-spectral:** Height × Width × Channels (satellite imagery)

**Preprocessing pipeline:**

**Normalization:**
- **Pixel value scaling:** [0,255] to [0,1] or [-1,1]
- **Channel-wise normalization:** Subtract mean, divide by standard deviation
- **Dataset statistics:** Use training set statistics for normalization

**Resizing and cropping:**
- **Fixed size:** Resize all images to standard dimensions
- **Aspect ratio preservation:** Pad or crop to maintain ratios
- **Multi-scale:** Use multiple resolutions for training

**Data augmentation:**
Artificially increase dataset diversity:
- **Geometric transformations:** Rotation, flipping, cropping, scaling
- **Photometric changes:** Brightness, contrast, saturation adjustments
- **Advanced techniques:** Mixup, CutMix, AutoAugment

**Color space conversions:**
- **RGB to HSV:** Separate color and intensity information
- **Grayscale conversion:** Reduce to single intensity channel
- **Lab color space:** Perceptually uniform color representation

#### Audio Data

**Digital audio representation:**
- **Sampling rate:** Number of samples per second (Hz)
- **Bit depth:** Number of bits per sample
- **Channels:** Mono (1) or stereo (2) or multi-channel

**Feature extraction:**

**Time-domain features:**
- **Zero crossing rate:** Frequency of signal sign changes
- **Energy:** Signal power over time
- **Spectral centroid:** Center of mass of frequency spectrum

**Frequency-domain features:**
- **Mel-frequency cepstral coefficients (MFCCs):** Perceptually motivated features
- **Spectrograms:** Time-frequency representations
- **Chroma features:** Pitch class profiles

**Preprocessing:**
- **Noise reduction:** Remove background noise
- **Normalization:** Standardize volume levels
- **Windowing:** Segment audio into overlapping frames
- **Pre-emphasis:** Boost high frequencies

---

## 4. Problem Formulation Process

### 4.1 Real-world to ML Translation

#### Problem Definition Framework

**Step 1: Business/Scientific Objective Identification**
- **What decision needs to be made?** Clear articulation of the end goal
- **Who will use the results?** Understanding stakeholder needs
- **What constitutes success?** Defining measurable outcomes
- **What are the constraints?** Time, budget, computational, ethical limitations

**Step 2: Data Availability Assessment**
- **What data is available?** Current data sources and quality
- **What data can be collected?** Feasible data collection strategies
- **What are the data limitations?** Privacy, access, quality constraints
- **How much data is needed?** Sample size requirements for desired accuracy

**Step 3: ML Problem Type Mapping**
- **Input-output relationship:** What predicts what?
- **Supervision availability:** Labeled vs unlabeled data
- **Output structure:** Categorical, numerical, structured
- **Temporal aspects:** Static vs dynamic predictions

#### Common Translation Challenges

**Ambiguous objectives:**
- **Multiple goals:** Balancing accuracy, fairness, interpretability
- **Stakeholder alignment:** Different groups want different outcomes
- **Success metrics:** What to optimize may not be clear
- **Long-term vs short-term:** Different optimization horizons

**Data-model mismatch:**
- **Data availability vs requirements:** Limited data for complex models
- **Feature engineering needs:** Gap between raw data and useful features
- **Temporal alignment:** Historical data may not reflect current patterns
- **Scale mismatch:** Laboratory vs production data differences

**Solution approaches:**
- **Iterative refinement:** Start simple, increase complexity gradually
- **Prototype development:** Build minimal viable model first
- **Stakeholder engagement:** Regular feedback and requirement clarification
- **Risk assessment:** Identify and mitigate potential failure modes

### 4.2 Success Metrics Definition

#### Alignment with Business Objectives

**Performance metrics must reflect business value:**

**Classification examples:**
- **Medical diagnosis:** Minimize false negatives (missed diseases) even if false positives increase
- **Fraud detection:** Balance catching fraud vs customer experience
- **Content moderation:** Consider costs of both over-moderation and under-moderation

**Regression examples:**
- **Demand forecasting:** Asymmetric costs of over-prediction vs under-prediction
- **Price optimization:** Revenue impact more important than prediction accuracy
- **Quality control:** Different tolerances for different types of defects

**Custom metrics:**
Sometimes standard ML metrics don't capture business value:
- **Profit-based metrics:** Weight predictions by economic impact
- **Temporal metrics:** Early detection bonus, late penalty
- **Fairness metrics:** Equal opportunity across demographic groups
- **Robustness metrics:** Performance under distribution shift

#### Multi-objective optimization

**Common trade-offs:**
- **Accuracy vs Interpretability:** Complex models vs explainable models
- **Performance vs Fairness:** Overall accuracy vs equitable outcomes
- **Precision vs Recall:** Conservative vs liberal prediction strategies
- **Training time vs Inference time:** Complex training for fast inference

**Handling approaches:**
- **Pareto optimization:** Find non-dominated solutions
- **Weighted objectives:** Combine metrics with business-driven weights
- **Constraint optimization:** Optimize primary metric subject to constraints
- **Sequential optimization:** Optimize metrics in order of priority

---

## 5. Key Questions and Answers

### Beginner Level Questions

**Q1: What's the difference between classification and regression?**
**A:** 
- **Classification:** Predicts discrete categories/classes (spam/not spam, dog/cat/bird)
- **Regression:** Predicts continuous numerical values (house price, temperature, stock price)
- **Key distinction:** The type of output variable determines the problem type
- **Mixed cases:** Some problems can be formulated either way (age prediction vs age group classification)

**Q2: When would you use unsupervised learning instead of supervised learning?**
**A:** Use unsupervised learning when:
- **No labeled data available:** You have data but no target variables
- **Exploratory analysis:** Want to discover hidden patterns or structure
- **Feature learning:** Need better representations for downstream tasks
- **Anomaly detection:** Identify unusual patterns without knowing what "unusual" looks like
- **Data preprocessing:** Dimensionality reduction, clustering for data organization

**Q3: What is the difference between multi-class and multi-label classification?**
**A:**
- **Multi-class:** Each instance belongs to exactly one class out of multiple classes (mutually exclusive)
  - Example: Email classification (spam, promotions, personal, work)
- **Multi-label:** Each instance can belong to multiple classes simultaneously (not mutually exclusive)  
  - Example: Movie genres (a movie can be both action AND comedy AND thriller)

**Q4: How do you handle categorical data with many unique values?**
**A:** For high-cardinality categorical features:
- **Frequency-based grouping:** Combine rare categories into "Other" category
- **Target encoding:** Replace categories with target variable statistics (with proper CV)
- **Embedding layers:** Learn dense representations in deep learning models
- **Feature hashing:** Map categories to fixed number of hash buckets
- **Drop rare categories:** Remove categories with very few examples

### Intermediate Level Questions

**Q5: Explain the difference between MCAR, MAR, and MNAR missing data mechanisms.**
**A:**
- **MCAR (Missing Completely at Random):** Missingness is independent of both observed and unobserved data. Example: Lab equipment randomly fails.
- **MAR (Missing at Random):** Missingness depends only on observed data. Example: Younger people more likely to skip income questions.
- **MNAR (Missing Not at Random):** Missingness depends on unobserved data. Example: High earners refuse to report income.
Understanding the mechanism affects the choice of imputation strategy and potential bias in analysis.

**Q6: Why might you choose semi-supervised learning over supervised learning?**
**A:** Semi-supervised learning is beneficial when:
- **Labeled data is expensive:** Medical imaging where expert annotation is costly
- **Large amounts of unlabeled data:** Web scraping provides lots of unlabeled text/images
- **Labeling is subjective:** Different annotators might give different labels
- **Temporal dynamics:** New unlabeled data continuously arrives
- **Performance gains:** Unlabeled data provides information about data distribution

**Q7: How do you decide between different problem formulations for the same real-world problem?**
**A:** Consider:
- **Data availability:** What type of labels/targets do you have?
- **Business requirements:** What decisions will be made with predictions?
- **Evaluation criteria:** How will success be measured?
- **Computational constraints:** Training time, inference speed, memory requirements
- **Interpretability needs:** How important is explaining predictions?
- **Start simple:** Begin with simplest formulation, add complexity as needed

### Advanced Level Questions

**Q8: How does the choice of loss function relate to the assumptions about data distribution and noise?**
**A:** Loss functions encode assumptions about the problem structure:
- **Mean Squared Error:** Assumes Gaussian noise, penalizes large errors heavily
- **Mean Absolute Error:** Assumes Laplace noise, more robust to outliers
- **Huber loss:** Combines MSE and MAE, robust to outliers while maintaining smoothness
- **Cross-entropy:** Assumes categorical distribution, measures information loss
- **Focal loss:** Addresses class imbalance by focusing on hard examples
The choice should match the actual data generating process and business requirements.

**Q9: Explain the theoretical relationship between semi-supervised and self-supervised learning.**
**A:** Both leverage unlabeled data but differently:
- **Semi-supervised:** Uses small labeled set + large unlabeled set, assumes smoothness/cluster/manifold assumptions
- **Self-supervised:** Creates labels from data structure itself, learns general representations
- **Connection:** Both exploit data structure, but self-supervised doesn't require any external labels
- **Modern trend:** Self-supervised pretraining followed by supervised fine-tuning
- **Theoretical foundation:** Both try to learn from data distribution P(X) to improve performance on P(Y|X)

**Q10: How do you handle concept drift in different types of learning problems?**
**A:** Concept drift occurs when P(Y|X) or P(X) changes over time:

**Detection strategies:**
- **Statistical tests:** Compare recent vs historical data distributions
- **Performance monitoring:** Track model performance over time
- **Drift detectors:** ADWIN, DDM, EDDM algorithms

**Adaptation strategies:**
- **Periodic retraining:** Retrain on recent data windows
- **Online learning:** Continuously update model with new data  
- **Ensemble methods:** Combine models trained on different time periods
- **Domain adaptation:** Transfer learning techniques for distribution shift

**Problem-specific considerations:**
- **Classification:** Monitor class distribution changes, decision boundary shifts
- **Regression:** Track residual patterns, feature importance changes
- **Time series:** Detect structural breaks, seasonality changes

---

## 6. Tricky Questions for Deep Understanding

### Conceptual Paradoxes

**Q1: Why might a supervised learning problem be better solved with unsupervised methods?**
**A:** This counterintuitive situation occurs when:

**Label noise:** If labels are unreliable or noisy, unsupervised methods might find better natural groupings than following noisy supervision.

**Representation learning:** Sometimes unsupervised feature learning (like word embeddings) creates better representations than direct supervised learning on the target task.

**Sample complexity:** With very few labeled examples, unsupervised clustering might provide better generalization than overfitting to limited supervision.

**Hidden structure:** The supervised labels might not capture the most important structure in the data. Unsupervised methods might reveal more meaningful patterns.

**Example:** Document clustering might find more coherent topic groups than classification based on noisy crowd-sourced labels.

**Q2: How can the same dataset be used for both classification and regression problems?**
**A:** This depends on how you define the target variable:

**Examples:**
- **Age prediction:** Regression (predict exact age) vs Classification (predict age group: child/adult/senior)
- **Customer value:** Regression (predict lifetime value) vs Classification (high/medium/low value)
- **Medical diagnosis:** Regression (predict biomarker level) vs Classification (normal/abnormal)

**Trade-offs:**
- **Regression:** Preserves information, provides more granular predictions
- **Classification:** May be more robust, easier to interpret, aligns with decision-making

**Hybrid approaches:**
- **Ordinal regression:** Treats ordered categories with regression-like methods
- **Threshold methods:** Train regression model, apply thresholds for classification
- **Multi-task learning:** Simultaneously predict continuous value and category

**Q3: When might increasing training data hurt performance?**
**A:** Counterintuitively, more data can sometimes hurt:

**Label noise accumulation:** If new data has higher noise rate, it can degrade model performance.

**Distribution shift:** New data from different distribution can hurt if not handled properly.

**Computational constraints:** With limited compute, using more data might force simpler models or fewer epochs.

**Overfitting with model complexity:** If model capacity increases with data but inappropriately, overfitting might worsen.

**Class imbalance amplification:** If new data skews class distribution further, it might hurt minority class performance.

### Technical Subtleties

**Q4: Why might a model perform well on validation data but fail in production?**
**A:** This indicates distribution shift between development and deployment:

**Temporal shift:** Model trained on historical data, but patterns change over time.

**Selection bias:** Validation data not representative of actual users/conditions.

**Data leakage:** Information about target leaked into features during development but not available in production.

**Adversarial environments:** Users adapt behavior in response to model (feedback loops).

**Infrastructure differences:** Different data processing, latency constraints, or input quality.

**Batch vs online:** Model validated on batch data but deployed for real-time inference.

**Solutions:** Proper train/validation/test splits, temporal validation, A/B testing, continuous monitoring.

**Q5: Explain why some features might be predictive individually but not useful in ensemble models.**
**A:** This relates to feature redundancy and model capacity:

**Correlation with existing features:** If feature provides similar information to features already in model, marginal benefit is small.

**Non-linear interactions:** Feature might only be useful in specific combinations that the model can't capture.

**Different scale requirements:** Feature might require different preprocessing or model type than rest of ensemble.

**Overfitting contribution:** Feature might improve training performance but hurt generalization.

**Computational cost:** Feature might provide small benefit at high computational cost, making it inefficient.

**Example:** In text classification, individual rare words might be predictive, but if the model already captures document topics through common words, rare words add little value.

### Philosophical Questions

**Q6: Is the distinction between supervised and unsupervised learning fundamental, or just a matter of how we frame problems?**
**A:** This touches on deep questions about learning and intelligence:

**Fundamental perspective:**
- **Different objectives:** Supervised minimizes prediction error, unsupervised maximizes likelihood or finds structure
- **Different inductive biases:** Require different assumptions about world structure
- **Different mathematical frameworks:** Different optimization problems and theoretical foundations

**Unified perspective:**
- **Self-supervision bridge:** Shows that supervision can come from data structure itself
- **Generative models:** Unsupervised models can be used for supervised tasks
- **Representation learning:** Both aim to learn useful representations
- **Information theory:** Both maximize mutual information between inputs and learned representations

**Modern view:** The distinction is blurring as we understand that all learning is about capturing structure, whether that structure is defined by external labels or inherent data patterns.

**Q7: How do we know if we've formulated a machine learning problem correctly?**
**A:** This is both technical and philosophical:

**Technical indicators:**
- **Baseline performance:** Does model beat reasonable baselines?
- **Learning curves:** Does performance improve with more data/training?
- **Generalization:** Does model work on held-out data?
- **Error analysis:** Are errors interpretable and actionable?

**Business indicators:**
- **Stakeholder satisfaction:** Does model solve the actual business problem?
- **ROI measurement:** Does model provide value commensurate with investment?
- **User adoption:** Do end users actually use the system?
- **Downstream impact:** Does model improve decision-making?

**Philosophical considerations:**
- **Problem well-posedness:** Is there enough information to make accurate predictions?
- **Ethical implications:** Does problem formulation respect human values and rights?
- **Unintended consequences:** What are potential negative side effects?
- **Alternative formulations:** Are there better ways to frame the problem?

**Iterative refinement:** Problem formulation is rarely right the first time. Success often comes from iterative improvement based on feedback and results.

---

## Summary and Integration

Problem formulation is the cornerstone of successful machine learning projects. The framework presented here provides a systematic approach to:

### Key Takeaways:

1. **Learning Paradigm Selection:** Understanding when to use supervised, unsupervised, semi-supervised, or self-supervised learning depends on data availability, problem structure, and business objectives.

2. **Problem Type Mapping:** Correctly identifying whether your problem is classification, regression, clustering, or density estimation determines appropriate algorithms and evaluation metrics.

3. **Data Type Considerations:** Structured vs unstructured data require different preprocessing pipelines and modeling approaches.

4. **Success Metrics Alignment:** Evaluation metrics must reflect business value, not just statistical accuracy.

### Decision Framework:

**Start with business objectives** → **Assess data availability** → **Map to ML problem type** → **Choose appropriate methods** → **Define success metrics** → **Iterate and refine**

### Future Considerations:

As the field evolves, the boundaries between different problem types and learning paradigms continue to blur. Foundation models and transfer learning are making it easier to leverage pre-trained representations across different problem types. However, understanding these fundamental distinctions remains crucial for making informed decisions about model selection, evaluation, and deployment.

The ability to correctly formulate problems separates successful ML practitioners from those who apply techniques without understanding. This foundational knowledge enables you to ask the right questions, choose appropriate methods, and design solutions that actually solve real-world problems.

---

## Next Steps

In the next module, we'll explore the business and industry context of deep learning, examining how these problem formulation principles apply across different sectors and how to build business cases for machine learning projects.