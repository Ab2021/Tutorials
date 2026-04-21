# 🧠 ML Coding Problems Universe — Deep-ML + TensorTonic
### Hands-On & Machine Coding Round Grind Reference
#### Sources: deep-ml.com | tensortonic.com (via GitHub solution repos)

> **How to use this file:**
> - `⭐` = Frequently asked in DS interviews (high priority)
> - `🔥` = Core ML building block (must know cold)
> - `💡` = Appears on both platforms = especially important
> - Study in ORDER: Linear Algebra → Statistics → ML Algorithms → DL → NLP → MLOps

---

## 📊 COVERAGE SUMMARY

| Platform | Problems | Focus |
|---|---|---|
| **TensorTonic** | ~160+ problems (combined from 2 repos) | Core ML from scratch, DL building blocks |
| **Deep-ML** | 100+ problems | Linear algebra, ML, statistics, NLP, vision |
| **Combined unique** | ~200+ distinct problems | Full ML coding spectrum |

---

---

# ═══════════════════════════════════════════════
# SECTION 1: LINEAR ALGEBRA & MATRIX OPERATIONS
# ═══════════════════════════════════════════════

> **Interview Relevance:** High — almost every ML coding round starts with matrix ops. Build these first.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 1 | Matrix Transpose | Both | Easy | 🔥 |
| 2 | Matrix Multiplication / Dot Product | Both | Easy | 🔥💡 |
| 3 | Matrix Inverse | Both | Medium | ⭐ |
| 4 | Matrix Trace | Both | Easy | 🔥 |
| 5 | Matrix Normalization (Frobenius norm) | Both | Easy | ⭐ |
| 6 | Make Diagonal Matrix | TensorTonic | Easy | ⭐ |
| 7 | Eigenvalues Computation | TensorTonic | Hard | ⭐ |
| 8 | Covariance Matrix | Both | Medium | 🔥💡 |
| 9 | Euclidean Distance | Both | Easy | 🔥💡 |
| 10 | Cosine Similarity | Both | Easy | 🔥💡 |
| 11 | Vector Norm (L1, L2, Inf) | TensorTonic | Easy | 🔥 |
| 12 | Normalize 3D Vector | Both | Easy | ⭐ |
| 13 | Angle Between 3D Vectors | Both | Easy | ⭐ |
| 14 | Dot Product | Both | Easy | 🔥 |
| 15 | Bilinear Interpolation | TensorTonic | Medium | ⭐ |
| 16 | Homogeneous Transform | TensorTonic | Medium | Medium |
| 17 | PCA Projection (SVD-based) | TensorTonic | Hard | ⭐ |
| 18 | Linear Regression Closed Form (Normal Eq.) | TensorTonic | Medium | 🔥⭐ |
| 19 | Ridge Regression | TensorTonic (Repo2) | Medium | ⭐ |
| 20 | Rank Transform | TensorTonic | Easy | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 2: STATISTICS & PROBABILITY
# ═══════════════════════════════════════════════

> **Interview Relevance:** Very High — statistical coding is a Flipkart DDS staple.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 21 | Mean, Median, Mode | Both | Easy | 🔥💡 |
| 22 | Percentiles & Quantiles | TensorTonic | Easy | ⭐ |
| 23 | Standard Deviation & Variance | Deep-ML | Easy | 🔥 |
| 24 | Bootstrapped Mean Estimation | TensorTonic | Medium | ⭐ |
| 25 | Autocorrelation | TensorTonic | Medium | ⭐ |
| 26 | Chi-Squared Independence Test | TensorTonic | Medium | ⭐ |
| 27 | Bernoulli PMF | TensorTonic | Easy | ⭐ |
| 28 | Binomial PMF and CDF | Both | Easy | 🔥💡 |
| 29 | Poisson PMF and CDF | TensorTonic | Easy | ⭐ |
| 30 | Geometric PMF & Mean | TensorTonic | Easy | ⭐ |
| 31 | Expected Value (Discrete) | Both | Easy | 🔥 |
| 32 | Expected Calibration Error (ECE) | Both | Medium | ⭐ |
| 33 | Gaussian Naive Bayes | TensorTonic | Medium | 🔥⭐ |
| 34 | Percent Change | TensorTonic | Easy | ⭐ |
| 35 | Minmax Normalization | Both | Easy | 🔥💡 |
| 36 | Robust Scaling (IQR-based) | TensorTonic | Easy | ⭐ |
| 37 | Z-score Normalization | Deep-ML | Easy | 🔥 |
| 38 | Imputate Missing Values (mean/median/mode) | TensorTonic | Easy | 🔥 |
| 39 | Binning (equal-width, equal-frequency) | Both | Medium | ⭐ |
| 40 | Cohen's Kappa Score | TensorTonic | Medium | ⭐ |
| 41 | Differencing (time-series stationarity) | TensorTonic | Easy | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 3: CORE ML ALGORITHMS FROM SCRATCH
# ═══════════════════════════════════════════════

> **Interview Relevance:** HIGHEST — this is the main HO round content.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 42 | Logistic Regression Training (Gradient Descent) | TensorTonic | Medium | 🔥⭐ |
| 43 | Linear Regression (Gradient Descent) | Deep-ML | Medium | 🔥 |
| 44 | Linear Regression Closed Form (OLS) | TensorTonic | Medium | 🔥⭐ |
| 45 | Ridge Regression (L2) | TensorTonic | Medium | ⭐ |
| 46 | Decision Tree Split (Gini / Info Gain) | TensorTonic | Hard | 🔥⭐ |
| 47 | Entropy at Node | TensorTonic | Easy | 🔥 |
| 48 | Information Gain | TensorTonic (Repo2) | Medium | 🔥⭐ |
| 49 | Random Forest Vote (Majority) | TensorTonic | Medium | ⭐ |
| 50 | Majority Classifier (baseline) | TensorTonic | Easy | ⭐ |
| 51 | K-Means Assignment Step | Both | Medium | 🔥💡 |
| 52 | K-Means Centroid Update | Both | Medium | 🔥💡 |
| 53 | K-Nearest Neighbors Distance | TensorTonic | Easy | 🔥 |
| 54 | PCA Projection (eigen decomposition) | TensorTonic | Hard | ⭐ |
| 55 | Gradient Descent on Quadratic | TensorTonic | Easy | 🔥 |
| 56 | Gradient Clipping (norm-based) | Both | Medium | ⭐ |
| 57 | K-Fold Split | TensorTonic | Medium | ⭐ |
| 58 | Batch Generator (mini-batch for SGD) | TensorTonic | Easy | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 4: DEEP LEARNING BUILDING BLOCKS
# ═══════════════════════════════════════════════

> **Interview Relevance:** High — DL coding is in Flipkart's HO & DMM rounds.

## 4A: Activation Functions (ALL must be done cold)

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 59 | Sigmoid (numerically stable) | TensorTonic | Easy | 🔥⭐ |
| 60 | ReLU Activation | Both | Easy | 🔥💡 |
| 61 | Leaky ReLU | Both | Easy | 🔥💡 |
| 62 | ELU (Exponential Linear Unit) | Both | Easy | ⭐ |
| 63 | SELU | TensorTonic | Medium | ⭐ |
| 64 | GELU | Both | Medium | 🔥⭐ |
| 65 | Swish Activation | TensorTonic | Easy | ⭐ |
| 66 | Tanh Activation | TensorTonic | Easy | 🔥 |
| 67 | Softmax Function | TensorTonic | Easy | 🔥⭐ |

## 4B: Loss Functions

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 68 | Mean Squared Error (MSE) | Both | Easy | 🔥💡 |
| 69 | Huber Loss (smooth L1) | TensorTonic | Medium | ⭐ |
| 70 | Focal Loss (class-imbalanced detection) | Both | Medium | 🔥⭐ |
| 71 | Dice Loss (segmentation) | Both | Medium | ⭐ |
| 72 | Contrastive Loss (SimCLR/CL) | TensorTonic | Hard | ⭐ |
| 73 | InfoNCE Loss (contrastive) | TensorTonic | Hard | ⭐ |
| 74 | Cosine Embedding Loss | TensorTonic | Medium | ⭐ |
| 75 | Policy Gradient Loss (REINFORCE) | TensorTonic | Hard | ⭐ |
| 76 | R² Score (coefficient of determination) | TensorTonic | Easy | 🔥 |

## 4C: Neural Network Layers & Operations

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 77 | Linear Layer Forward Pass (W·x + b) | Both | Easy | 🔥⭐ |
| 78 | Batch Normalization | Both | Medium | 🔥⭐ |
| 79 | Dropout (training mode with inverted scaling) | Both | Medium | 🔥⭐ |
| 80 | Max Pooling 2D | TensorTonic | Medium | ⭐ |
| 81 | MaxPool Forward (with indices for backprop) | TensorTonic | Hard | ⭐ |
| 82 | Global Average Pooling | TensorTonic | Easy | ⭐ |
| 83 | Conv2D (image filtering/cross-correlation) | TensorTonic | Hard | 🔥⭐ |
| 84 | Simple CNN Layer | TensorTonic | Hard | ⭐ |
| 85 | RNN Step Forward | Both | Medium | 🔥⭐ |
| 86 | GRU Cell Forward | Both | Hard | 🔥⭐ |
| 87 | LSTM Cell Forward | Deep-ML | Hard | 🔥⭐ |
| 88 | AlexNet Architecture | TensorTonic | Hard | ⭐ |
| 89 | Pad Sequences (variable-length inputs) | TensorTonic | Easy | ⭐ |

## 4D: Weight Initialization

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 90 | He Initialization (for ReLU networks) | TensorTonic | Easy | 🔥⭐ |
| 91 | Xavier/Glorot Initialization | Deep-ML | Easy | 🔥 |

## 4E: Optimizers (must implement from scratch)

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 92 | SGD with Momentum | Deep-ML | Medium | 🔥 |
| 93 | Nesterov Momentum | TensorTonic | Medium | ⭐ |
| 94 | Adagrad Optimizer | Both | Medium | 🔥💡 |
| 95 | RMSProp Optimizer | TensorTonic | Medium | 🔥⭐ |
| 96 | Adam Optimizer | Both | Medium | 🔥💡 |
| 97 | AdamW Optimizer (Adam + weight decay) | TensorTonic | Medium | 🔥⭐ |
| 98 | Nadam Optimizer (Adam + Nesterov) | Both | Hard | ⭐ |

## 4F: Learning Rate Schedulers

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 99 | Linear LR Scheduler | Both | Easy | ⭐ |
| 100 | Cosine Annealing LR | TensorTonic | Medium | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 5: TRANSFORMERS & ATTENTION MECHANISMS
# ═══════════════════════════════════════════════

> **Interview Relevance:** Very High — Flipkart GenAI focus makes this critical.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 101 | Positional Encoding (sinusoidal) | Both | Medium | 🔥💡 |
| 102 | Scaled Dot-Product Attention | Deep-ML | Hard | 🔥⭐ |
| 103 | Multi-Head Attention | Deep-ML | Hard | 🔥⭐ |
| 104 | Causal Masking (for autoregressive LM) | Both | Medium | 🔥💡 |
| 105 | Cross-Entropy Loss for LM | Deep-ML | Medium | 🔥 |
| 106 | Perplexity Computation | TensorTonic | Medium | 🔥⭐ |
| 107 | BLEU Score | TensorTonic | Medium | ⭐ |
| 108 | Text Chunking (fixed size + overlap) | TensorTonic | Easy | 🔥⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 6: NLP & TEXT PROCESSING
# ═══════════════════════════════════════════════

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 109 | Bag of Words Vectorizer | Both | Easy | 🔥💡 |
| 110 | TF-IDF from Scratch | Deep-ML | Medium | 🔥⭐ |
| 111 | Word Count Dictionary | TensorTonic | Easy | 🔥 |
| 112 | Remove Stopwords | TensorTonic | Easy | ⭐ |
| 113 | Edit Distance (Levenshtein) | TensorTonic | Medium | 🔥⭐ |
| 114 | Jaccard Similarity | TensorTonic | Easy | 🔥⭐ |
| 115 | BLEU Score | TensorTonic | Hard | ⭐ |
| 116 | Perplexity (language model evaluation) | TensorTonic | Medium | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 7: EVALUATION METRICS (MUST MASTER ALL)
# ═══════════════════════════════════════════════

> **Interview Relevance:** CRITICAL — you'll be asked to implement these live.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 117 | Classification Metrics (P, R, F1, Acc) | Both | Easy | 🔥💡 |
| 118 | F1 Micro-Averaged (multi-class) | TensorTonic | Medium | ⭐ |
| 119 | Confusion Matrix (normalized) | Both | Easy | 🔥💡 |
| 120 | AUC-ROC (trapezoidal rule) | Both | Medium | 🔥💡 |
| 121 | R² Score | TensorTonic | Easy | 🔥 |
| 122 | NDCG@K (ranking metric) | TensorTonic | Hard | 🔥⭐ |
| 123 | Precision@K and Recall@K | TensorTonic | Medium | 🔥⭐ |
| 124 | Catalog Coverage (recommender) | TensorTonic | Medium | ⭐ |
| 125 | Rating Normalization (user-level) | TensorTonic | Easy | ⭐ |
| 126 | Popularity Ranking | TensorTonic | Easy | ⭐ |
| 127 | Expected Calibration Error (ECE) | Both | Medium | 🔥⭐ |
| 128 | Cohen's Kappa | TensorTonic | Medium | ⭐ |
| 129 | IoU (Intersection over Union) for boxes | Both | Medium | 🔥⭐ |
| 130 | Compute Advantage (RL) | TensorTonic | Medium | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 8: COMPUTER VISION OPERATIONS
# ═══════════════════════════════════════════════

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 131 | Color to Grayscale (luminance formula) | TensorTonic | Easy | ⭐ |
| 132 | Image Histogram | TensorTonic | Easy | ⭐ |
| 133 | Conv2D with Padding & Stride | TensorTonic | Hard | ⭐ |
| 134 | MaxPool 2D Forward | TensorTonic | Medium | ⭐ |
| 135 | Anchor Box Generation | Both | Hard | ⭐ |
| 136 | IoU Bounding Box | Both | Medium | 🔥⭐ |
| 137 | Morphological Operations (erosion/dilation) | TensorTonic | Medium | ⭐ |
| 138 | Bilinear Interpolation (image resize) | TensorTonic | Medium | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 9: REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 139 | Epsilon-Greedy Policy | TensorTonic | Easy | ⭐ |
| 140 | Monte Carlo Policy Evaluation | Both | Medium | ⭐ |
| 141 | Compute Advantage Function | TensorTonic | Medium | ⭐ |
| 142 | Policy Gradient Loss (REINFORCE) | TensorTonic | Hard | ⭐ |
| 143 | Priority Replay Sampling | TensorTonic | Hard | ⭐ |

---

# ═══════════════════════════════════════════════
# SECTION 10: MLOPS, DATA ENGINEERING & MONITORING
# ═══════════════════════════════════════════════

> **Interview Relevance:** Very High — Flipkart cares about production-ready thinking.

| # | Problem | Platform | Difficulty | Priority |
|---|---|---|---|---|
| 144 | ETL Schema Validation | TensorTonic | Medium | 🔥⭐ |
| 145 | ETL Dependency Orchestration (DAG order) | TensorTonic | Hard | 🔥⭐ |
| 146 | Monitoring Metrics Selection | TensorTonic | Medium | 🔥⭐ |
| 147 | Retraining Trigger Design | TensorTonic | Hard | 🔥⭐ |
| 148 | Data Drift Detection (PSI/KL) | Deep-ML | Hard | 🔥⭐ |
| 149 | Feature Store Backfill Logic | Deep-ML | Hard | ⭐ |

---

---

# ═══════════════════════════════════════════════
# DEEP-ML EXCLUSIVE PROBLEM CATEGORIES
# (Not in TensorTonic repos — from deep-ml.com)
# ═══════════════════════════════════════════════

> Deep-ML has 100+ problems organized into 5 categories. Key problems below.

## Deep-ML: Linear Algebra Category

| # | Problem | Difficulty |
|---|---|---|
| DML-1 | Matrix times Matrix | Easy 🔥 |
| DML-2 | Transpose of a Matrix | Easy 🔥 |
| DML-3 | Reshape Matrix | Easy |
| DML-4 | Calculate Eigenvalues | Hard ⭐ |
| DML-5 | Calculate Mean by Row or Column | Easy 🔥 |
| DML-6 | Scalar Multiplication | Easy |
| DML-7 | Cross Product | Easy |
| DML-8 | Linear Regression Using Normal Equation | Medium 🔥⭐ |
| DML-9 | Singular Value Decomposition (SVD) | Hard ⭐ |
| DML-10 | Determinant of a 4x4 Matrix | Hard |
| DML-11 | Matrix times Vector | Easy 🔥 |
| DML-12 | Linear Regression Using Gradient Descent | Medium 🔥 |
| DML-13 | Feature Scaling | Easy 🔥 |
| DML-14 | Orthogonal Projection | Hard ⭐ |
| DML-15 | Sparse Matrix Representation | Medium |
| DML-16 | Image Rotation (matrix multiplication) | Hard |

## Deep-ML: Machine Learning Category

| # | Problem | Difficulty |
|---|---|---|
| DML-17 | Logistic Regression Binary Classification | Medium 🔥⭐ |
| DML-18 | Implementing Decision Trees | Hard 🔥⭐ |
| DML-19 | K-Means Clustering | Medium 🔥⭐ |
| DML-20 | Principal Component Analysis (PCA) | Hard ⭐ |
| DML-21 | Random Forest Classification | Hard ⭐ |
| DML-22 | K-Nearest Neighbors (KNN) | Medium 🔥 |
| DML-23 | Implement AdaBoost | Hard ⭐ |
| DML-24 | Bag of Words | Easy 🔥 |
| DML-25 | TF-IDF | Medium 🔥⭐ |
| DML-26 | Softmax Activation | Easy 🔥⭐ |
| DML-27 | Ridge Regression | Medium ⭐ |
| DML-28 | Lasso Regression | Medium ⭐ |
| DML-29 | Naive Bayes Classifier | Medium 🔥 |
| DML-30 | Implement DBSCAN | Hard |
| DML-31 | Cross-Validation Data Splitter | Medium ⭐ |
| DML-32 | Confusion Matrix | Easy 🔥 |
| DML-33 | Precision, Recall, F1 Score | Easy 🔥⭐ |
| DML-34 | Mean Squared Error | Easy 🔥 |
| DML-35 | Single Neuron | Easy 🔥 |
| DML-36 | Single Neuron with Backpropagation | Medium 🔥⭐ |
| DML-37 | Feature Importance via Permutation | Medium ⭐ |
| DML-38 | Gradient Boosting Implementation | Hard ⭐ |
| DML-39 | Gini Impurity vs Entropy | Medium 🔥 |

## Deep-ML: Deep Learning Category

| # | Problem | Difficulty |
|---|---|---|
| DML-40 | Simple Conv Layer | Hard 🔥⭐ |
| DML-41 | ReLU Activation | Easy 🔥 |
| DML-42 | Sigmoid Activation | Easy 🔥 |
| DML-43 | Backpropagation for MLP | Hard 🔥⭐ |
| DML-44 | Xavier Initialization | Easy 🔥 |
| DML-45 | LSTM Cell | Hard 🔥⭐ |
| DML-46 | Transformer Positional Encoding | Medium 🔥⭐ |
| DML-47 | Multi-Head Attention | Hard 🔥⭐ |
| DML-48 | Batch Normalization | Medium 🔥⭐ |
| DML-49 | Layer Normalization | Medium 🔥⭐ |
| DML-50 | Cross-Entropy Loss | Easy 🔥 |
| DML-51 | Binary Cross-Entropy | Easy 🔥 |
| DML-52 | Leaky ReLU | Easy 🔥 |
| DML-53 | Dropout Layer | Medium 🔥⭐ |
| DML-54 | Adam Optimizer Step | Medium 🔥⭐ |
| DML-55 | Learning Rate Scheduler | Medium ⭐ |
| DML-56 | Early Stopping Criterion | Medium ⭐ |

## Deep-ML: Statistics & Probability Category

| # | Problem | Difficulty |
|---|---|---|
| DML-57 | Calculate Pearson Correlation | Medium ⭐ |
| DML-58 | Hypothesis Testing (t-test) | Medium ⭐ |
| DML-59 | Log Transform | Easy |
| DML-60 | Box-Cox Transformation | Medium |
| DML-61 | Confidence Interval | Medium ⭐ |
| DML-62 | Bayesian Update (Beta-Binomial) | Hard ⭐ |
| DML-63 | A/B Test Power Calculation | Medium 🔥⭐ |
| DML-64 | Simpson's Paradox Detection | Hard ⭐ |
| DML-65 | Bootstrap Confidence Interval | Medium ⭐ |

---

---

# ═══════════════════════════════════════════════
# SECTION 11: 🔥 TOP 30 MOST FREQUENTLY ASKED
# (FLIPKART / FAANG ML INTERVIEW HOT LIST)
# ═══════════════════════════════════════════════

> Based on cross-platform frequency, interview reports, and both platforms' "most solved" rankings.

| Rank | Problem | Why It's Hot |
|---|---|---|
| 1 | **Softmax + Cross-Entropy Loss** | Every NLP/DL system uses it; tests numerical stability |
| 2 | **Scaled Dot-Product Attention** | Core of all LLM/GenAI work; direct Flipkart signal |
| 3 | **Backpropagation for MLP** | Tests gradient chain rule mastery |
| 4 | **Linear Regression (Normal Eq. + GD)** | Most fundamental ML problem on both platforms |
| 5 | **Logistic Regression from Scratch** | Classification building block |
| 6 | **K-Means Clustering** | Unsupervised workhorse, tested on both platforms |
| 7 | **Adam Optimizer** | Modern DL standard; derivation in every DMM interview |
| 8 | **Batch Normalization** | In every DL coding interview |
| 9 | **LSTM Cell Forward** | Time series, fraud, sequential modeling |
| 10 | **Focal Loss** | Imbalanced data — fraud detection bread and butter |
| 11 | **NDCG@K** | Ranking metric, Flipkart Search/RecSys critical |
| 12 | **Precision@K, Recall@K** | RecSys evaluation standard |
| 13 | **TF-IDF from Scratch** | NLP baseline, entity matching |
| 14 | **Edit Distance (Levenshtein)** | Fuzzy matching, entity resolution |
| 15 | **Positional Encoding** | Transformer prerequisite |
| 16 | **Decision Tree Split (Info Gain/Gini)** | Tree-based models foundation |
| 17 | **AUC-ROC from scratch** | Fraud/ranking evaluation |
| 18 | **IoU (Bounding Box)** | Object detection / logistics vision systems |
| 19 | **Dropout Layer** | Regularization building block |
| 20 | **Conv2D Forward Pass** | Computer vision building block |
| 21 | **Expected Calibration Error (ECE)** | Model production monitoring |
| 22 | **Confusion Matrix (normalized)** | Classification evaluation standard |
| 23 | **PCA from SVD** | Dimensionality reduction |
| 24 | **GRU Cell Forward** | Lighter alternative to LSTM |
| 25 | **Cosine Similarity** | Embedding search, RAG retrieval |
| 26 | **Jaccard Similarity** | Entity matching, MinHashLSH |
| 27 | **ETL Dependency Orchestration** | Airflow DAG design coding |
| 28 | **Retraining Trigger Design** | MLOps production system design |
| 29 | **Text Chunking (RAG)** | LLM pipeline building block |
| 30 | **Perplexity Computation** | LLM evaluation metric |

---

---

# ═══════════════════════════════════════════════
# SECTION 12: GRIND CHEAT SHEET — MUST-KNOW IMPLEMENTATIONS
# ═══════════════════════════════════════════════

## 🧮 Linear Algebra Snippets

```python
# Matrix multiply (without numpy.matmul)
def matmul(A, B):
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2, "Dimension mismatch"
    C = [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
    return C

# Transpose
def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

# Frobenius norm
import math
def frobenius_norm(A):
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))
```

## 📉 Loss Functions Snippets

```python
import numpy as np

# Binary Cross-Entropy (numerically stable)
def bce_loss(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Focal Loss (for class imbalance)
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return np.mean(focal_weight * bce)

# Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    return np.mean(np.where(is_small, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta)))
```

## 🔄 Optimizer Snippets

```python
# Adam Optimizer Step (from scratch)
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return updated
```

## 🤖 Attention Snippet

```python
# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # (batch, heads, seq, seq)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # Fill masked positions with -inf
    weights = softmax(scores, axis=-1)
    return weights @ V

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)   # Numerically stable
    return e_x / e_x.sum(axis=axis, keepdims=True)
```

## 📊 AUC-ROC from Scratch

```python
def auc_roc(y_true, y_scores):
    """Trapezoidal rule AUC-ROC. O(n log n)."""
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list, fpr_list = [0], [0]
    P = y_true.sum()
    N = len(y_true) - P
    
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
    
    tpr_list.append(1); fpr_list.append(1)
    
    # Trapezoidal integration
    auc = sum(
        0.5 * (tpr_list[i+1] + tpr_list[i]) * (fpr_list[i+1] - fpr_list[i])
        for i in range(len(tpr_list) - 1)
    )
    return auc
```

## 📈 NDCG@K

```python
def ndcg_at_k(relevances, k):
    """
    relevances: list of relevance scores for ranked items (position 0 = top ranked)
    k: cutoff
    """
    def dcg(rel, k):
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel[:k]))
    
    actual_dcg = dcg(relevances, k)
    ideal_dcg = dcg(sorted(relevances, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

---

## 🎯 STUDY PLAN FOR HO ROUND

| Day | Topics | Priority Problems |
|---|---|---|
| Day 1 | Linear Algebra | #1-20: Transpose, MatMul, Covariance, PCA, Linear Regression Normal Eq. |
| Day 2 | Activations + Losses | #59-76: All activations, BCE, Focal, Huber, MSE |
| Day 3 | Optimizers + Layers | #77-100: Adam, BatchNorm, Dropout, LSTM, GRU |
| Day 4 | Attention + Transformers | #101-108: Attention, Positional Encoding, Causal Mask |
| Day 5 | Evaluation Metrics | #117-130: AUC, NDCG, ECE, Precision@K |
| Day 6 | ML Algorithms | #42-57: Logistic Reg, Decision Tree, K-Means from scratch |
| Day 7 | NLP + Production | #109-116, #144-147: TF-IDF, Edit Distance, ETL, Monitoring |

---

*Sources:*
- *TensorTonic via https://github.com/yitaochen/TensorTonic-Solutions (139 problems)*
- *TensorTonic via https://github.com/TranThinh2003/TensorTonic-Solutions (74 problems)*
- *Deep-ML via https://www.deep-ml.com/problems (100+ problems)*
- *Compiled and annotated for Flipkart Senior Data Scientist interview grind*
