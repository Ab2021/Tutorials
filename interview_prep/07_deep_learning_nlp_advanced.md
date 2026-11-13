# Deep Learning, NLP, & Advanced Data Science Interview Guide
## Senior & Lead Data Scientist Roles | General Mills & London Stock Exchange Group (LSEG) India

---

## Executive Overview & Interview Landscape

Senior Data Science roles at **General Mills (Mumbai/Powai Digital Hub)** and **LSEG (London Stock Exchange Group, Bangalore/Mumbai Tech Hubs)** demand a sophisticated blend of:
1. **Mathematical Depth**: First-principles understanding of neural backpropagation, matrix attention complexity, loss formulations, and gradient dynamics.
2. **Production-Grade Engineering**: Writing modular, clean, and vectorized PyTorch/NumPy code that scales to distributed training, mixed precision, and low-latency inference microservices.
3. **Domain Application**:
   - **General Mills (CPG/FMCG)**: Consumer sentiment analysis on product reviews, social listening, multi-SKU retail demand forecasting (LSTM/Transformers vs Prophet), shelf planogram computer vision.
   - **LSEG (FinTech / Financial Markets)**: Financial NLP on unstructured earnings transcripts, 10-K regulatory filings, high-throughput sentiment scoring, market anomaly detection, and hallucination-free RAG agents.
4. **MLOps & Reliability**: Continuous drift monitoring (PSI, KS-statistic), quantization (INT8/ONNX), and robust guardrails.

---

## Complete Table of Contents

- [Section 1: Deep Learning Fundamentals & PyTorch Core (Q1 - Q5)](#section-1-deep-learning-fundamentals--pytorch-core)
  - [Q1 [Medium]: Neural Network from Scratch with Backpropagation in NumPy](#q1-medium-neural-network-from-scratch-with-backpropagation-in-numpy)
  - [Q2 [Easy/Medium]: Production PyTorch Training Pipeline with AMP, Scheduler & Early Stopping](#q2-easymedium-production-pytorch-training-pipeline-with-amp-scheduler--early-stopping)
  - [Q3 [Medium]: Custom Loss Function: Focal Loss for Extreme Class Imbalance](#q3-medium-custom-loss-function-focal-loss-for-extreme-class-imbalance)
  - [Q4 [Hard]: Gradient Vanishing/Exploding, He/Xavier Initialization & PyTorch Gradient Clipping](#q4-hard-gradient-vanishingexploding-hexavier-initialization--pytorch-gradient-clipping)
  - [Q5 [Medium]: Deep Learning Model Optimization: Dynamic Quantization (INT8) & ONNX Runtime](#q5-medium-deep-learning-model-optimization-dynamic-quantization-int8--onnx-runtime)
- [Section 2: NLP Foundations & Modern Embeddings (Q6 - Q8)](#section-2-nlp-foundations--modern-embeddings)
  - [Q6 [Easy]: End-to-End Text Preprocessing & Vectorized TF-IDF from Scratch](#q6-easy-end-to-end-text-preprocessing--vectorized-tf-idf-from-scratch)
  - [Q7 [Medium]: Word2Vec Continuous Bag-of-Words (CBOW) Architecture in PyTorch](#q7-medium-word2vec-continuous-bag-of-words-cbow-architecture-in-pytorch)
  - [Q8 [Medium/Hard]: Byte-Pair Encoding (BPE) Tokenization & Sinusoidal Positional Encoding](#q8-mediumhard-byte-pair-encoding-bpe-tokenization--sinusoidal-positional-encoding)
- [Section 3: Transformer Architecture & Large Language Models (Q9 - Q13)](#section-3-transformer-architecture--large-language-models)
  - [Q9 [Hard]: Multi-Head Self-Attention Layer from Scratch in PyTorch](#q9-hard-multi-head-self-attention-layer-from-scratch-in-pytorch)
  - [Q10 [Medium]: Fine-Tuning BERT for Sequence Classification with Differential Learning Rates](#q10-medium-fine-tuning-bert-for-sequence-classification-with-differential-learning-rates)
  - [Q11 [Hard]: Financial Named Entity Recognition (Token Classification) with Hugging Face BERT](#q11-hard-financial-named-entity-recognition-token-classification-with-hugging-face-bert)
  - [Q12 [Medium/Hard]: Production Retrieval-Augmented Generation (RAG) with Dense Retrieval & FAISS](#q12-mediumhard-production-retrieval-augmented-generation-rag-with-dense-retrieval--faiss)
  - [Q13 [Medium]: Structured LLM Outputs via Function Calling & Pydantic Validation](#q13-medium-structured-llm-outputs-via-function-calling--pydantic-validation)
- [Section 4: Domain-Specific NLP: LSEG Finance & General Mills FMCG (Q14 - Q16)](#section-4-domain-specific-nlp-lseg-finance--general-mills-fmcg)
  - [Q14 [Hard]: LSEG Focus: Financial Sentiment Analysis on Earnings Calls using FinBERT](#q14-hard-lseg-focus-financial-sentiment-analysis-on-earnings-calls-using-finbert)
  - [Q15 [Medium]: General Mills Focus: FMCG Aspect-Based Sentiment Analysis (ABSA) for Product Reviews](#q15-medium-general-mills-focus-fmcg-aspect-based-sentiment-analysis-absa-for-product-reviews)
  - [Q16 [Medium]: FMCG Focus: Social Media Brand Topic Modeling with Sentence Transformers & BERTopic](#q16-medium-fmcg-focus-social-media-brand-topic-modeling-with-sentence-transformers--bertopic)
- [Section 5: Deep Learning for Time Series Forecasting (Q17 - Q19)](#section-5-deep-learning-for-time-series-forecasting)
  - [Q17 [Hard]: Multivariate LSTM for Multi-SKU Retail Demand Forecasting](#q17-hard-multivariate-lstm-for-multi-sku-retail-demand-forecasting)
  - [Q18 [Medium]: Meta Prophet vs Deep Learning: Trend Changepoints, Seasonality & Exogenous Regressors](#q18-medium-meta-prophet-vs-deep-learning-trend-changepoints-seasonality--exogenous-regressors)
  - [Q19 [Hard]: Temporal Transformer Forecaster with Multi-Horizon Prediction & Attention Masking](#q19-hard-temporal-transformer-forecaster-with-multi-horizon-prediction--attention-masking)
- [Section 6: Computer Vision in Retail & FMCG (Q20 - Q21)](#section-6-computer-vision-in-retail--fmcg)
  - [Q20 [Medium]: Retail Shelf Planogram Audit: Transfer Learning with PyTorch ResNet](#q20-medium-retail-shelf-planogram-audit-transfer-learning-with-pytorch-resnet)
  - [Q21 [Medium]: Object Detection Bounding Box Evaluation: IoU & mAP Implementation](#q21-medium-object-detection-bounding-box-evaluation-iou--map-implementation)
- [Section 7: MLOps, Drift Detection & Production Deployment (Q22 - Q24)](#section-7-mlops-drift-detection--production-deployment)
  - [Q22 [Medium]: Statistical Data Drift Detection: Population Stability Index (PSI) & KS-Test](#q22-medium-statistical-data-drift-detection-population-stability-index-psi--ks-test)
  - [Q23 [Medium]: High-Throughput FastAPI Inference Engine with Dynamic Micro-Batching](#q23-medium-high-throughput-fastapi-inference-engine-with-dynamic-micro-batching)
  - [Q24 [Medium/Hard]: LLM Guardrails: Hallucination Verification & PII Anonymization Pipeline](#q24-mediumhard-llm-guardrails-hallucination-verification--pii-anonymization-pipeline)
- [Architecture & Algorithm Cheat Sheet](#architecture--algorithm-cheat-sheet)

---

# Section 1: Deep Learning Fundamentals & PyTorch Core

---

### Q1 [Medium]: Neural Network from Scratch with Backpropagation in NumPy

#### Context & Business Framing
In senior technical interviews at both General Mills and LSEG, interviewers test if candidates understand deep learning mathematically or merely treat PyTorch/TensorFlow as black boxes. You are asked to implement a two-layer feedforward network (Dense $\to$ ReLU $\to$ Dense $\to$ Softmax) trained with categorical cross-entropy loss from scratch using pure NumPy.

#### Mathematical Foundation
For input matrix $X \in \mathbb{R}^{N \times D}$, hidden weights $W_1 \in \mathbb{R}^{D \times H}$, bias $b_1 \in \mathbb{R}^{1 \times H}$, output weights $W_2 \in \mathbb{R}^{H \times C}$, bias $b_2 \in \mathbb{R}^{1 \times C}$:
1. **Forward Pass**:
   $$Z_1 = X W_1 + b_1$$
   $$A_1 = \max(0, Z_1) \quad (\text{ReLU})$$
   $$Z_2 = A_1 W_2 + b_2$$
   $$\hat{Y}_{ij} = \frac{e^{Z_{2, ij}}}{\sum_{k=1}^C e^{Z_{2, ik}}} \quad (\text{Softmax with numerical stability shift } Z - \max(Z))$$
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{ic} \log(\hat{Y}_{ic} + \epsilon)$$

2. **Backward Pass (Chain Rule)**:
   $$\frac{\partial \mathcal{L}}{\partial Z_2} = \frac{1}{N} (\hat{Y} - Y) \in \mathbb{R}^{N \times C}$$
   $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \left(\frac{\partial \mathcal{L}}{\partial Z_2}\right), \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial Z_{2, i}}$$
   $$\frac{\partial \mathcal{L}}{\partial A_1} = \left(\frac{\partial \mathcal{L}}{\partial Z_2}\right) W_2^T$$
   $$\frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \odot \mathbb{I}(Z_1 > 0)$$
   $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \left(\frac{\partial \mathcal{L}}{\partial Z_1}\right), \quad \frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial Z_{1, i}}$$

#### Complete Python Implementation
```python
import numpy as np

class TwoLayerNeuralNet:
    """
    Two-layer Neural Network with He normal initialization, 
    ReLU activation, Softmax, and Cross-Entropy Loss.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 42):
        np.random.seed(seed)
        # He (Kaiming) Initialization for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # Xavier/Glorot Initialization for Output layer
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / (hidden_dim + num_classes))
        self.b2 = np.zeros((1, num_classes))
        
        # Cache for backprop
        self.cache = {}

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (prevents overflow in exp)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = np.dot(A1, self.W2) + self.b2
        probs = self._softmax(Z2)
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'probs': probs}
        return probs

    def compute_loss(self, probs: np.ndarray, y_true_one_hot: np.ndarray, eps: float = 1e-12) -> float:
        N = probs.shape[0]
        # Cross entropy loss
        loss = -np.sum(y_true_one_hot * np.log(probs + eps)) / N
        return float(loss)

    def backward(self, y_true_one_hot: np.ndarray) -> dict:
        X = self.cache['X']
        A1 = self.cache['A1']
        Z1 = self.cache['Z1']
        probs = self.cache['probs']
        N = X.shape[0]

        # Gradient at output layer (dZ2 = probs - Y)
        dZ2 = (probs - y_true_one_hot) / N
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0).astype(float)  # ReLU derivative
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def step(self, grads: dict, lr: float):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.05, batch_size: int = 32):
        N = X.shape[0]
        num_classes = self.W2.shape[1]
        y_one_hot = np.eye(num_classes)[y]
        
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb, yb = X_shuffled[start:end], y_shuffled[start:end]
                
                probs = self.forward(xb)
                grads = self.backward(yb)
                self.step(grads, lr)
                
            full_probs = self.forward(X)
            epoch_loss = self.compute_loss(full_probs, y_one_hot)
            acc = np.mean(np.argmax(full_probs, axis=1) == y)
            history.append((epoch_loss, acc))
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | Acc: {acc*100:.2f}%")
        return history

# Verification Run
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=600, n_features=20, n_informative=15, n_classes=3, random_state=42)
    model = TwoLayerNeuralNet(input_dim=20, hidden_dim=32, num_classes=3)
    history = model.fit(X, y, epochs=60, lr=0.1, batch_size=32)
```

#### Key Interview Follow-ups & Gotchas
- **Why subtract $\max(Z)$ before exp?**: Without this, float64 overflows at $e^{710} \approx \infty$, returning `NaN`.
- **Softmax + Cross-Entropy derivative simplification**: Interviewers love asking how $\frac{\partial \mathcal{L}}{\partial Z_2}$ simplifies so neatly to $\hat{Y} - Y$. You should explain that the Jacobian of softmax multiplied by the gradient of the log-likelihood cancels out terms, avoiding manual Kronecker delta expansions.

---

### Q2 [Easy/Medium]: Production PyTorch Training Pipeline with AMP, Scheduler & Early Stopping

#### Context & Business Framing
For senior enterprise DS roles at General Mills (demand modeling) and LSEG (text feature extractors), training on multi-gigabyte datasets requires production practices: Automatic Mixed Precision (`torch.cuda.amp`), dynamic learning rate scheduling (OneCycleLR / CosineAnnealing), gradient clipping, and robust early stopping with state checkpoints.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

class TabularDataset(Dataset):
    """Memory-efficient PyTorch Dataset for tabular financial or FMCG features."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class EarlyStopping:
    """Early stopping to prevent overfitting and checkpoint best model weights."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class ProductionMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_production_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[torch.device] = None
) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopper = EarlyStopping(patience=4)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if early_stopper(val_loss, model):
            print(f"[*] Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
            model.load_state_dict(torch.load("best_model.pt"))
            break

    return model
```

#### Key Interview Follow-ups
- **Why `optimizer.zero_grad(set_to_none=True)`?**: Setting gradients to `None` frees tensor memory immediately rather than writing zeros into existing memory buffers, reducing memory overhead and execution time by 5-10%.
- **Why AMP (`torch.cuda.amp.autocast` + `GradScaler`)?**: Half-precision (FP16) operations cut GPU VRAM in half and use Tensor Cores, but small gradients underflow to zero. `GradScaler` multiplies loss by a scale factor ($2^{16}$) before backward pass, then unscales before the optimizer step.

---

### Q3 [Medium]: Custom Loss Function: Focal Loss for Extreme Class Imbalance

#### Context & Business Framing
In financial fraud detection at LSEG or rare supply chain disruption detection at General Mills, positive instances may comprise $<0.5\%$ of records. Standard Cross-Entropy produces gradients dominated by easy negative examples. Focal Loss addresses this by down-weighting well-classified examples.

#### Mathematical Foundation
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where $p_t$ is the model's estimated probability for the ground-truth class, $\alpha_t \in [0, 1]$ balances class frequencies, and $\gamma \ge 0$ is the focusing parameter. When $\gamma = 0$, Focal Loss reduces to standard Cross-Entropy. When an example is easily classified ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, zeroing out the gradient contribution.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Numerically stable Multi-Class Focal Loss implemented in PyTorch.
    Supports class weighting (alpha) and focusing parameter (gamma).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Shape: (C,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized model outputs of shape (N, C)
            targets: Class labels of shape (N,)
        """
        # log_softmax is much more numerically stable than log(softmax(logits))
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather probability of true target class: shape (N, 1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Modulating factor: (1 - pt)^gamma
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = -modulating_factor * log_pt

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Unit Test & Numerical Validation
if __name__ == "__main__":
    torch.manual_seed(42)
    N, C = 4, 3
    logits = torch.randn(N, C, requires_grad=True)
    targets = torch.tensor([0, 2, 1, 0])
    alpha = torch.tensor([0.2, 0.5, 0.3])

    criterion = MultiClassFocalLoss(alpha=alpha, gamma=2.0)
    loss = criterion(logits, targets)
    loss.backward()

    print(f"Computed Focal Loss: {loss.item():.4f}")
    assert logits.grad is not None, "Gradient backward pass failed!"
    print("Gradients successfully propagated through Focal Loss.")
```

---

### Q4 [Hard]: Gradient Vanishing/Exploding, He/Xavier Initialization & PyTorch Gradient Clipping

#### Context & Mathematical Derivation
When training deep feedforward or recurrent networks, gradients backpropagating through $L$ layers multiply weight matrices:
$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=2}^L W_l^T \operatorname{diag}(\sigma'(z_l))$$
If the singular values of $W_l$ are $< 1$, the product exponentially collapses to 0 (Vanishing Gradients). If $> 1$, the product explodes to $\pm \infty$.

**Weight Initialization Derivation**:
To preserve variance of activations across layers $\operatorname{Var}(z_l) = \operatorname{Var}(z_{l-1})$:
- **Xavier / Glorot (for Tanh / Sigmoid)**: $W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$
- **He / Kaiming (for ReLU)**: Because ReLU zeros out half the variance ($\mathbb{E}[\max(0, x)^2] = \frac{1}{2} \operatorname{Var}(x)$), variance must be doubled:
  $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import math

class CustomLinearInit(nn.Module):
    """Demonstrates manual implementation of Kaiming Normal and Xavier Normal init."""
    def __init__(self, in_features: int, out_features: int, non_linearity: str = 'relu'):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self._initialize_weights(non_linearity)

    def _initialize_weights(self, non_linearity: str):
        in_dim = self.weight.size(1)
        out_dim = self.weight.size(0)

        with torch.no_grad():
            if non_linearity == 'relu':
                # He/Kaiming Normal: std = sqrt(2 / in_dim)
                std = math.sqrt(2.0 / in_dim)
            elif non_linearity in ['tanh', 'sigmoid']:
                # Xavier/Glorot Normal: std = sqrt(2 / (in_dim + out_dim))
                std = math.sqrt(2.0 / (in_dim + out_dim))
            else:
                std = 1.0 / math.sqrt(in_dim)
            self.weight.normal_(0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias

def custom_clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Manual implementation of torch.nn.utils.clip_grad_norm_
    Scales all gradients proportionally if total L2 norm exceeds max_norm.
    """
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    # Calculate total norm across all parameter gradients
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type
    )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)
            
    return float(total_norm)
```

---

### Q5 [Medium]: Deep Learning Model Optimization: Dynamic Quantization (INT8) & ONNX Runtime

#### Context & Business Framing
In trading systems at LSEG and real-time inventory scanners at General Mills, neural networks must evaluate in $<10\text{ms}$ on commodity CPU instances. Converting FP32 weights to INT8 reduces model footprint by 4x and accelerates inference via AVX-512/VNNI CPU vector instructions.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import os
import time

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embeds)
        return self.fc(hn[-1])

def quantize_and_benchmark():
    # 1. Instantiate FP32 model
    model_fp32 = SentimentClassifier()
    model_fp32.eval()

    # 2. PyTorch Post-Training Dynamic Quantization (Linear and LSTM layers to qint8)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )

    # 3. Model Size Comparison
    torch.save(model_fp32.state_dict(), "temp_fp32.pt")
    torch.save(model_int8.state_dict(), "temp_int8.pt")
    size_fp32 = os.path.getsize("temp_fp32.pt") / (1024 * 1024)
    size_int8 = os.path.getsize("temp_int8.pt") / (1024 * 1024)
    print(f"Model Size FP32: {size_fp32:.2f} MB | INT8: {size_int8:.2f} MB (Compression: {size_fp32/size_int8:.1f}x)")

    # 4. Latency Benchmark
    dummy_input = torch.randint(0, 5000, (64, 50))  # Batch=64, SeqLen=50
    
    # Warmup
    for _ in range(10):
        _ = model_fp32(dummy_input)
        _ = model_int8(dummy_input)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model_fp32(dummy_input)
    fp32_time = (time.perf_counter() - t0) * 10

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model_int8(dummy_input)
    int8_time = (time.perf_counter() - t0) * 10

    print(f"Avg Latency per batch (100 runs) -> FP32: {fp32_time:.2f} ms | INT8: {int8_time:.2f} ms")

    # Clean up
    os.remove("temp_fp32.pt")
    os.remove("temp_int8.pt")

if __name__ == "__main__":
    quantize_and_benchmark()
```


---

# Section 2: NLP Foundations & Modern Embeddings

---

### Q6 [Easy]: End-to-End Text Preprocessing & Vectorized TF-IDF from Scratch

#### Context & Business Framing
Before throwing a heavy transformer at a problem, both LSEG (document indexing) and General Mills (consumer reviews) rely on high-throughput sparse vector representations. Candidates must demonstrate how text normalization, n-gram generation, and term-weighting math work under the hood.

#### Mathematical Foundation
For term $t$ and document $d$ in corpus $D$:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \quad (\text{Smooth IDF)}$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Vectors are then $L_2$-normalized: $v_{\text{norm}} = \frac{v}{\|v\|_2}$.

#### Complete Python Implementation
```python
import re
import math
from collections import Counter, defaultdict
import numpy as np

class CustomTFIDFVectorizer:
    """
    Scratch implementation of TF-IDF Vectorizer with n-grams,
    sublinear TF scaling, and L2 normalization.
    """
    def __init__(self, ngram_range=(1, 1), min_df: int = 1, max_df: float = 1.0, sublinear_tf: bool = True):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = {}

    def _clean_and_tokenize(self, text: str) -> list[str]:
        # Lowercase and strip punctuation/digits
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        # Build n-grams
        min_n, max_n = self.ngram_range
        all_ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                all_ngrams.append(" ".join(tokens[i:i + n]))
        return all_ngrams

    def fit(self, raw_documents: list[str]):
        N = len(raw_documents)
        doc_freq = Counter()
        
        tokenized_corpus = []
        for doc in raw_documents:
            tokens = set(self._clean_and_tokenize(doc))
            doc_freq.update(tokens)
            tokenized_corpus.append(tokens)

        # Apply min_df and max_df filtering
        vocab = {}
        idx = 0
        for term, freq in doc_freq.items():
            df_ratio = freq / N
            if freq >= self.min_df and df_ratio <= self.max_df:
                vocab[term] = idx
                idx += 1

        self.vocabulary_ = vocab

        # Compute smooth IDF: ln((1 + N) / (1 + df)) + 1
        self.idf_ = {}
        for term, term_idx in self.vocabulary_.items():
            df = doc_freq[term]
            self.idf_[term] = math.log((1 + N) / (1 + df)) + 1.0
        return self

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        N = len(raw_documents)
        V = len(self.vocabulary_)
        matrix = np.zeros((N, V), dtype=np.float32)

        for row_idx, doc in enumerate(raw_documents):
            tokens = self._clean_and_tokenize(doc)
            term_counts = Counter(tokens)

            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    col_idx = self.vocabulary_[term]
                    # Sublinear TF: 1 + log(count)
                    tf = (1.0 + math.log(count)) if self.sublinear_tf and count > 0 else count
                    matrix[row_idx, col_idx] = tf * self.idf_[term]

            # L2 Normalization per document vector
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm

        return matrix

    def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

# Verification Run
if __name__ == "__main__":
    corpus = [
        "General Mills Cheerios cereal contains whole grain oats.",
        "Cheerios is an iconic breakfast brand by General Mills.",
        "LSEG provides real-time market data and financial analytics.",
        "Financial market participants use LSEG platforms for fixed income pricing."
    ]
    vectorizer = CustomTFIDFVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Top 5 Vocab entries: {list(vectorizer.vocabulary_.items())[:5]}")
```

---

### Q7 [Medium]: Word2Vec Continuous Bag-of-Words (CBOW) Architecture in PyTorch

#### Context & Mathematical Derivation
Continuous Bag-of-Words (CBOW) predicts a center target word $w_t$ given context words within a symmetric sliding window $C = [w_{t-k}, \dots, w_{t+k}] \setminus \{w_t\}$.
1. **Input Context Vector**:
   $$\bar{v} = \frac{1}{2k} \sum_{c \in C} v_c, \quad v_c \in \mathbb{R}^d$$
2. **Scoring & Objective**:
   $$z = W_{\text{out}} \bar{v} + b \in \mathbb{R}^{|V|}$$
   $$\mathcal{L} = -\log \frac{\exp(z_{\text{target}})}{\sum_{j=1}^{|V|} \exp(z_j)}$$
In production, negative sampling replaces full softmax to reduce gradient computation from $\mathcal{O}(|V|)$ to $\mathcal{O}(K)$.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    """
    Continuous Bag-of-Words (CBOW) PyTorch Model.
    Averages context word embeddings and projects to target vocabulary.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        # context_indices shape: (Batch_Size, 2*Window_Size)
        embeds = self.embeddings(context_indices)          # (Batch_Size, 2*Window_Size, embedding_dim)
        context_mean = torch.mean(embeds, dim=1)           # (Batch_Size, embedding_dim)
        logits = self.linear(context_mean)                 # (Batch_Size, vocab_size)
        return logits

    def get_embedding(self, word_idx: int) -> torch.Tensor:
        return self.embeddings.weight[word_idx].detach()

def build_cbow_dataset(tokenized_text: list[str], window_size: int = 2):
    word_to_idx = {w: i for i, w in enumerate(set(tokenized_text))}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    data = []
    for i in range(window_size, len(tokenized_text) - window_size):
        context = [tokenized_text[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = tokenized_text[i]
        data.append(([word_to_idx[w] for w in context], word_to_idx[target]))
    return data, word_to_idx, idx_to_word

# Training Demonstration
if __name__ == "__main__":
    raw_text = "lseg reports earnings increase and general mills boosts cereal manufacturing volume".split()
    data, w2i, i2w = build_cbow_dataset(raw_text, window_size=2)
    
    model = CBOWModel(vocab_size=len(w2i), embedding_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            ctx_tensor = torch.tensor([context], dtype=torch.long)
            target_tensor = torch.tensor([target], dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(ctx_tensor)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"Final CBOW Loss: {total_loss / len(data):.4f}")
    # Cosine similarity check between two tokens
    v1 = model.get_embedding(w2i['earnings'])
    v2 = model.get_embedding(w2i['volume'])
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    print(f"Cosine Similarity (earnings, volume): {sim:.4f}")
```

---

### Q8 [Medium/Hard]: Byte-Pair Encoding (BPE) Tokenization & Sinusoidal Positional Encoding

#### Context & Mathematical Derivation
LLMs like GPT-4 and Llama don't use words or characters; they rely on subword tokenizers (BPE).
**BPE Algorithm**:
1. Split training text into individual characters + end-of-word marker `</w>`.
2. Count frequencies of all adjacent character pairs.
3. Iteratively merge the most frequent pair and add the new merged token to vocabulary.

**Sinusoidal Positional Encoding (Vaswani et al., 2017)**:
Since self-attention is permutation-invariant, positional information must be injected:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
This allows the model to easily learn relative positions because $PE_{pos + k}$ can be expressed as a linear function of $PE_{pos}$.

#### Complete Python Implementation
```python
import collections
import torch
import torch.nn as nn
import math

class BPETokenizerTrainer:
    """Byte-Pair Encoding vocabulary learner from scratch."""
    def __init__(self, num_merges: int = 10):
        self.num_merges = num_merges
        self.merges = {}

    def get_stats(self, vocab: dict) -> dict:
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair: tuple, vocab_in: dict) -> dict:
        vocab_out = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in vocab_in.items():
            w_out = word.replace(bigram, replacement)
            vocab_out[w_out] = freq
        return vocab_out

    def train(self, corpus: list[str]):
        # Initial character-level split with </w>
        vocab = collections.defaultdict(int)
        for doc in corpus:
            for word in doc.strip().split():
                vocab[" ".join(list(word)) + " </w>"] += 1

        for i in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges[best_pair] = "".join(best_pair)
            print(f"Merge {i+1:02d}: {best_pair} -> {self.merges[best_pair]} (Freq: {pairs[best_pair]})")
        return vocab

class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. Sinusoidal Positional Encoding module."""
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        
        # div_term = exp(2i * -(ln(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as non-trainable persistent buffer
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch_Size, Seq_Len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Verification
if __name__ == "__main__":
    trainer = BPETokenizerTrainer(num_merges=5)
    sample_corpus = ["low lower newest lowest newer wide wider"]
    learned_vocab = trainer.train(sample_corpus)
    
    pos_enc = SinusoidalPositionalEncoding(d_model=64)
    dummy_tokens = torch.randn(2, 10, 64)
    encoded = pos_enc(dummy_tokens)
    print(f"Positional Encoded Output Shape: {encoded.shape}")
```

---

# Section 3: Transformer Architecture & Large Language Models

---

### Q9 [Hard]: Multi-Head Self-Attention Layer from Scratch in PyTorch

#### Context & Mathematical Derivation
The Transformer's core engine is Multi-Head Attention.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
Where:
- $Q = X W^Q, K = X W^K, V = X W^V$
- $d_k$ is the dimension per attention head: $d_k = d_{\text{model}} / h$
- $M$ is an optional causal mask where invalid future positions are filled with $-\infty$.
- $\sqrt{d_k}$ scaling prevents dot products from growing excessively large, which pushes softmax into regions with vanishingly small gradients.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Production-grade Multi-Head Self-Attention layer in PyTorch.
    Supports causal/autoregressive masking and returns attention weights.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined Q, K, V projection layer for higher memory throughput
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False, key_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_causal: If True, prevents current tokens from attending to future tokens
            key_padding_mask: Boolean tensor (batch_size, seq_len) where True indicates pad tokens
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projection: (B, S, 3 * D)
        qkv = self.qkv_projection(x)
        # Split into Q, K, V: each is (B, S, D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 2. Reshape into multi-heads: (B, H, S, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention Scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply Causal Masking (Lower Triangular)
        if is_causal:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 5. Apply Key Padding Mask
        if key_padding_mask is not None:
            # key_padding_mask shape: (B, S) -> broadcast to (B, 1, 1, S)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # 6. Softmax over last dimension
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Multiply with Values: (B, H, S, d_k)
        context = torch.matmul(attn_weights, v)

        # 8. Concatenate heads back to (B, S, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 9. Output projection
        output = self.out_projection(context)
        return output, attn_weights

# Verification
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 4, 16, 64
    x = torch.randn(B, S, D)
    mha = MultiHeadSelfAttention(d_model=D, num_heads=8)
    
    out, weights = mha(x, is_causal=True)
    print(f"MHA Output Shape: {out.shape}")
    print(f"Attention Weights Shape: {weights.shape}")
    assert out.shape == (B, S, D), "Output shape mismatch!"
```

#### Key Interview Follow-ups: FlashAttention
- **Why does standard attention scale as $\mathcal{O}(N^2)$ in memory?**: Standard attention stores the $N \times N$ attention matrix in GPU HBM (High Bandwidth Memory) during backward pass.
- **How does FlashAttention solve this?**: FlashAttention uses GPU SRAM tiling (blocks of Q, K, V loaded into fast SRAM), computes softmax incrementally without materializing the full $N \times N$ matrix in HBM, reducing memory from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ and speeding up execution by 2x to 4x.

---

### Q10 [Medium]: Fine-Tuning BERT for Sequence Classification with Differential Learning Rates

#### Context & Business Framing
When fine-tuning pre-trained models like BERT or RoBERTa for financial sentiment or consumer feedback categorization, lower layers capture general linguistic syntax while top classification layers need rapid adaptation. Interviewers expect you to configure layer-wise learning rate decay (LLRD).

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW

def build_differential_optimizer(model: nn.Module, base_lr: float = 2e-5, head_lr: float = 1e-4, weight_decay: float = 0.01):
    """
    Sets up differential learning rates: lower learning rate for transformer backbone,
    higher learning rate for the task-specific classification head.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    backbone_params_decay = []
    backbone_params_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "score" in name:
            head_params.append(param)
        elif any(nd in name for nd in no_decay):
            backbone_params_no_decay.append(param)
        else:
            backbone_params_decay.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params_decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": backbone_params_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]
    return AdamW(optimizer_grouped_parameters)

# Demonstration of loading and training setup
def mock_finetuning_pipeline():
    model_name = "bert-base-uncased"
    # Load model with 3 output labels (Negative, Neutral, Positive)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        optimizer = build_differential_optimizer(model)
        print("[+] Differential optimizer initialized successfully with 3 parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: LR={group['lr']}, Weight Decay={group['weight_decay']}, Num Tensors={len(group['params'])}")
    except Exception as e:
        print(f"Hugging Face offline or download skipped: {e}")

if __name__ == "__main__":
    mock_finetuning_pipeline()
```

---

### Q11 [Hard]: Financial Named Entity Recognition (Token Classification) with Hugging Face BERT

#### Context & Business Framing
At LSEG, parsing unstructured corporate disclosures (e.g., 10-K, press releases) requires extracting financial entities: `ORG` (Company), `MONEY` (Revenue/EBITDA), `PERCENT` (Growth), and `TICKER`. Token classification models must properly align subword tokens (WordPiece) with word-level IOB labels.

#### Complete Python Implementation
```python
import torch
from transformers import BertTokenizerFast

def align_labels_with_tokens(labels: list[str], word_ids: list[int | None], label_to_id: dict) -> list[int]:
    """
    Aligns word-level NER labels with subword WordPiece tokens.
    Assigns -100 to special tokens and non-initial subwords to ignore in PyTorch CrossEntropyLoss.
    """
    aligned_label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens [CLS], [SEP]
            aligned_label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # First subword token of a word
            label_name = labels[word_idx]
            aligned_label_ids.append(label_to_id[label_name])
        else:
            # Non-first subword of the same word (e.g., 'Cheer', '##ios')
            aligned_label_ids.append(-100)
        previous_word_idx = word_idx

    return aligned_label_ids

# Demonstration
if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Financial sentence sample
    sentence = "General Mills acquired Blue Buffalo for 8 billion dollars".split()
    ner_tags = ["B-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "O", "B-MONEY", "I-MONEY", "I-MONEY"]
    
    label_to_id = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-MONEY": 3, "I-MONEY": 4}
    
    tokenized_inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt")
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    aligned_labels = align_labels_with_tokens(ner_tags, word_ids, label_to_id)

    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
    print(f"{'Token':<15} | {'Word ID':<10} | {'Aligned Label ID':<15}")
    print("-" * 45)
    for tok, w_id, l_id in zip(tokens, word_ids, aligned_labels):
        print(f"{tok:<15} | {str(w_id):<10} | {l_id:<15}")
```

---

### Q12 [Medium/Hard]: Production Retrieval-Augmented Generation (RAG) with Dense Retrieval & FAISS

#### Context & Business Framing
LSEG and General Mills use RAG systems to query proprietary policy manuals, earnings transcripts, and supply chain SOPs. A senior DS is expected to build a production retrieval pipeline: chunking documents with overlap, computing dense embeddings, indexing via FAISS, and re-ranking candidates with a Cross-Encoder.

#### Complete Python Implementation
```python
import numpy as np
from typing import List, Dict

class VectorStoreRAG:
    """
    RAG pipeline demonstrating recursive chunking, dense embedding simulation,
    exact cosine similarity retrieval, and cross-encoder re-ranking.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.chunks = []
        self.index_matrix = None  # Normalized embeddings

    def chunk_document(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += (chunk_size - overlap)
        return chunks

    def mock_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic pseudo-embedding generator for runnable demonstration."""
        np.random.seed(42)
        embeds = []
        for text in texts:
            # Deterministic hash-based seed
            val = sum(ord(c) for c in text[:20])
            np.random.seed(val)
            vec = np.random.randn(self.embedding_dim).astype(np.float32)
            vec /= np.linalg.norm(vec)  # L2 normalize for cosine similarity
            embeds.append(vec)
        return np.vstack(embeds)

    def index_documents(self, documents: List[str]):
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        self.chunks = all_chunks
        self.index_matrix = self.mock_embed(all_chunks)
        print(f"[+] Indexed {len(self.chunks)} chunks across corpus.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.mock_embed([query])[0]
        # Inner product on normalized vectors is exact Cosine Similarity
        cosine_scores = np.dot(self.index_matrix, query_vec)
        top_indices = np.argsort(cosine_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": int(idx),
                "score": float(cosine_scores[idx]),
                "text": self.chunks[idx]
            })
        return results

    def cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Simulates Cross-Encoder re-ranking. 
        Bi-encoders do separate retrieval; Cross-encoders concatenate [Query; Chunk]
        and pass through full self-attention for precise semantic relevance scoring.
        """
        for item in candidates:
            # Bonus score if query keywords appear in chunk
            overlap_count = sum(1 for word in query.lower().split() if word in item["text"].lower())
            item["reranked_score"] = item["score"] + (0.15 * overlap_count)
            
        return sorted(candidates, key=lambda x: x["reranked_score"], reverse=True)

# Verification
if __name__ == "__main__":
    docs = [
        "General Mills reported Q3 organic net sales grew 4% driven by cereal and pet food segments including Blue Buffalo.",
        "LSEG London Stock Exchange Group acquired Refinitiv to strengthen data analytics, news distribution, and desktop terminals.",
        "Supply chain disruptions in North America impacted General Mills Häagen-Dazs ice cream production logistics."
    ]
    rag = VectorStoreRAG()
    rag.index_documents(docs)
    
    query = "What drove organic net sales growth for General Mills?"
    raw_results = rag.search(query, top_k=2)
    reranked = rag.cross_encoder_rerank(query, raw_results)
    
    print("\nTop Retrived & Re-ranked Chunk:")
    print(f"Score: {reranked[0]['reranked_score']:.4f} | Text: {reranked[0]['text']}")
```

---

### Q13 [Medium]: Structured LLM Outputs via Function Calling & Pydantic Validation

#### Context & Business Framing
In production LLM applications, parsing free-form LLM text via regex is brittle. At LSEG (extracting key financial guidance) and General Mills (extracting consumer defect reports), enterprise architectures mandate JSON schema validation using Pydantic and function calling.

#### Complete Python Implementation
```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import json

class FinancialMetric(BaseModel):
    metric_name: str = Field(description="Name of the financial KPI, e.g., Net Revenue, EBITDA")
    value: float = Field(description="Numerical value reported")
    unit: str = Field(description="Unit of measurement, e.g., USD, Millions, Percentage")
    fiscal_period: str = Field(description="Period e.g., Q3 2024, FY2023")

class EarningsReportExtraction(BaseModel):
    company_name: str = Field(description="Standardized corporation name")
    ticker: Optional[str] = Field(default=None, description="Stock ticker symbol")
    sentiment_label: str = Field(description="Overall sentiment: Positive, Neutral, or Negative")
    metrics: List[FinancialMetric] = Field(description="Extracted financial metrics")

    @field_validator("sentiment_label")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        valid = {"Positive", "Neutral", "Negative"}
        if v.capitalize() not in valid:
            raise ValueError(f"sentiment_label must be one of {valid}")
        return v.capitalize()

def simulate_llm_json_extractor(raw_transcript: str) -> EarningsReportExtraction:
    """
    Demonstrates parsing and strictly validating JSON responses from LLM APIs.
    """
    # Simulated JSON returned by an LLM function calling endpoint
    mock_llm_response = """
    {
        "company_name": "General Mills, Inc.",
        "ticker": "GIS",
        "sentiment_label": "Positive",
        "metrics": [
            {"metric_name": "Net Sales", "value": 5.1, "unit": "Billion USD", "fiscal_period": "Q3 2024"},
            {"metric_name": "Operating Profit Growth", "value": 14.0, "unit": "Percentage", "fiscal_period": "Q3 2024"}
        ]
    }
    """
    parsed_json = json.loads(mock_llm_response)
    validated_data = EarningsReportExtraction(**parsed_json)
    return validated_data

if __name__ == "__main__":
    report = simulate_llm_json_extractor("...")
    print(f"Validated Extraction for: {report.company_name} ({report.ticker})")
    print(f"Sentiment: {report.sentiment_label}")
    for m in report.metrics:
        print(f"  -> {m.metric_name}: {m.value} {m.unit} [{m.fiscal_period}]")
```


---

# Section 4: Domain-Specific NLP: LSEG Finance & General Mills FMCG

---

### Q14 [Hard]: LSEG Focus: Financial Sentiment Analysis on Earnings Calls using FinBERT

#### Context & Business Framing
General domain sentiment lexicons fail on financial text. For example:
- *"Operating liabilities liabilities declined by 14%."* $\to$ Standard NLP sees "liabilities declined" as negative; financial NLP knows this is **Positive** (debt reduction).
- FinBERT (Araci, 2019) is pre-trained on corporate 10-K/10-Q filings and earnings call transcripts. At LSEG, transcripts are parsed into two sections:
  1. **Management Presentation**: Formal, scripted guidance.
  2. **Analyst Q&A**: Unscripted, reveals underlying executive hesitation or optimism.

#### Complete Python Implementation
```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

class FinancialSentimentAnalyzer:
    """
    FinBERT inference pipeline for earnings call transcript analysis.
    Computes sentiment distribution, polarity scores, and handles text chunking.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}

    def analyze_sentences(self, sentences: List[str], batch_size: int = 16) -> List[Dict]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            for text, prob in zip(batch_texts, probs):
                top_label_idx = int(prob.argmax())
                label = self.id2label[top_label_idx]
                
                # Financial Polarity Score: P(Positive) - P(Negative) in range [-1.0, 1.0]
                # Find indices for positive and negative
                pos_idx = [k for k, v in self.id2label.items() if v.lower() == 'positive'][0]
                neg_idx = [k for k, v in self.id2label.items() if v.lower() == 'negative'][0]
                polarity = float(prob[pos_idx] - prob[neg_idx])

                results.append({
                    "sentence": text,
                    "sentiment": label,
                    "confidence": float(prob[top_label_idx]),
                    "polarity_score": round(polarity, 4),
                    "probs": {self.id2label[k]: round(float(prob[k]), 4) for k in self.id2label}
                })
        return results

    def segment_and_score_call(self, management_text: List[str], qa_text: List[str]) -> Dict:
        mgmt_scores = self.analyze_sentences(management_text)
        qa_scores = self.analyze_sentences(qa_text)

        mgmt_mean_polarity = sum(s["polarity_score"] for s in mgmt_scores) / max(len(mgmt_scores), 1)
        qa_mean_polarity = sum(s["polarity_score"] for s in qa_scores) / max(len(qa_scores), 1)

        # LSEG Signal: Management vs Q&A Sentiment Divergence
        # If Management is highly positive (+0.6) but Q&A is negative (-0.2), flags a guidance hedge risk!
        divergence = mgmt_mean_polarity - qa_mean_polarity

        return {
            "management_sentiment_score": round(mgmt_mean_polarity, 4),
            "qa_sentiment_score": round(qa_mean_polarity, 4),
            "sentiment_divergence_signal": round(divergence, 4),
            "risk_flag": divergence > 0.4
        }

# Verification Run
if __name__ == "__main__":
    try:
        analyzer = FinancialSentimentAnalyzer()
        mgmt_statements = [
            "Operating margins expanded by 180 basis points due to manufacturing efficiencies.",
            "We are raising our full-year organic net sales guidance."
        ]
        qa_statements = [
            "Raw material headwinds in packaging and wheat prices may compress margins next quarter.",
            "Retail volumes in European channels remain slightly subdued."
        ]
        report = analyzer.segment_and_score_call(mgmt_statements, qa_statements)
        print("LSEG Earnings Call Sentiment Summary:")
        print(report)
    except Exception as e:
        print(f"HuggingFace model load note: {e}")
```

---

### Q15 [Medium]: General Mills Focus: FMCG Aspect-Based Sentiment Analysis (ABSA) for Product Reviews

#### Context & Business Framing
For General Mills brands (Cheerios, Nature Valley, Pillsbury, Blue Buffalo), an overall 3-star rating on Amazon or Walmart.com is uninformative. The product team needs granular signals:
- *“The cereal tastes great, but the box arrived crushed and stale.”*
  - **Taste**: Positive
  - **Packaging**: Negative
  - **Freshness**: Negative

#### Complete Python Implementation
```python
import re
from typing import List, Dict

class FMCGAspectSentimentMiner:
    """
    Rule-guided and syntactic dependency aspect extractor for FMCG consumer reviews.
    Extracts aspects (Taste, Packaging, Price, Texture) and associates modifier polarities.
    """
    ASPECT_LEXICON = {
        "taste": ["taste", "flavor", "sweetness", "delicious", "tasty", "bland", "salty"],
        "packaging": ["box", "bag", "seal", "package", "crushed", "torn", "leak", "container"],
        "texture": ["crunchy", "crispy", "soggy", "hard", "chewy", "stale", "texture"],
        "value": ["price", "cost", "expensive", "cheap", "worth", "value", "deal"]
    }

    POSITIVE_WORDS = {"good", "great", "excellent", "love", "loved", "fresh", "crisp", "tasty", "delicious", "worth"}
    NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "stale", "soggy", "crushed", "broken", "expensive", "bland"}

    def __init__(self):
        # Reverse lookup map
        self.term_to_aspect = {}
        for aspect, keywords in self.ASPECT_LEXICON.items():
            for kw in keywords:
                self.term_to_aspect[kw] = aspect

    def extract_aspect_sentiment(self, review_text: str) -> Dict[str, Dict]:
        # Split text into clauses/sentences
        clauses = re.split(r'[,.;!]+|but|however|although', review_text.lower())
        aspect_scores = {aspect: {"positive": 0, "negative": 0} for aspect in self.ASPECT_LEXICON}

        for clause in clauses:
            words = clause.strip().split()
            found_aspects = set()
            
            # Detect aspects in this clause
            for word in words:
                if word in self.term_to_aspect:
                    found_aspects.add(self.term_to_aspect[word])

            if not found_aspects:
                continue

            # Determine sentiment of the clause
            pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
            neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
            
            # Negation flipping (e.g. "not good")
            has_negation = any(neg in words for neg in ["not", "never", "no", "hardly"])
            if has_negation:
                pos_count, neg_count = neg_count, pos_count

            for aspect in found_aspects:
                aspect_scores[aspect]["positive"] += pos_count
                aspect_scores[aspect]["negative"] += neg_count

        # Compute net polarity per aspect
        summary = {}
        for aspect, counts in aspect_scores.items():
            net = counts["positive"] - counts["negative"]
            if counts["positive"] + counts["negative"] > 0:
                sentiment = "Positive" if net > 0 else ("Negative" if net < 0 else "Neutral")
                summary[aspect] = {
                    "sentiment": sentiment,
                    "net_polarity": net,
                    "pos_mentions": counts["positive"],
                    "neg_mentions": counts["negative"]
                }
        return summary

# Verification
if __name__ == "__main__":
    miner = FMCGAspectSentimentMiner()
    sample_review = "The cereal flavor is delicious and fresh, but the box arrived completely crushed and not worth the expensive price."
    print("Extracted Aspect Breakdown:")
    for aspect, data in miner.extract_aspect_sentiment(sample_review).items():
        print(f"  [{aspect.upper()}] -> Sentiment: {data['sentiment']} (Pos: {data['pos_mentions']}, Neg: {data['neg_mentions']})")
```

---

### Q16 [Medium]: FMCG Focus: Social Media Brand Topic Modeling with Sentence Transformers & BERTopic

#### Context & Mathematical Foundation
Traditional LDA (Latent Dirichlet Allocation) fails on short, noisy social media tweets or Reddit threads because documents share few exact word co-occurrences.
**BERTopic / Neural Topic Modeling Architecture**:
1. **Sentence Embeddings**: Pre-trained Transformer (`all-MiniLM-L6-v2`) produces dense representations $e_i \in \mathbb{R}^{384}$.
2. **Dimensionality Reduction**: UMAP projects high-dimensional embeddings to 5D space to retain local manifold structure.
3. **Density Clustering**: HDBSCAN clusters dense areas and separates noise.
4. **Class-based TF-IDF (c-TF-IDF)**:
   $$W_{t, c} = \|\text{tf}_{t, c}\| \times \log\left(1 + \frac{A}{f_t}\right)$$
   Where $\text{tf}_{t, c}$ is frequency of term $t$ in cluster $c$, $f_t$ is frequency across all clusters, and $A$ is the average number of words per cluster.

#### Complete Python Implementation
```python
import numpy as np
from collections import Counter
import math

class ScratchCTFIDF:
    """
    Class-based TF-IDF (c-TF-IDF) implementation from scratch.
    Aggregates text chunks by cluster and identifies top defining keywords per topic.
    """
    def __init__(self, top_n_words: int = 5):
        self.top_n_words = top_n_words
        self.topic_words_ = {}

    def fit_transform(self, documents: list[str], labels: list[int]) -> dict[int, list[tuple[str, float]]]:
        unique_labels = [lbl for lbl in set(labels) if lbl != -1]  # Exclude noise label -1
        
        # 1. Concatenate all documents in each cluster into a single class-document
        cluster_docs = {lbl: [] for lbl in unique_labels}
        for doc, lbl in zip(documents, labels):
            if lbl != -1:
                cluster_docs[lbl].extend(doc.lower().split())

        # 2. Compute total word counts across all clusters
        all_words = []
        for words in cluster_docs.values():
            all_words.extend(words)
        total_vocab = set(all_words)
        global_word_freq = Counter(all_words)

        avg_words_per_cluster = np.mean([len(w) for w in cluster_docs.values()])

        # 3. Compute c-TF-IDF scores
        for lbl in unique_labels:
            words = cluster_docs[lbl]
            cluster_tf = Counter(words)
            scores = {}

            for word, tf in cluster_tf.items():
                # c-TF-IDF formula
                idf = math.log(1.0 + (avg_words_per_cluster / (global_word_freq[word] + 1e-9)))
                scores[word] = tf * idf

            # Extract top N keywords for this cluster
            top_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:self.top_n_words]
            self.topic_words_[lbl] = top_words

        return self.topic_words_

# Demonstration
if __name__ == "__main__":
    # Simulated FMCG tweets / social mentions
    tweets = [
        "Cheerios heart healthy oats breakfast breakfast oats",
        "love the oats and healthy Cheerios breakfast",
        "Pillsbury cookie dough baking fresh warm cookies",
        "baking fresh warm Pillsbury chocolate chip dough",
        "Nature Valley protein bar granola snack snack",
        "healthy protein snack bar on the go"
    ]
    # Simulated cluster labels from HDBSCAN
    pseudo_clusters = [0, 0, 1, 1, 2, 2]

    ctfidf = ScratchCTFIDF(top_n_words=3)
    topics = ctfidf.fit_transform(tweets, pseudo_clusters)
    for topic_id, words in topics.items():
        print(f"Topic {topic_id}: {[w for w, score in words]}")
```

---

# Section 5: Deep Learning for Time Series Forecasting

---

### Q17 [Hard]: Multivariate LSTM for Multi-SKU Retail Demand Forecasting

#### Context & Business Framing
At General Mills, forecasting weekly sales demand for 10,000+ SKUs across Walmart, Target, and Kroger requires multivariate deep learning models. Key features include historical sales lags, promotional discounts (`on_promo`), markdown percentages, and calendar seasonality.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesWindowDataset(Dataset):
    """
    Creates rolling lookback windows for multivariate sequential forecasting.
    Input: (lookback_len, num_features) -> Output: (forecast_horizon, num_targets)
    """
    def __init__(self, data: np.ndarray, lookback: int = 12, horizon: int = 4):
        self.X, self.y = [], []
        # data format: column 0 is target demand, remaining columns are features
        for i in range(len(data) - lookback - horizon + 1):
            self.X.append(data[i:i + lookback, :])
            self.y.append(data[i + lookback:i + lookback + horizon, 0])  # Target: next 'horizon' sales

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class MultivariateDemandLSTM(nn.Module):
    """
    Production Multivariate LSTM Forecaster with Dropout and Multi-Horizon Linear Head.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 2, horizon: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon)  # Direct multi-step output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch_Size, Lookback, Num_Features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take hidden state of last time step: (Batch_Size, hidden_dim)
        last_hidden = lstm_out[:, -1, :]
        forecast = self.fc(last_hidden)
        return forecast

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    # 52 weeks of synthetic retail data with 4 features: [Sales, Price, PromoFlag, Seasonality]
    weeks = 104
    synthetic_sales = 500 + 50 * np.sin(np.linspace(0, 8*np.pi, weeks)) + np.random.randn(weeks) * 10
    price = 3.99 - np.random.choice([0.0, 0.5, 1.0], size=weeks, p=[0.7, 0.2, 0.1])
    promo = (price < 3.5).astype(float)
    seasonality = np.cos(np.linspace(0, 4*np.pi, weeks))
    
    dataset_matrix = np.column_stack([synthetic_sales, price, promo, seasonality])
    ds = TimeSeriesWindowDataset(dataset_matrix, lookback=12, horizon=4)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = MultivariateDemandLSTM(num_features=4, hidden_dim=32, num_layers=2, horizon=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(15):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Train MSE Loss: {total_loss / len(loader):.4f}")
```

---

### Q18 [Medium]: Meta Prophet vs Deep Learning: Trend Changepoints, Seasonality & Exogenous Regressors

#### Context & Business Framing
In senior interviews, interviewers frequently ask: *"When should you choose Prophet over an LSTM/Transformer, and how do both handle retail promotion shocks?"*
- **Prophet's Decomposable Additive Model**:
  $$y(t) = g(t) + s(t) + h(t) + \sum_{j} \beta_j X_{j, t} + \epsilon_t$$
  Where $g(t)$ is piecewise linear or logistic growth with $L_1$-regularized changepoints, $s(t)$ is Fourier seasonality, $h(t)$ is holiday effects, and $X_{j, t}$ are user regressors (e.g., trade promotions).
- **Trade-off Summary**:
  - **Prophet**: Interpretable parameters, handles missing dates effortlessly, fast CPU inference, fits individual SKU curves without big data.
  - **Deep Learning (LSTM/TFT)**: Capable of learning cross-series correlations (cannibalization across SKUs), handles complex non-linear feature interactions, but requires larger data volumes and GPU training.

#### Complete Python Implementation
```python
import pandas as pd
import numpy as np

def generate_retail_sku_series(days: int = 365) -> pd.DataFrame:
    """Generates synthetic FMCG daily sales data with holiday shocks and promo flags."""
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    base_demand = 1200
    # Day-of-week seasonality (highest on Saturday/Sunday)
    dow_effect = np.array([0, 10, 20, 30, 80, 250, 200])[dates.dayofweek]
    
    # Promotional discounts (binary flag)
    np.random.seed(42)
    promo_flag = np.random.choice([0, 1], size=days, p=[0.85, 0.15])
    promo_lift = promo_flag * 450.0

    noise = np.random.normal(0, 30, size=days)
    sales = base_demand + dow_effect + promo_lift + noise

    return pd.DataFrame({
        "ds": dates,
        "y": sales,
        "is_promotion": promo_flag
    })

def fit_prophet_model(df: pd.DataFrame):
    """
    Prophet pipeline configured with retail promotion regressor and custom seasonality.
    """
    try:
        from prophet import Prophet
        
        # Initialize Prophet with explicit changepoint sensitivity
        m = Prophet(
            growth='linear',
            changepoint_prior_scale=0.05,  # Higher = more flexible trend
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        # Add trade promotion as exogenous regressor
        m.add_regressor('is_promotion', prior_scale=10.0, mode='additive')
        
        m.fit(df)
        
        # Future dataframe for next 14 days
        future = m.make_future_dataframe(periods=14)
        # Assume promotions on weekend days in future horizon
        future['is_promotion'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        
        forecast = m.predict(future)
        print("[+] Prophet Model Fitted Successfully.")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'is_promotion']].tail(5))
        return m, forecast
    except ImportError:
        print("[*] prophet library not installed. Simulating decomposed regression output:")
        print(f"Mean Sales: {df['y'].mean():.2f} | Promo Lift: ~450 units")

if __name__ == "__main__":
    df = generate_retail_sku_series()
    fit_prophet_model(df)
```

---

### Q19 [Hard]: Temporal Transformer Forecaster with Multi-Horizon Prediction & Attention Masking

#### Context & Mathematical Foundation
Unlike NLP where attention is over token vocabularies, Time-Series Transformers apply attention over continuous temporal feature vectors.
1. **Input Projection**: Projects continuous features $x_t \in \mathbb{R}^F$ to $d_{\text{model}}$ via 1D Conv or Linear layer.
2. **Causal Attention Mask**: Ensures that when forecasting at time $t$, the model cannot attend to future periods $t' > t$.
3. **Temporal Self-Attention**:
   $$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}} + M_{\text{causal}}\right) V$$

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    """
    Time Series Transformer Forecaster with Feature Projection,
    Temporal Encoding, Transformer Encoder Stack, and Direct Multi-Step Head.
    """
    def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4, num_layers: int = 2, forecast_horizon: int = 7, dropout: float = 0.1):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = TemporalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Input_Dim)
        x_proj = self.feature_proj(x)
        x_encoded = self.pos_encoder(x_proj)
        
        # Transformer representation
        memory = self.transformer_encoder(x_encoded)
        
        # Global pooling across sequence dimension
        pooled_repr = torch.mean(memory, dim=1)
        predictions = self.head(pooled_repr)  # (Batch, forecast_horizon)
        return predictions

# Verification
if __name__ == "__main__":
    B, S, F = 16, 30, 5  # Batch=16, 30-day lookback, 5 economic/FMCG features
    dummy_ts = torch.randn(B, S, F)
    ts_model = TimeSeriesTransformer(input_dim=F, d_model=32, n_heads=4, num_layers=2, forecast_horizon=7)
    
    forecasts = ts_model(dummy_ts)
    print(f"Transformer 7-day Ahead Forecast Shape: {forecasts.shape}")
    assert forecasts.shape == (B, 7), "Forecast shape invalid!"
```


---

# Section 6: Computer Vision in Retail & FMCG

---

### Q20 [Medium]: Retail Shelf Planogram Audit: Transfer Learning with PyTorch ResNet

#### Context & Business Framing
At General Mills, retail execution teams audit supermarket shelves to ensure brands (Cheerios, Nature Valley, Pillsbury) occupy agreed-upon shelf slots (Planogram compliance) and are not out of stock. Computer vision models trained on mobile phone photos taken by field reps classify packaging types and detect misplaced SKUs.

#### Complete Python Implementation
```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FMCGShelfClassifier:
    """
    Transfer Learning pipeline using ResNet-50 for retail SKU classification.
    Freezes convolutional backbone and fine-tunes custom multi-class head.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        # Freeze all feature extractor layers to preserve ImageNet representations
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classification head with custom classifier for FMCG SKUs
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.model = self.model.to(self.device)

    @staticmethod
    def get_data_transforms():
        """Data augmentation pipeline for retail shelf photos under varying store lighting."""
        return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def train_head_only(self, train_loader: DataLoader, epochs: int = 3, lr: float = 1e-3):
        # Only parameters of the new fc head require gradients
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1:02d} | Head Loss: {total_loss / max(len(train_loader), 1):.4f}")

    def unfreeze_top_layers(self, num_layers_to_unfreeze: int = 2):
        """Allows fine-tuning top residual blocks (layer4) with lower learning rate."""
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        print("[+] Unfroze ResNet layer4 for end-to-end fine-tuning.")

# Verification Run
if __name__ == "__main__":
    classifier = FMCGShelfClassifier(num_classes=5, pretrained=False)
    dummy_imgs = torch.randn(4, 3, 224, 224)
    out = classifier.model(dummy_imgs)
    print(f"Shelf Classifier Output Logits Shape: {out.shape}")
```

---

### Q21 [Medium]: Object Detection Bounding Box Evaluation: IoU & mAP Implementation

#### Context & Mathematical Foundation
In retail shelf audits, bounding boxes are predicted around cereal boxes or snack bars.
1. **Intersection over Union (IoU)**:
   $$\text{IoU} = \frac{\text{Area}(\text{Box}_A \cap \text{Box}_B)}{\text{Area}(\text{Box}_A \cup \text{Box}_B)} = \frac{\text{Area}(\text{Intersection})}{\text{Area}_A + \text{Area}_B - \text{Area}(\text{Intersection})}$$
2. **Non-Maximum Suppression (NMS)**:
   Discards overlapping duplicate boxes that have $\text{IoU} > \text{threshold}$ with a higher-confidence box.

#### Complete Python Implementation
```python
import numpy as np

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes IoU between two bounding boxes in [x1, y1, x2, y2] format.
    """
    # Coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0.0

def non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """
    Scratch implementation of Non-Maximum Suppression (NMS).
    Returns list of preserved box indices.
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break

        # Compute IoU between top box and remaining candidate boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes with IoU less than threshold
        remaining_indices = np.where(ious <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return keep

# Demonstration
if __name__ == "__main__":
    # Boxes: [x1, y1, x2, y2]
    candidate_boxes = np.array([
        [50, 50, 150, 200],    # Box A: High score
        [52, 48, 148, 198],    # Box B: High overlap with A
        [220, 50, 320, 200]    # Box C: Separate object
    ])
    confidences = np.array([0.95, 0.88, 0.91])

    kept_indices = non_maximum_suppression(candidate_boxes, confidences, iou_threshold=0.5)
    print(f"Indices retained after NMS: {kept_indices}")
    print(f"Retained Boxes: {candidate_boxes[kept_indices].tolist()}")
```

---

# Section 7: MLOps, Drift Detection & Production Deployment

---

### Q22 [Medium]: Statistical Data Drift Detection: Population Stability Index (PSI) & KS-Test

#### Context & Mathematical Foundation
Once models are deployed (e.g., LSEG credit score models or General Mills demand predictors), feature distributions shift over time.
1. **Population Stability Index (PSI)**:
   $$\text{PSI} = \sum_{b=1}^B \left( \% \text{Actual}_b - \% \text{Expected}_b \right) \times \ln\left(\frac{\% \text{Actual}_b}{\% \text{Expected}_b}\right)$$
   - $\text{PSI} < 0.1$: No significant drift; model is stable.
   - $0.1 \le \text{PSI} \le 0.25$: Moderate drift; warning triggered.
   - $\text{PSI} > 0.25$: Significant population drift; model retraining required.
2. **Kolmogorov-Smirnov (KS) Test**:
   Two-sample test calculating maximum vertical divergence between baseline and current empirical CDFs:
   $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$

#### Complete Python Implementation
```python
import numpy as np
from scipy.stats import ks_2samp

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> tuple[float, str]:
    """
    Computes Population Stability Index (PSI) using quantile binning on baseline.
    """
    # 1. Compute quantiles on baseline to create equal-frequency bins
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(baseline, percentiles)
    # Ensure distinct bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # 2. Count observed counts in each bucket
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # 3. Convert to percentages with epsilon smoothing to prevent div by zero
    eps = 1e-6
    expected_pct = (baseline_counts / len(baseline)) + eps
    actual_pct = (current_counts / len(current)) + eps

    # Normalize back to 1.0
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # 4. PSI Formula
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    if psi_value < 0.10:
        status = "STABLE: No significant drift detected."
    elif psi_value <= 0.25:
        status = "WARNING: Moderate drift detected. Monitor closely."
    else:
        status = "ALERT: Critical population drift detected. Retraining mandated!"

    return float(psi_value), status

def evaluate_drift_report(feature_name: str, baseline_data: np.ndarray, production_data: np.ndarray):
    psi, status = calculate_psi(baseline_data, production_data)
    ks_stat, p_val = ks_2samp(baseline_data, production_data)
    
    print(f"\n--- Drift Diagnostics for [{feature_name}] ---")
    print(f"PSI Metric: {psi:.4f} -> {status}")
    print(f"KS-Statistic: {ks_stat:.4f} | p-value: {p_val:.4e} {'(Drift p < 0.05)' if p_val < 0.05 else ''}")

# Demonstration
if __name__ == "__main__":
    np.random.seed(42)
    baseline_demand = np.random.normal(loc=100, scale=15, size=2000)
    current_stable = np.random.normal(loc=101, scale=15, size=1500)
    current_drifted = np.random.normal(loc=125, scale=22, size=1500)

    evaluate_drift_report("Weekly Cereal Unit Sales (Stable Period)", baseline_demand, current_stable)
    evaluate_drift_report("Weekly Cereal Unit Sales (Inflation/Promotion Shock)", baseline_demand, current_drifted)
```

---

### Q23 [Medium]: High-Throughput FastAPI Inference Engine with Dynamic Micro-Batching

#### Context & Business Framing
In production at LSEG, millions of news headlines stream per minute. Serving predictions one sentence at a time creates massive GPU under-utilization and CPU-GPU transfer bottlenecks. A Senior Data Scientist must understand dynamic request queueing and micro-batching.

#### Complete Python Implementation
```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import asyncio
import time
from typing import List

app = FastAPI(title="LSEG Financial News Scoring Service", version="1.0.0")

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    sentiment: str
    score: float
    latency_ms: float

class DynamicBatcher:
    """
    Asynchronous micro-batcher that groups incoming concurrent HTTP requests
    into mini-batches before invoking the GPU neural model.
    """
    def __init__(self, max_batch_size: int = 32, max_wait_sec: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_sec
        self.queue: List[tuple[str, asyncio.Future]] = []
        self.lock = asyncio.Lock()
        self.worker_task = None

    async def predict(self, text: str) -> dict:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        async with self.lock:
            self.queue.append((text, future))
            if len(self.queue) >= self.max_batch_size:
                # Trigger batch immediately
                self._dispatch_batch()
            elif self.worker_task is None or self.worker_task.done():
                self.worker_task = asyncio.create_task(self._wait_and_dispatch())

        return await future

    async def _wait_and_dispatch(self):
        await asyncio.sleep(self.max_wait_sec)
        async with self.lock:
            self._dispatch_batch()

    def _dispatch_batch(self):
        if not self.queue:
            return
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        texts = [b[0] for b in batch]
        futures = [b[1] for b in batch]

        # Execute batched inference
        results = self._mock_gpu_forward(texts)
        for fut, res in zip(futures, results):
            if not fut.done():
                fut.set_result(res)

    def _mock_gpu_forward(self, texts: List[str]) -> List[dict]:
        # Batch tensor inference simulated
        return [{"sentiment": "POSITIVE" if "gain" in t.lower() else "NEUTRAL", "score": 0.88} for t in texts]

batcher = DynamicBatcher()

@app.post("/predict", response_model=InferenceResponse)
async def predict_sentiment(req: InferenceRequest):
    t0 = time.perf_counter()
    res = await batcher.predict(req.text)
    latency = (time.perf_counter() - t0) * 1000.0
    return InferenceResponse(sentiment=res["sentiment"], score=res["score"], latency_ms=round(latency, 2))

@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "gpu_available": False}
```

---

### Q24 [Medium/Hard]: LLM Guardrails: Hallucination Verification & PII Anonymization Pipeline

#### Context & Business Framing
In enterprise finance (LSEG) and corporate consumer databases (General Mills), LLMs must never expose PII (Personally Identifiable Information like PAN, credit card numbers, or names) and must never hallucinate financial commitments or product ingredient claims.

#### Complete Python Implementation
```python
import re
from typing import Dict, Any

class LLMGuardrailPipeline:
    """
    Production guardrail engine combining regex-based PII redaction
    and context-faithfulness verification.
    """
    PII_REGEX_PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE_IN": r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b',
        "PAN_IN": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
    }

    def sanitize_pii(self, text: str) -> tuple[str, Dict[str, int]]:
        redacted_text = text
        redaction_counts = {}
        for pii_type, pattern in self.PII_REGEX_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                redaction_counts[pii_type] = len(matches)
                redacted_text = re.sub(pattern, f"<REDACTED_{pii_type}>", redacted_text)
        return redacted_text, redaction_counts

    def verify_faithfulness_heuristic(self, generated_answer: str, source_context: str) -> Dict[str, Any]:
        """
        NLI / Heuristic Faithfulness verification:
        Extracts numbers and entities from generated answer and checks if they exist in source context.
        """
        # Extract numerical quantities and percentages from answer
        answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', generated_answer))
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', source_context))

        unsupported_facts = answer_numbers - context_numbers
        is_faithful = len(unsupported_facts) == 0

        return {
            "is_faithful": is_faithful,
            "hallucination_risk": "HIGH" if not is_faithful else "LOW",
            "unsupported_numerical_claims": list(unsupported_facts)
        }

# Verification
if __name__ == "__main__":
    guard = LLMGuardrailPipeline()
    
    # 1. Test PII Redaction
    raw_user_prompt = "Contact analyst Rahul at rahul.kumar@lseg.com or mobile +91 9876543210 with PAN ABCDE1234F"
    clean_prompt, counts = guard.sanitize_pii(raw_user_prompt)
    print("Sanitized Prompt:")
    print(clean_prompt)
    print(f"Redactions: {counts}")

    # 2. Test Hallucination Verification
    context = "General Mills Q3 operating profit reached 820 million USD with 5% margin increase."
    good_answer = "The reported operating profit was 820 million USD."
    hallucinated_answer = "The operating profit was 950 million USD with 12% margin increase."

    check1 = guard.verify_faithfulness_heuristic(good_answer, context)
    check2 = guard.verify_faithfulness_heuristic(hallucinated_answer, context)
    
    print("\nFaithfulness Check 1 (Accurate):", check1)
    print("Faithfulness Check 2 (Hallucinated):", check2)
```

---

# Architecture & Algorithm Cheat Sheet

| Architecture / Technique | Primary Use Case | Time Complexity (Train) | Memory Footprint | Key Advantage | Major Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | Short sequential patterns | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(T)$ | Simple recurrent state | Vanishing/exploding gradients |
| **LSTM / GRU** | FMCG Demand Forecasting, Sequential Tabular | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(T)$ | Cell state gates prevent vanishing gradient | Sequential compute; cannot parallelize across time |
| **Transformer (Standard)** | NLP, Long-Horizon Forecasting | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T^2)$ | Full parallel training, global self-attention | Quadratic memory bottleneck for long sequences |
| **FlashAttention-2** | Large Language Models (LLM) | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T)$ (SRAM tiling) | 2x-4x faster wall-clock, linear GPU memory | Requires modern CUDA hardware (A100/H100/L4) |
| **Mamba / SSM** | Ultra-long document parsing | $\mathcal{O}(T \cdot d)$ | $\mathcal{O}(T)$ | Linear scaling in sequence length | Less mature ecosystem for fine-tuning |
| **Meta Prophet** | Single-SKU business forecasting | $\mathcal{O}(N \cdot K)$ (L-BFGS / MAP) | Minimal (CPU-friendly) | Decomposable trend & holiday interpretation | Does not capture cross-SKU cannibalization |

---

## Senior Data Scientist Interview Preparation Checklist

### 1. Mathematical Rigor (Whiteboard & Coding)
- [ ] Derive backpropagation equations for a 2-layer MLP on paper.
- [ ] Explain why self-attention divides by $\sqrt{d_k}$.
- [ ] Contrast Weight Decay in Adam vs AdamW (L2 penalty vs decoupled parameter decay).
- [ ] Explain the difference between Data Parallel (DP) and Distributed Data Parallel (DDP).

### 2. Domain Preparation: General Mills (CPG / FMCG)
- [ ] Be ready to discuss multivariate demand forecasting with trade promotions.
- [ ] Explain how you would evaluate out-of-stock risk using asymmetric loss functions (penalizing stockouts more than overstock).
- [ ] Describe consumer review topic mining and aspect-based sentiment pipelines.

### 3. Domain Preparation: LSEG (Financial Markets)
- [ ] Explain how FinBERT differs from general BERT in tokenization and financial sentiment polarity.
- [ ] Walk through a production RAG system architecture for regulatory filings (10-K, 10-Q).
- [ ] Discuss latency optimization: ONNX Runtime, Dynamic Quantization, and vLLM PagedAttention.
- [ ] Detail statistical drift monitoring (PSI thresholds, KS two-sample test).
