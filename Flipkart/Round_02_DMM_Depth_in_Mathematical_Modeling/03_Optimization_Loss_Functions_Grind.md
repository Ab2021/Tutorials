# 📉 DMM GRIND — Optimization & Custom Loss Functions (Questions 76-100)
### Depth in Mathematical Modeling

> At Senior/Staff levels, you aren't just calling `model.compile(loss='mse')`. You design custom loss functions and optimize them.

---

## ═══════════════════════════════════════
## SECTION H: OPTIMIZATION & LOSS FUNCTIONS
## ═══════════════════════════════════════

### Q76: Derive the Softmax and Cross-Entropy gradients. Why do they work so well together?
**Expected Answer:**
**Softmax:** $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
**Cross-Entropy:** $L = -\sum_k y_k \log(\hat{y}_k)$ (where $y$ is one-hot).
Gradient w.r.t logits $z_i$:
By chain rule: $\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_i}$
We know $\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$.
Softmax Jacobian:
If $i = k$: $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$.
If $i \neq k$: $\frac{\partial \hat{y}_k}{\partial z_i} = -\hat{y}_k \hat{y}_i$.
Substitute into chain rule:
$$\frac{\partial L}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \hat{y}_i(1 - \hat{y}_i) - \sum_{k \neq i} \frac{y_k}{\hat{y}_k} (-\hat{y}_k \hat{y}_i)$$
$$= -y_i + y_i\hat{y}_i + \sum_{k \neq i} y_k \hat{y}_i = -y_i + \hat{y}_i \sum_k y_k$$
Since $y$ is a probability distribution, $\sum y_k = 1$.
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$
**Why it's beautiful:** The gradient is simply the error (prediction - target). If you predict 0.99 and target is 1.0, gradient is -0.01 (small nudge). If you predict 0.1 and target is 1.0, gradient is -0.9 (huge nudge). It naturally scales the learning signal!

### Q77: Design a custom loss function for an asymmetric business problem: Missing fraud costs ₹10,000, False positive costs ₹500.
**Expected Answer:**
Standard Binary Cross-Entropy (BCE) treats False Positives (FP) and False Negatives (FN) equally.
Let $y \in \{0,1\}$ (1 is fraud).
Let $c_{FN} = 10000$ and $c_{FP} = 500$.
**Cost-Sensitive Loss:**
$$L(y, \hat{y}) = - [ c_{FN} \cdot y \log(\hat{y}) + c_{FP} \cdot (1-y) \log(1-\hat{y}) ]$$
This forces the model to heavily penalize predicting a low probability when $y=1$.
Alternatively, for tree-based models like XGBoost, you write a custom objective function returning the gradient and hessian. The gradient of this cost-sensitive loss is:
$$g = \frac{\partial L}{\partial \hat{y}} = - \frac{c_{FN} \cdot y}{\hat{y}} + \frac{c_{FP} \cdot (1-y)}{1-\hat{y}}$$
Using the log-odds (logit $z$) simplifies the math to $g = c_{FN} \cdot y(\hat{y}-1) + c_{FP} \cdot (1-y)\hat{y}$.

### Q78: Explain Contrastive Loss vs Triplet Loss. When to use which in Representation Learning?
**Expected Answer:**
**Contrastive Loss (SimCLR, CLIP):** Operates on pairs $(x_i, x_j)$. If they are similar (positive pair, $y=1$), pull embeddings together. If dissimilar ($y=0$), push apart up to a margin $m$.
$$L_{cont} = y \|f(x_i) - f(x_j)\|^2 + (1-y) \max(0, m - \|f(x_i) - f(x_j)\|)^2$$
**Triplet Loss (FaceNet):** Operates on triplets (Anchor, Positive, Negative). Forces distance(A,P) + margin < distance(A,N).
$$L_{triplet} = \max(0, \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \text{margin})$$
**Comparison:**
- Triplet loss focuses on relative distances (A is closer to P than N), which is better for ranking.
- Contrastive loss pushes absolute distances, which is sometimes harder to tune but simpler to train with in-batch negatives.
- **Modern standard:** InfoNCE (Normalized Temperature-scaled Cross Entropy), a generalization of contrastive loss to multiple negatives in a batch (used in CLIP and SimCLR).

### Q79: Formulate the InfoNCE loss used in modern embedding models (SimCLR, Sentence-BERT).
**Expected Answer:**
$$L_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(z_i, z_j) / \tau)}$$
- $\text{sim}(\cdot, \cdot)$ is cosine similarity.
- $z_i^+$ is the positive pair (e.g., augmented version of same image, or relevant doc for a query).
- The denominator sums over all $N$ negatives in the batch (plus the positive).
- $\tau$ is the temperature parameter.
**Why Temperature $\tau$?**
It scales the logits before the softmax.
- High $\tau$ (e.g., 1.0): Softens probabilities, model pays attention to all negatives.
- Low $\tau$ (e.g., 0.07): Sharpens probabilities, model focuses heavily on penalizing the "hardest" negatives (the ones most similar to the anchor but mathematically negative). This makes training much more efficient.

### Q80: In XGBoost, how do you define a custom evaluation metric vs a custom objective function? Give an example for Mean Absolute Percentage Error (MAPE).
**Expected Answer:**
- **Objective function:** Used DURING training to build trees. Requires providing the first (Gradient) and second (Hessian) derivatives of the loss w.r.t the prediction.
- **Evaluation metric:** Used AFTER trees are built to monitor performance (e.g., early stopping). Only requires returning a single float score.
**MAPE Formulation:**
$$L(y, \hat{y}) = \left| \frac{y - \hat{y}}{y} \right|$$
**Problem for Objective:** The absolute value is not smooth (non-differentiable at 0). You cannot efficiently use MAPE as an objective function in XGBoost without smoothing it (e.g., Huber loss approximation of MAPE).
**However, as an Evaluation Metric:** It's trivial.
```python
def mape_eval(preds, dtrain):
    labels = dtrain.get_label()
    # Mask to avoid division by zero
    mask = labels != 0
    mape = np.mean(np.abs((labels[mask] - preds[mask]) / labels[mask]))
    return 'mape', mape, False  # False means we want to minimize it
```

---
*End of Optimization Grind.*
