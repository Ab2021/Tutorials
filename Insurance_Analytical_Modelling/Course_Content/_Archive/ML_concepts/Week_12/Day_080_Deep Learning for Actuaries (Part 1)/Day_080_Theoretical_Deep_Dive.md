# Deep Learning for Actuaries (Part 1) - Theoretical Deep Dive

## Overview
"Deep Learning is only for Images and Text." **False.**
This session covers the breakthrough technique of **Entity Embeddings** (Guo & Berkhahn), which allows Neural Networks to beat XGBoost on tabular data. We also explore **Neural Mortality Forecasting**.

---

## 1. Conceptual Foundation

### 1.1 The Categorical Variable Problem

*   **Scenario:** You have a "Zip Code" variable (30,000 levels).
*   **One-Hot Encoding:** Creates 30,000 sparse columns. Memory explosion. No relationship between Zip 10001 and 10002.
*   **Label Encoding:** Assigns 1, 2, 3... Implies an order that doesn't exist.
*   **Target Encoding:** Replaces Zip with Average Claim Cost. Loses information.

### 1.2 Entity Embeddings (The Solution)

*   **Idea:** Map each category to a dense vector of low dimension (e.g., size 10).
*   **Mechanism:** The network *learns* the best vector representation for each Zip Code during training.
*   **Result:**
    *   Zip Codes with similar risk profiles will have vectors that are close in Euclidean space.
    *   The network learns the "Geography" of risk automatically.

### 1.3 The Rossmann Store Sales Legend

*   **Context:** A Kaggle competition to predict sales.
*   **Outcome:** The 3rd place winner used a simple MLP with Entity Embeddings.
*   **Significance:** It proved that NNs can handle tabular data with high-cardinality features better than Tree models in some cases.

---

## 2. Mathematical Framework

### 2.1 Embedding Layer

*   Input: Integer index $i$ (e.g., Zip Code #42).
*   Operation: Lookup row $i$ in a Weight Matrix $W$ of size $(N_{categories} \times D_{embedding})$.
*   Output: Vector $v \in \mathbb{R}^D$.
*   *Note:* The matrix $W$ is a trainable parameter. Backpropagation updates it.

### 2.2 Neural Lee-Carter (Mortality)

*   **Classical Lee-Carter:** $\ln(m_{x,t}) = \alpha_x + \beta_x \kappa_t$.
*   **Neural Extension:**
    *   Input: Age ($x$), Year ($t$).
    *   Embeddings: Embed Age and Year.
    *   Hidden Layers: Capture non-linear interactions.
    *   Output: $\ln(m_{x,t})$.
*   *Benefit:* Can capture cohort effects and non-linear trends that LC misses.

---

## 3. Theoretical Properties

### 3.1 Dimensionality Reduction

*   How big should the embedding be?
*   **Rule of Thumb:** $D = \min(50, (N_{categories} + 1) // 2)$.
*   *Example:* For 50 US States, use size 10. For 30,000 Zip Codes, use size 50.

### 3.2 Semantic Similarity

*   After training, you can plot the embeddings (using t-SNE or PCA).
*   **Observation:** You will see clusters.
    *   "Urban Zips" cluster together.
    *   "Coastal Zips" cluster together.
*   *Actuarial Use:* You can extract these embeddings and feed them into a GLM! (Hybrid Model).

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyTorch Embedding Model

```python
import torch
import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, num_zips, num_occupations):
        super().__init__()
        # Embeddings
        self.embed_zip = nn.Embedding(num_zips, 50) # 30k -> 50
        self.embed_occ = nn.Embedding(num_occupations, 10) # 100 -> 10
        
        # Dense Layers
        self.fc1 = nn.Linear(50 + 10 + 5, 128) # 5 numeric features
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x_cat, x_num):
        # x_cat[:, 0] is Zip, x_cat[:, 1] is Occupation
        e1 = self.embed_zip(x_cat[:, 0])
        e2 = self.embed_occ(x_cat[:, 1])
        
        # Concatenate embeddings and numeric features
        x = torch.cat([e1, e2, x_num], dim=1)
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        return self.output(x)
```

### 4.2 Extracting Embeddings

```python
# Get the learned matrix
zip_weights = model.embed_zip.weight.data.numpy()

# Find nearest neighbors for Zip #100
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(zip_weights)
distances, indices = nbrs.kneighbors([zip_weights[100]])

print(f"Zip Codes similar to #100: {indices}")
```

---

## 5. Evaluation & Validation

### 5.1 Embedding Visualization

*   Use **t-SNE** to project the 50-dimensional Zip vectors into 2D.
*   Color the points by "Average Claim Cost".
*   *Validation:* If the colors are mixed randomly, the embedding failed. If you see a gradient of colors, it learned the risk.

### 5.2 Generalization Gap

*   Embeddings can overfit if the category is rare (only 1 claim in Zip 99999).
*   *Fix:* Use Dropout on the embedding layer.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Unseen Categories**
    *   What if the Test set has a Zip Code not in the Training set?
    *   *Fix:* Reserve index 0 for "Unknown/Other". During training, randomly map some Zips to 0.

2.  **Trap: Embedding Size too Large**
    *   If size = Number of Categories, it's just One-Hot Encoding (no compression).
    *   If size = 1, it's just Label Encoding (too compressed).

### 6.2 Implementation Challenges

1.  **Indexing:**
    *   PyTorch Embedding requires inputs to be LongTensor (Integers) from $0$ to $N-1$.
    *   You must preprocess your data to map "NY", "CA"... to 0, 1...

---

## 7. Advanced Topics & Extensions

### 7.1 Transformer for Tabular Data (TabTransformer)

*   Instead of concatenating embeddings, feed them into a Transformer Encoder (Self-Attention).
*   Allows the model to learn context-aware interactions between categories.

### 7.2 Neural GLMs

*   Use a Neural Network to learn the features (Embeddings), but the final layer is a GLM (Poisson/Gamma).
*   Best of both worlds: Deep Feature Engineering + Actuarial Output Distribution.

---

## 8. Regulatory & Governance Considerations

### 8.1 "The Map"

*   **Regulator:** "Why is Zip A priced same as Zip B?"
*   **Actuary:** "Because the Embedding Layer learned they are similar."
*   **Regulator:** "Show me."
*   **Actuary:** Show the t-SNE plot. "See, they are neighbors in risk space."

---

## 9. Practical Example

### 9.1 Worked Example: The "Occupation" Factor

**Scenario:**
*   Pricing Life Insurance.
*   **Variable:** Occupation (500 codes).
*   **GLM:** Grouped into 5 classes (White Collar, Blue Collar...). (Loss of info).
*   **NN with Embeddings:** Kept all 500 codes.
*   **Result:** The NN found that "Skydiving Instructor" (rare) clusters with "Deep Sea Welder". The GLM had lumped them into "Other".
*   **Impact:** More accurate pricing for niche risks.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Embeddings** turn categories into vectors.
2.  **Rossmann** proved it works for tabular data.
3.  **Neural Mortality** beats Lee-Carter.

### 10.2 When to Use This Knowledge
*   **High Cardinality:** When you have variables with >100 levels.
*   **Hybrid Models:** Use NN to get embeddings, then put them in XGBoost.

### 10.3 Critical Success Factors
1.  **Pre-processing:** Map categories to 0..N-1 correctly.
2.  **Regularization:** Dropout on embeddings is key.

### 10.4 Further Reading
*   **Guo & Berkhahn:** "Entity Embeddings of Categorical Variables".
*   **Richman & Wüthrich:** "Nagging Predictors" (Neural Networks in Actuarial Science).

---

## Appendix

### A. Glossary
*   **Cardinality:** Number of unique values.
*   **Dense Vector:** A vector with mostly non-zero values.
*   **t-SNE:** t-Distributed Stochastic Neighbor Embedding.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Embedding Lookup** | $v = W[i]$ | Vector Retrieval |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
