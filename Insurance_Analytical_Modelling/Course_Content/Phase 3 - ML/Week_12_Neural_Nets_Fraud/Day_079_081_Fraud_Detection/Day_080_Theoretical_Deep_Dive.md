# Deep Learning for Actuaries (Part 1) - Entity Embeddings & Categorical Data - Theoretical Deep Dive

## Overview
"Actuarial data is 90% categorical."
Zip Code, Car Model, Occupation, ICD-10 Code.
Traditional One-Hot Encoding explodes the dimensionality and loses the *relationship* between categories.
This day focuses on **Entity Embeddings** (Guo & Berkhahn, 2016), a technique that revolutionized how Neural Networks handle tabular data, making them competitive with GBMs.

---

## 1. Conceptual Foundation

### 1.1 The Curse of Dimensionality

*   **Scenario:** You have a "Zip Code" feature with 40,000 levels.
*   **One-Hot Encoding:** Creates 40,000 columns. The matrix is 99.9% sparse.
*   **Problem:**
    *   Memory hog.
    *   The model treats Zip 90210 and 90211 as completely orthogonal (Distance = $\sqrt{2}$). It doesn't know they are neighbors.

### 1.2 The Embedding Solution

*   **Idea:** Map each category to a dense vector of low dimension (e.g., $d=5$).
*   **Mechanism:**
    *   Input: Integer ID (e.g., 42).
    *   Lookup Table: Returns row 42 of a Weight Matrix $W \in \mathbb{R}^{40000 \times 5}$.
    *   Output: `[0.1, -0.4, 0.9, 0.2, 0.0]`.
*   **Learning:** The values in $W$ are learned via backpropagation. The network *learns* the "meaning" of the Zip Code.

---

## 2. Mathematical Framework

### 2.1 Embedding Dimension Selection

How big should the vector be?
*   **Rule of Thumb:** $d = \min(50, (N+1)/2)$.
*   **FastAI Heuristic:** $d = \min(600, 1.6 \times N^{0.56})$.
*   **Trade-off:**
    *   Too small: Cannot capture complexity (Underfitting).
    *   Too large: Overfitting and slow training.

### 2.2 Distance in Embedding Space

*   **Euclidean Distance:** $\|v_A - v_B\|_2$.
*   **Interpretation:** If Zip A and Zip B have similar risk profiles (e.g., urban density, theft rates), their embedding vectors will be close in the vector space.
*   **Visualization:** t-SNE or PCA can project these 5D vectors to 2D. You often see "Clusters" of similar risks (e.g., Coastal Zips cluster together).

---

## 3. Theoretical Properties

### 3.1 Transfer Learning with Embeddings

*   **Pre-training:** Train a network to predict "Claim Frequency".
*   **Extraction:** Extract the learned embeddings for "Car Model".
*   **Reuse:** Use these embeddings as features in a *different* model (e.g., a GBM for Severity).
*   **Benefit:** The GBM now knows that "Honda Civic" and "Toyota Corolla" are similar, without needing millions of rows to learn it from scratch.

### 3.2 Handling New Categories (Cold Start)

*   **Problem:** A new Car Model appears in the test set.
*   **Solution:**
    *   Map it to an "Unknown" token (trained with dropout).
    *   Or map it to the average embedding of its parent category (e.g., "Sedan").

---

## 4. Modeling Artifacts & Implementation

### 4.1 Keras Embedding Layer

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_embedding_model(vocab_size, embedding_dim):
    # Input: Integer Index
    input_cat = layers.Input(shape=(1,), name='zip_code')
    
    # Embedding Layer
    # input_dim = Vocabulary Size + 1 (for Unknown)
    # output_dim = Vector Size
    embed = layers.Embedding(input_dim=vocab_size+1, 
                             output_dim=embedding_dim, 
                             name='zip_embedding')(input_cat)
    
    # Flatten: (Batch, 1, Dim) -> (Batch, Dim)
    embed = layers.Flatten()(embed)
    
    # Concatenate with continuous inputs...
    # ...
    return models.Model(inputs=input_cat, outputs=embed)

# Example
model = build_embedding_model(vocab_size=40000, embedding_dim=10)
model.summary()
```

### 4.2 Visualizing Embeddings

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get weights
weights = model.get_layer('zip_embedding').get_weights()[0]

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
weights_2d = tsne.fit_transform(weights)

# Plot
plt.scatter(weights_2d[:, 0], weights_2d[:, 1], alpha=0.5)
plt.title("Zip Code Embeddings Space")
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 Nearest Neighbors Analysis

*   **Validation:** Pick a known category (e.g., "Ford F-150").
*   **Query:** Find the 5 nearest neighbors in embedding space.
*   **Expected Result:** "Chevy Silverado", "Ram 1500", "Toyota Tundra".
*   **Bad Result:** "Mini Cooper", "Porsche 911". (Implies the model failed to learn the semantics).

### 5.2 Impact on Gini

*   **Experiment:**
    *   Model A: One-Hot Encoding.
    *   Model B: Embeddings.
*   **Result:** Model B usually has higher Gini *and* trains 10x faster.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Ordinal Variables

*   **Question:** Should "Driver Age" be continuous or embedded?
*   **Answer:**
    *   Continuous: Assumes smoothness.
    *   Embedding: Allows arbitrary non-linearity (e.g., Age 16 is high risk, 25 is low, 80 is high).
    *   **Best Practice:** Bin Age and use Embeddings. It captures the U-shape perfectly.

### 6.2 The "Unknown" Token

*   **Pitfall:** Forgetting to reserve index 0 for "Unknown/Missing".
*   **Consequence:** The model crashes on new data.

---

## 7. Advanced Topics & Extensions

### 7.1 Autoencoders for Imputation

*   **Problem:** Missing Data.
*   **Method:** Denoising Autoencoder (DAE).
    *   Input: Row with missing values masked (set to 0).
    *   Target: Original Row.
    *   Loss: Reconstruction Error.
*   **Result:** The network learns to "fill in the blanks" based on correlations.

### 7.2 TabNet

*   **Architecture:** A deep learning architecture specifically for tabular data.
*   **Mechanism:** Uses "Sequential Attention" to select features at each step, mimicking a Decision Tree.
*   **Performance:** Often beats XGBoost on large datasets.

---

## 8. Regulatory & Governance Considerations

### 8.1 Proxy Discrimination via Embeddings

*   **Risk:** Zip Code embeddings might reconstruct "Race" or "Income" with high accuracy.
*   **Test:** Train a linear probe on the embeddings to predict Race.
*   **Mitigation:** Adversarial Debiasing. Penalize the network if the embeddings can predict the protected attribute.

---

## 9. Practical Example

### 9.1 The "Occupation" Factor

**Scenario:** You have "Occupation" data (2,000 codes).
**Traditional:** Group into "Blue Collar", "White Collar", "Student". (Loss of information).
**Deep Learning:** Train an Embedding Model.
**Discovery:**
*   The model clusters "Surgeon" with "Pilot" (High Stress, High Income, Low Frequency, High Severity).
*   It clusters "Bartender" with "Musician" (Late night driving).
*   **Result:** Much more granular risk segmentation than the manual groups.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Embeddings** turn categories into math.
2.  **Dimensionality Reduction** is built-in.
3.  **Semantic Similarity** is learned automatically.

### 10.2 When to Use This Knowledge
*   **High Cardinality:** When you have > 100 categories.
*   **Unstructured Data:** When mixing Tabular data with Text/Images.

### 10.3 Critical Success Factors
1.  **Pre-processing:** Map rare categories to "Other" before training.
2.  **Regularization:** Embeddings can overfit. Use L2 regularization on the embedding matrix.

### 10.4 Further Reading
*   **Guo & Berkhahn:** "Entity Embeddings of Categorical Variables".
*   **FastAI Course:** "Tabular Data" section.

---

## Appendix

### A. Glossary
*   **Latent Space:** The hidden vector space where the embeddings live.
*   **Lookup Table:** The efficient implementation of an embedding layer.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Embedding** | $v = W \cdot e_i$ | Lookup |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
