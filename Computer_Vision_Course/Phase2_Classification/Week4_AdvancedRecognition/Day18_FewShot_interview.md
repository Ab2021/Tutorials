# Day 18 Interview Questions: Few-Shot Learning

## Q1: What is "N-way K-shot" classification?
**Answer:**
A standard evaluation protocol for few-shot learning.
*   **N-way:** The number of classes in the test episode (e.g., 5 classes).
*   **K-shot:** The number of labeled examples provided per class (e.g., 1 or 5).
*   **Query Set:** The unlabeled images to classify.
*   5-way 1-shot is much harder than 5-way 5-shot.

## Q2: Explain the core idea of Prototypical Networks.
**Answer:**
*   It assumes there exists an embedding space where points cluster around a single **prototype** (mean vector) for each class.
*   Training involves learning a non-linear mapping (CNN) that minimizes the distance between query points and their correct class prototype, while maximizing distance to incorrect prototypes.

## Q3: How does MAML differ from standard Transfer Learning?
**Answer:**
*   **Transfer Learning:** Train on Task A, then fine-tune on Task B. The initialization is optimized for Task A.
*   **MAML:** Explicitly optimizes the initialization so that it is **easy to fine-tune** on *any* task. It maximizes the performance *after* fine-tuning, not before.

## Q4: Why is Cosine Similarity often used instead of Euclidean Distance in FSL?
**Answer:**
*   In high-dimensional spaces, magnitude of vectors can vary significantly and introduce noise.
*   **Cosine Similarity** focuses on the *angle* (direction) of the feature vector, which captures semantic content better than magnitude.
*   Often, features are L2-normalized, making Euclidean distance and Cosine similarity monotonic to each other.

## Q5: What is the "Base Class" vs "Novel Class" split?
**Answer:**
*   **Base Classes:** A large set of classes with many images used for training (meta-training).
*   **Novel Classes:** A disjoint set of classes used for testing (meta-testing).
*   Crucially, the model has never seen the Novel classes during training. It must generalize to them using only the K support examples.

## Q6: What is the difference between Episodic Training and Batch Training?
**Answer:**
*   **Batch Training:** Standard classification (minimize Cross-Entropy on random batches).
*   **Episodic Training:** Mimics the test condition during training.
    *   Sample N classes, K support, Q queries.
    *   Compute loss on queries.
    *   Update model.
    *   Ensures the model learns to "compare" and "adapt" rather than just memorize class features.

## Q7: Why is Second-Order derivative required in MAML?
**Answer:**
*   MAML optimizes the initialization $\theta$.
*   The loss depends on the adapted weights $\theta' = \theta - \alpha \nabla L$.
*   To compute gradient w.r.t $\theta$, we need to differentiate through the update step: $\nabla_\theta L(\theta - \alpha \nabla L)$.
*   This involves the gradient of a gradient (Hessian).
*   *Note: First-order MAML ignores this term for efficiency.*

## Q8: What is "Transductive Inference"?
**Answer:**
Using the statistics of the **unlabeled query set** to help classification.
*   Example: If we know the query set contains 5 dogs and 5 cats, we can adjust our predictions to satisfy this distribution, rather than predicting "dog" for everything.

## Q9: Why does SimpleShot (Transfer Learning) often beat Meta-Learning?
**Answer:**
*   Meta-learning algorithms (like MAML) can be unstable and hard to train.
*   Training a deep ResNet on all Base classes learns extremely robust features.
*   Simply using these robust features with a Nearest Neighbor classifier is often more effective than learning a complex adaptation strategy on weaker features.

## Q10: Implement the prototype calculation in PyTorch.
**Answer:**
```python
def get_prototypes(embeddings, labels, n_way):
    # embeddings: (N_total, D)
    # labels: (N_total,)
    prototypes = []
    for i in range(n_way):
        # Select embeddings belonging to class i
        mask = labels == i
        class_embeddings = embeddings[mask]
        # Compute mean
        proto = class_embeddings.mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)
```
