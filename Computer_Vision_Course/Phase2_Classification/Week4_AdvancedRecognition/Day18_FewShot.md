# Day 18: Few-Shot Learning & Meta-Learning

## 1. Problem Formulation
**Goal:** Learn to recognize new classes given only a few examples.
**N-way K-shot:**
*   **Support Set:** $N$ classes, $K$ examples each (Total $N \times K$ images).
*   **Query Set:** Unlabeled images from the $N$ classes to classify.
*   **Example:** 5-way 1-shot = 5 classes, 1 example each.

## 2. Metric Learning
**Idea:** Learn a feature space where similar classes are close and different classes are far.

### Siamese Networks
*   Train on pairs of images.
*   **Contrastive Loss:** Minimize distance for same class, maximize for different.

### Prototypical Networks (Snell et al., 2017)
**Idea:** Represent each class by the mean (prototype) of its support examples.
1.  **Embedding:** Map support images to feature space $f_\theta(x)$.
2.  **Prototype:** $c_k = \frac{1}{K} \sum_{(x,y) \in S_k} f_\theta(x)$.
3.  **Classification:** Assign query $x_q$ to nearest prototype (Euclidean distance).
    $$ p(y=k|x_q) = \frac{\exp(-d(f_\theta(x_q), c_k))}{\sum_{k'} \exp(-d(f_\theta(x_q), c_{k'}))} $$

```python
def prototype_loss(support, query, n_way, k_shot, q_query):
    # support: (n_way, k_shot, D)
    # query: (n_way, q_query, D)
    
    # 1. Compute Prototypes
    prototypes = support.mean(dim=1) # (n_way, D)
    
    # 2. Compute Distances
    # Expand for broadcasting
    # query: (n_way*q_query, 1, D)
    # prototypes: (1, n_way, D)
    query_flat = query.view(-1, 1, query.shape[-1])
    proto_flat = prototypes.unsqueeze(0)
    
    dists = torch.pow(query_flat - proto_flat, 2).sum(dim=2) # (N*Q, N)
    
    # 3. Log Softmax & Loss
    log_p_y = F.log_softmax(-dists, dim=1)
    
    # Targets: 0, 0... 1, 1...
    target = torch.arange(n_way).repeat_interleave(q_query).to(device)
    
    loss = F.nll_loss(log_p_y, target)
    return loss
```

## 3. Optimization-Based Meta-Learning

### MAML (Model-Agnostic Meta-Learning)
**Idea:** Learn a set of initial weights $\theta$ that can be quickly adapted to a new task with few gradient steps.
*   **Inner Loop:** Adapt $\theta$ to specific task $i$: $\theta'_i = \theta - \alpha \nabla_\theta L_{S_i}(\theta)$.
*   **Outer Loop:** Update $\theta$ to minimize loss of adapted weights on query set: $\theta \leftarrow \theta - \beta \nabla_\theta \sum L_{Q_i}(\theta'_i)$.
*   **Intuition:** Find a parameter initialization that is "close" to the optimal parameters for many different tasks.

## 4. Relation Networks
**Idea:** Learn a non-linear distance metric.
*   Concatenate feature of query and feature of support.
*   Pass through a "Relation Module" (MLP/CNN) to output a similarity score $[0, 1]$.

## Summary
Few-shot learning moves away from "training on a dataset" to "learning to learn". Prototypical Networks (Metric) and MAML (Optimization) are the foundational approaches.
