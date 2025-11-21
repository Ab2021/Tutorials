# Day 29 Deep Dive: Second-Order Meta-Learning

## 1. Why Second-Order Derivatives?
MAML's outer loop optimizes:
$$ \theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}(\theta') $$
where $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$ (inner update).

To compute $\nabla_\theta \mathcal{L}(\theta')$, we need:
$$\frac{d\mathcal{L}(\theta')}{d\theta} = \frac{d\mathcal{L}}{d\theta'} \frac{d\theta'}{d\theta} $$
This involves the **Hessian** (second derivative of the loss).
*   Computationally expensive.
*   Requires storing the computation graph.

## 2. First-Order MAML (FOMAML)
Approximation: Ignore $\frac{d\theta'}{d\theta}$ (treat $\theta'$ as constant):
$$ \nabla_\theta \mathcal{L}(\theta') \approx \nabla_{\theta'} \mathcal{L}(\theta') $$
*   Much faster (no second-order derivatives).
*   Works almost as well in practice.

## 3. Reptile: A Simpler Alternative
Reptile (Nichol et al., 2018) simplifies MAML:
1.  Sample a task.
2.  Take multiple SGD steps on the task to get $\theta'$.
3.  Meta-update: $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$.
*   No second-order derivatives.
*   Empirically similar performance to MAML.

## 4. Context-Based Meta-RL
Instead of adapting parameters, encode the task in a **context vector**:
*   **PEARL:** Uses a VAE to encode the task from experience.
*   **CAVIA:** Context vector is updated, not the full policy.
*   **Advantage:** No inner-loop optimization at test time.
