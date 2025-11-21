# Day 24: Trust Region Policy Optimization (TRPO)

## 1. The Theoretical Foundation
TRPO is the rigorous predecessor to PPO.
**Core Idea:** Maximize expected return while constraining the policy update using KL divergence.
$$ \max_\theta \mathbb{E}[r_t(\theta) A_t] $$
$$ \text{subject to } \mathbb{E}[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta $$
*   This ensures **monotonic improvement**: The new policy is guaranteed to be at least as good as the old one.
*   $\delta$ is a small constant (e.g., 0.01).

## 2. Second-Order Optimization
To solve this constrained problem, TRPO uses:
*   **Natural Gradient:** A second-order method that uses the Fisher Information Matrix $F$.
    $$ \theta \leftarrow \theta + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g $$
    where $g$ is the policy gradient.
*   **Conjugate Gradient:** To approximate $F^{-1} g$ without computing the full inverse (computationally expensive).
*   **Line Search:** To ensure the KL constraint is satisfied.

## 3. Fisher Information Matrix
The Fisher Information Matrix measures the curvature of the KL divergence:
$$ F_{ij} = \mathbb{E}[\frac{\partial \log \pi}{\partial \theta_i} \frac{\partial \log \pi}{\partial \theta_j}] $$
*   Computing $F$ exactly is prohibitive for large networks.
*   TRPO approximates $F^{-1} g$ using conjugate gradient with Hessian-vector products.

## 4. Code Sketch (High-Level)
```python
def trpo_update(policy, states, actions, advantages):
    # 1. Compute policy gradient
    loss = -(log_probs * advantages).mean()
    grads = torch.autograd.grad(loss, policy.parameters())
    g = torch.cat([grad.view(-1) for grad in grads])
    
    # 2. Compute Fisher-vector product for conjugate gradient
    def fisher_vector_product(v):
        kl = kl_divergence(old_policy, new_policy)
        grads_kl = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads_kl])
        kl_v = (flat_grad_kl * v).sum()
        grads_kl_v = torch.autograd.grad(kl_v, policy.parameters())
        return torch.cat([grad.view(-1) for grad in grads_kl_v])
    
    # 3. Solve F^{-1} g using Conjugate Gradient
    step_dir = conjugate_gradient(fisher_vector_product, g)
    
    # 4. Line search to satisfy KL constraint
    step_size = line_search(policy, step_dir, delta=0.01)
    
    # 5. Update parameters
    update_params(policy, step_size * step_dir)
```

### Key Takeaways
*   TRPO provides theoretical guarantees (monotonic improvement).
*   Too complex for practitioners; replaced by PPO.
*   Important for understanding trust regions and natural gradients.
