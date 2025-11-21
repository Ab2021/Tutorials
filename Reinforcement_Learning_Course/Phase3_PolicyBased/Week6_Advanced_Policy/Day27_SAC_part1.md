# Day 27 Deep Dive: Automatic Temperature Tuning

## 1. The Temperature Problem
The temperature $\alpha$ controls the exploration-exploitation tradeoff:
$$ J = \mathbb{E}[r + \alpha H(\pi)] $$
*   High $\alpha$: More exploration (higher entropy).
*   Low $\alpha$: More exploitation (lower entropy).
*   Fixed $\alpha$ is suboptimal: Early in training, you want high exploration; later, you want exploitation.

## 2. Automatic Tuning
SAC treats $\alpha$ as a **learnable parameter**:
$$ \max_\alpha \mathbb{E}[\alpha (H(\pi(\cdot|s_t)) - H_{target})] $$
where $H_{target}$ is a target entropy (hyperparameter).
*   **Heuristic:** $H_{target} = -\dim(\mathcal{A})$ (negative action dimensionality).
*   $\alpha$ is updated to match the current policy's entropy to the target.
*   This automates the exploration schedule.

## 3. Log-Alpha Trick
In practice, we optimize $\log \alpha$ instead of $\alpha$ for numerical stability:
$$ \alpha = e^{\log \alpha} $$
$$ \log \alpha \leftarrow \log \alpha - \beta \nabla_{\log \alpha} (H(\pi) - H_{target}) $$
This ensures $\alpha > 0$ and avoids numerical issues.

## 4. Squashing Function
Actions are sampled from an unbounded Gaussian, then squashed using $\tanh$:
$$ a = \tanh(\mu + \sigma \epsilon) $$
This maps actions to $[-1, 1]$.
**Important:** The log-probability must account for the squashing:
$$ \log \pi(a|s) = \log \mathcal{N}(a'|\mu, \sigma) - \sum_i \log(1 - \tanh^2(a'_i)) $$
where $a' = \tanh^{-1}(a)$.

## 5. SAC vs. TD3 Summary
| Feature | TD3 | SAC |
|---------|-----|-----|
| **Policy** | Deterministic | Stochastic |
| **Exploration** | Action Noise | Entropy Max |
| **Temperature** | Fixed Noise | Learnable $\alpha$ |
| **Robustness** | Good | Excellent |
| **Sample Efficiency** | Good | Better |
| **Complexity** | Medium | High |
