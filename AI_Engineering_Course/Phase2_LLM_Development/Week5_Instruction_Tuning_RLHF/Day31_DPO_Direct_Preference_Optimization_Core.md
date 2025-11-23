# Day 31: DPO (Direct Preference Optimization)
## Core Concepts & Theory

### The Problem with RLHF

RLHF has a fundamental complexity: it requires training TWO models (Reward Model + Policy) and using a complex RL algorithm (PPO).
- **Instability:** PPO is notoriously hard to tune. Hyperparameters matter immensely.
- **Reward Hacking:** The policy can exploit bugs in the Reward Model.
- **Computational Cost:** Training RM, then running PPO for thousands of iterations.

**Question:** Can we skip the Reward Model and PPO entirely?
**Answer:** Yes. DPO (Direct Preference Optimization).

### 1. The Core Insight of DPO

**RLHF Objective:**
$$ \max_\pi \mathbb{E}_{x, y \sim \pi} [R(x, y) - \beta \cdot KL(\pi || \pi_{ref})] $$

**Key Observation:**
The optimal policy $\pi^*$ that maximizes this objective has a closed-form solution:
$$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right) $$
where $Z(x)$ is the partition function.

**Rearranging:**
$$ R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x) $$

**Substituting into Bradley-Terry:**
$$ P(y_w > y_l | x) = \sigma(R(x, y_w) - R(x, y_l)) $$
$$ = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right) $$

The $\log Z(x)$ terms cancel!

**DPO Loss:**
$$ L_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right] $$

**Result:** We can directly optimize the policy $\pi_\theta$ on preference data WITHOUT training a separate Reward Model!

### 2. DPO Algorithm

**Input:**
- Preference dataset: $(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$.
- Reference policy: $\pi_{ref}$ (frozen SFT model).

**Training:**
1. **Forward Pass:**
   - Compute $\log \pi_\theta(y_w|x)$ and $\log \pi_\theta(y_l|x)$.
   - Compute $\log \pi_{ref}(y_w|x)$ and $\log \pi_{ref}(y_l|x)$ (no gradients).
2. **Loss:**
   $$ L = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right) $$
3. **Backward:** Compute gradients and update $\pi_\theta$.
4. **Repeat:** For 1-3 epochs over the preference dataset.

**Hyperparameters:**
- **$\beta$:** 0.1-0.5. Controls the strength of the KL penalty.
- **Learning Rate:** 1e-6 to 5e-7.
- **Batch Size:** 32-128.

### 3. Advantages of DPO

| Aspect | RLHF | DPO |
| :--- | :--- | :--- |
| **Models to Train** | 2 (RM + Policy) | 1 (Policy) |
| **Algorithm** | PPO (complex) | Supervised Learning (simple) |
| **Stability** | Unstable (PPO tuning) | Stable |
| **Reward Hacking** | Possible | Impossible (no RM) |
| **Compute** | High (RL iterations) | Low (few epochs) |
| **Performance** | Slightly Better | Comparable |

### 4. Implicit Reward Function

Even though DPO doesn't train an explicit Reward Model, we can extract an *implicit* reward:
$$ R_{DPO}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} $$

This reward is always "grounded" in the reference policy, preventing reward hacking.

### 5. DPO Variants

**IPO (Identity Preference Optimization):**
- Uses a different loss function (MSE instead of log-sigmoid).
- More stable for small datasets.

**KTO (Kahneman-Tversky Optimization):**
- Based on prospect theory.
- Handles binary feedback (thumbs up/down) instead of pairwise comparisons.

**ORPO (Odds Ratio Preference Optimization):**
- Combines SFT and DPO into a single stage.
- Trains on both demonstrations and preferences simultaneously.

### 6. When to Use DPO vs. RLHF

**Use DPO if:**
- You want simplicity and stability.
- You have a fixed preference dataset.
- You don't need to inspect the reward function.

**Use RLHF if:**
- You need iterative improvement (collect new data, retrain RM, run PPO again).
- You want to inspect and debug the reward function.
- You have the resources for complex RL training.

### Real-World Adoption

- **Zephyr-7B:** Trained with DPO. Outperforms LLaMA-2-Chat-70B on MT-Bench.
- **Mistral-7B-Instruct:** Uses DPO.
- **Anthropic (Claude):** Still uses RLHF (Constitutional AI variant).
- **OpenAI (GPT-4):** Uses RLHF.

### Summary

**DPO is a game-changer:**
- Eliminates the need for a Reward Model.
- Eliminates the need for PPO.
- Achieves comparable performance to RLHF with 10x less complexity.
- Has become the default for open-source models (2024+).

### Next Steps
In the Deep Dive, we will derive the DPO loss from first principles and implement a complete DPO training loop in PyTorch.
