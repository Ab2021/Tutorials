# Day 33: Advanced RLHF Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Iterative RLHF: Distribution Shift Analysis

**Problem:** After PPO training, the policy $\pi_{new}$ generates responses from a different distribution than $\pi_{SFT}$.
The Reward Model was trained on data from $\pi_{SFT}$, so it may be inaccurate on $\pi_{new}$ outputs.

**Distribution Shift:**
$$ D_{KL}(P_{data}^{new} || P_{data}^{old}) > \epsilon $$

**Solution:** Iterative RLHF
1. Train RM on data from $\pi_{SFT}$.
2. Run PPO -> get $\pi_1$.
3. Collect new preferences on data from $\pi_1$.
4. Train new RM on combined data (old + new).
5. Run PPO -> get $\pi_2$.
6. Repeat.

**Convergence:**
After 3-5 iterations, the policy stabilizes. Further iterations yield diminishing returns.

### 2. Multi-Objective RLHF: Pareto Frontier

**Scalarization (Weighted Sum):**
$$ R_{total} = w_1 R_1 + w_2 R_2 $$
**Problem:** Choosing weights is arbitrary. Different weights give different trade-offs.

**Pareto Optimization:**
Find all policies where improving one objective hurts another.
**Pareto Frontier:** Set of all Pareto-optimal policies.

**Algorithm (MORL - Multi-Objective RL):**
1. Train multiple policies with different weight vectors.
2. Plot them in objective space $(R_1, R_2)$.
3. The convex hull is the Pareto frontier.

**User Choice:**
At inference time, let the user choose the trade-off (e.g., "Be more creative" vs "Be more factual").

### 3. Process Supervision: Step-Level Rewards

**Outcome Reward:**
$$ R_{outcome}(x, y) = \begin{cases} 1 & \text{if final answer is correct} \\ 0 & \text{otherwise} \end{cases} $$

**Process Reward:**
$$ R_{process}(x, y) = \sum_{t=1}^T r_t $$
where $r_t \in \{-1, 0, +1\}$ is the reward for step $t$.

**Labeling:**
Humans annotate each reasoning step:
- +1: Correct step.
- 0: Neutral (e.g., rephrasing).
- -1: Incorrect step.

**Training:**
Train a Process Reward Model (PRM) to predict $r_t$ for each step.
Use PRM scores during PPO to reward correct reasoning.

**Benefits:**
- **Generalization:** Model learns the reasoning process, not just the answer pattern.
- **Interpretability:** Can identify which step failed.

**Challenges:**
- **Cost:** Annotating every step is 10x more expensive than outcome labels.
- **Granularity:** Defining "steps" is subjective.

### 4. Best-of-N: Expected Max Reward

**Analysis:**
Let $R_i \sim \text{Reward}(x, y_i)$ where $y_i \sim \pi(y|x)$.
The expected maximum reward over $N$ samples is:
$$ \mathbb{E}[\max_{i=1}^N R_i] \approx \mu + \sigma \sqrt{2 \log N} $$
where $\mu$ is the mean reward and $\sigma$ is the standard deviation.

**Implication:**
- Doubling $N$ increases expected reward by $\sigma \sqrt{2 \log 2} \approx 0.83\sigma$.
- Diminishing returns: Going from N=16 to N=32 gives less gain than N=1 to N=16.

### 5. Reward Model Ensemble: Uncertainty Quantification

**Epistemic Uncertainty:**
Uncertainty due to limited training data.
Captured by variance across ensemble members.

**Aleatoric Uncertainty:**
Inherent randomness in the data.
Cannot be reduced by more data.

**Ensemble Variance:**
$$ \text{Var}(R) = \frac{1}{K} \sum_{k=1}^K (R_k - \bar{R})^2 $$
High variance = uncertain prediction = risky to trust.

**Uncertainty-Aware PPO:**
$$ R_{total} = R_{mean} - \lambda \cdot \text{Var}(R) $$
Penalize high-uncertainty responses.

### Code: Multi-Objective RLHF (Weighted Sum)

```python
import torch

def multi_objective_reward(responses, reward_models, weights):
    """
    Compute weighted sum of multiple reward models.
    
    responses: [B, L] - batch of responses
    reward_models: list of RM models
    weights: [K] - weight for each objective
    """
    total_reward = 0
    
    for rm, w in zip(reward_models, weights):
        r = rm(responses)  # [B]
        total_reward += w * r
    
    return total_reward

# Example: Helpfulness vs Conciseness
rm_helpful = load_reward_model("helpful")
rm_concise = load_reward_model("concise")

# User preference: 70% helpful, 30% concise
weights = [0.7, 0.3]
reward = multi_objective_reward(responses, [rm_helpful, rm_concise], weights)
```

### 6. Offline RL: Conservative Q-Learning (CQL)

**Problem:** Standard Q-learning overestimates values for out-of-distribution actions.

**CQL Objective:**
$$ \min_Q \mathbb{E}_{s,a \sim D} [(Q(s,a) - r)^2] + \alpha \mathbb{E}_s \left[\log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim \pi_\beta} [Q(s,a)]\right] $$

**Interpretation:**
- First term: Standard TD error.
- Second term: Penalize high Q-values for actions not in the dataset.

**Result:** Conservative estimates. Safer for offline training.

### 7. Process Supervision Example (Math)

**Problem:** "Solve: 2x + 3 = 11"

**Correct Reasoning:**
- Step 1: "Subtract 3 from both sides: 2x = 8" ✓ (+1)
- Step 2: "Divide by 2: x = 4" ✓ (+1)
- Final: "x = 4" ✓ (+1)
- **Total Reward:** +3

**Incorrect Reasoning:**
- Step 1: "Divide by 2: x + 3 = 5.5" ✗ (-1)
- Step 2: "Subtract 3: x = 2.5" ✗ (-1)
- Final: "x = 2.5" ✗ (-1)
- **Total Reward:** -3

**PRM Training:**
Train a model to predict +1/-1 for each step.
Use these predictions as rewards during PPO.
