# Day 30: RLHF Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Bradley-Terry Model

The Reward Model is based on the Bradley-Terry model from statistics.
**Assumption:** Each response has a "true quality" score $R(x, y)$.
The probability that response $y_1$ is preferred over $y_2$ is:
$$ P(y_1 > y_2 | x) = \frac{e^{R(x, y_1)}}{e^{R(x, y_1)} + e^{R(x, y_2)}} = \sigma(R(x, y_1) - R(x, y_2)) $$
where $\sigma$ is the sigmoid function.

**Training Loss:**
Given a dataset of comparisons $(x, y_w, y_l)$ where $y_w$ is preferred:
$$ L = -\log \sigma(R(x, y_w) - R(x, y_l)) $$
This is equivalent to Binary Cross-Entropy.

### 2. PPO Clipped Objective

**Vanilla Policy Gradient:**
$$ L = \mathbb{E} [r_t \cdot \nabla \log \pi(a_t | s_t)] $$
**Problem:** Large updates can destroy the policy.

**PPO Clipped:**
$$ L^{CLIP} = \mathbb{E} \left[ \min \left( \frac{\pi(a|s)}{\pi_{old}(a|s)} A, \text{clip}\left(\frac{\pi(a|s)}{\pi_{old}(a|s)}, 1-\epsilon, 1+\epsilon\right) A \right) \right] $$
- $A$: Advantage (how much better this action is than average).
- $\epsilon$: Clip range (e.g., 0.2).
**Effect:** Prevents the ratio $\frac{\pi}{\pi_{old}}$ from going beyond $[0.8, 1.2]$. This limits the step size.

### 3. KL Penalty vs. KL Constraint

**KL Penalty (Soft):**
$$ R_{total} = R_{RM}(x, y) - \beta \cdot KL(\pi(y|x) || \pi_{ref}(y|x)) $$
**KL Constraint (Hard):**
$$ \max R_{RM}(x, y) \quad \text{s.t.} \quad KL(\pi || \pi_{ref}) < \delta $$
**Adaptive $\beta$:**
If $KL$ is too high, increase $\beta$ (penalize drift more).
If $KL$ is too low, decrease $\beta$ (allow more exploration).

### 4. Reward Hacking

**Problem:** The model finds adversarial inputs that maximize the Reward Model's score without actually being good.
**Example:**
- RM was trained to prefer longer responses.
- Model learns to generate infinite repetition: "The answer is yes. The answer is yes. The answer is yes..."
**Mitigation:**
- Length Penalty.
- Ensemble of RMs.
- Human-in-the-loop monitoring.

### Code: Simple Reward Model

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        self.value_head = nn.Linear(self.transformer.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Get last hidden state
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        # Pool: Take the last token's hidden state
        last_hidden = hidden[:, -1, :]
        
        # Scalar reward
        reward = self.value_head(last_hidden).squeeze(-1)
        return reward

# Training Loop (Simplified)
def train_reward_model(model, dataloader, optimizer):
    for batch in dataloader:
        # batch: {prompt_ids, chosen_ids, rejected_ids}
        r_chosen = model(batch['chosen_ids'], batch['chosen_mask'])
        r_rejected = model(batch['rejected_ids'], batch['rejected_mask'])
        
        # Bradley-Terry Loss
        loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
