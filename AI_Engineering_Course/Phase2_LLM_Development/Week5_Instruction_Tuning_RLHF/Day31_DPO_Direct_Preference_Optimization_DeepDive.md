# Day 31: DPO (Direct Preference Optimization)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Mathematical Derivation of DPO

**Starting Point: RLHF Objective**
$$ \max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} [R(x, y)] - \beta \cdot D_{KL}(\pi(\cdot|x) || \pi_{ref}(\cdot|x)) $$

**Optimal Solution (Partition Function):**
Using Lagrange multipliers, the optimal policy is:
$$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right) $$
where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)$ is the partition function.

**Solving for Reward:**
$$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right) $$
$$ Z(x) \pi^*(y|x) = \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right) $$
$$ \log Z(x) + \log \pi^*(y|x) = \log \pi_{ref}(y|x) + \frac{1}{\beta} R(x, y) $$
$$ R(x, y) = \beta \left[\log \pi^*(y|x) - \log \pi_{ref}(y|x) + \log Z(x)\right] $$

**Bradley-Terry Model:**
$$ P(y_w > y_l | x) = \frac{\exp(R(x, y_w))}{\exp(R(x, y_w)) + \exp(R(x, y_l))} = \sigma(R(x, y_w) - R(x, y_l)) $$

**Substituting:**
$$ R(x, y_w) - R(x, y_l) = \beta \left[\log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right] $$
Note: $\log Z(x)$ cancels out!

**DPO Loss:**
$$ L_{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right] $$

### 2. Gradient Analysis

**Gradient of DPO Loss:**
$$ \nabla_\theta L_{DPO} = -\beta \mathbb{E} \left[\sigma(-\Delta) \left(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right)\right] $$
where $\Delta = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$.

**Interpretation:**
- If $\Delta > 0$ (model already prefers $y_w$), $\sigma(-\Delta)$ is small. Small gradient.
- If $\Delta < 0$ (model prefers $y_l$), $\sigma(-\Delta)$ is large. Large gradient.
- The gradient pushes the model to increase $\pi(y_w|x)$ and decrease $\pi(y_l|x)$.

### 3. Implicit Regularization

**Implicit KL Penalty:**
The term $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ acts as an implicit KL penalty.
If $\pi_\theta$ drifts too far from $\pi_{ref}$, this ratio becomes large, and the loss increases.
**Result:** DPO automatically prevents mode collapse without explicit KL computation.

### 4. Length Bias in DPO

**Problem:**
If $y_w$ is longer than $y_l$, then $\log \pi_\theta(y_w|x)$ is the sum of more log probabilities.
This can bias the model to prefer longer responses.

**Solution:**
Length-normalized DPO:
$$ \Delta = \beta \left[\frac{1}{|y_w|} \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \frac{1}{|y_l|} \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right] $$

### 5. DPO vs. SFT on Winners

**Naive Approach:** Just do SFT on the winning responses $y_w$.
**Problem:** This ignores the *contrastive* signal. The model doesn't learn what NOT to do.
**DPO:** Explicitly increases $\pi(y_w)$ and decreases $\pi(y_l)$. Much more sample-efficient.

### Code: Complete DPO Training Loop

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    Compute DPO loss for a batch of preferences.
    batch: {
        'prompt_ids': [B, L_prompt],
        'chosen_ids': [B, L_chosen],
        'rejected_ids': [B, L_rejected]
    }
    """
    # Concatenate prompt + chosen
    chosen_input_ids = torch.cat([batch['prompt_ids'], batch['chosen_ids']], dim=1)
    rejected_input_ids = torch.cat([batch['prompt_ids'], batch['rejected_ids']], dim=1)
    
    # Forward pass through policy
    chosen_logits = policy_model(chosen_input_ids).logits
    rejected_logits = policy_model(rejected_input_ids).logits
    
    # Forward pass through reference (no grad)
    with torch.no_grad():
        chosen_ref_logits = ref_model(chosen_input_ids).logits
        rejected_ref_logits = ref_model(rejected_input_ids).logits
    
    # Compute log probabilities
    chosen_logprobs = compute_logprobs(chosen_logits, batch['chosen_ids'])
    rejected_logprobs = compute_logprobs(rejected_logits, batch['rejected_ids'])
    chosen_ref_logprobs = compute_logprobs(chosen_ref_logits, batch['chosen_ids'])
    rejected_ref_logprobs = compute_logprobs(rejected_ref_logits, batch['rejected_ids'])
    
    # DPO loss
    pi_logratios = chosen_logprobs - rejected_logprobs
    ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs
    logits = beta * (pi_logratios - ref_logratios)
    
    loss = -F.logsigmoid(logits).mean()
    
    # Metrics
    accuracy = (logits > 0).float().mean()
    
    return loss, accuracy

def compute_logprobs(logits, labels):
    """Compute log probabilities of labels given logits."""
    # Shift logits and labels for next-token prediction
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for the labels
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # Sum over sequence
    return per_token_logps.sum(dim=1)

# Training Loop
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model.eval()  # Freeze reference

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)

for epoch in range(3):
    for batch in dataloader:
        loss, acc = dpo_loss(model, ref_model, batch, beta=0.1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.2%}")
```

### 6. Theoretical Guarantees

**Theorem (Rafailov et al., 2023):**
Under mild assumptions, DPO converges to the same optimal policy as RLHF.
**Implication:** DPO is not an approximation; it's an exact reparameterization of the RLHF objective.
