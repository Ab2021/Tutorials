# Day 36: RLHF & Alignment - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Reinforcement Learning from Human Feedback, PPO, and DPO

## 1. Theoretical Foundation: Alignment

Pre-training (Next Token Prediction) makes models **capable** but not **helpful**.
They might generate toxic content, lie, or be vague.
**Alignment**: Making the model behave according to human intent (Helpful, Honest, Harmless).

## 2. The RLHF Pipeline (InstructGPT)

1.  **SFT (Supervised Fine-Tuning)**: Train on high-quality (Prompt, Response) pairs.
2.  **Reward Modeling (RM)**:
    *   Collect comparison data: "Which response is better, A or B?"
    *   Train a Reward Model $r(x, y)$ to predict human preference.
3.  **PPO (Proximal Policy Optimization)**:
    *   Optimize the LLM policy $\pi$ to maximize reward $r$.
    *   Constraint: Don't drift too far from SFT model (KL Divergence).

## 3. DPO (Direct Preference Optimization)

RLHF is complex (needs 4 models in memory: Actor, Critic, Ref, Reward). Unstable.
**DPO (2023)**:
*   Derives the optimal policy analytically.
*   Optimizes the LLM directly on preference data using a simple classification loss.
*   No Reward Model needed. No PPO.
$$ L_{DPO} = - \log \sigma (\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}) $$

## 4. Implementation: DPO with TRL

Hugging Face `trl` (Transformer Reinforcement Learning) library.

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2") # Frozen copy

# 2. Data (Prompt, Chosen, Rejected)
dataset = [
    {"prompt": "Hi", "chosen": "Hello! How can I help?", "rejected": "What do you want?"}
]

# 3. Train
dpo_config = DPOConfig(beta=0.1)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=dpo_config
)

trainer.train()
```

## 5. Constitutional AI

Instead of human feedback, use AI feedback (RLAIF).
1.  Ask LLM to critique its own response based on a "Constitution" (e.g., "Do not be racist").
2.  Revise the response.
3.  Train on the revised data.
