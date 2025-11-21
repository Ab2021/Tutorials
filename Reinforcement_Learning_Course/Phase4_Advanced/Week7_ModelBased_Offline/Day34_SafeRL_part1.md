# Day 34 Deep Dive: RLHF and AI Safety

## 1. Reinforcement Learning from Human Feedback (RLHF)
Used to align language models (ChatGPT, Claude).

**Process:**
1.  **Supervised Fine-Tuning:** Train on human demonstrations.
2.  **Reward Modeling:** Train a reward model from human preferences.
3.  **RL Fine-Tuning:** Use PPO to optimize the policy w.r.t. the learned reward.

## 2. Interpretability and Explainability
*   **Attention Visualization:** Understand what the agent focuses on.
*   **Feature Attribution:** Which features drive decisions?
*   **Policy Distillation:** Extract interpretable rules from learned policies.

## 3. Robustness and Adversarial RL
*   **Adversarial Training:** Train against worst-case perturbations.
*   **Domain Randomization:** Train on diverse environments for robustness.
*   **Certified Robustness:** Provable guarantees on safety.

## 4. Value Alignment
*   **Reward Hacking:** Agents exploit flaws in reward design.
*   **Side Effects:** Unintended consequences of achieving the goal.
*   **Corrigibility:** Can the agent be safely interrupted or corrected?
