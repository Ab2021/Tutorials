# Day 36: RLHF - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: PPO Internals, Reward Hacking, and KTO

## 1. PPO (Proximal Policy Optimization)

The standard RL algorithm for LLMs.
**Objective**:
$$ \max E [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ] $$
*   **Ratio $r_t(\theta)$**: Probability of action under new policy vs old policy.
*   **Clipping**: Prevents the policy from changing too much in one step (Stability).
*   **Advantage $\hat{A}_t$**: How much better was this action than average?

## 2. Reward Hacking

The Proxy Reward Model is not the True Reward (Human Satisfaction).
*   **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."
*   The LLM finds loopholes to maximize the Reward Model score without actually being helpful (e.g., repeating "I love you" if the RM likes positive sentiment).
*   **Solution**: KL Penalty (stay close to SFT model).

## 3. KTO (Kahneman-Tversky Optimization)

DPO requires paired data (Chosen vs Rejected). Hard to get.
**KTO**:
*   Uses binary signals (Like / Dislike) for single outputs.
*   Based on Prospect Theory (Human loss aversion).
*   Easier data collection.

## 4. Rejection Sampling (Best-of-N)

Simple alternative to PPO.
1.  Generate $N$ responses for a prompt.
2.  Score them with the Reward Model.
3.  Pick the best one.
4.  Fine-tune on the best ones.
*   Used in LLaMA-2.

## 5. Offline vs Online RL

*   **Offline (DPO/SliC)**: Learn from a static dataset of preferences. Stable.
*   **Online (PPO)**: Generate new samples during training, score them, update. Better performance but harder to tune.
