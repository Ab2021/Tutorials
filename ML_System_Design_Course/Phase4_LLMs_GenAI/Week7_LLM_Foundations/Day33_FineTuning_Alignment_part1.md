# Day 33 (Part 1): Advanced Alignment (DPO/KTO)

> **Phase**: 6 - Deep Dive
> **Topic**: Steering the Model
> **Focus**: DPO Derivation, KTO, and Safety
> **Reading Time**: 60 mins

---

## 1. DPO Derivation

### 1.1 The Math
*   Start with RL objective: Maximize Reward - KL Divergence.
*   Optimal policy $\pi^*(y|x) \propto \pi_{ref}(y|x) e^{R(x,y)}$.
*   Rearrange: $R(x,y) = \log \frac{\pi^*}{\pi_{ref}}$.
*   Substitute R into Bradley-Terry model.
*   **Result**: Loss depends only on $\pi$ and $\pi_{ref}$. No Reward Model needed.

---

## 2. KTO (Kahneman-Tversky Optimization)

### 2.1 The Idea
*   DPO needs pairs (Winner, Loser).
*   **KTO**: Needs only (Prompt, Output, Label: Good/Bad). Unpaired data.
*   **Benefit**: Much easier to collect data (Like/Dislike button).

---

## 3. Tricky Interview Questions

### Q1: Why does RLHF lead to "Mode Collapse"?
> **Answer**:
> *   Model finds a specific style (e.g., "As an AI language model...") that gets high reward.
> *   It loses diversity.
> *   **Fix**: Higher KL penalty.

### Q2: Constitutional AI (Anthropic)?
> **Answer**:
> *   **Phase 1**: Supervised Learning on "Helpful" data.
> *   **Phase 2**: RLAIF (RL from AI Feedback).
> *   Model critiques its own harmful outputs based on a "Constitution" (Rules). Generates its own preference data.

### Q3: Rejection Sampling (Best-of-N)?
> **Answer**:
> *   Generate N outputs.
> *   Score with Reward Model.
> *   Pick best.
> *   Train on that. (Simple alternative to PPO).

---

## 4. Practical Edge Case: Reward Hacking
*   **Example**: Model learns to write long answers because humans prefer length.
*   **Fix**: Length penalty in Reward Model.

