# Day 30: RLHF Fundamentals
## Core Concepts & Theory

### The Alignment Problem

SFT teaches the model *what* to say (format, style).
But it doesn't teach the model *how good* the answer is.
- **Problem:** Given "Explain quantum physics", there are infinite valid responses. Which one is best?
- **Solution:** RLHF (Reinforcement Learning from Human Feedback).

### 1. The RLHF Pipeline

**Three Stages:**

**Stage 1: Supervised Fine-Tuning (SFT)**
- Train on (Prompt, Response) pairs.
- Creates a baseline policy $\pi_{SFT}$.

**Stage 2: Reward Model (RM) Training**
- Collect human preferences: Given Prompt, show 2 responses (A, B). Human picks the better one.
- Train a classifier to predict: $P(A > B | Prompt, A, B)$.
- This is the Reward Model $R(x, y)$.

**Stage 3: RL Fine-tuning (PPO)**
- Use the Reward Model as the reward function.
- Train the policy $\pi$ to maximize: $\mathbb{E}[R(x, y)]$ where $y \sim \pi(x)$.
- Add a KL penalty to prevent the model from drifting too far from $\pi_{SFT}$.

### 2. Preference Data Collection

**Pairwise Comparison:**
- **Prompt:** "Write a poem about cats."
- **Response A:** "Cats are cute and fluffy."
- **Response B:** "Whiskers dance in moonlit grace, / Silent paws in shadowed space."
- **Human:** B > A.

**Ranking:**
- Show 4 responses. Human ranks them: B > D > A > C.

**Likert Scale:**
- Rate each response 1-5. (Less common, harder to calibrate).

### 3. The Reward Model

**Architecture:** Same as the base LLM, but the final layer outputs a scalar (reward score) instead of logits.
**Training:**
- **Input:** (Prompt, Response_A, Response_B, Label).
- **Loss:** Binary Cross-Entropy or Ranking Loss.
$$ L = -\log \sigma(R(x, y_w) - R(x, y_l)) $$
where $y_w$ is the preferred (winner) response, $y_l$ is the rejected (loser) response.

### 4. PPO (Proximal Policy Optimization)

**Goal:** Update the policy to maximize reward without breaking the model.
**Objective:**
$$ \max_\pi \mathbb{E}_{x, y \sim \pi} [R(x, y) - \beta \cdot KL(\pi || \pi_{ref})] $$
- $R(x, y)$: Reward from RM.
- $KL(\pi || \pi_{ref})$: KL divergence from the reference policy (SFT model). Prevents mode collapse.

### Summary of RLHF

| Stage | Input | Output | Goal |
| :--- | :--- | :--- | :--- |
| **SFT** | (Prompt, Response) | Policy $\pi_{SFT}$ | Baseline |
| **RM** | (Prompt, A, B, Winner) | Reward Model $R$ | Preference Predictor |
| **PPO** | Prompts | Policy $\pi_{RLHF}$ | Maximize Reward |

### Next Steps
In the Deep Dive, we will derive the PPO loss function and implement a simple Reward Model in PyTorch.
