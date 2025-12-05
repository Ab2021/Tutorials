# Day 80: Inverse Reinforcement Learning (IRL)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> Don't copy the expert's *actions*. Copy the expert's *intent*.
> - **Focus:** Recovering the Reward Function $R(S)$ from demonstrations. Maximum Entropy IRL.
> - **Code:** A "GridWorld" implementation of MaxEnt IRL. (Full robotic IRL is too slow for a daily exercise).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** BC (Mimicking Actions) with IRL (Mimicking Intent/Reward).
2.  **Derive** the Maximum Entropy IRL objective (Feature Matching).
3.  **Implement** MaxEnt IRL to infer rewards in a Grid Navigation task.
4.  **Discuss** the ambiguity of IRL (Many reward functions explain the same behavior).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- CPU is sufficient for GridWorld.

### Software Environment
```bash
pip install numpy matplotlib gymnasium
```

### Prior Knowledge
- Reinforcement Learning (MDPs, Value Iteration).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Why" of IRL

Behavior Cloning is brittle.
*   **Example:** Expert drives efficiently but slows down for mud.
*   **BC:** "If I see mud, press brake."
*   **IRL:** Infers $Reward = -Distance - 10 \cdot Mud$.
*   **Transfer:** In a new map with *Ice* (which looks different but expert avoids it), BC fails. IRL (if features capture "Slippery") works.

### 🔹 Part 2: Feature Matching

We assume Reward is linear in features:
$$ R(s) = w^T \phi(s) $$
*   $\phi(s)$: Features (Distance to goal, Distance to obstacle, Lane center).
*   $w$: Weights (Unknown).
*   **Goal:** Find $w$ such that the Expected Feature Counts of the Policy match the Expert's Feature Counts.
    $$ \mathbb{E}_{\pi}[\phi(s)] = \mathbb{E}_{expert}[\phi(s)] $$

### 🔹 Part 3: Maximum Entropy (MaxEnt)

There are many $w$ that satisfy the matching (e.g., $w=0$, implies all policies optimal).
*   **Principle:** Choose the distribution over paths with highest entropy (least committed) subject to the method constraints.
*   **Result:** $P(\tau) \propto \exp(\sum R(s_t))$.

---

## 💻 Implementation: MaxEnt IRL in GridWorld

We use a simple 5x5 grid.
*   **Expert:** Goes to Goal (4,4) avoiding Puddles.
*   **Learner:** Watches expert. Infers Puddles are bad. Requires no explicit "Puddle Penalty" definition—it learns it.

### 🛠️ Project Structure
```text
day80_irl/
├── src/
│   ├── gridworld.py (Environment)
│   ├── expert.py (Generates Demos)
│   └── maxent_irl.py (Training)
└── plots/
    └── reward_map.png
```

### 👨‍💻 MaxEnt IRL Logic (`src/maxent_irl.py`)

Simplified Algorithm:
1.  Calculate Expert Feature Expectations $\hat{\phi}_E$.
2.  Loop:
    *   Solve MDP with current weights $w$ (Value Iteration).
    *   Calculate Policy Feature Expectations $\hat{\phi}_\pi$.
    *   Update $w \leftarrow w + \alpha (\hat{\phi}_E - \hat{\phi}_\pi)$.

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_expert_features(trajectories, feature_dim):
    # Sum features across all expert paths
    feat_exp = np.zeros(feature_dim)
    for traj in trajectories:
        for state_feats in traj:
            feat_exp += state_feats
    return feat_exp / len(trajectories)

def maxent_irl(expert_trajectories, mdp, lr=0.1, epochs=50):
    n_states, d_features = mdp.features.shape
    weights = np.random.uniform(size=(d_features,))
    
    expert_feat_exp = compute_expert_features(expert_trajectories, d_features)

    for i in range(epochs):
        # 1. Compute Rewards given current weights
        rewards = mdp.features @ weights
        
        # 2. Solve MDP (Soft Value Iteration for MaxEnt)
        # In simple case, just get optimal policy
        policy = mdp.solve(rewards)
        
        # 3. Compute Expected State Visitation D (Forward Pass)
        # This is the "Policy Feature Expectation" part
        expected_svf = mdp.get_state_visitation_freq(policy)
        policy_feat_exp = mdp.features.T @ expected_svf
        
        # 4. Gradient Ascent
        grad = expert_feat_exp - policy_feat_exp
        weights += lr * grad
        
        print(f"Epoch {i}: Norm Grad {np.linalg.norm(grad):.4f}")

    return weights
```

### 👨‍💻 GridWorld Environment (`src/gridworld.py`)

Contains:
*   State Features: One-hot vector? Or semantics (IsPuddle, IsGoal).
*   Dynamics: Move Up/Down/Left/Right.

---

## 🔬 Lab Exercise: "The Puddle Jumper"

### 1. Lab Objectives
- **Setup:** GridWorld with Goal at (4,4) and a "Puddle" at (2,2).
- **Features:**
    *   $\phi_1 = 1$ (Constant step cost).
    *   $\phi_2 = IsPuddle(s)$.
    *   $\phi_3 = IsGoal(s)$.
- **Expert:** Provided with $R = -1 -10 \cdot \phi_2 + 100 \cdot \phi_3$.
- **Task:** Run Expert to generate 10 paths.
- **Run IRL:** Feed paths to `maxent_irl`.
- **Result:** Check learned `weights`.
    *   Expect $w_3$ (Goal) high positive.
    *   Expect $w_2$ (Puddle) high negative.
    *   Expect $w_1$ (Step) small negative.

---

## 🚀 Project: "Style Transfer"

**Goal:** Learn "Aggressive" vs "Passive" driving styles.
1.  **Expert A (Aggressive):** Drives close to obstacles, high speed.
2.  **Expert B (Passive):** Keeps large distance, low speed.
3.  **IRL:**
    *   Train $w_A$ on A's data. Reward map shows low penalty for obstacles.
    *   Train $w_B$ on B's data. Reward map shows high penalty ("Fear Zones") around obstacles.
4.  **Transplant:** Run Planner with $w_A$ on a *new* map. The robot drives aggressively.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Ambiguity"
*   **Symptom:** Weights oscillate or don't converge.
*   **Cause:** $2R$ produces same policy as $R$. Or $R+Constant$. The solution space is infinite.
*   **Fix:** Regularization (L1/L2 on weights). Fix the "Step Cost" weight to -1.0 to anchor the scale.

#### 2. "Computation Time"
*   **Cause:** We solve the MDP in the inner loop! For large worlds, this is impossible.
*   **Fix:** Use **Deep IRL (GAIL)** or approximate the policy updates.

---

## ⚡ Optimization: GAIL (Generative Adversarial Imitation Learning)

Modern IRL uses GANs.
*   **Discriminator:** Tries to distinguish Expert pairs $(s,a)$ from Policy pairs.
*   **Generator (Policy):** Tries to fool the Discriminator.
*   **Reward:** The output of the Discriminator $D(s,a)$ *is* the reward function.
*   Scales to high dimensions (Images).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** If the expert is suboptimal, what does IRL learn?
    *   **A:** It learns a reward function that *makes* the expert looks optimal (MaxEnt assumption). It essentially "rationalizes" the mistakes.
2.  **Q:** Why not just use RL?
    *   **A:** RL requires you to write the Reward function. "Drive safely" is hard to write mathematically. Demonstrating it is easier.

### Challenge Task
> **Task:** Bad Teacher.
> 1. Generate Expert paths that loop in circles before hitting goal.
> 2. Run IRL.
> 3. Learner will think "Circling" gives positive reward.
> 4. To fix this, you need **Inverse Inverse RL** (inferring expert beliefs) or simply better data.

---

## 📚 Further Reading
- **Ziebart (2008):** "Maximum Entropy Inverse Reinforcement Learning".
- **GAIL (2016):** "Generative Adversarial Imitation Learning".

---

**Day 80 Complete**
