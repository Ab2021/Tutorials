# Day 81: Offline RL (Conservative Q-Learning)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> Robots are expensive. Data is cheap.
> - **Focus:** Learning from static datasets (no interaction). The problem of Overestimation. CQL (Conservative Q-Learning).
> - **Code:** A PyTorch implementation of CQL Loss on a static D4RL dataset.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Offline RL (Batch RL) vs Online RL.
2.  **Diagnose** "Extrapolation Error": Why standard Q-learning fails on static data.
3.  **Implement** the CQL Regularization term: $\min Q(s, a_{non-dataset}) - \max Q(s, a_{dataset})$.
4.  **Train** an agent on a "Replay Buffer" collected by another policy.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU.

### Software Environment
```bash
pip install d4rl # Datasets for Deep Data-Driven RL
pip install torch
```

### Prior Knowledge
- Q-Learning (Bellman Equation).
- Day 80 (IRL).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Offline Promise

*   **Online RL:** Robot tries -> Fails -> Learns. (Risky, Slow).
*   **Offline RL:** Use 1,000 hours of existing logs (from humans or old robots). Learn optimal policy. Deploy.
*   **Analogy:** Learning to play chess by reading a book of GM games, without playing a single move.

### 🔹 Part 2: The Action Distribution Shift

Bellman Update:
$$ Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a') $$
*   We query the Q-function for the *best* action $a'$.
*   If $a'$ is an action we *never saw* in the dataset, the Q-function might hallucinate a huge value (e.g., "Teleport").
*   Since we max over $a'$, we select these hallucinations.
*   **Result:** The policy picks crazy actions.

### 🔹 Part 3: Conservative Q-Learning (CQL)

Idea: Be pessimistic. If I haven't seen it, assume it's bad.
New Objective: $\text{Standard Bellman Error} + \alpha \cdot \text{CQL Term}$
$$ \mathcal{L}_{CQL} = \mathbb{E}_{s \sim D} \left[ \log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim D}[Q(s,a)] \right] $$
*   **Minimize** Q-values of random actions.
*   **Maximize** Q-values of actions in the dataset.
*   This creates a "Valley" of low value around the data distribution.

---

## 💻 Implementation: CQL Agent

We extend a standard SAC (Soft Actor-Critic) agent with CQL loss.

### 🛠️ Project Structure
```text
day81_offline_rl/
├── src/
│   ├── cql_agent.py
│   └── train_offline.py
└── data/
    └── dataset.hdf5
```

### 👨‍💻 CQL Loss Implementation (`src/cql_agent.py`)

```python
import torch
import torch.nn.functional as F

class CQLAgent:
    def __init__(self, q_net, target_q_net, optimizer):
        self.q_net = q_net
        self.optimizer = optimizer
        self.cql_weight = 1.0

    def compute_loss(self, states, actions, rewards, next_states, dones):
        # 1. Standard Bellman Error (TD Error)
        with torch.no_grad():
            next_actions = self.actor(next_states) # From Policy
            target_q = self.target_q_net(next_states, next_actions)
            target = rewards + (1 - dones) * 0.99 * target_q
        
        current_q = self.q_net(states, actions)
        bellman_loss = F.mse_loss(current_q, target)
        
        # 2. CQL Regularization
        # We need to sample "Random" actions to minimize their Q
        batch_size = states.shape[0]
        random_actions = torch.rand((batch_size, 10, action_dim)).to(states.device) # 10 samples
        
        # Also sample from current Policy
        curr_actions, log_pi = self.actor(states, num_samples=10)
        
        # Combine
        cql_q_rand = self.q_net(states, random_actions)
        cql_q_curr = self.q_net(states, curr_actions)
        cql_q_data = current_q.unsqueeze(1) # Dataset actions
        
        # LogSumExp (Push down non-data actions)
        # Loss = log( exp(Q_rand) + exp(Q_curr) ) - Q_data
        cat_q = torch.cat([cql_q_rand, cql_q_curr], dim=1)
        cql_loss = (torch.logsumexp(cat_q, dim=1) - cql_q_data).mean()
        
        total_loss = bellman_loss + self.cql_weight * cql_loss
        return total_loss

    def update(self, batch):
        loss = self.compute_loss(*batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 🔬 Lab Exercise: "The Safe Driver"

### 1. Lab Objectives
- **Dataset:** 100 trajectories of a robot driving randomly (often crashing).
- **Goal:** Learn a safe driving policy.
- **Run Standard Q-Learning:**
    *   Result: Robot assumes it can "phase through walls" because Q-values for wall-entry were implicitly extrapolated high.
- **Run CQL:**
    *   Result: Robot sticks *strictly* to the safe paths seen in the dataset. It learns to avoid the crashes even though it never interacts.

---

## 🚀 Project: "From Logs to Riches"

**Goal:** Use the "Expert Data" from Day 78 (BC) to train an RL agent.
1.  **Data:** Day 78 HDF5 file. (It contains "Good" driving).
2.  **Augment:** Add some "Random" noise data (Bad driving) to the buffer.
3.  **Train:** CQL Agent.
4.  **Compare:**
    *   BC Agent: Tries to average Good + Bad data? (If dataset is mixed).
    *   CQL Agent: Finds the *Best* behavior in the mixed dataset (Stitching).
    *   Example: Path A goes Left. Path B goes Right. BC averages $\to$ Middle (Crash). CQL picks max Q $\to$ Left or Right.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Q-values diverge to negative infinity"
*   **Cause:** CQL pushes Q-values down. If `cql_weight` is too high, they keep dropping.
*   **Fix:** Use specific "CQL with Importance Sampling" or just tune `alpha`.

#### 2. "Action Constraints"
*   **Symptom:** Policy outputs values outside [-1, 1].
*   **Fix:** Always Tanh squash the output of the Actor network.

---

## ⚡ Optimization: IQL (Implicit Q-Learning)

CQL requires sampling actions (expensive).
*   **IQL:** Completely in-sample algorithms.
*   Uses Expectile Regression to estimate specific quantiles of the value function.
*   Much faster training, very stable. State-of-the-Art for Offline RL.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can Offline RL do better than the Demonstrator?
    *   **A:** Yes! Through "Stitching". If Expert A does Start $\to$ Middle, and Expert B does Middle $\to$ Goal. Offline RL can combine them to optimal Start $\to$ Goal, even if no single expert did it.
2.  **Q:** Why not just run Online RL?
    *   **A:** Safety. You can't let a robot flail around on a highway to learn driving.
3.  **Q:** What is the "Optimism via Uncertainty" in Online RL?
    *   **A:** "If I don't know, it might be awesome!" (Exploration). Offline RL is opposite: "If I don't know, it's terrible" (Pessimism).

### Challenge Task
> **Task:** Mixed Quality Data.
> 1. Dataset: 10% Expert, 90% Random.
> 2. BC fails (copies random).
> 3. CQL should recover near-expert performance by assigning low value to the random trajectories.

---

## 📚 Further Reading
- **CQL Paper (Kumar et al., 2020):** "Conservative Q-Learning for Offline Reinforcement Learning".
- **D4RL:** Benchmarks for Deep Data-Driven RL.

---

**Day 81 Complete**
