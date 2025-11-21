# Day 31: Model-Based Reinforcement Learning

## 1. Model-Free vs. Model-Based RL
*   **Model-Free (DQN, PPO, SAC):** Learn the policy/value function directly from experience. Sample-inefficient but robust.
*   **Model-Based:** Learn a **model** of the environment dynamics, then use it for planning. Sample-efficient but can suffer from model errors.

**Environment Model:**
$$ T(s' | s, a) = P(s_{t+1} = s' | s_t = s, a_t = a) $$
$$ R(s, a) = \mathbb{E}[r_t | s_t = s, a_t = a] $$

## 2. Dyna-Q: Integrating Planning and Learning
Dyna-Q (Sutton, 1990) combines:
*   **Direct RL:** Learn Q-values from real experience.
*   **Model Learning:** Learn a model from real experience.
*   **Planning:** Use the model to simulate experience and update Q-values.

**Algorithm:**
1.  Take action $a$ in state $s$, observe $r, s'$.
2.  **Direct RL:** $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
3.  **Model Learning:** $T(s, a) \leftarrow s'$, $R(s, a) \leftarrow r$ (store in table or learn neural net).
4.  **Planning:** Repeat $n$ times:
    *   Sample $(s, a)$ from previously visited states.
    *   Simulate $s', r$ using the model.
    *   Update: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.

## 3. Benefits of Model-Based RL
*   **Sample Efficiency:** Can "imagine" many experiences without interacting with the environment.
*   **Transfer Learning:** A learned model can generalize to new tasks (if the dynamics are the same).
*   **Planning:** Can solve new tasks by planning with the model (no retraining).

## 4. Challenges
*   **Model Error:** If the model is inaccurate, planning will be suboptimal (compounding errors).
*   **Model Complexity:** Modeling high-dimensional observations (images) is difficult.
*   **Exploration:** How to explore efficiently to learn a good model?

## 5. Code Example: Dyna-Q
```python
import numpy as np

class DynaQ:
    def __init__(self, n_states, n_actions, planning_steps=5):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # Model: (s, a) -> (r, s')
        self.planning_steps = planning_steps
        self.visited = set()  # Track visited (s, a) pairs
        
    def update(self, s, a, r, s_next, alpha=0.1, gamma=0.99):
        # Direct RL update
        td_target = r + gamma * np.max(self.Q[s_next])
        self.Q[s, a] += alpha * (td_target - self.Q[s, a])
        
        # Model Learning
        self.model[(s, a)] = (r, s_next)
        self.visited.add((s, a))
        
        # Planning
        for _ in range(self.planning_steps):
            # Sample random previously visited (s, a)
            s_plan, a_plan = random.choice(list(self.visited))
            r_plan, s_next_plan = self.model[(s_plan, a_plan)]
            
            # Simulated update
            td_target_plan = r_plan + gamma * np.max(self.Q[s_next_plan])
            self.Q[s_plan, a_plan] += alpha * (td_target_plan - self.Q[s_plan, a_plan])
    
    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
```

## 6. Modern Model-Based RL
*   **World Models (Ha & Schmidhuber, 2018):** Learn a compressed latent model, train policy in imagination.
*   **Dreamer (Hafner et al., 2020):** State-of-the-art MBRL using latent dynamics models and actor-critic.
*   **MuZero (Schrittwieser et al., 2020):** Learns a model for **planning only** (not dynamics), achieves superhuman Atari.

### Key Takeaways
*   Model-Based RL is sample-efficient but sensitive to model errors.
*   Dyna-Q integrates planning with learning.
*   Modern MBRL uses learned latent models for imagination-based learning.
