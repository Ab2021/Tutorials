# Day 186: Learning-Based Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Stop coding the physics. Let the robot learn it.
> - **Focus:** Reinforcement Learning (RL), Proximal Policy Optimization (PPO), and Imitation Learning.
> - **Code:** `ppo_swingup.py`. Using Stable Baselines3 to learn a Swing-Up policy for a Pendulum (classic gym env).
> - **Concept:** The Reward Hypothesis.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** the MDP tuple $(S, A, R, P, \gamma)$.
2.  **Train** a PPO agent in OpenAI Gym.
3.  **Explain** the difference between Model-Free (RL) and Model-Based (MPC).
4.  **Implement** Domain Randomization to bridge Sim-to-Real.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU.

### Software Environment
```bash
pip install gymnasium stable-baselines3 shimmy
```

### Prior Knowledge
- Neural Networks (PyTorch).
- Optimal Control (Day 185).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Reinforcement Learning (Model-Free)

Instead of solving $F = ma$, we try random things, get points (Reward), and update our brain (Policy $\pi$).
*   **Critic ($V$):** How good is this state?
*   **Actor ($\pi$):** What should I do?

### 🔹 Part 2: PPO (Proximal Policy Optimization)

The standard for Continuous Control.
*   **Idea:** Don't change the policy *too much* in one step (Trust Region).
*   **Advantage:** Stable, works out of the box.

### 🔹 Part 3: Sim-to-Real Gap

Policies trained in Sim usually fail in Real Life because:
1.  Friction is wrong.
2.  Delays are missing.
*   **Fix:** **Domain Randomization**. Train on 1000 varieties of friction/mass. The policy learns to be robust to physics parameters.

---

## 💻 Implementation: Pendulum Swing-Up

A classic control task. Hard for Linear Control (Linearization at bottom is 0). Easy for RL.

### 🛠️ Project Structure
```text
day186_rl/
├── src/
│   ├── train_ppo.py
│   └── eval_agent.py
└── models/
    └── ppo_pendulum.zip
```

### 👨‍💻 Training Script (`src/train_ppo.py`)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def make_env():
    # Use standard gym env
    return gym.make("Pendulum-v1", render_mode=None) # g=10.0

def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_env()
    
    # PPO Hyperparams
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )
    
    print("Starting Training...")
    # Pendulum is solved in ~20k steps usually
    model.learn(total_timesteps=50000)
    
    model.save("models/ppo_pendulum")
    print("Model Saved.")
    env.close()

if __name__ == "__main__":
    main()
```

### 👨‍💻 Evaluation & Viz (`src/eval_agent.py`)

```python
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

def main():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    model = PPO.load("models/ppo_pendulum")
    
    obs, info = env.reset()
    frames = []
    rewards = []
    
    print("Evaluating...")
    for _ in range(200):
        # Deterministic=True helps for eval
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frames.append(env.render())
        rewards.append(reward)
        
        if terminated or truncated:
            obs, info = env.reset()
            break
            
    env.close()
    
    # Plot Reward
    plt.plot(rewards)
    plt.title("Reward per Step")
    plt.ylabel("Reward")
    plt.xlabel("Time")
    plt.savefig("eval_reward.png")
    print("Evaluation Complete. Check output.")
    
    # In real code, we'd save the video frames to .mp4

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Randomize It!"

### 1. Lab Objectives
- **Create:** A custom Gym Wrapper `RandomizedEnv`.
- **Logic:** In `reset()`, vary the `mass` and `length` of the pendulum by $\pm 20\%$.
- **Train:** Train Agent A on standard, Agent B on Randomized.
- **Test:** Test both on a "Heavy" pendulum (Mass + 50%).
- **Result:** Agent A fails. Agent B succeeds.

---

## 🚀 Project: "Walker-2D"

**Goal:** Teach a biped to walk.
1.  **Env:** `BipedalWalker-v3` (Box2D).
2.  **Training:** This is harder. Needs ~1M steps.
3.  **Reward:** Forward velocity - Energy - Impact.
4.  **Result:** The robot learns a gait (sometimes goofy, but effective).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Reward Hacking"
*   **Scenario:** You penalized torque.
*   **Result:** Robot stands still and does nothing. Zero torque = Minimized penalty.
*   **Fix:** Ensure the "Task Reward" (Going to goal) outweighs the "Energy Cost" significantly.

#### 2. "Simpson's Paradox (Sim vs Real)"
*   **Scenario:** Works in Sim ($v=10$), Fails in Real.
*   **Cause:** Sim acts at 1000Hz with zero delay. Real robot has 50ms USB latency.
*   **Fix:** Add `ActionDelayWrapper` to training. Feed $O_t$ but execute $A_{t-1}$.

---

## ⚡ Optimization: Vectorized Environments

Training on 1 CPU core is slow.
*   `SubprocVecEnv`: Run 16 simulations in parallel.
*   **GPU Env (Isaac Gym/Brax):** Run 4096 simulations on 1 GPU. PPO becomes lightning fast.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Continuous vs Discrete Action Space?
    *   **A:** Discrete (DQN) = Up/Down/Left/Right. Continuous (PPO/SAC) = $v \in [-1.0, 1.0]$. Robots need Continuous.
2.  **Q:** What is "Exploration"?
    *   **A:** Trying actions you think are bad, just in case they are actually good (or lead to good states).

### Challenge Task
> **Task:** "Curriculum Learning".
> 1. Start with Gravity = 0.5g (Moon). Easy to swing up.
> 2. Once solved, increase Gravity to 1.0g.
> 3. Observe faster convergence than starting at 1.0g directly.

---

## 📚 Further Reading
- **OpenAI Spinning Up:** Best intro to RL.
- **Hugging Face Deep RL Course:** Free verified course.

---

## 🔗 External Resources
### 📜 Open Source Libraries
- [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) - PyTorch Reinforcement Learning implementations.
- [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Training framework and zoo of pretrained agents.

### 📺 Video Tutorials
- [Deep Reinforcement Learning for Walking Robots](https://www.youtube.com/results?search_query=Deep+Reinforcement+Learning+for+Walking+Robots) - MathWorks tutorial using Simscape.
- [Intro to RL for Robotics](https://www.youtube.com/results?search_query=Introduction+to+Reinforcement+Learning+Robotics) - Theoretical foundations.

---

**Day 186 Complete**
