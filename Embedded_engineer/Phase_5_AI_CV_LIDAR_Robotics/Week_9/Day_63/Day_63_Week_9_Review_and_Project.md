# Day 63: Week 9 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> We have a Digital Twin that behaves like the Real Robot. Now we prove it.
> - **Goal:** A complete Sim-to-Real pipeline. Train in Sim $\to$ Deploy on Real.
> - **Code:** Training a Reinforcement Learning agent (Wall Follower) and transferring it.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** URDF modeling, Gazebo Physics, and Sensor Noise into a training environment.
2.  **Train** a simple RL policy (PPO) in Simulation using Stable Baselines3.
3.  **Deploy** the trained `.zip` model onto a physical robot without retraining.
4.  **Evaluate** the robustness of the policy against Real World disturbances.

---

## 📚 Week 9 Review: The Simulation Stack

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **57** | **URDF/SDF** | Describe the Robot | Xacro |
| **58** | **Physics** | Simulate Gravity/Contact | Gazebo |
| **59** | **Sensors** | Simulate Beams/Cameras | Plugins |
| **60** | **Randomization** | Bridge the Reality Gap | DR |
| **61** | **Digital Twin** | Sync Real $\leftrightarrow$ Sim | ROS Bridge |
| **62** | **SysID** | Calibrate Physics | Least Squares |

### The "Golden Rule" of Simulation
*   **Garbage In, Garbage Out.** If your Inertia Matrix is wrong, your Controller gains will be wrong for the real robot.
*   **All models is wrong, some are useful.** Don't aim for atomic perfection. Aim for *Behavioral Fidelity*.

---

## 🚀 Weekly Capstone: "Sim-to-Real Wall Follower"

**Scenario:** A 2-wheeled robot.
**Task:**
1.  **Sim Env:** Create a winding corridor in Gazebo.
2.  **Training:** Train PPO to keep distance $d=0.5m$ from the right wall.
3.  **Randomization:** Randomize friction and wall textures during training.
4.  **Deploy:** Put real robot in a real corridor. Run the neural net.

### 🛠️ Project Structure
```text
week9_capstone/
├── sim_pkg/
│   ├── envs/
│   │   └── gazebo_gym_env.py
│   └── train_ppo.py
├── real_pkg/
│   └── deploy_policy.py
└── models/
    └── wall_follower_policy.zip
```

### 👨‍💻 Component 1: The Gym Environment (`sim_pkg/envs/gazebo_gym_env.py`)

Using `gymnasium` interface wrapping ROS topics.

```python
import gymnasium as gym
import numpy as np

class GazeboWallEnv(gym.Env):
    def __init__(self):
        # Observation: [Range_Front, Range_Right, Range_Back]
        self.observation_space = gym.spaces.Box(0, 10, shape=(3,))
        # Action: [Linear_Vel, Angular_Vel]
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        
    def step(self, action):
        # 1. Send Action to Gazebo (cmd_vel)
        self.pub_vel.publish(action_to_twist(action))
        
        # 2. Get State (Lidar)
        scan = self.get_scan() 
        state = process_scan(scan)
        
        # 3. Calculate Reward
        # Reward = 1.0 - abs(Dist_Right - 0.5)
        # Penalty if collision (Dist < 0.1)
        r = get_range_right(scan)
        reward = 1.0 - abs(r - 0.5)
        if min(scan) < 0.1:
            reward = -100
            done = True
        else:
            done = False
            
        return state, reward, done, {}
```

### 👨‍💻 Component 2: Domain Randomization Hook

Inside `reset()`:

```python
    def reset(self):
        # Randomize Physics (Friction)
        friction = np.random.uniform(0.5, 1.0)
        self.set_friction_service(friction)
        
        # Randomize Sensor Noise (Update Plugin Param)
        # (Simplified: Add noise in python processing)
        self.noise_level = np.random.uniform(0.0, 0.05)
        
        return self.get_state()
```

### 👨‍💻 Component 3: Deployment Node (`real_pkg/deploy_policy.py`)

Runs on the Physical Robot (Jetson Nano).

```python
from stable_baselines3 import PPO
import rclpy
from geometry_msgs.msg import Twist

model = PPO.load("wall_follower_policy.zip")

def timer_callback():
    # 1. Get Real Lidar
    scan = get_real_scan()
    obs = process_scan(scan) # Must match Sim processing exactly!
    
    # 2. Inference
    action, _ = model.predict(obs, deterministic=True)
    
    # 3. Act
    vel_msg = Twist()
    vel_msg.linear.x = float(action[0])
    vel_msg.angular.z = float(action[1])
    pub.publish(vel_msg)
```

---

## 📝 Self-Assessment Quiz

1.  **Safety:**
    *   What happens if the Neural Net outputs `linear.x = 100 m/s`?
    *   **A:** The Simulator might handle it (or crash). The Real Robot will burn motors.
    *   **Fix:** **Action Clipping** and Safety Layers (Day 46) are mandatory *after* the Neural Net output.

2.  **Latency:**
    *   Sim is synchronous (Step $\to$ Obs $\to$ Step). Real is asynchronous (Lidar comes when it wants).
    *   How to handle?
    *   **A:** Run inference at fixed rate (e.g., 20Hz). Use most recent scan. Add Observation Buffering (stack last 3 frames) to capture velocity/delay.

---

## ⏭️ Look Ahead: Week 10
We have simulated single robots. Now we need them to talk to the Cloud.
**Week 10: Swarm Robotics.**
*   Multi-Robot Systems (MRS).
*   Map Merging.
*   Leader Election.
*   Mesh Networking.

*(Wait, Week 10 was originally "Swarm Optimization" in some outlines, but based on Phase 5 progression, Weeks 5-6 covered Swarm. Week 10 should be "Advanced Deployment & Cloud"? Let's check user's original outline. Ah, Week 6 was Swarm. Week 10 should be **Capstone & Final Integration** or **Cloud Robotics**).*

*Correction:* Based on the current flow:
Week 10 will focus on **Cloud Robotics & Fleet Management**.
*   AWS RoboMaker / Google Cloud Robotics.
*   Docker/Kubernetes for Robots.
*   Over-the-Air (OTA) Updates.
*   Centralized Fleet Monitoring.

---

**Week 9 Complete**
