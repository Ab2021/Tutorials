# Day 84: Week 12 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> We have moved from "Programming" (Weeks 1-11) to "Teaching" (Week 12).
> - **Goal:** Synthesize BC, RL, and Sim-to-Real into a cohesive workflow.
> - **Code:** A `train_pipeline.py` that orchestrates Data Collection $\to$ BC Training $\to$ Sim Evaluation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the right Learning Algorithm (BC vs RL) based on data availability and risk.
2.  **Pipeline** the training process: Data $\to$ Train $\to$ Eval $\to$ Deploy.
3.  **Critique** the "Black Box" nature of Neural Policies vs Classically Planned logic.

---

## 📚 Week 12 Review: The Learning Stack

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **78** | **Imitation Learning** | Behavior Cloning (Supervised) | `torch`, `rosbag` |
| **79** | **DAgger** | Interactive corrections fix drift | `Human-in-Loop` |
| **80** | **Inverse RL** | Learning the Reward Function | `MaxEnt IRL` |
| **81** | **Offline RL** | Learning from static datasets (CQL) | `d4rl`, `CQL` |
| **82** | **Diffusion Policies** | Multimodal Trajectory Generation | `Diffusers` |
| **83** | **Sim-to-Real** | Domain Randomization closes the gap | `PyBullet`, `Gym` |

### The "Spectrum of Autonomy"
1.  **Classical:** 100% Manually Coded (Nav2, MoveIt). Interpretable. Brittle.
2.  **Imitation:** 100% Copied from Human. Smooth. Generalizes poorly.
3.  **RL:** 100% Learned from Trial & Error. Robust. Hard to train (Safety).

---

## 🚀 Weekly Capstone: "The Self-Correction Loop"

**Scenario:** Mobile Robot Navigation in a Crowded Office.
**Mission:** "Go to the Coffee Machine".
**Method:** Learning from Demonstration + Sim-to-Real.

### 🛠️ Project Structure
```text
week12_capstone/
├── src/
│   ├── record_demo.py
│   ├── train.py
│   └── eval_sim.py
├── config/
│   └── diffusion_config.yaml
└── models/
    └── policy_v1.pth
```

### 👨‍💻 Workflow

1.  **Step 1: Calibration (Sim-to-Real)**
    *   Measure Real Robot max accel.
    *   Update URDF mass.
    *   Set Sim Domain Randomization bounds (Friction $\pm 0.2$).
    
2.  **Step 2: Collect Demonstrations**
    *   Human drives the robot in Sim (10 laps).
    *   Save `demos.h5`.
    
3.  **Step 3: Train Diffusion Policy**
    *   Input: Lidar Scan (1D).
    *   Output: Velocity (2D).
    *   Epochs: 100.
    
4.  **Step 4: DAgger (Sim)**
    *   Run Policy.
    *   Human intervenes if it gets stuck.
    *   Retrain.
    
5.  **Step 5: Deployment**
    *   Load `policy_v1.pth` on Real Robot.
    *   Safety Watchdog: If `min_dist < 0.2m`, Override Stop.

### 👨‍💻 Training Script Snippet (`src/train.py`)

Integrating everything.

```python
import torch
from diffusion_policy import DiffusionModel, NoisePredNet
from dataset import RobotDataset

def train():
    # 1. Data
    ds = RobotDataset("data/demos.h5")
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    # 2. Model
    net = NoisePredNet(action_dim=2, obs_dim=360) # Lidar
    model = DiffusionModel(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    # 3. Loop
    for epoch in range(100):
        total_loss = 0
        for obs, act in loader:
            loss = model.loss(act, obs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {total_loss/len(loader)}")
        
    torch.save(net.state_dict(), "models/policy_v1.pth")
```

---

## 📝 Self-Assessment Quiz

1.  **Learning:**
    *   Why is "Offline RL" safer than "Online RL"?
    *   **A:** Online RL requires the robot to explore (try bad actions). Offline learning happens on a computer using past data.
2.  **Diffusion:**
    *   How does Diffusion handle the "Fork in the Road"?
    *   **A:** It represents the distribution as particles. 50 particles go left, 50 go right. None go straight.
3.  **Sim-to-Real:**
    *   What is "System Identification"?
    *   **A:** Calibrating the Sim parameters (Mass, Friction) to match Reality *before* doing Randomization. Reduces the range needed for DR.

---

## ⏭️ Look Ahead: Week 13
We return to the factory floor.
**Week 13: Industrial Robotics & Cobots.**
*   MoveIt Pro, ROS-Industrial, PLCs, and Safety Controllers.
*   We will learn how to integrate ROS 2 with "Old School" automation (Modbus/Profinet).

---

**Week 12 Complete**
