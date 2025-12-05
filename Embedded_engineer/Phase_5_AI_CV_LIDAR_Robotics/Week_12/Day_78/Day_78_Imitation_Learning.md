# Day 78: Imitation Learning (Behavior Cloning)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> Don't code the policy. Show the policy.
> - **Focus:** Collecting Expert Demonstrations, Behavior Cloning (BC) as Supervised Learning, and the Covariate Shift problem.
> - **Code:** A PyTorch BC agent trained on human demonstrations (teleop data).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Record** a dataset of human demonstrations (Observation, Action) pairs.
2.  **Train** a Convolutional Neural Network (CNN) to map Images $\to$ Velocity Commands.
3.  **Identify** the "Covariate Shift" failure mode (Drift leads to unknown states).
4.  **Deploy** the trained policy on a simulated robot.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Gamepad (Xbox/PS4) for Teleoperation.
- GPU for Training.

### Software Environment
```bash
pip install torch torchvision numpy h5py
sudo apt install ros-humble-teleop-twist-joy
```

### Prior Knowledge
- PyTorch (Week 1).
- Teleoperation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Paradigm Shift

*   **Classical:** Design Algorithm $\to$ Code $\to$ Robot.
*   **Robot Learning:** Demonstrate $\to$ Learn Function $\to$ Robot.
*   **Behavior Cloning (BC):** Treat the problem as Supervised Learning.
    *   Dataset: $D = \{ (o_t, a_t) \}_{t=1}^N$.
    *   Loss: $\mathcal{L}(\theta) = \sum || \pi_\theta(o_t) - a_t ||^2$.
    *   Goal: Minimize MSE between policy output and expert action.

### 🔹 Part 2: The Data Problem (Covariate Shift)

Why does standard BC fail?
1.  **Training:** Expert stays on the perfect line. Data contains only "Center of Lane" images.
2.  **Testing:** Robot drifts slightly (error $\epsilon$). Now it is "Off-Center".
3.  **Observation:** The policy *never saw* Off-Center images during training. It panics (or predicts nonsense).
4.  **Result:** Robot crashes. Error accumulates quadratically $O(T^2)$.

### 🔹 Part 3: Architectures

*   **Visuomotor Policy:**
    *   Input: RGB Image ($224 \times 224 \times 3$).
    *   Backbone: ResNet18 (Pretrained) / EfficientNet.
    *   Head: Fully Connected $\to$ Output: Linear Vel ($v$), Angular Vel ($\omega$).
*   **Recurrent Policies (LSTM/Transformer):**
    *   Input: Sequence of images $o_{t-k} \dots o_t$.
    *   Helps with "State Estimation" (inferring velocity from static images).

---

## 💻 Implementation: Behavioral Cloning Agent

We will:
1.  Record Data (rosbag).
2.  Train PyTorch Model.
3.  Deploy Inference Node.

### 🛠️ Project Structure
```text
day78_bc/
├── src/
│   ├── record_data.py
│   ├── train_bc.py
│   └── deploy_policy.py
├── data/
│   └── demos.h5 (Collected Data)
└── models/
    └── bc_policy.pth
```

### 👨‍💻 Data Recorder (`src/record_data.py`)

Listens to Images and Joy commands. Syncs them.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import h5py
import numpy as np
import cv2

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        
        self.bridge = CvBridge()
        self.current_vel = None
        self.data_obs = []
        self.data_act = []
        self.recording = False

    def cmd_cb(self, msg):
        self.current_vel = [msg.linear.x, msg.angular.z]
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.recording = True
        else:
            self.recording = False

    def img_cb(self, msg):
        if not self.recording or self.current_vel is None: return
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_img = cv2.resize(cv_img, (128, 128)) # Downsample
        
        self.data_obs.append(cv_img)
        self.data_act.append(self.current_vel)
        
    def save(self):
        with h5py.File('data/demos.h5', 'w') as f:
            f.create_dataset('observations', data=np.array(self.data_obs))
            f.create_dataset('actions', data=np.array(self.data_act))
        print(f"Saved {len(self.data_obs)} samples.")

def main():
    rclpy.init()
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save()
```

### 👨‍💻 Training (`src/train_bc.py`)

A simple CNN.

```python
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader

class RobotDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.obs = torch.from_numpy(f['observations'][:]).float().permute(0,3,1,2) / 255.0
            self.act = torch.from_numpy(f['actions'][:]).float()
            
    def __len__(self): return len(self.obs)
    def __getitem__(self, i): return self.obs[i], self.act[i]

class BCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*10*10, 100), nn.ReLU(), # Flatten size depends on input
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 2) # v, w
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# Training Loop (Standard PyTorch)
# Loss: nn.MSELoss()
# Optimizer: Adam(lr=1e-4)
```

---

## 🔬 Lab Exercise: "The Clone Teleop"

### 1. Lab Objectives
- Open Gazebo world with a simple track.
- **Drive:** Completed 5 laps manually (Expert Data). `python record_data.py`.
- **Train:** Run `train_bc.py` for 50 epochs.
- **Deploy:** Run `deploy_policy.py`.
- **Observation:** The robot should drive autonomously.
- **Challenge:** Push the robot slightly off course. Does it recover? Likely NO (Covariate Shift).

---

## 🚀 Project: "Follow the Leader"

**Goal:** Train a robot to follow a human (simulated cylinder).
1.  **Data:** Teleoperate the robot to keep the cylinder in the center of the image at a fixed distance (1m).
2.  **Dataset:** ~5000 frames.
3.  **Behavior:**
    *   If Person Left $\to$ Turn Left.
    *   If Person Right $\to$ Turn Right.
    *   If Person Close $\to$ Stop.
    *   If Person Far $\to$ Move Forward.
4.  **BC:** The network learns these rules implicitly.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Wobbles"
*   **Cause:** Training data was "jerky" (Keyboard control). Expert actions alternated 0 and 1.
*   **Fix:** Use a Joystick (Analog). Smooth the training labels (Moving Average) before training.

#### 2. "Model outputs zero"
*   **Cause:** Image normalization error. (0-255 vs 0-1).
*   **Fix:** Ensure `deploy_policy.py` divides pixel values by 255.0.

---

## ⚡ Optimization: Data Augmentation

To fight Covariate Shift *without* new data:
*   **Augmentation:** Randomly Crop, Rotate, or Add Noise to training images.
*   **Effect:** Behaves like "Synthetic Drifts".
*   **Result:** Policy becomes more robust to slight view changes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is BC called "Supervised Learning"?
    *   **A:** Because we have ground truth labels (Expert Actions) for every input.
2.  **Q:** Does BC learn the *dynamics* of the robot?
    *   **A:** No. It learns a mapping from State to Action. It doesn't know "Physics".
3.  **Q:** What if the expert makes a mistake?
    *   **A:** BC will copy the mistake. "Garbage In, Garbage Out".

### Challenge Task
> **Task:** Multi-Modal Inputs.
> 1. Modify the network to accept Image + Lidar Scan (360).
> 2. Concatenate the flattened Lidar vector with the CNN features before the FC layers.
> 3. Does it drive better? (Lidar provides precise depth info).

---

## 📚 Further Reading
- **ALVINN (1989):** First BC for Self-Driving.
- **NVIDIA End-to-End Deep Learning:** The modern standard paper.

---

**Day 78 Complete**
