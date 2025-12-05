# Day 79: DAgger (Dataset Aggregation)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> BC fails when the drift starts. DAgger embraces the drift.
> - **Focus:** The DAgger Loop (Train -> Rollout -> Expert Correction -> Aggregation -> Retrain).
> - **Code:** An interactive "Human-in-the-Loop" training script where the user intervenes to correct the robot.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why DAgger is a "No Regret" algorithm (Provable bounds).
2.  **Implement** an Intervention Recorder: Records only when the human overrides the policy.
3.  **Execute** the DAgger workflow iteratively to improve policy robustness.
4.  **Visualize** the decrease in user interventions over iterations.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Gamepad (Essential for corrections).

### Software Environment
```bash
# Same as Day 78
pip install torch
```

### Prior Knowledge
- Day 78 (BC shortcomings).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Distribution Mismatch

*   **Train Dist ($P_{expert}$):** Perfect driving. Center of lane.
*   **Test Dist ($P_{policy}$):** Sloppy driving. Sometimes near the wall.
*   **Problem:** The Policy doesn't know what to do near the wall, because the Expert never went there.
*   **Solution:** We Must force the robot to go near the wall, *then show it how to recover*.

### 🔹 Part 2: The DAgger Algorithm

1.  **Initialize:** $D \leftarrow$ Human Demonstrations. Train $\pi_1$ on $D$.
2.  **Loop ($i=1 \dots N$):**
    *   Run $\pi_i$ on the robot.
    *   **Expert acts** to produce correct actions $a^*$ for the states $s$ visited by $\pi_i$.
    *   (Variant: Expert just watches and takes over when $\pi_i$ fails).
    *   Get new dataset $D_i = \{(s, a^*)\}$.
    *   Aggregate: $D \leftarrow D \cup D_i$.
    *   Train $\pi_{i+1}$ on $D$.

### 🔹 Part 3: Human-Gated DAgger (HG-DAgger)

Standard DAgger requires the expert to provide actions *all the time* (even when robot is right).
*   **HG-DAgger:**
    *   Robot drives.
    *   Expert holds "Deadman Switch" or just grabs the joystick.
    *   If Joystick Active: Record (Image, Joystick Action). Override Robot.
    *   If Joystick Inactive: Robot drives. Don't record (or record (Image, Robot Action) as positives?). usually we only care about corrections.

---

## 💻 Implementation: HG-DAgger Recorder

We modify the recorder from Day 78.

### 🛠️ Project Structure
```text
day79_dagger/
├── src/
│   ├── dagger_loop.py
│   └── train_dagger.py
├── models/
│   └── policy_iter_0.pth
└── data/
    └── iter_1.h5
```

### 👨‍💻 DAgger Loop Node (`src/dagger_loop.py`)

Runs the model. Listens to Joystick.
*   If Joy == 0: Publishes Model Command.
*   If Joy != 0: Publishes Joy Command. Saves (Image, Joy) to buffer.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np
import h5py

# Import Network from Day 78
from day78_bc.network import BCNetwork 

class DAggerNode(Node):
    def __init__(self):
        super().__init__('dagger_node')
        # Params
        self.declare_parameter('model_path', '')
        self.model_path = self.get_parameter('model_path').value
        
        # Load Policy
        self.device = torch.device('cuda')
        self.policy = BCNetwork().to(self.device)
        self.policy.load_state_dict(torch.load(self.model_path))
        self.policy.eval()

        # IO
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.human_active = False
        self.human_vel = [0.0, 0.0]
        
        self.new_data_obs = []
        self.new_data_act = []

    def joy_cb(self, msg):
        # Check deadman or movement threshold
        lin = msg.axes[1]
        ang = msg.axes[0]
        if abs(lin) > 0.05 or abs(ang) > 0.05:
            self.human_active = True
            self.human_vel = [lin, ang]
        else:
            self.human_active = False

    def img_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 1. Inference
        img_t = self.preprocess(cv_img).to(self.device)
        with torch.no_grad():
            pred = self.policy(img_t) # [v, w]
            pred_vel = pred.cpu().numpy()[0]
            
        cmd = Twist()
        
        if self.human_active:
            # 2. Correction Mode
            cmd.linear.x = float(self.human_vel[0])
            cmd.angular.z = float(self.human_vel[1])
            
            # Save Data!
            self.new_data_obs.append(cv_img)
            self.new_data_act.append(self.human_vel)
            self.get_logger().info("Recording Correction!", throttle_duration_sec=1.0)
        else:
            # 3. Autonomy Mode
            cmd.linear.x = float(pred_vel[0])
            cmd.angular.z = float(pred_vel[1])
            
        self.pub_cmd.publish(cmd)
        
    def preprocess(self, cv_img):
        # Resize, Normalize, Tensorize...
        pass

    def save_dataset(self):
        # Save to HDF5...
        pass
```

### 👨‍💻 Training Loop

Just re-runs the training script from Day 78, but points to `data/*.h5` (All collected files).

---

## 🔬 Lab Exercise: "The Wandering Robot"

### 1. Lab Objectives
- **Iter 0:** Train on 1 minute of perfect driving.
- **Run:** Robot will crash into walls.
- **Iter 1:** Capture corrections (Pulling robot away from walls). Train.
- **Run:** Robot now "wiggles" away from walls but might over-correct.
- **Iter 2:** Correct the oscillations.
- **Result:** Robust policy that stays in lane better than pure BC.

---

## 🚀 Project: "Canyon Run"

**Goal:** Navigate a winding canyon.
1.  **Sim:** Gazebo Canyon world.
2.  **Challenge:** Sharp turns.
3.  **Procedure:**
    *   Start with BC.
    *   Robot likely fails at the first sharp turn.
    *   Use DAgger to show it: "When you see a wall filling the screen, Turn HARD Left".
    *   After 3 iterations, it should clear the canyon.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Human reaction time lag"
*   **Symptom:** The recorded labels are slightly delayed compared to images.
*   **Cause:** Human brain takes 200ms to react.
*   **Fix:** Shift labels backward by ~5-10 frames in the dataset before training.

#### 2. "Expert fights the robot"
*   **Symptom:** Robot tries left, Human pushes right. Training data has high variance.
*   **Fix:** In HG-DAgger, when human engages, robot output is ignored. This is clean. But ensure the transition is smooth.

---

## ⚡ Optimization: SafeDAgger

DAgger involves the robot *almost crashing*. Dangerous on real hardware.
*   **SafeDAgger:**
    *   Learn a safety policy (collision predictor).
    *   If Policy $\pi_{safe}(s)$ predicts crash, automatically switch to Human Control.
    *   Reduces burden on human attention.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does DAgger require more data than BC?
    *   **A:** Often *less* total data for *better* performance, because the data is more "Task Relevant" (covers the difficult states).
2.  **Q:** What is "No Regret"?
    *   **A:** Over time, the algorithm performs as well as the best policy in the class, regardless of the sequence of states visited.
3.  **Q:** Can we use DAgger for manipulation?
    *   **A:** Yes. "Teleoperate to correct the grasp".

### Challenge Task
> **Task:** Dataset Balancing.
> 1. DAgger collects a lot of "Recovery" data.
> 2. The dataset becomes imbalanced (90% driving straight, 10% hard turns).
> 3. **Fix:** Use Weighted Loss or Oversampling for the "Action > Threshold" samples.

---

## 📚 Further Reading
- **DAgger Paper (Ross et al., 2011):** "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning".
- **HG-DAgger:** Human Gated DAgger.

---

**Day 79 Complete**
