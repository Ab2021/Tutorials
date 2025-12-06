# Day 201: Learning System Integration
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> Don't hardcode the grip. Learn it.
> - **Focus:** Integrating a Learning-Based component (Reinforcement Learning or Grasp Prediction) into the ROS System.
> - **Code:** `grasp_network.py` (Inference Wrapper) and `rl_grasp_node.py`.
> - **Concept:** Cloud Robotics / Offline RL.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why analytic grasping (geometry-based) fails on deformable fruit.
2.  **Deploy** a pre-trained RL policy (PyTorch) as a ROS 2 Service.
3.  **Bridge** the gap between ROS messages (`sensor_msgs/Image`) and Tensor inputs (`torch.Tensor`).
4.  **Implement** a "Force-Feedback" reward signal for online fine-tuning.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install torch torchvision
```

### Prior Knowledge
- Reinforcement Learning (Day 186).
- ROS Services (Day 12).

---

## 📖 Theoretical Integration

### 🔹 The Heuristic Failure
Traditional code: "Go to Centroid. Close gripper."
*   **Result:** Smashes the fruit or slips off the side.

### 🔹 The Learned Solution
**Input:** Depth Crop of the fruit.
**Output:** $(x, y, z, \theta, \text{aperture})$.
**Training:**
*   **Sim:** Try 10,000 grasps in Isaac Gym.
*   **Reward:** Binary (Lifted object successfully?).
*   **Policy:** $\pi(Depth) \to \text{GraspPose}$.

### 🔹 ROS Integration Pattern
The Deep Learning model is heavy (loads of VRAM). Don't put it in the control loop.
*   **Structure:** Make it a Service Server.
*   **Client:** Harvest FSM sends `DetectGraspRequest(image)`.
*   **Server:** DL Model runs inference, returns `DetectGraspResponse(pose)`.

---

## 💻 Implementation: The Grasp Server

### 🛠️ Project Structure
```text
agribot_learning/
├── srv/
│   └── PredictGrasp.srv
├── src/
│   ├── grasp_server.py
│   └── policy_net.py
└── launch/
    └── learning.launch.py
```

### 👨‍💻 Service Definition (`srv/PredictGrasp.srv`)

```text
sensor_msgs/Image depth_image
sensor_msgs/RegionOfInterest roi
---
geometry_msgs/Pose grasp_pose
float32 confidence
```

### 👨‍💻 Neural Policy Wrapper (`src/policy_net.py`)

A mock CNN that outputs offsets based on depth features.

```python
import torch
import torch.nn as nn
import numpy as np

class GraspCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple conv for demo
        self.conv = nn.Conv2d(1, 16, 3)
        self.fc = nn.Linear(16 * 28 * 28, 4) # x, y, z, theta

    def forward(self, depth_crop):
        # ... forward pass ...
        # Returning dummy for now
        return torch.tensor([0.0, 0.0, 0.0, 0.0]) # Offsets

class GraspPredictor:
    def __init__(self):
        self.model = GraspCNN()
        self.model.eval()
        print("Grasp Policy Loaded.")
        
    def predict(self, depth_numpy):
        # Preprocess
        tensor = torch.from_numpy(depth_numpy).float().unsqueeze(0).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            offsets = self.model(tensor)
            
        return offsets.numpy()[0]
```

### 👨‍💻 Grasp Service Node (`src/grasp_server.py`)

```python
import rclpy
from rclpy.node import Node
from agribot_learning.srv import PredictGrasp
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from policy_net import GraspPredictor
import numpy as np

class GraspService(Node):
    def __init__(self):
        super().__init__('grasp_service')
        self.srv = self.create_service(PredictGrasp, 'predict_grasp', self.handle_req)
        self.predictor = GraspPredictor()
        self.bridge = CvBridge()
        
    def handle_req(self, request, response):
        depth_img = self.bridge.imgmsg_to_cv2(request.depth_image, "passthrough")
        x, y, w, h = request.roi.x_offset, request.roi.y_offset, request.roi.width, request.roi.height
        
        # Crop
        crop = depth_img[y:y+h, x:x+w]
        
        if crop.size == 0:
            self.get_logger().warn("Empty crop!")
            return response
            
        # Inference
        offsets = self.predictor.predict(crop) # [dx, dy, dz, dtheta]
        
        # Base Pose (Center of ROI)
        # In reality, you'd deproject the ROI centers using camera info
        # Here we just output the RELATIVE adjustment
        
        response.grasp_pose.position.x = float(offsets[0])
        response.grasp_pose.position.y = float(offsets[1])
        response.grasp_pose.position.z = float(offsets[2])
        response.confidence = 0.95
        
        print(f"Predicted optimal grasp offset: {offsets}")
        return response

def main():
    rclpy.init()
    node = GraspService()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Service Call"

### 1. Lab Objectives
- **Modify:** `harvest_fsm.py` from Day 200.
- **Insert:** Inside "PICKING", before moving the arm:
    1.  Wait for service `predict_grasp`.
    2.  Send the cropped depth image (from the perception node queue? You might need to restructure data flow).
    3.  Receive `grasp_pose`.
    4.  Apply `grasp_pose` as an offset/rotation to the centroid target.
- **Run:** Verify that the robot calls the service and adjusts its hand orientation.

---

## 🚀 Project Integration

1.  **Safety:** If the RL network outputs a grasp pose that is *upside down* or causes a collision, MoveIt planning will fail (Good!).
2.  **Fallback:** If Service fails or confidence < 0.5, fall back to "Centroid Grasp" (Heuristic).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Tensor Shape Mismatch"
*   **Cause:** Crop size varies (fruit at edge vs fruit in center). CNN expects fixed 224x224.
*   **Fix:** **Resize/Pad.** `cv2.resize(crop, (224, 224))` before converting to tensor.

#### 2. "Service Hangs"
*   **Cause:** FSM is not async. `client.call()` blocks the ROS spin loop.
*   **Fix:** Use `client.call_async()` and a future. Or run the call in a separate thread. Ideally, keep FSM logic simple.

---

## ⚡ Optimization: Latency Cloaking

Inference takes 100ms.
*   **Trick:** Start calling the grasp service *while* the robot is approaching the approach_waypoint.
*   **Result:** By the time the robot arrives at 10cm standoff, the exact grasp pose is ready.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use a Service instead of a Topic?
    *   **A:** Grasping is Request-Response. We only need a grasp calculation when we are ready to pick, not a continuous stream of random grasp poses.
2.  **Q:** What is "Sim-to-Real" gap in grasping?
    *   **A:** Friction. Sim friction is perfect Coulomb. Real fruit is sticky/slippery and soft. RL policies often fail here without Domain Randomization.

### Challenge Task
> **Task:** "Data Collection".
> 1. Set up the robot to "Try Grasp".
> 2. Force Sensor > Threshold? Label = Success. Else Label = Fail.
> 3. Save (Image, Label) to disk.
> 4. Use this dataset to finetune the CNN. (Self-Supervised Learning).

---

## 📚 Further Reading
- **GraspNet:** 1 Billion Grasp Dataset.
- **DexNet:** UC Berkeley's Grasping logic.

---

**Day 201 Complete**
