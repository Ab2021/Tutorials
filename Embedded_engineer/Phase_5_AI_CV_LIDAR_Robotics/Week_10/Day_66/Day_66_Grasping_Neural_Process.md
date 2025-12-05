# Day 66: Grasping Neural Process (GNP)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> Grasping is not just geometry; it's physics.
> - **Focus:** 6-DOF Grasp Detection (GraspNet-1Billion), Neural Processes for Uncertainty, and Implicit Representations.
> - **Code:** A PyTorch inference node that takes a PointCloud and outputs 6D Grasp Candidates with confidence scores.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the difference between geometric (GPD) and data-driven (GraspNet) grasping.
2.  **Implement** a 6-DOF Grasp Sampler using a Variational Autoencoder (VAE) or similar generative model.
3.  **Deploy** a PyTorch Grasping Node in ROS 2.
4.  **Visualize** Grasp candidates in Rviz as Marker arrays.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Depth Camera (RealSense).
- GPU (CUDA 11+).

### Software Environment
```bash
pip install graspnetAPI open3d
# Model weights must be downloaded (e.g., GraspNet baseline)
```

### Prior Knowledge
- PointNet++ (Day 4/Week 1).
- SO(3) Rotation Group.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Grasping Problem

A grasp $g \in SE(3)$ is defined by Center ($t$), Orientation ($R$), Width ($w$), and Depth ($d$).
*   **Analytic Grasping:** Force Closure, Friction Cones. Requires perfect mesh.
*   **Data-Driven Grasping:** Learn $Q(g | P)$ (Quality of grasp $g$ given Point Cloud $P$).
    *   **Generative Approach:** Sample potential grasps $g \sim \pi(g|P)$.
    *   **Evaluative Approach:** Score the samples $S = Q(g)$.

### 🔹 Part 2: Grasping Neural Process (GNP)

Why "Neural Process"? Mappings from *Functions* to *Functions*.
*   **Context:** Data from observed objects.
*   **Meta-Learning:** Learning to grasp *classes* of objects rather than specific instances.
*   **Uncertainty:** GNPs provide a distribution over grasps, allowing us to select only "High Confidence" actions.

### 🔹 Part 3: Architecture (GraspNet Simplified)

1.  **Backbone:** PointNet++ extracts local features per point.
2.  **Approach Net:** Predicts an approach vector $v$ for each point.
3.  **Rotation Net:** Predicts in-plane rotation $\theta$.
4.  **Width Net:** Predicts gripper width.
5.  **Score Net:** Predicts success probability (0-1).

---

## 💻 Implementation: PyTorch Grasp Sampler

We will build a simplified Grasp Sampler based on GraspNet-1Billion baseline.

### 🛠️ Project Structure
```text
day66_gnp/
├── weights/
│   └── checkpoint.pth
├── src/
│   ├── grasp_net.py
│   └── inference_node.py
└── launch/
    └── grasp_demo.launch.py
```

### 👨‍💻 Grasp Network (`src/grasp_net.py`)

Using a PointNet backbone.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, c_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, c_dim)

    def forward(self, x):
        # x: (B, 3, N)
        B, C, N = x.shape
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0] # Global Max Pool (B, 1024, 1)
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # (B, c_dim) - Global Context
        return x

class GraspGenerator(nn.Module):
    def __init__(self, z_dim=2, c_dim=128):
        super().__init__()
        # Decoder: Latent Z + Context -> Grasp Parameters
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7) # tx, ty, tz, qw, qx, qy, qz (Pose)
        )
    
    def forward(self, context, z):
        inp = torch.cat([context, z], dim=1)
        return self.decoder(inp)

class NeuralGrasp(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.generator = GraspGenerator()
        
    def sample_grasps(self, points, num_samples=100):
        # points: (B, 3, N)
        B = points.shape[0]
        context = self.encoder(points) # (B, 128)
        
        # Expand context for samples
        context = context.repeat_interleave(num_samples, dim=0) # (B*S, 128)
        
        # Sample Latent Z
        z = torch.randn(B * num_samples, 2).to(points.device)
        
        grasps = self.generator(context, z) # (B*S, 7)
        return grasps.view(B, num_samples, 7)
```

### 👨‍💻 ROS 2 Inference Node (`src/inference_node.py`)

Subscribes to PointCloud2, Inference, Publishes Markers.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import torch
from grasp_net import NeuralGrasp

class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_net_node')
        self.sub = self.create_subscription(PointCloud2, '/camera/depth/points', self.cb, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/grasp_markers', 10)
        
        self.model = NeuralGrasp()
        self.model.eval() # Load weights...
        self.device = torch.device('cuda')
        self.model.to(self.device)

    def cb(self, msg):
        # 1. Convert ROS -> Numpy
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_np = np.array(list(gen)) # (N, 3)
        
        if len(points_np) < 100: return
        
        # Downsample
        idx = np.random.choice(len(points_np), 1024)
        points_np = points_np[idx]
        
        # 2. Tensorize
        points_t = torch.from_numpy(points_np).float().permute(1,0).unsqueeze(0).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            grasps = self.model.sample_grasps(points_t, num_samples=50)
            
        # 4. Publish Visualization
        self.publish_markers(grasps[0].cpu().numpy(), msg.header)

    def publish_markers(self, grasps, header):
        ma = MarkerArray()
        for i, g in enumerate(grasps):
            m = Marker()
            m.header = header
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = float(g[0])
            m.pose.position.y = float(g[1])
            m.pose.position.z = float(g[2])
            m.pose.orientation.w = float(g[3])
            m.pose.orientation.x = float(g[4])
            m.pose.orientation.y = float(g[5])
            m.pose.orientation.z = float(g[6])
            m.scale.x = 0.1 # Length
            m.scale.y = 0.01 # Width
            m.scale.z = 0.01
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.a = 0.8
            ma.markers.append(m)
        self.pub_markers.publish(ma)

def main():
    rclpy.init()
    rclpy.spin(GraspNode())
```

---

## 🔬 Lab Exercise: The "Grasp Heatmap"

### 1. Lab Objectives
- Point camera at a messy table.
- Color code the grasp markers by "Quality" (Score).
- **Green:** High probability (>0.8).
- **Red:** Low probability (<0.2).
- **Observation:** Notice how grasps cluster around edges and handles. Flat surfaces have no grasps (slippery).

---

## 🚀 Project: "AnyGrasp Integration"

**Goal:** Integrate the official `AnyGrasp` (or `GraspNet`) SDK.
1.  **Container:** Use a Docker container for `GraspNet` because valid environments are hard to build (Minkowski Engine dependency).
2.  **Bridge:** Use ROS 2 Bridge to send PointClouds to Docker and receive Grasp Poses.
3.  **Execute:** Pick the best grasp and send to `moveit_task_constructor` (Day 65) as the `GeneratePose` stage.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Grasp Inside Object"
*   **Symptom:** Gripper crashes into object during approach.
*   **Cause:** Grasp Center is estimated *at* the surface, but gripper fingers need to envelop it.
*   **Fix:** **Retreat**. Move the grasp frame *backwards* along the approach vector by `finger_length/2`.

#### 2. "Floating Code"
*   **Symptom:** Grasps appear in mid-air.
*   **Cause:** Noisy PointCloud (Flying pixels).
*   **Fix:** Statistical Outlier Removal (Open3D) before inference.

---

## ⚡ Optimization: 6-DOF Grasping

Traditional Top-Down grasping (4-DOF: XYZ + Yaw) is easier but limited.
*   **6-DOF:** Allows grasping from the side (Pitch/Roll).
*   **Complexity:** Search space is huge.
*   **Solution:** Constrain sampling to Surface Normals. The approach vector should generally oppose the surface normal.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Antipodal" property?
    *   **A:** Two fingers pressing against each other on opposite sides of the object to create friction stability.
2.  **Q:** VAE vs GAN for Grasping?
    *   **A:** VAE provides a smooth latent space to sample diverse grasps. GANs can be mode-collapsed (only finding one good grasp).
3.  **Q:** Why not just use Analytical Force Closure?
    *   **A:** Requires a perfect Mesh. We usually only have a partial Point Cloud (Single View).

### Challenge Task
> **Task:** Thin Objects.
> 1. Try to grasp a credit card or a sheet of paper.
> 2. Most depth cameras fail (too thin, noise).
> 3. **Fix:** Use RGB clues (Edge detection) + Plane Segmentation.

---

## 📚 Further Reading
- **GraspNet-1Billion:** The dataset and benchmark.
- **Contact-GraspNet:** 6-DOF Grasping.

---

**Day 66 Complete**
