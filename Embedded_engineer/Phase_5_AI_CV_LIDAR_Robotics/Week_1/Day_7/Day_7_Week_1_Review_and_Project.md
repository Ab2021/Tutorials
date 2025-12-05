# Day 7: Week 1 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> The first week was heavy on theory (Transformers, Point Clouds, GNNs). Today is about **Synthesis**.
> - **Goal:** Build a cohesive "Perception Output" ROS 2 message that aggregates 2D, 3D, and Graph data.
> - **Code:** A complete System Integration project.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** concepts from CNNs, FPNs, PointNets, and GNNs into a unified perception pipeline.
2.  **Architect** a multi-threaded ROS 2 system that handles varying sensor rates ($30\text{Hz}$ Camera vs $10\text{Hz}$ LiDAR).
3.  **Deploy** optimized inference engines (TensorRT/ONNX) for real-time performance.
4.  **Evaluate** the end-to-end latency of the perception stack.

---

## 📚 Week 1 Review: The "Perception Stack"

We have built individual components. Now we assemble the Voltron.

| Day | Component | Function | Input | Output |
|-----|-----------|----------|-------|--------|
| **1-3** | **2D Vision** | Detect & Track | RGB Image | 2D Bounding Boxes + IDs |
| **4** | **3D Vision** | Shape Analysis | Point Cloud | 3D Centroids + Classes |
| **5** | **Relational** | Scene Graph | Objects | Relations ("Cup on Table") |
| **6** | **Learning** | Adaptation | Unlabeled Logs | Robust Featuers |

### Key Takeaways
1.  **Latency:** A 99% accurate model running at 1 FPS is useless for a drone. Balance is key (MobileNet, Tiny-BiFPN).
2.  **Representation:** Converting unstructured data (pixels, points) into structured data (graphs, objects) is the core job of the perception engineer.
3.  **Data:** Self-Supervised Learning allows us to scale beyond human labelling.

---

## 🚀 Weekly Capstone: The "UniPercept" Node

**Scenario:** You are building the perception system for a **Service Robot** in a hospital.
**Requirements:**
1.  Detect people and equipment (Wheelchairs, IV Stands) in 2D.
2.  Fuse with LiDAR to get precise 3D location.
3.  Build a local Scene Graph ("PersonA is near BedB").
4.  Publish a `WorldState` custom message.

### 🛠️ Project Structure
```text
week1_capstone/
├── perception_pkg/
│   ├── config/
│   │   └── perception_params.yaml
│   ├── msg/
│   │   ├── Object3D.msg
│   │   └── SceneGraph.msg
│   ├── src/
│   │   ├── detector_2d.py     # YOLOx / MobileNet-FPN
│   │   ├── segmentor_3d.py    # PointNet++
│   │   ├── fusion_engine.py   # 2D-3D Fusion
│   │   └── graph_builder.py   # GNN
│   └── perception_node.py     # Main ROS 2 Node
├── launch/
│   └── perception_system.launch.py
└── test/
    └── latency_test.py
```

### 👨‍💻 Code Implementation: The Fusion Engine

We will implement the critical `fusion_engine.py` which takes asynchronous 2D and 3D data and fuses them.

```python
import numpy as np
import message_filters
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class FusionEngine:
    def __init__(self, camera_matrix, extrinsic_matrix):
        self.K = camera_matrix # 3x3 Intrinsic
        self.T = extrinsic_matrix # 4x4 Extrinsic (LiDAR -> Camera)
        
    def project_lidar_to_image(self, points_3d):
        """
        Project 3D points [N, 3] to 2D pixels [N, 2]
        """
        # 1. Transform LiDAR Frame -> Camera Frame
        ones = np.ones((points_3d.shape[0], 1))
        points_homo = np.hstack([points_3d, ones])
        points_cam = (self.T @ points_homo.T).T # [N, 4]
        
        # 2. Filter Z > 0 (In front of camera)
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]
        
        # 3. Project to Image Plane
        uv_homo = (self.K @ points_cam[:, :3].T).T
        uv = uv_homo[:, :2] / uv_homo[:, 2:3]
        
        return uv, points_cam[:, :3], mask

    def fuse(self, detections_2d, point_cloud):
        """
        Associate 2D boxes with 3D clusters.
        Simple Heuristic: Frustum Culling.
        """
        fused_objects = []
        
        # Convert ROS PC2 to Numpy
        pc_np = np.array(list(pc2.read_points(point_cloud, field_names=("x", "y", "z"))))
        
        # Project all points
        uv_pixels, xyz_cam, valid_mask = self.project_lidar_to_image(pc_np)
        
        for det in detections_2d.detections:
            bbox = det.bbox # center_x, center_y, size_x, size_y
            x1 = bbox.center.x - bbox.size_x / 2
            x2 = bbox.center.x + bbox.size_x / 2
            y1 = bbox.center.y - bbox.size_y / 2
            y2 = bbox.center.y + bbox.size_y / 2
            
            # Find points inside this 2D box
            in_box_mask = (uv_pixels[:, 0] >= x1) & (uv_pixels[:, 0] <= x2) & \
                          (uv_pixels[:, 1] >= y1) & (uv_pixels[:, 1] <= y2)
            
            relevant_points = xyz_cam[in_box_mask]
            
            if len(relevant_points) > 5:
                # Estimate 3D Centroid (Median is robust to outliers)
                centroid = np.median(relevant_points, axis=0)
                fused_objects.append({
                    "class_id": det.results[0].id,
                    "score": det.results[0].score,
                    "position": centroid
                })
                
        return fused_objects
```

### 👨‍💻 Code Implementation: Main Node (`perception_node.py`)

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
# Custom imports
from fusion_engine import FusionEngine
from detector_2d import Detector2D

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Components
        self.detector = Detector2D("mobilenet_fpn.pth")
        self.fusion = FusionEngine(K_matrix, T_matrix)
        
        # Buffers
        self.latest_depth = None
        
        # Subscribers (Approximate Time Sync is alternative, but here we do manual for clarity)
        self.create_subscription(PointCloud2, '/lidar/points', self.lidar_cb, 10)
        self.create_subscription(Image, '/camera/rgb', self.camera_cb, 10)
        
    def lidar_cb(self, msg):
        self.latest_depth = msg
        
    def camera_cb(self, msg):
        if self.latest_depth is None: return
        
        # 1. Run 2D Detection (Inference)
        detections = self.detector.detect(msg)
        
        # 2. Fuse with latest LiDAR
        objects_3d = self.fusion.fuse(detections, self.latest_depth)
        
        # 3. Publish
        self.get_logger().info(f"Published {len(objects_3d)} fused objects")
        # pub.publish(objects_3d)
```

---

## ⚡ Latency Benchmarking

To ensure "Real-Time" performance, we measure the breakdown.

| Stage | Target (ms) | Description |
|-------|-------------|-------------|
| **Pre-process** | 5 ms | Resize, Normalize |
| **2D Inference** | 15 ms | MobileNet / TensorRT |
| **Point Cloud IO** | 10 ms | Read PC2, Filter |
| **Fusion / Projection** | 5 ms | Matrix Math (NumPy/GPU) |
| **Graph Update** | 2 ms | Graph Adjacency Update |
| **Total** | **37 ms** | **~27 FPS** |

*Challenge:* If latency > 100ms, use `nodelets` (C++) or separate inference processes to avoid Python GIL.

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Which architecture is best for detecting small distant objects? (FPN)
    *   Which architecture is best for 3D unordered points? (PointNet++)

2.  **Fusion:**
    *   What is the difference between Early Fusion and Late Fusion?
        *   *Early:* Fuse raw LiDAR points (projected as channels) with RGB pixels -> CNN.
        *   *Late:* Detect in 2D, Detect in 3D, then match outputs (what we did above).

3.  **Deployment:**
    *   Why do we filter LiDAR points with $x > 0$ (or $z > 0$ depending on frame)?
        *   To remove points behind the camera that cannot be projected.

---

## ⏭️ Look Ahead: Week 2
Now that we can **SEE** (Perception), we need to **KNOW** where we are.
**Week 2: Advanced SLAM & State Estimation.**
*   LiDAR Odometry (LOAM).
*   Visual-Inertial Odometry (VINS).
*   Handling dynamic environments.

---

**Week 1 Complete**
