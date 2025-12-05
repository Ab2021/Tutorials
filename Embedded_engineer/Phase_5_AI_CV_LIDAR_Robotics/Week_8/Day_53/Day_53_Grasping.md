# Day 53: Grasping (GraspNet)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> We can move to a point. But can we pick up a coffee mug?
> - **Focus:** 6-DOF Grasp Pose Detection, Point Clouds, and GraspNet (Deep Learning).
> - **Code:** Inferring valid gripper poses from a depth image of a cluttered bin.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Force Closure and Form Closure (Physics of grasping).
2.  **Deploy** a pre-trained GraspNet model to generate thousands of candidate grasps.
3.  **Filter** grasps based on collision (Gripper vs Box) and Kinematics (Reachability).
4.  **Execute** the "Best" grasp: Approach $\to$ Close $\to$ Lift.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- RGB-D Camera (Simulated).

### Software Environment
```bash
pip install open3d numpy scipy
# Install GraspNet-baseline (Requires PyTorch/CUDA)
```

### Prior Knowledge
- Point Clouds (Day 8).
- Collision Checking.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Analytical vs Data-Driven

1.  **Analytical (Old School):**
    *   Model object as Mesh. Calculate Friction Cone.
    *   **Force Closure:** Can I resist any external force?
    *   *Cons:* Requires perfect 3D model. Slow.
2.  **Data-Driven (Modern):**
    *   Input: Point Cloud. Output: $(x, y, z, r, p, y, w)$ for N grasps.
    *   Train on massive datasets (GraspNet-1B, Acronym).
    *   *Pros:* Works on unknown objects. Handles partial view.

### 🔹 Part 2: The 6-DOF Grasp

A grasp is defined by:
*   **Center:** $(x,y,z)$ between fingers.
*   **Approach Vector:** The direction the gripper approaches (Red Arrow).
*   **Width:** How wide to open fingers.
*   **Score:** Probability of success ($0.0 \to 1.0$).

### 🔹 Part 3: GraspNet Architecture

PointNet++ Backbone $\to$ Feature Extraction.
*   **Seed Sampling:** Pick 1000 points on the object surface.
*   **Cylinder Search:** Check cylinder volume around seed for friction/collision.
*   **Evaluation:** Regress the grasp score.

---

## 💻 Implementation: Contact-GraspNet Inference

Using `contact_graspnet` (TensorFlow/PyTorch) to predict grasps on a point cloud.

### 🛠️ Project Structure
```text
day53_grasping/
├── data/
│   ├── mug.pcd (Point cloud)
├── src/
│   ├── grasp_inference.py
│   └── visualize_grasps.py
└── run_demo.py
```

### 👨‍💻 Grasp Inference (`src/grasp_inference.py`)

Pseudo-code adapting standard GraspNet API.

```python
import numpy as np
import open3d as o3d
# import contact_graspnet.model as grasp_model

class GraspDetector:
    def __init__(self):
        print("Loading GraspNet Weights...")
        # self.model = grasp_model.load("checkpoint.pth")
        
    def predict(self, pcd):
        # 1. Preprocess Point Cloud
        points = np.asarray(pcd.points)
        
        # Downsample for speed
        if len(points) > 20000:
            pcd = pcd.voxel_down_sample(0.005)
            points = np.asarray(pcd.points)
            
        print(f"Processing {len(points)} points...")
        
        # 2. Inference (Simulated Output)
        # Returns: (N, 4, 4) Homogeneous Matrices, (N,) Scores
        grasps, scores = self.simulated_inference(points)
        
        # 3. Filter Low Scores
        mask = scores > 0.8
        return grasps[mask], scores[mask]
        
    def simulated_inference(self, points):
        # Heuristic: Find surface normals, create grasp aligned with normal
        # This simulates what the NN does
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        
        normals = np.asarray(pcd.normals)
        N = len(points)
        
        grasps = []
        scores = []
        
        for i in range(0, N, 100): # Sample every 100th point
            center = points[i]
            approach = -normals[i] # Approach against the surface
            binormal = np.cross(approach, [0,0,1])
            if np.linalg.norm(binormal) < 0.1:
                binormal = np.cross(approach, [0,1,0])
            binormal /= np.linalg.norm(binormal)
            axis = np.cross(approach, binormal)
            
            # Construct Rotation Matrix [approach, binormal, axis]
            R = np.column_stack((binormal, axis, approach))
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center
            
            grasps.append(T)
            scores.append(np.random.rand()) # Random confidence
            
        return np.array(grasps), np.array(scores)
```

### 👨‍💻 Visualization (`src/visualize_grasps.py`)

Draw gripper meshes at pose.

```python
import open3d as o3d
import numpy as np

def draw_gripper(T, color=[0, 1, 0]):
    # Create a U-shape mesh representing gripper
    mesh = o3d.geometry.TriangleMesh.create_box(0.1, 0.02, 0.02)
    mesh.paint_uniform_color(color)
    mesh.transform(T)
    return mesh

def show_grasps(pcd, grasps):
    geometries = [pcd]
    
    # Draw top 5 grasps
    for i in range(min(5, len(grasps))):
        gripper = draw_gripper(grasps[i])
        geometries.append(gripper)
        
    o3d.visualization.draw_geometries(geometries)
```

### 👨‍💻 Main (`run_demo.py`)

```python
import open3d as o3d
from src.grasp_inference import GraspDetector
from src.visualize_grasps import show_grasps

# Load Point Cloud
pcd = o3d.io.read_point_cloud("data/mug.pcd")

detector = GraspDetector()
grasps, scores = detector.predict(pcd)

print(f"Found {len(grasps)} valid grasps.")
show_grasps(pcd, grasps)
```

---

## 🔬 Lab Exercise: The "Unreachable" Grasp

### 1. Lab Objectives
- The NN predicts a perfect grasp on the *back* of the mug.
- **Problem:** The robot is in *front*. It cannot reach the back without hitting the mug itself.
- **Task:** Filter grasps using IK.
    - `for grasp in grasps:`
        - `sol = compute_ik(grasp)`
        - `if sol: valid_grasps.append(grasp)`
- **Result:** Only front-facing grasps survive.

---

## 🚀 Project: "Smart Bin Picking"

**Goal:** Clean a table scattered with blocks.
1.  **Scan:** Move head to capture PC.
2.  **Detect:** Run GraspNet. Get top score grasp.
3.  **Plan:** MoveIt plans path to "Pre-Grasp" pose (10cm away along approach vector).
4.  **Execute:** Move Pre-Grasp $\to$ Move Grasp $\to$ Close Gripper $\to$ Lift.
5.  **Loop:** Repeat until no grasps found.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Gripper hits object"
*   **Symptom:** Robot moves to grasp, but fingers smash into object before closing.
*   **Cause:** Depth sensor noise. The object is closer than measured. Or Grasp Center is too deep inside object.
*   **Fix:** Back off 2cm. Or use **Force Control** (stop when touch).

#### 2. "Object Slips"
*   **Symptom:** Lift up, object falls.
*   **Cause:** Low friction. Or grasp width too wide.
*   **Fix:** Increase grip force. Add rubber pads. Choose grasps closer to Center of Mass (CoM).

---

## ⚡ Optimization: 4-DOF Grasping

For Top-Down picking (Suction cup or parallel jaw on SCARA), we don't need 6-DOF.
*   $(x, y, z, \theta_{yaw})$.
*   Much faster inference (2D CNN on Depth Map).
*   **GG-CNN** (Generative Grasping CNN).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Antipodal" grasping?
    *   **A:** Two fingers pressing against each other on opposite sides of the object surface (Stable).
2.  **Q:** Why Point Clouds? Why not Images?
    *   **A:** Grasping is 3D geometry. We need to know collision depths. RGB loses scale.
3.  **Q:** Pre-Grasp Pose?
    *   **A:** A waypoint aligned with the grasp approach vector. Allows the planner to align the gripper *before* the dangerous Final Approach.

### Challenge Task
> **Task:** Suction Grasping.
> 1. Instead of generic Force Closure.
> 2. Find "Flat Surfaces".
> 3. Calculate Surface Normal.
> 4. Score = Flatness * Area.

---

## 📚 Further Reading
- **GraspNet-1B:** Benchmark for General Object Grasping.
- **Dex-Net:** Dataset generation for grasping.

---

**Day 53 Complete**
