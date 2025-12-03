# Day 125: Sensor Fusion with Deep Learning (BEV Fusion)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 125 Focus:**
> Kalman Filters (Week 17) fuse *objects*. But what if we fuse *features*? **BEV Fusion** is the state-of-the-art approach. It takes Camera images, "lifts" them into 3D, and fuses them with Lidar features in a shared Bird's Eye View (BEV) space.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the limitation of Late Fusion (Information Loss).
2.  **Understand** the "Lift-Splat-Shoot" (LSS) operation.
3.  **Project** 2D Image features into 3D Frustum.
4.  **Fuse** Camera and Lidar feature maps.
5.  **Visualize** the transformation from Perspective View to BEV.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 124:** PointPillars (BEV representation).
-   **Camera Geometry:** Intrinsic/Extrinsic Matrix.

### Hardware Requirements
-   **GPU:** Mandatory.

### Software Stack
-   **Python:** `torch`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Viewpoint Mismatch

-   **Camera:** Perspective View (2D). Objects get smaller with distance.
-   **Lidar:** Orthographic/3D View. Distance is preserved.
-   **Problem:** How to align them?
    -   *Old Way:* Project Lidar to Image (loses depth).
    -   *New Way:* Project Image to BEV (LSS).

### 🔹 Part 2: Lift-Splat-Shoot (LSS)

1.  **Lift:** Predict a depth distribution for each pixel $D(u, v)$.
    -   Create a point cloud of features: $F(u, v) \times D(u, v)$.
    -   This creates a "Frustum" of features in 3D.
2.  **Splat:** "Splat" (Sum/Pool) these 3D features onto the BEV grid.
3.  **Shoot:** (Not really a step, just part of the name).

### 🔹 Part 3: BEV Fusion Architecture

1.  **Camera Stream:** Image -> ResNet -> LSS -> Camera BEV Features.
2.  **Lidar Stream:** Point Cloud -> PointPillars -> Lidar BEV Features.
3.  **Fusion Module:** Concatenate (Cam_BEV, Lidar_BEV) -> Conv2d.
4.  **Head:** Detection Head (CenterPoint/TransFusion).

---

## 💻 Implementation: Perspective to BEV Projection

**Scenario:**
-   Input: A dummy feature map ($H \times W$).
-   Task: Project it to BEV assuming flat ground (Homography).

### 🛠️ Setup
Create `week18_day125` and `bev_proj.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day125
cd ~/ros2_ws/src/week18_day125
touch bev_proj.py
```

### 👨‍💻 Code: IPM (Inverse Perspective Mapping)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_ipm_matrix(width, height):
    # Define source points (Trapezoid on the road in image)
    src_pts = np.float32([
        [200, 400],  # Top Left
        [440, 400],  # Top Right
        [0, 480],    # Bottom Left
        [640, 480]   # Bottom Right
    ])
    
    # Define dest points (Rectangle in BEV)
    dst_pts = np.float32([
        [200, 0],    # Top Left
        [440, 0],    # Top Right
        [200, 480],  # Bottom Left
        [440, 480]   # Bottom Right
    ])
    
    # Calculate Homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M

def main():
    # 1. Create a Synthetic Road Image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw Road Lines (converging)
    cv2.line(img, (0, 480), (280, 240), (255, 255, 255), 2) # Left
    cv2.line(img, (640, 480), (360, 240), (255, 255, 255), 2) # Right
    
    # Draw a "Car" (Rectangle)
    cv2.rectangle(img, (300, 350), (340, 380), (0, 0, 255), -1)
    
    # 2. Apply IPM (Simple BEV Projection)
    # Note: LSS is better because it handles depth, but IPM is the geometric basis.
    
    # We need to pick points carefully for IPM.
    # Let's assume a standard camera setup.
    src = np.float32([[240, 300], [400, 300], [0, 480], [640, 480]])
    dst = np.float32([[200, 0], [440, 0], [200, 640], [440, 640]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    bev_img = cv2.warpPerspective(img, M, (640, 640))
    
    # 3. Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Front Camera View")
    # Draw src points
    for pt in src:
        plt.plot(pt[0], pt[1], 'r.')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
    plt.title("Bird's Eye View (IPM)")
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Stretched Car

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   The converging road lines become parallel in BEV. (Good).
    -   The "Car" (Red box) gets stretched vertically. (Bad).
3.  **Analysis:**
    -   IPM assumes everything lies on the flat ground plane ($Z=0$).
    -   The car has height ($Z > 0$).
    -   Pixels at height are projected incorrectly to "far away" on the ground.
    -   **Solution:** This is why **LSS (Lift-Splat-Shoot)** is needed. It predicts depth per pixel, so it knows the car pixels are close, not far.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Calibration Errors
**Symptom:** Camera and Lidar features don't align.
**Cause:** Extrinsic Matrix ($T_{cam}^{lidar}$) is slightly off.
**Solution:** Online Calibration (Self-Calibration) or very rigid mounting.

#### 2. Depth Ambiguity
**Symptom:** Ghost objects in BEV.
**Cause:** Camera depth prediction is hard. Monocular depth is ill-posed.
**Solution:** Use Stereo Cameras or rely on Lidar for depth and Camera for semantic (color).

---

## ⚡ Optimization & Best Practices

### 1. Transformers (ViT)
Instead of CNNs, use Transformers for Fusion.
-   **TransFusion:** Use Lidar queries to attend to Camera features.
-   "Is there a car here (Lidar)? Let me check the color (Camera)."

### 2. Temporal Fusion
Fuse features from $t-1, t-2, t$.
-   Helps with occlusion.
-   If a car is hidden behind a truck now, but was visible 1s ago, the temporal memory remembers it.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is BEV a good representation for Fusion?
    *   **A:** It is the native space for planning and control. Cars don't drive in "Image Space".
2.  **Q:** What is the main drawback of IPM?
    *   **A:** It assumes a flat world. Vertical objects get distorted (The "Flat Earth Assumption").
3.  **Q:** How does LSS solve the flat earth problem?
    *   **A:** By predicting a depth distribution. It places features at their estimated 3D position, not just on the ground.

### Challenge Task
**Task:** Feature Concatenation.
1.  Create two tensors: `cam_bev` ($1 \times 64 \times 100 \times 100$) and `lidar_bev` ($1 \times 64 \times 100 \times 100$).
2.  Concatenate: `fused = torch.cat([cam_bev, lidar_bev], dim=1)`.
3.  Apply a $1 \times 1$ Conv to mix them back to 64 channels.

---

## 📚 Further Reading & References
-   [Lift, Splat, Shoot Paper](https://arxiv.org/abs/2008.05711)
-   [BEVFusion Paper](https://arxiv.org/abs/2205.13542)

---

**Day 125 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
