# Day 83: Advanced Capstone - Sensor Fusion (Lidar + Camera)
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎯 Learning Objectives
1.  **Understand** the strengths/weaknesses of Camera vs Lidar vs Radar.
2.  **Perform** Extrinsic Calibration between a Camera and a Lidar.
3.  **Project** 3D Lidar points onto a 2D Camera image.
4.  **Implement** "Early Fusion" (fusing raw data) vs "Late Fusion" (fusing object detections).
5.  **Visualize** the fused output (Colored Point Cloud) using ROS 2 / Rviz.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera (e.g., IMX219), Lidar (e.g., Velodyne/Ouster or RPLIDAR for 2D).
*   **Software:** ROS 2 (Humble), OpenCV, PCL (Point Cloud Library).
*   **Knowledge:** Homogeneous Transformation Matrices ($4 \times 4$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Fusion?
*   **Camera:** High Resolution, Color, Texture. Good for Classification ("That is a Stop Sign"). Bad at Distance.
*   **Lidar:** Perfect Distance, 3D Shape. Bad at Color/Texture ("Is that a Stop Sign or a Red Balloon?").
*   **Fusion:** Combine both to get "Colored 3D Points" or "Distance-Aware Object Detection".

### 🔹 Part 2: Calibration (The Hard Part)
*   We need the **Extrinsic Matrix** ($T_{cam\_lidar}$) that transforms a point from Lidar Frame to Camera Frame.
*   **Method:** Use a Checkerboard.
    *   Camera sees the corners (2D).
    *   Lidar sees the board plane (3D).
    *   Algorithm (PnP) solves for Rotation ($R$) and Translation ($t$).

### 🔹 Part 3: Projection Math
To project a 3D point $P_{lidar} (X, Y, Z)$ onto the image pixel $(u, v)$:
1.  **Transform:** $P_{cam} = T_{cam\_lidar} \times P_{lidar}$
2.  **Project:** $p_{image} = K \times P_{cam}$ (where $K$ is Camera Intrinsic Matrix).
3.  **Normalize:** $u = p_x / p_z$, $v = p_y / p_z$.

---

## 💻 Implementation Examples

### Example 1: Projection Function (Python)

```python
import numpy as np
import cv2

def project_lidar_to_camera(points_3d, K, T_ext):
    """
    points_3d: Nx3 array of Lidar points
    K: 3x3 Intrinsic Matrix
    T_ext: 4x4 Extrinsic Matrix (Lidar -> Camera)
    """
    # 1. Convert to Homogeneous (Nx4)
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # 2. Transform to Camera Frame
    points_cam = (T_ext @ points_hom.T).T # Nx4
    
    # 3. Filter points behind camera (Z < 0)
    points_cam = points_cam[points_cam[:, 2] > 0]
    
    # 4. Project to Image Plane (Homogeneous 2D)
    points_img = (K @ points_cam[:, :3].T).T # Nx3
    
    # 5. Normalize (u, v)
    u = points_img[:, 0] / points_img[:, 2]
    v = points_img[:, 1] / points_img[:, 2]
    
    return np.stack([u, v], axis=1), points_cam[:, 2] # Return UV and Depth
```

### Example 2: Coloring Lidar Points

Overlaying depth on the image.

```python
def draw_depth_overlay(image, uv_points, depth_values):
    h, w = image.shape[:2]
    
    for (u, v), z in zip(uv_points, depth_values):
        if 0 <= u < w and 0 <= v < h:
            # Color map: Near = Red, Far = Blue
            color = get_jet_color(z, min_dist=1.0, max_dist=50.0)
            cv2.circle(image, (int(u), int(v)), 2, color, -1)
            
    return image
```

### Example 3: Late Fusion (Object Level)

Merging YOLO Bounding Box with Lidar Cluster.

```python
def fuse_objects(camera_objects, lidar_clusters):
    fused_objects = []
    
    for cam_obj in camera_objects:
        # Find Lidar points inside the Bounding Box
        points_in_box = find_points_in_roi(lidar_points, cam_obj.bbox)
        
        if len(points_in_box) > 0:
            # Median depth of points
            dist = np.median(points_in_box[:, 2])
            fused_objects.append({
                "class": cam_obj.class_name,
                "distance": dist
            })
            
    return fused_objects
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Manual Calibration

**Objective:** Estimate $T_{ext}$ roughly.

**Steps:**
1.  Place Lidar and Camera side-by-side.
2.  Measure offset with a ruler ($t_x, t_y, t_z$).
3.  Assume Rotation is Identity (if aligned perfectly).
4.  Run projection.
5.  **Tweak:** Adjust $t_x$ manually until the Lidar points align with the object in the image.

### Lab 2: The "Paint" Test

**Objective:** Verify alignment.

**Steps:**
1.  Stand 5m away.
2.  Hold a large cardboard box.
3.  Visualize the projection.
4.  **Goal:** The Lidar points corresponding to the box should fall exactly *on* the box pixels in the image. If they float in the air next to it, calibration is off.

### Lab 3: Frustum Culling

**Objective:** Optimize processing.

**Steps:**
1.  Lidar is $360^\circ$. Camera is $60^\circ$.
2.  Discard all Lidar points outside the Camera's Field of View (Frustum).
3.  **Math:** Check if projected $(u, v)$ is within $[0, W]$ and $[0, H]$.

---

## 🐛 Debugging Fusion Issues

### Debug 1: Time Synchronization

**Symptom:** Alignment is good when static, but fails when moving.

**Cause:**
*   Camera and Lidar capture at different times.
*   **Fix:** Use PTP (Precision Time Protocol) to sync clocks. Interpolate Lidar motion (Ego-Motion Compensation) to the Camera timestamp.

### Debug 2: Parallax Error

**Symptom:** Objects close to camera are misaligned.

**Cause:**
*   Physical distance between sensors (Baseline).
*   **Fix:** Accurate translation vector ($t$) in calibration. Occlusion handling (Lidar sees behind the object that Camera sees? No, usually Lidar is blocked too).

---

## ⚡ Performance Optimization

### Optimization 1: Project on GPU

*   Projecting 100,000 Lidar points in Python is slow.
*   Use CUDA (or Vertex Shader) to project points.

### Optimization 2: Depth Map Generation

*   Instead of sparse points, create a dense Depth Map image.
*   Upsample the sparse Lidar points (Bilateral Filter) to fill the gaps.
*   Allows using standard CNNs (RGB-D) for processing.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Early Fusion" vs "Late Fusion"?** (Early = Raw Data / Pixels + Points. Late = Bounding Boxes + Clusters).
2.  **Why is Lidar usually lower resolution than Camera?** (Scanning mechanics / Physics limits).
3.  **What is the "PnP Problem"?** (Perspective-n-Point. Finding pose from 3D-2D correspondences).
4.  **Why do we need to filter $Z > 0$?** (Points behind the camera mathematically project to the screen but are invalid).

### Practical Challenges

1.  **Implement "Distance Label":** Run YOLO. For every "Car" detected, calculate the distance using the Lidar points inside the box and draw "Car: 12.5m" on the screen.
2.  **Visualize in Rviz:** Publish the Camera Image and the Colored Point Cloud to ROS 2 Rviz.

---

## 📚 Further Reading & Resources

### Papers
*   **"PointPainting" (Vora et al.):** A classic Early Fusion technique.

### Tools
*   **Autoware:** Open-source autonomous driving stack (uses Fusion).

---

## 🎓 Summary

Today we covered:
- ✅ **Fusion:** Best of both worlds.
- ✅ **Calibration:** Aligning coordinate systems.
- ✅ **Projection:** 3D to 2D math.
- ✅ **Visualization:** Coloring points.
- ✅ **Sync:** The importance of time.

**Next:** Day 84 - Advanced Capstone: Cloud Integration (AWS IoT).

---

**Day 83 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
