# Day 68: 3D Multi-Object Tracking (AB3DMOT)
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 68 Focus:**
> 2D tracking is good for cameras, but self-driving cars live in a 3D world. We need to know if the car is 10m or 50m away. **AB3DMOT** (A Baseline for 3D Multi-Object Tracking) takes the principles of SORT and applies them to 3D LiDAR detections.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the 3D State Vector: $[x, y, z, \theta, l, w, h, v_x, v_y, v_z]$.
2.  **Implement** 3D IoU (Intersection over Union) for oriented bounding boxes.
3.  **Design** a 3D Kalman Filter with a Constant Velocity model.
4.  **Code** a 3D Tracker that associates LiDAR detections over time.
5.  **Visualize** 3D tracks in a simulated 3D space.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 66:** SORT.
-   **Geometry:** 3D Rotations (Yaw), Polygon Intersection.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `shapely` (for polygon intersection).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The 3D State Space

In 2D, we tracked $[u, v, s, r]$.
In 3D, we track the physical properties of the object:
$$ \mathbf{x} = [x, y, z, \theta, l, w, h, v_x, v_y, v_z] $$
-   $x, y, z$: Center position (meters).
-   $\theta$: Yaw angle (radians).
-   $l, w, h$: Dimensions (Length, Width, Height).
-   $v_x, v_y, v_z$: Linear velocities.
-   Note: We usually assume dimensions are constant ($\dot{l}=\dot{w}=\dot{h}=0$).

### 🔹 Part 2: 3D IoU

Calculating overlap of two rotated 3D boxes is hard.
**Approximation:**
1.  **BEV Overlap:** Project to 2D (Birds Eye View). Calculate intersection of two rotated rectangles (using `shapely` or Separating Axis Theorem).
2.  **Height Overlap:** Calculate intersection of Z-intervals $[z_{min}, z_{max}]$.
3.  **3D IoU:** $\frac{\text{Area}_{BEV} \times \text{Height}_{overlap}}{\text{Vol}_A + \text{Vol}_B - \text{Intersection}}$.

### 🔹 Part 3: Orientation Correction

The Kalman Filter assumes linear state evolution.
Angle $\theta$ is circular ($-\pi$ to $\pi$).
-   If $\theta$ jumps from $3.1$ to $-3.1$, the difference is $-6.2$ (Huge error!).
-   **Fix:** Always normalize angle difference to $[-\pi, \pi]$.

---

## 💻 Implementation: AB3DMOT

**Scenario:**
-   **Input:** 3D Detections `[[x, y, z, theta, l, w, h], ...]`.
-   **Output:** 3D Tracks with IDs.

### 🛠️ Setup
Create `week10_day68` and `ab3dmot.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day68
cd ~/ros2_ws/src/week10_day68
touch ab3dmot.py
```

### 👨‍💻 Code: 3D Tracker

```python
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

# --- 3D IoU Helper ---
def poly_iou(poly1, poly2):
    # poly: List of (x,y) tuples
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if not p1.intersects(p2):
        return 0.0
    try:
        inter_area = p1.intersection(p2).area
        union_area = p1.area + p2.area - inter_area
        return inter_area / union_area
    except:
        return 0.0

def get_corners_bev(x, y, l, w, theta):
    # Returns 4 corners of the rectangle in BEV
    c = np.cos(theta)
    s = np.sin(theta)
    # Rotation matrix
    R = np.array([[c, -s], [s, c]])
    # Local corners (centered at 0)
    corners_local = np.array([
        [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
    ]).T
    # Rotate and Translate
    corners_global = R @ corners_local + np.array([[x], [y]])
    return corners_global.T # 4x2

def iou3d(box1, box2):
    # box: [x, y, z, theta, l, w, h]
    
    # 1. BEV Overlap
    poly1 = get_corners_bev(box1[0], box1[1], box1[4], box1[5], box1[3])
    poly2 = get_corners_bev(box2[0], box2[1], box2[4], box2[5], box2[3])
    bev_iou = poly_iou(poly1, poly2)
    
    if bev_iou == 0: return 0.0
    
    # 2. Height Overlap
    z1_min = box1[2] - box1[6]/2
    z1_max = box1[2] + box1[6]/2
    z2_min = box2[2] - box2[6]/2
    z2_max = box2[2] + box2[6]/2
    
    inter_h = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    union_h = max(z1_max, z2_max) - min(z1_min, z2_min)
    
    # 3. Approx 3D IoU (Assuming volume is proportional to BEV area * height)
    # This is a simplification. True 3D IoU is Vol_Inter / Vol_Union
    # Vol_Inter = Area_Inter * Inter_H
    # Vol_Union = Vol1 + Vol2 - Vol_Inter
    
    vol1 = box1[4] * box1[5] * box1[6]
    vol2 = box2[4] * box2[5] * box2[6]
    vol_inter = (bev_iou * (box1[4]*box1[5] + box2[4]*box2[5]) / 2) * inter_h # Rough approx of area
    # Better: Use shapely area directly
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    area_inter = p1.intersection(p2).area
    vol_inter = area_inter * inter_h
    
    return vol_inter / (vol1 + vol2 - vol_inter)

# --- 3D Kalman Filter ---
class KalmanBox3D:
    count = 0
    def __init__(self, bbox):
        # bbox: [x, y, z, theta, l, w, h]
        self.id = KalmanBox3D.count
        KalmanBox3D.count += 1
        self.time_since_update = 0
        self.hits = 1
        
        # State: [x, y, z, theta, l, w, h, vx, vy, vz]
        self.x = np.zeros((10, 1))
        self.x[:7, 0] = bbox
        
        self.P = np.eye(10) * 1.0
        self.P[7:, 7:] *= 100.0 # High velocity uncertainty
        
    def predict(self):
        # Constant Velocity
        F = np.eye(10)
        F[0, 7] = 0.1 # dt
        F[1, 8] = 0.1
        F[2, 9] = 0.1
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(10) * 0.01
        self.time_since_update += 1
        
        return self.x[:7, 0]

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        
        z = np.array(bbox).reshape(7, 1)
        H = np.eye(7, 10)
        R = np.eye(7) * 0.1
        
        # Angle Wrap
        diff = z[3, 0] - self.x[3, 0]
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        z[3, 0] = self.x[3, 0] + diff
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(10) - K @ H) @ self.P

class AB3DMOT:
    def __init__(self):
        self.trackers = []
        
    def update(self, dets):
        # 1. Predict
        trks = []
        for t in self.trackers:
            trks.append(t.predict())
            
        # 2. Association
        if len(trks) == 0:
            for d in dets:
                self.trackers.append(KalmanBox3D(d))
            return
            
        iou_matrix = np.zeros((len(dets), len(trks)))
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou3d(det, trk)
                
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > 0.1: # Threshold
                matched_indices.append((r, c))
                
        # 3. Update
        matched_det_idx = [m[0] for m in matched_indices]
        matched_trk_idx = [m[1] for m in matched_indices]
        
        for d_idx, t_idx in matched_indices:
            self.trackers[t_idx].update(dets[d_idx])
            
        # 4. Create
        for d_idx in range(len(dets)):
            if d_idx not in matched_det_idx:
                self.trackers.append(KalmanBox3D(dets[d_idx]))
                
        # 5. Delete
        self.trackers = [t for t in self.trackers if t.time_since_update < 5]

def main():
    tracker = AB3DMOT()
    
    # Simulation: Car driving in circle
    # Radius 10m, Speed 0.1 rad/step
    
    plt.figure(figsize=(8, 8))
    
    for i in range(50):
        theta = i * 0.1
        x = 10 * np.cos(theta)
        y = 10 * np.sin(theta)
        z = 0
        yaw = theta + np.pi/2 # Tangent
        
        # Detection [x, y, z, theta, l, w, h]
        det = [x, y, z, yaw, 4.0, 2.0, 1.5]
        
        tracker.update([det])
        
        # Viz (BEV)
        plt.cla()
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        
        # Draw Det
        poly = get_corners_bev(x, y, 4.0, 2.0, yaw)
        plt.fill(poly[:, 0], poly[:, 1], 'g', alpha=0.3, label='Det')
        
        # Draw Track
        for t in tracker.trackers:
            state = t.x[:7, 0]
            poly_t = get_corners_bev(state[0], state[1], state[4], state[5], state[3])
            plt.plot(poly_t[:, 0], poly_t[:, 1], 'b', linewidth=2, label=f'ID:{t.id}')
            # Close loop
            plt.plot([poly_t[-1, 0], poly_t[0, 0]], [poly_t[-1, 1], poly_t[0, 1]], 'b', linewidth=2)
            
        plt.title(f"Step {i}")
        plt.pause(0.1)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Spin

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The Blue Box (Track) follows the Green Box (Detection) in a circle.
3.  **Experiment:**
    -   Add noise to the Yaw angle in the detection: `yaw = theta + np.pi/2 + np.random.randn()*0.5`.
    -   **Result:** The track should smooth out the jittery orientation.
    -   **Challenge:** If the noise is too large ($>\pi/2$), the IoU might drop to zero (boxes perpendicular). The track might be lost.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Angle Discontinuity
**Symptom:** Track spins wildly (360 degrees) when crossing $\pm \pi$.
**Cause:** Kalman Filter treats angle as a linear number. $\pi \to -\pi$ looks like a huge jump.
**Solution:** Normalize the innovation (residual) $y = z - Hx$ to $[-\pi, \pi]$.

#### 2. Dimension Collapse
**Symptom:** Box size shrinks to zero.
**Cause:** Unobservable dimensions. If you only see the front of the car, you can't estimate length.
**Solution:** Use priors (Average Car Size) or accumulate points over time.

---

## ⚡ Optimization & Best Practices

### 1. CenterPoint
Modern 3D detectors (CenterPoint) output velocity directly ($v_x, v_y$).
-   Use this measured velocity in the Kalman Update!
-   Makes tracking much more robust than relying on position difference.

### 2. Quaternion vs Euler
For full 3D rotation (drones), use Quaternions.
For cars (flat ground), Yaw is usually enough.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is 3D IoU harder than 2D?
    *   **A:** Because of rotation. Axis-aligned boxes are easy (min/max). Rotated boxes require polygon clipping.
2.  **Q:** What happens if we don't normalize the angle?
    *   **A:** The filter will try to "catch up" to the jump by spinning the box rapidly, causing huge errors for several frames.
3.  **Q:** Can we use Camera for 3D Tracking?
    *   **A:** Yes, if we have Monocular 3D Detection (Day 50). But depth uncertainty makes it harder than LiDAR.

### Challenge Task
**Task:** Velocity Update.
1.  Modify the `update` method to accept measured velocity $[v_x, v_y]$.
2.  Update the $H$ matrix to observe velocity states.
3.  Observe faster convergence when the car accelerates.

---

## 📚 Further Reading & References
-   [AB3DMOT Paper (Weng et al.)](https://arxiv.org/abs/2008.08063)
-   [Shapely Documentation](https://shapely.readthedocs.io/en/stable/)

---

**Day 68 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
