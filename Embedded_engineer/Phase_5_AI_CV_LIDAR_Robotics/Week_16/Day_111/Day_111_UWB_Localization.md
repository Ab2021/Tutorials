# Day 111: Ultra-Wideband (UWB) Localization
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> GPS doesn't work indoors. UWB does.
> - **Focus:** Ranging Physics (Two-Way Ranging), Anchor vs Tag architecture, Trilateration mathematics (Intersection of circles), and fusing UWB with Odometry.
> - **Code:** A Python node that simulates receiving Ranges from 3 Anchors (with noise) and uses Non-Linear Least Squares (`scipy.optimize`) to compute the $(X, Y)$ position.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** RSSI (Signal Strength - Inaccurate) vs TWR (Time of Flight - Accurate).
2.  **Deploy** a UWB system (Decawave DWM1000 examples) conceptually.
3.  **Solve** the Trilateration problem using optimization (Sphere intersection).
4.  **Integrate** UWB Range messages into `robot_localization` (EKF) as a "GPS-like" input.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Decawave/Qorvo UWB Modules (Optional for simulation).

### Software Environment
```bash
pip install scipy matplotlib numpy
```

### Prior Knowledge
- Geometry of Circles.
- Kalman Filtering (Week 6).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How UWB Works

*   **Pulse Radio:** Sends nanosecond pulses across very wide spectrum (500MHz+).
*   **Time of Flight:** Speed of light $c$. $d = c \times t_{flight}$.
*   **Precision:** 10cm accuracy (indoors, through walls).
*   **Method:** Two-Way Ranging (TWR).
    1.  Tag sends Poll.
    2.  Anchor sends Resp.
    3.  Tag sends Final.
    4.  Timestamps allow cancelling out clock drift.

### 🔹 Part 2: Trilateration

If you know you are distance $d_1$ from $(x_1, y_1)$, you are on a circle.
If you know $d_2$ from $(x_2, y_2)$, you are at the intersection points (2 points).
If you know $d_3$, you are at a unique point.

Mathematically, minimize error:
$$ E = \sum_{i=1}^{N} (\sqrt{(x-x_i)^2 + (y-y_i)^2} - d_i)^2 $$

### 🔹 Part 3: NLOS (Non-Line of Sight)

Metal shelves reflect UWB.
*   **Result:** Path length increases (Bounce).
*   **Error:** Measured range > True range.
*   **Fix:** Robust EKF that rejects outliers (Mahalanobis distance).

---

## 💻 Implementation: UWB Solver

We will simulate a robot moving in a room with 4 anchors.

### 🛠️ Project Structure
```text
day111_uwb/
├── src/
│   ├── uwb_sim.py
│   └── trilateration_node.py
└── launch/
    └── localization.launch.py
```

### 👨‍💻 Solver Node (`src/trilateration_node.py`)

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.optimize import least_squares

class UWBSolver(Node):
    def __init__(self):
        super().__init__('uwb_solver')
        # Input: [id, range, id, range...] or formatted message
        self.sub = self.create_subscription(Float32MultiArray, '/uwb/ranges', self.cb, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/uwb/pose', 10)
        
        # Anchor Positions (Known Map)
        # x, y, z
        self.anchors = {
            0: [0.0, 0.0, 2.0],
            1: [10.0, 0.0, 2.0],
            2: [0.0, 10.0, 2.0],
            3: [10.0, 10.0, 2.0]
        }
        
        self.last_guess = [5.0, 5.0] # Center of room

    def cb(self, msg):
        # Format: [id0, range0, id1, range1, ...]
        data = np.array(msg.data)
        ids = data[0::2]
        ranges = data[1::2]
        
        if len(ids) < 3:
            return # Need 3 anchors for 2D, 4 for 3D

        # Prepared data for optimizer
        anchor_pos = []
        measured_dists = []
        
        for i, uid in enumerate(ids):
            if uid in self.anchors:
                anchor_pos.append(self.anchors[uid][:2]) # Use 2D for now
                measured_dists.append(ranges[i])

        if len(anchor_pos) < 3: return

        # Optimization Function
        def residuals(x):
            # x is [user_x, user_y]
            res = []
            for i, anc in enumerate(anchor_pos):
                pred_dist = np.linalg.norm(x - anc)
                res.append(pred_dist - measured_dists[i])
            return res

        # Solve
        res_lsq = least_squares(residuals, self.last_guess)
        est_pos = res_lsq.x
        self.last_guess = est_pos # Update seed
        
        # Publish Pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = est_pos[0]
        pose_msg.pose.pose.position.y = est_pos[1]
        
        # Covariance (Static for now, but least_squares provides Jacobian)
        pose_msg.pose.covariance[0] = 0.1 # X var
        pose_msg.pose.covariance[7] = 0.1 # Y var
        
        self.pub.publish(pose_msg)

def main():
    rclpy.init()
    rclpy.spin(UWBSolver())
```

### 👨‍💻 Simulator (`src/uwb_sim.py`)

A script to generate noisy ranges based on mouse position or path.
*(User implementation: Simple Publisher)*

---

## 🔬 Lab Exercise: "The Loose Anchor"

### 1. Lab Objectives
- **Setup:** Launch Solver.
- **Input:** Fake range data for (5,5).
    *   A1(0,0): 7.07
    *   A2(10,0): 7.07
    *   A3(0,10): 7.07
- **Result:** Output (5.0, 5.0).
- **Perturb:** Add +1m to A1 range (Simulate NLOS).
- **Result:** Output shifts to (5.3, 5.3). Triangulation gets "pulled" by the error.
- **Fix:** Use Huber Loss in `least_squares(..., loss='huber')` to ignore outliers.

---

## 🚀 Project: "Indoor GPS Fusion"

**Goal:** Robot Navigation without Lidar.
1.  **Inputs:** Wheel Odometry (Smooth, Drifts) + UWB (Noisy, No Drift).
2.  **Node:** `ekf_localization_node`.
3.  **Config:**
    *   `odom0`: /odom
    *   `pose0`: /uwb/pose (Treat UWB as absolute pose).
4.  **Result:** The UWB "snaps" the robot back to the map if Odometry drifts too far.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Z-Axis Error"
*   **Scenario:** Anchors are on ceiling (Z=3m). Robot on floor (Z=0).
*   **Math:** $d_{2D} = \sqrt{d_{measured}^2 - (Z_{anc} - Z_{rob})^2}$.
*   **Fix:** Always project 3D range to 2D plane before trilateration if doing 2D loc.

#### 2. "Flipping"
*   **Scenario:** 3 Anchors in a line (Collinear).
*   **Result:** Infinite solutions or Mirror solutions.
*   **Fix:** Ensure Anchors form a Convex Hull around the operating area.

---

## ⚡ Optimization: Graph Optimization (GTSAM)

Instead of instant Least Squares, use Factor Graphs.
*   Node: Robot Pose at $t$.
*   Factor: Range measurement.
*   Smooths trajectory over time window. Significantly more stable than frame-by-frame LS.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use WiFi RSSI?
    *   **A:** RSSI fluctuates by 10dB just by a person walking by. Accuracy is ~3-5 meters. UWB is Time-based, so signal strength drop doesn't affect distance calc (only SNR).
2.  **Q:** Anchor Calibration?
    *   **A:** You must know exact XYZ of anchors. If Anchor 1 is actually 10cm off, your robot is 10cm off.
3.  **Q:** How many anchors max?
    *   **A:** UWB is time-slotted. Typical update rate 10Hz. Adding more anchors reduces update rate per tag.

### Challenge Task
> **Task:** Follow Me.
> 1. Put UWB Tag in pocket.
> 2. Put UWB Tag on Robot.
> 3. Robot maintains distance $d=1.5m$ and Angle=0 (if using multi-antenna AoA or just crude following).

---

## 📚 Further Reading
- **Decawave Application Notes:** "APS006: Channel Effects".
- **Robot Localization:** "Integrating GPS/UWB".

---

**Day 111 Complete**
