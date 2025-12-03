# Day 174: Final Assessment (Exam)
## Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career

---

> **📝 Day 174 Focus:**
> It's time to prove your worth. This is the **Final Exam**. It covers Perception, Localization, Planning, Control, and System Architecture. Treat this like a real certification exam. No Google. No ChatGPT. Just you and the problem.

---

## 🎯 Exam Structure

*   **Part 1: Theory (Multiple Choice)** - 20 Questions (30 mins).
*   **Part 2: Coding (Implementation)** - 2 Problems (60 mins).
*   **Part 3: System Design (Architecture)** - 1 Problem (30 mins).

---

## 📝 Part 1: Theory (Sample Questions)

1.  **Which sensor is best for detecting the velocity of a moving object in fog?**
    *   A) Lidar
    *   B) Camera
    *   C) Radar
    *   D) Ultrasonic

2.  **In a Kalman Filter, if the Measurement Noise Covariance ($R$) is zero, what happens?**
    *   A) The filter ignores the measurement.
    *   B) The filter ignores the prediction.
    *   C) The filter becomes unstable.
    *   D) The filter works perfectly.

3.  **What is the primary advantage of RRT* over standard RRT?**
    *   A) Faster search.
    *   B) Asymptotic optimality (finds the shortest path eventually).
    *   C) Deterministic behavior.
    *   D) Easier to implement.

4.  **In MPC, what is the "Horizon"?**
    *   A) The distance the car can see.
    *   B) The number of future time steps the optimizer considers.
    *   C) The limit of the steering angle.
    *   D) The loop rate of the controller.

5.  **Which coordinate frame is fixed to the world?**
    *   A) `base_link`
    *   B) `odom`
    *   C) `map`
    *   D) `camera_optical_frame`

*(Answers: 1:C, 2:B, 3:B, 4:B, 5:C)*

---

## 💻 Part 2: Coding Challenge

### Problem A: The Particle Filter Update
**Task:** Implement the `update` step of a Particle Filter in Python.
**Input:**
*   `particles`: List of `[x, y, theta, weight]`
*   `measurement`: `[dist_to_landmark]`
*   `landmark_pos`: `[lx, ly]`
*   `sensor_std`: Standard deviation of sensor noise.

**Constraints:**
*   Use Gaussian probability density function.
*   Normalize weights at the end.

```python
import numpy as np

def update_particles(particles, measurement, landmark_pos, sensor_std):
    # Your Code Here
    weights = []
    for p in particles:
        # 1. Calculate distance from particle to landmark
        dist = np.hypot(p[0] - landmark_pos[0], p[1] - landmark_pos[1])
        
        # 2. Calculate Gaussian likelihood
        # P(z|x) = (1 / (std * sqrt(2pi))) * exp(-0.5 * ((z - dist)/std)^2)
        prob = (1.0 / (sensor_std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((measurement - dist) / sensor_std)**2)
        
        # 3. Update weight
        p[3] *= prob
        weights.append(p[3])
        
    # 4. Normalize
    total_weight = sum(weights)
    if total_weight > 0:
        for p in particles:
            p[3] /= total_weight
            
    return particles
```

### Problem B: The Stanley Controller
**Task:** Implement the Steering Law for the Stanley Controller.
**Input:**
*   `cte`: Cross Track Error (distance from path).
*   `heading_error`: $\psi_{track} - \psi_{car}$.
*   `v`: Velocity.
*   `k`: Gain parameter.

**Formula:** $\delta = \psi_e + \arctan(\frac{k \cdot e}{v})$

```python
import math

def stanley_control(cte, heading_error, v, k):
    # Your Code Here
    # Prevent division by zero
    if abs(v) < 0.1:
        v = 0.1
        
    # Calculate cross track term
    cross_track_steering = math.atan2(k * cte, v)
    
    # Total steering
    steer = heading_error + cross_track_steering
    
    # Clip to max steering (e.g., 30 deg)
    max_steer = math.radians(30)
    steer = max(min(steer, max_steer), -max_steer)
    
    return steer
```

---

## 📐 Part 3: System Design

**Scenario:**
Design a **Highway Autopilot** for a Truck.
**Requirements:**
1.  Maintain lane center.
2.  Keep safe distance (ACC).
3.  Handle "Cut-in" vehicles.
4.  Operate in Rain.

**Deliverable:**
Draw the block diagram. List sensors and algorithms.

**Solution Sketch:**
*   **Sensors:**
    *   **Radar (Long Range):** For ACC (robust to rain).
    *   **Camera (Windshield):** For Lane Detection (Deep Learning).
    *   **Lidar (Optional):** For precise Cut-in detection.
*   **Localization:**
    *   GPS/IMU + Lane Matching (Lateral accuracy).
*   **Perception:**
    *   YOLO (Cars/Trucks).
    *   LaneNet (Lane Lines).
    *   Radar Clustering (Objects).
*   **Fusion:**
    *   Kalman Filter (Track Radar Objects + Camera Bounding Boxes).
*   **Planning:**
    *   FSM (Keep Lane, Change Lane).
    *   Polynomial Trajectory (Jerk minimization).
*   **Control:**
    *   MPC (Handles heavy mass/inertia of truck).

---

## 🧠 Self-Evaluation

*   **Score < 60%:** Review Week 1-10. Focus on Python/C++ basics.
*   **Score 60-80%:** Review Week 15-20. Focus on Planning/Control math.
*   **Score > 80%:** You are ready for a Senior Engineer role.

---

**Day 174 Complete** | Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career
