# Day 133: Week 19 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 133 Focus:**
> We have covered the "Where am I?" problem from all angles: GPS, IMU, Visual Odometry, Lidar Odometry, Particle Filters, and Graph SLAM. Today, we build a **Full SLAM System** that maps an unknown environment and localizes within it.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Front-End (Odometry) and Back-End (Optimization).
2.  **Manage** the Keyframe Database.
3.  **Detect** Loop Closures using geometric verification.
4.  **Visualize** the growing map and trajectory in real-time.
5.  **Evaluate** the map consistency (Loop Closure error).

---

## 📚 Week 19 Review

### 1. Sensors
-   **GNSS:** Absolute position (Low freq, blocked).
-   **IMU:** High freq, drifts fast.
-   **Lidar/Camera:** Relative motion (Odometry).

### 2. Algorithms
-   **Filter-Based:** EKF, Particle Filter. Good for real-time localization.
-   **Optimization-Based:** Graph SLAM. Good for building consistent maps.
-   **Registration:** ICP, NDT. Aligning clouds.

### 3. Maps
-   **Occupancy Grid:** For navigation.
-   **Point Cloud:** For localization.
-   **HD Map (Lanelet2):** For rules/planning.

---

## 🛠️ Capstone Project: Mini-SLAM

**Goal:** Robot explores a 2D world with landmarks.
**Pipeline:**
1.  **Odometry:** Noisy motion updates.
2.  **Landmark Detection:** Range/Bearing to landmarks.
3.  **Data Association:** Identify landmarks.
4.  **Graph Optimization:** Correct trajectory when re-visiting landmarks.

### Package Structure
Create `week19_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week19_project
cd ~/ros2_ws/src/week19_project
touch mini_slam.py
```

### 👨‍💻 Code: The SLAM Engine

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- 1. Simulation Environment ---
class World:
    def __init__(self):
        # Landmarks (x, y)
        self.landmarks = np.array([
            [10, 10], [10, 50], [50, 50], [50, 10], [30, 30]
        ])
        self.robot_pose = np.array([0.0, 0.0, 0.0]) # x, y, theta

    def move(self, v, w, dt):
        # Noise
        v_n = v + np.random.normal(0, 0.1)
        w_n = w + np.random.normal(0, 0.05)
        
        theta = self.robot_pose[2]
        self.robot_pose[0] += v_n * np.cos(theta) * dt
        self.robot_pose[1] += v_n * np.sin(theta) * dt
        self.robot_pose[2] += w_n * dt
        
        return v_n, w_n # Return noisy control (Odometry)

    def sense(self):
        # Detect landmarks within range 30m
        measurements = []
        for i, lm in enumerate(self.landmarks):
            dx = lm[0] - self.robot_pose[0]
            dy = lm[1] - self.robot_pose[1]
            dist = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx) - self.robot_pose[2]
            
            if dist < 30.0:
                # Add noise
                dist_n = dist + np.random.normal(0, 0.2)
                phi_n = phi + np.random.normal(0, 0.05)
                measurements.append((i, dist_n, phi_n)) # ID, Range, Bearing
                
        return measurements

# --- 2. SLAM System ---
class GraphSLAM:
    def __init__(self):
        self.nodes = [] # Robot Poses (x, y, theta)
        self.edges = [] # Odometry constraints
        self.landmark_edges = [] # (node_idx, lm_idx, dist, phi)
        self.landmarks_est = {} # lm_idx -> [x, y] (Initial guess)

    def add_node(self, pose):
        self.nodes.append(pose)

    def add_odom_edge(self, idx1, idx2, dx, dy, dtheta):
        self.edges.append((idx1, idx2, dx, dy, dtheta))

    def add_landmark_edge(self, node_idx, lm_idx, dist, phi):
        self.landmark_edges.append((node_idx, lm_idx, dist, phi))
        
        # Initialize landmark position if new
        if lm_idx not in self.landmarks_est:
            rx, ry, rtheta = self.nodes[node_idx]
            lx = rx + dist * np.cos(rtheta + phi)
            ly = ry + dist * np.sin(rtheta + phi)
            self.landmarks_est[lm_idx] = np.array([lx, ly])

    def optimize(self):
        # State Vector: [RobotPoses (N*3), LandmarkPoses (M*2)]
        n_nodes = len(self.nodes)
        lm_ids = list(self.landmarks_est.keys())
        n_lms = len(lm_ids)
        lm_map = {id: i for i, id in enumerate(lm_ids)} # Map ID to index
        
        x0 = np.hstack([np.array(self.nodes).flatten(), 
                        np.array([self.landmarks_est[id] for id in lm_ids]).flatten()])
        
        def error_func(x):
            # Unpack
            poses = x[:n_nodes*3].reshape(-1, 3)
            lms = x[n_nodes*3:].reshape(-1, 2)
            residuals = []
            
            # 1. Odometry Constraints
            for (i, j, dx, dy, dth) in self.edges:
                pi, pj = poses[i], poses[j]
                # Expected relative motion
                # Rel in i frame: R_i^T * (p_j - p_i)
                c, s = np.cos(pi[2]), np.sin(pi[2])
                dx_est = c*(pj[0]-pi[0]) + s*(pj[1]-pi[1])
                dy_est = -s*(pj[0]-pi[0]) + c*(pj[1]-pi[1])
                dth_est = pj[2] - pi[2]
                
                residuals.extend([dx_est - dx, dy_est - dy, dth_est - dth])
                
            # 2. Landmark Constraints
            for (node_idx, lm_idx, dist, phi) in self.landmark_edges:
                if lm_idx not in lm_map: continue
                lm_i = lm_map[lm_idx]
                
                rx, ry, rth = poses[node_idx]
                lx, ly = lms[lm_i]
                
                dist_est = np.sqrt((lx-rx)**2 + (ly-ry)**2)
                phi_est = np.arctan2(ly-ry, lx-rx) - rth
                
                residuals.extend([dist_est - dist, phi_est - phi])
                
            # Anchor
            residuals.extend(poses[0] * 100)
            
            return np.array(residuals)
            
        print(f"Optimizing Graph: {n_nodes} Poses, {n_lms} Landmarks...")
        res = least_squares(error_func, x0, verbose=1)
        
        # Update State
        opt_poses = res.x[:n_nodes*3].reshape(-1, 3)
        opt_lms = res.x[n_nodes*3:].reshape(-1, 2)
        
        self.nodes = list(opt_poses)
        for i, id in enumerate(lm_ids):
            self.landmarks_est[id] = opt_lms[i]

def main():
    world = World()
    slam = GraphSLAM()
    
    # Initial Pose
    slam.add_node(np.array([0.0, 0.0, 0.0]))
    
    dt = 1.0
    steps = 60
    
    # Path: Square
    
    print("Running Simulation...")
    for t in range(steps):
        # Move
        v = 2.0
        w = 0.15 # Turn left
        v_n, w_n = world.move(v, w, dt)
        
        # Add Node (Dead Reckoning)
        prev_pose = slam.nodes[-1]
        theta = prev_pose[2]
        new_pose = prev_pose + np.array([v_n*np.cos(theta)*dt, v_n*np.sin(theta)*dt, w_n*dt])
        slam.add_node(new_pose)
        
        # Add Odom Edge
        slam.add_odom_edge(t, t+1, v_n*dt, 0, w_n*dt) # Simplified relative
        
        # Sense
        measurements = world.sense()
        for (id, dist, phi) in measurements:
            slam.add_landmark_edge(t+1, id, dist, phi)
            
    # Optimize
    slam.optimize()
    
    # Plot
    est_poses = np.array(slam.nodes)
    est_lms = np.array(list(slam.landmarks_est.values()))
    gt_lms = world.landmarks
    
    plt.figure(figsize=(10, 10))
    plt.plot(est_poses[:, 0], est_poses[:, 1], 'b-', label='Trajectory')
    plt.scatter(est_lms[:, 0], est_lms[:, 1], c='r', marker='x', s=100, label='Est Landmarks')
    plt.scatter(gt_lms[:, 0], gt_lms[:, 1], c='g', marker='o', facecolors='none', s=150, label='GT Landmarks')
    plt.title("Mini-SLAM Result")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Loop Closure Check
-   **Scenario:** The robot drives in a circle and sees Landmark 0 again.
-   **Observation:** The trajectory should close (End point near Start point).
-   **Without Optimization:** The trajectory would spiral out due to drift.
-   **With Optimization:** The red 'x' (Est Landmark) should align with the green 'o' (GT Landmark).

### 2. Landmark Consistency
-   **Check:** Are there 5 red crosses?
-   **Error:** If there are 10 red crosses, it means Data Association failed (it thought Landmark 0 was a new Landmark 5).
-   **Fix:** Ensure ID matching is correct (In this code, we cheat and use GT IDs).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Theory
1.  **Q:** What is the "Frontend" of SLAM?
    *   **A:** The part that processes sensor data to find constraints (Odometry, Feature Matching).
2.  **Q:** What is the "Backend" of SLAM?
    *   **A:** The optimizer (Graph Solver) that minimizes the error in the constraints.

### Section 2: Implementation
3.  **Q:** Why do we optimize Landmarks and Poses together?
    *   **A:** Because uncertainty in the robot pose causes uncertainty in the landmark position, and observing a landmark reduces uncertainty in the robot pose. They are coupled.
4.  **Q:** What happens if we remove the Anchor constraint?
    *   **A:** The matrix becomes singular (Infinite solutions). The whole map can shift to infinity.

---

## 🏆 Conclusion

Congratulations on completing Week 19!
-   You have mastered **Localization & SLAM**.
-   You can find the robot's position using Satellites, IMUs, Cameras, Lidars, and Maps.

**Next Week:** We teach the car to *act*. **Planning & Control**. Path Planning, Trajectory Generation, and MPC.

---

**Day 133 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
