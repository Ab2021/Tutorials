# Day 28: Week 4 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 28 Focus:**
> We have explored the entire Localization stack: from satellite signals in space (GPS) to point clouds on the ground (Lidar), and the maps that bind them together. Today, we unify these concepts into a robust **Multi-Modal Localization System** that can survive tunnels, urban canyons, and featureless highways.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Synthesize** Lidar, Visual, and GNSS localization techniques into a coherent architecture.
2.  **Architect** a hierarchical localization system: Global (GPS) -> Local (Map) -> Odometry (IMU/Wheel).
3.  **Implement** a Particle Filter (MCL) that fuses Odom and Lidar to localize in a 2D grid map.
4.  **Simulate** sensor failures (GPS loss) and observe how the system recovers (or drifts).
5.  **Evaluate** your mastery of Week 4 concepts through a comprehensive assessment.

---

## 📚 Week 4 Review

### 1. Lidar SLAM (Geometry)
-   **ICP:** Matches point clouds. Good for local odometry.
-   **LOAM:** Separates high-freq odometry (Edge/Plane matching) from low-freq mapping.
-   **NDT:** Matches scans to a grid of Gaussians. Robust and smooth.

### 2. Visual SLAM (Appearance)
-   **ORB-SLAM:** Sparse feature matching.
-   **Scale Ambiguity:** Monocular cameras don't know size. Need Stereo or IMU.
-   **Loop Closure:** Bag of Words (BoW) detects revisited places to fix drift.

### 3. Sensor Fusion (Optimization)
-   **Factor Graphs:** Represent SLAM as a graph of constraints.
-   **GTSAM:** Optimizes the graph to find the most likely trajectory.
-   **VIO/LIO:** Tightly couples IMU with Vision/Lidar for robustness.

### 4. Global Localization (Absolute)
-   **GNSS/RTK:** Provides absolute lat/lon. Subject to multipath/blocking.
-   **HD Maps (Lanelet2):** Provides semantic context (Lanes, Signs) to aid localization.

---

## 🛠️ Capstone Project: The "Unstoppable" Localizer

**Goal:** Build a Localization System that fuses:
1.  **Odometry:** Noisy velocity commands ($v, \omega$).
2.  **GPS:** Noisy absolute position ($x, y$). Available intermittently.
3.  **Landmarks (Map):** Range/Bearing to known landmarks (simulating Lidar/Vision map matching).

**Algorithm:** Augmented Monte Carlo Localization (AMCL).
-   Why AMCL? It handles non-linearities and global uncertainty better than EKF.

### Architecture
-   **Motion Model:** Probabilistic velocity motion model.
-   **Sensor Model:** Beam model (for landmarks) + Gaussian model (for GPS).
-   **Resampling:** Low Variance Sampling.

### Package Structure
Create `week4_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week4_project
cd ~/ros2_ws/src/week4_project
touch amcl_fusion.py
```

### 👨‍💻 Code: AMCL with GPS Fusion

```python
import numpy as np
import matplotlib.pyplot as plt
import math

class AMCL:
    def __init__(self, num_particles, map_landmarks):
        self.N = num_particles
        self.landmarks = map_landmarks # [[x, y], ...]
        
        # Initialize particles randomly (Global Localization)
        self.particles = np.random.uniform(0, 100, (self.N, 3)) # x, y, theta
        self.particles[:, 2] = np.random.uniform(0, 2*np.pi, self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self, u, dt):
        # u = [v, w]
        v, w = u
        
        # Add noise to control
        v_noisy = v + np.random.normal(0, 0.5, self.N)
        w_noisy = w + np.random.normal(0, 0.1, self.N)
        
        theta = self.particles[:, 2]
        
        # Motion Model (Bicycle/Unicycle)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = v_noisy / w_noisy
            self.particles[:, 0] += term1 * (np.sin(theta + w_noisy*dt) - np.sin(theta))
            self.particles[:, 1] += term1 * (np.cos(theta) - np.cos(theta + w_noisy*dt))
            self.particles[:, 2] += w_noisy * dt
            
        # Handle straight line case (w ~ 0)
        mask = np.abs(w_noisy) < 1e-5
        self.particles[mask, 0] += v_noisy[mask] * dt * np.cos(theta[mask])
        self.particles[mask, 1] += v_noisy[mask] * dt * np.sin(theta[mask])
        self.particles[mask, 2] += 0
        
        # Normalize angle
        self.particles[:, 2] %= 2 * np.pi

    def update_landmarks(self, z_landmarks):
        # z_landmarks = [[id, dist], ...]
        if not z_landmarks: return
        
        sigma_dist = 2.0
        
        for i in range(self.N):
            prob = 1.0
            px, py = self.particles[i, 0], self.particles[i, 1]
            
            for lm_id, meas_dist in z_landmarks:
                lm_x, lm_y = self.landmarks[lm_id]
                pred_dist = np.sqrt((px - lm_x)**2 + (py - lm_y)**2)
                
                # Gaussian Likelihood
                prob *= np.exp(- (meas_dist - pred_dist)**2 / (2 * sigma_dist**2))
            
            self.weights[i] *= prob
            
        self.normalize_weights()

    def update_gps(self, z_gps):
        # z_gps = [x, y]
        # GPS is absolute, so we weigh particles based on distance to GPS fix
        if z_gps is None: return
        
        sigma_gps = 5.0 # GPS is noisy
        
        dist_sq = (self.particles[:, 0] - z_gps[0])**2 + (self.particles[:, 1] - z_gps[1])**2
        prob = np.exp(- dist_sq / (2 * sigma_gps**2))
        
        self.weights *= prob
        self.normalize_weights()

    def normalize_weights(self):
        sum_w = np.sum(self.weights)
        if sum_w < 1e-9:
            # Particle deprivation! Re-initialize some particles around GPS if available
            # For now, just uniform reset
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= sum_w

    def resample(self):
        # Low Variance Sampling
        new_particles = np.empty_like(self.particles)
        beta = 0.0
        mw = np.max(self.weights)
        index = int(np.random.random() * self.N)
        
        for i in range(self.N):
            beta += np.random.random() * 2.0 * mw
            while beta > self.weights[index]:
                beta -= self.weights[index]
                index = (index + 1) % self.N
            new_particles[i] = self.particles[index]
            
        self.particles = new_particles

    def estimate(self):
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        return mean

def run_simulation():
    # Map: 4 Landmarks in a 100x100 world
    landmarks = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])
    
    amcl = AMCL(num_particles=500, map_landmarks=landmarks)
    
    # Ground Truth Robot
    robot_pose = np.array([10.0, 10.0, 0.0])
    
    # Trajectory
    path_gt = []
    path_est = []
    
    plt.figure(figsize=(8, 8))
    
    for t in range(100):
        # 1. Move Robot
        u = [2.0, 0.1] # v=2, w=0.1 (Circle)
        dt = 0.5
        
        robot_pose[0] += u[0]*dt * np.cos(robot_pose[2])
        robot_pose[1] += u[0]*dt * np.sin(robot_pose[2])
        robot_pose[2] += u[1]*dt
        path_gt.append(robot_pose[:2].copy())
        
        # 2. Simulate Sensors
        # Landmarks (Lidar)
        z_lm = []
        for i, lm in enumerate(landmarks):
            dist = np.sqrt((robot_pose[0]-lm[0])**2 + (robot_pose[1]-lm[1])**2)
            if dist < 30.0: # Max range
                z_lm.append([i, dist + np.random.normal(0, 0.5)])
                
        # GPS (Available every 10 steps)
        z_gps = None
        if t % 10 == 0:
            z_gps = robot_pose[:2] + np.random.normal(0, 3.0)
            
        # 3. AMCL Cycle
        amcl.predict(u, dt)
        amcl.update_landmarks(z_lm)
        amcl.update_gps(z_gps) # Fuse GPS
        amcl.resample()
        
        est = amcl.estimate()
        path_est.append(est)
        
        # 4. Visualization
        if t % 5 == 0:
            plt.clf()
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            
            # Particles
            plt.scatter(amcl.particles[:, 0], amcl.particles[:, 1], s=1, c='r', alpha=0.3)
            
            # Landmarks
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=100, marker='*', c='k')
            
            # Paths
            pgt = np.array(path_gt)
            pest = np.array(path_est)
            plt.plot(pgt[:, 0], pgt[:, 1], 'b-', label='Ground Truth')
            plt.plot(pest[:, 0], pest[:, 1], 'g--', label='AMCL Estimate')
            
            if z_gps is not None:
                plt.scatter(z_gps[0], z_gps[1], s=50, c='m', marker='x', label='GPS Fix')
                
            plt.legend()
            plt.title(f"Step {t} | GPS: {'Yes' if z_gps is not None else 'No'}")
            plt.pause(0.1)
            
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🧪 Verification & Testing

### 1. GPS Recovery
**Scenario:**
1.  Robot gets "Kidnapped" (Teleport it in simulation but not in AMCL).
2.  AMCL particles will be stuck at the old location.
3.  **Trigger:** When GPS signal arrives (Step 10, 20...), the `update_gps` function will down-weight the stuck particles and up-weight any random particles near the GPS fix.
4.  **Result:** The cloud should jump to the new location.

### 2. Tunnel Simulation
**Scenario:**
1.  Disable GPS (`z_gps = None` always).
2.  Disable Landmarks (Empty `z_lm`).
3.  **Result:** The particles should spread out (increase uncertainty) purely based on Odometry noise. This is "Dead Reckoning".

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Lidar SLAM
1.  **Q:** In LOAM, why do we separate Edge points and Planar points?
    *   **A:** To constrain different degrees of freedom. Edges constrain translation perpendicular to the edge. Planes constrain translation perpendicular to the plane.
2.  **Q:** What is the advantage of NDT over ICP?
    *   **A:** NDT uses a continuous probability function (differentiable), allowing Newton's method optimization, and doesn't require expensive nearest neighbor search.

### Section 2: Visual SLAM
3.  **Q:** How does Loop Closure correct drift?
    *   **A:** It adds a constraint between the current pose and an old pose. The optimizer (Bundle Adjustment / Pose Graph) then distributes the accumulated error across the entire loop.
4.  **Q:** What is the "Covisibility Graph"?
    *   **A:** A graph where nodes are Keyframes and edges represent shared Map Points. It defines the local neighborhood for optimization.

### Section 3: GPS & Maps
5.  **Q:** Why can't we use raw GPS for autonomous lane keeping?
    *   **A:** Standard GPS has 3-5m error. A lane is 3.5m wide. You would be in the next lane. RTK or Map-matching is required.
6.  **Q:** What is the difference between the Geometric and Semantic layers of an HD Map?
    *   **A:** Geometric = Point cloud/Mesh (Shape). Semantic = Lane lines, Speed limits, Traffic rules (Meaning).

---

## 🏆 Conclusion

Congratulations on completing Week 4!
-   You have mastered the art of knowing "Where am I?".
-   You can build maps and localize in them using Lidar, Cameras, and GPS.
-   You have built a robust fusion engine.

**Next Week:** We move to **Planning & Control**. Now that we know where we are and what is around us, we need to decide **Where to go** and **How to steer**.

---

**Day 28 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
