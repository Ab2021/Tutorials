# Day 110: Radar Point Cloud Generation
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 110 Focus:**
> We have Range, Velocity, and Angle. Now we combine them. A **Radar Point Cloud** is sparse (unlike LiDAR) and noisy. We must filter the raw detections, convert coordinates, and cluster points to find "Objects".

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** the 3 FFT steps (Range, Doppler, Angle) into a pipeline.
2.  **Filter** detections based on SNR (Signal-to-Noise Ratio).
3.  **Transform** Spherical coordinates $(R, \theta, \phi)$ to Cartesian $(X, Y, Z)$.
4.  **Cluster** points using DBSCAN (Density-Based Spatial Clustering).
5.  **Visualize** the final Radar Point Cloud in 3D.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 108:** CFAR.
-   **Day 109:** AoA.
-   **Linear Algebra:** Coordinate Transforms.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `scikit-learn` (for DBSCAN).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Processing Chain

1.  **ADC Data:** Raw samples $[N_{samples} \times N_{chirps} \times N_{rx}]$.
2.  **Range FFT:** $[N_{range} \times N_{chirps} \times N_{rx}]$.
3.  **Doppler FFT:** $[N_{range} \times N_{vel} \times N_{rx}]$.
4.  **CFAR:** Detect peaks in Range-Doppler map. Output: List of $(R, V)$ indices.
5.  **Angle FFT:** For each detected peak, perform FFT across antennas. Output: $\theta$.
6.  **Point Cloud:** List of $\{R, V, \theta, SNR\}$.

### 🔹 Part 2: Coordinate Transformation

Radar gives Polar coordinates. We need Cartesian for fusion.
-   $x = R \cos(\theta_{el}) \sin(\theta_{az})$
-   $y = R \cos(\theta_{el}) \cos(\theta_{az})$
-   $z = R \sin(\theta_{el})$
-   $v_x = v_{radial} \sin(\theta_{az})$
-   $v_y = v_{radial} \cos(\theta_{az})$

### 🔹 Part 3: Clustering

Radar points are "jumpy". A car might generate 3 points: one on the bumper, one on the wheel, one ghost.
-   **DBSCAN:** Groups points that are close together ($< 1.5m$) and have similar velocity ($< 2 m/s$).
-   **Centroid:** The average position of the cluster becomes the "Object".

---

## 💻 Implementation: Radar Pipeline

**Scenario:**
-   Simulate 3 objects (Car, Pedestrian, Wall).
-   Generate noisy detections.
-   Cluster them.

### 🛠️ Setup
Create `week16_day110` and `radar_pcl.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day110
cd ~/ros2_ws/src/week16_day110
touch radar_pcl.py
```

### 👨‍💻 Code: Point Cloud Generation & Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class RadarPoint:
    def __init__(self, r, v, azimuth, snr):
        self.r = r
        self.v = v
        self.azimuth = azimuth # Degrees
        self.snr = snr
        
        # Cartesian
        theta = np.radians(self.azimuth)
        self.x = r * np.sin(theta)
        self.y = r * np.cos(theta)

def generate_mock_data():
    points = []
    
    # Object 1: Car at (10, 50), V=20
    # Generate 5 points around it
    for i in range(5):
        r = 51.0 + np.random.normal(0, 0.5)
        az = 11.3 + np.random.normal(0, 1.0) # atan(10/50) approx 11.3 deg
        v = 20.0 + np.random.normal(0, 0.2)
        points.append(RadarPoint(r, v, az, 25.0))
        
    # Object 2: Pedestrian at (-5, 20), V=-2
    for i in range(3):
        r = 20.6 + np.random.normal(0, 0.2)
        az = -14.0 + np.random.normal(0, 2.0)
        v = -2.0 + np.random.normal(0, 0.5)
        points.append(RadarPoint(r, v, az, 15.0))
        
    # Noise: Random points
    for i in range(10):
        r = np.random.uniform(0, 60)
        az = np.random.uniform(-45, 45)
        v = np.random.uniform(-30, 30)
        points.append(RadarPoint(r, v, az, 5.0)) # Low SNR
        
    return points

def filter_and_cluster(points):
    # 1. SNR Filter
    valid_points = [p for p in points if p.snr > 10.0]
    
    if not valid_points:
        return [], []
    
    # 2. Prepare for DBSCAN
    # Feature vector: [x, y, v]
    # We scale velocity to make it comparable to distance
    # e.g., 1 m/s difference ~= 1 m distance
    features = np.array([[p.x, p.y, p.v] for p in valid_points])
    
    # 3. DBSCAN
    # eps=2.0 (Search radius), min_samples=2
    clustering = DBSCAN(eps=3.0, min_samples=2).fit(features)
    labels = clustering.labels_
    
    # 4. Extract Objects (Centroids)
    objects = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1: continue # Noise
        
        cluster_indices = np.where(labels == label)[0]
        cluster_pts = features[cluster_indices]
        
        centroid = np.mean(cluster_pts, axis=0)
        objects.append(centroid) # [x, y, v]
        
    return valid_points, objects, labels

def main():
    raw_points = generate_mock_data()
    print(f"Raw Points: {len(raw_points)}")
    
    valid_pts, objects, labels = filter_and_cluster(raw_points)
    print(f"Filtered Points: {len(valid_pts)}")
    print(f"Detected Objects: {len(objects)}")
    
    # Plotting
    plt.figure(figsize=(8, 8))
    
    # Plot Raw Noise (Gray)
    noise_pts = [p for p in raw_points if p.snr <= 10.0]
    if noise_pts:
        nx = [p.x for p in noise_pts]
        ny = [p.y for p in noise_pts]
        plt.scatter(nx, ny, c='gray', marker='.', label='Noise (Low SNR)')
        
    # Plot Clustered Points
    if len(valid_pts) > 0:
        vx = [p.x for p in valid_pts]
        vy = [p.y for p in valid_pts]
        # Color by Cluster Label
        colors = ['red' if l == -1 else f'C{l}' for l in labels]
        plt.scatter(vx, vy, c=colors, marker='o', label='Valid Points')
        
    # Plot Centroids
    for obj in objects:
        plt.scatter(obj[0], obj[1], c='black', marker='x', s=200, linewidth=3, label='Object Centroid')
        plt.text(obj[0]+1, obj[1], f"V={obj[2]:.1f}m/s")
        
    plt.xlim(-30, 30)
    plt.ylim(0, 60)
    plt.xlabel("Lateral (m)")
    plt.ylabel("Longitudinal (m)")
    plt.title("Radar Clustering (DBSCAN)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Ghost Filter

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Gray dots (Noise) are ignored.
    -   Colored dots (Valid) are grouped.
    -   Black X (Centroid) marks the car and pedestrian.
3.  **Experiment:**
    -   Lower SNR threshold to 2.0.
    -   **Result:** Noise points get clustered into fake objects ("Ghosts").
    -   **Lesson:** SNR filtering is the first line of defense.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Multipath Reflections
**Symptom:** A target appears "underground" or double range.
**Cause:** Radar bounces off the road -> Car -> Radar.
**Solution:** Check Elevation Angle (if 4D Radar) or track consistency (Multipath is unstable).

#### 2. Velocity Ambiguity
**Symptom:** Fast car detected as receding.
**Cause:** Aliasing (Day 107).
**Solution:** Use "Chinese Remainder Theorem" with multiple PRFs (Pulse Repetition Frequencies) to resolve ambiguity.

---

## ⚡ Optimization & Best Practices

### 1. Weighted Centroid
Instead of simple mean:
$$ X_{obj} = \frac{\sum x_i \cdot SNR_i}{\sum SNR_i} $$
-   Points with higher SNR pull the centroid closer.
-   More accurate than simple average.

### 2. Tracking (Kalman Filter)
Clustering happens per frame.
-   **Tracking** links clusters across time.
-   If an object disappears for 1 frame (scintillation), the tracker keeps it alive.
-   We will cover this in Week 17 (Sensor Fusion).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we include Velocity in clustering?
    *   **A:** To separate a moving car from a stationary guardrail next to it. Spatially they are close, but velocities are different.
2.  **Q:** What is DBSCAN's advantage over K-Means?
    *   **A:** DBSCAN doesn't need to know the number of clusters ($k$) beforehand. It finds as many as needed.
3.  **Q:** What is "RCS" (Radar Cross Section)?
    *   **A:** The reflectivity of an object. Truck > Car > Pedestrian > Stealth Fighter.

### Challenge Task
**Task:** Radial Velocity Correction.
1.  Radar measures $v_{radial}$.
2.  Real velocity vector $\vec{v}$ might be different.
3.  If a car moves perpendicular to you, $v_{radial} = 0$.
4.  Radar is blind to cross-traffic velocity! (Unless using advanced tracking).

---

## 📚 Further Reading & References
-   [DBSCAN Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
-   [Radar Data Processing Chain](https://www.ti.com/lit/wp/spry328/spry328.pdf)

---

**Day 110 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
