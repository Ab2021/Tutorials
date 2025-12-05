# Day 124: Sonar Processing (Imaging Sonar)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> Light doesn't travel underwater. Sound does.
> - **Focus:** Forward Looking Sonar (FLS), Side Scan Sonar (SSS), Acoustic Shadows, and Polar-to-Cartesian transformation.
> - **Code:** A Python node that reads a raw sonar beam image (Range vs Angle), thresholds it for high-intensity returns (Objects), and projects points into 3D space (`PointCloud2`) for Octomap.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between FLS (Video-like, Low Res) and Side-Scan (Map-like, non-realtime).
2.  **Explain** the Geometry of Sonar: $R = c \times t / 2$. Resolution depends on frequency (High Freq = High Res = Low Range).
3.  **Correct** for Geometric Distortions (Slant Range).
4.  **Visualize** Sonar clouds in Rviz.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (UUV Sim with Blueview Sonar). Or Dataset (`marine_msgs`).

### Software Environment
```bash
sudo apt install ros-humble-marine-msgs
```

### Prior Knowledge
- Speed of Sound in Water ($c \approx 1500 m/s$).
- Polar Coordinates.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Acoustic Image

An "Acoustic Camera" (FLS) emits sound in a fan shape (e.g., $130^\circ$ horizontal, $20^\circ$ vertical).
*   **X-Axis:** Bearing (Angle).
*   **Y-Axis:** Range (Time).
*   **Pixel Value:** Intensity of Return.
    *   **Bright:** Hard object (Metal, Rock).
    *   **Dark:** Water or Shadow.
    *   **Shadow:** Behind a rock, there is no sound. A "hole" in the image.

### 🔹 Part 2: Geometric Transformation

To put this on a map, we convert Polar $(r, \theta)$ to Cartesian $(x, y)$.
$$ x = r \cos(\theta) $$
$$ y = r \sin(\theta) $$
(Standard math, but coordinate frames differ. Sonar usually: X is Forward, Y is Right).

### 🔹 Part 3: Noise & Multipath

*   **Multipath:** Sound bounces off surface, hits object, comes back. "Ghost object".
*   **Speckle:** Interference pattern. Looks like "Salt and Pepper" noise.
*   **Filter:** Median Filter is essential.

---

## 💻 Implementation: Sonar to PointCloud

We simulate reading a `marine_msgs/SonarImageData` (or generic Image with metadata).

### 🛠️ Project Structure
```text
day124_sonar/
├── src/
│   ├── sonar_proc.py
└── data/
    ├── sonar_sample.png (Simulated)
```

### 👨‍💻 Processor Node (`src/sonar_proc.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct

class SonarProc(Node):
    def __init__(self):
        super().__init__('sonar_proc')
        
        self.sub = self.create_subscription(Image, '/blueview/image', self.cb, 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/sonar/cloud', 10)
        self.bridge = CvBridge()
        
        # Sonar Parameters (Example: Blueview P900)
        self.min_range = 0.5
        self.max_range = 30.0
        self.fov = np.deg2rad(90) # +/- 45 deg
        self.num_beams = 512 # Image Width
        self.num_bins = 512 # Image Height (Range bins)
        
        # Threshold (Only map bright objects)
        self.intensity_thresh = 200

    def cb(self, msg):
        # 1. Get Image
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        
        # 2. Denoise
        img_blur = cv2.medianBlur(img, 5)
        
        # 3. Extract Points
        # Find pixels > Threshold
        y_indices, x_indices = np.where(img_blur > self.intensity_thresh)
        intensities = img_blur[y_indices, x_indices]
        
        # 4. Map to Geometry
        # X index -> Angle
        # 0 -> -FOV/2, Width -> +FOV/2
        angles = (x_indices / self.num_beams) * self.fov - (self.fov / 2.0)
        
        # Y index -> Range
        # Height -> Min Range (Usually Top is Near?), Check manual.
        # Let's assume Top (0) is Near, Bottom (H) is Far
        ranges = self.min_range + (y_indices / self.num_bins) * (self.max_range - self.min_range)
        
        # Polar to Cartesian (in Sonar Frame: X Forward, Y Left)
        # x = range * cos(angle)
        # y = range * sin(angle)
        
        # Wait. Sonar convention:
        # Range is radial distance.
        pts_x = ranges * np.cos(angles)
        pts_y = ranges * np.sin(angles)
        pts_z = np.zeros_like(pts_x) # 2D Sonar is flat fan
        
        # 5. Pack PointCloud2
        points = []
        for i in range(len(pts_x)):
            # struct.pack: float, float, float, intensity
            # Note: PointCloud2 standard is X, Y, Z
            points.append([pts_x[i], pts_y[i], pts_z[i], float(intensities[i])])
            
        self.publish_cloud(points, msg.header)

    def publish_cloud(self, points, header):
        # Simplified PC2 creation
        # In production, use point_cloud2.create_cloud
        import sensor_msgs_py.point_cloud2 as pc2
        
        header.frame_id = "sonar_link"
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        pc2_msg = pc2.create_cloud(header, fields, points)
        self.pub_cloud.publish(pc2_msg)

def main():
    rclpy.init()
    rclpy.spin(SonarProc())
```

---

## 🔬 Lab Exercise: "The Minehunter"

### 1. Lab Objectives
- **Scenario:** Scattered mines (Spheres) on the seabed.
- **Run:** Node.
- **Visualize:** Rviz. Set Decay Time to 10s to accumulate points.
- **Action:** Drive AUV forward.
- **Observe:** Bright arcs appearing (the mines).
- **Challenge:** Notice the "Smearing". The sonar beam is wide vertically ($20^\circ$). A mine on the floor looks the same as a mine 2m up.
- **Lesson:** 2D Sonar lacks Elevation data. You need 3D Multibeam or Profiling Sonar for Z.

---

## 🚀 Project: "Acoustic SLAM"

**Goal:** Build a map using Sonar.
1.  **Node:** `slam_toolbox`.
2.  **Input:** `/sonar/cloud` projected to `/scan` (LaserScan).
3.  **Trick:** Compress the PointCloud into a single 2D LaserScan by taking the *closest* high-intensity point in each column.
4.  **Result:** Occupancy Grid of the seabed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Slant Range Distortion"
*   **Cause:** Sonar measures range to target ($R$), not horizontal distance ($D$).
*   **Math:** $D = \sqrt{R^2 - Altitude^2}$.
*   **Fix:** You need an Altimeter (DVL) to know altitude off bottom to correct this side-scan distortion.

#### 2. "Null Returns"
*   **Cause:** Absorptive materials (Mud) or angled surfaces reflecting sound *away*.
*   **Result:** Blind spots.
*   **Fix:** Approach objects from different angles.

---

## ⚡ Optimization: GPU Processing

Processing 512x512 sonar images at 15Hz in Python is slow.
*   **CUDA:** Use CuPy or CUDA C++ to perform thresholding and coordinate transform.
*   FFT-based Beamforming happens on FPGA usually, but post-processing should be on GPU for autonomy.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is sound used instead of Light?
    *   **A:** Light attenuates in meters. Sound travels kilometers.
2.  **Q:** What is "Gain"?
    *   **A:** Amplification of the return signal. Too high = Noise saturation. Too low = Miss targets. Time-Varying Gain (TVG) increases gain for distant returns to compensate for attenuation.
3.  **Q:** Continuous Wave (CW) vs Chirp?
    *   **A:** Chirp (FMCW) gives better range resolution and noise rejection, similar to Radar.

### Challenge Task
> **Task:** Shadow Classification.
> 1. Detect the bright spot (Object).
> 2. Detect the dark streak behind it (Shadow).
> 3. Measure Shadow Length ($L$).
> 4. Estimate Object Height ($H$) using geometry ($H \approx \frac{L \times Altitude}{Range}$).

---

## 📚 Further Reading
- **Discovery of Sound in the Sea (DOSITS):** Physics guide.
- **Blue Robotics:** Ping360 Docs (Low cost scanning sonar).

---

**Day 124 Complete**
