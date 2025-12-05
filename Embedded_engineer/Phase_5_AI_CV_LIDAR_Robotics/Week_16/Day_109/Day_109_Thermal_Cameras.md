# Day 109: Thermal Cameras (LWIR)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> Seeing heat is a superpower.
> - **Focus:** Long-Wave Infrared (LWIR) physics, Uncooled Microbolometers, Radiometry (Temperature measurement) vs Imaging, and FFC (Flat Field Correction).
> - **Code:** A "Person Detector" that works in Absolute Darkness using simple thresholding on 16-bit thermal data, with False Color visualization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Near-Infrared (NIR - used in Night Vision goggles) and Long-Wave Infrared (LWIR - Thermal).
2.  **Handle** 16-bit `mono16` images where values represent raw sensor counts or Kelvin.
3.  **Apply** AGC (Auto Gain Control) normalization to convert 16-bit to 8-bit RGB for human viewing.
4.  **Perform** FFC (Flat Field Correction) trigger management for shutter-based cameras.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- FLIR Lepton / Boson / Teledyne DALSA. (Or Dataset).

### Software Environment
```bash
sudo apt install ros-humble-flir-camera-driver
pip install opencv-python
```

### Prior Knowledge
- Blackbody Radiation.
- Bit Depth (8 vs 16).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics of Heat

Everything emits radiation.
*   **LWIR (8-14 $\mu m$):** Humans peak here (37°C / 310K).
*   **Sensor:** Microbolometer. A grid of resistors that change resistance when heated by incoming photons.
*   **Uncooled:** No cyrogenic cooling needed (unlike MWIR/SWIR used in missiles). Small, cheap, but noisy.

### 🔹 Part 2: The Data Format

Thermal cameras are *Monochrome*.
*   **Format:** `mono16` (unsigned 16-bit integer).
*   **Radiometric:** Value $X$ maps to Temperature $T$.
    *   $T_{kelvin} = \text{GenericCount} \times 0.04$ (Example factor).
*   **Non-Radiometric:** Value $X$ is just relative intensity.

### 🔹 Part 3: Flat Field Correction (FFC)

*   **Problem:** The sensor heats up unevenly. Center might look hotter than corners (Vignetting/Drift).
*   **Solution:** A mechanical shutter closes for 0.5s. Sensor sees uniform temperature (the shutter). It recalibrates offsets.
*   **Robot Issue:** During FFC, the image freezes! Avoid FFC during critical maneuvers.

---

## 💻 Implementation: Human Detector (Night)

We will process a `mono16` stream.
1.  **Normalize:** 16-bit $\to$ 8-bit (for Viz).
2.  **Threshold:** 16-bit analysis (for Detection).

### 🛠️ Project Structure
```text
day109_thermal/
├── src/
│   ├── thermal_proc.py
└── launch/
    ├── flir.launch.py
```

### 👨‍💻 Thermal Processor (`src/thermal_proc.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ThermalProc(Node):
    def __init__(self):
        super().__init__('thermal_proc')
        self.sub = self.create_subscription(Image, '/flir/image_raw', self.cb, 10)
        self.pub_viz = self.create_publisher(Image, '/flir/false_color', 10)
        self.bridge = CvBridge()
        
        # Radiometric Parameters (Example)
        # raw value of 8000 = 20C, 8100 = 30C
        self.counts_per_C = 10.0 
        self.offset_C = -273.15 # if raw is Kelvin * 100

    def cb(self, msg):
        # 1. Convert ROS -> OpenCV
        # Ensure we keep 16-bit depth
        frame_16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        
        # 2. Visualization (AGC - Auto Gain Control)
        # Find min/max in CURRENT frame to stretch contrast
        min_val, max_val, _, _ = cv2.minMaxLoc(frame_16)
        
        # Avoid divide by zero
        if max_val == min_val: max_val += 1
        
        # Normalize to 0-255
        frame_8 = cv2.convertScaleAbs(frame_16, alpha=(255.0/(max_val-min_val)), beta=-(min_val*255.0/(max_val-min_val)))
        
        # Apply Colormap (IRONBOW is standard for Thermal)
        heatmap = cv2.applyColorMap(frame_8, cv2.COLORMAP_JET)
        
        # 3. Detection (Logic on 16-bit raw data)
        # Let's say Human Body > 30C. 
        # If camera is Radiometric (Raw=Kelvin*100): 30C = 303.15K = 30315
        threshold_val = 30315 # Example
        
        # Create Mask
        _, mask = cv2.threshold(frame_16, threshold_val, 65535, cv2.THRESH_BINARY)
        mask_8 = mask.astype(np.uint8)
        
        # Contours
        contours, _ = cv2.findContours(mask_8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            if cv2.contourArea(c) > 500: # Filter noise
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(heatmap, (x,y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(heatmap, "HUMAN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Publish
        self.pub_viz.publish(self.bridge.cv2_to_imgmsg(heatmap, "bgr8"))

def main():
    rclpy.init()
    rclpy.spin(ThermalProc())
```

---

## 🔬 Lab Exercise: "The Coffee Cup"

### 1. Lab Objectives
- **Setup:** Place a hot cup of water and a cold metal bottle.
- **Run:** The node.
- **Observation:**
    *   Cup glows Red/White.
    *   Metal bottle usually looks Black (Cold) OR Reflective (Mirrors heat from room).
- **Challenge:** Thermal reflection. Metal surfaces act like mirrors in LWIR. You might see your own heat reflection in a cold metal sheet. This confuses robots.

---

## 🚀 Project: "Multi-Spectral Fusion"

**Goal:** Combine RGB + Thermal.
1.  **Hardware:** Stereo Rig (RGB Left, Thermal Right).
2.  **Calibration:** Use a "Heated Checkerboard" (Print checkerboard on metal, heat it with lamp).
3.  **Process:**
    *   Rectify RGB.
    *   Rectify Thermal.
    *   Overlay Thermal on RGB (Alpha blending).
4.  **Result:** You see the visual texture (RGB) AND the heat signature. Perfect for search and rescue.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Image is Gray and Flat"
*   **Cause:** Low contrast range (e.g., looking at a wall where variation is 0.1C).
*   **Fix:** Histogram Equalization (CLAHE) brings out the noise, but shows details.

#### 2. "Solar Damage"
*   **Warning:** NEVER point an uncooled thermal camera at the SUN. The lens focuses heat and burns the Bolometer permanently. You will have a dead pixel trail forever.

---

## ⚡ Optimization: 14-bit over 8-bit Video

Sending `mono16` over WiFi is heavy (2 bytes/pixel).
Sending converted `bgr8` is heavier (3 bytes/pixel).
*   **Trick:** Encode 16-bit into PNG (Lossless compression works well on thermal gradients). `image_transport` with `compressedDepth` plugin can handle 16-bit depth effectively.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can thermal see through glass?
    *   **A:** **NO.** Glass is opaque to LWIR. It acts like a mirror. You cannot see a person inside a car through the windshield. (This is a major limitation vs RGB).
2.  **Q:** What is Emissivity?
    *   **A:** Efficiency of radiating heat. Skin is high (0.98). Polished Aluminum is low (0.05). Low emissivity objects usually reflect surroundings.
3.  **Q:** Why does the image freeze every few minutes?
    *   **A:** FFC (NUC - Non-Uniformity Correction). The shutter is closed.

### Challenge Task
> **Task:** Fire Detection.
> 1. Light a candle.
> 2. Detect the saturation point.
> 3. Warning: Fires are HOT. Some cameras saturate at 150C. Industrial ones go to 1000C. Check your camera specs.

---

## 📚 Further Reading
- **FLIR Science:** "Infrared Explained".
- **ROS 2 Driver:** `spinnaker_camera_driver` now supports FLIR thermal (sometimes).

---

**Day 109 Complete**
