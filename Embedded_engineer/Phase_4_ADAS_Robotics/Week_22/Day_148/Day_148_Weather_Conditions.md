# Day 148: Weather Conditions (Rain, Snow, Fog)
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 148 Focus:**
> Autonomous cars work great in sunny California. But what about a snowstorm in Norway? **Weather** is a major edge case. Rain creates reflections, snow hides lane markings, and fog blinds Lidars. Today, we learn to handle these adverse conditions.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Analyze** the impact of rain, snow, and fog on Camera, Lidar, and Radar.
2.  **Implement** Data Augmentation to simulate weather in training data.
3.  **Apply** De-raining / De-hazing algorithms (Image Processing).
4.  **Filter** Lidar noise caused by snowflakes/raindrops.
5.  **Evaluate** model performance under degraded conditions.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Physics:** Light scattering (Mie/Rayleigh).
-   **Deep Learning:** CNNs, Data Augmentation.

### Hardware Requirements
-   **None:** Simulation based (CARLA Weather).

### Software Stack
-   **Python:** `opencv-python`, `albumentations` (for augmentation).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sensor Degradation

-   **Camera:**
    -   **Rain:** Drops on lens (blur), streaks in air (occlusion), wet road (reflections).
    -   **Fog:** Low contrast, depth perception loss.
    -   **Snow:** Whiteout (overexposure), covered signs.
-   **Lidar:**
    -   **Rain/Snow:** Laser hits drops/flakes $\to$ False positives (Ghost obstacles).
    -   **Wet Road:** Specular reflection $\to$ No ground return (Black hole).
-   **Radar:**
    -   **Robust:** Radar is mostly immune to weather (Radio waves penetrate fog/rain). This is why Radar is critical for redundancy.

### 🔹 Part 2: Mitigation Strategies

1.  **Hardware:** Wipers, Heaters, Hydrophobic coatings.
2.  **Software (Pre-processing):**
    -   **Radius Outlier Removal (ROR):** Filter isolated Lidar points (snowflakes).
    -   **De-hazing:** Dark Channel Prior algorithm.
3.  **Software (Learning):**
    -   Train on weather data (Real or Sim).
    -   **Domain Adaptation:** GANs to convert "Sunny" images to "Rainy".

---

## 💻 Implementation: Weather Augmentation

**Scenario:**
-   We have a dataset of sunny images.
-   We want to train a lane detector that works in rain.
-   Task: Generate synthetic rain and fog.

### 🛠️ Setup
Create `week22_day148` and `weather_aug.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day148
cd ~/ros2_ws/src/week22_day148
pip install albumentations
touch weather_aug.py
```

### 👨‍💻 Code: Synthetic Weather

```python
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def add_rain(image):
    transform = A.Compose([
        A.RandomRain(
            brightness_coefficient=0.9, 
            drop_width=1, 
            blur_value=5, 
            p=1.0
        ),
    ])
    augmented = transform(image=image)['image']
    return augmented

def add_fog(image):
    transform = A.Compose([
        A.RandomFog(
            fog_coef_lower=0.3, 
            fog_coef_upper=0.8, 
            alpha_coef=0.1, 
            p=1.0
        ),
    ])
    augmented = transform(image=image)['image']
    return augmented

def add_snow(image):
    transform = A.Compose([
        A.RandomSnow(
            brightness_coeff=2.5, 
            snow_point_lower=0.3, 
            snow_point_upper=0.5, 
            p=1.0
        ),
    ])
    augmented = transform(image=image)['image']
    return augmented

def lidar_snow_filter(points, radius=1.0, min_neighbors=5):
    # Simple Radius Outlier Removal (ROR)
    # Points: Nx3 numpy array
    # In production, use PCL or Open3D for speed
    
    # Mock implementation for concept
    # Real implementation requires KD-Tree
    return points # Placeholder

def main():
    # 1. Create a Dummy Image (Road)
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (100, 100, 100) # Grey road
    # Draw Lane Lines
    cv2.line(img, (200, 400), (280, 200), (255, 255, 255), 5)
    cv2.line(img, (400, 400), (320, 200), (255, 255, 255), 5)
    # Draw Sky
    cv2.rectangle(img, (0, 0), (600, 200), (255, 200, 100), -1)
    
    # 2. Apply Augmentations
    rainy = add_rain(img)
    foggy = add_fog(img)
    snowy = add_snow(img)
    
    # 3. Visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original (Sunny)")
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(rainy, cv2.COLOR_BGR2RGB))
    plt.title("Synthetic Rain")
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(foggy, cv2.COLOR_BGR2RGB))
    plt.title("Synthetic Fog")
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(snowy, cv2.COLOR_BGR2RGB))
    plt.title("Synthetic Snow")
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Lidar Noise

### Lab Objectives
1.  **Simulate Lidar Snow:**
    -   Generate random points in 3D space (Snowflakes).
    -   Add them to a clean Point Cloud.
2.  **Filter:**
    -   Use `open3d.geometry.PointCloud.remove_radius_outlier`.
    -   **Observation:** Snowflakes are sparse (few neighbors). Real objects are dense. The filter removes the snow but keeps the car.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Over-Augmentation
**Symptom:** The image is unrecognizable (pure white/grey).
**Cause:** Fog coefficient too high.
**Solution:** Tune parameters. The model still needs to see features to learn.

#### 2. False Negatives
**Symptom:** Object detector misses cars in rain.
**Cause:** Rain streaks look like edges, confusing the CNN.
**Solution:** Train with rain augmentation. Use Radar fusion (Radar sees through rain).

---

## ⚡ Optimization & Best Practices

### 1. Sensor Fusion Weights
Adaptive Fusion.
-   Sunny: Trust Camera (0.8) + Lidar (0.2).
-   Foggy: Trust Radar (0.6) + Lidar (0.3) + Camera (0.1).
-   Detect weather condition first, then adjust Kalman Filter measurement noise matrices ($R$).

### 2. Thermal Cameras
-   **LWIR (Long-Wave Infrared):** Sees heat.
-   Unaffected by light/shadow/fog.
-   Excellent for pedestrian detection at night/fog.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does Lidar fail in heavy rain?
    *   **A:** Water absorbs IR light (905nm/1550nm). Signal attenuates. Also, drops scatter the beam, causing "ghost" points close to the sensor.
2.  **Q:** What is "Data Augmentation"?
    *   **A:** Artificially increasing the diversity of training data by applying transformations (crop, rotate, weather) to prevent overfitting.
3.  **Q:** Which sensor is most robust to weather?
    *   **A:** Radar (Millimeter wave).

### Challenge Task
**Task:** De-Hazing.
1.  Implement the "Dark Channel Prior" algorithm (He et al.).
2.  It assumes that in non-sky patches, at least one color channel has very low intensity. Fog adds whiteness, violating this.
3.  Use this to estimate transmission map and recover the clear image.

---

## 📚 Further Reading & References
-   [Albumentations Documentation](https://albumentations.ai/)
-   [Waymo Open Dataset (Weather)](https://waymo.com/open/)

---

**Day 148 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
