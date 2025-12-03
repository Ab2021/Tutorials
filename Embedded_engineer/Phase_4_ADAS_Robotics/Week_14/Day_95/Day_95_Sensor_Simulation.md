# Day 95: Sensor Simulation (LiDAR, Camera, Radar)
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 95 Focus:**
> A simulator is only as good as its sensors. If the simulated LiDAR is "perfect" (no noise, infinite range), your algorithms will fail in the real world. Today, we configure realistic sensors in CARLA, adding noise, dropouts, and distortion.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Configure** Camera intrinsics (FOV, Focal Length) and distortion.
2.  **Simulate** a rotating LiDAR (Channels, Range, Dropoff).
3.  **Model** Radar detections (Velocity, Azimuth, Elevation).
4.  **Inject** Noise: Gaussian noise for Radar, lens flare for Camera, atmospheric attenuation for LiDAR.
5.  **Visualize** multi-sensor data in a unified 3D view.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 16:** Camera Models.
-   **Day 46:** LiDAR Physics.
-   **Day 53:** Radar Physics.

### Hardware Requirements
-   **GPU:** Essential for Camera/LiDAR rendering.

### Software Stack
-   **CARLA:** Python API.
-   **Open3D:** For point cloud viz.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Camera Simulation

CARLA uses UE4's rendering engine.
-   **Parameters:**
    -   `image_size_x`, `image_size_y`.
    -   `fov`: Field of View (Horizontal).
    -   `fstop`, `shutter_speed`, `iso`: Exposure control.
    -   `lens_circle_multiplier`: Simulates Bokeh/Lens Flare.
-   **Output:** Raw RGB, Depth (Logarithmic), Semantic Segmentation (Ground Truth).

### 🔹 Part 2: LiDAR Simulation

Ray-casting based.
-   **Parameters:**
    -   `channels`: 16, 32, 64, 128 (Vertical resolution).
    -   `range`: Max distance (e.g., 100m).
    -   `points_per_second`: Horizontal resolution.
    -   `rotation_frequency`: Hz (e.g., 10Hz).
    -   `noise_stddev`: Gaussian noise added to distance.
    -   `dropoff_general_rate`: Probability of losing a point (Rain/Absorptive surface).

### 🔹 Part 3: Radar Simulation

Not ray-tracing (too slow). Uses bounding boxes + ray casting for occlusion.
-   **Output:** List of detections (not an image).
    -   `depth`: Distance.
    -   `azimuth`, `elevation`: Angles.
    -   `velocity`: Doppler shift (Radial speed).
-   **Noise:** Radar is noisy! Ghost targets are not simulated by default in CARLA, but noise in measurements is.

---

## 💻 Implementation: The Sensor Rig

**Scenario:**
-   Spawn a vehicle.
-   Attach:
    -   **RGB Camera** (Roof).
    -   **LiDAR** (Roof).
    -   **Radar** (Front Bumper).
-   Drive and visualize all 3 streams.

### 🛠️ Setup
Create `week14_day95` and `sensor_rig.py`.

```bash
mkdir -p ~/ros2_ws/src/week14_day95
cd ~/ros2_ws/src/week14_day95
touch sensor_rig.py
```

### 👨‍💻 Code: Multi-Sensor Setup

```python
import carla
import random
import time
import numpy as np
import cv2
import open3d as o3d
import queue

# --- Sensor Callbacks ---
def process_img(image, q):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3] # BGR
    q.put(i3)

def process_lidar(lidar_data, q):
    # Data is float32: x, y, z, intensity
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    q.put(points[:, :3]) # XYZ only

def process_radar(radar_data, q):
    # Data: velocity, azimuth, elevation, depth
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    q.put(points)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # Queues for sync
    img_q = queue.Queue()
    lidar_q = queue.Queue()
    radar_q = queue.Queue()
    
    actors = []
    
    try:
        # 1. Spawn Ego
        bp = bp_lib.filter('model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego = world.spawn_actor(bp, spawn_point)
        ego.set_autopilot(True)
        actors.append(ego)
        
        # 2. Camera Setup
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        cam.listen(lambda data: process_img(data, img_q))
        actors.append(cam)
        
        # 3. LiDAR Setup (Velodyne 64 style)
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('noise_stddev', '0.02') # 2cm noise
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego)
        lidar.listen(lambda data: process_lidar(data, lidar_q))
        actors.append(lidar)
        
        # 4. Radar Setup
        radar_bp = bp_lib.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '10')
        radar_bp.set_attribute('range', '50')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=ego)
        radar.listen(lambda data: process_radar(data, radar_q))
        actors.append(radar)
        
        # 5. Visualization Loop
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='LiDAR + Radar', width=640, height=480)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        
        print("Running Sensor Rig...")
        while True:
            # Get Data
            if not img_q.empty():
                img = img_q.get()
                cv2.imshow("Camera", img)
                cv2.waitKey(1)
                
            if not lidar_q.empty():
                pts = lidar_q.get()
                # Update Open3D
                pcd.points = o3d.utility.Vector3dVector(pts)
                # Colorize by height
                colors = np.zeros_like(pts)
                colors[:, 1] = 1.0 # Green
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                
            if not radar_q.empty():
                # Radar points are sparse
                r_pts = radar_q.get()
                # Convert Polar to Cartesian
                # vel, azi, ele, depth
                # x = depth * cos(azi) * cos(ele)
                # y = depth * sin(azi) * cos(ele)
                # z = depth * sin(ele)
                cart_pts = []
                for p in r_pts:
                    depth = p[3]
                    azi = p[1]
                    ele = p[2]
                    x = depth * np.cos(azi) * np.cos(ele)
                    y = depth * np.sin(azi) * np.cos(ele)
                    z = depth * np.sin(ele)
                    cart_pts.append([x, y, z])
                
                if len(cart_pts) > 0:
                    print(f"Radar Detections: {len(cart_pts)}")
                    # Ideally, visualize these as large red dots in Open3D
                    # For now, just print
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        for actor in actors:
            actor.destroy()
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Noise Test

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Camera: Clear image.
    -   LiDAR: Dense point cloud (Green). Walls are fuzzy (due to `noise_stddev=0.02`).
    -   Radar: Prints detections only when cars/walls are in front.
3.  **Experiment:**
    -   Increase `noise_stddev` to `0.2` (20cm).
    -   **Result:** The walls in the LiDAR view become thick and blurry. This simulates a cheap LiDAR or bad weather.
    -   **Impact:** SLAM algorithms might fail to match scans if noise is too high.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Empty Queue"
**Symptom:** Visualization lags or freezes.
**Cause:** Python `queue` is thread-safe but slow. Visualization loop is slower than sensor rate (20Hz vs 10Hz).
**Solution:** Use `queue.get_nowait()` and drop old frames if the queue is full. Only visualize the latest data.

#### 2. Radar Coordinates
**Symptom:** Radar points appear sideways.
**Cause:** CARLA Radar uses `azimuth` relative to the sensor's forward vector.
**Solution:** Ensure your Polar-to-Cartesian math matches CARLA's coordinate system (X-forward, Y-right, Z-up).

---

## ⚡ Optimization & Best Practices

### 1. Semantic LiDAR
CARLA offers `sensor.lidar.ray_cast_semantic`.
-   **Output:** XYZ + Class Label (Road, Car, Pedestrian).
-   **Use:** Perfect for training segmentation networks (Ground Truth) without manual labeling.

### 2. Sensor Synchronization
In the code above, sensors drift (Camera at t=1.01, LiDAR at t=1.02).
-   **Solution:** Use **Synchronous Mode** (Day 93).
-   In the loop: `world.tick()`. Then read all queues. They will correspond to the exact same simulation frame.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Radar simulation faster than LiDAR?
    *   **A:** Radar outputs sparse detections (objects), not a dense point cloud. It doesn't need to ray-cast millions of points.
2.  **Q:** What is "Dropoff Rate" in LiDAR?
    *   **A:** The probability that a laser pulse is lost (absorbed by black asphalt or scattered by rain) and returns no reading.
3.  **Q:** How do I simulate a "Fisheye" camera?
    *   **A:** Use the `sensor.camera.fisheye` blueprint and configure the lens distortion parameters.

### Challenge Task
**Task:** Sensor Fusion Visualization.
1.  Project LiDAR points onto the Camera image.
2.  Use the extrinsic matrix (LiDAR to Camera).
3.  Use the intrinsic matrix (Camera to Image).
4.  Draw colored dots on the image corresponding to depth.

---

## 📚 Further Reading & References
-   [CARLA Sensors Reference](https://carla.readthedocs.io/en/latest/ref_sensors/)
-   [Open3D Visualization](http://www.open3d.org/docs/release/tutorial/visualization/index.html)

---

**Day 95 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
