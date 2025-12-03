# Day 93: CARLA Simulator Setup
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 93 Focus:**
> Gazebo is great for robotics (arms, indoor rovers). But for **Autonomous Driving**, the king is **CARLA**. It provides photorealistic rendering, realistic physics, and a rich library of assets (cars, pedestrians, buildings). Today, we start our CARLA journey.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Client-Server architecture of CARLA.
2.  **Install** and Run the CARLA Simulator (Server).
3.  **Connect** a Python Client to spawn a vehicle.
4.  **Control** the vehicle (Throttle, Steer, Brake) via API.
5.  **Differentiate** between Synchronous and Asynchronous modes.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Python:** OOP.
-   **Networking:** TCP/IP (Client connects to localhost:2000).

### Hardware Requirements
-   **GPU:** NVIDIA GTX 1060 or better (Required for rendering).
-   **RAM:** 16GB+.

### Software Stack
-   **CARLA:** 0.9.13 or newer.
-   **Python:** `carla` egg file.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Architecture

CARLA is based on **Unreal Engine 4 (UE4)**.
-   **Server:** Runs the physics and rendering. Listens on port 2000.
-   **Client:** Your Python script. Sends commands ("Steer 0.5") and receives data ("Image 1024x768").
-   **Assets:** Maps (Town01, Town02...), Blueprints (Tesla Model 3, Pedestrian...).

### 🔹 Part 2: Sync vs Async

-   **Asynchronous (Default):**
    -   Server runs as fast as possible (e.g., 60 FPS).
    -   Client sends commands whenever.
    -   **Problem:** If Client is slow (processing AI), it misses frames. The car crashes because the server kept moving time forward.
-   **Synchronous:**
    -   Server waits for a "Tick" from the Client.
    -   **Benefit:** Deterministic. The simulation time only advances when your AI is ready. Essential for training ML models.

### 🔹 Part 3: The Python API

Everything is an object.
-   `World`: The simulation environment.
-   `Blueprint`: The template (e.g., "vehicle.tesla.model3").
-   `Actor`: The spawned instance.
-   `Sensor`: Attached to an actor (Camera, Lidar).

---

## 💻 Implementation: Hello CARLA

**Scenario:**
-   Spawn a Mercedes.
-   Set it to "Autopilot" (Built-in traffic manager).
-   Attach a Camera.
-   Drive for 10 seconds.

### 🛠️ Setup
Assuming CARLA is installed in `/opt/carla-simulator`.
Create `week14_day93` and `hello_carla.py`.

```bash
mkdir -p ~/ros2_ws/src/week14_day93
cd ~/ros2_ws/src/week14_day93
touch hello_carla.py
```

### 👨‍💻 Code: Spawning and Driving

```python
import glob
import os
import sys
import time
import random
import numpy as np
import cv2

# --- Import CARLA Egg ---
# Find the .egg file in the CARLA install directory
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    actor_list = []
    
    try:
        # 1. Connect to Server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # 2. Get World
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 3. Pick a Car
        bp = blueprint_library.filter('model3')[0] # Tesla Model 3
        
        # 4. Pick a Spawn Point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # 5. Spawn Actor
        vehicle = world.spawn_actor(bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned {vehicle.type_id} at {spawn_point.location}")
        
        # 6. Set Autopilot (Let CARLA drive it)
        vehicle.set_autopilot(True)
        
        # 7. Attach Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Position: On the hood
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        # 8. Define Callback for Camera Data
        def process_img(image):
            # Convert raw data to numpy array
            i = np.array(image.raw_data)
            i2 = i.reshape((600, 800, 4)) # BGRA
            i3 = i2[:, :, :3] # BGR
            cv2.imshow("CARLA Camera", i3)
            cv2.waitKey(1)
            
        camera.listen(lambda image: process_img(image))
        
        # 9. Run Simulation Loop
        print("Running simulation for 20 seconds...")
        time.sleep(20)
        
    finally:
        # 10. Cleanup (Very Important!)
        print("Destroying actors...")
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Traffic Jam

### Lab Objectives
1.  Start CARLA Server: `./CarlaUE4.sh` (in the install dir).
2.  Run the script: `python3 hello_carla.py`.
3.  **Observation:**
    -   A window opens showing the camera view from the Tesla.
    -   The car drives itself, obeying traffic lights and avoiding walls.
4.  **Experiment:**
    -   Spawn 50 cars instead of 1.
    -   `world.spawn_actor(bp, random.choice(spawn_points))` in a loop.
    -   **Result:** You see other cars interacting. The "Autopilot" handles basic collision avoidance.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Connection Refused"
**Symptom:** Client cannot connect.
**Cause:** Server is not running or firewall blocking port 2000.
**Solution:** Check if `./CarlaUE4.sh` is running. Check `netstat -an | grep 2000`.

#### 2. "Orphaned Actors"
**Symptom:** You restart the script, but the old car is still there, piled on top of the new one.
**Cause:** Script crashed before `actor.destroy()` was called.
**Solution:**
    -   Always use `try...finally`.
    -   Or use the `config.py` script provided by CARLA to clean the world.

---

## ⚡ Optimization & Best Practices

### 1. No Rendering Mode
If training an RL agent, you don't need to see the UE4 window.
-   Run server with `./CarlaUE4.sh -RenderOffScreen`.
-   Saves GPU for the actual sensor computation.

### 2. Fixed Time Step
For reproducible experiments (and RL):
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05 # 20 FPS
world.apply_settings(settings)

# In loop:
world.tick()
```

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What happens if I don't destroy actors?
    *   **A:** They stay in the server forever (until server restart). This consumes memory and CPU.
2.  **Q:** Why use Synchronous Mode for ML training?
    *   **A:** To ensure the model receives every frame in order, and the physics doesn't "jump" while the model is thinking (inference time).
3.  **Q:** What is a Blueprint?
    *   **A:** A definition of an actor (e.g., "vehicle.audi.tt"). You spawn an Actor *from* a Blueprint.

### Challenge Task
**Task:** Manual Control.
1.  Disable `set_autopilot`.
2.  Use `vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))`.
3.  Make the car drive in a circle.

---

## 📚 Further Reading & References
-   [CARLA Python API Reference](https://carla.readthedocs.io/en/latest/python_api/)
-   [CARLA First Steps](https://carla.readthedocs.io/en/latest/tuto_first_steps/)

---

**Day 93 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
