# Day 154: Week 22 Review & Project (Stress Testing)
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 154 Focus:**
> We have learned how the world tries to break our robot. Rain, Glare, Shadows, Cones, Sirens, and Hackers. Today, we build a **Stress Test Suite** (The Torture Chamber) to validate that our system fails gracefully, not catastrophically.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Design** a comprehensive Test Plan for Edge Cases.
2.  **Implement** a Scenario Runner in CARLA with randomized adverse conditions.
3.  **Analyze** Failure Modes (False Positives vs False Negatives).
4.  **Calculate** Safety Metrics (Miles Per Disengagement).
5.  **Demonstrate** system robustness (or lack thereof).

---

## 📚 Week 22 Review

### 1. Environmental Conditions
-   **Weather:** Rain/Snow degrades Lidar/Camera. Augmentation helps.
-   **Lighting:** Glare/Night requires HDR and Gamma Correction.

### 2. Complex Scenes
-   **Occlusions:** Tracking memory (Kalman Filter) is essential.
-   **Shadows:** Texture analysis distinguishes them from obstacles.
-   **Construction:** Perception-based navigation (Cones) replaces Maps.

### 3. Rare & Malicious
-   **Emergency Vehicles:** Audio/Visual detection + Yield logic.
-   **Adversarial:** Sensor fusion defeats single-sensor attacks.

---

## 🛠️ Capstone Project: The Torture Chamber

**Goal:** Drive 500m in CARLA under randomized severe conditions.
**Success Criteria:** No collisions, No lane departures, Stop for "Emergency".

### Package Structure
Create `week22_project` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week22_project
mkdir -p week22_project/scenarios
```

### 👨‍💻 Code: The Scenario Runner

We will use a Python script to control CARLA's environment and spawn edge cases.

```python
import carla
import random
import time
import numpy as np

def main():
    # 1. Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 2. Set Weather (The Storm)
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=90.0, # Heavy Rain
        precipitation_deposits=50.0, # Puddles
        wind_intensity=50.0,
        sun_azimuth_angle=45.0,
        sun_altitude_angle=10.0, # Low sun (Glare)
        fog_density=30.0
    )
    world.set_weather(weather)
    print("Weather set to: Heavy Rain + Fog + Glare")
    
    # 3. Spawn Ego Vehicle
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    ego_vehicle = world.try_spawn_actor(ego_bp, spawn_points[0])
    
    # 4. Spawn Edge Cases
    
    # A. Construction Zone (Cones)
    cone_bp = bp_lib.find('static.prop.constructioncone')
    # Spawn a line of cones blocking the right lane
    start_loc = spawn_points[0].location
    for i in range(10, 50, 5):
        loc = carla.Location(x=start_loc.x + i, y=start_loc.y + 2, z=start_loc.z)
        world.try_spawn_actor(cone_bp, carla.Transform(loc))
        
    # B. Emergency Vehicle (Police)
    police_bp = bp_lib.find('vehicle.dodge.charger_police')
    police_loc = spawn_points[1]
    police_car = world.try_spawn_actor(police_bp, police_loc)
    police_car.set_autopilot(True)
    # Turn on Siren/Lights (CARLA API specific)
    # police_car.set_light_state(...)
    
    # C. Occluder (Bus)
    bus_bp = bp_lib.find('vehicle.mitsubishi.fusorosa')
    bus_loc = carla.Transform(carla.Location(x=start_loc.x + 60, y=start_loc.y, z=start_loc.z))
    world.try_spawn_actor(bus_bp, bus_loc)
    
    print("Scenario Setup Complete. Good Luck!")
    
    # Monitor Loop
    try:
        while True:
            # Here we would log metrics
            # dist_to_obstacle = ...
            time.sleep(1)
    except KeyboardInterrupt:
        print("Cleaning up...")
        # Destroy actors...

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Rain Test
-   **Observation:** Camera is blurry. Lidar has noise points.
-   **Pass:** Perception stack filters noise and still detects the lead car.
-   **Fail:** Phantom braking due to rain noise.

### 2. The Glare Test
-   **Observation:** Sun is low. Camera image is white-washed.
-   **Pass:** Lane detection uses HDR/Gamma and finds lines. Or system hands over to Map/Lidar.
-   **Fail:** Car drifts out of lane.

### 3. The Cone Test
-   **Observation:** Lane blocked by cones.
-   **Pass:** Planner detects blockage, switches to "Construction Mode", and changes lanes.
-   **Fail:** Car hits cones or stops indefinitely.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Theory
1.  **Q:** What is the "Long Tail" problem?
    *   **A:** The infinite number of rare, weird edge cases that make up the last 1% of autonomous driving development.
2.  **Q:** How does Fog affect Lidar?
    *   **A:** Scattering. The beam hits water droplets and returns early (false positive) or scatters away (false negative).

### Section 2: Strategy
3.  **Q:** If all sensors fail (Whiteout Blizzard), what should the car do?
    *   **A:** **Minimal Risk Condition (MRC)**. Turn on hazards, slow down gradually, pull over if possible, and stop.

---

## 🏆 Conclusion

You have survived Week 22.
You now understand that making a car drive is easy. Making it **safe** is hard.
Next week, we move to **Testing & Validation** (MIL/SIL/HIL), where we formalize this process into industry-standard V-Model testing.

---

**Day 154 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
