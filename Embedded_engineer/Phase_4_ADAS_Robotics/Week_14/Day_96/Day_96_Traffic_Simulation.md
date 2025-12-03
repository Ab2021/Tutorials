# Day 96: Traffic Simulation
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 96 Focus:**
> An empty city is easy to drive in. A real city has chaos: Jaywalkers, aggressive taxis, and broken traffic lights. Today, we populate our CARLA world with **Smart Traffic** to stress-test our AV algorithms.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Deploy** the Traffic Manager (TM) to control NPC vehicles.
2.  **Configure** NPC behaviors (Aggressiveness, Speeding, Lane changes).
3.  **Spawn** Pedestrians with AI controllers (Walking, Crossing).
4.  **Synchronize** Traffic Lights for green waves.
5.  **Create** a "Chaos Scenario" with high density and rule violations.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 93:** CARLA Python API.
-   **NavMesh:** Navigation Mesh for pedestrians.

### Hardware Requirements
-   **CPU:** Traffic simulation is CPU-intensive.

### Software Stack
-   **CARLA:** `TrafficManager` API.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Traffic Manager (TM)

The TM is a module in CARLA that controls all non-player vehicles (NPCs) in "Autopilot" mode.
-   **Centralized:** One TM controls hundreds of cars.
-   **Hybrid Physics:**
    -   **Far away:** Simplified physics (teleportation) to save CPU.
    -   **Nearby:** Full physics (suspension, friction).
-   **Parameters:**
    -   `global_percentage_speed_difference`: -30% means drive 30% faster than limit.
    -   `ignore_lights_percentage`: % chance to run a red light.
    -   `distance_to_leading_vehicle`: Tailgating distance.

### 🔹 Part 2: Pedestrian Navigation

Pedestrians don't use lanes; they use the **NavMesh** (Sidewalks + Crosswalks).
-   **Walker Controller:** An AI controller attached to the pedestrian actor.
-   **Behavior:** Random walk, go to target, cross street.
-   **Risk:** Pedestrians can be set to "Jaywalk" (cross where there is no crosswalk).

---

## 💻 Implementation: Urban Chaos

**Scenario:**
-   Spawn 50 Vehicles.
-   Spawn 50 Pedestrians.
-   Set 10% of cars to be "Aggressive" (Speeding, ignoring lights).
-   Set 10% of pedestrians to run across the street.

### 🛠️ Setup
Create `week14_day96` and `traffic_gen.py`.

```bash
mkdir -p ~/ros2_ws/src/week14_day96
cd ~/ros2_ws/src/week14_day96
touch traffic_gen.py
```

### 👨‍💻 Code: Traffic Generator

```python
import carla
import random
import time
import logging

def main():
    # 1. Connect
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 2. Setup Traffic Manager
    tm = client.get_trafficmanager(8000) # Port 8000
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_hybrid_physics_mode(True) # Optimization
    tm.set_hybrid_physics_radius(70.0) # Physics only within 70m of Ego
    
    # 3. Spawn Vehicles
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    
    spawn_points = world.get_map().get_spawn_points()
    number_of_vehicles = 50
    
    vehicles_list = []
    
    print(f"Spawning {number_of_vehicles} vehicles...")
    
    for _ in range(number_of_vehicles):
        bp = random.choice(vehicle_bps)
        spawn_point = random.choice(spawn_points)
        
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, tm.get_port())
            vehicles_list.append(vehicle)
            
            # Configure Behavior (Randomly)
            if random.random() < 0.1: # 10% Aggressive
                tm.vehicle_percentage_speed_difference(vehicle, -30) # 30% faster
                tm.ignore_lights_percentage(vehicle, 100) # Run red lights
                tm.distance_to_leading_vehicle(vehicle, 0.5) # Tailgate
                print(f"Spawned Aggressive Driver: {vehicle.id}")
            else: # 90% Normal
                tm.vehicle_percentage_speed_difference(vehicle, 0) # Speed limit
                
    # 4. Spawn Pedestrians
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')
    
    number_of_walkers = 50
    walkers_list = []
    controllers_list = []
    
    print(f"Spawning {number_of_walkers} pedestrians...")
    
    # 4.1. Spawn Locations on Sidewalk
    all_locs = []
    for i in range(number_of_walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            all_locs.append(loc)
            
    # 4.2. Spawn Walkers
    for i in range(len(all_locs)):
        bp = random.choice(walker_bps)
        loc = all_locs[i]
        trans = carla.Transform(loc)
        
        walker = world.try_spawn_actor(bp, trans)
        if walker:
            walkers_list.append(walker)
            
            # Attach Controller
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            controllers_list.append(controller)
            
            # Start Walking
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            
            # Configure Speed
            if random.random() < 0.1: # 10% Runners
                controller.set_max_speed(4.0) # Run
            else:
                controller.set_max_speed(1.4) # Walk

    print("Traffic Simulation Running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
            # Optional: Reshuffle destinations for walkers if they arrive
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        client.apply_batch([carla.command.DestroyActor(x) for x in controllers_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Red Light Runner

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Cars are driving around.
    -   Pedestrians are walking on sidewalks.
    -   **Watch closely:** Look for the "Aggressive" drivers (they drive fast).
3.  **Experiment:**
    -   Find an intersection.
    -   Wait for a Red Light.
    -   You might see an aggressive car blow through it.
    -   **Impact:** If your AV assumes "Green means Safe", it will crash. Your AV must check "Green AND Intersection Clear".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Spawn Failed"
**Symptom:** `try_spawn_actor` returns None.
**Cause:** Spawn point blocked by another vehicle or collision.
**Solution:** This is normal. The script handles it by checking `if vehicle:`. Just try more spawn points.

#### 2. Pedestrians Stuck
**Symptom:** Walkers standing still.
**Cause:** Controller not started or target location unreachable.
**Solution:** Ensure `controller.start()` is called and `go_to_location` uses a valid NavMesh point.

---

## ⚡ Optimization & Best Practices

### 1. Hybrid Physics
Simulating 100 cars with full physics kills the CPU.
-   **Hybrid Mode:** Only cars within `radius` (e.g., 70m) of the *Hero Vehicle* (tagged with `role_name='hero'`) get full physics. Others are teleported.
-   **Note:** You must register your Ego vehicle with the TM for this to work.

### 2. Seeded Randomness
For reproducible tests:
-   `tm.set_random_device_seed(42)`
-   `random.seed(42)`
-   Ensures the same cars spawn in the same places every time.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `set_autopilot(True)` and `TrafficManager`?
    *   **A:** `set_autopilot(True)` hands control of the vehicle to the Traffic Manager. The TM is the *brain* behind the autopilot.
2.  **Q:** How do pedestrians know where to walk?
    *   **A:** They use the **NavMesh** (Navigation Mesh), a simplified map of walkable areas (sidewalks) baked into the map.
3.  **Q:** Why do we need aggressive drivers in simulation?
    *   **A:** To test **Defensive Driving**. Real roads are not perfect; AVs must handle rule-breakers.

### Challenge Task
**Task:** The Jaywalker.
1.  Modify the script to force a pedestrian to cross the road *outside* a crosswalk.
2.  Use `controller.go_to_location()` with a target on the other side of the road.
3.  Observe if cars stop for them (TM usually respects physics and will brake if it sees an obstacle).

---

## 📚 Further Reading & References
-   [CARLA Traffic Manager](https://carla.readthedocs.io/en/latest/adv_traffic_manager/)
-   [Pedestrian Navigation](https://carla.readthedocs.io/en/latest/tuto_G_pedestrian_control/)

---

**Day 96 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
