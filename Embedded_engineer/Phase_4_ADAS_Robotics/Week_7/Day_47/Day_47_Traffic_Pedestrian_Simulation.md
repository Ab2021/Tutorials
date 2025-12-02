# Day 47: Traffic & Pedestrian Simulation (Actors)
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 47 Focus:**
> A self-driving car that can't handle pedestrians is a danger to society. Testing with real people is unethical (and illegal). Today, we populate our Gazebo world with **Actors**—animated humans and vehicles that follow scripted paths—to create challenging scenarios like jaywalking and traffic jams.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** an Actor in SDF: Skin (Mesh), Animation, and Trajectory.
2.  **Script** a "Jaywalker" scenario where a pedestrian crosses the road unexpectedly.
3.  **Create** a "Traffic Jam" using simple vehicle models.
4.  **Control** actors dynamically (Stop/Go) using a custom Gazebo plugin (Conceptual).
5.  **Analyze** the limitations of Gazebo Actors (No physics interaction).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 45:** SDF Basics.
-   **Animation:** Keyframes, Skeleton animation (concept).

### Hardware Requirements
-   **GPU:** Recommended for rendering animated meshes.

### Software Stack
-   **Gazebo Classic:** Actor support.
-   **Assets:** Standard Gazebo models (`walk.dae`, `moonwalk.dae`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is an Actor?

In Gazebo, an **Actor** is a special entity.
-   **Visual:** It has a mesh (Skin) and animations (Skeleton).
-   **Movement:** It follows a pre-defined trajectory (Waypoints).
-   **Physics:** **None.** It is a "Ghost". It passes through walls and robots.
    -   *Workaround:* You can attach a `<collision>` box to it, but it won't ragdoll. It will act like a moving brick wall.

### 🔹 Part 2: Scripting Trajectories

Defined in SDF using `<script>`.
-   **Loop:** True/False.
-   **Delay Start:** Wait X seconds before moving.
-   **Trajectory:** A list of Waypoints $(Time, Pose)$.
    -   Gazebo interpolates between waypoints.

### 🔹 Part 3: Dynamic Control

Standard actors are "dumb" (Open Loop).
To make a pedestrian stop when a car approaches, we need a **Plugin**.
-   **`libActorPlugin.so`:** C++ code that overrides the trajectory.
-   It can subscribe to a ROS topic (`/actor/cmd`) and change the animation speed or target pose.

---

## 💻 Implementation: The "Jaywalker" Scenario

We will create a world `traffic.world` where:
1.  **Ego Car:** Starts at (0, 0).
2.  **Pedestrian:** Starts on the sidewalk, waits 5 seconds, then crosses the road.
3.  **Traffic Car:** Drives in the opposite lane.

### 🛠️ Setup
Use `week7_day47` (copy from `week7_day46`).

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week7_day47
mkdir worlds
touch worlds/traffic.world
```

### 👨‍💻 Code: traffic.world

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- 1. Road -->
    <road name="main_road">
      <width>7.0</width>
      <point>-50 0 0</point>
      <point>50 0 0</point>
    </road>

    <!-- 2. The Jaywalker -->
    <actor name="jaywalker">
      <skin>
        <filename>walk.dae</filename>
        <scale>1.0</scale>
      </skin>
      <animation name="walking">
        <filename>walk.dae</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <delay_start>0.0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          <!-- Stand on sidewalk for 5s -->
          <waypoint>
            <time>0.0</time>
            <pose>10 -5 0 0 0 1.57</pose> <!-- Facing Road -->
          </waypoint>
          <waypoint>
            <time>5.0</time>
            <pose>10 -5 0 0 0 1.57</pose>
          </waypoint>
          
          <!-- Cross the road (5s to cross 10m) -->
          <waypoint>
            <time>10.0</time>
            <pose>10 5 0 0 0 1.57</pose>
          </waypoint>
          
          <!-- Walk back -->
          <waypoint>
            <time>15.0</time>
            <pose>10 -5 0 0 0 -1.57</pose>
          </waypoint>
        </trajectory>
      </script>
      
      <!-- Hack: Add Collision (Invisible Box) -->
      <!-- Note: Gazebo 11 Actors don't support collision tags natively well. 
           We usually attach a separate model to the actor using a plugin, 
           or just rely on Lidar seeing the mesh. -->
    </actor>

    <!-- 3. Traffic Car (Simple Box) -->
    <actor name="traffic_car">
      <skin>
        <filename>model://pickup/meshes/pickup.dae</filename> <!-- Standard Gazebo Model -->
        <scale>1.0</scale>
      </skin>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="driving">
          <waypoint>
            <time>0.0</time>
            <pose>50 2 0 0 0 3.14</pose> <!-- Right Lane, driving West -->
          </waypoint>
          <waypoint>
            <time>10.0</time>
            <pose>-50 2 0 0 0 3.14</pose>
          </waypoint>
        </trajectory>
      </script>
    </actor>

  </world>
</sdf>
```

### 👨‍💻 Code: launch_traffic.launch.py

Standard launch file pointing to `traffic.world`.

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_day47
source install/setup.bash
ros2 launch week7_day47 launch_traffic.launch.py
```

---

## 🔬 Lab Exercise: Lidar vs Ghost

### Lab Objectives
1.  Launch the world.
2.  **Observation:**
    -   See the "Jaywalker" standing at x=10, y=-5.
    -   See the "Traffic Car" driving past.
3.  **Lidar Check:**
    -   Spawn your RoboCar.
    -   Open Rviz2.
    -   Drive towards the Jaywalker.
    -   **Question:** Does the Lidar see him?
    -   **Answer:** Yes! The Lidar ray-casting works on the visual mesh of the actor.
4.  **Collision Check:**
    -   Drive *into* the Jaywalker.
    -   **Result:** You pass right through him. He is a ghost.
    -   **Lesson:** For testing "Emergency Braking", this is fine (Lidar sees him). For testing "Impact Physics", this is useless.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Actor Skating
**Symptom:** Actor moves but feet don't match speed (Moonwalking).
**Cause:** Trajectory speed $\neq$ Animation speed.
**Solution:** Adjust waypoint times. If animation is 1m/s, waypoints must be 1m apart per second.

#### 2. Actor Sinks into Ground
**Symptom:** Feet are underground.
**Cause:** Origin of mesh is at center, not feet.
**Solution:** Add a Z-offset in the `<pose>` tag (e.g., `0 0 1.0`).

#### 3. Missing Meshes
**Symptom:** Actor is invisible.
**Cause:** `walk.dae` not found.
**Solution:** Ensure `gazebo_models` is downloaded (`~/.gazebo/models`).

---

## ⚡ Optimization & Best Practices

### 1. Traffic Manager
For complex traffic (50 cars), writing SDF scripts is insane.
Use a **Traffic Manager** (like in CARLA).
-   Or write a Python script that generates the `.world` file with random trajectories.

### 2. Pedestrian Plugins
To make pedestrians "Smart" (avoid robot):
-   Use `libgazebo_ros_actor_command.so`.
-   Or switch to **Ignition Gazebo (Gazebo New)**, which has better actor support.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Can an Actor interact with the physics engine (e.g., push a box)?
    *   **A:** No. Actors are purely visual animations that follow a path. They have no mass or collision by default.
2.  **Q:** How does the Lidar see the Actor if it has no collision?
    *   **A:** Gazebo's Ray Sensor checks against the *Visual* geometry if configured, or specific collision masks. Actors are special; ray sensors are hardcoded to see their skins.
3.  **Q:** What is the difference between `<skin>` and `<animation>`?
    *   **A:** Skin is the 3D mesh (Body). Animation is the motion file (Bones moving). They must match (same skeleton).

### Challenge Task
**Task:** The Crowd.
1.  Write a Python script `generate_crowd.py`.
2.  Generate an SDF snippet with 10 actors.
3.  Randomize their start positions and walking directions.
4.  Insert into the world.

---

## 📚 Further Reading & References
-   [Gazebo Actor Tutorial](http://gazebosim.org/tutorials?tut=actor&cat=build_robot)
-   [Writing a Custom Actor Plugin](http://gazebosim.org/tutorials?tut=actor&cat=build_robot#Scriptingtrajectory)

---

**Day 47 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
