# Day 45: Creating Custom Worlds (Building Editor)
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 45 Focus:**
> A robot in an empty void is boring. To test ADAS algorithms, we need roads, intersections, buildings, and traffic signs. Today, we become architects. We will use the **Gazebo Building Editor** and **SDF** to construct a realistic "City Block" environment for our RoboCar.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Navigate** the Gazebo Building Editor to construct floor plans.
2.  **Create** custom models (e.g., Traffic Signs) using the Model Editor.
3.  **Understand** the relationship between URDF (ROS) and SDF (Gazebo).
4.  **Populate** a world with static objects (Trees, Hydrants) and dynamic actors (Walking Humans).
5.  **Export** and organize your custom world as a ROS 2 package.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 44:** Gazebo Basics.
-   **XML:** SDF syntax.

### Hardware Requirements
-   **Mouse:** Essential for 3D editing.

### Software Stack
-   **Gazebo Classic:** Building Editor tool.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: URDF vs SDF

-   **URDF (Unified Robot Description Format):**
    -   Tree structure (Parent -> Child).
    -   Used by ROS (Rviz, TF, MoveIt).
    -   Cannot describe a "World" (only a robot).
-   **SDF (Simulation Description Format):**
    -   Graph structure (Loops allowed).
    -   Used by Gazebo.
    -   Can describe Robots, Worlds, Lights, Physics.

**The Pipeline:**
When you spawn a URDF in Gazebo, ROS converts it to SDF automatically. But for static environments (Buildings), we write SDF directly (or use the GUI).

### 🔹 Part 2: The Building Editor

A built-in tool in Gazebo to create 3D structures from 2D floor plans.
-   **Walls:** Define the perimeter.
-   **Windows/Doors:** Cut holes in walls.
-   **Textures:** Brick, Wood, Concrete.
-   **Export:** Saves as a Model (SDF + Meshes).

### 🔹 Part 3: The Model Editor

Used to create composite objects.
-   **Example:** A "Stop Sign".
    -   Link 1: Pole (Cylinder).
    -   Link 2: Sign (Octagon Box).
    -   Joint: Fixed.
    -   Material: Stop Sign Texture.

---

## 💻 Implementation: Building "SimCity"

We will create a custom world package `city_sim` and build a simple city block.

### 🛠️ Setup
Create `week7_day45` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week7_day45
cd week7_day45
mkdir worlds models maps
```

### 👨‍💻 Workflow: Using the GUI (Instructions)

Since we cannot click the mouse for you, here is the step-by-step guide to build the world.

#### Step 1: The Office Building
1.  Open Gazebo: `gazebo`.
2.  **Edit -> Building Editor** (Ctrl+B).
3.  **Create Walls:** Draw a rectangle (10m x 10m).
4.  **Add Features:** Add a Door and two Windows.
5.  **Add Texture:** Right-click wall -> Open Inspector -> Select "Brick".
6.  **Save:** File -> Save As -> `office_building`.
    -   Save to `~/ros2_ws/src/week7_day45/models/office_building`.
7.  **Exit** Building Editor.

#### Step 2: The Traffic Sign
1.  **Edit -> Model Editor** (Ctrl+M).
2.  **Add Pole:** Simple Shape -> Cylinder. Resize to thin pole.
3.  **Add Sign:** Simple Shape -> Box. Resize to flat square. Place on top.
4.  **Joint:** Joint tab -> Create Fixed Joint between Pole and Sign.
5.  **Save:** File -> Save As -> `stop_sign`.
    -   Save to `~/ros2_ws/src/week7_day45/models/stop_sign`.

#### Step 3: Assembling the World
1.  **Insert:** Use the "Insert" tab to place your `office_building` and `stop_sign`.
2.  **Roads:** Insert -> Road. Draw a loop around the building.
3.  **Save World:** File -> Save World As -> `city.world`.
    -   Save to `~/ros2_ws/src/week7_day45/worlds/city.world`.

### 👨‍💻 Code: city.world (SDF Source)

If you prefer code (or want to verify the GUI output), here is what the SDF looks like.

```xml
<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <!-- Sun & Ground -->
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- 1. Road -->
    <road name="main_street">
      <width>7.34</width> <!-- 2 lanes -->
      <point>0 0 0</point>
      <point>50 0 0</point>
      <point>50 50 0</point>
      <point>0 50 0</point>
      <point>0 0 0</point>
    </road>

    <!-- 2. Custom Building (Include) -->
    <!-- Note: You must add the model path to GAZEBO_MODEL_PATH -->
    <include>
      <uri>model://office_building</uri>
      <pose>25 25 0 0 0 0</pose>
    </include>

    <!-- 3. Stop Sign -->
    <include>
      <uri>model://stop_sign</uri>
      <pose>45 5 0 0 0 1.57</pose>
    </include>

    <!-- 4. Dynamic Actor (Walking Human) -->
    <actor name="pedestrian">
      <skin>
        <filename>walk.dae</filename>
      </skin>
      <animation name="walking">
        <filename>walk.dae</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <delay_start>0.000000</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          <waypoint>
            <time>0.0</time>
            <pose>10 0 0 0 0 0</pose>
          </waypoint>
          <waypoint>
            <time>5.0</time>
            <pose>20 0 0 0 0 0</pose>
          </waypoint>
          <waypoint>
            <time>10.0</time>
            <pose>10 0 0 0 0 0</pose>
          </waypoint>
        </trajectory>
      </script>
    </actor>

  </world>
</sdf>
```

### 👨‍💻 Code: launch_city.launch.py

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_path = get_package_share_directory('week7_day45')
    
    # Add our models folder to GAZEBO_MODEL_PATH
    # This is crucial so Gazebo finds 'office_building'
    model_path = os.path.join(pkg_path, 'models')
    
    # Note: In a real script, you append to the existing path.
    # For simplicity, we assume this is the primary path or handled by hook.
    # Better way: Use hooks in package.xml.
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': os.path.join(pkg_path, 'worlds', 'city.world')}.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
install(DIRECTORY worlds models
  DESTINATION share/${PROJECT_NAME}
)
```

### 👨‍💻 Code: package.xml (Export Hook)

To make Gazebo find your models automatically:

```xml
<export>
  <build_type>ament_cmake</build_type>
  <gazebo_ros gazebo_model_path="${prefix}/models"/>
</export>
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_day45
source install/setup.bash
ros2 launch week7_day45 launch_city.launch.py
```

---

## 🔬 Lab Exercise: The Pedestrian

### Lab Objectives
1.  Launch the city world.
2.  **Observation:** You should see a human figure walking back and forth between x=10 and x=20.
3.  **Interaction:** Spawn your RoboCar (from Day 44) into this world.
    ```bash
    ros2 run gazebo_ros spawn_entity.py -topic robot_description -entity robocar -x 0 -y 0
    ```
4.  **Drive:** Drive the car. Try to hit the pedestrian.
    -   *Result:* The pedestrian is a "Ghost" (Visual only) unless you add `<collision>` tags to the actor SDF (which is complex). For now, just avoid them!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Models Not Found (Black/Missing)
**Symptom:** Gazebo errors "Unable to find uri[model://office_building]".
**Cause:** `GAZEBO_MODEL_PATH` environment variable does not include your package's `models` folder.
**Solution:** Source the setup file (`source install/setup.bash`). Verify the `package.xml` export tag.

#### 2. Textures Missing (Purple/Black Checkerboard)
**Symptom:** Walls look weird.
**Cause:** Gazebo cannot find the material scripts.
**Solution:** Ensure the `materials` folder inside your model is structured correctly (`scripts`, `textures`).

#### 3. Shadows Flickering
**Symptom:** Weird visual artifacts.
**Cause:** Graphics driver issues or multiple light sources.
**Solution:** Disable shadows in Gazebo (Scene -> Shadows: False).

---

## ⚡ Optimization & Best Practices

### 1. Static vs Dynamic
Mark buildings as `<static>true</static>`.
-   **Why?** The physics engine ignores them for dynamics (gravity/forces), treating them as immovable obstacles. Saves massive CPU.

### 2. Heightmaps
For terrain (hills, mountains), don't use meshes. Use **Heightmaps**.
-   Grayscale image (White = High, Black = Low).
-   Gazebo optimizes collision for heightmaps efficiently.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Can I use a URDF file for a building?
    *   **A:** Technically yes, but SDF is better. URDF is for articulated robots (chains). SDF is for environments (graphs).
2.  **Q:** What is an "Actor" in Gazebo?
    *   **A:** A scripted entity (like a walking human) that plays an animation and follows a trajectory, but usually doesn't interact fully with physics (no ragdoll).
3.  **Q:** How do I share my Gazebo world with a friend?
    *   **A:** Package it as a ROS 2 package. Include the `.world` file and all custom `models`. Use `package.xml` exports to handle paths.

### Challenge Task
**Task:** Traffic Lights.
1.  Create a Traffic Light model (Box + 3 Spheres).
2.  Add a plugin `libTrafficLightPlugin.so` (Conceptual) or simply toggle the `<emissive>` property of the spheres to simulate Red/Green/Yellow.

---

## 📚 Further Reading & References
-   [Gazebo Building Editor Tutorial](http://gazebosim.org/tutorials?tut=building_editor)
-   [Gazebo Actors](http://gazebosim.org/tutorials?tut=actor&cat=build_robot)

---

**Day 45 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
