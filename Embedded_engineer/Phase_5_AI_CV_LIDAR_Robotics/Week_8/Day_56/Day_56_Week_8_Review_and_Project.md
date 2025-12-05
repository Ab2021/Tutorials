# Day 56: Week 8 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> We have mastered the Arm. Now we combine it with the Eyes and the Brain.
> - **Goal:** A "Mobile Manipulator" (e.g., Tiago, Fetch) stack.
> - **Code:** "Fetch the Coke Can". Navigation $\to$ Detection $\to$ Visual Servoing $\to$ Grasping $\to$ Force-Aware Placement.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Orchestrate** a complex state machine transitioning between Base Navigation and Arm Manipulation.
2.  **Mitigate** "Base Placement Errors" using Visual Servoing (Arm compensates for Base inaccuracy).
3.  **Implement** a robust Pick & Place pipeline with failure recovery (e.g., "Grasp Failed" $\to$ "Retry").
4.  **Demonstrate** End-to-End Mobile Manipulation in simulation.

---

## 📚 Week 8 Review: The Manipulation Stack

| Day | Topic | Key Concept | Use Case |
|-----|-------|-------------|----------|
| **50** | **Kinematics** | DH Parameters / FK | Where is the hand? |
| **51** | **IK** | Jacobian / Newton-Raphson | How to reach (x,y,z)? |
| **52** | **Planning** | MoveIt 2 / OMPL / Collision | How to move safely? |
| **53** | **Grasping** | GraspNet / PointNet | Where to pinch? |
| **54** | **Force** | Admittance Control | Touching gently |
| **55** | **Servoing** | IBVS / Interaction Matrix | Precision alignment |

### The "Pick" Paradox
*   **Open Loop:** Plan IK $\to$ Execute. Fast but fails if object moves 1mm.
*   **Closed Loop:** Visual Servoing. Slow but precise.
*   **Hybrid:** Plan Approach (MoveIt) $\to$ Servo Terminal (IBVS) $\to$ Grasp (Force).

---

## 🚀 Weekly Capstone: "The Kitchen Assistant"

**Scenario:** A Tiago Robot.
**Task:**
1.  **Nav:** Drive to Kitchen Counter.
2.  **Perceive:** Find "Mustard Bottle".
3.  **Approach:** Move Arm to pre-grasp.
4.  **Refine:** Servo to align gripper.
5.  **Pick:** Lift object.
6.  **Place:** Put on Dining Table (Detect contact).

### 🛠️ Project Structure
```text
week8_capstone/
├── src/
│   ├── behavior_tree.py
│   ├── arm_controller.py
│   ├── perception_server.py
│   └── mobile_base.py
└── run_demo.py
```

### 👨‍💻 Component 1: The Behavior Tree (`src/behavior_tree.py`)

Using `py_trees` (ROS 2 standard).

```python
import py_trees

def create_tree():
    # Root
    root = py_trees.composites.Sequence("FetchTask")
    
    # 1. Navigation
    nav = Action("NavigateToTable")
    
    # 2. Perception
    detect = Action("DetectObject")
    
    # 3. Manipulation
    manip = py_trees.composites.Selector("PickPipeline")
    
    # Try Simple Pick
    plan_execution = py_trees.composites.Sequence("PlanAndPick")
    plan = Action("MoveItPlan")
    execute = Action("ServoAndGrasp")
    plan_execution.add_children([plan, execute])
    
    # Fallback: Ask Human
    ask = Action("AskForHelp")
    
    manip.add_children([plan_execution, ask])
    
    root.add_children([nav, detect, manip])
    return root
```

### 👨‍💻 Component 2: Arm Controller (`src/arm_controller.py`)

Fuses MoveIt and Servoing.

```python
class ArmInterface:
    def pick_object(self, object_pose):
        # 1. Pre-Grasp (10cm away)
        pre_pose = offset(object_pose, -0.1)
        self.move_group.set_pose_target(pre_pose)
        success = self.move_group.go()
        if not success: return "PLAN_FAILED"
        
        # 2. Switch to Servo Mode (Velocity Control)
        self.switch_controller("servo_controller")
        
        # 3. Servo Loop (Approach)
        while dist > 0.02:
            err = self.get_visual_error()
            v_cmd = self.compute_ibvs(err)
            v_cmd.linear.z = 0.05 # Forward crawl
            self.pub_vel.publish(v_cmd)
            
        # 4. Force Guarded Move (Touch)
        while force_z < 2.0:
             self.pub_vel.publish(crawl_forward)
             
        # 5. Grasp
        self.gripper.close()
        
        return "SUCCESS"
```

### 👨‍💻 Component 3: Mobile Base Integration (`src/mobile_base.py`)

Handles "Base Placement Reasoning".
*   If object is out of reach, move the base?
*   Yes. `InverseReachabilityMap`.

```python
class MobileBase:
    def position_for_grasp(self, object_pose):
        # We need the object to be in the "Dexterous Workspace" of the arm.
        # Ideally 0.5m in front, 0.8m high.
        
        target_base_pose = object_pose
        target_base_pose.x -= 0.5 
        
        self.nav.navigate_to(target_base_pose)
```

---

## 📝 Self-Assessment Quiz

1.  **Singularities:**
    *   Why avoid singularities?
        *   **A:** Infinite velocities required. Robot usually E-Stops. MoveIt planners try to avoid them, but Servoing might drift into them.
        *   **Fix:** Manipulability Index maximization.

2.  **Whole Body Control:**
    *   Why move Base and Arm separately? Why not together?
        *   **A:** "Mobile Manipulation" (Coupled) is harder. Base has low accuracy/high jitter. Arm has high accuracy. Usually better to Park Base $\to$ Move Arm.

3.  **Visual Servoing:**
    *   Why did we switch controllers?
        *   **A:** MoveIt uses `FollowJointTrajectory`. Servoing uses `JointGroupVelocity`. You can't run two controllers on same joints simultaneously (Resource Conflict).

---

## ⏭️ Look Ahead: Week 9
We have covered the Robot. Now the Environment.
**Week 9: Simulation & Sim-to-Real.**
*   Gazebo / Isaac Sim.
*   URDF / SDF Modeling.
*   Physics Engines.
*   Domain Randomization.

---

**Week 8 Complete**
