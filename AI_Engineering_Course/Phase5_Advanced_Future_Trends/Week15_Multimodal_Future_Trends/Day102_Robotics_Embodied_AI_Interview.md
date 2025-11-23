# Day 102: Robotics & Embodied AI
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is "Data" the bottleneck in Robotics?

**Answer:**
*   **Text:** We have the whole internet.
*   **Robot Data:** We only have data from specific labs. Robots break. Teleoperation is slow.
*   **Solution:** Open X-Embodiment Dataset (aggregating data from 20+ labs).

#### Q2: What is the "Reality Gap"?

**Answer:**
The difference between Simulation and Reality.
*   Sim physics is perfect. Real physics has dust, loose wires, and wear-and-tear.
*   A policy trained in Sim often fails in Real.

#### Q3: How do you ensure Safety in LLM-controlled robots?

**Answer:**
*   **Constrained Decoding:** Ensure the output tokens are valid actions.
*   **Safety Layer:** A separate, non-AI collision avoidance system (Lidar) that overrides the LLM if it gets too close to a human.

#### Q4: Explain "Affordance".

**Answer:**
What actions are possible on an object?
*   Cup -> Graspable, Pourable.
*   Button -> Pushable.
*   VLA models learn affordances from large-scale video data.

### Production Challenges

#### Challenge 1: Latency

**Scenario:** Robot moves at 1 m/s. LLM takes 1s to think. Robot crashes.
**Root Cause:** Inference speed.
**Solution:**
*   **High-Frequency Control:** Run a low-level controller at 100Hz (PID) and the high-level LLM planner at 1Hz.
*   **Distillation:** Distill the VLA into a smaller model.

#### Challenge 2: Hardware Variability

**Scenario:** Model trained on Franka arm. You have a UR5 arm.
**Root Cause:** Embodiment mismatch.
**Solution:**
*   **Cross-Embodiment Training:** Train on data from many different robots so the model learns "Generalized Manipulation" rather than specific joint angles.

#### Challenge 3: Long-Horizon Planning

**Scenario:** "Clean the kitchen."
**Root Cause:** Thousands of steps. Error accumulation.
**Solution:**
*   **Hierarchical Planning:**
    *   Planner: "1. Pick up trash. 2. Wipe table."
    *   Policy: Executes "Pick up trash".

### System Design Scenario: Warehouse Sorting Robot

**Requirement:** Sort packages by color.
**Design:**
1.  **Vision:** Camera overhead.
2.  **VLM:** Detect package color and coordinates.
3.  **Planner:** Assign packages to bins.
4.  **Motion Planning:** RRT* (Rapidly-exploring Random Tree) to avoid collisions.
5.  **Grasp:** Suction gripper.

### Summary Checklist for Production
*   [ ] **E-Stop:** Physical Emergency Stop button is mandatory.
*   [ ] **Watchdog:** Timer that halts the robot if the model hangs.
