# Day 141: Cobot Safety Standards (ISO 10218/15066)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> A robot shall not harm a human being.
> - **Focus:** Risk Assessment, ISO 10218-1/2, ISO/TS 15066, Power and Force Limiting (PFL), Speed and Separation Monitoring (SSM).
> - **Code:** A Python-based `SafetyMonitor` class that checks robot state (Velocity, Torque, Momentum) against ISO limits and triggers a Category 0 Stop if violated.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Interpret** the ISO 15066 standard for biomechanical limits (How hard can a robot punch?).
2.  **Implement** the 4 Collaborative Modes: Stop, Hand Guiding, SSM, PFL.
3.  **Compute** the Kinetic Energy limit ($E_k < 0.5 mv^2$) for safe transient contact.
4.  **Design** a Safety PLC logic simulator.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). Real Cobots (UR, Franka) have this built-in.

### Software Environment
```bash
pip install numpy
```

### Prior Knowledge
- PID Control (Day 26).
- Rigid Body Dynamics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Standards

*   **ISO 10218:** General Industrial Robot Safety. Mandates "Safety Rated" stops.
*   **ISO/TS 15066:** Specific to Collaborative Robots. Defines *pain thresholds*.
    *   **Quasi-Static Contact:** Pinching (Clamping). High risk. Limit: ~150N (Hands).
    *   **Transient Contact:** Bumping. Low risk. Limit: 2x Force, provided duration < 0.5s.

### 🔹 Part 2: The 4 Modes of Collaboration

1.  **Safety-Rated Monitored Stop (SRMS):** Robot stops when human enters zone. Resumes when human leaves.
2.  **Hand Guiding:** Robot moves only when operator holds the handle.
3.  **Speed and Separation Monitoring (SSM):** Robot slows down as human gets closer.
4.  **Power and Force Limiting (PFL):** Robot can hit you, but it won't hurt. (The "True" Cobot).

### 🔹 Part 3: Stopping Time Mechanics

$$ D_{stop} = v \cdot t_{reaction} + \frac{v^2}{2a_{max}} + D_{uncertainty} $$
*   If you detect a human at distance $X$, you must prove you can stop before touching them (or hit them softly).

---

## 💻 Implementation: The Safety Supervisor

We simulate a robot arm moving, and a Safety Monitor watching it.

### 🛠️ Project Structure
```text
day141_safety/
├── src/
│   ├── safety_monitor.py
└── output/
    ├── stop_trace.png
```

### 👨‍💻 Safety Logic (`src/safety_monitor.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class RobotArm:
    def __init__(self):
        self.mass = 10.0 # kg (Effective mass at End Effector)
        self.velocity = 0.0 # m/s
        self.position = 0.0 # m
        self.force = 0.0 # N
        self.state = "RUNNING" # RUNNING, STOPPING, STOPPED
        
    def step(self, cmd_vel, dt):
        if self.state == "STOPPED":
            self.velocity = 0.0
            self.force = 0.0
            return
            
        if self.state == "STOPPING":
            # Decelerate max brake
            brake_acc = -5.0 # m/s^2
            if self.velocity > 0:
                self.velocity += brake_acc * dt
                if self.velocity < 0: self.velocity = 0
            else:
                self.velocity = 0
                self.state = "STOPPED"
        else:
            # Normal operation
            # Simple dynamics: Vel follows Cmd roughly
            acc = (cmd_vel - self.velocity) * 2.0
            self.velocity += acc * dt
            
        self.position += self.velocity * dt
        # Kinetic Energy = 0.5 * m * v^2
        pass

class SafetyMonitor:
    def __init__(self, robot):
        self.robot = robot
        
        # ISO 15066 Limits (Simplified)
        self.max_force = 150.0 # N
        self.max_energy = 10.0 # Joules (Transient contact)
        self.min_separation = 0.5 # m (for SSM)
        
        # Human
        self.human_pos = 2.0 # m
        
    def check(self):
        # 1. PFL Check (Energy)
        energy = 0.5 * self.robot.mass * (self.robot.velocity**2)
        
        if energy > self.max_energy:
            print(f"SAFETY VIOLATION: Kinetic Energy {energy:.2f}J > {self.max_energy}J")
            return "CAT_0_STOP"
            
        # 2. SSM Check (Separation)
        dist = self.human_pos - self.robot.position
        
        if dist < self.min_separation:
            # Calculate required stop distance
            # D_stop = v^2 / 2a + reaction_dist...
            # For this sim, just trigger stop
            print(f"SAFETY WARNING: Human too close ({dist:.2f}m)")
            return "REDUCE_SPEED"
            
        return "OK"

def main():
    robot = RobotArm()
    monitor = SafetyMonitor(robot)
    
    dt = 0.01
    times = []
    vels = []
    states = []
    
    print("Starting Safety Simulation...")
    
    for t in np.arange(0, 5.0, dt):
        # Robot tries to accelerate to 2.0 m/s
        cmd = 2.0
        
        # Monitor
        status = monitor.check()
        
        if status == "CAT_0_STOP":
            robot.state = "STOPPING"
        elif status == "REDUCE_SPEED":
            cmd = 0.25 # Safe slow speed
            
        # Physics
        robot.step(cmd, dt)
        
        # Human walks closer at t=2.0
        if t > 2.0:
            monitor.human_pos -= 0.5 * dt # Walking towards robot
            
        times.append(t)
        vels.append(robot.velocity)
        states.append(1 if robot.state == "RUNNING" else 0)
        
    # Plot
    fig, ax1 = plt.subplots()
    
    ax1.plot(times, vels, 'b-', label='Velocity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(times, states, 'r--', label='State (1=Run, 0=Stop)')
    ax2.set_ylabel('Safety State', color='r')
    
    plt.title("ISO 15066 Safety Response")
    plt.savefig("output/stop_trace.png")
    print("Sim Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Collision Test"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Robot accelerates. At $t>2$, human approaches. Monitor triggers `REDUCE_SPEED` (SSM). Robot slows to 0.25m/s.
- **Modify:** Set `human_pos` to jump instantly to `robot.position` (Teleport/Blindspot).
- **Result:** Robot stops, but might impact if Velocity was high.
- **Calculate:** If Robot $V=1m/s$, $M=10kg$, Impact Energy = 5J. Is it safe? Yes ($<10J$).
- **Calculate:** If Robot $V=2m/s$, $E=20J$. Not safe.

---

## 🚀 Project: "Safety Bubble Visualization"

**Goal:** Visualize dynamic safety zones in Rviz.
1.  **Inputs:** Robot Joint velocities.
2.  **Compute:** Impact potential for every link.
3.  **Draw:** Red spheres around the robot. Radius $\propto$ Velocity.
4.  **Lidar:** Point cloud of Human.
5.  **Logic:** If Cloud intersects Sphere $\to$ STOP.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "False Positives"
*   **Cause:** Noise in velocity estimate. Derivative of position is noisy.
*   **Result:** Random E-Stops.
*   **Fix:** Low-pass filter velocity signal.

#### 2. "Stopping Distance is too long"
*   **Cause:** Heavy Tools.
*   **Fix:** Reduce max speed or reduce tool weight.

---

## ⚡ Optimization: Dual Channel Safety

Real safety systems use redundancy.
*   **Channel A:** Calculates limit.
*   **Channel B:** Calculates limit independently.
*   **Logic:** If A != B $\to$ STOP.
*   **Hardware:** Safety PLC (yellow box) separate from Robot Controller.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between "Protective Stop" and "Emergency Stop"?
    *   **A:** Protective is programmatically triggered (e.g., Lidar field violation), robot can resume auto. Emergency is Button Press, cuts power, requires manual reset.
2.  **Q:** What is PFL?
    *   **A:** Power and Force Limiting. The robot hardware is designed (springs/sensors) to feel collisions and yield.
3.  **Q:** Can a robot be 100% safe?
    *   **A:** No. If you hold a knife and run into a stationary robot, you get hurt. Safety is "Acceptable Risk".

### Challenge Task
> **Task:** Payload Estimation.
> 1. Robot picks up object. Mass changes $10kg \to 15kg$.
> 2. Stopping distance increases.
> 3. Update the Safety Monitor to account for variable mass.

---

## 📚 Further Reading
- **Universal Robots:** "Safety Functions Guide".
- **ISO 15066:** Official specification (Paid, but summaries exist).

---

**Day 141 Complete**
