# Day 166: Control & Maneuvering
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 166 Focus:**
> The Planner gave us a path with zig-zags and gear shifts. Now the **Controller** must execute it. Highway PID won't work here. We need **MPC** to handle the kinematic constraints and stop exactly 5cm from the wall.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Implement** a Low-Speed MPC (Model Predictive Control).
2.  **Handle** Gear Shifting (Drive $\leftrightarrow$ Reverse) logic.
3.  **Achieve** high stopping accuracy (< 5cm).
4.  **Tune** MPC weights for smooth steering at low speeds.
5.  **Integrate** the Controller with the Hybrid A* Planner.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 137:** MPC.
-   **Day 164:** Parking Planner.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `cvxpy` (Optimization solver).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Parking vs Highway Control

-   **Highway:** High speed, small steering angles. Linear models work well.
-   **Parking:** Low speed, large steering angles (lock-to-lock). Non-linear kinematics dominate.
-   **Singularity:** At $v=0$, steering has no effect on position. The controller must handle stopping and starting smoothly.

### 🔹 Part 2: Gear Management

-   The path from Hybrid A* contains "Cusps" (Reversals).
-   **Logic:**
    1.  Track path until Cusp.
    2.  Stop ($v=0$).
    3.  Hold Brake.
    4.  Shift Gear ($D \to R$).
    5.  Release Brake.
    6.  Track next segment.

### 🔹 Part 3: Stopping Accuracy

-   **Challenge:** Actuator lag and friction.
-   **Solution:**
    -   **Feedforward:** Predict friction torque.
    -   **Final Approach:** Switch to "Distance Control" instead of "Velocity Control" for the last 50cm.

---

## 💻 Implementation: Parking MPC

**Scenario:**
-   Track a reference path that includes a reversal.
-   State: $[x, y, \theta, v]$.
-   Control: $[a, \delta]$ (Accel, Steer).

### 🛠️ Setup
Create `week24_capstone/control`.

### 👨‍💻 Code: MPC with Reversal Handling

```python
import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt

class ParkingMPC:
    def __init__(self):
        self.T = 10 # Horizon
        self.dt = 0.1
        self.L = 2.5 # Wheelbase
        
    def solve(self, state, ref_traj):
        # state: [x, y, theta, v]
        # ref_traj: List of [x, y, theta, v] for next T steps
        
        # Variables
        x = cp.Variable((4, self.T + 1))
        u = cp.Variable((2, self.T)) # [accel, steer]
        
        cost = 0
        constraints = []
        
        # Initial State
        constraints += [x[:, 0] == state]
        
        for t in range(self.T):
            # Cost
            cost += cp.sum_squares(x[:, t+1] - ref_traj[t]) * 10.0 # Tracking
            cost += cp.sum_squares(u[:, t]) * 1.0 # Effort
            cost += cp.sum_squares(u[:, t] - (u[:, t-1] if t>0 else 0)) * 10.0 # Smoothness
            
            # Model (Linearized Bicycle)
            # x_next = x + v*cos(theta)*dt
            # Linearized around ref_traj[t]
            xr = ref_traj[t]
            vr = xr[3]
            thr = xr[2]
            
            # A, B matrices (Jacobians)
            # Simplified for demo (Ideally compute properly)
            # x_next = Ax + Bu
            constraints += [x[0, t+1] == x[0, t] + vr * np.cos(thr) * self.dt] # Very rough approx
            constraints += [x[1, t+1] == x[1, t] + vr * np.sin(thr) * self.dt]
            constraints += [x[2, t+1] == x[2, t] + (vr / self.L) * u[1, t] * self.dt]
            constraints += [x[3, t+1] == x[3, t] + u[0, t] * self.dt]
            
            # Constraints
            constraints += [u[0, t] <= 2.0, u[0, t] >= -2.0] # Accel limits
            constraints += [u[1, t] <= 0.5, u[1, t] >= -0.5] # Steer limits
            
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            return u[:, 0].value
        else:
            print("MPC Fail")
            return [0.0, 0.0]

def main():
    mpc = ParkingMPC()
    
    # Reference Path: Drive straight, Stop, Reverse
    # 0-2s: v=2
    # 2-3s: v=0 (Stop)
    # 3-5s: v=-2 (Reverse)
    ref_traj = []
    curr_x = 0
    for i in range(50):
        if i < 20: v = 2.0
        elif i < 30: v = 0.0
        else: v = -2.0
        curr_x += v * 0.1
        ref_traj.append([curr_x, 0, 0, v])
        
    # Simulate
    state = np.array([0.0, 0.0, 0.0, 0.0])
    history_x = []
    history_v = []
    
    for i in range(40): # Run for 4s
        # Get Horizon
        horizon = []
        for t in range(10):
            idx = min(i + t, len(ref_traj)-1)
            horizon.append(ref_traj[idx])
            
        # Solve
        ctrl = mpc.solve(state, horizon)
        
        # Update Plant
        acc, steer = ctrl
        state[3] += acc * 0.1
        state[2] += (state[3] / 2.5) * np.tan(steer) * 0.1
        state[0] += state[3] * np.cos(state[2]) * 0.1
        state[1] += state[3] * np.sin(state[2]) * 0.1
        
        history_x.append(state[0])
        history_v.append(state[3])
        
    # Plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history_x, label='Actual X')
    plt.plot([p[0] for p in ref_traj[:40]], '--', label='Ref X')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history_v, label='Actual V')
    plt.plot([p[3] for p in ref_traj[:40]], '--', label='Ref V')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Gear Shift

### Lab Objectives
1.  **Run the script.**
    -   **Observation:** The car accelerates, stops, and reverses.
2.  **Analyze the Stop:**
    -   Look at the velocity plot around $t=2s$.
    -   Does it reach exactly 0?
    -   If not, the gear shift logic (in a real car) would fail (Grinding gears).
3.  **Implement Logic:**
    -   Wrap the MPC in a state machine.
    -   `if abs(v) < 0.01 and target_v < 0: shift_gear(REVERSE)`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Oscillation at Stop
**Symptom:** Car jitters back and forth at the goal.
**Cause:** MPC tries to correct small position errors but overshoots due to min throttle.
**Solution:** **Deadband**. If `dist_to_goal < 5cm`, set `cmd = 0` and apply Handbrake.

#### 2. Steering while Stopped
**Symptom:** Wheels turn while $v=0$.
**Cause:** Valid in simulation, bad for tires in reality (Dry Steering).
**Solution:** Add constraint: `delta_steer <= k * abs(v)`. Only allow steering when moving.

---

## ⚡ Optimization & Best Practices

### 1. Time-Varying Linearization
-   Linearize the bicycle model at *each step* of the horizon based on the reference trajectory.
-   Essential for accurate reversing control.

### 2. Soft Constraints
-   Make the "Stop Line" a hard constraint? No, solver might fail.
-   Make it a very high cost Soft Constraint.
-   Always ensure the solver finds *some* solution, even if imperfect.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is "Dry Steering" bad?
    *   **A:** It wears out the tires and puts high load on the steering motor.
2.  **Q:** How does MPC handle the gear shift delay?
    *   **A:** It doesn't know about it unless modeled. The State Machine must handle the delay (wait 0.5s) before re-engaging the controller.
3.  **Q:** What is the "Singularity" in the bicycle model?
    *   **A:** At low speeds, the kinematic equations divide by velocity or become ill-conditioned.

### Challenge Task
**Task:** Parallel Parking Controller.
1.  Use the path from Day 164 (Parallel Park).
2.  Tune MPC to track the S-curve.
3.  Ensure the final heading error is $< 1^\circ$.

---

## 📚 Further Reading & References
-   [MPC for Autonomous Parking](https://ieeexplore.ieee.org/document/8317726)
-   [CVXPY Documentation](https://www.cvxpy.org/)

---

**Day 166 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
