# Day 32: Control Theory (PID, MPC)
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 32 Focus:**
> The Planner says "Go here at 50mph." The Controller's job is to make the physics obey. It calculates the exact steering angle and throttle voltage to minimize the error between the *Reference* (Plan) and the *State* (Reality). Today, we master the workhorse **PID** and the modern standard **MPC**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Implement** a PID Controller for Longitudinal Control (Cruise Control).
2.  **Tune** P, I, and D gains to minimize overshoot and settling time.
3.  **Explain** the Model Predictive Control (MPC) formulation: Cost Function, Constraints, and Horizon.
4.  **Implement** a Linear MPC for Lateral Control (Lane Keeping) using Python.
5.  **Contrast** PID (Reactive) vs MPC (Predictive).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Derivatives and Integrals.
-   **Optimization:** Minimizing a quadratic cost function.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `matplotlib`, `cvxpy` (for MPC optimization).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: PID Control

The simplest feedback controller.
Error $e(t) = \text{Target} - \text{Actual}$.

$$ u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt} $$

-   **P (Proportional):** "Steer harder if error is big." (Present).
-   **I (Integral):** "Steer harder if error persists." (Past). Fixes steady-state error (e.g., misaligned steering wheel).
-   **D (Derivative):** "Steer less if error is decreasing fast." (Future). Dampens overshoot.

### 🔹 Part 2: Model Predictive Control (MPC)

PID is reactive. It doesn't know the road curves ahead.
MPC looks into the future.

1.  **Model:** Use a kinematic model to predict the car's state for the next $N$ steps.
    $$ x_{t+1} = x_t + v_t \cos(\theta_t) dt $$
2.  **Cost Function:** Define what "Good" means.
    $$ J = \sum_{t=0}^{N} (e_{cte}^2 + e_{psi}^2 + v_{error}^2) + \sum (u_{steer}^2) + \sum (\Delta u_{steer}^2) $$
    -   Minimize Cross Track Error ($e_{cte}$).
    -   Minimize Heading Error ($e_{psi}$).
    -   Minimize Actuation (Don't turn wheel wildly).
    -   Minimize Jerk (Don't change steering too fast).
3.  **Constraints:**
    -   Steering angle $\in [-30^\circ, 30^\circ]$.
    -   Throttle $\in [0, 1]$.
4.  **Optimization:** Solve for the sequence of inputs $u_0, u_1, \dots, u_N$ that minimizes $J$.
5.  **Receding Horizon:** Apply $u_0$. Discard the rest. Repeat at next step.

---

## 💻 Implementation: PID vs MPC

We will implement two scripts: `pid_cruise.py` and `mpc_steer.py`.

### 🛠️ Setup
Create `week5_day32`.

```bash
mkdir -p ~/ros2_ws/src/week5_day32
cd ~/ros2_ws/src/week5_day32
pip install cvxpy
touch pid_cruise.py mpc_steer.py
```

### 👨‍💻 Code: PID Cruise Control

```python
import numpy as np
import matplotlib.pyplot as plt

class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

def run_pid():
    # Simulation Params
    dt = 0.1
    t_end = 20.0
    target_speed = 30.0 # m/s
    
    # Car Params
    mass = 1000.0
    drag_coeff = 0.1
    
    # Controller
    pid = PID(Kp=50.0, Ki=2.0, Kd=10.0, dt=dt)
    
    # State
    speed = 0.0
    speeds = []
    targets = []
    times = np.arange(0, t_end, dt)
    
    for t in times:
        # 1. Calculate Error
        error = target_speed - speed
        
        # 2. Get Control (Force)
        force = pid.update(error)
        
        # 3. Apply Physics (F = ma)
        # Drag force
        drag = drag_coeff * speed**2
        accel = (force - drag) / mass
        
        speed += accel * dt
        
        speeds.append(speed)
        targets.append(target_speed)
        
    plt.plot(times, speeds, label='Speed')
    plt.plot(times, targets, 'r--', label='Target')
    plt.title("PID Cruise Control")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_pid()
```

### 👨‍💻 Code: Linear MPC for Path Tracking

We use `cvxpy` to solve a simplified Linear MPC problem.
State: $[y, v_y, \psi, r]$ (Lateral pos, Lat vel, Heading, Yaw rate).
Input: $\delta$ (Steering angle).

*Note: For simplicity, we assume constant longitudinal speed and linear tire model.*

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def run_mpc():
    # Horizon
    T = 10
    dt = 0.1
    
    # Model (Bicycle Model Linearized)
    # x(k+1) = A x(k) + B u(k)
    # State: [y, psi] (Simplified: Lateral Error, Heading Error)
    # Input: [steer]
    v = 10.0 # m/s constant
    L = 2.5 # Wheelbase
    
    # Kinematic Model:
    # y_dot = v * sin(psi) ~= v * psi
    # psi_dot = v/L * tan(delta) ~= v/L * delta
    
    A = np.array([[1.0, v*dt],
                  [0.0, 1.0]])
    B = np.array([[0.0],
                  [v/L*dt]])
                  
    nx = 2
    nu = 1
    
    # Variables
    x = cp.Variable((nx, T+1))
    u = cp.Variable((nu, T))
    
    # Initial State (Error)
    x_init = np.array([1.0, 0.2]) # 1m off center, 0.2 rad heading error
    
    # Cost Matrices
    Q = np.diag([10.0, 1.0]) # Penalize y heavily
    R = np.diag([1.0])       # Penalize steering effort
    
    cost = 0
    constraints = [x[:, 0] == x_init]
    
    for t in range(T):
        cost += cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R)
        constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
        constraints += [cp.abs(u[:, t]) <= 0.5] # Max steer 0.5 rad
        
    # Terminal Cost
    cost += cp.quad_form(x[:, T], Q)
    
    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    # Visualization
    time = np.arange(0, (T+1)*dt, dt)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, x.value[0, :], label='Lateral Error (y)')
    plt.plot(time, x.value[1, :], label='Heading Error (psi)')
    plt.grid()
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.step(time[:-1], u.value[0, :], label='Steering (rad)')
    plt.grid()
    plt.legend()
    
    plt.suptitle("Linear MPC Path Tracking")
    plt.show()

if __name__ == "__main__":
    run_mpc()
```

---

## 🔬 Lab Exercise: Tuning

### Lab Objectives
1.  **PID Tuning:**
    -   Set $K_p=10, K_i=0, K_d=0$. Observe oscillation.
    -   Increase $K_d$. Observe damping.
    -   Add $K_i$. Observe zero steady-state error.
2.  **MPC Tuning:**
    -   Increase $Q[0,0]$ (Lateral Error weight).
    -   *Result:* Car steers aggressively to return to center.
    -   Increase $R$ (Steering weight).
    -   *Result:* Car steers gently, accepting some error to save tires/comfort.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Integral Windup (PID)
**Symptom:** Huge overshoot after a long period of error (e.g., going up a hill then cresting).
**Cause:** The Integral term accumulated a massive value.
**Solution:** Clamp the integral term (`self.integral = max(min(self.integral, limit), -limit)`).

#### 2. Infeasible Problem (MPC)
**Symptom:** Solver returns `None` or `Infeasible`.
**Cause:** Constraints are impossible to satisfy (e.g., "Reach 100m in 1s" with "Max Speed 1m/s").
**Solution:** Soften constraints (use slack variables) or check initial conditions.

#### 3. Instability
**Symptom:** Car oscillates wildly.
**Cause:** Latency. The controller calculates $u_t$ based on $x_t$, but applies it at $t+\delta$.
**Solution:** Model the latency in the MPC prediction ($x_{t+1}$ depends on $u_{t-1}$).

---

## ⚡ Optimization & Best Practices

### 1. Feedforward Control
Don't rely only on feedback.
If you know the road curves, **Feedforward** the steering angle needed for that curvature ($\delta_{ff} = \arctan(L \kappa)$).
PID/MPC then only corrects the *residual* error.

### 2. Nonlinear MPC (NMPC)
Linear MPC works for highway driving (small angles).
For parking or racing (drifting), use **NMPC** with the full kinematic/dynamic model.
-   Solvers: Ipopt, ACADO.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the role of the 'D' term in PID?
    *   **A:** To predict future error based on current rate of change, providing damping and reducing overshoot.
2.  **Q:** Why is MPC better than PID for autonomous driving?
    *   **A:** MPC handles constraints (Lane boundaries, Actuator limits) explicitly and plans for the future (Curve ahead).
3.  **Q:** What is the "Horizon" in MPC?
    *   **A:** How far into the future the controller predicts (e.g., 2 seconds). Too short = unstable. Too long = computationally expensive.

### Challenge Task
**Task:** Stanley Controller.
1.  Implement the **Stanley Controller** (Geometric path tracking).
    $$ \delta(t) = \psi(t) + \arctan \left( \frac{k e(t)}{v(t)} \right) $$
2.  Compare it with PID for lane keeping.

---

## 📚 Further Reading & References
-   [Control of Mobile Robots (Coursera)](https://www.coursera.org/learn/mobile-robot)
-   [Model Predictive Control for Autonomous Vehicles (Falcone)](https://ieeexplore.ieee.org/document/4162483)

---

**Day 32 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
