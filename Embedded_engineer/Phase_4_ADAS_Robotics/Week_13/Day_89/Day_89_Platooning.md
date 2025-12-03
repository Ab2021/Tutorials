# Day 89: Platooning (CACC)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 89 Focus:**
> Trucks driving close together save fuel (aerodynamic drag). But human drivers can't react fast enough to drive safely at 5-meter gaps. **Platooning** uses V2V to synchronize braking instantly. This is **CACC (Cooperative Adaptive Cruise Control)**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between ACC (Radar-based) and CACC (V2V-based).
2.  **Define** String Stability (Does the oscillation grow or die out?).
3.  **Derive** the CACC Control Law (Feedforward term from Leader).
4.  **Simulate** a 3-truck platoon reacting to emergency braking.
5.  **Implement** Platoon Logic: Join, Follow, Split.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Control Theory:** PID, Feedforward.
-   **Physics:** Drag force $F_d = \frac{1}{2} \rho v^2 C_d A$.
-   **Day 85:** V2V Latency.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ACC vs CACC

-   **ACC (Adaptive Cruise Control):**
    -   Sensor: Radar.
    -   Reaction: Reacts to *distance error*.
    -   Delay: Radar processing + Actuation (~0.5s).
    -   Result: Needs large gaps (2s headway) to be stable.
-   **CACC (Cooperative ACC):**
    -   Sensor: Radar + V2V (BSM).
    -   Reaction: Reacts to *Leader's Acceleration*.
    -   Delay: V2V (~0.02s).
    -   Result: Can drive with 0.5s headway safely.

### 🔹 Part 2: String Stability

Imagine a chain of cars. Car 1 brakes slightly.
-   **Unstable:** Car 2 brakes harder. Car 3 brakes even harder. Car 10 crashes. (Ghost Traffic Jam).
-   **Stable:** The braking disturbance *attenuates* as it propagates down the string.
-   **Condition:** $|H(j\omega)| \le 1$ where $H$ is the transfer function $\frac{a_i}{a_{i-1}}$.

### 🔹 Part 3: The CACC Control Law

$$ u_i = k_p (d_{error}) + k_d (v_{error}) + u_{feedforward} $$
-   $d_{error} = d_{actual} - d_{desired}$.
-   $v_{error} = v_{leader} - v_{ego}$.
-   $u_{feedforward} = a_{leader}$ (Received via V2V).
-   The feedforward term is the magic. We brake *before* the gap closes.

---

## 💻 Implementation: Platoon Simulator

**Scenario:**
-   **Leader (Truck 0):** Profiles a velocity (Accelerate -> Constant -> Brake).
-   **Follower 1 (Truck 1):** Uses CACC to follow Truck 0.
-   **Follower 2 (Truck 2):** Uses CACC to follow Truck 1.
-   **Comparison:** We will toggle Feedforward (V2V) to see the difference between ACC and CACC.

### 🛠️ Setup
Create `week13_day89` and `platoon_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day89
cd ~/ros2_ws/src/week13_day89
touch platoon_sim.py
```

### 👨‍💻 Code: CACC Controller

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
DT = 0.01
KP = 0.5 # Distance Gain
KD = 1.0 # Velocity Gain
HEADWAY = 0.5 # Seconds (Aggressive!)
STANDSTILL_DIST = 5.0 # Meters

class Truck:
    def __init__(self, id, x_start, use_v2v=True):
        self.id = id
        self.x = x_start
        self.v = 0.0
        self.a = 0.0
        self.use_v2v = use_v2v
        
        # History
        self.x_hist = []
        self.v_hist = []
        self.a_hist = []
        
    def update_physics(self, u_cmd):
        # Actuator Lag (First order lag)
        tau = 0.1
        self.a += (u_cmd - self.a) * (DT / tau)
        
        # Kinematics
        self.v += self.a * DT
        self.x += self.v * DT
        
        # Log
        self.x_hist.append(self.x)
        self.v_hist.append(self.v)
        self.a_hist.append(self.a)

    def control(self, leader):
        # Desired Distance
        d_des = STANDSTILL_DIST + HEADWAY * self.v
        d_act = leader.x - self.x
        
        # Errors
        e_dist = d_act - d_des
        e_vel = leader.v - self.v
        
        # PD Control (ACC)
        u = KP * e_dist + KD * e_vel
        
        # Feedforward (CACC)
        if self.use_v2v:
            # We receive leader's acceleration instantly (simulated)
            u += leader.a
            
        return u

def main():
    # Setup Platoon
    # Truck 0 (Leader) at 20m
    # Truck 1 at 10m
    # Truck 2 at 0m
    
    USE_V2V = True # Toggle this to see ACC vs CACC
    
    trucks = [
        Truck(0, 40.0, use_v2v=False), # Leader (Open Loop)
        Truck(1, 20.0, use_v2v=USE_V2V),
        Truck(2, 0.0, use_v2v=USE_V2V)
    ]
    
    time_steps = np.arange(0, 30, DT)
    
    print(f"Simulating Platoon (V2V={'ON' if USE_V2V else 'OFF'})...")
    
    for t in time_steps:
        # 1. Leader Profile
        u_leader = 0.0
        if 1.0 < t < 5.0: u_leader = 2.0 # Accel
        elif 15.0 < t < 20.0: u_leader = -4.0 # Hard Brake
        
        trucks[0].update_physics(u_leader)
        
        # 2. Followers
        for i in range(1, len(trucks)):
            leader = trucks[i-1]
            u_cmd = trucks[i].control(leader)
            trucks[i].update_physics(u_cmd)
            
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    # Position
    for trk in trucks:
        axs[0].plot(time_steps, trk.x_hist, label=f'Truck {trk.id}')
    axs[0].set_title('Position')
    axs[0].set_ylabel('m')
    axs[0].legend()
    axs[0].grid()
    
    # Velocity
    for trk in trucks:
        axs[1].plot(time_steps, trk.v_hist, label=f'Truck {trk.id}')
    axs[1].set_title('Velocity')
    axs[1].set_ylabel('m/s')
    axs[1].grid()
    
    # Acceleration (Check Stability)
    for trk in trucks:
        axs[2].plot(time_steps, trk.a_hist, label=f'Truck {trk.id}')
    axs[2].set_title('Acceleration (Check Overshoot)')
    axs[2].set_ylabel('m/s^2')
    axs[2].grid()
    
    # Calculate Min Gap
    min_gap_1 = np.min(np.array(trucks[0].x_hist) - np.array(trucks[1].x_hist))
    min_gap_2 = np.min(np.array(trucks[1].x_hist) - np.array(trucks[2].x_hist))
    print(f"Min Gap (0-1): {min_gap_1:.2f}m")
    print(f"Min Gap (1-2): {min_gap_2:.2f}m")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Crash Test

### Lab Objectives
1.  Run with `USE_V2V = True`.
    -   **Observation:** When Leader brakes (-4 m/s²), Followers brake almost instantly.
    -   **Result:** Gaps remain stable. No crash.
2.  Run with `USE_V2V = False` (Standard ACC).
    -   **Observation:**
        -   Leader brakes.
        -   Follower 1 waits for gap to close (Delay). Then brakes hard.
        -   Follower 2 waits for Follower 1. Then brakes **even harder** (Overshoot).
    -   **Result:** String Instability. The acceleration peaks grow downstream. Potential crash (Gap < 0).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Packet Loss
**Symptom:** V2V feedforward is missing for 0.5s.
**Cause:** Radio interference.
**Solution:** Fallback to ACC. If V2V is lost, increase the Headway immediately (e.g., 0.5s -> 1.5s) to maintain safety.

#### 2. Actuator Lag
**Symptom:** Oscillations even with V2V.
**Cause:** If the truck's brakes take 0.5s to engage (`tau`), knowing about the braking 0.5s early just barely cancels the lag.
**Solution:** Better low-level controllers (brake pre-fill) or larger headways.

---

## ⚡ Optimization & Best Practices

### 1. Platoon Management
How to form a platoon?
-   **Discovery:** Truck A broadcasts "I can lead".
-   **Join Request:** Truck B sends "Request to join".
-   **Gap Closing:** Truck B accelerates to close gap to 5m.
-   **Steady State:** Switch to CACC mode.

### 2. Cut-In Handling
What if a car cuts in between the trucks?
-   **Radar Detection:** Radar sees the new object.
-   **Split:** The platoon must logically split into two. Truck B becomes the Leader of the rear platoon.
-   **Gap Opening:** Truck B brakes to create safe gap for the car.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does CACC allow shorter gaps?
    *   **A:** Because it removes the reaction time delay. We brake when the leader brakes, not when we see the leader getting closer.
2.  **Q:** What is "String Stability"?
    *   **A:** The property that spacing errors do not amplify as they propagate down the platoon.
3.  **Q:** Does CACC work with different truck brands?
    *   **A:** Only if they implement the same V2X standard (e.g., EN 16397). Interoperability is a huge challenge.

### Challenge Task
**Task:** Communication Delay.
1.  Modify `control` to use `leader.a` from `t - delay`.
2.  Set `delay = 0.2` (200ms).
3.  Observe if the platoon becomes unstable. (It might!).

---

## 📚 Further Reading & References
-   [PATH Program (UC Berkeley) - CACC Pioneers](https://path.berkeley.edu/)
-   [Grand Cooperative Driving Challenge (GCDC)](https://www.gcdc.net/)

---

**Day 89 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
