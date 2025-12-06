# Day 157: Platooning (String Stability)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> Stop the Phantom Traffic Jam.
> - **Focus:** ACC vs CACC (Cooperative ACC), String Stability (Does the error grow or shrink downstream?), Time Headway vs Constant Distance, and Feedforward control using V2V.
> - **Code:** A Python script `platoon_stability.py` simulating 5 cars in a line. Compare ACC (Unstable, oscillations grow) vs CACC (Stable, oscillations damp out) when the leader brakes.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** String Stability: $\|e_i\|_\infty < \|e_{i-1}\|_\infty$ (Error downstream < Error upstream).
2.  **Explain** why Radar-only ACC is theoretically unstable for short headways (< 1.5s).
3.  **Implement** CACC using V2V acceleration data ($u_i = k_p e + k_d \dot{e} + u_{i-1}$).
4.  **Simulate** a "Ghost Traffic Jam" caused by reaction delay.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- PID Control.
- Laplacians/Transfer Functions (Basic Control Theory).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Phantom Jam

You are on a highway. Sudden braking. You stop. Then accelerate. There was no accident. Why?
*   **Instability:** Car 1 brakes slightly. Car 2 sees it late, brakes harder. Car 3 brakes emergency. Car 4 stops.
*   **Cause:** Sensor Delay + Actuation Delay > Time Headway.

### 🔹 Part 2: CACC (Cooperative ACC)

How to fix it?
*   **ACC:** Uses Radar distance ($d$) and relative velocity ($\dot{d}$). Feedback only.
*   **CACC:** Uses V2V to get Leader's **Acceleration** ($a_{lead}$).
*   **Feedforward:** $u_{ego} = \dots + a_{lead}$.
*   **Result:** You brake *simultaneously* with the leader, not *after* you see them slow down.

### 🔹 Part 3: Spacing Policies

*   **Constant Distance Policy (CDP):** Danger. Maintaining 5m gap at all speeds is unstable usually.
*   **Constant Time Headway (CTH):** $D = \tau \cdot v + d_0$. Safer.

---

## 💻 Implementation: Traffic String Simulation

We simulate 5 cars. Car 0 is the Leader. Cars 1-4 follow.

### 🛠️ Project Structure
```text
day157_platoon/
├── src/
│   ├── platoon_stability.py
└── output/
    ├── stability_plot.png
```

### 👨‍💻 Stability Sim (`src/platoon_stability.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class Car:
    def __init__(self, id, pos, method="ACC"):
        self.id = id
        self.x = pos
        self.v = 20.0 # m/s (start speed)
        self.a = 0.0
        self.method = method
        
        # Params
        self.tau = 0.8 # Desired Time Headway (seconds)
        self.d0 = 5.0 # Standstill gap
        self.kp = 1.0
        self.kd = 3.0 # High damping needed for ACC
        self.ka = 1.0 # Feedforward gain (only for CACC)
        
    def update(self, leader, dt):
        # 1. Measurement
        dist = leader.x - self.x
        rel_v = leader.v - self.v
        
        # 2. Desired Dist (Constant Time Headway)
        d_des = self.d0 + self.tau * self.v
        
        # 3. Error
        err_dist = dist - d_des
        
        # 4. Control Law
        # u = k_p * (d - d_des) + k_d * (v_lead - v)
        cmd_acc = self.kp * err_dist + self.kd * rel_v
        
        # CACC Feedforward
        if self.method == "CACC":
            cmd_acc += self.ka * leader.a
            
        # 5. Actuation Lag (First order lag)
        # alpha = dt / (lag + dt)
        # But for simple stability demo, let's use instant accel calc, with saturation
        self.a = np.clip(cmd_acc, -5.0, 3.0)
        
        # 6. Physics
        self.v += self.a * dt
        self.x += self.v * dt

def run_simulation(method):
    dt = 0.05
    steps = 400
    
    # Init Platoon
    # 5 Cars. Spaced by 25m approx (20m/s * 1s + 5m)
    leader = Car(0, 150.0, "Leader")
    followers = [Car(i, 150.0 - i*25.0, method) for i in range(1, 6)]
    
    # History
    history = {i: [] for i in range(6)}
    
    for t in range(steps):
        time_s = t * dt
        
        # Leader Behavior:
        # Cruising, then Sinusoidal Brake/Accel disturbance
        if 5.0 < time_s < 10.0:
            leader.a = -2.0 * np.sin(2 * np.pi * (time_s - 5.0) / 5.0)
        else:
            leader.a = 0.0
            
        leader.v += leader.a * dt
        leader.x += leader.v * dt
        
        history[0].append(leader.v)
        
        # Follower Updates
        prev = leader
        for f in followers:
            f.update(prev, dt)
            history[f.id].append(f.v)
            prev = f
            
    return history, steps*dt

def plot_results(acc_hist, cacc_hist, duration):
    t = np.linspace(0, duration, len(acc_hist[0]))
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ACC
    axs[0].set_title("Standard ACC (Radar Only)")
    for i in range(6):
        linewidth = 2 if i==0 or i==5 else 1
        label = "Leader" if i==0 else f"Car {i}"
        axs[0].plot(t, acc_hist[i], label=label, linewidth=linewidth)
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].grid()
    axs[0].legend()
    
    # Plot CACC
    axs[1].set_title("CACC (Radar + V2V Feedforward)")
    for i in range(6):
        linewidth = 2 if i==0 or i==5 else 1
        label = "Leader" if i==0 else f"Car {i}"
        axs[1].plot(t, cacc_hist[i], label=label, linewidth=linewidth)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid()
    
    plt.savefig("output/stability_plot.png")
    print("Sim Complete. Plot Saved.")

def main():
    print("Simulating ACC...")
    hist_acc, dur = run_simulation("ACC")
    
    print("Simulating CACC...")
    hist_cacc, _ = run_simulation("CACC")
    
    plot_results(hist_acc, hist_cacc, dur)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Crash"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe (ACC):** Car 5 oscillates *more* than the Leader. This is String Instability.
- **Observe (CACC):** Car 5 tracks tight. Amplitude dampens.
- **Modify:** Reduce `tau` (Time Headway) to 0.4s in ACC mode.
- **Result:** Crash! (Velocity graphs intersect or Distance < 0). ACC needs larger gaps to be stable. CACC allows smaller gaps (High density).

---

## 🚀 Project: "Platoon Join Maneuver"

**Goal:** FSM.
1.  **State:** `FREE_AGENT` searches for platoon.
2.  **Request:** Sends "Join Request" to Platoon Leader (V2V).
3.  **Ack:** Leader says "Okay, join at tail, Slot 4".
4.  **Action:** Ego speeds up to catch the tail, then switches to CACC mode.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Packet Loss in CACC"
*   **Cause:** V2V fails. $a_{lead}$ becomes 0.
*   **Result:** Degrades to ACC. Might become unstable if gap is small.
*   **Fix:** Graceful degradation. If packet lost, increase `tau` (Gap) immediately.

#### 2. "Integral Windup"
*   **Cause:** If using PID and distance error saturates.
*   **Fix:** Clamp Integrator.

---

## ⚡ Optimization: Energy Saving

Drafting (Aerodynamic Drag Reduction).
*   **Physics:** Drag $\propto v^2$.
*   **Benefit:** Following at 5m saves ~20% fuel.
*   **Challenge:** 5m at 20m/s is 0.25s gap. Human reaction is 1.0s. Impossible for humans. Only CACC is safe.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Feedforward in this context?
    *   **A:** Sending the Leader's Control Input (Braking/Torque) directly to followers, bypassing the delay of waiting for physics (distance change) to occur.
2.  **Q:** Centralized vs Decentralized Platoon?
    *   **A:** Centralized: Leader tells every car what to do. Decentralized: Each car looks at the one in front (Predecessor Following). CACC is usually decentralized (or Predecessor-Leader Following).
3.  **Q:** String Stability Condition?
    *   **A:** The transfer function magnitude $|\Gamma(j\omega)| \le 1$ for all frequencies $\omega$.

### Challenge Task
> **Task:** Heterogeneous Platoon.
> 1. Mix Trucks (Slow braking) and Cars (Fast braking).
> 2. Order matters!
> 3. Does putting the Truck at the front or back improve safety? (Answer: Light cars should trail heavy trucks? Or Heavy lead? Better braking should be at the rear to avoid collisions? Actually, worst braking vehicle should be Leader. If Leader can brake 1g and Follower only 0.5g, collision is guaranteed).

---

## 📚 Further Reading
- **PATH Program (Berkeley):** Pioneers of Platooning (1990s).
- **Rajamani:** "Vehicle Dynamics and Control" (Standard textbook on CACC).

---

**Day 157 Complete**
