# Day 158: Intersection Management (V2I)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> No traffic lights. No waiting.
> - **Focus:** SPaT (Signal Phase and Timing), MAP Messages (Intersection Topology), and AIM (Autonomous Intersection Management) - Reservation Tiles in Time-Space.
> - **Code:** A Python script `aim_manager.py` that simulates a 4-way intersection. Cars send "Reservation Requests" ($t_{arrival}$, $v_{arrival}$). The Intersection Manager approves or delays them to prevent collisions without stopping flow.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Parse** V2I messages: MAP (Where the lanes are) and SPaT (When the light turns green).
2.  **Explain** the "Green Wave" speed advisory ($V = Dist / TimeToGreen$).
3.  **Implement** a Tile-Based Reservation System for lightless intersections.
4.  **Visualize** Space-Time conflicts in a junction.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (In real life: RSU - Road Side Unit).

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Trajectory Prediction.
- Resource Allocation (Mutex).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Legacy V2I (Traffic Lights)

*   **Fixed Time:** Signal is dumb.
*   **Actuated:** Signal uses induction loops.
*   **V2I (SPaT):** Signal broadcasts: "Phase 2 (North-South Green) will end in 12.5 seconds."
*   **GLOSA (Green Light Optimized Speed Advisory):** Car calculates: "If I drive at 45km/h, I will hit Green. If I drive 60km/h, I will hit Red. I should slow down to save fuel."

### 🔹 Part 2: AIM (Autonomous Intersection Management)

Why stop at all?
Divide the intersection into a **Grid of Tiles** ($2m \times 2m$).
*   Car A requests: "I need Tile (5,5) at $t=10.1s$."
*   Manager checks: "Is Tile (5,5) booked at $t=10.1$?"
    *   **No:** BOOKED. Send "Confirm".
    *   **Yes:** REJECT. Send "Try entering 2s later".

### 🔹 Part 3: Space-Time Reservation

Collision check is not just Geometry ($X, Y$). It is ($X, Y, Time$).
Two cars can occupy the same spot $(5,5)$, just not at the same time.

---

## 💻 Implementation: The Intersection Manager

We simulate a 4-way Crossing.

### 🛠️ Project Structure
```text
day158_aim/
├── src/
│   ├── aim_manager.py
└── output/
    ├── reservation_grid.png
```

### 👨‍💻 AIM Simulator (`src/aim_manager.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class ReservationGrid:
    def __init__(self, size=20, tile_size=2.0):
        # 20x20m intersection
        self.size = size
        self.tile_size = tile_size
        self.grid_dim = int(size / tile_size)
        # Reservations: key=(x_idx, y_idx), value=List of (t_start, t_end)
        self.reservations = {}

    def get_tiles(self, path_x, path_y):
        # Convert path points to tile indices
        tiles = set()
        for x, y in zip(path_x, path_y):
            if -self.size/2 <= x < self.size/2 and -self.size/2 <= y < self.size/2:
                idx_x = int((x + self.size/2) / self.tile_size)
                idx_y = int((y + self.size/2) / self.tile_size)
                tiles.add((idx_x, idx_y))
        return list(tiles)

    def is_available(self, tiles, time_window):
        # Check conflicts
        t_req_start, t_req_end = time_window
        
        for tile in tiles:
            if tile in self.reservations:
                for (start, end) in self.reservations[tile]:
                    # Overlap Check
                    if max(t_req_start, start) < min(t_req_end, end):
                        return False # Conflict
        return True

    def book(self, tiles, time_window):
        t_start, t_end = time_window
        for tile in tiles:
            if tile not in self.reservations:
                self.reservations[tile] = []
            self.reservations[tile].append((t_start, t_end))
        return True

class CarAgent:
    def __init__(self, id, start_pos, velocity, direction):
        self.id = id
        self.pos = np.array(start_pos, dtype=float) # x, y
        self.vel = np.array(velocity, dtype=float)
        self.direction = direction # 'NS' or 'EW'
        self.size = 2.0 # Car length
        
    def generate_path(self, t_entry):
        # Simple Linear Path
        # Crossing time = width / speed
        # t_entry is when front bumper enters intersection box
        
        path_x = []
        path_y = []
        
        # Simulate trajectory for 3 seconds of crossing
        for dt in np.arange(0, 3.0, 0.1):
            p = self.pos + self.vel * dt
            path_x.append(p[0])
            path_y.append(p[1])
            
        return path_x, path_y

def main():
    manager = ReservationGrid()
    
    # Scene: 4-Way Intersection (-10 to 10 coords)
    
    # Car 1: North -> South. Enters at t=0.
    car1 = CarAgent(1, [0, 15], [0, -10], 'NS')
    
    # Car 2: East -> West. Enters at t=0.5. (Collision Course?)
    # Collision point is roughly (0,0).
    # Car 1 hits (0,0) at t=1.5s approx.
    # Car 2 needs to hit (0,0) at same time to crash.
    car2 = CarAgent(2, [15, 0], [-10, 0], 'EW')
    
    cars = [car1, car2]
    
    print("--- AIM Reservation Requests ---")
    
    for car in cars:
        # 1. Prediction (When does it enter/leave?)
        # Let's request Reservation for [t_entry, t_exit]
        
        # Distance to entry edge (Box is -10 to 10)
        # Entry Line: depending on direction
        
        # Simplified: Request tiles for specific absolute times based on const velocity
        # Path samples
        path_x, path_y = car.generate_path(0)
        tiles = manager.get_tiles(path_x, path_y)
        
        # Estimate crossing time window (Start entering at current dist/vel, end at...)
        # Assume cars are just entering box now (t=0 simulation time for car pos)
        # Car 1: At (0, 15). Box Y=10. Dist=5m. Speed=10. Enters t=0.5.
        # Exits Y=-10. Dist=25m. Exits t=2.5.
        
        dist_to_entry = abs(max(abs(car.pos[0]), abs(car.pos[1])) - 10.0)
        t_entry = dist_to_entry / np.linalg.norm(car.vel)
        t_exit = (dist_to_entry + 20.0) / np.linalg.norm(car.vel)
        
        # Safety Buffer
        time_window = (t_entry + 0.0, t_exit + 0.5) 
        
        print(f"Car {car.id} Request: Tiles={len(tiles)}, Time={time_window[0]:.2f}-{time_window[1]:.2f}s")
        
        if manager.is_available(tiles, time_window):
            manager.book(tiles, time_window)
            print(f" -> APPROVED.")
        else:
            print(f" -> REJECTED (Conflict). Slow down!")
            
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw Grid
    ticks = np.arange(-10, 11, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True)
    
    # Draw Reservations
    # For visualization, just coloring tiles red/blue if booked
    
    for tile in manager.reservations:
        # tile is (idx_x, idx_y)
        # Convert to center xy
        cx = -10.0 + tile[0]*2.0 + 1.0
        cy = -10.0 + tile[1]*2.0 + 1.0
        
        # Check who booked? (Simplification: just color it)
        rect = plt.Rectangle((cx-0.9, cy-0.9), 1.8, 1.8, color='red', alpha=0.3)
        ax.add_patch(rect)
        
    # Draw Car Paths
    for car in cars:
        px, py = car.generate_path(0)
        ax.plot(px, py, 'k--', label=f'Path {car.id}')
        ax.arrow(car.pos[0], car.pos[1], car.vel[0], car.vel[1], head_width=1, color='blue')
        
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_title("Intersection Reservation Grid")
    plt.legend()
    plt.savefig("output/reservation_grid.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Deadlock"

### 1. Lab Objectives
- **Run:** Sim.
- **Scenario:** 4 Cars enter exactly at the same time from N, S, E, W.
- **Logic:** FCFS (First Come First Serve).
- **Result:** Car 1 books. Car 2 (conflicts) rejected. Car 3 (parallel to 1) approved. Car 4 rejected.
- **Fail:** If policy is circular (Round Robin), complex timing ensues.
- **Task:** Implement "Right of Way" logic. If rejected, retry with `t_start += 2.0s`.

---

## 🚀 Project: "Virtual Traffic Light"

**Goal:** Centralized Signal in the Cloud.
1.  **State:** RSU maintains a Virtual Cycle (Green/Red).
2.  **Msg:** RSU broadcasts SPaT.
3.  **Car:** Displays Red Light on Dashboard.
4.  **Benefit:** No physical lights maintenance. Adaptive phase length based on real-time queue length (V2I queue reporting).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Clock Drift"
*   **Cause:** Car A Clock != RSU Clock.
*   **Result:** A thinks it has Green, RSU thinks it's Red.
*   **Fix:** GPS Time / NTP. All V2X messages must use GPS Epoch time (TAI).

#### 2. "Packet Delay"
*   **Cause:** Request takes 200ms. Response takes 200ms.
*   **Result:** Car enters intersection before receiving Approval.
*   **Fix:** Reservation Horizon. Must book 5 seconds in advance. If no ACK by 2s out, Brake.

---

## ⚡ Optimization: Swarm Crossing

Instead of blocking the whole path, break the car into Time-Points.
Allows "weaving" through traffic. Cars pass within inches of each other at 50km/h. (Scary for humans, efficient for robots).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is SPaT?
    *   **A:** Signal Phase and Timing. "Lane 1 Green State: 15s remaining".
2.  **Q:** What is MAP?
    *   **A:** Topology. "Lane 1 connects to Lane 3 (Through) and Lane 4 (Right Turn)".
3.  **Q:** What is DSRC Channel 172?
    *   **A:** The Safety Channel (CCH). Always reserved for BSMs. Service channels (SCH) used for maps/downloading.

### Challenge Task
> **Task:** Ambulance Priority.
> 1. Emergency Vehicle sends `SpecialVehicleExtensions` in BSM.
> 2. AIM Manager detects `LightbarInUse`.
> 3. Manager cancels all conflicting reservations and checks "Preemption".
> 4. Grants Green Wave to Ambulance.

---

## 📚 Further Reading
- **Peter Stone (UT Austin):** AIM System papers.
- **SAE J2735:** MAP/SPaT definitions.

---

**Day 158 Complete**
