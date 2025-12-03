# Day 87: CAM/DENM Messages (European Standard)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 87 Focus:**
> While the US uses BSM (SAE J2735), Europe uses **CAM (Cooperative Awareness Message)** and **DENM (Decentralized Environmental Notification Message)** defined by ETSI. The concepts are similar, but the trigger logic is smarter. Today, we cross the Atlantic.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** US (BSM) vs EU (CAM/DENM) standards.
2.  **Explain** the CAM generation rules (Dynamics-based triggering).
3.  **Analyze** the DENM structure for event handling (Roadworks, Accident).
4.  **Implement** a CAM Triggering Logic (Send only when moving).
5.  **Simulate** a DENM broadcast for a "Hazardous Location".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 86:** BSM and ASN.1.
-   **ETSI ITS:** European Telecommunications Standards Institute.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: CAM (Cooperative Awareness Message)

Equivalent to BSM, but smarter.
-   **BSM:** Sends at fixed 10Hz (mostly).
-   **CAM:** Sends based on **Dynamics**.
    -   $T_{gen}$ (Generation Interval) is variable (0.1s to 1.0s).
    -   **Trigger Rules:** Send a CAM if:
        -   Heading change > $4^\circ$.
        -   Position change > 4m.
        -   Speed change > 0.5 m/s.
        -   Or time since last CAM > 1.0s (Heartbeat).
-   **Benefit:** Reduces channel load when vehicles are stopped or moving steadily on a highway.

### 🔹 Part 2: DENM (Decentralized Environmental Notification Message)

Used for **Events**.
-   **Use Cases:** Roadworks, Stationary Vehicle, Weather, Wrong Way Driver.
-   **Lifecycle:**
    -   **Trigger:** Event detected.
    -   **Update:** Event parameters changed.
    -   **Termination:** Event over.
-   **Relevance Area:** Defines where the message is valid (e.g., 500m radius).
-   **GeoNetworking:** Packets are forwarded by other cars to reach the relevance area (Multi-hop).

### 🔹 Part 3: ETSI ITS Stack

-   **Facilities Layer:** CAM/DENM generation.
-   **Networking & Transport:** GeoNetworking (BTP/GN).
-   **Access:** ITS-G5 (802.11p).

---

## 💻 Implementation: CAM Trigger & DENM Generator

**Scenario:**
-   **CAM:** A car drives in a circle. It should send CAMs frequently (due to heading change). Then it stops. CAM rate drops to 1Hz.
-   **DENM:** The car detects "Ice on Road". Broadcasts a DENM.

### 🛠️ Setup
Create `week13_day87` and `etsi_its.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day87
cd ~/ros2_ws/src/week13_day87
touch etsi_its.py
```

### 👨‍💻 Code: CAM Trigger Logic

```python
import time
import math
import numpy as np
import matplotlib.pyplot as plt

class VehicleState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0 # Degrees
        self.speed = 0.0 # m/s
        self.last_cam_time = 0.0
        self.last_cam_state = None # (x, y, heading, speed)

class CAMGenerator:
    def __init__(self):
        self.cam_log = [] # List of timestamps
        
    def check_trigger(self, state, current_time):
        # ETSI EN 302 637-2 Rules
        
        # 1. Time Check (Min 100ms, Max 1000ms)
        dt = current_time - state.last_cam_time
        if dt < 0.1: # T_GenMin
            return False
            
        if dt >= 1.0: # T_GenMax (Heartbeat)
            return True
            
        # 2. Dynamics Check
        if state.last_cam_state is None:
            return True
            
        last_x, last_y, last_hdg, last_spd = state.last_cam_state
        
        # Position Change
        dist = math.sqrt((state.x - last_x)**2 + (state.y - last_y)**2)
        if dist > 4.0:
            return True
            
        # Heading Change
        hdg_diff = abs(state.heading - last_hdg)
        if hdg_diff > 4.0:
            return True
            
        # Speed Change
        spd_diff = abs(state.speed - last_spd)
        if spd_diff > 0.5:
            return True
            
        return False

    def generate(self, state, current_time):
        if self.check_trigger(state, current_time):
            # Send CAM
            # print(f"CAM Sent at {current_time:.2f}s")
            state.last_cam_time = current_time
            state.last_cam_state = (state.x, state.y, state.heading, state.speed)
            self.cam_log.append(current_time)
            return True
        return False

class DENMGenerator:
    def __init__(self):
        self.active_events = {}
        
    def trigger_event(self, event_type, position, duration=10.0):
        event_id = len(self.active_events) + 1
        denm = {
            'stationID': 1234,
            'sequenceNumber': 0,
            'detectionTime': time.time(),
            'validityDuration': duration,
            'eventPosition': position,
            'causeCode': event_type # e.g., 1=Accident, 2=Roadworks
        }
        self.active_events[event_id] = denm
        print(f"DENM Triggered: {event_type} at {position}")
        return denm

def main():
    # Simulation: Car driving
    # 0-5s: Stopped
    # 5-10s: Accelerating straight
    # 10-20s: Turning (Circle)
    # 20-30s: Constant Speed Straight
    
    state = VehicleState()
    cam_gen = CAMGenerator()
    denm_gen = DENMGenerator()
    
    dt = 0.01 # 100Hz Physics
    time_steps = np.arange(0, 30, dt)
    
    cam_sent_times = []
    
    print("Simulating Vehicle Dynamics & CAM Generation...")
    
    for t in time_steps:
        # Physics Update
        if t < 5.0:
            # Stopped
            state.speed = 0.0
        elif t < 10.0:
            # Accelerate
            state.speed += 2.0 * dt # 2 m/s^2
            state.x += state.speed * dt
        elif t < 20.0:
            # Turn
            state.heading += 10.0 * dt # 10 deg/s
            state.x += state.speed * math.cos(math.radians(state.heading)) * dt
            state.y += state.speed * math.sin(math.radians(state.heading)) * dt
        else:
            # Constant Speed
            state.x += state.speed * math.cos(math.radians(state.heading)) * dt
            state.y += state.speed * math.sin(math.radians(state.heading)) * dt
            
        # CAM Logic
        if cam_gen.generate(state, t):
            cam_sent_times.append(t)
            
        # DENM Logic (Simulate Ice detection at t=15)
        if abs(t - 15.0) < dt/2:
            denm_gen.trigger_event("ICE_ON_ROAD", (state.x, state.y))
            
    # Analysis
    intervals = np.diff(cam_sent_times)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cam_sent_times[:-1], intervals, 'b.-')
    plt.axhline(0.1, color='r', linestyle='--', label='Min Interval (0.1s)')
    plt.axhline(1.0, color='g', linestyle='--', label='Max Interval (1.0s)')
    plt.title("CAM Generation Interval vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Interval (s)")
    plt.legend()
    plt.grid()
    
    # Annotate phases
    plt.text(2, 0.5, "Stopped (1Hz)", ha='center')
    plt.text(7.5, 0.2, "Accel (High Rate)", ha='center')
    plt.text(15, 0.2, "Turning (High Rate)", ha='center')
    plt.text(25, 0.2, "Steady (Low Rate)", ha='center') # Actually, if speed is high, pos change triggers it
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Efficiency Gain

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   **0-5s (Stopped):** Interval is exactly 1.0s (Heartbeat).
    -   **5-10s (Accel):** Interval drops to ~0.2s (Triggered by Speed/Pos change).
    -   **10-20s (Turn):** Interval stays low (Triggered by Heading change).
    -   **20-30s (Steady):** Interval depends on speed. If speed is 10 m/s, position changes 4m every 0.4s. So rate is ~2.5Hz.
3.  **Comparison:**
    -   BSM sends at 10Hz constant (300 messages in 30s).
    -   CAM sends adaptively (Count the dots, likely < 100 messages).
    -   **Result:** CAM saves bandwidth!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Jittery CAMs
**Symptom:** CAM rate fluctuates wildly.
**Cause:** Noisy GPS/IMU data triggering thresholds.
**Solution:** Filter the vehicle state (Kalman Filter) before checking triggers.

#### 2. DENM Termination
**Symptom:** Warning persists after hazard is gone.
**Cause:** `validityDuration` expired but not explicitly terminated.
**Solution:** The originating station should send a `negation` DENM (ActionID + Termination) when the event is cleared.

---

## ⚡ Optimization & Best Practices

### 1. Relevance Area
Don't broadcast DENM to the whole world.
-   **Geocasting:** Define a destination area (e.g., Circle(Lat, Lon, R=500m)).
-   Only cars inside or approaching this area process the message.

### 2. Aggregation
If 10 cars detect "Ice", we don't want 10 DENMs.
-   **Data Fusion:** The Roadside Unit (RSU) or Cloud aggregates reports and broadcasts a single authoritative DENM.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of CAM over BSM?
    *   **A:** Channel efficiency. It reduces congestion by sending fewer messages when dynamics are low.
2.  **Q:** When is a DENM sent?
    *   **A:** Only when an event occurs (Event-driven).
3.  **Q:** What is the "Heartbeat" rule?
    *   **A:** Even if nothing changes, send a CAM every 1.0s so neighbors know you are still there (and didn't crash/lose power).

### Challenge Task
**Task:** Wrong Way Driver DENM.
1.  Check if `heading` is opposite to the lane direction (from Map).
2.  If `diff > 150` degrees, trigger DENM `causeCode=WrongWay`.
3.  Set `validityDuration` short (e.g., 5s) and repeat while condition holds.

---

## 📚 Further Reading & References
-   [ETSI EN 302 637-2 (CAM Specification)](https://www.etsi.org/deliver/etsi_en/302600_302699/30263702/01.03.02_60/en_30263702v010302p.pdf)
-   [ETSI EN 302 637-3 (DENM Specification)](https://www.etsi.org/deliver/etsi_en/302600_302699/30263703/01.02.01_60/en_30263703v010201p.pdf)

---

**Day 87 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
