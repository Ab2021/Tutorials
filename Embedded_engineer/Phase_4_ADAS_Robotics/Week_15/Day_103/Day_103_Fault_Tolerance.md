# Day 103: Fault Tolerant Control
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 103 Focus:**
> When a fault happens (and it will), the system must not kill anyone. **Fault Tolerance** is the art of surviving failures. We move from "Fail-Safe" (turning off) to "Fail-Operational" (limping home).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Fail-Silent, Fail-Safe, and Fail-Operational.
2.  **Implement** a Watchdog Timer to detect software freezes.
3.  **Design** a "Limp Home" mode (Degraded performance).
4.  **Execute** a Safe Stop Maneuver (Pull over to shoulder).
5.  **Simulate** a sensor failure and the subsequent recovery.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 100:** HARA (Safety Goals).
-   **Control Theory:** State Machines.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** State Machine logic.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Failure Modes

1.  **Fail-Silent:** System stops outputting. (e.g., ECU resets).
2.  **Fail-Safe:** System enters a safe state. (e.g., Traffic light turns Red-Blinking).
3.  **Fail-Operational:** System continues working. (e.g., Aircraft avionics, L4 Robotaxi steering).

### 🔹 Part 2: Detection Mechanisms

How do we know we failed?
-   **Watchdog:** Hardware timer that resets the CPU if software hangs.
-   **Heartbeat:** Periodic message between ECUs ("I'm alive").
-   **Range Check:** `if speed > 300 km/h: Error`.
-   **Plausibility Check:** `if wheel_speed > 0 and gps_speed == 0: Error`.

### 🔹 Part 3: Safe Stop (MRM)

**Minimum Risk Maneuver (MRM):**
-   If the driver doesn't take over (Level 3) or can't (Level 4), the car must stop safely.
-   **Strategy:**
    1.  Turn on hazards.
    2.  Slow down gently (-2 m/s²).
    3.  Steer to shoulder (if possible) or stay in lane (if blind).
    4.  Stop and unlock doors.

---

## 💻 Implementation: The Safety Monitor

**Scenario:**
-   **System:** Cruise Control.
-   **Fault:** Radar stops sending data (Heartbeat loss).
-   **Reaction:** Transition to "Degraded Mode" (Coast) -> "Safe Stop" (Brake).

### 🛠️ Setup
Create `week15_day103` and `safety_monitor.py`.

```bash
mkdir -p ~/ros2_ws/src/week15_day103
cd ~/ros2_ws/src/week15_day103
touch safety_monitor.py
```

### 👨‍💻 Code: State Machine with Watchdog

```python
import time
import threading

class Watchdog:
    def __init__(self, timeout, callback):
        self.timeout = timeout
        self.callback = callback
        self.timer = None
        self.kick()

    def kick(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.callback)
        self.timer.start()

    def stop(self):
        if self.timer:
            self.timer.cancel()

class CruiseControl:
    def __init__(self):
        self.state = "OFF" # OFF, ACTIVE, DEGRADED, SAFE_STOP
        self.speed = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.radar_alive = False
        
        # Watchdog: If Radar doesn't update for 0.5s, trigger fault
        self.wd = Watchdog(0.5, self.on_radar_loss)
        
    def on_radar_loss(self):
        print("!!! CRITICAL: Radar Heartbeat Lost !!!")
        self.trigger_fault()
        
    def trigger_fault(self):
        if self.state == "ACTIVE":
            print("Transitioning to DEGRADED mode.")
            self.state = "DEGRADED"
            
    def update_radar(self, data):
        self.radar_alive = True
        self.wd.kick() # Reset timer
        # Process data...
        
    def control_loop(self):
        print(f"State: {self.state} | Speed: {self.speed:.1f}")
        
        if self.state == "ACTIVE":
            # Normal P-Control
            self.throttle = 0.5
            self.brake = 0.0
            self.speed += 1.0 # Simulate accel
            
        elif self.state == "DEGRADED":
            # Coasting (Throttle 0)
            self.throttle = 0.0
            self.brake = 0.0
            self.speed *= 0.99 # Drag
            
            # If speed drops below 10, stop completely
            if self.speed < 10.0:
                self.state = "SAFE_STOP"
                
        elif self.state == "SAFE_STOP":
            # Active Braking
            self.throttle = 0.0
            self.brake = 1.0
            self.speed -= 2.0
            if self.speed < 0: self.speed = 0
            
            print(">>> HAZARDS ON <<<")

def main():
    cc = CruiseControl()
    cc.state = "ACTIVE"
    cc.speed = 50.0
    
    try:
        # Simulation Loop
        for i in range(20):
            time.sleep(0.1)
            
            # Simulate Radar working for first 1 second
            if i < 10:
                cc.update_radar("Object detected")
            else:
                # Radar dies (cable cut)
                pass
                
            cc.control_loop()
            
    except KeyboardInterrupt:
        pass
    finally:
        cc.wd.stop()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Heartbeat

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   `t=0.0s`: State ACTIVE. Speed increases.
    -   `t=1.0s`: Radar stops updating.
    -   `t=1.5s`: Watchdog fires ("Radar Heartbeat Lost"). State -> DEGRADED.
    -   `t=1.6s`: Throttle cuts to 0. Speed drops.
    -   `t=2.5s`: Speed low. State -> SAFE_STOP. Hazards ON.
3.  **Experiment:**
    -   Change Watchdog timeout to 2.0s.
    -   **Result:** The car drives "blind" for 2 seconds before reacting. This violates the FTTI (Fault Tolerant Time Interval) if the safety goal requires 500ms reaction.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. False Positives
**Symptom:** Watchdog triggers randomly.
**Cause:** System load is high, causing the "Kick" thread to be delayed.
**Solution:** Run Watchdog on a high-priority thread or dedicated hardware (External Watchdog).

#### 2. Recovery
**Symptom:** Radar comes back, but system stays in SAFE_STOP.
**Cause:** Latching faults.
**Solution:** Safety systems usually require a "Key Cycle" (Restart) to reset critical faults. Do not auto-recover from ASIL D faults while driving.

---

## ⚡ Optimization & Best Practices

### 1. 2-out-of-3 Voting (TMR)
For Fail-Operational systems (e.g., Space Shuttle, Waymo Steering):
-   Run 3 identical computers.
-   Vote: If A says "Left", B says "Left", C says "Right" -> Go Left. Ignore C.
-   **Cost:** 3x Hardware.

### 2. Graceful Degradation
Don't just give up.
-   **Lidar fails?** Switch to Camera-only mode (Limit speed to 50 km/h).
-   **GPS fails?** Switch to Odometry/SLAM (Limit range to 5 km).
-   Keep the customer moving if safe.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a Watchdog and a Heartbeat?
    *   **A:** A **Watchdog** monitors the *local* execution (did I hang?). A **Heartbeat** monitors a *remote* node (is the other ECU alive?).
2.  **Q:** Why is Fail-Operational required for Level 4?
    *   **A:** Because there is no driver to take over. If the steering fails, the car must still be able to steer to the shoulder.
3.  **Q:** What is an MRM?
    *   **A:** Minimum Risk Maneuver. The "Plan B" to get the car to a safe state.

### Challenge Task
**Task:** Redundant Sensor Check.
1.  `check_sensors(radar_dist, camera_dist)`.
2.  If `abs(radar_dist - camera_dist) > 5.0m`: Trigger Fault.
3.  Which sensor is wrong? You don't know.
4.  Action: Disengage ACC (Fail-Safe).

---

## 📚 Further Reading & References
-   [Safe Stop Strategies](https://www.adas-validation.com/safe-stop-maneuver)
-   [Watchdog Timers in Embedded Systems](https://barrgroup.com/embedded-systems/how-to/watchdog-timer-software-setup)

---

**Day 103 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
