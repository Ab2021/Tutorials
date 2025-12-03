# Day 173: Advanced Topics (V2X, Fleet)
## Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career

---

> **📝 Day 173 Focus:**
> A single robot is smart. A swarm of robots is genius. Today, we look at the **Connected Future**. V2X (Vehicle-to-Everything), Fleet Management, and the Cloud Infrastructure that powers Robotaxis.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** V2V, V2I, and V2P communication (DSRC vs C-V2X).
2.  **Design** a Fleet Management System (FMS) architecture.
3.  **Understand** OTA (Over-the-Air) update security.
4.  **Simulate** a V2X scenario (Green Light Optimal Speed Advisory).
5.  **Discuss** Teleoperation and Remote Assistance.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Networking:** TCP/IP, MQTT, 5G.
-   **Cryptography:** Public Key Infrastructure (PKI).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **MQTT Broker:** Mosquitto.
-   **Simulation:** CARLA (supports V2X).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: V2X (Vehicle-to-Everything)

-   **V2V (Vehicle-to-Vehicle):** "I'm braking!" (Forward Collision Warning).
-   **V2I (Vehicle-to-Infrastructure):** "Light will turn Red in 3s." (SPaT - Signal Phase and Timing).
-   **V2P (Vehicle-to-Pedestrian):** Phone broadcasts location to car.
-   **Standards:**
    -   **DSRC (802.11p):** WiFi-based. Low latency. Short range.
    -   **C-V2X (5G):** Cellular. High bandwidth. Long range.

### 🔹 Part 2: Fleet Management

-   **Dispatch:** "Car 101, pick up User A at Location B."
-   **Routing:** "Traffic jam on Main St. Reroute all cars."
-   **Health:** "Car 102, tire pressure low. Return to base."
-   **Protocol:** MQTT is standard for IoT fleets.

### 🔹 Part 3: OTA (Over-the-Air) Updates

-   **A/B Partitioning:** Update Partition B while running on A. Reboot to B. If fail, rollback to A.
-   **Security:** Signed binaries. Encrypted channels.
-   **Uptane:** The security framework for automotive OTA.

---

## 💻 Implementation: V2I Traffic Light Assistant

**Scenario:**
-   Traffic Light broadcasts its state via MQTT.
-   Car receives state and adjusts speed to catch the Green light (GLOSA).

### 🛠️ Setup
Create `week25_advanced`. Install MQTT.

```bash
pip install paho-mqtt
```

### 👨‍💻 Code: The Infrastructure (Traffic Light)

```python
import paho.mqtt.client as mqtt
import time
import json

def traffic_light_sim():
    client = mqtt.Client("TrafficLight_001")
    client.connect("test.mosquitto.org", 1883, 60)
    
    state = "RED"
    time_to_change = 10
    
    while True:
        # Publish SPaT (Signal Phase and Timing)
        payload = {
            "id": "TL_001",
            "state": state,
            "time_to_change": time_to_change,
            "timestamp": time.time()
        }
        client.publish("v2x/traffic_lights", json.dumps(payload))
        print(f"Broadcasting: {state} ({time_to_change}s)")
        
        time.sleep(1)
        time_to_change -= 1
        
        if time_to_change == 0:
            state = "GREEN" if state == "RED" else "RED"
            time_to_change = 10

if __name__ == "__main__":
    traffic_light_sim()
```

### 👨‍💻 Code: The Vehicle (GLOSA)

```python
import paho.mqtt.client as mqtt
import json

current_speed = 10.0 # m/s
dist_to_light = 100.0 # m

def on_message(client, userdata, msg):
    global current_speed
    data = json.loads(msg.payload.decode())
    
    if data['id'] == "TL_001":
        state = data['state']
        ttc = data['time_to_change'] # Time to Change
        
        # GLOSA Logic (Green Light Optimal Speed Advisory)
        if state == "RED":
            # Light is Red. Will turn Green in 'ttc' seconds.
            # We are 100m away.
            # Required Speed = Dist / Time
            target_speed = dist_to_light / (ttc + 2.0) # Arrive 2s after Green
            print(f"Light RED. Slow down to {target_speed:.1f} m/s to catch Green.")
        else:
            print("Light GREEN. Maintain speed.")

def vehicle_client():
    client = mqtt.Client("Car_101")
    client.connect("test.mosquitto.org", 1883, 60)
    client.subscribe("v2x/traffic_lights")
    client.on_message = on_message
    
    print("Listening for V2X messages...")
    client.loop_forever()

if __name__ == "__main__":
    vehicle_client()
```

---

## 🔬 Lab Exercise: The Fleet Dashboard

### Lab Objectives
1.  **Run the Scripts:**
    -   Start `traffic_light.py` in Terminal 1.
    -   Start `vehicle.py` in Terminal 2.
    -   **Observation:** The car calculates the optimal speed to avoid stopping.
2.  **Scale Up:**
    -   Run 5 instances of `vehicle.py`.
    -   **Concept:** This is how a fleet coordinates.
3.  **Teleoperation:**
    -   Imagine the car gets stuck.
    -   Send an MQTT message: `v2x/car/101/cmd` -> `{"action": "remote_control"}`.
    -   Stream video back to operator.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Latency
**Symptom:** Car receives "Red Light" message too late.
**Cause:** Cloud MQTT has 100ms+ latency.
**Solution:** **Edge Computing**. The MQTT broker should be on the street corner (MEC - Multi-Access Edge Computing), not in AWS.

#### 2. Security
**Symptom:** Hacker injects "Green Light" message.
**Cause:** No authentication.
**Solution:** Use TLS certificates. Every message must be signed by a trusted CA.

---

## ⚡ Optimization & Best Practices

### 1. Hybrid V2X
-   Use DSRC for safety-critical (Braking).
-   Use 5G for infotainment/maps (High bandwidth).

### 2. Predictive Maintenance
-   Upload logs to cloud.
-   Analyze battery voltage trends.
-   Predict failure *before* it happens.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is GLOSA?
    *   **A:** Green Light Optimal Speed Advisory. Telling the driver/car the perfect speed to pass through a green wave.
2.  **Q:** Why is OTA dangerous?
    *   **A:** If the update fails, the car is "bricked" (unusable). Requires robust rollback mechanisms.
3.  **Q:** What is the difference between Teleoperation and Remote Assistance?
    *   **A:** Teleop = Driving with a joystick (High latency risk). Assistance = Giving high-level commands ("Go Left") to the AI.

### Challenge Task
**Task:** Ambulance Preemption.
1.  Modify the Traffic Light script.
2.  Listen for `v2x/ambulance/approaching`.
3.  If received, force state to GREEN immediately.

---

## 📚 Further Reading & References
-   [5G Automotive Association (5GAA)](https://5gaa.org/)
-   [Uptane Security Framework](https://uptane.github.io/)

---

**Day 173 Complete** | Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career
