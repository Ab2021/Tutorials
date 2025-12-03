# Day 157: Hardware-in-the-Loop (HIL)
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 157 Focus:**
> SIL tests the code, but not the chip. **Hardware-in-the-Loop (HIL)** puts the code on the real ECU (Electronic Control Unit). The ECU thinks it's driving a car, but its wires are connected to a Simulator (dSPACE/NI) that fakes the sensors.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the HIL architecture (DUT vs Simulator).
2.  **Explain** Real-Time constraints in HIL.
3.  **Simulate** a HIL setup using a Raspberry Pi (ECU) and PC (Plant).
4.  **Inject** Hardware Faults (Cable disconnect, Voltage drop).
5.  **Validate** CAN bus communication timing.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Embedded Systems:** GPIO, UART/CAN.
-   **Networking:** UDP/TCP.

### Hardware Requirements
-   **Raspberry Pi (or Jetson):** Acts as the ECU.
-   **PC:** Acts as the Simulator.
-   **Ethernet Cable:** To connect them.

### Software Stack
-   **Python:** `socket` (for simple HIL comms).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The HIL Bench

-   **DUT (Device Under Test):** The ADAS Controller (e.g., NVIDIA Orin).
-   **HIL Simulator:** A powerful Real-Time PC (e.g., dSPACE Scalexio).
-   **I/O Interface:**
    -   **Digital/Analog:** Fakes wheel speeds, voltages.
    -   **Bus:** Fakes CAN/Ethernet messages from other ECUs (Restbus Simulation).

### 🔹 Part 2: Real-Time Requirement

The Simulator **MUST** be faster than the ECU.
-   If the ECU runs at 100Hz (10ms), the Simulator must calculate physics and send sensor data in < 10ms.
-   If the Simulator lags, the ECU will detect a "Sensor Timeout" and trigger a failsafe, invalidating the test.

### 🔹 Part 3: Fault Injection

HIL is the *only* safe way to test hardware failures:
-   **Short Circuit:** Short CAN_H to GND.
-   **Open Circuit:** Cut the Lidar wire.
-   **Packet Loss:** Drop 50% of CAN frames.
-   **Goal:** Verify the ECU detects the fault and enters Safe Mode.

---

## 💻 Implementation: Poor Man's HIL

**Scenario:**
-   **ECU (Raspberry Pi):** Runs the Lane Keep Assist (LKA) logic.
-   **Simulator (PC):** Runs the Vehicle Physics and Camera generation.
-   **Interface:** UDP over Ethernet (Simulating CAN).

### 🛠️ Setup
Create `week23_day157` and `hil_ecu.py` (for Pi) and `hil_sim.py` (for PC).

```bash
mkdir -p ~/ros2_ws/src/week23_day157
cd ~/ros2_ws/src/week23_day157
touch hil_ecu.py hil_sim.py
```

### 👨‍💻 Code: The ECU (Target)

Run this on the Raspberry Pi (or a separate terminal).

```python
import socket
import struct
import time

# ECU Configuration
IP = "0.0.0.0"
PORT = 5000

def lka_logic(lateral_error):
    # Simple P-Controller
    kp = -0.5
    steer_cmd = kp * lateral_error
    return max(min(steer_cmd, 1.0), -1.0) # Clip -1 to 1

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))
    print(f"ECU (LKA) listening on {PORT}...")
    
    while True:
        # 1. Receive Sensor Data (Lateral Error)
        data, addr = sock.recvfrom(1024)
        lat_error = struct.unpack('f', data)[0]
        
        # 2. Compute Control
        start_time = time.time()
        steer = lka_logic(lat_error)
        compute_time = time.time() - start_time
        
        # 3. Send Actuator Command
        msg = struct.pack('f', steer)
        sock.sendto(msg, addr)
        
        print(f"Err: {lat_error:.2f} -> Steer: {steer:.2f} (Time: {compute_time*1000:.2f}ms)")

if __name__ == "__main__":
    main()
```

### 👨‍💻 Code: The Simulator (Plant)

Run this on the PC.

```python
import socket
import struct
import time
import math

# Simulator Configuration
ECU_IP = "127.0.0.1" # Use Localhost for demo, or Pi IP
ECU_PORT = 5000
DT = 0.01 # 100Hz

def vehicle_model(y, steer, v=20.0):
    # Kinematic Bicycle Model (Lateral)
    # y_dot = v * sin(heading) approx v * heading
    # heading_dot = v / L * tan(steer)
    # Simplified: y_next = y + v * steer * dt (Very rough)
    y_next = y + v * steer * DT * 0.1 # Gain scaling
    return y_next

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1) # 100ms timeout
    
    y = 1.0 # Initial Lateral Error (1m off center)
    t = 0
    
    print("Starting HIL Simulation...")
    
    try:
        while t < 10.0:
            # 1. Send Sensor Data
            msg = struct.pack('f', y)
            sock.sendto(msg, (ECU_IP, ECU_PORT))
            
            # 2. Measure Round Trip Time
            start = time.time()
            
            try:
                # 3. Receive Control
                data, _ = sock.recvfrom(1024)
                steer = struct.unpack('f', data)[0]
                rtt = (time.time() - start) * 1000
                
                # 4. Update Physics
                y = vehicle_model(y, steer)
                
                print(f"T={t:.2f} | Y={y:.2f} | Steer={steer:.2f} | RTT={rtt:.2f}ms")
                
            except socket.timeout:
                print("TIMEOUT: ECU did not respond!")
                # Failsafe: Coast
                
            time.sleep(DT)
            t += DT
            
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Fault Injection

### Lab Objectives
1.  **Run Normal:**
    -   Start ECU. Start Sim.
    -   **Observation:** `Y` decreases to 0. The LKA works.
2.  **Inject Latency (Timing Fault):**
    -   Add `time.sleep(0.2)` in `hil_ecu.py` (Simulate heavy CPU load).
    -   **Result:** Simulator prints "TIMEOUT".
    -   **Lesson:** Real-time systems must meet deadlines.
3.  **Inject Noise (Sensor Fault):**
    -   In Sim, send `y + random.normal()`.
    -   **Result:** Steering jitters. Verify ECU filtering.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Network Jitter
**Symptom:** RTT varies wildly (1ms to 50ms).
**Cause:** Ethernet/WiFi is not deterministic.
**Solution:** Use **EtherCAT** or dedicated Real-Time Ethernet for professional HIL. For UDP, use a direct cable connection.

#### 2. Clock Drift
**Symptom:** Simulation runs faster/slower than real time.
**Cause:** `time.sleep(DT)` is inaccurate in Python (Windows/Linux).
**Solution:** Use a Real-Time OS (RTOS) or C++ `std::chrono` for the Simulator loop.

---

## ⚡ Optimization & Best Practices

### 1. FPGA Acceleration
For High-Fidelity HIL (e.g., Radar simulation).
-   CPU is too slow to generate raw Radar signals (GHz).
-   Use FPGA to generate RF signals in real-time.

### 2. Automated HIL
Integrate with Jenkins.
-   Jenkins flashes the ECU firmware.
-   Jenkins starts the dSPACE script.
-   Jenkins parses the report.
-   "Nightly HIL Run".

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why can't we just use SIL?
    *   **A:** SIL doesn't test the hardware (CAN transceivers, CPU thermal throttling, Memory bandwidth).
2.  **Q:** What is "Restbus Simulation"?
    *   **A:** Simulating the "Rest of the Bus". If you test the ADAS ECU, the HIL must pretend to be the Engine ECU, Brake ECU, and Dashboard to keep the ADAS ECU happy.
3.  **Q:** What happens if the HIL Simulator crashes?
    *   **A:** The ECU detects a loss of signal and should enter a safe state (e.g., disengage Autopilot).

### Challenge Task
**Task:** Watchdog Timer.
1.  Implement a Watchdog in the ECU.
2.  If no UDP packet received for 200ms, print "COMM LOST - SAFE STOP".
3.  Kill the Simulator script to test it.

---

## 📚 Further Reading & References
-   [dSPACE HIL Systems](https://www.dspace.com/en/pub/home/products/systems/hil_simulators.cfm)
-   [National Instruments HIL](https://www.ni.com/en-us/innovations/automotive/hardware-in-the-loop.html)

---

**Day 157 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
