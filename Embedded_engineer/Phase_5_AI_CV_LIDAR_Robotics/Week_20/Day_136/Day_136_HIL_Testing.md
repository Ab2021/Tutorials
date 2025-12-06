# Day 136: Hardware-in-Loop (HIL) Testing
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> Trick the Brain.
> - **Focus:** HIL vs SIL (Software-in-Loop), Connecting real embedded hardware (Jetson/ESP32) to a Simulation (Gazebo/Unity), Latency measurement, and validating Real-Time constraints.
> - **Code:** A Python bridge `hil_bridge.py` that connects a "Plant" (Simulated Motor) via UDP/Serial to a "Controller" (Pseudo-Embedded code), measuring the Round Trip Time (RTT).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between PIL (Processor-in-Loop), HIL (Hardware-in-Loop), and SIL (Software-in-Loop).
2.  **Configure** a HIL setup where the Autopilot (Pixelhawk/Jetson) thinks it's flying, but the IMU data comes from a USB cable.
3.  **Measure** the crucial latency budget (Sim $\to$ Transport $\to$ Embedded $\to$ Transport $\to$ Sim).
4.  **Implement** a UDP Bridge for high-speed HIL.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Laptop (Simulator).
- Optional: RasPi / Arduino (Device Under Test). We will simulate the DUT as a separate process if hardware is missing.

### Software Environment
```bash
pip install pyserial
```

### Prior Knowledge
- Serial/UDP Communication.
- Control Theory (Phase Margin reduction due to delay).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Trust Hierarchy

1.  **SIL (Software-in-Loop):** Run C++ Controller as a node *inside* the PC simulation. (Fastest, easy debug).
2.  **HIL (Hardware-in-Loop):** Run C++ Controller on the *actual robot computer*. Connect weird wires to the PC Sim.
    *   **Why?** Because the PC is an i9 CPU (3GHz). The Robot is an ARM Cortex-M4 (300MHz). Code that runs on PC might freeze the Robot.
3.  **Real:** Expensive crash risk.

### 🔹 Part 2: Timing is Everything

Dynamic System Stability depends on Loop Time.
*   Sim Time must wait for Hardware Time.
*   **Lockstep:** Sim pauses until it gets a command from HW. (Safe, but slows down time).
*   **Real-Time:** Sim runs freely. If HW is too slow, the Sim crashes (Virtual Crash). Validation!

---

## 💻 Implementation: The UDP Bridge

We simulate a "Plant" (Physics) on Port 5000 and a "Controller" (Embedded) on Port 5001.

### 🛠️ Project Structure
```text
day136_hil/
├── src/
│   ├── plant_sim.py   (The "World")
│   ├── embedded_controller.py (The "Brain")
└── output/
    ├── latency_log.txt
```

### 👨‍💻 The Plant (Simulator) - `src/plant_sim.py`

```python
import socket
import struct
import time
import math
import matplotlib.pyplot as plt

# Simulates a Motor Position
# Receives: Voltage (float)
# Sends: Position (float), Velocity (float)

HOST = '127.0.0.1'
PORT_RX = 5000
PORT_TX = 5001

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT_RX))
    sock.settimeout(0.01) # Non-blocking-ish
    
    print(f"Plant (Sim) listening on {PORT_RX}, sending to {PORT_TX}")
    
    # Physics State
    pos = 0.0
    vel = 0.0
    voltage = 0.0
    
    dt = 0.01
    t = 0
    history = []
    
    # Run loop
    try:
        while True:
            t_start = time.perf_counter()
            
            # 1. Receive Control Input (Simulating Hardware Driver)
            try:
                data, addr = sock.recvfrom(1024)
                # Unpack float (Voltage)
                voltage = struct.unpack('f', data)[0]
            except socket.timeout:
                pass # Hold previous voltage (Zero Order Hold)
            
            # 2. Physics Step (Motor Model)
            # Torque = k * V - b * w
            torque = 1.0 * voltage - 0.1 * vel
            acc = torque # Inertia = 1
            
            vel += acc * dt
            pos += vel * dt
            
            # 3. Send Sensor Data (Simulating Encoder)
            # Pack 2 floats: pos, vel
            packet = struct.pack('ff', pos, vel)
            sock.sendto(packet, (HOST, PORT_TX))
            
            # Log
            history.append(pos)
            
            # Real Time Regulation
            t_sim = time.perf_counter() - t_start
            if t_sim < dt:
                time.sleep(dt - t_sim)
                
            t += dt
            if t > 5.0: break
            
    except KeyboardInterrupt:
        pass
        
    plt.plot(history)
    plt.title("Plant Response (HIL)")
    plt.savefig("output/hil_response.png")

if __name__ == "__main__":
    main()
```

### 👨‍💻 The Controller (Embedded) - `src/embedded_controller.py`

Run this in a separate terminal.

```python
import socket
import struct
import time

HOST = '127.0.0.1'
PORT_RX = 5001 # Listening for Sensors
PORT_TX = 5000 # Sending Commands

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT_RX))
    
    print(f"Controller (Embedded) listening on {PORT_RX}")
    
    target = 10.0
    kp = 5.0
    kd = 1.0
    
    latencies = []
    
    try:
        while True:
            # Block until Sensor Data arrives (Interrupt driven)
            t_recv = time.perf_counter()
            data, addr = sock.recvfrom(1024)
            
            # Unpack
            pos, vel = struct.unpack('ff', data)
            
            # Control Law (PID)
            error = target - pos
            voltage = kp * error - kd * vel
            
            # Limit
            if voltage > 10: voltage = 10
            if voltage < -10: voltage = -10
            
            # Send Command
            packet = struct.pack('f', voltage)
            sock.sendto(packet, (HOST, PORT_TX))
            
            # Latency Measurement
            # Time spent in "Interrupt"
            t_calc = (time.perf_counter() - t_recv) * 1000 # ms
            latencies.append(t_calc)
            
            print(f"Pos: {pos:.2f} | Cmd: {voltage:.2f} | Latency: {t_calc:.3f}ms")
            
    except KeyboardInterrupt:
        import numpy as np
        print(f"\nAvg Compute Time: {np.mean(latencies):.3f} ms")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Lag Monster"

### 1. Lab Objectives
- **Run:** Start `embedded_controller.py`. Then start `plant_sim.py`.
- **Observe:** The system works. Latency is tiny (<1ms) because lookback is local.
- **Fail:** Add `time.sleep(0.05)` inside the Controller loop.
- **Result:** 50ms delay. The Plant response becomes oscillatory or unstable.
- **Lesson:** Delays reduce Phase Margin. HIL exposes "Code that is chemically correct but temporally wrong."

---

## 🚀 Project: "PX4 SITL bridge"

**Goal:** Connect PX4 Flight Stack to a custom Python Sim.
1.  **Protocol:** MAVLink (via UDP 14550).
2.  **Sim:** Send `HIL_SENSOR` messages (Gyro, Accel, Baro).
3.  **PX4:** Calculates attitude/thrust.
4.  **Sim:** Receives `HIL_ACTUATOR_CONTROLS` (Motor PWMs).
5.  **Result:** Validates the entire PX4 config (Mixers, PIDs) without crashing a drone.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Packet Loss"
*   **Cause:** UDP drops packets if buffer is full.
*   **Fix:** Ensure Sim and Controller run fast enough to drain buffers. Or use TCP (but TCP has Nagle Algorithm delay/jitter). UDP is preferred for Control.

#### 2. "Floating Point Endianness"
*   **Cause:** Sending raw bytes between ARM (Little Endian) and PowerPC (Big Endian) or different implementations.
*   **Fix:** Always use `struct` with standard endian (`<f` or `>f`).

---

## ⚡ Optimization: FPGA HIL

For high speed (100kHz) motor control HIL.
*   Simulate the Motor Model *on an FPGA* (National Instruments CompactRIO / Speedgoat).
*   PC Simulation is too slow (1kHz limit). FPGA gives $<1 \mu s$ latency.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just trust SIL?
    *   **A:** SIL doesn't check if the CPU is overloaded. You might have a memory leak or 100% CPU usage on the real board that SIL won't show.
2.  **Q:** What is "Real-Time Factor"?
    *   **A:** $RTF = T_{sim} / T_{real}$. If RTF < 1.0, Sim is slower than reality. For HIL, your PC must maintain RTF >= 1.0 or the embedded controller gets confused.
3.  **Q:** Can I HIL a localized Vision system?
    *   **A:** Hard. Rendering images and sending them over HDMI to the Jetson is slow. Usually we bypass the camera and inject Feature Lists or Object Detections directly (Sensor Bypass HIL).

### Challenge Task
> **Task:** Jitter Test.
> 1. In `plant_sim.py`, add random sleep `sleep(random(0, 0.02))`.
> 2. This simulates non-real-time OS jitter (Windows/Linux).
> 3. Observe effect on stability.

---

## 📚 Further Reading
- **PX4 Docs:** Hardware in the Loop simulation.
- **dSPACE:** Industrial HIL standards.

---

**Day 136 Complete**
