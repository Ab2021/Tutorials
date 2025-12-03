# Day 97: Hardware-in-the-Loop (HIL)
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 97 Focus:**
> Simulation on a PC is "Software-in-the-Loop" (SIL). But code running on an i9 CPU behaves differently than on an embedded ECU (Jetson/Raspberry Pi). **Hardware-in-the-Loop (HIL)** connects the real ECU to the simulator, tricking it into thinking it's driving a real car.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between SIL (Software), HIL (Hardware), and VIL (Vehicle) testing.
2.  **Architect** a HIL setup: Simulator <-> CAN Interface <-> ECU.
3.  **Simulate** CAN Bus messages (Speed, RPM) from CARLA.
4.  **Implement** a bridge to send Steering Commands from ECU to CARLA.
5.  **Analyze** the impact of latency and jitter in HIL.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 21:** CAN Bus (SocketCAN).
-   **Day 93:** CARLA API.

### Hardware Requirements
-   **Embedded Board:** Raspberry Pi or Jetson Nano (The "ECU").
-   **CAN Adapter:** USB-to-CAN (e.g., PEAK) or Virtual CAN (`vcan`).

### Software Stack
-   **Python:** `python-can`.
-   **Linux:** `can-utils`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The HIL Loop

1.  **Simulator (PC):** Renders the world, calculates physics.
    -   Output: "Current Speed is 50 km/h".
2.  **Interface (CAN):** Converts "50 km/h" into a CAN Frame (`ID 0x100, Data 0x32`).
3.  **ECU (Embedded):** Receives CAN Frame. Runs control logic (ACC/LKA).
    -   Output: "Steer +5 degrees".
4.  **Interface (CAN):** Converts CAN Frame (`ID 0x200, Data 0x05`) into API call.
5.  **Simulator (PC):** Applies steering to the virtual car.

### 🔹 Part 2: Real-Time Constraints

In SIL, if the CPU is slow, the simulation clock slows down.
In HIL, **Time is Real**. The ECU clock runs at 1 second per second.
-   **Requirement:** The Simulator MUST run in Real-Time (or faster).
-   **Synchronization:** If Simulator lags, the ECU will detect a "CAN Timeout" and trigger a fault.

---

## 💻 Implementation: Virtual HIL Bridge

**Scenario:**
-   **PC (Simulator):** Runs CARLA. Sends Speed on `vcan0`.
-   **ECU (Script):** Listens on `vcan0`. Sends Steering command back.
-   **Bridge:** Connects CARLA API to `vcan0`.

### 🛠️ Setup
1.  Setup Virtual CAN on Linux:
    ```bash
    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0
    ```
2.  Create `week14_day97` and `hil_bridge.py`.

```bash
mkdir -p ~/ros2_ws/src/week14_day97
cd ~/ros2_ws/src/week14_day97
touch hil_bridge.py
touch ecu_firmware.py
```

### 👨‍💻 Code: The Bridge (PC Side)

```python
import carla
import can
import struct
import time
import threading

# --- CAN Configuration ---
# ID 0x100: Vehicle Status (Speed) [Sender: Sim]
# ID 0x200: Control Command (Steer, Throttle) [Sender: ECU]

class HILBridge:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.bus = can.interface.Bus(channel='vcan0', bustype='socketcan')
        self.running = True
        
    def send_status_loop(self):
        while self.running:
            # Get Speed
            v = self.vehicle.get_velocity()
            speed_ms = (v.x**2 + v.y**2 + v.z**2)**0.5
            speed_kmh = speed_ms * 3.6
            
            # Pack CAN Frame (ID 0x100, 4 bytes float)
            data = struct.pack('>f', speed_kmh)
            msg = can.Message(arbitration_id=0x100, data=data, is_extended_id=False)
            
            try:
                self.bus.send(msg)
            except can.CanError:
                pass
                
            time.sleep(0.02) # 50Hz

    def receive_control_loop(self):
        while self.running:
            msg = self.bus.recv(timeout=1.0)
            if msg and msg.arbitration_id == 0x200:
                # Unpack (Steer: float, Throttle: float)
                steer, throttle = struct.unpack('>ff', msg.data)
                
                # Apply to CARLA
                control = carla.VehicleControl()
                control.steer = steer
                control.throttle = throttle
                self.vehicle.apply_control(control)
                # print(f"Applied Control: S={steer:.2f} T={throttle:.2f}")

    def start(self):
        t1 = threading.Thread(target=self.send_status_loop)
        t2 = threading.Thread(target=self.receive_control_loop)
        t1.start()
        t2.start()
        
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            t1.join()
            t2.join()

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Spawn Ego
    bp = world.get_blueprint_library().filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    
    print("HIL Bridge Started. Connect your ECU now.")
    
    bridge = HILBridge(vehicle)
    bridge.start()
    
    vehicle.destroy()

if __name__ == "__main__":
    main()
```

### 👨‍💻 Code: The ECU Firmware (Embedded Side)

```python
import can
import struct
import time

# Simple Lane Keep Assist (Mock)
# If Speed > 10, Steer slightly left (Circle)

def main():
    bus = can.interface.Bus(channel='vcan0', bustype='socketcan')
    print("ECU Started. Waiting for Vehicle Status...")
    
    target_speed = 30.0
    
    while True:
        msg = bus.recv(timeout=1.0)
        if msg and msg.arbitration_id == 0x100:
            # Decode Speed
            speed_kmh = struct.unpack('>f', msg.data)[0]
            print(f"ECU Rx: Speed = {speed_kmh:.1f} km/h")
            
            # Control Logic (P-Controller for Cruise Control)
            error = target_speed - speed_kmh
            throttle = max(0.0, min(1.0, 0.5 + 0.1 * error))
            
            # Steer Logic (Open Loop Circle)
            steer = -0.1 # Turn Left
            
            # Send Control
            data = struct.pack('>ff', steer, throttle)
            cmd_msg = can.Message(arbitration_id=0x200, data=data, is_extended_id=False)
            bus.send(cmd_msg)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Virtual Drive

### Lab Objectives
1.  Start CARLA.
2.  Run `python3 hil_bridge.py` (This acts as the car interface).
3.  Run `python3 ecu_firmware.py` (This acts as the ECU).
4.  **Observation:**
    -   The ECU prints the speed received from CARLA.
    -   The CARLA car starts moving (Throttle applied by ECU).
    -   The car drives in a circle (Steer applied by ECU).
5.  **Latency Test:**
    -   Add `time.sleep(0.1)` in the ECU loop.
    -   **Result:** The control loop becomes unstable. The car might oscillate or crash. This demonstrates the importance of low latency in HIL.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No Buffer Space Available"
**Symptom:** CAN send fails.
**Cause:** `vcan` buffer is full because no one is reading, or reading too slow.
**Solution:** `sudo ip link set vcan0 txqueuelen 1000`. Or ensure the receiver is running.

#### 2. Synchronization Drift
**Symptom:** ECU thinks 10s passed, Simulator thinks 8s passed.
**Cause:** Simulator running < Real Time.
**Solution:**
    -   Lower graphics settings.
    -   Use **Synchronous Mode** in CARLA, but this requires the Bridge to "Tick" the world, effectively slaving the Simulator to the ECU clock (or a master clock).

---

## ⚡ Optimization & Best Practices

### 1. FPGA Bridge
Python is too slow for high-frequency HIL (e.g., Engine Control at 1kHz).
-   **Solution:** Use an FPGA (Field Programmable Gate Array) to handle CAN I/O and buffering.
-   The PC sends a UDP packet to the FPGA. The FPGA blasts it out as CAN frames with microsecond precision.

### 2. Sensor Injection
Sending raw Camera images over CAN is impossible (Bandwidth).
-   **Direct Injection:** Connect the ECU's HDMI Input to the PC's GPU Output.
-   The ECU "sees" the simulated world as if it were a camera feed.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of HIL over SIL?
    *   **A:** It validates the **Hardware** (ECU processor, CAN transceiver, thermal issues) and the **Real-Time** performance of the software.
2.  **Q:** Can I use `vcan` for real HIL?
    *   **A:** No. `vcan` is internal to Linux. For real HIL, you need a physical USB-to-CAN adapter connected to the physical ECU pins.
3.  **Q:** What happens if the Simulator crashes during HIL?
    *   **A:** The ECU sees "Sensor Timeout" and should enter Safe Mode (e.g., Emergency Brake). This tests the ECU's fault handling.

### Challenge Task
**Task:** Emergency Brake HIL.
1.  Add a "Collision Sensor" to CARLA.
2.  Send `Collision=True` (ID 0x300) when hit.
3.  Update ECU: If `Collision` received, set `Throttle=0`, `Brake=1`.

---

## 📚 Further Reading & References
-   [dSPACE HIL Systems](https://www.dspace.com/en/inc/home/products/systems/hil_simulators.cfm)
-   [SocketCAN Documentation](https://www.kernel.org/doc/html/latest/networking/can.html)

---

**Day 97 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
