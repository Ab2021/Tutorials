# Day 162: Threat Modeling (STRIDE)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> Think like a hacker.
> - **Focus:** The STRIDE model (Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation), Attack Surfaces in Robotics (CAN Bus, Sensors, WiFi), and Bus Arbitration logic.
> - **Code:** A Python script `can_attack.py` simulating a CAN bus. Implement a "Replay Attack" (Record steering command, replay it later) and a "DoS Attack" (Inject ID 0x00 messages to hog the bus).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Analyze** a robotic system using the STRIDE framework.
2.  **Explain** why the CAN Bus is inherently insecure (No encryption, No authentication).
3.  **Simulate** a Replay Attack on a critical control message.
4.  **Implement** a basic specialized firewall (CAN Gateway) to filter malicious IDs.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (In real life: CANable, PeakCAN).

### Software Environment
```bash
pip install python-can
# Or just standard library for simulation
```

### Prior Knowledge
- Hexadecimal.
- Basic Networking.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: STRIDE for Robots

1.  **Spoofing:** "I am the Lidar." (Fake data).
2.  **Tampering:** "Speed is 100." -> Man-In-The-Middle -> "Speed is 0."
3.  **Repudiation:** "I didn't crash." (Deleting logs).
4.  **Information Disclosure:** "Here is the map of the secret base." (Data leak).
5.  **Denial of Service:** "Ping Ping Ping..." (Battery drain / CPU freeze).
6.  **Elevation of Privilege:** Guest WiFi -> Root Access on Drive By Wire.

### 🔹 Part 2: The CAN Bus Vulnerability

Controller Area Network (CAN) connects the Brakes, Steering, and Engine.
*   **Broadcast:** Everyone hears everything.
*   **Arbitration:** Lowest ID wins. ID `0x000` overrides `0x100`.
*   **No Auth:** If you plug into the OBD-II port, you are God.

---

## 💻 Implementation: The CAN Hack

We simulate a bus with an `EngineECU` (Sender) and `Dashboard` (Receiver).
The `Attacker` sniffs and injects messages.

### 🛠️ Project Structure
```text
day162_security/
├── src/
│   ├── can_attack.py
└── output/
    ├── bus_log.txt
```

### 👨‍💻 CAN Simulation (`src/can_attack.py`)

```python
import time
import threading
import queue
import random

# Message Structure: ID (11-bit), Data (8 bytes)
class CANMessage:
    def __init__(self, arbitration_id, data):
        self.arbitration_id = arbitration_id
        self.data = data
        self.timestamp = time.time()
        
    def __str__(self):
        return f"ID: 0x{self.arbitration_id:03X} | Data: {self.data.hex().upper()}"

class CANBus:
    def __init__(self):
        self.bus = queue.PriorityQueue()
        # Priority Queue mimics Arbitration (Lowest ID first)
        
    def send(self, msg):
        self.bus.put((msg.arbitration_id, msg))
        
    def listen(self):
        if not self.bus.empty():
            return self.bus.get()[1]
        return None

class EngineECU:
    def __init__(self, bus):
        self.bus = bus
        self.rpm = 1000
        self.running = True
        
    def loop(self):
        while self.running:
            # ID 0x1AB = Engine RPM
            data = int(self.rpm).to_bytes(2, 'big') + b'\x00'*6
            msg = CANMessage(0x1AB, data)
            self.bus.send(msg)
            
            self.rpm += random.randint(-10, 20)
            time.sleep(0.5) # 2 Hz

class Attacker:
    def __init__(self, bus):
        self.bus = bus
        self.recorded_msgs = []
        
    def sniff(self, duration=2.0):
        print("[HACKER] Sniffing bus...")
        start = time.time()
        while time.time() - start < duration:
            # In a real sim, we'd peep without removing. 
            # Here, we assume we are just tapping the wire.
            pass
            
    def replay_attack(self):
        print("\n[HACKER] Executing REPLAY ATTACK (Engine Off Spoof)...")
        # Fake 0 RPM message
        # We record nothing, we just forge it (Spoofing)
        # OR Replay: We recorded a "0 RPM" message when car was off yesterday
        
        fake_data = b'\x00\x00\x00\x00\x00\x00\x00\x00' # 0 RPM
        msg = CANMessage(0x1AB, fake_data)
        
        # Flooding
        for _ in range(5):
            self.bus.send(msg)
            time.sleep(0.1)

    def dos_attack(self):
        print("\n[HACKER] Executing DoS ATTACK (Priority Override)...")
        # ID 0x000 (Highest Priority possible)
        zero_msg = CANMessage(0x000, b'\xFF'*8)
        for _ in range(10):
            self.bus.send(zero_msg)
            # No sleep. Flood.

class Dashboard:
    def __init__(self, bus):
        self.bus = bus
        self.running = True
        
    def loop(self):
        print("Dashboard Active.")
        while self.running:
            msg = self.bus.listen()
            if msg:
                if msg.arbitration_id == 0x1AB:
                    rpm = int.from_bytes(msg.data[:2], 'big')
                    print(f"[DASH] Engine RPM: {rpm}")
                elif msg.arbitration_id == 0x000:
                    print(f"[DASH] SYSTEM ERROR! (DoS Detected)")
                else:
                    print(f"[DASH] Unknown Msg: {msg}")
            time.sleep(0.1)

def main():
    bus = CANBus()
    
    engine = EngineECU(bus)
    dash = Dashboard(bus)
    hacker = Attacker(bus)
    
    # Start Threads
    t1 = threading.Thread(target=engine.loop)
    t2 = threading.Thread(target=dash.loop)
    
    t1.start()
    t2.start()
    
    time.sleep(2)
    
    # Attack 1: Replay/Spoof
    hacker.replay_attack()
    
    time.sleep(2)
    
    # Attack 2: DoS
    hacker.dos_attack()
    
    time.sleep(2)
    
    engine.running = False
    dash.running = False
    t1.join()
    t2.join()
    print("Sim Finished.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Countermeasure"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Capabilities of the Hacker to alter Dashboard readings.
- **Modify:** Implement a Rolling Counter.
- **Engine:** Add a byte that increments $0 \to 255$.
- **Dashboard:** If `new_counter <= old_counter`, reject message.
- **Result:** Replay attacks fail (Old counter value is rejected). Spoofing only works if Hacker guesses the next counter.

---

## 🚀 Project: "Intrusion Detection System (IDS)"

**Goal:** Detect anomalies.
1.  **Metric:** Frequency of ID `0x1AB`. Normal = 2Hz.
2.  **Attack:** If Frequency > 10Hz, raise "Flooding Alert".
3.  **Metric:** Data Range. Normal RPM = $0-8000$.
4.  **Attack:** If RPM = 65535, raise "Sensor Fault/Hack".
5.  **Task:** Write a Python Daemon that monitors `bus` and prints alerts.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Arbitration Simulation"
*   **Cause:** Python Queue isn't exactly real-time wire-OR logic.
*   **Logic:** In real CAN, if two nodes transmit, the one writing `0` when other writes `1` wins. The loser stops transmitting immediately.
*   **Sim:** `PriorityQueue` approximates this by serving Lower ID first.

#### 2. "Bus Off"
*   **Cause:** Too many transmit errors.
*   **Result:** Node disconnects itself.
*   **Sim:** Not modeled here.

---

## ⚡ Optimization: Gateway Firewall

Physical Segmentation.
*   **Architecture:** Put the Head Unit (Bluetooth/WiFi) on `Bus A`. Put the Brakes on `Bus B`. Connect them via a Gateway.
*   **Gateway Rule:** "Allow Music metadata from A to B. BLOCK Steering commands from A to B."
*   **Result:** Even if WiFi is hacked, Brakes are safe.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Replay Attack effective?
    *   **A:** Because standard CAN messages are identical every time (Static). "Door Open" is always `0x123 [01]`. If you record it, you can open the door later.
2.  **Q:** What is SecOC?
    *   **A:** Secure Onboard Communication. Adds a CMAC (Authenticator) to the CAN frame. Requires keys.
3.  **Q:** What is the "Jeep Hack" (2015)?
    *   **A:** Miller & Valasek hacked the Entertaintment System via Sprint Cellular, pivoted to the CAN bus, and disabled the brakes remotely.

### Challenge Task
> **Task:** Time-based Authentication.
> 1. Use `timestamp` in message payload.
> 2. If `msg.time < now - 1.0s`, reject it.
> 3. Requires clock sync!

---

## 📚 Further Reading
- **Car Hacking Handbook:** The bible of automotive security.
- **Miller & Valasek Whitepaper:** Analyzing the attack surfaces of the Jeep Cherokee.

---

**Day 162 Complete**
