# Day 85: V2X Standards (DSRC, C-V2X)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 85 Focus:**
> Sensors (Camera, LiDAR) are limited by Line-of-Sight. They can't see around corners or through trucks. **V2X (Vehicle-to-Everything)** is the "X-Ray Vision" of ADAS. It allows cars to talk to each other (V2V) and to the infrastructure (V2I). Today, we compare the two rival standards: DSRC and C-V2X.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** DSRC (802.11p) and C-V2X (LTE/5G) technologies.
2.  **Define** the communication modes: V2V, V2I, V2P, V2N.
3.  **Analyze** the performance metrics: Latency (< 20ms) and Range (~300m).
4.  **Simulate** a V2V network using Python sockets.
5.  **Explain** the "Day 1" use cases: Emergency Brake Light, Intersection Assist.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Networking:** UDP/IP, Bandwidth, Latency.
-   **Radio Physics:** Frequency (5.9 GHz), Multipath.

### Hardware Requirements
-   **None:** Pure algorithm day. (Real V2X radios like Cohda Wireless are expensive).

### Software Stack
-   **Python:** `socket`, `threading`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Standards War

1.  **DSRC (Dedicated Short Range Communication):**
    -   **Base:** WiFi (IEEE 802.11p).
    -   **Pros:** Proven, low latency, decentralized (no SIM card needed).
    -   **Cons:** Limited range, older tech.
    -   **Status:** Being phased out in US/China in favor of C-V2X.
2.  **C-V2X (Cellular V2X):**
    -   **Base:** 4G LTE (Release 14) / 5G NR (Release 16).
    -   **Mode 4 (PC5):** Direct communication (like Walkie-Talkie). No cell tower needed.
    -   **Mode 3 (Uu):** Network communication (via Tower).
    -   **Pros:** Longer range, better non-line-of-sight, future proof (5G).

### 🔹 Part 2: Communication Types

-   **V2V (Vehicle-to-Vehicle):** "I am braking!" (Forward Collision Warning).
-   **V2I (Vehicle-to-Infrastructure):** "Light is Red." (SPaT - Signal Phase and Timing).
-   **V2P (Vehicle-to-Pedestrian):** Phone broadcasts "I am crossing."
-   **V2N (Vehicle-to-Network):** Cloud traffic updates (Waze style).

### 🔹 Part 3: The 5.9 GHz Spectrum

Governments reserved 75 MHz of spectrum at 5.9 GHz specifically for safety.
-   **Safety Channel:** Broadcasts BSM (Basic Safety Messages) 10 times per second (10Hz).
-   **Service Channel:** Map downloads, tolling.

---

## 💻 Implementation: V2V Simulator

**Scenario:**
-   **Car A (Sender):** Broadcasts its position and speed.
-   **Car B (Receiver):** Listens. If Car A brakes hard, Car B warns the driver.
-   **Network:** UDP Multicast (Simulating radio broadcast).

### 🛠️ Setup
Create `week13_day85` and `v2x_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day85
cd ~/ros2_ws/src/week13_day85
touch v2x_sim.py
```

### 👨‍💻 Code: V2V UDP Broadcast

```python
import socket
import struct
import time
import threading
import json
import random

# --- Configuration ---
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
IS_SENDER = True # Toggle this to run as Sender or Receiver

class V2XNode:
    def __init__(self, node_id, role):
        self.node_id = node_id
        self.role = role # SENDER or RECEIVER
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        if role == 'RECEIVER':
            self.sock.bind(('', MCAST_PORT))
            mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        self.running = True
        self.speed = 30.0 # m/s
        self.brakes_active = False

    def broadcast_bsm(self):
        # Basic Safety Message (BSM) Simulation
        while self.running:
            # Simulate Driving
            if random.random() < 0.1: # 10% chance to brake hard
                self.brakes_active = True
                self.speed -= 5.0
            else:
                self.brakes_active = False
                self.speed = min(30.0, self.speed + 1.0)
                
            msg = {
                'id': self.node_id,
                'ts': time.time(),
                'lat': 37.7749,
                'lon': -122.4194,
                'speed': self.speed,
                'brakes': self.brakes_active
            }
            
            data = json.dumps(msg).encode('utf-8')
            self.sock.sendto(data, (MCAST_GRP, MCAST_PORT))
            print(f"[Tx] ID:{self.node_id} | Spd:{self.speed:.1f} | Brake:{self.brakes_active}")
            
            time.sleep(0.1) # 10 Hz

    def receive_bsm(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))
                
                # Process Message
                latency = (time.time() - msg['ts']) * 1000.0
                
                print(f"[Rx] From {msg['id']} | Spd:{msg['speed']:.1f} | Latency:{latency:.1f}ms")
                
                if msg['brakes']:
                    print(">>> WARNING: CAR AHEAD BRAKING! <<<")
                    
            except Exception as e:
                print(e)

    def start(self):
        if self.role == 'SENDER':
            t = threading.Thread(target=self.broadcast_bsm)
            t.start()
        else:
            t = threading.Thread(target=self.receive_bsm)
            t.start()
            
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.running = False

def main():
    # To test: Run this script in two separate terminals.
    # Terminal 1: python v2x_sim.py (Sender)
    # Terminal 2: Modify code to IS_SENDER = False, then run (Receiver)
    
    # For this single-file demo, we will spawn threads for both
    print("Starting V2X Simulation (Sender + Receiver in one process)...")
    
    sender = V2XNode("Car_A", "SENDER")
    receiver = V2XNode("Car_B", "RECEIVER")
    
    t1 = threading.Thread(target=sender.broadcast_bsm)
    t2 = threading.Thread(target=receiver.receive_bsm)
    
    t1.start()
    t2.start()
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
        
    sender.running = False
    receiver.running = False
    t1.join()
    t2.join()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Emergency Brake

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Car A sends BSMs at 10Hz.
    -   Car B receives them.
    -   Occasionally, Car A brakes (`brakes: true`).
    -   Car B prints `>>> WARNING: CAR AHEAD BRAKING! <<<`.
3.  **Latency Check:**
    -   Observe the latency printout. In localhost, it's < 1ms.
    -   In real DSRC, it's ~5ms.
    -   In Cloud (V2N), it would be ~100ms (Too slow for braking!). This proves why we need Direct V2V.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Multicast Blocked
**Symptom:** Receiver gets nothing.
**Cause:** Firewall or Router blocking UDP Multicast.
**Solution:** Allow UDP port 5007. Or use `127.0.0.1` (Unicast) for local testing.

#### 2. Packet Loss
**Symptom:** Messages skipped.
**Cause:** UDP is unreliable. Radio interference.
**Solution:** V2X apps must be robust to packet loss. Don't wait for retransmission (there is no ACK in broadcast). Just wait for the next BSM (0.1s later).

---

## ⚡ Optimization & Best Practices

### 1. Congestion Control
If 1000 cars are in a jam, and all talk at 10Hz, the channel jams.
-   **DCC (Decentralized Congestion Control):** If channel load is high, reduce rate to 2Hz or reduce power (range).

### 2. Security
How do I know "Car A" is real and not a hacker spoofing a crash?
-   **PKI (Public Key Infrastructure):** Every BSM is signed with a digital certificate.
-   Certificates change every 5 minutes to prevent tracking (Privacy).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why use 5.9 GHz and not 2.4 GHz (WiFi)?
    *   **A:** 5.9 GHz is a licensed band (less interference). 2.4 GHz is crowded (Microwaves, Bluetooth).
2.  **Q:** What is the difference between Uu and PC5 interfaces in C-V2X?
    *   **A:** **PC5** is Direct (Car-to-Car). **Uu** is Network (Car-to-Tower).
3.  **Q:** Can V2X replace Radar?
    *   **A:** No. V2X only works if the *other* car has V2X. Radar works on everything. They are complementary.

### Challenge Task
**Task:** Intersection Collision Warning.
1.  Simulate Car A moving North, Car B moving East.
2.  Calculate "Time to Collision" (TTC) based on position and speed in BSM.
3.  If TTC < 3s, trigger warning.

---

## 📚 Further Reading & References
-   [5GAA (5G Automotive Association)](https://5gaa.org/)
-   [OmniAir Consortium (Certification)](https://omniair.org/)

---

**Day 85 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
