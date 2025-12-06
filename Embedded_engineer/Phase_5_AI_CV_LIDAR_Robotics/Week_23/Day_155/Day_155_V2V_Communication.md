# Day 155: V2V Communication (DSRC/C-V2X)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> Don't guess. Ask the other car.
> - **Focus:** DSRC (802.11p) vs C-V2X (Cellular), The Basic Safety Message (BSM), Latency requirements (<100ms), and preventing "Hidden Node" collisions.
> - **Code:** A Python script `v2v_radio.py` that simulates a UDP Broadcast network. Two "Cars" exchange BSM packets (Pos, Vel, Brake Status), triggering a warning if a collision is imminent.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** DSRC (WiFi-based, short range) and C-V2X (LTE/5G based, long range/direct).
2.  **Structure** a BSM (Basic Safety Message) according to SAE J2735.
3.  **Implement** a Collision Warning algorithm based on received V2V data.
4.  **Simulate** Packet Loss and Latency in a V2V network.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (In real life: OBU - On Board Unit like Cohda Wireless).

### Software Environment
```bash
pip install numpy
```

### Prior Knowledge
- Networking (UDP/IP).
- Kinematics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: V2X Taxonomy

*   **V2V (Vehicle-to-Vehicle):** "I am here, braking hard!" (Collision Avoidance).
*   **V2I (Vehicle-to-Infrastructure):** "Traffic Light is Red in 5s." (SPaT - Signal Phase & Timing).
*   **V2P (Vehicle-to-Pedestrian):** "Phone detects car approaching."

### 🔹 Part 2: The Standards War

*   **DSRC (802.11p):** The old guard. Verified, robust, no subscription fee. Range ~300m.
*   **C-V2X (PC5/Uu):** The new challenger (Qualcomm). Uses 5G radio. Longer range, better non-line-of-sight. PC5 mode works without cell towers.

### 🔹 Part 3: The BSM (Basic Safety Message)

Broadcasted at 10Hz.
*   **Core Data:**
    *   Latitude, Longitude, Elevation.
    *   Speed, Heading.
    *   Size (Length/Width).
    *   Brake Status (ABS Active?).

---

## 💻 Implementation: The Digital Radio

We simulate 2 cars talking over UDP Multicast (Loopback).

### 🛠️ Project Structure
```text
day155_v2v/
├── src/
│   ├── v2v_radio.py
└── output/
    ├── collision_log.txt
```

### 👨‍💻 V2V Simulation (`src/v2v_radio.py`)

```python
import socket
import struct
import time
import threading
import json
import math

# Configuration
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

class CarNode:
    def __init__(self, car_id, start_x, start_y, vel_x):
        self.id = car_id
        self.x = start_x
        self.y = start_y
        self.vx = vel_x
        self.vy = 0.0
        self.brake_active = False
        self.running = True
        
        # Network
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        
        # Receiver
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind(('', MCAST_PORT))
        mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
        self.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.recv_sock.settimeout(0.1)

    def tx_loop(self):
        while self.running:
            # Create BSM (JSON for simplicity, Binary in real life)
            bsm = {
                "id": self.id,
                "x": self.x,
                "y": self.y,
                "vx": self.vx,
                "brake": self.brake_active,
                "ts": time.time()
            }
            msg = json.dumps(bsm).encode('utf-8')
            self.sock.sendto(msg, (MCAST_GRP, MCAST_PORT))
            time.sleep(0.1) # 10 Hz

    def rx_loop(self):
        while self.running:
            try:
                data, addr = self.recv_sock.recvfrom(1024)
                bsm = json.loads(data.decode('utf-8'))
                
                if bsm['id'] != self.id:
                    self.process_v2v(bsm)
                    
            except socket.timeout:
                pass

    def process_v2v(self, bsm):
        # Collision Warning Logic
        # Calculate TTC (Time to Collision)
        dx = bsm['x'] - self.x
        dv = bsm['vx'] - self.vx
        
        # If cars are approaching (dv < 0 if bsm is slower/stopped ahead)
        if abs(dx) < 50.0: # Close range
            if bsm['brake']:
                print(f"[{self.id}] WARNING: Car {bsm['id']} ahead is BRAKING! Dist={dx:.1f}")
                
            if dv != 0:
                ttc = -dx / dv
                if 0 < ttc < 3.0:
                    print(f"[{self.id}] CRITICAL: Forward Collision Warning! TTC={ttc:.1f}s")

    def update_physics(self):
        # Move car
        self.x += self.vx * 0.1

    def start(self):
        threading.Thread(target=self.tx_loop).start()
        threading.Thread(target=self.rx_loop).start()

def main():
    # Scenario:
    # Car A: x=0, v=20 (Fast)
    # Car B: x=100, v=10 (Slow) -> Then Brakes
    
    car_a = CarNode("Car_A", start_x=0.0, start_y=0.0, vel_x=20.0)
    car_b = CarNode("Car_B", start_x=100.0, start_y=0.0, vel_x=10.0)
    
    car_a.start()
    car_b.start()
    
    print("Simulation Started. Cars moving...")
    
    try:
        for t in range(50): # 5 seconds
            car_a.update_physics()
            car_b.update_physics()
            
            # Event: At t=20, Car B hits brakes
            if t == 20:
                print("\n>>> EVENT: Car B Slams Brakes! <<<\n")
                car_b.brake_active = True
                car_b.vx = 0.0 # Instant stop for simplicity
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        car_a.running = False
        car_b.running = False
        print("Simulation Stopped.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Packet Loss"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Car A detects Car B braking immediately.
- **Fail:** In `tx_loop`, add `if random() < 0.5: continue`. (50% packet loss).
- **Result:** Warning might be delayed by 0.2 - 0.5s.
- **Criticality:** At 30m/s, 0.5s delay = 15m blind travel.
- **Mitigation:** High Repetition Rate (10-20Hz) ensures *some* packet gets through.

---

## 🚀 Project: "Emergency Electronic Brake Light (EEBL)"

**Goal:** See through walls.
1.  **Scenario:** Car A follows Car B follows Car C.
2.  **Fog/Truck:** Car A cannot see Car C (Occluded).
3.  **Action:** Car C brakes hard. Sends BSM.
4.  **Reaction:** Car A receives BSM from C *before* Car B even reacts.
5.  **Task:** Simulate a 3-car chain and time the reaction of A with vs without V2V.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Localhost Multicast"
*   **Cause:** OS firewall blocks UDP Multicast.
*   **Fix:** Allow Python in firewall or use `127.0.0.1` Unicast for testing.

#### 2. "GPS Drift"
*   **Cause:** Car A reports X=100 (+/- 5m). Car B thinks A is in the left lane.
*   **Fix:** Filter V2V positions with a particle filter or match to HD Map lanes.

---

## ⚡ Optimization: Congestion Control

If 1000 cars jam a highway, 1000 BSMs/sec crashes the WiFi channel.
*   **DCC (Decentralized Congestion Control):**
    *   If Channel Load > 60%, reduce BSM rate to 2Hz.
    *   Reduce Tx Power (Talk only to neighbors).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use Cloud (4G)?
    *   **A:** Latency. Cloud RTT is ~100ms. DSRC is ~2ms. For emergency braking, ms matter. Also, dead zones.
2.  **Q:** Security?
    *   **A:** What if a hacker sends "Fake Brakes"? V2X uses PKI Certificates. Every message is signed. Cars verify signatures before trusting data.
3.  **Q:** Privacy?
    *   **A:** BSMs contain no License Plate or VIN. ID is a random MAC address that rotates every 5 minutes.

### Challenge Task
> **Task:** Intersection Assist (IMA).
> 1. Car A approaching N-S.
> 2. Car B approaching E-W.
> 3. Calculate "Distance to Intersection".
> 4. If both arrive at same time, Warn.

---

## 📚 Further Reading
- **SAE J2735:** DSRC Message Dictionary.
- **5GAA:** 5G Automotive Association whitepapers.

---

**Day 155 Complete**
