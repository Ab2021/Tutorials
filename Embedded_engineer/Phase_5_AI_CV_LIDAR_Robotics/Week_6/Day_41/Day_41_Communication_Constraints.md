# Day 41: Communication Constraints (Mesh Networks)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> Robots assume infinite Wifi. In caves, Wifi is zero.
> - **Focus:** Ad-Hoc Mesh Networks, Delay Tolerant Networking (DTN), and Bandwidth Allocation.
> - **Code:** Simulating a Relay Chain where Robot B passes messages from A to C.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the difference between Infrastructure Mode (Router) and Ad-Hoc Mesh (Peer-to-Peer).
2.  **Simulate** Packet Loss and Latency using `tc` (Traffic Control).
3.  **Implement** a Store-and-Forward (DTN) mechanism for disconnected robots.
4.  **Design** a "Relay Chain" behavior to maintain connectivity in a long tunnel.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Multiple WiFi Adapters (Optional).

### Software Environment
```bash
# On Linux
sudo apt install batctl # BATMAN-adv mesh
pip install simpy # Discrete Event Simulation
```

### Prior Knowledge
- TCP/IP vs UDP.
- ROS 2 QoS (Reliability, History).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Mesh Networks (BATMAN / OLSR)

In a cave, there is no Router.
**Mesh:** Every node is a router.
*   **Routing:** How does Robot A know that Robot C is reachable via Robot B?
*   **Protocols:**
    *   **OLSR (Optimized Link State Routing):** Proactive. Everyone floods routing tables. High overhead, low latency.
    *   **AODV (Ad-hoc On-demand):** Reactive. "Who knows where C is?" ... "I do". Low overhead, high setup latency.
    *   **BATMAN (Better Approach To Mobile Ad-hoc Networking):** Layer 2 Mesh. Looks like a giant virtual ethernet switch.

### 🔹 Part 2: Communication Models

1.  **Disc Model:** $P(Rx) = 1$ if $d < R$, else $0$. (Too simple).
2.  **Log-Normal Shadowing:** Signal strength decays with distance + random noise (walls).
3.  **Packet Loss:** $P(Loss) \propto \text{Congestion}$.

### 🔹 Part 3: Delay Tolerant Networking (DTN)

If Robot A explores a dead zone, it loses connection.
**Store-and-Forward:**
1.  Robot A buffers data (Map).
2.  Robot A returns to comms range.
3.  Robot A syncs buffer with Robot B.
4.  Robot B carries data to Base Station.
*   "Data Mules".

---

## 💻 Implementation: Relay Chain Simulation

Scenario: Operator -> Relay 1 -> Relay 2 -> Explorer.
We simulate the "Chain Break" and recovery.

### 🛠️ Project Structure
```text
day41_comms/
├── src/
│   ├── node.py
│   ├── network_sim.py
└── run_chain.py
```

### 👨‍💻 Network Simulator (`src/network_sim.py`)

Using `simpy` to simulate transmission time and range checks.

```python
import simpy
import random
import math

class WirelessMedium:
    def __init__(self, env):
        self.env = env
        self.nodes = []
        self.range = 50.0 # meters
        
    def register(self, node):
        self.nodes.append(node)
        
    def broadcast(self, sender, packet):
        # Determine who hears it
        for receiver in self.nodes:
            if receiver == sender: continue
            
            dist = math.dist(sender.pos, receiver.pos)
            if dist <= self.range:
                # Add Latency (Speed of Light + Processing) + Jitter
                latency = 0.01 + random.uniform(0, 0.005)
                self.env.process(self.deliver(receiver, packet, latency))
                
    def deliver(self, receiver, packet, delay):
        yield self.env.timeout(delay)
        receiver.receive(packet)

class Node:
    def __init__(self, env, medium, name, pos):
        self.env = env
        self.medium = medium
        self.name = name
        self.pos = pos
        self.inbox = []
        self.medium.register(self)
        
    def send(self, data):
        print(f"[{self.env.now:.2f}] {self.name} sending: {data}")
        self.medium.broadcast(self, {'from': self.name, 'data': data})
        
    def receive(self, packet):
        print(f"[{self.env.now:.2f}] {self.name} received from {packet['from']}: {packet['data']}")
        self.inbox.append(packet)
```

### 👨‍💻 Simulation Logic (`run_chain.py`)

```python
import simpy
from src.network_sim import WirelessMedium, Node

def relay_logic(env, node):
    while True:
        if node.inbox:
            packet = node.inbox.pop(0)
            # Retransmit if meant for downstream
            # Simple Flood Logic: If I haven't seen this ID, re-send
            yield env.timeout(0.5) # Processing delay
            node.send(f"Relayed: {packet['data']}")
        else:
            yield env.timeout(0.1)

env = simpy.Environment()
medium = WirelessMedium(env)

# Chain: Base (0,0) -> R1 (40,0) -> R2 (80,0) -> Explorer (120,0)
base = Node(env, medium, "Base", [0, 0])
r1 = Node(env, medium, "Relay1", [40, 0])
r2 = Node(env, medium, "Relay2", [80, 0])
expl = Node(env, medium, "Explorer", [120, 0])

# Start Processes
env.process(relay_logic(env, r1))
env.process(relay_logic(env, r2))

def mission(env):
    yield env.timeout(1.0)
    base.send("Mission Start")
    
    # 2. Explorer moves out of range (simulate movement)
    yield env.timeout(5.0)
    expl.pos = [200, 0] # Break link
    r2.send("Ping Explorer?") # Won't reach

env.process(mission(env))
env.run(until=10.0)
```

### 3. Expected Output
1.  **Time 1.0:** Base sends. R1 hears it (dist=40). R2/Expl don't.
2.  **Time 1.5:** R1 relays. R2 hears it (dist=40 from R1). Base hears echo.
3.  **Time 2.0:** R2 relays. Expl hears it.
4.  **Time 6.0:** Expl moves. R2 pings. Expl hears nothing. Link Broken.

---

## 🔬 Lab Exercise: Packet Loss & QoS

### 1. Lab Objectives
- In ROS 2, `Reliable` (TCP-like) retries until success. `BestEffort` (UDP-like) drops.
- **Task:** Send high-res video (10MB/s) and Odometry (1KB/s) over a simulated bad link (20% loss).
- **Observe:**
    - Reliable Video: Latency spikes to seconds (Retries clog network).
    - BestEffort Video: Glitchy frames, but low latency. Good for teleop.
    - Reliable Odom: Essential. Never use BestEffort for tf frames!

---

## 🚀 Project: "Comm-Aware Path Planning"

**Goal:** Robot must reach Goal, but maintain Signal Strength > -70dBm.
1.  **CostMap:** Add a "Signal Layer".
    *   Likelihood field based on known Router position.
    *   Or Gaussian Process Regression (GPR) to learn signal map.
2.  **Planner:** A* finds path that stays in "White" zones (Good Signal) and avoids "Black" zones (Dead zones).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Hidden Terminal Problem"
*   **Scenario:** A and C transmit to B. A and C cannot hear each other.
*   **Result:** They transmit simultaneously. Packets collide at B. B hears noise.
*   **Fix:** RTS/CTS (Request To Send / Clear To Send) handshake in Wifi.

#### 2. "Bandwidth Saturation"
*   **Symptom:** Lidar works, Camera freezes.
*   **Cause:** Wifi 2.4GHz has ~5MB/s real throughput. Lidar (1MB) + Cam (4MB) = Limit.
*   **Fix:** Throttling. Don't send 30FPS. Send 5FPS if network is busy. Adaptive Resolution.

---

## ⚡ Optimization: Multi-Frequency

Use diverse radios.
*   **5GHz (Wifi):** High Bandwidth (Video), Short Range, Blocked by walls.
*   **900MHz (LoRa/Telemetry):** Low Bandwidth (Status), Long Range, Penetrates walls.
*   **Strategy:** Send video on 5GHz. If 5GHz drops, switch to 900MHz (Text Only Control).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Data Mule"?
    *   **A:** A robot that physically carries data from A to B because no wireless link exists.
2.  **Q:** Difference between "Access Point" and "Mesh Node"?
    *   **A:** AP connects clients to wired internet. Mesh Node forwards traffic for other nodes wirelessly.
3.  **Q:** Why is Latency bad for Control?
    *   **A:** Delay reduces Phase Margin. If delay > 200ms, teleoperation becomes oscillating/unstable for humans.

### Challenge Task
> **Task:** Relay Positioning.
> 1. Robot A explores. Relay B follows.
> 2. Control Law for B: Minimize $E = (Pos_A + Pos_Base)/2$.
> 3. B automatically positions itself halfway to act as a repeater.

---

## 📚 Further Reading
- **BATMAN-adv:** Open Mesh Protocol.
- **Delay Tolerant Networks (RFC 4838).**

---

**Day 41 Complete**
