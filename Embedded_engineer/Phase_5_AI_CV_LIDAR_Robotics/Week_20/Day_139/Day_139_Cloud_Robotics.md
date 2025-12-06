# Day 139: Cloud Robotics (Kubernetes/Fog ROS)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> The Robot is just a sensor. The Brain is in the Cloud.
> - **Focus:** Fog Computing, Cloud SLAM, Kubernetes (K8s) for fleet management, 5G constraints, and the "Thin Client" robot architecture.
> - **Code:** A "Cloud SLAM" proxy using Websockets. `robot_client.py` sends scans. `cloud_server.py` (simulating a K8s Pod) processes them and sends back the Map.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Justify** Cloud Robotics: Why carry a GPU if you have 5G? (Battery, Weight, Cost).
2.  **Architect** a Fog System: Critical loops (Balance) on Edge, Heavy loops (SLAM/Grasp Planning) on Cloud.
3.  **Deploy** ROS 2 nodes as Docker Containers in a Kubernetes Cluster (Simulated).
4.  **Handle** Network Latency & Disconnection (Graceful degradation).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Internet connection (Localhost for sim).

### Software Environment
```bash
pip install websockets asyncio numpy matplotlib
```

### Prior Knowledge
- Docker (Day 101).
- Websockets.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Thin Client vs Thick Client

*   **Thick Client (Autonomous):** Compute onboard (Jetson). Robust to WiFi loss. Expensive battery.
*   **Thin Client (Cloud):** Compute offboard (AWS/GCP). Cheap. Infinite compute. Dies if WiFi fails.
*   **Fog Robotics:** The Hybrid. Safety on Edge. Intelligence in Cloud.

### 🔹 Part 2: Kubernetes (K8s)

The OS for the Cloud.
*   **Pod:** A container (ROS Node).
*   **Service:** A Load Balancer (ROS Master/Discovery).
*   **Deployment:** "Run 100 replicas of the SLAM node for 100 robots".

### 🔹 Part 3: The Latency Budget

*   **Round Trip Time (RTT):**
    *   WiFi: 5-50ms (Variable).
    *   5G: 5-10ms (Stable).
    *   Starlink: 40-100ms.
*   **Control Loop:** 100Hz requires 10ms. Cloud cannot do Balance Control. It can do Map Updates (1Hz).

---

## 💻 Implementation: Cloud SlaaS (SLAM as a Service)

We emulate a system where the Robot blindly sends Lidar data, and the Cloud returns the Map.

### 🛠️ Project Structure
```text
day139_cloud/
├── src/
│   ├── cloud_server.py
│   ├── robot_client.py
└── output/
    ├── map_update.png
```

### 👨‍💻 The Cloud Server (`src/cloud_server.py`)

Simulates a heavy map optimization process.

```python
import asyncio
import websockets
import json
import numpy as np
import time

# Simulating a K8s Pod
PORT = 8765

async def slam_handler(websocket, path):
    print("Cloud: Robot Connected.")
    map_data = [] # Global Map
    
    try:
        async for message in websocket:
            # 1. Receive Scan (JSON or Binary)
            data = json.loads(message)
            scan = np.array(data['scan'])
            pos = np.array(data['odom'])
            
            # 2. Heavy Computation (Simulate Latency)
            # Scan Matching / Bundle Adjustment
            await asyncio.sleep(0.1) # 100ms Processing Time
            
            # 3. Update Map (Simple accumulation)
            # Transform scan to global frame
            angle = pos[2]
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            global_points = R @ scan.T + pos[:2].reshape(2,1)
            
            # Just store center for "Map"
            map_data.append(pos[:2].tolist())
            
            # 4. Reply with Updated Map Info
            response = {
                "status": "Map Updated",
                "nodes": len(map_data),
                "opt_score": 0.99
            }
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print("Cloud: Robot Disconnected.")

async def main():
    start_server = websockets.serve(slam_handler, "localhost", PORT)
    print(f"Cloud SLAM Service running on ws://localhost:{PORT}")
    await start_server
    await asyncio.Future() # Run forever

if __name__ == "__main__":
    asyncio.run(main())
```

### 👨‍💻 The Robot Client (`src/robot_client.py`)

Sends data, handles latency.

```python
import asyncio
import websockets
import json
import numpy as np
import time
import matplotlib.pyplot as plt

async def robot_loop():
    uri = "ws://localhost:8765"
    
    # Simulated Trajectory
    t = np.linspace(0, 10, 100)
    x = t * np.cos(t)
    y = t * np.sin(t)
    theta = t
    
    latencies = []
    
    try:
        async with websockets.connect(uri) as websocket:
            for i in range(len(t)):
                # 1. Sense
                # Dummy Lidar (Circle of points)
                lidar = np.random.rand(10, 2) # Local frame
                odom = [x[i], y[i], theta[i]]
                
                payload = {
                    "scan": lidar.tolist(),
                    "odom": odom,
                    "timestamp": time.time()
                }
                
                # 2. Upload
                t_send = time.time()
                await websocket.send(json.dumps(payload))
                
                # 3. Wait for Map Update?
                # Using 'await' makes it Blocking.
                # In real system, we'd use a callback or separate thread.
                response = await websocket.recv()
                t_recv = time.time()
                
                rtt = (t_recv - t_send) * 1000 # ms
                latencies.append(rtt)
                
                print(f"Step {i}: RTT={rtt:.1f}ms | Cloud: {response}")
                
                # Simulate Robot Speed (10Hz)
                await asyncio.sleep(0.1)
                
    except ConnectionRefusedError:
        print("Cloud Unreachable! Switching to Safe Mode.")

    # Plot Latency
    plt.plot(latencies)
    plt.title("Cloud SLAM Latency (RTT)")
    plt.xlabel("Step")
    plt.ylabel("Time (ms)")
    plt.axhline(100, color='r', linestyle='--', label='Max Budget')
    plt.legend()
    plt.savefig("output/cloud_latency.png")

if __name__ == "__main__":
    asyncio.run(robot_loop())
```

---

## 🔬 Lab Exercise: "The Disconnect"

### 1. Lab Objectives
- **Run:** Server, then Client.
- **Observe:** RTT is ~100ms (dominated by simulated processing).
- **Kill:** Stop the Server mid-run.
- **Client:** Must catch `ConnectionRefused` or `ConnectionClosed`.
- **Action:** Switch to "Safety Mode" (Stop moving).
- **Modify:** Make the client "Asynchronous". Don't wait for reply. Just stream data. Read replies when they come.

---

## 🚀 Project: "Cloud Grasping"

**Goal:** Pick up unknown object.
1.  **Robot:** Sends Image to Cloud.
2.  **Cloud:** Runs `GraspNet-1M` (Huge model, 10GB VRAM).
3.  **Cloud:** Returns 6D Grasp Pose.
4.  **Robot:** Executes trajectory.
5.  **Benefit:** Robot needs only a Raspberry Pi. No GPU needed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Serialization Overhead"
*   **Cause:** JSON is slow. Converting 100k points to ASCII text (`"1.234"`) is heavy.
*   **Fix:** Use Binary formats like `Protobuf`, `Flatbuffers`, or `CBOR`. Or raw bytes.

#### 2. "Bandwidth Saturation"
*   **Cause:** Streaming 4K Raw Video.
*   **Fix:** Edge Compression. H.264/H.265 encoding on the Robot. Send only compressed video.

---

## ⚡ Optimization: 5G Slicing

Telcos offer "Network Slicing".
*   **Slice A:** High Bandwidth (Video).
*   **Slice B:** Ultra-Low Latency (Control).
*   Guarantees QoS (Quality of Service) for robots even if everyone else is watching Netflix.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Fog" computing?
    *   **A:** Compute nodes located on the local LAN (e.g., a powerful Server in the Warehouse), not in an Amazon Datacenter. Lower latency.
2.  **Q:** Why Kubernetes for Robots?
    *   **A:** Auto-scaling. If you turn on 50 robots, K8s automatically spins up 50 SLAM pods. If 10 turn off, it frees resources.
3.  **Q:** Security?
    *   **A:** Critical. TLS/SSL for Websockets. VPN for ROS 2 DDS. If cloud is hacked, fleet is hijacked.

### Challenge Task
> **Task:** Compression.
> 1. Use `zlib`.
> 2. Compress the scan data before sending.
> 3. Decompress on Server.
> 4. Measure CPU time vs Bandwidth saved.

---

## 📚 Further Reading
- **AWS RoboMaker:** Cloud robotics platform.
- **Open Robotics:** "ROS 2 on Kubernetes".

---

**Day 139 Complete**
