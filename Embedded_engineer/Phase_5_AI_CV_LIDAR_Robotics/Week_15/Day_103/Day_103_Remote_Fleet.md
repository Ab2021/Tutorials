# Day 103: Remote Fleet Management
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> The robot is in Japan. You are in San Francisco. Talk to it.
> - **Focus:** Overcoming NAT/Firewalls, VPNs (Husarnet/Tailscale), and IoT Bridges (Zenoh/MQTT).
> - **Code:** A demo using `zenoh-bridge-dds` to link two ROS 2 domains across the internet (simulated via Docker networks to represent WAN).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Discovery Problem over the Internet (Multicast doesn't route over WAN).
2.  **Deploy** Husarnet (P2P VPN) to create a flat IPv6 network for Robots.
3.  **Bridge** ROS 2 to Zenoh (Zero Overhead Protocol) for extremely bandwidth-efficient teleop.
4.  **Architect** a Fleet Management System (Cloud Dashboard $\leftrightarrow$ Robots).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Two Linux machines (or VMs/Containers) to simulate "Robot" and "Cloud".

### Software Environment
```bash
sudo apt install cargo
cargo install zenoh-bridge-dds
# Or Husarnet Client
```

### Prior Knowledge
- Networking (IP, Subnets).
- ROS 2 DDS.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The WAN Problem

*   **LAN (Local Area Network):** ROS 2 works great (Discovery via Multicast).
*   **WAN (Wide Area Network):** Multicast is blocked. NAT (Network Address Translation) hides robots behind routers.
*   **Solutions:**
    1.  **VPN (Virtual Private Network):** Husarnet/WireGuard. Makes the WAN look like a LAN.
    2.  **Bridging:** DDS Router (eProsima) or Zenoh. Forward specific topics via TCP/QUIC to a central server.

### 🔹 Part 2: Husarnet (The Easy Way)

*   P2P VPN optimized for ROS.
*   Uses IPv6.
*   Zero configuration. Just install client, join group.
*   **Result:** `ping robot-1`. `export ROS_DOMAIN_ID=0`. It works magically.

### 🔹 Part 3: Eclipse Zenoh (The Efficient Way)

*   DDS is heavy (Discovery traffic). Zenoh is light (Byte-level overhead).
*   **Architecture:**
    *   Robot runs `zenoh-bridge-dds`.
    *   Cloud runs `zenoh-bridge-dds`.
    *   They connect via TCP/QUIC.
    *   Example: Robot Pub `/cam` $\to$ Bridge $\to$ Internet $\to$ Bridge $\to$ Cloud Sub `/cam`.

---

## 💻 Implementation: Zenoh Bridge

We will simulate a "Field Robot" and a "Control Center".

### 🛠️ Project Structure
```text
day103_remote/
├── docker-compose.yml 
├── robot_config.json5
└── cloud_config.json5
```

### 👨‍💻 Orchestration (`docker-compose.yml`)

We create two isolated networks to prove they can talk.

```yaml
version: '3'

services:
  # --- THE ROBOT (Network A) ---
  robot_talker:
    image: osrf/ros:humble-desktop
    command: ros2 run demo_nodes_cpp talker
    networks:
      - robot_net
    environment:
      - ROS_DOMAIN_ID=42

  robot_bridge:
    image: eclipse/zenoh-bridge-dds:latest
    command: -d 42 -e tcp/0.0.0.0:7447
    ports:
      - "7447:7447" # Open port for Cloud to connect
    networks:
      - robot_net

  # --- THE CLOUD (Network B) ---
  cloud_listener:
    image: osrf/ros:humble-desktop
    command: ros2 run demo_nodes_cpp listener
    networks:
      - cloud_net
    environment:
      - ROS_DOMAIN_ID=0 # Different Domain!

  cloud_bridge:
    image: eclipse/zenoh-bridge-dds:latest
    # Connect to Robot's exposed port (In real world, this would be a Public IP)
    command: -d 0 -e tcp/robot_bridge:7447 
    networks:
      - cloud_net

networks:
  robot_net:
  cloud_net:
```

### 👨‍💻 Filter Config (`robot_config.json5`)

Don't send EVERYTHING. Bandwidth costs money (LTE).

```json5
{
  plugins: {
    dds: {
      allow: {
        // Only allow Telemetry and Camera
        publishers: ["/robot_status", "/camera/compressed"],
        subscribers: ["/cmd_vel"]
      }
    }
  }
}
```

---

## 🔬 Lab Exercise: "The Ping Across The World"

### 1. Lab Objectives
- **Start:** `docker-compose up`.
- **Observe:** `robot_talker` publishes `Hello World`.
- **Observe:** `cloud_listener` hears `Hello World`.
- **Proof:** They are on different ROS Domains (42 vs 0) and isolated Docker networks. The Bridge is tunneling the data.
- **Latency Test:** Add artificial delay (using `tc` command) to simulate 4G connection. Zenoh handles it well.

---

## 🚀 Project: "LTE Teleoperation"

**Goal:** Drive a turtlebot from home using a joystick.
1.  **Robot:** Raspberry Pi with 4G Modem. Runs Husarnet. Run `ros2 launch turtlebot3_bringup ...`.
2.  **Home:** Laptop on WiFi. Runs Husarnet. Run `ros2 run teleop_twist_keyboard`.
3.  **Config:** `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. (Husarnet works best with Cyclone or FastDDS).
4.  **Video:** `/camera/image_raw` takes 5MB/s. Too big for 4G?
    *   **Fix:** Use `/camera/image_compressed` or `image_transport`.
    *   **Result:** Usable lag (<200ms).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "MTU Issues" (VPN)
*   **Symptom:** Small messages work, Large messages (Lidar/Camera) drop.
*   **Cause:** VPN adds headers. Packet size exceeds 1500 bytes. Packet fragmentation fails.
*   **Fix:** Reduce MTU on the ROS interface or enable `fragmentation` in DDS config.

#### 2. "Discovery works but no data"
*   **Cause:** Firewalls blocking UDP ports. Husarnet usually punches holes, but corporate NATs are strict.
*   **Fix:** Use a Relay Server (Husarnet Base Station).

---

## ⚡ Optimization: Zenoh-Plugin-Ros1

What if you have a legacy ROS 1 robot?
*   Zenoh has a plugin for ROS 1 too.
*   You can bridge ROS 1 Robot $\leftrightarrow$ Zenoh $\leftrightarrow$ ROS 2 Cloud.
*   Easiest way to modernize a fleet without rewriting robot code.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just forward port 11311 (ROS 1 Master) or 7400 (ROS 2 Disc)?
    *   **A:** **SECURITY.** Exposing ROS ports to the internet allows anyone to drive your robot. VPNs provide encryption and auth.
2.  **Q:** What is the difference between VPN and Bridge?
    *   **A:** VPN connects *Machines* (All ports open). Bridge connects *Topics* (Application level filtering). Bridging is more secure for exposing specific data to 3rd parties.
3.  **Q:** Does Zenoh use ROS messages?
    *   **A:** Zenoh is data-agnostic. It just moves bytes. The `zenoh-bridge-dds` does the translation between DDS types and Zenoh payload.

### Challenge Task
> **Task:** Cloud Logging.
> 1. Robot publishes `/battery_state`.
> 2. Cloud node subscribes and writes to a Database (InfluxDB).
> 3. Verify data integrity over 1 hour.

---

## 📚 Further Reading
- **Husarnet Docs:** "Using ROS 2 with Husarnet".
- **Zenoh.io:** Whitepaper on "Zero Overhead".

---

**Day 103 Complete**
