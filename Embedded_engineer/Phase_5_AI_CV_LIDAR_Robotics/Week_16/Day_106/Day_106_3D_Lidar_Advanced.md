# Day 106: 3D Lidar Advanced
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> Don't just use `ros2 launch velodyne...`. Understand the bits.
> - **Focus:** 3D Lidar physics (ToF, Rotating Prisms, MEMS), Packet decoding (UDP), and `PointCloud2` field structure (XYZIR).
> - **Code:** A Python implementation of a "Lidar Driver" that binds to a UDP port, reads raw Velodyne-like packets, calculates XYZ from Azimuth/Elevation, and publishes a structured PointCloud.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Rotating 3D Lidars (Velodyne/Ouster) and Solid State Lidars (Livox).
2.  **Decode** Raw UDP Packets (Firing Blocks, Azimuth bytes).
3.  **Construct** a `sensor_msgs/PointCloud2` message manually including `ring` and `intensity` fields.
4.  **Visualize** intensity calibration (Reflectivity of Asphalt vs Paint).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Velodyne VLP-16 (or Dataset/PCAP file).

### Software Environment
```bash
sudo apt install ros-humble-velodyne-driver ros-humble-pcl-ros
pip install dpkt # For PCAP parsing
```

### Prior Knowledge
- Spherical Coordinates ($r, \theta, \phi$).
- UDP Networking.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How 3D Lidar Works

*   **Mechanism:** A laser diode fires. A mirror spins. The time of flight (ToF) determines distance.
*   **Channels:** VLP-16 has 16 laser pairs arranged vertically (Elevation $\omega$). The motor rotates (Azimuth $\alpha$).
*   **Math:**
    $$ x = r \cos(\omega) \sin(\alpha) $$
    $$ y = r \cos(\omega) \cos(\alpha) $$
    $$ z = r \sin(\omega) $$

### 🔹 Part 2: The Data Stream

Lidars pump massive data (~100 Mbps).
*   **Protocol:** UDP Broadcast.
*   **Packet Structure (Velodyne Example):**
    *   Header (1206 bytes payload).
    *   12 Blocks per packet.
    *   2 Firing sequences per block (Upper/Lower bank).
    *   Timestamps and Factory Bytes.

### 🔹 Part 3: PointCloud2 Structure

It's a binary blob.
*   **Fields:** `x(float32)`, `y(float32)`, `z(float32)`, `intensity(float32)`, `ring(uint16)`.
*   **Row Step:** Bytes per row.
*   **Is Dense:** True if no `NaN` values.

---

## 💻 Implementation: Custom Lidar Driver

We will parse a sample PCAP file (Wireshark capture of Lidar Traffic) or simulated UDP stream.

### 🛠️ Project Structure
```text
day106_lidar/
├── src/
│   ├── simple_driver.py
│   └── udp_sim.py
└── data/
    └── sample_vlp16.pcap (User provides)
```

### 👨‍💻 Driver Node (`src/simple_driver.py`)

Parses raw firing data into XYZ.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import socket
import struct

class LidarDriver(Node):
    def __init__(self):
        super().__init__('custom_lidar_driver')
        self.pub = self.create_publisher(PointCloud2, 'velodyne_points', 10)
        
        # Hardcoded Vertical Angles for VLP-16 (Degrees)
        self.vertical_angles = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        self.vertical_angles = np.deg2rad(self.vertical_angles)
        
        # UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 2368)) # Standard Velodyne Port
        
        self.timer = self.create_wall_timer(0.001, self.read_packet) # Poll fast

    def read_packet(self):
        try:
            data, addr = self.sock.recvfrom(2048)
            # 1206 bytes standard
            if len(data) == 1206:
                points = self.parse_velodyne(data)
                self.publish_cloud(points)
        except BlockingIOError:
            pass

    def parse_velodyne(self, data):
        # Simply parsing logic (Simplified)
        points = []
        # Offset 0-1200 contains 12 blocks
        for i in range(12):
            # Block Header (2 bytes) + Azimuth (2 bytes)
            offset = i * 100
            azimuth_int = struct.unpack_from('<H', data, offset+2)[0]
            azimuth = np.deg2rad(azimuth_int / 100.0)
            
            # 32 Firings (2 sequences of 16 lasers)
            for j in range(32):
                # Distance (2 bytes) + Reflectivity (1 byte)
                dist_offset = offset + 4 + (j * 3)
                distance_mm = struct.unpack_from('<H', data, dist_offset)[0]
                distance_m = distance_mm / 1000.0
                intensity = struct.unpack_from('<B', data, dist_offset+2)[0]
                
                # Laser ID (0-15) -> Vertical Angle
                laser_id = j % 16
                omega = self.vertical_angles[laser_id]
                
                if distance_m > 0.1: # Min Range
                    # Spherical to Cartesian
                    # Note: Velodyne Coordinate system might differ slightly in Rotation direction
                    xy_proj = distance_m * np.cos(omega)
                    x = xy_proj * np.sin(azimuth)
                    y = xy_proj * np.cos(azimuth)
                    z = distance_m * np.sin(omega)
                    
                    points.append([x, y, z, float(intensity)])
        
        return points

    def publish_cloud(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "velodyne"
        
        msg.height = 1
        msg.width = len(points)
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16 # 4 floats * 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        
        # Serialize list to binary
        buffer = []
        for p in points:
            buffer += struct.pack('ffff', *p)
        
        msg.data = bytearray(buffer)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(LidarDriver())
```

---

## 🔬 Lab Exercise: "Playback"

### 1. Lab Objectives
- **Download:** A sample Velodyne PCAP (from Velodyne website or ROS bags).
- **Replay:** Use `udpreplay` (on Linux) to blast the PCAP to localhost UDP port 2368.
- **Run:** `ros2 run day106_lidar simple_driver`.
- **View:** Rviz.
- **Result:** You see the point cloud appearing. You successfully reverse-engineered the driver.

---

## 🚀 Project: "Intensity Calibration"

**Goal:** Detect Lane Markings.
1.  **Observation:** Road is black (Low Intensity). Paint is white (High Intensity).
2.  **Filter:** Only keep points with `intensity > 200`.
3.  **Result:** You get a sparse cloud of just the lane lines and traffic signs.
4.  **Application:** Use clear white tape on the floor and try to follow it using Lidar only (no camera).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Warped Cloud"
*   **Cause:** Packet loss or Timing issues. If you drop UDP packets, the Azimuth jumps, causing tearing.
*   **Fix:** Increase Kernel UDP Buffer size (`sysctl -w net.core.rmem_max=26214400`).

#### 2. "Upside Down"
*   **Cause:** Coordinate frame mismatch. VLP-16 often mounted upside down on drones.
*   **Fix:** Static Transform Publisher (`static_transform_publisher 0 0 0 0 3.14159 0 base_link velodyne`).

---

## ⚡ Optimization: PointCloud2 Modifier

Using Python struct pack is slow (CPU intensive for 300k points/sec).
*   **Fast Way:** Use `numpy` structured arrays.
*   `points = np.zeros(N, dtype=[('x', 'f4'), ('y', 'f4')...])`.
*   `msg.data = points.tobytes()`.
*   100x speedup over iteration.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Ring" field?
    *   **A:** The index of the laser (0-15). Useful for processing the cloud as "Scan Lines" rather than a blob.
2.  **Q:** Why use UDP instead of TCP for Lidar?
    *   **A:** Latency. With TCP, if a packet is lost, the stream halts to retransmit. In Lidar, old data is useless. We want the newest packet immediately.
3.  **Q:** Difference between VLP-16 and Ouster OS-1?
    *   **A:** VLP-16 rotates firing emitters. Ouster uses a digital shutter (Solid State-ish) and structured light/flash principles, often outputting organized clouds naturally.

### Challenge Task
> **Task:** Deskewing.
> 1. Robot is moving fast.
> 2. Point at start of scan is at $t=0$. Point at end is at $t=0.1s$.
> 3. Robot moved 10cm. The cloud is stretched.
> 4. **Fix:** Use IMU/Odom velocity to un-warp the points based on their discrete packet timestamp.

---

## 📚 Further Reading
- **Velodyne Interface Specification:** The bible for packet structures.
- **PCL ROS:** How to handle PointCloud2 efficiently in C++.

---

**Day 106 Complete**
