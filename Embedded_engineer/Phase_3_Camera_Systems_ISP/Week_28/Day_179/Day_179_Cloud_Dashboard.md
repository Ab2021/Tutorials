# Day 179: Cloud & UI Dashboard
## Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project

---

## 🎯 Learning Objectives
1.  **Connect** the Robot to the Cloud (AWS IoT Core) using MQTT.
2.  **Stream** Low-Latency Video to a Browser using WebRTC (Aiortc).
3.  **Build** a React/HTML Dashboard to visualize Telemetry (Speed, Battery, Map) and Control the robot.
4.  **Implement** "Over-The-Air" (OTA) updates for the robot software.
5.  **Secure** the connection using TLS Certificates.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Robot connected to WiFi.
*   **Software:** AWS Account, Node.js (for Dashboard), Python `awsiotsdk`, `aiortc`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MQTT (Message Queuing Telemetry Transport)
*   **Pub/Sub:** Robot publishes to `robot/telemetry`. Dashboard subscribes to `robot/telemetry`.
*   **Shadows:** A JSON document in the cloud that represents the "Desired" and "Reported" state.
    *   Dashboard sets `desired: { mode: "PATROL" }`.
    *   Robot reads it, changes mode, sets `reported: { mode: "PATROL" }`.
*   **QoS (Quality of Service):** Use QoS 1 (At least once) for commands.

### 🔹 Part 2: WebRTC (Real-Time Communication)
*   **Why not RTSP?** RTSP latency is 2-5 seconds. WebRTC is < 500ms.
*   **Signaling:** Exchanging SDP (Session Description Protocol) via MQTT or HTTP to establish the P2P connection.
*   **STUN/TURN:** Servers to punch through NAT/Firewalls.

### 🔹 Part 3: The Dashboard
*   **Frontend:** React.js or simple HTML/JS.
*   **Backend:** AWS Lambda or a local Node.js server to bridge MQTT to Websockets (if not using AWS IoT directly over WS).

---

## 💻 Implementation Examples

### Example 1: AWS IoT MQTT Client (Python)

```python
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import json
import time

def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    print(f"Received message from topic '{topic}': {payload}")
    # Handle Command (e.g., Emergency Stop)

# Connect
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint="xxx-ats.iot.us-east-1.amazonaws.com",
    cert_filepath="certs/certificate.pem.crt",
    pri_key_filepath="certs/private.pem.key",
    ca_filepath="certs/AmazonRootCA1.pem",
    client_id="Robot01",
    clean_session=False,
    keep_alive_secs=30)

connect_future = mqtt_connection.connect()
connect_future.result()

# Loop
while True:
    telemetry = {
        "battery": 12.4,
        "speed": 0.5,
        "location": {"x": 10, "y": 20}
    }
    mqtt_connection.publish(
        topic="robot/telemetry",
        payload=json.dumps(telemetry),
        qos=mqtt.QoS.AT_LEAST_ONCE)
    time.sleep(1)
```

### Example 2: WebRTC Streamer (Aiortc)

```python
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

class CameraStreamTrack(VideoStreamTrack):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.camera.get_frame() # From HAL
        
        # Convert to AV Frame
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

async def run(pc, signaling):
    # Signaling logic (Exchange SDP via MQTT)
    # ...
    pass
```

### Example 3: Simple Dashboard (HTML/JS)

```html
<!DOCTYPE html>
<html>
<body>
    <h1>Robot Dashboard</h1>
    <div id="video-container">
        <video id="remote-video" autoplay playsinline></video>
    </div>
    <div id="telemetry">
        <p>Battery: <span id="battery">--</span> V</p>
    </div>
    <button onclick="sendCommand('START')">START</button>
    <button onclick="sendCommand('STOP')">STOP</button>

    <script>
        // Connect to AWS IoT via MQTT.js over Websockets
        // Handle WebRTC PeerConnection
    </script>
</body>
</html>
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Remote Control

**Objective:** Drive from the browser.

**Steps:**
1.  Add "WASD" key listeners to the Dashboard.
2.  Send JSON commands: `{"cmd": "move", "x": 1, "y": 0}` via MQTT.
3.  Robot subscribes and passes to `MotorDriver`.
4.  **Latency Test:** Press 'W'. Measure time until wheels spin. Target < 200ms.

### Lab 2: Battery Graph

**Objective:** Visualize drain.

**Steps:**
1.  Robot publishes battery voltage every 1s.
2.  Dashboard uses Chart.js to plot it.
3.  **Alert:** If voltage < 10.5V, flash the screen RED.

### Lab 3: The "Kill Switch"

**Objective:** Safety first.

**Steps:**
1.  Add a big RED button on the Dashboard.
2.  When pressed, send `STOP` with QoS 1.
3.  Robot MUST acknowledge receipt.
4.  If Robot loses WiFi (MQTT Disconnect), it should auto-stop after 2 seconds (Heartbeat logic).

---

## 🐛 Debugging Connectivity

### Debug 1: "Video Lag (2 seconds)"

**Symptom:** WebRTC feels like RTSP.

**Cause:**
*   Buffer bloat in the encoder.
*   **Fix:** Set `zerolatency` tune in H.264 encoder. Reduce bitrate.

### Debug 2: "MQTT Disconnects"

**Symptom:** Robot goes offline randomly.

**Cause:**
*   Weak WiFi signal.
*   Keep-Alive timeout too short.
*   **Fix:** Increase Keep-Alive. Use a WiFi range extender.

---

## ⚡ Performance Optimization

### Optimization 1: Edge Processing

*   Don't send *every* frame to the cloud. Bandwidth is expensive.
*   Only send the video stream when a user is watching.
*   Send *metadata* (e.g., "Person Detected") constantly.

### Optimization 2: Protobuf

*   Replace JSON with Protocol Buffers for telemetry.
*   Reduces payload size by 50%. Saves data costs.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "NAT Traversal"?** (Technique to establish a connection between two devices behind different routers. STUN/TURN servers help).
2.  **Why use TLS?** (Encryption. Prevents hackers from hijacking your robot).
3.  **Difference between "Shadow" and "Topic"?** (Topic is a transient message channel. Shadow is a persistent state document).

### Practical Challenges

1.  **Map Visualization:** If the robot uses SLAM (Simultaneous Localization and Mapping), send the Occupancy Grid (Map) to the dashboard and render it on a Canvas.
2.  **Voice Control:** Integrate AWS Alexa. "Alexa, tell the robot to patrol."

---

## 📚 Further Reading & Resources

### Documentation
*   **AWS IoT Core Documentation.**
*   **WebRTC for the Curious.**

---

## 🎓 Summary

Today we covered:
- ✅ **MQTT:** Telemetry & Control.
- ✅ **WebRTC:** Low-latency video.
- ✅ **Dashboard:** The cockpit.
- ✅ **Security:** TLS & Certs.
- ✅ **Heartbeat:** Safety monitoring.

**Next:** Day 180 - Final Demo & Graduation.

---

**Day 179 Complete** | Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project


