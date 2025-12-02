# Day 84: Advanced Capstone - Cloud Integration (AWS IoT)
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎯 Learning Objectives
1.  **Connect** an embedded camera to AWS IoT Core using MQTT (MQTTS).
2.  **Implement** "Thing Shadows" to manage device state (e.g., Remote Configuration).
3.  **Upload** Event Snapshots (Images) to AWS S3 using Signed URLs.
4.  **Stream** Live Video to AWS Kinesis Video Streams (KVS).
5.  **Handle** Offline Scenarios: Store-and-Forward logic.
6.  **Secure** the connection using X.509 Certificates.

---

## 📚 Prerequisites & Preparation
*   **Account:** AWS Free Tier Account.
*   **Software:** AWS IoT Device SDK (Python/C++), Boto3.
*   **Knowledge:** MQTT, JSON, Public Key Infrastructure (PKI).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cloud Architecture
*   **Control Plane (MQTT):** Low bandwidth. Used for commands ("Reboot", "Update Model") and status ("Temp=45C").
*   **Data Plane (S3/KVS):** High bandwidth. Used for images and video.
*   **Edge Compute:** The camera decides *what* to send. Don't send 24/7 4K video (too expensive). Send only "Events".

### 🔹 Part 2: Security (Mutual Auth)
*   **Server Auth:** Device verifies AWS (Root CA).
*   **Client Auth:** AWS verifies Device (Client Certificate + Private Key).
*   **Policies:** IAM Roles define what the device can do (e.g., `s3:PutObject` allowed, `s3:DeleteObject` denied).

### 🔹 Part 3: Kinesis Video Streams (KVS)
*   A managed service for ingesting video.
*   Supports WebRTC for low-latency (< 1s) live viewing.
*   Supports HLS/DASH for playback of recorded history.

---

## 💻 Implementation Examples

### Example 1: MQTT Telemetry (Python)

Sending heartbeat.

```python
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json
import time

# 1. Configure Client
myMQTTClient = AWSIoTMQTTClient("TrafficCam_01")
myMQTTClient.configureEndpoint("xyz-ats.iot.us-east-1.amazonaws.com", 8883)
myMQTTClient.configureCredentials("root-CA.crt", "private.key", "cert.pem")

# 2. Connect
myMQTTClient.connect()

# 3. Publish Loop
while True:
    payload = {
        "temperature": read_temp(),
        "fps": read_fps(),
        "uptime": read_uptime()
    }
    myMQTTClient.publish("cameras/TrafficCam_01/status", json.dumps(payload), 1)
    time.sleep(60)
```

### Example 2: Uploading Snapshot to S3

Using `boto3` to upload a file.

```python
import boto3

s3 = boto3.client('s3')
bucket_name = "traffic-cam-storage"

def upload_event(image_path, event_id):
    key = f"events/{event_id}.jpg"
    
    # Upload
    s3.upload_file(image_path, bucket_name, key)
    
    # Generate Signed URL (valid for 1 hour)
    url = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': bucket_name, 'Key': key},
                                    ExpiresIn=3600)
    return url
```

### Example 3: KVS GStreamer Plugin

Streaming to Kinesis using the C++ SDK plugin (`kvssink`).

```bash
gst-launch-1.0 v4l2src ! video/x-raw,width=1920,height=1080 ! \
  nvvideoconvert ! nvv4l2h264enc ! h264parse ! \
  kvssink stream-name="TrafficCam_01" \
  access-key="AKIA..." secret-key="SECRET..." \
  aws-region="us-east-1"
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Remote Configuration (Shadows)

**Objective:** Change camera resolution from the cloud.

**Steps:**
1.  In AWS Console, edit the Thing Shadow: `{"desired": {"resolution": "720p"}}`.
2.  Device receives Delta message.
3.  Device applies change (restarts pipeline).
4.  Device reports new state: `{"reported": {"resolution": "720p"}}`.
5.  **Result:** Sync achieved.

### Lab 2: The "Smart Doorbell" Flow

**Objective:** Press button -> Notification.

**Steps:**
1.  Detect "Person" (YOLO).
2.  Save `snapshot.jpg`.
3.  Upload to S3.
4.  Publish MQTT message: `{"event": "Person", "url": "s3://..."}`.
5.  (Cloud Side): Lambda function triggers SNS to send SMS to your phone.

### Lab 3: Offline Buffering

**Objective:** Handle network loss.

**Steps:**
1.  Disconnect WiFi.
2.  Trigger events.
3.  **Logic:** Save JSON metadata and Images to local SD card (`/queue`).
4.  Reconnect WiFi.
5.  **Logic:** A background thread reads `/queue` and uploads everything.

---

## 🐛 Debugging Cloud Issues

### Debug 1: TLS Handshake Failure

**Symptom:** Connection refused.

**Cause:**
*   Clock Skew. If the camera time is 1970, the certificate is "not yet valid".
*   **Fix:** Ensure NTP is running (`chronyd`).

### Debug 2: High Latency in KVS

**Symptom:** 5-second delay.

**Cause:**
*   GOP Size (Keyframe interval) is too large.
*   **Fix:** Set Encoder `iframeinterval` to 30 (1 second) or less. KVS fragments video at keyframes.

---

## ⚡ Performance Optimization

### Optimization 1: Edge Filtering

*   Don't upload every "Car". Only upload "Speeding Cars" or "Stolen Cars" (License Plate Match).
*   Saves $$$ on S3 storage and Data Transfer.

### Optimization 2: MQTT Keep-Alive

*   Adjust Keep-Alive interval based on network stability.
*   Too short = Battery drain. Too long = Ghost disconnects.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is a "Thing Shadow"?** (A JSON document persisting the device's state, even if offline).
2.  **Why use MQTT over HTTP?** (Lighter weight, Pub/Sub pattern, better for unstable networks).
3.  **What is "X.509"?** (Standard for Public Key Certificates).
4.  **Cost:** If you upload 1GB to S3, how much does it cost? (Roughly $0.023/GB storage, plus request costs).

### Practical Challenges

1.  **Implement "OTA Update":** Use MQTT to send a URL to a new firmware binary (`update.bin`). The device downloads it, verifies signature, writes to partition, and reboots.
2.  **Secure the Keys:** Store the Private Key in a Secure Element (TPM/HSM) so it cannot be extracted from the file system.

---

## 📚 Further Reading & Resources

### Documentation
*   **AWS IoT Core Developer Guide.**
*   **Amazon Kinesis Video Streams C++ Producer SDK.**

---

## 🎓 Summary

Today we covered:
- ✅ **IoT Core:** MQTT & Shadows.
- ✅ **S3:** Storing evidence.
- ✅ **KVS:** Streaming video.
- ✅ **Security:** Certificates & Policies.
- ✅ **Resilience:** Offline queuing.

**Next:** Day 85 - Advanced Capstone: Edge AI Optimization (TensorRT).

---

**Day 84 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
