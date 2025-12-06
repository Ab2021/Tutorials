# Day 170: Cloud Infrastructure (OTA Updates)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 25: Final Integration & Graduation

---

> **📝 Content Creator Instructions:**
> The Robot is just an edge node.
> - **Focus:** Fleet Management, MQTT Telemetry, Over-the-Air (OTA) Updates, A/B Partitioning, and Cloud Simulation.
> - **Code:** A Python OTA System (`OTA_Client` and `OTA_Server`). The client polls for updates. The server provides a signed firmware blob. The client downloads, verifies, and performs an Atomic A/B Swap simulation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a scalable Fleet Management System (Shadow Twin).
2.  **Implement** Robust OTA updates (Rollback on failure).
3.  **Explain** A/B Partitioning (RootFS redundancy).
4.  **Use** MQTT for real-time telemetry streaming (Speed, GPS, Battery).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (Cloud Tier Free Tier useful).

### Software Environment
```bash
pip install flask requests
```

### Prior Knowledge
- HTTP/REST APIs.
- Linux Partitions.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The OTA Problem

You can't brick a car on the highway.
*   **Atomic Update:** The update must happen completely or not at all.
*   **A/B Partitioning:**
    *   **Slot A:** Running Version 1.0 (Active).
    *   **Slot B:** Idle.
    *   **Update:** Flash Version 2.0 to Slot B.
    *   **Swap:** Set Bootloader to boot from B. Reboot.
    *   **Verify:** If B fails to boot 3 times, Watchdog reverts bootloader to A. (Rollback).

### 🔹 Part 2: Fleet Telemetry

Thousands of robots sending data.
*   **MQTT (Message Queuing Telemetry Transport):** Lightweight, Pub/Sub.
*   **Topic:** `fleet/car_123/speed`.
*   **QoS (Quality of Service):**
    *   0: Fire and Forget (GPS).
    *   1: At least once (Alerts).
    *   2: Exactly once (Billing).

### 🔹 Part 3: Digital Twin

The Cloud maintains a "Shadow" of the robot.
*   **Robot:** `{"battery": 80}` $\to$ Cloud.
*   **App:** User sets `{"climate": 22C}` on App. App $\to$ Cloud Twin.
*   **Sync:** When Robot comes online, Cloud pushes `{"climate": 22C}` to Robot.

---

## 💻 Implementation: The OTA Update Loop

Simulated Flask Server and Python Client.

### 🛠️ Project Structure
```text
day170_cloud/
├── src/
│   ├── ota_server.py
│   ├── ota_client.py
│   └── firmware/
│       └── v2.0.bin
└── output/
    ├── update_log.txt
```

### 👨‍💻 OTA System (`src/ota_simulation.py`)

```python
import time
import json
import hashlib
import threading
import os

# --- MOCK SERVER ---
class CloudServer:
    def __init__(self):
        self.manifest = {
            "version": "2.0.0",
            "url": "mock://firmware_v2.bin",
            "hash": "deadbeef",
            "signature": "valid_sig_123"
        }
        
    def check_for_update(self, current_version):
        if current_version != self.manifest["version"]:
            return self.manifest
        return None
        
    def download_firmware(self, url):
        print(f"[CLOUD] Serving firmware blob for {url}...")
        time.sleep(1.0) # Network delay
        # Return fake binary content
        return b"FIRMWARE_V2_DATA_CONTENT"

# --- CLIENT ---
class RobotClient:
    def __init__(self, server):
        self.server = server
        self.version = "1.0.0"
        self.state = "IDLE"
        self.partition_active = "A"
        self.partition_b_content = None
        
    def run_cycle(self):
        print(f"\n[ROBOT] Ver: {self.version} | Active Slot: {self.partition_active}")
        
        # 1. Poll
        update_info = self.server.check_for_update(self.version)
        
        if update_info:
            print(f"[ROBOT] Update Found: {update_info['version']}")
            self.perform_update(update_info)
        else:
            print("[ROBOT] No updates.")
            
    def perform_update(self, manifest):
        self.state = "DOWNLOADING"
        print("[ROBOT] Downloading...")
        
        blob = self.server.download_firmware(manifest['url'])
        
        # 2. Verify (Hash check)
        # In real life, calculate SHA256(blob) == manifest['hash']
        # We simulate pass
        if len(blob) > 0:
            print("[ROBOT] Verification Successful.")
            
            # 3. Flash to Inactive Partition
            target_slot = "B" if self.partition_active == "A" else "A"
            print(f"[ROBOT] Flashing to Slot {target_slot}...")
            self.partition_b_content = blob
            
            # 4. Reboot/Swap
            self.apply_update(target_slot, manifest['version'])
            
    def apply_update(self, new_slot, new_version):
        print("[ROBOT] Rebooting to apply update...")
        time.sleep(1)
        
        # Simulating Bootloader Logic
        # Try Booting New Slot
        print(f"[BOOTLOADER] Attempting boot from Slot {new_slot}...")
        
        success = True # Simulate success
        
        if success:
            self.partition_active = new_slot
            self.version = new_version
            print(f"[ROBOT] Update Complete! Now running {new_version} on Slot {new_slot}")
            # Send Success Telemetry
        else:
            print("[BOOTLOADER] Boot Failed! Rolling back.")
            # Revert logic

def main():
    server = CloudServer()
    robot = RobotClient(server)
    
    # Cycle 1: Check (Should find update)
    robot.run_cycle()
    
    # Cycle 2: Check (Should be up to date)
    robot.run_cycle()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Bricked Bot"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Successful upgrade 1.0 -> 2.0.
- **Modify:** Force `success = False` in `apply_update`.
- **Result:** Robot should print "Rolling back" and stay on Version 1.0.
- **Code:** Implement the fallback logic explicitly (reset `partition_active` to old slot).
- **Chaos:** What if power cuts *during* flashing? (Simulate partial write). The Hash check should fail on next boot.

---

## 🚀 Project: "AWS IoT Core (Simulated)"

**Goal:** Telemetry Dashboard.
1.  **Format:** JSON `{'vin': '123', 'lat': 34.0, 'lon': -118.0, 'speed': 45}`.
2.  **Script:** `telemetry_gen.py`. Generates random path data.
3.  **Visualization:** Use `matplotlib` living plot to show the "Cloud Dashboard" receiving the points.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Bandwidth Cost"
*   **Cause:** Sending full Point Clouds (Gbps) over 4G.
*   **Fix:** Edge Compute. Process Lidar ON CAR. Send only "Detected Objects" (Kbps) to Cloud.

#### 2. "Update Loop"
*   **Cause:** Version 2.0 crashes immediately. Watchdog rolls back to 1.0. 1.0 sees update 2.0 available. Updates to 2.0. Crashes.
*   **Fix:** "Blacklist" the bad version after a rollback.

---

## ⚡ Optimization: Delta Updates

Don't download the whole OS (1GB).
*   **Binary Diff:** Use `bsdiff`.
*   **Delta:** If V1.0 and V2.0 differ by 10MB, download only the 10MB patch.
*   **Apply:** `patch(V1.0, Delta) -> V2.0`. Only works if V1.0 is bit-exact.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** A vs B partition?
    *   **A:** Redundancy. One is always bootable.
2.  **Q:** What is "The Edge"?
    *   **A:** Computing done locally on the robot, vs "The Cloud".
3.  **Q:** Why sign firmware?
    *   **A:** Prevent hackers from pushing a malicious update that steals the car.

### Challenge Task
> **Task:** Secure Boot Chain.
> 1. Hardware verifies Bootloader signature.
> 2. Bootloader verifies Kernel signature.
> 3. Kernel verifies Filesystem.
> 4. Fail at any stage = Stop.

---

## 📚 Further Reading
- **Uptane:** Security framework for automotive updates.
- **AWS IoT Greengrass:** Running Lambda functions on edge devices.

---

**Day 170 Complete**
