# Day 72: RTK (Real-Time Kinematic) GPS
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 72 Focus:**
> 5 meters of error is fine for Google Maps. It's fatal for a self-driving car staying in a 3-meter lane. **RTK (Real-Time Kinematic)** GPS uses a reference station to cancel out atmospheric errors, achieving **2 cm accuracy**. Today, we learn how to get precision.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Code-based Positioning (Standard GPS) vs Carrier Phase Positioning (RTK).
2.  **Explain** the role of the Base Station and Rover.
3.  **Decode** RTCM messages (the language of corrections).
4.  **Implement** an NTRIP Client to fetch corrections from the internet.
5.  **Visualize** the difference between "Float" and "Fixed" RTK solutions.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 71:** GNSS Fundamentals.
-   **Networking:** TCP/IP (for NTRIP).

### Hardware Requirements
-   **RTK GNSS Module:** (Optional) U-Blox ZED-F9P (The gold standard for hobbyists/research).
-   **Antenna:** Multi-band active antenna.

### Software Stack
-   **Python:** `socket`, `base64`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why is GPS inaccurate?

The signal travels 20,000 km.
-   **Ionosphere:** Charged particles slow down the radio wave. Delay varies by day/night. (~5m error).
-   **Troposphere:** Weather (humidity) affects speed. (~0.5m error).
-   **Orbit/Clock:** Satellite position isn't perfect. (~1m error).

### 🔹 Part 2: Differential GPS (DGPS) & RTK

**Concept:**
1.  Place a **Base Station** at a *known* fixed location.
2.  Base calculates: "I am at X, but GPS says X+5. Error is +5."
3.  Base sends "Error = +5" to the **Rover** (Car).
4.  Rover subtracts 5 from its reading.

**RTK (Carrier Phase):**
Instead of just timing the code (0s and 1s), RTK counts the *number of waves* (Carrier Phase).
-   Wavelength of L1 signal $\approx 19$ cm.
-   If we can align the phase to 1% of a wave, we get millimeter precision.
-   **Integer Ambiguity Resolution:** The hardest math problem in GPS. Solving for the unknown number of whole cycles ($N$) between sat and receiver.

### 🔹 Part 3: NTRIP (Networked Transport of RTCM via Internet Protocol)

How does the Rover get corrections?
-   **Radio:** 915 MHz / 433 MHz (Line of Sight).
-   **Internet (NTRIP):** Base sends data to a Caster (Server). Rover connects to Caster via 4G/5G.

---

## 💻 Implementation: NTRIP Client

**Scenario:**
-   **Goal:** Connect to a public NTRIP Caster (e.g., RTK2GO) and receive RTCM data.
-   **Note:** We won't decode RTCM (it's binary and complex). We will just forward it to the GPS module (which does the decoding).

### 🛠️ Setup
Create `week11_day72` and `ntrip_client.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day72
cd ~/ros2_ws/src/week11_day72
touch ntrip_client.py
```

### 👨‍💻 Code: NTRIP Client

```python
import socket
import base64
import time
import sys

# --- Configuration ---
# Use a free caster like rtk2go.com
CASTER_HOST = "rtk2go.com"
CASTER_PORT = 2101
MOUNTPOINT = "STR_MOUNTAIN_VIEW" # Example mountpoint (Check rtk2go.com for active ones)
USER = "email@example.com" # Often required for free casters
PASSWORD = "none"

# NMEA GGA string is required by some casters to send virtual corrections (VRS)
# Send your approximate location
GGA_NMEA = "$GPGGA,123519,3723.2475,N,12202.2475,W,1,08,0.9,545.4,M,46.9,M,,*47\r\n"

class NTRIPClient:
    def __init__(self):
        self.sock = None
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((CASTER_HOST, CASTER_PORT))
            
            # HTTP GET Request
            user_pass = f"{USER}:{PASSWORD}"
            auth = base64.b64encode(user_pass.encode()).decode()
            
            req = (
                f"GET /{MOUNTPOINT} HTTP/1.0\r\n"
                f"User-Agent: NTRIP PythonClient/1.0\r\n"
                f"Authorization: Basic {auth}\r\n"
                f"Accept: */*\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            
            self.sock.sendall(req.encode())
            
            # Check Response
            response = self.sock.recv(4096) # Read headers
            if b"ICY 200 OK" in response or b"HTTP/1.0 200 OK" in response:
                print("Connected to Caster!")
                
                # Send GGA (needed for VRS)
                self.sock.sendall(GGA_NMEA.encode())
                return True
            else:
                print(f"Connection Failed: {response.decode('ascii', errors='ignore')}")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False

    def read_stream(self):
        # Read binary RTCM data
        try:
            data = self.sock.recv(1024)
            if not data:
                return None
            return data
        except:
            return None

    def close(self):
        if self.sock:
            self.sock.close()

def main():
    client = NTRIPClient()
    
    if not client.connect():
        sys.exit(1)
        
    print(f"Streaming RTCM from {CASTER_HOST}/{MOUNTPOINT}...")
    
    total_bytes = 0
    start_time = time.time()
    
    try:
        while True:
            data = client.read_stream()
            if data:
                total_bytes += len(data)
                # In a real system, you would write 'data' to the GPS Serial Port
                # gps_serial.write(data)
                
                # Print stats every second
                if time.time() - start_time > 1.0:
                    print(f"Received {len(data)} bytes. Total: {total_bytes/1024:.1f} KB")
                    start_time = time.time()
            else:
                print("Stream ended.")
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
        client.close()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: RTK Fix

### Lab Objectives
1.  **Setup:** If you have a ZED-F9P, connect it to U-Center (Windows) or use the script above to pipe data to it.
2.  **Observation:**
    -   **No RTCM:** Status = "3D Fix". Accuracy $\approx$ 1.5m. Position drifts slowly.
    -   **With RTCM:** Status changes to "3D/DGNSS" -> "Float" -> "Fixed".
    -   **RTK Fixed:** Accuracy drops to 0.02m (2cm). Position is rock solid.
3.  **Experiment:**
    -   Cover the antenna with your hand.
    -   **Result:** Status drops to "Float" or "3D Fix". Accuracy degrades instantly. RTK is fragile!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Float" but never "Fixed"
**Symptom:** Accuracy is ~20cm, but not 2cm.
**Cause:** Signal not strong enough (SNR < 35 dBHz) or Base Station too far (> 20km).
**Solution:** Get a better view of the sky. Find a closer mountpoint.

#### 2. Connection Refused
**Symptom:** NTRIP client fails.
**Cause:** Wrong Mountpoint or IP. Caster might be down.
**Solution:** Use a browser to check the Caster's Sourcetable (usually port 2101).

---

## ⚡ Optimization & Best Practices

### 1. Dual Antenna (GPS Compass)
RTK gives position, but not heading (unless moving).
-   Use **Two Antennas** separated by 1 meter.
-   Calculate heading from the relative position of Antenna 2 vs Antenna 1.
-   Gives accurate heading even when stationary (unlike magnetic compass, which is affected by metal).

### 2. Sensor Fusion
RTK is great, but it updates at 10Hz and can be blocked by bridges.
-   **Always** fuse RTK with IMU and Odometry (Kalman Filter).
-   IMU fills the gaps between GPS updates.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between "Float" and "Fixed"?
    *   **A:** **Float:** The integer ambiguity is estimated as a float (decimal). Good accuracy (decimeter). **Fixed:** The integer is solved (exact count). Best accuracy (centimeter).
2.  **Q:** Does RTK work in tunnels?
    *   **A:** No. It needs satellite signals.
3.  **Q:** What is the maximum distance to the Base Station?
    *   **A:** Typically < 20-30 km. Beyond that, atmospheric conditions at Base and Rover are too different to cancel out.

### Challenge Task
**Task:** Parse RTCM Packet Type.
1.  RTCM packets start with `0xD3`.
2.  Read the first byte of `data`. If `0xD3`, read length.
3.  Extract Message ID (e.g., 1077 for GPS MSM7). Print the ID.

---

## 📚 Further Reading & References
-   [RTKLIB (Open Source RTK)](http://www.rtklib.com/)
-   [U-Blox ZED-F9P Interface Description](https://www.u-blox.com/en/product/zed-f9p-module)

---

**Day 72 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
