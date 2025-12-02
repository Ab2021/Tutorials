# Day 27: GPS/GNSS & RTK
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 27 Focus:**
> SLAM drifts. Maps can be outdated. But the sky (usually) doesn't lie. **GNSS (Global Navigation Satellite System)** provides absolute positioning anywhere on Earth. Today, we go beyond the 5-meter accuracy of your phone and explore **RTK (Real-Time Kinematic)**, which achieves centimeter-level precision for autonomous driving.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the principle of Trilateration and how GNSS receivers calculate position.
2.  **Identify** sources of error: Ionospheric delay, Multipath, and Clock drift.
3.  **Master** the concept of Differential GPS (DGPS) and RTK (Carrier Phase).
4.  **Parse** NMEA 0183 sentences ($GPGGA, $GPRMC) using Python.
5.  **Visualize** GPS traces on an interactive map using `folium`.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Geometry:** Spheres and intersection.
-   **Communication:** Serial/UART (Day 1).

### Hardware Requirements
-   **GNSS Receiver:** USB GPS module (e.g., u-blox NEO-M8N) or a log file.
-   **RTK:** Requires a Base Station and Rover (conceptually).

### Software Stack
-   **Python Libraries:** `pyserial`, `folium`, `pynmea2`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How GNSS Works

**GNSS** is the umbrella term for:
-   **GPS** (USA)
-   **GLONASS** (Russia)
-   **Galileo** (Europe)
-   **BeiDou** (China)

#### 1.1 Trilateration
1.  Satellite sends a signal at time $t_1$ containing its position $(x_s, y_s, z_s)$.
2.  Receiver gets it at time $t_2$.
3.  Distance (Pseudorange) $d = c \times (t_2 - t_1)$.
4.  Receiver is on a sphere of radius $d$ around the satellite.
5.  Intersection of 3 spheres = 2 points (one in space, one on Earth).
6.  Intersection of 4 spheres = Unique point + Time correction.

#### 1.2 Sources of Error
-   **Ionosphere:** Charged particles slow down radio waves. (~5m error).
-   **Ephemeris:** Satellite orbit errors. (~2m).
-   **Clock:** Atomic clocks drift slightly.
-   **Multipath:** Signal bounces off buildings (Ghost signals). Major issue in cities ("Urban Canyon").

---

### 🔹 Part 2: RTK (Real-Time Kinematic)

Standard GPS accuracy: ~3-5 meters.
Autonomous Driving requirement: < 10 cm.

**Solution:** Use a **Base Station** at a known fixed location.

1.  Base Station sees satellites and calculates its position.
2.  It compares calculated pos with *known* pos. Difference = Error.
3.  Base sends **Corrections** (RTCM format) to the **Rover** (Car) via radio/internet (NTRIP).
4.  Rover subtracts the error.

#### 2.1 Carrier Phase
Standard GPS uses the *code* (bits) to measure time.
RTK uses the *carrier wave* (1.5 GHz sine wave).
-   Wavelength $\lambda \approx 19$ cm.
-   If we count the number of waves, we get cm-level precision.
-   **Ambiguity Resolution:** We don't know the *total* number of integer cycles ($N$) between sat and receiver. Solving for $N$ is the "Integer Ambiguity Problem".
    -   **Float Solution:** $N$ is estimated as a float (Accuracy: 50cm).
    -   **Fixed Solution:** $N$ is locked to an integer (Accuracy: 1cm).

---

### 🔹 Part 3: NMEA Protocol

GPS receivers output ASCII text over Serial (UART).
Standard: **NMEA 0183**.

**Example:** `$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47`

-   `$GPGGA`: Global Positioning System Fix Data.
-   `123519`: UTC Time (12:35:19).
-   `4807.038,N`: Latitude 48 deg 07.038' N.
-   `01131.000,E`: Longitude 11 deg 31.000' E.
-   `1`: Fix Quality (0=Invalid, 1=GPS, 2=DGPS, 4=RTK Fixed).
-   `08`: Number of Satellites.
-   `0.9`: HDOP (Horizontal Dilution of Precision). Lower is better.
-   `545.4,M`: Altitude (Meters).

---

## 💻 Implementation: GPS Logger & Visualizer

We will write a script `gps_tool.py` that:
1.  Reads NMEA from a serial port (or simulates it).
2.  Parses GPGGA sentences.
3.  Logs coordinates to a CSV.
4.  Generates an HTML map using `folium`.

### 🛠️ Setup
Create `week4_day27` and `gps_tool.py`.

```bash
mkdir -p ~/ros2_ws/src/week4_day27
cd ~/ros2_ws/src/week4_day27
pip install pyserial pynmea2 folium
touch gps_tool.py
```

### 👨‍💻 Code: NMEA Parser & Mapper

```python
import serial
import pynmea2
import time
import csv
import folium
import os

class GPSLogger:
    def __init__(self, port, baudrate=9600, simulate=False):
        self.simulate = simulate
        self.coordinates = [] # List of (lat, lon)
        
        if not simulate:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                print(f"Connected to {port}")
            except serial.SerialException as e:
                print(f"Error: {e}")
                self.simulate = True
                print("Switching to Simulation Mode")
        
        # Simulation Data (A drive around a block)
        self.sim_data = [
            "$GPGGA,120000,3746.000,N,12225.000,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120001,3746.001,N,12225.000,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120002,3746.002,N,12225.000,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120003,3746.002,N,12225.001,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120004,3746.002,N,12225.002,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120005,3746.001,N,12225.002,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120006,3746.000,N,12225.002,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120007,3746.000,N,12225.001,W,1,08,0.9,10.0,M,0,M,,*47",
            "$GPGGA,120008,3746.000,N,12225.000,W,1,08,0.9,10.0,M,0,M,,*47",
        ]

    def read_nmea(self):
        if self.simulate:
            for line in self.sim_data:
                yield line
                time.sleep(0.5)
        else:
            while True:
                try:
                    line = self.ser.readline().decode('ascii', errors='replace')
                    if line:
                        yield line
                except Exception as e:
                    print(e)
                    break

    def run(self, duration=10):
        start_time = time.time()
        print("Logging GPS data...")
        
        with open('gps_log.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Latitude', 'Longitude', 'Altitude', 'Quality'])
            
            for line in self.read_nmea():
                if time.time() - start_time > duration:
                    break
                    
                if line.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(line)
                        lat = msg.latitude
                        lon = msg.longitude
                        alt = msg.altitude
                        qual = msg.gps_qual
                        
                        print(f"Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt}, Qual: {qual}")
                        
                        self.coordinates.append((lat, lon))
                        writer.writerow([msg.timestamp, lat, lon, alt, qual])
                        
                    except pynmea2.ParseError:
                        continue

    def generate_map(self):
        if not self.coordinates:
            print("No coordinates to map.")
            return
            
        # Center map on first point
        start_coords = self.coordinates[0]
        m = folium.Map(location=start_coords, zoom_start=18)
        
        # Draw path
        folium.PolyLine(self.coordinates, color="blue", weight=2.5, opacity=1).add_to(m)
        
        # Add markers
        folium.Marker(self.coordinates[0], popup="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(self.coordinates[-1], popup="End", icon=folium.Icon(color='red')).add_to(m)
        
        m.save("gps_trace.html")
        print("Map saved to gps_trace.html")

if __name__ == "__main__":
    # Replace 'COM3' or '/dev/ttyUSB0' with your port
    logger = GPSLogger(port='COM3', simulate=True)
    logger.run(duration=5) # Run for 5 seconds (simulation is fast)
    logger.generate_map()
```

---

## 🔬 Lab Exercise: Urban Canyon Effect

### Lab Objectives
1.  Take a GPS receiver (or phone logging app) for a walk.
2.  **Scenario A:** Open field / Park.
    -   *Observation:* Path is smooth. Accuracy < 3m.
3.  **Scenario B:** Downtown / Between tall buildings.
    -   *Observation:* Path jumps around (Multipath). Altitude fluctuates wildly.
4.  **Analysis:** Plot `HDOP` (Horizontal Dilution of Precision) from the NMEA log. High HDOP = Bad Geometry = Poor Accuracy.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. No Fix (Quality = 0)
**Symptom:** Receiver outputs commas `,,,,,`.
**Cause:**
-   Indoors (No signal).
-   Cold Start (Needs to download Almanac, takes 12 mins).
**Solution:** Go outside. Wait.

#### 2. Drift while stationary
**Symptom:** Position wanders in a 5m radius.
**Cause:** Normal GPS noise.
**Solution:** Use a Kalman Filter (Day 9) with a Motion Model (Velocity = 0) to filter this out.

#### 3. Coordinate Format
**Symptom:** Position is miles off.
**Cause:** NMEA uses `DDMM.MMMM` (Degrees + Minutes). Maps use `DD.DDDD` (Decimal Degrees).
**Solution:** `pynmea2` handles this conversion. If manual: `Decimal = Degrees + Minutes/60`.

---

## ⚡ Optimization & Best Practices

### 1. Dual Antenna GPS
Use two antennas on the car (front/back).
-   Allows calculating **Heading** (Yaw) even when stationary.
-   Single antenna only knows heading when moving.

### 2. NTRIP Client
To get RTK corrections without your own base station:
-   Connect to a public NTRIP caster (e.g., RTK2GO).
-   Send corrections to the GPS module via Serial.

### 3. Sensor Fusion
Never trust GPS alone for control.
-   GPS (1Hz, Absolute, Noisy) + IMU (200Hz, Relative, Smooth) + Odom.
-   Use the Factor Graph from Day 24.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the minimum number of satellites needed for a 3D fix?
    *   **A:** 4 (X, Y, Z, Time).
2.  **Q:** What does "RTK Fixed" mean?
    *   **A:** The integer ambiguity is resolved. Accuracy is ~1cm. "RTK Float" means it's still estimating (Accuracy ~50cm).
3.  **Q:** Why is Z (Altitude) usually less accurate than X/Y?
    *   **A:** Satellites are only above you, not below you. The geometry (DOP) is worse vertically.

### Challenge Task
**Task:** Geofencing.
1.  Define a polygon (e.g., a parking lot) using Lat/Lon.
2.  Read live GPS.
3.  Check if the point is inside the polygon (Ray Casting algorithm).
4.  Trigger an alarm if outside.

---

## 📚 Further Reading & References
-   [GPS Compendium (u-blox)](https://www.u-blox.com/en/gps-compendium) - Excellent technical guide.
-   [NMEA 0183 Reference](https://gpsd.gitlab.io/gpsd/NMEA.html)

---

**Day 27 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
