# Day 71: GNSS Fundamentals (GPS, NMEA)
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 71 Focus:**
> To drive autonomously, you need to know where you are. **GNSS (Global Navigation Satellite System)** is the only sensor that provides absolute position (Latitude, Longitude) anywhere on Earth. Today, we decode the signals from space.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the principle of Trilateration using Time-of-Flight (ToF).
2.  **Decode** NMEA 0183 sentences (`$GPGGA`, `$GPRMC`) to extract position and velocity.
3.  **Convert** Geodetic coordinates (Lat/Lon) to Cartesian coordinates (UTM).
4.  **Analyze** sources of GPS error: Multipath, Atmospheric delay, and Dilution of Precision (DOP).
5.  **Implement** a Python GPS driver that parses serial data.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Geography:** Latitude, Longitude, Altitude.
-   **Serial Communication:** UART/RS-232.

### Hardware Requirements
-   **GPS Module:** (Optional) U-Blox NEO-6M or similar.
-   **USB-TTL Adapter:** To connect to PC.

### Software Stack
-   **Python:** `pyserial`, `pynmea2`, `utm`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How GPS Works (Trilateration)

Satellites broadcast their position ($x_s, y_s, z_s$) and the time of transmission ($t_s$).
The receiver measures the time of arrival ($t_r$).
Distance (Pseudorange) $\rho = c \times (t_r - t_s)$.
We need 4 satellites to solve for 4 unknowns: $x, y, z$ (Receiver Position) and $\delta t$ (Receiver Clock Bias).
$$ (x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2 = (c(t_r - t_s - \delta t))^2 $$

### 🔹 Part 2: NMEA 0183 Standard

GPS modules output ASCII text over UART.
**$GPGGA (Global Positioning System Fix Data):**
`$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47`
-   `123519`: UTC Time (12:35:19).
-   `4807.038,N`: Latitude 48 deg 07.038' N.
-   `01131.000,E`: Longitude 11 deg 31.000' E.
-   `1`: Fix Quality (1=GPS, 2=DGPS, 4=RTK, 0=Invalid).
-   `08`: Number of Satellites.
-   `0.9`: HDOP (Horizontal Dilution of Precision).

### 🔹 Part 3: Coordinate Systems

-   **WGS84 (World Geodetic System 1984):** Ellipsoid. Lat/Lon/Alt. Good for maps, bad for math (meters).
-   **UTM (Universal Transverse Mercator):** Projects Earth onto 60 zones. $x$ (Easting), $y$ (Northing) in meters.
    -   Essential for robotics (calculating distance/speed in meters).

---

## 💻 Implementation: GPS Driver & UTM Converter

**Scenario:**
-   **Input:** Simulated NMEA stream (or real Serial port).
-   **Output:** UTM coordinates $(x, y)$ and status.

### 🛠️ Setup
Create `week11_day71` and `gps_driver.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day71
cd ~/ros2_ws/src/week11_day71
touch gps_driver.py
```

### 👨‍💻 Code: GPS Driver

```python
import serial
import pynmea2
import utm
import time
import math

class GPSDriver:
    def __init__(self, port=None, baud=9600, sim_mode=True):
        self.sim_mode = sim_mode
        if not sim_mode:
            self.ser = serial.Serial(port, baud, timeout=1)
        
        # Simulation State (San Francisco)
        self.lat = 37.7749
        self.lon = -122.4194
        self.heading = 0.0
        
    def read(self):
        if self.sim_mode:
            return self.simulate_nmea()
        else:
            line = self.ser.readline().decode('ascii', errors='replace')
            return line.strip()

    def simulate_nmea(self):
        # Simulate moving car
        self.lat += 0.00001 * math.cos(self.heading)
        self.lon += 0.00001 * math.sin(self.heading)
        self.heading += 0.01
        
        # Format as GPGGA
        # Lat: DDMM.MMMM
        lat_deg = int(self.lat)
        lat_min = (abs(self.lat) - abs(lat_deg)) * 60
        lat_str = f"{abs(lat_deg):02d}{lat_min:07.4f}"
        ns = 'N' if self.lat >= 0 else 'S'
        
        lon_deg = int(self.lon)
        lon_min = (abs(self.lon) - abs(lon_deg)) * 60
        lon_str = f"{abs(lon_deg):03d}{lon_min:07.4f}"
        ew = 'E' if self.lon >= 0 else 'W'
        
        # Checksum calculation (Simplified, pynmea2 handles parsing)
        # Construct raw string for pynmea2 to parse
        msg = pynmea2.GGA('GP', 'GGA', (
            time.strftime("%H%M%S"),
            lat_str, ns,
            lon_str, ew,
            '1', '08', '1.0', '100.0', 'M', '0.0', 'M', '', ''
        ))
        return str(msg)

    def parse(self, nmea_sentence):
        try:
            msg = pynmea2.parse(nmea_sentence)
            if isinstance(msg, pynmea2.types.talker.GGA):
                return {
                    'type': 'GGA',
                    'lat': msg.latitude,
                    'lon': msg.longitude,
                    'alt': msg.altitude,
                    'sats': msg.num_sats,
                    'hdop': msg.horizontal_dil
                }
            elif isinstance(msg, pynmea2.types.talker.RMC):
                return {
                    'type': 'RMC',
                    'speed': msg.spd_over_grnd * 1.852, # Knots to km/h
                    'course': msg.true_course
                }
        except pynmea2.ParseError:
            return None
        return None

def main():
    gps = GPSDriver(sim_mode=True)
    
    print("Starting GPS Driver...")
    print("Lat, Lon -> UTM Easting, UTM Northing")
    
    try:
        while True:
            line = gps.read()
            data = gps.parse(line)
            
            if data and data['type'] == 'GGA':
                # Convert to UTM
                easting, northing, zone_num, zone_letter = utm.from_latlon(data['lat'], data['lon'])
                
                print(f"[{data['sats']} Sats] "
                      f"Lat: {data['lat']:.6f}, Lon: {data['lon']:.6f} -> "
                      f"UTM: {easting:.2f} E, {northing:.2f} N (Zone {zone_num}{zone_letter})")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Urban Canyon

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The UTM coordinates change smoothly as the simulated car moves.
3.  **Experiment:**
    -   Simulate "Multipath" noise. Add random jumps to `lat`/`lon` in `simulate_nmea`.
    -   **Result:** The UTM path becomes jagged.
    -   **Challenge:** Implement a "Jump Filter". If the distance between two consecutive points corresponds to > 200 km/h, ignore the new point.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Serial Permission Denied
**Symptom:** `PermissionError: [Errno 13]`.
**Cause:** User not in `dialout` group (Linux).
**Solution:** `sudo usermod -a -G dialout $USER`.

#### 2. No Fix (Empty fields)
**Symptom:** `,,,,,` in NMEA.
**Cause:** Antenna not connected or indoors.
**Solution:** Go outside. Or check antenna voltage (active antennas need 3.3V).

---

## ⚡ Optimization & Best Practices

### 1. Baud Rate
Default NMEA is 9600 baud (1 Hz).
-   **Configure** the module (using UBX commands for U-Blox) to 115200 baud and 10 Hz update rate.
-   Essential for high-speed robotics.

### 2. Binary Protocol (UBX)
Parsing ASCII NMEA is slow and loses precision.
-   Use binary protocols (e.g., UBX) for lower latency and more data (e.g., carrier phase).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is HDOP?
    *   **A:** Horizontal Dilution of Precision. A multiplier for error. Low HDOP (< 1.0) is good (satellites spread out). High HDOP (> 5.0) is bad (satellites clustered).
2.  **Q:** Why convert to UTM?
    *   **A:** Because calculating "Distance = $\sqrt{\Delta x^2 + \Delta y^2}$" works in UTM (meters). It does NOT work in Lat/Lon (degrees).
3.  **Q:** How many satellites are needed for a 3D fix?
    *   **A:** Minimum 4. (3 for position, 1 for time).

### Challenge Task
**Task:** Waypoint Navigation.
1.  Define a target waypoint (Lat/Lon).
2.  Calculate the bearing to the target using `math.atan2`.
3.  Print "Turn Left" or "Turn Right" based on current heading.

---

## 📚 Further Reading & References
-   [GPS Compendium (u-blox)](https://www.u-blox.com/en/gps-compendium)
-   [NMEA 0183 Reference](https://www.gpsinformation.org/dale/nmea.htm)

---

**Day 71 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
