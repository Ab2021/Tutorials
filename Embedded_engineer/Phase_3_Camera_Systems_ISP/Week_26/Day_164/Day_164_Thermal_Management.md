# Day 164: Thermal Management
## Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power

---

## 🎯 Learning Objectives
1.  **Understand** the relationship between Power, Heat, and Performance.
2.  **Monitor** Thermal Zones using the Linux Thermal Framework (`/sys/class/thermal`).
3.  **Implement** Thermal Throttling policies (Passive vs Active).
4.  **Design** a Cooling Solution (Heatsink/Fan) for a camera module.
5.  **Stress Test** the system to find the Thermal Limit.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Jetson/Pi, Fan (PWM controlled).
*   **Software:** `stress-ng`, `thermal-engine`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Thermodynamics 101
*   **Heat Generation:** $P = C \times V^2 \times f$. (Power depends on Voltage squared and Frequency).
*   **Junction Temperature ($T_j$):** The temp inside the silicon. Max usually $105^\circ C$.
*   **Thermal Resistance ($\theta_{JA}$):** Resistance from Junction to Ambient.
*   **Goal:** Keep $T_j < T_{max}$.

### 🔹 Part 2: Linux Thermal Framework
*   **Thermal Zones:** Sensors (CPU, GPU, PMIC).
*   **Cooling Devices:** Fan (Active), CPU Frequency Scaling (Passive).
*   **Trip Points:**
    *   **Passive:** Start throttling (e.g., $85^\circ C$).
    *   **Active:** Turn on fan (e.g., $60^\circ C$).
    *   **Critical:** Emergency Shutdown (e.g., $100^\circ C$).

---

## 💻 Implementation Examples

### Example 1: Reading Temperatures (Python)

```python
import os
import glob

def get_temperatures():
    temps = {}
    zones = glob.glob('/sys/class/thermal/thermal_zone*')
    
    for zone in zones:
        try:
            with open(os.path.join(zone, 'type'), 'r') as f:
                name = f.read().strip()
            with open(os.path.join(zone, 'temp'), 'r') as f:
                temp = int(f.read().strip()) / 1000.0
            temps[name] = temp
        except:
            continue
    return temps

print(get_temperatures())
# Output: {'CPU-therm': 45.5, 'GPU-therm': 42.0}
```

### Example 2: Controlling a PWM Fan

Using Jetson GPIO.

```python
import Jetson.GPIO as GPIO
import time

FAN_PIN = 12 # PWM Pin
GPIO.setmode(GPIO.BOARD)
GPIO.setup(FAN_PIN, GPIO.OUT, initial=GPIO.HIGH)
pwm = GPIO.PWM(FAN_PIN, 100) # 100Hz
pwm.start(0)

def set_fan_speed(temp):
    if temp < 50:
        speed = 0
    elif temp < 70:
        speed = 50
    else:
        speed = 100
    pwm.ChangeDutyCycle(speed)

# Loop
while True:
    temps = get_temperatures()
    set_fan_speed(temps['CPU-therm'])
    time.sleep(1)
```

### Example 3: Defining Trip Points (Device Tree)

Configuring the kernel to handle it automatically.

```dts
thermal-zones {
    cpu-thermal {
        polling-delay-passive = <1000>;
        polling-delay = <1000>;
        thermal-sensors = <&tsens0>;

        trips {
            cpu_alert0: cpu-alert0 {
                temperature = <85000>; // 85C
                hysteresis = <2000>;
                type = "passive";
            };
            cpu_crit: cpu-crit {
                temperature = <100000>; // 100C
                hysteresis = <2000>;
                type = "critical";
            };
        };

        cooling-maps {
            map0 {
                trip = <&cpu_alert0>;
                cooling-device = <&cpu0 THERMAL_NO_LIMIT THERMAL_NO_LIMIT>;
            };
        };
    };
};
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Stress Testing

**Objective:** Heat it up.

**Steps:**
1.  Run `stress-ng --cpu 4 --io 2 --vm 1 --vm-bytes 1G --timeout 60s`.
2.  Monitor temperature script from Example 1.
3.  **Plot:** Temp vs Time.
4.  **Observe:** How fast does it rise? Does it plateau?

### Lab 2: Throttling Observation

**Objective:** See performance drop.

**Steps:**
1.  Run a heavy AI inference loop. Measure FPS.
2.  Cover the heatsink (simulate bad airflow).
3.  Watch Temp rise to $85^\circ C$.
4.  **Observe:** FPS drops as CPU/GPU frequency is throttled by the kernel.

### Lab 3: Fan Curve Tuning

**Objective:** Silence vs Cooling.

**Steps:**
1.  Implement a PID controller for the fan speed instead of simple steps.
2.  **Goal:** Keep temp at $65^\circ C$ with minimum noise.

---

## 🐛 Debugging Thermal Issues

### Debug 1: "System Shuts Down Randomly"

**Symptom:** Black screen, reboot.

**Cause:**
*   Hitting Critical Trip Point ($100^\circ C$).
*   **Fix:** Check heatsink contact. Apply new thermal paste. Check fan connector.

### Debug 2: "Sensor Noise"

**Symptom:** Image sensor gets noisy/grainy after 10 minutes.

**Cause:**
*   Sensor overheating. Dark Current doubles every $8^\circ C$.
*   **Fix:** The sensor needs a heatsink too! Or a thermal pad connecting it to the metal case.

---

## ⚡ Performance Optimization

### Optimization 1: Power Mode Selection

*   Jetson has modes: `MAXN` (Max Power), `15W`, `10W`.
*   Use `sudo nvpmodel -m <mode>` to limit the max power/heat.
*   Sometimes `15W` mode is better than `MAXN` because it sustains a steady frequency instead of throttling up and down.

### Optimization 2: Thermal Spreading

*   Use a Graphite Sheet or Copper Heat Spreader to distribute heat from the hot spot (SoC) to the rest of the case.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Dark Current"?** (Noise generated by heat in the image sensor, even with no light).
2.  **Difference between Active and Passive Cooling?** (Active = Fan/Pump. Passive = Heatsink/Case).
3.  **What is "Hysteresis"?** (The gap between turning ON and turning OFF. Prevents the fan from toggling rapidly at the threshold).

### Practical Challenges

1.  **Thermal Camera:** If you have a FLIR Lepton, connect it and visualize the heat distribution of your PCB.
2.  **Log to Cloud:** Send temperature telemetry to AWS IoT (from Day 84). Alert if a camera in the field is overheating.

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux Thermal Subsystem Documentation.**
*   **Jetson Thermal Design Guide.**

---

## 🎓 Summary

Today we covered:
- ✅ **Thermodynamics:** Heat kills electronics.
- ✅ **Monitoring:** `/sys/class/thermal`.
- ✅ **Control:** Fans and Throttling.
- ✅ **Device Tree:** Configuring trips.
- ✅ **Dark Current:** Why sensors need cooling.

**Next:** Day 165 - Boot Time Optimization.

---

**Day 164 Complete** | Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power


