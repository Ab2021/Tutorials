# Day 61: Android Automotive OS (AAOS) Overview
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Understand** the difference between Android Auto (Projection) and Android Automotive OS (Native).
2.  **Analyze** the AAOS Architecture: Car Service, HAL, VHAL, EVS.
3.  **Explore** the Camera Service in AAOS (Camera2 API vs EVS).
4.  **Set up** the AAOS build environment (Cuttlefish/Emulator).
5.  **Interact** with Car Properties (Gear, Speed, Turn Signals) via `adb`.
6.  **Debug** AAOS services using `dumpsys` and `logcat`.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** PC with 16GB+ RAM (for Emulator) or a supported board (Pixel, Raspberry Pi with GloDroid).
*   **Software:** Android Studio, AOSP Source Code (repo), Cuttlefish.
*   **Knowledge:** Android Binder, HAL (HIDL/AIDL), Linux Kernel.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is AAOS?
*   **Android Auto:** Your phone runs the OS, the car head unit is just a display (Remote Display).
*   **Android Automotive OS (AAOS):** Android runs *directly* on the car's hardware (IVY - In-Vehicle Infotainment). It controls HVAC, Radio, Cameras, and interacts with the CAN bus.

### 🔹 Part 2: AAOS Architecture
1.  **Application Framework:** Standard Android APIs + `android.car.*` APIs.
2.  **Car Service:** A system service that manages vehicle-specific functions.
3.  **Car API:** Java/Kotlin library for apps to talk to Car Service.
4.  **Vehicle HAL (VHAL):** The bridge between Android and the Vehicle Network (CAN/LIN).
5.  **Exterior View System (EVS):** A special high-priority, low-latency camera stack for Rear View Camera (RVC) and Surround View. *Crucial for safety compliance (boot in < 2s).*

### 🔹 Part 3: Camera Stack in AAOS
AAOS has **two** camera stacks:
1.  **Android Camera Service (Camera2 API):**
    *   Used by: Zoom, TikTok, Dashcam apps.
    *   Features: Rich ISP, Face Detect, 3A.
    *   Latency: High.
    *   Boot Time: Slow (starts after Zygote).
2.  **EVS (Exterior View System):**
    *   Used by: Reversing Camera, Surround View.
    *   Features: Raw/YUV streaming, Overlay.
    *   Latency: Ultra-low.
    *   Boot Time: Fast (starts directly from `init`).

---

## 💻 Implementation Examples

### Example 1: Building Cuttlefish (AAOS Emulator)

If you don't have a $2000 dev board, use Cuttlefish.

```bash
# 1. Install Dependencies
sudo apt install git-core gnupg flex bison gperf build-essential \
  zip curl zlib1g-dev gcc-multilib g++-multilib libc6-dev-i386 \
  lib32ncurses5-dev x11proto-core-dev libx11-dev lib32z1-dev \
  libgl1-mesa-dev libxml2-utils xsltproc unzip fontconfig

# 2. Download AOSP
mkdir aaos; cd aaos
repo init -u https://android.googlesource.com/platform/manifest -b android-14.0.0_r1
repo sync -j8

# 3. Build for Cuttlefish (x86_64)
source build/envsetup.sh
lunch aosp_cf_x86_64_auto-userdebug
m

# 4. Run
launch_cvd --start_webrtc=true
```

### Example 2: Interacting with Vehicle HAL (VHAL)

Simulating car signals (e.g., putting car in Reverse).

```bash
# List all vehicle properties
adb shell dumpsys android.hardware.automotive.vehicle.IVehicle/default

# Set Gear to Reverse (Gear ID: 4)
# Property ID for GEAR_SELECTION: 289408000
adb shell cmd car_service inject-vhal-event 289408000 4

# Set Speed to 60 km/h
# Property ID for PERF_VEHICLE_SPEED: 291504647
adb shell cmd car_service inject-vhal-event 291504647 60.0
```

### Example 3: Car API (Java)

Accessing car properties in an App.

```java
import android.car.Car;
import android.car.hardware.property.CarPropertyManager;
import android.car.VehiclePropertyIds;

public class CarSpeedMonitor {
    private CarPropertyManager mPropertyManager;

    public void init(Context context) {
        Car car = Car.createCar(context);
        mPropertyManager = (CarPropertyManager) car.getCarManager(Car.PROPERTY_SERVICE);
        
        // Register Callback for Speed
        mPropertyManager.registerCallback(new CarPropertyManager.CarPropertyEventCallback() {
            @Override
            public void onChangeEvent(CarPropertyValue value) {
                float speed = (float) value.getValue();
                Log.d("AAOS", "Current Speed: " + speed);
            }
        }, VehiclePropertyIds.PERF_VEHICLE_SPEED, CarPropertyManager.SENSOR_RATE_NORMAL);
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Exploring the AAOS UI

**Objective:** Familiarize with the interface.

**Steps:**
1.  Launch Cuttlefish or Emulator.
2.  Open the "Kitchen Sink" app (Developer tool).
3.  Test "HVAC" controls.
4.  Test "Sensor" readings.
5.  **Observation:** AAOS has a Status Bar, Navigation Bar (usually on side or bottom), and supports Multi-Display.

### Lab 2: Triggering the Rear View Camera

**Objective:** Understand the Reverse Gear trigger.

**Steps:**
1.  Open the Emulator.
2.  Use `adb` to inject the "Reverse Gear" event (Example 2).
3.  **Observation:** The screen should switch to the EVS Camera view (if implemented) or show a "Rear View" indicator.
4.  Inject "Drive" gear (ID: 8).
5.  **Observation:** Screen returns to Home.

### Lab 3: Analyzing `dumpsys`

**Objective:** Debug services.

**Steps:**
1.  Run `adb shell dumpsys car_service`.
2.  Look for `CarProjectionService`, `CarPowerManagementService`.
3.  **Task:** Find the current Power State (ON, SHUTDOWN_PREPARE, SUSPEND).

---

## 🐛 Debugging Techniques

### Debug 1: Emulator not starting

**Symptom:** `launch_cvd` fails.

**Cause:**
*   Virtualization (KVM) not enabled in BIOS.
*   User not in `kvm` group.
*   **Fix:** `sudo usermod -aG kvm $USER` and reboot.

### Debug 2: VHAL Property Ignored

**Symptom:** Injecting event does nothing.

**Cause:**
*   The property is Read-Only.
*   The property is not supported by the specific VHAL implementation (Default vs Vendor).
*   **Fix:** Check `get-vhal-prop` to see if the property exists and is writable.

---

## ⚡ Performance Optimization

### Optimization 1: Boot Optimization

*   AAOS must boot fast.
*   **Technique:** "Garage Mode". When the car is turned off, the OS doesn't fully shut down immediately. It enters a low-power maintenance mode to install updates, then suspends to RAM (STR) or disk.
*   **Cold Boot:** < 4 seconds required for RVC.

### Optimization 2: Early Audio/Video

*   Radio and Camera must work *before* Android fully loads.
*   **Technique:** Use EVS for camera and a separate microcontroller (or DSP) for audio routing.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why can't we just use the standard Camera2 API for the Reverse Camera?** (Too slow to boot, high latency, lower priority than system UI).
2.  **What is the role of the VHAL?**
3.  **How does AAOS handle multiple displays (Cluster + Infotainment)?** (Cluster is often a separate OS or a protected display layer).
4.  **What is "Garage Mode"?**

### Practical Challenges

1.  **Create a "Speed Warning" App:** Write an app that turns the screen red if `PERF_VEHICLE_SPEED` > 120 km/h.
2.  **Modify VHAL:** Edit the `DefaultVehicleHal.cpp` in AOSP to add a custom property (e.g., `CABIN_TEMPERATURE`).

---

## 📚 Further Reading & Resources

### Documentation
*   **source.android.com:** Android Automotive OS.
*   **Google Codelabs:** Building Apps for Android Automotive.

### Tools
*   **Sniff:** CAN bus sniffer (if using real hardware).

---

## 🎓 Summary

Today we covered:
- ✅ **AAOS:** Native Android for Cars.
- ✅ **Architecture:** Car Service, VHAL, EVS.
- ✅ **Development:** Building AOSP/Cuttlefish.
- ✅ **Interaction:** Injecting VHAL events.
- ✅ **Safety:** Why EVS exists.

**Next:** Day 62 - EVS (Exterior View System) Architecture Deep Dive.

---

**Day 61 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
