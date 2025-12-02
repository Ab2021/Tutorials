# Day 176: Hardware Integration (Camera + Sensors + Actuators)
## Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project

---

## 🎯 Learning Objectives
1.  **Interface** the Motor Driver (L298N/PCA9685) with Jetson GPIO/I2C.
2.  **Integrate** the Camera Module (IMX219) and verify streaming.
3.  **Add** Auxiliary Sensors: Ultrasonic (HC-SR04) or IMU (MPU6050).
4.  **Implement** a Hardware Abstraction Layer (HAL) in Python.
5.  **Test** the complete electrical system (Power, Control, Sensing).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Assembled Robot Chassis.
*   **Software:** `Jetson.GPIO`, `smbus2` (for I2C).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Motor Control (PWM)
*   **H-Bridge:** Circuit to control direction of DC motors.
*   **PWM (Pulse Width Modulation):** Controls speed.
    *   Duty Cycle 100% = Max Speed.
    *   Duty Cycle 50% = Half Speed.
*   **Differential Drive:**
    *   Turn Left = Right Motor Forward, Left Motor Backward (or Slower).
    *   Turn Right = Left Motor Forward, Right Motor Backward.

### 🔹 Part 2: Sensor Fusion (IMU)
*   **Accelerometer:** Measures gravity/acceleration. Noisy.
*   **Gyroscope:** Measures rotation rate. Drifts over time.
*   **Complementary Filter:** Combines both to get stable Pitch/Roll.
    *   $\theta = \alpha \times (\theta_{gyro} + \omega \times dt) + (1-\alpha) \times \theta_{accel}$.

### 🔹 Part 3: Hardware Abstraction Layer (HAL)
*   Code shouldn't know it's running on a Jetson or a Pi.
*   `class MotorDriver`: `set_speed(left, right)`.
*   `class Camera`: `get_frame()`.
*   Allows swapping hardware without rewriting logic.

---

## 💻 Implementation Examples

### Example 1: Motor Driver HAL (L298N)

```python
import Jetson.GPIO as GPIO

class MotorDriver:
    def __init__(self, enA, in1, in2, enB, in3, in4):
        self.pins = [enA, in1, in2, enB, in3, in4]
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pins, GPIO.OUT)
        
        self.pwmA = GPIO.PWM(enA, 1000) # 1kHz
        self.pwmB = GPIO.PWM(enB, 1000)
        self.pwmA.start(0)
        self.pwmB.start(0)
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4

    def set_speed(self, left, right):
        # Left Motor
        GPIO.output(self.in1, GPIO.HIGH if left > 0 else GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW if left > 0 else GPIO.HIGH)
        self.pwmA.ChangeDutyCycle(abs(left) * 100)

        # Right Motor
        GPIO.output(self.in3, GPIO.HIGH if right > 0 else GPIO.LOW)
        GPIO.output(self.in4, GPIO.LOW if right > 0 else GPIO.HIGH)
        self.pwmB.ChangeDutyCycle(abs(right) * 100)

    def stop(self):
        self.set_speed(0, 0)
        
    def cleanup(self):
        self.pwmA.stop()
        self.pwmB.stop()
        GPIO.cleanup()
```

### Example 2: Camera HAL (GStreamer)

```python
import cv2

class Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
```

### Example 3: Ultrasonic Sensor (HC-SR04)

```python
import time

class Ultrasonic:
    def __init__(self, trig, echo):
        self.trig = trig
        self.echo = echo
        GPIO.setup(trig, GPIO.OUT)
        GPIO.setup(echo, GPIO.IN)

    def get_distance(self):
        GPIO.output(self.trig, True)
        time.sleep(0.00001)
        GPIO.output(self.trig, False)

        start = time.time()
        stop = time.time()

        while GPIO.input(self.echo) == 0:
            start = time.time()
        while GPIO.input(self.echo) == 1:
            stop = time.time()

        elapsed = stop - start
        distance = (elapsed * 34300) / 2 # cm
        return distance
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Donut" Test

**Objective:** Calibrate Motors.

**Steps:**
1.  Command `set_speed(0.5, 0.5)`.
2.  Does the robot go straight? Usually no (one motor is weaker).
3.  **Fix:** Add a trim factor. `right_speed *= 0.95`.
4.  Command `set_speed(0.5, -0.5)`.
5.  Does it spin in place?

### Lab 2: Camera Latency Check

**Objective:** Verify pipeline.

**Steps:**
1.  Start Camera HAL.
2.  Show frame with `cv2.imshow`.
3.  Wave hand. Is it snappy?
4.  **Note:** `cv2.imshow` is slow over SSH (X11 Forwarding). Use a local display or WebRTC for true test.

### Lab 3: Sensor Noise

**Objective:** Filter Ultrasonic.

**Steps:**
1.  Read distance 100 times.
2.  Plot values. You will see spikes (e.g., 2000cm).
3.  **Fix:** Implement a Median Filter (Window size 5).

---

## 🐛 Debugging Hardware

### Debug 1: "Motors Hum but Don't Move"

**Symptom:** High pitched noise.

**Cause:**
*   PWM frequency too high? Or Duty Cycle too low (Stall Torque).
*   Battery voltage too low.
*   **Fix:** Increase min duty cycle (e.g., start at 30%). Check battery.

### Debug 2: "Camera Green Screen"

**Symptom:** Green or Pink output.

**Cause:**
*   Loose CSI cable.
*   Wrong Bayer format in GStreamer.
*   **Fix:** Reseat cable. Check `nvarguscamerasrc` settings.

---

## ⚡ Performance Optimization

### Optimization 1: Threaded Camera

*   The `get_frame()` call blocks until a frame arrives.
*   Move capture to a separate thread that constantly updates a `self.latest_frame` variable.
*   Main loop reads `self.latest_frame` instantly (non-blocking).

### Optimization 2: Soft-Start Motors

*   Jumping from 0 to 100% speed causes current spikes and wheel slip.
*   Ramp up speed: 0 -> 20 -> 40 -> ... -> 100 over 500ms.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why use a Motor Driver? Why not connect motors to GPIO?** (GPIO provides 3.3V @ 20mA. Motors need 12V @ 2A. GPIO would burn instantly).
2.  **What is "Back EMF"?** (Voltage generated by the spinning motor. Can damage electronics. Flyback diodes prevent this).
3.  **Why use "BCM" pin numbering?** (Broadcom chip numbering. Standard for Pi/Jetson GPIO libraries).

### Practical Challenges

1.  **Odometer:** Use the IMU or Wheel Encoders to calculate "Distance Traveled".
2.  **Battery Monitor:** Use an ADC (Analog to Digital Converter) to read battery voltage. If < 10V, stop motors and flash LED.

---

## 📚 Further Reading & Resources

### Documentation
*   **Jetson GPIO Library.**
*   **L298N Datasheet.**

---

## 🎓 Summary

Today we covered:
- ✅ **Motors:** PWM Control.
- ✅ **Sensors:** Ultrasonic & IMU.
- ✅ **HAL:** Abstracting hardware.
- ✅ **Integration:** Putting it together.
- ✅ **Testing:** Donuts and Latency.

**Next:** Day 177 - Perception Pipeline (AI + CV).

---

**Day 176 Complete** | Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project


