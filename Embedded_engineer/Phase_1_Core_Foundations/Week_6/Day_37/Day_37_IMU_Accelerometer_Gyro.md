# Day 37: IMU (Accelerometer/Gyro)
## Phase 1: Core Embedded Engineering Foundations | Week 6: Sensors and Actuators

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the physics of MEMS Accelerometers and Gyroscopes.
2.  **Interface** an IMU (Inertial Measurement Unit) like MPU6050 (I2C) or LIS3DSH (SPI).
3.  **Convert** raw register values into physical units (g-force, deg/sec).
4.  **Implement** a Complementary Filter to fuse Accelerometer and Gyro data for stable pitch/roll estimation.
5.  **Calibrate** sensor offsets (Zero-G trim).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (LIS3DSH onboard).
    *   Optional: MPU6050 (I2C) module.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Serial Plotter
*   **Prior Knowledge:**
    *   Day 31 (SPI) or Day 33 (I2C)
    *   Day 24 (Filtering)
*   **Datasheets:**
    *   [LIS3DSH Datasheet](https://www.st.com/resource/en/datasheet/lis3dsh.pdf)
    *   [MPU6050 Register Map](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MEMS Physics

#### 1.1 Accelerometer
Measures **Proper Acceleration** (including gravity).
*   **Static:** Measures tilt (Gravity vector).
*   **Dynamic:** Measures movement/vibration.
*   **Problem:** Sensitive to vibration. Noisy.

#### 1.2 Gyroscope
Measures **Angular Velocity** (Rate of rotation).
*   **Output:** Degrees per second (dps).
*   **Integration:** $\theta = \int \omega dt$.
*   **Problem:** Drift. The integral accumulates error over time.

### 🔹 Part 2: Sensor Fusion
We need the stability of the Accelerometer (long term) and the responsiveness of the Gyro (short term).
*   **Complementary Filter:**
    *   $\theta_{new} = \alpha \times (\theta_{old} + \omega \times dt) + (1 - \alpha) \times \theta_{accel}$
    *   Usually $\alpha \approx 0.98$.
    *   Trust Gyro for short term, correct with Accel for long term.

---

## 💻 Implementation: LIS3DSH Tilt Sensor

> **Instruction:** We will use the onboard SPI Accelerometer to calculate Pitch and Roll.

### 🛠️ Hardware/System Configuration
*   **SPI1:** Configured as Day 31.

### 👨‍💻 Code Implementation

#### Step 1: Read and Convert (`imu.c`)

```c
#include <math.h>
#include "lis3dsh.h" // From Day 31

typedef struct {
    float ax, ay, az; // g
    float pitch, roll; // degrees
} IMU_Data_t;

void IMU_ReadAccel(IMU_Data_t *data) {
    int16_t raw_x, raw_y, raw_z;
    LIS3DSH_ReadAxes(&raw_x, &raw_y, &raw_z);
    
    // Sensitivity for +/- 2g range is usually 0.061 mg/LSB (Check datasheet!)
    // Actually LIS3DSH 2g sensitivity is 0.06 mg/digit.
    float sensitivity = 0.061f / 1000.0f; 
    
    data->ax = raw_x * sensitivity;
    data->ay = raw_y * sensitivity;
    data->az = raw_z * sensitivity;
}
```

#### Step 2: Calculate Angles
Using trigonometry (assuming static case):
*   $Roll = \text{atan2}(Y, Z)$
*   $Pitch = \text{atan2}(-X, \sqrt{Y^2 + Z^2})$

```c
void IMU_CalculateAngles(IMU_Data_t *data) {
    // Convert to degrees (180/PI = 57.29)
    data->roll  = atan2f(data->ay, data->az) * 57.29f;
    data->pitch = atan2f(-data->ax, sqrtf(data->ay*data->ay + data->az*data->az)) * 57.29f;
}
```

#### Step 3: Main Loop
```c
int main(void) {
    SPI1_Init();
    LIS3DSH_Init(); // Enable axes, 100Hz
    
    IMU_Data_t imu;
    
    while(1) {
        IMU_ReadAccel(&imu);
        IMU_CalculateAngles(&imu);
        
        printf("Pitch: %.2f, Roll: %.2f\r\n", imu.pitch, imu.roll);
        Delay_ms(50);
    }
}
```

---

## 🔬 Lab Exercise: Lab 37.1 - Digital Spirit Level

### 1. Lab Objectives
- Use the LEDs on the Discovery Board to indicate tilt.
- Flat = All Off. Tilt Forward = Top LED On.

### 2. Step-by-Step Guide

#### Phase A: Logic
*   Pitch > 5 deg -> Orange LED (PD13).
*   Pitch < -5 deg -> Green LED (PD12).
*   Roll > 5 deg -> Red LED (PD14).
*   Roll < -5 deg -> Blue LED (PD15).

#### Phase B: Implementation
```c
void Update_LEDs(float pitch, float roll) {
    GPIOD->ODR &= ~(0xF000); // Clear LEDs
    
    if (pitch > 5.0f)  GPIOD->ODR |= (1 << 13);
    if (pitch < -5.0f) GPIOD->ODR |= (1 << 12);
    if (roll > 5.0f)   GPIOD->ODR |= (1 << 14);
    if (roll < -5.0f)  GPIOD->ODR |= (1 << 15);
}
```

### 3. Verification
Tilt the board. The "ball" (lit LED) should roll towards the lowest point (or highest, depending on logic).

---

## 🧪 Additional / Advanced Labs

### Lab 2: MPU6050 Gyro Integration
- **Goal:** Read Gyro data via I2C.
- **Task:**
    1.  Init MPU6050 (Wake up, set range +/- 250 dps).
    2.  Read Gyro X, Y, Z.
    3.  Integrate: `Angle += Gyro * dt`.
    4.  Observe Drift: Keep board still. Angle will slowly increase.

### Lab 3: Complementary Filter
- **Goal:** Fuse Accel + Gyro.
- **Task:**
    1.  Implement the formula: `Angle = 0.98 * (Angle + Gyro*dt) + 0.02 * AccelAngle`.
    2.  Shake the board. The angle should remain stable (unlike Accel only).
    3.  Tilt the board. The angle should track (unlike Gyro only).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Wrong Values (e.g., 36000)
*   **Cause:** Reading `int16_t` as `uint16_t`.
*   **Solution:** Ensure variables are signed.

#### 2. Noise
*   **Cause:** MEMS sensors are noisy.
*   **Solution:** Enable the sensor's internal Low Pass Filter (LPF) via registers (e.g., `CTRL_REG` or `CONFIG`).

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Data Ready Interrupt:** Instead of polling, configure the IMU to pull a GPIO pin Low when new data is ready. Trigger SPI read in the EXTI ISR.

### Code Quality
- **Calibration:** Always implement a "Calibrate" routine that averages 1000 samples while flat to find the Zero-G offset. Subtract this offset from raw readings.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't we use Accelerometer alone for yaw (heading)?
    *   **A:** Gravity points down. Rotating around the Z-axis (Yaw) doesn't change the gravity vector relative to the sensor (unless it's tilted). You need a Magnetometer for Yaw.
2.  **Q:** What is "Gimbal Lock"?
    *   **A:** A mathematical singularity in Euler Angles (Pitch/Roll/Yaw) when Pitch = 90. Quaternions avoid this.

### Challenge Task
> **Task:** Implement "Shake Detection". Detect a sudden spike in total acceleration magnitude ($|A| = \sqrt{x^2+y^2+z^2}$) > 2g. Toggle an LED.

---

## 📚 Further Reading & References
- [A Guide to using IMU (Accelerometer and Gyroscope Devices)](https://ozzmaker.com/accelerometer-gyroscope/)

---
