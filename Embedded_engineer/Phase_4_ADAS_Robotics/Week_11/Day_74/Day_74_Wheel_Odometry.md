# Day 74: Wheel Odometry (Dead Reckoning)
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 74 Focus:**
> Before GPS, sailors used "Dead Reckoning" (Deduced Reckoning). Robots do the same. By counting how many times the wheels turn, we can calculate how far we have traveled. This is **Wheel Odometry**. It's the backbone of indoor navigation.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Derive** the Forward Kinematics for a Differential Drive robot.
2.  **Calculate** position $(x, y, \theta)$ from encoder ticks.
3.  **Model** Odometry Errors: Systematic (Wheel size) vs Non-Systematic (Slip).
4.  **Implement** an Odometry Node in Python that publishes `nav_msgs/Odometry`.
5.  **Calibrate** wheel diameter and track width using the UMBmark method.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Trigonometry:** Sine, Cosine.
-   **Kinematics:** $v = \omega r$.

### Hardware Requirements
-   **Encoders:** Optical or Magnetic (Hall Effect) on motors.

### Software Stack
-   **Python:** `math`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Differential Drive Kinematics

Two wheels, Left ($L$) and Right ($R$), separated by Track Width $b$.
-   **Linear Velocity ($v$):** Average of wheel speeds.
    $$ v = \frac{v_R + v_L}{2} $$
-   **Angular Velocity ($\omega$):** Difference of wheel speeds.
    $$ \omega = \frac{v_R - v_L}{b} $$
-   **Wheel Speed from Ticks:**
    $$ v_{wheel} = \frac{\Delta \text{ticks}}{\Delta t} \times \frac{2\pi r}{N} $$
    ($r$ = radius, $N$ = ticks per revolution).

### 🔹 Part 2: Integration (Dead Reckoning)

Given $v_k$ and $\omega_k$ at step $k$:
$$ x_{k+1} = x_k + v_k \cos(\theta_k) \Delta t $$
$$ y_{k+1} = y_k + v_k \sin(\theta_k) \Delta t $$
$$ \theta_{k+1} = \theta_k + \omega_k \Delta t $$
*Note: For better accuracy, use Runge-Kutta integration or the exact arc solution.*

### 🔹 Part 3: Sources of Error

1.  **Systematic (Calibration):**
    -   **Unequal Wheel Diameters:** Robot drives in a curve instead of straight.
    -   **Uncertain Wheelbase ($b$):** Robot turns more/less than expected.
2.  **Non-Systematic (Random):**
    -   **Wheel Slip:** Wheel turns but robot doesn't move (ice/oil).
    -   **Rough Terrain:** Bumps change effective radius.

---

## 💻 Implementation: Odometry Calculator

**Scenario:**
-   **Input:** Left and Right encoder ticks.
-   **Output:** Robot Pose $(x, y, \theta)$.

### 🛠️ Setup
Create `week11_day74` and `odometry.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day74
cd ~/ros2_ws/src/week11_day74
touch odometry.py
```

### 👨‍💻 Code: Odometry Class

```python
import math
import matplotlib.pyplot as plt

class Odometry:
    def __init__(self, wheel_radius, track_width, ticks_per_rev):
        self.r = wheel_radius
        self.b = track_width
        self.N = ticks_per_rev
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.prev_ticks_l = 0
        self.prev_ticks_r = 0
        self.first_run = True
        
    def update(self, ticks_l, ticks_r, dt):
        if self.first_run:
            self.prev_ticks_l = ticks_l
            self.prev_ticks_r = ticks_r
            self.first_run = False
            return self.x, self.y, self.theta
            
        # 1. Calculate Delta Ticks
        d_ticks_l = ticks_l - self.prev_ticks_l
        d_ticks_r = ticks_r - self.prev_ticks_r
        
        # Handle wrap-around if necessary (not implemented here)
        
        self.prev_ticks_l = ticks_l
        self.prev_ticks_r = ticks_r
        
        # 2. Calculate Wheel Distances
        d_l = 2 * math.pi * self.r * (d_ticks_l / self.N)
        d_r = 2 * math.pi * self.r * (d_ticks_r / self.N)
        
        # 3. Calculate Robot Motion
        d_center = (d_l + d_r) / 2.0
        d_theta = (d_r - d_l) / self.b
        
        # 4. Integrate Pose (Runge-Kutta 2nd Order / Midpoint)
        # Use theta + d_theta/2 for better accuracy
        self.x += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y += d_center * math.sin(self.theta + d_theta / 2.0)
        self.theta += d_theta
        
        # Normalize Theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        return self.x, self.y, self.theta

def main():
    # Robot Specs
    R = 0.05 # 5cm radius
    B = 0.20 # 20cm track width
    TPR = 1000 # Ticks per rev
    
    odom = Odometry(R, B, TPR)
    
    # Simulation: Square Path (1m x 1m)
    # 1. Forward 1m
    # 2. Turn Left 90 deg
    # 3. Forward 1m
    # ...
    
    path_x = []
    path_y = []
    
    ticks_l = 0
    ticks_r = 0
    dt = 0.1
    
    print("Simulating Square Path...")
    
    for leg in range(4):
        # Move Forward 1m
        dist = 1.0
        steps = int(dist / (0.2 * dt)) # 0.2 m/s
        for _ in range(steps):
            # Both wheels turn same speed
            d_ticks = (0.2 * dt) / (2 * math.pi * R) * TPR
            ticks_l += d_ticks
            ticks_r += d_ticks
            
            x, y, th = odom.update(ticks_l, ticks_r, dt)
            path_x.append(x)
            path_y.append(y)
            
        # Turn Left 90 deg (pi/2)
        # d_theta = (dr - dl) / b => pi/2 = (d - (-d))/b = 2d/b => d = pi/2 * b / 2
        arc_len = (math.pi / 2.0) * B / 2.0
        steps = int(arc_len / (0.1 * dt)) # Turn speed
        for _ in range(steps):
            d_ticks = (0.1 * dt) / (2 * math.pi * R) * TPR
            ticks_l -= d_ticks # Left wheel back
            ticks_r += d_ticks # Right wheel fwd
            
            x, y, th = odom.update(ticks_l, ticks_r, dt)
            path_x.append(x)
            path_y.append(y)
            
    # Plot
    plt.plot(path_x, path_y, label="Odometry")
    plt.axis('equal')
    plt.grid()
    plt.title("Wheel Odometry Simulation")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The UMBmark Calibration

### Lab Objectives
1.  **Concept:** Systematic errors (wrong radius/wheelbase) cause drift even on perfect ground.
2.  **Procedure:**
    -   Program robot to drive a 4m x 4m square.
    -   Measure the final position error $(x_{err}, y_{err})$.
    -   **Type A Error (Wheelbase):** If robot turns too much/little (Square becomes Trapezoid).
    -   **Type B Error (Diameter):** If robot drives too far/short (Square scales up/down).
3.  **Correction:**
    -   $b_{new} = b_{old} \times \frac{90^\circ}{90^\circ - \alpha_{err}}$.
    -   $r_{new} = r_{old} \times \frac{L_{actual}}{L_{measured}}$.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Integer Overflow
**Symptom:** Position jumps wildly.
**Cause:** Encoder ticks (int32) wrap around.
**Solution:** Handle overflow logic. `if new < old - 10000: wrap detected`.

#### 2. Polarity Reversal
**Symptom:** Robot thinks it's turning right when turning left.
**Cause:** Encoder A/B phases swapped or motor wired backwards.
**Solution:** Swap A/B wires or multiply ticks by -1 in software.

---

## ⚡ Optimization & Best Practices

### 1. High Frequency
Run odometry loop as fast as possible (>= 50Hz).
-   Integration error minimizes as $dt \to 0$.
-   Use interrupts (hardware timers) to count ticks, not polling.

### 2. Gyro Fusion
Wheel odometry is terrible for rotation (slippage amplifies angle error).
-   **Best Practice:** Use Encoders for $v$ (Distance) and Gyro for $\omega$ (Angle).
-   This "Fused Odometry" is extremely robust indoors.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Differential Drive called "Differential"?
    *   **A:** Because steering is achieved by the *difference* in speed between the two wheels.
2.  **Q:** What happens if one wheel slips on ice?
    *   **A:** The encoder counts ticks, so the robot *thinks* it moved. But it didn't. This creates a permanent offset in the map.
3.  **Q:** Why use Runge-Kutta integration?
    *   **A:** Simple Euler ($x += v \cos \theta$) assumes straight lines between steps. RK accounts for the curvature *during* the step.

### Challenge Task
**Task:** Ackermann Odometry.
1.  Derive kinematics for a car (Steering Angle $\delta$).
2.  $R = L / \tan(\delta)$.
3.  $\omega = v / R$.
4.  Implement `AckermannOdometry` class.

---

## 📚 Further Reading & References
-   [UMBmark: A Method for Measuring, Comparing, and Correcting Dead-reckoning Errors](http://www-personal.umich.edu/~johannb/Papers/umbmark.pdf)
-   [Probabilistic Robotics (Thrun)](https://docs.ros.org/en/melodic/api/robot_pose_ekf/html/)

---

**Day 74 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
