# Day 178: Control & Logic (State Machine)
## Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project

---

## 🎯 Learning Objectives
1.  **Implement** a Finite State Machine (FSM) to manage robot behavior (Idle, Follow, Avoid, Stop).
2.  **Design** a PID Controller for smooth line following.
3.  **Integrate** Perception outputs (Lane Deviation, Object Distance) into the Control Loop.
4.  **Handle** Safety Events (e.g., "Lost Line", "Battery Low").
5.  **Tune** the PID gains ($K_p, K_i, K_d$) for the specific robot chassis.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Robot with working Motors and Perception Pipeline.
*   **Software:** Python `statemachine` library (optional) or custom class.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Finite State Machine (FSM)
*   **States:** Distinct modes of operation.
    *   **IDLE:** Waiting for start command.
    *   **FOLLOW_LANE:** Normal operation.
    *   **OBSTACLE_STOP:** Object detected in path.
    *   **INTERSECTION:** QR code detected, deciding turn.
*   **Transitions:** Events that cause state change.
    *   `Start Button` -> IDLE to FOLLOW.
    *   `Object < 0.5m` -> FOLLOW to STOP.
    *   `Object > 0.8m` -> STOP to FOLLOW.

### 🔹 Part 2: PID Control
*   **Error ($e$):** `Lane Deviation` (0.0 = Center).
*   **Proportional ($P$):** $K_p \times e$. Steer harder if error is large.
*   **Integral ($I$):** $K_i \times \int e$. Fix steady-state error (e.g., if robot is always slightly left).
*   **Derivative ($D$):** $K_d \times \frac{de}{dt}$. Dampen the oscillation. Don't overshoot.
*   **Output:** `Steering Angle` (Differential Speed).

### 🔹 Part 3: Safety Watchdogs
*   If Perception thread dies, the `latest_state` becomes stale.
*   **Logic:** If `timestamp` of state is > 200ms old, STOP immediately.

---

## 💻 Implementation Examples

### Example 1: The PID Class

```python
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output
```

### Example 2: The State Machine

```python
import time

class RobotController:
    def __init__(self, perception, motors):
        self.perception = perception
        self.motors = motors
        self.state = "IDLE"
        self.pid = PID(Kp=1.0, Ki=0.0, Kd=0.1)
        self.base_speed = 0.5

    def run(self):
        while True:
            state_data = self.perception.get_state()
            
            # Safety Check
            if time.time() - state_data['timestamp'] > 0.2:
                self.motors.stop()
                print("SAFETY STOP: Stale Data")
                continue

            # State Logic
            if self.state == "IDLE":
                self.motors.stop()
                # Wait for MQTT command to start...
                
            elif self.state == "FOLLOW":
                # Check Obstacles
                if self.check_obstacle(state_data['objects']):
                    self.state = "STOPPED"
                    continue
                
                # PID Steering
                error = state_data['deviation']
                steering = self.pid.compute(error, 0.033) # 30fps = 0.033s
                
                left = self.base_speed + steering
                right = self.base_speed - steering
                
                # Clamp
                left = max(min(left, 1.0), -1.0)
                right = max(min(right, 1.0), -1.0)
                
                self.motors.set_speed(left, right)

            elif self.state == "STOPPED":
                self.motors.stop()
                if not self.check_obstacle(state_data['objects']):
                    self.state = "FOLLOW"

            time.sleep(0.01)

    def check_obstacle(self, objects):
        for obj in objects:
            if obj['distance'] < 0.5: # 50cm
                return True
        return False
```

### Example 3: Handling QR Commands

```python
# Inside loop
if state_data['qrs']:
    cmd = state_data['qrs'][0] # {"action": "stop"}
    if cmd['action'] == 'stop':
        self.state = "IDLE"
    elif cmd['action'] == 'slow':
        self.base_speed = 0.3
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: PID Tuning (Ziegler-Nicholsish)

**Objective:** Stop the wobble.

**Steps:**
1.  Set $K_i = 0, K_d = 0$. Increase $K_p$ until robot oscillates constantly.
2.  Set $K_p$ to half that value.
3.  Increase $K_d$ until oscillation stops (Damping).
4.  Increase $K_i$ slightly if it fails to center perfectly on curves.

### Lab 2: Obstacle Stop Test

**Objective:** Don't crash.

**Steps:**
1.  Robot moves forward.
2.  Jump in front of it.
3.  **Verify:** Robot stops within 20cm.
4.  **Latency Check:** If it hits you, check the pipeline latency.

### Lab 3: "Lost Line" Recovery

**Objective:** What if the line breaks?

**Steps:**
1.  Tape a line with a 10cm gap.
2.  **Logic:** If no line detected, keep previous steering angle (or go straight) for 1 second.
3.  If still no line after 1 second, STOP and spin to search.

---

## 🐛 Debugging Control

### Debug 1: "Robot Spins in Circles"

**Symptom:** PID output saturates.

**Cause:**
*   Sign error. Positive error should cause Negative steering (or vice versa).
*   **Fix:** Flip the sign of `steering` in `left = base + steering`.

### Debug 2: "Integral Windup"

**Symptom:** Robot overshoots massively after a long curve.

**Cause:**
*   The $I$ term accumulated too much error.
*   **Fix:** Clamp the integral term to a max value. Reset integral to 0 when error crosses zero.

---

## ⚡ Performance Optimization

### Optimization 1: Frequency Matching

*   Perception runs at 30Hz. Control loop runs at 100Hz?
*   No point running Control faster than Perception unless you have other sensors (Encoders/IMU) that update faster.
*   If using IMU, run Control at 100Hz to stabilize orientation, while Vision updates Lane Deviation at 30Hz.

### Optimization 2: Soft Stop

*   Instead of `set_speed(0, 0)` (Hard Stop, robot tips over), use a deceleration ramp.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Hysteresis" in state transitions?** (Prevents rapid toggling. Stop at 0.5m, Start only when clear > 0.6m).
2.  **Why do we need the $D$ term in PID?** (To predict the future. If error is decreasing fast, start counter-steering *before* we cross the center).
3.  **What is a "Race Condition" in FSM?** (Two events happening at once. e.g., "Stop Sign" and "Obstacle". Priority logic is needed).

### Practical Challenges

1.  **Figure 8:** Program the robot to follow a Figure-8 track. Tune PID for both Left and Right turns.
2.  **Emergency Stop:** Wire a physical button to a GPIO. If pressed, cut power to motors (Software or Hardware).

---

## 📚 Further Reading & Resources

### Documentation
*   **"Feedback Control for Computer Systems" (Book).**
*   **Python State Machine Library.**

---

## 🎓 Summary

Today we covered:
- ✅ **FSM:** Managing states.
- ✅ **PID:** Smooth control.
- ✅ **Safety:** Watchdogs and Hysteresis.
- ✅ **Integration:** Connecting Vision to Motors.
- ✅ **Tuning:** The art of $K_p, K_i, K_d$.

**Next:** Day 179 - Cloud & UI Dashboard.

---

**Day 178 Complete** | Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project


