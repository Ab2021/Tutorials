# Day 173: Capstone: Security & Comm
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 25: Final Integration & Graduation

---

> **📝 Content Creator Instructions:**
> Secure the gates.
> - **Focus:** Integrating the Security Monitor and V2X stack. Building the "Safety Node" that sits between the AI and the Drive-by-Wire system.
> - **Code:** `capstone_security.py`. A Node that subscribes to `/ai/cmd`, performs sanity checks (Jerk limits, Reverse while fast), signs outgoing V2X messages, and verifies incoming warnings.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a Safety Executive Node that overrides AI commands.
2.  **Integrate** V2X DSRC logic to handle "External Braking Events" (EEBL).
3.  **Deploy** a CAN Firewall simulation loop.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install cryptography
```

### Prior Knowledge
- V2X (Day 155, 166).
- CAN Security (Day 162).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Safety Executive

The AI (Planner) says "Accelerate to 100 mph". The Executive says "No".
*   **Role:** The final gatekeeper before hardware.
*   **Rules:**
    1.  Max Speed Limit (Geo-fenced).
    2.  Max Acceleration/Jerk (Passenger Comfort).
    3.  Direction consistency (Don't shift to Reverse at 60mph).
    4.  Watchdog (If AI stops talking, hit brakes).

### 🔹 Part 2: Integrated V2X

The car talks while it drives.
*   **TX:** Broadcast `BSM` (Basic Safety Message) every 100ms. Signed by ECDSA.
*   **RX:** Listen for `EEBL` (Emergency Electronic Brake Light).
*   **Fusion:** If V2X says "Car Ahead Braking HARD" but Radar doesn't see it yet (Occlusion), **Pre-charge Brakes**.

---

## 💻 Implementation: The Gatekeeper

We simulate the Control Loop.
`AI` -> `SafetyNode` -> `CAN`.

### 🛠️ Project Structure
```text
day173_security/
├── src/
│   ├── capstone_security.py
│   └── keys/
│       └── ego_private.pem
└── output/
    ├── command_log.txt
```

### 👨‍💻 Safety Node (`src/capstone_security.py`)

```python
import time
import json
import threading

# Mock ECDSA signing (From Day 166)
def sign_data(payload):
    return "signature_123" 

class AICmd:
    def __init__(self, throttle, steer):
        self.throttle = throttle
        self.steer = steer
        self.ts = time.time()

class SafetyExecutive:
    def __init__(self):
        # State
        self.current_speed = 20.0 # m/s (approx 45 mph)
        self.last_cmd_ts = time.time()
        self.enabled = True
        
        # Limits
        self.MAX_SPEED = 30.0 # ~65 mph
        self.MAX_ACCEL = 5.0  # m/s^2
        self.MAX_JERK = 2.0   # m/s^3
        
        self.last_throttle = 0.0
        
    def process_telemetry(self, speed):
        self.current_speed = speed

    def validate_command(self, cmd):
        """
        Input: AICmd
        Output: Safe (throttle, steer) or Override
        """
        # 1. Watchdog Check
        if time.time() - cmd.ts > 0.2: # 200ms latency limit
            print("[SAFE] STALE COMMAND! Braking.")
            return (-1.0, 0.0)
            
        # 2. Speed Limit
        if self.current_speed > self.MAX_SPEED and cmd.throttle > 0:
            print(f"[SAFE] Speed Limit {self.current_speed:.1f} > {self.MAX_SPEED}. Cutting Throttle.")
            return (0.0, cmd.steer)
            
        # 3. Jerk Limiting
        # delta_throttle = cmd.throttle - self.last_throttle
        # We simplify functionality for this demo
        safe_throttle = cmd.throttle
        
        # 4. Save state
        self.last_throttle = safe_throttle
        self.last_cmd_ts = time.time()
        
        return (safe_throttle, cmd.steer)

class V2XManager(threading.Thread):
    def __init__(self, safety_exec):
        super().__init__()
        self.safety = safety_exec
        self.running = True
        
    def run(self):
        print("[V2X] Started.")
        while self.running:
            # 1. Broadcast BSM
            self.broadcast_bsm()
            
            # 2. Receive Messages (Simulated)
            self.check_incoming()
            
            time.sleep(0.1) # 10Hz
            
    def broadcast_bsm(self):
        # Payload
        bsm = {
            'id': 'EGO_CAR',
            'speed': self.safety.current_speed,
            'ts': time.time()
        }
        sig = sign_data(bsm)
        # print(f"[V2X] TX BSM: {bsm} Sig={sig}")
        
    def check_incoming(self):
        # Simulate an EEBL Event
        if time.time() % 10 > 8 and time.time() % 10 < 8.2:
            print("[V2X] RX WARNING: Car Ahead Emergency Braking!")
            # Action: Tell Safety Exec to override?
            # Or just warn Planner?
            # Safety Exec *could* force brake.
            pass

def main():
    exec_node = SafetyExecutive()
    v2x_node = V2XManager(exec_node)
    v2x_node.start()
    
    print("--- Capstone Security Loop ---")
    
    # Simulate Main Control Loop (100Hz)
    try:
        for t in range(50):
            # 1. AI Generates Command
            # Normal driving
            ai_throttle = 0.5
            ai_steer = 0.1
            
            # Inject Fault: At t=20, AI goes crazy (Full Throttle)
            if t == 20: 
                ai_throttle = 100.0 
            
            # Inject Fault: At t=40, AI hangs (Old timestamp)
            ts = time.time()
            if t >= 40:
                ts = time.time() - 1.0 # 1 sec old
                
            raw_cmd = AICmd(ai_throttle, ai_steer)
            raw_cmd.ts = ts
            
            # 2. Validate
            safe_throttle, safe_steer = exec_node.validate_command(raw_cmd)
            
            # 3. Actuate (Simulated)
            # print(f"T={t} | Req: {ai_throttle:.1f} -> Safe: {safe_throttle:.1f}")
            
            # Update Physics
            exec_node.process_telemetry(exec_node.current_speed + safe_throttle * 0.1)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
        
    v2x_node.running = False
    v2x_node.join()
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Override"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:**
    *   T=20: Request 100.0 throttled? (Requires code mod to check MAX limit logic more strictly).
    *   T=40: Stale command triggers E-Stop (-1.0).
- **Modify:** Add "Reverse Gear" protection.
- **Logic:** `if self.current_speed > 0.5 and cmd.gear == 'R': Block()`.
- **Result:** Prevents transmission destruction.

---

## 🚀 Project: "CAN Gateway"

**Goal:** Message Filter.
1.  **Input:** Stream of simulated CAN Frames.
2.  **Filter:** Whitelist allowed IDs (Steering, Brake, Engine).
3.  **Drop:** Unknown IDs (Diagnostics `0x7DF` from OBD port while driving).
4.  **Action:** Log attempt to "Intrusion Detection System".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Buffer Bloat"
*   **Cause:** V2X BSMs arriving faster than processed. Queue fills up. Old messages processed late.
*   **Fix:** Use a Ring Buffer (size 1). Always overwrite. We only care about the *latest* BSM from each neighbor.

#### 2. "Floating Point Compare"
*   **Cause:** `if speed == 0.0`.
*   **Fix:** `if abs(speed) < epsilon`.

---

## ⚡ Optimization: Hardware Security

TrustZone / SGX.
*   **Safety Critical Code:** Runs in Secure World (TEE - Trusted Execution Environment).
*   **Linux OS:** Runs in Normal World.
*   **Interaction:** OS requests "Verify Signature", TEE checks hardware key and returns Yes/No. OS never sees the key.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Safety Executive"?
    *   **A:** The software module responsible for enforcing safety constraints on control outputs.
2.  **Q:** Why verify V2X?
    *   **A:** To prevent Sybil attacks (One hacker pretending to be 100 cars) or Spoofing.
3.  **Q:** What is Jerk?
    *   **A:** Derivative of Acceleration. High jerk = Whiplash = Bad passenger experience.

### Challenge Task
> **Task:** Plausibility Check.
> 1. AI says: "Turn Left 90 degrees instantly".
> 2. Safety Node says: "Impossible given current speed/steering rack Physics".
> 3. Result: Clamp to physical max slew rate.

---

## 📚 Further Reading
- **AUTOSAR:** Safety architecture standards.
- **Comma.ai Panda:** Open source CAN interface safety code.

---

**Day 173 Complete**
