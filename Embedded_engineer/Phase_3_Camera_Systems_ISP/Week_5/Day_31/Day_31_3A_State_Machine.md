# Day 31: 3A State Machine & Synchronization
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Design** a central 3A State Machine to coordinate AE, AWB, and AF.
2.  **Implement** synchronization logic (e.g., "Don't AF while AE is unstable").
3.  **Handle** Pre-Flash and Flash sequences.
4.  **Manage** 3A Locks (AE-Lock, AWB-Lock) for video recording.
5.  **Debug** race conditions and state deadlocks.
6.  **Analyze** the frame delay between "Request" and "Result" (Pipeline Latency).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with full 3A control.
*   **Software:** C/C++ Compiler, State Machine Framework (or simple switch-case).
*   **Knowledge:** Finite State Machines (FSM), Event-Driven Programming.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Need for Coordination
AE, AWB, and AF are not independent.
*   **AE vs AWB:** If AE is oscillating (brightness changing), AWB stats will fluctuate, causing color shifts.
*   **AE vs AF:** If the image is too dark, AF cannot find contrast. AE must converge *before* AF starts.
*   **AF vs AE:** Moving the lens changes the effective aperture (breathing) or FoV, slightly affecting AE stats.

### 🔹 Part 2: The 3A State Machine
We need a Supervisor FSM.
*   **States:** `IDLE`, `PREVIEW`, `CONVERGING`, `LOCKED`, `CAPTURE`.
*   **Triggers:** `Scene_Change`, `Half_Press`, `Full_Press`.

### 🔹 Part 3: Pipeline Latency (The "Frame Delay")
When you write a new Exposure Gain at Frame N, it doesn't apply instantly.
*   **Frame N:** Software calculates Gain. Writes I2C.
*   **Frame N+1:** Sensor integrates with *old* gain (double buffering).
*   **Frame N+2:** Sensor integrates with *new* gain.
*   **Frame N+3:** ISP receives Frame N+2.
*   **Result:** You see the effect of your change 2-3 frames later. The Control Loop MUST account for this delay to avoid oscillation.

---

## 💻 Implementation Examples

### Example 1: The 3A State Definitions

```cpp
enum class AeState {
    INACTIVE,
    SEARCHING,
    CONVERGED,
    LOCKED,
    FLASH_REQUIRED,
    PRECAPTURE
};

enum class AfState {
    INACTIVE,
    PASSIVE_SCAN, // Continuous AF
    ACTIVE_SCAN,  // Triggered AF
    FOCUSED,
    FAILED,
    LOCKED
};

enum class AwbState {
    INACTIVE,
    SEARCHING,
    CONVERGED,
    LOCKED
};

struct ThreeAState {
    AeState ae;
    AfState af;
    AwbState awb;
};
```

### Example 2: The Supervisor Logic (Coordination)

```cpp
/**
 * @brief 3A Supervisor Loop
 * Called every frame (SOF)
 */
void run_3a_supervisor(ThreeAState& state, const Stats& stats) {
    
    // 1. Check Scene Change
    bool scene_changed = detect_scene_change(stats);
    if (scene_changed && state.ae == AeState::LOCKED) {
        // If locked (e.g. video), maybe don't unlock? 
        // Or unlock if change is drastic.
        state.ae = AeState::SEARCHING;
        state.awb = AwbState::SEARCHING;
    }
    
    // 2. Coordinate AE and AF
    if (state.af == AfState::ACTIVE_SCAN) {
        // AF is running.
        // If AE is searching, it might mess up AF stats.
        // But we usually let them run in parallel unless low light.
        
        if (is_low_light(stats)) {
            // Force AE to settle first
            if (state.ae != AeState::CONVERGED) {
                pause_af(); 
            } else {
                resume_af();
            }
        }
    }
    
    // 3. Run Algorithms
    run_ae(state.ae, stats);
    run_awb(state.awb, stats);
    run_af(state.af, stats);
}
```

### Example 3: Handling Pipeline Latency

We use a Queue to track "Pending" requests.

```cpp
struct ExposureRequest {
    int frame_id;
    float gain;
    int exposure_time;
};

std::deque<ExposureRequest> pending_requests;

void run_ae_with_latency(int current_frame_id, float current_luma) {
    // 1. Check if the current stats reflect our last request
    if (!pending_requests.empty()) {
        ExposureRequest last_req = pending_requests.back();
        int delay = 2; // Sensor delay
        
        if (current_frame_id < last_req.frame_id + delay) {
            // The stats we see now are from BEFORE our last change took effect.
            // Do NOT calculate new exposure yet. Wait.
            return;
        }
        
        // Request applied. Remove from queue.
        pending_requests.pop_back();
    }
    
    // 2. Calculate New Exposure
    // ... (PID Logic) ...
    
    // 3. Apply and Queue
    apply_exposure(new_gain, new_time);
    pending_requests.push_back({current_frame_id, new_gain, new_time});
}
```

### Example 4: Pre-Capture Sequence (Flash)

When user presses shutter (Full Press):
1.  **Lock 3A:** Stop all hunting.
2.  **Pre-Flash:** Fire a short flash burst.
3.  **Measure:** Calculate necessary Flash Power.
4.  **Main Flash:** Fire high power flash + Capture.
5.  **Unlock:** Return to preview.

```cpp
void handle_capture_sequence() {
    // 1. Lock
    state.ae = AeState::LOCKED;
    state.awb = AwbState::LOCKED;
    
    // 2. Pre-Flash
    fire_flash(LOW_POWER);
    wait_for_frame();
    Stats pre_flash_stats = get_stats();
    
    // 3. Calculate Main Flash Power
    // Difference between Ambient (locked) and Pre-Flash stats
    float flash_needed = calculate_flash_power(ambient_stats, pre_flash_stats);
    
    // 4. Capture
    configure_flash(flash_needed);
    trigger_capture();
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Latency Characterization

**Objective:** Measure the exact frame delay of your system.

**Steps:**
1.  Point camera at a static scene.
2.  Frame 0: Send command to double the Gain (2x).
3.  Record mean Luma for Frame 0, 1, 2, 3, 4, 5.
4.  **Observation:**
    *   Frame 0: No change.
    *   Frame 1: No change (or partial).
    *   Frame 2: Brightness doubles.
5.  **Result:** Latency = 2 frames. Hardcode this into your AE loop.

### Lab 2: The "Race Condition"

**Objective:** Observe AE/AWB fighting.

**Steps:**
1.  Point camera at a colored screen (e.g., Blue) that changes brightness.
2.  Enable aggressive AE and AWB.
3.  Change brightness rapidly.
4.  **Observation:** Colors might shift wildly (Blue -> Grey -> Blue) because AWB reacts to the brightness change (which affects saturation perception) before AE settles.
5.  **Fix:** Slow down AWB or lock AWB during large AE jumps.

### Lab 3: AF Triggering

**Objective:** Implement "Touch to Focus".

**Steps:**
1.  User clicks on coordinates (x, y).
2.  **State Machine:**
    *   Set AF ROI to (x, y).
    *   Set `AF_State = ACTIVE_SCAN`.
    *   Wait for `AF_State == FOCUSED`.
    *   Set `AE_State = LOCKED` (optional, AE lock on focus point).
3.  **Result:** Camera focuses on the selected object and stays there.

---

## 🐛 Debugging Techniques

### Debug 1: Deadlock

**Symptom:** Camera stuck in "Searching" forever.

**Cause:**
*   AE is waiting for AF to finish.
*   AF is waiting for AE to converge.
*   **Fix:** Timeouts. If AF takes > 2 seconds, force `AF_FAILED` and let AE resume.

### Debug 2: Flash Exposure Wrong

**Symptom:** Flash pictures are white or black.

**Cause:**
*   Timing mismatch. The flash fired *between* frames (during blanking) or the sensor exposure window didn't overlap with the flash pulse.
*   **Fix:** Use hardware strobe synchronization (FSYNC/STROBE pins).

---

## ⚡ Performance Optimization

### Optimization 1: Parallel Execution

*   AE, AWB, and AF stats calculation happens in Hardware (ISP).
*   The Control Algorithms run on CPU (ARM).
*   Run them in parallel threads?
    *   **No.** Usually run sequentially in a single "3A Thread" to avoid race conditions and ensure deterministic state transitions.

### Optimization 2: Event-Driven

*   Don't poll. Wait for the "Stats Ready" interrupt from the ISP.
*   Run the 3A loop immediately after stats are available to minimize latency.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the difference between "Passive Scan" (CAF) and "Active Scan" (Touch AF)?**
2.  **Why do we need a "Pre-Flash"? Why not just fire the main flash?**
3.  **How does "Pipeline Depth" affect the stability of the PID controller?**
4.  **What is "AE Lock" and when is it used?**

### Practical Challenges

1.  **Implement a "Zero Shutter Lag" (ZSL) state machine:**
    *   Always capture full-res images into a ring buffer.
    *   When shutter pressed, pick the frame *from the past* (Frame N-1) which is already captured.
2.  **Design a state diagram** for a video recording session: Preview -> Record (Locks) -> Stop (Unlocks).

---

## 📚 Further Reading & Resources

### Standards
*   **Android Camera2 API:** Study the State Machine diagrams for AE, AF, AWB (very detailed).

### Code
*   **libcamera:** Source code for `ipa/raspberrypi/controller`.

---

## 🎓 Summary

Today we covered:
- ✅ **3A Coordination:** The Supervisor FSM.
- ✅ **Latency:** The killer of control loops.
- ✅ **Sequences:** Pre-flash and Capture.
- ✅ **States:** Searching, Converged, Locked.
- ✅ **Synchronization:** Preventing conflicts.

**Next:** Day 32 - High Dynamic Range (HDR) Control.

---

**Day 31 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
