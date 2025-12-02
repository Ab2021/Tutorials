# Day 69: ASIL Requirements & Decomposition
## Phase 3: Camera Systems & ISP | Week 12: Advanced Automotive Topics

---

## 🎯 Learning Objectives
1.  **Understand** ASIL Decomposition: Breaking a high ASIL requirement into redundant lower ASIL components (e.g., D = B + B).
2.  **Analyze** Hardware Metrics: SPFM (Single Point Fault Metric) and LFM (Latent Fault Metric).
3.  **Implement** "Freedom from Interference" (FFI) using Memory Protection Units (MPU) and Virtualization.
4.  **Design** a Dual-Path Camera System for ASIL D compliance (e.g., Autonomous Emergency Braking).
5.  **Trace** Requirements from Safety Goal to Software Unit.

---

## 📚 Prerequisites & Preparation
*   **Knowledge:** ISO 26262 Basics (Day 68), Probability Theory.
*   **Context:** Designing a Front Camera for ADAS (AEB/LKA).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ASIL Decomposition
*   Achieving ASIL D on a single component is expensive and difficult.
*   **Algebra:** `ASIL D = ASIL B(D) + ASIL B(D)` or `ASIL D = ASIL C(D) + ASIL A(D)`.
*   **Example:** Instead of one super-safe microcontroller (ASIL D), use two medium-safe microcontrollers (ASIL B) checking each other. If they disagree -> Safe State.
*   **Condition:** The two paths must be **Independent** (Common Cause Failure analysis required).

### 🔹 Part 2: Hardware Metrics (The Numbers)
*   **SPFM (Single Point Fault Metric):** Percentage of faults that are detected/covered.
    *   ASIL B: > 90%
    *   ASIL D: > 99%
*   **LFM (Latent Fault Metric):** Percentage of "sleeping" faults (faults that don't cause error immediately but break the safety mechanism) detected.
    *   ASIL B: > 60%
    *   ASIL D: > 90%
*   **PMHF (Probabilistic Metric for Hardware Failures):**
    *   ASIL D: < 10 FIT.

### 🔹 Part 3: Freedom from Interference (FFI)
*   If QM software (Infotainment) runs on the same chip as ASIL B software (Cluster), the QM code must NOT crash or corrupt the ASIL code.
*   **Mechanisms:**
    *   **Spatial:** MPU/MMU (Memory Protection).
    *   **Temporal:** Watchdog / Time Partitioning (Scheduler).
    *   **Communication:** E2E Protection (CRC/Sequence ID).

---

## 💻 Implementation Examples

### Example 1: ASIL Decomposition (Dual Path)

**Safety Goal:** "Vehicle shall brake if pedestrian detected." (ASIL D).

**Architecture:**
1.  **Path A (Main):** AI Model (YOLO) on GPU detects pedestrian. (ASIL B).
2.  **Path B (Checker):** Optical Flow / Radar on DSP detects obstacle. (ASIL B).
3.  **Voter:** If (A says BRAKE) OR (B says BRAKE) -> BRAKE.
    *   *Note:* For "Unintended Braking" (don't brake for ghost), logic is AND. For "Missed Braking" (don't kill pedestrian), logic is OR.

### Example 2: Memory Protection (FFI)

Configuring an MPU (Memory Protection Unit) on an RTOS.

```c
// Define Regions
#define REGION_SAFETY  0x20000000 // Safety Critical Data
#define REGION_QM      0x20001000 // Non-Critical Data

void configure_mpu() {
    // Region 0: Safety (Read/Write for Safety Task, No Access for QM Task)
    MPU->RBAR = REGION_SAFETY | VALID | REGION_NUMBER_0;
    MPU->RASR = ENABLE | SIZE_4KB | AP_PRIV_RW_USER_NO; // Privileged access only
    
    // Region 1: QM (Read/Write for Everyone)
    MPU->RBAR = REGION_QM | VALID | REGION_NUMBER_1;
    MPU->RASR = ENABLE | SIZE_4KB | AP_RW_USER_RW;
    
    MPU->CTRL = ENABLE | PRIVDEFENA;
}
```

### Example 3: Program Flow Monitoring (Logical Watchdog)

Ensuring code executes in the correct order (A -> B -> C).

```c
int checkpoint = 0;

void task_step_A() {
    do_work_A();
    checkpoint = 1;
}

void task_step_B() {
    if (checkpoint != 1) trigger_fault("SEQUENCE_ERROR_B");
    do_work_B();
    checkpoint = 2;
}

void task_step_C() {
    if (checkpoint != 2) trigger_fault("SEQUENCE_ERROR_C");
    do_work_C();
    checkpoint = 3;
}

void watchdog_check() {
    if (checkpoint != 3) trigger_fault("INCOMPLETE_SEQUENCE");
    checkpoint = 0; // Reset
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Designing a Decomposition

**Objective:** Architecture Design.

**Scenario:** Lane Keeping Assist (LKA).
**Risk:** Unintended Steering (Steering into traffic). ASIL D.
**Task:** Decompose into:
1.  **Camera Path (ASIL B):** Detects Lane Lines.
2.  **Steering Controller (ASIL B):** Calculates Torque, limits max torque (Sanity Check).
3.  **Result:** If Camera requests 10Nm torque (insane), Controller limits to 3Nm.

### Lab 2: Fault Injection (Memory Corruption)

**Objective:** Test FFI.

**Steps:**
1.  Create two threads: `SafetyThread` and `QMThread`.
2.  `SafetyThread` holds a critical variable `brake_status`.
3.  `QMThread` tries to write to `&brake_status` (via a pointer bug).
4.  **Result:**
    *   Without MPU: `brake_status` corrupted. Disaster.
    *   With MPU: CPU triggers `MemManage_Fault`. System resets to Safe State.

### Lab 3: Timing Analysis

**Objective:** Temporal FFI.

**Steps:**
1.  Run a high-priority Safety Task (10ms period).
2.  Run a low-priority QM Task (Infinite loop bug).
3.  **Result:** Scheduler preempts QM Task. Safety Task runs on time.
4.  **Failure:** If interrupts are disabled by QM Task -> FFI Violation.

---

## 🐛 Debugging Safety Issues

### Debug 1: "Bus Fault" / "Hard Fault"

**Symptom:** System crashes immediately.

**Cause:**
*   MPU violation. Code tried to access protected memory.
*   **Fix:** Analyze the Fault Status Register (CFSR) to find the instruction address.

### Debug 2: Decomposition Failure

**Symptom:** Both paths fail simultaneously.

**Cause:**
*   **Common Cause Failure:** Both paths use the *same* reference voltage, or the *same* clock, or the *same* library code.
*   **Fix:** Ensure diversity. Use different clocks, different algorithms, different developers.

---

## ⚡ Performance Optimization

### Optimization 1: Lock-Step Cores

*   Instead of software decomposition (slow), use Hardware Lock-Step (e.g., Cortex-R5F Dual Core).
*   Both cores run the same code cycle-by-cycle. Hardware compares outputs.
*   **Benefit:** ASIL D coverage with zero software overhead.

### Optimization 2: ECC Memory

*   Error Correcting Code (ECC) RAM corrects 1-bit flips and detects 2-bit errors.
*   Essential for ASIL B/D.
*   **Performance:** Slight latency penalty on write, but enables high SPFM.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Freedom from Interference" (FFI)?**
2.  **Why is "Diversity" important in decomposition?** (To avoid Common Cause Failures).
3.  **What is the difference between a "Watchdog" and a "Program Flow Monitor"?** (Watchdog checks time; Flow Monitor checks logic/sequence).
4.  **Can I use a Linux Kernel for ASIL D?** (No. Linux is QM. You need a Safety Hypervisor or Safety Island).

### Practical Challenges

1.  **Implement a "Safe Integer" Class:** A C++ class that stores the value and its inverse (`val` and `~val`). On every read, check `val == ~inv`.
2.  **Review Code:** Look at a C function. Identify potential "Division by Zero" or "Array Out of Bounds" risks.

---

## 📚 Further Reading & Resources

### Standards
*   **AUTOSAR Safety:** Standard software architecture for automotive.

---

## 🎓 Summary

Today we covered:
- ✅ **Decomposition:** D = B + B.
- ✅ **Metrics:** SPFM, LFM, PMHF.
- ✅ **FFI:** MPU, Partitioning.
- ✅ **Diversity:** Avoiding common causes.
- ✅ **Lock-Step:** Hardware redundancy.

**Next:** Day 70 - Camera Security (Authentication & Encryption).

---

**Day 69 Complete** | Phase 3: Camera Systems & ISP | Week 12: Advanced Automotive Topics
