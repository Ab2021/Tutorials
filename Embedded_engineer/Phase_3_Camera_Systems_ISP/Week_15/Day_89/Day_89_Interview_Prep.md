# Day 89: Interview Preparation (Technical & System Design)
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎯 Learning Objectives
1.  **Master** the Top 20 Technical Questions for Camera/Embedded roles.
2.  **Practice** "Whiteboard Coding" for Image Processing algorithms (C/C++).
3.  **Solve** System Design problems (e.g., "Design a Dashcam").
4.  **Understand** the "STAR" method for behavioral questions.
5.  **Simulate** a Mock Interview scenario.

---

## 📚 Prerequisites & Preparation
*   **Mindset:** You are the expert. Explain *why*, not just *what*.
*   **Tools:** Pen and Paper (or Whiteboard). No IDE.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: C/C++ & OS Fundamentals
*   **Volatile:** Why is it needed for registers? (Prevents compiler optimization).
*   **Interrupts:** Can you sleep in an ISR? (No). Why? (No process context).
*   **Memory:** Stack vs Heap. What is a Memory Leak? How to find it (Valgrind)?
*   **Concurrency:** Mutex vs Semaphore vs Spinlock. When to use Spinlock? (Short wait, ISR context).

### 🔹 Part 2: Camera Domain Knowledge
*   **MIPI CSI-2:** Explain the Physical Layer (D-PHY) vs Protocol Layer. What is a "Virtual Channel"?
*   **ISP:** Explain the pipeline order. Why Demosaic *after* Denoise? (Usually Denoise is first on RAW to avoid correlating noise).
*   **V4L2:** What is the difference between `mmap` and `userptr`?
*   **SerDes:** How does GMSL handle bidirectional control? (Back channel).

### 🔹 Part 3: System Design
*   **Framework:** Requirements -> Hardware Selection -> Software Architecture -> Data Flow -> Power/Performance.
*   **Trade-offs:** Latency vs Quality. Power vs Performance. Cost vs Features.

---

## 💻 Implementation Examples

### Example 1: Whiteboard Coding - Bayer Demosaic (Nearest Neighbor)

**Question:** Write a C function to convert a Bayer GRBG image to RGB.

```c
// Input: raw (W x H), Output: rgb (W x H x 3)
void demosaic_nn(uint8_t* raw, uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height - 1; y += 2) {
        for (int x = 0; x < width - 1; x += 2) {
            // GRBG Pattern:
            // G R
            // B G
            
            uint8_t G1 = raw[y * width + x];
            uint8_t R  = raw[y * width + (x+1)];
            uint8_t B  = raw[(y+1) * width + x];
            uint8_t G2 = raw[(y+1) * width + (x+1)];
            
            // Pixel (x, y) - Green
            rgb[(y*width + x)*3 + 0] = R;
            rgb[(y*width + x)*3 + 1] = G1;
            rgb[(y*width + x)*3 + 2] = B;
            
            // Pixel (x+1, y) - Red
            rgb[(y*width + x+1)*3 + 0] = R;
            rgb[(y*width + x+1)*3 + 1] = (G1 + G2) / 2;
            rgb[(y*width + x+1)*3 + 2] = B;
            
            // ... (Repeat for B and G2 pixels)
        }
    }
}
```
*   **Interviewer Follow-up:** "This is slow. How to optimize?"
*   **Answer:** SIMD (NEON/SSE). Loop Unrolling.

### Example 2: System Design - Battery Powered Doorbell

**Question:** Design a video doorbell that lasts 6 months on a 5000mAh battery.

**Answer Structure:**
1.  **Requirements:** 6 months = 4300 hours. Avg Current = 5000mAh / 4300h = ~1.1mA.
2.  **Problem:** WiFi uses 200mA. Camera uses 200mA. We can't run 24/7.
3.  **Architecture:**
    *   **PIR Sensor:** Uses 10uA. Wakes up the system.
    *   **Fast Boot:** Linux takes 10s. Too slow. Use RTOS (FreeRTOS) or "Suspend to RAM".
    *   **Pre-Roll:** We need video *before* the button press. Circular Buffer in RAM? No, RAM needs power.
4.  **Solution:**
    *   PIR wakes SoC.
    *   SoC boots in < 1s (RTOS).
    *   Capture snapshot.
    *   Connect WiFi (High Power).
    *   Send Notification.
    *   Sleep.

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "20 Questions" Drill

**Objective:** Rapid fire answers.

**List:**
1.  What is I2C clock stretching?
2.  How does DMA work?
3.  What is a Device Tree?
4.  Explain "Rolling Shutter" effect.
5.  What is "Gamma Correction"?
6.  Difference between H.264 and H.265?
7.  What is a "Ring Buffer"?
8.  How to debug a Kernel Panic?
9.  What is "Priority Inversion"?
10. Explain "Bayer Pattern".

**Task:** Write 1-sentence answers for all. Time yourself (10 mins).

### Lab 2: Code Review Simulation

**Objective:** Find the bug.

**Code:**
```c
void process_image(uint8_t* img) {
    uint8_t* buffer = malloc(1024);
    if (img[0] == 0) return; // BUG: Memory Leak
    memcpy(buffer, img, 1024);
    free(buffer);
}
```
**Task:** Identify the leak. Identify potential buffer overflow (if img is < 1024).

### Lab 3: Mock Interview

**Objective:** Speak out loud.

**Steps:**
1.  Record yourself answering: "Tell me about a challenging bug you faced."
2.  **Use STAR:**
    *   **S:** "I was working on the EVS system..."
    *   **T:** "The video was tearing at 60fps..."
    *   **A:** "I used ftrace to analyze the VSYNC interrupt and found the GPU was holding the lock too long..."
    *   **R:** "I optimized the shader, reduced render time by 2ms, and fixed the tearing."

---

## 🐛 Debugging Interview Nerves

### Debug 1: "I don't know the answer"

**Symptom:** Panic.

**Fix:**
*   Don't lie.
*   Say: "I haven't worked with that specific chip, but based on my experience with X, I assume it works like Y..."
*   Show your *thinking process*.

### Debug 2: "I got stuck on the code"

**Symptom:** Brain freeze.

**Fix:**
*   Start with Brute Force. "The naive solution is O(N^2)..."
*   Then optimize. "We can improve this to O(N) using a Hash Map."

---

## ⚡ Performance Optimization

### Optimization 1: Know the Company

*   If interviewing at **Tesla**: Study Rolling Shutter, HDR, and C++ Optimization.
*   If interviewing at **GoPro**: Study ISP Tuning, Color Science, and Video Codecs.
*   If interviewing at **NVIDIA**: Study CUDA, TensorRT, and Computer Architecture.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we use `const` in C++?** (Safety, and helps compiler optimize).
2.  **What is "Padding" in structures?** (Alignment for CPU access efficiency).
3.  **How does an Image Sensor convert photons to electrons?** (Photoelectric effect in the Photodiode).

### Practical Challenges

1.  **Solve LeetCode:** "Rotate Image" (Matrix rotation) in C++.
2.  **Design a System:** "Design a Baby Monitor that detects crying". (Audio processing + Camera + Cloud).

---

## 📚 Further Reading & Resources

### Books
*   **"Cracking the Coding Interview"** (Classic).
*   **"Making Embedded Systems"** by Elecia White.

---

## 🎓 Summary

Today we covered:
- ✅ **Technical:** C, OS, Camera fundamentals.
- ✅ **Coding:** Whiteboard practice.
- ✅ **Design:** Architecture thinking.
- ✅ **Behavioral:** STAR method.
- ✅ **Strategy:** Know your audience.

**Next:** Day 90 - Phase 3 Final Assessment & Graduation.

---

**Day 89 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
