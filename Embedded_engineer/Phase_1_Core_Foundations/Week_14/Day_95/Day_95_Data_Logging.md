# Day 95: Data Logging Strategies
## Phase 1: Core Embedded Engineering Foundations | Week 14: File Systems and Storage

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
1.  **Compare** CSV vs Binary logging formats in terms of size, speed, and readability.
2.  **Implement** a Double Buffering scheme to decouple high-speed sampling from slow SD writes.
3.  **Integrate** RTC timestamps into log entries.
4.  **Handle** "Log Rotation" (creating new files when size limit reached).
5.  **Design** a power-safe logging mechanism that minimizes data loss on power cut.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   SD Card.
    *   RTC (Internal).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   FatFs (Day 93).
*   **Prior Knowledge:**
    *   Day 20 (RTC)
    *   Day 75 (Double Buffering)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Latency Problem
SD Cards are fast on average but have terrible worst-case latency.
*   **Average Write:** 500KB/s.
*   **Worst Case:** 200ms pause (internal garbage collection / wear leveling).
*   **Impact:** If you sample at 1kHz and block for 200ms, you lose 200 samples.
*   **Solution:** Buffer RAM.

### 🔹 Part 2: CSV vs Binary
*   **CSV (Comma Separated Values):**
    *   Format: `2023-10-27,12:00:01,23.5,1013`
    *   Pros: Human readable, Excel compatible.
    *   Cons: Slow (`sprintf`), large size (ASCII overhead).
*   **Binary:**
    *   Format: `[Timestamp:4][Temp:4][Press:4]`
    *   Pros: Fast (memcpy), compact.
    *   Cons: Needs a parser tool on PC.

### 🔹 Part 3: Log Rotation
One giant 2GB file is hard to open.
*   **Strategy:** `log_001.csv`, `log_002.csv`...
*   **Trigger:** File size > 10MB or Time > 24h.

---

## 💻 Implementation: The Logger Task

> **Instruction:** Implement a high-performance logger using FreeRTOS.

### 👨‍💻 Code Implementation

#### Step 1: Data Structure
```c
typedef struct {
    uint32_t timestamp;
    float    temperature;
    float    pressure;
    float    accel[3];
} LogEntry_t;

#define BUF_SIZE 512 // Entries, not bytes
LogEntry_t logBuffer[BUF_SIZE];
volatile int head = 0;
volatile int tail = 0;
```

#### Step 2: Sampling (ISR or High Prio Task)
```c
void Sample_Sensors(void) {
    LogEntry_t entry;
    entry.timestamp = RTC_GetTime();
    entry.temperature = BSP_Temp_Read();
    // ... read others ...
    
    int next = (head + 1) % BUF_SIZE;
    if (next != tail) {
        logBuffer[head] = entry;
        head = next;
        
        // Notify Logger Task if half full
        if ((head % (BUF_SIZE/2)) == 0) {
            xTaskNotifyGive(hLoggerTask);
        }
    } else {
        // Overflow! LED Error.
    }
}
```

#### Step 3: Logger Task (Low Prio)
```c
void vTaskLogger(void *p) {
    FIL fil;
    f_open(&fil, "data.bin", FA_WRITE | FA_OPEN_APPEND);
    
    while(1) {
        // Wait for notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // Write all available data
        while(tail != head) {
            // Calculate contiguous block
            int count;
            if (head > tail) count = head - tail;
            else count = BUF_SIZE - tail;
            
            UINT bw;
            f_write(&fil, &logBuffer[tail], count * sizeof(LogEntry_t), &bw);
            
            // Sync to disk periodically (e.g., every 100 writes)
            f_sync(&fil);
            
            tail = (tail + count) % BUF_SIZE;
        }
    }
}
```

---

## 💻 Implementation: CSV Formatter

> **Instruction:** If CSV is required, format it in the Logger Task.

### 👨‍💻 Code Implementation

#### Step 1: CSV Write
```c
void Write_CSV(FIL *fp, LogEntry_t *entry) {
    char line[64];
    int len = snprintf(line, sizeof(line), "%lu,%.2f,%.2f\n",
                       entry->timestamp,
                       entry->temperature,
                       entry->pressure);
    
    UINT bw;
    f_write(fp, line, len, &bw);
}
```
**Note:** `snprintf` is slow. Do not do this in the ISR!

---

## 🔬 Lab Exercise: Lab 95.1 - Latency Test

### 1. Lab Objectives
- Simulate SD card latency.
- Verify buffer handles it.

### 2. Step-by-Step Guide

#### Phase A: Simulation
In `diskio.c`, add a delay:
```c
DRESULT disk_write(...) {
    if ((rand() % 100) == 0) {
        HAL_Delay(200); // 1% chance of 200ms delay
    }
    // ... write ...
}
```

#### Phase B: Run
1.  Sample at 100Hz (10ms period).
2.  Run for 1 minute.
3.  **Observation:** `head` moves away from `tail` during the delay, but catches up later. No data loss.

### 3. Verification
If `head == tail` (Overflow) occurs, increase `BUF_SIZE`. 200ms @ 100Hz = 20 entries. Buffer should be > 50.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Log Rotation
- **Goal:** Split files.
- **Task:**
    1.  Check `f_size(&fil)`.
    2.  If > 1MB:
        *   `f_close(&fil)`.
        *   Increment index (`log_002.bin`).
        *   `f_open` new file.

### Lab 3: Power Loss Protection
- **Goal:** Minimize corruption.
- **Task:**
    1.  Call `f_sync()` every 1 second.
    2.  Or, monitor PVD (Programmable Voltage Detector). If VDD drops, call `f_close()` immediately (using backup cap energy).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. File size is 0
*   **Cause:** Forgot `f_close` or `f_sync` before power off. FAT table wasn't updated.
*   **Solution:** Use `f_sync` regularly.

#### 2. Slow Performance
*   **Cause:** Writing 1 byte at a time.
*   **Solution:** Buffer 4KB in RAM, then write one big chunk. SD cards hate small writes.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Pre-allocation:** Use `f_expand` (FatFs) to pre-allocate a large file (e.g., 100MB). This prevents fragmentation and avoids FAT table updates during writing (faster).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Binary logging faster?
    *   **A:** No CPU time spent converting float to string (`ftoa`). Smaller data size means less SPI traffic.
2.  **Q:** What is "Wear Leveling"?
    *   **A:** The SD controller moves data around so that one sector isn't written too often (Flash wears out). This causes the random latency.

### Challenge Task
> **Task:** Implement "Ring Log". If disk full, delete the oldest file (`log_000.bin`) to make room for the new one (`log_100.bin`). Dashcam style.

---

## 📚 Further Reading & References
- [High Speed Data Logging with FatFs](http://elm-chan.org/fsw/ff/doc/appnote.html#fs1)

---
