# Day 20: Real-Time Clock (RTC)
## Phase 1: Core Embedded Engineering Foundations | Week 3: Timers and GPIO

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
1.  **Understand** the Backup Domain architecture and how it survives VDD loss (using VBAT).
2.  **Configure** the LSE (Low Speed External) oscillator for high-precision timekeeping.
3.  **Implement** a Calendar (Date/Time) with BCD (Binary Coded Decimal) format handling.
4.  **Set** an RTC Alarm to wake the system from low-power modes.
5.  **Store** critical data in the Backup Registers (BKP).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   CR2032 Battery (connected to VBAT pin, optional)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 13 (Clock Tree)
    *   Day 4 (Bit Manipulation - BCD)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (RTC Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Backup Domain

The STM32 has a special power domain called the **Backup Domain**.
*   **Components:** RTC, LSE Oscillator, Backup Registers (4KB on F4), Power Management (PWR).
*   **Power Source:**
    *   If $V_{DD}$ is present (> 1.8V), it runs from $V_{DD}$.
    *   If $V_{DD}$ is lost, it switches to $V_{BAT}$ (Coin cell).
*   **Protection:** Write access is disabled by default to prevent accidental corruption during power-up/down.

### 🔹 Part 2: RTC Architecture

#### 2.1 Clock Source (LSE)
The RTC needs a 32.768 kHz crystal (LSE).
*   **Prescalers:**
    *   **Asynchronous (7-bit):** Default 127. Reduces power.
    *   **Synchronous (15-bit):** Default 255.
    *   $f_{CK\_SPRE} = \frac{f_{LSE}}{(PREDIV\_A + 1) \times (PREDIV\_S + 1)} = \frac{32768}{128 \times 256} = 1 Hz$

#### 2.2 BCD Format
Registers `TR` (Time) and `DR` (Date) use BCD.
*   **Example:** 23 Seconds.
    *   Binary: `0001 0111` (0x17)
    *   BCD: `0010 0011` (0x23) -> Nibble 2 is '2', Nibble 3 is '3'.
    *   *Why?* Easier to display on LCDs without division logic.

### 🔹 Part 3: Alarms
The RTC has two Alarms (A and B). They can trigger:
*   **Interrupt:** `RTC_Alarm_IRQn`.
*   **Wakeup:** Can wake the MCU from Stop or Standby mode.

---

## 💻 Implementation: Digital Clock with Backup

> **Instruction:** We will initialize the RTC. If it's already running (from Backup), we won't reset the time.

### 🛠️ Hardware/System Configuration
STM32F4 Discovery.

### 👨‍💻 Code Implementation

#### Step 1: Enable Access (`rtc.c`)

```c
#include "stm32f4xx.h"

void RTC_Init(void) {
    // 1. Enable Power Clock
    RCC->APB1ENR |= (1 << 28); // PWREN

    // 2. Allow Access to Backup Domain
    PWR->CR |= (1 << 8); // DBP (Disable Backup Protection)

    // 3. Check if RTC is already initialized (INITS bit in ISR)
    if (RTC->ISR & (1 << 4)) {
        // Already running, don't reset time!
        return;
    }

    // 4. Enable LSE (32.768 kHz)
    RCC->BDCR |= (1 << 0); // LSEON
    while(!(RCC->BDCR & (1 << 1))); // Wait for LSERDY

    // 5. Select LSE as RTC Clock
    RCC->BDCR &= ~(0x3 << 8);
    RCC->BDCR |= (0x1 << 8); // 01: LSE

    // 6. Enable RTC
    RCC->BDCR |= (1 << 15); // RTCEN

    // 7. Enter Initialization Mode
    RTC->ISR = 0xFFFFFFFF; // Unprotect keys (Key writing sequence needed usually)
    // Actually, write protection is unlocked by writing keys to WPR
    RTC->WPR = 0xCA;
    RTC->WPR = 0x53;

    RTC->ISR |= (1 << 7); // INIT
    while(!(RTC->ISR & (1 << 6))); // Wait for INITF

    // 8. Set Prescalers (Default)
    RTC->PRER = (127 << 16) | 255;

    // 9. Set Time (12:00:00)
    // TR: HT(1) HU(2) : M(0) : S(0)
    RTC->TR = (1 << 20) | (2 << 16); 

    // 10. Exit Initialization Mode
    RTC->ISR &= ~(1 << 7);
    
    // 11. Lock Write Protection
    RTC->WPR = 0xFF;
}
```

#### Step 2: Read Time
```c
typedef struct {
    uint8_t Hours;
    uint8_t Minutes;
    uint8_t Seconds;
} Time_t;

void RTC_GetTime(Time_t *time) {
    uint32_t tr = RTC->TR;
    
    // Decode BCD
    time->Seconds = ((tr & 0x70) >> 4) * 10 + (tr & 0xF);
    time->Minutes = ((tr & 0x7000) >> 12) * 10 + ((tr & 0xF00) >> 8);
    time->Hours   = ((tr & 0x300000) >> 20) * 10 + ((tr & 0xF0000) >> 16);
}
```

#### Step 3: Main Loop
```c
#include <stdio.h>

int main(void) {
    // Init UART...
    RTC_Init();
    
    Time_t now;

    while(1) {
        RTC_GetTime(&now);
        printf("Time: %02d:%02d:%02d\r", now.Hours, now.Minutes, now.Seconds);
        
        for(int i=0; i<1000000; i++);
    }
}
```

---

## 🔬 Lab Exercise: Lab 20.1 - The Alarm Clock

### 1. Lab Objectives
- Set an alarm to trigger 10 seconds from now.
- Blink an LED in the Alarm ISR.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Unlock RTC (`WPR`).
2.  Disable Alarm A (`ALRAE` = 0).
3.  Wait for write access (`ALRAWF`).

#### Phase B: Coding
```c
void Set_Alarm_In_10s(void) {
    Time_t now;
    RTC_GetTime(&now);
    
    uint8_t target_sec = (now.Seconds + 10) % 60;
    
    // Convert to BCD
    uint8_t bcd_sec = (target_sec / 10) << 4 | (target_sec % 10);
    
    // Configure Alarm A
    // Mask Date, Hours, Minutes (Ignore them) -> MSK4, MSK3, MSK2 = 1
    // Match Seconds -> MSK1 = 0
    RTC->ALRMAR = (1<<31) | (1<<30) | (1<<23) | bcd_sec;
    
    // Enable Alarm A
    RTC->CR |= (1 << 8); // ALRAE
    
    // Enable Interrupt
    RTC->CR |= (1 << 12); // ALRAIE
    EXTI->IMR |= (1 << 17); // EXTI Line 17 is RTC Alarm
    EXTI->RTSR |= (1 << 17);
    NVIC_EnableIRQ(RTC_Alarm_IRQn);
}

void RTC_Alarm_IRQHandler(void) {
    if (RTC->ISR & (1 << 8)) { // ALRAF
        RTC->ISR &= ~(1 << 8); // Clear Flag
        EXTI->PR |= (1 << 17); // Clear EXTI
        
        printf("ALARM TRIGGERED!\n");
    }
}
```

### 3. Verification
Run the code. Wait 10 seconds. Observe the output.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Backup Registers
- **Goal:** Count resets.
- **Task:**
    1.  Read `RTC->BKP0R`.
    2.  Increment it.
    3.  Write it back.
    4.  Print "Boot Count: X".
    5.  Press Reset button. The count should increase.
    6.  Power cycle (remove USB). If VBAT is connected, count persists. If not, it resets to 0.

### Lab 3: Tamper Detection
- **Goal:** Detect case opening.
- **Task:**
    1.  Configure Tamper Pin (PC13).
    2.  Enable Tamper Interrupt.
    3.  When PC13 changes state, the Backup Registers are automatically erased (Security feature).
    4.  Verify `BKP0R` becomes 0.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. RTC Not Counting
*   **Cause:** LSE not ready (Crystal failure).
*   **Solution:** Check `LSERDY`. If using Discovery board without crystal (some revisions), use LSI instead.

#### 2. Alarm Not Firing
*   **Cause:** EXTI Line 17 not configured.
*   **Detail:** RTC Alarm is internally connected to EXTI 17. You must unmask it in EXTI controller to get an interrupt.

---

## ⚡ Optimization & Best Practices

### Low Power Design
- **Calibration:** The LSE crystal can drift. Use the `CALR` register to add/subtract clock cycles to compensate for temperature drift (if you have a temperature sensor).

### Code Quality
- **Timeouts:** Always use timeouts when waiting for flags like `INITF` or `LSERDY`. If the hardware fails, the code shouldn't hang forever.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is BCD?
    *   **A:** Binary Coded Decimal. 4 bits per decimal digit. 0x19 = 25 decimal.
2.  **Q:** Why do we need to unlock the RTC registers?
    *   **A:** To prevent runaway code from corrupting the time or calibration data.

### Challenge Task
> **Task:** Implement a "Unix Timestamp" converter. Convert the RTC BCD Date/Time to a 32-bit `time_t` (Seconds since Jan 1, 1970) and vice-versa.

---

## 📚 Further Reading & References
- [STM32 RTC Application Note (AN3371)](https://www.st.com/resource/en/application_note/dm00025071-using-the-hardware-realtime-clock-rtc-in-stm32-f0-f2-f3-f4-and-l1-series-of-mcus-stmicroelectronics.pdf)

---
