# Day 40: Touch Screens
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
1.  **Explain** the difference between Resistive (Pressure) and Capacitive (Charge) touch screens.
2.  **Interface** a Resistive Touch Controller (XPT2046/ADS7843) via SPI.
3.  **Calibrate** a touch screen (Coordinate Transformation).
4.  **Implement** a simple "Paint" application on the LCD.
5.  **Debounce** touch inputs to prevent ghost clicks.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   TFT LCD with Resistive Touch (e.g., ILI9341 + XPT2046)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 31 (SPI)
    *   Day 36 (LCD)
*   **Datasheets:**
    *   [XPT2046 Datasheet](https://www.waveshare.com/w/upload/9/95/XPT2046_EN.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Resistive Touch
*   **Construction:** Two flexible sheets coated with resistive material, separated by air gap.
*   **Operation:** Pressing brings sheets in contact.
    *   **Measure X:** Apply voltage to X-plate. Measure voltage on Y-plate (acts as wiper).
    *   **Measure Y:** Apply voltage to Y-plate. Measure voltage on X-plate.
*   **Controller:** Chips like XPT2046 automate this switching and ADC conversion.

### 🔹 Part 2: Calibration
Raw ADC values (0-4095) do not match Screen Pixels (0-320).
*   **Linear Transformation:**
    *   $X_{screen} = A \times X_{raw} + B$
    *   $Y_{screen} = C \times Y_{raw} + D$
*   **3-Point Calibration:** More accurate, handles rotation and skew.

---

## 💻 Implementation: XPT2046 Driver

> **Instruction:** SPI2 (PB13/PB14/PB15) + CS (PC5) + IRQ (PC4).

### 🛠️ Hardware/System Configuration
*   **SCK:** PB13
*   **MISO:** PB14
*   **MOSI:** PB15
*   **CS:** PC5
*   **IRQ:** PC4 (Input Pull-Up)

### 👨‍💻 Code Implementation

#### Step 1: SPI Init (Reuse Day 31 logic for SPI2)
Ensure Baud Rate is low (< 2 MHz) for XPT2046.

#### Step 2: Read Function (`touch.c`)

```c
#include "stm32f4xx.h"

#define CMD_READ_X 0xD0 // 12-bit, Differential
#define CMD_READ_Y 0x90

uint16_t TP_Read(uint8_t cmd) {
    uint16_t val;
    
    TP_CS_LOW();
    SPI2_Transfer(cmd);
    // Read 16 bits (Result is in top 12 bits usually, or shift required)
    uint8_t h = SPI2_Transfer(0);
    uint8_t l = SPI2_Transfer(0);
    TP_CS_HIGH();
    
    val = ((h << 8) | l) >> 3; // XPT2046 returns 12 bits left aligned? Check datasheet.
    // Actually XPT2046: Bit 15=Start, ... Result is bits 14-3.
    // So ((h<<8)|l) >> 3 is correct for 12-bit.
    
    return val;
}

void TP_GetXY(uint16_t *x, uint16_t *y) {
    *x = TP_Read(CMD_READ_X);
    *y = TP_Read(CMD_READ_Y);
}
```

#### Step 3: Calibration Map
```c
// Simple 2-point map (needs manual tuning)
#define X_MIN 200
#define X_MAX 3800
#define Y_MIN 300
#define Y_MAX 3700

void TP_GetPixel(int *px, int *py) {
    uint16_t raw_x, raw_y;
    TP_GetXY(&raw_x, &raw_y);
    
    *px = (raw_x - X_MIN) * 320 / (X_MAX - X_MIN);
    *py = (raw_y - Y_MIN) * 240 / (Y_MAX - Y_MIN);
    
    if (*px < 0) *px = 0;
    if (*px > 319) *px = 319;
    // ... clamp Y ...
}
```

---

## 🔬 Lab Exercise: Lab 40.1 - Paint App

### 1. Lab Objectives
- Draw on the screen using a stylus.

### 2. Step-by-Step Guide

#### Phase A: Logic
1.  Check IRQ pin (Low = Touched).
2.  If Touched, Read X, Y.
3.  Convert to Pixel X, Y.
4.  Draw Pixel on LCD.

#### Phase B: Implementation
```c
int main(void) {
    LCD_Init();
    TP_Init();
    
    while(1) {
        if ((GPIOC->IDR & (1 << 4)) == 0) { // IRQ Low
            int x, y;
            TP_GetPixel(&x, &y);
            LCD_DrawPixel(x, y, RED);
        }
    }
}
```

### 3. Verification
Touching the screen leaves a trail of red pixels.

---

## 🧪 Additional / Advanced Labs

### Lab 2: On-Screen Buttons
- **Goal:** Create a GUI.
- **Task:**
    1.  Draw a rectangle "CLEAR".
    2.  If Touch X,Y is inside rectangle: Clear Screen.

### Lab 3: Median Filtering
- **Goal:** Remove noise.
- **Task:**
    1.  Read X 5 times.
    2.  Sort and pick median.
    3.  Greatly improves drawing smoothness.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. X/Y Swapped or Inverted
*   **Cause:** Touch panel orientation vs LCD orientation.
*   **Solution:** Swap X/Y in software. `x = 320 - x` to invert.

#### 2. Jittery Cursor
*   **Cause:** SPI noise or power supply noise.
*   **Solution:** Lower SPI speed. Add Median Filter.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **IRQ Interrupt:** Instead of polling IRQ pin, use EXTI. Only read SPI when interrupt fires.

### Code Quality
- **Calibration Routine:** Don't hardcode Min/Max. Write a routine that asks user to "Touch Top-Left" and "Touch Bottom-Right", saves values to Flash/EEPROM.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does Resistive touch require pressure?
    *   **A:** To physically deform the top layer to contact the bottom layer. Capacitive detects change in field (finger proximity).
2.  **Q:** Can XPT2046 measure pressure?
    *   **A:** Yes (Z-axis). By measuring resistance across the contact point.

### Challenge Task
> **Task:** Implement "Gesture Detection". Detect a "Swipe Right" (Start X < 50, End X > 250, Time < 500ms).

---

## 📚 Further Reading & References
- [Touch Screen Controller Theory](https://www.ti.com/lit/an/sbaa155a/sbaa155a.pdf)

---
