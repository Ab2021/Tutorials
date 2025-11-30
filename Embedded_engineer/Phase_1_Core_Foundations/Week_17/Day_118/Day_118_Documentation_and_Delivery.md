# Day 118: Documentation & Delivery
## Phase 1: Core Embedded Engineering Foundations | Week 17: Final Project - The Smart Home Hub

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
1.  **Generate** API documentation automatically using Doxygen.
2.  **Write** a concise User Manual for the Smart Home Hub.
3.  **Manage** Firmware Versions (Semantic Versioning) and Changelogs.
4.  **Conduct** a Code Review using a structured checklist.
5.  **Package** the final release (Binaries + Docs) for delivery.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   None (Documentation Day).
*   **Software Required:**
    *   [Doxygen](https://www.doxygen.nl/index.html)
    *   [Graphviz](https://graphviz.org/) (for call graphs).
*   **Prior Knowledge:**
    *   Completed Project Code (Days 113-117).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Code vs Documentation
*   **Code:** Tells the machine *what* to do.
*   **Comments:** Tell the human *why* it's done that way.
*   **Docs:** Tell the user/developer *how* to use it without reading the code.

### 🔹 Part 2: Doxygen Standard
Standard format for C comments.
*   `/** ... */` block.
*   `@brief`: Short summary.
*   `@param`: Argument description.
*   `@return`: Return value description.
*   `@note`: Important warnings.

### 🔹 Part 3: Semantic Versioning (SemVer)
Format: `MAJOR.MINOR.PATCH` (e.g., 1.2.0).
*   **MAJOR:** Incompatible API changes.
*   **MINOR:** Backwards-compatible functionality.
*   **PATCH:** Backwards-compatible bug fixes.

---

## 💻 Implementation: Doxygen Setup

> **Instruction:** Document the `bsp_wifi.h` file.

### 👨‍💻 Code Implementation

#### Step 1: Annotated Header
```c
/**
 * @file bsp_wifi.h
 * @brief Board Support Package for ESP8266 WiFi Module.
 * @author Your Name
 * @date 2023-10-27
 * @version 1.0.0
 */

#ifndef BSP_WIFI_H
#define BSP_WIFI_H

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Initializes the WiFi module.
 * 
 * Resets the ESP8266 via GPIO and sends AT command to verify communication.
 * 
 * @return true if module responded "OK".
 * @return false if timeout or error.
 * @note This function blocks for up to 1000ms.
 */
bool BSP_WiFi_Init(void);

/**
 * @brief Connects to an Access Point.
 * 
 * @param[in] ssid Null-terminated string of the SSID.
 * @param[in] pass Null-terminated string of the Password.
 * @return true if connected successfully.
 */
bool BSP_WiFi_ConnectAP(const char *ssid, const char *pass);

#endif
```

#### Step 2: Doxyfile Configuration
1.  Run `doxygen -g` to generate `Doxyfile`.
2.  Edit `Doxyfile`:
    *   `PROJECT_NAME = "Smart Home Hub"`
    *   `INPUT = ./Drivers ./App`
    *   `RECURSIVE = YES`
    *   `HAVE_DOT = YES` (If Graphviz installed).
    *   `CALL_GRAPH = YES`
3.  Run `doxygen`. Open `html/index.html`.

---

## 💻 Implementation: Release Management

> **Instruction:** Prepare Release v1.0.0.

### 👨‍💻 Code Implementation

#### Step 1: Version Header (`version.h`)
```c
#ifndef VERSION_H
#define VERSION_H

#define FW_VERSION_MAJOR 1
#define FW_VERSION_MINOR 0
#define FW_VERSION_PATCH 0

#define FW_VERSION_STRING "1.0.0"

#endif
```

#### Step 2: Print Version on Boot
```c
// app_main.c
#include "version.h"

void App_Init(void) {
    printf("Smart Home Hub v%s\n", FW_VERSION_STRING);
    printf("Build Date: %s %s\n", __DATE__, __TIME__);
}
```

#### Step 3: Changelog (`CHANGELOG.md`)
```markdown
# Changelog

## [1.0.0] - 2023-10-27
### Added
- Initial Release.
- WiFi/MQTT support.
- Sensor Logging to SD Card.
- Bluetooth Command Interface.

### Known Issues
- WiFi reconnection takes 10s (Blocking).
```

---

## 🔬 Lab Exercise: Lab 118.1 - The Code Review

### 1. Lab Objectives
- Review your own code (or a peer's) against a checklist.
- Fix violations.

### 2. Step-by-Step Guide

#### Phase A: The Checklist
1.  **Style:** Indentation consistent? Naming convention (`BSP_`, `App_`) followed?
2.  **Safety:** Any infinite loops without timeout? Any buffer overflows (`strcpy` vs `strncpy`)?
3.  **Resources:** Are files closed (`f_close`)? Memory freed?
4.  **Comments:** Are complex logic blocks explained?
5.  **Magic Numbers:** Are `1000`, `0x40` replaced with `#define`?

#### Phase B: Execution
1.  Scan `bsp_wifi.c`.
2.  *Finding:* `HAL_Delay(5000)` inside `ConnectAP`.
3.  *Action:* Add `@note Blocks for 5s` in Doxygen or refactor to FSM.

### 3. Verification
Code should be cleaner and safer.

---

## 🧪 Additional / Advanced Labs

### Lab 2: User Manual
- **Goal:** Write for the End User.
- **Task:** Create `UserManual.pdf`.
    1.  **Installation:** "Connect 5V power..."
    2.  **Setup:** "Edit `config.ini` on SD Card with your WiFi details."
    3.  **Usage:** "LED Green = Connected. LED Red = Error."
    4.  **Troubleshooting:** "If Red LED blinks, check SD Card."

### Lab 3: Binary Patching
- **Goal:** Embed checksum.
- **Task:**
    1.  Write a Python script to calculate CRC32 of `main.bin`.
    2.  Append CRC32 to the end of the binary.
    3.  Bootloader can verify this before booting.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Doxygen Empty
*   **Cause:** `INPUT` path wrong or files don't have `@file` tag.
*   **Solution:** Ensure `INPUT` points to source folders and `RECURSIVE = YES`.

#### 2. Version Mismatch
*   **Cause:** Forgot to update `version.h` before build.
*   **Solution:** Use a build script (Makefile) to auto-generate `version.h` from Git Tag.
    *   `git describe --tags > version.h`

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Self-Documenting Code:** `bool is_wifi_connected` is better than `bool flag; // wifi status`. If code is clear, you need fewer comments.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `@brief` and `@details`?
    *   **A:** `@brief` is a one-liner summary. `@details` (or body text) provides full explanation.
2.  **Q:** Why use `__DATE__` and `__TIME__`?
    *   **A:** Preprocessor macros that insert the compilation timestamp. Useful for verifying if the board is running the latest build.

### Challenge Task
> **Task:** "Automated Release". Write a script that:
> 1.  Updates `version.h`.
> 2.  Builds the project.
> 3.  Generates Doxygen.
> 4.  Zips `main.bin` + `docs/` into `Release_v1.0.zip`.

---

## 📚 Further Reading & References
- [Doxygen Manual](https://www.doxygen.nl/manual/index.html)
- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---
