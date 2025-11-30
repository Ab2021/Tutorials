# Day 69: OTA Updates with RTOS
## Phase 1: Core Embedded Engineering Foundations | Week 10: Advanced RTOS & IoT

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
1.  **Design** a Dual-Bank or Slot-based Flash partition scheme for OTA.
2.  **Implement** an HTTP Client to download a large binary file in chunks.
3.  **Write** the downloaded chunks to the "Next Slot" in Internal Flash.
4.  **Verify** the firmware integrity (CRC/SHA) before rebooting.
5.  **Trigger** the update process remotely via an MQTT command.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Python HTTP Server (`python -m http.server`).
*   **Prior Knowledge:**
    *   Day 53 (Bootloader)
    *   Day 65 (Sockets)
*   **Datasheets:**
    *   [STM32F4 Flash Programming Manual](https://www.st.com/resource/en/programming_manual/pm0081-stm32f40xxx-and-stm32f41xxx-flash-memory-programming-stmicroelectronics.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: OTA Architecture
*   **Single Bank (Ping-Pong):**
    *   **Slot A:** Active App.
    *   **Slot B:** Download Area.
    *   **Bootloader:** Decides whether to run A or B. Or copies B to A (safer but slower).
*   **Process:**
    1.  **Notification:** "New FW available at URL X".
    2.  **Download:** Fetch X chunk by chunk.
    3.  **Storage:** Write to Slot B.
    4.  **Verification:** Check CRC.
    5.  **Activation:** Set a flag (in EEPROM/Backup Register). Reboot.
    6.  **Bootloader:** Sees flag. Swaps/Jumps.

### 🔹 Part 2: HTTP Range Requests
To download a 100KB file with 1KB RAM buffer:
*   `GET /app.bin HTTP/1.1`
*   `Range: bytes=0-1023`
*   Server replies with first 1KB.
*   Repeat for `1024-2047`, etc.
*   **Or:** Just use a standard TCP stream and process data as it arrives (Streaming).

---

## 💻 Implementation: OTA Agent

> **Instruction:** Download `app.bin` from `192.168.1.5:8000` and write to `0x0808 0000` (Sector 8, Slot B).

### 👨‍💻 Code Implementation

#### Step 1: Flash Writer Helper
```c
#define SLOT_B_ADDR 0x08080000

void OTA_Write_Chunk(uint32_t offset, uint8_t *data, uint16_t len) {
    Flash_Unlock();
    
    // If start of sector, erase
    if (offset == 0) {
        Flash_EraseSector(8); // Sector 8 (128KB)
        // Note: If App > 128KB, need to erase Sector 9, 10...
    }
    
    uint32_t addr = SLOT_B_ADDR + offset;
    for (int i=0; i<len; i+=4) {
        Flash_WriteWord(addr + i, *(uint32_t*)(data + i));
    }
    
    Flash_Lock();
}
```

#### Step 2: HTTP Downloader Task
```c
void vTaskOTA(void *p) {
    char *url = (char*)p;
    int sock;
    struct sockaddr_in server_addr;
    char buf[1024];
    uint32_t total_received = 0;
    
    // 1. Connect
    sock = socket(AF_INET, SOCK_STREAM, 0);
    // ... Connect logic ...
    
    // 2. Send GET
    sprintf(buf, "GET %s HTTP/1.1\r\nHost: 192.168.1.5\r\n\r\n", url);
    send(sock, buf, strlen(buf), 0);
    
    // 3. Skip Header
    // Read until \r\n\r\n
    // (Simplified logic here)
    
    // 4. Receive Body
    int len;
    while ((len = recv(sock, buf, sizeof(buf), 0)) > 0) {
        OTA_Write_Chunk(total_received, (uint8_t*)buf, len);
        total_received += len;
        printf("Downloaded: %lu\n", total_received);
    }
    
    close(sock);
    
    // 5. Verify & Reboot
    if (Verify_CRC(SLOT_B_ADDR, total_received)) {
        Set_Boot_Flag(BOOT_FROM_SLOT_B);
        NVIC_SystemReset();
    }
}
```

---

## 🔬 Lab Exercise: Lab 69.1 - The Update

### 1. Lab Objectives
- Host a firmware file.
- Trigger OTA.
- Observe Flash write.

### 2. Step-by-Step Guide

#### Phase A: Prepare Firmware
1.  Compile a "Blinky Fast" app.
2.  Rename `app.bin` to `update.bin`.
3.  Start Python Server: `python -m http.server 8000`.

#### Phase B: Trigger
1.  Send MQTT message: `stm32/cmd` -> `{"ota": "/update.bin"}`.
2.  STM32 parses JSON. Starts `vTaskOTA`.
3.  **Observation:** Terminal shows "Downloaded: 1024", "Downloaded: 2048"...
4.  **Observation:** System Resets.
5.  **Observation:** Bootloader checks flag. Copies Slot B to Slot A (or jumps to B).
6.  **Observation:** LED blinks fast.

### 3. Verification
Check Flash content using CubeProgrammer. Sector 8 should contain the new binary.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Resume Capability
- **Goal:** Handle network interruption.
- **Task:**
    1.  Save `total_received` to Backup Register.
    2.  If connection lost, reconnect.
    3.  Send `Range: bytes=total_received-` header.
    4.  Append to Flash.

### Lab 3: Rollback
- **Goal:** Safety.
- **Task:**
    1.  After update, set "Trial State".
    2.  If system crashes (Watchdog), Bootloader sees "Trial Failed".
    3.  Bootloader reverts to Old Slot.
    4.  If system runs for 5 mins, App marks "Update Confirmed".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flash Write Error
*   **Cause:** Interrupts firing during Flash Write? (Usually fine on F4, but stalls CPU).
*   **Cause:** Voltage drop.
*   **Solution:** Disable interrupts during write if timing is critical, or ensure Vcc is stable.

#### 2. HTTP Header Parsing
*   **Issue:** `recv` might return Header + Body in one chunk.
*   **Solution:** Robust parser to find `\r\n\r\n` and start writing from there.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Security:** Always use HTTPS for OTA. Verify the signature of the binary (RSA/ECDSA) to prevent malicious firmware injection.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not write directly to Slot A (Active App)?
    *   **A:** If power fails during write, you have no app. Device is bricked (unless Bootloader can recover via USB/UART).
2.  **Q:** How does the Bootloader know where to jump?
    *   **A:** It reads a configuration sector (or last page of Flash) or checks the validity (CRC) of Slot A vs Slot B.

### Challenge Task
> **Task:** Implement "Delta OTA". Use a library (like `bsdiff`) to generate a patch between v1 and v2. Download only the patch (small). Reconstruct v2 on the device. (Advanced!).

---

## 📚 Further Reading & References
- [FreeRTOS OTA Agent](https://www.freertos.org/ota/index.html)

---
