# Day 201: Firmware Loading (request_firmware)
## Phase 2: Linux Kernel & Device Drivers | Week 30: Device Model & Sysfs

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
1.  **Explain** why firmware is loaded from userspace.
2.  **Use** `request_firmware` to fetch binary blobs.
3.  **Understand** the search paths (`/lib/firmware`).
4.  **Implement** asynchronous loading (`request_firmware_nowait`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 199 (Uevents).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Firmware Problem
*   **Context:** Many devices (WiFi, GPU, FPGA) need proprietary microcode to function.
*   **Issue:** We cannot embed this code in the GPL kernel (licensing + size).
*   **Solution:** Keep it in the filesystem (`/lib/firmware`) and load it on demand.

### 🔹 Part 2: The Mechanism
1.  Driver calls `request_firmware(&fw, "my_blob.bin", dev)`.
2.  Kernel creates a sysfs node `/sys/class/firmware/.../loading` and sends a Uevent.
3.  Userspace (udev/systemd) sees the event.
4.  Userspace finds the file and writes it to the sysfs node.
5.  Kernel wakes up the driver with the data.

---

## 💻 Implementation: Firmware Loader Driver

> **Instruction:** Create a driver that asks for "my_firmware.bin" and prints its size.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/firmware.h>
#include <linux/platform_device.h>

static int my_probe(struct platform_device *pdev) {
    const struct firmware *fw;
    int ret;
    
    pr_info("MyFW: Requesting firmware...\n");
    
    // Synchronous call (Blocks until userspace provides it or timeout)
    ret = request_firmware(&fw, "my_firmware.bin", &pdev->dev);
    if (ret) {
        pr_err("MyFW: Failed to load firmware: %d\n", ret);
        return ret;
    }
    
    pr_info("MyFW: Loaded %zu bytes\n", fw->size);
    
    // Process fw->data here (e.g., write to device)
    // print_hex_dump(KERN_INFO, "FW: ", DUMP_PREFIX_OFFSET, 16, 1, fw->data, 64, true);
    
    release_firmware(fw);
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 201.1 - Providing the Blob

### 1. Lab Objectives
- Create a dummy firmware file.
- Load the driver and verify it finds the file.

### 2. Step-by-Step Guide
1.  **Create Blob:**
    ```bash
    echo "This is my dummy firmware data" | sudo tee /lib/firmware/my_firmware.bin
    ```
2.  **Load Driver:** `insmod my_fw_driver.ko`.
3.  **Check Log:**
    ```
    dmesg | tail
    # Should see: "MyFW: Loaded 30 bytes"
    ```
4.  **Test Failure:**
    *   Delete the file.
    *   Reload driver.
    *   It should hang for 60s (default timeout) then fail.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Asynchronous Loading
- **Goal:** Don't block `probe`.
- **Task:**
    1.  Use `request_firmware_nowait`.
    2.  Provide a callback function.
    3.  The kernel will call your callback when the firmware is ready.
    4.  Essential for drivers that load during boot when the filesystem might not be mounted yet.

### Lab 3: Custom Path
- **Goal:** Load from a non-standard location?
- **Task:**
    *   Actually, you can't easily change the kernel search path.
    *   But you can use `echo -n /path/to/file > /sys/class/firmware/.../data` manually if you catch the uevent and stop the automatic loader.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Direct firmware load failed"
*   **Cause:** File not found in `/lib/firmware`.
*   **Fix:** Check filename and permissions.

#### 2. Driver hangs during boot
*   **Cause:** Using synchronous `request_firmware` in `probe` before rootfs is mounted.
*   **Fix:** Use `request_firmware_nowait` or ensure the driver is a module (loaded after boot).

---

## ⚡ Optimization & Best Practices

### `release_firmware`
*   **Critical:** Always call `release_firmware(fw)` when done. The kernel allocates vmalloc memory for the blob. Leaking it is bad.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `request_firmware` and `request_firmware_direct`?
    *   **A:** `direct` bypasses the Uevent/Userspace fallback and only looks in the filesystem directly. It's faster but less flexible (can't fetch from network).
2.  **Q:** Can I modify the firmware data in memory?
    *   **A:** No, `fw->data` is `const`. If you need to modify it (e.g., endian swap), copy it to a new buffer first.

### Challenge Task
> **Task:** "The FPGA Programmer".
> *   Simulate an FPGA driver.
> *   Load a 1KB "bitstream".
> *   Verify a checksum (CRC32) of the data.
> *   If valid, print "FPGA Configured".

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/firmware/request_firmware.rst](https://www.kernel.org/doc/html/latest/driver-api/firmware/request_firmware.html)

---
