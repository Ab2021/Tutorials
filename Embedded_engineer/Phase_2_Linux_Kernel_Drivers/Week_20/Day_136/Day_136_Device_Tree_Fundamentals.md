# Day 136: Device Tree Fundamentals
## Phase 2: Linux Kernel & Device Drivers | Week 20: Platform Drivers & Device Tree

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
1.  **Explain** why Device Tree replaced manual `platform_device` registration (the "Board File" era).
2.  **Write** a DTS node describing a custom peripheral.
3.  **Compile** DTS to DTB using `dtc`.
4.  **Load** a Device Tree Overlay (DTO) to add hardware at runtime.
5.  **Inspect** the running Device Tree via `/proc/device-tree`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `device-tree-compiler`.
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 135 (Platform Drivers).
    *   Day 125 (DTS Intro).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Board File" Problem
In Day 135, we wrote `pcd_device_setup.c`. This is a "Board File".
*   **Issue:** Every time a hardware engineer changes a pin or adds a sensor, the kernel source code must be recompiled.
*   **Scale:** With thousands of ARM boards, the `arch/arm` directory became a mess.
*   **Solution:** **Device Tree**. A separate binary (`.dtb`) describing the hardware. The kernel remains generic.

### 🔹 Part 2: DTS Syntax Recap
*   **Nodes:** `{ ... }` blocks.
*   **Properties:** `key = value;`.
*   **Cells:** 32-bit integers `<0x1000>`.
*   **Strings:** `"string"`.
*   **Byte Arrays:** `[00 11 22]`.

### 🔹 Part 3: Addressing
*   `#address-cells`: How many 32-bit integers make up an address (usually 1 or 2).
*   `#size-cells`: How many 32-bit integers make up a size.
*   `reg`: The actual Address and Size list.

---

## 💻 Implementation: The DTS Overlay

> **Instruction:** Instead of `pcd_device_setup.c`, we will write `pcd_overlay.dts`.

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`pcd_overlay.dts`)

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            /* Device 1 */
            pcd0: pseudo-char-device@1000 {
                compatible = "org,pseudo-char-device";
                reg = <0x1000 0x20>; /* Start 0x1000, Size 0x20 */
                interrupts = <10>;
                status = "okay";
                label = "pcd-dev-1";
            };

            /* Device 2 */
            pcd1: pseudo-char-device@2000 {
                compatible = "org,pseudo-char-device";
                reg = <0x2000 0x20>;
                interrupts = <11>;
                status = "okay";
                label = "pcd-dev-2";
            };
        };
    };
};
```

#### Step 2: Compile
```bash
dtc -@ -I dts -O dtb -o pcd_overlay.dtbo pcd_overlay.dts
```

#### Step 3: Apply Overlay (ConfigFS)
Ensure ConfigFS is mounted.
```bash
mount -t configfs none /sys/kernel/config
mkdir /sys/kernel/config/device-tree/overlays/pcd
cat pcd_overlay.dtbo > /sys/kernel/config/device-tree/overlays/pcd/dtbo
```

#### Step 4: Verify
Check if the nodes appeared.
```bash
ls /proc/device-tree/pseudo-char-device@1000
# Output: compatible, reg, interrupts, status...
```

---

## 💻 Implementation: Updating the Driver

> **Instruction:** The driver needs to know which `compatible` string to look for.

### 👨‍💻 Code Implementation

#### Step 1: Update `pcd_platform_driver.c`

```c
#include <linux/of.h> // Open Firmware (Device Tree) header
#include <linux/of_device.h>

// ... probe and remove functions (Same as Day 135) ...

// Match Table
static const struct of_device_id pcd_dt_ids[] = {
    { .compatible = "org,pseudo-char-device", },
    { .compatible = "another,compatible-id", },
    { } // Null terminator
};
MODULE_DEVICE_TABLE(of, pcd_dt_ids);

static struct platform_driver pcd_driver = {
    .probe = pcd_probe,
    .remove = pcd_remove,
    .driver = {
        .name = "pseudo-char-device",
        .owner = THIS_MODULE,
        .of_match_table = pcd_dt_ids, // Link the match table
    }
};

// ... init and exit ...
```

#### Step 2: Test
1.  Load `pcd_platform_driver.ko`.
2.  Apply the Overlay.
3.  **Result:** `probe` is called!
    ```text
    PCD Driver: Probe called for device pseudo-char-device.0
    PCD Driver: MEM Start=0x1000, End=0x101F
    ```

---

## 🔬 Lab Exercise: Lab 136.1 - Extracting Custom Properties

### 1. Lab Objectives
- Add a custom property `org,device-serial-num = <12345>;` to the DTS.
- Read it in the driver using `of_property_read_u32`.

### 2. Step-by-Step Guide
1.  **Edit DTS:** Add the property to `pcd0`. Recompile and re-apply.
2.  **Edit Driver (`probe`):**
    ```c
    u32 serial_num = 0;
    if (!of_property_read_u32(pdev->dev.of_node, "org,device-serial-num", &serial_num)) {
        printk("Serial: %d\n", serial_num);
    } else {
        printk("Serial property missing\n");
    }
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Phandles and References
- **Goal:** Reference another node.
- **Task:**
    1.  Create a dummy clock node.
    2.  In `pcd0`, add `clocks = <&my_clock>;`.
    3.  In driver, use `of_parse_phandle` to find the clock node.

### Lab 3: Disabling a Device
- **Goal:** Use `status` property.
- **Task:**
    1.  Set `status = "disabled";` in DTS for `pcd1`.
    2.  Apply overlay.
    3.  Verify `probe` is ONLY called for `pcd0`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "dtc: syntax error"
*   **Cause:** Missing semicolon `;` is the #1 cause.

#### 2. Driver doesn't probe
*   **Cause:** Typo in `compatible` string. It must match character-for-character.
*   **Cause:** Forgot `MODULE_DEVICE_TABLE`.

#### 3. "Overlay failed to apply"
*   **Cause:** The `target-path` in the overlay does not exist in the base tree.

---

## ⚡ Optimization & Best Practices

### Naming Convention
*   **Compatible Strings:** `"vendor,device"`. Example: `"ti,omap4-gpio"`.
*   **Node Names:** `generic-name@address`. Example: `serial@1000`, not `uart@1000`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `platform_get_resource` and parsing DTS manually?
    *   **A:** `platform_get_resource` is the generic API. The kernel core parses the DTS `reg` property and populates the `platform_device` resources for you. You should prefer the generic API.
2.  **Q:** Can I have multiple compatible strings?
    *   **A:** Yes. `compatible = "my,new-ver", "my,old-ver";`. The kernel tries to match from left to right.

### Challenge Task
> **Task:** "The RGB LED".
> *   DTS: Node with 3 GPIO references (`gpios = <&gpio 1 0>, <&gpio 2 0>, <&gpio 3 0>;`).
> *   Driver: Parse the array of GPIOs and control them.

---

## 📚 Further Reading & References
- [Device Tree Usage (eLinux.org)](https://elinux.org/Device_Tree_Usage)
- [Kernel Documentation: devicetree/bindings/](https://www.kernel.org/doc/Documentation/devicetree/bindings/)

---
