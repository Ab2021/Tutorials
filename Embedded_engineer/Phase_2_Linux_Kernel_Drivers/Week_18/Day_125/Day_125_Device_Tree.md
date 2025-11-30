# Day 125: Device Tree (DTS) Fundamentals
## Phase 2: Linux Kernel & Device Drivers | Week 18: Linux Kernel Fundamentals

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
1.  **Read and Write** Device Tree Source (DTS) files.
2.  **Compile** DTS to DTB using the Device Tree Compiler (`dtc`).
3.  **Explain** the relationship between Device Tree nodes and Kernel Drivers (`compatible` string).
4.  **Debug** the running Device Tree via `/proc/device-tree`.
5.  **Create** and **Apply** Device Tree Overlays (DTO) for runtime hardware changes.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `device-tree-compiler` (`sudo apt install device-tree-compiler`).
    *   Linux Kernel Source (for reference DTS files).
*   **Prior Knowledge:**
    *   Basic hardware concepts (Memory addresses, IRQs).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is the Device Tree?
In the x86 world, hardware is discoverable (PCIe enumeration, USB plug-and-play, ACPI).
In the Embedded world (ARM, RISC-V), hardware is often hardwired and non-discoverable. The kernel has no way of knowing "There is a UART at address 0x1000" unless we tell it.
*   **Old Way:** Hardcoded C files (`arch/arm/mach-xxx/board-yyy.c`). This led to "Linus's Rant" and code bloat.
*   **New Way:** **Device Tree**. A data structure describing the hardware topology. Passed to the kernel at boot time.

### 🔹 Part 2: DTS Syntax
It looks like JSON but with C-style comments.
```dts
/dts-v1/;

/ {
    model = "My Custom Board";
    compatible = "myvendor,myboard", "arm,cortex-a53";

    cpus {
        cpu@0 {
            compatible = "arm,cortex-a53";
            device_type = "cpu";
            reg = <0>;
        };
    };

    memory@80000000 {
        device_type = "memory";
        reg = <0x80000000 0x20000000>; /* 512MB at 2GB offset */
    };

    uart0: serial@10000000 {
        compatible = "arm,pl011", "arm,primecell";
        reg = <0x10000000 0x1000>;
        interrupts = <0 5 4>;
    };
};
```
*   **Nodes:** `serial@10000000` (Name + Address).
*   **Properties:** Key-value pairs (`compatible`, `reg`).
*   **Labels:** `uart0:` (Used to reference this node elsewhere).
*   **Phandles:** Pointers to other nodes (e.g., `clocks = <&clk_ref>`).

### 🔹 Part 3: The `compatible` String
This is the magic glue.
1.  **DTS:** `compatible = "arm,pl011";`
2.  **Driver C Code:**
    ```c
    static const struct of_device_id pl011_ids[] = {
        { .compatible = "arm,pl011", },
        { }
    };
    MODULE_DEVICE_TABLE(of, pl011_ids);
    ```
3.  **Kernel Core:** During boot, it walks the DT. When it sees a node, it looks for a loaded driver with a matching `compatible` string. If found, it calls the driver's `probe()` function.

---

## 💻 Implementation: Compiling and Decompiling

> **Instruction:** Practice using the `dtc` tool.

### 👨‍💻 Command Line Steps

#### Step 1: Create a Simple DTS
Save the example above as `test.dts`.

#### Step 2: Compile to DTB (Binary)
```bash
dtc -I dts -O dtb -o test.dtb test.dts
```
*   `-I`: Input format.
*   `-O`: Output format.
*   The `.dtb` file is what the bootloader loads.

#### Step 3: Decompile DTB to DTS
Useful for reverse engineering a binary blob.
```bash
dtc -I dtb -O dts -o dump.dts test.dtb
```
Compare `test.dts` and `dump.dts`. They should be semantically identical.

---

## 💻 Implementation: Exploring the Live Tree

> **Instruction:** If you have a running Linux system (even your host PC or the QEMU VM), the kernel exposes the tree.

### 👨‍💻 Command Line Steps

```bash
cd /proc/device-tree
ls -F
```
You will see directories representing nodes and files representing properties.
```bash
cat model
# Output: linux,dummy-virt (or similar for QEMU)

cd cpus/cpu@0
cat compatible
# Output: arm,cortex-a57
```
**Key Insight:** This is the *live* view. If you apply an overlay, this directory updates.

---

## 🔬 Lab Exercise: Lab 125.1 - Adding a Custom LED Node

### 1. Lab Objectives
- Modify the QEMU DTS to add a "virtual" LED.
- Verify the kernel sees it.

### 2. Step-by-Step Guide

#### Phase A: Extract QEMU DTS
QEMU generates its DTB on the fly. We need to dump it.
```bash
qemu-system-aarch64 -M virt -machine dumpdtb=qemu.dtb
dtc -I dtb -O dts -o qemu.dts qemu.dtb
```

#### Phase B: Edit DTS
Open `qemu.dts`. Find the root node `/ { ... };`. Add this:
```dts
    my_leds {
        compatible = "gpio-leds";
        led0 {
            label = "status_led";
            gpios = <&pl061 5 0>; /* Assuming pl061 exists, check phandle! */
            default-state = "on";
        };
    };
```
*Note: You need to find the correct phandle for the GPIO controller in the dumped file. It might be `phandle = <0x8001>`. Use that.*

#### Phase C: Compile and Boot
1.  `dtc -I dts -O dtb -o custom.dtb qemu.dts`
2.  Run QEMU with your custom DTB:
    ```bash
    qemu-system-aarch64 \
        -M virt \
        -dtb custom.dtb \
        -kernel Image \
        ...
    ```

#### Phase D: Verify
Inside the VM:
```bash
ls /sys/class/leds/
# Should see "status_led"
```

---

## 💻 Implementation: Device Tree Overlays (DTO)

> **Instruction:** Modifying the main DTS requires a reboot. Overlays allow runtime changes (like plugging in a Raspberry Pi HAT).

### 👨‍💻 Code Implementation

#### Step 1: Create Overlay Source (`overlay.dts`)
```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            my_overlay_node {
                compatible = "test,overlay";
                status = "okay";
            };
        };
    };
};
```

#### Step 2: Compile Overlay
```bash
dtc -@ -I dts -O dtb -o overlay.dtbo overlay.dts
```
*   `-@`: Enable symbols (required for overlays).

#### Step 3: Apply (ConfigFS)
On the target (requires `CONFIG_OF_OVERLAY` and `CONFIG_CONFIGFS_FS`):
```bash
mount -t configfs none /sys/kernel/config
mkdir /sys/kernel/config/device-tree/overlays/my_test
cat overlay.dtbo > /sys/kernel/config/device-tree/overlays/my_test/dtbo
```
Check `dmesg`. The kernel should report the new node.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Pin Muxing (Pinctrl)
- **Goal:** Understand how pins are assigned to functions.
- **Task:**
    1.  Locate a `pinctrl` node in a real DTS (e.g., Raspberry Pi).
    2.  Trace how a UART node references it: `pinctrl-0 = <&uart0_pins>;`.
    3.  Explain what happens if two devices claim the same pins (Conflict!).

### Lab 3: Disabling Hardware
- **Goal:** Disable a device via DTS.
- **Task:**
    1.  Create an overlay that targets an existing node (e.g., a serial port).
    2.  Set `status = "disabled";`.
    3.  Apply and verify the device disappears from `/dev`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Error: syntax error"
*   **Cause:** Missing semicolon `;` or mismatched braces `{}`.
*   **Tip:** `dtc` gives line numbers. Check carefully.

#### 2. "phandle reference not found"
*   **Cause:** You referenced `&gpio0` but `gpio0` label doesn't exist in the scope.
*   **Solution:** Ensure the label is defined or use the full path.

#### 3. Driver doesn't probe
*   **Cause:** Typo in `compatible` string.
*   **Cause:** Driver not compiled into kernel (check `.config`).
*   **Cause:** `status` property is not `"okay"`.

---

## ⚡ Optimization & Best Practices

### DTS Organization
- **dtsi files:** Include files. Put common SoC definitions in `soc.dtsi` and board-specifics in `board.dts`.
- **Labels:** Use labels heavily to override nodes without rewriting the whole hierarchy.
    ```dts
    /* In board.dts */
    &uart0 {
        status = "okay";
    };
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the purpose of the `reg` property?
    *   **A:** It defines the memory-mapped address and size of the device's registers.
2.  **Q:** Can I pass parameters to the kernel via DTS?
    *   **A:** Yes, via the `/chosen` node (e.g., `bootargs`).

### Challenge Task
> **Task:** "The I2C Sensor". Write a DTS node for a hypothetical I2C temperature sensor at address `0x48` on `i2c1`. It should use the `ti,tmp102` compatible string and have an interrupt line connected to GPIO bank 3, pin 14, active low.

---

## 📚 Further Reading & References
- [Device Tree Specification](https://www.devicetree.org/specifications/)
- [Kernel Documentation: Documentation/devicetree](https://www.kernel.org/doc/Documentation/devicetree/)

---
