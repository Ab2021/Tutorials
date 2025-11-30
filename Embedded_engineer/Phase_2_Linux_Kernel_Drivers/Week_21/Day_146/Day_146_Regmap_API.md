# Day 146: The Regmap API - Abstraction Layer
## Phase 2: Linux Kernel & Device Drivers | Week 21: I2C and SPI Drivers

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
1.  **Explain** why `regmap` is the standard for I2C/SPI drivers (Abstraction, Caching, Debugging).
2.  **Configure** `regmap_config` for various device types (8/16/32-bit registers).
3.  **Implement** Regmap Caching (`REGCACHE_RBTREE`) to reduce bus traffic.
4.  **Handle** Regmap IRQs (`regmap_add_irq_chip`) for devices with interrupt controllers.
5.  **Debug** register access using `/sys/kernel/debug/regmap`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 142 (I2C) & Day 144 (SPI).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem Regmap Solves
Before Regmap, drivers were full of:
*   `if (is_spi) spi_write(...) else i2c_smbus_write(...)`
*   Manual caching logic.
*   Manual endianness swapping.
*   Manual debug printks.

**Regmap** unifies this. You initialize it with the bus (I2C/SPI/MMIO), and then use `regmap_read` / `regmap_write`. The core handles the rest.

### 🔹 Part 2: Caching Strategies
*   **REGCACHE_NONE:** No caching. Always go to hardware.
*   **REGCACHE_RBTREE:** Red-Black Tree. Good for sparse maps (registers at 0x00, 0x100, 0x5000).
*   **REGCACHE_FLAT:** Array. Good for dense maps (0x00 to 0x10). Fast but wastes memory if sparse.

### 🔹 Part 3: Regmap IRQ
Many PMICs and Audio Codecs have their own Interrupt Controller (Status Reg + Mask Reg).
*   Regmap provides a helper to handle this.
*   It reads the Status Reg, checks the Mask, and calls the appropriate nested IRQ handler.

---

## 💻 Implementation: Advanced Configuration

> **Instruction:** Configuring Regmap for a complex device (16-bit address, 32-bit data).

### 👨‍💻 Code Implementation

```c
static const struct regmap_config my_config = {
    .reg_bits = 16,  // 16-bit Register Address
    .val_bits = 32,  // 32-bit Register Value
    .pad_bits = 0,
    
    .max_register = 0xFFFF,
    .cache_type = REGCACHE_RBTREE,
    
    // Endianness
    .val_format_endian = REGMAP_ENDIAN_BIG, // Device sends MSB first
    
    // Ranges
    .volatile_reg = is_volatile_reg, // Callback: Don't cache these (e.g., Status)
    .readable_reg = is_readable_reg, // Callback: Is this readable?
};

static bool is_volatile_reg(struct device *dev, unsigned int reg) {
    switch (reg) {
        case REG_STATUS:
        case REG_INT_SRC:
            return true; // Always read from HW
        default:
            return false; // Use Cache
    }
}
```

---

## 💻 Implementation: Regmap IRQ Chip

> **Instruction:** Handling a device that has an interrupt pin and internal status registers.

### 👨‍💻 Code Implementation

#### Step 1: Define the IRQ Chip
```c
static const struct regmap_irq my_irqs[] = {
    [0] = { .reg_offset = 0, .mask = BIT(0) }, // IRQ 0: Bit 0 of Status
    [1] = { .reg_offset = 0, .mask = BIT(1) }, // IRQ 1: Bit 1 of Status
};

static const struct regmap_irq_chip my_irq_chip = {
    .name = "my_device_irq",
    .status_base = REG_INT_STATUS,
    .mask_base = REG_INT_MASK,
    .num_regs = 1,
    .irqs = my_irqs,
    .num_irqs = ARRAY_SIZE(my_irqs),
};
```

#### Step 2: Initialize in Probe
```c
struct regmap_irq_chip_data *irq_data;
int irq = platform_get_irq(pdev, 0);

ret = devm_regmap_add_irq_chip(dev, map, irq, 
                               IRQF_ONESHOT | IRQF_TRIGGER_LOW, 
                               0, &my_irq_chip, &irq_data);

// Request the virtual IRQ for a specific event
int virq = regmap_irq_get_virq(irq_data, 0); // Get Linux IRQ for Bit 0
request_threaded_irq(virq, NULL, my_handler, ...);
```

---

## 🔬 Lab Exercise: Lab 146.1 - Debugfs Exploration

### 1. Lab Objectives
- Load a regmap driver.
- Use debugfs to read/write registers without writing code.

### 2. Step-by-Step Guide
1.  Mount debugfs: `mount -t debugfs none /sys/kernel/debug`.
2.  Navigate: `cd /sys/kernel/debug/regmap/0-0050/` (Example).
3.  **Read All:** `cat registers`.
4.  **Write:** `echo "0x10 0xAB" > registers`. (Writes 0xAB to reg 0x10).
5.  **Check Cache:** `cat access`. See Hit/Miss ratio.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Paging support
- **Goal:** Handle devices with "Pages" (Windowing).
- **Task:**
    1.  Use `regmap_config.ranges`.
    2.  Or use `regmap_add_range`.
    3.  Some devices use a "Page Select" register to access more than 256 bytes. Regmap supports this via `.selector_reg`.

### Lab 3: Multi-Byte Read
- **Goal:** Read a FIFO.
- **Task:**
    1.  `regmap_raw_read(map, FIFO_REG, buf, 128);`
    2.  Note: `raw_read` bypasses the cache and endianness conversion. Useful for data streams.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Regmap init failed"
*   **Cause:** Invalid config. e.g., `reg_bits` not 8/16/32.
*   **Cause:** Bus driver (I2C/SPI) not ready.

#### 2. IRQ Storm
*   **Cause:** `mask_base` incorrect. The driver fails to mask the interrupt, so the line stays low.
*   **Cause:** `ack_base` needed? Some chips need you to Write 1 to Clear. Regmap supports `.ack_base`.

---

## ⚡ Optimization & Best Practices

### `regmap_update_bits`
*   Read-Modify-Write cycle.
*   `regmap_update_bits(map, REG, MASK, VAL);`
*   **Optimization:** If the value in cache matches, it does NOTHING. Saves a bus transaction!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `regmap_write` and `regmap_raw_write`?
    *   **A:** `regmap_write` handles endianness and caching. `regmap_raw_write` sends the bytes exactly as provided, bypassing most logic.
2.  **Q:** Why use `regmap_add_irq_chip`?
    *   **A:** It demultiplexes a single physical IRQ line into multiple virtual IRQs within Linux. This allows you to write clean handlers for "Button Press", "Battery Low", etc., instead of one giant ISR that checks status bits.

### Challenge Task
> **Task:** "The Audio Codec Simulator".
> *   Create a dummy driver with a dense register map (0x00 to 0xFF).
> *   Mark 0x10 as Volatile (Volume VU Meter).
> *   Mark 0x20 as Read-Only (Chip ID).
> *   Verify via debugfs that writing to 0x20 fails or is ignored.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/regmap.rst](https://www.kernel.org/doc/html/latest/driver-api/regmap.html)

---
