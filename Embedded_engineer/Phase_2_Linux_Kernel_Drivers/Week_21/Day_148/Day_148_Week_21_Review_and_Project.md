# Day 148: Week 21 Review & Project - The Universal Sensor Hub
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
1.  **Synthesize** Week 21 concepts (I2C, SPI, Regmap, Input).
2.  **Architect** a multi-protocol driver (Core Logic + I2C Glue + SPI Glue).
3.  **Implement** a complete Input Device driver for a 3-Axis Accelerometer.
4.  **Demonstrate** proper locking and concurrency in a bus driver.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   All tools from Days 142-147.
*   **Prior Knowledge:**
    *   Week 21 Content.

---

## 🔄 Week 21 Review

### 1. I2C Subsystem (Days 142-143)
*   **Architecture:** Adapter (Master) <-> Client (Slave).
*   **API:** `i2c_transfer` (Raw), `i2c_smbus_*` (Register).
*   **Drivers:** `i2c_driver`, `probe`, `id_table`.

### 2. SPI Subsystem (Days 144-145)
*   **Architecture:** Controller <-> Device.
*   **API:** `spi_sync`, `spi_async`, `spi_message`, `spi_transfer`.
*   **Speed:** Higher throughput, Full Duplex.

### 3. Regmap (Day 146)
*   **Abstraction:** Hides I2C/SPI differences.
*   **Features:** Caching, Endianness, IRQ Chip.

### 4. Input Subsystem (Day 147)
*   **Purpose:** Standardize user input.
*   **Events:** `EV_KEY`, `EV_ABS`.
*   **Tools:** `evtest`.

---

## 🛠️ Project: The "UniAccel" Driver

### 📋 Project Requirements
Create a driver for a hypothetical Accelerometer (`uniaccel`).
1.  **Multi-Bus:** Support both I2C and SPI interfaces.
2.  **Regmap:** Use Regmap for register access.
3.  **Input Device:** Report X, Y, Z acceleration as `ABS_X`, `ABS_Y`, `ABS_Z`.
4.  **Interrupts:** Support "Data Ready" interrupt to trigger reading.
5.  **Sysfs:** Expose sampling rate configuration.

### 🏗️ Architecture
*   `uniaccel_core.c`: The main logic (Regmap -> Input).
*   `uniaccel_i2c.c`: `i2c_driver` that calls Core.
*   `uniaccel_spi.c`: `spi_driver` that calls Core.
*   `uniaccel.h`: Shared definitions.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: Shared Header (`uniaccel.h`)

```c
#ifndef UNIACCEL_H
#define UNIACCEL_H

#include <linux/regmap.h>
#include <linux/device.h>

struct uniaccel_data {
    struct device *dev;
    struct regmap *map;
    struct input_dev *input;
    int irq;
};

int uniaccel_core_probe(struct device *dev, struct regmap *map, int irq, const char *name);
void uniaccel_core_remove(struct device *dev);

extern const struct regmap_config uniaccel_regmap_config;

#endif
```

### 🔹 Phase 2: Core Logic (`uniaccel_core.c`)

```c
#include <linux/module.h>
#include <linux/input.h>
#include <linux/interrupt.h>
#include "uniaccel.h"

#define REG_CHIP_ID 0x00
#define REG_DATA_X  0x02
#define REG_DATA_Y  0x04
#define REG_DATA_Z  0x06

const struct regmap_config uniaccel_regmap_config = {
    .reg_bits = 8,
    .val_bits = 16, // 16-bit data
    .val_format_endian = REGMAP_ENDIAN_LITTLE,
    .max_register = 0xFF,
};
EXPORT_SYMBOL_GPL(uniaccel_regmap_config);

static irqreturn_t uniaccel_isr(int irq, void *d) {
    struct uniaccel_data *data = d;
    int ret;
    u32 val_x, val_y, val_z;

    // Bulk read or individual reads
    regmap_read(data->map, REG_DATA_X, &val_x);
    regmap_read(data->map, REG_DATA_Y, &val_y);
    regmap_read(data->map, REG_DATA_Z, &val_z);

    input_report_abs(data->input, ABS_X, (s16)val_x);
    input_report_abs(data->input, ABS_Y, (s16)val_y);
    input_report_abs(data->input, ABS_Z, (s16)val_z);
    input_sync(data->input);

    return IRQ_HANDLED;
}

int uniaccel_core_probe(struct device *dev, struct regmap *map, int irq, const char *name) {
    struct uniaccel_data *data;
    int ret;

    data = devm_kzalloc(dev, sizeof(*data), GFP_KERNEL);
    if (!data) return -ENOMEM;

    data->dev = dev;
    data->map = map;
    data->irq = irq;
    dev_set_drvdata(dev, data);

    // Input Device Init
    data->input = devm_input_allocate_device(dev);
    data->input->name = name;
    set_bit(EV_ABS, data->input->evbit);
    input_set_abs_params(data->input, ABS_X, -32768, 32767, 0, 0);
    input_set_abs_params(data->input, ABS_Y, -32768, 32767, 0, 0);
    input_set_abs_params(data->input, ABS_Z, -32768, 32767, 0, 0);

    ret = input_register_device(data->input);
    if (ret) return ret;

    // IRQ
    ret = devm_request_threaded_irq(dev, irq, NULL, uniaccel_isr,
                                    IRQF_TRIGGER_RISING | IRQF_ONESHOT,
                                    name, data);
    if (ret) return ret;

    dev_info(dev, "UniAccel Core Probed\n");
    return 0;
}
EXPORT_SYMBOL_GPL(uniaccel_core_probe);

void uniaccel_core_remove(struct device *dev) {
    dev_info(dev, "UniAccel Core Removed\n");
}
EXPORT_SYMBOL_GPL(uniaccel_core_remove);

MODULE_LICENSE("GPL");
```

### 🔹 Phase 3: I2C Glue (`uniaccel_i2c.c`)

```c
#include <linux/module.h>
#include <linux/i2c.h>
#include "uniaccel.h"

static int uniaccel_i2c_probe(struct i2c_client *client, const struct i2c_device_id *id) {
    struct regmap *map;
    
    map = devm_regmap_init_i2c(client, &uniaccel_regmap_config);
    if (IS_ERR(map)) return PTR_ERR(map);
    
    return uniaccel_core_probe(&client->dev, map, client->irq, "uniaccel-i2c");
}

static void uniaccel_i2c_remove(struct i2c_client *client) {
    uniaccel_core_remove(&client->dev);
}

static const struct i2c_device_id uniaccel_i2c_id[] = { { "uniaccel", 0 }, { } };
MODULE_DEVICE_TABLE(i2c, uniaccel_i2c_id);

static struct i2c_driver uniaccel_i2c_driver = {
    .driver = { .name = "uniaccel-i2c" },
    .probe = uniaccel_i2c_probe,
    .remove = uniaccel_i2c_remove,
    .id_table = uniaccel_i2c_id,
};
module_i2c_driver(uniaccel_i2c_driver);
MODULE_LICENSE("GPL");
```

### 🔹 Phase 4: SPI Glue (`uniaccel_spi.c`)

```c
#include <linux/module.h>
#include <linux/spi/spi.h>
#include "uniaccel.h"

static int uniaccel_spi_probe(struct spi_device *spi) {
    struct regmap *map;
    
    map = devm_regmap_init_spi(spi, &uniaccel_regmap_config);
    if (IS_ERR(map)) return PTR_ERR(map);
    
    return uniaccel_core_probe(&spi->dev, map, spi->irq, "uniaccel-spi");
}

static void uniaccel_spi_remove(struct spi_device *spi) {
    uniaccel_core_remove(&spi->dev);
}

static struct spi_driver uniaccel_spi_driver = {
    .driver = { .name = "uniaccel-spi" },
    .probe = uniaccel_spi_probe,
    .remove = uniaccel_spi_remove,
};
module_spi_driver(uniaccel_spi_driver);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile all 3 modules. Load Core first.

### 👨‍💻 Command Line Steps

1.  **Compile:** Ensure Makefile builds `uniaccel_core.o`, `uniaccel_i2c.o`, `uniaccel_spi.o`.
2.  **Load:**
    ```bash
    insmod uniaccel_core.ko
    insmod uniaccel_i2c.ko
    insmod uniaccel_spi.ko
    ```
3.  **Simulate:**
    *   Create I2C device via `new_device`.
    *   Or SPI device via Overlay.
4.  **Verify:**
    *   `dmesg`: "UniAccel Core Probed".
    *   `evtest`: Check for events.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Modularity** | Clean separation of Core vs Bus logic. | Logic mixed in Bus files. | Monolithic file. |
| **Regmap** | Correct config for endianness/bits. | Manual I2C/SPI calls. | No Regmap. |
| **Input** | Correct ABS params and event reporting. | Wrong event types. | No Input device. |

---

## 🔮 Looking Ahead: Week 22
Next week, we dive into **V4L2 (Video for Linux 2)**.
*   Camera Sensors.
*   Video Buffers.
*   Streaming API.

---
