# Day 139: Platform Driver with Device Tree - Advanced Integration
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
1.  **Support** multiple hardware variants in a single driver using `of_device_id.data`.
2.  **Retrieve** variant-specific configuration using `of_device_get_match_data`.
3.  **Handle** multiple instances of the same device (e.g., 4 UARTs) cleanly.
4.  **Implement** a robust initialization sequence that handles missing properties gracefully.
5.  **Debug** probe deferral issues (`-EPROBE_DEFER`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 135 (Platform Drivers).
    *   Day 138 (OF API).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: One Driver, Many Variants
Imagine you have a "Temp Sensor V1" and "Temp Sensor V2".
*   V1 has a 10-bit ADC.
*   V2 has a 12-bit ADC and a different register offset.
*   **Bad Approach:** Write `sensor_v1.c` and `sensor_v2.c`. Code duplication!
*   **Good Approach:** Write `sensor_driver.c` that checks `compatible` string.
    *   `"vendor,sensor-v1"` -> Load V1 config.
    *   `"vendor,sensor-v2"` -> Load V2 config.

### 🔹 Part 2: Probe Deferral
What if your driver needs a GPIO, but the GPIO driver hasn't loaded yet?
*   `devm_gpiod_get` returns `-EPROBE_DEFER`.
*   **Action:** Return this error from `probe`.
*   **Kernel:** Puts your device in a "Deferred List" and tries to probe it again later (after other drivers load).

---

## 💻 Implementation: The Multi-Variant Driver

> **Instruction:** We will create a driver that behaves differently based on the compatible string.

### 👨‍💻 Code Implementation

#### Step 1: DTS Overlay (`variants.dts`)

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            /* Variant 1 */
            sensor_v1: sensor@1000 {
                compatible = "org,sensor-v1";
                reg = <0x1000 0x20>;
            };

            /* Variant 2 */
            sensor_v2: sensor@2000 {
                compatible = "org,sensor-v2";
                reg = <0x2000 0x20>;
            };
        };
    };
};
```

#### Step 2: Driver (`variant_driver.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/of_device.h>

// Configuration Structure
struct sensor_config {
    const char *name;
    int resolution_bits;
    int register_offset;
};

// Config for V1
static const struct sensor_config config_v1 = {
    .name = "Sensor V1 (Legacy)",
    .resolution_bits = 10,
    .register_offset = 0x00,
};

// Config for V2
static const struct sensor_config config_v2 = {
    .name = "Sensor V2 (High Res)",
    .resolution_bits = 12,
    .register_offset = 0x10,
};

// Match Table with Data
static const struct of_device_id sensor_dt_ids[] = {
    { .compatible = "org,sensor-v1", .data = &config_v1 },
    { .compatible = "org,sensor-v2", .data = &config_v2 },
    { }
};
MODULE_DEVICE_TABLE(of, sensor_dt_ids);

static int sensor_probe(struct platform_device *pdev) {
    const struct sensor_config *cfg;
    
    // Retrieve the config based on which compatible string matched
    cfg = of_device_get_match_data(&pdev->dev);
    if (!cfg) {
        return -ENODEV;
    }
    
    printk(KERN_INFO "Probe: %s\n", cfg->name);
    printk(KERN_INFO "  Resolution: %d bits\n", cfg->resolution_bits);
    printk(KERN_INFO "  Offset: 0x%x\n", cfg->register_offset);
    
    // ... Initialize hardware using cfg->register_offset ...
    
    return 0;
}

static struct platform_driver sensor_driver = {
    .probe = sensor_probe,
    .driver = {
        .name = "universal-sensor",
        .of_match_table = sensor_dt_ids,
    }
};

module_platform_driver(sensor_driver);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Handling Multiple Instances

> **Instruction:** The driver must not use global variables for device state!

### 👨‍💻 Code Implementation

#### Step 1: Private Data Structure
```c
struct sensor_private {
    struct device *dev;
    void __iomem *base_addr;
    int irq;
    // ... other per-device state ...
};
```

#### Step 2: Probe Allocation
```c
static int sensor_probe(struct platform_device *pdev) {
    struct sensor_private *priv;
    
    // Allocate per-instance structure
    priv = devm_kzalloc(&pdev->dev, sizeof(*priv), GFP_KERNEL);
    if (!priv) return -ENOMEM;
    
    priv->dev = &pdev->dev;
    
    // Save it in the platform_device for later (e.g., in remove or suspend)
    platform_set_drvdata(pdev, priv);
    
    // ...
    return 0;
}
```

#### Step 3: Usage in Remove
```c
static int sensor_remove(struct platform_device *pdev) {
    struct sensor_private *priv = platform_get_drvdata(pdev);
    
    printk("Removing device at %p\n", priv->base_addr);
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 139.1 - Probe Deferral Simulation

### 1. Lab Objectives
- Simulate a missing dependency.
- Return `-EPROBE_DEFER`.
- Observe `dmesg` to see the kernel retrying.

### 2. Step-by-Step Guide
1.  In `probe`:
    ```c
    static int attempt = 0;
    if (attempt < 3) {
        printk("Dependency missing, deferring probe (attempt %d)\n", attempt);
        attempt++;
        return -EPROBE_DEFER;
    }
    printk("Dependency found, probing success!\n");
    ```
2.  Load driver.
3.  Check logs. You might see the message multiple times as other drivers load and trigger a retry.

---

## 🧪 Additional / Advanced Labs

### Lab 2: `of_match_device` vs `of_device_get_match_data`
- **Goal:** Understand the difference.
- **Task:**
    1.  Use `of_match_device(sensor_dt_ids, &pdev->dev)`.
    2.  It returns a `struct of_device_id *`.
    3.  Access data via `match->data`.
    4.  *Note: `of_device_get_match_data` is the modern helper that does this in one step.*

### Lab 3: Required vs Optional Properties
- **Goal:** Robust parsing.
- **Task:**
    1.  Make "resolution-bits" an optional property in DTS.
    2.  In driver, check if property exists.
    3.  If yes, override the default from `config_vX`.
    4.  If no, use default.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Null Pointer Dereference" in Probe
*   **Cause:** Using global variables for device state. When the second device probes, it overwrites the pointer used by the first device.
*   **Fix:** ALWAYS use `devm_kzalloc` and `platform_set_drvdata`.

#### 2. Driver loads but nothing happens
*   **Cause:** You loaded the driver, but forgot to apply the DTS overlay.
*   **Cause:** You applied the overlay, but the `status` is not `"okay"`.

---

## ⚡ Optimization & Best Practices

### Const Correctness
*   Mark your configuration structures and match tables as `const`. They should not change at runtime.
    ```c
    static const struct of_device_id ...
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** When does `-EPROBE_DEFER` happen automatically?
    *   **A:** When you use APIs like `devm_gpiod_get`, `devm_clk_get`, `devm_regulator_get` and the provider driver (GPIO/Clk/Regulator) is not yet ready.
2.  **Q:** Can I pass a function pointer in `.data`?
    *   **A:** Yes! This is a powerful pattern. You can pass a pointer to a "reset" function specific to that hardware variant.

### Challenge Task
> **Task:** "The Hybrid Driver".
> *   Support both Device Tree AND Platform Data (Legacy C struct).
> *   In `probe`:
>     1.  Check `dev_get_platdata`. If exists, use it.
>     2.  Else, check `of_device_get_match_data`. If exists, use it.
>     3.  Else, Error.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/driver-model/driver.rst](https://www.kernel.org/doc/html/latest/driver-api/driver-model/driver.html)
- [LWN: Probe Deferral](https://lwn.net/Articles/485194/)

---
