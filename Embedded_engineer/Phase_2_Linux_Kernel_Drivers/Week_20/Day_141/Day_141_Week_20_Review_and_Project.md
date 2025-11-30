# Day 141: Week 20 Review & Project - The Smart Sensor Platform Driver
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
1.  **Synthesize** Week 20 concepts (Platform Bus, Device Tree, OF API, Pinctrl).
2.  **Architect** a production-ready Platform Driver that binds via Device Tree.
3.  **Implement** robust resource handling using Managed APIs (`devm_*`).
4.  **Expose** driver attributes to User Space via Sysfs.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   All tools from Days 135-140.
*   **Prior Knowledge:**
    *   Week 20 Content.

---

## 🔄 Week 20 Review

### 1. Platform Device Model (Day 135)
*   **Bus:** Virtual bus for on-chip peripherals.
*   **Binding:** Matching `driver.name` (Legacy) or `compatible` (DT).
*   **Probe/Remove:** The lifecycle methods.

### 2. Device Tree (Days 136-138)
*   **DTS:** Hardware description.
*   **Properties:** `reg`, `interrupts`, `compatible`.
*   **OF API:** `of_property_read_*`, `of_match_node`.

### 3. Advanced Integration (Day 139)
*   **Match Data:** Supporting multiple variants (`v1`, `v2`).
*   **Instances:** Handling multiple devices via `platform_set_drvdata`.

### 4. Pinctrl & GPIO (Day 140)
*   **Pinctrl:** Muxing pins (UART vs GPIO).
*   **Gpiolib:** Descriptor-based access (`gpiod_get`).

---

## 🛠️ Project: The "Smart Sensor" Driver

### 📋 Project Requirements
Create a driver `smart_sensor` for a hypothetical memory-mapped sensor.
1.  **DTS Overlay:**
    *   Node: `smart-sensor@5000`.
    *   Reg: `0x5000` size `0x100`.
    *   IRQ: `12`.
    *   Property: `sensor-sensitivity = <100>;`.
    *   GPIO: `enable-gpios`.
2.  **Driver:**
    *   Match `org,smart-sensor`.
    *   Parse `sensor-sensitivity`.
    *   Toggle Enable GPIO in `probe`.
    *   Handle IRQ (count interrupts).
    *   Expose `sensitivity` and `irq_count` in Sysfs.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: DTS Overlay

**`smart_sensor.dts`**
```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            my_sensor: smart-sensor@5000 {
                compatible = "org,smart-sensor";
                reg = <0x5000 0x100>;
                interrupt-parent = <&gic>; /* Or &intc in QEMU */
                interrupts = <0 12 4>;
                sensor-sensitivity = <100>;
                enable-gpios = <&gpio0 5 0>; /* Active High */
                status = "okay";
            };
        };
    };
};
```

### 🔹 Phase 2: Driver Structure & Probe

**`smart_sensor.c`**
```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/gpio/consumer.h>
#include <linux/interrupt.h>

struct sensor_data {
    void __iomem *base;
    int irq;
    u32 sensitivity;
    struct gpio_desc *enable_gpio;
    int irq_count;
    struct device *dev;
};

// Sysfs Show Function
static ssize_t sensitivity_show(struct device *dev, struct device_attribute *attr, char *buf) {
    struct sensor_data *data = dev_get_drvdata(dev);
    return sprintf(buf, "%d\n", data->sensitivity);
}
static DEVICE_ATTR_RO(sensitivity);

static ssize_t irq_count_show(struct device *dev, struct device_attribute *attr, char *buf) {
    struct sensor_data *data = dev_get_drvdata(dev);
    return sprintf(buf, "%d\n", data->irq_count);
}
static DEVICE_ATTR_RO(irq_count);

// Interrupt Handler
static irqreturn_t sensor_isr(int irq, void *dev_id) {
    struct sensor_data *data = dev_id;
    data->irq_count++;
    return IRQ_HANDLED;
}

// Probe
static int sensor_probe(struct platform_device *pdev) {
    struct sensor_data *data;
    struct resource *res;
    int ret;
    
    dev_info(&pdev->dev, "Probing Smart Sensor...\n");
    
    // 1. Allocate Data
    data = devm_kzalloc(&pdev->dev, sizeof(*data), GFP_KERNEL);
    if (!data) return -ENOMEM;
    data->dev = &pdev->dev;
    platform_set_drvdata(pdev, data);
    
    // 2. Get Memory
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    data->base = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(data->base)) return PTR_ERR(data->base);
    
    // 3. Parse DT Property
    if (of_property_read_u32(pdev->dev.of_node, "sensor-sensitivity", &data->sensitivity)) {
        data->sensitivity = 50; // Default
        dev_warn(&pdev->dev, "Sensitivity not found, using default 50\n");
    }
    
    // 4. Get GPIO
    data->enable_gpio = devm_gpiod_get(&pdev->dev, "enable", GPIOD_OUT_LOW);
    if (IS_ERR(data->enable_gpio)) return PTR_ERR(data->enable_gpio);
    
    // 5. Enable Sensor
    gpiod_set_value(data->enable_gpio, 1);
    
    // 6. Get IRQ
    data->irq = platform_get_irq(pdev, 0);
    if (data->irq < 0) return data->irq;
    
    ret = devm_request_irq(&pdev->dev, data->irq, sensor_isr, 0, "smart_sensor", data);
    if (ret) return ret;
    
    // 7. Create Sysfs Files
    device_create_file(&pdev->dev, &dev_attr_sensitivity);
    device_create_file(&pdev->dev, &dev_attr_irq_count);
    
    dev_info(&pdev->dev, "Probe Success. Sensitivity=%d\n", data->sensitivity);
    return 0;
}

// Remove
static int sensor_remove(struct platform_device *pdev) {
    struct sensor_data *data = platform_get_drvdata(pdev);
    
    gpiod_set_value(data->enable_gpio, 0); // Disable
    
    device_remove_file(&pdev->dev, &dev_attr_sensitivity);
    device_remove_file(&pdev->dev, &dev_attr_irq_count);
    
    dev_info(&pdev->dev, "Removed\n");
    return 0;
}

// Match Table
static const struct of_device_id sensor_ids[] = {
    { .compatible = "org,smart-sensor" },
    { }
};
MODULE_DEVICE_TABLE(of, sensor_ids);

static struct platform_driver sensor_driver = {
    .probe = sensor_probe,
    .remove = sensor_remove,
    .driver = {
        .name = "smart-sensor",
        .of_match_table = sensor_ids,
    }
};

module_platform_driver(sensor_driver);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile, Load Overlay, Load Driver.

### 👨‍💻 Command Line Steps

1.  **Apply Overlay:** `cat smart_sensor.dtbo > /sys/kernel/config/...`
2.  **Load Driver:** `insmod smart_sensor.ko`
3.  **Check Logs:** `dmesg` should show "Probe Success".
4.  **Check Sysfs:**
    ```bash
    cd /sys/bus/platform/devices/5000.smart-sensor/
    cat sensitivity
    # Output: 100
    cat irq_count
    # Output: 0
    ```
5.  **Trigger IRQ (Simulated):** If using QEMU monitor or a test script to trigger IRQ 12.
6.  **Unload:** `rmmod smart_sensor` -> "Removed".

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **DT Integration** | Correctly parses all properties and handles defaults. | Misses properties or hardcodes values. | Fails to match. |
| **Resource Mgmt** | Uses `devm_` everywhere. Clean remove. | Manual `kfree` mixed with `devm`. | Memory leaks. |
| **Sysfs** | Attributes work and reflect internal state. | Attributes missing or crash on read. | No Sysfs. |

---

## 🔮 Looking Ahead: Week 21
Next week, we move to **Bus Drivers (I2C & SPI)**.
*   We will apply the Platform Driver concepts to I2C/SPI Clients.
*   `i2c_driver` and `spi_driver` work very similarly to `platform_driver`.
*   We will write drivers for real sensors (accelerometers, EEPROMs).

---
