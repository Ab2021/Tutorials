# Day 140: Pinctrl and GPIO Subsystems
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
1.  **Distinguish** between Pinctrl (Muxing/Config) and GPIO (High/Low).
2.  **Define** Pin Groups and States in Device Tree (`pinctrl-0`, `pinctrl-names`).
3.  **Consume** GPIOs in a driver using the descriptor-based API (`gpiod_*`).
4.  **Implement** a driver that requests specific pin configurations (e.g., Pull-Up, Drive Strength).
5.  **Debug** pin states using `/sys/kernel/debug/pinctrl`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 136 (DTS).
    *   Digital Logic (Pull-ups, Open Drain).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pinctrl Subsystem
Modern SoCs have thousands of functions but limited pins.
*   **Pin Muxing:** Selecting which internal controller (UART, I2C, GPIO) controls a physical pin.
*   **Pin Config:** Setting electrical properties (Pull-up, Pull-down, Drive Strength, Slew Rate).
*   **The State Machine:**
    *   `default`: Active state.
    *   `sleep`: Low power state (e.g., inputs disconnected).
    *   `idle`: Intermediate state.

### 🔹 Part 2: The GPIO Subsystem (gpiolib)
Once a pin is muxed as "GPIO", this subsystem takes over.
*   **Legacy API:** `gpio_request`, `gpio_set_value` (Integer based). **DEPRECATED**.
*   **Descriptor API:** `gpiod_get`, `gpiod_set_value` (Opaque struct based). **RECOMMENDED**.
    *   Handles Active High/Low automatically based on DTS flags.

---

## 💻 Implementation: Pin Configuration in DTS

> **Instruction:** We will define a pin group for a hypothetical UART and a GPIO LED.

### 👨‍💻 Code Implementation

#### Step 1: Pinctrl Node (SoC specific)
*Note: The syntax depends heavily on the SoC driver (e.g., `pinctrl-single` or vendor specific).*

```dts
/* In the Pinctrl Controller Node */
&pinctrl {
    /* Group for UART */
    uart0_pins: uart0-pins {
        pins = "P1.0", "P1.1";
        function = "uart0";
        bias-pull-up;
    };

    /* Group for LED */
    led_pins: led-pins {
        pins = "P2.5";
        function = "gpio";
        drive-strength = <8>; /* mA */
    };
};
```

#### Step 2: Device Node Usage
```dts
&uart0 {
    pinctrl-names = "default", "sleep";
    pinctrl-0 = <&uart0_pins>; /* Active */
    pinctrl-1 = <&uart0_sleep_pins>; /* Sleep */
    status = "okay";
};

my_led {
    compatible = "gpio-leds";
    pinctrl-names = "default";
    pinctrl-0 = <&led_pins>;
    
    led0 {
        gpios = <&gpio2 5 GPIO_ACTIVE_HIGH>;
    };
};
```

---

## 💻 Implementation: The GPIO Consumer Driver

> **Instruction:** Write a driver that controls a GPIO using the modern API.

### 👨‍💻 Code Implementation

#### Step 1: Driver Source (`gpio_consumer.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/gpio/consumer.h>
#include <linux/delay.h>

struct my_device_data {
    struct gpio_desc *led;
    struct gpio_desc *btn;
};

static int my_probe(struct platform_device *pdev) {
    struct my_device_data *data;
    struct device *dev = &pdev->dev;
    
    data = devm_kzalloc(dev, sizeof(*data), GFP_KERNEL);
    if (!data) return -ENOMEM;
    
    // 1. Get GPIOs
    // Looks for "status-gpios" in DTS
    data->led = devm_gpiod_get(dev, "status", GPIOD_OUT_LOW);
    if (IS_ERR(data->led)) {
        dev_err(dev, "Failed to get LED GPIO\n");
        return PTR_ERR(data->led);
    }
    
    // Looks for "user-gpios" in DTS
    data->btn = devm_gpiod_get(dev, "user", GPIOD_IN);
    if (IS_ERR(data->btn)) {
        return PTR_ERR(data->btn);
    }
    
    // 2. Use them
    gpiod_set_value(data->led, 1); // Turn ON
    
    if (gpiod_get_value(data->btn)) {
        dev_info(dev, "Button is pressed during boot!\n");
    }
    
    platform_set_drvdata(pdev, data);
    return 0;
}

static int my_remove(struct platform_device *pdev) {
    struct my_device_data *data = platform_get_drvdata(pdev);
    
    gpiod_set_value(data->led, 0); // Turn OFF
    return 0;
}

static const struct of_device_id my_match[] = {
    { .compatible = "org,gpio-consumer", },
    { }
};
MODULE_DEVICE_TABLE(of, my_match);

static struct platform_driver my_driver = {
    .probe = my_probe,
    .remove = my_remove,
    .driver = {
        .name = "gpio-consumer",
        .of_match_table = my_match,
    }
};

module_platform_driver(my_driver);
MODULE_LICENSE("GPL");
```

#### Step 2: DTS Entry
```dts
my_consumer {
    compatible = "org,gpio-consumer";
    status-gpios = <&gpio0 10 GPIO_ACTIVE_HIGH>;
    user-gpios = <&gpio0 11 GPIO_ACTIVE_LOW>;
};
```

---

## 🔬 Lab Exercise: Lab 140.1 - Dynamic Pinctrl

### 1. Lab Objectives
- Manually switch pinctrl states in the driver.
- Switch between "default" and "sleep" states.

### 2. Step-by-Step Guide
1.  In `probe`:
    ```c
    struct pinctrl *p = devm_pinctrl_get(dev);
    struct pinctrl_state *s_default = pinctrl_lookup_state(p, "default");
    struct pinctrl_state *s_sleep = pinctrl_lookup_state(p, "sleep");
    ```
2.  Switch:
    ```c
    pinctrl_select_state(p, s_sleep); // Pins go to sleep config
    msleep(1000);
    pinctrl_select_state(p, s_default); // Pins wake up
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: GPIO Arrays
- **Goal:** Handle multiple LEDs.
- **Task:**
    1.  DTS: `leds-gpios = <&gpio0 1 0>, <&gpio0 2 0>;`
    2.  Driver: `devm_gpiod_get_array(...)`.
    3.  Iterate: `for (i=0; i < descs->ndescs; i++) gpiod_set_value(descs->desc[i], 1);`

### Lab 3: Debouncing via Gpiolib
- **Goal:** Use software debounce.
- **Task:**
    1.  `gpiod_set_debounce(btn, 200);`
    2.  Note: This might fail if the hardware or driver doesn't support it.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Pin already requested"
*   **Cause:** Two devices in DTS claim the same `pinctrl-0` pins.
*   **Fix:** Check your DTS. Only one active device can own the pins.

#### 2. LED Logic Inverted
*   **Cause:** DTS says `GPIO_ACTIVE_HIGH`, but hardware is Active Low.
*   **Fix:** Change DTS to `GPIO_ACTIVE_LOW`. The driver code (`gpiod_set_value(led, 1)`) remains the same (logical "ON").

### Debugfs
```bash
cat /sys/kernel/debug/pinctrl/pinctrl-maps
cat /sys/kernel/debug/gpio
```

---

## ⚡ Optimization & Best Practices

### Hogging Pins
*   **GPIO Hogs:** If a pin needs to be set to a specific value permanently (e.g., enabling a regulator) and no driver controls it, use a "hog" in the GPIO controller node.
    ```dts
    &gpio0 {
        enable-wifi {
            gpio-hog;
            gpios = <10 GPIO_ACTIVE_HIGH>;
            output-high;
        };
    };
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if I don't define `pinctrl-0`?
    *   **A:** The pins remain in their boot-up state (or whatever the bootloader left them in). This might work, but is unreliable.
2.  **Q:** Why use `gpiod_set_value` instead of `gpio_set_value`?
    *   **A:** `gpiod` handles polarity (Active High/Low) automatically. `gpio` sets the raw physical level, which forces the driver to know about board schematics.

### Challenge Task
> **Task:** "The Traffic Light".
> *   DTS: 3 LEDs (Red, Yellow, Green) defined as an array.
> *   Driver: Cycle through them (R->Y->G) using a timer.
> *   Constraint: Use `gpiod_get_array` and `gpiod_set_array_value`.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/gpio/consumer.rst](https://www.kernel.org/doc/html/latest/driver-api/gpio/consumer.html)
- [Kernel Documentation: driver-api/pinctl.rst](https://www.kernel.org/doc/html/latest/driver-api/pinctl.html)

---
