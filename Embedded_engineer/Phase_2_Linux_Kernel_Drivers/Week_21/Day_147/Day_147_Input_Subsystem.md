# Day 147: The Input Subsystem
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
1.  **Explain** the Input Subsystem architecture (Device, Handler, Event).
2.  **Register** an Input Device (`input_register_device`).
3.  **Report** events (`input_report_key`, `input_report_abs`) from an ISR.
4.  **Map** physical keys to Linux Keycodes (`KEY_POWER`, `KEY_A`).
5.  **Test** input devices using `evtest` and `getevent`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `evtest` (`sudo apt install evtest`).
*   **Prior Knowledge:**
    *   Day 132 (Interrupts).
    *   Day 140 (GPIO).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Architecture
The Input Subsystem unifies all user input devices (Keyboard, Mouse, Joystick, Touchscreen).
1.  **Input Device Driver:** Talks to hardware (e.g., reads GPIO, reads I2C registers). Reports "Events".
2.  **Input Core:** Normalizes events.
3.  **Input Handlers:** Interface to userspace.
    *   `evdev`: The standard handler. Creates `/dev/input/eventX`.
    *   `mousedev`: Emulates PS/2 mouse.
    *   `joydev`: Joystick interface.

### 🔹 Part 2: The Event Protocol
Userspace reads `struct input_event` from `/dev/input/eventX`.
*   `type`: EV_KEY, EV_REL (Relative Mouse), EV_ABS (Touchscreen).
*   `code`: KEY_A, REL_X, ABS_MT_POSITION_X.
*   `value`: 1 (Press), 0 (Release), Coordinate.

---

## 💻 Implementation: GPIO Button Driver

> **Instruction:** We will create a driver that turns a GPIO interrupt into a Keyboard Key press.

### 👨‍💻 Code Implementation

#### Step 1: Driver Structure
```c
#include <linux/module.h>
#include <linux/input.h>
#include <linux/gpio/consumer.h>
#include <linux/interrupt.h>
#include <linux/platform_device.h>

struct my_button_data {
    struct input_dev *input;
    struct gpio_desc *gpio;
    int irq;
};
```

#### Step 2: Interrupt Handler
```c
static irqreturn_t button_isr(int irq, void *dev_id) {
    struct my_button_data *data = dev_id;
    int state;
    
    // Read GPIO state
    state = gpiod_get_value(data->gpio);
    
    // Report Event
    // KEY_POWER: The keycode
    // state: 1 = Pressed, 0 = Released
    input_report_key(data->input, KEY_POWER, state);
    input_sync(data->input); // Commit the event frame
    
    return IRQ_HANDLED;
}
```

#### Step 3: Probe Function
```c
static int button_probe(struct platform_device *pdev) {
    struct my_button_data *data;
    int error;
    
    data = devm_kzalloc(&pdev->dev, sizeof(*data), GFP_KERNEL);
    
    // 1. Get GPIO
    data->gpio = devm_gpiod_get(&pdev->dev, "button", GPIOD_IN);
    if (IS_ERR(data->gpio)) return PTR_ERR(data->gpio);
    
    // 2. Allocate Input Device
    data->input = devm_input_allocate_device(&pdev->dev);
    if (!data->input) return -ENOMEM;
    
    data->input->name = "My GPIO Button";
    data->input->id.bustype = BUS_HOST;
    
    // 3. Declare Capabilities
    // We generate Key events (EV_KEY)
    set_bit(EV_KEY, data->input->evbit);
    // We generate KEY_POWER
    set_bit(KEY_POWER, data->input->keybit);
    
    // 4. Register Input Device
    error = input_register_device(data->input);
    if (error) return error;
    
    // 5. Request IRQ
    data->irq = gpiod_to_irq(data->gpio);
    error = devm_request_any_context_irq(&pdev->dev, data->irq, button_isr,
                                         IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING,
                                         "my-button", data);
    if (error < 0) return error;
    
    return 0;
}
```

---

## 💻 Implementation: Polled Input Device

> **Instruction:** What if the device doesn't have an interrupt? Use `input-polldev`.

### 👨‍💻 Code Implementation

```c
#include <linux/input-polldev.h>

static void my_poll(struct input_polled_dev *dev) {
    struct my_data *data = dev->private;
    int val = read_sensor_value();
    
    input_report_abs(dev->input, ABS_X, val);
    input_sync(dev->input);
}

// In Probe:
// poll_dev = devm_input_allocate_polled_device(...);
// poll_dev->poll = my_poll;
// poll_dev->poll_interval = 50; // ms
// input_register_polled_device(poll_dev);
```

---

## 🔬 Lab Exercise: Lab 147.1 - Testing with evtest

### 1. Lab Objectives
- Load the button driver.
- Use `evtest` to see the events.

### 2. Step-by-Step Guide
1.  Load module.
2.  Run `sudo evtest`.
3.  Select "My GPIO Button".
4.  Press the physical button (or trigger GPIO in QEMU).
5.  **Output:**
    ```text
    Event: time 123.456, type 1 (EV_KEY), code 116 (KEY_POWER), value 1
    Event: time 123.456, -------------- SYN_REPORT ------------
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Touchscreen Driver (I2C)
- **Goal:** Report Absolute Coordinates.
- **Task:**
    1.  `set_bit(EV_ABS, input->evbit);`
    2.  `input_set_abs_params(input, ABS_X, 0, 1024, 0, 0);`
    3.  In ISR:
        ```c
        input_report_abs(input, ABS_X, x);
        input_report_abs(input, ABS_Y, y);
        input_report_key(input, BTN_TOUCH, 1);
        input_sync(input);
        ```

### Lab 3: Multitouch (MT) Protocol
- **Goal:** Support 2 fingers.
- **Task:**
    1.  Use `input_mt_init_slots(input, 2, 0)`.
    2.  Report:
        ```c
        input_mt_slot(input, 0);
        input_mt_report_slot_state(input, MT_TOOL_FINGER, true);
        input_report_abs(input, ABS_MT_POSITION_X, x1);
        ```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. No Events in evtest
*   **Cause:** Forgot `input_sync()`. The kernel buffers events until a Sync is received.
*   **Cause:** Forgot to set capability bits (`set_bit`). The core filters out unexpected events.

#### 2. "Debouncing"
*   **Problem:** One press generates 50 events.
*   **Fix:** Use `gpio_set_debounce` if hardware supports it, or use a timer in the driver.

---

## ⚡ Optimization & Best Practices

### `gpio-keys` Driver
*   **Pro Tip:** For simple GPIO buttons, **DO NOT WRITE A DRIVER**.
*   Use the existing `gpio-keys` driver in the kernel.
*   Just configure it in Device Tree:
    ```dts
    gpio-keys {
        compatible = "gpio-keys";
        power_btn {
            label = "Power";
            gpios = <&gpio0 5 GPIO_ACTIVE_LOW>;
            linux,code = <KEY_POWER>;
        };
    };
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `EV_SYN`?
    *   **A:** Synchronization Event. It tells userspace "This batch of updates (X, Y, Button) happened at the same time".
2.  **Q:** Can I simulate input from userspace?
    *   **A:** Yes, using `/dev/uinput`. Useful for testing applications without hardware.

### Challenge Task
> **Task:** "The Virtual Joystick".
> *   Write a driver that exposes a Joystick (`EV_ABS`).
> *   Use a timer to move the X/Y axes in a circle automatically.
> *   Verify with `jstest` (part of `joystick` package).

---

## 📚 Further Reading & References
- [Kernel Documentation: input/input-programming.rst](https://www.kernel.org/doc/html/latest/input/input-programming.html)
- [Linux Input Event Codes](https://github.com/torvalds/linux/blob/master/include/uapi/linux/input-event-codes.h)

---
