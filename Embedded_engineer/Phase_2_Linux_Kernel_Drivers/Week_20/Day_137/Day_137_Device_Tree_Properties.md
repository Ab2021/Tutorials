# Day 137: Device Tree Properties - Deep Dive
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
1.  **Decode** complex `reg` properties using `#address-cells` and `#size-cells`.
2.  **Implement** Address Translation using the `ranges` property.
3.  **Map** Interrupts from a device to an Interrupt Controller using `interrupt-parent`.
4.  **Parse** GPIO bindings (`gpios` property).
5.  **Debug** property parsing errors in the kernel.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 136 (DTS Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Address Cells and Size Cells
The `reg` property is a list of raw numbers. How the kernel interprets them depends on the **parent** node.
*   `#address-cells`: Number of 32-bit cells for the Base Address.
*   `#size-cells`: Number of 32-bit cells for the Length.

**Example 1: 32-bit System**
```dts
parent {
    #address-cells = <1>;
    #size-cells = <1>;
    
    child {
        reg = <0x1000 0x20>; /* Addr=0x1000, Size=0x20 */
    };
};
```

**Example 2: 64-bit System**
```dts
parent {
    #address-cells = <2>; /* High + Low */
    #size-cells = <2>;
    
    child {
        reg = <0x0 0x80000000  0x0 0x100000>; 
        /* Addr=0x80000000 (4GB space), Size=1MB */
    };
};
```

### 🔹 Part 2: The `ranges` Property (Address Translation)
If a bus (like PCIe or I2C) has its own address space, we need to map it to the CPU's memory map.
*   Format: `<ChildAddr ParentAddr Size>`
*   `ranges;` (Empty): 1:1 Mapping (Identity).
*   `ranges = <0x0 0x40000000 0x1000>;`: Child address 0x0 maps to CPU address 0x40000000.

### 🔹 Part 3: Interrupt Mapping
*   `interrupt-parent`: Phandle to the Interrupt Controller (GIC, NVIC, GPIO).
*   `interrupts`: List of specifiers.
    *   Format depends on the controller.
    *   ARM GIC: `<Type ID Flags>` (e.g., `<GIC_SPI 34 IRQ_TYPE_LEVEL_HIGH>`).

---

## 💻 Implementation: Complex Device Node

> **Instruction:** We will create a simulated "Bus" node with address translation and a child device.

### 👨‍💻 Code Implementation

#### Step 1: DTS Overlay (`complex_overlay.dts`)

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            
            /* A Custom Bus */
            my_bus: my-custom-bus@40000000 {
                compatible = "simple-bus";
                #address-cells = <1>;
                #size-cells = <1>;
                
                /* Map Bus 0x0 -> CPU 0x40000000, Size 0x1000 */
                ranges = <0x0 0x40000000 0x1000>;
                
                /* Child Device on the Bus */
                my_dev: child-dev@100 {
                    compatible = "org,child-device";
                    reg = <0x100 0x20>; /* Bus Addr 0x100 -> CPU 0x40000100 */
                    interrupt-parent = <&gic>; /* Assuming GIC exists */
                    interrupts = <0 40 4>;     /* SPI 40, Level High */
                };
            };
        };
    };
};
```
*Note: In QEMU Virt, the GIC phandle might be different (e.g., `&intc`). Check the base DTS.*

#### Step 2: Driver (`child_driver.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>

static int child_probe(struct platform_device *pdev) {
    struct resource *res;
    
    // 1. Get Memory (Kernel automatically handles translation!)
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (res) {
        printk("Child: Phys Addr Start=0x%llx\n", res->start);
        // Expected: 0x40000100 (Not 0x100)
    }
    
    // 2. Get IRQ
    int irq = platform_get_irq(pdev, 0);
    printk("Child: Virtual IRQ=%d\n", irq);
    
    return 0;
}

// ... Standard remove, driver struct, module_init ...
// Compatible: "org,child-device"
```

---

## 💻 Implementation: GPIO Properties

> **Instruction:** Parsing GPIOs is slightly different. We use the GPIO Descriptor API.

### 👨‍💻 Code Implementation

#### Step 1: DTS Update
```dts
my_dev: child-dev@100 {
    ...
    /* Name "led", Index 0, Active Low */
    led-gpios = <&gpio0 5 1>; 
    /* Name "btn", Index 0, Active High */
    btn-gpios = <&gpio0 6 0>; 
};
```

#### Step 2: Driver Update
```c
#include <linux/gpio/consumer.h>

struct gpio_desc *led_gpio;
struct gpio_desc *btn_gpio;

static int child_probe(struct platform_device *pdev) {
    // Get GPIO by name "led" -> looks for "led-gpios" property
    led_gpio = devm_gpiod_get(&pdev->dev, "led", GPIOD_OUT_LOW);
    if (IS_ERR(led_gpio)) {
        printk("Failed to get LED GPIO\n");
        return PTR_ERR(led_gpio);
    }
    
    btn_gpio = devm_gpiod_get(&pdev->dev, "btn", GPIOD_IN);
    
    // Toggle LED
    gpiod_set_value(led_gpio, 1);
    
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 137.1 - Manual Parsing

### 1. Lab Objectives
- Use `of_address_to_resource` manually.
- Use `of_irq_get` manually.
- Compare results with `platform_get_resource`.

### 2. Step-by-Step Guide
1.  In `probe`:
    ```c
    struct device_node *np = pdev->dev.of_node;
    struct resource r;
    
    if (of_address_to_resource(np, 0, &r) == 0) {
        printk("Manual: Start=0x%llx\n", r.start);
    }
    ```
2.  Verify it matches the Platform API result.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Multiple Reg Ranges
- **Goal:** Handle a device with 2 memory regions (Control Regs + Data Buffer).
- **Task:**
    1.  DTS: `reg = <0x100 0x20>, <0x200 0x1000>;`
    2.  Driver:
        *   `platform_get_resource(pdev, IORESOURCE_MEM, 0)` -> Control.
        *   `platform_get_resource(pdev, IORESOURCE_MEM, 1)` -> Data.

### Lab 3: String Lists
- **Goal:** Parse a list of strings.
- **Task:**
    1.  DTS: `mode-names = "fast", "slow", "turbo";`
    2.  Driver: `of_property_read_string_index(np, "mode-names", 1, &str);` -> "slow".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Invalid resource"
*   **Cause:** Parent node missing `#address-cells` or `#size-cells`. The kernel defaults to 2 and 1 respectively (usually), which might misinterpret your data.

#### 2. "GPIO lookup failed"
*   **Cause:** You requested "led", but DTS property is `led-gpio` (singular) instead of `led-gpios` (plural). The kernel supports both, but suffix matters.
*   **Cause:** Phandle to GPIO controller is wrong.

---

## ⚡ Optimization & Best Practices

### Documentation
*   **Bindings:** Before inventing a new property, check `Documentation/devicetree/bindings/`. Use standard properties (`label`, `status`, `reg`) whenever possible.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `ranges;` (empty) mean?
    *   **A:** It means the child bus addresses are identical to the parent bus addresses (1:1 mapping).
2.  **Q:** Why do we need `interrupt-parent`?
    *   **A:** A board might have multiple interrupt controllers (Main GIC, GPIO Controller, PMIC). We need to specify which one handles this device's IRQ line.

### Challenge Task
> **Task:** "The Nested Bus".
> *   Create a Grandparent Bus (0x80000000).
> *   Create a Parent Bus (0x1000 offset).
> *   Create a Child Device (0x10 offset).
> *   Calculate the final CPU Physical Address manually and verify with the driver.

---

## 📚 Further Reading & References
- [Device Tree Specification: Properties](https://www.devicetree.org/specifications/)
- [Kernel API: of.h](https://www.kernel.org/doc/html/latest/core-api/kernel-api.html#open-firmware-and-device-tree)

---
