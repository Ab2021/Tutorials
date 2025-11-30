# Day 138: The Open Firmware (OF) API
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
1.  **Traverse** the Device Tree using `of_find_*` functions.
2.  **Iterate** over child nodes using `for_each_child_of_node`.
3.  **Check** device availability using `of_device_is_available`.
4.  **Parse** custom node structures (e.g., a list of LEDs or Buttons).
5.  **Manage** Reference Counting (`of_node_put`) to avoid memory leaks.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 137 (DTS Properties).
    *   Linked Lists.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is "OF"?
"Open Firmware" is the standard that defined Device Tree. In Linux, all Device Tree related functions start with `of_`.
*   **Header:** `<linux/of.h>`
*   **Structure:** `struct device_node` represents a node in the tree.

### 🔹 Part 2: Finding Nodes
Sometimes you need to find a node that isn't your child (e.g., a system-wide controller).
*   `of_find_node_by_name(from, name)`: Search by name.
*   `of_find_node_by_path(path)`: Search by full path (`/soc/gpio@...`).
*   `of_find_compatible_node(from, type, compat)`: Search by compatible string.

### 🔹 Part 3: Reference Counting
The Device Tree is a dynamic structure (due to Overlays). Nodes are reference counted.
*   **Rule:** Every `of_find_*` function increments the refcount.
*   **Rule:** You MUST call `of_node_put(node)` when you are done with it.
*   **Exception:** `devm_` functions or managed iteration macros often handle this, but be careful.

---

## 💻 Implementation: The LED Parser

> **Instruction:** We will create a driver that parses a custom "gpio-leds" style node.
> *   Parent: `my-leds`
> *   Children: `led-1`, `led-2`, etc.

### 👨‍💻 Code Implementation

#### Step 1: DTS Overlay (`leds_overlay.dts`)

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target-path = "/";
        __overlay__ {
            my_leds: my-custom-leds {
                compatible = "org,my-led-controller";
                
                red_led {
                    label = "System Fault";
                    color-id = <1>; // 1=Red
                    default-state = "off";
                };
                
                green_led {
                    label = "System OK";
                    color-id = <2>; // 2=Green
                    default-state = "on";
                };
            };
        };
    };
};
```

#### Step 2: Driver (`led_parser.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>

static int led_probe(struct platform_device *pdev) {
    struct device_node *parent = pdev->dev.of_node;
    struct device_node *child;
    const char *label;
    u32 color;
    int count = 0;

    printk("LED Parser: Probing...\n");

    // Iterate over children
    for_each_child_of_node(parent, child) {
        // Check if enabled
        if (!of_device_is_available(child)) {
            continue;
        }

        printk("--- Found Child Node: %s ---\n", child->name);

        // Read Label
        if (of_property_read_string(child, "label", &label) == 0) {
            printk("  Label: %s\n", label);
        }

        // Read Color
        if (of_property_read_u32(child, "color-id", &color) == 0) {
            printk("  Color ID: %d\n", color);
        }
        
        count++;
    }

    printk("LED Parser: Found %d LEDs\n", count);
    return 0;
}

static const struct of_device_id led_dt_ids[] = {
    { .compatible = "org,my-led-controller", },
    { }
};
MODULE_DEVICE_TABLE(of, led_dt_ids);

static struct platform_driver led_driver = {
    .probe = led_probe,
    .driver = {
        .name = "my-led-controller",
        .of_match_table = led_dt_ids,
    }
};

module_platform_driver(led_driver); // Macro for init/exit
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Searching for Nodes

> **Instruction:** Find a node by path from a completely unrelated driver.

### 👨‍💻 Code Implementation

```c
void find_the_leds(void) {
    struct device_node *np;
    
    // 1. Find by Path
    np = of_find_node_by_path("/my-custom-leds");
    if (!np) {
        printk("Node not found!\n");
        return;
    }
    
    printk("Found node: %s\n", np->full_name);
    
    // 2. Do something...
    
    // 3. Release Reference (CRITICAL!)
    of_node_put(np);
}
```

---

## 🔬 Lab Exercise: Lab 138.1 - Count Enabled Nodes

### 1. Lab Objectives
- Modify the DTS to add `status = "disabled";` to the Red LED.
- Verify the driver ignores it.

### 2. Step-by-Step Guide
1.  Update DTS:
    ```dts
    red_led {
        ...
        status = "disabled";
    };
    ```
2.  Recompile and Apply.
3.  Check dmesg. `count` should be 1.
4.  Remove `if (!of_device_is_available(child))` from driver.
5.  Check dmesg. `count` should be 2.

---

## 🧪 Additional / Advanced Labs

### Lab 2: `of_get_next_child`
- **Goal:** Understand manual iteration.
- **Task:** Re-implement `for_each_child_of_node` using `of_get_next_child`.
    ```c
    child = NULL;
    while ((child = of_get_next_child(parent, child)) != NULL) {
        // ...
    }
    ```
    *Note: `of_get_next_child` automatically handles refcounting (puts previous, gets next).*

### Lab 3: Matching Data
- **Goal:** Use `of_device_get_match_data`.
- **Task:**
    1.  In `of_device_id` table, set `.data = (void *)SOME_CONFIG_STRUCT`.
    2.  In `probe`, retrieve it: `data = of_device_get_match_data(&pdev->dev);`.
    3.  Useful for supporting multiple hardware revisions with one driver.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Memory Leaks
*   **Cause:** Breaking out of `for_each_child_of_node` without putting the child.
*   **Fix:**
    ```c
    for_each_child_of_node(parent, child) {
        if (error) {
            of_node_put(child); // MUST DO THIS
            return error;
        }
    }
    ```

#### 2. Null Pointer Dereference
*   **Cause:** Assuming `pdev->dev.of_node` is not NULL.
*   **Fix:** Always check if `of_node` exists (it might be NULL if the device was created manually via `platform_device_register`).

---

## ⚡ Optimization & Best Practices

### Scoped Iterators (Kernel 6.x+)
*   Modern kernels introduce `for_each_child_of_node_scoped`.
*   It uses `__cleanup` attribute (RAII) to automatically call `of_node_put` when the variable goes out of scope.
*   **Recommendation:** Use it if your kernel version supports it.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `of_node_put` do?
    *   **A:** Decrements the reference count of the node. If it hits zero, the kernel frees the memory associated with that node structure.
2.  **Q:** Can I modify the Device Tree from the driver code?
    *   **A:** Generally NO. The DT is treated as Read-Only configuration. If you need to change hardware state, use the appropriate subsystem APIs (GPIO, Clock, Regulator), not raw DT manipulation.

### Challenge Task
> **Task:** "The Dependency Finder".
> *   DTS: Node A has `my-dep = <&node_b>;`.
> *   Driver:
>     1.  Find Node A.
>     2.  Parse `my-dep` to get the phandle.
>     3.  Resolve phandle to `struct device_node *` of Node B.
>     4.  Print the name of Node B.

---

## 📚 Further Reading & References
- [Kernel API: of.h](https://www.kernel.org/doc/html/latest/core-api/kernel-api.html#open-firmware-and-device-tree)
- [LWN: Device Tree Overlays](https://lwn.net/Articles/448502/)

---
