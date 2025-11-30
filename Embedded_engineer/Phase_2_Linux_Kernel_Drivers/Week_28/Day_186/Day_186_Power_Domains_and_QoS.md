# Day 186: Power Domains and QoS
## Phase 2: Linux Kernel & Device Drivers | Week 28: Power Management

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
1.  **Explain** the concept of Power Domains (GenPD).
2.  **Understand** how devices are grouped into domains.
3.  **Use** PM QoS to set constraints (e.g., "CPU must not sleep deeper than C1").
4.  **Implement** Device Links to model dependencies.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   SoC-based board (like Raspberry Pi) is better for this than a PC.
*   **Software Required:**
    *   Kernel source with `CONFIG_PM_GENERIC_DOMAINS`.
*   **Prior Knowledge:**
    *   Day 184 (Runtime PM).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Power Domains (GenPD)
*   **Problem:** On an SoC, multiple devices (e.g., Camera, ISP, VPU) often share a single power rail or clock domain. You can't turn off the rail until *all* devices are idle.
*   **Solution:** Generic Power Domains (GenPD).
    *   Drivers use Runtime PM.
    *   GenPD intercepts the calls.
    *   When all devices in the domain are suspended, GenPD turns off the physical switch.

### 🔹 Part 2: PM QoS (Quality of Service)
*   **Problem:** A driver needs low latency (e.g., Audio, Network). Deep sleep states (C-states) have high exit latency.
*   **Solution:** PM QoS allows drivers to request constraints.
    *   `cpu_dma_latency`: "I need response in < 100us".
    *   Kernel prevents CPU from entering deep C-states that violate this.

---

## 💻 Implementation: PM QoS Request

> **Instruction:** Write a module that prevents the CPU from sleeping too deeply.

### 👨‍💻 Code Implementation

```c
#include <linux/pm_qos.h>

static struct pm_qos_request my_qos_req;

static int my_probe(struct platform_device *pdev) {
    // Add a constraint: Max latency 0us (Keep CPU fully active)
    // Or a realistic value like 100us.
    cpu_latency_qos_add_request(&my_qos_req, 100);
    
    pr_info("MyQoS: Constraint added (100us)\n");
    return 0;
}

static int my_remove(struct platform_device *pdev) {
    cpu_latency_qos_remove_request(&my_qos_req);
    pr_info("MyQoS: Constraint removed\n");
    return 0;
}
```

---

## 💻 Implementation: Device Links

> **Instruction:** Model a dependency where Device A (Consumer) needs Device B (Supplier) to be active.

### 👨‍💻 Code Implementation

```c
// In Consumer Driver Probe
struct device *supplier = ...; // Find supplier device (e.g., via phandle)

// Create link
// DL_FLAG_PM_RUNTIME: When Consumer resumes, Supplier must resume first.
// DL_FLAG_AUTOREMOVE_CONSUMER: Remove link when consumer unbinds.
struct device_link *link = device_link_add(dev, supplier, 
                                           DL_FLAG_PM_RUNTIME | 
                                           DL_FLAG_AUTOREMOVE_CONSUMER);

if (!link) {
    return -EINVAL;
}
```

---

## 🔬 Lab Exercise: Lab 186.1 - QoS vs Idle States

### 1. Lab Objectives
- Check available C-states.
- Load QoS driver.
- Observe C-state usage change.

### 2. Step-by-Step Guide
1.  **Check Idle Info:**
    ```bash
    cpupower idle-info
    # Shows C-states and their exit latencies (e.g., C1: 1us, C2: 20us, C3: 100us).
    ```
2.  **Monitor:**
    ```bash
    cpupower monitor
    ```
3.  **Load Driver:**
    *   Set constraint to 10us.
4.  **Observe:**
    *   The CPU should stop entering states with latency > 10us (e.g., C2, C3).
    *   Power consumption will rise.

---

## 🧪 Additional / Advanced Labs

### Lab 2: GenPD Debugging
- **Goal:** Inspect Power Domains.
- **Task:**
    1.  `cat /sys/kernel/debug/pm_genpd/summary`.
    2.  See the hierarchy of domains and devices.
    3.  See status (on/off).

### Lab 3: Network Latency
- **Goal:** Real-world use case.
- **Task:**
    1.  Ping a device. Measure latency.
    2.  Load QoS driver (0 latency).
    3.  Ping again. Latency jitter should decrease (but power usage increases).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Constraint not met"
*   **Cause:** Hardware limitations.
*   **Fix:** Check `cpuidle` driver capabilities.

#### 2. Device Links loop
*   **Cause:** Circular dependency (A needs B, B needs A).
*   **Fix:** Kernel will warn/fail. Redesign architecture.

---

## ⚡ Optimization & Best Practices

### Dynamic QoS
*   Don't hold a constraint forever.
*   Add request in `open()`, remove in `release()`.
*   Or update value dynamically: `cpu_latency_qos_update_request(&req, new_val)`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `device_link_add` and manual `pm_runtime_get(supplier)`?
    *   **A:** Device Links are managed by the core. They handle edge cases, circular deps, and system suspend ordering automatically. Manual handling is error-prone.
2.  **Q:** Why use GenPD?
    *   **A:** It abstracts the hardware reality (shared power rails) from the drivers. Drivers just say "I need power", GenPD handles the "If A and B are idle, turn off Rail X" logic.

### Challenge Task
> **Task:** "The Latency Sensitive Audio".
> *   Write a dummy audio driver.
> *   When "playing" (simulated), set QoS to 50us.
> *   When "stopped", remove QoS.

---

## 📚 Further Reading & References
- [Kernel Documentation: power/pm_qos_interface.rst](https://www.kernel.org/doc/html/latest/power/pm_qos_interface.html)
- [Kernel Documentation: driver-api/device_link.rst](https://www.kernel.org/doc/html/latest/driver-api/device_link.html)

---
