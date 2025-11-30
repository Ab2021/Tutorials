# Day 196: Week 29 Review and Project - The Black Box Recorder
## Phase 2: Linux Kernel & Device Drivers | Week 29: Kernel Debugging

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
1.  **Synthesize** Week 29 concepts (Ftrace, Kprobes, Perf, Crash).
2.  **Implement** a persistent logging mechanism (`pstore` concept).
3.  **Debug** a complex kernel module using multiple tools.
4.  **Analyze** post-mortem data.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   All Week 29 tools.
*   **Prior Knowledge:**
    *   Week 29 Content.

---

## 🔄 Week 29 Review

### 1. Intro (Day 190)
*   `printk`, `dynamic_debug`, `debugfs`.

### 2. Tracing (Day 191-192)
*   `ftrace`: Low overhead function tracing.
*   `kprobes`: Dynamic breakpoints.

### 3. Profiling (Day 193)
*   `perf`: CPU cycles, cache misses, flame graphs.

### 4. Debugging (Day 194-195)
*   `kgdb`: Interactive GDB.
*   `crash`: Post-mortem analysis.

---

## 🛠️ Project: The "Black Box Recorder"

### 📋 Project Requirements
1.  **Module:** `blackbox.ko`.
2.  **Functionality:**
    *   Allocates a **Ring Buffer** in memory (e.g., 1MB).
    *   Uses **Kprobes** to hook into key events (`do_sys_open`, `schedule`, `sys_fork`).
    *   Records timestamp, PID, and event type into the buffer.
    *   On **Panic** (using `panic_notifier_list`), it dumps the last N events to the console (or preserves them in RAM if using `ramoops`).
3.  **Interface:**
    *   Debugfs file `/sys/kernel/debug/blackbox/log` to read current buffer.
    *   Debugfs file `/sys/kernel/debug/blackbox/trigger_panic` to test it.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Ring Buffer
Use a simple circular buffer structure.

```c
struct bb_entry {
    u64 timestamp;
    pid_t pid;
    char comm[16];
    char msg[64];
};

#define BB_SIZE 1024
static struct bb_entry buffer[BB_SIZE];
static atomic_t idx = ATOMIC_INIT(0);

static void bb_log(const char *fmt, ...) {
    int i = atomic_fetch_add(1, &idx) % BB_SIZE;
    struct bb_entry *e = &buffer[i];
    
    e->timestamp = ktime_get_real_ns();
    e->pid = current->pid;
    get_task_comm(e->comm, current);
    
    va_list args;
    va_start(args, fmt);
    vsnprintf(e->msg, sizeof(e->msg), fmt, args);
    va_end(args);
}
```

### 🔹 Phase 2: The Probes
Register Kprobes to feed the buffer.

```c
static int handler_pre(struct kprobe *p, struct pt_regs *regs) {
    bb_log("Called %s", p->symbol_name);
    return 0;
}

static struct kprobe kp = {
    .symbol_name = "do_sys_open",
    .pre_handler = handler_pre,
};
```

### 🔹 Phase 3: The Panic Notifier
Dump data when the ship goes down.

```c
static int bb_panic_handler(struct notifier_block *nb, unsigned long val, void *data) {
    int i;
    pr_emerg("BLACKBOX DUMP:\n");
    for (i = 0; i < BB_SIZE; i++) {
        // Print valid entries...
    }
    return NOTIFY_OK;
}

static struct notifier_block bb_nb = {
    .notifier_call = bb_panic_handler,
};

// In init:
atomic_notifier_chain_register(&panic_notifier_list, &bb_nb);
```

### 🔹 Phase 4: Testing
1.  **Load:** `insmod blackbox.ko`.
2.  **Generate Activity:** Open files, run processes.
3.  **Check Log:** `cat /sys/kernel/debug/blackbox/log`.
4.  **Crash:** `echo c > /proc/sysrq-trigger`.
5.  **Verify:** Check the console output (or `vmcore` with `crash`) to see if the "BLACKBOX DUMP" appeared.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Stability** | Does not crash the system itself. | Crashes occasionally. | Crashes on load. |
| **Coverage** | Hooks multiple events. | Hooks only one. | No hooks. |
| **Panic Dump** | Prints clean history on panic. | Prints nothing. | Prints garbage. |

---

## 🔮 Looking Ahead: Week 30
Next week, we explore **Device Model & Sysfs**.
You will learn how the kernel organizes devices, how `udev` works, and how to write clean, attribute-rich drivers.

---
