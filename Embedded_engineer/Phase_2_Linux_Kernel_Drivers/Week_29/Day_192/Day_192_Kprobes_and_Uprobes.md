# Day 192: Kprobes and Uprobes
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
1.  **Explain** how Kprobes (Kernel Probes) work (Breakpoints/Trampolines).
2.  **Write** a Kprobe module to intercept any kernel function.
3.  **Use** Kretprobes to inspect return values.
4.  **Understand** Uprobes for userspace tracing.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel with `CONFIG_KPROBES`.
*   **Prior Knowledge:**
    *   Day 191 (Ftrace).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Kprobes Mechanism
*   **Concept:** Allows you to break into any kernel routine and collect information non-disruptively.
*   **How it works:**
    1.  You register a probe at an address (symbol).
    2.  Kernel copies the instruction at that address and replaces it with a Breakpoint (int3 on x86).
    3.  When CPU hits breakpoint, trap handler calls your `pre_handler`.
    4.  Kernel single-steps the original instruction.
    5.  Kernel calls your `post_handler` (optional).
    6.  Execution continues.

### 🔹 Part 2: Kretprobes
*   **Concept:** Probes the *return* of a function.
*   **Mechanism:**
    1.  Entry probe hijacks the return address on the stack.
    2.  Replaces it with a trampoline address.
    3.  Function runs.
    4.  Function returns to trampoline.
    5.  Trampoline calls your handler, then jumps to real return address.

---

## 💻 Implementation: A Kprobe Module

> **Instruction:** We will spy on `do_sys_open` to see every file opened by the system.

### 👨‍💻 Code Implementation

```c
#include <linux/kprobes.h>
#include <linux/module.h>

static struct kprobe kp = {
    .symbol_name = "do_sys_open",
};

// Called before the probed instruction is executed
static int handler_pre(struct kprobe *p, struct pt_regs *regs) {
    // On x86_64, first arg is in DI, second in SI.
    // do_sys_open(int dfd, const char __user *filename, ...)
    // We can't easily read user memory here safely without more work,
    // but we can print the process name.
    pr_info("Kprobe: Process %s (PID %d) called do_sys_open\n",
            current->comm, task_pid_nr(current));
    return 0;
}

static int my_init(void) {
    kp.pre_handler = handler_pre;
    
    int ret = register_kprobe(&kp);
    if (ret < 0) {
        pr_err("Register kprobe failed: %d\n", ret);
        return ret;
    }
    pr_info("Kprobe registered\n");
    return 0;
}

static void my_exit(void) {
    unregister_kprobe(&kp);
    pr_info("Kprobe unregistered\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: A Kretprobe Module

> **Instruction:** Spy on the return value of `do_sys_open` to see if it failed.

### 👨‍💻 Code Implementation

```c
static struct kretprobe rp = {
    .kp.symbol_name = "do_sys_open",
    .maxactive = 20, // Max concurrent probes
};

static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs) {
    int retval = regs_return_value(regs);
    if (retval < 0) {
        pr_info("Kretprobe: Open failed with error %d\n", retval);
    }
    return 0;
}

static int my_init(void) {
    rp.handler = ret_handler;
    register_kretprobe(&rp);
    return 0;
}
// ...
```

---

## 🔬 Lab Exercise: Lab 192.1 - Tracing via Sysfs (No Code)

### 1. Lab Objectives
- Use the Ftrace interface to create kprobes without writing C code.

### 2. Step-by-Step Guide
1.  **Navigate:** `cd /sys/kernel/tracing`.
2.  **Define Probe:**
    ```bash
    echo 'p:myprobe do_sys_open' > kprobe_events
    ```
3.  **Enable:**
    ```bash
    echo 1 > events/kprobes/myprobe/enable
    ```
4.  **View:**
    ```bash
    cat trace
    ```
5.  **Define Retprobe:**
    ```bash
    echo 'r:myretprobe do_sys_open $retval' > kprobe_events
    ```
    *   `$retval` fetches the return register.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Uprobes (User Space)
- **Goal:** Trace `malloc` in libc.
- **Task:**
    1.  Find offset of `malloc`: `nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep malloc`.
    2.  `echo 'p:libc_malloc /lib/.../libc.so.6:0x<offset>' > uprobe_events`.
    3.  Enable and trace.
    4.  You can see every time any program calls malloc!

### Lab 3: Jprobes (Deprecated but interesting)
- **Note:** Jprobes are removed in newer kernels (replaced by Ftrace callbacks), but understanding the concept (Jump Probes) is useful for history.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. System Freeze
*   **Cause:** Calling a function in the probe handler that triggers the probe again (Infinite Recursion).
*   **Fix:** `kprobe` handlers disable preemption and interrupts. Don't call `printk` if probing `printk`! Use `kprobe_busy()` checks or recursion protection.

#### 2. "Symbol not found"
*   **Cause:** Function is `static` or inlined.
*   **Fix:** Check `/proc/kallsyms`. If not there, you can't probe it by name (only by address, which is brittle).

---

## ⚡ Optimization & Best Practices

### eBPF (Extended Berkeley Packet Filter)
*   Writing Kprobe modules in C is "Old School".
*   Modern way: **eBPF**. You write a restricted C script, the kernel verifies it (safety), and JIT compiles it.
*   Tools: `bpftrace`, `bcc`.
*   Example: `bpftrace -e 'kprobe:do_sys_open { printf("%s\n", comm); }'` replaces the entire module we wrote above!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I change the arguments of a function using Kprobes?
    *   **A:** Yes, in `pre_handler`, you can modify `regs`. This allows "Fault Injection" (e.g., force a filename to be NULL).
2.  **Q:** Why is `kretprobe` limited by `maxactive`?
    *   **A:** It needs to allocate a "shadow stack" to store the real return addresses. If too many functions enter before returning, it runs out of slots.

### Challenge Task
> **Task:** "The Error Injector".
> *   Write a Kprobe for `my_driver_function`.
> *   In the handler, modify the instruction pointer (`regs->ip`) to skip the function body and return `-EIO` immediately.
> *   (Note: This is very advanced and architecture specific. Easier to use `kretprobe` and modify return value, but that runs *after* the function).

---

## 📚 Further Reading & References
- [Kernel Documentation: trace/kprobes.rst](https://www.kernel.org/doc/html/latest/trace/kprobes.html)
- [Brendan Gregg's BPF Tools](https://www.brendangregg.com/ebpf.html)

---
