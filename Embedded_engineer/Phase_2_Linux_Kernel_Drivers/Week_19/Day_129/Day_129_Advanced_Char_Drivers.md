# Day 129: Advanced Char Drivers - IOCTL
## Phase 2: Linux Kernel & Device Drivers | Week 19: Character Device Drivers

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
1.  **Explain** why `read()` and `write()` are insufficient for hardware control.
2.  **Define** custom IOCTL commands using `_IO`, `_IOR`, `_IOW`, `_IOWR` macros.
3.  **Implement** the `.unlocked_ioctl` file operation in a kernel driver.
4.  **Invoke** IOCTL commands from a User Space C application.
5.  **Control** simulated hardware (GPIO LED) using IOCTL.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 128 (Char Drivers).
    *   Bitwise operations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with Read/Write
`read()` and `write()` are for **Data Flow**.
*   **Scenario:** You have a Serial Port.
*   **Data:** The characters you send/receive.
*   **Control:** Baud rate (9600 vs 115200), Parity, Stop bits.
*   **Problem:** How do you set the baud rate? You can't "write" the number 9600 to the file, because that would be sent as data!
*   **Solution:** `ioctl` (Input/Output Control). A "side channel" for configuration.

### 🔹 Part 2: The IOCTL Command Number
An IOCTL command is a 32-bit integer, but it's structured. It's not just a random number.
It encodes:
1.  **Magic Number (8 bits):** Unique ID for your driver (to prevent conflicts).
2.  **Sequence Number (8 bits):** Command ID (0, 1, 2...).
3.  **Direction (2 bits):** Read, Write, or None.
4.  **Size (14 bits):** Size of the data argument.

### 🔹 Part 3: The Macros
Linux provides macros to generate these numbers:
*   `_IO(type, nr)`: No data transfer (e.g., RESET).
*   `_IOR(type, nr, datatype)`: Driver -> User (Read).
*   `_IOW(type, nr, datatype)`: User -> Driver (Write).
*   `_IOWR(type, nr, datatype)`: Bidirectional.

---

## 💻 Implementation: The LED Controller Driver

> **Instruction:** We will create a driver that simulates an LED. We can turn it ON, OFF, or query its status via IOCTL.

### 👨‍💻 Code Implementation

#### Step 1: Header File (`led_ioctl.h`)
This header must be shared between Kernel (Driver) and User (App).
```c
#ifndef LED_IOCTL_H
#define LED_IOCTL_H

#include <linux/ioctl.h>

#define LED_MAGIC 'k' // Magic number

// Define Commands
#define LED_ON      _IO(LED_MAGIC, 1)
#define LED_OFF     _IO(LED_MAGIC, 2)
#define LED_GET_STATUS _IOR(LED_MAGIC, 3, int)
#define LED_SET_BLINK  _IOW(LED_MAGIC, 4, int) // Pass blink delay in ms

#endif
```

#### Step 2: Driver Code (`led_driver.c`)

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include "led_ioctl.h"

#define DEVICE_NAME "led_dev"

static dev_t dev_num;
static struct cdev my_cdev;
static int led_status = 0; // 0=OFF, 1=ON
static int blink_delay = 0;

static long my_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    int val;

    switch(cmd) {
        case LED_ON:
            printk(KERN_INFO "LED Driver: LED turned ON\n");
            led_status = 1;
            break;

        case LED_OFF:
            printk(KERN_INFO "LED Driver: LED turned OFF\n");
            led_status = 0;
            break;

        case LED_GET_STATUS:
            printk(KERN_INFO "LED Driver: Reporting Status %d\n", led_status);
            if (copy_to_user((int __user *)arg, &led_status, sizeof(int))) {
                return -EFAULT;
            }
            break;

        case LED_SET_BLINK:
            if (copy_from_user(&val, (int __user *)arg, sizeof(int))) {
                return -EFAULT;
            }
            printk(KERN_INFO "LED Driver: Blink Delay set to %d ms\n", val);
            blink_delay = val;
            break;

        default:
            return -ENOTTY; // Invalid IOCTL command
    }
    return 0;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = my_ioctl, // Note: unlocked_ioctl, not ioctl
};

// ... Init and Exit functions (Same as Day 128) ...
// (Omitted for brevity, assume standard cdev registration)
```

#### Step 3: User Space App (`led_app.c`)

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "led_ioctl.h"

int main(int argc, char *argv[]) {
    int fd;
    int status;
    int delay = 500;

    fd = open("/dev/led_dev", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return -1;
    }

    printf("Turning LED ON...\n");
    ioctl(fd, LED_ON);
    sleep(1);

    printf("Checking Status...\n");
    ioctl(fd, LED_GET_STATUS, &status);
    printf("LED Status: %d\n", status);

    printf("Setting Blink Delay...\n");
    ioctl(fd, LED_SET_BLINK, &delay);

    printf("Turning LED OFF...\n");
    ioctl(fd, LED_OFF);

    close(fd);
    return 0;
}
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile everything and run on Target.

### 👨‍💻 Command Line Steps

1.  **Compile Driver:** `make` (produces `led_driver.ko`).
2.  **Compile App:** `aarch64-linux-gnu-gcc -o led_app led_app.c`.
3.  **Transfer:** Copy `.ko` and `led_app` to QEMU RootFS.
4.  **Load:** `insmod led_driver.ko`.
5.  **Create Node:** `mknod /dev/led_dev c <major> 0`.
6.  **Run:** `./led_app`.

**Expected Output:**
```text
Turning LED ON...
Checking Status...
LED Status: 1
Setting Blink Delay...
Turning LED OFF...
```
**Kernel Log (`dmesg`):**
```text
LED Driver: LED turned ON
LED Driver: Reporting Status 1
LED Driver: Blink Delay set to 500 ms
LED Driver: LED turned OFF
```

---

## 🔬 Lab Exercise: Lab 129.1 - Structure Passing

### 1. Lab Objectives
- Pass a C structure via IOCTL instead of a simple integer.
- Scenario: Configure a PWM channel (Frequency + Duty Cycle).

### 2. Step-by-Step Guide

#### Phase A: Define Struct
In `led_ioctl.h`:
```c
struct pwm_config {
    int frequency;
    int duty_cycle;
};
#define SET_PWM _IOW(LED_MAGIC, 5, struct pwm_config)
```

#### Phase B: Driver Implementation
In `my_ioctl`:
```c
struct pwm_config cfg;
case SET_PWM:
    if (copy_from_user(&cfg, (struct pwm_config __user *)arg, sizeof(cfg)))
        return -EFAULT;
    printk("PWM: Freq=%d, Duty=%d\n", cfg.frequency, cfg.duty_cycle);
    break;
```

#### Phase C: App Implementation
```c
struct pwm_config my_cfg = {1000, 50}; // 1kHz, 50%
ioctl(fd, SET_PWM, &my_cfg);
```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Capability Checks
- **Goal:** Restrict IOCTLs to root users only.
- **Task:**
    1.  Include `<linux/capability.h>`.
    2.  In `my_ioctl`, check `if (!capable(CAP_SYS_ADMIN)) return -EPERM;`.
    3.  Try running `led_app` as a non-root user. It should fail.

### Lab 3: The `compat_ioctl`
- **Goal:** Support 32-bit User Space on 64-bit Kernel.
- **Task:** Research `.compat_ioctl`. If your struct has pointers or `long` types, their size differs between 32/64 bit. You need a translation layer.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Inappropriate ioctl for device" (ENOTTY)
*   **Cause:** You called an IOCTL number that the driver doesn't handle (the `default` case in switch).
*   **Cause:** Magic number mismatch.

#### 2. "Bad address" (EFAULT)
*   **Cause:** You passed a pointer to `ioctl` but the driver treated it as a value (or vice versa).
*   **Tip:** Always verify `_IOR` vs `_IOW` definitions match how you use `arg`.

---

## ⚡ Optimization & Best Practices

### Type Safety
- Use `__user` annotation for pointers in IOCTL arguments. Sparse (static analysis tool) will catch bugs where you access user memory directly.

### Documentation
- Document your IOCTLs! In `Documentation/userspace-api/ioctl/`. If you don't document it, nobody knows how to use your driver.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is `unlocked_ioctl` called "unlocked"?
    *   **A:** In older kernels, `ioctl` held the Big Kernel Lock (BKL). `unlocked_ioctl` does not; the driver author is responsible for locking (using mutexes/spinlocks) to prevent race conditions.
2.  **Q:** Can I use `ioctl` for transferring large amounts of data (e.g., video frames)?
    *   **A:** Bad idea. `copy_from_user` is slow for MBs of data. Use `mmap` for large buffers.

### Challenge Task
> **Task:** "The Register Editor". Create a driver that maps a fake "register bank" (array of 10 integers). Use IOCTLs to:
> 1.  `REG_WRITE`: Write value V to Index I.
> 2.  `REG_READ`: Read value from Index I.
> 3.  `REG_DUMP`: Copy the whole array to user space.

---

## 📚 Further Reading & References
- [Kernel Documentation: ioctl-number.rst](https://www.kernel.org/doc/Documentation/ioctl/ioctl-number.rst)
- [Linux Device Drivers 3, Chapter 6: Advanced Char Drivers](https://lwn.net/Kernel/LDD3/)

---
