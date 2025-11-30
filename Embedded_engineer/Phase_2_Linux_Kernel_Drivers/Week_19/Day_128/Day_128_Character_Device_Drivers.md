# Day 128: Character Device Drivers - The Basics
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
1.  **Explain** the architecture of a Character Device Driver.
2.  **Allocate** Major and Minor numbers dynamically.
3.  **Implement** the `file_operations` structure (`open`, `release`, `read`, `write`).
4.  **Register** a Character Device (`cdev`) with the kernel.
5.  **Transfer** data between User Space and Kernel Space safely (`copy_to_user`, `copy_from_user`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Kernel Headers/Source.
    *   `gcc`, `make`.
*   **Prior Knowledge:**
    *   Day 126 (Kernel Modules).
    *   C Pointers and Structures.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Character Device?
In Linux, "Everything is a File".
*   **Character Device (`cdev`):** Accessed as a stream of bytes (like a file). Examples: Serial ports (`/dev/ttyS0`), Keyboards, Sound cards, GPIOs.
*   **Block Device (`blkdev`):** Accessed in blocks (sectors). Examples: HDD, SSD, SD Cards.
*   **Network Interface (`netif`):** Accessed via sockets. Examples: `eth0`, `wlan0`.

### 🔹 Part 2: Major and Minor Numbers
*   **Major Number:** Identifies the **Driver** (e.g., "The Serial Driver").
*   **Minor Number:** Identifies the **Device Instance** (e.g., "The first serial port").
*   **Allocation:**
    *   **Static:** Hardcoding numbers (Bad practice, leads to conflicts).
    *   **Dynamic:** `alloc_chrdev_region()` (Best practice).

### 🔹 Part 3: The `file_operations` Structure
This is the interface between the Virtual File System (VFS) and your driver.
```c
struct file_operations {
    struct module *owner;
    int (*open) (struct inode *, struct file *);
    int (*release) (struct inode *, struct file *);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    // ... many others (ioctl, mmap, poll)
};
```

### 🔹 Part 4: User-Kernel Data Transfer
You **CANNOT** just dereference a user pointer in kernel space.
*   User memory might be swapped out.
*   User memory might be read-only.
*   Security: User might pass a kernel address to trick the driver.
*   **Solution:**
    *   `copy_from_user(to, from, n)`: User -> Kernel.
    *   `copy_to_user(to, from, n)`: Kernel -> User.

---

## 💻 Implementation: The "Echo" Driver

> **Instruction:** We will create a driver that acts like a memory buffer. You write to it, and you can read it back.

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`echo_driver.c`)

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h> // For copy_to/from_user
#include <linux/slab.h>    // For kmalloc

#define DEVICE_NAME "echo_dev"
#define BUF_SIZE 1024

// Global Variables
static dev_t dev_num; // Holds Major:Minor
static struct cdev my_cdev;
static char *kernel_buffer;

// --- File Operations ---

static int my_open(struct inode *inode, struct file *file) {
    printk(KERN_INFO "EchoDev: Device opened\n");
    return 0;
}

static int my_release(struct inode *inode, struct file *file) {
    printk(KERN_INFO "EchoDev: Device closed\n");
    return 0;
}

static ssize_t my_read(struct file *file, char __user *user_buf, size_t count, loff_t *offset) {
    size_t datalen = strlen(kernel_buffer);

    if (*offset >= datalen) return 0; // EOF

    if (count > datalen - *offset) count = datalen - *offset; // Trim count

    if (copy_to_user(user_buf, kernel_buffer + *offset, count)) {
        return -EFAULT;
    }

    *offset += count;
    printk(KERN_INFO "EchoDev: Read %zu bytes\n", count);
    return count;
}

static ssize_t my_write(struct file *file, const char __user *user_buf, size_t count, loff_t *offset) {
    if (count >= BUF_SIZE) count = BUF_SIZE - 1; // Safety cap

    if (copy_from_user(kernel_buffer, user_buf, count)) {
        return -EFAULT;
    }

    kernel_buffer[count] = '\0'; // Null terminate
    *offset += count;
    printk(KERN_INFO "EchoDev: Wrote %zu bytes: %s\n", count, kernel_buffer);
    return count;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
    .read = my_read,
    .write = my_write,
};

// --- Init & Exit ---

static int __init echo_init(void) {
    int ret;

    // 1. Allocate Major/Minor
    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ERR "EchoDev: Failed to allocate major number\n");
        return ret;
    }
    printk(KERN_INFO "EchoDev: Major: %d, Minor: %d\n", MAJOR(dev_num), MINOR(dev_num));

    // 2. Initialize cdev
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;

    // 3. Add cdev to Kernel
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        printk(KERN_ERR "EchoDev: Failed to add cdev\n");
        return ret;
    }

    // 4. Allocate Buffer
    kernel_buffer = kmalloc(BUF_SIZE, GFP_KERNEL);
    if (!kernel_buffer) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev_num, 1);
        return -ENOMEM;
    }
    strcpy(kernel_buffer, "Hello from Kernel!"); // Initial content

    printk(KERN_INFO "EchoDev: Loaded successfully\n");
    return 0;
}

static void __exit echo_exit(void) {
    kfree(kernel_buffer);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    printk(KERN_INFO "EchoDev: Unloaded\n");
}

module_init(echo_init);
module_exit(echo_exit);
MODULE_LICENSE("GPL");
```

#### Step 2: Makefile
Standard module Makefile (same as Day 126).

---

## 💻 Implementation: Testing the Driver

> **Instruction:** Load the driver and interact with it using standard shell commands.

### 👨‍💻 Command Line Steps

#### Step 1: Load
```bash
insmod echo_driver.ko
dmesg | tail
# Output: EchoDev: Major: 240, Minor: 0
```
Note the Major number (e.g., 240).

#### Step 2: Create Device Node
Since we haven't implemented `udev` auto-creation yet (that's later), we must manually create the node.
```bash
mknod /dev/echo_dev c 240 0
chmod 666 /dev/echo_dev
```
*   `c`: Character device.
*   `240`: Major number (from dmesg).
*   `0`: Minor number.

#### Step 3: Read
```bash
cat /dev/echo_dev
# Output: Hello from Kernel!
```

#### Step 4: Write
```bash
echo "Embedded Linux Rocks" > /dev/echo_dev
```
Check dmesg: `EchoDev: Wrote 21 bytes: Embedded Linux Rocks`.

#### Step 5: Read Again
```bash
cat /dev/echo_dev
# Output: Embedded Linux Rocks
```

---

## 🔬 Lab Exercise: Lab 128.1 - Multiple Minors

### 1. Lab Objectives
- Modify the driver to support **2 separate buffers**.
- `/dev/echo0` -> Buffer A.
- `/dev/echo1` -> Buffer B.

### 2. Step-by-Step Guide
1.  Change `alloc_chrdev_region` to request 2 numbers.
2.  Change `cdev_add` to add 2 devices.
3.  In `open()`, check `iminor(inode)`.
    *   If 0, point `file->private_data` to Buffer A.
    *   If 1, point `file->private_data` to Buffer B.
4.  In `read/write`, use `file->private_data` instead of the global `kernel_buffer`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Automatic Device Creation (Class)
- **Goal:** Make `/dev/echo_dev` appear automatically without `mknod`.
- **Task:**
    1.  Use `class_create()` in `init`.
    2.  Use `device_create()` after `cdev_add`.
    3.  Use `device_destroy()` and `class_destroy()` in `exit`.

### Lab 3: Seek Support (`llseek`)
- **Goal:** Allow `lseek()` to work.
- **Task:** Implement `.llseek` in `fops`. Update `file->f_pos`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Bad address" (EFAULT)
*   **Cause:** You tried to access user memory directly or `copy_to_user` failed (invalid pointer passed by user).

#### 2. "Device or resource busy"
*   **Cause:** You tried to `insmod` but the Major number is taken (if static) or you forgot to `rmmod` the previous version.

#### 3. System Hang on `cat`
*   **Cause:** Your `read` function returns `count` but doesn't update `*offset`, or doesn't return 0 at EOF. `cat` keeps reading forever.

---

## ⚡ Optimization & Best Practices

### Security
- **Always validate inputs:** If user writes 1GB, don't try to `kmalloc` 1GB. Cap it.
- **Zeroing memory:** When allocating buffers that go to user space, use `kzalloc` to avoid leaking kernel data (security risk).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need `copy_from_user`? Why not `memcpy`?
    *   **A:** `memcpy` doesn't check if the address is valid in the current process's virtual address space. `copy_from_user` handles page faults and permissions.
2.  **Q:** What happens if I forget `cdev_del` in exit?
    *   **A:** The kernel might crash if someone tries to access the device after the module code is removed from memory.

### Challenge Task
> **Task:** "The Scrambler". Create a driver that takes a string, reverses it in kernel space, and returns the reversed string when read.

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 3: Char Drivers](https://lwn.net/Kernel/LDD3/)
- [Kernel API: fs.h](https://www.kernel.org/doc/html/latest/filesystems/vfs.html)

---
