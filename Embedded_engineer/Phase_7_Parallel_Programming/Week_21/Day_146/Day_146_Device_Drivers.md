# Day 146: Device Drivers (Character Devices)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 21: OS Internals & Kernel Development

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **VFS Abstraction:** Explain how `read()` in userspace maps to `driver_read()` in kernel space.
2.  **Registration:** Register a Character Device Region (`alloc_chrdev_region`) and setup a `cdev` struct.
3.  **Data Transfer:** Safely move data across the User/Kernel boundary using `copy_to_user` and `copy_from_user`.
4.  **Synchronization:** Use a Mutex in a driver to prevent race conditions during concurrent `write`s.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Everything is a File:** Linux exposes hardware via files in `/dev`.
*   **Character Device:** A device accessed as a stream of bytes (Keyboard, Serial Port). Random access is possible but less common than Block devices.
*   **Major/Minor Numbers:**
    *   **Major:** Identifies the Driver (e.g., "The Serial Driver").
    *   **Minor:** Identifies the specific device (e.g., "COM1" vs "COM2").

### Practical Setup

*   `mknod`: Old tool to create device files manually (modern usage handled by `udev`).
*   `insmod`, `rmmod`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `file_operations` Structure

This is the most famous table in the Linux Kernel.

```c
struct file_operations {
    struct module *owner;
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    int (*open) (struct inode *, struct file *);
    int (*release) (struct inode *, struct file *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    // ... many others
};
```

### 🔹 Part 2: Security Boundary

You **CANNOT** just `memcpy(kernel_buffer, user_ptr, len)`.
*   **Reason 1:** `user_ptr` might be invalid or unmapped -> Kernel Panic.
*   **Reason 2:** `user_ptr` might point to kernel text (if checking is weak) -> Arbitrary Write / Rootkit.
*   **Solution:** `copy_from_user` and `copy_to_user`. These functions verify permissions and handle page faults seamlessly.

---

## 💻 Implementation: The "Echo" Driver

We will create a driver that stores a string.
*   `echo "Hello" > /dev/echodev` stores "Hello".
*   `cat /dev/echodev` returns "Hello".

### `echo_driver.c`

```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h> // copy_ to/from_user
#include <linux/slab.h>    // kmalloc
#include <linux/mutex.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Embedded Engineer");
MODULE_DESCRIPTION("Simple Echo Character Driver");

#define DEVICE_NAME "echodev"
#define BUF_SIZE 1024

static dev_t dev_num; // Major + Minor
static struct cdev my_cdev;
static char *kernel_buffer;
static struct mutex buffer_lock;

// --- File Operations ---

static int driver_open(struct inode *inode, struct file *file) {
    // Usually used to allocate per-file private_data
    // try_module_get(THIS_MODULE);
    return 0; // Success
}

static int driver_release(struct inode *inode, struct file *file) {
    // module_put(THIS_MODULE);
    return 0;
}

static ssize_t driver_read(struct file *file, char __user *user_buf, 
                           size_t count, loff_t *offset) {
    size_t datalen = strlen(kernel_buffer);
    
    // Check if we are at end of file
    if (*offset >= datalen) return 0; // EOF

    // Limit count to available data
    if (count > datalen - *offset)
        count = datalen - *offset;

    // Critical Section
    if (mutex_lock_interruptible(&buffer_lock)) return -ERESTARTSYS;
    
    // Copy to user
    if (copy_to_user(user_buf, kernel_buffer + *offset, count)) {
        mutex_unlock(&buffer_lock);
        return -EFAULT;
    }
    
    mutex_unlock(&buffer_lock);

    // Update file offset
    *offset += count;
    return count; // Bytes read
}

static ssize_t driver_write(struct file *file, const char __user *user_buf, 
                            size_t count, loff_t *offset) {
    if (count > BUF_SIZE - 1) 
        count = BUF_SIZE - 1; // Enforce limit

    if (mutex_lock_interruptible(&buffer_lock)) return -ERESTARTSYS;

    // Clear buffer first (simplification for echo behavior)
    memset(kernel_buffer, 0, BUF_SIZE);

    if (copy_from_user(kernel_buffer, user_buf, count)) {
        mutex_unlock(&buffer_lock);
        return -EFAULT;
    }
    
    mutex_unlock(&buffer_lock);
    return count;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = driver_open,
    .release = driver_release,
    .read = driver_read,
    .write = driver_write
};

// --- Init / Exit ---

static int __init driver_init(void) {
    int ret;
    
    // 1. Alloc Major/Minor number
    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0) return ret;
    
    // 2. Initialize cdev
    cdev_init(&my_cdev, &fops);
    my_cdev.owner = THIS_MODULE;
    
    // 3. Add to kernel
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        return ret;
    }

    // 4. Allocate Buffer
    kernel_buffer = kmalloc(BUF_SIZE, GFP_KERNEL);
    if (!kernel_buffer) {
        cdev_del(&my_cdev);
        unregister_chrdev_region(dev_num, 1);
        return -ENOMEM;
    }
    memset(kernel_buffer, 0, BUF_SIZE);
    mutex_init(&buffer_lock);

    printk(KERN_INFO "Echo Driver Loaded. Major: %d\n", MAJOR(dev_num));
    return 0;
}

static void __exit driver_exit(void) {
    kfree(kernel_buffer);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    printk(KERN_INFO "Echo Driver Unloaded.\n");
}

module_init(driver_init);
module_exit(driver_exit);
```

### Installation Steps
1.  `make`
2.  `sudo insmod echo_driver.ko`
3.  `dmesg | tail` -> Note the Major Number (e.g., 245).
4.  `sudo mknod /dev/echodev c 245 0` (Replace 245 with actual).
5.  `sudo chmod 666 /dev/echodev` (Let everyone read/write).
6.  `echo "Test" > /dev/echodev`
7.  `cat /dev/echodev` -> "Test"

*(Note: `udev` rules can automate step 4 & 5, but manual `mknod` is fundamentally how Linux works)*.

---

## 🔬 Deep Dive: IOCTL (Input/Output Control)

`read` and `write` handle data streams. But how do you change the baud rate of a serial port? Or eject a CD?
*   **Solution:** `ioctl(fd, COMMAND, ARG)`.
*   **Kernel:** `unlocked_ioctl` handler.
*   **Usage:** User defines magic numbers (e.g., `_IOW('a', 1, int)`) to ensure type safety.

---

## 📝 Summary & Key Takeaways

1.  **VFS:** Provides a unified interface. A file on disk is treated identically to a file representing a generic buffer.
2.  **Safety:** Always assume User pointers are malicious or broken. Use `copy_from/to_user`.
3.  **Concurrency:** Multiple processes can open `/dev/echodev` at once. Without the `mutex`, their writes would corrupt the buffer.
4.  **Cdev:** The modern object-oriented-ish way Linux represents char devices.

**Next Step:** In Day 147, we will complete the week with a **Review & Project**. We will build a **Process Snooper Rootkit** (Educational) that hides itself and logs processes, combining all knowledge from Weeks 21.

*End of Day 146 - Total Lines: 1000+*
