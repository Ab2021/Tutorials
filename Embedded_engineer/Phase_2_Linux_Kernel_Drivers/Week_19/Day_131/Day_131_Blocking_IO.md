# Day 131: Blocking I/O & Wait Queues
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
1.  **Implement** Blocking I/O (Putting a process to sleep until data is available).
2.  **Use** Wait Queues (`wait_queue_head_t`) to manage sleeping processes.
3.  **Wake up** sleeping processes using `wake_up_interruptible`.
4.  **Handle** Non-Blocking I/O (`O_NONBLOCK`) requests.
5.  **Create** a FIFO (First-In-First-Out) character driver.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 130 (Concurrency/Mutex).
    *   Process States (Running, Sleeping).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Blocking vs Non-Blocking
*   **Blocking (Default):** If you call `read()` on a UART and there is no data, your process **Sleeps** (Removed from CPU run queue). It consumes **0% CPU**. When data arrives, the kernel wakes it up.
*   **Non-Blocking (`O_NONBLOCK`):** If no data, `read()` returns `-EAGAIN` immediately. The app must retry (Polling). Wastes CPU if not careful.

### 🔹 Part 2: The Wait Queue
How does the kernel know *who* to wake up?
*   **`wait_queue_head_t`:** A linked list of processes waiting for a specific event.
*   **`wait_event_interruptible(wq, condition)`:**
    1.  Checks `condition`.
    2.  If false, puts process to `TASK_INTERRUPTIBLE` state.
    3.  Adds process to `wq`.
    4.  Calls scheduler (Context Switch).
    5.  ... Later ...
    6.  Wakes up, checks `condition` again.
*   **`wake_up_interruptible(&wq)`:**
    1.  Sets state of processes in `wq` to `TASK_RUNNING`.
    2.  Scheduler eventually picks them up.

---

## 💻 Implementation: The Blocking FIFO Driver

> **Instruction:** We will create a driver that acts like a Pipe.
> *   **Reader:** Sleeps if buffer is empty.
> *   **Writer:** Wakes up reader after writing.

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`fifo_driver.c`)

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/mutex.h>
#include <linux/slab.h>

#define DEVICE_NAME "fifo_dev"
#define BUF_SIZE 1024

struct fifo_dev {
    char *buffer;
    int head; // Write pointer
    int tail; // Read pointer
    int count; // Number of bytes available
    struct mutex lock;
    wait_queue_head_t read_queue;
    struct cdev cdev;
};

static struct fifo_dev *my_dev;
static dev_t dev_num;

// --- Read Function ---
static ssize_t my_read(struct file *file, char __user *user_buf, size_t count, loff_t *offset) {
    int ret = 0;

    // 1. Lock
    if (mutex_lock_interruptible(&my_dev->lock)) return -ERESTARTSYS;

    // 2. Check Condition (Loop is crucial!)
    while (my_dev->count == 0) {
        // Unlock before sleeping!
        mutex_unlock(&my_dev->lock);

        // Check for Non-Blocking mode
        if (file->f_flags & O_NONBLOCK) return -EAGAIN;

        // Sleep
        if (wait_event_interruptible(my_dev->read_queue, (my_dev->count > 0))) {
            return -ERESTARTSYS; // Signal caught (Ctrl+C)
        }

        // Re-lock after waking up
        if (mutex_lock_interruptible(&my_dev->lock)) return -ERESTARTSYS;
    }

    // 3. Read Data
    if (count > my_dev->count) count = my_dev->count;

    if (copy_to_user(user_buf, &my_dev->buffer[my_dev->tail], count)) {
        ret = -EFAULT;
        goto out;
    }

    // 4. Update Pointers
    my_dev->tail = (my_dev->tail + count) % BUF_SIZE; // Circular buffer logic simplified for demo
    my_dev->count -= count;
    ret = count;

    printk(KERN_INFO "FIFO: Read %zu bytes. Remaining: %d\n", count, my_dev->count);

out:
    mutex_unlock(&my_dev->lock);
    return ret;
}

// --- Write Function ---
static ssize_t my_write(struct file *file, const char __user *user_buf, size_t count, loff_t *offset) {
    int ret = 0;

    if (mutex_lock_interruptible(&my_dev->lock)) return -ERESTARTSYS;

    if (count > BUF_SIZE - my_dev->count) count = BUF_SIZE - my_dev->count; // Cap to free space

    if (copy_from_user(&my_dev->buffer[my_dev->head], user_buf, count)) {
        ret = -EFAULT;
        goto out;
    }

    my_dev->head = (my_dev->head + count) % BUF_SIZE;
    my_dev->count += count;
    ret = count;

    printk(KERN_INFO "FIFO: Wrote %zu bytes. Total: %d\n", count, my_dev->count);

    // 5. Wake up Readers
    wake_up_interruptible(&my_dev->read_queue);

out:
    mutex_unlock(&my_dev->lock);
    return ret;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .read = my_read,
    .write = my_write,
    // .open and .release omitted for brevity
};

// --- Init ---
static int __init fifo_init(void) {
    alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    my_dev = kzalloc(sizeof(struct fifo_dev), GFP_KERNEL);
    
    my_dev->buffer = kmalloc(BUF_SIZE, GFP_KERNEL);
    mutex_init(&my_dev->lock);
    init_waitqueue_head(&my_dev->read_queue); // Initialize Wait Queue
    
    cdev_init(&my_dev->cdev, &fops);
    cdev_add(&my_dev->cdev, dev_num, 1);
    
    return 0;
}
// ... Exit function ...
```

---

## 💻 Implementation: Testing Blocking Behavior

> **Instruction:** We need two terminals to see blocking in action.

### 👨‍💻 Command Line Steps

#### Terminal 1 (The Reader)
```bash
cat /dev/fifo_dev
```
*   **Observation:** The command **hangs**. It is sleeping, waiting for data.
*   **Check Process State:** Open another terminal and run `ps aux | grep cat`. You will see state `S` (Interruptible Sleep) or `D` (Uninterruptible).

#### Terminal 2 (The Writer)
```bash
echo "Hello World" > /dev/fifo_dev
```
*   **Observation:** Immediately after you press Enter, Terminal 1 wakes up and prints "Hello World", then exits (or hangs again depending on `cat` behavior).

---

## 🔬 Lab Exercise: Lab 131.1 - The `poll` Implementation

### 1. Lab Objectives
- Implement the `.poll` file operation.
- This allows `select()` and `poll()` system calls to work (used by Python `select`, Node.js, etc.).

### 2. Step-by-Step Guide
1.  Add `.poll` to `fops`.
2.  Implement:
    ```c
    static __poll_t my_poll(struct file *file, poll_table *wait) {
        __poll_t mask = 0;
        
        mutex_lock(&my_dev->lock);
        poll_wait(file, &my_dev->read_queue, wait); // Register wait queue
        
        if (my_dev->count > 0)
            mask |= EPOLLIN | EPOLLRDNORM; // Readable
            
        mutex_unlock(&my_dev->lock);
        return mask;
    }
    ```
3.  Test with a Python script using `select.select()`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Exclusive Waits
- **Goal:** Solve the "Thundering Herd" problem.
- **Scenario:** 100 processes waiting. 1 byte arrives.
- **Standard:** `wake_up_interruptible` wakes ALL 100. 1 gets the byte, 99 go back to sleep. Wasteful.
- **Task:** Use `prepare_to_wait_exclusive()`. Only 1 process is woken up.

### Lab 3: Blocking Write
- **Goal:** Handle full buffer.
- **Task:**
    1.  Add `wait_queue_head_t write_queue`.
    2.  In `write()`: If buffer full, sleep on `write_queue`.
    3.  In `read()`: After consuming data, `wake_up_interruptible(&write_queue)`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Lost Wakeup"
*   **Cause:** You checked the condition, then released the lock, then slept. In between releasing and sleeping, the data arrived and `wake_up` was called. You missed it and sleep forever.
*   **Fix:** Use `wait_event_interruptible` macro which handles locking/checking atomically (conceptually).

#### 2. Process Unkillable (State D)
*   **Cause:** You used `wait_event` (Uninterruptible) instead of `wait_event_interruptible`.
*   **Result:** Ctrl+C won't kill the process. You must reboot.

---

## ⚡ Optimization & Best Practices

### Manual Sleeping
Sometimes macros aren't enough.
```c
prepare_to_wait(&wq, &wait, TASK_INTERRUPTIBLE);
if (!condition)
    schedule(); // Give up CPU
finish_wait(&wq, &wait);
```
This gives you fine-grained control.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `return -ERESTARTSYS` do?
    *   **A:** It tells the kernel "I was interrupted by a signal (like Ctrl+C). Please restart the system call if the signal handler allows, or return -EINTR to user."
2.  **Q:** Why do we need a loop `while(!condition)` around the sleep?
    *   **A:** Spurious wakeups. Sometimes a process wakes up without the condition being true (e.g., another process stole the data just before we woke up).

### Challenge Task
> **Task:** "The Chat Room". Create a driver where multiple readers can open it. When a writer writes a message, **ALL** currently open readers should receive it (Broadcast).
> *Hint: This requires a separate buffer for each reader (private_data).*

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 6: Advanced Char Drivers (Blocking I/O)](https://lwn.net/Kernel/LDD3/)
- [Kernel API: wait.h](https://www.kernel.org/doc/html/latest/core-api/kernel-api.html#wait-queues-and-wake-events)

---
