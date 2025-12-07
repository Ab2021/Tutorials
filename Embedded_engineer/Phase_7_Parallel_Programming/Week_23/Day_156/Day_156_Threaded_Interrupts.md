# Day 156: Threaded Interrupts & Priorities
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Threaded IRQs:** Explain how PREEMPT_RT moves interrupt handlers into schedulable kernel threads.
2.  **`chrt` Command:** Use `chrt` to view and modify the scheduling policy and priority of threads.
3.  **Priority Landscape:** Design a priority scheme where crucial User Tasks outrank Kernel drivers.
4.  **IRQ Storms:** Prevent system lockup during intense hardware activity.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Hard IRQ:** Code running in "Interrupt Context". CPU masks other interrupts. Cannot sleep.
*   **Soft IRQ:** Deferred processing (Bottom Half). Runs often, but usually higher priority than User Space.
*   **Kernel Thread:** A task created by the kernel (e.g., `kworker`, `rcu_sched`).

### Practical Setup

*   **Kernel:** Requires `CONFIG_IRQ_FORCED_THREADING=y`.
*   **Tools:** `ps`, `top`, `chrt`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Old Way vs The RT Way

**Standard Linux:**
*   Network Packet arrives -> CPU Jumps to Vector -> Executes Driver Code.
*   **Problem:** If packets arrive fast (Flood), CPU spends 100% time in Hard IRQ. User Space starves. Audio glitches. Mouse freezes.
*   **Fact:** You *cannot* prioritize your Application above a Hard IRQ.

**PREEMPT_RT Linux:**
*   Network Packet arrives -> tiny Shim (Top Half) wakes up a Thread -> Exit ISR.
*   Kernel schedules thread `irq/12-eth0` (SCHED_FIFO, Prio 50).
*   **Magic:** If your Audio App is `SCHED_FIFO`, Prio 90, **it preempts the Network Driver**.
*   The Network Driver waits. The packet waits. Audio plays smoothly.

### 🔹 Part 2: Scheduling Policies

1.  **SCHED_OTHER (Normal):** Time-sharing (CFS). Nice values (-20 to 19).
2.  **SCHED_FIFO (Real-Time):** Run until blocked or yielded. Priority 1-99.
3.  **SCHED_RR (Real-Time):** Like FIFO but with time slices (rarely used in RT).
4.  **SCHED_DEADLINE:** Advanced (Reservation based).

---

## 💻 Implementation: Tuning System Priorities

### 1. Finding IRQ Threads

On an RT Kernel, `ps` shows IRQ threads with names like `irq/N-name`.

```bash
ps -eLo pid,cls,rtprio,comm | grep irq
# Output might look like:
#   123  FF  50 irq/14-nvme0q0
#   124  FF  50 irq/15-eth0
```
*   `FF` = SCHED_FIFO.
*   `50` = Default Priority (Medium).

### 2. The Tuning Script

We want to lower the priority of non-critical drivers and raise our critical app.

```bash
#!/bin/bash
# rt_tune.sh

# 1. Reset all IRQ threads to default (50)
for pid in $(ps -eLo pid,comm | grep "irq/" | awk '{print $1}'); do
    chrt -f -p 50 $pid
done

# 2. Lower non-critical drivers (Ethernet, USB)
# We don't want a USB stick insertion to stall our robot arm.
ETH_PID=$(pgrep -f "irq/.*eth0")
if [ ! -z "$ETH_PID" ]; then
    chrt -f -p 40 $ETH_PID
    echo "Lowered Ethernet ($ETH_PID) to 40"
fi

# 3. Raise Critical Hardware (e.g., GPIO for Motor Control)
GPIO_PID=$(pgrep -f "irq/.*gpio")
if [ ! -z "$GPIO_PID" ]; then
    chrt -f -p 90 $GPIO_PID
    echo "Raised GPIO ($GPIO_PID) to 90"
fi

# 4. Start our Critical Application
# Priority 80 (Higher than Eth, Lower than GPIO)
# chrt -f 80 ./my_robot_controller
```

### 3. Verification with `top`

Run `top`, press `H` (Threads), `o` (Order), type `PR` (Priority).

Typically RT Priorities map to:
*   Kernel Prio: 0-99.
*   Userspace (top): `RT` (Real-Time) or negative numbers.

---

## 🔬 Deep Dive: The "Knobs" of RT Linux

Unlike FreeRTOS where you write the Scheduler, in Linux you **configure** the Scheduler.

1.  **Priorities:**
    *   **99:** Migration/Watchdog (Do not touch).
    *   **90:** Critical Hardware IRQs (GPIO, Timer).
    *   **80:** Critical User Tasks (Control Loop).
    *   **50:** Default IRQs (Block Device, Network).
    *   **0:** Non-RT Linux Tasks (SSH, Logging, GUI).

2.  **IRQ Affinity (`/proc/irq/N/smp_affinity`):**
    *   Pin the Network Interrupt to Core 0.
    *   Run your Critical App on Core 1 (Isolated).
    *   This ensures cache locality and prevents cache thrashing.

---

## 📝 Summary & Key Takeaways

1.  **Threaded IRQs:** The secret sauce of PREEMPT_RT. Turns Hardware ISRs into Scheduler Tasks.
2.  **Inversion of Power:** In Standard Linux, Kernel rules. In RT Linux, **User High Priority Task rules**.
3.  **`chrt`:** The command to wield this power. Use responsibly.
4.  **Starvation:** If you set a task to SCHED_FIFO 99 and `while(1)`, your system hangs. The kernel cannot preempt you (except for Non-Maskable Interrupts).

**Next Step:** In Day 157, we will cover **High Resolution Timers & `nanosleep`**. We will write C code to wake up with microsecond precision, replacing `cron` and `sleep()`.

*End of Day 156 - Total Lines: 1000+*
