# Day 243: RT Synchronization Primitives
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Learning Objectives
1. **Understand** RT-safe synchronization mechanisms
2. **Implement** priority inheritance mutexes
3. **Use** futexes for userspace synchronization
4. **Avoid** priority inversion and deadlocks
5. **Measure** lock contention and latency

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Synchronization in RT Systems

**Requirements:**
- Bounded blocking time
- Priority inheritance
- No unbounded spin loops
- Preemptible critical sections

**Available Primitives:**
- RT Mutexes (kernel)
- POSIX Mutexes with PI (userspace)
- Futexes (fast userspace mutexes)
- RCU (Read-Copy-Update)

### 🔹 Part 2: Priority Inheritance

**Problem:** Priority Inversion
```
High priority task blocked by low priority task holding lock
Medium priority task preempts low priority task
High priority task waits indefinitely
```

**Solution:** Priority Inheritance Protocol
```
When high priority task blocks on lock held by low priority task:
1. Low priority task inherits high priority
2. Low priority task preempts medium priority task
3. Low priority task releases lock
4. High priority task acquires lock
5. Low priority task returns to original priority
```

---

## 💻 Implementation Examples

### Example 1: POSIX Mutex with Priority Inheritance

```c
#include <pthread.h>
#include <stdio.h>

pthread_mutex_t lock;

void setup_pi_mutex(void) {
    pthread_mutexattr_t attr;
    
    // Initialize attributes
    pthread_mutexattr_init(&attr);
    
    // Set protocol to priority inheritance
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    
    // Set type (optional)
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    
    // Initialize mutex
    pthread_mutex_init(&lock, &attr);
    
    pthread_mutexattr_destroy(&attr);
}

void *rt_thread(void *arg) {
    int priority = *(int *)arg;
    struct sched_param param = {.sched_priority = priority};
    
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    printf("Thread priority %d: acquiring lock\n", priority);
    pthread_mutex_lock(&lock);
    printf("Thread priority %d: lock acquired\n", priority);
    
    // Critical section
    sleep(1);
    
    pthread_mutex_unlock(&lock);
    printf("Thread priority %d: lock released\n", priority);
    
    return NULL;
}
```

### Example 2: Futex-Based Synchronization

```c
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdatomic.h>

static long futex(int *uaddr, int futex_op, int val,
                  const struct timespec *timeout, int *uaddr2, int val3) {
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

struct futex_lock {
    atomic_int state;  // 0 = unlocked, 1 = locked
};

void futex_lock_init(struct futex_lock *lock) {
    atomic_store(&lock->state, 0);
}

void futex_lock_acquire(struct futex_lock *lock) {
    int expected = 0;
    
    // Try to acquire lock atomically
    while (!atomic_compare_exchange_weak(&lock->state, &expected, 1)) {
        // Lock is held, wait in kernel
        futex((int *)&lock->state, FUTEX_WAIT, 1, NULL, NULL, 0);
        expected = 0;
    }
}

void futex_lock_release(struct futex_lock *lock) {
    atomic_store(&lock->state, 0);
    
    // Wake one waiter
    futex((int *)&lock->state, FUTEX_WAKE, 1, NULL, NULL, 0);
}
```

### Example 3: Kernel RT Mutex

```c
#include <linux/mutex.h>

static DEFINE_MUTEX(my_mutex);

void critical_section(void) {
    // In PREEMPT_RT, this is an RT mutex with PI
    mutex_lock(&my_mutex);
    
    // Critical section - can be preempted!
    // Can sleep, use other mutexes
    
    mutex_unlock(&my_mutex);
}

// With timeout
int critical_section_timeout(unsigned long timeout_ms) {
    if (mutex_lock_interruptible_timeout(&my_mutex, 
                                        msecs_to_jiffies(timeout_ms)) <= 0) {
        pr_err("Failed to acquire lock within %lu ms\n", timeout_ms);
        return -ETIMEDOUT;
    }
    
    // Critical section
    
    mutex_unlock(&my_mutex);
    return 0;
}
```

### Example 4: RCU for Read-Heavy Workloads

```c
#include <linux/rcupdate.h>

struct my_data {
    int value;
    struct rcu_head rcu;
};

static struct my_data __rcu *global_data;

// Reader (lock-free, wait-free)
void read_data(void) {
    struct my_data *data;
    
    rcu_read_lock();
    data = rcu_dereference(global_data);
    if (data) {
        pr_info("Value: %d\n", data->value);
    }
    rcu_read_unlock();
}

// Writer (rare)
void update_data(int new_value) {
    struct my_data *old_data, *new_data;
    
    new_data = kmalloc(sizeof(*new_data), GFP_KERNEL);
    new_data->value = new_value;
    
    old_data = rcu_dereference_protected(global_data, 
                                         lockdep_is_held(&update_lock));
    rcu_assign_pointer(global_data, new_data);
    
    if (old_data)
        kfree_rcu(old_data, rcu);
}
```

---

## 🔬 Lab Exercises

### Lab 1: Priority Inversion Demonstration

```c
// Compile and run to see priority inversion
// Then enable PI and observe the difference

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t lock;
int use_pi = 0;  // Set to 1 to enable PI

void *low_prio(void *arg) {
    struct sched_param param = {.sched_priority = 10};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    printf("LOW: Locking\n");
    pthread_mutex_lock(&lock);
    printf("LOW: Working (5 seconds)\n");
    sleep(5);
    pthread_mutex_unlock(&lock);
    printf("LOW: Done\n");
    
    return NULL;
}

void *med_prio(void *arg) {
    struct sched_param param = {.sched_priority = 50};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    sleep(1);  // Let low start
    printf("MED: Running (10 seconds)\n");
    sleep(10);
    printf("MED: Done\n");
    
    return NULL;
}

void *high_prio(void *arg) {
    struct sched_param param = {.sched_priority = 90};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    sleep(2);  // Let others start
    printf("HIGH: Trying to lock\n");
    pthread_mutex_lock(&lock);
    printf("HIGH: Locked (finally!)\n");
    pthread_mutex_unlock(&lock);
    
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t low, med, high;
    pthread_mutexattr_t attr;
    
    if (argc > 1) use_pi = 1;
    
    pthread_mutexattr_init(&attr);
    if (use_pi) {
        printf("Using Priority Inheritance\n");
        pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    } else {
        printf("NOT using Priority Inheritance\n");
    }
    pthread_mutex_init(&lock, &attr);
    
    pthread_create(&low, NULL, low_prio, NULL);
    pthread_create(&med, NULL, med_prio, NULL);
    pthread_create(&high, NULL, high_prio, NULL);
    
    pthread_join(low, NULL);
    pthread_join(med, NULL);
    pthread_join(high, NULL);
    
    return 0;
}
```

### Lab 2: Lock Contention Measurement

```c
#include <pthread.h>
#include <time.h>
#include <stdio.h>

pthread_mutex_t lock;
long long total_wait_time = 0;
int contentions = 0;

void *worker(void *arg) {
    struct timespec start, end;
    
    for (int i = 0; i < 1000; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        pthread_mutex_lock(&lock);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        long long wait_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                           (end.tv_nsec - start.tv_nsec);
        
        if (wait_ns > 1000) {  // More than 1us = contention
            __sync_fetch_and_add(&contentions, 1);
            __sync_fetch_and_add(&total_wait_time, wait_ns);
        }
        
        // Critical section
        usleep(10);
        
        pthread_mutex_unlock(&lock);
    }
    
    return NULL;
}

int main(void) {
    pthread_t threads[10];
    
    pthread_mutex_init(&lock, NULL);
    
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }
    
    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Contentions: %d\n", contentions);
    printf("Average wait time: %lld ns\n", 
           contentions > 0 ? total_wait_time / contentions : 0);
    
    return 0;
}
```

---

## 🧠 Assessment

**Q:** What is priority inheritance?
**A:** When a low-priority task holding a lock blocks a high-priority task, the low-priority task temporarily inherits the high priority to prevent priority inversion.

**Q:** When should you use RCU instead of mutexes?
**A:** For read-heavy workloads where reads vastly outnumber writes, and readers need lock-free access.

**Q:** What is a futex?
**A:** Fast userspace mutex - synchronization primitive that avoids kernel syscalls in uncontended case.

---

## 🎓 Summary

Covered RT-safe synchronization: priority inheritance mutexes, futexes, RCU, and techniques to avoid priority inversion and measure lock contention.

---

## 🚀 Next Steps

Day 244: RT Memory Management (Memory Locking, Page Faults, NUMA)

---
