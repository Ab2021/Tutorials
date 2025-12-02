# Day 37: Real-Time OS (RTOS) Concepts
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 37 Focus:**
> A general-purpose OS (Linux/Windows) focuses on *throughput* (doing as much as possible). An RTOS focuses on *determinism* (doing it on time). For an airbag or a brake controller, "fast enough" isn't good enough. It must be **on time, every time**. Today, we master **FreeRTOS**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between Hard Real-Time (Airbag) and Soft Real-Time (Video Streaming).
2.  **Explain** Preemptive Scheduling, Context Switching, and Task Priorities.
3.  **Implement** Tasks in FreeRTOS and manage their priorities.
4.  **Synchronize** tasks using Semaphores and Mutexes (and avoid Priority Inversion).
5.  **Communicate** between tasks using Queues.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **C/C++:** Pointers, Structs.
-   **Concurrency:** Threads, Race Conditions.

### Hardware Requirements
-   **Microcontroller:** ESP32, STM32, or Arduino (optional).
-   **Simulation:** We will use the **FreeRTOS POSIX Port** to run on Linux/Windows.

### Software Stack
-   **FreeRTOS:** We will simulate it or use a library wrapper.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is Real-Time?

**Real-Time $\neq$ Fast.**
Real-Time means **Deterministic**. The system guarantees a response within a specific deadline.
-   **Hard Real-Time:** Missing a deadline is a total system failure (e.g., ABS Brakes).
-   **Soft Real-Time:** Missing a deadline degrades quality (e.g., Dropped frame in Netflix).

### 🔹 Part 2: The Scheduler

The heart of the RTOS. It decides which task runs on the CPU.
-   **Preemptive:** High priority task *immediately* interrupts a low priority task.
-   **Round Robin:** Tasks of equal priority share time slices.
-   **Context Switch:** Saving the state (Registers, Stack) of Task A and loading Task B. Takes time (~10us).

### 🔹 Part 3: IPC (Inter-Process Communication)

Tasks need to talk. Global variables are dangerous (Race Conditions).

#### 3.1 Queues
Thread-safe FIFO buffers. Copy by value.
-   Task A sends data to Queue.
-   Task B blocks (sleeps) until data arrives.

#### 3.2 Semaphores
Signaling mechanism.
-   **Binary:** Like a flag (0 or 1). Used for synchronization (ISR -> Task).
-   **Counting:** Counts resources (e.g., 5 empty slots in a buffer).

#### 3.3 Mutexes
Mutual Exclusion. Used to protect shared resources (e.g., I2C bus).
-   **Priority Inversion:** Low priority task holds Mutex. High priority task waits. Medium priority task preempts Low. High is blocked by Medium!
-   **Solution:** **Priority Inheritance**. Low inherits High's priority while holding the Mutex.

---

## 💻 Implementation: FreeRTOS Simulation

We will use a mock structure that mimics FreeRTOS syntax to demonstrate the logic in standard C++ (since running actual FreeRTOS requires specific kernel headers/porting).
*Note: If you have an ESP32, you can run this directly.*

### 🛠️ Setup
Create `week6_day37` and `rtos_demo.cpp`.

```bash
mkdir -p ~/ros2_ws/src/week6_day37
cd ~/ros2_ws/src/week6_day37
touch rtos_demo.cpp
```

### 👨‍💻 Code: RTOS Logic (Simulated with std::thread)

We simulate FreeRTOS behavior using C++ `std::thread` and `std::mutex` to teach the concepts without needing cross-compilation.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <vector>
#include <string>

// --- FreeRTOS Abstraction Layer (Simulation) ---

// 1. Queue Wrapper
template <typename T>
class Queue {
public:
    void send(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(item);
        cv_.notify_one(); // Wake up receiver
    }

    T receive() {
        std::unique_lock<std::mutex> lock(mutex_);
        // Block until data is available
        cv_.wait(lock, [this] { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        return item;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// 2. Mutex Wrapper
class Mutex {
public:
    void take() {
        mutex_.lock();
    }
    void give() {
        mutex_.unlock();
    }
private:
    std::mutex mutex_;
};

// --- Application Code ---

struct SensorData {
    int id;
    float value;
};

// Global Objects
Queue<SensorData> sensorQueue;
Mutex serialMutex;

// Task 1: Sensor Producer (High Priority conceptually)
void vTaskSensor(int id) {
    while (true) {
        // Simulate reading hardware
        float val = (float)rand() / RAND_MAX * 100.0;
        
        SensorData data = {id, val};
        
        // Protect Serial Output
        serialMutex.take();
        std::cout << "[Sensor " << id << "] Read: " << val << " -> Sending to Queue\n";
        serialMutex.give();
        
        // Send to Queue
        sensorQueue.send(data);
        
        // vTaskDelay (Sleep)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// Task 2: Processing Consumer (Lower Priority)
void vTaskProcess() {
    while (true) {
        // Block until data arrives
        SensorData data = sensorQueue.receive();
        
        // Process
        serialMutex.take();
        std::cout << "[Process] Received from Sensor " << data.id 
                  << ". Value: " << data.value << "\n";
        serialMutex.give();
        
        // Simulate heavy computation
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int main() {
    std::cout << "--- RTOS Simulation Started ---\n";
    
    // Create Tasks
    std::thread t1(vTaskSensor, 1);
    std::thread t2(vTaskSensor, 2);
    std::thread t3(vTaskProcess);
    
    // Start Scheduler (Join threads)
    t1.join();
    t2.join();
    t3.join();
    
    return 0;
}
```

### 👨‍💻 Code: Actual FreeRTOS Syntax (Reference)

If you were running on an ESP32, the code would look like this:

```cpp
/*
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

QueueHandle_t xQueue;

void vTaskSensor(void *pvParameters) {
    int id = (int)pvParameters;
    SensorData data;
    data.id = id;
    
    while(1) {
        data.value = analogRead(0);
        xQueueSend(xQueue, &data, portMAX_DELAY);
        vTaskDelay(500 / portTICK_PERIOD_MS);
    }
}

void vTaskProcess(void *pvParameters) {
    SensorData data;
    while(1) {
        if(xQueueReceive(xQueue, &data, portMAX_DELAY)) {
            printf("Got data: %f\n", data.value);
        }
    }
}

void app_main() {
    xQueue = xQueueCreate(10, sizeof(SensorData));
    xTaskCreate(vTaskSensor, "Sensor1", 2048, (void*)1, 2, NULL);
    xTaskCreate(vTaskProcess, "Process", 2048, NULL, 1, NULL);
}
*/
```

---

## 🔬 Lab Exercise: Priority Inversion

### Lab Objectives
1.  **Concept:** Create 3 tasks: Low, Medium, High.
2.  **Scenario:**
    -   Low takes Mutex.
    -   High preempts Low, tries to take Mutex, Blocks.
    -   Medium preempts Low (because Med > Low).
    -   **Result:** High is waiting for Low, but Low is blocked by Medium. High is effectively lower priority than Medium!
3.  **Fix:** Use `std::mutex` (which often implements Priority Inheritance in OS) or simulate the fix by boosting Low's priority manually.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stack Overflow
**Symptom:** Random crashes or memory corruption.
**Cause:** Task stack size too small (local variables exceeded limit).
**Solution:** Increase stack size in `xTaskCreate`. Use `uxTaskGetStackHighWaterMark()` to check usage.

#### 2. Deadlock
**Symptom:** System hangs.
**Cause:** Task A holds Mutex 1, waits for Mutex 2. Task B holds Mutex 2, waits for Mutex 1.
**Solution:** Always acquire mutexes in the same order. Use timeouts (`xSemaphoreTake(..., 100ms)`).

#### 3. Starvation
**Symptom:** Low priority task never runs.
**Cause:** High priority tasks are always busy (no `vTaskDelay`).
**Solution:** Ensure high priority tasks sleep or yield.

---

## ⚡ Optimization & Best Practices

### 1. Interrupt Service Routines (ISRs)
Keep ISRs short!
-   Don't do math in ISR.
-   Don't print in ISR.
-   **Pattern:** ISR gives a Semaphore. A High Priority Task takes the Semaphore and does the work ("Deferred Interrupt Processing").

### 2. Zero Copy Queues
Copying large structs into queues is slow.
-   **Solution:** Send *pointers* to the data in the queue.
-   *Warning:* Ensure the data remains valid (use static buffers or memory pools).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a Mutex and a Binary Semaphore?
    *   **A:** A Mutex has ownership (only the taker can give it back) and priority inheritance. A Binary Semaphore has no ownership (ISR can give, Task can take).
2.  **Q:** Why is `malloc` bad in Real-Time systems?
    *   **A:** It is non-deterministic (fragmentation search time varies). Use Static Allocation or Memory Pools.
3.  **Q:** What is "Jitter"?
    *   **A:** The variation in the delay of a task execution. (e.g., Task runs every 10ms +/- 1ms).

### Challenge Task
**Task:** Rate Monotonic Scheduling (RMS).
1.  Create 3 tasks with periods: 100ms, 200ms, 500ms.
2.  Assign priorities according to RMS: Shorter period = Higher priority.
3.  Verify that all deadlines are met.

---

## 📚 Further Reading & References
-   [Mastering the FreeRTOS Real Time Kernel (Official Book)](https://www.freertos.org/Documentation/RTOS_book.html)
-   [RTOS Concepts (Embedded.com)](https://www.embedded.com/)

---

**Day 37 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
