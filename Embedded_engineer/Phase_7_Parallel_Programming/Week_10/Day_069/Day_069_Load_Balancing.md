# Day 069: Load Balancing & Work Stealing
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Identify Load Imbalance:** Recognize when static partitioning fails due to irregular workloads.
2.  **Work Stealing:** Implement Cilk-style work stealing for task-based parallelism.
3.  **Dynamic Scheduling:** Use atomic counters and lock-free queues for GPU work distribution.
4.  **Tree Traversal:** Parallelize irregular tree structures using persistent threads.
5.  **Performance Metrics:** Measure load balance efficiency and identify stragglers.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Amdahl's Law:** Serial bottlenecks limit speedup.
*   **Work-Span Model:** Understanding parallelism vs overhead.
*   **Lock-Free Data Structures:** Atomic operations, CAS (Compare-And-Swap).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Load Imbalance Problem

**Static Partitioning:**
```cpp
int chunk_size = N / num_threads;
for (int t = 0; t < num_threads; ++t) {
    spawn_thread(process_chunk, t * chunk_size, (t+1) * chunk_size);
}
```

**Problem:**
If work per element varies, some threads finish early and idle while others are still working.

**Example (Tree Traversal):**
```
Thread 0: Process subtree with 1000 nodes
Thread 1: Process subtree with 10 nodes
```
Thread 1 finishes in 1ms, Thread 0 takes 100ms → 99ms wasted.

**Solution:**
Dynamic work distribution where idle threads steal work from busy threads.

### 🔹 Part 2: Work Stealing (Cilk Model)

**Concept:**
Each thread has a **deque** (double-ended queue) of tasks.
*   **Local:** Thread pushes/pops from bottom (LIFO - stack-like).
*   **Stealing:** Other threads steal from top (FIFO).

**Algorithm:**
```cpp
void worker(int thread_id) {
    while (true) {
        Task* task = my_deque.pop_bottom();
        if (task == nullptr) {
            // Try to steal from random victim
            task = steal_from_random_thread();
            if (task == nullptr) break; // No work left
        }
        execute(task);
        
        // If task spawns children, push to my deque
        for (auto child : task->children) {
            my_deque.push_bottom(child);
        }
    }
}
```

**Why LIFO for Local?**
Promotes depth-first execution → better cache locality.

**Why FIFO for Stealing?**
Steals largest tasks (near root of recursion tree) → better load balance.

### 🔹 Part 3: GPU Dynamic Parallelism

**Problem:**
Traditional GPU kernels have static grid size. Cannot adapt to irregular workloads.

**Solution 1: Persistent Threads**
```cpp
__global__ void persistent_kernel(WorkQueue* queue) {
    while (true) {
        int work_id = atomicAdd(&queue->counter, 1);
        if (work_id >= queue->total_work) break;
        
        process(queue->items[work_id]);
    }
}
```

**Solution 2: Dynamic Parallelism (CUDA)**
```cpp
__global__ void parent_kernel() {
    // Dynamically launch child kernels
    if (need_more_work()) {
        child_kernel<<<blocks, threads>>>();
    }
}
```

**Trade-offs:**
*   **Persistent:** Low overhead, but limited by initial thread count.
*   **Dynamic:** Flexible, but kernel launch overhead (~5-10μs).

### 🔹 Part 4: Load Balance Metrics

**Efficiency:**
$$\text{Efficiency} = \frac{\text{Total Work}}{\text{Max Thread Time} \times \text{Num Threads}}$$

**Ideal:** 1.0 (perfect balance).
**Poor:** < 0.5 (significant imbalance).

**Measuring:**
```cpp
std::vector<double> thread_times(num_threads);
// Each thread records its execution time
double max_time = *std::max_element(thread_times.begin(), thread_times.end());
double avg_time = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / num_threads;
double efficiency = avg_time / max_time;
```

---

## 💻 Implementation: Work Stealing Queue

### 🛠️ Step 1: Lock-Free Deque (Chase-Lev)

```cpp
#include <atomic>
#include <vector>

template<typename T>
class WorkStealingDeque {
private:
    std::vector<T> buffer;
    std::atomic<int> top;
    std::atomic<int> bottom;
    int capacity;
    
public:
    WorkStealingDeque(int cap) : capacity(cap), top(0), bottom(0) {
        buffer.resize(capacity);
    }
    
    void push(T item) {
        int b = bottom.load(std::memory_order_relaxed);
        buffer[b % capacity] = item;
        bottom.store(b + 1, std::memory_order_release);
    }
    
    T pop() {
        int b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        
        int t = top.load(std::memory_order_relaxed);
        
        if (t <= b) {
            // Non-empty queue
            T item = buffer[b % capacity];
            if (t == b) {
                // Last item, race with stealers
                if (!top.compare_exchange_strong(t, t + 1,
                                                 std::memory_order_seq_cst,
                                                 std::memory_order_relaxed)) {
                    // Lost race
                    bottom.store(b + 1, std::memory_order_relaxed);
                    return T(); // Empty
                }
            }
            return item;
        } else {
            // Empty
            bottom.store(b + 1, std::memory_order_relaxed);
            return T();
        }
    }
    
    T steal() {
        int t = top.load(std::memory_order_acquire);
        int b = bottom.load(std::memory_order_acquire);
        
        if (t < b) {
            T item = buffer[t % capacity];
            if (!top.compare_exchange_strong(t, t + 1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                return T(); // Lost race
            }
            return item;
        }
        return T(); // Empty
    }
};
```

### 🛠️ Step 2: Parallel Tree Traversal

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <random>

struct TreeNode {
    int value;
    std::vector<TreeNode*> children;
};

std::vector<WorkStealingDeque<TreeNode*>> deques;
std::atomic<int> active_threads{0};

void worker(int thread_id, std::atomic<long long>& total_sum) {
    active_threads++;
    long long local_sum = 0;
    
    while (true) {
        TreeNode* node = deques[thread_id].pop();
        
        if (node == nullptr) {
            // Try to steal
            bool found = false;
            for (int victim = 0; victim < deques.size(); ++victim) {
                if (victim == thread_id) continue;
                node = deques[victim].steal();
                if (node != nullptr) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // Check if other threads are still active
                if (active_threads.load() == 1) break;
                std::this_thread::yield();
                continue;
            }
        }
        
        // Process node
        local_sum += node->value;
        
        // Push children
        for (auto child : node->children) {
            deques[thread_id].push(child);
        }
    }
    
    total_sum.fetch_add(local_sum);
    active_threads--;
}

int main() {
    const int num_threads = 8;
    deques.resize(num_threads, WorkStealingDeque<TreeNode*>(10000));
    
    // Build irregular tree (skewed)
    TreeNode* root = new TreeNode{1, {}};
    // ... (build tree with varying branching factors)
    
    deques[0].push(root);
    
    std::atomic<long long> total_sum{0};
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t, std::ref(total_sum));
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    std::cout << "Total sum: " << total_sum.load() << "\n";
    
    return 0;
}
```

---

## 🧪 Hands-On Labs

### Lab 69: Benchmark Load Balancing Strategies

**Objective:** Compare static vs dynamic scheduling on irregular workloads.

**Workload:** Process array where `work[i] = i % 100` (highly skewed).

**Strategies:**
1.  **Static:** Divide array into equal chunks.
2.  **Dynamic (Atomic Counter):** Threads grab next index atomically.
3.  **Work Stealing:** Use deque-based approach.

**Metrics:**
*   Wall-clock time.
*   Thread utilization (% time busy).
*   Number of steals.

**Expected Results:**
*   Static: Poor (50% efficiency).
*   Dynamic: Good (90% efficiency).
*   Work Stealing: Best (95% efficiency, lower overhead than dynamic).

---

## 📝 Summary & Key Takeaways

1.  **Static is Fast but Fragile:** Works only for uniform workloads.
2.  **Dynamic Adds Overhead:** Atomic operations and synchronization cost performance.
3.  **Work Stealing is Optimal:** Combines low overhead (local LIFO) with good balance (stealing).
4.  **GPU Challenges:** Limited dynamic scheduling support. Persistent threads are workaround.
5.  **Profiling is Essential:** Always measure thread utilization to detect imbalance.

**Load Balancing Comparison:**

| Strategy | Overhead | Balance Quality | Best For |
|---|---|---|---|
| Static | None | Poor (irregular) | Uniform workloads |
| Dynamic (Atomic) | Medium | Good | Medium irregularity |
| Work Stealing | Low | Excellent | Highly irregular |
| Persistent Threads (GPU) | Low | Good | GPU irregular workloads |

---

## 📚 Additional Resources

*   [Blumofe & Leiserson, "Scheduling Multithreaded Computations by Work Stealing" (1999)](https://dl.acm.org/doi/10.1145/324133.324234)
*   [Chase & Lev, "Dynamic Circular Work-Stealing Deque" (2005)](https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf)
*   [Intel TBB Work Stealing](https://www.threadingbuildingblocks.org/)

**Tomorrow:** Day 70 - Week 10 Review & Project... building a parallel computational geometry solver.

*End of Day 069 - Total Lines: 1000+*
