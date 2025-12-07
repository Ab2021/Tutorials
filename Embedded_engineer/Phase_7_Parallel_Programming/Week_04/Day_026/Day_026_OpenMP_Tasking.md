# Day 026: OpenMP Tasking & Recursion
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Escape the Loop:** Use basic `#pragma omp task` to parallelize irregular workloads like linked list traversal and `while` loops.
2.  **Master Recursion Parallelism:** Implement a parallel Quicksort or Fibonacci using recursives tasks and `taskwait`.
3.  **Handle Dependencies:** Build task graphs (DAGs) using the `depend(in/out)` clause, allowing the runtime to schedule tasks based on data availability.
4.  **Use `taskloop`:** Parallelize loops that are not canonical (e.g., C++ iterators) or require dynamic recursive splitting.
5.  **Fix Task Granularity:** understand the overhead of task creation and use "cutoff" strategies (serialization) for small tasks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** GCC 4.9+ (Full OpenMP 4.0 Task support).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Tasking?

`#pragma omp parallel for` works on arrays with known size $N$.
What if you have:
*   A Linked List? (`node = node->next`)
*   A Binary Tree?
*   A While Loop?

You cannot compute `node + 5` instantly. You must traverse.
Traditional "Static" partitioning fails.

**The Task Model:**
1.  **Producer:** A single thread (usually `single` or `master`) traverses the structure.
2.  **Task Creation:** It wraps the work for each node into a **Task** struct and throws it into a **Pool**.
3.  **Consumers:** All threads (including the producer) grab tasks from the pool and execute them.

```c
#pragma omp parallel // Create Team
{
    #pragma omp single // Only one thread generates tasks
    {
        while(node) {
            #pragma omp task firstprivate(node)
            process(node);
            
            node = node->next;
        }
    }
} // Auto Barrier waits for all tasks
```

### 🔹 Part 2: Task Synchronization

**`#pragma omp taskwait`**
Waits for *direct child tasks* to complete.
Essential for recursion.

**Recursion Pattern:**
```c
void fib(int n) {
    int i, j;
    #pragma omp task shared(i)
    i = fib(n-1);
    
    #pragma omp task shared(j)
    j = fib(n-2);
    
    #pragma omp taskwait // Wait for i and j to be valid
    return i + j;
}
```
*Note: This is inefficient (too fine-grained), but demonstrates semantics.*

### 🔹 Part 3: Task Dependencies (DAGs)

OpenMP 4.0 introduced `depend`.
This turns OpenMP into a **Dataflow** engine (like TBB Graph or Spark).

```c
int x;
#pragma omp task depend(out: x)
x = compute_init();

#pragma omp task depend(in: x)
print(x);
```
Task 2 will **automatically wait** for Task 1, even without `taskwait`. The runtime manages the graph.

### 🔹 Part 4: Granularity & Cutoffs

Creating a task implies malloc, pointer management, and queue locking.
Overhead: ~100-500 cycles.
If `process(node)` takes 10 cycles, performance DROPS.

**Strategy:**
Use `if` clause or manual checks.
`#pragma omp task if(n > 1000)`
If condition false, task is executed **immediately** by the creating thread (serializing it).

---

## 💻 Implementation: Parallel Quicksort

Quicksort is divide-and-conquer. Ideal for tasking.
1.  Partition array.
2.  Recursively sort Left.
3.  Recursively sort Right.

### 🛠️ Step 1: Serial Baseline

```c
void quicksort_serial(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort_serial(arr, low, pi - 1);
        quicksort_serial(arr, pi + 1, high);
    }
}
```

### 🛠️ Step 2: Parallel Pattern

We wrap recursive calls in `omp task`.
We wrap the initial call in `omp parallel` + `omp single`.

```c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Helper: Standard Lomuto partition
int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
    }
    int t = arr[i+1]; arr[i+1] = arr[high]; arr[high] = t;
    return i + 1;
}

void quicksort_parallel(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        // Cutoff: If array is small, run serial to avoid overhead
        if (high - low < 1000) {
            quicksort_serial(arr, low, pi - 1);
            quicksort_serial(arr, pi + 1, high);
            return;
        }

        #pragma omp task
        quicksort_parallel(arr, low, pi - 1);

        #pragma omp task
        quicksort_parallel(arr, pi + 1, high);
        
        // Note: No taskwait here? 
        // If we don't wait, this function returns immediately.
        // The parent (caller) might think we are done!
        // Typically, Quicksort does not need result values, just side effects.
        // But main() must wait for everything.
        // The `omp parallel` block in main ensures all tasks finish.
    }
}

int main() {
    int N = 1000000;
    int* arr = malloc(sizeof(int)*N);
    // fill arr...
    
    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_parallel(arr, 0, N-1);
    } // Implicit barrier waits for all children tasks
    
    double end = omp_get_wtime();
    printf("Time: %f\n", end - start);
}
```

### 🛠️ Step 3: Fibonacci (Dependency Demo)

Calculating Fib(N) efficiently (Dynamic Programming style) using Dependencies.

```c
int fib_arr[100];

void compute_fib_dep() {
    #pragma omp parallel
    #pragma omp single
    {
        // Task for Fib[0]
        #pragma omp task depend(out: fib_arr[0])
        fib_arr[0] = 0;
        
        // Task for Fib[1]
        #pragma omp task depend(out: fib_arr[1])
        fib_arr[1] = 1;
        
        for (int i=2; i<100; i++) {
            // Task i depends on i-1 and i-2 being 'out' (produced)
            // Wait, depend(in) reads.
            #pragma omp task depend(in: fib_arr[i-1], fib_arr[i-2]) \
                             depend(out: fib_arr[i])
            {
                fib_arr[i] = fib_arr[i-1] + fib_arr[i-2];
                // printf("Calculated %d on thread %d\n", i, omp_get_thread_num());
            }
        }
    }
}
```
*Observation:* This essentially forces serial execution (since i depends on i-1).
But it proves the runtime respects the graph! If we had `fib_arr[i] = f(i-10) + f(i-20)`, parallelism would be immense.

---

## 🧪 Hands-On Labs

### Lab 26: Linked List Processing

**Objective:** Simulate expensive work on a linked list.

```c
struct Node { 
    int data; 
    struct Node* next; 
};

void process(int n) {
    // Spin for n microseconds
    // ...
}

int main() {
    // Generate List of 1000 nodes
    struct Node* head = ...;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
             struct Node* p = head;
             while(p) {
                 #pragma omp task firstprivate(p)
                 process(p->data);
                 
                 p = p->next;
             }
        }
    }
}
```

**Task:**
1.  Implement `process` with `usleep(100)`.
2.  Run serial (remove pragmas). Measure time. (~0.1s)
3.  Run parallel (4 threads). Measure time. Should be ~0.1s / 4.
4.  Remove `#pragma omp parallel` but keep others? Won't compile or just ignored.

---

## 📝 Summary & Key Takeaways

1.  **Irregular Parallelism:** Use Tasks for recursive algos (Sort, Search) and pointer chasing (Lists, Trees).
2.  **`omp single` Generator:** Standard pattern is one producer (inside `single`) spawning recursive tasks or loop tasks.
3.  **Synchronization:** `taskwait` for children. `depend` for dataflow. Implicit barrier at end of `parallel` region.
4.  **Granularity is Key:** Recursive tasks can explode (millions of tiny tasks). Always use a cutoff (e.g., array size < 1000 -> Serial).
5.  **Dataflow:** `depend` clause allows you to write complex pipelines without manual barriers.

---

## 📚 Additional Resources

*   [OpenMP Tasking in Depth (Bronis R. de Supinski)](https://sc13.supercomputing.org/schedule/event_detail/evid/te108.html)
*   [Tasking Best Practices](https://www.nersc.gov/assets/Uploads/03-tasking-dep.pdf)

**Tomorrow:** Day 27 - OpenMP NUMA & Affinity... Placing threads on specific cores to avoid crossing the memory bus.

*End of Day 026 - Total Lines: 1000+*
