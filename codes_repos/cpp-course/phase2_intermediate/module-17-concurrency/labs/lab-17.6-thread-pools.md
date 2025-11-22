# Lab 17.6: Thread Pools

## Objective
Implement a thread pool for efficient task management and reuse.

## Instructions

### Step 1: Thread Pool Concept
A thread pool maintains a fixed number of worker threads that process tasks from a queue.

Create `thread_pool.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
    
public:
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task(); // Execute task
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    void enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.push(std::move(task));
        }
        cv.notify_one();
    }
};
```

### Step 2: Using the Thread Pool
```cpp
int main() {
    ThreadPool pool(4); // 4 worker threads
    
    for (int i = 0; i < 10; ++i) {
        pool.enqueue([i]() {
            std::cout << "Task " << i << " on thread " 
                      << std::this_thread::get_id() << "\n";
        });
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 0;
}
```

## Challenges

### Challenge 1: Return Values
Extend the thread pool to return futures for task results.

### Challenge 2: Priority Queue
Implement a priority-based task queue.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

// Challenge 1: Thread pool with futures
class AdvancedThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
    
public:
    AdvancedThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    ~AdvancedThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.emplace([task]() { (*task)(); });
        }
        
        cv.notify_one();
        return result;
    }
};

// Challenge 2: Priority thread pool
class PriorityThreadPool {
    struct Task {
        int priority;
        std::function<void()> func;
        
        bool operator<(const Task& other) const {
            return priority < other.priority; // Higher priority first
        }
    };
    
    std::vector<std::thread> workers;
    std::priority_queue<Task> tasks;
    
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
    
public:
    PriorityThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) return;
                        
                        task = std::move(tasks.top().func);
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    ~PriorityThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    void enqueue(int priority, std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.push({priority, std::move(task)});
        }
        cv.notify_one();
    }
};

int main() {
    std::cout << "=== Advanced Thread Pool ===\n";
    
    AdvancedThreadPool pool(4);
    
    std::vector<std::future<int>> results;
    
    for (int i = 0; i < 10; ++i) {
        results.push_back(pool.enqueue([i]() {
            return i * i;
        }));
    }
    
    for (auto& result : results) {
        std::cout << "Result: " << result.get() << "\n";
    }
    
    std::cout << "\n=== Priority Thread Pool ===\n";
    
    PriorityThreadPool priorityPool(2);
    
    priorityPool.enqueue(1, []() { std::cout << "Low priority\n"; });
    priorityPool.enqueue(10, []() { std::cout << "High priority\n"; });
    priorityPool.enqueue(5, []() { std::cout << "Medium priority\n"; });
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented basic thread pool
✅ Reused worker threads for multiple tasks
✅ Added future support (Challenge 1)
✅ Implemented priority queue (Challenge 2)

## Key Learnings
- Thread pools avoid thread creation overhead
- Worker threads wait on condition variable
- Tasks are queued and processed by available workers
- Futures enable retrieving task results
- Priority queues enable task prioritization

## Next Steps
Proceed to **Lab 17.7: Read-Write Locks**.
