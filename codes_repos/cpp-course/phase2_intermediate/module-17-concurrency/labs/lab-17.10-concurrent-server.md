# Lab 17.10: Concurrent Server (Capstone)

## Objective
Build a multi-threaded server that handles concurrent client requests.

## Instructions

### Step 1: Design
Create a concurrent server that:
- Accepts multiple client connections
- Processes requests in parallel
- Uses thread pool for efficiency
- Implements thread-safe request queue

Create `concurrent_server.cpp`.

### Step 2: Request Queue
```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>

class RequestQueue {
    std::queue<std::string> requests;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    
public:
    void push(const std::string& request) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            requests.push(request);
        }
        cv.notify_one();
    }
    
    bool pop(std::string& request) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !requests.empty() || done; });
        
        if (requests.empty()) return false;
        
        request = std::move(requests.front());
        requests.pop();
        return true;
    }
    
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
    }
};
```

### Step 3: Worker Thread Pool
```cpp
#include <thread>
#include <vector>

class WorkerPool {
    std::vector<std::thread> workers;
    RequestQueue& queue;
    
public:
    WorkerPool(size_t numWorkers, RequestQueue& q) : queue(q) {
        for (size_t i = 0; i < numWorkers; ++i) {
            workers.emplace_back([this, id = i]() {
                std::string request;
                while (queue.pop(request)) {
                    processRequest(id, request);
                }
            });
        }
    }
    
    ~WorkerPool() {
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
private:
    void processRequest(int workerId, const std::string& request) {
        std::cout << "Worker " << workerId 
                  << " processing: " << request << "\n";
        
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
};
```

## Challenges

### Challenge 1: Request Statistics
Track concurrent request statistics (total, in-progress, completed).

### Challenge 2: Priority Requests
Implement priority-based request handling.

### Challenge 3: Rate Limiting
Add per-client rate limiting.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <chrono>
#include <atomic>
#include <unordered_map>

// Challenge 1: Request statistics
class Statistics {
    std::atomic<size_t> totalRequests{0};
    std::atomic<size_t> inProgress{0};
    std::atomic<size_t> completed{0};
    
public:
    void requestStarted() {
        ++totalRequests;
        ++inProgress;
    }
    
    void requestCompleted() {
        --inProgress;
        ++completed;
    }
    
    void report() const {
        std::cout << "\n=== Statistics ===\n";
        std::cout << "Total: " << totalRequests << "\n";
        std::cout << "In Progress: " << inProgress << "\n";
        std::cout << "Completed: " << completed << "\n";
    }
};

// Challenge 2: Priority request queue
struct Request {
    int priority;
    std::string data;
    
    bool operator<(const Request& other) const {
        return priority < other.priority; // Higher priority first
    }
};

class PriorityRequestQueue {
    std::priority_queue<Request> requests;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    
public:
    void push(int priority, const std::string& data) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            requests.push({priority, data});
        }
        cv.notify_one();
    }
    
    bool pop(Request& request) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !requests.empty() || done; });
        
        if (requests.empty()) return false;
        
        request = requests.top();
        requests.pop();
        return true;
    }
    
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
    }
};

// Challenge 3: Rate limiter
class RateLimiter {
    struct ClientInfo {
        std::chrono::steady_clock::time_point lastRequest;
        int requestCount = 0;
    };
    
    std::unordered_map<std::string, ClientInfo> clients;
    std::mutex mtx;
    int maxRequestsPerSecond;
    
public:
    RateLimiter(int maxReqs) : maxRequestsPerSecond(maxReqs) {}
    
    bool allowRequest(const std::string& clientId) {
        std::lock_guard<std::mutex> lock(mtx);
        
        auto now = std::chrono::steady_clock::now();
        auto& info = clients[clientId];
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - info.lastRequest
        );
        
        if (elapsed.count() >= 1) {
            info.requestCount = 1;
            info.lastRequest = now;
            return true;
        }
        
        if (info.requestCount < maxRequestsPerSecond) {
            ++info.requestCount;
            return true;
        }
        
        return false; // Rate limit exceeded
    }
};

// Complete concurrent server
class ConcurrentServer {
    PriorityRequestQueue queue;
    std::vector<std::thread> workers;
    Statistics stats;
    RateLimiter rateLimiter;
    
public:
    ConcurrentServer(size_t numWorkers, int maxRequestsPerSecond) 
        : rateLimiter(maxRequestsPerSecond) {
        
        for (size_t i = 0; i < numWorkers; ++i) {
            workers.emplace_back([this, id = i]() {
                Request request;
                while (queue.pop(request)) {
                    stats.requestStarted();
                    processRequest(id, request);
                    stats.requestCompleted();
                }
            });
        }
    }
    
    ~ConcurrentServer() {
        queue.finish();
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    bool submitRequest(const std::string& clientId, int priority, const std::string& data) {
        if (!rateLimiter.allowRequest(clientId)) {
            std::cout << "Rate limit exceeded for client: " << clientId << "\n";
            return false;
        }
        
        queue.push(priority, data);
        return true;
    }
    
    void reportStats() const {
        stats.report();
    }
    
private:
    void processRequest(int workerId, const Request& request) {
        std::cout << "Worker " << workerId 
                  << " processing priority " << request.priority 
                  << ": " << request.data << "\n";
        
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
};

int main() {
    std::cout << "=== Concurrent Server ===\n";
    
    ConcurrentServer server(4, 10); // 4 workers, 10 req/sec limit
    
    // Simulate clients
    std::vector<std::thread> clients;
    
    for (int i = 0; i < 3; ++i) {
        clients.emplace_back([&, clientId = i]() {
            for (int j = 0; j < 20; ++j) {
                std::string request = "Client" + std::to_string(clientId) 
                                    + "_Req" + std::to_string(j);
                int priority = (j % 3) + 1; // Priority 1-3
                
                server.submitRequest("client" + std::to_string(clientId), 
                                   priority, request);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
    
    for (auto& client : clients) {
        client.join();
    }
    
    // Wait for processing to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    server.reportStats();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented concurrent request queue
✅ Created worker thread pool
✅ Added request statistics (Challenge 1)
✅ Implemented priority handling (Challenge 2)
✅ Added rate limiting (Challenge 3)

## Key Learnings
- Thread pools efficiently handle concurrent requests
- Priority queues enable request prioritization
- Atomics track statistics without locks
- Rate limiting prevents resource exhaustion
- Condition variables coordinate workers

## Rust Comparison
```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(100);
    
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            // Process request
        }
    });
}
```

## Next Steps
Congratulations! You've completed Module 17. Proceed to **Module 18: Build Systems and Package Managers**.
