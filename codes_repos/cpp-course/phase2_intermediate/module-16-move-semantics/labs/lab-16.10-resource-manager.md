# Lab 16.10: Resource Manager (Capstone)

## Objective
Build a complete resource manager using all move semantics concepts.

## Instructions

### Step 1: Design
Create a resource manager that:
- Manages file handles or memory buffers
- Implements Rule of Five correctly
- Uses move semantics for efficiency
- Supports resource pooling

Create `resource_manager.cpp`.

### Step 2: Resource Class
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

class FileResource {
    std::string filename;
    std::unique_ptr<std::fstream> file;
    bool is_open;
    
public:
    FileResource(const std::string& fname) 
        : filename(fname), is_open(false) {
        open();
    }
    
    ~FileResource() {
        close();
    }
    
    // Delete copy operations (move-only)
    FileResource(const FileResource&) = delete;
    FileResource& operator=(const FileResource&) = delete;
    
    // Move operations
    FileResource(FileResource&& other) noexcept 
        : filename(std::move(other.filename))
        , file(std::move(other.file))
        , is_open(other.is_open) {
        other.is_open = false;
    }
    
    FileResource& operator=(FileResource&& other) noexcept {
        if (this != &other) {
            close();
            filename = std::move(other.filename);
            file = std::move(other.file);
            is_open = other.is_open;
            other.is_open = false;
        }
        return *this;
    }
    
    void write(const std::string& data) {
        if (is_open && file) {
            *file << data;
        }
    }
    
private:
    void open() {
        file = std::make_unique<std::fstream>(
            filename, std::ios::out | std::ios::app
        );
        is_open = file->is_open();
    }
    
    void close() {
        if (is_open && file) {
            file->close();
            is_open = false;
        }
    }
};
```

### Step 3: Resource Pool
```cpp
template<typename T>
class ResourcePool {
    std::vector<T> resources;
    
public:
    template<typename... Args>
    void add(Args&&... args) {
        resources.emplace_back(std::forward<Args>(args)...);
    }
    
    T acquire() {
        if (resources.empty()) {
            throw std::runtime_error("No resources available");
        }
        T resource = std::move(resources.back());
        resources.pop_back();
        return resource;
    }
    
    void release(T&& resource) {
        resources.push_back(std::move(resource));
    }
    
    size_t size() const { return resources.size(); }
};
```

## Challenges

### Challenge 1: RAII Wrapper
Create an RAII wrapper that automatically returns resources to the pool.

### Challenge 2: Performance Analysis
Measure the performance benefits of move semantics in your manager.

### Challenge 3: Thread Safety
Add basic thread safety to the resource pool.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <chrono>
#include <mutex>

// File Resource (move-only)
class FileResource {
    std::string filename;
    std::unique_ptr<std::fstream> file;
    bool is_open;
    
public:
    FileResource(const std::string& fname = "") 
        : filename(fname), is_open(false) {
        if (!filename.empty()) {
            open();
        }
    }
    
    ~FileResource() { close(); }
    
    FileResource(const FileResource&) = delete;
    FileResource& operator=(const FileResource&) = delete;
    
    FileResource(FileResource&& other) noexcept 
        : filename(std::move(other.filename))
        , file(std::move(other.file))
        , is_open(other.is_open) {
        other.is_open = false;
    }
    
    FileResource& operator=(FileResource&& other) noexcept {
        if (this != &other) {
            close();
            filename = std::move(other.filename);
            file = std::move(other.file);
            is_open = other.is_open;
            other.is_open = false;
        }
        return *this;
    }
    
    void write(const std::string& data) {
        if (is_open && file) {
            *file << data << "\n";
        }
    }
    
    bool isOpen() const { return is_open; }
    
private:
    void open() {
        file = std::make_unique<std::fstream>(
            filename, std::ios::out | std::ios::app
        );
        is_open = file && file->is_open();
    }
    
    void close() {
        if (is_open && file) {
            file->close();
            is_open = false;
        }
    }
};

// Challenge 3: Thread-safe resource pool
template<typename T>
class ResourcePool {
    std::vector<T> resources;
    mutable std::mutex mtx;
    
public:
    template<typename... Args>
    void add(Args&&... args) {
        std::lock_guard<std::mutex> lock(mtx);
        resources.emplace_back(std::forward<Args>(args)...);
    }
    
    T acquire() {
        std::lock_guard<std::mutex> lock(mtx);
        if (resources.empty()) {
            throw std::runtime_error("No resources available");
        }
        T resource = std::move(resources.back());
        resources.pop_back();
        return resource;
    }
    
    void release(T&& resource) {
        std::lock_guard<std::mutex> lock(mtx);
        resources.push_back(std::move(resource));
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return resources.size();
    }
};

// Challenge 1: RAII Wrapper
template<typename T>
class PooledResource {
    T resource;
    ResourcePool<T>* pool;
    
public:
    PooledResource(T&& res, ResourcePool<T>* p) 
        : resource(std::move(res)), pool(p) {}
    
    ~PooledResource() {
        if (pool) {
            pool->release(std::move(resource));
        }
    }
    
    // Move-only
    PooledResource(const PooledResource&) = delete;
    PooledResource& operator=(const PooledResource&) = delete;
    
    PooledResource(PooledResource&& other) noexcept
        : resource(std::move(other.resource))
        , pool(other.pool) {
        other.pool = nullptr;
    }
    
    T* operator->() { return &resource; }
    T& operator*() { return resource; }
};

// Challenge 2: Performance benchmark
void benchmarkMoveVsCopy() {
    const int iterations = 1000;
    
    // Move benchmark
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<FileResource> vec;
    for (int i = 0; i < iterations; ++i) {
        FileResource res("temp_" + std::to_string(i) + ".txt");
        vec.push_back(std::move(res)); // Move
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Move semantics: " << duration.count() << "μs\n";
}

int main() {
    std::cout << "=== Resource Manager Demo ===\n\n";
    
    // Create resource pool
    ResourcePool<FileResource> pool;
    pool.add("log1.txt");
    pool.add("log2.txt");
    pool.add("log3.txt");
    
    std::cout << "Pool size: " << pool.size() << "\n";
    
    // Acquire and use resource
    {
        auto res = pool.acquire();
        res.write("Log entry 1");
        std::cout << "Acquired resource, pool size: " << pool.size() << "\n";
        
        // RAII wrapper automatically returns to pool
        PooledResource<FileResource> pooled(std::move(res), &pool);
        pooled->write("Log entry 2");
    } // Resource returned to pool here
    
    std::cout << "Resource returned, pool size: " << pool.size() << "\n";
    
    // Performance benchmark
    std::cout << "\n=== Performance Benchmark ===\n";
    benchmarkMoveVsCopy();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented move-only resource class
✅ Created resource pool with move semantics
✅ Built RAII wrapper (Challenge 1)
✅ Benchmarked performance (Challenge 2)
✅ Added thread safety (Challenge 3)

## Key Learnings
- Move semantics enable efficient resource management
- Move-only types prevent accidental copies
- RAII ensures resources are properly released
- Resource pooling benefits from move semantics
- Thread safety requires careful synchronization

## Rust Comparison
```rust
// Rust's ownership system handles this automatically
struct FileResource {
    file: File,
}

// No need for explicit move semantics
let res1 = FileResource::new("file.txt");
let res2 = res1; // res1 moved, no longer accessible
```

## Next Steps
Congratulations! You've completed Module 16. Proceed to **Module 17: Concurrency and Multithreading**.
