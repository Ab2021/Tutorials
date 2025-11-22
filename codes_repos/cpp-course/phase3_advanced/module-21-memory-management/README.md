# Module 21: Memory Management

## Overview
Advanced memory management techniques including custom allocators, memory pools, and optimization strategies for high-performance C++ applications.

## Learning Objectives
By the end of this module, you will be able to:
- Understand memory layout and alignment
- Implement custom allocators
- Create memory pools
- Manage object lifetimes efficiently
- Debug memory issues
- Optimize memory usage

## Key Concepts

### 1. Memory Layout
Understanding how objects are laid out in memory.

```cpp
struct Data {
    char a;      // 1 byte
    // 3 bytes padding
    int b;       // 4 bytes
    char c;      // 1 byte
    // 3 bytes padding
};
// Total: 12 bytes (with padding)
```

### 2. Custom Allocators
Implementing allocators for STL containers.

```cpp
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    
    T* allocate(size_t n) {
        return static_cast<T*>(pool.allocate(n * sizeof(T)));
    }
    
    void deallocate(T* p, size_t n) {
        pool.deallocate(p, n * sizeof(T));
    }
};

std::vector<int, PoolAllocator<int>> vec;
```

### 3. Memory Pools
Pre-allocated memory for fast allocation.

```cpp
class MemoryPool {
    std::vector<char> buffer;
    size_t offset = 0;
    
public:
    void* allocate(size_t size) {
        if (offset + size > buffer.size()) {
            throw std::bad_alloc();
        }
        void* ptr = &buffer[offset];
        offset += size;
        return ptr;
    }
};
```

### 4. Alignment
Ensuring proper memory alignment.

```cpp
alignas(64) struct CacheLine {
    int data[16];
};

void* aligned = std::aligned_alloc(64, size);
```

### 5. Memory Debugging
Tools and techniques for finding memory issues.

```cpp
// Valgrind, AddressSanitizer, etc.
#ifdef DEBUG
    void* operator new(size_t size) {
        void* p = malloc(size);
        std::cout << "Allocated: " << size << " bytes\n";
        return p;
    }
#endif
```

## Rust Comparison

### Memory Safety
**C++:**
```cpp
int* ptr = new int(42);
delete ptr; // Manual management
```

**Rust:**
```rust
let value = Box::new(42);
// Automatically freed
```

### Custom Allocators
**C++:**
```cpp
std::vector<int, CustomAllocator<int>> vec;
```

**Rust:**
```rust
// Uses global allocator or custom via #[global_allocator]
```

## Labs

1. **Lab 21.1**: Memory Layout and Alignment
2. **Lab 21.2**: Custom Allocator Basics
3. **Lab 21.3**: STL Allocator Interface
4. **Lab 21.4**: Memory Pool Implementation
5. **Lab 21.5**: Stack Allocator
6. **Lab 21.6**: Object Lifetime Management
7. **Lab 21.7**: Placement New
8. **Lab 21.8**: Memory Debugging Tools
9. **Lab 21.9**: Cache-Friendly Data Structures
10. **Lab 21.10**: High-Performance Allocator (Capstone)

## Additional Resources
- "Effective Modern C++" by Scott Meyers
- Valgrind documentation
- AddressSanitizer guide

## Next Module
After completing this module, proceed to **Module 22: Performance Optimization**.
