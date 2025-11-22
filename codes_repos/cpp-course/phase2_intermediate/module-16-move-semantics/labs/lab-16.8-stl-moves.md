# Lab 16.8: Move Semantics with STL

## Objective
Leverage move semantics with STL containers and algorithms for better performance.

## Instructions

### Step 1: Moving into Containers
Create `stl_moves.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec;
    
    std::string s1 = "Hello";
    vec.push_back(s1);              // Copy
    vec.push_back(std::move(s1));   // Move
    
    // Emplace constructs in-place (no move needed)
    vec.emplace_back("World");
    
    return 0;
}
```

### Step 2: Moving from Containers
```cpp
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec = {"A", "B", "C"};
    
    // Move element out
    std::string s = std::move(vec[0]);
    // vec[0] is now in valid but unspecified state
    
    // Move entire container
    std::vector<std::string> vec2 = std::move(vec);
    // vec is now empty (moved-from state)
    
    return 0;
}
```

### Step 3: Algorithms with Move
```cpp
#include <algorithm>
#include <vector>

int main() {
    std::vector<std::string> src = {"A", "B", "C"};
    std::vector<std::string> dst(3);
    
    // Move elements
    std::move(src.begin(), src.end(), dst.begin());
    // src elements are now in unspecified state
    
    return 0;
}
```

### Step 4: Move-Only Types in Containers
```cpp
#include <vector>
#include <memory>

int main() {
    std::vector<std::unique_ptr<int>> vec;
    
    // Can't copy unique_ptr, must move
    vec.push_back(std::make_unique<int>(42));
    
    auto ptr = std::make_unique<int>(99);
    vec.push_back(std::move(ptr));
    
    return 0;
}
```

## Challenges

### Challenge 1: Benchmark
Compare performance of copying vs moving large objects in vectors.

### Challenge 2: Custom Move-Aware Container
Implement a simple container that uses move semantics efficiently.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

// Challenge 1: Benchmark
class LargeObject {
    std::vector<int> data;
public:
    LargeObject() : data(10000, 42) {}
    
    LargeObject(const LargeObject& other) : data(other.data) {
        // Expensive copy
    }
    
    LargeObject(LargeObject&& other) noexcept 
        : data(std::move(other.data)) {
        // Cheap move
    }
};

void benchmarkCopy() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<LargeObject> vec;
    for (int i = 0; i < 1000; ++i) {
        LargeObject obj;
        vec.push_back(obj); // Copy
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Copy: " << duration.count() << "ms\n";
}

void benchmarkMove() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<LargeObject> vec;
    for (int i = 0; i < 1000; ++i) {
        LargeObject obj;
        vec.push_back(std::move(obj)); // Move
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Move: " << duration.count() << "ms\n";
}

// Challenge 2: Move-aware container
template<typename T>
class SimpleVector {
    T* data;
    size_t size;
    size_t capacity;
    
public:
    SimpleVector() : data(nullptr), size(0), capacity(0) {}
    
    ~SimpleVector() { delete[] data; }
    
    void push_back(const T& value) {
        ensureCapacity();
        data[size++] = value; // Copy
    }
    
    void push_back(T&& value) {
        ensureCapacity();
        data[size++] = std::move(value); // Move
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        ensureCapacity();
        new (&data[size++]) T(std::forward<Args>(args)...);
    }
    
private:
    void ensureCapacity() {
        if (size >= capacity) {
            size_t newCapacity = capacity == 0 ? 1 : capacity * 2;
            T* newData = new T[newCapacity];
            
            // Move existing elements
            for (size_t i = 0; i < size; ++i) {
                newData[i] = std::move(data[i]);
            }
            
            delete[] data;
            data = newData;
            capacity = newCapacity;
        }
    }
};

int main() {
    std::cout << "=== Benchmark ===\n";
    benchmarkCopy();
    benchmarkMove();
    
    std::cout << "\n=== Custom Container ===\n";
    SimpleVector<std::string> vec;
    
    std::string s = "Hello";
    vec.push_back(s);              // Copy
    vec.push_back(std::move(s));   // Move
    vec.emplace_back("World");     // Construct in-place
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used move semantics with STL containers
✅ Moved elements with algorithms
✅ Benchmarked copy vs move (Challenge 1)
✅ Implemented move-aware container (Challenge 2)

## Key Learnings
- `push_back` with `std::move` avoids copies
- `emplace_back` constructs in-place
- STL algorithms like `std::move` transfer elements
- Move-only types (like `unique_ptr`) require move semantics
- Containers automatically use move when available

## Next Steps
Proceed to **Lab 16.9: Common Pitfalls**.
