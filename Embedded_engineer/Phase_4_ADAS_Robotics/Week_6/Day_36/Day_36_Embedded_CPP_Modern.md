# Day 36: Embedded C++ for Robotics (Modern C++)
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 36 Focus:**
> Python is great for prototyping (Weeks 1-5), but real autonomous cars run on **C++**. It offers direct hardware access, deterministic performance, and zero-overhead abstractions. Today, we leave "C with Classes" behind and master **Modern C++ (C++11 to C++20)**, the language of ROS 2 and high-performance robotics.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the benefits of Modern C++ features: `auto`, `nullptr`, Range-based loops.
2.  **Manage** memory safely using Smart Pointers (`std::unique_ptr`, `std::shared_ptr`) and RAII.
3.  **Optimize** performance using Move Semantics (`std::move`) and `constexpr`.
4.  **Utilize** STL containers (`std::vector`, `std::array`) and Algorithms (`std::sort`, `std::transform`).
5.  **Write** a Modern C++ application with `CMake` build system.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **C Basics:** Pointers, Structs, Memory layout.
-   **OOP:** Classes, Inheritance, Polymorphism.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows with WSL2/MinGW).

### Software Stack
-   **Compiler:** `g++` (GCC 9+) or `clang`.
-   **Build System:** `cmake`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Modern" Revolution (C++11/14)

#### 1.1 Type Inference (`auto`)
Don't write `std::vector<int>::iterator it = v.begin()`.
Write `auto it = v.begin()`.
-   **Rule:** Use `auto` when the type is obvious (e.g., return of a function) or hard to spell.

#### 1.2 Range-Based For Loops
Don't write `for(int i=0; i<v.size(); ++i)`.
Write `for(const auto& item : v)`.
-   **Safety:** No index out of bounds errors.

#### 1.3 `nullptr`
`NULL` is just `0` (an integer). This causes ambiguity in function overloading.
`nullptr` is a specific type (`std::nullptr_t`). Always use it.

### 🔹 Part 2: Memory Management (RAII)

**RAII (Resource Acquisition Is Initialization):**
-   Acquire resource (memory, file handle, lock) in Constructor.
-   Release resource in Destructor.
-   **Result:** No memory leaks, even if exceptions are thrown.

#### 2.1 Smart Pointers
Never use `new` and `delete` manually.
-   **`std::unique_ptr`:** Exclusive ownership. Cannot be copied, only moved. Fast (no overhead).
-   **`std::shared_ptr`:** Shared ownership (Reference counting). Deletes object when count reaches 0. Slower (thread-safe counter).
-   **`std::weak_ptr`:** Non-owning reference to a shared_ptr. Breaks cyclic references.

### 🔹 Part 3: Performance Features

#### 3.1 Move Semantics (`std::move`)
Copying a large object (e.g., a Lidar point cloud) is slow.
**Move:** Steal the pointer from the old object and give it to the new one. The old object is left empty.
-   Happens automatically for temporary objects (R-values).
-   Force it with `std::move(obj)`.

#### 3.2 Compile-Time Evaluation (`constexpr`)
Compute values at compile time, not runtime.
-   `constexpr int factorial(int n) { ... }`
-   If called with a constant, the compiler replaces the function call with the result. Zero runtime cost.

---

## 💻 Implementation: Modern C++ Demo

We will write a C++ program `robot_manager.cpp` that demonstrates these concepts.

### 🛠️ Setup
Create `week6_day36`.

```bash
mkdir -p ~/ros2_ws/src/week6_day36
cd ~/ros2_ws/src/week6_day36
touch robot_manager.cpp CMakeLists.txt
```

### 👨‍💻 Code: robot_manager.cpp

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>

// 1. RAII Class
class LidarSensor {
public:
    LidarSensor(int id) : id_(id) {
        std::cout << "Lidar " << id_ << " Initialized (Resource Acquired)\n";
        data_ = new float[1000]; // Simulate raw memory allocation
    }

    ~LidarSensor() {
        std::cout << "Lidar " << id_ << " Shutdown (Resource Released)\n";
        delete[] data_; // RAII ensures this is called
    }

    // Delete Copy Constructor (Unique ownership simulation)
    LidarSensor(const LidarSensor&) = delete;
    LidarSensor& operator=(const LidarSensor&) = delete;

    // Allow Move Constructor
    LidarSensor(LidarSensor&& other) noexcept : id_(other.id_), data_(other.data_) {
        other.data_ = nullptr; // Steal the pointer
        std::cout << "Lidar " << id_ << " Moved\n";
    }

    void scan() {
        std::cout << "Lidar " << id_ << " Scanning...\n";
    }

private:
    int id_;
    float* data_;
};

// 2. Factory Function (std::make_unique)
std::unique_ptr<LidarSensor> create_lidar(int id) {
    return std::make_unique<LidarSensor>(id);
}

// 3. Lambda Function & STL Algorithm
void process_data() {
    std::vector<int> readings = {10, 5, 8, 3, 12};

    // Sort using Lambda
    std::sort(readings.begin(), readings.end(), [](int a, int b) {
        return a > b; // Descending
    });

    std::cout << "Sorted Readings: ";
    for (const auto& r : readings) {
        std::cout << r << " ";
    }
    std::cout << "\n";
}

// 4. Constexpr (Compile Time)
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

int main() {
    std::cout << "--- Modern C++ Demo ---\n";

    // Smart Pointers
    {
        std::cout << "[Scope Start]\n";
        auto lidar1 = create_lidar(1);
        lidar1->scan();
        
        // std::shared_ptr
        std::shared_ptr<LidarSensor> shared_lidar = std::make_shared<LidarSensor>(2);
        {
            auto another_ref = shared_lidar; // Count = 2
            std::cout << "Shared Count: " << shared_lidar.use_count() << "\n";
        } // another_ref dies, Count = 1
        std::cout << "Shared Count: " << shared_lidar.use_count() << "\n";
        
        std::cout << "[Scope End] -> Destructors called automatically\n";
    }

    // Move Semantics
    std::cout << "\n--- Move Semantics ---\n";
    std::vector<LidarSensor> sensors;
    sensors.push_back(LidarSensor(3)); // Temporary object moved into vector

    // STL & Lambdas
    std::cout << "\n--- STL & Lambdas ---\n";
    process_data();

    // Constexpr
    std::cout << "\n--- Constexpr ---\n";
    constexpr int val = factorial(5); // Computed at compile time
    std::cout << "Factorial(5) = " << val << "\n";

    return 0;
}
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(ModernCppDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(robot_manager robot_manager.cpp)
```

### 🛠️ Build & Run

```bash
mkdir build
cd build
cmake ..
make
./robot_manager
```

---

## 🔬 Lab Exercise: Memory Leak Check

### Lab Objectives
1.  Run the code. Observe the "Lidar Shutdown" messages.
2.  **Experiment:** Remove the `delete[] data_` from the destructor.
3.  **Tool:** Use `valgrind` to check for leaks.
    ```bash
    valgrind --leak-check=full ./robot_manager
    ```
    -   *Result:* Valgrind will report "definitely lost: 4,000 bytes".
4.  **Fix:** Put `delete[]` back. Run Valgrind again. "All heap blocks were freed".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Use of deleted function"
**Symptom:** Compiler error when trying to copy a `unique_ptr` or a class with deleted copy constructor.
**Cause:** You are trying to copy something that is unique.
**Solution:** Use `std::move()` to transfer ownership, or pass by reference (`const LidarSensor&`).

#### 2. Dangling Reference
**Symptom:** Segmentation Fault.
**Cause:** Returning a reference to a local variable, or using a raw pointer after the smart pointer has deleted the object.
**Solution:** Never return references to locals. Use `weak_ptr` if ownership is cyclic.

---

## ⚡ Optimization & Best Practices

### 1. Pass by Value vs Reference
-   **Small types (int, double):** Pass by value.
-   **Large types (vector, class):** Pass by `const reference` (`const std::vector<int>&`).
-   **Sinks (Function keeps the object):** Pass by value and `std::move` inside.

### 2. `std::array` vs `std::vector`
-   **`std::vector`:** Dynamic size, heap allocation. Use when size is unknown.
-   **`std::array`:** Fixed size, stack allocation. Zero overhead. Use for fixed buffers (e.g., `std::array<double, 3>` for XYZ).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `unique_ptr` and `shared_ptr`?
    *   **A:** `unique_ptr` has one owner (fast). `shared_ptr` has multiple owners (reference counted, slower).
2.  **Q:** Why is `auto` preferred in modern C++?
    *   **A:** It prevents type mismatch errors (e.g., float vs double) and makes code readable (no long iterator types).
3.  **Q:** What does `std::move` actually do?
    *   **A:** It casts an object to an R-value reference, allowing the move constructor to "steal" its resources instead of copying them.

### Challenge Task
**Task:** Custom Vector.
1.  Implement a simple `MyVector` class.
2.  Implement the **Rule of Five**: Destructor, Copy Constructor, Copy Assignment, Move Constructor, Move Assignment.
3.  Add print statements to see when copies vs moves happen.

---

## 📚 Further Reading & References
-   [Effective Modern C++ (Scott Meyers)](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) - The Bible of C++11/14.
-   [CppReference](https://en.cppreference.com/)

---

**Day 36 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
