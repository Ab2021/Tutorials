# C++ Quick Reference Guide

## 📌 Basic Syntax

### Hello World
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### Compilation
```bash
# GCC/Clang
g++ -std=c++20 -Wall -Wextra file.cpp -o output
clang++ -std=c++20 -Wall -Wextra file.cpp -o output

# MSVC (Developer Command Prompt)
cl /EHsc /std:c++20 file.cpp
```

---

## 🔤 Variables and Types

### Basic Types
```cpp
int x = 42;                  // Integer
double y = 3.14;             // Floating point
bool flag = true;            // Boolean
char c = 'A';                // Character
std::string s = "Hello";     // String (include <string>)

auto z = 42;                 // Type inference (int)
const int MAX = 100;         // Constant
constexpr int SIZE = 50;     // Compile-time constant
```

### Type Conversions
```cpp
int i = 42;
double d = static_cast<double>(i);     // Explicit conversion
int j = (int)d;                        // C-style cast (avoid)
```

---

## 🔄 Control Flow

### Conditionals
```cpp
if (x > 0) {
    // ...
} else if (x < 0) {
    // ...
} else {
    // ...
}

// Ternary operator
int result = (x > 0) ? 1 : -1;

// Switch
switch (x) {
    case 1:
        // ...
        break;
    case 2:
        // ...
        break;
    default:
        // ...
}
```

### Loops
```cpp
// For loop
for (int i = 0; i < 10; ++i) {
    std::cout << i << "\n";
}

// Range-based for loop
std::vector<int> vec = {1, 2, 3, 4, 5};
for (const auto& elem : vec) {
    std::cout << elem << "\n";
}

// While loop
while (condition) {
    // ...
}

// Do-while loop
do {
    // ...
} while (condition);
```

---

## 📦 Functions

### Function Definition
```cpp
// Declaration
int add(int a, int b);

// Definition
int add(int a, int b) {
    return a + b;
}

// With default arguments
void greet(const std::string& name = "World") {
    std::cout << "Hello, " << name << "!\n";
}

// Function overloading
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
```

### Lambda Functions
```cpp
auto add = [](int a, int b) { return a + b; };
int sum = add(3, 4);  // 7

// Capture by value
int x = 10;
auto lambda1 = [x]() { return x * 2; };

// Capture by reference
auto lambda2 = [&x]() { x *= 2; };

// Capture all
auto lambda3 = [=]() { /* all by value */ };
auto lambda4 = [&]() { /* all by reference */ };
```

---

## 📚 Arrays and Containers

### Arrays
```cpp
// C-style array
int arr[5] = {1, 2, 3, 4, 5};

// std::array (C++11)
#include <array>
std::array<int, 5> arr = {1, 2, 3, 4, 5};
```

### Vectors
```cpp
#include <vector>

std::vector<int> vec;           // Empty
std::vector<int> vec2(10);      // 10 elements, default initialized
std::vector<int> vec3(10, 5);   // 10 elements, all 5
std::vector<int> vec4 = {1, 2, 3, 4, 5};  // Initializer list

vec.push_back(42);              // Add element
vec.pop_back();                 // Remove last
vec.size();                     // Number of elements
vec.empty();                    // Check if empty
vec.clear();                    // Remove all
vec[0];                         // Access (no bounds check)
vec.at(0);                      // Access (bounds checked)
```

### Maps
```cpp
#include <map>
#include <unordered_map>

std::map<std::string, int> age_map;  // Ordered
age_map["Alice"] = 30;
age_map["Bob"] = 25;

std::unordered_map<std::string, int> hash_map;  // Hash table
hash_map["Charlie"] = 35;

// Check if key exists
if (age_map.count("Alice")) { /* exists */ }
if (age_map.find("Alice") != age_map.end()) { /* exists */ }

// Iterate
for (const auto& [key, value] : age_map) {  // C++17 structured bindings
    std::cout << key << ": " << value << "\n";
}
```

### Sets
```cpp
#include <set>
#include <unordered_set>

std::set<int> s = {3, 1, 4, 1, 5};  // {1, 3, 4, 5} - duplicates removed
s.insert(2);
s.erase(3);
s.count(4);  // 1 if exists, 0 otherwise
```

---

## 🎯 Pointers and References

### Pointers
```cpp
int x = 42;
int* ptr = &x;      // Pointer to x
*ptr = 50;          // Dereference and assign
int y = *ptr;       // Read through pointer

int* null_ptr = nullptr;  // Null pointer (C++11)

// Dynamic allocation
int* p = new int(42);
delete p;           // Must delete!

int* arr = new int[10];
delete[] arr;       // Array deletion
```

### References
```cpp
int x = 42;
int& ref = x;       // Reference to x
ref = 50;           // x is now 50

// Cannot be null, must be initialized
// Cannot be reassigned to refer to another variable

// Const references (common for function parameters)
const int& cref = x;  // Cannot modify through cref
```

---

## 🧱 Classes and Objects

### Basic Class
```cpp
class Rectangle {
private:
    double width;
    double height;

public:
    // Constructor
    Rectangle(double w, double h) : width(w), height(h) {}
    
    // Member function
    double area() const {
        return width * height;
    }
    
    // Getter
    double getWidth() const { return width; }
    
    // Setter
    void setWidth(double w) { width = w; }
};

// Usage
Rectangle rect(10.0, 5.0);
double a = rect.area();
```

### Struct (public by default)
```cpp
struct Point {
    double x;
    double y;
    
    double distance() const {
        return std::sqrt(x*x + y*y);
    }
};

Point p = {3.0, 4.0};
```

---

## 🧠 Smart Pointers

### unique_ptr
```cpp
#include <memory>

std::unique_ptr<int> ptr = std::make_unique<int>(42);
// Automatic cleanup, exclusive ownership
// Cannot be copied, can be moved

std::unique_ptr<int[]> arr = std::make_unique<int[]>(10);
```

### shared_ptr
```cpp
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::shared_ptr<int> ptr2 = ptr1;  // Shared ownership
// Deleted when last shared_ptr is destroyed
```

### weak_ptr
```cpp
std::shared_ptr<int> sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;  // Doesn't increase ref count

// Must lock to access
if (auto sp2 = wp.lock()) {
    // sp2 is valid shared_ptr
}
```

---

## ⚠️ Error Handling

### Exceptions
```cpp
try {
    if (error_condition) {
        throw std::runtime_error("Error message");
    }
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << "\n";
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
} catch (...) {
    std::cerr << "Unknown exception\n";
}
```

### Optional (C++17)
```cpp
#include <optional>

std::optional<int> find_value(bool exists) {
    if (exists) {
        return 42;
    }
    return std::nullopt;
}

auto result = find_value(true);
if (result.has_value()) {
    std::cout << result.value() << "\n";
}

// Or
if (result) {
    std::cout << *result << "\n";
}
```

---

## 🔀 Templates

### Function Templates
```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int i = max(3, 5);
double d = max(3.14, 2.71);
```

### Class Templates
```cpp
template<typename T>
class Box {
private:
    T value;
public:
    Box(T v) : value(v) {}
    T get() const { return value; }
};

Box<int> int_box(42);
Box<std::string> str_box("Hello");
```

---

## 🔄 Move Semantics

```cpp
class MyClass {
    int* data;
public:
    // Move constructor
    MyClass(MyClass&& other) noexcept 
        : data(other.data) {
        other.data = nullptr;
    }
    
    // Move assignment
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};

MyClass obj1;
MyClass obj2 = std::move(obj1);  // Move, not copy
```

---

## 🧵 Concurrency

### Threads
```cpp
#include <thread>

void work() {
    std::cout << "Working...\n";
}

std::thread t(work);
t.join();  // Wait for completion

// With lambda
std::thread t2([]() {
    std::cout << "Lambda thread\n";
});
t2.join();
```

### Mutexes
```cpp
#include <mutex>

std::mutex mtx;
int shared_data = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared_data;
}  // Automatically unlocks
```

### Atomic Operations
```cpp
#include <atomic>

std::atomic<int> counter(0);
counter++;  // Thread-safe
counter.fetch_add(1);
```

---

## 📝 File I/O

### Reading Files
```cpp
#include <fstream>
#include <string>

std::ifstream file("input.txt");
if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }
    file.close();
}
```

### Writing Files
```cpp
std::ofstream file("output.txt");
if (file.is_open()) {
    file << "Hello, File!\n";
    file << 42 << "\n";
    file.close();
}
```

---

## 🛠️ CMake Basics

### Simple CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(my_app main.cpp utils.cpp)

# Add include directories
target_include_directories(my_app PRIVATE include/)

# Link libraries
target_link_libraries(my_app PRIVATE pthread)
```

### Build Commands
```bash
cmake -S . -B build                    # Configure
cmake --build build                    # Build
cmake --build build --target my_app    # Build specific target
cmake --install build                  # Install
```

---

## 🔍 Common Algorithms

```cpp
#include <algorithm>
#include <numeric>
#include <vector>

std::vector<int> vec = {5, 2, 8, 1, 9};

// Sort
std::sort(vec.begin(), vec.end());

// Find
auto it = std::find(vec.begin(), vec.end(), 8);

// Count
int count = std::count(vec.begin(), vec.end(), 2);

// Transform
std::transform(vec.begin(), vec.end(), vec.begin(),
               [](int x) { return x * 2; });

// Accumulate (sum)
int sum = std::accumulate(vec.begin(), vec.end(), 0);

// For each
std::for_each(vec.begin(), vec.end(),
              [](int x) { std::cout << x << " "; });
```

---

## 🆚 Rust vs C++ Quick Comparison

| Feature | C++ | Rust |
|---------|-----|------|
| **Variable** | `int x = 5;` | `let x = 5;` |
| **Mutable** | `int x = 5;` (default) | `let mut x = 5;` |
| **Reference** | `int& r = x;` | `let r = &x;` |
| **Mutable Ref** | `int& r = x;` | `let r = &mut x;` |
| **Pointer** | `int* p = &x;` | `let p: *const i32 = &x as *const i32;` (unsafe) |
| **Vector** | `std::vector<int>` | `Vec<i32>` |
| **String** | `std::string` | `String` |
| **Option** | `std::optional<T>` (C++17) | `Option<T>` |
| **Result** | Exceptions or `std::expected` (C++23) | `Result<T, E>` |
| **Smart Ptr** | `std::unique_ptr<T>` | `Box<T>` |
| **Shared Ptr** | `std::shared_ptr<T>` | `Rc<T>` / `Arc<T>` |
| **Thread** | `std::thread` | `std::thread::spawn` |
| **Mutex** | `std::mutex` | `std::sync::Mutex` |

---

## 📚 Must-Know Resources

- [cppreference.com](https://en.cppreference.com/) - Complete C++ reference
- [Compiler Explorer](https://godbolt.org/) - See assembly output
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/) - Best practices
- [cplusplus.com](http://www.cplusplus.com/) - Tutorials and reference

---

**Keep this reference handy as you progress through the course!** 🚀
