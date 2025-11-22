# Lab 9.5: Non-Type Template Parameters

## Objective
Pass values (like integers) to templates instead of just types.

## Instructions

### Step 1: Static Array
Create `non_type.cpp`. Create a wrapper for a fixed-size array.

```cpp
#include <iostream>

template <typename T, int Size>
class Array {
    T data[Size]; // Size is known at compile time!
public:
    int getSize() { return Size; }
    
    T& operator[](int index) {
        return data[index];
    }
};
```

### Step 2: Usage
```cpp
int main() {
    Array<int, 5> arr;
    arr[0] = 10;
    std::cout << "Size: " << arr.getSize() << std::endl;
    
    // Array<int, 5> and Array<int, 10> are DIFFERENT types!
    return 0;
}
```

### Step 3: Why?
Unlike `std::vector` (dynamic size), this allocates on the **Stack**. It's faster but size must be constant.

## Challenges

### Challenge 1: Bounds Checking
Add bounds checking to `operator[]`. Throw exception or print error if index >= Size.

### Challenge 2: Default Value
Set default size to 10.
`template <typename T, int Size = 10>`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <stdexcept>

template <typename T, int Size = 10>
class Array {
    T data[Size];
public:
    T& operator[](int index) {
        if (index < 0 || index >= Size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    int size() const { return Size; }
};

int main() {
    Array<int, 5> arr;
    try {
        arr[10] = 5; // Throws
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    
    Array<double> defaultArr; // Size 10
    std::cout << "Default size: " << defaultArr.size() << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used integer as template parameter
✅ Created fixed-size stack array wrapper
✅ Implemented bounds checking (Challenge 1)
✅ Used default non-type argument (Challenge 2)

## Key Learnings
- Non-type parameters must be compile-time constants (integers, enums, pointers)
- Enables creating highly optimized, fixed-size structures
- `std::array` is the standard library equivalent

## Next Steps
Proceed to **Lab 9.6: Default Template Arguments** to refine your APIs.
