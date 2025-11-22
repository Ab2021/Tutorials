# Lab 2.8: Type Size and Alignment

## Objective
Understand how data types are laid out in memory, including padding and alignment.

## Instructions

### Step 1: Sizeof and Alignof
Create `alignment.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "int: size=" << sizeof(int) 
              << " align=" << alignof(int) << std::endl;
    std::cout << "double: size=" << sizeof(double) 
              << " align=" << alignof(double) << std::endl;
    std::cout << "char: size=" << sizeof(char) 
              << " align=" << alignof(char) << std::endl;
    return 0;
}
```

### Step 2: Struct Padding
Define a struct with mixed types:

```cpp
struct BadStruct {
    char c;     // 1 byte
    double d;   // 8 bytes
    int i;      // 4 bytes
};
// Expected size: 1 + 8 + 4 = 13?
// Actual size will likely be 24 due to padding!
```

Print its size:
```cpp
std::cout << "BadStruct size: " << sizeof(BadStruct) << std::endl;
```

### Step 3: Reordering for Efficiency
Create a `GoodStruct` with the same members but reordered to minimize padding (largest to smallest usually works best).

```cpp
struct GoodStruct {
    double d;   // 8 bytes
    int i;      // 4 bytes
    char c;     // 1 byte
    // Padding: 3 bytes to reach multiple of 8 (if double alignment is 8)
};
```

Print its size.

## Challenges

### Challenge 1: Manual Packing
Use `#pragma pack` to force 1-byte alignment.
```cpp
#pragma pack(push, 1)
struct PackedStruct {
    char c;
    double d;
    int i;
};
#pragma pack(pop)
```
Check the size. It should be exactly 13. Warning: Accessing unaligned data can be slow or crash on some architectures!

### Challenge 2: Offsetof
Use `offsetof` macro (include `<cstddef>`) to see exactly where members are located.
```cpp
#include <cstddef>
std::cout << "Offset of d: " << offsetof(BadStruct, d) << std::endl;
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstddef>

struct BadStruct {
    char c;   // 1 byte + 7 padding
    double d; // 8 bytes
    int i;    // 4 bytes + 4 padding
}; // Total 24

struct GoodStruct {
    double d; // 8 bytes
    int i;    // 4 bytes
    char c;   // 1 byte + 3 padding
}; // Total 16

int main() {
    std::cout << "BadStruct: " << sizeof(BadStruct) << std::endl;
    std::cout << "Offset c: " << offsetof(BadStruct, c) << std::endl;
    std::cout << "Offset d: " << offsetof(BadStruct, d) << std::endl;
    std::cout << "Offset i: " << offsetof(BadStruct, i) << std::endl;
    
    std::cout << "GoodStruct: " << sizeof(GoodStruct) << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed difference between `sizeof` and sum of members
✅ Optimized struct size by reordering
✅ Used `#pragma pack` (Challenge 1)
✅ Used `offsetof` to visualize padding (Challenge 2)

## Key Learnings
- Memory alignment requirements
- Struct padding
- Importance of member ordering
- Performance vs size tradeoffs (packed structs)

## Next Steps
Proceed to **Lab 2.9: Numeric Limits** to handle edge cases.
