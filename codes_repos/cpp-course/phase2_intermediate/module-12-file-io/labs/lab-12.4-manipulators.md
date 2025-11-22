# Lab 12.4: Stream Manipulators

## Objective
Control the format of input and output (hex, precision, width).

## Instructions

### Step 1: Boolean and Hex
Create `manipulators.cpp`.
Include `<iomanip>`.

```cpp
#include <iostream>
#include <iomanip>

int main() {
    bool b = true;
    std::cout << "Default: " << b << "\n"; // 1
    std::cout << "Boolalpha: " << std::boolalpha << b << "\n"; // true
    
    int x = 255;
    std::cout << "Hex: " << std::hex << x << "\n"; // ff
    std::cout << "Dec: " << std::dec << x << "\n"; // 255
    
    return 0;
}
```

### Step 2: Width and Fill
Align output in columns.

```cpp
std::cout << std::setw(10) << "Name" << std::setw(5) << "Age" << "\n";
std::cout << std::setw(10) << "Alice" << std::setw(5) << 30 << "\n";
std::cout << std::setfill('-') << std::setw(15) << "" << std::setfill(' ') << "\n";
```

### Step 3: Floating Point Precision
Control decimal places.

```cpp
double pi = 3.1415926535;
std::cout << std::fixed << std::setprecision(2) << pi << "\n"; // 3.14
```

## Challenges

### Challenge 1: Currency Format
Print a double as currency: `$1,234.56`.
(Note: Adding commas is tricky in standard C++, just focus on `fixed` and `precision` for now).

### Challenge 2: Hex Dump
Read a string and print the ASCII value of each character in Hex, separated by spaces.
`Hello` -> `48 65 6c 6c 6f`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <iomanip>
#include <string>

int main() {
    // Challenge 2: Hex Dump
    std::string s = "Hello";
    for (unsigned char c : s) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)c << " ";
    }
    std::cout << std::dec << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `boolalpha`, `hex`, `dec`
✅ Used `setw` and `setfill` for alignment
✅ Used `fixed` and `setprecision`
✅ Created a Hex Dump (Challenge 2)

## Key Learnings
- Manipulators modify the stream state
- Some are persistent (`hex`, `boolalpha`), some apply only to next item (`setw`)
- Essential for creating readable console tables

## Next Steps
Proceed to **Lab 12.5: Binary File I/O** for raw data.
