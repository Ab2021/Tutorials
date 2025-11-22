# Lab 12.8: Error Handling in Streams

## Objective
Detect and recover from stream errors (EOF, Fail, Bad).

## Instructions

### Step 1: Stream States
Create `stream_errors.cpp`.
- `good()`: All good.
- `eof()`: End of File.
- `fail()`: Logical error (e.g., read string into int).
- `bad()`: Serious error (e.g., disk full).

```cpp
#include <iostream>
#include <limits>

int main() {
    int x;
    std::cout << "Enter int: ";
    std::cin >> x;
    
    if (std::cin.fail()) {
        std::cout << "That's not an integer!\n";
    }
    return 0;
}
```

### Step 2: Recovery
If `cin` fails, it stays in fail state and refuses further input.
You must:
1. `cin.clear()`: Reset flags.
2. `cin.ignore(...)`: Discard bad input.

```cpp
if (std::cin.fail()) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Recovered. Try again.\n";
}
```

### Step 3: Exceptions
Force streams to throw exceptions on failure.

```cpp
std::cin.exceptions(std::ios::failbit | std::ios::badbit);
try {
    std::cin >> x;
} catch (const std::ios_base::failure& e) {
    std::cerr << "Caught stream error\n";
}
```

## Challenges

### Challenge 1: Robust Input Loop
Write a function `int getInt()` that loops until the user provides a valid integer.

### Challenge 2: File Check
Open a non-existent file. Check `fail()`.
Then try to write to a read-only file. Check `fail()`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <limits>

int getInt() {
    int x;
    while (true) {
        std::cout << "Enter number: ";
        if (std::cin >> x) return x;
        
        std::cout << "Invalid input.\n";
        std::cin.clear();
        std::cin.ignore(1000, '\n');
    }
}

int main() {
    int val = getInt();
    std::cout << "You entered: " << val << "\n";
    return 0;
}
```
</details>

## Success Criteria
✅ Checked stream state flags
✅ Recovered from bad input using `clear` and `ignore`
✅ Enabled stream exceptions
✅ Implemented robust input loop (Challenge 1)

## Key Learnings
- Streams have internal state
- `fail()` is common for parsing errors
- `clear()` is mandatory to reuse a failed stream
- `ignore()` flushes the buffer

## Next Steps
Proceed to **Lab 12.9: Filesystem Library** for path manipulation.
