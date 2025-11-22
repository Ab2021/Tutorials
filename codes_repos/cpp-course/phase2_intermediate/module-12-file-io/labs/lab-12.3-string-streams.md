# Lab 12.3: String Streams

## Objective
Use `std::stringstream` to format strings and parse data in memory.

## Instructions

### Step 1: Formatting (Output)
Create `string_streams.cpp`.
Build a string from variables.

```cpp
#include <iostream>
#include <sstream>

int main() {
    int id = 42;
    std::string name = "Robot";
    
    std::stringstream ss;
    ss << "ID: " << id << ", Name: " << name;
    
    std::string result = ss.str();
    std::cout << result << std::endl;
    
    return 0;
}
```

### Step 2: Parsing (Input)
Extract types from a string.

```cpp
std::string input = "100 3.14 Hello";
std::stringstream ss2(input);

int i;
double d;
std::string s;

ss2 >> i >> d >> s;
std::cout << i << " | " << d << " | " << s << std::endl;
```

### Step 3: Type Conversion (toString)
Write a generic function to convert anything to string.

```cpp
template <typename T>
std::string toString(const T& val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}
```

## Challenges

### Challenge 1: Word Count
Count words in a sentence using `stringstream`.
`while(ss >> word) count++;`

### Challenge 2: Validation
Check if a string is a valid integer.
Try to extract an int. Check if `ss.fail()` and if `ss.eof()` (to ensure no trailing garbage).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <sstream>
#include <string>

bool isInteger(const std::string& s) {
    std::stringstream ss(s);
    int i;
    ss >> i;
    // Check if extraction failed OR if there are leftover chars
    return !ss.fail() && ss.eof();
}

int main() {
    std::string text = "Hello world from C++";
    std::stringstream ss(text);
    std::string word;
    int count = 0;
    while(ss >> word) count++;
    std::cout << "Words: " << count << "\n";
    
    std::cout << "Is '123' int? " << isInteger("123") << "\n";
    std::cout << "Is '123a' int? " << isInteger("123a") << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `stringstream` for formatting
✅ Used `stringstream` for parsing
✅ Implemented generic `toString`
✅ Implemented integer validation (Challenge 2)

## Key Learnings
- `stringstream` behaves like `cin`/`cout` but for strings
- Essential for converting types (int <-> string)
- Useful for parsing space-separated data

## Next Steps
Proceed to **Lab 12.4: Stream Manipulators** to format output.
