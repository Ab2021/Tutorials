# Lab 12.2: Reading Line by Line

## Objective
Read files line-by-line using `std::getline`.

## Instructions

### Step 1: The Problem with `>>`
Create `lines.cpp`.
Create a file `data.txt` with:
```
Alice Smith
Bob Jones
```
Try reading with `>>`. It reads "Alice", then "Smith", treating them as separate items.

### Step 2: Using getline
Use `std::getline` to read until newline.

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream file("data.txt");
    std::string line;
    
    while (std::getline(file, line)) {
        std::cout << "Read line: " << line << std::endl;
    }
    return 0;
}
```

### Step 3: Custom Delimiter
Read comma-separated values (CSV).
`std::getline(file, token, ',')`

```cpp
std::string csvData = "apple,banana,cherry";
// (See Challenge 1)
```

## Challenges

### Challenge 1: CSV Parser
Create `data.csv`:
```
Name,Age,City
Alice,30,New York
Bob,25,London
```
Read and print each field separately.
*Hint: Read a line, then use `stringstream` on that line with `getline(ss, token, ',')`.*

### Challenge 2: Line Count
Write a utility that counts the number of lines in a file.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // Challenge 1: CSV
    std::ifstream file("data.csv");
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        
        while (std::getline(ss, token, ',')) {
            row.push_back(token);
        }
        
        for (const auto& col : row) std::cout << "[" << col << "] ";
        std::cout << "\n";
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::getline` for full lines
✅ Used custom delimiter
✅ Parsed CSV data (Challenge 1)
✅ Counted lines (Challenge 2)

## Key Learnings
- `>>` stops at whitespace
- `getline` stops at newline (default) or custom delimiter
- Combining `getline` with `stringstream` is powerful for parsing

## Next Steps
Proceed to **Lab 12.3: String Streams** to master in-memory formatting.
