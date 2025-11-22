# Lab 12.6: Random Access (Seek)

## Objective
Move the read/write position cursor within a file.

## Instructions

### Step 1: Seekg and Tellg (Input)
Create `seek.cpp`.
`seekg(offset, direction)` moves the cursor.
`tellg()` returns current position.

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ofstream out("letters.txt");
    out << "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    out.close();
    
    std::ifstream in("letters.txt");
    
    // Move to 5th character (index 4) from beginning
    in.seekg(4, std::ios::beg);
    std::cout << (char)in.get() << "\n"; // E
    
    // Move 3 chars back from current
    in.seekg(-3, std::ios::cur); // C
    
    // Move to 2nd char from end
    in.seekg(-2, std::ios::end);
    std::cout << (char)in.get() << "\n"; // Y
    
    return 0;
}
```

### Step 2: Seekp (Output)
`seekp` works for output streams.
Overwrite specific bytes.

```cpp
std::fstream fs("letters.txt", std::ios::in | std::ios::out);
fs.seekp(0, std::ios::beg);
fs.put('#'); // Overwrite 'A' with '#'
```

### Step 3: File Size
Calculate file size using seek.

```cpp
in.seekg(0, std::ios::end);
std::streampos size = in.tellg();
std::cout << "Size: " << size << " bytes\n";
```

## Challenges

### Challenge 1: Read Last Line
Write a function that efficiently reads the *last* line of a huge file without reading the whole file.
Start at end, seek back char by char until `\n` is found.

### Challenge 2: Patch Binary
Open `player.dat` from Lab 12.5.
Seek to the `health` field (offset = `sizeof(int)`).
Overwrite it with `0.0f`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <string>

std::string readLastLine(const std::string& filename) {
    std::ifstream in(filename, std::ios::ate); // Open at end
    if (!in) return "";
    
    std::streampos size = in.tellg();
    for(int i = 1; i <= size; ++i) {
        in.seekg(-i, std::ios::end);
        char c = in.get();
        if (c == '\n' && i > 1) { // Found newline (ignore trailing newline)
            std::string line;
            std::getline(in, line);
            return line;
        }
    }
    return ""; // File is one line
}

int main() {
    std::cout << readLastLine("letters.txt") << "\n";
    return 0;
}
```
</details>

## Success Criteria
✅ Used `seekg` and `seekp`
✅ Used `beg`, `cur`, `end`
✅ Calculated file size
✅ Implemented efficient last-line reader (Challenge 1)

## Key Learnings
- Random access allows efficient partial reads/writes
- `ios::ate` opens file at the end
- `tellg`/`tellp` gives current position

## Next Steps
Proceed to **Lab 12.7: Custom Stream Operators** to print your classes.
