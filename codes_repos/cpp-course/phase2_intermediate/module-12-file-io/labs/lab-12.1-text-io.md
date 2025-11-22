# Lab 12.1: Basic Text I/O

## Objective
Learn how to write to and read from text files.

## Instructions

### Step 1: Writing to a File
Create `text_io.cpp`.
Use `std::ofstream` to create a file.

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ofstream outFile("hello.txt");
    if (!outFile) {
        std::cerr << "Error creating file\n";
        return 1;
    }
    
    outFile << "Hello World\n";
    outFile << "C++ File I/O is easy\n";
    outFile << 12345 << "\n";
    
    outFile.close(); // Optional (destructor does it)
    std::cout << "File written.\n";
    
    return 0;
}
```

### Step 2: Reading from a File
Use `std::ifstream`.

```cpp
void readFile() {
    std::ifstream inFile("hello.txt");
    if (!inFile) {
        std::cerr << "File not found\n";
        return;
    }
    
    std::string word;
    while (inFile >> word) { // Reads word by word (stops at space)
        std::cout << word << std::endl;
    }
}
```

### Step 3: Append Mode
Open file in append mode to add data without overwriting.

```cpp
std::ofstream appendFile("hello.txt", std::ios::app);
appendFile << "Appended line\n";
```

## Challenges

### Challenge 1: User Input to File
Ask the user for their name and age. Save it to `user_info.txt`.

### Challenge 2: Copy File
Write a function `copyFile(source, dest)` that reads content from source and writes it to dest.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <string>

void copyFile(const std::string& src, const std::string& dest) {
    std::ifstream in(src);
    std::ofstream out(dest);
    
    if (in && out) {
        // Efficient copy using stream buffers
        out << in.rdbuf(); 
        std::cout << "Copied " << src << " to " << dest << "\n";
    } else {
        std::cerr << "Copy failed\n";
    }
}

int main() {
    // Challenge 1
    std::ofstream out("user_info.txt");
    std::string name;
    int age;
    std::cout << "Name: "; std::cin >> name;
    std::cout << "Age: "; std::cin >> age;
    out << "Name: " << name << "\nAge: " << age << "\n";
    out.close();
    
    // Challenge 2
    copyFile("user_info.txt", "user_copy.txt");
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and wrote to a file
✅ Read from a file
✅ Used Append mode
✅ Implemented file copy (Challenge 2)

## Key Learnings
- `ofstream` for output, `ifstream` for input
- `>>` operator reads whitespace-delimited words
- `std::ios::app` appends to end of file

## Next Steps
Proceed to **Lab 12.2: Reading Line by Line** for better text handling.
