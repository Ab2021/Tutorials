# Lab 12.9: Filesystem Library (C++17)

## Objective
Use `std::filesystem` to manage files and directories portably.

## Instructions

### Step 1: Paths
Create `fs_demo.cpp`.
Include `<filesystem>`. Namespace alias is common.

```cpp
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

int main() {
    fs::path p = "subdir/test.txt";
    std::cout << "Filename: " << p.filename() << "\n";
    std::cout << "Extension: " << p.extension() << "\n";
    std::cout << "Parent: " << p.parent_path() << "\n";
    return 0;
}
```

### Step 2: Operations
Check existence, create directories.

```cpp
if (!fs::exists("sandbox")) {
    fs::create_directory("sandbox");
    std::cout << "Created sandbox\n";
}

fs::copy("hello.txt", "sandbox/hello_copy.txt", fs::copy_options::overwrite_existing);
```

### Step 3: Directory Iterator
List files in a folder.

```cpp
for (const auto& entry : fs::directory_iterator(".")) {
    std::cout << entry.path() << (entry.is_directory() ? " [DIR]" : "") << "\n";
}
```

## Challenges

### Challenge 1: Recursive Size
Write a function that calculates the total size of a directory (recursively).
Use `fs::recursive_directory_iterator` and `fs::file_size(entry)`.

### Challenge 2: Extension Filter
List only `.txt` files in the current directory.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

uintmax_t dirSize(const fs::path& p) {
    uintmax_t size = 0;
    if (fs::exists(p) && fs::is_directory(p)) {
        for (const auto& entry : fs::recursive_directory_iterator(p)) {
            if (fs::is_regular_file(entry)) {
                size += fs::file_size(entry);
            }
        }
    }
    return size;
}

int main() {
    // Challenge 2
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.path().extension() == ".txt") {
            std::cout << "Text file: " << entry.path() << "\n";
        }
    }
    
    std::cout << "Total Size: " << dirSize(".") << " bytes\n";
    return 0;
}
```
</details>

## Success Criteria
✅ Used `fs::path` for parsing
✅ Created directories and copied files
✅ Iterated over directories
✅ Calculated recursive size (Challenge 1)

## Key Learnings
- `std::filesystem` replaces platform-specific code (Windows API / POSIX)
- `path` handles separators (`/` vs `\`) automatically
- Powerful iterators for traversing trees

## Next Steps
Proceed to **Lab 12.10: Config File Parser** to build a real tool.
