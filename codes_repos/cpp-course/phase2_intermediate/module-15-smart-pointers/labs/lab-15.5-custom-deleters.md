# Lab 15.5: Custom Deleters

## Objective
Use custom deleters for non-standard cleanup.

## Instructions

### Step 1: Function Deleter
Create `custom_deleters.cpp`.

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

void fileDeleter(FILE* f) {
    if (f) {
        std::cout << "Closing file\n";
        fclose(f);
    }
}

int main() {
    std::unique_ptr<FILE, decltype(&fileDeleter)> file(
        fopen("test.txt", "w"),
        fileDeleter
    );
    
    if (file) {
        fprintf(file.get(), "Hello\n");
    }
    
    return 0;
} // File automatically closed
```

### Step 2: Lambda Deleter
```cpp
auto deleter = [](int* p) {
    std::cout << "Deleting: " << *p << "\n";
    delete p;
};

std::unique_ptr<int, decltype(deleter)> p(new int(42), deleter);
```

### Step 3: Shared Pointer Deleter
`shared_ptr` stores deleter without type erasure.

```cpp
std::shared_ptr<FILE> file2(
    fopen("test2.txt", "w"),
    [](FILE* f) { if (f) fclose(f); }
);
```

## Challenges

### Challenge 1: Array Deleter
Create a deleter for C-style arrays allocated with `malloc`.

### Challenge 2: Resource Handle
Wrap a Windows HANDLE or POSIX file descriptor with smart pointer.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <cstdlib>

int main() {
    // Challenge 1: malloc/free
    auto deleter = [](int* p) {
        std::cout << "Freeing malloc'd memory\n";
        free(p);
    };
    
    std::unique_ptr<int, decltype(deleter)> p(
        static_cast<int*>(malloc(sizeof(int))),
        deleter
    );
    
    *p = 42;
    std::cout << "Value: " << *p << "\n";
    
    // shared_ptr with custom deleter (no template parameter needed)
    std::shared_ptr<int> sp(
        static_cast<int*>(malloc(sizeof(int))),
        [](int* p) { free(p); }
    );
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used function as deleter
✅ Used lambda as deleter
✅ Used deleter with `shared_ptr`
✅ Created malloc/free deleter (Challenge 1)

## Key Learnings
- Custom deleters enable RAII for any resource
- `unique_ptr` requires deleter type in template
- `shared_ptr` stores deleter via type erasure

## Next Steps
Proceed to **Lab 15.6: Make Functions**.
