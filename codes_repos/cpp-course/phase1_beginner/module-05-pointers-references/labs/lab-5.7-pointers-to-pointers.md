# Lab 5.7: Pointers to Pointers

## Objective
Understand multiple levels of indirection (`**`).

## Instructions

### Step 1: Double Pointer
Create `double_ptr.cpp`.

```cpp
#include <iostream>

int main() {
    int val = 100;
    int* ptr = &val;
    int** ptr2 = &ptr;
    
    std::cout << "Val: " << val << std::endl;
    std::cout << "*ptr: " << *ptr << std::endl;
    std::cout << "**ptr2: " << **ptr2 << std::endl;
    
    return 0;
}
```

### Step 2: Modifying via Double Pointer
Change `val` using `ptr2`.
```cpp
**ptr2 = 200;
```

### Step 3: Changing the Pointer
Change where `ptr` points using `ptr2`.
```cpp
int other = 500;
*ptr2 = &other; // Changes ptr
std::cout << "*ptr is now: " << *ptr << std::endl; // 500
```

## Challenges

### Challenge 1: Array of Strings (C-style)
`char* argv[]` is essentially `char**`.
Create an array of C-strings:
```cpp
const char* names[] = {"Alice", "Bob", "Charlie"};
const char** p = names;
```
Iterate using `p`.

### Challenge 2: Dynamic 2D Array (Preview)
Simulate a 2D array using pointers.
```cpp
int rows = 3;
int cols = 3;
int** table = new int*[rows];
for(int i=0; i<rows; ++i) table[i] = new int[cols];
```
(Don't forget to delete it later!)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 1
    const char* names[] = {"Alice", "Bob", "Charlie"};
    const char** p = names;
    
    for (int i = 0; i < 3; ++i) {
        std::cout << *(p + i) << std::endl;
    }
    
    // Challenge 2
    int** table = new int*[3];
    for(int i=0; i<3; ++i) {
        table[i] = new int[3];
        table[i][0] = i; // Set some values
    }
    
    std::cout << "Table[1][0]: " << table[1][0] << std::endl;
    
    // Cleanup
    for(int i=0; i<3; ++i) delete[] table[i];
    delete[] table;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `int**` to access value
✅ Used `int**` to modify `int*`
✅ Iterated array of strings (Challenge 1)
✅ Created dynamic 2D array (Challenge 2)

## Key Learnings
- `type**` holds address of `type*`
- Used for modifying pointers in functions
- Used for dynamic 2D arrays (though `std::vector` is better)

## Next Steps
Proceed to **Lab 5.8: Dynamic Memory** to manage the heap.
