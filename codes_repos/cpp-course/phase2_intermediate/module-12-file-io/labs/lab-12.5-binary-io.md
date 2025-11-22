# Lab 12.5: Binary File I/O

## Objective
Read and write raw binary data, bypassing text formatting.

## Instructions

### Step 1: Writing Binary
Create `binary_io.cpp`.
Open with `std::ios::binary`. Use `write()`.

```cpp
#include <iostream>
#include <fstream>

struct Player {
    int id;
    float health;
};

int main() {
    Player p = {1, 100.0f};
    
    std::ofstream out("player.dat", std::ios::binary);
    // Must cast address to char*
    out.write(reinterpret_cast<char*>(&p), sizeof(Player));
    out.close();
    
    return 0;
}
```

### Step 2: Reading Binary
Use `read()`.

```cpp
void loadPlayer() {
    Player p;
    std::ifstream in("player.dat", std::ios::binary);
    if(in.read(reinterpret_cast<char*>(&p), sizeof(Player))) {
        std::cout << "Loaded ID: " << p.id << "\n";
    }
}
```

### Step 3: Arrays
Write an array of integers.

```cpp
int arr[] = {10, 20, 30};
out.write(reinterpret_cast<char*>(arr), sizeof(arr));
```

## Challenges

### Challenge 1: Vector I/O
Write a `std::vector<int>` to binary.
Warning: You cannot write `&vector` because vector contains a pointer to heap memory!
You must write `vector.data()` and `vector.size() * sizeof(int)`.

### Challenge 2: String I/O
Strings are also dynamic. To save a string:
1. Write the length (int).
2. Write the characters.
To read:
1. Read length.
2. Resize string.
3. Read characters.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    // Write
    std::ofstream out("vec.bin", std::ios::binary);
    size_t size = v.size();
    out.write(reinterpret_cast<char*>(&size), sizeof(size)); // Write size first
    out.write(reinterpret_cast<char*>(v.data()), size * sizeof(int)); // Write data
    out.close();
    
    // Read
    std::vector<int> v2;
    std::ifstream in("vec.bin", std::ios::binary);
    size_t inSize;
    in.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
    v2.resize(inSize);
    in.read(reinterpret_cast<char*>(v2.data()), inSize * sizeof(int));
    
    for(int n : v2) std::cout << n << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::ios::binary`
✅ Used `write()` and `read()` with `reinterpret_cast`
✅ Serialized a struct
✅ Serialized a vector correctly (Challenge 1)

## Key Learnings
- Binary I/O is faster and smaller than text
- Never write pointers or classes with pointers (like `string`, `vector`) directly
- Always serialize size + data for dynamic structures

## Next Steps
Proceed to **Lab 12.6: Random Access** to jump around files.
