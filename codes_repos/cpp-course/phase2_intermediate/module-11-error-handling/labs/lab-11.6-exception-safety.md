# Lab 11.6: Exception Safety Levels

## Objective
Write code that guarantees specific behavior even when exceptions occur.

## Instructions

### Step 1: Basic Guarantee
Ensure no memory leaks and invariants are preserved.
Create `safety.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <memory>

class Buffer {
    std::vector<int> data;
public:
    void append(int val) {
        data.push_back(val); // Strong guarantee (vector provides it)
    }
};
```

### Step 2: Strong Guarantee (Transactional)
Either the operation succeeds completely, or it has no effect.

```cpp
class User {
    std::string name;
    int id;
public:
    void update(std::string newName, int newId) {
        // Problem: If newId assignment fails (unlikely for int, but imagine a complex type),
        // name might already be changed.
        
        // Fix: Copy and Swap Idiom
        std::string tempName = newName; // Might throw
        int tempId = newId;
        
        // No-throw swap
        std::swap(name, tempName);
        std::swap(id, tempId);
    }
};
```

### Step 3: No-Throw Guarantee
Functions that promise not to throw.

```cpp
void safeSwap(int& a, int& b) noexcept {
    int temp = a;
    a = b;
    b = temp;
}
```

## Challenges

### Challenge 1: Implement Strong Guarantee
Create a class `Playlist` with `vector<string> songs`.
Implement `void addSong(string s)` that throws if the song is "Justin Bieber".
Ensure that if it throws, the playlist is unchanged. (Vector push_back already does this, so try to do it manually with a raw array to see the difficulty, or just verify vector behavior).

### Challenge 2: Broken Guarantee
Write a function that modifies a global variable, then throws.
Verify the global variable is left in a changed state (Basic guarantee failed if invariant broken, or just poor design).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <string>

class Playlist {
    std::vector<std::string> songs;
public:
    void addSong(const std::string& s) {
        if (s == "Justin Bieber") throw std::runtime_error("Taste Error");
        songs.push_back(s); // Strong guarantee
    }
    
    void print() { for(auto& s : songs) std::cout << s << " "; std::cout << "\n"; }
};

int globalState = 0;
void broken() {
    globalState = 1;
    throw 1;
}

int main() {
    Playlist p;
    try {
        p.addSong("Queen");
        p.addSong("Justin Bieber");
    } catch(...) {
        std::cout << "Caught error\n";
    }
    p.print(); // Should contain Queen, but NOT Justin Bieber
    
    try { broken(); } catch(...) {}
    std::cout << "Global: " << globalState << " (Changed!)\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood Basic vs Strong guarantee
✅ Implemented Copy-and-Swap idiom
✅ Verified Strong Guarantee behavior (Challenge 1)
✅ Observed side effects of exceptions (Challenge 2)

## Key Learnings
- **Strong Guarantee** is ideal but can be expensive (requires copying)
- **Basic Guarantee** is the minimum requirement
- **No-Throw** is required for destructors and move operations

## Next Steps
Proceed to **Lab 11.7: The noexcept Specifier** to optimize performance.
