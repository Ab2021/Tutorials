# Lab 10.3: Maps and Sets

## Objective
Use Associative Containers to store key-value pairs and unique elements.

## Instructions

### Step 1: std::set
Stores unique elements, sorted.
Create `maps_sets.cpp`.

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;
    s.insert(3);
    s.insert(1);
    s.insert(2);
    s.insert(3); // Duplicate ignored
    
    for(int n : s) std::cout << n << " "; // Prints 1 2 3 (Sorted!)
    
    if(s.contains(2)) std::cout << "\nFound 2"; // C++20
    // Pre-C++20: if(s.find(2) != s.end()) ...
    
    return 0;
}
```

### Step 2: std::map
Stores Key-Value pairs, sorted by Key.

```cpp
#include <map>
#include <string>

void testMap() {
    std::map<std::string, int> scores;
    scores["Alice"] = 50;
    scores["Bob"] = 80;
    scores["Alice"] = 60; // Updates value
    
    for(auto const& [key, val] : scores) { // Structured Binding (C++17)
        std::cout << key << ": " << val << "\n";
    }
}
```

### Step 3: Unordered (Hash Map)
`std::unordered_map` is faster (O(1)) but not sorted.

```cpp
#include <unordered_map>
std::unordered_map<std::string, int> hash;
```

## Challenges

### Challenge 1: Word Frequency
Read a string "apple banana apple orange banana apple".
Count occurrences of each word using a map.

### Challenge 2: Custom Key
Use a struct `Point {int x, y;}` as a key in `std::map`.
You must overload `operator<` for `Point` because `map` needs to sort keys.
(For `unordered_map`, you'd need a Hash function).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <map>
#include <sstream>
#include <string>

struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};

int main() {
    // Challenge 1
    std::string text = "apple banana apple orange banana apple";
    std::map<std::string, int> counts;
    std::stringstream ss(text);
    std::string word;
    
    while(ss >> word) {
        counts[word]++;
    }
    
    for(auto [w, c] : counts) std::cout << w << ": " << c << "\n";
    
    // Challenge 2
    std::map<Point, std::string> locations;
    locations[{1, 2}] = "Home";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `set` for unique sorted data
✅ Used `map` for key-value storage
✅ Iterated using structured binding
✅ Implemented word frequency counter (Challenge 1)
✅ Used custom struct as map key (Challenge 2)

## Key Learnings
- `map`/`set` are Red-Black Trees (O(log n))
- `unordered_map`/`unordered_set` are Hash Tables (O(1) avg)
- Keys in `map` must be comparable (`<`)

## Next Steps
Proceed to **Lab 10.4: Iterators** to traverse them manually.
