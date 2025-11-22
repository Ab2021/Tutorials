# Lab 10.1: Vector Deep Dive

## Objective
Master `std::vector`, the workhorse of C++. Understand capacity, size, and performance.

## Instructions

### Step 1: Basic Operations
Create `vector_basics.cpp`.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3};
    v.push_back(4);
    v.push_back(5);
    
    // Access
    std::cout << "Element at 2: " << v[2] << std::endl;
    std::cout << "Front: " << v.front() << ", Back: " << v.back() << std::endl;
    
    return 0;
}
```

### Step 2: Capacity vs Size
Vectors grow dynamically. This involves reallocating memory.

```cpp
void printStats(const std::vector<int>& v) {
    std::cout << "Size: " << v.size() << ", Capacity: " << v.capacity() << std::endl;
}

// In main:
std::vector<int> v2;
for(int i=0; i<20; ++i) {
    v2.push_back(i);
    printStats(v2);
}
```
*Observe how capacity doubles (usually) when size exceeds it.*

### Step 3: Reserve
Avoid reallocations by reserving memory upfront.

```cpp
std::vector<int> v3;
v3.reserve(100); // Allocate for 100 ints immediately
// Now push_back won't reallocate until > 100
```

## Challenges

### Challenge 1: Emplace Back
Use `emplace_back` instead of `push_back` for a vector of objects.
It constructs the object *in place*, avoiding a copy/move.
Create a struct `Item` that prints when constructed/copied, and test it.

### Challenge 2: Erase-Remove Idiom (Pre-C++20)
Remove all even numbers from a vector.
1. `std::remove_if` moves elements to end.
2. `v.erase` chops off the end.
(In C++20, use `std::erase_if`).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

struct Item {
    Item(int x) { std::cout << "Construct " << x << "\n"; }
    Item(const Item&) { std::cout << "Copy\n"; }
};

int main() {
    // Challenge 1
    std::vector<Item> items;
    std::cout << "Push:\n";
    items.push_back(Item(1)); // Construct + Move/Copy
    
    std::cout << "Emplace:\n";
    items.emplace_back(2); // Construct in place
    
    // Challenge 2
    std::vector<int> nums = {1, 2, 3, 4, 5, 6};
    // Move evens to end, return iterator to new end
    auto newEnd = std::remove_if(nums.begin(), nums.end(), [](int n){ return n % 2 == 0; });
    // Erase garbage at end
    nums.erase(newEnd, nums.end());
    
    for(int n : nums) std::cout << n << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `push_back` and element access
✅ Observed Capacity growth
✅ Used `reserve` to optimize
✅ Understood `emplace_back` vs `push_back` (Challenge 1)
✅ Implemented Erase-Remove (Challenge 2)

## Key Learnings
- `vector` is a dynamic array
- Reallocations are expensive; use `reserve` if size is known
- `emplace_back` is more efficient for objects

## Next Steps
Proceed to **Lab 10.2: List and Deque** for alternative sequences.
