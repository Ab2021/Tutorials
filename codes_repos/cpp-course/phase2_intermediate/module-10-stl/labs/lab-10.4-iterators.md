# Lab 10.4: Iterators

## Objective
Understand how Iterators bridge Containers and Algorithms.

## Instructions

### Step 1: Basic Iteration
Create `iterators.cpp`.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {10, 20, 30, 40};
    
    // Explicit Iterator
    std::vector<int>::iterator it = v.begin();
    
    // Dereference
    std::cout << *it << std::endl; // 10
    
    // Increment
    it++;
    std::cout << *it << std::endl; // 20
    
    // Arithmetic (Random Access Iterator)
    it += 2;
    std::cout << *it << std::endl; // 40
    
    return 0;
}
```

### Step 2: Const Iterator
If you don't need to modify data, use `const_iterator` (or `cbegin()`).

```cpp
std::vector<int>::const_iterator cit = v.cbegin();
// *cit = 50; // Error: Read-only
```

### Step 3: Reverse Iterator
Traverse backwards.

```cpp
for(auto rit = v.rbegin(); rit != v.rend(); ++rit) {
    std::cout << *rit << " "; // 40 30 20 10
}
```

## Challenges

### Challenge 1: List Iterator
Try `it += 2` on a `std::list<int>::iterator`.
It will fail. List iterators are **Bidirectional**, not **Random Access**.
Use `std::advance(it, 2)` instead.

### Challenge 2: Insert via Iterator
Use `v.insert(it, 99)`.
Note that `insert` returns a *new* iterator to the inserted element.
The old iterator might be invalidated!

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <list>

int main() {
    std::list<int> l = {1, 2, 3};
    auto lit = l.begin();
    // lit += 2; // Error
    std::advance(lit, 2);
    std::cout << "List 3rd: " << *lit << "\n";
    
    std::vector<int> v = {1, 2, 3};
    auto it = v.begin();
    it = v.insert(it, 0); // Insert 0 at start
    // 'it' now points to 0.
    
    std::cout << "Vec front: " << *it << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used iterator to traverse vector
✅ Used reverse iterator
✅ Understood Random Access vs Bidirectional (Challenge 1)
✅ Handled iterator invalidation/return values (Challenge 2)

## Key Learnings
- Iterators behave like pointers
- Different containers have different iterator categories (Random Access, Bidirectional, Forward)
- Algorithms rely on these categories

## Next Steps
Proceed to **Lab 10.5: Basic Algorithms** to use them.
