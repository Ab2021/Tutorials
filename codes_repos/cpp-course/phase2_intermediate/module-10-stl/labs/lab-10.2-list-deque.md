# Lab 10.2: List and Deque

## Objective
Understand `std::list` (Linked List) and `std::deque` (Double Ended Queue) and when to use them.

## Instructions

### Step 1: std::list
Create `list_deque.cpp`.
Lists allow fast insertion/deletion anywhere, but slow random access (no `[]`).

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> l = {1, 2, 3};
    l.push_front(0); // Fast
    l.push_back(4);
    
    // Insert in middle
    auto it = l.begin();
    std::advance(it, 2); // Move iterator 2 steps (Slow!)
    l.insert(it, 99);
    
    for(int n : l) std::cout << n << " ";
    return 0;
}
```

### Step 2: std::deque
Deque ("Deck") is like a vector but allows fast push/pop at BOTH ends.
It is implemented as a list of small fixed-size arrays.

```cpp
#include <deque>

void testDeque() {
    std::deque<int> d = {1, 2, 3};
    d.push_front(0); // Fast! (Vector can't do this efficiently)
    d.push_back(4);
    
    std::cout << d[2] << std::endl; // Fast random access!
}
```

### Step 3: Comparison
- **Vector:** Fast random access, fast back insertion. Contiguous memory.
- **List:** Fast insert/delete anywhere. No random access. Node-based (cache unfriendly).
- **Deque:** Fast random access, fast front/back insertion. Not contiguous.

## Challenges

### Challenge 1: Splice
`std::list` has a unique method `splice`.
Create two lists. Move all elements from list2 into list1 without copying them.
`l1.splice(l1.end(), l2);`

### Challenge 2: Invalidating Pointers
Create a vector and a list.
Take a pointer to an element.
Add 1000 elements to both.
Check if the pointer is still valid.
(Vector pointer: Likely Invalidated due to realloc. List pointer: Valid).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <list>
#include <vector>

int main() {
    // Challenge 1
    std::list<int> l1 = {1, 2, 3};
    std::list<int> l2 = {4, 5, 6};
    
    l1.splice(l1.end(), l2); // O(1) operation
    
    std::cout << "L1 size: " << l1.size() << ", L2 size: " << l2.size() << "\n";
    
    // Challenge 2
    std::list<int> myList = {10};
    int* pList = &myList.front();
    
    std::vector<int> myVec = {10};
    int* pVec = &myVec.front();
    
    for(int i=0; i<1000; ++i) {
        myList.push_back(i);
        myVec.push_back(i);
    }
    
    std::cout << "List Ptr: " << *pList << " (Safe)\n";
    std::cout << "Vec Ptr: " << *pVec << " (Likely Garbage/Crash)\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::list` for front insertion
✅ Used `std::deque` for double-ended operations
✅ Used `splice` (Challenge 1)
✅ Demonstrated iterator invalidation differences (Challenge 2)

## Key Learnings
- Use `vector` by default (95% of cases)
- Use `deque` if you need to push/pop front AND back
- Use `list` only if you need to insert/remove in the middle frequently AND don't need random access

## Next Steps
Proceed to **Lab 10.3: Maps and Sets** for associative containers.
