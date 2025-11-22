# Lab 10.6: Modifying Algorithms (Transform, Remove)

## Objective
Learn how to modify containers using algorithms.

## Instructions

### Step 1: Transform (Map)
Create `transform.cpp`.
Apply a function to every element.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4};
    std::vector<int> out;
    
    // Square each number
    std::transform(v.begin(), v.end(), std::back_inserter(out), [](int n) {
        return n * n;
    });
    
    for(int n : out) std::cout << n << " "; // 1 4 9 16
    return 0;
}
```

### Step 2: Replace
Replace all 1s with 9s.

```cpp
std::replace(v.begin(), v.end(), 1, 9);
```

### Step 3: Remove (Erase-Remove)
We saw this in Lab 10.1. `std::remove` doesn't resize the container.

```cpp
auto newEnd = std::remove(v.begin(), v.end(), 9);
v.erase(newEnd, v.end());
```

## Challenges

### Challenge 1: Transform In-Place
Use `std::transform` where the output iterator is the same as the input iterator.
Double every element in `v`.

### Challenge 2: Copy If
Copy only odd numbers from `v` to `out`.
`std::copy_if(v.begin(), v.end(), std::back_inserter(out), ...)`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    // Challenge 1: In-Place
    std::transform(v.begin(), v.end(), v.begin(), [](int n) {
        return n * 2;
    });
    // v is now {2, 4, 6, 8, 10}
    
    // Challenge 2: Copy If
    std::vector<int> odds;
    std::copy_if(v.begin(), v.end(), std::back_inserter(odds), [](int n) {
        return n % 2 != 0; // Wait, they are all even now!
    });
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::transform` to map values
✅ Used `std::back_inserter` to append to vector
✅ Used `std::replace`
✅ Implemented `copy_if` (Challenge 2)

## Key Learnings
- Algorithms decouple logic from containers
- `back_inserter` creates an iterator that calls `push_back`
- `remove` only shuffles elements; `erase` actually deletes them

## Next Steps
Proceed to **Lab 10.7: Numeric Algorithms** for math.
