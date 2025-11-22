# Lab 10.5: Basic Algorithms (Sort, Find)

## Objective
Use standard algorithms from `<algorithm>` to process data.

## Instructions

### Step 1: Sort
Create `algorithms_basic.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {5, 2, 8, 1, 9};
    
    // Sort ascending
    std::sort(v.begin(), v.end());
    
    for(int n : v) std::cout << n << " ";
    std::cout << "\n";
    
    // Sort descending (Lambda)
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;
    });
    
    return 0;
}
```

### Step 2: Find
Search for an element.

```cpp
auto it = std::find(v.begin(), v.end(), 8);
if (it != v.end()) {
    std::cout << "Found 8 at index: " << std::distance(v.begin(), it) << std::endl;
} else {
    std::cout << "Not found\n";
}
```

### Step 3: Count
Count occurrences.

```cpp
int c = std::count(v.begin(), v.end(), 5);
```

## Challenges

### Challenge 1: Binary Search
`std::find` is O(N). If sorted, use `std::binary_search` (O(log N)).
Check if 9 exists using binary search.

### Challenge 2: Sort Structs
Create a `struct Person { string name; int age; };`.
Sort a vector of Persons by age.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Person {
    std::string name;
    int age;
};

int main() {
    std::vector<int> v = {1, 2, 5, 8, 9};
    
    // Challenge 1
    bool exists = std::binary_search(v.begin(), v.end(), 8);
    std::cout << "Exists? " << exists << "\n";
    
    // Challenge 2
    std::vector<Person> people = {{"Bob", 30}, {"Alice", 25}, {"Charlie", 35}};
    std::sort(people.begin(), people.end(), [](const Person& a, const Person& b) {
        return a.age < b.age;
    });
    
    for(auto& p : people) std::cout << p.name << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::sort` with and without comparator
✅ Used `std::find`
✅ Used `std::binary_search` (Challenge 1)
✅ Sorted custom objects (Challenge 2)

## Key Learnings
- `<algorithm>` contains highly optimized loops
- Never write your own sort (unless for learning)
- Lambdas make custom sorting easy

## Next Steps
Proceed to **Lab 10.6: Modifying Algorithms** to transform data.
