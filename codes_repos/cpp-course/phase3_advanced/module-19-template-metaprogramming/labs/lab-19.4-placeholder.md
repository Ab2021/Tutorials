# Lab 19.4: Tag Dispatch

## Objective
Use tag dispatch as an alternative to SFINAE for algorithm selection.

## Instructions

### Step 1: Iterator Category Tags
Create `tag_dispatch.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <iterator>

// Implementation for random access iterators
template<typename Iter>
void advanceImpl(Iter& it, int n, std::random_access_iterator_tag) {
    std::cout << "Random access advance\n";
    it += n;
}

// Implementation for bidirectional iterators
template<typename Iter>
void advanceImpl(Iter& it, int n, std::bidirectional_iterator_tag) {
    std::cout << "Bidirectional advance\n";
    if (n >= 0) {
        while (n--) ++it;
    } else {
        while (n++) --it;
    }
}

// Dispatch function
template<typename Iter>
void advance(Iter& it, int n) {
    advanceImpl(it, n, 
        typename std::iterator_traits<Iter>::iterator_category());
}
```

### Step 2: Custom Tags
```cpp
struct fast_tag {};
struct slow_tag {};

template<typename T>
struct algorithm_tag {
    using type = std::conditional_t<
        sizeof(T) <= 8,
        fast_tag,
        slow_tag
    >;
};
```

## Challenges

### Challenge 1: Distance Calculator
Implement `distance()` using tag dispatch for different iterator categories.

### Challenge 2: Sort Dispatcher
Create a sort function that dispatches to different algorithms based on container properties.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>

// Challenge 1: Distance calculator

template<typename Iter>
typename std::iterator_traits<Iter>::difference_type
distanceImpl(Iter first, Iter last, std::random_access_iterator_tag) {
    std::cout << "O(1) distance calculation\n";
    return last - first;
}

template<typename Iter>
typename std::iterator_traits<Iter>::difference_type
distanceImpl(Iter first, Iter last, std::input_iterator_tag) {
    std::cout << "O(n) distance calculation\n";
    typename std::iterator_traits<Iter>::difference_type n = 0;
    while (first != last) {
        ++first;
        ++n;
    }
    return n;
}

template<typename Iter>
auto distance(Iter first, Iter last) {
    return distanceImpl(first, last,
        typename std::iterator_traits<Iter>::iterator_category());
}

// Challenge 2: Sort dispatcher

struct quick_sort_tag {};
struct insertion_sort_tag {};

template<typename Container>
struct sort_tag {
    using type = std::conditional_t<
        (sizeof(typename Container::value_type) > 64),
        insertion_sort_tag,
        quick_sort_tag
    >;
};

template<typename Container>
void sortImpl(Container& c, quick_sort_tag) {
    std::cout << "Using quick sort\n";
    std::sort(c.begin(), c.end());
}

template<typename Container>
void sortImpl(Container& c, insertion_sort_tag) {
    std::cout << "Using insertion sort\n";
    // Simple insertion sort
    for (auto it = c.begin(); it != c.end(); ++it) {
        auto key = *it;
        auto j = it;
        while (j != c.begin() && *(--j) > key) {
            *(std::next(j)) = *j;
        }
        if (j != it) *(std::next(j)) = key;
    }
}

template<typename Container>
void smartSort(Container& c) {
    sortImpl(c, typename sort_tag<Container>::type());
}

int main() {
    // Challenge 1
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    std::cout << "Vector distance: " << distance(vec.begin(), vec.end()) << "\n";
    std::cout << "List distance: " << distance(lst.begin(), lst.end()) << "\n";
    
    // Challenge 2
    std::vector<int> data = {5, 2, 8, 1, 9};
    smartSort(data);
    
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented tag dispatch pattern
✅ Used iterator category tags
✅ Created custom tag types
✅ Implemented distance calculator (Challenge 1)
✅ Created sort dispatcher (Challenge 2)

## Key Learnings
- Tag dispatch is cleaner than SFINAE for some cases
- Tags enable compile-time algorithm selection
- Iterator categories are a standard use of tags
- Custom tags can represent any property

## Next Steps
Proceed to **Lab 19.5: Compile-Time Algorithms**.
