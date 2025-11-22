# Lab 10.7: Numeric Algorithms

## Objective
Use algorithms specifically designed for numeric operations (`<numeric>`).

## Instructions

### Step 1: Accumulate (Sum/Fold)
Create `numeric.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v = {1, 2, 3, 4};
    
    // Sum
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "Sum: " << sum << std::endl;
    
    // Product
    int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
    std::cout << "Product: " << product << std::endl;
    
    return 0;
}
```

### Step 2: Iota (Range Generation)
Fill a vector with 0, 1, 2, 3...

```cpp
std::vector<int> seq(10);
std::iota(seq.begin(), seq.end(), 0); // Start at 0
```

### Step 3: Inner Product
Dot product of two vectors.
`v1 . v2 = (1*1) + (2*2) + ...`

```cpp
int dot = std::inner_product(v.begin(), v.end(), v.begin(), 0);
```

## Challenges

### Challenge 1: String Concatenation
Use `accumulate` to join a vector of strings.
Start with `std::string("")`.

### Challenge 2: Partial Sum
Calculate running totals.
Input: `{1, 2, 3, 4}`
Output: `{1, 3, 6, 10}`
Use `std::partial_sum`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <string>

int main() {
    // Challenge 1
    std::vector<std::string> words = {"Hello", " ", "World"};
    std::string sentence = std::accumulate(words.begin(), words.end(), std::string(""));
    std::cout << sentence << "\n";
    
    // Challenge 2
    std::vector<int> nums = {1, 2, 3, 4};
    std::vector<int> result(4);
    std::partial_sum(nums.begin(), nums.end(), result.begin());
    
    for(int n : result) std::cout << n << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `accumulate` for sum and product
✅ Used `iota` to generate sequences
✅ Used `inner_product`
✅ Concatenated strings with accumulate (Challenge 1)

## Key Learnings
- `<numeric>` is separate from `<algorithm>`
- `accumulate` is a general-purpose Fold (Left Fold)
- `iota` is great for initializing test data

## Next Steps
Proceed to **Lab 10.8: String View** for performance.
