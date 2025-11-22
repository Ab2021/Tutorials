# Lab 09: Range Query Implementation with Prefix Sum

## Difficulty
🟢 **Easy**

## Estimated Time
45 minutes

## Learning Objectives
- Understand prefix sum (cumulative sum) technique
- Optimize range query operations from O(n) to O(1)
- Apply preprocessing for efficient queries
- Handle multiple query scenarios

## Prerequisites
- Day 4: Prefix Sum & Difference Arrays
- Understanding of array indexing
- Basic arithmetic operations

## Problem Statement

Given an integer array `nums`, implement a data structure that supports efficient range sum queries. A range sum query asks for the sum of elements between indices `left` and `right` (inclusive).

Implement the `NumArray` class:
- `NumArray(int[] nums)`: Initializes the object with the integer array `nums`
- `int sumRange(int left, int right)`: Returns the sum of elements between indices `left` and `right` inclusive

### Example 1
```
Input:
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]

Output:
[null, 1, -1, -3]

Explanation:
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
```

### Example 2
```
Input:
["NumArray", "sumRange", "sumRange"]
[[[1, 2, 3, 4, 5]], [0, 4], [1, 3]]

Output:
[null, 15, 9]
```

### Constraints
- `1 <= nums.length <= 10^4`
- `-10^5 <= nums[i] <= 10^5`
- `0 <= left <= right < nums.length`
- At most `10^4` calls will be made to `sumRange`

## Requirements

1. Implement the `NumArray` class with initialization and query methods
2. Optimize `sumRange` to run in O(1) time
3. Keep space complexity reasonable (O(n))
4. Handle edge cases (single element, negative numbers, etc.)

## Starter Code

```python
from typing import List

class NumArray:
    def __init__(self, nums: List[int]):
        """
        Initialize the data structure with the given array.
        
        Args:
            nums: Input integer array
        """
        # TODO: Implement initialization
        pass
    
    def sumRange(self, left: int, right: int) -> int:
        """
        Return the sum of elements between indices left and right inclusive.
        
        Args:
            left: Left index (inclusive)
            right: Right index (inclusive)
            
        Returns:
            Sum of elements in range [left, right]
        """
        # TODO: Implement range sum query
        pass

# Test cases
def test_num_array():
    # Test case 1
    obj = NumArray([-2, 0, 3, -5, 2, -1])
    assert obj.sumRange(0, 2) == 1
    assert obj.sumRange(2, 5) == -1
    assert obj.sumRange(0, 5) == -3
    
    # Test case 2
    obj2 = NumArray([1, 2, 3, 4, 5])
    assert obj2.sumRange(0, 4) == 15
    assert obj2.sumRange(1, 3) == 9
    assert obj2.sumRange(0, 0) == 1
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_num_array()
```

## Hints

<details>
<summary>Hint 1: Brute Force Approach</summary>

The naive solution would sum elements from `left` to `right` for each query:
```python
def sumRange(self, left, right):
    return sum(self.nums[left:right+1])
```
This is O(n) per query. With 10^4 queries, this could be slow. Can you preprocess the array to make queries faster?
</details>

<details>
<summary>Hint 2: Prefix Sum Concept</summary>

Create an array where `prefix[i]` = sum of all elements from index 0 to i.

Then: `sum(left, right) = prefix[right] - prefix[left-1]`

Example: `nums = [1, 2, 3, 4, 5]`
- `prefix = [1, 3, 6, 10, 15]`
- `sum(1, 3) = prefix[3] - prefix[0] = 10 - 1 = 9` ✓
</details>

<details>
<summary>Hint 3: Handling Edge Cases</summary>

What happens when `left = 0`? You'd need `prefix[-1]`, which doesn't exist. 

Solution: Add a dummy element at the beginning: `prefix[0] = 0`, then `prefix[i+1] = sum(nums[0:i+1])`
</details>

<details>
<summary>Hint 4: Construction Formula</summary>

Build prefix sum array:
```python
prefix[0] = 0
for i in range(len(nums)):
    prefix[i+1] = prefix[i] + nums[i]
```

Query formula:
```python
sumRange(left, right) = prefix[right+1] - prefix[left]
```
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Approach 1: Brute Force (Not Optimal)

```python
class NumArrayBruteForce:
    def __init__(self, nums: List[int]):
        self.nums = nums
    
    def sumRange(self, left: int, right: int) -> int:
        return sum(self.nums[left:right+1])
```

**Time Complexity**:
- Initialization: O(1)
- Query: O(n) where n = right - left + 1

**Space Complexity**: O(1) extra space

**Analysis**: With up to 10^4 queries on an array of size 10^4, worst case is O(10^8) operations - too slow!

---

### Approach 2: Prefix Sum (Optimal)

```python
class NumArray:
    def __init__(self, nums: List[int]):
        """
        Build prefix sum array during initialization.
        prefix[i] = sum of nums[0] to nums[i-1]
        """
        self.prefix = [0]  # prefix[0] = 0 (sum of zero elements)
        
        # Build prefix sum array
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
        
        # Alternative one-liner:
        # self.prefix = [0] + list(itertools.accumulate(nums))
    
    def sumRange(self, left: int, right: int) -> int:
        """
        Calculate range sum using prefix sum difference.
        
        sum(left, right) = prefix[right+1] - prefix[left]
        
        Intuition:
        - prefix[right+1] = sum of all elements from 0 to right
        - prefix[left] = sum of all elements from 0 to left-1
        - Difference gives sum from left to right
        """
        return self.prefix[right + 1] - self.prefix[left]
```

**Time Complexity**:
- Initialization: O(n) - build prefix array
- Query: O(1) - simple subtraction

**Space Complexity**: O(n) - store prefix array

**Analysis**: Perfect for multiple queries! Preprocessing takes O(n), but each query is O(1).

---

### Detailed Walkthrough

Let's trace through Example 1: `nums = [-2, 0, 3, -5, 2, -1]`

**Step 1: Build Prefix Sum Array**
```
nums   = [-2,  0,  3, -5,  2, -1]
indices:  0   1   2   3   4   5

prefix[0] = 0
prefix[1] = 0 + (-2) = -2
prefix[2] = -2 + 0 = -2
prefix[3] = -2 + 3 = 1
prefix[4] = 1 + (-5) = -4
prefix[5] = -4 + 2 = -2
prefix[6] = -2 + (-1) = -3

prefix = [0, -2, -2, 1, -4, -2, -3]
```

**Step 2: Query sumRange(0, 2)**
```
sum(0, 2) = prefix[3] - prefix[0]
          = 1 - 0
          = 1 ✓

Verification: nums[0] + nums[1] + nums[2] = -2 + 0 + 3 = 1 ✓
```

**Step 3: Query sumRange(2, 5)**
```
sum(2, 5) = prefix[6] - prefix[2]
          = -3 - (-2)
          = -1 ✓

Verification: nums[2] + nums[3] + nums[4] + nums[5] = 3 + (-5) + 2 + (-1) = -1 ✓
```

**Step 4: Query sumRange(0, 5)**
```
sum(0, 5) = prefix[6] - prefix[0]
          = -3 - 0
          = -3 ✓

Verification: sum of all elements = -3 ✓
```

---

### Alternative Implementation (More Explicit)

```python
class NumArrayVerbose:
    def __init__(self, nums: List[int]):
        n = len(nums)
        self.prefix = [0] * (n + 1)
        
        # Build prefix sum array
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
    
    def sumRange(self, left: int, right: int) -> int:
        # Sum from 0 to right
        sum_to_right = self.prefix[right + 1]
        
        # Sum from 0 to left-1
        sum_to_left_minus_1 = self.prefix[left]
        
        # Difference gives sum from left to right
        return sum_to_right - sum_to_left_minus_1
```

---

### Using Python's itertools (Elegant)

```python
import itertools

class NumArrayItertools:
    def __init__(self, nums: List[int]):
        # itertools.accumulate returns cumulative sum
        self.prefix = [0] + list(itertools.accumulate(nums))
    
    def sumRange(self, left: int, right: int) -> int:
        return self.prefix[right + 1] - self.prefix[left]
```

---

### Visualization

```
nums:    [-2,  0,  3, -5,  2, -1]
index:     0   1   2   3   4   5

prefix:  [0, -2, -2,  1, -4, -2, -3]
index:    0   1   2   3   4   5   6

Query sumRange(1, 4):
         ┌─────────────────┐
nums:   [-2,  0,  3, -5,  2, -1]
              └──────────┘
         
prefix[5] - prefix[1] = -2 - (-2) = 0
Verification: 0 + 3 + (-5) + 2 = 0 ✓
```

</details>

## Test Cases

```python
def comprehensive_tests():
    """Additional test cases for edge cases."""
    
    # Single element
    obj = NumArray([5])
    assert obj.sumRange(0, 0) == 5
    
    # All negative
    obj = NumArray([-1, -2, -3, -4])
    assert obj.sumRange(0, 3) == -10
    assert obj.sumRange(1, 2) == -5
    
    # All positive
    obj = NumArray([1, 2, 3, 4, 5])
    assert obj.sumRange(0, 4) == 15
    assert obj.sumRange(2, 4) == 12
    
    # Mixed with zeros
    obj = NumArray([0, 0, 0, 1, 0])
    assert obj.sumRange(0, 4) == 1
    assert obj.sumRange(0, 2) == 0
    
    # Large values
    obj = NumArray([100000, -100000, 50000])
    assert obj.sumRange(0, 2) == 50000
    assert obj.sumRange(0, 1) == 0
    
    print("✅ All comprehensive tests passed!")
```

## Extensions

1. **Mutable Array**: Extend the class to support `update(index, val)` that changes `nums[index]` to `val`. How does this affect the prefix sum array? Can you update it efficiently?

2. **2D Range Sum**: Extend to 2D arrays. Given a matrix, support queries for sum of any rectangular region. (Hint: Use 2D prefix sums)

3. **Range Maximum Query**: Instead of sum, find the maximum element in a range. Can prefix sums help here?

4. **Lazy Updates**: Support range updates (add a value to all elements in a range) along with range queries. (Hint: Use difference arrays)

5. **Circular Array**: Handle circular range queries where the range can wrap around (e.g., from index n-2 to index 1).

## Related Concepts
- [Day 4: Prefix Sum & Difference Arrays](../Day4_Prefix_Sum.md)
- [Lab 10: Range Update Operations](lab_10_range_update.md)
- [Lab 11: Matrix Spiral Traversal](lab_11_spiral_matrix.md)

## Real-World Applications

1. **Financial Analysis**: Calculate total revenue/expenses over date ranges
2. **Time Series Data**: Compute cumulative metrics (total sales, page views)
3. **Image Processing**: Integral images for fast region sum calculation
4. **Database Queries**: Optimize SUM aggregations over ranges
5. **Game Development**: Calculate damage/score over time intervals

## Key Takeaways

1. **Preprocessing**: Spending O(n) time upfront can make queries O(1)
2. **Space-Time Tradeoff**: Using O(n) extra space for O(1) queries
3. **Prefix Sum Formula**: `sum(left, right) = prefix[right+1] - prefix[left]`
4. **Dummy Element**: Adding `prefix[0] = 0` simplifies edge case handling
5. **Multiple Queries**: Prefix sums excel when there are many queries on static data

## Common Mistakes

1. **Off-by-One Errors**: Forgetting to use `right+1` instead of `right`
2. **Missing Dummy**: Not adding the initial 0 to prefix array
3. **Index Confusion**: Mixing up `nums` indices with `prefix` indices
4. **Negative Numbers**: Assuming sums are always positive
5. **Immutability**: Prefix sum approach assumes array doesn't change

---

**Next**: [Lab 10: Range Update Operations](lab_10_range_update.md)
