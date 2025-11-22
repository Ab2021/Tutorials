# Lab 06: Container With Most Water

## Difficulty
🟡 **Medium**

## Estimated Time
45 minutes

## Learning Objectives
- Master the two-pointer technique
- Understand greedy decision-making
- Optimize brute force solutions
- Analyze area maximization problems

## Prerequisites
- Day 3: Two Pointers & Sliding Window
- Understanding of array traversal
- Basic geometry (area calculation)

## Problem Statement

You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i-th` line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Note**: You may not slant the container.

### Example 1
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The vertical lines are represented by array [1,8,6,2,5,4,8,3,7].
The max area is between index 1 and 8: min(8,7) * (8-1) = 7 * 7 = 49
```

### Example 2
```
Input: height = [1,1]
Output: 1
```

### Example 3
```
Input: height = [4,3,2,1,4]
Output: 16
Explanation: Max area is between index 0 and 4: min(4,4) * (4-0) = 4 * 4 = 16
```

### Constraints
- `n == height.length`
- `2 <= n <= 10^5`
- `0 <= height[i] <= 10^4`

## Requirements

1. Implement a function `max_area(height: List[int]) -> int`
2. The solution must run in O(n) time
3. Use O(1) extra space
4. Handle edge cases (minimum length array, all same heights, etc.)

## Starter Code

```python
from typing import List

def max_area(height: List[int]) -> int:
    """
    Find the maximum area of water that can be contained.
    
    Args:
        height: List of integers representing line heights
        
    Returns:
        Maximum area of water
    """
    # TODO: Implement your solution here
    pass

# Test cases
def test_max_area():
    test_cases = [
        ([1,8,6,2,5,4,8,3,7], 49),
        ([1,1], 1),
        ([4,3,2,1,4], 16),
        ([1,2,1], 2),
        ([2,3,4,5,18,17,6], 17),
    ]
    
    for heights, expected in test_cases:
        result = max_area(heights)
        assert result == expected, f"Failed for {heights}: expected {expected}, got {result}"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_max_area()
```

## Hints

<details>
<summary>Hint 1: Brute Force Approach</summary>

The brute force solution checks all pairs of lines:
```python
max_area = 0
for i in range(len(height)):
    for j in range(i+1, len(height)):
        area = min(height[i], height[j]) * (j - i)
        max_area = max(max_area, area)
```
This is O(n²). Can you optimize it?
</details>

<details>
<summary>Hint 2: Two Pointer Strategy</summary>

Start with the widest container (leftmost and rightmost lines). The area is limited by the shorter line. To potentially find a larger area, you need to move the pointer at the shorter line inward.
</details>

<details>
<summary>Hint 3: Greedy Decision</summary>

Why move the shorter line's pointer?
- Moving the taller line's pointer will only decrease width and can't increase height (limited by the shorter line)
- Moving the shorter line's pointer might find a taller line, potentially increasing area despite decreased width
</details>

<details>
<summary>Hint 4: Area Calculation</summary>

Area = min(height[left], height[right]) * (right - left)

The water level is determined by the shorter of the two lines.
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Approach 1: Brute Force (Not Optimal)

```python
def max_area_brute_force(height: List[int]) -> int:
    """
    Check all possible pairs of lines.
    Time: O(n²), Space: O(1)
    """
    max_water = 0
    n = len(height)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Area = height * width
            # Height is limited by shorter line
            width = j - i
            h = min(height[i], height[j])
            area = h * width
            max_water = max(max_water, area)
    
    return max_water
```

**Time Complexity**: O(n²) - checking all pairs  
**Space Complexity**: O(1)  
**Verdict**: Too slow for large inputs (n up to 10^5)

---

### Approach 2: Two Pointers (Optimal)

```python
def max_area(height: List[int]) -> int:
    """
    Use two pointers starting from both ends.
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        h = min(height[left], height[right])
        current_area = h * width
        max_water = max(max_water, current_area)
        
        # Move the pointer at the shorter line
        # This is the greedy choice: we can only improve
        # by potentially finding a taller line
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

**Time Complexity**: O(n) - single pass through array  
**Space Complexity**: O(1) - only using two pointers

---

### Detailed Walkthrough

Let's trace through Example 1: `height = [1,8,6,2,5,4,8,3,7]`

```
Initial state:
left = 0, right = 8
height[0] = 1, height[8] = 7
area = min(1, 7) * (8 - 0) = 1 * 8 = 8
max_water = 8
Move left (1 < 7)

Step 2:
left = 1, right = 8
height[1] = 8, height[8] = 7
area = min(8, 7) * (8 - 1) = 7 * 7 = 49
max_water = 49
Move right (8 > 7)

Step 3:
left = 1, right = 7
height[1] = 8, height[7] = 3
area = min(8, 3) * (7 - 1) = 3 * 6 = 18
max_water = 49 (no change)
Move right (8 > 3)

... continues until left >= right

Final answer: 49
```

---

### Why This Works (Proof of Correctness)

**Claim**: The two-pointer approach never misses the optimal solution.

**Proof by Contradiction**:
1. Assume the optimal solution uses lines at positions `i` and `j` where `i < j`
2. Assume our algorithm skips this pair by moving a pointer before considering it
3. WLOG, say we moved the left pointer past `i` before considering the pair `(i, j)`
4. This means we were at state `(i, k)` where `k > j`, and we moved left because `height[i] < height[k]`
5. But if `height[i] < height[k]`, then:
   - Area at `(i, k)` = `height[i] * (k - i)` 
   - Area at `(i, j)` = `min(height[i], height[j]) * (j - i)` ≤ `height[i] * (j - i)`
   - Since `k > j`, we have `(k - i) > (j - i)`
   - Therefore, area at `(i, k)` ≥ area at `(i, j)`
6. This contradicts the assumption that `(i, j)` is optimal
7. Therefore, the algorithm never skips the optimal solution ∎

---

### Alternative Implementation (More Explicit)

```python
def max_area_verbose(height: List[int]) -> int:
    """
    Same algorithm with more explicit logic.
    """
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate dimensions
        width = right - left
        left_height = height[left]
        right_height = height[right]
        container_height = min(left_height, right_height)
        
        # Calculate area
        current_area = container_height * width
        
        # Update maximum
        if current_area > max_water:
            max_water = current_area
        
        # Greedy choice: move the shorter line's pointer
        if left_height < right_height:
            # Left line is shorter, move left pointer
            left += 1
        elif right_height < left_height:
            # Right line is shorter, move right pointer
            right -= 1
        else:
            # Both lines are equal height
            # Moving either is fine; we'll move left
            left += 1
    
    return max_water
```

</details>

## Test Cases

```python
def comprehensive_tests():
    """Additional test cases for edge cases."""
    
    # Minimum length
    assert max_area([1, 1]) == 1
    
    # All same height
    assert max_area([5, 5, 5, 5]) == 15  # min(5,5) * 3
    
    # Increasing heights
    assert max_area([1, 2, 3, 4, 5]) == 6  # min(1,5) * 4 or min(2,5) * 3
    
    # Decreasing heights
    assert max_area([5, 4, 3, 2, 1]) == 6
    
    # Peak in middle
    assert max_area([1, 3, 5, 3, 1]) == 4  # min(3,3) * 2 or min(1,1) * 4
    
    # Large values
    assert max_area([10000, 1, 1, 1, 10000]) == 40000  # min(10000,10000) * 4
    
    print("✅ All comprehensive tests passed!")
```

## Extensions

1. **3D Container**: Extend the problem to 3D. Given a 2D grid of heights, find the maximum volume of water that can be contained.

2. **Multiple Containers**: Find the top k containers (k pairs of lines with largest areas).

3. **Minimum Container**: Find the minimum area container that can still hold at least X units of water.

4. **Visualization**: Create a visual representation of the container using ASCII art or matplotlib.

5. **Proof Challenge**: Formally prove that the two-pointer greedy approach is optimal using induction.

## Related Concepts
- [Day 3: Two Pointers Pattern](../Day3_Two_Pointers.md)
- [Lab 07: Array Partitioning](lab_07_array_partitioning.md)
- [Lab 08: Sliding Window](lab_08_max_sum_subarray.md)

## Real-World Applications

1. **Resource Allocation**: Maximizing capacity between two points
2. **Network Bandwidth**: Finding optimal connection points
3. **Construction**: Determining optimal support beam placement
4. **Data Compression**: Finding optimal chunk sizes

## Key Takeaways

1. **Two Pointers**: Powerful technique for optimization problems on arrays
2. **Greedy Approach**: Making locally optimal choices can lead to global optimum
3. **Proof of Correctness**: Always verify that greedy choices don't miss optimal solutions
4. **Time-Space Tradeoff**: O(n) time with O(1) space is often achievable with clever algorithms

---

**Next**: [Lab 07: Array Partitioning Schemes](lab_07_array_partitioning.md)
