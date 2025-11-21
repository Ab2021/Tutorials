# Day 2: Arrays & Memory - Interview Preparation

> **Focus**: Array manipulation problems
> **Difficulty**: Easy to Medium

---

## Problem 1: Remove Duplicates from Sorted Array

**Question**: Given a sorted array, remove duplicates in-place. Return the new length.

```python
# Example:
# Input: [1, 1, 2, 2, 3]
# Output: 3 (array becomes [1, 2, 3, _, _])
```

**Solution: Two Pointers**
```python
def remove_duplicates(arr):
    if not arr:
        return 0
    
    write = 1  # Position to write next unique element
    for read in range(1, len(arr)):
        if arr[read] != arr[read-1]:
            arr[write] = arr[read]
            write += 1
    return write

# Time: O(n), Space: O(1)
```

---

## Problem 2: Rotate Array

**Question**: Rotate array to the right by k steps.

```python
# Example:
# Input: [1,2,3,4,5], k=2
# Output: [4,5,1,2,3]
```

**Solution 1: Extra Array - O(n) space**
```python
def rotate_extra_space(arr, k):
    n = len(arr)
    k = k % n
    return arr[-k:] + arr[:-k]
```

**Solution 2: Reversal - O(1) space**
```python
def rotate_in_place(arr, k):
    n = len(arr)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    reverse(0, n-1)      # Reverse entire array
    reverse(0, k-1)      # Reverse first k
    reverse(k, n-1)      # Reverse remaining
    return arr

# Example: [1,2,3,4,5], k=2
# Step 1: [5,4,3,2,1]
# Step 2: [4,5,3,2,1]
# Step 3: [4,5,1,2,3] ✓

# Time: O(n), Space: O(1)
```

---

## Problem 3: Move Zeros

**Question**: Move all zeros to the end while maintaining relative order.

```python
# Input: [0,1,0,3,12]
# Output: [1,3,12,0,0]
```

**Solution**:
```python
def move_zeros(arr):
    write = 0
    for read in range(len(arr)):
        if arr[read] != 0:
            arr[write] = arr[read]
            write += 1
    
    # Fill remaining with zeros
    for i in range(write, len(arr)):
        arr[i] = 0

# Time: O(n), Space: O(1)
```

---

## Problem 4: Best Time to Buy and Sell Stock

**Question**: Find maximum profit from one buy and one sell.

```python
# Input: [7,1,5,3,6,4]
# Output: 5 (buy at 1, sell at 6)
```

**Solution**:
```python
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        profit = price - min_price
        max_profit = max(max_profit, profit)
    
    return max_profit

# Time: O(n), Space: O(1)
```

---

## Problem 5: Product of Array Except Self

**Question**: Return array where output[i] = product of all elements except arr[i]. No division allowed.

```python
# Input: [1,2,3,4]
# Output: [24,12,8,6]
```

**Solution: Prefix & Suffix Products**
```python
def product_except_self(arr):
    n = len(arr)
    result = [1] * n
    
    # Left products
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= arr[i]
    
    # Right products
    right_product = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_product
        right_product *= arr[i]
    
    return result

# Time: O(n), Space: O(1) (output doesn't count)
```

---

## Problem 6: Container With Most Water

**Question**: Find two lines that together with x-axis form a container holding the most water.

```python
# Input: [1,8,6,2,5,4,8,3,7]
# Output: 49 (lines at index 1 and 8)
```

**Solution: Two Pointers**
```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Time: O(n), Space: O(1)
```

---

## Problem 7: Spiral Matrix

**Question**: Return all elements of matrix in spiral order.

```python
# Input: [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,3,6,9,8,7,4,5]
```

**Solution**:
```python
def spiral_order(matrix):
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Left (if still valid)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Up (if still valid)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result

# Time: O(m*n), Space: O(1)
```

---

## Problem 8: Maximum Subarray (Kadane's Algorithm)

**Question**: Find the contiguous subarray with the largest sum.

```python
# Input: [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6 (subarray [4,-1,2,1])
```

**Solution**:
```python
def max_subarray(arr):
    max_sum = arr[0]
    current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Time: O(n), Space: O(1)
```

---

## Key Patterns

1. **Two Pointers**: Remove duplicates, move zeros, container with water
2. **Reversal Trick**: Rotate array
3. **Prefix/Suffix**: Product except self
4. **Greedy**: Best time to buy/sell stock, Kadane's algorithm
5. **Boundary Tracking**: Spiral matrix

---

**Next**: [Day 3: Two Pointers & Sliding Window](Day3_Two_Pointers.md)
