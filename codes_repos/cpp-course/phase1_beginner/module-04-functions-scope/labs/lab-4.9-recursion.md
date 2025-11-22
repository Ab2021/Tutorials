# Lab 4.9: Recursion

## Objective
Understand recursion by writing functions that call themselves.

## Instructions

### Step 1: Factorial
Create `recursion.cpp`. Factorial of N is N * Factorial(N-1). Base case: 0! = 1.

```cpp
#include <iostream>

int factorial(int n) {
    if (n <= 1) return 1; // Base case
    return n * factorial(n - 1); // Recursive step
}

int main() {
    std::cout << "5! = " << factorial(5) << std::endl;
    return 0;
}
```

### Step 2: Stack Overflow
What happens if you forget the base case?
Or call `factorial(-1)`?
*Try it! The program will crash (Stack Overflow).*

### Step 3: Fibonacci
Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8...
F(n) = F(n-1) + F(n-2).

```cpp
int fib(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fib(n - 1) + fib(n - 2);
}
```

## Challenges

### Challenge 1: Print Digits
Write a recursive function `void printDigits(int n)` that prints digits of a number in order (e.g., 123 -> "1 2 3").
*Hint: Print `n / 10` recursively, then print `n % 10`.*

### Challenge 2: Sum of Array
Write a recursive function to sum an array.
`int sum(int* arr, int size)`
*Hint: arr[0] + sum(arr + 1, size - 1)*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void printDigits(int n) {
    if (n < 10) {
        std::cout << n << " ";
        return;
    }
    printDigits(n / 10);
    std::cout << (n % 10) << " ";
}

int sum(int* arr, int size) {
    if (size == 0) return 0;
    return arr[0] + sum(arr + 1, size - 1);
}

int main() {
    printDigits(12345);
    std::cout << std::endl;
    
    int arr[] = {1, 2, 3, 4, 5};
    std::cout << "Sum: " << sum(arr, 5) << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented recursive factorial
✅ Implemented recursive Fibonacci
✅ Understood base case importance
✅ Solved digit printing (Challenge 1)

## Key Learnings
- Recursion requires a base case
- Each call adds a frame to the stack
- Useful for tree/graph traversal (covered later)

## Next Steps
Proceed to **Lab 4.10: Lambda Expressions** for modern functional C++.
