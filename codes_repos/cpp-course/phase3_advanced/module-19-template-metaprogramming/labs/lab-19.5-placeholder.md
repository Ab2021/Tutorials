# Lab 19.5: Compile-Time Algorithms

## Objective
Implement algorithms that execute entirely at compile time using template metaprogramming.

## Instructions

### Step 1: Compile-Time Factorial
Create `compile_time_algorithms.cpp`.

```cpp
#include <iostream>

template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Usage
constexpr int fact5 = Factorial<5>::value; // Computed at compile time
```

### Step 2: Compile-Time Fibonacci
```cpp
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<0> { static constexpr int value = 0; };

template<>
struct Fibonacci<1> { static constexpr int value = 1; };
```

### Step 3: Compile-Time Array Operations
```cpp
template<typename T, size_t N>
struct Array {
    T data[N];
    
    constexpr T& operator[](size_t i) { return data[i]; }
    constexpr const T& operator[](size_t i) const { return data[i]; }
    constexpr size_t size() const { return N; }
};

template<typename T, size_t N>
constexpr auto sum(const Array<T, N>& arr) {
    T result = 0;
    for (size_t i = 0; i < N; ++i) {
        result += arr[i];
    }
    return result;
}
```

## Challenges

### Challenge 1: Compile-Time Prime Check
Implement a compile-time algorithm to check if a number is prime.

### Challenge 2: Compile-Time Sort
Create a compile-time sorting algorithm for arrays.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <array>

// Challenge 1: Compile-time prime check
template<int N, int D = N - 1>
struct IsPrime {
    static constexpr bool value = (N % D != 0) && IsPrime<N, D - 1>::value;
};

template<int N>
struct IsPrime<N, 1> {
    static constexpr bool value = true;
};

template<>
struct IsPrime<1, 0> {
    static constexpr bool value = false;
};

template<int N>
inline constexpr bool is_prime_v = IsPrime<N>::value;

// Challenge 2: Compile-time bubble sort
template<typename T, size_t N>
constexpr std::array<T, N> bubbleSort(std::array<T, N> arr) {
    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = 0; j < N - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                T temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

// Compile-time GCD
template<int A, int B>
struct GCD {
    static constexpr int value = GCD<B, A % B>::value;
};

template<int A>
struct GCD<A, 0> {
    static constexpr int value = A;
};

// Compile-time power
template<int Base, int Exp>
struct Power {
    static constexpr int value = Base * Power<Base, Exp - 1>::value;
};

template<int Base>
struct Power<Base, 0> {
    static constexpr int value = 1;
};

int main() {
    // Factorial
    std::cout << "5! = " << Factorial<5>::value << "\n";
    
    // Fibonacci
    std::cout << "Fib(10) = " << Fibonacci<10>::value << "\n";
    
    // Prime check
    std::cout << "Is 17 prime? " << is_prime_v<17> << "\n";
    std::cout << "Is 20 prime? " << is_prime_v<20> << "\n";
    
    // Compile-time sort
    constexpr std::array<int, 5> unsorted = {5, 2, 8, 1, 9};
    constexpr auto sorted = bubbleSort(unsorted);
    
    std::cout << "Sorted: ";
    for (int x : sorted) std::cout << x << " ";
    std::cout << "\n";
    
    // GCD
    std::cout << "GCD(48, 18) = " << GCD<48, 18>::value << "\n";
    
    // Power
    std::cout << "2^10 = " << Power<2, 10>::value << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented compile-time factorial
✅ Created compile-time Fibonacci
✅ Wrote compile-time array operations
✅ Implemented prime checker (Challenge 1)
✅ Created compile-time sort (Challenge 2)

## Key Learnings
- Template recursion enables compile-time computation
- `constexpr` functions execute at compile time
- Compile-time algorithms have zero runtime cost
- Template specialization provides base cases

## Next Steps
Proceed to **Lab 19.6: Variadic Template Patterns**.
