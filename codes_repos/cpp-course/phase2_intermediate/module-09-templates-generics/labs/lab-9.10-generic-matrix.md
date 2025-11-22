# Lab 9.10: Generic Matrix Class (Capstone)

## Objective
Build a fully generic Matrix class using templates, non-type parameters, and operator overloading.

## Instructions

### Step 1: Class Definition
Create `matrix.cpp`.
`template <typename T, int Rows, int Cols>`

```cpp
#include <iostream>
#include <array>

template <typename T, int Rows, int Cols>
class Matrix {
    std::array<std::array<T, Cols>, Rows> data;
public:
    Matrix() {
        // Initialize to zero
        for(auto& row : data) row.fill(T{});
    }
    
    T& at(int r, int c) { return data[r][c]; }
    const T& at(int r, int c) const { return data[r][c]; }
    
    void print() const {
        for(const auto& row : data) {
            for(const auto& val : row) std::cout << val << " ";
            std::cout << "\n";
        }
    }
};
```

### Step 2: Matrix Addition
Implement `operator+`.
Note: Can only add matrices of SAME dimensions.

```cpp
    Matrix operator+(const Matrix& other) const {
        Matrix result;
        for(int i=0; i<Rows; ++i)
            for(int j=0; j<Cols; ++j)
                result.at(i, j) = at(i, j) + other.at(i, j);
        return result;
    }
```

### Step 3: Matrix Multiplication
This is tricky! `(MxN) * (NxP) = (MxP)`.
The dimensions change.

```cpp
    template <int OtherCols>
    Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const {
        Matrix<T, Rows, OtherCols> result;
        for(int i=0; i<Rows; ++i) {
            for(int j=0; j<OtherCols; ++j) {
                for(int k=0; k<Cols; ++k) {
                    result.at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return result;
    }
```

## Challenges

### Challenge 1: Transpose
Implement `Matrix<T, Cols, Rows> transpose() const`.
Swaps rows and columns.

### Challenge 2: Concept Constraint
Use C++20 `requires` to ensure `T` supports arithmetic operations (`+`, `*`).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <array>
#include <concepts>

template <typename T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template <Arithmetic T, int Rows, int Cols>
class Matrix {
    std::array<std::array<T, Cols>, Rows> data;
public:
    Matrix() { for(auto& row : data) row.fill(0); }
    
    T& operator()(int r, int c) { return data[r][c]; }
    const T& operator()(int r, int c) const { return data[r][c]; }
    
    void print() const {
        for(const auto& row : data) {
            for(const auto& val : row) std::cout << val << " ";
            std::cout << "\n";
        }
    }
    
    // Transpose
    Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> res;
        for(int i=0; i<Rows; ++i)
            for(int j=0; j<Cols; ++j)
                res(j, i) = data[i][j];
        return res;
    }
    
    // Multiplication
    template <int OtherCols>
    Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const {
        Matrix<T, Rows, OtherCols> res;
        for(int i=0; i<Rows; ++i)
            for(int j=0; j<OtherCols; ++j)
                for(int k=0; k<Cols; ++k)
                    res(i, j) += data[i][k] * other(k, j);
        return res;
    }
};

int main() {
    Matrix<int, 2, 3> m1;
    m1(0,0)=1; m1(0,1)=2; m1(0,2)=3;
    m1(1,0)=4; m1(1,1)=5; m1(1,2)=6;
    
    std::cout << "M1:\n"; m1.print();
    
    std::cout << "Transpose:\n"; m1.transpose().print();
    
    Matrix<int, 3, 2> m2;
    m2(0,0)=7; m2(0,1)=8;
    m2(1,0)=9; m2(1,1)=1;
    m2(2,0)=2; m2(2,1)=3;
    
    std::cout << "M1 * M2:\n"; (m1 * m2).print();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented generic Matrix class
✅ Used non-type parameters for dimensions
✅ Implemented Matrix Multiplication (changing types)
✅ Implemented Transpose (Challenge 1)
✅ Used Concepts (Challenge 2)

## Key Learnings
- Templates allow complex mathematical types to be safe and efficient
- Template parameters can change return types (MxN * NxP -> MxP)
- `std::array` is a zero-overhead wrapper for C-arrays

## Next Steps
Congratulations! You've completed Module 9.

Proceed to **Module 10: Standard Template Library (STL)** to use the containers C++ provides for you.
