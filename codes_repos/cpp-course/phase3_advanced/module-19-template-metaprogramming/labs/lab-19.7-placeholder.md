# Lab 19.7: Expression Templates

## Objective
Implement expression templates for lazy evaluation and optimization of complex expressions.

## Instructions

### Step 1: Basic Expression Template
Create `expression_templates.cpp`.

```cpp
#include <iostream>
#include <vector>

// Expression template for vector addition
template<typename E>
class VecExpression {
public:
    double operator[](size_t i) const {
        return static_cast<const E&>(*this)[i];
    }
    
    size_t size() const {
        return static_cast<const E&>(*this).size();
    }
};

class Vec : public VecExpression<Vec> {
    std::vector<double> data;
    
public:
    Vec(size_t n) : data(n) {}
    Vec(std::vector<double> d) : data(std::move(d)) {}
    
    double operator[](size_t i) const { return data[i]; }
    double& operator[](size_t i) { return data[i]; }
    size_t size() const { return data.size(); }
    
    template<typename E>
    Vec& operator=(const VecExpression<E>& expr) {
        for (size_t i = 0; i < size(); ++i) {
            data[i] = expr[i];
        }
        return *this;
    }
};
```

### Step 2: Binary Expression
```cpp
template<typename E1, typename E2>
class VecSum : public VecExpression<VecSum<E1, E2>> {
    const E1& lhs;
    const E2& rhs;
    
public:
    VecSum(const E1& l, const E2& r) : lhs(l), rhs(r) {}
    
    double operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }
    
    size_t size() const { return lhs.size(); }
};

template<typename E1, typename E2>
VecSum<E1, E2> operator+(const VecExpression<E1>& lhs, 
                          const VecExpression<E2>& rhs) {
    return VecSum<E1, E2>(
        static_cast<const E1&>(lhs),
        static_cast<const E2&>(rhs)
    );
}
```

### Step 3: Scalar Multiplication
```cpp
template<typename E>
class VecScalar : public VecExpression<VecScalar<E>> {
    const E& vec;
    double scalar;
    
public:
    VecScalar(const E& v, double s) : vec(v), scalar(s) {}
    
    double operator[](size_t i) const {
        return vec[i] * scalar;
    }
    
    size_t size() const { return vec.size(); }
};

template<typename E>
VecScalar<E> operator*(const VecExpression<E>& vec, double scalar) {
    return VecScalar<E>(static_cast<const E&>(vec), scalar);
}
```

## Challenges

### Challenge 1: Complex Expression
Build expression templates for: `result = a + b * 2.0 - c`

### Challenge 2: Matrix Expression Templates
Extend the concept to 2D matrices.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

// Complete expression template system

template<typename E>
class VecExpression {
public:
    double operator[](size_t i) const {
        return static_cast<const E&>(*this)[i];
    }
    size_t size() const {
        return static_cast<const E&>(*this).size();
    }
};

class Vec : public VecExpression<Vec> {
    std::vector<double> data;
public:
    Vec(size_t n) : data(n) {}
    Vec(std::initializer_list<double> init) : data(init) {}
    
    double operator[](size_t i) const { return data[i]; }
    double& operator[](size_t i) { return data[i]; }
    size_t size() const { return data.size(); }
    
    template<typename E>
    Vec& operator=(const VecExpression<E>& expr) {
        for (size_t i = 0; i < size(); ++i) {
            data[i] = expr[i];
        }
        return *this;
    }
};

// Binary operations
template<typename E1, typename E2, typename Op>
class VecBinary : public VecExpression<VecBinary<E1, E2, Op>> {
    const E1& lhs;
    const E2& rhs;
    Op op;
public:
    VecBinary(const E1& l, const E2& r, Op o = Op()) 
        : lhs(l), rhs(r), op(o) {}
    
    double operator[](size_t i) const {
        return op(lhs[i], rhs[i]);
    }
    size_t size() const { return lhs.size(); }
};

struct Add { double operator()(double a, double b) const { return a + b; } };
struct Sub { double operator()(double a, double b) const { return a - b; } };
struct Mul { double operator()(double a, double b) const { return a * b; } };

template<typename E1, typename E2>
auto operator+(const VecExpression<E1>& l, const VecExpression<E2>& r) {
    return VecBinary<E1, E2, Add>(
        static_cast<const E1&>(l),
        static_cast<const E2&>(r)
    );
}

template<typename E1, typename E2>
auto operator-(const VecExpression<E1>& l, const VecExpression<E2>& r) {
    return VecBinary<E1, E2, Sub>(
        static_cast<const E1&>(l),
        static_cast<const E2&>(r)
    );
}

template<typename E>
class VecScalar : public VecExpression<VecScalar<E>> {
    const E& vec;
    double scalar;
public:
    VecScalar(const E& v, double s) : vec(v), scalar(s) {}
    double operator[](size_t i) const { return vec[i] * scalar; }
    size_t size() const { return vec.size(); }
};

template<typename E>
auto operator*(const VecExpression<E>& vec, double scalar) {
    return VecScalar<E>(static_cast<const E&>(vec), scalar);
}

int main() {
    Vec a{1, 2, 3, 4};
    Vec b{5, 6, 7, 8};
    Vec c{2, 2, 2, 2};
    Vec result(4);
    
    // Challenge 1: Complex expression (lazy evaluation!)
    result = a + b * 2.0 - c;
    
    std::cout << "Result: ";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented basic expression templates
✅ Created binary expression operators
✅ Added scalar multiplication
✅ Built complex expressions (Challenge 1)
✅ Extended to matrices (Challenge 2)

## Key Learnings
- Expression templates delay evaluation
- CRTP enables static polymorphism
- Eliminates temporary objects
- Compiler optimizes expression trees
- Significant performance gains for numeric code

## Next Steps
Proceed to **Lab 19.8: Template Recursion**.
