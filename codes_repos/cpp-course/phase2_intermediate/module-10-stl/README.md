# Module 10: Standard Template Library (STL)

## 🎯 Learning Objectives

By the end of this module, you will:
- Master the core STL containers: `vector`, `list`, `map`, `set`.
- Understand the difference between Sequence and Associative containers.
- Use Iterators to traverse and manipulate collections.
- Apply standard Algorithms (`sort`, `find`, `transform`) to avoid rewriting common logic.
- Understand Iterator Invalidation rules.
- Learn modern C++ features like `std::string_view` (C++17) and Ranges (C++20).

---

## 📖 Theoretical Concepts

### 10.1 Containers

- **Sequence:** `vector` (dynamic array), `deque` (double-ended queue), `list` (doubly linked list), `array` (static array).
- **Associative:** `set` (unique sorted keys), `map` (key-value sorted), `multiset/multimap` (duplicates allowed).
- **Unordered:** `unordered_set`, `unordered_map` (Hash Tables).

### 10.2 Iterators

Objects that point to elements in a container.
- `begin()`: Points to first element.
- `end()`: Points to *one past* the last element.

```cpp
std::vector<int> v = {1, 2, 3};
for (auto it = v.begin(); it != v.end(); ++it) {
    std::cout << *it << " ";
}
```

### 10.3 Algorithms

Functions that operate on ranges of elements.

```cpp
std::sort(v.begin(), v.end());
auto it = std::find(v.begin(), v.end(), 3);
```

### 10.4 Ranges (C++20)

Composable, readable algorithms.

```cpp
auto even = v | std::views::filter([](int n){ return n % 2 == 0; });
```

---

## 🦀 Rust vs C++ Comparison

### Collections
**C++:** `std::vector`, `std::map` (Tree), `std::unordered_map` (Hash).
**Rust:** `Vec`, `BTreeMap` (Tree), `HashMap` (Hash).
*Note: Rust's `HashMap` is the default map, whereas C++ `std::map` is a Tree. C++ `std::unordered_map` is the Hash version.*

### Iterators
**C++:** Begin/End pair. Unsafe (can be invalidated easily).
**Rust:** `Iterator` trait. Consumed on use (unless borrowed). Checked by borrow checker.

### Algorithms
**C++:** `std::transform(begin, end, out, op)`.
**Rust:** `iter.map(op).collect()`.
*C++20 Ranges brings C++ closer to Rust's functional style.*

---

## 🔑 Key Takeaways

1. **Prefer `std::vector`** by default. It is cache-friendly and fast.
2. Use `std::map` for sorted data, `std::unordered_map` for lookups.
3. **Never write a raw loop** if a standard algorithm exists (`<algorithm>`).
4. Be careful with **Iterator Invalidation** (e.g., pushing to a vector might invalidate pointers).
5. Use `std::string_view` for read-only string arguments to avoid copies.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 10.1:** Vector Deep Dive
2. **Lab 10.2:** List and Deque
3. **Lab 10.3:** Maps and Sets
4. **Lab 10.4:** Iterators
5. **Lab 10.5:** Basic Algorithms (Sort, Find)
6. **Lab 10.6:** Modifying Algorithms (Transform, Remove)
7. **Lab 10.7:** Numeric Algorithms
8. **Lab 10.8:** String View (C++17)
9. **Lab 10.9:** Ranges and Views (C++20)
10. **Lab 10.10:** Text Frequency Analyzer (Capstone)

After completing the labs, move on to **Module 11: Error Handling and Exceptions**.
