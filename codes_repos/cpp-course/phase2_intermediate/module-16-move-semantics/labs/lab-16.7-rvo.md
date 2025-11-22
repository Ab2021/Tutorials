# Lab 16.7: Return Value Optimization (RVO)

## Objective
Understand RVO, NRVO, and when the compiler eliminates copies/moves.

## Instructions

### Step 1: Copy Elision (RVO)
The compiler can eliminate copies when returning temporaries.

Create `rvo.cpp`.

```cpp
#include <iostream>

class Widget {
public:
    Widget() { std::cout << "Constructed\n"; }
    Widget(const Widget&) { std::cout << "Copied\n"; }
    Widget(Widget&&) noexcept { std::cout << "Moved\n"; }
};

Widget makeWidget() {
    return Widget(); // RVO: no copy, no move!
}

int main() {
    Widget w = makeWidget(); // Only "Constructed" printed
    return 0;
}
```

### Step 2: Named Return Value Optimization (NRVO)
```cpp
Widget makeNamedWidget() {
    Widget w; // Named object
    return w; // NRVO may eliminate copy/move
}

int main() {
    Widget w = makeNamedWidget(); // Likely only "Constructed"
    return 0;
}
```

### Step 3: When RVO Doesn't Apply
```cpp
Widget makeConditional(bool flag) {
    Widget w1, w2;
    return flag ? w1 : w2; // Can't apply NRVO (multiple returns)
}

Widget dontDoThis() {
    Widget w;
    return std::move(w); // Prevents NRVO! Don't do this!
}
```

### Step 4: Guaranteed Copy Elision (C++17)
```cpp
Widget w = Widget(); // Guaranteed elision since C++17
```

## Challenges

### Challenge 1: Measure RVO
Create a class that tracks constructions and verify RVO.

### Challenge 2: Multiple Return Paths
Implement a function with multiple returns and observe behavior.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

// Challenge 1: Tracking class
class Tracker {
    std::string name;
    static int constructed;
    static int copied;
    static int moved;
    
public:
    Tracker(std::string n = "default") : name(n) {
        ++constructed;
        std::cout << "Constructed: " << name << "\n";
    }
    
    Tracker(const Tracker& other) : name(other.name) {
        ++copied;
        std::cout << "Copied: " << name << "\n";
    }
    
    Tracker(Tracker&& other) noexcept : name(std::move(other.name)) {
        ++moved;
        std::cout << "Moved: " << name << "\n";
    }
    
    ~Tracker() {
        std::cout << "Destroyed: " << name << "\n";
    }
    
    static void report() {
        std::cout << "\n=== Statistics ===\n";
        std::cout << "Constructed: " << constructed << "\n";
        std::cout << "Copied: " << copied << "\n";
        std::cout << "Moved: " << moved << "\n";
    }
};

int Tracker::constructed = 0;
int Tracker::copied = 0;
int Tracker::moved = 0;

// RVO example
Tracker makeTracker() {
    return Tracker("RVO");
}

// NRVO example
Tracker makeNamedTracker() {
    Tracker t("NRVO");
    return t;
}

// Challenge 2: Multiple return paths
Tracker makeConditional(bool flag) {
    if (flag) {
        return Tracker("Path1");
    } else {
        return Tracker("Path2");
    }
}

int main() {
    std::cout << "=== RVO ===\n";
    Tracker t1 = makeTracker();
    
    std::cout << "\n=== NRVO ===\n";
    Tracker t2 = makeNamedTracker();
    
    std::cout << "\n=== Conditional (no NRVO) ===\n";
    Tracker t3 = makeConditional(true);
    
    Tracker::report();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood RVO and NRVO
✅ Avoided `std::move` in return statements
✅ Tracked copy elision (Challenge 1)
✅ Observed multiple return paths (Challenge 2)

## Compiler Flags
```bash
# Disable RVO to see copies/moves
g++ -fno-elide-constructors rvo.cpp -o rvo

# Enable optimizations (RVO enabled by default)
g++ -O2 rvo.cpp -o rvo
```

## Key Learnings
- RVO eliminates copies when returning temporaries
- NRVO eliminates copies for named return values
- C++17 guarantees copy elision in some cases
- Never use `std::move` in return statements
- Multiple return paths prevent NRVO

## Next Steps
Proceed to **Lab 16.8: Move Semantics with STL**.
