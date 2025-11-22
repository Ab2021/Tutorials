# Lab 1.8: Namespace Basics

## Objective
Learn how to use namespaces to organize code and prevent name collisions.

## Instructions

### Step 1: Create a Namespace
Create `shapes.cpp`:

```cpp
#include <iostream>

namespace Math {
    int add(int a, int b) {
        return a + b;
    }
}

namespace Physics {
    int add(int a, int b) {
        return a + b + 10; // Different implementation
    }
}

int main() {
    // TODO: Call Math::add
    
    // TODO: Call Physics::add
    
    return 0;
}
```

### Step 2: Using Declarations
Modify `main` to use `using` declarations:

```cpp
using Math::add;
std::cout << add(5, 5) << std::endl; // Calls Math::add
```

### Step 3: Nested Namespaces
Create a nested namespace structure:

```cpp
namespace Game {
    namespace Graphics {
        void render() {
            std::cout << "Rendering..." << std::endl;
        }
    }
}
```

Call it in `main`: `Game::Graphics::render();`

## Challenges

### Challenge 1: Namespace Aliasing
Type `Game::Graphics::render()` is long. Create an alias:
```cpp
namespace GFX = Game::Graphics;
```
Now call `GFX::render()`.

### Challenge 2: Anonymous Namespaces
Create an anonymous namespace (unnamed) in a source file.
```cpp
namespace {
    void internal_helper() {
        // ...
    }
}
```
Try to call `internal_helper` from another file (you shouldn't be able to). This is how we make functions "private" to a file in C++.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

namespace Math {
    int add(int a, int b) { return a + b; }
}

namespace Physics {
    int add(int a, int b) { return a + b + 10; }
}

namespace Game {
    namespace Graphics {
        void render() { std::cout << "Rendering..." << std::endl; }
    }
}

namespace GFX = Game::Graphics; // Alias

int main() {
    std::cout << "Math: " << Math::add(2, 3) << std::endl;
    std::cout << "Physics: " << Physics::add(2, 3) << std::endl;
    
    GFX::render();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and used namespaces
✅ Resolved name collisions
✅ Used nested namespaces
✅ Created namespace aliases

## Key Learnings
- Namespaces prevent naming conflicts
- How to access namespace members (`::`)
- `using` declarations
- Namespace aliases for readability

## Next Steps
Proceed to **Lab 1.9: Command-line Arguments** to interact with your program from the shell.
