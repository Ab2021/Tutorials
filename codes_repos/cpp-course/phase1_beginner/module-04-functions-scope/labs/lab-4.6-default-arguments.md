# Lab 4.6: Default Arguments

## Objective
Learn how to make function parameters optional by providing default values.

## Instructions

### Step 1: Basic Defaults
Create `defaults.cpp`. Write a `log` function.

```cpp
#include <iostream>
#include <string>

void log(std::string msg, int level = 1) {
    std::cout << "[" << level << "] " << msg << std::endl;
}

int main() {
    log("System starting..."); // Uses default level 1
    log("Error occurred!", 3); // Uses level 3
    return 0;
}
```

### Step 2: Multiple Defaults
Defaults must be at the end.

```cpp
void createWindow(int width, int height, std::string title = "Untitled", bool fullscreen = false) {
    std::cout << "Creating " << title << " (" << width << "x" << height << ")";
    if (fullscreen) std::cout << " [FULLSCREEN]";
    std::cout << std::endl;
}
```

### Step 3: Calling
```cpp
createWindow(800, 600);
createWindow(1920, 1080, "My Game");
createWindow(1920, 1080, "My Game", true);
```

## Challenges

### Challenge 1: Forward Declaration
If you use a prototype, where does the default value go?
Try putting it in both. (Error!)
Try putting it only in definition. (Error!)
**Rule:** Put it in the declaration (prototype) only.

### Challenge 2: Skipping Arguments
Can you skip the middle argument?
`createWindow(800, 600, , true);` // Error?
C++ doesn't support named arguments like Python. You must provide all preceding arguments.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

// Prototype with defaults
void log(std::string msg, int level = 1);

int main() {
    log("Test");
    return 0;
}

// Definition without defaults
void log(std::string msg, int level) {
    std::cout << "[" << level << "] " << msg << std::endl;
}
```
</details>

## Success Criteria
✅ Implemented function with default arguments
✅ Called function with and without optional args
✅ Understood "trailing arguments only" rule
✅ Placed defaults in prototype correctly (Challenge 1)

## Key Learnings
- Default arguments simplify APIs
- They must be the last parameters
- Define them in the header/prototype, not the implementation

## Next Steps
Proceed to **Lab 4.7: Inline Functions** to optimize small functions.
