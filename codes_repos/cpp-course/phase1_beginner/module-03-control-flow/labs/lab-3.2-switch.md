# Lab 3.2: Switch Statement Menu System

## Objective
Create a menu-driven program using `switch` statements.

## Instructions

### Step 1: Display Menu
Create `menu.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "=== Main Menu ===" << std::endl;
    std::cout << "1. Start Game" << std::endl;
    std::cout << "2. Options" << std::endl;
    std::cout << "3. Exit" << std::endl;
    std::cout << "Select: ";
    
    int choice;
    std::cin >> choice;
    
    // TODO: Handle choice
    
    return 0;
}
```

### Step 2: Switch Logic
Implement the `switch` statement:

```cpp
switch (choice) {
    case 1:
        std::cout << "Starting game..." << std::endl;
        break;
    case 2:
        std::cout << "Opening options..." << std::endl;
        break;
    case 3:
        std::cout << "Exiting..." << std::endl;
        break;
    default:
        std::cout << "Invalid choice!" << std::endl;
}
```

### Step 3: Fallthrough
Demonstrate fallthrough by adding cases 4 and 5 that do the same thing (e.g., "Hidden Feature").

```cpp
case 4:
case 5:
    std::cout << "You found a secret!" << std::endl;
    break;
```

## Challenges

### Challenge 1: Enum Switch
Define an `enum class MenuOption { Start=1, Options, Exit };`.
Cast the input integer to `MenuOption` and switch on that.

### Challenge 2: [[fallthrough]]
Use the C++17 `[[fallthrough]]` attribute to explicitly mark intentional fallthrough.
```cpp
case 1:
    std::cout << "Initializing..." << std::endl;
    [[fallthrough]];
case 2:
    std::cout << "Loading..." << std::endl;
    break;
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

enum class MenuOption { Start=1, Options, Exit };

int main() {
    int input;
    std::cin >> input;
    MenuOption choice = static_cast<MenuOption>(input);
    
    switch (choice) {
        case MenuOption::Start:
            std::cout << "Start" << std::endl;
            break;
        case MenuOption::Options:
            std::cout << "Options" << std::endl;
            break;
        case MenuOption::Exit:
            std::cout << "Exit" << std::endl;
            break;
        default:
            std::cout << "Invalid" << std::endl;
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented menu with switch
✅ Handled default case
✅ Used fallthrough correctly
✅ Switched on an enum (Challenge 1)

## Key Learnings
- `switch` syntax and `break`
- `default` case importance
- Grouping cases for shared logic

## Next Steps
Proceed to **Lab 3.3: While Loop** to make the menu repeat.
