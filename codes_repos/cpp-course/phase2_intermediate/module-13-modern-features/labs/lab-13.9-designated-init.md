# Lab 13.9: Designated Initializers (C++20)

## Objective
Initialize structs by explicitly naming fields.

## Instructions

### Step 1: Traditional Initialization
Create `designated_init.cpp`.

```cpp
#include <iostream>

struct Config {
    int port;
    std::string host;
    bool debug;
};

int main() {
    // Old way: Must match order
    Config c1{8080, "localhost", true};
    
    return 0;
}
```

### Step 2: Designated Initializers
Name the fields explicitly (C++20).

```cpp
Config c2{
    .port = 9090,
    .host = "127.0.0.1",
    .debug = false
};
```

### Step 3: Partial Initialization
You can skip fields (they get default-initialized).

```cpp
Config c3{
    .port = 3000
    // host and debug use default values
};
```

## Challenges

### Challenge 1: Order Matters
In C++20, designated initializers must be in declaration order.
Try to initialize out of order and observe the error.

### Challenge 2: Nested Structs
```cpp
struct Server {
    Config config;
    int maxConnections;
};

Server s{
    .config = {.port = 8080, .host = "0.0.0.0", .debug = true},
    .maxConnections = 100
};
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

struct Config {
    int port = 8080;
    std::string host = "localhost";
    bool debug = false;
};

struct Server {
    Config config;
    int maxConnections = 50;
};

int main() {
    // Designated initializers
    Config c{
        .port = 9090,
        .host = "127.0.0.1",
        .debug = true
    };
    
    std::cout << "Port: " << c.port << ", Host: " << c.host << "\n";
    
    // Challenge 2: Nested
    Server s{
        .config = {.port = 3000, .host = "0.0.0.0"},
        .maxConnections = 200
    };
    
    std::cout << "Server port: " << s.config.port << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used designated initializers
✅ Understood field order requirement
✅ Used partial initialization
✅ Initialized nested structs (Challenge 2)

## Key Learnings
- Designated initializers improve readability
- Fields must be in declaration order (C++ restriction, unlike C)
- Great for configuration structs

## Next Steps
Proceed to **Lab 13.10: Spaceship Operator** for automatic comparisons.
