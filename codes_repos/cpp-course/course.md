# C++ Programming Course: From Basics to Advanced

## 📚 Course Overview

Welcome to the comprehensive C++ programming course! This course is designed to take you from a complete beginner to an advanced C++ developer. C++ is a powerful, high-performance programming language that powers everything from operating systems to game engines, offering unparalleled control and efficiency.

### Course Objectives
- Master modern C++ (C++11/14/17/20/23)
- Understand memory management and RAII
- Build safe, performant applications
- Learn production-level software engineering
- Develop real-world C++ systems

### Prerequisites
- Basic programming knowledge (any language)
- Familiarity with command-line interfaces
- A computer with a C++ compiler installed

---

## 🦀 Rust vs C++: Understanding Both Languages

Throughout this course, we'll include comparisons with Rust to help you understand the strengths and trade-offs of each language.

### When to Choose C++
- Legacy code integration required
- Maximum platform compatibility needed
- Existing C++ ecosystem (libraries, tools)
- Game development, graphics programming
- Incremental adoption of modern features

### When to Choose Rust
- New projects prioritizing memory safety
- Fearless concurrency is critical
- Web assembly or embedded systems
- Want compile-time safety guarantees
- Prefer integrated tooling (Cargo)

### Key Philosophical Differences
- **C++:** "Trust the programmer" - flexibility and power
- **Rust:** "Safety by default" - compiler-enforced correctness
- **Both:** Zero-cost abstractions and excellent performance

---

## 📖 Course Modules

### Module 1: Getting Started with C++
**Duration:** 4-6 hours  
**Folder:** `phase1_beginner/module-01-getting-started/`

**Topics:**
- What is C++ and its evolution (C++98 to C++23)
- Installing compilers (GCC, Clang, MSVC)
- Understanding the compilation process
- Introduction to build systems (Make, CMake)
- Your first C++ program: "Hello, World!"
- Basic project structure

**Labs:**
- Lab 1.1: Compiler installation and verification
- Lab 1.2: Hello World variations
- Lab 1.3: Building a simple calculator
- Lab 1.4: Understanding compilation flags
- Lab 1.5: Multi-file programs
- Lab 1.6: Basic CMake project
- Lab 1.7: Working with header files
- Lab 1.8: Namespace basics
- Lab 1.9: Command-line arguments
- Lab 1.10: Build configurations (Debug vs Release)

**Rust vs C++:**
- Cargo vs CMake/Make
- Single compiler (rustc) vs multiple (GCC/Clang/MSVC)
- Integrated tooling comparison

---

### Module 2: Variables and Types
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-02-variables-types/`

**Topics:**
- Fundamental types: int, float, double, char, bool
- Type inference with `auto` (C++11)
- Constants: `const` and `constexpr`
- Type conversions (implicit and explicit)
- Strongly-typed enums (enum class)
- `sizeof` operator and type limits
- User-defined literals

**Labs:**
- Lab 2.1: Type exploration and limits
- Lab 2.2: Auto type deduction exercises
- Lab 2.3: Const and constexpr practice
- Lab 2.4: Type conversion safety
- Lab 2.5: Enum class vs traditional enums
- Lab 2.6: Temperature converter with strong types
- Lab 2.7: Custom literals
- Lab 2.8: Type size and alignment
- Lab 2.9: Numeric limits exploration
- Lab 2.10: Building a type-safe units library

**Rust vs C++:**
- Type inference: `auto` vs automatic inference in Rust
- Immutability: `const` vs Rust's default immutability
- Enums: C++ enum class vs Rust's powerful enums

---

### Module 3: Control Flow
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-03-control-flow/`

**Topics:**
- if/else statements and ternary operator
- Switch statements and case labels
- Loops: for, while, do-while
- Range-based for loops (C++11)
- Break and continue
- Goto (and why to avoid it)
- Structured bindings (C++17)

**Labs:**
- Lab 3.1: Conditional logic exercises
- Lab 3.2: Switch statement practice
- Lab 3.3: Loop variations
- Lab 3.4: Range-based for with containers
- Lab 3.5: FizzBuzz challenge
- Lab 3.6: Prime number generator
- Lab 3.7: Number guessing game
- Lab 3.8: Pattern printing
- Lab 3.9: Structured bindings with pairs
- Lab 3.10: Control flow optimization

**Rust vs C++:**
- Pattern matching (Rust) vs switch/if-else (C++)
- Loop labels comparison
- Expression-based control flow in Rust

---

### Module 4: Functions and Scope
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-04-functions-scope/`

**Topics:**
- Function declarations and definitions
- Parameters: pass by value, reference, const reference
- Return types and return value optimization (RVO)
- Function overloading
- Default arguments
- Inline functions
- Scope and lifetime
- Storage duration (auto, static, dynamic)

**Labs:**
- Lab 4.1: Function basics
- Lab 4.2: Pass by value vs reference
- Lab 4.3: Const correctness in parameters
- Lab 4.4: Function overloading
- Lab 4.5: Default arguments usage
- Lab 4.6: Inline functions and performance
- Lab 4.7: Scope and lifetime exploration
- Lab 4.8: Static local variables
- Lab 4.9: Mathematical functions library
- Lab 4.10: String manipulation utilities

**Rust vs C++:**
- Parameter passing: Rust's ownership vs C++ references
- Function overloading (C++) vs traits (Rust)
- Return value optimization comparison

---

### Module 5: Pointers and References
**Duration:** 8-10 hours  
**Folder:** `phase1_beginner/module-05-pointers-references/`

**Topics:**
- Pointer basics and syntax
- References and their rules
- nullptr (C++11) vs NULL
- Pointer arithmetic
- Const pointers and pointers to const
- References vs pointers
- Dangling pointers and references
- Best practices for safe pointer usage

**Labs:**
- Lab 5.1: Pointer fundamentals
- Lab 5.2: Reference basics
- Lab 5.3: Const correctness with pointers
- Lab 5.4: Pointer arithmetic exercises
- Lab 5.5: Avoiding dangling pointers
- Lab 5.6: Pointers vs references comparison
- Lab 5.7: Swap function implementations
- Lab 5.8: Pointer-based linked list
- Lab 5.9: Memory address exploration
- Lab 5.10: Safe pointer patterns

**Rust vs C++:**
- Raw pointers (C++) vs Rust references
- Borrow checker vs manual safety in C++
- Lifetimes in Rust vs scope in C++
- When to use unsafe Rust vs C++ pointers

---

### Module 6: Memory Management Basics
**Duration:** 8-10 hours  
**Folder:** `phase1_beginner/module-06-memory-basics/`

**Topics:**
- Stack vs heap memory
- Dynamic allocation: new and delete
- Arrays: new[] and delete[]
- Memory leaks and how to prevent them
- Introduction to RAII (Resource Acquisition Is Initialization)
- Scope-based resource management
- Common memory errors

**Labs:**
- Lab 6.1: Stack vs heap exploration
- Lab 6.2: Dynamic allocation basics
- Lab 6.3: Array allocation and deallocation
- Lab 6.4: Detecting memory leaks
- Lab 6.5: RAII introduction with simple wrapper
- Lab 6.6: Scope-based lifetime management
- Lab 6.7: Building a simple vector class
- Lab 6.8: Memory error examples and fixes
- Lab 6.9: Dynamic string implementation
- Lab 6.10: Resource management patterns

**Rust vs C++:**
- Ownership system (Rust) vs manual new/delete (C++)
- RAII in C++ vs automatic drops in Rust
- No garbage collector in both languages
- Safety guarantees: compile-time (Rust) vs runtime (C++ sanitizers)

---

### Module 7: Classes and Objects
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-07-classes-objects/`

**Topics:**
- Class definitions and instantiation
- Constructors (default, parameterized, copy)
- Destructors and cleanup
- Member functions and methods
- Access specifiers: private, public, protected
- The `this` pointer
- Const member functions
- Static members

**Labs:**
- Lab 7.1: Basic class creation
- Lab 7.2: Constructor variations
- Lab 7.3: Destructor and RAII
- Lab 7.4: Member function implementation
- Lab 7.5: Access control practice
- Lab 7.6: This pointer usage
- Lab 7.7: Const member functions
- Lab 7.8: Static members
- Lab 7.9: Building a BankAccount class
- Lab 7.10: Rectangle class with methods

**Rust vs C++:**
- Classes (C++) vs structs with impl blocks (Rust)
- Inheritance in C++ vs composition in Rust
- Methods and self in Rust vs this in C++

---

### Module 8: STL Containers
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-08-stl-containers/`

**Topics:**
- std::vector - dynamic arrays
- std::string - string handling
- std::map and std::unordered_map
- std::set and std::unordered_set
- std::array - fixed-size arrays (C++11)
- Iterators and iterator types
- Range-based for loops with containers
- Common algorithms with containers

**Labs:**
- Lab 8.1: Vector operations
- Lab 8.2: String manipulation
- Lab 8.3: Map for key-value storage
- Lab 8.4: Set for unique elements
- Lab 8.5: Iterator usage
- Lab 8.6: Building a phonebook with map
- Lab 8.7: Text analysis with containers
- Lab 8.8: Container comparison and selection
- Lab 8.9: Custom sorting with containers
- Lab 8.10: Implementing a word counter

**Rust vs C++:**
- Vec vs std::vector
- String vs std::string
- HashMap vs std::unordered_map
- Ownership with containers comparison

---

### Module 9: Error Handling
**Duration:** 6-8 hours  
**Folder:** `phase1_beginner/module-09-error-handling/`

**Topics:**
- Exceptions: throw, try, catch
- Exception types and hierarchy
- Stack unwinding
- RAII and exception safety
- noexcept specification (C++11)
- std::optional (C++17)
- std::expected (C++23)
- Error codes vs exceptions

**Labs:**
- Lab 9.1: Basic exception handling
- Lab 9.2: Custom exception types
- Lab 9.3: Exception safety with RAII
- Lab 9.4: noexcept practice
- Lab 9.5: std::optional usage
- Lab 9.6: File operations with exceptions
- Lab 9.7: Exception vs error codes
- Lab 9.8: Strong exception guarantee
- Lab 9.9: Building safe resource handlers
- Lab 9.10: Error handling strategies

**Rust vs C++:**
- Result<T, E> (Rust) vs exceptions (C++)
- ? operator (Rust) vs try-catch (C++)
- std::optional vs Option<T>
- Error handling philosophies

---

### Module 10: Modules and Organization
**Duration:** 5-7 hours  
**Folder:** `phase1_beginner/module-10-modules-organization/`

**Topics:**
- Header files (.h) and source files (.cpp)
- Include guards and #pragma once
- Namespaces and name resolution
- Translation units and linking
- Forward declarations
- C++20 modules (modern approach)
- Library creation and usage
- Organizing large projects

**Labs:**
- Lab 10.1: Header and source file separation
- Lab 10.2: Include guards vs pragma once
- Lab 10.3: Namespace usage
- Lab 10.4: Forward declarations
- Lab 10.5: Creating a static library
- Lab 10.6: Creating a shared library
- Lab 10.7: C++20 modules basics
- Lab 10.8: Multi directory project
- Lab 10.9: Build system organization
- Lab 10.10: Reusable math library project

**Rust vs C++:**
- Cargo modules vs C++ headers
- Module system (C++20) vs Rust's module system
- Visibility: pub in Rust vs public in C++
- Build system integration comparison

---

## 🎓 Phase 2: Intermediate Modules

### Module 11: Smart Pointers and RAII
**Duration:** 6-8 hours  
**Folder:** `phase2_intermediate/module-01-smart-pointers/`

**Topics:**
- std::unique_ptr - exclusive ownership
- std::shared_ptr - shared ownership
- std::weak_ptr - breaking cycles
- Custom deleters
- make_unique and make_shared
- RAII patterns in depth
- Resource management best practices

**Rust vs C++:**
- Box<T> vs std::unique_ptr
- Rc<T> vs std::shared_ptr
- Weak<T> vs std::weak_ptr
- Ownership semantics comparison

---

### Module 12: Templates Fundamentals
**Duration:** 8-10 hours  
**Folder:** `phase2_intermediate/module-02-templates/`

**Topics:**
- Function templates
- Class templates
- Template specialization
- Template parameters (type and non-type)
- SFINAE basics
- Type traits
- Template template parameters

**Rust vs C++:**
- Generics (Rust) vs templates (C++)
- Monomorphization in both languages
- Trait bounds vs template constraints
- Concepts (C++20) vs traits

---

### Module 13: Move Semantics
**Duration:** 8-10 hours  
**Folder:** `phase2_intermediate/module-03-move-semantics/`

**Topics:**
- Lvalue and rvalue references
- Move constructors and move assignment
- std::move and std::forward
- Perfect forwarding
- Rule of five
- Copy elision and RVO
- Universal references

**Rust vs C++:**
- Move semantics in C++ vs ownership transfer in Rust
- Explicit std::move vs implicit moves in Rust
- Copy trait vs copy constructors

---

### Module 14: Operator Overloading
**Duration:** 5-6 hours  
**Folder:** `phase2_intermediate/module-04-operator-overloading/`

**Topics:**
- Arithmetic operators (+, -, *, /)
- Comparison operators (==, !=, <, >, etc.)
- Stream operators (<< and >>)
- Function call operator ()
- Conversion operators
- Increment/decrement operators
- Best practices and pitfalls

**Rust vs C++:**
- Traits (Add, Sub, etc.) vs operator overloading
- Explicit trait implementation vs operator functions
- Deref trait vs operator* and operator->

---

### Module 15: Testing and TDD
**Duration:** 6-8 hours  
**Folder:** `phase2_intermediate/module-05-testing-tdd/`

**Topics:**
- Google Test framework
- Unit testing best practices
- Test fixtures and setup/teardown
- Mocking with Google Mock
- Test-driven development workflow
- Code coverage tools
- Integration testing

---

### Module 16: Documentation and APIs
**Duration:** 5-6 hours  
**Folder:** `phase2_intermediate/module-06-documentation-apis/`

**Topics:**
- Doxygen comments and generation
- API design principles
- Header-only libraries
- ABI compatibility
- Semantic versioning
- Public vs private interfaces
- Documentation best practices

---

### Module 17: Design Patterns
**Duration:** 8-10 hours  
**Folder:** `phase2_intermediate/module-07-design-patterns/`

**Topics:**
- SOLID principles
- Creational: Factory, Builder, Singleton
- Structural: Adapter, Decorator, Facade
- Behavioral: Observer, Strategy, Command
- Modern C++ idioms
- Pattern anti-patterns

---

### Module 18: Concurrency Fundamentals
**Duration:** 8-10 hours  
**Folder:** `phase2_intermediate/module-08-concurrency/`

**Topics:**
- std::thread creation and management
- Mutexes and locks
- lock_guard and unique_lock
- Condition variables
- std::atomic operations
- Thread-safe data structures
- Deadlock prevention

**Rust vs C++:**
- Send and Sync traits vs manual thread safety
- Fearless concurrency in Rust vs careful programming in C++
- Data race prevention: compile-time vs runtime

---

### Module 19: File I/O and Serialization
**Duration:** 5-6 hours  
**Folder:** `phase2_intermediate/module-09-file-io/`

**Topics:**
- File streams (ifstream, ofstream)
- Binary I/O operations
- Serialization patterns
- JSON libraries (nlohmann/json)
- XML and other formats
- File system operations (C++17)

---

### Module 20: Build Systems and Tools
**Duration:** 6-8 hours  
**Folder:** `phase2_intermediate/module-10-build-tools/`

**Topics:**
- CMake advanced features
- Package managers: vcpkg, Conan
- Static analysis tools
- Address and undefined behavior sanitizers
- Continuous integration
- Cross-compilation basics

---

## 🚀 Phase 3: Advanced Modules

### Module 21: Advanced Templates and Metaprogramming
**Duration:** 10-12 hours  
**Folder:** `phase3_advanced/module-01-advanced-templates/`

**Topics:**
- Variadic templates
- Template metaprogramming techniques
- Concepts (C++20)
- constexpr and consteval
- Compile-time computation
- CRTP (Curiously Recurring Template Pattern)

**Rust vs C++:**
- Const generics vs non-type template parameters
- Compile-time computation capabilities
- Procedural macros vs templates

---

### Module 22: Coroutines and Async Programming
**Duration:** 10-12 hours  
**Folder:** `phase3_advanced/module-02-coroutines/`

**Topics:**
- C++20 coroutines introduction
- co_await, co_yield, co_return
- Promise and awaitable types
- Generators and async tasks
- Coroutine handle and state
- Building async frameworks

**Rust vs C++:**
- async/await in both languages
- tokio (Rust) vs coroutine libraries (C++)
- Futures comparison

---

### Module 23: Advanced Memory Management
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-03-advanced-memory/`

**Topics:**
- Custom allocators
- Memory pools and arenas
- Placement new
- Memory-mapped files
- Cache-friendly data structures
- Small buffer optimization

---

### Module 24: Modern C++ Features (C++20/23)
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-04-modern-features/`

**Topics:**
- Ranges library
- Modules (C++20)
- std::format (C++20)
- Designated initializers
- Three-way comparison (spaceship operator)
- std::span
- Calendar and time zones

---

### Module 25: Performance Optimization
**Duration:** 10-12 hours  
**Folder:** `phase3_advanced/module-05-performance/`

**Topics:**
- Profiling tools (perf, vtune, gprof)
- Benchmarking with Google Benchmark
- SIMD programming
- Cache optimization techniques
- Branch prediction optimization
- Link-time optimization

**Rust vs C++:**
- Performance parity in most cases
- Optimization opportunities in both
- Profiling tools ecosystem

---

### Module 26: Network Programming
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-06-network/`

**Topics:**
- Socket programming basics
- TCP and UDP protocols
- Boost.Asio library
- HTTP client and server
- WebSocket implementation
- Protocol design

---

### Module 27: GUI Development
**Duration:** 10-12 hours  
**Folder:** `phase3_advanced/module-07-gui/`

**Topics:**
- Qt framework basics
- Dear ImGui for tools
- wxWidgets overview
- Event-driven programming
- MVC and MVVM patterns
- UI best practices

---

### Module 28: Database Integration
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-08-database/`

**Topics:**
- SQLite integration
- PostgreSQL with libpq
- ORM libraries
- Connection pooling
- Transaction management
- Migration strategies

---

### Module 29: Systems Programming
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-09-systems/`

**Topics:**
- Operating system interfaces
- Process management
- Inter-process communication
- Signal handling
- System calls
- Memory-mapped I/O

---

### Module 30: Production Deployment
**Duration:** 8-10 hours  
**Folder:** `phase3_advanced/module-10-production/`

**Topics:**
- Docker containerization
- CI/CD pipelines
- Static vs dynamic linking
- Cross-platform builds
- Debugging in production
- Monitoring and logging
- Security best practices

---

## 🎯 Final Projects

After completing all modules, choose one or more final projects:

1. **Command-Line Tool**: Build a file search utility with advanced filters
2. **Web Server**: Create a multi-threaded HTTP server
3. **Game Engine**: Develop a 2D game engine with rendering
4. **Database System**: Implement a key-value store with transactions
5. **Chat Application**: Build a concurrent chat server and client
6. **Compiler**: Create a simple language compiler or interpreter
7. **GUI Application**: Desktop application with Qt or wxWidgets

---

## 📚 Additional Resources

### Official Documentation
- [C++ Reference](https://en.cppreference.com/)
- [ISO C++](https://isocpp.org/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Compiler Explorer](https://godbolt.org/)

### Books
- "Effective Modern C++" by Scott Meyers
- "C++ Primer" by Stanley Lippman
- "The C++ Programming Language" by Bjarne Stroustrup

### Community Resources
- [r/cpp](https://www.reddit.com/r/cpp/)
- [C++ Slack](https://cpplang.slack.com/)
- [CppCon Talks](https://www.youtube.com/user/CppCon)

### Practice Platforms
- [LeetCode](https://leetcode.com/)
- [HackerRank C++](https://www.hackerrank.com/domains/cpp)
- [Exercism C++ Track](https://exercism.org/tracks/cpp)

---

## 🗓️ Recommended Learning Path

### Beginner Track (Weeks 1-8)
- Modules 1-10 (Phase 1)
- Focus on fundamentals and memory management
- Complete all labs

### Intermediate Track (Weeks 9-20)
- Modules 11-20 (Phase 2)
- Build progressively complex projects
- Study design patterns deeply

### Advanced Track (Weeks 21-32)
- Modules 21-30 (Phase 3)
- Work on final projects
- Contribute to open source C++ projects

---

## 💡 Tips for Success

1. **Practice Daily**: Code in C++ every day, consistency is key
2. **Read Compiler Messages**: Modern compilers give excellent diagnostics
3. **Use Sanitizers**: AddressSanitizer and UBSanitizer catch bugs early
4. **Join the Community**: Engage with other C++ developers
5. **Build Real Projects**: Apply concepts to actual applications
6. **Read Quality Code**: Study well-written C++ on GitHub
7. **Learn Modern C++**: Focus on C++11 and later features
8. **Master RAII**: This is the foundation of safe C++ code

---

## 📝 Course Structure

Each module contains:
- `README.md`: Detailed theoretical concepts and examples
- `labs/`: Hands-on exercises with starter code (10 labs per module)
- Solutions included in lab files under collapsible sections

---

## 🚀 Getting Started

1. Install a C++ compiler (see [GETTING_STARTED.md](./GETTING_STARTED.md))
2. Set up CMake for build management
3. Start with Module 1: `cd phase1_beginner/module-01-getting-started`
4. Read the module README.md
5. Complete labs in order
6. Check solutions only after attempting

---

**Happy Learning! 🚀**
