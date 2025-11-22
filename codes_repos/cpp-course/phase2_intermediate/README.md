# Phase 2: Intermediate - Software Engineering with C++

## 🎯 Phase Overview

Phase 2 focuses on applying software engineering principles with C++, covering advanced language features, design patterns, testing, and professional development practices.

**Duration:** 80-100 hours  
**Modules:** 10  
**Labs:** 100  
**Level:** Intermediate

---

## 📚 Modules

### [Module 01: Smart Pointers and RAII](./module-01-smart-pointers/)
**Duration:** 6-8 hours | **Labs:** 10

- std::unique_ptr for exclusive ownership
- std::shared_ptr for shared ownership
- std::weak_ptr for breaking cycles
- Custom deleters
- RAII patterns in depth

### [Module 02: Templates Fundamentals](./module-02-templates/)
**Duration:** 8-10 hours | **Labs:** 10

- Function templates
- Class templates
- Template specialization
- Type traits
- SFINAE introduction

### [Module 03: Move Semantics](./module-03-move-semantics/)
**Duration:** 8-10 hours | **Labs:** 10

- Lvalue and rvalue references
- Move constructors and assignment
- std::move and std::forward
- Perfect forwarding
- Rule of five

### [Module 04: Operator Overloading](./module-04-operator-overloading/)
**Duration:** 5-6 hours | **Labs:** 10

- Arithmetic operators
- Comparison operators
- Stream operators (<< and >>)
- Function call operator
- Best practices

### [Module 05: Testing and TDD](./module-05-testing-tdd/)
**Duration:** 6-8 hours | **Labs:** 10

- Google Test framework
- Unit testing best practices
- Mocking with Google Mock
- Test-driven development
- Code coverage

### [Module 06: Documentation and APIs](./module-06-documentation-apis/)
**Duration:** 5-6 hours | **Labs:** 10

- Doxygen comments
- API design principles
- Header-only libraries
- ABI compatibility
- Documentation generation

### [Module 07: Design Patterns](./module-07-design-patterns/)
**Duration:** 8-10 hours | **Labs:** 10

- SOLID principles
- Creational patterns
- Structural patterns
- Behavioral patterns
- Modern C++ idioms

### [Module 08: Concurrency Fundamentals](./module-08-concurrency/)
**Duration:** 8-10 hours | **Labs:** 10

- std::thread management
- Mutexes and locks
- Condition variables
- std::atomic operations
- Thread-safe design

### [Module 09: File I/O and Serialization](./module-09-file-io/)
**Duration:** 5-6 hours | **Labs:** 10

- File streams
- Binary I/O
- JSON serialization
- File system operations (C++17)
- Serialization patterns

### [Module 10: Build Systems and Tools](./module-10-build-tools/)
**Duration:** 6-8 hours | **Labs:** 10

- Advanced CMake
- Package managers (vcpkg, Conan)
- Static analysis tools
- Sanitizers
- Continuous integration

---

## 🎓 Learning Outcomes

After completing Phase 2, you will:

✅ **Master Advanced C++**
- Use smart pointers for safe memory management
- Write generic code with templates
- Implement move semantics efficiently
- Overload operators correctly

✅ **Apply Software Engineering**
- Design robust, testable APIs
- Implement design patterns
- Write comprehensive tests
- Use TDD methodology

✅ **Build Production Code**
- Create thread-safe applications
- Handle I/O and serialization
- Use modern build systems
- Apply static analysis

✅ **Professional Practices**
- Generate documentation
- Manage dependencies
- Set up CI/CD pipelines
- Follow C++ best practices

---

## 📊 Progress Tracking

Track your progress through Phase 2:

- [ ] Module 01: Smart Pointers and RAII (10 labs)
- [ ] Module 02: Templates Fundamentals (10 labs)
- [ ] Module 03: Move Semantics (10 labs)
- [ ] Module 04: Operator Overloading (10 labs)
- [ ] Module 05: Testing and TDD (10 labs)
- [ ] Module 06: Documentation and APIs (10 labs)
- [ ] Module 07: Design Patterns (10 labs)
- [ ] Module 08: Concurrency Fundamentals (10 labs)
- [ ] Module 09: File I/O and Serialization (10 labs)
- [ ] Module 10: Build Systems and Tools (10 labs)

**Completion:** 0/100 labs

---

## 🦀 Rust vs C++ in Phase 2

### Smart Pointers
- **C++:** std::unique_ptr, std::shared_ptr, manual implementation
- **Rust:** Box<T>, Rc<T>, Arc<T> built-in with ownership
- **Learning:** C++ offers more control, Rust enforces correctness

### Templates vs Generics
- **C++:** Full template metaprogramming capabilities
- **Rust:** Generics with trait bounds, simpler model
- **Strength:** C++ templates more powerful, Rust generics more predictable

### Concurrency
- **C++:** Thread safety is programmer's responsibility
- **Rust:** Send/Sync traits enforce thread safety at compile-time
- **Trade-off:** C++ flexibility vs Rust safety

### Testing
- **C++:** External frameworks (Google Test, Catch2)
- **Rust:** Built-in cargo test
- **Ecosystem:** Rust has integrated testing from the start

---

## 🚀 Getting Started

1. Complete **Phase 1** before starting Phase 2
2. Review [Phase 1 README](../phase1_beginner/) concepts
3. Start with [Module 01: Smart Pointers](./module-01-smart-pointers/)
4. Complete all 10 labs per module
5. Build progressively complex projects
6. Apply concepts in real applications

---

## 💡 Study Tips

1. **Combine Concepts** - Build projects using multiple modules
2. **Read Standard Library** - Study STL source code
3. **Practice TDD** - Write tests first from Module 05 onward
4. **Use Modern Features** - Embrace C++11/14/17/20
5. **Join Code Reviews** - Learn from other developers
6. **Contribute to OSS** - Apply skills to real projects

---

## ⏭️ Next Phase

After completing Phase 2, proceed to:
**[Phase 3: Advanced - Production Systems](../phase3_advanced/)**

---

**Keep Building!** 🚀
