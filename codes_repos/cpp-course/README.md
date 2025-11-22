# 🚀 Comprehensive C++ Programming Course

## Three-Phase Learning Path

Welcome to the most comprehensive C++ programming course! This course is structured in three progressive phases, taking you from complete beginner to production-ready C++ developer.

---

## 📚 Course Structure

### 🟢 [Phase 1: Beginner - Foundation](./phase1_beginner/)
**Duration:** 60-80 hours | **Modules:** 10 | **Labs:** 100

Master the fundamentals of C++ programming.

| Module | Topic | Labs | Focus |
|--------|-------|------|-------|
| 01 | Getting Started | 10 | Installation, compilers, first programs |
| 02 | Variables and Types | 10 | Data types, auto, constexpr, enums |
| 03 | Control Flow | 10 | Loops, conditionals, switch statements |
| 04 | Functions and Scope | 10 | Parameters, overloading, scope rules |
| 05 | Pointers and References | 10 | Pointers, references, const correctness |
| 06 | Memory Management Basics | 10 | Stack/heap, new/delete, RAII intro |
| 07 | Classes and Objects | 10 | Class definitions, constructors, destructors |
| 08 | STL Containers | 10 | vector, string, map, set, iterators |
| 09 | Error Handling | 10 | Exceptions, try-catch, optional, expected |
| 10 | Modules and Organization | 10 | Headers, namespaces, C++20 modules |

### 🟡 [Phase 2: Intermediate - Software Engineering](./phase2_intermediate/)
**Duration:** 80-100 hours | **Modules:** 10 | **Labs:** 100

Apply software engineering principles with C++.

| Module | Topic | Labs | Focus |
|--------|-------|------|-------|
| 01 | Smart Pointers and RAII | 10 | unique_ptr, shared_ptr, weak_ptr, RAII |
| 02 | Templates Fundamentals | 10 | Function/class templates, specialization |
| 03 | Move Semantics | 10 | Rvalue refs, move constructors, forwarding |
| 04 | Operator Overloading | 10 | Arithmetic, comparison, stream operators |
| 05 | Testing and TDD | 10 | Google Test, unit tests, mocking |
| 06 | Documentation and APIs | 10 | Doxygen, API design, header-only libs |
| 07 | Design Patterns | 10 | SOLID, Factory, Observer, Strategy |
| 08 | Concurrency Fundamentals | 10 | std::thread, mutexes, atomics |
| 09 | File I/O and Serialization | 10 | Streams, binary I/O, JSON |
| 10 | Build Systems and Tools | 10 | CMake, vcpkg, Conan, sanitizers |

### 🔴 [Phase 3: Advanced - Production Systems](./phase3_advanced/)
**Duration:** 60-80 hours | **Modules:** 10 | **Labs:** 100

Build production-ready systems and applications.

| Module | Topic | Labs | Focus |
|--------|-------|------|-------|
| 01 | Advanced Templates | 10 | Metaprogramming, concepts, variadic |
| 02 | Coroutines and Async | 10 | C++20 coroutines, async patterns |
| 03 | Advanced Memory | 10 | Custom allocators, memory pools |
| 04 | Modern C++ Features | 10 | Ranges, modules, format, C++20/23 |
| 05 | Performance Optimization | 10 | Profiling, SIMD, cache optimization |
| 06 | Network Programming | 10 | Sockets, Boost.Asio, HTTP servers |
| 07 | GUI Development | 10 | Qt, Dear ImGui, event-driven design |
| 08 | Database Integration | 10 | SQLite, PostgreSQL, ORMs, pooling |
| 09 | Systems Programming | 10 | OS interfaces, IPC, system calls |
| 10 | Production Deployment | 10 | Docker, CI/CD, cross-platform builds |

---

## 📊 Course Statistics

- **Total Modules:** 30 (10 per phase)
- **Total Labs:** 300 (10 per module)
- **Total Learning Time:** 200-260 hours
- **Projects:** 50+ real-world applications
- **Exercises:** 1000+ hands-on exercises

---

## 🚀 Quick Start

### Prerequisites
- Basic programming knowledge (any language)
- Command-line familiarity
- 200+ hours of dedicated learning time

### Installation

**Windows (MSVC):**
Download Visual Studio Community Edition from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/)

**Windows (MinGW-w64):**
```powershell
winget install -e --id MSYS2.MSYS2
# Then install GCC via MSYS2
```

**Linux:**
```bash
sudo apt-get install build-essential cmake  # Ubuntu/Debian
sudo dnf install gcc-c++ cmake              # Fedora
```

**macOS:**
```bash
xcode-select --install
brew install cmake
```

**Verify:**
```bash
g++ --version     # or clang++ --version
cmake --version
```

### Learning Path

1. **Start with Phase 1** - Complete all 10 modules sequentially
2. **Progress to Phase 2** - After mastering Phase 1
3. **Advance to Phase 3** - For production-level skills

**Recommended Pace:**
- Phase 1: 2-3 months (part-time) or 3-4 weeks (full-time)
- Phase 2: 2-3 months (part-time) or 4-5 weeks (full-time)
- Phase 3: 2-3 months (part-time) or 3-4 weeks (full-time)

---

## 📖 How to Use This Course

### For Self-Learners
1. Read [GETTING_STARTED.md](./GETTING_STARTED.md)
2. Start with Phase 1, Module 01
3. Complete all labs before moving forward
4. Use [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) as needed
5. Join the C++ community for support

### For Instructors
- Use as complete curriculum
- Adapt labs for classroom settings
- Assign projects from each phase
- Leverage community resources

### For Teams
- Onboard new C++ developers
- Structured learning path
- Hands-on practice
- Production-ready skills

---

## 🎯 Learning Outcomes

### After Phase 1 (Beginner)
✅ Understand C++ fundamentals  
✅ Master pointers and references  
✅ Write safe, correct code with RAII  
✅ Handle errors properly  
✅ Organize code with headers and namespaces

### After Phase 2 (Intermediate)
✅ Apply software engineering principles  
✅ Design robust APIs  
✅ Implement design patterns  
✅ Write comprehensive tests  
✅ Build concurrent applications

### After Phase 3 (Advanced)
✅ Build production systems  
✅ Optimize performance  
✅ Develop web services  
✅ Integrate databases  
✅ Deploy to production

---

## 🦀 Rust vs C++ Comparison

Throughout this course, you'll find comparisons between Rust and C++ to help you understand:

### Memory Safety
- **Rust:** Ownership system enforced at compile-time, borrow checker prevents data races
- **C++:** RAII pattern, smart pointers, optional runtime checks (sanitizers)
- **Tradeoff:** Rust enforces safety, C++ gives flexibility with discipline required

### Concurrency
- **Rust:** Send/Sync traits, fearless concurrency guarantees
- **C++:** Thread safety is programmer's responsibility
- **Best Practice:** C++ can be safe with proper patterns and tools

### Learning Curve
- **Rust:** Steep initially due to ownership, then smooth
- **C++:** Gentler start, but endless depth and complexity
- **Recommendation:** Both are powerful; choose based on your needs

### Performance
- **Both:** Zero-cost abstractions, excellent performance
- **C++:** More mature ecosystem, broader tooling
- **Rust:** Modern design, focusing on safety without sacrificing speed

---

## 📚 Additional Resources

### Official Documentation
- [C++ Reference](https://en.cppreference.com/)
- [ISO C++ Website](https://isocpp.org/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

### Community
- [C++ Subreddit](https://www.reddit.com/r/cpp/)
- [Stack Overflow C++ Tag](https://stackoverflow.com/questions/tagged/c++)
- [C++ Slack](https://cpplang.slack.com/)

### Practice
- [LeetCode C++ Problems](https://leetcode.com/problemset/all/)
- [HackerRank C++](https://www.hackerrank.com/domains/cpp)
- [Exercism C++ Track](https://exercism.org/tracks/cpp)

---

## 🏆 Certification Path

Complete all three phases to achieve:
- **Beginner Certification**: Phase 1 complete
- **Intermediate Certification**: Phases 1-2 complete
- **Advanced Certification**: All phases complete

---

## 🤝 Contributing

This course is continuously improved. Contributions welcome!

---

## 📝 License

Educational use - Free and open

---

**Start Your C++ Journey Today!** 🚀

Begin with [Phase 1: Getting Started](./phase1_beginner/module-01-getting-started/)
