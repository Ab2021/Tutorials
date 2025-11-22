# Module 12: File I/O and Streams

## 🎯 Learning Objectives

By the end of this module, you will:
- Master the C++ Stream hierarchy (`iostream`, `fstream`, `sstream`).
- Read and write text files using `ifstream` and `ofstream`.
- Handle binary file I/O.
- Use String Streams for in-memory formatting and parsing.
- Manipulate stream state (flags, precision, width).
- overload `<<` and `>>` operators for custom types.
- Understand the C++ filesystem library (`std::filesystem` in C++17).

---

## 📖 Theoretical Concepts

### 12.1 Stream Hierarchy

- **`ios_base`**: Base class for all streams.
- **`istream`**: Input stream (`cin`, `ifstream`).
- **`ostream`**: Output stream (`cout`, `ofstream`).
- **`iostream`**: Bidirectional (`fstream`).

### 12.2 File Streams (`<fstream>`)

```cpp
std::ofstream out("file.txt");
out << "Hello";

std::ifstream in("file.txt");
std::string s;
in >> s;
```

### 12.3 String Streams (`<sstream>`)

Useful for converting between strings and numbers.

```cpp
std::stringstream ss;
ss << "Age: " << 25;
std::string result = ss.str();
```

### 12.4 Filesystem (`<filesystem>`)

Standardized way to manipulate paths and directories (C++17).

```cpp
namespace fs = std::filesystem;
if (fs::exists("path/to/file")) { ... }
```

---

## 🦀 Rust vs C++ Comparison

### File I/O
**C++:** Uses stream operators `<<` and `>>`. RAII handles closing.
**Rust:** Uses `std::fs::File` and `std::io::Read/Write` traits. No `<<` operator for files.

### Path Handling
**C++:** `std::filesystem::path` (C++17).
**Rust:** `std::path::Path` and `PathBuf`.

### Serialization
**C++:** Manual `<<` overloading or libraries (like Boost.Serialization).
**Rust:** `Serde` crate is the de-facto standard (derive `Serialize`/`Deserialize`).

---

## 🔑 Key Takeaways

1.  Always check `if (file.is_open())` or `if (file)` after opening.
2.  Use `std::getline(file, string)` to read whole lines.
3.  Use `std::filesystem` for portable path manipulation.
4.  Binary I/O requires `std::ios::binary` and `read()`/`write()` methods (not `<<`/`>>`).
5.  Stream manipulators (`std::hex`, `std::setw`) modify stream formatting.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1.  **Lab 12.1:** Basic Text I/O
2.  **Lab 12.2:** Reading Line by Line
3.  **Lab 12.3:** String Streams
4.  **Lab 12.4:** Stream Manipulators
5.  **Lab 12.5:** Binary File I/O
6.  **Lab 12.6:** Random Access (Seek)
7.  **Lab 12.7:** Custom Stream Operators
8.  **Lab 12.8:** Error Handling in Streams
9.  **Lab 12.9:** Filesystem Library (C++17)
10. **Lab 12.10:** Config File Parser (Capstone)

After completing the labs, move on to **Module 13: Modern C++ Features**.
