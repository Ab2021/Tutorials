# Module 25: Serialization

## Overview
Data serialization and deserialization techniques for storing and transmitting structured data.

## Learning Objectives
By the end of this module, you will be able to:
- Implement binary serialization
- Work with JSON and XML
- Use Protocol Buffers
- Apply MessagePack
- Create custom serialization frameworks
- Handle versioning and compatibility

## Key Concepts

### 1. Binary Serialization
Efficient binary data formats.

```cpp
class Serializer {
public:
    template<typename T>
    void write(const T& value) {
        buffer.insert(buffer.end(), 
            reinterpret_cast<const char*>(&value),
            reinterpret_cast<const char*>(&value) + sizeof(T));
    }
    
    template<typename T>
    T read() {
        T value;
        std::memcpy(&value, &buffer[offset], sizeof(T));
        offset += sizeof(T);
        return value;
    }
};
```

### 2. JSON Handling
Working with JSON data.

```cpp
#include <nlohmann/json.hpp>

nlohmann::json j = {
    {"name", "John"},
    {"age", 30},
    {"skills", {"C++", "Python"}}
};

std::string serialized = j.dump();
auto parsed = nlohmann::json::parse(serialized);
```

### 3. Protocol Buffers
Google's data interchange format.

```protobuf
message Person {
    string name = 1;
    int32 age = 2;
    repeated string skills = 3;
}
```

```cpp
Person person;
person.set_name("John");
person.set_age(30);
std::string serialized = person.SerializeAsString();
```

### 4. MessagePack
Efficient binary serialization.

```cpp
#include <msgpack.hpp>

struct Person {
    std::string name;
    int age;
    MSGPACK_DEFINE(name, age);
};

Person p{"John", 30};
msgpack::sbuffer buffer;
msgpack::pack(buffer, p);
```

### 5. Custom Serialization
Building your own serialization framework.

```cpp
template<typename T>
concept Serializable = requires(T t, Serializer& s) {
    { t.serialize(s) } -> std::same_as<void>;
    { T::deserialize(s) } -> std::same_as<T>;
};
```

## Rust Comparison

### JSON
**C++:**
```cpp
nlohmann::json j = {{"key", "value"}};
```

**Rust:**
```rust
use serde_json::json;
let j = json!({"key": "value"});
```

### Serialization Traits
**C++:**
```cpp
concept Serializable = ...
```

**Rust:**
```rust
use serde::{Serialize, Deserialize};
#[derive(Serialize, Deserialize)]
struct Person { }
```

## Labs

1. **Lab 25.1**: Binary Serialization Basics
2. **Lab 25.2**: Endianness Handling
3. **Lab 25.3**: JSON with nlohmann/json
4. **Lab 25.4**: XML Parsing
5. **Lab 25.5**: Protocol Buffers
6. **Lab 25.6**: MessagePack
7. **Lab 25.7**: Custom Serialization Framework
8. **Lab 25.8**: Versioning and Compatibility
9. **Lab 25.9**: Schema Evolution
10. **Lab 25.10**: Data Exchange System (Capstone)

## Additional Resources
- nlohmann/json documentation
- Protocol Buffers guide
- MessagePack specification

## Next Module
After completing this module, proceed to **Module 26: Reflection and Metaprogramming**.
