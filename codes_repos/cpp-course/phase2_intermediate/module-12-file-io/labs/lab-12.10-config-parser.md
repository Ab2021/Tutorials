# Lab 12.10: Config File Parser (Capstone)

## Objective
Build a class that reads a configuration file (`key=value`), parses it, and allows querying values.

## Instructions

### Step 1: Config Format
Create `config.txt`:
```
host=127.0.0.1
port=8080
debug=true
# This is a comment
timeout=5.5
```

### Step 2: ConfigParser Class
Create `config_parser.cpp`.
Use `std::map<string, string>` to store data.

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

class Config {
    std::map<std::string, std::string> data;
public:
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file");
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments
            
            std::stringstream ss(line);
            std::string key, val;
            if (std::getline(ss, key, '=') && std::getline(ss, val)) {
                data[key] = val;
            }
        }
    }
    
    std::string get(const std::string& key) {
        if (data.count(key)) return data[key];
        throw std::runtime_error("Key not found: " + key);
    }
};
```

### Step 3: Typed Getters
Add `getInt`, `getDouble`, `getBool`.
Use `stoi`, `stod`, etc.

```cpp
    int getInt(const std::string& key) {
        return std::stoi(get(key));
    }
```

## Challenges

### Challenge 1: Trim Whitespace
Handle `key = value` (spaces around equals).
Implement a `trim` function (from Lab 10.8) and apply it to key and value before storing.

### Challenge 2: Save Config
Implement `void save(filename)` that writes the map back to a file.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <algorithm>

// Helper trim
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}

class Config {
    std::map<std::string, std::string> data;
public:
    void load(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = trim(line.substr(0, delimiterPos));
                std::string value = trim(line.substr(delimiterPos + 1));
                data[key] = value;
            }
        }
    }
    
    std::string getString(const std::string& key) { return data.at(key); }
    int getInt(const std::string& key) { return std::stoi(data.at(key)); }
    
    void save(const std::string& filename) {
        std::ofstream file(filename);
        for(auto const& [key, val] : data) {
            file << key << "=" << val << "\n";
        }
    }
};

int main() {
    // Create dummy file
    std::ofstream out("config.ini");
    out << "port = 9090\n# Comment\nhost=localhost";
    out.close();
    
    Config cfg;
    cfg.load("config.ini");
    std::cout << "Port: " << cfg.getInt("port") << "\n";
    std::cout << "Host: " << cfg.getString("host") << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Parsed key-value pairs
✅ Handled comments and empty lines
✅ Implemented typed getters
✅ Handled whitespace trimming (Challenge 1)
✅ Implemented save functionality (Challenge 2)

## Key Learnings
- Parsing text files is a common task
- String manipulation (trim, split) is essential
- `std::map` is perfect for key-value storage

## Next Steps
Congratulations! You've completed Module 12.

Proceed to **Module 13: Modern C++ Features** to learn about `auto`, `nullptr`, `constexpr`, and more.
