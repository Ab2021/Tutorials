# Lab 15.10: Resource Manager (Capstone)

## Objective
Build a resource manager using smart pointers to manage textures, sounds, or other assets.

## Instructions

### Step 1: Resource Class
Create `resource_manager.cpp`.

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Texture {
    std::string filename;
public:
    Texture(std::string file) : filename(file) {
        std::cout << "Loading texture: " << filename << "\n";
    }
    
    ~Texture() {
        std::cout << "Unloading texture: " << filename << "\n";
    }
    
    void use() const {
        std::cout << "Using texture: " << filename << "\n";
    }
};
```

### Step 2: Resource Manager
Use `shared_ptr` for shared ownership, `weak_ptr` for cache.

```cpp
class ResourceManager {
    std::map<std::string, std::weak_ptr<Texture>> cache;
    
public:
    std::shared_ptr<Texture> load(const std::string& filename) {
        // Check cache
        auto it = cache.find(filename);
        if (it != cache.end()) {
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << "\n";
                return sp;
            }
        }
        
        // Load new
        std::cout << "Cache miss: " << filename << "\n";
        auto texture = std::make_shared<Texture>(filename);
        cache[filename] = texture;
        return texture;
    }
    
    void cleanup() {
        for (auto it = cache.begin(); it != cache.end(); ) {
            if (it->second.expired()) {
                std::cout << "Removing expired: " << it->first << "\n";
                it = cache.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

### Step 3: Usage
```cpp
int main() {
    ResourceManager rm;
    
    {
        auto tex1 = rm.load("player.png");
        auto tex2 = rm.load("player.png"); // Cache hit
        tex1->use();
    } // Textures destroyed
    
    rm.cleanup(); // Remove expired entries
    
    return 0;
}
```

## Challenges

### Challenge 1: Reference Counting
Add a method to report how many active references exist for each resource.

### Challenge 2: Preloading
Add a `preload` method that loads resources but doesn't return them (keeps them in cache).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Texture {
    std::string filename;
public:
    Texture(std::string file) : filename(file) {
        std::cout << "Loading: " << filename << "\n";
    }
    ~Texture() {
        std::cout << "Unloading: " << filename << "\n";
    }
    void use() const {
        std::cout << "Using: " << filename << "\n";
    }
};

class ResourceManager {
    std::map<std::string, std::weak_ptr<Texture>> cache;
    std::map<std::string, std::shared_ptr<Texture>> preloaded;
    
public:
    std::shared_ptr<Texture> load(const std::string& filename) {
        auto it = cache.find(filename);
        if (it != cache.end()) {
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << "\n";
                return sp;
            }
        }
        
        std::cout << "Loading: " << filename << "\n";
        auto texture = std::make_shared<Texture>(filename);
        cache[filename] = texture;
        return texture;
    }
    
    void preload(const std::string& filename) {
        if (preloaded.count(filename)) return;
        preloaded[filename] = load(filename);
    }
    
    void unloadPreloaded(const std::string& filename) {
        preloaded.erase(filename);
    }
    
    void reportUsage() {
        std::cout << "\n=== Resource Usage ===\n";
        for (const auto& [name, wp] : cache) {
            if (auto sp = wp.lock()) {
                std::cout << name << ": " << sp.use_count() << " refs\n";
            }
        }
    }
    
    void cleanup() {
        for (auto it = cache.begin(); it != cache.end(); ) {
            if (it->second.expired()) {
                it = cache.erase(it);
            } else {
                ++it;
            }
        }
    }
};

int main() {
    ResourceManager rm;
    
    rm.preload("background.png");
    rm.reportUsage();
    
    {
        auto tex1 = rm.load("player.png");
        auto tex2 = rm.load("player.png");
        auto tex3 = rm.load("background.png");
        
        rm.reportUsage();
    }
    
    rm.reportUsage();
    rm.cleanup();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented resource caching with `weak_ptr`
✅ Prevented duplicate loading
✅ Automatic cleanup of unused resources
✅ Implemented reference counting report (Challenge 1)
✅ Implemented preloading (Challenge 2)

## Key Learnings
- `weak_ptr` is perfect for caches
- Smart pointers enable automatic resource management
- Shared ownership allows flexible resource sharing

## Next Steps
Congratulations! You've completed Module 15.

Proceed to **Module 16: Move Semantics (Deep Dive)** to master performance optimization.
