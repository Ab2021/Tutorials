# Lab 20.5: Adapter Pattern

## Objective
Implement the Adapter pattern to make incompatible interfaces work together.

## Instructions

### Step 1: Class Adapter
Create `adapter_pattern.cpp`.

```cpp
#include <iostream>
#include <string>

// Target interface
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& filename) = 0;
};

// Adaptee (incompatible interface)
class AdvancedMediaPlayer {
public:
    virtual ~AdvancedMediaPlayer() = default;
    virtual void playVlc(const std::string& filename) = 0;
    virtual void playMp4(const std::string& filename) = 0;
};

class VlcPlayer : public AdvancedMediaPlayer {
public:
    void playVlc(const std::string& filename) override {
        std::cout << "Playing VLC file: " << filename << "\n";
    }
    void playMp4(const std::string&) override {}
};

class Mp4Player : public AdvancedMediaPlayer {
public:
    void playVlc(const std::string&) override {}
    void playMp4(const std::string& filename) override {
        std::cout << "Playing MP4 file: " << filename << "\n";
    }
};

// Adapter
class MediaAdapter : public MediaPlayer {
    std::unique_ptr<AdvancedMediaPlayer> player;
    std::string audioType;
    
public:
    MediaAdapter(const std::string& type) : audioType(type) {
        if (type == "vlc") {
            player = std::make_unique<VlcPlayer>();
        } else if (type == "mp4") {
            player = std::make_unique<Mp4Player>();
        }
    }
    
    void play(const std::string& filename) override {
        if (audioType == "vlc") {
            player->playVlc(filename);
        } else if (audioType == "mp4") {
            player->playMp4(filename);
        }
    }
};
```

## Challenges

### Challenge 1: Object Adapter
Implement using composition instead of inheritance.

### Challenge 2: Two-Way Adapter
Create an adapter that works in both directions.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <string>

// Modern interfaces
class NewPrinter {
public:
    virtual ~NewPrinter() = default;
    virtual void printDocument(const std::string& doc) = 0;
};

// Legacy interface
class LegacyPrinter {
public:
    void print(const char* text) {
        std::cout << "Legacy print: " << text << "\n";
    }
};

// Challenge 1: Object Adapter
class PrinterAdapter : public NewPrinter {
    std::unique_ptr<LegacyPrinter> legacy;
    
public:
    PrinterAdapter() : legacy(std::make_unique<LegacyPrinter>()) {}
    
    void printDocument(const std::string& doc) override {
        legacy->print(doc.c_str());
    }
};

// Challenge 2: Two-way adapter
class TwoWayAdapter : public NewPrinter, public LegacyPrinter {
    std::string buffer;
    
public:
    // NewPrinter interface
    void printDocument(const std::string& doc) override {
        buffer = doc;
        print(buffer.c_str());
    }
    
    // Can also be used as LegacyPrinter
    using LegacyPrinter::print;
};

int main() {
    // Object adapter
    std::unique_ptr<NewPrinter> printer = std::make_unique<PrinterAdapter>();
    printer->printDocument("Hello from adapter!");
    
    // Two-way adapter
    TwoWayAdapter twoWay;
    twoWay.printDocument("New interface");
    twoWay.print("Legacy interface");
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented class adapter
✅ Created object adapter (Challenge 1)
✅ Built two-way adapter (Challenge 2)

## Key Learnings
- Adapter makes incompatible interfaces compatible
- Object adapter uses composition
- Class adapter uses inheritance
- Useful for integrating legacy code

## Next Steps
Proceed to **Lab 20.7: Observer Pattern**.
