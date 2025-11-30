# Day 97: Configuration Management
## Phase 1: Core Embedded Engineering Foundations | Week 14: File Systems and Storage

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a robust Configuration Management system for embedded devices.
2.  **Implement** a Key-Value (KV) store using LittleFS for flexible parameter storage.
3.  **Serialize** and **Deserialize** configuration structs (Binary vs JSON).
4.  **Implement** a "Factory Reset" mechanism to restore default settings.
5.  **Handle** configuration versioning (migrating data after firmware update).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Flash Storage (Internal or External).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   LittleFS (Day 96).
*   **Prior Knowledge:**
    *   Day 96 (LittleFS)
    *   Day 6 (Structs)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Where to store Config?
1.  **Hardcoded:** `const char *ssid = "MyWiFi";` (Bad. Requires recompile).
2.  **Raw Flash:** Struct at fixed address `0x080E0000`. (Fragile. One byte shift breaks everything).
3.  **EEPROM:** Dedicated chip. (Robust but extra cost).
4.  **File System:** `config.json` or `/kv/ssid`. (Flexible, scalable).

### 🔹 Part 2: The KV Store Pattern
Instead of one big struct, store each parameter as a file.
*   `/kv/wifi_ssid` -> "HomeNet"
*   `/kv/wifi_pass` -> "Secret123"
*   `/kv/volume` -> "80"
*   **Pros:** Atomic updates (changing volume doesn't corrupt wifi). Easy to add new keys.
*   **Cons:** Slower than raw struct.

### 🔹 Part 3: Versioning
Firmware v1.0 has `struct Config { int vol; }`.
Firmware v2.0 adds `int brightness`.
*   If v2.0 reads v1.0 data, `brightness` is garbage.
*   **Solution:** Magic Number + Version Header.
    *   `[MAGIC][VER][LEN][DATA...]`

---

## 💻 Implementation: KV Store (LittleFS)

> **Instruction:** Implement `KV_Set` and `KV_Get`.

### 👨‍💻 Code Implementation

#### Step 1: Helper Functions
```c
#include "lfs.h"

// Helper to build path: "wifi_ssid" -> "/kv/wifi_ssid"
void KV_GetPath(const char *key, char *path) {
    snprintf(path, 64, "/kv/%s", key);
}

// Ensure /kv exists
void KV_Init(void) {
    lfs_mkdir(&lfs, "/kv");
}
```

#### Step 2: Set (Write)
```c
int KV_Set(const char *key, void *data, size_t size) {
    char path[64];
    KV_GetPath(key, path);
    
    lfs_file_t f;
    int err = lfs_file_open(&lfs, &f, path, LFS_O_WRONLY | LFS_O_CREAT | LFS_O_TRUNC);
    if (err) return err;
    
    lfs_file_write(&lfs, &f, data, size);
    lfs_file_close(&lfs, &f);
    return 0;
}

int KV_SetString(const char *key, const char *str) {
    return KV_Set(key, (void*)str, strlen(str) + 1); // Include null terminator
}

int KV_SetInt(const char *key, int val) {
    return KV_Set(key, &val, sizeof(int));
}
```

#### Step 3: Get (Read)
```c
int KV_Get(const char *key, void *data, size_t size) {
    char path[64];
    KV_GetPath(key, path);
    
    lfs_file_t f;
    int err = lfs_file_open(&lfs, &f, path, LFS_O_RDONLY);
    if (err) return err; // Key not found
    
    lfs_ssize_t read_len = lfs_file_read(&lfs, &f, data, size);
    lfs_file_close(&lfs, &f);
    
    return (read_len == size) ? 0 : -1;
}

int KV_GetInt(const char *key, int default_val) {
    int val;
    if (KV_Get(key, &val, sizeof(int)) == 0) {
        return val;
    }
    return default_val;
}
```

---

## 💻 Implementation: Configuration Manager

> **Instruction:** Load/Save a global config struct using the KV store (or a single file).

### 👨‍💻 Code Implementation

#### Step 1: The Struct
```c
typedef struct {
    char wifi_ssid[32];
    char wifi_pass[32];
    uint8_t volume;
    uint8_t brightness;
} AppConfig_t;

AppConfig_t gConfig;

const AppConfig_t kDefaultConfig = {
    .wifi_ssid = "SetupAP",
    .wifi_pass = "password",
    .volume = 50,
    .brightness = 100
};
```

#### Step 2: Load/Save
```c
void Config_Save(void) {
    KV_Set("app_config", &gConfig, sizeof(AppConfig_t));
}

void Config_Load(void) {
    if (KV_Get("app_config", &gConfig, sizeof(AppConfig_t)) != 0) {
        // Not found or size mismatch
        printf("Config not found. Loading Defaults.\n");
        memcpy(&gConfig, &kDefaultConfig, sizeof(AppConfig_t));
        Config_Save(); // Create file
    }
}
```

#### Step 3: Factory Reset
```c
void Config_FactoryReset(void) {
    // Option 1: Delete file
    lfs_remove(&lfs, "/kv/app_config");
    
    // Option 2: Format FS (Nuclear option)
    // lfs_format(&lfs, &cfg);
    
    NVIC_SystemReset();
}
```

---

## 🔬 Lab Exercise: Lab 97.1 - Persistence Test

### 1. Lab Objectives
- Change Volume.
- Reset Board.
- Verify Volume persists.

### 2. Step-by-Step Guide

#### Phase A: CLI Integration
Add `set_vol <val>` to CLI.
```c
// In CLI Task
gConfig.volume = atoi(arg);
Config_Save();
printf("Saved.\n");
```

#### Phase B: Test
1.  Boot. `Config_Load` runs. Default Vol = 50.
2.  CLI: `set_vol 80`.
3.  Reset.
4.  Boot. `Config_Load` runs.
5.  CLI: `status`. Vol should be 80.

### 3. Verification
If `Config_Load` always loads defaults, check `KV_Get` return value. Maybe `sizeof(AppConfig_t)` changed?

---

## 🧪 Additional / Advanced Labs

### Lab 2: JSON Config
- **Goal:** Human readable config.
- **Task:**
    1.  Use `cJSON` library.
    2.  `Config_Save`: Serialize struct to JSON string -> Write to file.
    3.  `Config_Load`: Read file -> Parse JSON -> Fill struct.
    4.  **Benefit:** You can add fields to JSON without breaking old firmware readers (forward compatibility).

### Lab 3: Secure Storage
- **Goal:** Protect WiFi Pass.
- **Task:**
    1.  Encrypt data before `KV_Set` (AES-128).
    2.  Key stored in MCU Unique ID or OTP area.
    3.  **Note:** If someone dumps Flash, they see garbage.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Struct Padding
*   **Cause:** Compiler adds padding bytes between `char` and `int`.
*   **Result:** Binary mismatch between firmware versions compiled with different optimization levels.
*   **Solution:** Use `__attribute__((packed))` or serialize field-by-field.

#### 2. Partial Write
*   **Cause:** Power loss during `Config_Save`.
*   **Result:** Corrupt struct.
*   **Solution:** Use "Atomic Rename".
    1.  Write to `config.tmp`.
    2.  `lfs_rename("config.tmp", "config.dat")`.
    3.  Rename is atomic in LittleFS.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Shadow Config:** Keep a copy of the config in RAM (`gConfig`). Only write to Flash when changed (and maybe with a delay/debounce) to save Flash cycles. Don't write on every volume knob tick!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is `lfs_rename` atomic?
    *   **A:** It updates the directory entry pointer in one go. Either it points to the old file or the new file. Never "half" a file.
2.  **Q:** What is "Wear Leveling" in this context?
    *   **A:** If we update `config.dat` 1000 times, LittleFS ensures we don't write to the same physical Flash block 1000 times. It moves the file around.

### Challenge Task
> **Task:** Implement "Safe Mode". If the device crashes 3 times in a row (check Boot Count in FDR), load `kDefaultConfig` automatically on the 4th boot.

---

## 📚 Further Reading & References
- [cJSON Library](https://github.com/DaveGamble/cJSON)

---
