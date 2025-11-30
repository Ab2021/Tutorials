# Day 250: Linux Capabilities and Privilege Reduction
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Learning Objectives
1. **Understand** Linux capabilities system
2. **Use** capabilities to reduce privileges
3. **Implement** capability-aware applications
4. **Debug** capability-related issues
5. **Apply** principle of least privilege

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Capabilities Overview

**Linux Capabilities** divide root privileges into distinct units that can be independently enabled/disabled.

**Key Capabilities:**
- `CAP_NET_BIND_SERVICE` - Bind to ports < 1024
- `CAP_NET_RAW` - Use RAW/PACKET sockets
- `CAP_SYS_ADMIN` - Various admin operations
- `CAP_SYS_TIME` - Set system clock
- `CAP_CHOWN` - Change file ownership
- `CAP_KILL` - Send signals to any process
- `CAP_SETUID/SETGID` - Change UID/GID

### 🔹 Part 2: Capability Sets

Each process has 5 capability sets:
1. **Permitted:** Capabilities process may use
2. **Effective:** Currently active capabilities
3. **Inheritable:** Capabilities preserved across execve
4. **Bounding:** Limit on capabilities
5. **Ambient:** Capabilities preserved for non-root

---

## 💻 Implementation Examples

### Example 1: Drop Capabilities

```c
#include <sys/capability.h>
#include <sys/prctl.h>
#include <stdio.h>
#include <unistd.h>

void drop_capabilities(void) {
    cap_t caps;
    
    // Get current capabilities
    caps = cap_get_proc();
    
    // Clear all capabilities
    cap_clear(caps);
    
    // Set (empty) capabilities
    if (cap_set_proc(caps) != 0) {
        perror("cap_set_proc");
    }
    
    cap_free(caps);
    
    // Prevent regaining capabilities
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl");
    }
}

int main(void) {
    printf("UID: %d, EUID: %d\n", getuid(), geteuid());
    
    // Drop all capabilities
    drop_capabilities();
    
    // Try to bind to port 80 (should fail)
    // ...
    
    return 0;
}

// Compile: gcc -o drop_caps drop_caps.c -lcap
```

### Example 2: Selective Capabilities

```c
#include <sys/capability.h>
#include <stdio.h>

int set_capabilities(cap_value_t *caps, int ncaps) {
    cap_t new_caps;
    
    new_caps = cap_init();
    
    // Set capabilities in permitted and effective sets
    if (cap_set_flag(new_caps, CAP_PERMITTED, ncaps, caps, CAP_SET) != 0 ||
        cap_set_flag(new_caps, CAP_EFFECTIVE, ncaps, caps, CAP_SET) != 0) {
        cap_free(new_caps);
        return -1;
    }
    
    if (cap_set_proc(new_caps) != 0) {
        cap_free(new_caps);
        return -1;
    }
    
    cap_free(new_caps);
    return 0;
}

int main(void) {
    cap_value_t caps[] = {CAP_NET_BIND_SERVICE, CAP_NET_RAW};
    
    // Keep only network capabilities
    if (set_capabilities(caps, 2) != 0) {
        perror("set_capabilities");
        return 1;
    }
    
    printf("Capabilities set successfully\n");
    
    // Now can bind to port 80 but cannot chown files
    
    return 0;
}
```

---

## 🧠 Assessment

**Q:** Why use capabilities instead of setuid root?
**A:** Capabilities provide fine-grained control, reducing attack surface by granting only needed privileges.

**Q:** Can capabilities be used to escape containers?
**A:** Some capabilities (like CAP_SYS_ADMIN) are powerful enough to escape if not properly restricted.

---

## 🎓 Summary

Covered Linux capabilities, privilege reduction, and implementing least-privilege applications.

---

## 🚀 Next Steps

Day 251: Kernel Hardening Techniques

---
