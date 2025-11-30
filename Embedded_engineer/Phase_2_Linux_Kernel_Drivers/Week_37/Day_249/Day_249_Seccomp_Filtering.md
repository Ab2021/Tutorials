# Day 249: Seccomp and System Call Filtering
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Learning Objectives
1. **Understand** seccomp (secure computing mode)
2. **Implement** seccomp-BPF filters
3. **Use** libseccomp for policy creation
4. **Debug** seccomp violations
5. **Apply** seccomp to containerized applications

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Seccomp Modes

**Seccomp (Secure Computing Mode)** restricts system calls a process can make.

**Modes:**
1. **SECCOMP_MODE_STRICT:** Only read, write, exit, sigreturn allowed
2. **SECCOMP_MODE_FILTER:** BPF-based filtering (most useful)

### 🔹 Part 2: Seccomp-BPF

Uses Berkeley Packet Filter (BPF) to filter syscalls based on:
- Syscall number
- Arguments
- Architecture

**Filter Actions:**
- `SECCOMP_RET_KILL_PROCESS` - Kill entire process
- `SECCOMP_RET_KILL_THREAD` - Kill thread
- `SECCOMP_RET_TRAP` - Send SIGSYS
- `SECCOMP_RET_ERRNO` - Return error
- `SECCOMP_RET_TRACE` - Notify tracer
- `SECCOMP_RET_ALLOW` - Allow syscall

---

## 💻 Implementation Examples

### Example 1: Basic Seccomp Filter

```c
#include <stdio.h>
#include <seccomp.h>
#include <unistd.h>

int main(void) {
    scmp_filter_ctx ctx;
    
    // Create filter context (default deny)
    ctx = seccomp_init(SCMP_ACT_KILL);
    if (!ctx) {
        perror("seccomp_init");
        return 1;
    }
    
    // Allow specific syscalls
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit), 0);
    
    // Load filter
    if (seccomp_load(ctx) < 0) {
        perror("seccomp_load");
        seccomp_release(ctx);
        return 1;
    }
    
    printf("Seccomp filter loaded\n");
    
    // This works (write allowed)
    write(1, "Hello\n", 6);
    
    // This would kill process (open not allowed)
    // open("/tmp/test", O_RDONLY);
    
    seccomp_release(ctx);
    return 0;
}

// Compile: gcc -o seccomp_basic seccomp_basic.c -lseccomp
```

### Example 2: Argument-Based Filtering

```c
#include <seccomp.h>
#include <fcntl.h>
#include <stdio.h>

int main(void) {
    scmp_filter_ctx ctx;
    
    ctx = seccomp_init(SCMP_ACT_ALLOW);  // Default allow
    
    // Block open() of /etc/passwd
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EACCES), SCMP_SYS(open), 1,
                    SCMP_A0(SCMP_CMP_EQ, (scmp_datum_t)"/etc/passwd"));
    
    // Block socket creation (network access)
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(socket), 0);
    
    // Block execve (prevent spawning processes)
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EACCES), SCMP_SYS(execve), 0);
    
    seccomp_load(ctx);
    
    // Test
    int fd = open("/etc/passwd", O_RDONLY);
    if (fd < 0) {
        perror("open /etc/passwd");  // Should fail
    }
    
    fd = open("/etc/hosts", O_RDONLY);
    if (fd >= 0) {
        printf("Opened /etc/hosts successfully\n");
        close(fd);
    }
    
    seccomp_release(ctx);
    return 0;
}
```

### Example 3: Seccomp for Sandboxing

```c
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>

// Sandbox a function
void sandbox_function(void (*func)(void)) {
    scmp_filter_ctx ctx;
    
    ctx = seccomp_init(SCMP_ACT_KILL);
    
    // Allow minimal syscalls
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(brk), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(mmap), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(munmap), 0);
    
    seccomp_load(ctx);
    
    // Run sandboxed function
    func();
    
    seccomp_release(ctx);
}

void untrusted_code(void) {
    printf("Running in sandbox\n");
    
    // This would kill process
    // system("ls");
}

int main(void) {
    sandbox_function(untrusted_code);
    return 0;
}
```

---

## 🔬 Lab Exercises

### Lab 1: Container Seccomp Profile

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "open", "close", "stat", "fstat"],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["socket", "connect", "bind"],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["execve"],
      "action": "SCMP_ACT_ERRNO",
      "args": []
    }
  ]
}
```

```bash
# Use with Docker
docker run --security-opt seccomp=profile.json myimage
```

---

## 🧠 Assessment

**Q:** What is the difference between seccomp strict mode and filter mode?
**A:** Strict allows only 4 syscalls (read/write/exit/sigreturn). Filter uses BPF for custom policies.

**Q:** Can seccomp be bypassed?
**A:** No, once loaded it cannot be removed. Child processes inherit filters.

---

## 🎓 Summary

Covered seccomp modes, BPF filtering, libseccomp usage, and practical sandboxing techniques.

---

## 🚀 Next Steps

Day 250: Linux Capabilities and Privilege Reduction

---
