# Day 251: Kernel Hardening Techniques
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Learning Objectives
1. **Understand** kernel hardening options
2. **Configure** kernel security features
3. **Implement** KASLR, stack protection, and other mitigations
4. **Use** kernel self-protection features
5. **Measure** security posture

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Kernel Hardening Features

**Key Hardening Mechanisms:**
- **KASLR** (Kernel Address Space Layout Randomization)
- **Stack Canaries** (Stack overflow protection)
- **DEP/NX** (Data Execution Prevention)
- **SMEP/SMAP** (Supervisor Mode Execution/Access Prevention)
- **Kernel Page Table Isolation** (KPTI/Meltdown mitigation)
- **Control Flow Integrity** (CFI)

### 🔹 Part 2: Kernel Configuration

```bash
# Essential hardening options
CONFIG_SECURITY=y
CONFIG_SECURITY_DMESG_RESTRICT=y
CONFIG_SECURITY_PERF_EVENTS_RESTRICT=y
CONFIG_SECURITY_YAMA=y
CONFIG_HARDENED_USERCOPY=y
CONFIG_FORTIFY_SOURCE=y
CONFIG_STACKPROTECTOR=y
CONFIG_STACKPROTECTOR_STRONG=y
CONFIG_RANDOMIZE_BASE=y
CONFIG_RANDOMIZE_MEMORY=y
CONFIG_PAGE_TABLE_ISOLATION=y
CONFIG_RETPOLINE=y
CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y
CONFIG_INIT_ON_FREE_DEFAULT_ON=y
```

---

## 💻 Implementation Examples

### Example 1: Runtime Kernel Hardening

```bash
# Restrict dmesg
echo 1 > /proc/sys/kernel/dmesg_restrict

# Restrict perf events
echo 2 > /proc/sys/kernel/perf_event_paranoid

# Restrict ptrace
echo 1 > /proc/sys/kernel/yama/ptrace_scope

# Disable kptr_restrict
echo 2 > /proc/sys/kernel/kptr_restrict

# Harden BPF JIT
echo 1 > /proc/sys/net/core/bpf_jit_harden

# Disable unprivileged BPF
echo 1 > /proc/sys/kernel/unprivileged_bpf_disabled

# Make permanent in /etc/sysctl.conf
cat >> /etc/sysctl.conf << EOF
kernel.dmesg_restrict = 1
kernel.perf_event_paranoid = 2
kernel.yama.ptrace_scope = 1
kernel.kptr_restrict = 2
net.core.bpf_jit_harden = 1
kernel.unprivileged_bpf_disabled = 1
EOF
```

### Example 2: Kernel Module Signing

```bash
# Generate signing key
openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv \
    -outform DER -out MOK.der -nodes -days 36500 \
    -subj "/CN=My Module Signing Key/"

# Sign module
/usr/src/linux/scripts/sign-file sha256 MOK.priv MOK.der mymodule.ko

# Verify signature
modinfo mymodule.ko | grep sig

# Enable module signature verification
CONFIG_MODULE_SIG=y
CONFIG_MODULE_SIG_FORCE=y
CONFIG_MODULE_SIG_ALL=y
CONFIG_MODULE_SIG_SHA256=y
```

---

## 🔬 Lab Exercises

### Lab 1: Security Audit

```bash
# Check kernel hardening status
checksec --kernel

# Check ASLR
cat /proc/sys/kernel/randomize_va_space
# 2 = full randomization

# Check stack protection
readelf -s /boot/vmlinuz-$(uname -r) | grep stack_chk

# Check SMEP/SMAP (x86)
grep -E 'smep|smap' /proc/cpuinfo

# Check KPTI
dmesg | grep -i "page table isolation"
```

### Lab 2: Exploit Mitigation Test

```c
// Test stack canary
#include <string.h>

void vulnerable(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // Buffer overflow
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable(argv[1]);
    }
    return 0;
}

// Compile with protections
// gcc -fstack-protector-strong -o test test.c

// Test overflow
// ./test $(python -c 'print "A"*100')
// Should abort with stack smashing detected
```

---

## 🧠 Assessment

**Q:** What is KASLR?
**A:** Kernel Address Space Layout Randomization - randomizes kernel memory addresses to prevent exploitation.

**Q:** What is the purpose of stack canaries?
**A:** Detect stack buffer overflows by placing a random value before return address.

**Q:** What is KPTI?
**A:** Kernel Page Table Isolation - separates kernel and user page tables to mitigate Meltdown.

---

## 🎓 Summary

Covered kernel hardening configuration, runtime security settings, module signing, and exploit mitigations.

---

## 🚀 Next Steps

Day 252: Week 37 Review and Project - Secure Application Framework

---
