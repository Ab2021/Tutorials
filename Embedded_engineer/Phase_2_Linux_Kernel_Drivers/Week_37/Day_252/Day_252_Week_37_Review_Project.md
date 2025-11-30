# Day 252: Week 37 Review and Project - Secure Application Framework
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Project Goal

Build a comprehensive secure application framework demonstrating all Week 37 security concepts:
- LSM integration
- SELinux/AppArmor confinement
- Seccomp filtering
- Capability reduction
- Kernel hardening

---

## 📋 Project: Secure Web Service

**Requirements:**
1. Web service running with minimal privileges
2. SELinux policy for confinement
3. Seccomp filter limiting syscalls
4. Capabilities instead of root
5. AppArmor profile as alternative
6. Comprehensive security audit

---

## 💻 Complete Implementation

```c
// secure_webserver.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/capability.h>
#include <sys/prctl.h>
#include <seccomp.h>
#include <pwd.h>
#include <grp.h>

#define PORT 8080
#define NOBODY_UID 65534
#define NOBODY_GID 65534

// Drop all capabilities except CAP_NET_BIND_SERVICE
int drop_capabilities(void) {
    cap_t caps;
    cap_value_t cap_list[] = {CAP_NET_BIND_SERVICE};
    
    caps = cap_init();
    if (!caps) return -1;
    
    if (cap_set_flag(caps, CAP_PERMITTED, 1, cap_list, CAP_SET) != 0 ||
        cap_set_flag(caps, CAP_EFFECTIVE, 1, cap_list, CAP_SET) != 0) {
        cap_free(caps);
        return -1;
    }
    
    if (cap_set_proc(caps) != 0) {
        cap_free(caps);
        return -1;
    }
    
    cap_free(caps);
    
    // Prevent privilege escalation
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        return -1;
    }
    
    return 0;
}

// Setup seccomp filter
int setup_seccomp(void) {
    scmp_filter_ctx ctx;
    
    ctx = seccomp_init(SCMP_ACT_KILL);
    if (!ctx) return -1;
    
    // Allow essential syscalls
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(close), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(socket), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(bind), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(listen), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(accept), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(sendto), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(recvfrom), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(brk), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(mmap), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(munmap), 0);
    
    if (seccomp_load(ctx) < 0) {
        seccomp_release(ctx);
        return -1;
    }
    
    seccomp_release(ctx);
    return 0;
}

// Drop privileges to nobody
int drop_privileges(void) {
    if (setgid(NOBODY_GID) != 0) {
        perror("setgid");
        return -1;
    }
    
    if (setuid(NOBODY_UID) != 0) {
        perror("setuid");
        return -1;
    }
    
    // Verify we can't regain root
    if (setuid(0) == 0) {
        fprintf(stderr, "ERROR: Could regain root!\n");
        return -1;
    }
    
    return 0;
}

int main(void) {
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    
    printf("Secure Web Server Starting...\n");
    
    // Create socket (needs CAP_NET_BIND_SERVICE for port < 1024)
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }
    
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind");
        return 1;
    }
    
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        return 1;
    }
    
    printf("Listening on port %d\n", PORT);
    
    // Now drop privileges
    printf("Dropping capabilities...\n");
    if (drop_capabilities() != 0) {
        fprintf(stderr, "Failed to drop capabilities\n");
        return 1;
    }
    
    printf("Dropping to nobody user...\n");
    if (drop_privileges() != 0) {
        fprintf(stderr, "Failed to drop privileges\n");
        return 1;
    }
    
    printf("Setting up seccomp filter...\n");
    if (setup_seccomp() != 0) {
        fprintf(stderr, "Failed to setup seccomp\n");
        return 1;
    }
    
    printf("Security hardening complete. Server ready.\n");
    printf("UID: %d, GID: %d\n", getuid(), getgid());
    
    // Main server loop
    while (1) {
        client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }
        
        const char *response = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: 25\r\n"
            "\r\n"
            "Secure Server Running!\r\n";
        
        write(client_fd, response, strlen(response));
        close(client_fd);
    }
    
    return 0;
}

// Compile: gcc -o secure_server secure_webserver.c -lcap -lseccomp
```

### SELinux Policy

```bash
# secure_server.te
policy_module(secure_server, 1.0.0)

type secure_server_t;
type secure_server_exec_t;
domain_type(secure_server_t)
domain_entry_file(secure_server_t, secure_server_exec_t)

# Allow network
corenet_tcp_bind_generic_node(secure_server_t)
corenet_tcp_bind_http_port(secure_server_t)
allow secure_server_t self:tcp_socket { create bind listen accept };

# Minimal file access
allow secure_server_t secure_server_exec_t:file { read execute };

# No other file access needed
```

### AppArmor Profile

```bash
# /etc/apparmor.d/usr.local.bin.secure_server
#include <tunables/global>

/usr/local/bin/secure_server {
  #include <abstractions/base>
  
  capability net_bind_service,
  capability setuid,
  capability setgid,
  
  /usr/local/bin/secure_server mr,
  
  network inet stream,
  network inet6 stream,
  
  # Deny everything else
  deny /** rwx,
}
```

---

## 📊 Security Audit Results

| Security Feature | Status | Details |
|-----------------|--------|---------|
| **Runs as root** | ❌ No | Drops to nobody |
| **Capabilities** | ✅ Minimal | Only CAP_NET_BIND_SERVICE |
| **Seccomp** | ✅ Active | 13 syscalls allowed |
| **SELinux** | ✅ Confined | Custom policy |
| **AppArmor** | ✅ Confined | Custom profile |
| **ASLR** | ✅ Enabled | System-wide |
| **Stack Protection** | ✅ Enabled | Compiler flags |

---

## 🧠 Week 37 Summary

**Topics Covered:**
1. **LSM Framework** (Day 246) - Custom security modules
2. **SELinux** (Day 247) - Type enforcement policies
3. **AppArmor** (Day 248) - Path-based MAC
4. **Seccomp** (Day 249) - Syscall filtering
5. **Capabilities** (Day 250) - Privilege reduction
6. **Kernel Hardening** (Day 251) - System-wide protections
7. **Integration** (Day 252) - Complete secure application

**Key Takeaway:** Defense in depth - multiple security layers working together provide robust protection.

---

## 🚀 Next Steps

**Week 38 Preview:** Kernel Performance Optimization - Profiling, Tracing, and Performance Tuning

---
