# Day 247: SELinux Deep Dive
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Learning Objectives
1. **Understand** SELinux architecture and type enforcement
2. **Write** SELinux policies for custom applications
3. **Debug** SELinux denials using audit logs
4. **Configure** SELinux contexts and booleans
5. **Optimize** SELinux performance

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SELinux Architecture

**Security-Enhanced Linux (SELinux)** implements mandatory access control using security labels and type enforcement.

**Core Concepts:**
- **Security Context:** `user:role:type:level`
- **Type Enforcement (TE):** Primary access control mechanism
- **Subjects:** Processes (domains)
- **Objects:** Files, sockets, etc. (types)
- **Policy:** Rules defining allowed interactions

**SELinux Decision Flow:**
```
Process → Access Object
    ↓
DAC Check (traditional permissions)
    ↓ (if allowed)
SELinux Check (security context + policy)
    ↓
Allow or Deny (logged to audit)
```

### 🔹 Part 2: Security Contexts

```bash
# View file context
ls -Z /etc/passwd
# Output: system_u:object_r:passwd_file_t:s0

# View process context
ps -eZ | grep sshd
# Output: system_u:system_r:sshd_t:s0-s0:c0.c1023

# Context format:
# user:role:type:level
# - user: SELinux user (system_u, user_u, etc.)
# - role: Role (system_r, user_r, etc.)
# - type: Type/Domain (most important for TE)
# - level: MLS/MCS level (optional)
```

### 🔹 Part 3: Type Enforcement Policy

```
# Basic TE rule syntax
allow <source_type> <target_type>:<class> { <permissions> };

# Example: Allow httpd to read web content
allow httpd_t httpd_sys_content_t:file { read getattr open };

# Example: Allow httpd to bind to port 80
allow httpd_t http_port_t:tcp_socket { bind };
```

---

## 💻 Implementation Examples

### Example 1: Creating Custom SELinux Policy

```bash
# 1. Create policy module directory
mkdir myapp_policy
cd myapp_policy

# 2. Create type enforcement file (myapp.te)
cat > myapp.te << 'EOF'
policy_module(myapp, 1.0.0)

# Declare domain for myapp
type myapp_t;
type myapp_exec_t;
domain_type(myapp_t)
domain_entry_file(myapp_t, myapp_exec_t)

# Allow myapp to execute
allow myapp_t myapp_exec_t:file { execute execute_no_trans };

# Allow myapp to read config files
type myapp_conf_t;
files_type(myapp_conf_t)
allow myapp_t myapp_conf_t:file { read open getattr };

# Allow myapp to write log files
type myapp_log_t;
logging_log_file(myapp_log_t)
allow myapp_t myapp_log_t:file { create write append open };

# Allow myapp to use network
corenet_tcp_bind_generic_node(myapp_t)
corenet_tcp_bind_http_port(myapp_t)
allow myapp_t self:tcp_socket { create bind listen accept };

# Allow transitions from init
init_daemon_domain(myapp_t, myapp_exec_t)
EOF

# 3. Create file contexts file (myapp.fc)
cat > myapp.fc << 'EOF'
/usr/local/bin/myapp  -- gen_context(system_u:object_r:myapp_exec_t,s0)
/etc/myapp(/.*)?         gen_context(system_u:object_r:myapp_conf_t,s0)
/var/log/myapp(/.*)?     gen_context(system_u:object_r:myapp_log_t,s0)
EOF

# 4. Compile and install policy
make -f /usr/share/selinux/devel/Makefile myapp.pp
sudo semodule -i myapp.pp

# 5. Restore file contexts
sudo restorecon -Rv /usr/local/bin/myapp /etc/myapp /var/log/myapp

# 6. Verify
semodule -l | grep myapp
ls -Z /usr/local/bin/myapp
```

### Example 2: Debugging SELinux Denials

```bash
# 1. Generate denial
./myapp  # Triggers SELinux denial

# 2. View audit log
sudo ausearch -m AVC -ts recent
# Output:
# type=AVC msg=audit(1234567890.123:456): avc:  denied  { read } for  
#   pid=1234 comm="myapp" name="data.txt" dev="sda1" ino=789012 
#   scontext=system_u:system_r:myapp_t:s0 
#   tcontext=system_u:object_r:user_home_t:s0 
#   tclass=file permissive=0

# 3. Use audit2allow to generate policy
sudo ausearch -m AVC -ts recent | audit2allow

# Output:
# #============= myapp_t ==============
# allow myapp_t user_home_t:file read;

# 4. Create policy module from denials
sudo ausearch -m AVC -ts recent | audit2allow -M myapp_fix
sudo semodule -i myapp_fix.pp

# 5. Alternative: Use audit2why for explanation
sudo ausearch -m AVC -ts recent | audit2why
```

### Example 3: SELinux Booleans

```bash
# List all booleans
getsebool -a

# Check specific boolean
getsebool httpd_can_network_connect
# Output: httpd_can_network_connect --> off

# Enable boolean (temporary)
sudo setsebool httpd_can_network_connect on

# Enable boolean (permanent)
sudo setsebool -P httpd_can_network_connect on

# View boolean description
semanage boolean -l | grep httpd_can_network_connect
```

### Example 4: Custom Application with SELinux

```c
// myapp.c - Application with SELinux awareness
#include <stdio.h>
#include <selinux/selinux.h>
#include <selinux/context.h>

int main(void) {
    char *con;
    
    // Check if SELinux is enabled
    if (is_selinux_enabled()) {
        printf("SELinux is enabled\n");
        
        // Get current context
        if (getcon(&con) == 0) {
            printf("Current context: %s\n", con);
            freecon(con);
        }
        
        // Get file context
        if (getfilecon("/etc/passwd", &con) == 0) {
            printf("/etc/passwd context: %s\n", con);
            freecon(con);
        }
        
        // Check access
        if (selinux_check_access("myapp_t", "passwd_file_t", 
                                "file", "read", NULL) == 0) {
            printf("Access would be allowed\n");
        } else {
            printf("Access would be denied\n");
        }
    } else {
        printf("SELinux is disabled\n");
    }
    
    return 0;
}

// Compile: gcc -o myapp myapp.c -lselinux
```

---

## 🔬 Lab Exercises

### Lab 1: Write Policy for Custom Daemon

```bash
# Create simple daemon
cat > /usr/local/bin/mydaemon << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): Running" >> /var/log/mydaemon.log
    sleep 60
done
EOF
chmod +x /usr/local/bin/mydaemon

# Create SELinux policy
# (Follow Example 1 structure)

# Test daemon
sudo /usr/local/bin/mydaemon &

# Check for denials
sudo ausearch -m AVC -ts recent
```

### Lab 2: Confine Existing Application

```bash
# Example: Confine custom Python script

# 1. Identify what the script does
strace -f python3 myscript.py 2>&1 | grep -E "open|connect|bind"

# 2. Run in permissive mode for the domain
sudo semanage permissive -a myapp_t

# 3. Exercise all functionality
python3 myscript.py

# 4. Generate policy from audit log
sudo ausearch -m AVC -ts recent | audit2allow -M myscript_policy

# 5. Review and refine policy
cat myscript_policy.te

# 6. Install and test in enforcing mode
sudo semodule -i myscript_policy.pp
sudo semanage permissive -d myapp_t
```

---

## 🧠 Assessment

**Q:** What is the difference between targeted and strict SELinux policy?
**A:** Targeted confines specific daemons, leaving most processes unconfined. Strict confines everything, requiring comprehensive policy.

**Q:** How does SELinux handle file creation?
**A:** New files inherit the type of the parent directory, or use type_transition rules to assign specific types.

**Q:** What is the purpose of SELinux booleans?
**A:** Allow runtime policy changes without recompiling, enabling/disabling specific features.

---

## 🎓 Summary

Covered SELinux architecture, type enforcement, policy writing, debugging denials, and practical application confinement.

---

## 🚀 Next Steps

Day 248: AppArmor Profiles and Path-Based MAC

---
