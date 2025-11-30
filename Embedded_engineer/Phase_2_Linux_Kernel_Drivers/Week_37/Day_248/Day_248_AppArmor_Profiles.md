# Day 248: AppArmor Profiles and Path-Based MAC
## Phase 2: Linux Kernel & Device Drivers | Week 37: Kernel Security

---

## 🎯 Learning Objectives
1. **Understand** AppArmor's path-based access control
2. **Write** AppArmor profiles for applications
3. **Use** AppArmor learning mode
4. **Debug** AppArmor denials
5. **Compare** AppArmor vs SELinux approaches

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: AppArmor vs SELinux

| Feature | AppArmor | SELinux |
|---------|----------|---------|
| **Model** | Path-based | Label-based |
| **Complexity** | Lower | Higher |
| **Policy** | Per-application profiles | System-wide type enforcement |
| **Learning** | Built-in learning mode | Requires audit2allow |
| **Flexibility** | Easier for simple cases | More powerful for complex policies |

### 🔹 Part 2: AppArmor Profile Structure

```
# Profile header
/usr/bin/myapp {
  # Include abstractions
  #include <abstractions/base>
  
  # Capabilities
  capability net_bind_service,
  
  # File access rules
  /etc/myapp/** r,              # Read config
  /var/log/myapp/** w,          # Write logs
  /usr/bin/myapp mr,            # Execute self
  
  # Network access
  network inet stream,
  
  # IPC
  signal send set=(term) peer=/usr/bin/helper,
}
```

---

## 💻 Implementation Examples

### Example 1: Basic AppArmor Profile

```bash
# Create profile for custom application
sudo aa-genprof /usr/local/bin/myapp

# This starts interactive mode:
# 1. Run application in another terminal
# 2. Exercise all functionality
# 3. Review and approve/deny each access
# 4. Save profile

# Manual profile creation
sudo nano /etc/apparmor.d/usr.local.bin.myapp

# Profile content:
cat > /etc/apparmor.d/usr.local.bin.myapp << 'EOF'
#include <tunables/global>

/usr/local/bin/myapp {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  
  # Capabilities
  capability net_bind_service,
  capability setuid,
  capability setgid,
  
  # Binary
  /usr/local/bin/myapp mr,
  
  # Configuration
  /etc/myapp/** r,
  owner /etc/myapp/myapp.conf rw,
  
  # Logs
  /var/log/myapp/ rw,
  /var/log/myapp/** rw,
  
  # Runtime data
  /var/run/myapp.pid rw,
  
  # Network
  network inet stream,
  network inet6 stream,
  
  # Shared libraries
  /lib/x86_64-linux-gnu/** mr,
  /usr/lib/x86_64-linux-gnu/** mr,
}
EOF

# Load profile
sudo apparmor_parser -r /etc/apparmor.d/usr.local.bin.myapp

# Verify
sudo aa-status | grep myapp
```

### Example 2: Using Learning Mode

```bash
# Put profile in complain mode (learning)
sudo aa-complain /usr/local/bin/myapp

# Run application and exercise all features
/usr/local/bin/myapp --test-all-features

# Review logged accesses
sudo aa-logprof

# This shows each denied access and asks:
# (A)llow / (D)eny / (I)gnore / (G)lob / Glob with (E)xtension / (N)ew / Abo(r)t

# After approving, profile is updated automatically

# Switch to enforce mode
sudo aa-enforce /usr/local/bin/myapp
```

### Example 3: Advanced Profile Features

```bash
# Profile with variables and conditionals
cat > /etc/apparmor.d/usr.bin.webapp << 'EOF'
#include <tunables/global>

@{APP_HOME}=/var/www/webapp
@{APP_USER}=www-data

/usr/bin/webapp {
  #include <abstractions/base>
  #include <abstractions/web-data>
  
  # Capabilities
  capability dac_override,
  capability chown,
  
  # Binary and libraries
  /usr/bin/webapp mr,
  /usr/lib/webapp/** mr,
  
  # Application files
  @{APP_HOME}/** r,
  owner @{APP_HOME}/uploads/** rw,
  owner @{APP_HOME}/cache/** rw,
  
  # Database socket
  /var/run/postgresql/.s.PGSQL.* rw,
  
  # Network (only specific ports)
  network inet stream,
  network inet6 stream,
  
  # Child processes
  /usr/bin/convert Px -> webapp//convert,
  
  # Deny specific paths
  deny /etc/shadow r,
  deny /root/** rwx,
  
  # Subprofile for image converter
  profile convert {
    #include <abstractions/base>
    /usr/bin/convert mr,
    /tmp/** rw,
    @{APP_HOME}/uploads/** r,
  }
}
EOF
```

---

## 🔬 Lab Exercises

### Lab 1: Confine Web Server

```bash
# Install Apache
sudo apt-get install apache2

# Generate profile
sudo aa-genprof /usr/sbin/apache2

# In another terminal, test Apache
curl http://localhost/
curl http://localhost/test.php

# Review and save profile
# Switch to enforce mode
sudo aa-enforce /usr/sbin/apache2

# Test that it still works
curl http://localhost/
```

### Lab 2: Debug Profile Issues

```bash
# Application fails with AppArmor
./myapp
# Error: Permission denied

# Check AppArmor logs
sudo dmesg | grep apparmor
sudo journalctl -xe | grep apparmor

# Or use aa-notify
sudo aa-notify -s 1 -v

# Fix by adding missing permissions
sudo aa-logprof
```

---

## 🧠 Assessment

**Q:** When should you use AppArmor vs SELinux?
**A:** AppArmor for simpler, application-specific confinement. SELinux for system-wide, fine-grained mandatory access control.

**Q:** What is the advantage of path-based MAC?
**A:** Easier to understand and write policies. No need to label files.

**Q:** What is complain mode?
**A:** Learning mode where violations are logged but not enforced, useful for policy development.

---

## 🎓 Summary

Covered AppArmor profiles, learning mode, path-based access control, and practical application confinement.

---

## 🚀 Next Steps

Day 249: Seccomp and System Call Filtering

---
