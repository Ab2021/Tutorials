# Day 86: System Hardening & Security Audit
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎯 Learning Objectives
1.  **Identify** the Attack Surface of an embedded camera (Network, Physical, Supply Chain).
2.  **Harden** the Linux OS: Disable unused services, lock down SSH, configure Firewall.
3.  **Secure** Data at Rest: Implement LUKS (Linux Unified Key Setup) encryption.
4.  **Disable** Debug Interfaces: UART Console, JTAG, U-Boot shell.
5.  **Perform** a Security Audit using tools like `lynis` and `nmap`.
6.  **Implement** a "Factory Reset" mechanism that securely wipes keys.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Linux Camera System (Jetson/Pi).
*   **Software:** `iptables`, `cryptsetup`, `lynis`.
*   **Knowledge:** Linux Permissions (chmod/chown), SSH Keys.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Attack Surface
*   **Network:** Open ports (SSH, Telnet, HTTP). Default passwords (`root:root`).
*   **Physical:** UART console (gives root shell without password). SD Card removal (reading data on PC).
*   **Application:** Command Injection in the Web Interface.

### 🔹 Part 2: Hardening Strategy (Defense in Depth)
1.  **Least Privilege:** Run the camera app as `camera_user`, not `root`.
2.  **Minimize Footprint:** Remove `gcc`, `gdb`, `python` (if possible) from the production image.
3.  **Read-Only RootFS:** Prevents malware from persisting after reboot.

### 🔹 Part 3: Secure Storage (LUKS)
*   **Problem:** If someone steals the camera, they can read the SD card.
*   **Solution:** Full Disk Encryption (FDE).
*   **Key Management:** The decryption key must be stored in the TPM (Trusted Platform Module) or derived from a hardware unique ID.

---

## 💻 Implementation Examples

### Example 1: Firewall Configuration (`iptables`)

Allow only RTSP (554) and HTTPS (443). Drop everything else.

```bash
# 1. Set Default Policy: DROP
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 2. Allow Loopback
iptables -A INPUT -i lo -j ACCEPT

# 3. Allow Established Connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 4. Allow Specific Ports
iptables -A INPUT -p tcp --dport 443 -j ACCEPT  # HTTPS
iptables -A INPUT -p tcp --dport 554 -j ACCEPT  # RTSP
iptables -A INPUT -p tcp --dport 8883 -j ACCEPT # MQTT (TLS)

# 5. Save
iptables-save > /etc/iptables/rules.v4
```

### Example 2: Disabling UART Console

Prevent access via the serial cable.

```bash
# In /boot/cmdline.txt or extlinux.conf
# Remove: console=ttyS0,115200

# In /etc/inittab or systemd
systemctl disable serial-getty@ttyS0.service
```

### Example 3: LUKS Encryption

Encrypting the `/data` partition.

```bash
# 1. Format
cryptsetup luksFormat /dev/mmcblk0p3

# 2. Open
cryptsetup luksOpen /dev/mmcblk0p3 secure_data

# 3. Create Filesystem
mkfs.ext4 /dev/mapper/secure_data

# 4. Mount
mount /dev/mapper/secure_data /mnt/data
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Nmap" Scan

**Objective:** Hack yourself.

**Steps:**
1.  Run `nmap -A <camera_ip>` from your PC.
2.  **Result:** It should show ONLY ports 443 and 554 open.
3.  **Fail:** If it shows Port 22 (SSH) open with "OpenSSH 7.6 (Ubuntu)", you are vulnerable.
4.  **Fix:** Change SSH port, disable Password Auth, or firewall it.

### Lab 2: Privilege Escalation Check

**Objective:** Can the app become root?

**Steps:**
1.  Log in as `camera_user`.
2.  Try `sudo ls /root`. Should fail.
3.  Check SUID binaries: `find / -perm -4000`.
4.  **Risk:** If `vim` or `find` has SUID bit set, you can get a root shell.

### Lab 3: Physical Access Test

**Objective:** UART Console.

**Steps:**
1.  Connect USB-TTL cable to the UART pins.
2.  Boot the device.
3.  **Goal:** You should see boot logs (maybe), but NO login prompt. Or the prompt should require a strong password.
4.  **U-Boot:** Press keys during boot. If you get a `=>` prompt, you can change `init=/bin/sh` and bypass all security. **Fix:** Set `bootdelay=-2` in U-Boot.

---

## 🐛 Debugging Security Issues

### Debug 1: "Locked Out"

**Symptom:** You disabled SSH and Console, now you can't fix a bug.

**Cause:**
*   Over-zealous hardening.
*   **Fix:** Implement a "Backdoor" (e.g., a specific USB drive with a signed key file that re-enables SSH temporarily).

### Debug 2: Boot Failure with LUKS

**Symptom:** System hangs asking for passphrase.

**Cause:**
*   No keyboard attached to enter password.
*   **Fix:** Use a keyfile stored in the initramfs (if RootFS is encrypted) or use a TPM to auto-unlock.

---

## ⚡ Performance Optimization

### Optimization 1: Hardware Crypto for LUKS

*   Ensure the kernel uses the hardware AES engine (`aes-ce` on ARM).
*   Check `/proc/crypto`.
*   Software encryption slows down disk I/O significantly.

### Optimization 2: Minimal Kernel

*   Compile a kernel with `CONFIG_MODULE_SIG=y` (Module Signing).
*   Prevents loading malicious kernel modules (Rootkits).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Defense in Depth"?** (Multiple layers of security. If Firewall fails, Authentication saves you. If Auth fails, Permissions save you).
2.  **Why is "Read-Only RootFS" good for reliability AND security?** (Prevents corruption on power loss, and prevents malware persistence).
3.  **What is a "Side Channel Attack"?** (Measuring power consumption or timing to guess the encryption key).
4.  **Difference between `su` and `sudo`?**

### Practical Challenges

1.  **Write a Hardening Script:** A bash script that automatically applies all the fixes (disables services, sets permissions, configures firewall).
2.  **Audit with Lynis:** Install `lynis` and run a system audit. Aim for a score > 80.

---

## 📚 Further Reading & Resources

### Standards
*   **CIS Benchmarks for Linux.**
*   **OWASP IoT Top 10.**

---

## 🎓 Summary

Today we covered:
- ✅ **Attack Surface:** Know your enemy.
- ✅ **Hardening:** Locking the doors.
- ✅ **Encryption:** Protecting the loot.
- ✅ **Physical Security:** Glue the UART.
- ✅ **Audit:** Checking your work.

**Next:** Day 87 - Documentation & User Manuals.

---

**Day 86 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
