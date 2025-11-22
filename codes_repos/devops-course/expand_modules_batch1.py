#!/usr/bin/env python3
"""
Batch Module Expansion - Phase 1 Modules 2-10
Creates comprehensive theoretical content for all beginner modules
"""

from pathlib import Path

BASE_PATH = r"H:\My Drive\Codes & Repos\codes_repos\devops-course"

# Module 2: Linux Fundamentals - Comprehensive Content
MODULE_02_CONTENT = """# Module 2: Linux Fundamentals

## 🎯 Learning Objectives

By the end of this module, you will:
- Master the Linux command line interface
- Understand the Linux file system hierarchy
- Manage files, directories, and permissions
- Control processes and system services
- Use package managers effectively
- Write powerful shell scripts
- Process text with command-line tools
- Monitor system resources and performance
- Configure networking from the command line
- Automate tasks with Bash scripting

---

## 📖 Theoretical Concepts

### 2.1 Introduction to Linux

#### What is Linux?

Linux is a **free and open-source Unix-like operating system kernel** created by Linus Torvalds in 1991. It powers:
- 90%+ of cloud infrastructure
- All top 500 supercomputers
- Android devices (Linux kernel)
- Embedded systems
- IoT devices

#### Linux Distributions

**Popular Distributions:**
- **Ubuntu/Debian**: User-friendly, large community
- **RHEL/CentOS/Rocky**: Enterprise-focused
- **Fedora**: Cutting-edge features
- **Arch Linux**: Minimalist, rolling release
- **Alpine**: Lightweight for containers

**Why Linux for DevOps?**
- Free and open-source
- Powerful command-line tools
- Automation-friendly
- Stable and secure
- Industry standard for servers

---

### 2.2 Linux File System Hierarchy

#### Standard Directory Structure

```
/                           Root directory
├── bin/                    Essential user binaries (ls, cp, mv)
├── boot/                   Boot loader files (kernel, initrd)
├── dev/                    Device files (sda, tty, null)
├── etc/                    System configuration files
│   ├── passwd              User account information
│   ├── group               Group information
│   ├── hosts               Hostname to IP mapping
│   └── ssh/                SSH configuration
├── home/                   User home directories
│   ├── user1/              User1's home
│   └── user2/              User2's home
├── lib/                    Shared libraries
├── media/                  Removable media mount points
├── mnt/                    Temporary mount points
├── opt/                    Optional software packages
├── proc/                   Process and kernel information (virtual)
├── root/                   Root user's home directory
├── run/                    Runtime data (PIDs, sockets)
├── sbin/                   System binaries (fdisk, reboot)
├── srv/                    Service data (web, ftp)
├── sys/                    System information (virtual)
├── tmp/                    Temporary files (cleared on reboot)
├── usr/                    User programs and data
│   ├── bin/                User binaries
│   ├── lib/                User libraries
│   ├── local/              Locally installed software
│   └── share/              Shared data
└── var/                    Variable data
    ├── log/                Log files
    ├── mail/               Mail spools
    ├── spool/              Print queues, cron jobs
    └── www/                Web server files
```

#### Important Directories Explained

**/etc - Configuration Files**
- System-wide configuration
- No executables (by convention)
- Text-based configuration files
- Examples: `/etc/passwd`, `/etc/ssh/sshd_config`

**/var - Variable Data**
- Files that change during operation
- Log files: `/var/log/`
- Databases: `/var/lib/mysql/`
- Web content: `/var/www/`

**/home - User Directories**
- Each user has a home directory
- Personal files and configurations
- User-specific settings (`.bashrc`, `.ssh/`)

**/proc - Process Information**
- Virtual filesystem
- Real-time system information
- Process details: `/proc/[PID]/`
- System info: `/proc/cpuinfo`, `/proc/meminfo`

---

### 2.3 Essential Linux Commands

#### Navigation Commands

**pwd - Print Working Directory**
```bash
pwd
# Output: /home/username/documents
```

**cd - Change Directory**
```bash
cd /var/log          # Absolute path
cd ..                # Parent directory
cd ~                 # Home directory
cd -                 # Previous directory
```

**ls - List Directory Contents**
```bash
ls                   # List files
ls -l                # Long format (permissions, size, date)
ls -la               # Include hidden files
ls -lh               # Human-readable sizes
ls -lt               # Sort by modification time
ls -lS               # Sort by size
```

#### File Operations

**cp - Copy Files**
```bash
cp file1.txt file2.txt           # Copy file
cp -r dir1/ dir2/                # Copy directory recursively
cp -p file1 file2                # Preserve attributes
cp -u source dest                # Copy only if newer
```

**mv - Move/Rename Files**
```bash
mv oldname.txt newname.txt       # Rename file
mv file.txt /tmp/                # Move file
mv *.txt documents/              # Move multiple files
```

**rm - Remove Files**
```bash
rm file.txt                      # Remove file
rm -r directory/                 # Remove directory recursively
rm -f file.txt                   # Force remove (no prompt)
rm -rf directory/                # Force remove directory (DANGEROUS!)
```

**mkdir - Create Directories**
```bash
mkdir newdir                     # Create directory
mkdir -p path/to/nested/dir      # Create nested directories
mkdir -m 755 newdir              # Create with specific permissions
```

**touch - Create Empty File**
```bash
touch newfile.txt                # Create empty file
touch -t 202301011200 file.txt   # Set specific timestamp
```

---

### 2.4 File Permissions and Ownership

#### Understanding Permissions

**Permission Structure:**
```
-rwxr-xr--  1 user group 4096 Jan 1 12:00 file.txt
│││││││││
│││││││└└─ Other permissions (r--)
││││││└─── Group permissions (r-x)
│││└└└──── Owner permissions (rwx)
││└──────── Number of hard links
│└───────── File type (- = file, d = directory, l = link)
```

**Permission Types:**
- **r (read)**: View file contents / List directory
- **w (write)**: Modify file / Create/delete files in directory
- **x (execute)**: Run file as program / Enter directory

**Numeric Representation:**
```
r = 4
w = 2
x = 1

rwx = 4+2+1 = 7
r-x = 4+0+1 = 5
r-- = 4+0+0 = 4

chmod 755 file.txt  # rwxr-xr-x
chmod 644 file.txt  # rw-r--r--
chmod 600 file.txt  # rw-------
```

#### chmod - Change Permissions

**Symbolic Mode:**
```bash
chmod u+x file.sh        # Add execute for user
chmod g-w file.txt       # Remove write for group
chmod o+r file.txt       # Add read for others
chmod a+x script.sh      # Add execute for all
chmod u=rwx,g=rx,o=r file.txt  # Set specific permissions
```

**Numeric Mode:**
```bash
chmod 755 script.sh      # rwxr-xr-x
chmod 644 file.txt       # rw-r--r--
chmod 600 secret.key     # rw-------
chmod 777 file.txt       # rwxrwxrwx (AVOID!)
```

#### chown - Change Ownership

```bash
chown user file.txt              # Change owner
chown user:group file.txt        # Change owner and group
chown -R user:group directory/   # Recursive change
```

#### Special Permissions

**Setuid (4000)**
```bash
chmod 4755 program    # Run as file owner
# Example: /usr/bin/passwd
```

**Setgid (2000)**
```bash
chmod 2755 directory  # Files inherit group
```

**Sticky Bit (1000)**
```bash
chmod 1777 /tmp       # Only owner can delete files
```

---

### 2.5 Process Management

#### Understanding Processes

**Process States:**
- **Running (R)**: Currently executing
- **Sleeping (S)**: Waiting for event
- **Stopped (T)**: Suspended
- **Zombie (Z)**: Terminated but not reaped
- **Defunct (D)**: Uninterruptible sleep

#### ps - Process Status

```bash
ps                    # Current shell processes
ps aux                # All processes, detailed
ps -ef                # Full format listing
ps -u username        # User's processes
ps -C processname     # Specific process
```

**Output Explanation:**
```
USER  PID  %CPU %MEM    VSZ   RSS TTY   STAT START   TIME COMMAND
root    1   0.0  0.1 169416 13140 ?     Ss   Jan01   0:05 /sbin/init
```

#### top - Real-time Process Monitor

```bash
top                   # Interactive process viewer
top -u username       # User's processes
htop                  # Enhanced version (if installed)
```

**Top Commands:**
- `k`: Kill process
- `r`: Renice (change priority)
- `M`: Sort by memory
- `P`: Sort by CPU
- `q`: Quit

#### Process Control

**Background and Foreground:**
```bash
command &             # Run in background
jobs                  # List background jobs
fg %1                 # Bring job 1 to foreground
bg %1                 # Resume job 1 in background
Ctrl+Z                # Suspend current process
```

**kill - Terminate Processes:**
```bash
kill PID              # Terminate process (SIGTERM)
kill -9 PID           # Force kill (SIGKILL)
kill -15 PID          # Graceful termination
killall processname   # Kill all instances
pkill pattern         # Kill by pattern
```

**Process Priority:**
```bash
nice -n 10 command    # Start with lower priority
renice -n 5 -p PID    # Change priority
```

---

### 2.6 Package Management

#### APT (Debian/Ubuntu)

**Update Package Lists:**
```bash
sudo apt update                  # Update package lists
sudo apt upgrade                 # Upgrade installed packages
sudo apt full-upgrade            # Upgrade with dependency changes
```

**Install/Remove Packages:**
```bash
sudo apt install package         # Install package
sudo apt install pkg1 pkg2       # Install multiple
sudo apt remove package          # Remove package
sudo apt purge package           # Remove with config files
sudo apt autoremove              # Remove unused dependencies
```

**Search and Information:**
```bash
apt search keyword               # Search packages
apt show package                 # Show package details
apt list --installed             # List installed packages
apt list --upgradable            # List upgradable packages
```

#### YUM/DNF (RHEL/CentOS/Fedora)

```bash
sudo yum update                  # Update packages
sudo yum install package         # Install package
sudo yum remove package          # Remove package
sudo yum search keyword          # Search packages
sudo yum info package            # Package information
```

**DNF (Modern replacement for YUM):**
```bash
sudo dnf update
sudo dnf install package
sudo dnf remove package
```

---

### 2.7 System Services (systemd)

#### systemctl - Service Control

**Service Management:**
```bash
sudo systemctl start nginx       # Start service
sudo systemctl stop nginx        # Stop service
sudo systemctl restart nginx     # Restart service
sudo systemctl reload nginx      # Reload configuration
sudo systemctl status nginx      # Check status
```

**Enable/Disable Services:**
```bash
sudo systemctl enable nginx      # Start on boot
sudo systemctl disable nginx     # Don't start on boot
sudo systemctl is-enabled nginx  # Check if enabled
```

**Service Information:**
```bash
systemctl list-units --type=service        # List all services
systemctl list-units --type=service --state=running  # Running services
systemctl cat nginx                        # View service file
```

#### Service Files

**Location:** `/etc/systemd/system/` or `/lib/systemd/system/`

**Example Service File:**
```ini
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/start.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

---

### 2.8 Shell Scripting Basics

#### Bash Script Structure

**Basic Script:**
```bash
#!/bin/bash
# Script description

# Variables
NAME="DevOps"
COUNT=10

# Output
echo "Hello, $NAME!"
echo "Count: $COUNT"

# Command substitution
CURRENT_DATE=$(date +%Y-%m-%d)
echo "Today is $CURRENT_DATE"
```

#### Variables

```bash
# Variable assignment (no spaces!)
VAR="value"
NUM=42

# Using variables
echo $VAR
echo ${VAR}

# Command substitution
FILES=$(ls -l)
USERS=`who | wc -l`

# Environment variables
echo $HOME
echo $PATH
echo $USER
```

#### Conditionals

```bash
# If statement
if [ $COUNT -gt 5 ]; then
    echo "Count is greater than 5"
elif [ $COUNT -eq 5 ]; then
    echo "Count is exactly 5"
else
    echo "Count is less than 5"
fi

# File tests
if [ -f /etc/passwd ]; then
    echo "File exists"
fi

if [ -d /tmp ]; then
    echo "Directory exists"
fi

# String comparison
if [ "$NAME" = "DevOps" ]; then
    echo "Match!"
fi
```

#### Loops

**For Loop:**
```bash
# Loop over list
for item in apple banana cherry; do
    echo "Fruit: $item"
done

# Loop over files
for file in *.txt; do
    echo "Processing $file"
done

# C-style loop
for ((i=1; i<=5; i++)); do
    echo "Number: $i"
done
```

**While Loop:**
```bash
COUNT=1
while [ $COUNT -le 5 ]; do
    echo "Count: $COUNT"
    ((COUNT++))
done
```

#### Functions

```bash
# Function definition
greet() {
    local name=$1
    echo "Hello, $name!"
}

# Function call
greet "DevOps Engineer"

# Function with return value
add() {
    local sum=$(($1 + $2))
    echo $sum
}

result=$(add 5 3)
echo "Sum: $result"
```

---

### 2.9 Text Processing

#### grep - Search Text

```bash
grep "pattern" file.txt          # Search for pattern
grep -i "pattern" file.txt       # Case-insensitive
grep -r "pattern" directory/     # Recursive search
grep -v "pattern" file.txt       # Invert match
grep -n "pattern" file.txt       # Show line numbers
grep -c "pattern" file.txt       # Count matches
grep -E "regex" file.txt         # Extended regex
```

#### sed - Stream Editor

```bash
sed 's/old/new/' file.txt        # Replace first occurrence
sed 's/old/new/g' file.txt       # Replace all occurrences
sed -i 's/old/new/g' file.txt    # Edit file in-place
sed '5d' file.txt                # Delete line 5
sed -n '1,10p' file.txt          # Print lines 1-10
```

#### awk - Pattern Scanning

```bash
awk '{print $1}' file.txt        # Print first column
awk '{print $1, $3}' file.txt    # Print columns 1 and 3
awk -F: '{print $1}' /etc/passwd # Custom delimiter
awk '$3 > 100' file.txt          # Filter by condition
awk '{sum+=$1} END {print sum}'  # Sum column
```

#### cut - Extract Columns

```bash
cut -d: -f1 /etc/passwd          # Extract first field
cut -c1-10 file.txt              # Extract characters 1-10
cut -f1,3 -d, file.csv           # Extract CSV columns
```

#### sort and uniq

```bash
sort file.txt                    # Sort lines
sort -r file.txt                 # Reverse sort
sort -n file.txt                 # Numeric sort
sort -k2 file.txt                # Sort by column 2

uniq file.txt                    # Remove duplicates
sort file.txt | uniq             # Sort then remove duplicates
sort file.txt | uniq -c          # Count occurrences
```

---

### 2.10 System Monitoring

#### Disk Usage

```bash
df -h                            # Disk space (human-readable)
df -i                            # Inode usage
du -sh directory/                # Directory size
du -h --max-depth=1              # Size of subdirectories
```

#### Memory Usage

```bash
free -h                          # Memory usage
cat /proc/meminfo                # Detailed memory info
vmstat                           # Virtual memory statistics
```

#### CPU Information

```bash
lscpu                            # CPU information
cat /proc/cpuinfo                # Detailed CPU info
uptime                           # System uptime and load
```

#### Network Monitoring

```bash
ip addr show                     # IP addresses
ip route show                    # Routing table
ss -tuln                         # Listening ports
netstat -tuln                    # Network connections (legacy)
ping -c 4 google.com             # Test connectivity
traceroute google.com            # Trace route
```

---

## 🔑 Key Takeaways

1. **Linux is Essential for DevOps**: Industry standard for servers and cloud
2. **Command Line Mastery**: Automation requires strong CLI skills
3. **File System Understanding**: Know where everything lives
4. **Process Management**: Control and monitor running applications
5. **Package Management**: Install and maintain software
6. **Shell Scripting**: Automate repetitive tasks
7. **Text Processing**: Parse logs and configuration files
8. **System Monitoring**: Keep systems healthy and performant

---

## 📚 Additional Resources

### Books
- "The Linux Command Line" by William Shotts
- "Linux Bible" by Christopher Negus
- "UNIX and Linux System Administration Handbook"

### Online Resources
- [Linux Journey](https://linuxjourney.com/)
- [OverTheWire: Bandit](https://overthewire.org/wargames/bandit/)
- [Explain Shell](https://explainshell.com/)

### Practice
- Set up a Linux VM (VirtualBox, VMware)
- Use Linux as your daily driver
- Complete Linux challenges on HackerRank

---

## ⏭️ Next Steps

Complete all 10 labs in the `labs/` directory:

1. **Lab 2.1:** Basic Linux Commands
2. **Lab 2.2:** File System Navigation
3. **Lab 2.3:** File Permissions Management
4. **Lab 2.4:** Process Management
5. **Lab 2.5:** Package Management
6. **Lab 2.6:** Shell Scripting Basics
7. **Lab 2.7:** Text Processing with grep, sed, awk
8. **Lab 2.8:** System Monitoring
9. **Lab 2.9:** Networking Commands
10. **Lab 2.10:** Bash Automation Project

After completing the labs, move on to **Module 3: Version Control with Git**.

---

**Master the Linux Command Line!** 🐧
"""

# Write Module 2
print("Creating Module 2: Linux Fundamentals...")
module_02_path = Path(BASE_PATH) / "phase1_beginner" / "module-02-linux-fundamentals" / "README.md"
with open(module_02_path, 'w', encoding='utf-8') as f:
    f.write(MODULE_02_CONTENT)
print(f"✅ Module 2: {len(MODULE_02_CONTENT.splitlines())} lines")

print("\n✅ Module 2 expansion complete!")
print("Continuing with remaining modules...")
