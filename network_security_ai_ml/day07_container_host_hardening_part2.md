# Day 7: Container & Host Hardening - Part 2

## Host Hardening for AI/ML

### Operating System Hardening

**Ubuntu/Debian Hardening Script:**
```bash
#!/bin/bash
# AI/ML Host Hardening Script for Ubuntu/Debian

set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/ml-hardening.log
}

log "Starting AI/ML host hardening process..."

# Update system packages
log "Updating system packages..."
apt-get update && apt-get upgrade -y
apt-get autoremove -y
apt-get autoclean

# Install security tools
log "Installing security packages..."
apt-get install -y \
    fail2ban \
    ufw \
    auditd \
    apparmor \
    apparmor-utils \
    rkhunter \
    chkrootkit \
    lynis \
    aide \
    unattended-upgrades

# Configure automatic security updates
log "Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";
EOF

# Enable automatic updates
echo 'APT::Periodic::Update-Package-Lists "1";' > /etc/apt/apt.conf.d/20auto-upgrades
echo 'APT::Periodic::Unattended-Upgrade "1";' >> /etc/apt/apt.conf.d/20auto-upgrades

# Configure SSH hardening
log "Hardening SSH configuration..."
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

cat > /etc/ssh/sshd_config << 'EOF'
# SSH Configuration for AI/ML Hosts
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# Authentication
PermitRootLogin no
MaxAuthTries 3
MaxSessions 10
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# Security settings
X11Forwarding no
PrintMotd no
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 60
AllowUsers mluser datauser

# Logging
SyslogFacility AUTH
LogLevel INFO
EOF

# Restart SSH service
systemctl restart sshd

# Configure firewall
log "Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (adjust port if changed)
ufw allow 22/tcp

# Allow common ML/AI ports
ufw allow 8080/tcp  # Jupyter/ML serving
ufw allow 8888/tcp  # Jupyter notebook
ufw allow 6006/tcp  # TensorBoard
ufw allow 8000/tcp  # ML API endpoints

# Enable firewall
ufw --force enable

# Configure fail2ban
log "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure audit system
log "Configuring audit system..."
cat > /etc/audit/rules.d/ml-audit.rules << 'EOF'
# AI/ML Security Audit Rules

# Monitor authentication events
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor system calls
-a always,exit -F arch=b64 -S execve -k exec
-a always,exit -F arch=b32 -S execve -k exec

# Monitor network configuration
-w /etc/network/ -p wa -k network
-w /etc/hosts -p wa -k network
-w /etc/hostname -p wa -k network

# Monitor ML/AI specific directories  
-w /opt/ml/ -p wa -k ml-access
-w /data/ -p wa -k data-access
-w /models/ -p wa -k model-access

# Monitor Docker/container runtime
-w /usr/bin/docker -p x -k docker
-w /var/lib/docker -p wa -k docker
-w /etc/docker -p wa -k docker

# Monitor privileged commands
-w /bin/su -p x -k privileged
-w /usr/bin/sudo -p x -k privileged
-w /usr/bin/passwd -p x -k privileged

# Monitor file deletions
-a always,exit -F arch=b64 -S unlink -S unlinkat -S rename -S renameat -F success=1 -k delete
-a always,exit -F arch=b32 -S unlink -S unlinkat -S rename -S renameat -F success=1 -k delete
EOF

systemctl enable auditd
systemctl start auditd

# Kernel parameter hardening
log "Configuring kernel parameters..."
cat > /etc/sysctl.d/99-ml-security.conf << 'EOF'
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# IPv6 security (disable if not used)
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# Memory protection
kernel.randomize_va_space = 2
kernel.exec-shield = 1
kernel.core_uses_pid = 1

# Process restrictions
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1

# File system security
fs.suid_dumpable = 0
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
EOF

sysctl -p /etc/sysctl.d/99-ml-security.conf

# Configure AppArmor profiles
log "Configuring AppArmor..."
systemctl enable apparmor
systemctl start apparmor

# Create ML-specific AppArmor profile
cat > /etc/apparmor.d/usr.bin.python3.ml << 'EOF'
#include <tunables/global>

/usr/bin/python3.ml flags=(attach_disconnected) {
  #include <abstractions/base>
  #include <abstractions/python>
  
  # Allow Python execution
  /usr/bin/python3* ix,
  
  # ML/AI specific paths
  /opt/ml/** r,
  /data/** rw,
  /models/** r,
  /tmp/ml-* rw,
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_ptrace,
  deny capability sys_module,
  
  # Network restrictions
  network inet stream,
  network inet6 stream,
  deny network inet dgram,
  deny network inet6 dgram,
  
  # System restrictions
  deny /etc/shadow r,
  deny /etc/passwd w,
  deny /etc/group w,
  deny /root/** rw,
  deny /home/*/.ssh/** rw,
  
  # Proc restrictions
  deny @{PROC}/sys/kernel/** w,
  deny @{PROC}/*/mem r,
  deny @{PROC}/kmem r,
  deny @{PROC}/kcore r,
}
EOF

apparmor_parser -r /etc/apparmor.d/usr.bin.python3.ml

# Set up log monitoring
log "Configuring log monitoring..."
cat > /etc/logrotate.d/ml-security << 'EOF'
/var/log/ml-*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    copytruncate
    postrotate
        /usr/bin/systemctl reload rsyslog > /dev/null 2>&1 || true
    endrotate
}
EOF

# Configure AIDE (file integrity monitoring)
log "Initializing AIDE database..."
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create daily AIDE check
cat > /usr/local/bin/aide-check.sh << 'EOF'
#!/bin/bash
AIDE_LOG="/var/log/aide-check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting AIDE integrity check..." >> $AIDE_LOG
/usr/bin/aide --check >> $AIDE_LOG 2>&1

if [ $? -ne 0 ]; then
    echo "[$DATE] AIDE detected file system changes!" >> $AIDE_LOG
    # Send alert (configure as needed)
    logger -p auth.crit "AIDE detected file system changes on $(hostname)"
fi
EOF

chmod +x /usr/local/bin/aide-check.sh

# Add to crontab
echo "0 2 * * * root /usr/local/bin/aide-check.sh" >> /etc/crontab

# Disable unused services
log "Disabling unused services..."
services_to_disable=(
    "bluetooth"
    "cups"
    "avahi-daemon"
    "whoopsie"
    "apport"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-enabled "$service" 2>/dev/null; then
        systemctl disable "$service"
        systemctl stop "$service"
        log "Disabled service: $service"
    fi
done

# Remove unnecessary packages
log "Removing unnecessary packages..."
apt-get remove --purge -y \
    telnet \
    rsh-client \
    rsh-redone-client \
    talk \
    ntalk \
    finger \
    netcat-traditional

log "Host hardening completed successfully!"
log "Please reboot the system to ensure all changes take effect."
```

**RHEL/CentOS Hardening:**
```bash
#!/bin/bash
# AI/ML Host Hardening Script for RHEL/CentOS

# Enable SELinux
sed -i 's/SELINUX=.*/SELINUX=enforcing/' /etc/selinux/config

# Configure firewalld
systemctl enable firewalld
systemctl start firewalld

# Default zones
firewall-cmd --set-default-zone=drop
firewall-cmd --zone=public --add-service=ssh --permanent
firewall-cmd --zone=public --add-port=8080/tcp --permanent  # ML services
firewall-cmd --zone=public --add-port=8888/tcp --permanent  # Jupyter
firewall-cmd --reload

# Install and configure fail2ban
yum install -y epel-release
yum install -y fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/secure
maxretry = 3
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure audit
systemctl enable auditd
systemctl start auditd

# Add ML-specific audit rules
cat >> /etc/audit/rules.d/audit.rules << 'EOF'
# ML/AI monitoring
-w /opt/ml -p wa -k ml_access
-w /data -p wa -k data_access
-w /models -p wa -k model_access

# Container monitoring
-w /usr/bin/docker -p x -k docker_exec
-w /usr/bin/podman -p x -k podman_exec
EOF

service auditd restart
```

## CIS Benchmarks Implementation

### CIS Controls for AI/ML Hosts

**CIS Control 1: Inventory and Control of Hardware Assets**
```yaml
# Ansible playbook for hardware inventory
---
- name: CIS Control 1 - Hardware Asset Inventory
  hosts: ml_hosts
  become: yes
  tasks:
  
  - name: Gather hardware facts
    setup:
      gather_subset:
        - hardware
        - network
        - virtual
  
  - name: Install hardware discovery tools
    package:
      name:
        - lshw
        - hwinfo
        - pciutils
        - usbutils
      state: present
  
  - name: Generate hardware inventory
    shell: |
      lshw -json > /var/log/hardware-inventory.json
      lspci -v > /var/log/pci-devices.txt
      lsusb -v > /var/log/usb-devices.txt
      dmidecode > /var/log/system-info.txt
    args:
      creates: /var/log/hardware-inventory.json
  
  - name: Check for unauthorized devices
    shell: |
      # Check for USB devices
      lsusb | grep -v "root hub" | wc -l
    register: usb_count
    
  - name: Alert on unauthorized USB devices
    debug:
      msg: "WARNING: {{ usb_count.stdout }} USB devices detected"
    when: usb_count.stdout|int > 0
  
  - name: Document GPU inventory
    shell: |
      if command -v nvidia-smi &> /dev/null; then
        nvidia-smi -L > /var/log/gpu-inventory.txt
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /var/log/gpu-details.csv
      fi
    ignore_errors: yes
```

**CIS Control 2: Inventory and Control of Software Assets**
```bash
#!/bin/bash
# Software inventory for AI/ML hosts

# Create software inventory
dpkg-query -l > /var/log/installed-packages.txt
snap list > /var/log/snap-packages.txt 2>/dev/null || true
pip list > /var/log/python-packages.txt 2>/dev/null || true
conda list > /var/log/conda-packages.txt 2>/dev/null || true

# Check for unauthorized software
UNAUTHORIZED_SOFTWARE=(
    "p2p-clients"
    "file-sharing"
    "games"
    "entertainment"
)

for software in "${UNAUTHORIZED_SOFTWARE[@]}"; do
    if dpkg -l | grep -i "$software"; then
        echo "WARNING: Unauthorized software detected: $software" | logger
    fi
done

# Docker image inventory
if command -v docker &> /dev/null; then
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" > /var/log/docker-images.txt
fi

# Kubernetes image inventory
if command -v kubectl &> /dev/null; then
    kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | sort -u > /var/log/k8s-images.txt
fi
```

**CIS Control 3: Continuous Vulnerability Management**
```yaml
# Vulnerability scanning automation
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vulnerability-scan
  namespace: security
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trivy-scanner
            image: aquasec/trivy:latest
            command:
            - /bin/sh
            - -c
            - |
              # Scan host filesystem
              trivy fs --format json --output /shared/host-scan.json /host
              
              # Scan container images
              trivy image --format json --output /shared/image-scan.json \
                $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(ml-|ai-)")
              
              # Generate summary report
              trivy --quiet --format table --output /shared/summary.txt \
                $(docker images --format "{{.Repository}}:{{.Tag}}" | head -10)
              
              # Upload results to central logging
              curl -X POST -H "Content-Type: application/json" \
                -d @/shared/host-scan.json \
                http://security-logging-service:8080/vulnerability-reports
            volumeMounts:
            - name: host-fs
              mountPath: /host
              readOnly: true
            - name: shared-reports
              mountPath: /shared
            - name: docker-socket
              mountPath: /var/run/docker.sock
              readOnly: true
          volumes:
          - name: host-fs
            hostPath:
              path: /
          - name: shared-reports
            emptyDir: {}
          - name: docker-socket
            hostPath:
              path: /var/run/docker.sock
          restartPolicy: OnFailure
```

**CIS Control 4: Controlled Use of Admin Privileges**
```bash
#!/bin/bash
# Admin privilege monitoring and control

# Configure sudo logging
cat > /etc/sudoers.d/ml-logging << 'EOF'
# Log all sudo commands
Defaults logfile="/var/log/sudo.log"
Defaults log_input, log_output
Defaults iolog_dir="/var/log/sudo-io"
Defaults timestamp_timeout=0

# ML team sudo rules
%ml-admins ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/bin/kubectl, /bin/systemctl restart ml-*
%ml-users ALL=(ALL) NOPASSWD: /usr/bin/docker ps, /usr/bin/docker logs, /usr/bin/kubectl get
EOF

# Monitor privilege escalation
cat > /usr/local/bin/privilege-monitor.sh << 'EOF'
#!/bin/bash
# Monitor for unauthorized privilege escalation

LOG_FILE="/var/log/privilege-monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Check for new SUID/SGID files
find / -type f \( -perm -4000 -o -perm -2000 \) -exec ls -la {} \; > /tmp/current-suid.txt 2>/dev/null

if [ -f /var/lib/suid-baseline.txt ]; then
    if ! diff -q /var/lib/suid-baseline.txt /tmp/current-suid.txt > /dev/null; then
        echo "[$DATE] ALERT: New SUID/SGID files detected!" >> $LOG_FILE
        diff /var/lib/suid-baseline.txt /tmp/current-suid.txt >> $LOG_FILE
        logger -p auth.crit "New SUID/SGID files detected on $(hostname)"
    fi
else
    cp /tmp/current-suid.txt /var/lib/suid-baseline.txt
    echo "[$DATE] Baseline SUID/SGID list created" >> $LOG_FILE
fi

# Monitor sudo usage
if [ -f /var/log/sudo.log ]; then
    tail -n 100 /var/log/sudo.log | grep "$(date +'%b %d')" | while read line; do
        if echo "$line" | grep -E "(su |sudo |doas )" > /dev/null; then
            echo "[$DATE] Privilege escalation: $line" >> $LOG_FILE
        fi
    done
fi
EOF

chmod +x /usr/local/bin/privilege-monitor.sh

# Add to crontab
echo "*/5 * * * * root /usr/local/bin/privilege-monitor.sh" >> /etc/crontab
```

## Kernel Security and Lockdown

### Kernel Hardening Configuration

**Kernel Module Restrictions:**
```bash
#!/bin/bash
# Kernel module security configuration

# Disable module loading after boot
echo "kernel.modules_disabled = 1" >> /etc/sysctl.d/99-kernel-security.conf

# Create module whitelist
cat > /etc/modprobe.d/ml-whitelist.conf << 'EOF'
# AI/ML required modules
install nvidia /sbin/modprobe --ignore-install nvidia
install nvidia_drm /sbin/modprobe --ignore-install nvidia_drm  
install nvidia_modeset /sbin/modprobe --ignore-install nvidia_modeset

# Block dangerous modules
install dccp /bin/true
install sctp /bin/true
install rds /bin/true
install tipc /bin/true
install n-hdlc /bin/true
install ax25 /bin/true
install netrom /bin/true
install x25 /bin/true
install rose /bin/true
install decnet /bin/true
install econet /bin/true
install af_802154 /bin/true
install ipx /bin/true
install appletalk /bin/true
install psnap /bin/true
install p8023 /bin/true
install p8022 /bin/true
install can /bin/true
install atm /bin/true
install cramfs /bin/true
install freevxfs /bin/true
install jffs2 /bin/true
install hfs /bin/true
install hfsplus /bin/true
install squashfs /bin/true
install udf /bin/true
install bluetooth /bin/true
install btusb /bin/true
install firewire-core /bin/true
install thunderbolt /bin/true
EOF

# Configure kernel lockdown
echo "lockdown=confidentiality" >> /etc/default/grub
update-grub

# Enable KASLR and other security features
sed -i 's/GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="kaslr mitigations=on"/' /etc/default/grub
update-grub
```

**Secure Boot Configuration:**
```bash
#!/bin/bash
# Secure Boot setup for AI/ML hosts

# Check if Secure Boot is supported
if [ -d /sys/firmware/efi ]; then
    echo "EFI system detected, configuring Secure Boot..."
    
    # Install mokutil for key management
    apt-get install -y mokutil
    
    # Check Secure Boot status
    mokutil --sb-state
    
    # Generate machine owner key
    openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -outform DER -out MOK.der -nodes -days 36500 -subj "/CN=ML Host MOK/"
    
    # Import the key
    mokutil --import MOK.der
    
    # Sign custom modules (NVIDIA drivers, etc.)
    for module in /lib/modules/$(uname -r)/updates/dkms/*.ko; do
        if [ -f "$module" ]; then
            kmodsign sha512 MOK.priv MOK.der "$module"
        fi
    done
    
    echo "Secure Boot configured. Reboot required to complete setup."
else
    echo "Legacy BIOS system detected. Consider upgrading to UEFI for Secure Boot support."
fi
```

### GRUB Security Configuration

**GRUB Hardening:**
```bash
#!/bin/bash
# GRUB security configuration

# Set GRUB password
read -s -p "Enter GRUB password: " grub_password
echo

# Generate GRUB password hash
grub_hash=$(echo -e "$grub_password\n$grub_password" | grub-mkpasswd-pbkdf2 | tail -1 | cut -d' ' -f7)

# Configure GRUB security
cat > /etc/grub.d/40_custom << EOF
#!/bin/sh
exec tail -n +3 \$0

set superusers="admin"
password_pbkdf2 admin $grub_hash

menuentry 'Ubuntu (Recovery Mode)' --class ubuntu --class gnu-linux --class gnu --class os --unrestricted {
    # Recovery mode entry
}
EOF

# Update GRUB timeout and hide menu
sed -i 's/GRUB_TIMEOUT=.*/GRUB_TIMEOUT=3/' /etc/default/grub
sed -i 's/#GRUB_HIDDEN_TIMEOUT=.*/GRUB_HIDDEN_TIMEOUT=0/' /etc/default/grub

# Add kernel security parameters
sed -i 's/GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="audit=1 kaslr intel_iommu=on amd_iommu=on"/' /etc/default/grub

update-grub

echo "GRUB security configuration completed."
```

This completes the host hardening section. Would you like me to continue with the remaining topics (Container Orchestration Security, Automated Patch Management, etc.) or proceed to Day 8?