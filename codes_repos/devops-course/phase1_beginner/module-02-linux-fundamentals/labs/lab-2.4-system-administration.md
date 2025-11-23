# Lab 2.4: Linux System Administration

## Objective
Perform essential Linux system administration tasks.

## Learning Objectives
- Manage users and permissions
- Configure system services
- Monitor system resources
- Automate administrative tasks

---

## User Management

```bash
# Create user
sudo useradd -m -s /bin/bash john
sudo passwd john

# Add to sudo group
sudo usermod -aG sudo john

# Set permissions
sudo chown -R john:john /home/john
sudo chmod 755 /home/john

# View user info
id john
groups john
```

## Service Management

```bash
# Systemd service management
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl status nginx
sudo systemctl restart nginx

# View logs
sudo journalctl -u nginx -f

# Create custom service
cat <<EOF | sudo tee /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
ExecStart=/usr/bin/myapp
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start myapp
```

## Resource Monitoring

```bash
# CPU and memory
top
htop

# Disk usage
df -h
du -sh /var/log/*

# Network
netstat -tulpn
ss -tulpn

# Process management
ps aux | grep nginx
kill -9 PID
```

## Automation

```bash
# Cron job
crontab -e
# Add: 0 2 * * * /usr/local/bin/backup.sh

# Backup script
cat <<'EOF' > /usr/local/bin/backup.sh
#!/bin/bash
tar -czf /backup/data-$(date +%Y%m%d).tar.gz /data
find /backup -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup.sh
```

## Success Criteria
✅ Users managed  
✅ Services configured  
✅ Resources monitored  
✅ Tasks automated  

**Time:** 45 min
