# Lab 2.5: Shell Scripting for Automation

## Objective
Write shell scripts to automate common DevOps tasks.

## Learning Objectives
- Write robust shell scripts
- Handle errors properly
- Use functions and loops
- Automate deployment tasks

---

## Deployment Script

```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
APP_NAME="myapp"
DEPLOY_DIR="/opt/${APP_NAME}"
BACKUP_DIR="/backup/${APP_NAME}"
LOG_FILE="/var/log/${APP_NAME}-deploy.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Backup function
backup() {
    log "Creating backup..."
    tar -czf "${BACKUP_DIR}/backup-$(date +%Y%m%d-%H%M%S).tar.gz" \
        -C "$DEPLOY_DIR" . || error_exit "Backup failed"
    log "Backup completed"
}

# Deploy function
deploy() {
    local version=$1
    log "Deploying version $version..."
    
    # Download
    wget -q "https://releases.example.com/${APP_NAME}-${version}.tar.gz" \
        -O /tmp/${APP_NAME}.tar.gz || error_exit "Download failed"
    
    # Extract
    tar -xzf /tmp/${APP_NAME}.tar.gz -C "$DEPLOY_DIR" \
        || error_exit "Extraction failed"
    
    # Restart service
    systemctl restart ${APP_NAME} || error_exit "Service restart failed"
    
    log "Deployment completed"
}

# Health check
health_check() {
    log "Running health check..."
    for i in {1..5}; do
        if curl -sf http://localhost:8080/health > /dev/null; then
            log "Health check passed"
            return 0
        fi
        sleep 2
    done
    error_exit "Health check failed"
}

# Main
main() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 <version>"
        exit 1
    fi
    
    backup
    deploy "$1"
    health_check
    
    log "Deployment successful!"
}

main "$@"
```

## Monitoring Script

```bash
#!/bin/bash

# Monitor system resources
while true; do
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    MEM=$(free | grep Mem | awk '{print ($3/$2) * 100}')
    DISK=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    
    if (( $(echo "$CPU > 80" | bc -l) )); then
        echo "HIGH CPU: ${CPU}%"
    fi
    
    if (( $(echo "$MEM > 80" | bc -l) )); then
        echo "HIGH MEMORY: ${MEM}%"
    fi
    
    if [ "$DISK" -gt 80 ]; then
        echo "HIGH DISK: ${DISK}%"
    fi
    
    sleep 60
done
```

## Success Criteria
✅ Scripts written  
✅ Error handling implemented  
✅ Automation working  
✅ Deployments automated  

**Time:** 45 min
