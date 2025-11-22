# Lab 2.10: Bash Automation Project (Capstone)

## 🎯 Objective

Combine everything you've learned in Module 2 to build a production-grade **Server Health Monitor Script**. This script will check disk, memory, and CPU usage, and generate a color-coded report. If critical thresholds are met, it will simulate sending an alert.

## 📋 Prerequisites

-   Completed Labs 2.1 - 2.9.
-   A "Can Do" attitude!

## 📚 Background

### The Scenario

You are a DevOps Engineer at "CloudCorp". Your manager wants a daily report on the health of 100 Linux servers. Manually checking them is impossible. You need a script that:
1.  Runs automatically.
2.  Checks vital signs.
3.  Alerts only when things are bad.

---

## 🔨 Hands-On Implementation

### Step 1: Setup the Script 📝

Create `health_check.sh` and add the shebang and variables.

```bash
#!/bin/bash

# Thresholds
DISK_THRESHOLD=90
MEM_THRESHOLD=90
CPU_THRESHOLD=4.0 # Load average

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Server Health Check ===${NC}"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "---------------------------"
```

### Step 2: Check Disk Usage 💾

Add this function to check the root partition.

```bash
check_disk() {
    # Get usage percentage of /
    USAGE=$(df / | grep / | awk '{ print $5 }' | sed 's/%//g')
    
    if [ "$USAGE" -ge "$DISK_THRESHOLD" ]; then
        echo -e "${RED}[CRITICAL] Disk Usage: $USAGE% (Threshold: $DISK_THRESHOLD%)${NC}"
    else
        echo -e "${GREEN}[OK] Disk Usage: $USAGE%${NC}"
    fi
}
```

### Step 3: Check Memory Usage 🧠

Add this function to check used memory.

```bash
check_mem() {
    # Calculate used memory percentage
    TOTAL=$(free | grep Mem: | awk '{print $2}')
    USED=$(free | grep Mem: | awk '{print $3}')
    # Bash doesn't do floating point math easily, so we use awk
    PERCENT=$(awk "BEGIN {print ($USED/$TOTAL)*100}")
    PERCENT_INT=${PERCENT%.*} # Convert to integer
    
    if [ "$PERCENT_INT" -ge "$MEM_THRESHOLD" ]; then
        echo -e "${RED}[CRITICAL] Memory Usage: $PERCENT_INT% (Threshold: $MEM_THRESHOLD%)${NC}"
    else
        echo -e "${GREEN}[OK] Memory Usage: $PERCENT_INT%${NC}"
    fi
}
```

### Step 4: Check Load Average ⚡

Add this function.

```bash
check_load() {
    # Get 1-minute load average
    LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk -F, '{ print $1 }' | xargs)
    
    # Compare floating point numbers using awk
    IS_HIGH=$(awk "BEGIN {print ($LOAD > $CPU_THRESHOLD)}")
    
    if [ "$IS_HIGH" -eq 1 ]; then
        echo -e "${RED}[CRITICAL] Load Average: $LOAD (Threshold: $CPU_THRESHOLD)${NC}"
    else
        echo -e "${GREEN}[OK] Load Average: $LOAD${NC}"
    fi
}
```

### Step 5: Main Execution 🚀

Call the functions.

```bash
check_disk
check_mem
check_load

echo "---------------------------"
echo "Check Complete."
```

### Step 6: Test It

1.  Make executable: `chmod +x health_check.sh`
2.  Run it: `./health_check.sh`

---

## 🎯 Challenges

### Challenge 1: The Alert Simulation (Difficulty: ⭐⭐⭐)

**Task:**
Modify the script so that if *any* check is CRITICAL, it appends a line to a file named `alerts.log` with the timestamp and the error message.

### Challenge 2: Cron Scheduling (Difficulty: ⭐⭐⭐)

**Task:**
Schedule this script to run every hour using `cron`.
1.  Run `crontab -e`.
2.  Add the line: `0 * * * * /path/to/health_check.sh >> /var/log/health_check.log 2>&1`

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 1 Solution</summary>

Add a logging function:

```bash
log_alert() {
    MSG=$1
    echo "$(date) - $MSG" >> alerts.log
}
```

Call it inside the `if` blocks:
```bash
if [ "$USAGE" -ge "$DISK_THRESHOLD" ]; then
    MSG="[CRITICAL] Disk Usage: $USAGE%"
    echo -e "${RED}$MSG${NC}"
    log_alert "$MSG"
...
```
</details>

---

## 🔑 Key Takeaways

1.  **Modular Code**: Using functions (`check_disk`) makes scripts readable and reusable.
2.  **Exit Codes**: In a real monitoring plugin (like for Nagios), you would `exit 1` (Warning) or `exit 2` (Critical) so the monitoring system knows the status.
3.  **Automation**: You just built a tool that saves hours of manual work. That is DevOps.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 2: Linux Fundamentals.
You now possess the CLI skills to survive in a server environment.

Proceed to **Module 3: Version Control with Git**.
