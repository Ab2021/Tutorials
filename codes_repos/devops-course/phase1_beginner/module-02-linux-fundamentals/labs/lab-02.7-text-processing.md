# Lab 2.7: Text Processing (grep, sed, awk)

## 🎯 Objective

Master the "Holy Trinity" of Linux text processing. You will learn how to search (`grep`), modify (`sed`), and analyze (`awk`) text streams, which is essential for parsing logs and configuration files.

## 📋 Prerequisites

-   Completed Lab 2.6.
-   Access to a Linux terminal.

## 📚 Background

### The Pipeline (`|`)

The power of Linux comes from piping the output of one command into the input of another.
`command1 | command2 | command3`

### The Trinity

1.  **grep**: Global Regular Expression Print. **Finds lines.**
2.  **sed**: Stream Editor. **Changes text.**
3.  **awk**: Aho, Weinberger, and Kernighan. **Processes columns.**

---

## 🔨 Hands-On Implementation

### Part 1: Setup Data 📄

Create a sample log file `access.log`:
```bash
cat > access.log << EOF
192.168.1.1 - - [22/Nov/2025:10:00:00] "GET /index.html HTTP/1.1" 200 1024
192.168.1.2 - - [22/Nov/2025:10:01:00] "POST /login HTTP/1.1" 401 512
192.168.1.1 - - [22/Nov/2025:10:02:00] "GET /about.html HTTP/1.1" 200 2048
10.0.0.5 - - [22/Nov/2025:10:03:00] "GET /secret.txt HTTP/1.1" 403 128
192.168.1.3 - - [22/Nov/2025:10:04:00] "GET /image.jpg HTTP/1.1" 200 5000
EOF
```

### Part 2: grep (Searching) 🔍

1.  **Find successful requests (200):**
    ```bash
    grep " 200 " access.log
    ```

2.  **Find failed requests (Not 200):**
    ```bash
    grep -v " 200 " access.log
    ```

3.  **Count occurrences:**
    ```bash
    grep -c "192.168.1.1" access.log
    ```

4.  **Case insensitive:**
    ```bash
    grep -i "get" access.log
    ```

### Part 3: sed (Editing) ✏️

1.  **Replace text (Display only):**
    Replace "GET" with "RETRIEVE".
    ```bash
    sed 's/GET/RETRIEVE/g' access.log
    ```
    *Note:* `s` = substitute, `g` = global (all occurrences).

2.  **Delete lines:**
    Delete the 2nd line.
    ```bash
    sed '2d' access.log
    ```

3.  **Edit in-place (Save changes):**
    ⚠️ Modifies the file!
    ```bash
    cp access.log access_backup.log
    sed -i 's/HTTP\/1.1/HTTP\/2.0/g' access.log
    cat access.log
    ```

### Part 4: awk (Columns & Analysis) 📊

1.  **Print specific column:**
    Print only IP addresses (1st column).
    ```bash
    awk '{print $1}' access.log
    ```

2.  **Print multiple columns:**
    Print IP and Status Code (9th column).
    ```bash
    awk '{print $1, $9}' access.log
    ```

3.  **Filter by value:**
    Print lines where Status Code is 403.
    ```bash
    awk '$9 == 403 {print $0}' access.log
    ```

4.  **Sum a column:**
    Calculate total bytes transferred (10th column).
    ```bash
    awk '{sum += $10} END {print sum}' access.log
    ```

---

## 🎯 Challenges

### Challenge 1: Log Analysis Script (Difficulty: ⭐⭐⭐)

**Task:**
Create a script `analyze_logs.sh` that reads `access.log` and outputs:
1.  Total number of requests.
2.  Number of 404 errors (add a 404 line to your log first).
3.  The unique list of IP addresses.

### Challenge 2: Data Cleaning (Difficulty: ⭐⭐)

**Task:**
You have a file `data.csv`:
```csv
Name,  Age,  City
John,  25,  New York
Alice, 30,  London
Bob,   22,  Paris
```
Use `sed` to remove all spaces after commas.
Output should be: `Name,Age,City`...

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
#!/bin/bash
LOG="access.log"
echo "Total Requests: $(wc -l < $LOG)"
echo "404 Errors: $(grep -c " 404 " $LOG)"
echo "Unique IPs:"
awk '{print $1}' $LOG | sort | uniq
```

**Challenge 2:**
```bash
sed 's/, /,/g' data.csv
```
</details>

---

## 🔑 Key Takeaways

1.  **Regex is King**: Learning Regular Expressions makes `grep` and `sed` infinitely more powerful.
2.  **Awk is a Language**: `awk` is actually a full programming language. You can write loops and functions in it.
3.  **Don't Reinvent**: Before writing a Python script to parse a file, check if a one-line `awk` or `sed` command can do it.

---

## ⏭️ Next Steps

We can process text. Now let's keep an eye on the system's health.

Proceed to **Lab 2.8: System Monitoring**.
