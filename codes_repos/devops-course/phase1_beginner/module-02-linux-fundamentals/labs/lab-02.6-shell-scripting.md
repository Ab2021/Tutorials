# Lab 2.6: Shell Scripting Basics

## 🎯 Objective

Write your first Bash scripts to automate tasks. Learn about the Shebang, variables, loops, and conditionals.

## 📋 Prerequisites

-   Completed Lab 2.5.
-   Text editor (nano, vim, or VS Code).

## 📚 Background

### What is a Shell Script?

It's a text file containing a list of commands that the shell executes in order. It turns manual typing into automation.

**The Shebang (`#!`)**:
The first line `#!/bin/bash` tells the system which interpreter to use.

---

## 🔨 Hands-On Implementation

### Part 1: Hello World (The Right Way) 👋

1.  **Create the file:**
    ```bash
    nano hello.sh
    ```

2.  **Add content:**
    ```bash
    #!/bin/bash
    # This is a comment
    echo "Hello, World!"
    echo "Current date is: $(date)"
    ```

3.  **Make executable:**
    ```bash
    chmod +x hello.sh
    ```

4.  **Run it:**
    ```bash
    ./hello.sh
    ```

### Part 2: Variables & Input 📦

1.  **Create `greet.sh`:**
    ```bash
    #!/bin/bash
    
    NAME="DevOps Engineer"  # No spaces around =
    echo "Welcome, $NAME"
    
    echo "What is your name?"
    read USER_NAME
    echo "Hello $USER_NAME, nice to meet you!"
    ```

2.  **Run it:**
    ```bash
    ./greet.sh
    ```

### Part 3: Conditionals (If/Else) 🔀

1.  **Create `check_file.sh`:**
    ```bash
    #!/bin/bash
    
    FILENAME="config.txt"
    
    if [ -f "$FILENAME" ]; then
        echo "$FILENAME exists."
    else
        echo "$FILENAME does not exist. Creating it..."
        touch "$FILENAME"
    fi
    ```
    *Note:* Spaces inside `[ ... ]` are mandatory!

### Part 4: Loops (For/While) 🔁

1.  **Create `backup.sh`:**
    ```bash
    #!/bin/bash
    
    FILES="file1.txt file2.txt file3.txt"
    
    # Create dummy files
    touch $FILES
    
    echo "Starting backup..."
    
    for FILE in $FILES; do
        echo "Backing up $FILE to $FILE.bak"
        cp "$FILE" "$FILE.bak"
    done
    
    echo "Backup complete."
    ```

---

## 🎯 Challenges

### Challenge 1: The System Report (Difficulty: ⭐⭐)

**Task:**
Create a script `sys_report.sh` that outputs:
1.  Hostname.
2.  Current User.
3.  Disk Usage (summary).
4.  Memory Usage.

Format the output nicely with headers (e.g., `=== Memory Info ===`).

### Challenge 2: The User Creator (Difficulty: ⭐⭐⭐)

**Task:**
Create a script `create_user.sh` that:
1.  Takes a username as an argument (`./create_user.sh john`).
2.  Checks if the user already exists.
3.  If not, creates the user.
4.  If yes, prints an error.
*Hint: Check `/etc/passwd` or use `id` command.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1 (System Report):**
```bash
#!/bin/bash
echo "=== System Report ==="
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""
echo "=== Disk Usage ==="
df -h /
echo ""
echo "=== Memory Usage ==="
free -h
```

**Challenge 2 (User Creator):**
```bash
#!/bin/bash

USERNAME=$1

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username>"
    exit 1
fi

if id "$USERNAME" &>/dev/null; then
    echo "User $USERNAME already exists!"
else
    echo "Creating user $USERNAME..."
    # sudo useradd -m $USERNAME
    echo "User created (Simulated)."
fi
```
</details>

---

## 🔑 Key Takeaways

1.  **Shebang is vital**: Always start with `#!/bin/bash`.
2.  **Permissions**: Don't forget `chmod +x`.
3.  **Variables**: `$VAR` to use, `VAR=val` to set. No spaces!
4.  **Exit Codes**: Scripts return 0 for success, non-zero for failure.

---

## ⏭️ Next Steps

We can write scripts. Now let's learn how to manipulate text data efficiently.

Proceed to **Lab 2.7: Text Processing (grep, sed, awk)**.
