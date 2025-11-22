# Lab 2.1: Basic Linux Commands & Navigation

## 🎯 Objective

Master the fundamental commands required to navigate and interact with the Linux command line interface (CLI). This is the foundation for all future DevOps work.

## 📋 Prerequisites

-   Access to a Linux terminal (WSL, Mac Terminal, or Linux VM).
-   Completed Module 1.

## 📚 Background

### The Shell

The **Shell** is a program that takes commands from the keyboard and gives them to the operating system to perform. The most common shell is **Bash** (Bourne Again SHell).

**Anatomy of a Command:**
`command -options arguments`
Example: `ls -la /home`
-   `ls`: The command (List).
-   `-la`: Options (Long format, All files).
-   `/home`: Argument (The directory to list).

---

## 🔨 Hands-On Implementation

### Part 1: Where am I? (Navigation) 🗺️

1.  **Print Working Directory (`pwd`):**
    ```bash
    pwd
    # Output example: /home/username
    ```
    *Concept:* In Linux, there are no drive letters (C:, D:). Everything starts from Root (`/`).

2.  **List Contents (`ls`):**
    ```bash
    ls          # Simple list
    ls -l       # Long format (permissions, owner, size, date)
    ls -a       # Show hidden files (starting with .)
    ls -la      # Combine them
    ```
    *Note:* Hidden files usually store configurations (e.g., `.bashrc`).

3.  **Change Directory (`cd`):**
    ```bash
    cd /tmp     # Go to absolute path
    pwd
    cd ..       # Go up one level (Parent directory)
    cd ~        # Go to Home directory (Shortcut)
    cd -        # Go to Previous directory (Back button)
    ```

### Part 2: Creating & Destroying (File Management) 🏗️

1.  **Create Directories (`mkdir`):**
    ```bash
    mkdir devops_lab
    cd devops_lab
    mkdir -p project/src/main  # Create parent directories automatically
    ```

2.  **Create Files (`touch`):**
    ```bash
    touch file1.txt
    touch .hiddenfile
    ```

3.  **Copying (`cp`):**
    ```bash
    cp file1.txt file2.txt        # Copy file
    cp -r project project_backup  # Copy directory (Recursive)
    ```

4.  **Moving/Renaming (`mv`):**
    ```bash
    mv file2.txt newname.txt      # Rename
    mv newname.txt project/       # Move
    ```

5.  **Removing (`rm`):**
    ⚠️ **DANGER ZONE** ⚠️
    ```bash
    rm file1.txt                  # Remove file
    rm -r project_backup          # Remove directory (Recursive)
    # rm -rf /                    # NEVER RUN THIS (Deletes everything)
    ```

### Part 3: Viewing Content 👀

1.  **Cat (`cat`):**
    Concatenate and print. Good for small files.
    ```bash
    echo "Hello World" > hello.txt
    cat hello.txt
    ```

2.  **Less (`less`):**
    Pager for large files. Allows scrolling.
    ```bash
    less /var/log/syslog  # (Or any large file)
    # Press 'q' to exit
    ```

3.  **Head & Tail:**
    ```bash
    head -n 5 hello.txt   # First 5 lines
    tail -n 5 hello.txt   # Last 5 lines
    tail -f /var/log/syslog # Follow file updates in real-time (Crucial for logs!)
    ```

---

## 🎯 Challenges

### Challenge 1: The Directory Tree (Difficulty: ⭐⭐)

**Task:**
Create the following structure using a single command (hint: `mkdir` options):
```
lab2
├── app
│   ├── css
│   └── js
├── config
└── docs
```

### Challenge 2: The Hidden Message (Difficulty: ⭐⭐)

**Task:**
1.  Create a hidden file named `.secret` in your home directory.
2.  Add the text "I am a DevOps Engineer" to it.
3.  Verify it exists.
4.  Verify the content.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
mkdir -p lab2/{app/{css,js},config,docs}
# OR
mkdir -p lab2/app/css lab2/app/js lab2/config lab2/docs
```

**Challenge 2:**
```bash
cd ~
echo "I am a DevOps Engineer" > .secret
ls -la # Verify existence
cat .secret # Verify content
```
</details>

---

## 🔑 Key Takeaways

1.  **Tab Completion**: Always press `Tab` to auto-complete paths. It prevents typos.
2.  **Man Pages**: Stuck? Type `man <command>` (e.g., `man ls`) to read the manual.
3.  **Case Sensitivity**: `File.txt` and `file.txt` are different files in Linux.

---

## ⏭️ Next Steps

Now that we can move around, let's understand the structure of the world we are in.

Proceed to **Lab 2.2: Linux File System Hierarchy**.
