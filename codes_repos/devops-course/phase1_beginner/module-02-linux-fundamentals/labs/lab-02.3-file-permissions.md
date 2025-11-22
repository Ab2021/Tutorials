# Lab 2.3: File Permissions & Ownership

## 🎯 Objective

Master the Linux permission model. You will learn how to read, modify, and secure files using `chmod`, `chown`, and understand the numeric (octal) permission system.

## 📋 Prerequisites

-   Completed Lab 2.2.
-   Access to a Linux terminal.

## 📚 Background

### The Permission String

When you run `ls -l`, you see something like `-rwxr-xr--`.
This breaks down into:
1.  **Type**: `-` (File), `d` (Directory), `l` (Link).
2.  **User (Owner)**: `rwx` (Read, Write, Execute).
3.  **Group**: `r-x` (Read, Execute).
4.  **Others**: `r--` (Read only).

### Numeric Mode (Octal)

-   **Read (r)** = 4
-   **Write (w)** = 2
-   **Execute (x)** = 1

Example: `rwx` = 4+2+1 = **7**. `r-x` = 4+0+1 = **5**.
So `rwxr-xr--` is **754**.

---

## 🔨 Hands-On Implementation

### Part 1: Reading Permissions 📖

1.  **Create a test file:**
    ```bash
    touch secret.txt
    ls -l secret.txt
    ```
    *Output:* `-rw-r--r-- 1 user group 0 ...`
    *Analysis:* User can Read/Write. Group can Read. Others can Read.

### Part 2: Modifying Permissions (`chmod`) 🔒

1.  **Remove Read access for Others:**
    ```bash
    chmod o-r secret.txt
    ls -l secret.txt
    ```
    *Output:* `-rw-r-----`

2.  **Add Execute access for User:**
    ```bash
    chmod u+x secret.txt
    ls -l secret.txt
    ```
    *Output:* `-rwxr-----`

3.  **Using Numbers (The DevOps Way):**
    Set it so ONLY the user can read/write (no one else).
    User = `rw-` (4+2=6). Group = `---` (0). Others = `---` (0).
    ```bash
    chmod 600 secret.txt
    ls -l secret.txt
    ```

4.  **Make a script executable:**
    Create a script:
    ```bash
    echo 'echo "Hello"' > script.sh
    ./script.sh
    # Output: Permission denied
    ```
    Fix it:
    ```bash
    chmod 755 script.sh
    ./script.sh
    # Output: Hello
    ```
    *Note:* 755 (rwxr-xr-x) is the standard for scripts/programs.

### Part 3: Changing Ownership (`chown`) 👤

*Note: You usually need `sudo` to give files to other users.*

1.  **Create a file:**
    ```bash
    touch app.log
    ```

2.  **Change Owner:**
    (Assuming `root` exists, which it always does).
    ```bash
    sudo chown root app.log
    ls -l app.log
    ```
    *Result:* You can no longer edit this file without sudo!

3.  **Change Group:**
    ```bash
    sudo chown :root app.log
    ```

4.  **Change Both:**
    ```bash
    sudo chown root:root app.log
    ```

5.  **Reclaim Ownership:**
    ```bash
    sudo chown $USER:$USER app.log
    ```

---

## 🎯 Challenges

### Challenge 1: The "Public" Folder (Difficulty: ⭐⭐)

**Task:**
Create a directory named `public_share`.
Configure permissions so that:
1.  The Owner has full control (rwx).
2.  The Group has full control (rwx).
3.  Others have Read and Execute (r-x), but CANNOT Write.

**Verify:**
Try to create a file inside it as a different user (if possible) or check `ls -ld public_share`.

### Challenge 2: The Sticky Bit (Difficulty: ⭐⭐⭐)

**Background:** The `/tmp` directory allows everyone to write files, but prevents you from deleting *other people's* files. This is done via the **Sticky Bit**.

**Task:**
1.  Create a directory `shared_space`.
2.  Set permissions to 777 (`chmod 777 shared_space`).
3.  Apply the sticky bit (`chmod +t shared_space`).
4.  Verify the permission string shows a `t` at the end (e.g., `drwxrwxrwt`).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
mkdir public_share
chmod 775 public_share
ls -ld public_share
# Output: drwxrwxr-x ...
```

**Challenge 2:**
```bash
mkdir shared_space
chmod 1777 shared_space
# OR
chmod 777 shared_space
chmod +t shared_space
ls -ld shared_space
# Output: drwxrwxrwt ...
```
</details>

---

## 🔑 Key Takeaways

1.  **Least Privilege**: Always grant the minimum permissions necessary. Don't use `chmod 777` just to make it work!
2.  **Directories need Execute**: To `cd` into a directory, you need Execute (x) permission, not just Read.
3.  **Root is God**: The root user ignores permissions.

---

## ⏭️ Next Steps

We can control files. Now let's control the programs running on the system.

Proceed to **Lab 2.4: Process Management**.
