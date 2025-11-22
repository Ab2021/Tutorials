# Lab 2.5: Package Management

## 🎯 Objective

Learn how to install, update, and remove software on Linux. You will understand the difference between package managers (APT, YUM/DNF) and how to manage repositories.

## 📋 Prerequisites

-   Completed Lab 2.4.
-   Access to a Linux terminal with `sudo` privileges.

## 📚 Background

### The App Store for Servers

Linux distributions use **Package Managers** to handle software. They:
1.  Download software from **Repositories** (central servers).
2.  Resolve **Dependencies** (installing libraries needed by the app).
3.  Install/Upgrade/Remove the software.

**Common Managers:**
-   **Debian/Ubuntu**: `apt` (Advanced Package Tool). Files end in `.deb`.
-   **RHEL/CentOS/Fedora**: `yum` or `dnf`. Files end in `.rpm`.
-   **Alpine**: `apk`.

---

## 🔨 Hands-On Implementation

*Note: Instructions below assume Ubuntu/Debian. Equivalents for CentOS are provided in comments.*

### Part 1: Updating the Catalog 📚

Before installing, always update your local list of available software.

1.  **Update Lists:**
    ```bash
    sudo apt update
    # CentOS: sudo dnf check-update
    ```

2.  **Upgrade Installed Packages:**
    ```bash
    sudo apt upgrade -y
    # CentOS: sudo dnf upgrade -y
    ```
    *Note:* `-y` automatically answers "Yes" to prompts.

### Part 2: Installing Software 📥

1.  **Install Nginx:**
    ```bash
    sudo apt install nginx -y
    # CentOS: sudo dnf install nginx -y
    ```

2.  **Verify Installation:**
    ```bash
    nginx -v
    systemctl status nginx
    ```

### Part 3: Searching & Info 🔍

1.  **Search for a package:**
    ```bash
    apt search python3-pip
    ```

2.  **Get Package Info:**
    ```bash
    apt show nginx
    ```
    *Output:* Shows version, maintainer, size, and description.

### Part 4: Removing Software 🗑️

1.  **Remove (Keep Configs):**
    ```bash
    sudo apt remove nginx
    ```
    *Note:* This leaves configuration files in `/etc/nginx`. Good for temporary removal.

2.  **Purge (Delete Everything):**
    ```bash
    sudo apt purge nginx
    # CentOS: sudo dnf remove nginx (dnf doesn't distinguish remove/purge as clearly)
    ```

3.  **Clean Up Dependencies:**
    Removes libraries that were installed for Nginx but are no longer needed.
    ```bash
    sudo apt autoremove
    ```

---

## 🎯 Challenges

### Challenge 1: Installing from a .deb file (Difficulty: ⭐⭐)

**Scenario:** Sometimes software isn't in the official repo (e.g., Google Chrome, VS Code). You download a `.deb` file.

**Task:**
1.  Download a sample `.deb` (or simulate it).
    *Example:* `wget https://github.com/sharkdp/bat/releases/download/v0.18.3/bat_0.18.3_amd64.deb` (A better `cat` command).
2.  Install it using `dpkg` or `apt`.
    *Hint:* `sudo apt install ./filename.deb`

### Challenge 2: Adding a Repository (Difficulty: ⭐⭐⭐)

**Scenario:** You want the latest version of Terraform, not the old one in the Ubuntu repo.

**Task:**
1.  Find the instructions to add the HashiCorp repository.
2.  Add the GPG key.
3.  Add the repo URL to sources.
4.  Update and install Terraform.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
wget https://github.com/sharkdp/bat/releases/download/v0.18.3/bat_0.18.3_amd64.deb
sudo apt install ./bat_0.18.3_amd64.deb
bat --version
```

**Challenge 2 (HashiCorp Example):**
```bash
# 1. Install dependencies
sudo apt install -y gnupg software-properties-common

# 2. Add GPG Key
wget -O- https://apt.releases.hashicorp.com/gpg | \
gpg --dearmor | \
sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg

# 3. Add Repo
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
sudo tee /etc/apt/sources.list.d/hashicorp.list

# 4. Install
sudo apt update
sudo apt install terraform
```
</details>

---

## 🔑 Key Takeaways

1.  **Always Update First**: `apt update` ensures you don't get 404 errors for old versions.
2.  **Sudo is Required**: Installing software affects the whole system.
3.  **Automation**: In scripts, always use `-y` (`apt install -y pkg`) so the script doesn't hang waiting for "Yes/No".

---

## ⏭️ Next Steps

We can install tools. Now let's combine them into powerful scripts.

Proceed to **Lab 2.6: Shell Scripting Basics**.
