# Lab 7.1: Introduction to IaC & Terraform Setup

## 🎯 Objective

Understand the paradigm shift from "ClickOps" (Manual) to "DevOps" (Code). You will install Terraform and verify your environment is ready for Infrastructure as Code.

## 📋 Prerequisites

-   VS Code installed.
-   Terminal access.

## 📚 Background

### What is Infrastructure as Code (IaC)?
-   **Manual**: Login to AWS Console -> Click "Launch Instance" -> Select Ubuntu -> Click "Next"...
    -   *Pros*: Easy for beginners.
    -   *Cons*: Slow, error-prone, not reproducible.
-   **IaC**: Write code (`server.tf`) -> Run `terraform apply`.
    -   *Pros*: Fast, version-controlled, consistent.

### Declarative vs Imperative
-   **Imperative (Scripting)**: "Make a server. Then install Nginx. Then start it." (Step-by-step).
-   **Declarative (Terraform)**: "I want a server with Nginx running." (Desired State). Terraform figures out *how* to do it.

### Idempotency
Running the same code twice should produce the same result.
-   *Bash*: `mkdir folder` -> Fails the second time (Folder exists).
-   *Terraform*: `resource "folder"` -> Does nothing the second time (Already exists).

---

## 🔨 Hands-On Implementation

### Part 1: Install Terraform 🛠️

1.  **Windows (Chocolatey):**
    ```powershell
    choco install terraform
    ```
    *Or download from [terraform.io](https://www.terraform.io/downloads).*

2.  **Linux (Ubuntu):**
    ```bash
    sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
    wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt-get install terraform
    ```

3.  **Mac (Homebrew):**
    ```bash
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform
    ```

4.  **Verify:**
    ```bash
    terraform -version
    ```
    *Output:* `Terraform v1.x.x`.

### Part 2: Setup AWS CLI (Prerequisite for later) ☁️

Even though Terraform does the work, it uses your local AWS credentials.

1.  **Install AWS CLI:**
    [Download Installer](https://aws.amazon.com/cli/).

2.  **Configure:**
    ```bash
    aws configure
    ```
    -   **Access Key ID**: (From your AWS IAM User).
    -   **Secret Access Key**: (From your AWS IAM User).
    -   **Region**: `us-east-1` (or your preferred region).
    -   **Format**: `json`.

3.  **Verify:**
    ```bash
    aws sts get-caller-identity
    ```
    *Result:* Should show your User ARN.

### Part 3: The "ClickOps" vs "IaC" Challenge 🥊

**Task:** Create a file named `manual.txt` manually.
1.  Right-click -> New File -> `manual.txt`.
2.  Write "Created manually".
3.  Save.

**Reflection:**
-   How do you prove who created it?
-   How do you ensure it has the exact same content on your colleague's laptop?
-   If you delete it, can you restore it exactly as it was instantly?

*IaC solves these problems.*

---

## 🎯 Challenges

### Challenge 1: Terraform Autocomplete (Difficulty: ⭐)

**Task:**
Install the **HashiCorp Terraform** extension in VS Code.
It provides syntax highlighting and autocomplete.
*Verification:* Open a `.tf` file. The icon should be purple.

### Challenge 2: Version Manager (Difficulty: ⭐⭐⭐)

**Task:**
Research **tfenv** (Terraform Version Manager).
Install it and use it to install a specific version of Terraform (e.g., `1.0.0`).
*Why?* Different projects might use different Terraform versions.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Search "Terraform" in VS Code Extensions Marketplace. Install the one by HashiCorp.

**Challenge 2:**
```bash
git clone https://github.com/tfutils/tfenv.git ~/.tfenv
echo 'export PATH="$HOME/.tfenv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
tfenv install 1.0.0
tfenv use 1.0.0
```
</details>

---

## 🔑 Key Takeaways

1.  **Terraform is the Standard**: It supports AWS, Azure, Google, Kubernetes, GitHub, and more.
2.  **State**: Terraform remembers what it created.
3.  **Plan**: Always run `terraform plan` before `terraform apply`.

---

## ⏭️ Next Steps

We have the tool. Let's write some code.

Proceed to **Lab 7.2: Terraform Basics**.
