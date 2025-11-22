# Lab 1.4: Setting Up the DevOps Toolchain

## 🎯 Objective

Install, configure, and verify the essential "Starter Pack" of DevOps tools. By the end of this lab, you will have a fully functioning local environment ready for the rest of the course.

## 📋 Prerequisites

-   Administrator/Root access to your computer.
-   Stable internet connection.
-   ~10GB of free disk space (mostly for Docker images).

## 🧰 The Toolchain

We will install the following core tools:
1.  **VS Code**: The Integrated Development Environment (IDE).
2.  **Git**: Version Control System.
3.  **Docker**: Containerization Platform.
4.  **AWS CLI**: Cloud Interface (even if you don't have an account yet).
5.  **Terraform**: Infrastructure as Code tool.
6.  **Kubectl**: Kubernetes Command Line Tool.

---

## 🔨 Hands-On Implementation

### Part 1: The IDE (VS Code) 💻

Visual Studio Code is the industry standard for DevOps due to its rich ecosystem of extensions.

1.  **Download & Install:**
    -   Visit [code.visualstudio.com](https://code.visualstudio.com/).
    -   Download the installer for your OS (Windows/Mac/Linux).
    -   Run the installer. **Important:** On Windows, check "Add to PATH" and "Add 'Open with Code' action".

2.  **Essential Extensions:**
    Open VS Code, go to the Extensions view (`Ctrl+Shift+X`), and install:
    -   **HashiCorp Terraform** (ID: `hashicorp.terraform`)
    -   **Docker** (ID: `ms-azuretools.vscode-docker`)
    -   **Kubernetes** (ID: `ms-kubernetes-tools.vscode-kubernetes-tools`)
    -   **YAML** (ID: `redhat.vscode-yaml`)
    -   **Markdown All in One** (ID: `yzhang.markdown-all-in-one`) - Great for documentation!

3.  **Configuration:**
    Enable "Auto Save" to prevent losing work:
    -   `File` > `Auto Save`.

### Part 2: Version Control (Git) 🌳

1.  **Install:**
    -   **Windows:** Download [Git for Windows](https://git-scm.com/download/win). Run installer with default settings.
    -   **Mac:** `brew install git` (requires Homebrew) or download installer.
    -   **Linux:** `sudo apt install git` or `sudo yum install git`.

2.  **Configuration (Global):**
    Open your terminal (PowerShell/Bash) and run:
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    git config --global init.defaultBranch main
    git config --global core.editor "code --wait"
    ```

3.  **Verification:**
    ```bash
    git --version
    # Output should be git version 2.x.x
    ```

### Part 3: Containerization (Docker) 🐳

This is often the trickiest installation. Follow carefully.

1.  **Install:**
    -   **Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/). *Note: Requires WSL2 (Windows Subsystem for Linux) enabled.*
    -   **Mac:** Install Docker Desktop for Mac.
    -   **Linux:** Follow official instructions for Docker Engine (Docker Desktop is optional on Linux).

2.  **Start Docker:**
    -   Launch the Docker Desktop application.
    -   Wait for the engine to start (Whale icon in system tray stops animating).

3.  **Verification:**
    Run the "Hello World" container to test the entire pipeline (Client -> Daemon -> Registry -> Runtime).
    ```bash
    docker run hello-world
    ```
    *Expected Output:* "Hello from Docker! This message shows that your installation appears to be working correctly."

### Part 4: Cloud Interface (AWS CLI) ☁️

Even if you use Azure/GCP, AWS CLI is a great tool to learn CLI structure.

1.  **Install:**
    -   **Windows:** Download MSI installer from AWS docs.
    -   **Mac:** `brew install awscli`.
    -   **Linux:** `sudo apt install awscli` (or via pip).

2.  **Verification:**
    ```bash
    aws --version
    # Output: aws-cli/2.x.x ...
    ```

3.  **Configuration (Optional for now):**
    If you have an AWS account:
    ```bash
    aws configure
    # Enter Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output (json)
    ```

### Part 5: Infrastructure as Code (Terraform) 🏗️

1.  **Install:**
    -   **Windows:** Download binary from [terraform.io](https://www.terraform.io/downloads), extract `terraform.exe` to a folder (e.g., `C:\Apps\Terraform`), and **add that folder to your System PATH environment variable**.
    -   **Mac:** `brew install terraform`.
    -   **Linux:** Follow HashiCorp repo instructions.

2.  **Verification:**
    ```bash
    terraform --version
    # Output: Terraform v1.x.x
    ```

### Part 6: Orchestration (Kubectl) ☸️

The command line tool for Kubernetes.

1.  **Install:**
    -   **Windows:** Download `kubectl.exe` from Kubernetes docs and add to PATH (similar to Terraform). *Note: Docker Desktop for Windows includes kubectl automatically.*
    -   **Mac:** `brew install kubectl`.
    -   **Linux:** `sudo apt install kubectl`.

2.  **Verification:**
    ```bash
    kubectl version --client
    # Output: Client Version: v1.x.x
    ```

---

## 🧪 Integrated Toolchain Test

Let's verify everything works together with a "Smoke Test".

1.  **Create a Workspace:**
    ```bash
    mkdir devops-setup-test
    cd devops-setup-test
    ```

2.  **Create a Dockerfile (VS Code + Docker):**
    Open VS Code: `code .`
    Create a file named `Dockerfile`:
    ```dockerfile
    FROM alpine:latest
    CMD ["echo", "DevOps Toolchain Verified!"]
    ```

3.  **Build Image (Docker):**
    In the VS Code terminal (`Ctrl+` `):
    ```bash
    docker build -t toolchain-test .
    ```

4.  **Run Container:**
    ```bash
    docker run toolchain-test
    ```
    *Expected Output:* `DevOps Toolchain Verified!`

5.  **Version Control (Git):**
    ```bash
    git init
    git add Dockerfile
    git commit -m "Initial toolchain test"
    ```

---

## 🎯 Challenges

### Challenge 1: Customizing the Terminal (Difficulty: ⭐⭐)

**Objective:** A DevOps engineer lives in the terminal. Make it informative.

**Task:**
-   Install a custom prompt tool like **Starship** (starship.rs) or **Oh My Posh**.
-   Configure it to show:
    -   Current Git branch.
    -   Current directory.
    -   Execution time of commands.
    -   Exit code of last command (if failed).

**Why?** Seeing your git branch and status directly in the prompt prevents costly mistakes (like committing to `main` accidentally).

### Challenge 2: Docker Web Server (Difficulty: ⭐⭐⭐)

**Task:**
1.  Run an Nginx web server using Docker.
2.  Map port 8080 on your host to port 80 on the container.
3.  Create an `index.html` file on your host saying "My DevOps Setup Works".
4.  Mount that file into the container so it replaces the default Nginx page.
5.  Access `http://localhost:8080` in your browser to verify.

**Command Hint:**
`docker run -d -p ... -v ... nginx`

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 2 Solution</summary>

```bash
# 1. Create the HTML file
echo "<h1>My DevOps Setup Works</h1>" > index.html

# 2. Run Docker container
# -d: Detached mode (runs in background)
# -p 8080:80: Map host port 8080 to container port 80
# -v $(pwd)/index.html:/usr/share/nginx/html/index.html: Mount file (Linux/Mac/PowerShell)
# --name my-web: Name the container

# Linux/Mac/PowerShell
docker run -d -p 8080:80 -v "$(pwd)/index.html:/usr/share/nginx/html/index.html" --name my-web nginx

# 3. Verify
curl http://localhost:8080
# Or open browser to http://localhost:8080

# 4. Cleanup
docker stop my-web
docker rm my-web
```

</details>

---

## 🔑 Key Takeaways

1.  **PATH Variable**: Understanding the system PATH is crucial. If a tool says "command not found," it's usually a PATH issue.
2.  **Integration**: These tools don't live in isolation. VS Code edits the Dockerfile, Docker builds it, Git tracks it.
3.  **Versions Matter**: Always check `tool --version`. DevOps tools evolve fast; syntax changes.

---

## ⏭️ Next Steps

Your environment is ready. Now we need to learn how to work effectively with others.

Proceed to **Lab 1.5: Collaboration Practices**.
