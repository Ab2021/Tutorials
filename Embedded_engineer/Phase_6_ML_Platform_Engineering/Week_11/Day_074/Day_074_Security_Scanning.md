# Day 74: The Trojan Horse: Container Security
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 11: Container Optimization

---

> **🎯 Focus Area:** A 5GB container is a big hiding place for malware. Use **Trivy** to scan your Docker images for CVEs (Common Vulnerabilities and Exposures) before deploying.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Install** and run Trivy Scanner.
2.  **Interpret** CVE reports (Critical vs Low).
3.  **Generate** an SBOM (Software Bill of Materials) for compliance.
4.  **Mitigate** vulnerabilities by patching Base Images.
5.  **Explain** the concept of Image Signing (Cosign).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Docker.

### Software Environment
- `trivy` (Install via apt/brew).

---

## 📖 Theoretical Foundation

### 1. The Layers of Vulnerability
1.  **OS Layer:** Old `openssl` in `ubuntu:20.04`. (Fixed by `apt upgrade`).
2.  **App Layer:** Old `requests` library in `requirements.txt`. (Fixed by `pip install --upgrade`).
3.  **Base Image:** The Docker Hub image itself might be compromised.

### 2. Scanning Pipeline
*   **Static Analysis:** Look at the file system of the image without running it. Check versions against NVD (National Vulnerability Database).
*   **Dynamic Analysis:** Run the container and watch system calls (Falco). (Day 76).

### 3. Supply Chain Security (SLSA)
How do you know `pytorch/pytorch` really came from Meta?
**Cosign:** Signs the image hash with a private key. You verify the public key before pulling.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Scanning with Trivy

1.  **Install Trivy:**
    ```bash
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh
    ```

2.  **Scan an Image:**
    ```bash
    trivy image python:3.9
    ```
    **Output:**
    ```text
    Total: 45 (UNKNOWN: 0, LOW: 20, MEDIUM: 15, HIGH: 8, CRITICAL: 2)
    
    Library       Vulnerability  Severity  Installed Version  Fixed Version
    openssl       CVE-2023-1234  CRITICAL  1.1.1n             1.1.1o
    ```

3.  **Scan Filesystem (Code):**
    ```bash
    trivy fs .
    ```
    Finds secrets committed in git.

### 👨‍💻 Core Implementation: Fixing it

**Strategy 1: Upgrade Base Image**
Change `FROM python:3.9` to `FROM python:3.9.18-slim` (Newer patch version).

**Strategy 2: Distroless**
Distroless images have fewer packages -> fewer CVEs.
Scan a distroless image and compare:
```bash
trivy image gcr.io/distroless/python3-debian11
```
*Result: Often 0 Critical CVEs.*

### 👨‍💻 Automation (CI/CD)

Add this to GitHub Actions:
```yaml
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'my-app:latest'
          format: 'table'
          exit-code: '1' # Fail pipeline if Critical found
          ignore-unfixed: true
          vuln-type: 'os,library'
          severity: 'CRITICAL,HIGH'
```

---

## 🔬 Lab Exercise: "The SBOM"

### Task
Generate a bill of materials.
1.  Run `trivy image --format cylonedx --output sbom.json python:3.9`.
2.  Open `sbom.json`.
3.  **Observation:** It lists *every* package (sqlite, zlib, python packages).
4.  **Use Case:** When a new "Log4j" happens, you grep your SBOMs to see if you are affected instantly.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Shift Left:** Scan locally before pushing. Don't wait for the Security Team to yell at you.
2.  **Ignore Unfixed:** Many CVEs have no patch available. Filter them out to reduce noise (`--ignore-unfixed`).
3.  **Secrets:** Trivy also finds `AWS_ACCESS_KEY` hardcoded in files.

### API Summary
```bash
trivy image <image>
trivy fs <dir>
```

---

**Day 74 Complete** ✅

*Next: Day 75 - Caching & BuildKit - Speeding up Docker builds.*
