# Lab 18.2: Container Signing with Cosign

## 🎯 Objective

Supply Chain Security. How do you know the Docker image you are pulling is actually the one you built? What if a hacker pushed a malicious image with the same tag? **Cosign** allows you to digitally sign images.

## 📋 Prerequisites

-   Docker installed.
-   Cosign installed (`brew install cosign` or download binary).
-   A container registry (Docker Hub or TTL.sh).

## 📚 Background

### Sigstore
A project to make signing software easy.
-   **Cosign**: The tool to sign containers.
-   **Rekor**: The transparency log.
-   **Fulcio**: The CA (Certificate Authority).

---

## 🔨 Hands-On Implementation

### Part 1: Generate Keys 🔑

1.  **Generate Key Pair:**
    ```bash
    cosign generate-key-pair
    ```
    *Prompt:* Enter password.
    *Result:* `cosign.key` (Private) and `cosign.pub` (Public).

### Part 2: Build & Push Image 📤

We need an image to sign. We'll use `ttl.sh` (ephemeral registry) to avoid auth issues, or use Docker Hub.

1.  **Pull & Tag:**
    ```bash
    docker pull alpine
    export IMAGE=ttl.sh/my-signed-alpine-$RANDOM:1h
    docker tag alpine $IMAGE
    ```

2.  **Push:**
    ```bash
    docker push $IMAGE
    ```

### Part 3: Sign the Image ✍️

1.  **Sign:**
    ```bash
    cosign sign --key cosign.key $IMAGE
    ```
    *Prompt:* Enter password.
    *Result:* "Pushing signature to: ..."

### Part 4: Verify the Signature ✅

1.  **Verify:**
    ```bash
    cosign verify --key cosign.pub $IMAGE
    ```
    *Result:*
    ```json
    [
      {
        "critical": {
          "identity": { "docker-reference": "..." },
          "image": { "docker-manifest-digest": "..." },
          "type": "cosign container image signature"
        }
      }
    ]
    ```
    **Success!** The image is trusted.

### Part 5: Verify a Fake (Attack Simulation) 🏴‍☠️

1.  **Push a new image to the SAME tag:**
    ```bash
    docker pull busybox
    docker tag busybox $IMAGE
    docker push $IMAGE
    ```
    *Note:* We overwrote the tag with a different image (Busybox).

2.  **Verify:**
    ```bash
    cosign verify --key cosign.pub $IMAGE
    ```
    *Result:* **Error: no matching signatures found**.
    *Explanation:* The signature attached to the tag matches the *old* digest (Alpine), not the new one (Busybox). You detected the tamper!

---

## 🎯 Challenges

### Challenge 1: Keyless Signing (Difficulty: ⭐⭐⭐)

**Task:**
Cosign supports "Keyless" signing using OIDC (Google/GitHub/Microsoft login).
Run `cosign sign $IMAGE` (no key).
It will open a browser to login.
Verify using `cosign verify --certificate-identity=you@email.com ...`.

### Challenge 2: Kubernetes Policy (Difficulty: ⭐⭐⭐⭐)

**Task:**
Research **Kyverno** or **OPA Gatekeeper**.
Write a policy that blocks any Pod from starting unless the image is signed by your public key.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
Kyverno Policy:
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
spec:
  rules:
  - name: check-image
    match:
      resources:
        kinds: [Pod]
    verifyImages:
    - image: "*"
      key: |
        -----BEGIN PUBLIC KEY-----
        ...
```
</details>

---

## 🔑 Key Takeaways

1.  **Digest vs Tag**: Tags are mutable (can change). Digests (SHA256) are immutable. Signatures attach to Digests.
2.  **Toctou**: Time-of-check to Time-of-use. Verification ensures the image hasn't changed between when you scanned it and when you ran it.
3.  **SBOM**: Software Bill of Materials. You can also attach an SBOM to the image using Cosign (`cosign attach sbom ...`).

---

## ⏭️ Next Steps

We are secure. Now let's manage Data.

Proceed to **Module 19: Database Operations**.
