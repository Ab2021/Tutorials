# Security & Compliance (DevSecOps)

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of DevSecOps, including:
- **Shift Left**: Integrating security early in the CI/CD pipeline.
- **Scanning**: Implementing SAST (Static) and DAST (Dynamic) testing.
- **Secrets**: Managing credentials securely with **HashiCorp Vault**.
- **Supply Chain**: Signing container images with **Cosign** and generating **SBOMs**.
- **Compliance**: Enforcing Policy as Code with **OPA**.

---

## 📖 Theoretical Concepts

### 1. DevSecOps

Traditionally, security was a gate at the end of the process ("The Department of No").
**DevSecOps** integrates security practices into the DevOps workflow.
- **Automated**: Security tests run on every commit.
- **Shared Responsibility**: Developers own the security of their code.

### 2. Testing Methodologies

- **SAST (Static Application Security Testing)**: Analyzes source code for vulnerabilities (e.g., SQL Injection flaws) without running it. (Tool: SonarQube).
- **DAST (Dynamic Application Security Testing)**: Attacks the running application from the outside. (Tool: OWASP ZAP).
- **SCA (Software Composition Analysis)**: Checks dependencies (`package.json`) for known CVEs. (Tool: Snyk/Trivy).

### 3. Secrets Management (HashiCorp Vault)

Storing secrets in Git (even encrypted) is risky.
**Vault** provides:
- **Central Storage**: Single source of truth.
- **Dynamic Secrets**: Generate AWS keys on the fly that expire in 1 hour.
- **Encryption as a Service**: Encrypt data without managing keys.

### 4. Supply Chain Security

How do you know the image running in Prod is the one you built in CI?
- **Signing**: Cryptographically sign images with **Cosign**.
- **SBOM (Software Bill of Materials)**: A list of all ingredients (libraries) in your software. Required by many governments now.

---

## 🔧 Practical Examples

### Signing an Image (Cosign)

```bash
# Generate keys
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key user/demo:v1

# Verify image
cosign verify --key cosign.pub user/demo:v1
```

### Vault Dynamic Secrets (AWS)

```bash
# Enable AWS secrets engine
vault secrets enable aws

# Configure root creds
vault write aws/config/root \
    access_key=AKIA... \
    secret_key=...

# Generate dynamic creds
vault read aws/creds/my-role
```

### OPA Policy (Rego)

Deny pods running as root.

```rego
package kubernetes.admission

deny[msg] {
  input.request.kind.kind == "Pod"
  input.request.object.spec.securityContext.runAsNonRoot != true
  msg := "Pods must run as non-root"
}
```

---

## 🎯 Hands-on Labs

- [Lab 18.1: SAST with SonarQube](./labs/lab-18.1-sast-sonarqube.md)
- [Lab 18.2: Container Signing with Cosign](./labs/lab-18.2-container-signing.md)
- [Lab 18.3: Secrets Management](./labs/lab-18.3-secrets-management.md)
- [Lab 18.4: Vault Setup](./labs/lab-18.4-vault-setup.md)
- [Lab 18.5: Security Policies](./labs/lab-18.5-security-policies.md)
- [Lab 18.6: Compliance Automation](./labs/lab-18.6-compliance-automation.md)
- [Lab 18.7: Container Scanning](./labs/lab-18.7-container-scanning.md)
- [Lab 18.8: Sast Dast](./labs/lab-18.8-sast-dast.md)
- [Lab 18.9: Security Monitoring](./labs/lab-18.9-security-monitoring.md)
- [Lab 18.10: Incident Response](./labs/lab-18.10-incident-response.md)

---

## 📚 Additional Resources

### Official Documentation
- [HashiCorp Vault Docs](https://www.vaultproject.io/docs)
- [Sigstore Cosign](https://docs.sigstore.dev/cosign/overview/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

### Tools
- [SonarQube](https://www.sonarqube.org/)
- [Open Policy Agent (OPA)](https://www.openpolicyagent.org/)

---

## 🔑 Key Takeaways

1.  **Identity is the New Perimeter**: Firewalls aren't enough. Use strong Identity (OIDC/Vault).
2.  **Trust Nothing**: Verify signatures, scan dependencies, enforce policies.
3.  **Rotate Secrets**: Static keys leak. Use dynamic secrets with short TTLs.
4.  **Compliance as Code**: Don't write PDF policies. Write Rego policies that block non-compliant deployments.

---

## ⏭️ Next Steps

1.  Complete the labs to secure your pipeline and infrastructure.
2.  Proceed to **[Module 19: Database Operations](../module-19-database-operations/README.md)** to learn how to manage data reliability.
