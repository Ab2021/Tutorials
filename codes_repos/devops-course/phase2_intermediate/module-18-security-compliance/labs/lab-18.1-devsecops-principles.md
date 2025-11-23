# Lab 18.1: DevSecOps Principles

## Objective
Implement security practices throughout the DevOps lifecycle.

## Learning Objectives
- Shift security left
- Automate security scanning
- Implement security gates
- Practice least privilege

---

## Security in CI/CD

```yaml
name: Security Pipeline

on: [push]

jobs:
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Semgrep
        run: |
          pip install semgrep
          semgrep --config=auto .
  
  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: TruffleHog scan
        run: |
          docker run --rm -v "$PWD:/pwd" trufflesecurity/trufflehog:latest filesystem /pwd
  
  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: OWASP Dependency Check
        run: |
          dependency-check --scan . --format HTML
```

## Container Security

```dockerfile
# Use minimal base
FROM gcr.io/distroless/python3

# Non-root user
USER nonroot

# Read-only filesystem
COPY --chown=nonroot:nonroot app.py /app/
WORKDIR /app

CMD ["app.py"]
```

## Success Criteria
✅ SAST scanning in CI/CD  
✅ Secrets scanning automated  
✅ Dependency checks running  
✅ Containers running as non-root  

**Time:** 45 min
