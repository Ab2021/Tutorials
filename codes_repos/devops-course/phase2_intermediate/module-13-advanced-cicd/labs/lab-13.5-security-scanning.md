# Lab 13.5: Security Scanning in CI/CD

## Objective
Integrate security scanning into CI/CD pipelines.

## Learning Objectives
- Scan code for vulnerabilities (SAST)
- Scan containers for CVEs
- Check dependencies
- Enforce security gates

---

## SAST Scanning

```yaml
name: Security Scan

on: [push, pull_request]

jobs:
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: auto
      
      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Container Scanning

```yaml
  container-scan:
    steps:
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'  # Fail on vulnerabilities
      
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

## Dependency Scanning

```yaml
  dependency-scan:
    steps:
      - uses: actions/checkout@v3
      
      - name: Run npm audit
        run: npm audit --audit-level=high
      
      - name: Snyk scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          command: test
          args: --severity-threshold=high
```

## Secrets Scanning

```yaml
  secrets-scan:
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: TruffleHog scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
```

## Security Gate

```yaml
  security-gate:
    needs: [sast, container-scan, dependency-scan]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Check all scans passed
        run: |
          if [ "${{ needs.sast.result }}" != "success" ] || \
             [ "${{ needs.container-scan.result }}" != "success" ] || \
             [ "${{ needs.dependency-scan.result }}" != "success" ]; then
            echo "Security scans failed"
            exit 1
          fi
```

## Success Criteria
✅ SAST scanning in pipeline  
✅ Container vulnerabilities detected  
✅ Dependencies scanned  
✅ Security gate enforced  

**Time:** 45 min
