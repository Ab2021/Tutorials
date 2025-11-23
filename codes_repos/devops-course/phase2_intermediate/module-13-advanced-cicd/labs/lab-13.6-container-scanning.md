# Lab 13.6: Container Scanning in CI/CD

## Objective
Scan container images for vulnerabilities in CI/CD pipelines.

## Learning Objectives
- Scan images with Trivy
- Use Snyk for containers
- Implement security gates
- Generate security reports

---

## Trivy Scanning

```yaml
name: Container Security

on: [push]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL,HIGH'
      
      - name: Upload results
        if: always()
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

## Snyk Container Scan

```yaml
  snyk-scan:
    steps:
      - name: Build image
        run: docker build -t myapp:latest .
      
      - name: Run Snyk
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: myapp:latest
          args: --severity-threshold=high
```

## Anchore Scan

```yaml
  anchore-scan:
    steps:
      - name: Scan with Anchore
        uses: anchore/scan-action@v3
        with:
          image: myapp:latest
          fail-build: true
          severity-cutoff: high
```

## Success Criteria
✅ Images scanned for CVEs  
✅ High/critical vulns block builds  
✅ Reports generated  
✅ Security gate enforced  

**Time:** 40 min
