# Lab 06.8: Artifacts Management

## Objective
Manage build artifacts in CI/CD pipelines.

## Learning Objectives
- Upload/download artifacts
- Share artifacts between jobs
- Store test results
- Manage artifact retention

---

## Upload Artifacts

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm run build
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist-files
          path: dist/
          retention-days: 7
```

## Download Artifacts

```yaml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v3
        with:
          name: dist-files
          path: dist/
      
      - name: Deploy
        run: ./deploy.sh dist/
```

## Test Results

```yaml
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            test-results/**/*.xml
            coverage/**
```

## Docker Images as Artifacts

```yaml
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Save image
        run: docker save myapp:${{ github.sha }} > myapp.tar
      
      - name: Upload image
        uses: actions/upload-artifact@v3
        with:
          name: docker-image
          path: myapp.tar
```

## Success Criteria
✅ Artifacts uploaded  
✅ Artifacts shared between jobs  
✅ Test results stored  

**Time:** 35 min
