# Lab 06.4: Automated Testing in CI/CD

## Objective
Integrate automated testing into CI/CD pipelines.

## Learning Objectives
- Implement unit, integration, and e2e tests in pipelines
- Configure test reporting
- Set up code coverage
- Fail builds on test failures

---

## Unit Tests

```yaml
# .github/workflows/test.yaml
name: Automated Testing

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install pytest pytest-cov
      
      - name: Run unit tests
        run: pytest tests/unit/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Integration Tests

```yaml
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/test
```

## E2E Tests

```yaml
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start application
        run: docker-compose up -d
      - name: Run E2E tests
        run: |
          npm install cypress
          npx cypress run
      - name: Upload test videos
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: cypress-videos
          path: cypress/videos
```

## Test Reporting

```yaml
  - name: Publish test results
    uses: EnricoMi/publish-unit-test-result-action@v2
    if: always()
    with:
      files: |
        test-results/**/*.xml
```

## Success Criteria
✅ Unit tests running in CI  
✅ Integration tests with DB  
✅ E2E tests automated  
✅ Test reports published  

**Time:** 45 min
