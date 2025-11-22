# Lab 06.3: Pipeline Creation - Building a Complete CI Workflow

## Objective
Build a complete CI pipeline that includes linting, testing, security scanning, and artifact generation for a real-world application.

## Prerequisites
- Completed Labs 06.1 and 06.2
- GitHub account with a repository
- Basic understanding of GitHub Actions YAML syntax

## Learning Objectives
- Create multi-stage pipelines with dependencies
- Implement code quality checks (linting)
- Add security scanning to pipelines
- Generate and store build artifacts
- Understand pipeline optimization techniques

---

## Part 1: Project Setup

### Create a Node.js Application

```bash
mkdir complete-ci-pipeline
cd complete-ci-pipeline
git init

# Initialize Node.js project
npm init -y

# Install dependencies
npm install express
npm install --save-dev jest eslint

# Create app
cat > app.js << 'EOF'
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ message: 'Hello, CI/CD!' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date() });
});

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = { app, server };
EOF

# Create tests
cat > app.test.js << 'EOF'
const request = require('supertest');
const { app, server } = require('./app');

describe('API Endpoints', () => {
  afterAll(() => {
    server.close();
  });

  test('GET / returns welcome message', async () => {
    const response = await request(app).get('/');
    expect(response.statusCode).toBe(200);
    expect(response.body.message).toBe('Hello, CI/CD!');
  });

  test('GET /health returns healthy status', async () => {
    const response = await request(app).get('/health');
    expect(response.statusCode).toBe(200);
    expect(response.body.status).toBe('healthy');
  });
});
EOF

# Add test dependency
npm install --save-dev supertest

# Update package.json scripts
npm pkg set scripts.test="jest"
npm pkg set scripts.lint="eslint ."
npm pkg set scripts.start="node app.js"
```

---

## Part 2: Create Multi-Stage Pipeline

### Complete Workflow File

Create `.github/workflows/complete-ci.yml`:

```yaml
name: Complete CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  NODE_VERSION: '18.x'

jobs:
  # Job 1: Code Quality Checks
  lint:
    name: Code Linting
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run ESLint
        run: npm run lint
        continue-on-error: false

  # Job 2: Security Scanning
  security:
    name: Security Audit
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Run npm audit
        run: npm audit --audit-level=moderate
        continue-on-error: true
      
      - name: Check for vulnerabilities
        run: |
          echo "Checking dependencies for known vulnerabilities..."
          npm audit --json > audit-results.json || true
      
      - name: Upload audit results
        uses: actions/upload-artifact@v3
        with:
          name: security-audit
          path: audit-results.json

  # Job 3: Unit Tests
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: [lint]  # Only run if linting passes
    
    strategy:
      matrix:
        node-version: [16.x, 18.x, 20.x]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test -- --coverage
      
      - name: Upload coverage
        uses: actions/upload-artifact@v3
        if: matrix.node-version == '18.x'
        with:
          name: coverage-report
          path: coverage/

  # Job 4: Build Application
  build:
    name: Build & Package
    runs-on: ubuntu-latest
    needs: [test, security]  # Run after tests and security pass
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci --production
      
      - name: Create build artifact
        run: |
          mkdir -p dist
          cp -r node_modules dist/
          cp app.js package.json dist/
          tar -czf app-${{ github.sha }}.tar.gz dist/
      
      - name: Upload build artifact
        uses: actions/upload-artifact@v3
        with:
          name: application-bundle
          path: app-${{ github.sha }}.tar.gz
          retention-days: 30
      
      - name: Generate build info
        run: |
          echo "Build completed successfully!"
          echo "Commit: ${{ github.sha }}"
          echo "Branch: ${{ github.ref_name }}"
          echo "Artifact: app-${{ github.sha }}.tar.gz"
```

---

## Part 3: Understanding Pipeline Features

### Job Dependencies

```yaml
needs: [lint]  # This job waits for 'lint' to complete
```

**Execution Flow:**
```
lint ──┬──> test ──┬──> build
       │           │
security ──────────┘
```

### Matrix Strategy

```yaml
strategy:
  matrix:
    node-version: [16.x, 18.x, 20.x]
```

This creates **3 parallel jobs**, one for each Node version.

### Artifacts

Artifacts preserve files between jobs and after workflow completion.

```yaml
- uses: actions/upload-artifact@v3
  with:
    name: my-artifact
    path: file.txt
    retention-days: 30  # Keep for 30 days
```

---

## Part 4: Verification

### Push and Monitor

```bash
git add .
git commit -m "Add complete CI pipeline"
git push origin main
```

### Check Pipeline Execution

1. Go to **Actions** tab
2. Click on the latest workflow run
3. Observe the job graph:
   - `lint` and `security` run in parallel
   - `test` waits for `lint`
   - `build` waits for both `test` and `security`

### Download Artifacts

1. Scroll to bottom of workflow run page
2. See "Artifacts" section
3. Download `application-bundle` or `coverage-report`

---

## Part 5: Pipeline Optimization

### Caching Dependencies

Add caching to speed up workflows:

```yaml
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

**Before caching:** `npm install` takes 45 seconds  
**After caching:** `npm install` takes 5 seconds

### Conditional Execution

Run jobs only when specific files change:

```yaml
on:
  push:
    paths:
      - '**.js'
      - 'package.json'
      - '.github/workflows/**'
```

---

## Challenges

### Challenge 1: Add Code Coverage Threshold

Fail the build if coverage drops below 80%:

```yaml
- name: Check coverage threshold
  run: |
    COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
      echo "Coverage $COVERAGE% is below 80%"
      exit 1
    fi
```

### Challenge 2: Slack Notifications

Add a job that sends Slack notifications on failure:

<details>
<summary>Solution</summary>

```yaml
notify:
  name: Notify on Failure
  runs-on: ubuntu-latest
  needs: [build]
  if: failure()
  
  steps:
    - name: Send Slack notification
      uses: slackapi/slack-github-action@v1
      with:
        payload: |
          {
            "text": "Build failed for ${{ github.repository }}"
          }
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```
</details>

### Challenge 3: Deploy to Staging

Add a deployment job that runs only on `main` branch:

```yaml
deploy-staging:
  name: Deploy to Staging
  runs-on: ubuntu-latest
  needs: [build]
  if: github.ref == 'refs/heads/main'
  
  steps:
    - name: Download artifact
      uses: actions/download-artifact@v3
      with:
        name: application-bundle
    
    - name: Deploy (simulated)
      run: |
        echo "Deploying to staging environment..."
        echo "Artifact: $(ls *.tar.gz)"
```

---

## Success Criteria

✅ Pipeline has multiple jobs with dependencies  
✅ Linting runs before tests  
✅ Tests run on multiple Node versions  
✅ Security scanning completes  
✅ Build artifacts are generated and stored  
✅ Pipeline completes in under 5 minutes  

---

## Key Learnings

- **Job dependencies create execution order** - Use `needs` to control flow
- **Matrix builds test multiple configurations** - Catch version-specific bugs
- **Artifacts preserve build outputs** - Essential for deployment
- **Caching speeds up pipelines** - Reuse dependencies between runs
- **Parallel execution saves time** - Run independent jobs simultaneously

---

## Troubleshooting

### Issue: Jobs running in wrong order

**Solution:** Check `needs` dependencies. Jobs without `needs` run immediately.

### Issue: Artifact not found

**Solution:** Ensure `upload-artifact` completes before `download-artifact`. Check artifact name matches exactly.

### Issue: Pipeline too slow

**Solutions:**
- Add caching for dependencies
- Run jobs in parallel where possible
- Use `npm ci` instead of `npm install`

---

## Next Steps

- **Lab 06.4:** Add Docker image building
- **Lab 06.5:** Implement deployment automation

**Estimated Time:** 60 minutes  
**Difficulty:** Intermediate
