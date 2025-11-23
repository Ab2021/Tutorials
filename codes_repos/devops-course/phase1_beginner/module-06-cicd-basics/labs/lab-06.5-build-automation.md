# Lab 06.5: Build Automation

## Objective
Automate application builds across different languages and platforms.

## Learning Objectives
- Build Docker images in CI
- Cache build dependencies
- Build multi-platform images
- Optimize build times

---

## Docker Build

```yaml
name: Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: myapp:${{ github.sha }},myapp:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Multi-Platform Builds

```yaml
      - name: Build multi-platform
        uses: docker/build-push-action@v4
        with:
          platforms: linux/amd64,linux/arm64
          push: true
          tags: myapp:${{ github.sha }}
```

## Language-Specific Builds

### Node.js
```yaml
  - uses: actions/setup-node@v3
    with:
      node-version: '18'
      cache: 'npm'
  - run: npm ci
  - run: npm run build
```

### Go
```yaml
  - uses: actions/setup-go@v4
    with:
      go-version: '1.21'
      cache: true
  - run: go build -v ./...
```

### Python
```yaml
  - uses: actions/setup-python@v4
    with:
      python-version: '3.11'
      cache: 'pip'
  - run: pip install -r requirements.txt
  - run: python setup.py build
```

## Build Artifacts

```yaml
  - name: Upload artifacts
    uses: actions/upload-artifact@v3
    with:
      name: build-artifacts
      path: dist/
```

## Success Criteria
✅ Docker images built automatically  
✅ Multi-platform support  
✅ Build caching working  
✅ Artifacts uploaded  

**Time:** 40 min
