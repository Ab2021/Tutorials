# Lab 13.8: Release Management

## Objective
Implement release management and versioning strategies.

## Learning Objectives
- Use semantic versioning
- Automate changelog generation
- Create GitHub releases
- Manage release branches

---

## Semantic Versioning

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Get version
        id: version
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
      
      - name: Build
        run: |
          docker build -t myapp:${{ steps.version.outputs.VERSION }} .
          docker tag myapp:${{ steps.version.outputs.VERSION }} myapp:latest
```

## Changelog Generation

```yaml
      - name: Generate changelog
        uses: orhun/git-cliff-action@v2
        with:
          config: cliff.toml
          args: --latest
        env:
          OUTPUT: CHANGELOG.md
```

## GitHub Release

```yaml
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: CHANGELOG.md
          draft: false
          prerelease: false
```

## Release Branches

```bash
# Create release branch
git checkout -b release/v1.2.0

# Merge to main
git checkout main
git merge release/v1.2.0

# Tag release
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

## Success Criteria
✅ Semantic versioning used  
✅ Changelog automated  
✅ GitHub releases created  
✅ Release process documented  

**Time:** 40 min
