# Lab 27.1: Internal Platforms

## Objective
Build internal developer platforms (IDPs).

## Learning Objectives
- Design IDP architecture
- Implement self-service
- Create golden paths
- Measure developer experience

---

## Platform Components

```yaml
# Platform stack
components:
  - name: "Developer Portal"
    tool: "Backstage"
  - name: "CI/CD"
    tool: "GitHub Actions"
  - name: "Infrastructure"
    tool: "Terraform Cloud"
  - name: "Observability"
    tool: "Datadog"
  - name: "Secrets"
    tool: "Vault"
```

## Backstage Setup

```bash
npx @backstage/create-app

# Configure
cat > app-config.yaml << 'EOF'
app:
  title: Developer Platform

backend:
  database:
    client: pg
    connection:
      host: localhost
      user: postgres
      password: secret

catalog:
  locations:
    - type: url
      target: https://github.com/myorg/catalog/blob/main/catalog-info.yaml
EOF
```

## Service Template

```yaml
# template.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: nodejs-service
spec:
  type: service
  parameters:
    - title: Service Info
      properties:
        name:
          type: string
        owner:
          type: string
  steps:
    - id: fetch
      name: Fetch template
      action: fetch:template
      input:
        url: ./skeleton
    - id: publish
      name: Publish to GitHub
      action: publish:github
```

## Success Criteria
✅ IDP architecture designed  
✅ Backstage deployed  
✅ Service templates created  
✅ Self-service working  

**Time:** 50 min
