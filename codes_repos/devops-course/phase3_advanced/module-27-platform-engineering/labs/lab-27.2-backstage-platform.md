# Lab 27.2: Internal Developer Platform with Backstage

## Objective
Build an internal developer platform using Backstage.

## Learning Objectives
- Install Backstage
- Create software catalog
- Build custom plugins
- Implement self-service workflows

---

## Install Backstage

```bash
# Create Backstage app
npx @backstage/create-app@latest

cd my-backstage-app
yarn dev
```

## Software Catalog

```yaml
# catalog-info.yaml
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: my-service
  description: My microservice
  annotations:
    github.com/project-slug: myorg/my-service
    backstage.io/techdocs-ref: dir:.
spec:
  type: service
  lifecycle: production
  owner: team-a
  system: order-system
  providesApis:
    - order-api
  consumesApis:
    - payment-api
```

## Custom Template

```yaml
# template.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: nodejs-service
  title: Node.js Microservice
  description: Create a new Node.js microservice
spec:
  owner: platform-team
  type: service
  parameters:
    - title: Service Details
      required:
        - name
        - owner
      properties:
        name:
          title: Name
          type: string
        owner:
          title: Owner
          type: string
  steps:
    - id: fetch
      name: Fetch Template
      action: fetch:template
      input:
        url: ./skeleton
        values:
          name: ${{ parameters.name }}
    
    - id: publish
      name: Publish to GitHub
      action: publish:github
      input:
        repoUrl: github.com?repo=${{ parameters.name }}
```

## Success Criteria
✅ Backstage running  
✅ Software catalog populated  
✅ Templates working  
✅ Self-service enabled  

**Time:** 50 min
