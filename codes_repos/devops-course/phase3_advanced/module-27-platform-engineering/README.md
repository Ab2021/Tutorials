# Platform Engineering

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Platform Engineering, including:
- **Concepts**: Understanding the shift from DevOps to Platform Teams.
- **Internal Developer Platforms (IDPs)**: Building self-service portals with **Backstage**.
- **Golden Paths**: Creating opinionated, paved roads for developers.
- **Developer Experience (DX)**: Measuring and improving productivity.
- **Platform as a Product**: Treating your platform like a product with users (developers).

---

## 📖 Theoretical Concepts

### 1. What is Platform Engineering?

Platform Engineering is the discipline of designing and building toolchains and workflows that enable self-service capabilities for software engineering organizations.
- **Problem**: DevOps teams become bottlenecks ("Can you provision a database for me?").
- **Solution**: Build a platform where developers can provision their own databases via a UI/API.

### 2. Internal Developer Platform (IDP)

A curated set of tools, services, and workflows that developers use to build and deploy applications.
- **Service Catalog**: "What services are available?" (Postgres, Redis, Kafka).
- **Software Templates**: "Create a new microservice" button that scaffolds code, CI/CD, and infrastructure.
- **Documentation**: Centralized docs for all platform capabilities.

### 3. Backstage (by Spotify)

The leading open-source IDP framework.
- **Catalog**: Discover all services, APIs, and teams.
- **Templates**: Cookiecutter for infrastructure.
- **TechDocs**: Docs-as-code (Markdown in Git).
- **Plugins**: Extensible (Kubernetes, ArgoCD, PagerDuty).

### 4. Golden Paths

The "blessed" way to do something.
- **Example**: "To deploy a new service, use this Helm chart, this CI template, and this Terraform module."
- **Why**: Reduces cognitive load. Developers don't need to be experts in K8s.

---

## 🔧 Practical Examples

### Backstage Software Template

```yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: nodejs-microservice
  title: Node.js Microservice
spec:
  owner: platform-team
  type: service
  parameters:
    - title: Service Details
      properties:
        name:
          title: Name
          type: string
  steps:
    - id: fetch-base
      name: Fetch Base Template
      action: fetch:template
      input:
        url: ./skeleton
    - id: publish
      name: Publish to GitHub
      action: publish:github
```

### Platform API (Terraform Module)

```hcl
module "postgres_db" {
  source = "git::https://github.com/myorg/terraform-modules//postgres"

  db_name = "my-app-db"
  size    = "db.t3.micro"
}
```

---

## 🎯 Hands-on Labs

- [Lab 27.1: Backstage (Internal Developer Platform)](./labs/lab-27.1-backstage.md)
- [Lab 27.2: Golden Paths (Software Templates)](./labs/lab-27.2-software-templates.md)
- [Lab 27.3: Self Service Infrastructure](./labs/lab-27.3-self-service-infrastructure.md)
- [Lab 27.4: Golden Paths](./labs/lab-27.4-golden-paths.md)
- [Lab 27.5: Platform Apis](./labs/lab-27.5-platform-apis.md)
- [Lab 27.6: Developer Experience](./labs/lab-27.6-developer-experience.md)
- [Lab 27.7: Platform Metrics](./labs/lab-27.7-platform-metrics.md)
- [Lab 27.8: Platform Documentation](./labs/lab-27.8-platform-documentation.md)
- [Lab 27.9: Platform Adoption](./labs/lab-27.9-platform-adoption.md)
- [Lab 27.10: Platform Evolution](./labs/lab-27.10-platform-evolution.md)

---

## 📚 Additional Resources

### Official Documentation
- [Backstage Documentation](https://backstage.io/docs/overview/what-is-backstage)
- [Platform Engineering Manifesto](https://platformengineering.org/)

### Books
- "Team Topologies" by Matthew Skelton and Manuel Pais.

---

## 🔑 Key Takeaways

1.  **Platform Teams Enable, Not Control**: Don't force developers to use the platform. Make it so good they want to.
2.  **Measure Developer Productivity**: DORA metrics (Deployment Frequency, Lead Time, MTTR, Change Failure Rate).
3.  **Start Small**: Don't build everything. Start with the biggest pain point (e.g., "Provisioning a DB takes 2 weeks").
4.  **Documentation is Critical**: A platform without docs is useless.

---

## ⏭️ Next Steps

1.  Complete the labs to build your own IDP.
2.  Proceed to **[Module 28: Cost Optimization](../module-28-cost-optimization/README.md)** to reduce your cloud bill.
