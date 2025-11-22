#!/usr/bin/env python3
"""
Generate README files for all 30 modules
"""

import os
from pathlib import Path

BASE_PATH = r"H:\My Drive\Codes & Repos\codes_repos\devops-course"

# Module metadata with enhanced descriptions
MODULES = {
    # Phase 1
    "phase1_beginner/module-01-introduction-devops": {
        "title": "Introduction to DevOps",
        "topics": ["DevOps culture", "Principles and practices", "Toolchain overview", "Collaboration", "Continuous improvement"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-02-linux-fundamentals": {
        "title": "Linux Fundamentals",
        "topics": ["Command line basics", "File system", "Permissions", "Process management", "Shell scripting"],
        "duration": "8-10 hours"
    },
    "phase1_beginner/module-03-version-control-git": {
        "title": "Version Control with Git",
        "topics": ["Git basics", "Branching and merging", "Remote repositories", "Collaboration workflows", "Best practices"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-04-networking-basics": {
        "title": "Networking Basics",
        "topics": ["TCP/IP fundamentals", "DNS", "HTTP/HTTPS", "Load balancing", "Network security"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-05-docker-fundamentals": {
        "title": "Docker Fundamentals",
        "topics": ["Containers", "Images", "Dockerfile", "Docker Compose", "Networking and volumes"],
        "duration": "8-10 hours"
    },
    "phase1_beginner/module-06-cicd-basics": {
        "title": "CI/CD Basics",
        "topics": ["CI/CD concepts", "GitHub Actions", "Pipeline creation", "Automated testing", "Deployment automation"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-07-infrastructure-as-code-intro": {
        "title": "Infrastructure as Code - Introduction",
        "topics": ["IaC principles", "Terraform basics", "CloudFormation intro", "State management", "Tool comparison"],
        "duration": "8-10 hours"
    },
    "phase1_beginner/module-08-configuration-management": {
        "title": "Configuration Management",
        "topics": ["Ansible fundamentals", "Playbooks", "Roles", "Variables", "Best practices"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-09-monitoring-logging-basics": {
        "title": "Monitoring and Logging Basics",
        "topics": ["Monitoring concepts", "Prometheus", "Grafana", "Log management", "Alerting"],
        "duration": "6-8 hours"
    },
    "phase1_beginner/module-10-cloud-fundamentals-aws": {
        "title": "Cloud Fundamentals (AWS)",
        "topics": ["AWS services", "EC2", "S3", "VPC", "IAM", "Best practices"],
        "duration": "8-10 hours"
    },
    # Phase 2
    "phase2_intermediate/module-11-advanced-docker": {
        "title": "Advanced Docker",
        "topics": ["Multi-stage builds", "Optimization", "Security", "Orchestration", "Production best practices"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-12-kubernetes-fundamentals": {
        "title": "Kubernetes Fundamentals",
        "topics": ["K8s architecture", "Pods and deployments", "Services", "ConfigMaps and Secrets", "Troubleshooting"],
        "duration": "10-12 hours"
    },
    "phase2_intermediate/module-13-advanced-cicd": {
        "title": "Advanced CI/CD",
        "topics": ["Multi-stage pipelines", "Parallel jobs", "Testing strategies", "Deployment strategies", "Optimization"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-14-infrastructure-as-code-advanced": {
        "title": "Infrastructure as Code - Advanced",
        "topics": ["Terraform modules", "CloudFormation stacks", "Pulumi", "State management", "Enterprise patterns"],
        "duration": "10-12 hours"
    },
    "phase2_intermediate/module-15-configuration-management-advanced": {
        "title": "Configuration Management - Advanced",
        "topics": ["Ansible roles", "Galaxy", "Dynamic inventory", "Custom modules", "Enterprise automation"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-16-monitoring-observability": {
        "title": "Monitoring and Observability",
        "topics": ["Advanced Prometheus", "Distributed tracing", "SLOs/SLIs", "Custom metrics", "Observability patterns"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-17-logging-log-management": {
        "title": "Logging and Log Management",
        "topics": ["ELK Stack", "Fluentd", "Log parsing", "Centralized logging", "Best practices"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-18-security-compliance": {
        "title": "Security and Compliance",
        "topics": ["DevSecOps", "Vulnerability scanning", "Secrets management", "Compliance automation", "Security monitoring"],
        "duration": "8-10 hours"
    },
    "phase2_intermediate/module-19-database-operations": {
        "title": "Database Operations",
        "topics": ["RDS", "Backups", "Replication", "Migrations", "Performance tuning", "Disaster recovery"],
        "duration": "6-8 hours"
    },
    "phase2_intermediate/module-20-cloud-architecture-patterns": {
        "title": "Cloud Architecture Patterns",
        "topics": ["High availability", "Auto-scaling", "Disaster recovery", "Microservices", "Serverless intro"],
        "duration": "8-10 hours"
    },
    # Phase 3
    "phase3_advanced/module-21-advanced-kubernetes": {
        "title": "Advanced Kubernetes",
        "topics": ["Operators", "CRDs", "Helm", "Service mesh", "Production K8s"],
        "duration": "10-12 hours"
    },
    "phase3_advanced/module-22-gitops-argocd": {
        "title": "GitOps and ArgoCD",
        "topics": ["GitOps principles", "ArgoCD", "Progressive delivery", "Multi-cluster", "Best practices"],
        "duration": "8-10 hours"
    },
    "phase3_advanced/module-23-serverless-functions": {
        "title": "Serverless and Functions",
        "topics": ["Lambda", "API Gateway", "Event-driven", "Serverless framework", "Optimization"],
        "duration": "8-10 hours"
    },
    "phase3_advanced/module-24-advanced-monitoring": {
        "title": "Advanced Monitoring",
        "topics": ["Distributed tracing", "APM", "SLO/SLI/SLA", "Error budgets", "Monitoring at scale"],
        "duration": "8-10 hours"
    },
    "phase3_advanced/module-25-chaos-engineering": {
        "title": "Chaos Engineering",
        "topics": ["Chaos principles", "Failure injection", "Resilience testing", "Automated chaos", "Building resilience"],
        "duration": "6-8 hours"
    },
    "phase3_advanced/module-26-multi-cloud-hybrid": {
        "title": "Multi-Cloud and Hybrid",
        "topics": ["Multi-cloud strategy", "AWS/Azure/GCP", "Cloud-agnostic tools", "Hybrid cloud", "Management"],
        "duration": "8-10 hours"
    },
    "phase3_advanced/module-27-platform-engineering": {
        "title": "Platform Engineering",
        "topics": ["Internal platforms", "Developer portals", "Self-service", "Golden paths", "Platform evolution"],
        "duration": "8-10 hours"
    },
    "phase3_advanced/module-28-cost-optimization": {
        "title": "Cost Optimization",
        "topics": ["FinOps", "Cost visibility", "Rightsizing", "Reserved instances", "Governance"],
        "duration": "6-8 hours"
    },
    "phase3_advanced/module-29-incident-management": {
        "title": "Incident Management",
        "topics": ["Incident response", "On-call", "Runbooks", "Post-mortems", "Blameless culture"],
        "duration": "6-8 hours"
    },
    "phase3_advanced/module-30-production-deployment": {
        "title": "Production Deployment",
        "topics": ["Blue-green", "Canary releases", "Feature flags", "Zero-downtime", "Best practices"],
        "duration": "6-8 hours"
    }
}

def create_module_readme(module_path, metadata):
    """Create a comprehensive README for a module"""
    
    module_num = module_path.split('-')[1]
    title = metadata['title']
    topics = metadata['topics']
    duration = metadata['duration']
    
    content = f"""# Module {module_num}: {title}

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the core concepts of {title.lower()}
- Gain hands-on experience with industry-standard tools
- Apply best practices in real-world scenarios
- Build production-ready solutions

---

## 📖 Module Overview

**Duration:** {duration}  
**Difficulty:** {"Beginner" if "phase1" in module_path else "Intermediate" if "phase2" in module_path else "Advanced"}

### Topics Covered

{chr(10).join(f"- {topic}" for topic in topics)}

---

## 📚 Theoretical Concepts

### Introduction

[Comprehensive theoretical content will cover the fundamental concepts, principles, and best practices for {title.lower()}.]

### Key Concepts

[Detailed explanations of core concepts with examples and diagrams]

### Best Practices

[Industry-standard best practices and recommendations]

---

## 🔧 Practical Examples

### Example 1: Basic Implementation

```bash
# Example commands and code
echo "Practical examples will be provided"
```

### Example 2: Advanced Scenario

```bash
# More complex examples
echo "Advanced use cases and patterns"
```

---

## 🎯 Hands-on Labs

This module includes 10 comprehensive labs:

1. **Lab {module_num}.1** - Introduction and setup
2. **Lab {module_num}.2** - Core concepts
3. **Lab {module_num}.3** - Practical implementation
4. **Lab {module_num}.4** - Advanced features
5. **Lab {module_num}.5** - Integration patterns
6. **Lab {module_num}.6** - Security and best practices
7. **Lab {module_num}.7** - Troubleshooting
8. **Lab {module_num}.8** - Performance optimization
9. **Lab {module_num}.9** - Real-world scenarios
10. **Lab {module_num}.10** - Best practices and review

Complete all labs in the `labs/` directory before proceeding to the next module.

---

## 📚 Additional Resources

### Official Documentation
- [Link to official documentation]
- [Related tools and frameworks]

### Tutorials and Guides
- [Recommended tutorials]
- [Video courses]

### Community Resources
- [Forums and discussion groups]
- [GitHub repositories]

---

## 🔑 Key Takeaways

- [Key concept 1]
- [Key concept 2]
- [Key concept 3]
- [Best practice 1]
- [Best practice 2]

---

## ⏭️ Next Steps

1. Complete all 10 labs in the `labs/` directory
2. Review the key concepts and best practices
3. Apply what you've learned in a personal project
4. Proceed to the next module

---

**Keep Learning!** 🚀
"""
    
    # Write README
    readme_path = Path(BASE_PATH) / module_path / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return readme_path

def generate_all_readmes():
    """Generate all module READMEs"""
    
    print("Generating Module READMEs...")
    print("=" * 60)
    
    for module_path, metadata in MODULES.items():
        readme_path = create_module_readme(module_path, metadata)
        module_name = module_path.split('/')[-1]
        print(f"✓ Created README for {module_name}")
    
    print("=" * 60)
    print(f"✅ Successfully created {len(MODULES)} module READMEs!")

if __name__ == "__main__":
    generate_all_readmes()
