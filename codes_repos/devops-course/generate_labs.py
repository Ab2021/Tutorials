#!/usr/bin/env python3
"""
DevOps Course Lab Generator
Creates all 300 lab files with proper structure
"""

import os
from pathlib import Path

BASE_PATH = r"H:\My Drive\Codes & Repos\codes_repos\devops-course"

# Phase 1 - Enhanced with more IaC content
PHASE1_LABS = {
    "01-introduction-devops": [
        "devops-culture", "devops-principles", "devops-lifecycle", "toolchain-overview",
        "collaboration-practices", "automation-benefits", "continuous-improvement",
        "devops-metrics", "case-studies", "career-paths"
    ],
    "02-linux-fundamentals": [
        "basic-commands", "file-system", "permissions", "process-management",
        "package-management", "shell-scripting", "text-processing",
        "system-monitoring", "networking-commands", "bash-automation"
    ],
    "03-version-control-git": [
        "git-basics", "branching-merging", "remote-repositories", "git-workflow",
        "conflict-resolution", "git-tags", "git-stash", "git-rebase",
        "pull-requests", "git-best-practices"
    ],
    "04-networking-basics": [
        "tcp-ip-fundamentals", "dns-configuration", "http-https", "load-balancing",
        "firewalls", "network-troubleshooting", "ssl-tls", "reverse-proxy",
        "cdn-basics", "network-security"
    ],
    "05-docker-fundamentals": [
        "docker-installation", "container-basics", "docker-images", "dockerfile-creation",
        "docker-compose", "container-networking", "volume-management", "docker-registry",
        "multi-container-apps", "docker-security"
    ],
    "06-cicd-basics": [
        "cicd-concepts", "github-actions-intro", "pipeline-creation", "automated-testing",
        "build-automation", "deployment-automation", "pipeline-triggers", "artifacts-management",
        "environment-variables", "pipeline-best-practices"
    ],
    "07-infrastructure-as-code-intro": [
        "iac-concepts", "terraform-installation", "terraform-basics", "cloudformation-intro",
        "resource-creation", "state-management", "terraform-variables", "iac-comparison",
        "terraform-plan-apply", "infrastructure-versioning"
    ],
    "08-configuration-management": [
        "ansible-installation", "inventory-management", "ansible-playbooks", "ansible-modules",
        "variables-facts", "handlers-tasks", "ansible-roles-intro", "templates",
        "ansible-vault", "playbook-best-practices"
    ],
    "09-monitoring-logging-basics": [
        "monitoring-concepts", "prometheus-installation", "metrics-collection", "grafana-dashboards",
        "alerting-basics", "log-management", "log-aggregation", "basic-queries",
        "visualization", "monitoring-best-practices"
    ],
    "10-cloud-fundamentals-aws": [
        "aws-account-setup", "ec2-instances", "s3-storage", "vpc-networking",
        "iam-basics", "security-groups", "aws-cli", "cloudwatch-basics",
        "load-balancers", "aws-best-practices"
    ]
}

# Phase 2 - Enhanced IaC module
PHASE2_LABS = {
    "11-advanced-docker": [
        "multi-stage-builds", "image-optimization", "docker-security-scanning", "health-checks",
        "resource-limits", "docker-networking-advanced", "custom-networks", "docker-swarm-intro",
        "container-orchestration", "production-best-practices"
    ],
    "12-kubernetes-fundamentals": [
        "k8s-architecture", "pods-creation", "deployments", "services",
        "configmaps-secrets", "persistent-volumes", "namespaces", "ingress-controllers",
        "rolling-updates", "k8s-troubleshooting"
    ],
    "13-advanced-cicd": [
        "multi-stage-pipelines", "parallel-jobs", "matrix-builds", "integration-testing",
        "deployment-strategies", "pipeline-optimization", "caching-strategies", "self-hosted-runners",
        "pipeline-security", "advanced-workflows"
    ],
    "14-infrastructure-as-code-advanced": [
        "terraform-modules", "module-composition", "cloudformation-stacks", "pulumi-intro",
        "state-locking", "terraform-import", "cross-tool-comparison", "iac-testing",
        "terraform-cloud", "enterprise-patterns"
    ],
    "15-configuration-management-advanced": [
        "ansible-roles", "role-dependencies", "ansible-galaxy", "dynamic-inventory",
        "custom-modules", "ansible-tower-intro", "callback-plugins", "error-handling",
        "ansible-testing", "enterprise-automation"
    ],
    "16-monitoring-observability": [
        "prometheus-advanced", "custom-metrics", "service-discovery", "alertmanager",
        "grafana-advanced", "distributed-tracing", "jaeger-setup", "metrics-best-practices",
        "slo-sli-setup", "observability-patterns"
    ],
    "17-logging-log-management": [
        "elk-stack-setup", "elasticsearch-basics", "logstash-pipelines", "kibana-dashboards",
        "fluentd-setup", "log-parsing", "log-retention", "log-analysis",
        "centralized-logging", "logging-best-practices"
    ],
    "18-security-compliance": [
        "devsecops-principles", "vulnerability-scanning", "secrets-management", "vault-setup",
        "security-policies", "compliance-automation", "container-scanning", "sast-dast",
        "security-monitoring", "incident-response"
    ],
    "19-database-operations": [
        "rds-setup", "database-backups", "point-in-time-recovery", "read-replicas",
        "database-migrations", "connection-pooling", "database-monitoring", "performance-tuning",
        "disaster-recovery", "database-security"
    ],
    "20-cloud-architecture-patterns": [
        "high-availability", "auto-scaling", "disaster-recovery", "multi-region",
        "caching-strategies", "cdn-implementation", "microservices-architecture", "event-driven",
        "serverless-intro", "architecture-best-practices"
    ]
}

# Phase 3
PHASE3_LABS = {
    "21-advanced-kubernetes": [
        "custom-resources", "operators", "helm-charts", "helm-repositories",
        "service-mesh-istio", "network-policies", "pod-security", "cluster-autoscaling",
        "stateful-applications", "k8s-production"
    ],
    "22-gitops-argocd": [
        "gitops-principles", "argocd-installation", "application-deployment", "sync-strategies",
        "multi-cluster", "argocd-projects", "automated-rollbacks", "progressive-delivery",
        "flux-comparison", "gitops-best-practices"
    ],
    "23-serverless-functions": [
        "lambda-functions", "api-gateway", "event-driven-architecture", "step-functions",
        "serverless-framework", "sam-templates", "lambda-layers", "cold-start-optimization",
        "serverless-monitoring", "cost-optimization"
    ],
    "24-advanced-monitoring": [
        "distributed-tracing-advanced", "apm-tools", "custom-instrumentation", "slo-sli-sla",
        "error-budgets", "synthetic-monitoring", "real-user-monitoring", "performance-analysis",
        "capacity-planning", "monitoring-at-scale"
    ],
    "25-chaos-engineering": [
        "chaos-principles", "chaos-monkey", "failure-injection", "resilience-testing",
        "chaos-experiments", "blast-radius", "steady-state-hypothesis", "automated-chaos",
        "chaos-reporting", "building-resilience"
    ],
    "26-multi-cloud-hybrid": [
        "multi-cloud-strategy", "aws-azure-integration", "gcp-services", "cloud-agnostic-tools",
        "hybrid-cloud", "cloud-migration", "multi-cloud-networking", "cost-comparison",
        "vendor-lock-in", "multi-cloud-management"
    ],
    "27-platform-engineering": [
        "internal-platforms", "developer-portals", "self-service-infrastructure", "golden-paths",
        "platform-apis", "developer-experience", "platform-metrics", "platform-documentation",
        "platform-adoption", "platform-evolution"
    ],
    "28-cost-optimization": [
        "finops-principles", "cost-visibility", "resource-tagging", "rightsizing",
        "reserved-instances", "spot-instances", "cost-allocation", "budget-alerts",
        "waste-elimination", "cost-governance"
    ],
    "29-incident-management": [
        "incident-response", "on-call-setup", "pagerduty-integration", "runbooks",
        "post-mortems", "incident-communication", "escalation-policies", "incident-metrics",
        "blameless-culture", "continuous-improvement"
    ],
    "30-production-deployment": [
        "blue-green-deployment", "canary-releases", "feature-flags", "progressive-rollout",
        "rollback-strategies", "deployment-verification", "smoke-testing", "deployment-automation",
        "zero-downtime", "deployment-best-practices"
    ]
}

def create_lab_file(phase, module_num, module_name, lab_num, lab_name):
    """Create a single lab file with proper structure"""
    
    # Format lab name for title
    lab_title = lab_name.replace('-', ' ').title()
    
    # Determine next lab
    next_lab = lab_num + 1 if lab_num < 10 else "next module"
    
    content = f"""# Lab {module_num}.{lab_num}: {lab_title}

## Objective
Learn and practice {lab_title.lower()} in a hands-on environment.

## Prerequisites
- Completed previous labs in this module
- Required tools installed (see GETTING_STARTED.md)

## Instructions

### Step 1: Setup
[Detailed setup instructions will be provided]

### Step 2: Implementation
[Step-by-step implementation guide]

### Step 3: Verification
[How to verify the implementation works correctly]

## Challenges

### Challenge 1: Basic Implementation
[Challenge description and requirements]

### Challenge 2: Advanced Scenario
[More complex challenge building on the basics]

## Solution

<details>
<summary>Click to reveal solution</summary>

### Solution Steps

```bash
# Example commands
echo "Solution code will be provided here"
```

**Explanation:**
[Detailed explanation of the solution]

</details>

## Success Criteria
✅ [Criterion 1]
✅ [Criterion 2]
✅ [Criterion 3]

## Key Learnings
- [Key concept 1]
- [Key concept 2]
- [Best practice 1]

## Troubleshooting

### Common Issues
**Issue 1:** [Description]
- **Solution:** [Fix]

**Issue 2:** [Description]
- **Solution:** [Fix]

## Additional Resources
- [Link to official documentation]
- [Related tutorial or article]

## Next Steps
Proceed to **Lab {module_num}.{next_lab}** or complete the module assessment.
"""
    
    # Create file path
    labs_dir = Path(BASE_PATH) / phase / f"module-{module_num}-{module_name}" / "labs"
    labs_dir.mkdir(parents=True, exist_ok=True)
    
    lab_file = labs_dir / f"lab-{module_num}.{lab_num}-{lab_name}.md"
    
    # Write content
    with open(lab_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return lab_file

def generate_all_labs():
    """Generate all 300 lab files"""
    
    phases = [
        ("phase1_beginner", PHASE1_LABS),
        ("phase2_intermediate", PHASE2_LABS),
        ("phase3_advanced", PHASE3_LABS)
    ]
    
    total_labs = 0
    
    for phase_name, modules in phases:
        print(f"\n{'='*60}")
        print(f"Creating labs for {phase_name}")
        print(f"{'='*60}")
        
        for module_name, labs in modules.items():
            module_num = module_name.split('-')[0]
            
            print(f"\nModule {module_num}: {module_name}")
            
            for idx, lab_name in enumerate(labs, 1):
                lab_file = create_lab_file(phase_name, module_num, module_name, idx, lab_name)
                print(f"  ✓ Created lab {module_num}.{idx}: {lab_name}")
                total_labs += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully created {total_labs} lab files!")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("DevOps Course Lab Generator")
    print("="*60)
    generate_all_labs()
