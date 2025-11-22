# DevOps Course Structure Generator
# This script creates all modules and labs for the DevOps course

$basePath = "H:\My Drive\Codes & Repos\codes_repos\devops-course"

# Phase 1 Modules
$phase1Modules = @(
    @{num="01"; name="introduction-devops"; labs=@(
        "devops-culture", "devops-principles", "devops-lifecycle", "toolchain-overview",
        "collaboration-practices", "automation-benefits", "continuous-improvement",
        "devops-metrics", "case-studies", "career-paths"
    )},
    @{num="02"; name="linux-fundamentals"; labs=@(
        "basic-commands", "file-system", "permissions", "process-management",
        "package-management", "shell-scripting", "text-processing",
        "system-monitoring", "networking-commands", "bash-automation"
    )},
    @{num="03"; name="version-control-git"; labs=@(
        "git-basics", "branching-merging", "remote-repositories", "git-workflow",
        "conflict-resolution", "git-tags", "git-stash", "git-rebase",
        "pull-requests", "git-best-practices"
    )},
    @{num="04"; name="networking-basics"; labs=@(
        "tcp-ip-fundamentals", "dns-configuration", "http-https", "load-balancing",
        "firewalls", "network-troubleshooting", "ssl-tls", "reverse-proxy",
        "cdn-basics", "network-security"
    )},
    @{num="05"; name="docker-fundamentals"; labs=@(
        "docker-installation", "container-basics", "docker-images", "dockerfile-creation",
        "docker-compose", "container-networking", "volume-management", "docker-registry",
        "multi-container-apps", "docker-security"
    )},
    @{num="06"; name="cicd-basics"; labs=@(
        "cicd-concepts", "github-actions-intro", "pipeline-creation", "automated-testing",
        "build-automation", "deployment-automation", "pipeline-triggers", "artifacts-management",
        "environment-variables", "pipeline-best-practices"
    )},
    @{num="07"; name="infrastructure-as-code-intro"; labs=@(
        "iac-concepts", "terraform-installation", "terraform-basics", "resource-creation",
        "state-management", "terraform-variables", "terraform-outputs", "terraform-modules-intro",
        "terraform-plan-apply", "infrastructure-versioning"
    )},
    @{num="08"; name="configuration-management"; labs=@(
        "ansible-installation", "inventory-management", "ansible-playbooks", "ansible-modules",
        "variables-facts", "handlers-tasks", "ansible-roles-intro", "templates",
        "ansible-vault", "playbook-best-practices"
    )},
    @{num="09"; name="monitoring-logging-basics"; labs=@(
        "monitoring-concepts", "prometheus-installation", "metrics-collection", "grafana-dashboards",
        "alerting-basics", "log-management", "log-aggregation", "basic-queries",
        "visualization", "monitoring-best-practices"
    )},
    @{num="10"; name="cloud-fundamentals-aws"; labs=@(
        "aws-account-setup", "ec2-instances", "s3-storage", "vpc-networking",
        "iam-basics", "security-groups", "aws-cli", "cloudwatch-basics",
        "load-balancers", "aws-best-practices"
    )}
)

# Phase 2 Modules
$phase2Modules = @(
    @{num="11"; name="advanced-docker"; labs=@(
        "multi-stage-builds", "image-optimization", "docker-security-scanning", "health-checks",
        "resource-limits", "docker-networking-advanced", "custom-networks", "docker-swarm-intro",
        "container-orchestration", "production-best-practices"
    )},
    @{num="12"; name="kubernetes-fundamentals"; labs=@(
        "k8s-architecture", "pods-creation", "deployments", "services",
        "configmaps-secrets", "persistent-volumes", "namespaces", "ingress-controllers",
        "rolling-updates", "k8s-troubleshooting"
    )},
    @{num="13"; name="advanced-cicd"; labs=@(
        "multi-stage-pipelines", "parallel-jobs", "matrix-builds", "integration-testing",
        "deployment-strategies", "pipeline-optimization", "caching-strategies", "self-hosted-runners",
        "pipeline-security", "advanced-workflows"
    )},
    @{num="14"; name="infrastructure-as-code-advanced"; labs=@(
        "terraform-modules", "module-composition", "workspaces", "remote-state",
        "state-locking", "terraform-import", "data-sources", "provisioners",
        "terraform-cloud", "enterprise-patterns"
    )},
    @{num="15"; name="configuration-management-advanced"; labs=@(
        "ansible-roles", "role-dependencies", "ansible-galaxy", "dynamic-inventory",
        "custom-modules", "ansible-tower-intro", "callback-plugins", "error-handling",
        "ansible-testing", "enterprise-automation"
    )},
    @{num="16"; name="monitoring-observability"; labs=@(
        "prometheus-advanced", "custom-metrics", "service-discovery", "alertmanager",
        "grafana-advanced", "distributed-tracing", "jaeger-setup", "metrics-best-practices",
        "slo-sli-setup", "observability-patterns"
    )},
    @{num="17"; name="logging-log-management"; labs=@(
        "elk-stack-setup", "elasticsearch-basics", "logstash-pipelines", "kibana-dashboards",
        "fluentd-setup", "log-parsing", "log-retention", "log-analysis",
        "centralized-logging", "logging-best-practices"
    )},
    @{num="18"; name="security-compliance"; labs=@(
        "devsecops-principles", "vulnerability-scanning", "secrets-management", "vault-setup",
        "security-policies", "compliance-automation", "container-scanning", "sast-dast",
        "security-monitoring", "incident-response"
    )},
    @{num="19"; name="database-operations"; labs=@(
        "rds-setup", "database-backups", "point-in-time-recovery", "read-replicas",
        "database-migrations", "connection-pooling", "database-monitoring", "performance-tuning",
        "disaster-recovery", "database-security"
    )},
    @{num="20"; name="cloud-architecture-patterns"; labs=@(
        "high-availability", "auto-scaling", "disaster-recovery", "multi-region",
        "caching-strategies", "cdn-implementation", "microservices-architecture", "event-driven",
        "serverless-intro", "architecture-best-practices"
    )}
)

# Phase 3 Modules
$phase3Modules = @(
    @{num="21"; name="advanced-kubernetes"; labs=@(
        "custom-resources", "operators", "helm-charts", "helm-repositories",
        "service-mesh-istio", "network-policies", "pod-security", "cluster-autoscaling",
        "stateful-applications", "k8s-production"
    )},
    @{num="22"; name="gitops-argocd"; labs=@(
        "gitops-principles", "argocd-installation", "application-deployment", "sync-strategies",
        "multi-cluster", "argocd-projects", "automated-rollbacks", "progressive-delivery",
        "flux-comparison", "gitops-best-practices"
    )},
    @{num="23"; name="serverless-functions"; labs=@(
        "lambda-functions", "api-gateway", "event-driven-architecture", "step-functions",
        "serverless-framework", "sam-templates", "lambda-layers", "cold-start-optimization",
        "serverless-monitoring", "cost-optimization"
    )},
    @{num="24"; name="advanced-monitoring"; labs=@(
        "distributed-tracing-advanced", "apm-tools", "custom-instrumentation", "slo-sli-sla",
        "error-budgets", "synthetic-monitoring", "real-user-monitoring", "performance-analysis",
        "capacity-planning", "monitoring-at-scale"
    )},
    @{num="25"; name="chaos-engineering"; labs=@(
        "chaos-principles", "chaos-monkey", "failure-injection", "resilience-testing",
        "chaos-experiments", "blast-radius", "steady-state-hypothesis", "automated-chaos",
        "chaos-reporting", "building-resilience"
    )},
    @{num="26"; name="multi-cloud-hybrid"; labs=@(
        "multi-cloud-strategy", "aws-azure-integration", "gcp-services", "cloud-agnostic-tools",
        "hybrid-cloud", "cloud-migration", "multi-cloud-networking", "cost-comparison",
        "vendor-lock-in", "multi-cloud-management"
    )},
    @{num="27"; name="platform-engineering"; labs=@(
        "internal-platforms", "developer-portals", "self-service-infrastructure", "golden-paths",
        "platform-apis", "developer-experience", "platform-metrics", "platform-documentation",
        "platform-adoption", "platform-evolution"
    )},
    @{num="28"; name="cost-optimization"; labs=@(
        "finops-principles", "cost-visibility", "resource-tagging", "rightsizing",
        "reserved-instances", "spot-instances", "cost-allocation", "budget-alerts",
        "waste-elimination", "cost-governance"
    )},
    @{num="29"; name="incident-management"; labs=@(
        "incident-response", "on-call-setup", "pagerduty-integration", "runbooks",
        "post-mortems", "incident-communication", "escalation-policies", "incident-metrics",
        "blameless-culture", "continuous-improvement"
    )},
    @{num="30"; name="production-deployment"; labs=@(
        "blue-green-deployment", "canary-releases", "feature-flags", "progressive-rollout",
        "rollback-strategies", "deployment-verification", "smoke-testing", "deployment-automation",
        "zero-downtime", "deployment-best-practices"
    )}
)

function Create-ModuleStructure {
    param(
        [string]$phase,
        [array]$modules
    )
    
    $phasePath = Join-Path $basePath $phase
    
    foreach ($module in $modules) {
        $moduleName = "module-$($module.num)-$($module.name)"
        $modulePath = Join-Path $phasePath $moduleName
        $labsPath = Join-Path $modulePath "labs"
        
        # Create directories
        New-Item -ItemType Directory -Force -Path $modulePath | Out-Null
        New-Item -ItemType Directory -Force -Path $labsPath | Out-Null
        
        Write-Host "Created: $moduleName"
        
        # Create lab files
        for ($i = 0; $i -lt $module.labs.Count; $i++) {
            $labNum = $i + 1
            $labName = $module.labs[$i]
            $labFile = "lab-$($module.num).$labNum-$labName.md"
            $labPath = Join-Path $labsPath $labFile
            
            # Create placeholder lab file
            $labContent = @"
# Lab $($module.num).$labNum: $($labName -replace '-', ' ' | ForEach-Object {(Get-Culture).TextInfo.ToTitleCase($_)})

## Objective
[Lab objective will be added]

## Instructions

### Step 1: [Step Title]
[Instructions]

### Step 2: [Step Title]
[Instructions]

## Challenges

### Challenge 1: [Challenge Title]
[Challenge description]

## Solution

<details>
<summary>Click to reveal solution</summary>

[Solution content]

</details>

## Success Criteria
✅ [Criterion 1]
✅ [Criterion 2]

## Key Learnings
- [Learning 1]
- [Learning 2]

## Next Steps
Proceed to **Lab $($module.num).$($labNum + 1)** or the next module.
"@
            
            Set-Content -Path $labPath -Value $labContent -Encoding UTF8
        }
    }
}

# Create all modules
Write-Host "Creating Phase 1 modules..." -ForegroundColor Green
Create-ModuleStructure -phase "phase1_beginner" -modules $phase1Modules

Write-Host "`nCreating Phase 2 modules..." -ForegroundColor Yellow
Create-ModuleStructure -phase "phase2_intermediate" -modules $phase2Modules

Write-Host "`nCreating Phase 3 modules..." -ForegroundColor Red
Create-ModuleStructure -phase "phase3_advanced" -modules $phase3Modules

Write-Host "`n✅ Course structure created successfully!" -ForegroundColor Green
Write-Host "Total modules: 30"
Write-Host "Total labs: 300"
