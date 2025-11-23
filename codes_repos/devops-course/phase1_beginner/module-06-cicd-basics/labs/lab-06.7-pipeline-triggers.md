# Lab 06.7: Pipeline Triggers

## Objective
Configure different pipeline triggers for various scenarios.

## Learning Objectives
- Trigger on push/PR
- Schedule pipelines
- Manual triggers
- Webhook triggers

---

## Push Triggers

```yaml
on:
  push:
    branches: [main, develop, 'release/**']
    paths:
      - 'src/**'
      - 'Dockerfile'
    tags:
      - 'v*'
```

## Pull Request Triggers

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [main]
```

## Scheduled Triggers

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
    - cron: '0 0 * * 0'  # Weekly on Sunday
```

## Manual Triggers

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - staging
          - production
      version:
        description: 'Version to deploy'
        required: true
```

## Workflow Call

```yaml
on:
  workflow_call:
    inputs:
      config-path:
        required: true
        type: string
```

## Success Criteria
✅ Push triggers working  
✅ Scheduled jobs running  
✅ Manual dispatch configured  

**Time:** 30 min
