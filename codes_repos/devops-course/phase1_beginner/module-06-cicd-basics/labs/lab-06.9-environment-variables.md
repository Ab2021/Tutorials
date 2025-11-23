# Lab 06.9: Environment Variables and Secrets

## Objective
Manage environment variables and secrets securely in CI/CD.

## Learning Objectives
- Use environment variables
- Store secrets securely
- Use GitHub Secrets
- Manage different environments

---

## Environment Variables

```yaml
env:
  NODE_ENV: production
  API_URL: https://api.example.com

jobs:
  build:
    env:
      BUILD_ENV: staging
    steps:
      - run: echo "NODE_ENV is $NODE_ENV"
      - run: echo "BUILD_ENV is $BUILD_ENV"
```

## GitHub Secrets

```yaml
jobs:
  deploy:
    steps:
      - name: Deploy
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: ./deploy.sh
```

## Environment-Specific Secrets

```yaml
jobs:
  deploy-staging:
    environment: staging
    steps:
      - run: echo "Deploying to staging"
        env:
          API_KEY: ${{ secrets.STAGING_API_KEY }}
  
  deploy-production:
    environment: production
    steps:
      - run: echo "Deploying to production"
        env:
          API_KEY: ${{ secrets.PRODUCTION_API_KEY }}
```

## Vault Integration

```yaml
      - name: Import Secrets
        uses: hashicorp/vault-action@v2
        with:
          url: https://vault.example.com
          token: ${{ secrets.VAULT_TOKEN }}
          secrets: |
            secret/data/production database_url | DATABASE_URL ;
            secret/data/production api_key | API_KEY
```

## Success Criteria
✅ Environment variables configured  
✅ Secrets stored securely  
✅ Different env configs working  

**Time:** 35 min
