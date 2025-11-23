# Lab 10.9: Load Balancers

## Objective
Configure Application and Network Load Balancers.

## Learning Objectives
- Create Application Load Balancer
- Configure target groups
- Set up health checks
- Implement SSL/TLS

---

## Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name my-alb \
  --subnets subnet-1 subnet-2 \
  --security-groups sg-12345

# Create target group
aws elbv2 create-target-group \
  --name my-targets \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-12345 \
  --health-check-path /health

# Register targets
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=i-1234567890 Id=i-0987654321

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:...
```

## SSL/TLS

```bash
# HTTPS listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:... \
  --default-actions Type=forward,TargetGroupArn=arn:aws:...
```

## Success Criteria
✅ ALB created  
✅ Targets registered  
✅ Health checks working  
✅ SSL configured  

**Time:** 45 min
