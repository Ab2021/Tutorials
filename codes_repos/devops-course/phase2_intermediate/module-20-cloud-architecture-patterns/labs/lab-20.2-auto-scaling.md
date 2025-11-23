# Lab 20.2: Auto Scaling

## Objective
Implement auto-scaling for applications.

## Learning Objectives
- Configure Auto Scaling Groups
- Set scaling policies
- Use target tracking
- Monitor scaling activities

---

## Create Launch Template

```bash
aws ec2 create-launch-template \
  --launch-template-name my-template \
  --version-description v1 \
  --launch-template-data '{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "t2.micro",
    "SecurityGroupIds": ["sg-12345"]
  }'
```

## Create Auto Scaling Group

```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --launch-template LaunchTemplateName=my-template \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3 \
  --vpc-zone-identifier "subnet-1,subnet-2" \
  --target-group-arns arn:aws:elasticloadbalancing:...
```

## Scaling Policies

```bash
# Target tracking - CPU
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name my-asg \
  --policy-name cpu-target-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    },
    "TargetValue": 70.0
  }'

# Step scaling
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name my-asg \
  --policy-name scale-out \
  --policy-type StepScaling \
  --adjustment-type PercentChangeInCapacity \
  --step-adjustments '[
    {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 50}
  ]'
```

## Success Criteria
✅ ASG created  
✅ Scaling policies configured  
✅ Auto-scaling working  
✅ Metrics monitored  

**Time:** 40 min
