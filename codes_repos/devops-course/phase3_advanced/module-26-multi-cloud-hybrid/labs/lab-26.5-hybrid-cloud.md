# Lab 26.5: Hybrid Cloud

## Objective
Implement hybrid cloud architecture connecting on-prem and cloud.

## Learning Objectives
- Set up VPN connections
- Configure hybrid networking
- Implement data sync
- Manage hybrid workloads

---

## AWS Site-to-Site VPN

```bash
# Create customer gateway
aws ec2 create-customer-gateway \
  --type ipsec.1 \
  --public-ip 203.0.113.12 \
  --bgp-asn 65000

# Create VPN gateway
aws ec2 create-vpn-gateway --type ipsec.1

# Create VPN connection
aws ec2 create-vpn-connection \
  --type ipsec.1 \
  --customer-gateway-id cgw-12345 \
  --vpn-gateway-id vgw-67890
```

## Hybrid Kubernetes

```yaml
# On-prem node joins cloud cluster
apiVersion: v1
kind: Node
metadata:
  name: onprem-node-1
  labels:
    location: onprem
spec:
  taints:
  - key: location
    value: onprem
    effect: NoSchedule
```

## Data Sync

```bash
# AWS DataSync
aws datasync create-location-nfs \
  --server-hostname onprem-nas.local \
  --subdirectory /data

aws datasync create-task \
  --source-location-arn arn:aws:datasync:... \
  --destination-location-arn arn:aws:datasync:...
```

## Success Criteria
✅ VPN connection established  
✅ Hybrid networking working  
✅ Data syncing between environments  
✅ Workloads distributed  

**Time:** 50 min
