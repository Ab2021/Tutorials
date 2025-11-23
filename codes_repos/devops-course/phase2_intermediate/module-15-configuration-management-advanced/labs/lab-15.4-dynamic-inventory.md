# Lab 15.4: Dynamic Inventory

## Objective
Use dynamic inventory to automatically discover infrastructure.

## Learning Objectives
- Configure AWS dynamic inventory
- Use inventory plugins
- Filter and group hosts dynamically
- Integrate with cloud providers

---

## AWS Dynamic Inventory

```yaml
# aws_ec2.yml
plugin: aws_ec2
regions:
  - us-east-1
filters:
  tag:Environment: production
keyed_groups:
  - key: tags.Role
    prefix: role
hostnames:
  - tag:Name
```

## Use Dynamic Inventory

```bash
# List hosts
ansible-inventory -i aws_ec2.yml --list

# Run playbook
ansible-playbook -i aws_ec2.yml playbook.yml
```

## Custom Inventory Script

```python
#!/usr/bin/env python3
import json

inventory = {
    "webservers": {
        "hosts": ["web1.example.com", "web2.example.com"]
    },
    "_meta": {
        "hostvars": {
            "web1.example.com": {"ansible_host": "10.0.1.10"}
        }
    }
}

print(json.dumps(inventory))
```

## Success Criteria
✅ AWS dynamic inventory working  
✅ Hosts auto-discovered  
✅ Custom inventory script created  
✅ Groups configured correctly  

**Time:** 40 min
