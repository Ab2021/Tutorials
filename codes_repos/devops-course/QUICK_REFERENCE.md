# DevOps Quick Reference Guide

## 🐳 Docker Commands

### Container Management
```bash
# Run container
docker run -d --name myapp -p 8080:80 nginx

# List containers
docker ps                    # Running containers
docker ps -a                 # All containers

# Stop/Start/Restart
docker stop <container-id>
docker start <container-id>
docker restart <container-id>

# Remove container
docker rm <container-id>
docker rm -f <container-id>  # Force remove

# View logs
docker logs <container-id>
docker logs -f <container-id>  # Follow logs

# Execute commands in container
docker exec -it <container-id> /bin/bash
docker exec <container-id> ls /app
```

### Image Management
```bash
# Build image
docker build -t myapp:v1.0 .
docker build -t myapp:v1.0 -f Dockerfile.prod .

# List images
docker images
docker images -a

# Pull/Push images
docker pull nginx:latest
docker push myregistry/myapp:v1.0

# Remove images
docker rmi <image-id>
docker rmi $(docker images -q)  # Remove all

# Tag image
docker tag myapp:v1.0 myregistry/myapp:v1.0
```

### Docker Compose
```bash
# Start services
docker-compose up
docker-compose up -d         # Detached mode

# Stop services
docker-compose down
docker-compose down -v       # Remove volumes

# View logs
docker-compose logs
docker-compose logs -f service-name

# Scale services
docker-compose up -d --scale web=3
```

### Cleanup
```bash
# Remove unused data
docker system prune
docker system prune -a       # Remove all unused images

# Remove volumes
docker volume prune

# Remove networks
docker network prune
```

---

## ☸️ Kubernetes (kubectl) Commands

### Cluster Info
```bash
# Cluster information
kubectl cluster-info
kubectl get nodes
kubectl describe node <node-name>
```

### Pod Management
```bash
# List pods
kubectl get pods
kubectl get pods -n <namespace>
kubectl get pods --all-namespaces

# Describe pod
kubectl describe pod <pod-name>

# Create pod
kubectl run nginx --image=nginx

# Delete pod
kubectl delete pod <pod-name>

# View logs
kubectl logs <pod-name>
kubectl logs -f <pod-name>    # Follow logs
kubectl logs <pod-name> -c <container-name>

# Execute commands
kubectl exec -it <pod-name> -- /bin/bash
```

### Deployment Management
```bash
# Create deployment
kubectl create deployment nginx --image=nginx
kubectl apply -f deployment.yaml

# List deployments
kubectl get deployments

# Scale deployment
kubectl scale deployment nginx --replicas=3

# Update deployment
kubectl set image deployment/nginx nginx=nginx:1.21

# Rollout status
kubectl rollout status deployment/nginx
kubectl rollout history deployment/nginx
kubectl rollout undo deployment/nginx

# Delete deployment
kubectl delete deployment <deployment-name>
```

### Service Management
```bash
# Create service
kubectl expose deployment nginx --port=80 --type=LoadBalancer
kubectl apply -f service.yaml

# List services
kubectl get services
kubectl get svc

# Describe service
kubectl describe service <service-name>
```

### ConfigMap & Secrets
```bash
# Create ConfigMap
kubectl create configmap app-config --from-file=config.properties
kubectl create configmap app-config --from-literal=key=value

# Create Secret
kubectl create secret generic db-secret --from-literal=password=mypass
kubectl create secret docker-registry regcred --docker-server=<server> --docker-username=<user> --docker-password=<pass>

# View
kubectl get configmaps
kubectl get secrets
kubectl describe configmap <name>
```

### Namespace Management
```bash
# List namespaces
kubectl get namespaces

# Create namespace
kubectl create namespace dev

# Set default namespace
kubectl config set-context --current --namespace=dev
```

### Debugging
```bash
# Get events
kubectl get events --sort-by=.metadata.creationTimestamp

# Port forwarding
kubectl port-forward pod/<pod-name> 8080:80

# Copy files
kubectl cp <pod-name>:/path/to/file ./local-file
kubectl cp ./local-file <pod-name>:/path/to/file
```

---

## 🔧 Terraform Commands

### Initialization & Planning
```bash
# Initialize
terraform init

# Validate configuration
terraform validate

# Format code
terraform fmt
terraform fmt -recursive

# Plan changes
terraform plan
terraform plan -out=tfplan
```

### Apply & Destroy
```bash
# Apply changes
terraform apply
terraform apply -auto-approve
terraform apply tfplan

# Destroy infrastructure
terraform destroy
terraform destroy -auto-approve
terraform destroy -target=aws_instance.example
```

### State Management
```bash
# List resources
terraform state list

# Show resource
terraform state show aws_instance.example

# Move resource
terraform state mv aws_instance.old aws_instance.new

# Remove resource
terraform state rm aws_instance.example

# Pull state
terraform state pull
```

### Workspace Management
```bash
# List workspaces
terraform workspace list

# Create workspace
terraform workspace new dev

# Select workspace
terraform workspace select dev

# Delete workspace
terraform workspace delete dev
```

### Output & Variables
```bash
# Show outputs
terraform output
terraform output instance_ip

# Set variables
terraform apply -var="instance_type=t2.micro"
terraform apply -var-file="prod.tfvars"
```

---

## 📦 Ansible Commands

### Ad-hoc Commands
```bash
# Ping all hosts
ansible all -m ping

# Run command
ansible all -a "uptime"
ansible webservers -a "systemctl status nginx"

# Copy file
ansible all -m copy -a "src=/local/file dest=/remote/file"

# Install package
ansible all -m apt -a "name=nginx state=present" --become
```

### Playbook Management
```bash
# Run playbook
ansible-playbook playbook.yml

# Check syntax
ansible-playbook playbook.yml --syntax-check

# Dry run
ansible-playbook playbook.yml --check

# Limit to hosts
ansible-playbook playbook.yml --limit webservers

# Use specific inventory
ansible-playbook -i inventory.ini playbook.yml

# Verbose output
ansible-playbook playbook.yml -v
ansible-playbook playbook.yml -vvv  # More verbose
```

### Inventory Management
```bash
# List hosts
ansible all --list-hosts
ansible webservers --list-hosts

# Show inventory
ansible-inventory --list
ansible-inventory --graph
```

### Ansible Galaxy
```bash
# Install role
ansible-galaxy install geerlingguy.nginx

# List installed roles
ansible-galaxy list

# Create role structure
ansible-galaxy init my-role
```

---

## 🌩️ AWS CLI Commands

### EC2
```bash
# List instances
aws ec2 describe-instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]' --output table

# Start/Stop instances
aws ec2 start-instances --instance-ids i-1234567890abcdef0
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Create key pair
aws ec2 create-key-pair --key-name MyKeyPair --query 'KeyMaterial' --output text > MyKeyPair.pem
```

### S3
```bash
# List buckets
aws s3 ls

# List bucket contents
aws s3 ls s3://my-bucket

# Copy files
aws s3 cp file.txt s3://my-bucket/
aws s3 cp s3://my-bucket/file.txt ./
aws s3 sync ./local-dir s3://my-bucket/

# Create bucket
aws s3 mb s3://my-new-bucket

# Remove bucket
aws s3 rb s3://my-bucket --force
```

### IAM
```bash
# List users
aws iam list-users

# Create user
aws iam create-user --user-name john

# List roles
aws iam list-roles

# Attach policy
aws iam attach-user-policy --user-name john --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess
```

### CloudFormation
```bash
# Create stack
aws cloudformation create-stack --stack-name my-stack --template-body file://template.yaml

# List stacks
aws cloudformation list-stacks

# Describe stack
aws cloudformation describe-stacks --stack-name my-stack

# Delete stack
aws cloudformation delete-stack --stack-name my-stack
```

---

## 🔍 Git Commands

### Basic Operations
```bash
# Initialize repository
git init

# Clone repository
git clone <url>

# Check status
git status

# Add files
git add file.txt
git add .

# Commit changes
git commit -m "Commit message"
git commit -am "Add and commit"

# Push changes
git push origin main
git push -u origin feature-branch

# Pull changes
git pull origin main
```

### Branching
```bash
# List branches
git branch
git branch -a        # All branches

# Create branch
git branch feature-x

# Switch branch
git checkout feature-x
git switch feature-x  # Modern syntax

# Create and switch
git checkout -b feature-x
git switch -c feature-x

# Delete branch
git branch -d feature-x
git branch -D feature-x  # Force delete

# Merge branch
git merge feature-x
```

### History & Logs
```bash
# View log
git log
git log --oneline
git log --graph --oneline --all

# Show changes
git diff
git diff --staged
git diff branch1..branch2

# Show commit
git show <commit-hash>
```

### Undo Changes
```bash
# Discard changes
git checkout -- file.txt
git restore file.txt

# Unstage file
git reset HEAD file.txt
git restore --staged file.txt

# Undo commit
git reset --soft HEAD~1   # Keep changes
git reset --hard HEAD~1   # Discard changes

# Revert commit
git revert <commit-hash>
```

---

## 📊 Monitoring Commands

### Prometheus Queries (PromQL)
```promql
# CPU usage
rate(node_cpu_seconds_total{mode="idle"}[5m])

# Memory usage
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100

# HTTP request rate
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Linux System Monitoring
```bash
# CPU and memory
top
htop

# Disk usage
df -h
du -sh /path/to/dir

# Network
netstat -tulpn
ss -tulpn

# Processes
ps aux
ps aux | grep nginx

# System logs
journalctl -u nginx
journalctl -f
tail -f /var/log/syslog
```

---

## 🔐 Security Commands

### SSL/TLS
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Check certificate
openssl x509 -in cert.pem -text -noout

# Test SSL connection
openssl s_client -connect example.com:443
```

### SSH
```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy SSH key to server
ssh-copy-id user@server

# SSH with specific key
ssh -i ~/.ssh/id_rsa user@server
```

---

## 💡 Quick Tips

### One-Liners
```bash
# Find and kill process by port
lsof -ti:8080 | xargs kill -9

# Watch command output
watch -n 2 kubectl get pods

# JSON formatting
echo '{"key":"value"}' | jq .

# Find large files
find / -type f -size +100M 2>/dev/null

# Check port connectivity
nc -zv hostname 80
telnet hostname 80
```

---

**Keep this guide handy for quick reference!** 📖
