# Day 2.5: Drift Detection and Automated Compliance Scanning

## 🎯 Learning Objectives
By the end of this section, you will understand:
- Configuration drift detection principles and implementation
- Automated compliance scanning frameworks and tools
- Continuous security posture monitoring for AI/ML infrastructure
- Remediation strategies and automated response systems
- Integration with governance and risk management frameworks

---

## 📚 Theoretical Foundation

### 1. Introduction to Configuration Drift and Compliance

#### 1.1 Configuration Drift Fundamentals

**Definition and Impact**:
Configuration drift occurs when the actual state of infrastructure deviates from the intended, documented, or previously established baseline configuration. In AI/ML environments, this phenomenon poses significant security and operational risks due to the complex, distributed nature of machine learning infrastructure.

**Types of Configuration Drift**:
```
Intentional Drift:
- Authorized emergency changes during incidents
- Hotfixes applied directly to production systems
- Manual optimizations for performance tuning
- Temporary configurations for debugging

Unintentional Drift:
- Software updates changing default configurations
- Human error during manual configuration changes
- Automated processes with incorrect parameters
- System failures causing partial configuration loss

Malicious Drift:
- Unauthorized configuration changes by attackers
- Privilege escalation through configuration manipulation
- Backdoor installations via configuration modifications
- Data exfiltration through modified security settings
```

**AI/ML Specific Drift Scenarios**:
```
Training Infrastructure Drift:
- GPU cluster configurations changing during long training runs
- Network security groups modified for debugging and not reverted
- Storage permissions altered for data access optimization
- Container runtime configurations modified for performance

Model Serving Drift:
- Load balancer configurations changing without approval
- API gateway security policies being relaxed
- Database connection settings modified for troubleshooting
- Monitoring and logging configurations being disabled

Data Pipeline Drift:
- Data processing job configurations changing automatically
- ETL pipeline security settings being modified
- Data lake access permissions expanding over time
- Backup and retention policies being altered
```

#### 1.2 Compliance Framework Integration

**Regulatory Compliance Requirements**:
```
SOC 2 Type II Compliance:
- CC6.2: System changes require prior authorization
- CC6.3: Logical access provisions are removed in a timely manner
- CC8.1: Change management process is followed
- CC9.1: Risk management process identifies threats

GDPR Compliance:
- Article 25: Data protection by design and by default
- Article 32: Security of processing requirements
- Article 33: Breach notification within 72 hours
- Article 35: Data protection impact assessments

HIPAA Compliance:
- 164.308(a)(1)(ii)(D): Information access management
- 164.308(a)(5)(ii)(C): Automatic logoff procedures
- 164.312(a)(2)(i): Unique user identification
- 164.312(b): Audit controls and monitoring
```

**Industry Standards Integration**:
```
NIST Cybersecurity Framework:
- Identify: Asset management and risk assessment
- Protect: Access control and data security
- Detect: Anomaly detection and monitoring
- Respond: Response planning and mitigation
- Recover: Recovery planning and improvements

ISO 27001 Controls:
- A.12.1.2: Change management procedures
- A.12.6.1: Management of technical vulnerabilities
- A.16.1.1: Responsibilities and procedures
- A.16.1.2: Reporting information security events
```

### 2. Drift Detection Technologies and Frameworks

#### 2.1 Infrastructure State Monitoring

**Terraform State Drift Detection**:
```terraform
# Terraform configuration for automated drift detection
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Remote state configuration with locking
  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-west-2:123456789012:key/terraform-state"
    dynamodb_table = "terraform-state-lock"
  }
}

# Data source to detect drift in security groups
data "aws_security_group" "ml_training" {
  id = aws_security_group.ml_training.id
}

# Local value to compare current vs expected state
locals {
  expected_ingress_rules = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["10.0.0.0/8"]
    },
    {
      from_port = 8000
      to_port   = 8100
      protocol  = "tcp"
      self      = true
    }
  ]
  
  actual_ingress_rules = data.aws_security_group.ml_training.ingress
  
  # Detect drift in security group rules
  security_group_drift = length(setsubtract(
    local.expected_ingress_rules,
    local.actual_ingress_rules
  )) > 0 || length(setsubtract(
    local.actual_ingress_rules,
    local.expected_ingress_rules
  )) > 0
}

# CloudWatch alarm for drift detection
resource "aws_cloudwatch_metric_alarm" "configuration_drift" {
  alarm_name          = "ml-infrastructure-drift-detected"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ConfigurationDrift"
  namespace           = "ML/Infrastructure"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "Detects configuration drift in ML infrastructure"
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]
  
  tags = {
    Purpose = "drift-detection"
    Service = "ml-infrastructure"
  }
}

# Custom metric for drift detection
resource "aws_cloudwatch_log_metric_filter" "drift_metric" {
  name           = "ml-infrastructure-drift"
  log_group_name = aws_cloudwatch_log_group.drift_detection.name
  pattern        = "[timestamp, request_id, \"DRIFT_DETECTED\", ...]"
  
  metric_transformation {
    name      = "ConfigurationDrift"
    namespace = "ML/Infrastructure"
    value     = "1"
  }
}
```

**AWS Config Rules for Compliance**:
```terraform
# AWS Config configuration recorder
resource "aws_config_configuration_recorder" "ml_config_recorder" {
  name     = "ml-infrastructure-recorder"
  role_arn = aws_iam_role.config_role.arn
  
  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }
}

# Config delivery channel
resource "aws_config_delivery_channel" "ml_config_delivery" {
  name           = "ml-config-delivery-channel"
  s3_bucket_name = aws_s3_bucket.config_bucket.bucket
  s3_key_prefix  = "config"
  
  snapshot_delivery_properties {
    delivery_frequency = "Hourly"
  }
}

# Custom Config rules for ML infrastructure
resource "aws_config_config_rule" "ml_security_groups_restricted" {
  name = "ml-security-groups-no-unrestricted-access"
  
  source {
    owner             = "AWS"
    source_identifier = "INCOMING_SSH_DISABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.ml_config_recorder]
}

resource "aws_config_config_rule" "ml_ebs_encrypted" {
  name = "ml-ebs-volumes-encrypted"
  
  source {
    owner             = "AWS"
    source_identifier = "ENCRYPTED_VOLUMES"
  }
  
  depends_on = [aws_config_configuration_recorder.ml_config_recorder]
}

resource "aws_config_config_rule" "ml_s3_encrypted" {
  name = "ml-s3-buckets-encrypted"
  
  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.ml_config_recorder]
}

# Custom Config rule for ML-specific compliance
resource "aws_config_config_rule" "ml_instance_compliance" {
  name = "ml-instances-compliance-check"
  
  source {
    owner                = "AWS"
    source_identifier    = "LAMBDA"
    source_detail {
      event_source = "aws.config"
      message_type = "ConfigurationItemChangeNotification"
    }
  }
  
  lambda_function_arn = aws_lambda_function.ml_compliance_checker.arn
  
  depends_on = [aws_config_configuration_recorder.ml_config_recorder]
}
```

#### 2.2 Kubernetes Configuration Monitoring

**Polaris Policy Enforcement**:
```yaml
# Polaris configuration for Kubernetes security scanning
apiVersion: v1
kind: ConfigMap
metadata:
  name: polaris-config
  namespace: polaris
data:
  config.yaml: |
    checks:
      # Security checks
      hostIPCSet: warning
      hostNetworkSet: danger
      hostPIDSet: warning
      notReadOnlyRootFilesystem: warning
      privilegeEscalationAllowed: danger
      runAsRootAllowed: warning
      runAsPrivileged: danger
      insecureCapabilities: warning
      dangerousCapabilities: danger
      
      # ML-specific checks
      cpuRequestsMissing: warning
      cpuLimitsMissing: warning
      memoryRequestsMissing: warning
      memoryLimitsMissing: warning
      
      # Custom checks for ML workloads
      gpuResourcesSet: ignore
      
    exemptions:
      - controllerNames:
          - nvidia-device-plugin-daemonset
        rules:
          - runAsPrivileged
          - hostNetwork
      - controllerNames:
          - ml-training-*
        rules:
          - cpuLimitsMissing
          
    controllers_to_scan:
      - Deployments
      - StatefulSets
      - DaemonSets
      - Jobs
      - CronJobs
      - ReplicaSets
```

**OPA Gatekeeper Constraint Violations**:
```yaml
# Gatekeeper constraint for ML workload compliance
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- constraint-template.yaml
- constraint.yaml

# constraint-template.yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: mlworkloadcompliance
spec:
  crd:
    spec:
      names:
        kind: MLWorkloadCompliance
      validation:
        openAPIV3Schema:
          type: object
          properties:
            requiredLabels:
              type: array
              items:
                type: string
            allowedRepositories:
              type: array
              items:
                type: string
            maxResourceLimits:
              type: object
              properties:
                cpu:
                  type: string
                memory:
                  type: string
                nvidia.com/gpu:
                  type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package mlworkloadcompliance
        
        violation[{"msg": msg}] {
          required := input.parameters.requiredLabels
          provided := input.review.object.metadata.labels
          missing := required[_]
          not provided[missing]
          msg := sprintf("Missing required label: %v", [missing])
        }
        
        violation[{"msg": msg}] {
          image := input.review.object.spec.containers[_].image
          allowed := input.parameters.allowedRepositories
          not startswith(image, allowed[_])
          msg := sprintf("Container image %v not from approved repository", [image])
        }
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          limits := container.resources.limits
          max_limits := input.parameters.maxResourceLimits
          
          resource_over_limit(limits.cpu, max_limits.cpu)
          msg := sprintf("CPU limit %v exceeds maximum %v", [limits.cpu, max_limits.cpu])
        }
        
        resource_over_limit(current, max) {
          current_val := units.parse_bytes(current)
          max_val := units.parse_bytes(max)
          current_val > max_val
        }

---
# constraint.yaml
apiVersion: gatekeeper.sh/v1beta1
kind: MLWorkloadCompliance
metadata:
  name: ml-workload-compliance
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
      - apiGroups: ["apps"]
        kinds: ["Deployment", "StatefulSet"]
    namespaces: ["ml-training", "ml-inference"]
  parameters:
    requiredLabels: ["app", "version", "environment", "team"]
    allowedRepositories: ["registry.company.com/ml/", "gcr.io/ml-project/"]
    maxResourceLimits:
      cpu: "16"
      memory: "64Gi"
      nvidia.com/gpu: "8"
```

#### 2.3 Cloud-Native Compliance Scanning

**Falco Runtime Security Monitoring**:
```yaml
# Falco configuration for ML workload security
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-config
  namespace: falco
data:
  falco.yaml: |
    rules_file:
      - /etc/falco/falco_rules.yaml
      - /etc/falco/ml_custom_rules.yaml
      
    time_format_iso_8601: true
    json_output: true
    json_include_output_property: true
    
    # Output channels
    file_output:
      enabled: true
      keep_alive: false
      filename: /var/log/falco/falco_events.log
      
    stdout_output:
      enabled: true
      
    syslog_output:
      enabled: true
      
    # HTTP output for integration
    http_output:
      enabled: true
      url: "http://falco-sidekick.falco.svc.cluster.local:2801"
      
    # gRPC output for advanced integrations
    grpc_output:
      enabled: true
      bind_address: "0.0.0.0:5060"
      threadiness: 8
      
    # Priority levels
    priority: debug
    
    # Syscall event drops
    syscall_event_drops:
      threshold: 0.03
      actions:
        - log
        - alert
        
  ml_custom_rules.yaml: |
    # Custom rules for ML workloads
    
    - rule: ML Training Data Access
      desc: Detect unauthorized access to ML training data
      condition: >
        open_read and 
        fd.name startswith "/data/training/" and
        not k8s_ml_training_pod
      output: >
        Unauthorized ML training data access 
        (user=%user.name command=%proc.cmdline file=%fd.name 
        container=%container.name pod=%k8s.pod.name)
      priority: ERROR
      tags: [ml, data-access, security]
      
    - rule: ML Model Modification
      desc: Detect unauthorized modifications to ML models
      condition: >
        open_write and 
        fd.name startswith "/models/" and
        not k8s_ml_serving_pod
      output: >
        Unauthorized ML model modification 
        (user=%user.name command=%proc.cmdline file=%fd.name 
        container=%container.name pod=%k8s.pod.name)
      priority: CRITICAL
      tags: [ml, model-security, security]
      
    - rule: GPU Resource Access
      desc: Detect unauthorized GPU access attempts
      condition: >
        open and 
        fd.name startswith "/dev/nvidia" and
        not k8s_gpu_workload
      output: >
        Unauthorized GPU resource access 
        (user=%user.name command=%proc.cmdline device=%fd.name 
        container=%container.name pod=%k8s.pod.name)
      priority: WARNING
      tags: [ml, gpu-access, resource]
      
    - macro: k8s_ml_training_pod
      condition: >
        ka and k8s.pod.label.app startswith "ml-training"
        
    - macro: k8s_ml_serving_pod
      condition: >
        ka and k8s.pod.label.app startswith "ml-serving"
        
    - macro: k8s_gpu_workload
      condition: >
        ka and k8s.pod.label.resource-type = "gpu"
```

### 3. Automated Compliance Frameworks

#### 3.1 Infrastructure Compliance Automation

**Chef InSpec for Infrastructure Testing**:
```ruby
# InSpec profile for ML infrastructure compliance
# ml_infrastructure_profile/controls/security.rb

control 'ml-01' do
  title 'ML Training instances should not have public IP addresses'
  desc 'Ensure ML training instances are not directly accessible from internet'
  impact 0.8
  
  describe aws_ec2_instances.where { tags.any? { |tag| tag.key == 'Purpose' && tag.value == 'ml-training' } } do
    its('public_ip_addresses') { should all(be_nil) }
  end
end

control 'ml-02' do
  title 'ML data S3 buckets should be encrypted'
  desc 'Ensure all S3 buckets containing ML data use encryption'
  impact 1.0
  
  aws_s3_buckets.bucket_names.each do |bucket|
    next unless bucket.include?('ml-data') || bucket.include?('training-data')
    
    describe aws_s3_bucket(bucket) do
      it { should have_server_side_encryption }
    end
  end
end

control 'ml-03' do
  title 'ML infrastructure security groups should not allow unrestricted access'
  desc 'Security groups should not allow 0.0.0.0/0 access except for specific services'
  impact 0.9
  
  aws_security_groups.group_ids.each do |sg_id|
    sg = aws_security_group(sg_id)
    next unless sg.tags.any? { |tag| tag.key == 'Purpose' && tag.value.include?('ml') }
    
    describe sg do
      its('inbound_rules') { should_not include(have_attributes(source: '0.0.0.0/0', port_range: '22..22')) }
      its('inbound_rules') { should_not include(have_attributes(source: '0.0.0.0/0', port_range: '3389..3389')) }
    end
  end
end

control 'ml-04' do
  title 'ML databases should have backup retention configured'
  desc 'RDS instances used for ML should have appropriate backup retention'
  impact 0.7
  
  aws_rds_instances.db_instance_identifiers.each do |db_id|
    instance = aws_rds_instance(db_id)
    next unless instance.tags.any? { |tag| tag.key == 'Purpose' && tag.value.include?('ml') }
    
    describe instance do
      its('backup_retention_period') { should be >= 7 }
      its('storage_encrypted') { should be true }
    end
  end
end

control 'ml-05' do
  title 'ML workload containers should run as non-root'
  desc 'Kubernetes pods for ML workloads should not run as root user'
  impact 0.8
  
  k8s_pods(namespace: 'ml-training').each do |pod|
    describe pod do
      its('security_context.run_as_user') { should_not eq 0 }
      its('security_context.run_as_non_root') { should be true }
    end
  end
end
```

**Ansible for Compliance Remediation**:
```yaml
# ansible-playbook ml_compliance_remediation.yml
---
- name: ML Infrastructure Compliance Remediation
  hosts: ml_infrastructure
  become: yes
  vars:
    compliance_baseline:
      ssh_max_auth_tries: 3
      password_max_age: 90
      audit_log_retention: 365
      firewall_default_policy: deny
    
  tasks:
    - name: Ensure SSH configuration is compliant
      lineinfile:
        path: /etc/ssh/sshd_config
        regexp: "{{ item.regexp }}"
        line: "{{ item.line }}"
        state: present
        backup: yes
      loop:
        - { regexp: '^MaxAuthTries', line: 'MaxAuthTries {{ compliance_baseline.ssh_max_auth_tries }}' }
        - { regexp: '^PermitRootLogin', line: 'PermitRootLogin no' }
        - { regexp: '^PasswordAuthentication', line: 'PasswordAuthentication no' }
        - { regexp: '^PubkeyAuthentication', line: 'PubkeyAuthentication yes' }
      notify: restart sshd
      
    - name: Configure system audit logging
      template:
        src: audit.rules.j2
        dest: /etc/audit/rules.d/ml_security.rules
        owner: root
        group: root
        mode: '0640'
      notify: restart auditd
      
    - name: Ensure fail2ban is configured for ML services
      template:
        src: fail2ban_ml.conf.j2
        dest: /etc/fail2ban/jail.d/ml_services.conf
        owner: root
        group: root
        mode: '0644'
      notify: restart fail2ban
      
    - name: Configure system file permissions
      file:
        path: "{{ item.path }}"
        mode: "{{ item.mode }}"
        owner: "{{ item.owner | default('root') }}"
        group: "{{ item.group | default('root') }}"
      loop:
        - { path: '/etc/passwd', mode: '0644' }
        - { path: '/etc/shadow', mode: '0640' }
        - { path: '/etc/group', mode: '0644' }
        - { path: '/etc/gshadow', mode: '0640' }
        
    - name: Install and configure CIS benchmark tools
      package:
        name: "{{ item }}"
        state: present
      loop:
        - lynis
        - rkhunter
        - chkrootkit
        
    - name: Run CIS benchmark assessment
      command: lynis audit system --quick
      register: lynis_result
      changed_when: false
      
    - name: Generate compliance report
      template:
        src: compliance_report.html.j2
        dest: /var/log/compliance_report_{{ ansible_date_time.epoch }}.html
        owner: root
        group: root
        mode: '0644'
      vars:
        lynis_output: "{{ lynis_result.stdout }}"
        
  handlers:
    - name: restart sshd
      service:
        name: sshd
        state: restarted
        
    - name: restart auditd
      service:
        name: auditd
        state: restarted
        
    - name: restart fail2ban
      service:
        name: fail2ban
        state: restarted
```

#### 3.2 Continuous Compliance Monitoring

**Security Compliance Automation Pipeline**:
```python
#!/usr/bin/env python3
# compliance_monitor.py

import boto3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
from dataclasses import dataclass

@dataclass
class ComplianceViolation:
    resource_id: str
    resource_type: str
    violation_type: str
    severity: str
    description: str
    remediation_steps: List[str]
    timestamp: datetime

class ComplianceMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')
        self.rds = boto3.client('rds')
        self.config_client = boto3.client('config')
        self.sns = boto3.client('sns')
        
        # Compliance rules configuration
        self.compliance_rules = self._load_compliance_rules()
        
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from configuration"""
        return {
            'ec2_security': {
                'no_public_ip': {
                    'severity': 'HIGH',
                    'description': 'EC2 instances should not have public IP addresses',
                    'remediation': [
                        'Remove public IP assignment',
                        'Use NAT Gateway for outbound access',
                        'Implement VPN for administrative access'
                    ]
                },
                'security_group_restrictions': {
                    'severity': 'CRITICAL',
                    'description': 'Security groups should not allow unrestricted access',
                    'remediation': [
                        'Review and restrict security group rules',
                        'Implement least privilege access',
                        'Use specific IP ranges instead of 0.0.0.0/0'
                    ]
                }
            },
            's3_security': {
                'encryption_required': {
                    'severity': 'HIGH',
                    'description': 'S3 buckets must have encryption enabled',
                    'remediation': [
                        'Enable S3 bucket encryption',
                        'Use customer-managed KMS keys',
                        'Configure bucket policies for encryption enforcement'
                    ]
                },
                'public_access_blocked': {
                    'severity': 'CRITICAL',
                    'description': 'S3 buckets should block public access',
                    'remediation': [
                        'Enable S3 Block Public Access settings',
                        'Review and update bucket policies',
                        'Implement access logging and monitoring'
                    ]
                }
            }
        }
    
    async def scan_ec2_compliance(self) -> List[ComplianceViolation]:
        """Scan EC2 instances for compliance violations"""
        violations = []
        
        try:
            # Get all EC2 instances with ML-related tags
            response = self.ec2.describe_instances(
                Filters=[
                    {
                        'Name': 'tag:Purpose',
                        'Values': ['ml-training', 'ml-inference', 'ml-development']
                    },
                    {
                        'Name': 'instance-state-name',
                        'Values': ['running', 'stopped']
                    }
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    
                    # Check for public IP address
                    if instance.get('PublicIpAddress'):
                        violations.append(ComplianceViolation(
                            resource_id=instance_id,
                            resource_type='EC2 Instance',
                            violation_type='public_ip_assigned',
                            severity='HIGH',
                            description=f'Instance {instance_id} has public IP address',
                            remediation_steps=self.compliance_rules['ec2_security']['no_public_ip']['remediation'],
                            timestamp=datetime.utcnow()
                        ))
                    
                    # Check security groups
                    for sg in instance['SecurityGroups']:
                        sg_violations = await self._check_security_group_compliance(sg['GroupId'])
                        violations.extend(sg_violations)
            
        except Exception as e:
            self.logger.error(f"Error scanning EC2 compliance: {str(e)}")
        
        return violations
    
    async def _check_security_group_compliance(self, sg_id: str) -> List[ComplianceViolation]:
        """Check security group for compliance violations"""
        violations = []
        
        try:
            response = self.ec2.describe_security_groups(GroupIds=[sg_id])
            sg = response['SecurityGroups'][0]
            
            # Check inbound rules for unrestricted access
            for rule in sg['IpPermissions']:
                for ip_range in rule.get('IpRanges', []):
                    if ip_range.get('CidrIp') == '0.0.0.0/0':
                        # Check if it's SSH or RDP
                        if (rule.get('FromPort', 0) <= 22 <= rule.get('ToPort', 65535) or
                            rule.get('FromPort', 0) <= 3389 <= rule.get('ToPort', 65535)):
                            
                            violations.append(ComplianceViolation(
                                resource_id=sg_id,
                                resource_type='Security Group',
                                violation_type='unrestricted_admin_access',
                                severity='CRITICAL',
                                description=f'Security group {sg_id} allows unrestricted administrative access',
                                remediation_steps=self.compliance_rules['ec2_security']['security_group_restrictions']['remediation'],
                                timestamp=datetime.utcnow()
                            ))
            
        except Exception as e:
            self.logger.error(f"Error checking security group {sg_id}: {str(e)}")
        
        return violations
    
    async def scan_s3_compliance(self) -> List[ComplianceViolation]:
        """Scan S3 buckets for compliance violations"""
        violations = []
        
        try:
            response = self.s3.list_buckets()
            
            for bucket in response['Buckets']:
                bucket_name = bucket['Name']
                
                # Only check ML-related buckets
                if not any(keyword in bucket_name.lower() for keyword in ['ml', 'training', 'model', 'dataset']):
                    continue
                
                # Check encryption
                try:
                    self.s3.get_bucket_encryption(Bucket=bucket_name)
                except self.s3.exceptions.NoSuchEncryption:
                    violations.append(ComplianceViolation(
                        resource_id=bucket_name,
                        resource_type='S3 Bucket',
                        violation_type='encryption_disabled',
                        severity='HIGH',
                        description=f'S3 bucket {bucket_name} does not have encryption enabled',
                        remediation_steps=self.compliance_rules['s3_security']['encryption_required']['remediation'],
                        timestamp=datetime.utcnow()
                    ))
                
                # Check public access
                try:
                    public_access = self.s3.get_public_access_block(Bucket=bucket_name)
                    config = public_access['PublicAccessBlockConfiguration']
                    
                    if not all([
                        config.get('BlockPublicAcls', False),
                        config.get('IgnorePublicAcls', False),
                        config.get('BlockPublicPolicy', False),
                        config.get('RestrictPublicBuckets', False)
                    ]):
                        violations.append(ComplianceViolation(
                            resource_id=bucket_name,
                            resource_type='S3 Bucket',
                            violation_type='public_access_allowed',
                            severity='CRITICAL',
                            description=f'S3 bucket {bucket_name} allows public access',
                            remediation_steps=self.compliance_rules['s3_security']['public_access_blocked']['remediation'],
                            timestamp=datetime.utcnow()
                        ))
                        
                except self.s3.exceptions.NoSuchPublicAccessBlockConfiguration:
                    violations.append(ComplianceViolation(
                        resource_id=bucket_name,
                        resource_type='S3 Bucket',
                        violation_type='public_access_block_missing',
                        severity='CRITICAL',
                        description=f'S3 bucket {bucket_name} has no public access block configuration',
                        remediation_steps=self.compliance_rules['s3_security']['public_access_blocked']['remediation'],
                        timestamp=datetime.utcnow()
                    ))
                
        except Exception as e:
            self.logger.error(f"Error scanning S3 compliance: {str(e)}")
        
        return violations
    
    async def scan_rds_compliance(self) -> List[ComplianceViolation]:
        """Scan RDS instances for compliance violations"""
        violations = []
        
        try:
            response = self.rds.describe_db_instances()
            
            for db_instance in response['DBInstances']:
                db_id = db_instance['DBInstanceIdentifier']
                
                # Check if it's ML-related
                tags = self.rds.list_tags_for_resource(
                    ResourceName=db_instance['DBInstanceArn']
                )['TagList']
                
                is_ml_related = any(
                    tag['Key'] == 'Purpose' and 'ml' in tag['Value'].lower()
                    for tag in tags
                )
                
                if not is_ml_related:
                    continue
                
                # Check encryption
                if not db_instance.get('StorageEncrypted', False):
                    violations.append(ComplianceViolation(
                        resource_id=db_id,
                        resource_type='RDS Instance',
                        violation_type='storage_not_encrypted',
                        severity='HIGH',
                        description=f'RDS instance {db_id} does not have encrypted storage',
                        remediation_steps=[
                            'Enable encryption for RDS instance',
                            'Create encrypted snapshot and restore',
                            'Update application connection strings'
                        ],
                        timestamp=datetime.utcnow()
                    ))
                
                # Check backup retention
                if db_instance.get('BackupRetentionPeriod', 0) < 7:
                    violations.append(ComplianceViolation(
                        resource_id=db_id,
                        resource_type='RDS Instance',
                        violation_type='insufficient_backup_retention',
                        severity='MEDIUM',
                        description=f'RDS instance {db_id} has insufficient backup retention period',
                        remediation_steps=[
                            'Increase backup retention period to at least 7 days',
                            'Configure automated backups',
                            'Test backup restoration procedures'
                        ],
                        timestamp=datetime.utcnow()
                    ))
                
        except Exception as e:
            self.logger.error(f"Error scanning RDS compliance: {str(e)}")
        
        return violations
    
    async def generate_compliance_report(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        # Categorize violations by severity
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        resource_type_counts = {}
        
        for violation in violations:
            severity_counts[violation.severity] += 1
            resource_type_counts[violation.resource_type] = resource_type_counts.get(violation.resource_type, 0) + 1
        
        # Calculate compliance score
        total_resources_scanned = len(set(v.resource_id for v in violations)) + 100  # Assume 100 compliant resources
        compliance_score = max(0, (total_resources_scanned - len(violations)) / total_resources_scanned * 100)
        
        report = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'compliance_score': round(compliance_score, 2),
            'total_violations': len(violations),
            'severity_breakdown': severity_counts,
            'resource_type_breakdown': resource_type_counts,
            'violations': [
                {
                    'resource_id': v.resource_id,
                    'resource_type': v.resource_type,
                    'violation_type': v.violation_type,
                    'severity': v.severity,
                    'description': v.description,
                    'remediation_steps': v.remediation_steps,
                    'timestamp': v.timestamp.isoformat()
                }
                for v in violations
            ],
            'recommendations': self._generate_recommendations(violations)
        }
        
        return report
    
    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []
        
        # Analyze violation patterns
        critical_violations = [v for v in violations if v.severity == 'CRITICAL']
        if critical_violations:
            recommendations.append(
                f"Address {len(critical_violations)} critical violations immediately - these pose significant security risks"
            )
        
        # Security group recommendations
        sg_violations = [v for v in violations if v.resource_type == 'Security Group']
        if len(sg_violations) > 5:
            recommendations.append(
                "Consider implementing automated security group management and regular auditing"
            )
        
        # Encryption recommendations
        encryption_violations = [v for v in violations if 'encryption' in v.violation_type]
        if encryption_violations:
            recommendations.append(
                "Implement organization-wide encryption policies and automated enforcement"
            )
        
        # General recommendations
        if len(violations) > 20:
            recommendations.append(
                "High number of violations detected - consider implementing policy-as-code and automated remediation"
            )
        
        return recommendations
    
    async def send_alerts(self, violations: List[ComplianceViolation]):
        """Send alerts for compliance violations"""
        critical_violations = [v for v in violations if v.severity == 'CRITICAL']
        
        if critical_violations:
            message = {
                'alert_type': 'compliance_violation',
                'severity': 'CRITICAL',
                'count': len(critical_violations),
                'violations': [
                    {
                        'resource_id': v.resource_id,
                        'resource_type': v.resource_type,
                        'description': v.description
                    }
                    for v in critical_violations[:5]  # Limit to first 5
                ],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            try:
                self.sns.publish(
                    TopicArn=self.config['sns_topic_arn'],
                    Subject='Critical Compliance Violations Detected',
                    Message=json.dumps(message, indent=2)
                )
                self.logger.info(f"Alert sent for {len(critical_violations)} critical violations")
            except Exception as e:
                self.logger.error(f"Failed to send alert: {str(e)}")
    
    async def run_compliance_scan(self) -> Dict[str, Any]:
        """Run comprehensive compliance scan"""
        self.logger.info("Starting compliance scan...")
        
        # Run all compliance scans concurrently
        ec2_violations, s3_violations, rds_violations = await asyncio.gather(
            self.scan_ec2_compliance(),
            self.scan_s3_compliance(),
            self.scan_rds_compliance(),
            return_exceptions=True
        )
        
        # Combine all violations
        all_violations = []
        for violation_list in [ec2_violations, s3_violations, rds_violations]:
            if isinstance(violation_list, list):
                all_violations.extend(violation_list)
            else:
                self.logger.error(f"Error in compliance scan: {violation_list}")
        
        # Generate report
        report = await self.generate_compliance_report(all_violations)
        
        # Send alerts for critical violations
        await self.send_alerts(all_violations)
        
        self.logger.info(f"Compliance scan completed. Found {len(all_violations)} violations.")
        
        return report

async def main():
    """Main compliance monitoring function"""
    config = {
        'sns_topic_arn': 'arn:aws:sns:us-west-2:123456789012:ml-compliance-alerts',
        'scan_interval': 3600,  # 1 hour
        'report_s3_bucket': 'ml-compliance-reports'
    }
    
    monitor = ComplianceMonitor(config)
    
    while True:
        try:
            report = await monitor.run_compliance_scan()
            
            # Save report to S3
            s3_client = boto3.client('s3')
            report_key = f"compliance-reports/{datetime.utcnow().strftime('%Y/%m/%d')}/compliance-report-{int(time.time())}.json"
            
            s3_client.put_object(
                Bucket=config['report_s3_bucket'],
                Key=report_key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            
            print(f"Compliance scan completed. Score: {report['compliance_score']}%")
            print(f"Total violations: {report['total_violations']}")
            print(f"Report saved to s3://{config['report_s3_bucket']}/{report_key}")
            
            # Wait for next scan
            await asyncio.sleep(config['scan_interval'])
            
        except KeyboardInterrupt:
            print("Compliance monitoring stopped.")
            break
        except Exception as e:
            logging.error(f"Error in compliance monitoring: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

### 4. Automated Remediation Systems

#### 4.1 Self-Healing Infrastructure

**AWS Lambda Auto-Remediation**:
```python
#!/usr/bin/env python3
# lambda_auto_remediation.py

import boto3
import json
import logging
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for automated compliance remediation
    Triggered by CloudWatch Events from AWS Config
    """
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    try:
        # Parse the CloudWatch event
        detail = event.get('detail', {})
        config_item = detail.get('configurationItem', {})
        
        resource_type = config_item.get('resourceType')
        resource_id = config_item.get('resourceId')
        compliance_type = detail.get('newEvaluationResult', {}).get('complianceType')
        
        logger.info(f"Processing compliance event: {resource_type} {resource_id} - {compliance_type}")
        
        if compliance_type != 'NON_COMPLIANT':
            return {'statusCode': 200, 'body': 'Resource is compliant'}
        
        # Route to appropriate remediation function
        remediation_result = None
        
        if resource_type == 'AWS::EC2::SecurityGroup':
            remediation_result = remediate_security_group(resource_id, detail)
        elif resource_type == 'AWS::S3::Bucket':
            remediation_result = remediate_s3_bucket(resource_id, detail)
        elif resource_type == 'AWS::EC2::Instance':
            remediation_result = remediate_ec2_instance(resource_id, detail)
        elif resource_type == 'AWS::RDS::DBInstance':
            remediation_result = remediate_rds_instance(resource_id, detail)
        else:
            logger.warning(f"No remediation available for resource type: {resource_type}")
            return {'statusCode': 200, 'body': f'No remediation for {resource_type}'}
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Remediation completed',
                'resource_id': resource_id,
                'resource_type': resource_type,
                'result': remediation_result
            })
        }
        
    except Exception as e:
        logger.error(f"Error in auto-remediation: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def remediate_security_group(sg_id: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Remediate security group compliance violations"""
    ec2 = boto3.client('ec2')
    logger = logging.getLogger()
    
    try:
        # Get security group details
        response = ec2.describe_security_groups(GroupIds=[sg_id])
        sg = response['SecurityGroups'][0]
        
        remediation_actions = []
        
        # Check for unrestricted SSH access (0.0.0.0/0:22)
        for rule in sg['IpPermissions']:
            if (rule.get('FromPort', 0) <= 22 <= rule.get('ToPort', 65535) and
                any(ip_range.get('CidrIp') == '0.0.0.0/0' for ip_range in rule.get('IpRanges', []))):
                
                # Remove the unrestricted rule
                ec2.revoke_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[rule]
                )
                
                # Add restricted rule for company IP ranges
                company_cidrs = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
                for cidr in company_cidrs:
                    ec2.authorize_security_group_ingress(
                        GroupId=sg_id,
                        IpPermissions=[{
                            'IpProtocol': rule['IpProtocol'],
                            'FromPort': rule.get('FromPort', 22),
                            'ToPort': rule.get('ToPort', 22),
                            'IpRanges': [{'CidrIp': cidr, 'Description': 'Company network access'}]
                        }]
                    )
                
                remediation_actions.append(f"Replaced unrestricted SSH rule with company CIDR restrictions")
        
        # Tag the security group for tracking
        ec2.create_tags(
            Resources=[sg_id],
            Tags=[
                {'Key': 'AutoRemediated', 'Value': 'true'},
                {'Key': 'RemediationTimestamp', 'Value': str(int(time.time()))}
            ]
        )
        
        return {
            'status': 'success',
            'actions': remediation_actions
        }
        
    except Exception as e:
        logger.error(f"Error remediating security group {sg_id}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def remediate_s3_bucket(bucket_name: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Remediate S3 bucket compliance violations"""
    s3 = boto3.client('s3')
    logger = logging.getLogger()
    
    try:
        remediation_actions = []
        
        # Check and enable encryption
        try:
            s3.get_bucket_encryption(Bucket=bucket_name)
        except s3.exceptions.NoSuchEncryption:
            # Enable default encryption
            s3.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [{
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }]
                }
            )
            remediation_actions.append("Enabled S3 bucket encryption")
        
        # Check and configure public access block
        try:
            s3.get_public_access_block(Bucket=bucket_name)
        except s3.exceptions.NoSuchPublicAccessBlockConfiguration:
            s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )
            remediation_actions.append("Configured S3 public access block")
        
        # Enable versioning if not already enabled
        versioning = s3.get_bucket_versioning(Bucket=bucket_name)
        if versioning.get('Status') != 'Enabled':
            s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            remediation_actions.append("Enabled S3 bucket versioning")
        
        return {
            'status': 'success',
            'actions': remediation_actions
        }
        
    except Exception as e:
        logger.error(f"Error remediating S3 bucket {bucket_name}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def remediate_ec2_instance(instance_id: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Remediate EC2 instance compliance violations"""
    ec2 = boto3.client('ec2')
    logger = logging.getLogger()
    
    try:
        remediation_actions = []
        
        # Get instance details
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        # Check if instance has public IP (for ML training instances, this should be removed)
        if instance.get('PublicIpAddress'):
            # This requires stopping and starting the instance
            # For safety, we'll just tag it for manual review
            ec2.create_tags(
                Resources=[instance_id],
                Tags=[
                    {'Key': 'ComplianceIssue', 'Value': 'PublicIPDetected'},
                    {'Key': 'RequiresManualReview', 'Value': 'true'}
                ]
            )
            remediation_actions.append("Tagged instance for manual review - public IP detected")
        
        # Ensure instance monitoring is enabled
        if not instance.get('Monitoring', {}).get('State') == 'enabled':
            ec2.monitor_instances(InstanceIds=[instance_id])
            remediation_actions.append("Enabled detailed monitoring")
        
        return {
            'status': 'success',
            'actions': remediation_actions
        }
        
    except Exception as e:
        logger.error(f"Error remediating EC2 instance {instance_id}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def remediate_rds_instance(db_instance_id: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Remediate RDS instance compliance violations"""
    rds = boto3.client('rds')
    logger = logging.getLogger()
    
    try:
        remediation_actions = []
        
        # Get RDS instance details
        response = rds.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        db_instance = response['DBInstances'][0]
        
        # Check backup retention period
        if db_instance.get('BackupRetentionPeriod', 0) < 7:
            rds.modify_db_instance(
                DBInstanceIdentifier=db_instance_id,
                BackupRetentionPeriod=7,
                ApplyImmediately=False  # Apply during maintenance window
            )
            remediation_actions.append("Increased backup retention period to 7 days")
        
        # Enable performance insights if not enabled
        if not db_instance.get('PerformanceInsightsEnabled', False):
            rds.modify_db_instance(
                DBInstanceIdentifier=db_instance_id,
                EnablePerformanceInsights=True,
                ApplyImmediately=False
            )
            remediation_actions.append("Enabled Performance Insights")
        
        return {
            'status': 'success',
            'actions': remediation_actions
        }
        
    except Exception as e:
        logger.error(f"Error remediating RDS instance {db_instance_id}: {str(e)}")
        return {'status': 'error', 'message': str(e)}
```

#### 4.2 Kubernetes Auto-Remediation

**Kubernetes Operator for Compliance**:
```yaml
# compliance-operator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-compliance-operator
  namespace: compliance-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-compliance-operator
  template:
    metadata:
      labels:
        app: ml-compliance-operator
    spec:
      serviceAccountName: ml-compliance-operator
      containers:
      - name: operator
        image: ml-compliance-operator:latest
        env:
        - name: WATCH_NAMESPACE
          value: ""
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: OPERATOR_NAME
          value: "ml-compliance-operator"
        resources:
          limits:
            cpu: 200m
            memory: 256Mi
          requests:
            cpu: 100m
            memory: 128Mi
        volumeMounts:
        - name: config
          mountPath: /etc/compliance
      volumes:
      - name: config
        configMap:
          name: compliance-config

---
# compliance-operator-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-compliance-operator
  namespace: compliance-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-compliance-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "persistentvolumeclaims", "events", "configmaps", "secrets"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["deployments", "daemonsets", "replicasets", "statefulsets"]
  verbs: ["*"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies"]
  verbs: ["*"]
- apiGroups: ["policy"]
  resources: ["podsecuritypolicies"]
  verbs: ["*"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ml-compliance-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ml-compliance-operator
subjects:
- kind: ServiceAccount
  name: ml-compliance-operator
  namespace: compliance-system
```

**Compliance Operator Implementation**:
```python
#!/usr/bin/env python3
# compliance_operator.py

import asyncio
import logging
from kubernetes import client, config, watch
from typing import Dict, Any, List
import yaml
import json
import time

class MLComplianceOperator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        
        # Load compliance rules
        self.compliance_rules = self._load_compliance_rules()
        
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from ConfigMap"""
        try:
            config_map = self.v1.read_namespaced_config_map(
                name="compliance-config",
                namespace="compliance-system"
            )
            return yaml.safe_load(config_map.data.get('rules.yaml', '{}'))
        except Exception as e:
            self.logger.error(f"Failed to load compliance rules: {str(e)}")
            return {}
    
    async def watch_pod_events(self):
        """Watch for pod creation/modification events"""
        w = watch.Watch()
        
        for event in w.stream(self.v1.list_pod_for_all_namespaces):
            event_type = event['type']
            pod = event['object']
            
            if event_type in ['ADDED', 'MODIFIED']:
                await self._check_pod_compliance(pod)
    
    async def _check_pod_compliance(self, pod: client.V1Pod):
        """Check pod compliance and remediate if necessary"""
        try:
            namespace = pod.metadata.namespace
            name = pod.metadata.name
            
            # Skip system namespaces
            if namespace in ['kube-system', 'kube-public', 'compliance-system']:
                return
            
            violations = []
            
            # Check if pod is ML-related
            labels = pod.metadata.labels or {}
            if not any(label in ['ml-training', 'ml-inference', 'ml-development'] 
                      for label in labels.values()):
                return
            
            # Check security context
            if pod.spec.security_context:
                if pod.spec.security_context.run_as_user == 0:
                    violations.append({
                        'type': 'root_user',
                        'description': 'Pod is running as root user',
                        'severity': 'HIGH'
                    })
                
                if not pod.spec.security_context.run_as_non_root:
                    violations.append({
                        'type': 'non_root_not_enforced',
                        'description': 'Pod does not enforce non-root execution',
                        'severity': 'MEDIUM'
                    })
            
            # Check container security contexts
            for container in pod.spec.containers:
                if container.security_context:
                    if container.security_context.privileged:
                        violations.append({
                            'type': 'privileged_container',
                            'description': f'Container {container.name} is running in privileged mode',
                            'severity': 'CRITICAL'
                        })
                    
                    if container.security_context.allow_privilege_escalation:
                        violations.append({
                            'type': 'privilege_escalation',
                            'description': f'Container {container.name} allows privilege escalation',
                            'severity': 'HIGH'
                        })
                
                # Check resource limits
                if not container.resources or not container.resources.limits:
                    violations.append({
                        'type': 'no_resource_limits',
                        'description': f'Container {container.name} has no resource limits',
                        'severity': 'MEDIUM'
                    })
            
            # Check required labels
            required_labels = self.compliance_rules.get('required_labels', [])
            for required_label in required_labels:
                if required_label not in labels:
                    violations.append({
                        'type': 'missing_label',
                        'description': f'Pod is missing required label: {required_label}',
                        'severity': 'LOW'
                    })
            
            # Remediate violations if possible
            if violations:
                await self._remediate_pod_violations(namespace, name, violations)
            
        except Exception as e:
            self.logger.error(f"Error checking pod compliance: {str(e)}")
    
    async def _remediate_pod_violations(self, namespace: str, name: str, violations: List[Dict[str, Any]]):
        """Attempt to remediate pod violations"""
        try:
            # For high-severity violations, create alert
            critical_violations = [v for v in violations if v['severity'] in ['CRITICAL', 'HIGH']]
            
            if critical_violations:
                await self._create_compliance_alert(namespace, name, critical_violations)
            
            # For certain violations, we can auto-remediate by updating the deployment
            owner_references = []
            try:
                pod = self.v1.read_namespaced_pod(name=name, namespace=namespace)
                owner_references = pod.metadata.owner_references or []
            except Exception:
                pass
            
            # Find the deployment that owns this pod
            for owner_ref in owner_references:
                if owner_ref.kind == 'ReplicaSet':
                    rs = self.apps_v1.read_namespaced_replica_set(
                        name=owner_ref.name, 
                        namespace=namespace
                    )
                    
                    # Find the deployment that owns this replica set
                    rs_owners = rs.metadata.owner_references or []
                    for rs_owner in rs_owners:
                        if rs_owner.kind == 'Deployment':
                            await self._remediate_deployment(namespace, rs_owner.name, violations)
            
        except Exception as e:
            self.logger.error(f"Error remediating pod violations: {str(e)}")
    
    async def _remediate_deployment(self, namespace: str, deployment_name: str, violations: List[Dict[str, Any]]):
        """Remediate deployment configuration"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            modified = False
            
            # Add missing labels
            for violation in violations:
                if violation['type'] == 'missing_label':
                    label_name = violation['description'].split(': ')[1]
                    if not deployment.spec.template.metadata.labels:
                        deployment.spec.template.metadata.labels = {}
                    deployment.spec.template.metadata.labels[label_name] = 'auto-added'
                    modified = True
            
            # Add security contexts if missing
            has_security_violations = any(
                v['type'] in ['root_user', 'non_root_not_enforced'] 
                for v in violations
            )
            
            if has_security_violations:
                if not deployment.spec.template.spec.security_context:
                    deployment.spec.template.spec.security_context = client.V1PodSecurityContext()
                
                deployment.spec.template.spec.security_context.run_as_non_root = True
                deployment.spec.template.spec.security_context.run_as_user = 1000
                modified = True
            
            # Update the deployment if modifications were made
            if modified:
                # Add annotation to track auto-remediation
                if not deployment.spec.template.metadata.annotations:
                    deployment.spec.template.metadata.annotations = {}
                deployment.spec.template.metadata.annotations['compliance.ml/auto-remediated'] = str(int(time.time()))
                
                self.apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                self.logger.info(f"Auto-remediated deployment {deployment_name} in namespace {namespace}")
            
        except Exception as e:
            self.logger.error(f"Error remediating deployment: {str(e)}")
    
    async def _create_compliance_alert(self, namespace: str, resource_name: str, violations: List[Dict[str, Any]]):
        """Create compliance alert event"""
        try:
            event = client.CoreV1Event(
                metadata=client.V1ObjectMeta(
                    name=f"compliance-violation-{resource_name}-{int(time.time())}",
                    namespace=namespace
                ),
                involved_object=client.V1ObjectReference(
                    kind="Pod",
                    name=resource_name,
                    namespace=namespace
                ),
                reason="ComplianceViolation",
                message=f"Compliance violations detected: {json.dumps(violations)}",
                first_timestamp=client.V1Time(time.time()),
                last_timestamp=client.V1Time(time.time()),
                count=1,
                type="Warning",
                source=client.V1EventSource(component="ml-compliance-operator")
            )
            
            self.v1.create_namespaced_event(namespace=namespace, body=event)
            
        except Exception as e:
            self.logger.error(f"Error creating compliance alert: {str(e)}")
    
    async def run(self):
        """Main operator loop"""
        self.logger.info("Starting ML Compliance Operator")
        
        try:
            await asyncio.gather(
                self.watch_pod_events(),
                # Add other watchers as needed
            )
        except Exception as e:
            self.logger.error(f"Error in operator main loop: {str(e)}")

async def main():
    logging.basicConfig(level=logging.INFO)
    operator = MLComplianceOperator()
    await operator.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔍 Key Questions

### Beginner Level

1. **Q**: What is configuration drift and why is it particularly problematic in AI/ML environments?
   **A**: Configuration drift occurs when the actual infrastructure state deviates from the intended configuration. In AI/ML environments, this is problematic because training jobs can run for days/weeks, small configuration changes can affect model performance, and distributed systems have complex interdependencies.

2. **Q**: What are the main types of compliance scanning and when would you use each?
   **A**: 
   - **Static scanning**: Analyzes configurations before deployment (IaC templates)
   - **Runtime scanning**: Monitors live systems for compliance violations
   - **Continuous scanning**: Ongoing monitoring with automated remediation
   Use static for early detection, runtime for operational compliance, continuous for automated governance.

3. **Q**: How does AWS Config help with compliance monitoring, and what are its key components?
   **A**: AWS Config tracks resource configurations and evaluates them against rules. Key components: Configuration Recorder (tracks changes), Delivery Channel (sends data to S3), Config Rules (evaluate compliance), and Remediation Actions (automated fixes).

### Intermediate Level

4. **Q**: Design a drift detection strategy for a Kubernetes-based ML training platform that includes both infrastructure and application-level monitoring.
   **A**:
   ```
   Infrastructure Level:
   - Terraform state drift detection with automated scanning
   - AWS Config rules for cloud resource compliance
   - Kubernetes admission controllers (OPA Gatekeeper)
   
   Application Level:
   - Polaris for pod security scanning
   - Falco for runtime security monitoring
   - Custom controllers for ML-specific policies
   
   Integration:
   - Centralized alerting through SNS/Slack
   - Automated remediation with approval workflows
   - Compliance dashboards with real-time metrics
   ```

5. **Q**: How would you implement automated remediation while ensuring safety and avoiding cascading failures in a production ML environment?
   **A**: Implement graduated response with safety checks: warning alerts for minor issues, automatic remediation for safe changes with rollback capability, manual approval for high-risk changes, circuit breakers to stop automation during incidents, and comprehensive testing in staging environments.

6. **Q**: Explain how to implement compliance scanning for a multi-cloud ML infrastructure spanning AWS, Azure, and GCP.
   **A**: Use cloud-agnostic tools (Terraform, OPA), implement platform-specific scanners (AWS Config, Azure Policy, GCP Security Command Center), centralize results in SIEM system, create unified compliance dashboards, and maintain consistent policy definitions across platforms.

### Advanced Level

7. **Q**: Design a comprehensive compliance framework for a regulated healthcare AI platform that must meet HIPAA, SOC 2, and FDA requirements.
   **A**:
   ```
   Framework Components:
   - Risk-based compliance tiers with different scanning frequencies
   - Automated policy enforcement with emergency override procedures  
   - Immutable audit trails for all configuration changes
   - Real-time compliance monitoring with sub-5-minute alerting
   - Automated evidence collection for regulatory audits
   - ML model validation pipeline with compliance checkpoints
   - Data lineage tracking for healthcare AI models
   - Incident response with regulatory notification workflows
   ```

8. **Q**: How would you implement zero-trust compliance monitoring that continuously validates security posture without impacting ML training performance?
   **A**: Deploy lightweight agents with minimal CPU/memory footprint, use asynchronous scanning during training idle periods, implement sampling-based monitoring for high-frequency checks, use machine learning for anomaly detection to reduce false positives, and create separate monitoring networks to avoid interference.

### Tricky Questions

9. **Q**: You discover that your automated remediation system has been making unauthorized changes to production ML infrastructure, potentially compromising a critical research project. How would you investigate and prevent future occurrences?
   **A**:
   ```
   Immediate Response:
   - Disable automated remediation system immediately
   - Assess scope of unauthorized changes through audit logs
   - Restore systems from known good configurations
   - Notify stakeholders and document impact
   
   Investigation:
   - Review automation logic and approval workflows
   - Analyze change logs and configuration drift patterns
   - Identify root cause (bug, policy misconfiguration, security breach)
   - Assess data integrity and model training impact
   
   Prevention:
   - Implement multi-level approval for production changes
   - Add comprehensive testing for remediation logic
   - Create staging environment mirroring production
   - Implement change impact assessment before automation
   - Add human oversight for critical infrastructure changes
   ```

10. **Q**: Design a compliance scanning system that can handle the scale of a major tech company's AI infrastructure (100,000+ resources across 50+ AWS accounts) while maintaining sub-minute detection times for critical violations.
    **A**:
    ```
    Architecture:
    - Distributed scanning with regional deployment
    - Event-driven architecture using CloudWatch Events
    - Parallel processing with Lambda/ECS for scalability
    - Caching layer for frequently accessed configuration data
    - Hierarchical alerting with escalation procedures
    
    Implementation:
    - Multi-account AWS Config aggregator for centralization
    - Stream processing for real-time violation detection
    - Machine learning for pattern recognition and false positive reduction
    - Auto-scaling based on event volume and resource count
    - Geographic distribution for global infrastructure coverage
    
    Performance Optimization:
    - Pre-computed compliance baselines
    - Incremental scanning for changed resources only
    - Prioritized scanning based on resource criticality
    - Batch processing for non-critical compliance checks
    - Resource tagging for intelligent filtering
    ```

---

## 🛡️ Security Deep Dive

### Advanced Threat Scenarios

#### Compliance Bypass Attacks

**Attack Vectors**:
```
Common Bypass Techniques:
- Temporary configuration changes during compliance scan windows
- Exploitation of drift detection timing gaps
- Privilege escalation to disable monitoring agents
- Configuration rollback after compliance checks
- Supply chain attacks on compliance tools

Detection Strategies:
- Continuous monitoring with randomized scan intervals
- Immutable infrastructure with change tracking
- Behavioral analysis for anomalous configuration patterns
- Multi-layered compliance validation
- Real-time alerting for monitoring tool tampering
```

#### Advanced Persistent Threats (APTs)

**Long-term Configuration Manipulation**:
```
APT Tactics:
- Gradual configuration changes to avoid detection
- Legitimate-looking changes during maintenance windows
- Compromise of automation tools for persistent access
- Living-off-the-land techniques using native cloud tools

Countermeasures:
- Machine learning-based anomaly detection
- Configuration change correlation analysis
- Supply chain security for automation tools
- Zero-trust approach to configuration management
- Regular security assessments and red team exercises
```

### Compliance and Governance Integration

#### Regulatory Framework Mapping

**Multi-Framework Compliance**:
```
Framework Integration Strategy:
- Common control mapping across regulations
- Unified evidence collection for multiple audits
- Risk-based prioritization of compliance requirements
- Automated regulatory reporting generation
- Change impact assessment for regulatory controls
```

---

## 🚀 Performance Optimization

### Scanning Performance Optimization

#### Distributed Scanning Architecture

**Scale-Out Strategies**:
```python
# Example distributed scanning coordinator
class DistributedScanCoordinator:
    def __init__(self, regions: List[str], max_workers: int = 10):
        self.regions = regions
        self.max_workers = max_workers
        self.scan_results = {}
    
    async def coordinate_global_scan(self) -> Dict[str, Any]:
        """Coordinate compliance scanning across multiple regions"""
        
        # Distribute scanning tasks across regions
        scan_tasks = []
        for region in self.regions:
            task = asyncio.create_task(
                self.scan_region(region)
            )
            scan_tasks.append(task)
        
        # Wait for all scans to complete
        results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Aggregate results
        return self.aggregate_scan_results(results)
    
    async def scan_region(self, region: str) -> Dict[str, Any]:
        """Scan a specific region for compliance violations"""
        # Implementation would include region-specific scanning logic
        pass
```

#### Intelligent Scanning Optimization

**Smart Scanning Strategies**:
```
Optimization Techniques:
- Risk-based scanning frequency (critical resources scanned more often)
- Change-triggered scanning (scan only when resources change)
- Batch processing for non-critical compliance checks
- Caching of compliance results with TTL
- Parallel scanning of independent resource groups
- Machine learning for scan prioritization
```

---

## 📝 Practical Exercises

### Exercise 1: Comprehensive Drift Detection System
Design and implement a complete configuration drift detection system that:
- Monitors infrastructure across multiple cloud providers
- Detects application-level configuration drift in Kubernetes
- Provides real-time alerting with severity-based escalation
- Includes automated remediation with safety controls
- Generates compliance reports for audit purposes

### Exercise 2: Regulatory Compliance Automation
Create an automated compliance system for a healthcare AI platform that:
- Implements HIPAA, SOC 2, and FDA validation requirements
- Provides continuous monitoring with audit trail generation
- Includes automated evidence collection for regulatory audits
- Handles emergency access procedures with proper logging
- Integrates with existing security and monitoring tools

### Exercise 3: High-Scale Compliance Scanning
Develop a compliance scanning solution capable of:
- Scanning 100,000+ resources across 50+ AWS accounts
- Achieving sub-minute detection for critical violations
- Handling 10,000+ configuration changes per hour
- Providing real-time compliance dashboards
- Supporting multiple regulatory frameworks simultaneously

### Exercise 4: Zero-Trust Compliance Architecture
Design a zero-trust compliance monitoring system that:
- Continuously validates security posture
- Implements context-aware policy enforcement
- Provides behavioral analysis for anomaly detection
- Includes automated response to policy violations
- Maintains detailed audit trails for forensic analysis

---

## 🔗 Completion of Day 2

We have successfully completed all Day 2 topics covering Secure Infrastructure & Automation:

✅ **Day 2.1**: Infrastructure-as-Code Security with Terraform, Pulumi, and Crossplane
✅ **Day 2.2**: Policy-as-Code with OPA, Sentinel, and Gatekeeper  
✅ **Day 2.3**: IaC Module Hardening and Least Privilege IAM
✅ **Day 2.4**: Secret Management with Vault, AWS KMS, and Azure Key Vault
✅ **Day 2.5**: Drift Detection and Automated Compliance Scanning

Each section provides comprehensive theoretical foundations, practical implementation examples, and progressive learning materials specifically tailored for AI/ML security environments. The content maintains focus on defensive security applications while covering beginner to advanced concepts with detailed examples, challenging questions, and real-world scenarios.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "8", "content": "Create day02_001_infrastructure_as_code_security.md - Secure IaC with Terraform, Pulumi, Crossplane", "status": "completed", "priority": "high"}, {"id": "9", "content": "Create day02_002_policy_as_code.md - OPA, Sentinel, Gatekeeper implementation", "status": "completed", "priority": "high"}, {"id": "10", "content": "Create day02_003_iac_module_hardening.md - Least privilege IAM and security practices", "status": "completed", "priority": "high"}, {"id": "11", "content": "Create day02_004_secret_management.md - Vault, AWS KMS, Azure Key Vault integration", "status": "completed", "priority": "high"}, {"id": "12", "content": "Create day02_005_compliance_scanning.md - Drift detection and automated compliance", "status": "completed", "priority": "high"}]