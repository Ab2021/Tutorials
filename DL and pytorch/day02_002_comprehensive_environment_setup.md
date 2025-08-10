# Day 2.2: Comprehensive Environment Setup for PyTorch Development

## Overview
Setting up an optimal development environment is crucial for successful deep learning projects. This comprehensive guide covers all aspects of PyTorch environment setup, from basic local installations to advanced cloud configurations, containerized deployments, and production-ready environments. We'll explore best practices for reproducibility, version management, and development workflow optimization.

## Installation Strategies

### Local Installation with Package Managers

**Understanding PyTorch Distribution Channels**
PyTorch offers multiple installation channels, each with specific advantages and use cases:

**Official PyTorch Distribution**:
- **Primary Source**: pytorch.org provides official, tested releases
- **CUDA Compatibility**: Pre-built binaries for different CUDA versions
- **Platform Support**: Windows, macOS, Linux with optimized builds
- **Update Frequency**: Regular releases with latest features and bug fixes

**Conda Installation Strategy**
Conda provides the most robust package management for scientific Python:

**Conda-Forge vs PyTorch Channel**:
- **PyTorch Official Channel**: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- **Conda-Forge Alternative**: More comprehensive dependency resolution but potentially slower updates
- **Channel Priorities**: Understanding how conda resolves dependencies across channels

**Environment Isolation Benefits**:
```bash
# Create dedicated PyTorch environment
conda create -n pytorch_env python=3.9
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Advantages of Conda for Deep Learning**:
- **Binary Dependencies**: Handles complex C++/CUDA dependencies automatically
- **Version Locking**: Precise version control for reproducible environments
- **Scientific Stack**: Optimized builds for NumPy, SciPy, and other scientific packages
- **Cross-Platform**: Consistent behavior across operating systems

**Pip Installation Considerations**
While conda is often preferred, pip installation offers certain advantages:

**PyPI Distribution**:
```bash
# Basic PyTorch installation via pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Advantages of Pip Installation**:
- **Latest Releases**: Often gets updates faster than conda
- **Lighter Weight**: Smaller installation footprint
- **Virtual Environment Integration**: Works seamlessly with venv and virtualenv
- **Development Builds**: Access to nightly and experimental releases

**Potential Challenges**:
- **Dependency Conflicts**: More prone to version conflicts with system packages
- **CUDA Dependencies**: Manual management of CUDA toolkit versions
- **Binary Compatibility**: Platform-specific binary compatibility issues

### CUDA Toolkit Installation and Version Management

**Understanding CUDA Ecosystem**
CUDA (Compute Unified Device Architecture) is essential for GPU-accelerated deep learning:

**CUDA Component Architecture**:
- **CUDA Driver**: Low-level interface between OS and GPU hardware
- **CUDA Runtime**: High-level API for CUDA programming
- **CUDA Toolkit**: Complete development environment with compilers and libraries
- **cuDNN**: Deep neural network library optimized for NVIDIA GPUs

**CUDA Version Compatibility Matrix**
Understanding compatibility between PyTorch, CUDA, and GPU architectures:

**PyTorch CUDA Support Timeline**:
- **PyTorch 2.0+**: Supports CUDA 11.7, 11.8, 12.1+
- **PyTorch 1.13**: Supports CUDA 11.6, 11.7, 11.8
- **Legacy Support**: Older PyTorch versions support CUDA 10.2, 11.3

**GPU Architecture Compatibility**:
- **Compute Capability 8.6**: RTX 30 series, RTX 40 series
- **Compute Capability 8.0**: A100, RTX 3090
- **Compute Capability 7.5**: RTX 20 series, GTX 16 series
- **Legacy Support**: Older architectures require compatible CUDA versions

**CUDA Installation Methods**

**System-Wide CUDA Installation**:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

**Docker-Based CUDA Environment**:
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Conda CUDA Management**:
```bash
# Install CUDA toolkit through conda
conda install cudatoolkit=11.8 -c conda-forge
# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Version Management Best Practices**:
- **Environment Separation**: Different CUDA versions in separate conda environments
- **Version Pinning**: Lock CUDA versions in environment.yml files
- **Compatibility Testing**: Regular testing across different CUDA versions
- **Documentation**: Clear documentation of CUDA requirements for projects

### Docker Containerization for Reproducible Environments

**Container-Based Development Advantages**
Docker provides the ultimate solution for reproducible deep learning environments:

**Reproducibility Benefits**:
- **Identical Environments**: Same runtime environment across all machines
- **Version Locking**: Immutable environment specifications
- **Dependency Isolation**: No conflicts with host system packages
- **Collaborative Development**: Team members share identical development environments

**PyTorch Docker Images**
NVIDIA provides official Docker images optimized for deep learning:

**Base Image Categories**:
- **Runtime Images**: Minimal images for running trained models
- **Development Images**: Full development environment with compilers and tools
- **Jupyter Images**: Pre-configured Jupyter notebook environments
- **Framework Images**: Images optimized for specific deep learning frameworks

**Official PyTorch Container Example**:
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install additional packages
RUN conda install -c conda-forge \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    jupyter

# Install pip packages
RUN pip install \
    wandb \
    tensorboard \
    hydra-core \
    lightning

# Copy project code
COPY . /workspace/

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

**Multi-Stage Build Optimization**:
```dockerfile
# Build stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "src/main.py"]
```

**Docker Compose for Complex Environments**:
```yaml
version: '3.8'
services:
  pytorch-dev:
    build: .
    volumes:
      - .:/workspace
      - pytorch_cache:/root/.cache/torch
    ports:
      - "8888:8888"
      - "6006:6006"  # TensorBoard
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0,1

  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir=/logs --host=0.0.0.0
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs

volumes:
  pytorch_cache:
```

### Cloud Platform Setup

**AWS EC2 Deep Learning Configuration**

**Deep Learning AMI (Amazon Machine Images)**:
- **Pre-configured Environments**: PyTorch, TensorFlow, and other frameworks pre-installed
- **CUDA Optimization**: Optimized CUDA and cuDNN installations
- **Jupyter Integration**: Pre-configured Jupyter environments with GPU support
- **Version Variants**: Different AMI versions for different framework versions

**EC2 Instance Selection Strategy**:
```bash
# GPU-optimized instances for deep learning
# p4d.24xlarge: 8x A100 GPUs, 96 vCPUs, 1152 GB RAM
# p3.16xlarge: 8x V100 GPUs, 64 vCPUs, 488 GB RAM
# g4dn.xlarge: 1x T4 GPU, 4 vCPUs, 16 GB RAM (cost-effective)

# Launch instance with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c94855ba95b798c7 \
    --instance-type p3.2xlarge \
    --key-name my-key-pair \
    --security-groups deep-learning-sg
```

**EBS Optimization for Deep Learning**:
- **Storage Types**: GP3 for cost-effectiveness, io2 for high IOPS workloads
- **Size Planning**: Consider dataset size plus model checkpoints
- **Snapshot Strategy**: Regular snapshots for data protection
- **Multi-Attach**: Shared storage across multiple instances for distributed training

**Google Cloud Platform (GCP) Configuration**

**AI Platform and Compute Engine Setup**:
```bash
# Create VM with GPUs for PyTorch development
gcloud compute instances create pytorch-dev \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

**Google Colab Pro Integration**:
- **Hardware Access**: High-memory VMs and premium GPUs
- **Persistent Storage**: Google Drive integration for data persistence
- **Notebook Sharing**: Collaborative development capabilities
- **Runtime Management**: Extended runtime limits for long training jobs

**Microsoft Azure ML Configuration**

**Azure ML Compute Setup**:
```python
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute

# Configure compute cluster for PyTorch training
compute_config = AmlCompute.provisioning_configuration(
    vm_size='Standard_NC24s_v3',  # 4x V100 GPUs
    min_nodes=0,
    max_nodes=4,
    idle_seconds_before_scaledown=120
)

compute_target = ComputeTarget.create(
    workspace=ws,
    name='pytorch-cluster',
    provisioning_configuration=compute_config
)
```

**Azure Container Instances**:
- **Serverless GPU Computing**: Pay-per-use GPU instances
- **Container Integration**: Direct Docker container deployment
- **Scaling**: Automatic scaling based on workload demands
- **Cost Optimization**: Only pay for actual compute time used

## Development Environment Configuration

### Jupyter Lab/Notebook Optimization

**Advanced Jupyter Configuration**
Jupyter provides the primary interface for interactive deep learning development:

**Custom Jupyter Configuration**:
```python
# ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.token = ''  # Only for secure environments
c.ResourceUseDisplay.track_cpu_percent = True
c.ResourceUseDisplay.mem_limit = 16*1024**3  # 16GB memory tracking
```

**JupyterLab Extensions for Deep Learning**:
```bash
# Install essential JupyterLab extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter labextension install @jupyterlab/debugger
jupyter labextension install jupyterlab-tensorboard
jupyter labextension install jupyterlab-system-monitor
```

**GPU Monitoring in Jupyter**:
```python
# Install and configure GPU monitoring
!pip install jupyterlab-nvdashboard
jupyter labextension install jupyterlab-nvdashboard

# Real-time GPU monitoring widget
from jupyterlab_nvdashboard import GPUMemoryWidget, GPUUtilizationWidget
gpu_widget = GPUMemoryWidget()
display(gpu_widget)
```

**Memory Management in Jupyter**:
```python
# Memory optimization settings
import torch
import gc

# Clear CUDA cache periodically
def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Automatic memory management
%load_ext autoreload
%autoreload 2
%config IPCompleter.greedy=True
```

### VSCode with PyTorch Extensions

**VSCode Deep Learning Setup**
VSCode provides excellent support for PyTorch development with proper configuration:

**Essential Extensions**:
- **Python**: Official Python extension with debugging and IntelliSense
- **PyTorch Snippets**: Code snippets for common PyTorch patterns
- **Jupyter**: Native Jupyter notebook support within VSCode
- **Remote - SSH**: Development on remote GPU servers
- **Docker**: Container development support

**VSCode Settings Configuration**:
```json
{
    "python.defaultInterpreterPath": "/opt/conda/envs/pytorch/bin/python",
    "python.terminal.activateEnvironment": true,
    "jupyter.jupyterServerType": "local",
    "jupyter.allowUnauthorizedRemoteConnection": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

**Remote Development Configuration**:
```bash
# SSH configuration for remote development
# ~/.ssh/config
Host gpu-server
    HostName gpu-server.example.com
    User username
    Port 22
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Debugging PyTorch Code in VSCode**:
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "PyTorch Training Debug",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train.py",
            "args": ["--config", "configs/debug.yaml"],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}
```

### Remote Development Setup

**SSH-Based Remote Development**
Remote development is essential for accessing GPU resources and large datasets:

**SSH Tunneling for Jupyter**:
```bash
# Local machine: Create SSH tunnel
ssh -L 8888:localhost:8888 username@remote-server

# Remote machine: Start Jupyter
jupyter lab --no-browser --port=8888

# Access via http://localhost:8888 on local machine
```

**VS Code Remote SSH Setup**:
```bash
# Install Remote - SSH extension
code --install-extension ms-vscode-remote.remote-ssh

# Connect to remote server
code --remote ssh-remote+gpu-server /path/to/project
```

**File Synchronization Strategies**:
```bash
# Rsync for efficient file synchronization
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    local_project/ username@remote-server:~/remote_project/

# Unison for bidirectional synchronization
unison local_project/ ssh://username@remote-server//home/username/remote_project/
```

**Remote Jupyter Configuration**:
```python
# Remote Jupyter with password authentication
from jupyter_server.auth import passwd
password = passwd('your-secure-password')

# jupyter_server_config.py
c.ServerApp.password = password
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_origin = '*'
```

### Version Control Integration with Git

**Git Configuration for Deep Learning Projects**
Version control requires special considerations for deep learning projects:

**Git LFS (Large File Storage) Setup**:
```bash
# Install and initialize Git LFS
git lfs install

# Track large files (models, datasets, checkpoints)
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.npz"
git lfs track "data/**"
git add .gitattributes
```

**Comprehensive .gitignore for PyTorch**:
```gitignore
# PyTorch specific
*.pth
*.pt
checkpoints/
wandb/
lightning_logs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Data
data/
datasets/
*.csv
*.h5
*.hdf5

# Logs and outputs
logs/
outputs/
results/
tensorboard/

# Environment
.env
.conda/
conda-meta/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

**Pre-commit Hooks for Code Quality**:
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203"]
```

**Branch Strategy for ML Projects**:
- **main/master**: Stable, production-ready code
- **develop**: Integration branch for new features
- **feature/experiment-name**: Individual experiments and features
- **hotfix/issue-description**: Critical fixes for production issues

## Key Questions for Review

### Installation and Setup
1. **Package Manager Choice**: When should you choose conda over pip for PyTorch installation, and what are the trade-offs?

2. **CUDA Compatibility**: How do you determine the correct CUDA version for your specific GPU and PyTorch version combination?

3. **Environment Isolation**: Why is environment isolation crucial for deep learning projects, and what problems does it solve?

### Containerization and Cloud
4. **Docker Benefits**: What specific advantages does Docker provide for deep learning development and deployment?

5. **Cloud Platform Selection**: How do you choose between AWS, GCP, and Azure for deep learning workloads?

6. **Cost Optimization**: What strategies can minimize cloud computing costs for deep learning experiments?

### Development Environment
7. **IDE Selection**: What factors should influence the choice between Jupyter, VSCode, and other development environments?

8. **Remote Development**: What are the key considerations for setting up effective remote development workflows?

9. **Version Control**: How should version control be adapted for deep learning projects with large models and datasets?

### Advanced Configuration
10. **Performance Optimization**: How can development environment configuration impact training performance?

11. **Reproducibility**: What practices ensure complete reproducibility across different development environments?

12. **Team Collaboration**: How do you set up environments that enable effective team collaboration on deep learning projects?

## Best Practices and Advanced Configurations

### Environment Reproducibility

**Complete Environment Specification**
Reproducibility requires comprehensive environment documentation:

**Conda Environment Files**:
```yaml
# environment.yml
name: pytorch-project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=2.0.1
  - torchvision=0.15.2
  - torchaudio=2.0.2
  - pytorch-cuda=11.8
  - cudnn=8.7.0
  - numpy=1.24.3
  - pandas=2.0.3
  - scikit-learn=1.3.0
  - matplotlib=3.7.1
  - seaborn=0.12.2
  - jupyter=1.0.0
  - ipykernel=6.24.0
  - pip
  - pip:
    - wandb==0.15.5
    - tensorboard==2.13.0
    - hydra-core==1.3.2
    - lightning==2.0.5
```

**Docker Environment Specification**:
```dockerfile
# Multi-stage build for reproducible environments
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils \
    python3-pip git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Install PyTorch and dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy project code
COPY . /workspace/

# Set default command
CMD ["/bin/bash"]
```

**Requirements Lock Files**:
```bash
# Generate exact version lock file
pip freeze > requirements-lock.txt

# Or use pip-tools for better dependency management
pip install pip-tools
pip-compile requirements.in --output-file requirements-lock.txt
```

### Performance Optimization

**System-Level Optimizations**
Environment configuration significantly impacts training performance:

**CUDA and cuDNN Optimization**:
```bash
# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=0  # Disable for production
export TORCH_CUDNN_OPTIMIZATIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Deterministic ops

# Memory allocation optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**CPU Performance Tuning**:
```python
import torch

# Optimize CPU threading
torch.set_num_threads(4)  # Set based on available cores
torch.set_num_interop_threads(1)

# Enable optimized CPU operations
torch.backends.mkldnn.enabled = True
torch.backends.mkl.enabled = True
```

**Memory Management Configuration**:
```python
# Configure memory allocation strategy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Enable memory efficient attention (for compatible models)
torch.backends.cuda.enable_flash_sdp(True)
```

### Development Workflow Integration

**Automated Testing Setup**:
```python
# pytest configuration for PyTorch projects
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    integration: marks tests as integration tests
```

**Continuous Integration Pipeline**:
```yaml
# .github/workflows/test.yml
name: Test PyTorch Project

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, '3.10']
        pytorch-version: [1.13.1, 2.0.1]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install PyTorch ${{ matrix.pytorch-version }}
      run: |
        pip install torch==${{ matrix.pytorch-version }} \
                   torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Conclusion

Proper environment setup forms the foundation of successful deep learning projects. The comprehensive approach covered in this module ensures:

**Development Efficiency**: Well-configured environments enable faster iteration, easier debugging, and smoother collaboration across team members.

**Reproducibility**: Systematic environment management ensures that experiments can be reproduced across different machines, time periods, and team members.

**Performance Optimization**: Proper configuration of CUDA, system libraries, and runtime parameters can significantly impact training performance and resource utilization.

**Production Readiness**: Environment setup practices that scale from development through production deployment reduce friction in the ML lifecycle.

**Risk Mitigation**: Robust environment management prevents common issues related to dependency conflicts, version incompatibilities, and hardware-specific problems.

The investment in proper environment setup pays dividends throughout the entire deep learning project lifecycle, from initial experimentation through production deployment. By following these comprehensive guidelines, practitioners can focus on the core challenges of deep learning rather than wrestling with environmental issues.