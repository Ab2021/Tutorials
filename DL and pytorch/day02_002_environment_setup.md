# Day 2.2: Comprehensive Environment Setup and Development Configuration

## Course: Comprehensive Deep Learning with PyTorch - 45-Day Masterclass
### Day 2, Part 2: Installation Strategies and Development Environment Optimization

---

## Overview

Setting up an optimal PyTorch development environment is crucial for productive deep learning work. This module covers comprehensive installation strategies, development environment configuration, dependency management, and optimization techniques for different computing environments from local development to cloud-based training.

## Learning Objectives

By the end of this module, you will:
- Master multiple PyTorch installation strategies and dependency management
- Configure optimal development environments for different use cases
- Understand CUDA installation and GPU acceleration setup
- Set up reproducible environments using containers and virtual environments
- Optimize development workflows for maximum productivity

---

## 1. Installation Strategies and Dependency Management

### 1.1 PyTorch Installation Options

#### Package Manager Comparison

**Conda Installation (Recommended):**
Conda provides the most reliable PyTorch installation experience:

**Advantages:**
- **Binary compatibility:** Pre-compiled binaries optimized for different hardware
- **Dependency resolution:** Handles complex dependency graphs automatically
- **CUDA integration:** Seamless CUDA toolkit installation and version management
- **Environment isolation:** Strong environment isolation capabilities
- **Cross-platform:** Consistent experience across Windows, macOS, and Linux

**Installation Commands:**
```bash
# CPU-only installation
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU installation with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# GPU installation with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Channel Management:**
- **pytorch channel:** Official PyTorch packages
- **nvidia channel:** CUDA libraries and drivers
- **conda-forge:** Community packages with broader library support
- **Priority ordering:** Control package source precedence

**Pip Installation:**
Alternative installation method with different trade-offs:

**Advantages:**
- **Simplicity:** Single package manager for Python packages
- **Speed:** Faster installation for CPU-only setups
- **Compatibility:** Works with existing pip workflows
- **Minimal overhead:** Lighter than conda environment

**Installation Commands:**
```bash
# CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU installation with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU installation with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Considerations:**
- **CUDA compatibility:** Must manually ensure CUDA toolkit compatibility
- **Dependency conflicts:** More prone to dependency resolution issues
- **System libraries:** May require manual installation of system dependencies

#### Version Management Strategies

**Semantic Versioning Understanding:**
PyTorch follows semantic versioning (MAJOR.MINOR.PATCH):

**Version Components:**
- **Major (1.x):** Breaking changes and major new features
- **Minor (x.13):** New features with backward compatibility
- **Patch (x.x.1):** Bug fixes and minor improvements

**Release Cycle:**
- **Quarterly releases:** New minor versions every 3-4 months
- **Patch releases:** Bug fixes released as needed
- **LTS versions:** Long-term support versions for production use
- **Nightly builds:** Cutting-edge features for development

**Version Selection Guidelines:**

**For Production:**
```bash
# Use LTS versions for stability
conda install pytorch=1.13.1 torchvision torchaudio -c pytorch
```

**For Research:**
```bash
# Use latest stable for newest features
conda install pytorch torchvision torchaudio -c pytorch
```

**For Development:**
```bash
# Use nightly for cutting-edge features
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

#### Environment Isolation

**Conda Environments:**
Best practice for managing multiple projects:

**Environment Creation:**
```bash
# Create new environment with specific Python version
conda create -n pytorch_env python=3.10

# Activate environment
conda activate pytorch_env

# Install PyTorch in environment
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# List environments
conda env list

# Export environment
conda env export > environment.yml

# Create from exported environment
conda env create -f environment.yml
```

**Virtual Environments (venv):**
Alternative for pip-based installations:

```bash
# Create virtual environment
python -m venv pytorch_venv

# Activate (Linux/macOS)
source pytorch_venv/bin/activate

# Activate (Windows)
pytorch_venv\Scripts\activate

# Install PyTorch
pip install torch torchvision torchaudio

# Generate requirements file
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### 1.2 CUDA Installation and GPU Setup

#### CUDA Toolkit Installation

**CUDA Version Compatibility:**
Understanding CUDA compatibility is crucial for GPU acceleration:

**CUDA Driver vs Runtime:**
- **CUDA Driver:** Low-level driver for GPU communication
- **CUDA Runtime:** Libraries and tools for CUDA development
- **Compatibility:** Runtime version must be ≤ Driver version

**Checking CUDA Installation:**
```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA compiler version
nvcc --version

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

**CUDA Installation Methods:**

**Method 1: CUDA Toolkit from NVIDIA:**
```bash
# Download from NVIDIA website and install
# Ubuntu/Debian example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

**Method 2: Conda CUDA Installation:**
```bash
# Install CUDA toolkit via conda (easier)
conda install cudatoolkit=11.8 -c conda-forge
```

#### Multi-GPU Configuration

**GPU Detection and Management:**
```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# List GPU names
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set default GPU
torch.cuda.set_device(0)
```

**Memory Management:**
```python
# Check GPU memory
print(f"Allocated: {torch.cuda.memory_allocated(0)}")
print(f"Cached: {torch.cuda.memory_reserved(0)}")

# Clear cache
torch.cuda.empty_cache()

# Memory profiling
torch.cuda.memory_summary(device=0)
```

**Environment Variables:**
```bash
# Limit visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Set memory growth (for TensorFlow compatibility)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
```

### 1.3 Development Environment Optimization

#### IDE and Editor Configuration

**Visual Studio Code Setup:**
Optimal configuration for PyTorch development:

**Essential Extensions:**
- **Python:** Official Python extension with IntelliSense
- **Pylance:** Fast Python language server
- **Jupyter:** Notebook support within VS Code
- **Python Docstring Generator:** Automated docstring creation
- **GitLens:** Enhanced Git integration
- **Remote Development:** SSH, containers, WSL support

**VS Code Settings for PyTorch:**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "jupyter.askForKernelRestart": false,
    "files.associations": {
        "*.py": "python"
    }
}
```

**PyCharm Configuration:**
Professional IDE optimized for Python development:

**Advantages:**
- **Advanced debugging:** Sophisticated debugging capabilities
- **Refactoring tools:** Powerful code refactoring features
- **Database integration:** Built-in database tools
- **Version control:** Integrated Git workflow

**PyTorch-specific settings:**
- **Scientific mode:** Enable for data science workflows
- **Conda integration:** Seamless environment management
- **Jupyter support:** Built-in Jupyter notebook support
- **Code inspection:** Advanced code quality analysis

**Jupyter Lab/Notebook Configuration:**
Interactive development environment:

**Installation and Setup:**
```bash
# Install JupyterLab
conda install jupyterlab

# Install extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Configure for PyTorch
pip install ipywidgets matplotlib seaborn plotly
```

**Jupyter Configuration:**
```python
# Display configuration for better plots
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Memory profiling
%load_ext memory_profiler
```

#### Code Quality and Formatting

**Code Formatting Tools:**

**Black (Recommended):**
```bash
# Install black
pip install black

# Format code
black my_script.py

# Configuration in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
```

**isort for Import Sorting:**
```bash
# Install isort
pip install isort

# Sort imports
isort my_script.py

# Configuration
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
```

**Linting Tools:**

**Pylint Configuration:**
```bash
# Install pylint
pip install pylint

# Run pylint
pylint my_script.py

# Configuration in .pylintrc
[MASTER]
disable = C0114,C0116,R0903,R0913
```

**Flake8 Alternative:**
```bash
# Install flake8
pip install flake8

# Configuration in .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,dist
```

#### Pre-commit Hooks

**Automated Code Quality:**
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    -   id: black
-   repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
    -   id: isort
-   repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
    -   id: flake8

# Install hooks
pre-commit install
```

---

## 2. Containerization and Reproducible Environments

### 2.1 Docker for PyTorch Development

#### Base Image Selection

**Official PyTorch Images:**
NVIDIA provides optimized PyTorch containers:

**Image Types:**
- **pytorch/pytorch:** Official PyTorch images
- **nvcr.io/nvidia/pytorch:** NVIDIA-optimized images with latest CUDA
- **jupyter/pytorch-notebook:** Jupyter-ready PyTorch environment
- **custom builds:** Tailored for specific requirements

**Example Dockerfile:**
```dockerfile
# Use NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set working directory
WORKDIR /workspace

# Install additional packages
RUN pip install pandas scikit-learn matplotlib seaborn \
    jupyter jupyterlab plotly wandb tensorboard

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Expose ports for Jupyter
EXPOSE 8888 6006

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

**Multi-stage Builds:**
Optimize image size and build efficiency:

```dockerfile
# Development stage
FROM nvcr.io/nvidia/pytorch:23.08-py3 AS development
WORKDIR /workspace
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Production stage
FROM nvcr.io/nvidia/pytorch:23.08-py3 AS production
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=development /workspace/src ./src
CMD ["python", "src/main.py"]
```

#### Docker Compose for Complex Setups

**Multi-service Development:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  pytorch-dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    volumes:
      - .:/workspace
      - pytorch_cache:/root/.cache/torch
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    runtime: nvidia
    tty: true
    stdin_open: true

  mlflow:
    image: python:3.10
    command: bash -c "pip install mlflow && mlflow server --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlruns

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pytorch_cache:
  mlflow_data:
```

**GPU Access Configuration:**
```yaml
# GPU support in docker-compose
services:
  pytorch-gpu:
    image: nvcr.io/nvidia/pytorch:23.08-py3
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2.2 Environment Reproducibility

#### Version Pinning Strategies

**Requirements File Best Practices:**
```txt
# requirements.txt - Pin exact versions for reproducibility
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
jupyterlab==4.0.5

# Development dependencies
pytest==7.4.0
black==23.7.0
isort==5.12.0
pylint==2.17.5
```

**Conda Environment Files:**
```yaml
# environment.yml
name: pytorch-env
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10.12
  - pytorch=2.0.1
  - torchvision=0.15.2
  - torchaudio=2.0.2
  - pytorch-cuda=11.8
  - numpy=1.24.3
  - pandas=2.0.3
  - matplotlib=3.7.2
  - scikit-learn=1.3.0
  - jupyterlab=4.0.5
  - pip
  - pip:
    - wandb==0.15.8
    - tensorboard==2.13.0
```

#### Dependency Lock Files

**Pip-tools for Dependency Management:**
```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in with high-level dependencies
# requirements.in
torch
torchvision
torchaudio
numpy
pandas
matplotlib
scikit-learn
jupyterlab

# Generate locked requirements.txt
pip-compile requirements.in

# Install from locked file
pip-sync requirements.txt
```

**Poetry for Advanced Dependency Management:**
```toml
# pyproject.toml
[tool.poetry]
name = "pytorch-project"
version = "0.1.0"
description = "PyTorch deep learning project"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.0.1"
torchvision = "^0.15.2"
torchaudio = "^2.0.2"
numpy = "^1.24.3"
pandas = "^2.0.3"
matplotlib = "^3.7.2"
scikit-learn = "^1.3.0"
jupyterlab = "^4.0.5"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.7.0"
isort = "^5.12.0"
pylint = "^2.17.5"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 2.3 Development Workflow Optimization

#### Git Integration and Version Control

**Repository Structure:**
```
pytorch-project/
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── src/
│   ├── __init__.py
│   ├── models/
│   ├── data/
│   ├── training/
│   └── utils/
├── notebooks/
│   ├── exploration/
│   └── experiments/
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
│   ├── model_configs/
│   └── training_configs/
└── scripts/
    ├── train.py
    └── evaluate.py
```

**Gitignore for PyTorch:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# PyTorch
*.pth
*.pt
checkpoints/
wandb/
tensorboard_logs/

# Data
data/raw/
data/processed/
*.csv
*.h5
*.pkl

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

#### Automated Testing Setup

**Pytest Configuration:**
```python
# tests/conftest.py
import pytest
import torch
import numpy as np

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_data():
    torch.manual_seed(42)
    return torch.randn(32, 3, 224, 224)

@pytest.fixture
def sample_labels():
    torch.manual_seed(42)
    return torch.randint(0, 10, (32,))
```

**Test Examples:**
```python
# tests/test_models.py
import torch
import pytest
from src.models import SimpleNet

class TestSimpleNet:
    def test_forward_pass(self, device, sample_data):
        model = SimpleNet(num_classes=10).to(device)
        output = model(sample_data.to(device))
        assert output.shape == (32, 10)
    
    def test_gradient_flow(self, device, sample_data, sample_labels):
        model = SimpleNet(num_classes=10).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        
        output = model(sample_data.to(device))
        loss = criterion(output, sample_labels.to(device))
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
```

**GitHub Actions CI/CD:**
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements.txt
        pip install pytest black isort pylint
    
    - name: Lint with black
      run: black --check src/ tests/
    
    - name: Sort imports with isort
      run: isort --check-only src/ tests/
    
    - name: Lint with pylint
      run: pylint src/
    
    - name: Test with pytest
      run: pytest tests/ -v
```

---

## 3. Performance Optimization and Monitoring

### 3.1 Development Environment Performance

#### Memory Management

**Memory Profiling Tools:**
```python
# Memory profiling with memory_profiler
from memory_profiler import profile

@profile
def train_model():
    model = LargeModel()
    # Training code here
    pass

# PyTorch memory profiling
import torch.profiler

def trace_memory():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Your PyTorch code here
        pass
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

**Memory Optimization Strategies:**
```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint(self.expensive_layer, x)
        return x

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### Development Server Optimization

**Jupyter Lab Configuration:**
```python
# jupyter_lab_config.py
c = get_config()

# Server settings
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''

# Performance settings
c.ServerApp.tornado_settings = {
    'max_body_size': 1073741824,  # 1GB
    'max_buffer_size': 1073741824  # 1GB
}

# Resource limits
c.ResourceUseDisplay.mem_limit = 8589934592  # 8GB
c.ResourceUseDisplay.track_cpu_percent = True
```

**VS Code Settings for Performance:**
```json
{
    "python.analysis.memory.keepLibraryAst": false,
    "python.analysis.autoImportCompletions": true,
    "python.analysis.indexing": true,
    "files.watcherExclude": {
        "**/node_modules/**": true,
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/data/**": true,
        "**/checkpoints/**": true
    }
}
```

### 3.2 Monitoring and Logging

#### Comprehensive Logging Setup

**Python Logging Configuration:**
```python
import logging
import sys
from datetime import datetime

def setup_logging(log_level="INFO", log_file=None):
    """Setup comprehensive logging for PyTorch projects."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Usage
logger = setup_logging("DEBUG", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
```

**Integration with Training Loops:**
```python
import wandb
import tensorboard
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, project_name, experiment_name):
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(f'runs/{experiment_name}')
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            }
        )
    
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            wandb.log({key: value}, step=step)
            self.logger.info(f"Step {step}: {key} = {value:.4f}")
    
    def close(self):
        self.writer.close()
        wandb.finish()
```

#### System Resource Monitoring

**Resource Monitoring Script:**
```python
import psutil
import GPUtil
import threading
import time
from collections import deque

class SystemMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.running = False
        self.metrics = {
            'cpu_percent': deque(maxlen=100),
            'memory_percent': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100),
            'gpu_memory': deque(maxlen=100)
        }
    
    def start_monitoring(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        while self.running:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_util = gpus[0].load * 100
                gpu_memory = gpus[0].memoryUtil * 100
            else:
                gpu_util = 0
                gpu_memory = 0
            
            # Store metrics
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_percent'].append(memory_percent)
            self.metrics['gpu_utilization'].append(gpu_util)
            self.metrics['gpu_memory'].append(gpu_memory)
            
            time.sleep(self.interval)
    
    def get_current_metrics(self):
        return {
            'cpu': list(self.metrics['cpu_percent'])[-1] if self.metrics['cpu_percent'] else 0,
            'memory': list(self.metrics['memory_percent'])[-1] if self.metrics['memory_percent'] else 0,
            'gpu_util': list(self.metrics['gpu_utilization'])[-1] if self.metrics['gpu_utilization'] else 0,
            'gpu_memory': list(self.metrics['gpu_memory'])[-1] if self.metrics['gpu_memory'] else 0
        }
```

---

## 4. Key Questions and Answers

### Beginner Level Questions

**Q1: Should I use Conda or Pip for PyTorch installation?**
**A:** Conda is generally recommended for PyTorch because:
- **Better dependency management:** Handles CUDA and other system libraries automatically
- **Pre-compiled binaries:** Optimized binaries for different hardware configurations
- **Environment isolation:** Strong environment management capabilities
- **Cross-platform consistency:** Works reliably across Windows, macOS, and Linux
However, pip is fine for simple CPU-only installations or if you're already using pip-based workflows.

**Q2: How do I know if my GPU is working with PyTorch?**
**A:** Check GPU availability with these commands:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0))  # GPU name
```
If these return False or 0, check your CUDA installation and driver versions.

**Q3: What Python version should I use with PyTorch?**
**A:** Use Python 3.8 or newer. PyTorch supports:
- **Python 3.8:** Stable, widely supported
- **Python 3.9:** Good balance of features and stability
- **Python 3.10:** Latest features, good compatibility
- **Python 3.11:** Newest, but check library compatibility
Python 3.9 or 3.10 are recommended for most users.

**Q4: How much disk space do I need for a PyTorch environment?**
**A:** Typical space requirements:
- **Basic PyTorch installation:** 2-3 GB
- **Full development environment:** 5-8 GB
- **With datasets and models:** 20+ GB
- **CUDA toolkit:** Additional 3-4 GB
Plan for at least 10-15 GB for a comfortable development setup.

### Intermediate Level Questions

**Q5: How do I set up PyTorch for multiple projects with different requirements?**
**A:** Use environment isolation:
```bash
# Create separate environments for each project
conda create -n project1 python=3.9 pytorch=1.12
conda create -n project2 python=3.10 pytorch=2.0

# Or use different CUDA versions
conda create -n old_project pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda create -n new_project pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Q6: What's the difference between different CUDA versions in PyTorch?**
**A:** CUDA versions affect:
- **Performance:** Newer CUDA versions may have optimizations
- **Hardware support:** Older GPUs may not support newest CUDA
- **Library compatibility:** Some libraries require specific CUDA versions
- **Driver requirements:** Newer CUDA needs newer drivers
Choose the CUDA version that matches your GPU capabilities and other library requirements.

**Q7: How do I optimize my development environment for large models?**
**A:** Several strategies:
- **Use mixed precision training:** Reduces memory usage
- **Enable gradient checkpointing:** Trade computation for memory
- **Optimize data loading:** Use multiple workers and pin memory
- **Monitor memory usage:** Profile and optimize memory bottlenecks
- **Use efficient data formats:** HDF5, parquet for large datasets

### Advanced Level Questions

**Q8: How do I create a reproducible environment that works across different machines?**
**A:** Comprehensive reproducibility strategy:
```bash
# Pin exact versions
conda env export --no-builds > environment.yml

# Use Docker for complete reproducibility
# Include OS, system libraries, and exact package versions

# Set random seeds in code
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Use deterministic algorithms where possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Q9: How do I handle CUDA version conflicts between different libraries?**
**A:** Resolution strategies:
- **Use conda environments:** Isolate conflicting requirements
- **Check compatibility matrices:** Verify PyTorch-CUDA-library compatibility
- **Use official compatibility guides:** Follow PyTorch and NVIDIA recommendations
- **Consider Docker:** Containerize entire stack for complex compatibility
- **Pin specific versions:** Lock to known working combinations

**Q10: What's the best way to monitor resource usage during development?**
**A:** Multi-layered monitoring approach:
```python
# Built-in PyTorch profiling
torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
)

# System monitoring
import psutil, GPUtil
# Monitor CPU, memory, GPU usage

# Integration with experiment tracking
wandb.log({"gpu_memory": torch.cuda.memory_allocated()})

# Custom monitoring dashboard
# Combine metrics in real-time dashboard
```

---

## 5. Tricky Questions for Deep Understanding

### Environment Complexity

**Q1: Why might the same PyTorch code run at different speeds on the same hardware with different installations?**
**A:** Multiple factors affect performance beyond the code:

**Compilation and optimization differences:**
- **CUDA version:** Different CUDA versions have different optimizations
- **cuDNN version:** Deep learning library optimizations vary between versions
- **PyTorch compilation:** Different builds may have different optimizations enabled
- **CPU libraries:** BLAS/LAPACK implementations (MKL vs OpenBLAS) significantly affect performance

**System configuration:**
- **CPU governor settings:** Power management affects CPU performance
- **GPU boost clocks:** Thermal and power limits affect GPU performance
- **Memory configuration:** RAM speed and configuration affect data loading
- **Background processes:** Other applications competing for resources

**Python environment:**
- **Python version:** Newer Python versions may have performance improvements
- **Package versions:** NumPy, SciPy versions affect underlying computations
- **Environment overhead:** Conda vs pip vs native installations have different overhead

**Q2: How can virtual environments sometimes make code run faster rather than slower?**
**A:** This counterintuitive situation occurs due to:

**Dependency optimization:**
- **Cleaner dependencies:** Fewer conflicting packages mean better optimization
- **Optimized builds:** Environment-specific builds may be more optimized
- **Reduced overhead:** Fewer loaded modules reduce import and memory overhead

**System resource isolation:**
- **Memory management:** Isolated environments may have better memory locality
- **Process isolation:** Less interference from other Python processes
- **Cache efficiency:** Smaller working set may fit better in CPU cache

**Library version benefits:**
- **Newer optimizations:** Isolated environment may use newer, faster library versions
- **Better compatibility:** Matching library versions may enable better optimizations
- **Platform-specific builds:** Environment may have platform-specific optimized builds

### Installation Paradoxes

**Q3: Why might a "simpler" installation method sometimes cause more problems?**
**A:** Simple installation methods can hide complexity that later causes issues:

**Dependency resolution hiding:**
- **Pip limitations:** Pip doesn't handle system dependencies (CUDA, BLAS libraries)
- **Version conflicts:** Simple installs may not check for version compatibility
- **System library conflicts:** May not detect conflicting system libraries

**Platform-specific issues:**
- **Binary compatibility:** Pre-built binaries may not match your exact system
- **Optimization flags:** Generic builds may miss system-specific optimizations
- **Missing features:** Simple installs may omit optional but important features

**Hidden assumptions:**
- **System state:** Simple installs assume clean, standard system configuration
- **User permissions:** May fail silently with permission issues
- **Path configuration:** May not properly set up environment variables

**Q4: When might you want multiple PyTorch installations on the same system?**
**A:** Several legitimate scenarios:

**Development vs production:**
- **Different optimization levels:** Development with debug symbols, production optimized
- **Different backends:** Research with latest features, production with stable versions
- **Security considerations:** Production with minimal dependencies

**Multi-user systems:**
- **User-specific requirements:** Different users need different PyTorch versions
- **Project isolation:** Different projects with incompatible requirements
- **Teaching environments:** Multiple class versions or student projects

**Research requirements:**
- **Reproducibility:** Maintain exact versions for reproducing published results
- **Comparison studies:** Compare performance across different PyTorch versions
- **Legacy support:** Maintain old versions for existing projects

### Performance Philosophy

**Q5: Is it better to optimize the environment or the code?**
**A:** This is a classic systems optimization question with nuanced answers:

**Environment optimization benefits:**
- **One-time cost:** Setup once, benefits all code
- **Broader impact:** Benefits all applications, not just current project
- **Knowledge transfer:** Environment skills transfer across projects
- **Foundation building:** Creates solid base for all development

**Code optimization benefits:**
- **Algorithmic improvements:** Often provide orders of magnitude improvements
- **Portability:** Code optimizations work across different environments
- **Scalability:** Good algorithms scale better than environment optimizations
- **Problem-specific:** Can target specific bottlenecks in your use case

**Balanced approach:**
- **Start with environment:** Ensure you have solid foundation
- **Profile to identify:** Use data to identify actual bottlenecks
- **Optimize systematically:** Address biggest impact items first
- **Measure improvements:** Quantify benefits of each optimization

**Q6: How do you balance reproducibility with staying current with latest improvements?**
**A:** This tension requires strategic thinking:

**Reproducibility requirements:**
- **Research publishing:** Need exact version replication
- **Production stability:** Avoid breaking changes
- **Legal compliance:** Some industries require version documentation
- **Team coordination:** Everyone needs same working environment

**Innovation benefits:**
- **Performance improvements:** Newer versions often faster
- **Bug fixes:** Security and correctness improvements
- **New features:** Access to latest capabilities
- **Community support:** Better support for current versions

**Strategic approaches:**
- **Staged adoption:** Use latest for exploration, stable for production
- **Version testing:** Regularly test upgrades in isolated environments
- **Dependency management:** Use lockfiles for reproducibility, requirements files for flexibility
- **Documentation:** Maintain clear records of version choices and rationale

---

## Summary and Best Practices

### Environment Setup Decision Framework

**For Beginners:**
1. **Use Conda:** More reliable dependency management
2. **Start with CPU:** Get familiar before adding GPU complexity
3. **Use stable versions:** Avoid nightly builds until experienced
4. **Single environment:** Keep it simple initially

**For Research:**
1. **Environment per project:** Isolate different experiment requirements
2. **Version pinning:** Enable reproducible results
3. **Comprehensive logging:** Track environment alongside results
4. **Backup strategies:** Save environment configurations with results

**For Production:**
1. **Containerization:** Docker for deployment consistency
2. **Minimal dependencies:** Only include necessary packages
3. **Security scanning:** Regular vulnerability assessments
4. **Performance optimization:** Profile and optimize for target hardware

### Future-Proofing Strategies

**Technology Evolution:**
- **Monitor PyTorch roadmap:** Stay informed about upcoming changes
- **Participate in community:** Engage with PyTorch community for early insights
- **Experiment with nightly builds:** Test upcoming features safely
- **Maintain upgrade paths:** Plan for major version transitions

**Infrastructure Evolution:**
- **Cloud-native approaches:** Prepare for serverless and managed ML services
- **Hardware evolution:** Plan for new accelerators (TPUs, specialized chips)
- **Container orchestration:** Kubernetes and cloud ML platforms
- **Edge computing:** Optimize for mobile and IoT deployment

Understanding comprehensive environment setup enables productive PyTorch development while avoiding common pitfalls. The key is balancing simplicity with flexibility, reproducibility with innovation, and immediate productivity with long-term maintainability.

---

## Next Steps

In our next module, we'll explore the broader PyTorch ecosystem, diving into specialized libraries like TorchVision, TorchText, and PyTorch Lightning that build upon the solid foundation we've established.