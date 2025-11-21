# Day 10 Deep Dive: Production Deep Learning

## 1. Model Deployment Strategies

### ONNX Export
**Open Neural Network Exchange:** Framework-agnostic model format.

```python
import torch.onnx

def export_to_onnx(model, dummy_input, output_path='model.onnx'):
    """Export PyTorch model to ONNX."""
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

# Usage
model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, dummy_input)
```

### TorchScript
**JIT compilation for deployment.**

```python
def export_to_torchscript(model, example_input):
    """Export to TorchScript."""
    model.eval()
    
    # Tracing
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('model_traced.pt')
    
    # Scripting (for control flow)
    scripted_model = torch.jit.script(model)
    scripted_model.save('model_scripted.pt')
    
    return traced_model, scripted_model

# Load and use
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(input_tensor)
```

## 2. Model Optimization

### Quantization
**Reduce precision from FP32 to INT8.**

```python
import torch.quantization

def quantize_model(model, calibration_loader):
    """Post-training static quantization."""
    # Fuse modules
    model.eval()
    model_fused = torch.quantization.fuse_modules(
        model,
        [['conv', 'bn', 'relu']]
    )
    
    # Prepare for quantization
    model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_fused, inplace=True)
    
    # Calibrate
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model_fused(inputs)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_fused, inplace=False)
    
    return model_quantized

# Dynamic quantization (simpler, for RNNs/LSTMs)
model_dynamic = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
```

**Benefits:**
- 4× smaller model size
- 2-4× faster inference
- Minimal accuracy loss (<1%)

### Pruning
**Remove unnecessary weights.**

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """Structured pruning."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # L1 unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=amount)
            
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    return model

# Iterative pruning
def iterative_pruning(model, train_fn, prune_amount=0.2, iterations=5):
    """Iterative magnitude pruning."""
    for i in range(iterations):
        # Prune
        model = prune_model(model, amount=prune_amount)
        
        # Fine-tune
        train_fn(model, epochs=10)
        
        print(f"Iteration {i+1}: Pruned {prune_amount * 100}%")
    
    return model
```

### Knowledge Distillation (Production)
```python
class ProductionDistiller:
    """Production-ready knowledge distillation."""
    
    def __init__(self, teacher, student, temperature=3.0, alpha=0.3):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distill(self, train_loader, epochs=50, lr=0.001):
        """Distillation training loop."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        criterion_hard = nn.CrossEntropyLoss()
        criterion_soft = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(epochs):
            self.student.train()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                # Student predictions
                student_logits = self.student(inputs)
                
                # Hard loss
                loss_hard = criterion_hard(student_logits, labels)
                
                # Soft loss
                T = self.temperature
                loss_soft = criterion_soft(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T * T)
                
                # Combined loss
                loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        return self.student
```

## 3. Multi-GPU Training

### DataParallel
**Simple multi-GPU training.**

```python
# Wrap model
model = nn.DataParallel(model)
model = model.cuda()

# Train normally
for inputs, labels in train_loader:
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = model(inputs)  # Automatically distributed
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### DistributedDataParallel (Recommended)
**More efficient multi-GPU/multi-node training.**

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """Training function for each process."""
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch training
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

## 4. Experiment Tracking

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    """Track experiments with TensorBoard."""
    
    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def log_metrics(self, metrics, step=None):
        """Log scalar metrics."""
        step = step if step is not None else self.global_step
        
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        
        self.global_step += 1
    
    def log_images(self, tag, images, step=None):
        """Log images."""
        step = step if step is not None else self.global_step
        self.writer.add_images(tag, images, step)
    
    def log_model_graph(self, model, input_size):
        """Log model architecture."""
        dummy_input = torch.randn(1, *input_size)
        self.writer.add_graph(model, dummy_input)
    
    def log_hyperparameters(self, hparams, metrics):
        """Log hyperparameters and results."""
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close writer."""
        self.writer.close()

# Usage
tracker = ExperimentTracker('runs/resnet50_experiment')

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    tracker.log_metrics({
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

tracker.close()
```

## 5. Model Versioning and Registry

```python
import json
import hashlib
from pathlib import Path

class ModelRegistry:
    """Simple model registry for versioning."""
    
    def __init__(self, registry_dir='model_registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.metadata_file = self.registry_dir / 'metadata.json'
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def register_model(self, model, name, version, metrics, hyperparams):
        """Register a trained model."""
        # Save model
        model_path = self.registry_dir / f'{name}_v{version}.pth'
        torch.save(model.state_dict(), model_path)
        
        # Compute checksum
        with open(model_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # Store metadata
        self.metadata[f'{name}_v{version}'] = {
            'path': str(model_path),
            'version': version,
            'metrics': metrics,
            'hyperparams': hyperparams,
            'checksum': checksum,
            'timestamp': str(pd.Timestamp.now())
        }
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Registered {name} v{version}")
    
    def load_model(self, name, version, model_class):
        """Load a registered model."""
        key = f'{name}_v{version}'
        
        if key not in self.metadata:
            raise ValueError(f"Model {key} not found in registry")
        
        model_info = self.metadata[key]
        model = model_class()
        model.load_state_dict(torch.load(model_info['path']))
        
        return model, model_info
    
    def list_models(self):
        """List all registered models."""
        return list(self.metadata.keys())

# Usage
registry = ModelRegistry()

# Register model
registry.register_model(
    model=trained_model,
    name='resnet50_imagenet',
    version='1.0',
    metrics={'val_acc': 0.923, 'val_loss': 0.234},
    hyperparams={'lr': 0.001, 'batch_size': 32, 'epochs': 100}
)

# Load model
model, info = registry.load_model('resnet50_imagenet', '1.0', ResNet50)
print(f"Loaded model with accuracy: {info['metrics']['val_acc']}")
```

## 6. Production Inference Server

```python
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
model = load_production_model()
model.eval()
model = model.cuda()

# Preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Load image
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).cuda()
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        top5_prob, top5_idx = probabilities.topk(5, dim=1)
    
    # Format response
    predictions = []
    for prob, idx in zip(top5_prob[0], top5_idx[0]):
        predictions.append({
            'class': int(idx),
            'probability': float(prob)
        })
    
    return jsonify({'predictions': predictions})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Summary
Production deep learning requires deployment strategies (ONNX, TorchScript), optimization (quantization, pruning), multi-GPU training, experiment tracking, and robust inference serving.
