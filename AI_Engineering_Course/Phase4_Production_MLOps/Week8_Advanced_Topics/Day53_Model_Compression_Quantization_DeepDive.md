# Day 53: Model Compression & Quantization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Dynamic Quantization Implementation

```python
import torch
import torch.nn as nn
import torch.quantization as quant

def dynamic_quantize_model(model):
    """Apply dynamic quantization to model."""
    # Quantize Linear and LSTM layers
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )
    
    return quantized_model

# Usage
model = YourModel()
quantized_model = dynamic_quantize_model(model)

# Model is now 4x smaller, 2-3x faster
```

### 2. Static Quantization with Calibration

```python
class StaticQuantizer:
    def __init__(self, model, calibration_dataloader):
        self.model = model
        self.calibration_dataloader = calibration_dataloader
    
    def quantize(self):
        """Apply static quantization."""
        # 1. Fuse operations (Conv+BN+ReLU)
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']]
        )
        
        # 2. Specify quantization config
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 3. Prepare model for quantization
        torch.quantization.prepare(self.model, inplace=True)
        
        # 4. Calibrate on representative dataset
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in self.calibration_dataloader:
                self.model(inputs)
        
        # 5. Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
```

### 3. Quantization-Aware Training

```python
class QATTrainer:
    def __init__(self, model):
        self.model = model
        
        # Prepare for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
    
    def train(self, dataloader, optimizer, epochs=10):
        """Train with quantization awareness."""
        self.model.train()
        
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                loss.backward()
                optimizer.step()
            
            # Freeze batch norm after a few epochs
            if epoch >= 3:
                self.model.apply(torch.quantization.disable_observer)
        
        # Convert to quantized model
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
```

### 4. GPTQ Implementation

```python
def gptq_quantize_layer(weight, inputs, bits=4):
    """
    Quantize layer using GPTQ algorithm.
    
    Args:
        weight: (out_features, in_features)
        inputs: (num_samples, in_features)
        bits: quantization bits
    """
    # Compute Hessian (second-order information)
    H = torch.matmul(inputs.T, inputs) / inputs.shape[0]
    H_inv = torch.inverse(H + 1e-6 * torch.eye(H.shape[0], device=H.device))
    
    # Quantize weights column by column
    quantized_weight = torch.zeros_like(weight)
    error = torch.zeros_like(weight)
    
    for col in range(weight.shape[1]):
        # Current weight + accumulated error
        w_col = weight[:, col] + error[:, col]
        
        # Quantize
        q_col = quantize_to_bits(w_col, bits)
        quantized_weight[:, col] = q_col
        
        # Compute quantization error
        err = w_col - q_col
        
        # Propagate error to remaining columns using Hessian
        if col < weight.shape[1] - 1:
            error[:, col+1:] -= torch.outer(
                err,
                H_inv[col, col+1:] / H_inv[col, col]
            )
    
    return quantized_weight

def quantize_to_bits(tensor, bits):
    """Quantize tensor to specified bits."""
    n_levels = 2 ** bits
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (n_levels - 1)
    zero_point = min_val
    
    # Quantize
    quantized = torch.round((tensor - zero_point) / scale)
    quantized = quantized.clamp(0, n_levels - 1)
    
    # Dequantize
    dequantized = quantized * scale + zero_point
    
    return dequantized
```

### 5. AWQ Implementation

```python
class AWQQuantizer:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.activation_scales = {}
    
    def compute_activation_scales(self):
        """Compute activation magnitudes for each layer."""
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # Compute average activation magnitude
                self.activation_scales[name] = input[0].abs().mean(dim=0)
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run calibration
        self.model.eval()
        with torch.no_grad():
            for inputs in self.calibration_data:
                self.model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    def quantize(self, bits=4, protection_threshold=0.9):
        """Quantize with activation-aware protection."""
        self.compute_activation_scales()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Get activation scales
                scales = self.activation_scales[name]
                
                # Identify important weights (high activation)
                threshold = torch.quantile(scales, protection_threshold)
                important_mask = scales > threshold
                
                # Quantize
                weight = module.weight.data
                
                # Protect important weights (keep in FP16)
                quantized_weight = weight.clone()
                
                # Quantize unimportant weights aggressively
                unimportant_mask = ~important_mask
                quantized_weight[:, unimportant_mask] = quantize_to_bits(
                    weight[:, unimportant_mask],
                    bits
                )
                
                module.weight.data = quantized_weight
        
        return self.model
```

### 6. Magnitude Pruning

```python
class MagnitudePruner:
    def __init__(self, model, sparsity=0.5):
        self.model = model
        self.sparsity = sparsity
    
    def prune(self):
        """Prune weights with smallest magnitude."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Compute threshold
                threshold = torch.quantile(
                    weight.abs(),
                    self.sparsity
                )
                
                # Create mask
                mask = weight.abs() > threshold
                
                # Apply mask
                module.weight.data *= mask
        
        return self.model
    
    def iterative_prune(self, dataloader, optimizer, prune_steps=10):
        """Iteratively prune and fine-tune."""
        sparsity_per_step = self.sparsity / prune_steps
        
        for step in range(prune_steps):
            # Prune
            current_sparsity = sparsity_per_step * (step + 1)
            self.sparsity = current_sparsity
            self.prune()
            
            # Fine-tune
            self.model.train()
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return self.model
```

### 7. Knowledge Distillation

```python
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute distillation loss."""
        # Soft loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        soft_loss *= self.temperature ** 2
        
        # Hard loss (cross entropy)
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return loss
    
    def train(self, dataloader, optimizer, epochs=10):
        """Train student with distillation."""
        self.student.train()
        
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                
                # Teacher predictions (no grad)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                # Student predictions
                student_logits = self.student(inputs)
                
                # Distillation loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    targets
                )
                
                loss.backward()
                optimizer.step()
        
        return self.student
```

### 8. Low-Rank Factorization

```python
def low_rank_factorize(weight, rank):
    """Factorize weight matrix into low-rank factors."""
    # SVD decomposition
    U, S, V = torch.svd(weight)
    
    # Keep top-k singular values
    U_k = U[:, :rank]
    S_k = S[:rank]
    V_k = V[:, :rank]
    
    # Factorized matrices
    A = U_k @ torch.diag(torch.sqrt(S_k))
    B = torch.diag(torch.sqrt(S_k)) @ V_k.T
    
    return A, B

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
    
    def forward(self, x):
        return F.linear(F.linear(x, self.B), self.A)

def apply_low_rank_factorization(model, rank=64):
    """Replace Linear layers with low-rank factorization."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Factorize
            A, B = low_rank_factorize(module.weight.data, rank)
            
            # Replace with low-rank layer
            low_rank_layer = LowRankLinear(
                module.in_features,
                module.out_features,
                rank
            )
            low_rank_layer.A.data = A
            low_rank_layer.B.data = B
            
            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, low_rank_layer)
    
    return model
```
