# Day 52: Efficient Training Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Mixed Precision Training

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.device = device
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass in FP16
            with autocast():
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (in FP32)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

### 2. Gradient Accumulation Implementation

```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Normalize loss by accumulation steps
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(dataloader)
```

### 3. Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model, train_dataset):
    """Train with DDP."""
    setup_distributed(rank, world_size)
    
    # Move model to GPU
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        model.train()
        for inputs, targets in dataloader:
            inputs = inputs.to(rank)
            targets = targets.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
    
    cleanup_distributed()

# Launch with torch.multiprocessing
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_distributed,
        args=(world_size, model, train_dataset),
        nprocs=world_size,
        join=True
    )
```

### 4. FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def train_with_fsdp(rank, world_size, model):
    """Train with FSDP for memory efficiency."""
    setup_distributed(rank, world_size)
    
    # Wrap model with FSDP
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e6  # Wrap layers with >1M params
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank,
        mixed_precision=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop (same as DDP)
    for epoch in range(10):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
    
    cleanup_distributed()
```

### 5. Gradient Checkpointing

```python
import torch.utils.checkpoint as checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # Use gradient checkpointing
        x = x + checkpoint.checkpoint(self._self_attn, x)
        x = x + checkpoint.checkpoint(self.ffn, x)
        return x
    
    def _self_attn(self, x):
        return self.self_attn(x, x, x)[0]
```

### 6. DeepSpeed ZeRO

```python
import deepspeed

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: Partition params + gradients + optimizer states
        "offload_optimizer": {
            "device": "cpu"  # Offload to CPU for even more memory
        }
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training loop
for inputs, targets in dataloader:
    outputs = model_engine(inputs)
    loss = criterion(outputs, targets)
    model_engine.backward(loss)
    model_engine.step()
```

### 7. LoRA Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank = rank
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01
    
    def forward(self, x):
        # Original forward
        original_output = F.linear(x, self.weight)
        
        # LoRA forward
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        return original_output + self.scaling * lora_output

def apply_lora_to_model(model, rank=8):
    """Replace Linear layers with LoRA layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with LoRA layer
            lora_layer = LoRALayer(
                module.in_features,
                module.out_features,
                rank=rank
            )
            lora_layer.weight.data = module.weight.data.clone()
            
            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, lora_layer)
    
    return model
```

### 8. 8-bit Adam Optimizer

```python
import bitsandbytes as bnb

# Use 8-bit Adam
optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999)
)

# Training loop (same as regular Adam)
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### 9. Flash Attention Integration

```python
from flash_attn import flash_attn_func

class FlashAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Flash attention
        output = flash_attn_func(Q, K, V)
        
        # Reshape and project
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output
```
