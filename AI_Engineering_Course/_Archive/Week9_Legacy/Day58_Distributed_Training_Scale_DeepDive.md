# Day 58: Distributed Training at Scale
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Distributed Data Parallel (DDP) Implementation

```python
import torch
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
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Move model to GPU
        self.model = model.to(rank)
        
        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
    
    def train_epoch(self, dataloader, optimizer):
        """Train one epoch with DDP."""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.rank)
            targets = targets.to(self.rank)
            
            optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            loss.backward()  # Gradients automatically synchronized
            optimizer.step()
            
            total_loss += loss.item()
        
        # Average loss across GPUs
        total_loss = torch.tensor(total_loss).to(self.rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / self.world_size / len(dataloader)
        
        return avg_loss

def train_ddp(rank, world_size, model, train_dataset):
    """Main training function for DDP."""
    setup_distributed(rank, world_size)
    
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
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = DistributedTrainer(model, rank, world_size)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        loss = trainer.train_epoch(dataloader, optimizer)
        
        if rank == 0:  # Only print from rank 0
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    cleanup_distributed()

# Launch with torchrun
# torchrun --nproc_per_node=8 train.py
```

### 2. FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
import functools

class FSDPTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Auto-wrap policy (wrap layers with >1M params)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e6
        )
        
        # Mixed precision
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        
        # Wrap model with FSDP
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=rank,
            sharding_strategy="FULL_SHARD",  # ZeRO-3
            cpu_offload=None  # Can offload to CPU for even more memory
        )
    
    def train_epoch(self, dataloader, optimizer):
        """Train one epoch with FSDP."""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.rank)
            targets = targets.to(self.rank)
            
            optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

### 3. Pipeline Parallelism

```python
class PipelineParallelModel(nn.Module):
    def __init__(self, num_layers=24, num_gpus=4):
        super().__init__()
        self.num_gpus = num_gpus
        self.layers_per_gpu = num_layers // num_gpus
        
        # Split layers across GPUs
        self.layer_groups = nn.ModuleList()
        for gpu_id in range(num_gpus):
            layers = nn.ModuleList([
                TransformerLayer()
                for _ in range(self.layers_per_gpu)
            ])
            self.layer_groups.append(layers.to(gpu_id))
    
    def forward(self, x, num_microbatches=4):
        """Forward pass with pipeline parallelism."""
        batch_size = x.shape[0]
        microbatch_size = batch_size // num_microbatches
        
        # Split into micro-batches
        microbatches = x.split(microbatch_size, dim=0)
        
        outputs = []
        for mb in microbatches:
            # Process through pipeline
            for gpu_id, layers in enumerate(self.layer_groups):
                mb = mb.to(gpu_id)
                for layer in layers:
                    mb = layer(mb)
            
            outputs.append(mb)
        
        # Concatenate outputs
        return torch.cat(outputs, dim=0)
```

### 4. Tensor Parallelism (Megatron-LM Style)

```python
class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer."""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.out_features_per_partition = out_features // world_size
        
        # Each GPU has a slice of the weight matrix
        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_partition, in_features)
        )
    
    def forward(self, x):
        """Forward pass with column parallelism."""
        # All GPUs compute their partition
        output_parallel = F.linear(x, self.weight)
        
        # No communication needed (output is partitioned)
        return output_parallel

class RowParallelLinear(nn.Module):
    """Row-parallel linear layer."""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.in_features_per_partition = in_features // world_size
        
        # Each GPU has a slice of the weight matrix
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_partition)
        )
    
    def forward(self, x):
        """Forward pass with row parallelism."""
        # Input is partitioned across GPUs
        output_parallel = F.linear(x, self.weight)
        
        # All-reduce to combine results
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        
        return output_parallel

class TensorParallelAttention(nn.Module):
    """Tensor-parallel multi-head attention."""
    def __init__(self, hidden_size, num_heads, world_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.world_size = world_size
        
        # Column-parallel Q, K, V projections
        self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, world_size)
        self.k_proj = ColumnParallelLinear(hidden_size, hidden_size, world_size)
        self.v_proj = ColumnParallelLinear(hidden_size, hidden_size, world_size)
        
        # Row-parallel output projection
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, world_size)
    
    def forward(self, x):
        """Forward pass with tensor parallelism."""
        # Each GPU computes a subset of heads
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Attention (local to each GPU)
        attn_output = self._attention(Q, K, V)
        
        # Output projection (all-reduce inside)
        output = self.o_proj(attn_output)
        
        return output
```

### 5. 3D Parallelism Coordinator

```python
class ThreeDParallelism:
    def __init__(
        self,
        model,
        data_parallel_size=64,
        pipeline_parallel_size=8,
        tensor_parallel_size=8
    ):
        self.dp_size = data_parallel_size
        self.pp_size = pipeline_parallel_size
        self.tp_size = tensor_parallel_size
        
        # Total GPUs
        self.world_size = dp_size * pp_size * tp_size
        
        # Create process groups
        self._create_process_groups()
        
        # Apply parallelism
        self.model = self._parallelize_model(model)
    
    def _create_process_groups(self):
        """Create process groups for each parallelism dimension."""
        # Data parallel group
        self.dp_group = dist.new_group(ranks=self._get_dp_ranks())
        
        # Pipeline parallel group
        self.pp_group = dist.new_group(ranks=self._get_pp_ranks())
        
        # Tensor parallel group
        self.tp_group = dist.new_group(ranks=self._get_tp_ranks())
    
    def _parallelize_model(self, model):
        """Apply 3D parallelism to model."""
        # 1. Tensor parallelism (within layer)
        model = apply_tensor_parallelism(model, self.tp_group)
        
        # 2. Pipeline parallelism (across layers)
        model = apply_pipeline_parallelism(model, self.pp_group)
        
        # 3. Data parallelism (across batches)
        model = DDP(model, process_group=self.dp_group)
        
        return model
```

### 6. Gradient Compression

```python
class GradientCompressor:
    def __init__(self, compression_ratio=0.01):
        self.compression_ratio = compression_ratio
    
    def compress(self, tensor):
        """Compress gradient via top-k sparsification."""
        numel = tensor.numel()
        k = max(1, int(numel * self.compression_ratio))
        
        # Get top-k values and indices
        values, indices = torch.topk(tensor.abs().flatten(), k)
        
        # Create sparse representation
        compressed = {
            'values': values * tensor.flatten()[indices].sign(),
            'indices': indices,
            'shape': tensor.shape
        }
        
        return compressed
    
    def decompress(self, compressed):
        """Decompress gradient."""
        tensor = torch.zeros(compressed['shape'].numel())
        tensor[compressed['indices']] = compressed['values']
        tensor = tensor.reshape(compressed['shape'])
        
        return tensor
```
