# Day 29.3: Model Optimization and Deployment - Production-Ready PyTorch Systems

## Overview

Model Optimization and Deployment represent the critical final stages of deep learning system development where research prototypes are transformed into production-ready solutions through sophisticated optimization techniques, deployment strategies, and infrastructure patterns that ensure models meet stringent requirements for latency, throughput, memory efficiency, and reliability while maintaining prediction accuracy and business value. Understanding the mathematical foundations and practical implementation of model optimization techniques, from quantization and pruning to knowledge distillation and neural architecture search, alongside comprehensive deployment strategies including containerization, serverless computing, edge deployment, and cloud-native architectures, reveals how PyTorch models can be systematically optimized and deployed across diverse production environments with optimal performance characteristics. This comprehensive exploration examines the theoretical principles underlying model compression and acceleration techniques, the engineering practices for robust deployment infrastructure, the monitoring and maintenance strategies for production ML systems, and the emerging approaches to automated optimization that collectively enable the successful transition from experimental deep learning models to scalable, reliable, and efficient production AI systems.

## Model Compression and Quantization

### Mathematical Foundations of Quantization

**Uniform Quantization Theory**:
```python
import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np

class QuantizationTheory:
    """Mathematical foundations of neural network quantization"""
    
    @staticmethod
    def uniform_quantization(x, num_bits=8, signed=True):
        """
        Uniform quantization mapping continuous values to discrete levels
        
        For signed quantization: x ∈ [-2^(b-1), 2^(b-1) - 1]
        For unsigned: x ∈ [0, 2^b - 1]
        """
        if signed:
            qmin = -2**(num_bits - 1)
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        
        # Scale and zero-point calculation
        scale = (x.max() - x.min()) / (qmax - qmin)
        zero_point = qmin - x.min() / scale
        zero_point = torch.clamp(zero_point.round(), qmin, qmax)
        
        # Quantization
        x_quantized = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        
        # Dequantization for verification
        x_dequantized = scale * (x_quantized - zero_point)
        
        return x_quantized, scale, zero_point, x_dequantized
    
    @staticmethod
    def compute_quantization_error(x_original, x_quantized):
        """Compute quantization error metrics"""
        mse = torch.mean((x_original - x_quantized) ** 2)
        snr = 10 * torch.log10(torch.var(x_original) / mse)
        max_error = torch.max(torch.abs(x_original - x_quantized))
        
        return {
            'mse': mse.item(),
            'snr_db': snr.item(),
            'max_error': max_error.item(),
            'mean_abs_error': torch.mean(torch.abs(x_original - x_quantized)).item()
        }
    
    @staticmethod
    def calibrate_quantization_params(model, calibration_loader, num_bits=8):
        """Calibrate quantization parameters using representative data"""
        activation_stats = {}
        
        def collect_stats(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(output.detach().cpu())
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(collect_stats(name))
                hooks.append(hook)
        
        # Collect statistics
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= 100:  # Limit calibration data
                    break
                _ = model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute quantization parameters
        quantization_params = {}
        for name, activations in activation_stats.items():
            all_activations = torch.cat(activations, dim=0)
            scale = (all_activations.max() - all_activations.min()) / (2**num_bits - 1)
            zero_point = -all_activations.min() / scale
            quantization_params[name] = {'scale': scale, 'zero_point': zero_point}
        
        return quantization_params

# Advanced quantization techniques
class AdvancedQuantization:
    def __init__(self, model):
        self.model = model
        self.quantized_model = None
        
    def post_training_quantization(self, calibration_loader):
        """Post-training quantization (PTQ)"""
        # Prepare model for quantization
        self.model.eval()
        
        # Fuse operations for better performance
        fused_model = self._fuse_model()
        
        # Set quantization configuration
        fused_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(fused_model, inplace=False)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= 100:
                    break
                prepared_model(data)
        
        # Convert to quantized model
        self.quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return self.quantized_model
    
    def quantization_aware_training(self, train_loader, val_loader, num_epochs=10):
        """Quantization-Aware Training (QAT)"""
        # Fuse model
        fused_model = self._fuse_model()
        
        # Set QAT configuration
        fused_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(fused_model, inplace=False)
        
        # Training setup
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        prepared_model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = prepared_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 99:
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
                    running_loss = 0.0
            
            # Validation
            self._validate_model(prepared_model, val_loader)
        
        # Convert to quantized model
        prepared_model.eval()
        self.quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return self.quantized_model
    
    def _fuse_model(self):
        """Fuse conv-bn-relu patterns for optimization"""
        from torch.quantization import fuse_modules
        
        fused_model = torch.jit.script(self.model)  # Basic fusion approach
        return fused_model
    
    def _validate_model(self, model, val_loader):
        """Validate model performance"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        return accuracy
    
    def compare_models(self, original_model, quantized_model, test_loader):
        """Compare original and quantized model performance"""
        def evaluate_model(model, loader):
            model.eval()
            correct = 0
            total = 0
            inference_times = []
            
            with torch.no_grad():
                for data, target in loader:
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    outputs = model(data)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time))
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            avg_inference_time = np.mean(inference_times)
            
            return accuracy, avg_inference_time
        
        # Evaluate both models
        orig_acc, orig_time = evaluate_model(original_model, test_loader)
        quant_acc, quant_time = evaluate_model(quantized_model, test_loader)
        
        # Model size comparison
        orig_size = self._get_model_size(original_model)
        quant_size = self._get_model_size(quantized_model)
        
        comparison_results = {
            'original': {
                'accuracy': orig_acc,
                'inference_time_ms': orig_time,
                'model_size_mb': orig_size
            },
            'quantized': {
                'accuracy': quant_acc,
                'inference_time_ms': quant_time,
                'model_size_mb': quant_size
            },
            'improvements': {
                'accuracy_drop': orig_acc - quant_acc,
                'speed_up': orig_time / quant_time,
                'size_reduction': orig_size / quant_size
            }
        }
        
        return comparison_results
    
    def _get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024

# Custom quantized operations
class QuantizedLinear(nn.Module):
    """Custom quantized linear layer implementation"""
    
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        
        # Weight parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0))
        self.register_buffer('output_scale', torch.tensor(1.0))
        self.register_buffer('output_zero_point', torch.tensor(0))
    
    def quantize_weights(self):
        """Quantize weights to specified bit width"""
        qmin = -2**(self.num_bits - 1)
        qmax = 2**(self.num_bits - 1) - 1
        
        self.weight_scale = (self.weight.max() - self.weight.min()) / (qmax - qmin)
        self.weight_zero_point = torch.clamp(
            torch.round(qmin - self.weight.min() / self.weight_scale), 
            qmin, qmax
        )
    
    def forward(self, input):
        if self.training:
            # Simulate quantization during training
            weight_quantized = self._fake_quantize(
                self.weight, self.weight_scale, self.weight_zero_point
            )
            return torch.nn.functional.linear(input, weight_quantized, self.bias)
        else:
            # Use integer arithmetic for inference
            return self._quantized_linear(input)
    
    def _fake_quantize(self, x, scale, zero_point):
        """Fake quantization for training"""
        qmin = -2**(self.num_bits - 1)
        qmax = 2**(self.num_bits - 1) - 1
        
        x_quantized = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dequantized = scale * (x_quantized - zero_point)
        
        return x_dequantized
    
    def _quantized_linear(self, input):
        """Integer-based quantized linear operation"""
        # This would implement actual integer arithmetic
        # Simplified version using floating point simulation
        return torch.nn.functional.linear(input, self.weight, self.bias)
```

## Model Pruning and Sparsification

### Structured and Unstructured Pruning

**Advanced Pruning Techniques**:
```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from collections import OrderedDict

class AdvancedPruning:
    def __init__(self, model):
        self.model = model
        self.original_model = None
        self.pruning_history = []
        
    def magnitude_based_pruning(self, pruning_ratio=0.2):
        """Prune weights based on magnitude"""
        # Store original model for comparison
        if self.original_model is None:
            self.original_model = type(self.model)()
            self.original_model.load_state_dict(self.model.state_dict())
        
        # Apply magnitude-based pruning to all linear and conv layers
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Record pruning statistics
        stats = self._compute_pruning_stats()
        self.pruning_history.append({
            'method': 'magnitude_based',
            'ratio': pruning_ratio,
            'stats': stats
        })
        
        return self.model
    
    def structured_pruning(self, pruning_ratio=0.3):
        """Prune entire channels/neurons based on their importance"""
        
        def compute_channel_importance(conv_layer):
            """Compute importance scores for each channel"""
            with torch.no_grad():
                weight = conv_layer.weight.data
                # L1 norm of each filter
                importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
            return importance
        
        # Prune convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance_scores = compute_channel_importance(module)
                num_channels = importance_scores.size(0)
                num_to_prune = int(num_channels * pruning_ratio)
                
                if num_to_prune > 0:
                    # Get indices of least important channels
                    _, indices_to_prune = torch.topk(
                        importance_scores, 
                        num_to_prune, 
                        largest=False
                    )
                    
                    # Create structured pruning mask
                    prune.structured(
                        module,
                        'weight',
                        amount=num_to_prune,
                        dim=0,
                        importance_scores=importance_scores
                    )
        
        return self.model
    
    def gradient_based_pruning(self, train_loader, num_samples=1000):
        """Prune based on gradient information"""
        # Compute gradients
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Accumulate gradients
        gradient_accumulator = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                gradient_accumulator[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if sample_count >= num_samples:
                break
                
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    gradient_accumulator[name] += param.grad.abs()
            
            sample_count += data.size(0)
        
        # Normalize gradients
        for name in gradient_accumulator:
            gradient_accumulator[name] /= (sample_count / train_loader.batch_size)
        
        # Prune based on gradient magnitude
        for name, module in self.model.named_modules():
            param_name = f"{name}.weight"
            if param_name in gradient_accumulator:
                importance_scores = gradient_accumulator[param_name]
                prune.l1_unstructured(
                    module, 
                    'weight', 
                    amount=0.2,
                    importance_scores=importance_scores
                )
        
        return self.model
    
    def iterative_pruning(self, train_loader, val_loader, 
                         target_sparsity=0.9, num_iterations=5):
        """Iterative pruning with fine-tuning"""
        current_sparsity = 0.0
        sparsity_increment = target_sparsity / num_iterations
        
        for iteration in range(num_iterations):
            print(f"Pruning iteration {iteration + 1}/{num_iterations}")
            
            # Prune
            current_sparsity += sparsity_increment
            self.magnitude_based_pruning(sparsity_increment)
            
            # Fine-tune
            self._fine_tune_model(train_loader, val_loader, num_epochs=2)
            
            # Evaluate
            accuracy = self._evaluate_model(val_loader)
            print(f"Accuracy after iteration {iteration + 1}: {accuracy:.2f}%")
            
            # Check if accuracy dropped too much
            if self.original_model:
                original_accuracy = self._evaluate_model(val_loader, self.original_model)
                accuracy_drop = original_accuracy - accuracy
                if accuracy_drop > 5.0:  # Stop if accuracy drops more than 5%
                    print(f"Stopping early due to accuracy drop: {accuracy_drop:.2f}%")
                    break
        
        return self.model
    
    def _fine_tune_model(self, train_loader, val_loader, num_epochs=5, lr=1e-4):
        """Fine-tune pruned model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 99:
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
                    running_loss = 0.0
    
    def _evaluate_model(self, val_loader, model=None):
        """Evaluate model accuracy"""
        if model is None:
            model = self.model
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    def _compute_pruning_stats(self):
        """Compute statistics about current pruning state"""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()
            elif hasattr(module, 'weight'):
                total_params += module.weight.numel()
        
        sparsity = pruned_params / total_params if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'sparsity': sparsity,
            'compression_ratio': 1 / (1 - sparsity) if sparsity < 1 else float('inf')
        }
    
    def remove_pruning_masks(self):
        """Remove pruning masks and make pruning permanent"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
                
        return self.model
    
    def export_sparse_model(self, filepath):
        """Export model with sparse representation"""
        # Remove masks to make pruning permanent
        self.remove_pruning_masks()
        
        # Save model with sparse weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pruning_history': self.pruning_history,
            'model_architecture': str(self.model)
        }, filepath)

# Neural Architecture Search for Pruning
class AutoPruner:
    def __init__(self, model):
        self.model = model
        self.search_space = self._define_search_space()
        
    def _define_search_space(self):
        """Define search space for pruning ratios"""
        search_space = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Search space: pruning ratios from 0% to 90%
                search_space[name] = np.arange(0.0, 0.9, 0.1)
        return search_space
    
    def evolutionary_search(self, train_loader, val_loader, 
                          population_size=20, generations=10):
        """Use evolutionary algorithm to find optimal pruning configuration"""
        
        def create_random_individual():
            """Create random pruning configuration"""
            individual = {}
            for layer_name in self.search_space:
                individual[layer_name] = np.random.choice(self.search_space[layer_name])
            return individual
        
        def mutate_individual(individual, mutation_rate=0.2):
            """Mutate pruning configuration"""
            mutated = individual.copy()
            for layer_name in mutated:
                if np.random.random() < mutation_rate:
                    mutated[layer_name] = np.random.choice(self.search_space[layer_name])
            return mutated
        
        def crossover_individuals(parent1, parent2):
            """Crossover two pruning configurations"""
            child = {}
            for layer_name in parent1:
                child[layer_name] = parent1[layer_name] if np.random.random() < 0.5 else parent2[layer_name]
            return child
        
        def evaluate_individual(individual):
            """Evaluate pruning configuration"""
            # Apply pruning configuration
            model_copy = type(self.model)()
            model_copy.load_state_dict(self.model.state_dict())
            
            for layer_name, pruning_ratio in individual.items():
                for name, module in model_copy.named_modules():
                    if name == layer_name:
                        if pruning_ratio > 0:
                            prune.l1_unstructured(module, 'weight', amount=pruning_ratio)
            
            # Fine-tune briefly
            optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            model_copy.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  # Quick fine-tuning
                    break
                optimizer.zero_grad()
                output = model_copy(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            accuracy = self._evaluate_model_copy(model_copy, val_loader)
            
            # Compute compression ratio
            total_params = sum(p.numel() for p in self.model.parameters())
            remaining_params = sum(
                (getattr(module, 'weight_mask', torch.ones_like(module.weight)) != 0).sum().item()
                if hasattr(module, 'weight') else 0
                for module in model_copy.modules()
            )
            compression_ratio = total_params / remaining_params if remaining_params > 0 else float('inf')
            
            # Fitness = weighted combination of accuracy and compression
            fitness = accuracy + 10 * np.log(compression_ratio)  # Adjust weights as needed
            
            return fitness, accuracy, compression_ratio
        
        # Initialize population
        population = [create_random_individual() for _ in range(population_size)]
        
        best_individual = None
        best_fitness = -float('inf')
        
        # Evolution loop
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness, accuracy, compression = evaluate_individual(individual)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    print(f"New best: Fitness={fitness:.2f}, Accuracy={accuracy:.2f}%, Compression={compression:.2f}x")
            
            # Selection and reproduction
            new_population = []
            
            # Keep best individuals (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = population_size // 4
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover and mutation
                child = crossover_individuals(parent1, parent2)
                child = mutate_individual(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual, best_fitness
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for parent selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _evaluate_model_copy(self, model, val_loader, max_batches=50):
        """Evaluate model accuracy (limited batches for speed)"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total if total > 0 else 0
```

## Knowledge Distillation and Model Compression

### Teacher-Student Knowledge Transfer

**Advanced Knowledge Distillation Framework**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationFramework:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute knowledge distillation loss"""
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        student_loss = F.cross_entropy(student_logits, true_labels)
        
        # Combined loss
        total_loss = (
            self.alpha * distillation_loss + 
            (1 - self.alpha) * student_loss
        )
        
        return total_loss, distillation_loss, student_loss
    
    def train_student(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        """Train student model using knowledge distillation"""
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.student_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get teacher and student predictions
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                student_logits = self.student_model(data)
                
                # Compute distillation loss
                total_loss, distill_loss, student_loss = self.distillation_loss(
                    student_logits, teacher_logits, target
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                          f'Total Loss: {total_loss:.4f}, '
                          f'Distill Loss: {distill_loss:.4f}, '
                          f'Student Loss: {student_loss:.4f}')
            
            scheduler.step()
            
            # Validation phase
            val_accuracy = self._validate_student(val_loader)
            
            train_losses.append(epoch_loss / len(train_loader))
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
    
    def _validate_student(self, val_loader):
        """Validate student model"""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                outputs = self.student_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total

# Feature-based knowledge distillation
class FeatureDistillation:
    def __init__(self, teacher_model, student_model, feature_layers):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.feature_layers = feature_layers  # Dict mapping layer names to modules
        
        # Feature adaptation layers to match dimensions
        self.adaptation_layers = nn.ModuleDict()
        self._setup_adaptation_layers()
        
    def _setup_adaptation_layers(self):
        """Setup adaptation layers to match teacher-student feature dimensions"""
        for layer_name, (teacher_layer, student_layer) in self.feature_layers.items():
            # Get feature dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size
                teacher_feat = self._get_layer_output(self.teacher_model, teacher_layer, dummy_input)
                student_feat = self._get_layer_output(self.student_model, student_layer, dummy_input)
                
                teacher_dim = teacher_feat.shape[1]  # Assuming channel dimension
                student_dim = student_feat.shape[1]
            
            if teacher_dim != student_dim:
                # Add adaptation layer
                self.adaptation_layers[layer_name] = nn.Conv2d(
                    student_dim, teacher_dim, kernel_size=1, bias=False
                )
    
    def _get_layer_output(self, model, layer, input_tensor):
        """Get output of specific layer"""
        features = {}
        
        def hook(module, input, output):
            features['output'] = output
        
        handle = layer.register_forward_hook(hook)
        _ = model(input_tensor)
        handle.remove()
        
        return features['output']
    
    def feature_distillation_loss(self, student_features, teacher_features, layer_name):
        """Compute feature distillation loss"""
        # Adapt student features if necessary
        if layer_name in self.adaptation_layers:
            student_features = self.adaptation_layers[layer_name](student_features)
        
        # L2 loss between features
        loss = F.mse_loss(student_features, teacher_features.detach())
        
        return loss

# Attention-based knowledge distillation
class AttentionDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model
    
    def attention_map(self, feature_maps):
        """Generate attention maps from feature maps"""
        # Sum across channel dimension and normalize
        attention = torch.sum(feature_maps, dim=1, keepdim=True)
        attention = attention.view(attention.size(0), -1)
        attention = F.softmax(attention, dim=1)
        attention = attention.view_as(feature_maps[:, :1])
        
        return attention
    
    def attention_distillation_loss(self, student_features, teacher_features):
        """Compute attention-based distillation loss"""
        # Generate attention maps
        student_attention = self.attention_map(student_features)
        teacher_attention = self.attention_map(teacher_features)
        
        # L2 loss between attention maps
        loss = F.mse_loss(student_attention, teacher_attention.detach())
        
        return loss

# Progressive knowledge distillation
class ProgressiveDistillation:
    def __init__(self, teacher_model, student_architectures, num_stages=3):
        self.teacher_model = teacher_model
        self.student_architectures = student_architectures
        self.num_stages = num_stages
        self.intermediate_models = []
    
    def create_progressive_students(self):
        """Create intermediate models of increasing complexity"""
        # Create models with progressively increasing capacity
        for i in range(self.num_stages):
            # This would create models of increasing size
            # Implementation depends on specific architectures
            student = self.student_architectures[i]()
            self.intermediate_models.append(student)
        
        return self.intermediate_models
    
    def train_progressive(self, train_loader, val_loader):
        """Train students progressively"""
        current_teacher = self.teacher_model
        
        for stage, student in enumerate(self.intermediate_models):
            print(f"Training stage {stage + 1}/{len(self.intermediate_models)}")
            
            # Setup distillation framework
            distiller = KnowledgeDistillationFramework(
                current_teacher, student, temperature=4.0, alpha=0.7
            )
            
            # Train student
            results = distiller.train_student(
                train_loader, val_loader, num_epochs=10
            )
            
            print(f"Stage {stage + 1} final accuracy: {results['val_accuracies'][-1]:.2f}%")
            
            # Use trained student as teacher for next stage
            current_teacher = student
        
        return self.intermediate_models[-1]  # Return final student
```

## Advanced Deployment Strategies

### Containerization and Orchestration

**Production Deployment Framework**:
```python
import torch
import docker
import kubernetes
import yaml
from pathlib import Path
import json
import logging

class ModelDeploymentManager:
    def __init__(self, model_registry_path, deployment_config):
        self.model_registry_path = Path(model_registry_path)
        self.deployment_config = deployment_config
        self.docker_client = docker.from_env()
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def prepare_model_for_deployment(self, model_path, optimization_config=None):
        """Prepare model for production deployment"""
        model = torch.load(model_path, map_location='cpu')
        
        # Apply optimizations if specified
        if optimization_config:
            if optimization_config.get('quantization', False):
                model = self._apply_quantization(model)
            
            if optimization_config.get('pruning', False):
                model = self._apply_pruning(model, optimization_config['pruning'])
            
            if optimization_config.get('jit_compile', True):
                model = torch.jit.script(model)
        
        # Save optimized model
        optimized_path = self.model_registry_path / 'optimized_model.pth'
        torch.save(model, optimized_path)
        
        return optimized_path
    
    def create_docker_image(self, model_path, image_name, tag='latest'):
        """Create Docker image for model deployment"""
        dockerfile_content = f"""
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/
COPY models/ /models/

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "serve.py"]
"""
        
        # Create build context
        build_context = self.model_registry_path / 'docker_build'
        build_context.mkdir(exist_ok=True)
        
        # Write Dockerfile
        (build_context / 'Dockerfile').write_text(dockerfile_content)
        
        # Copy model and application files
        self._copy_deployment_files(build_context, model_path)
        
        # Build Docker image
        try:
            image, logs = self.docker_client.images.build(
                path=str(build_context),
                tag=f"{image_name}:{tag}",
                rm=True
            )
            
            for log in logs:
                if 'stream' in log:
                    self.logger.info(log['stream'].strip())
            
            self.logger.info(f"Docker image built successfully: {image_name}:{tag}")
            return image
            
        except docker.errors.BuildError as e:
            self.logger.error(f"Docker build failed: {e}")
            raise
    
    def _copy_deployment_files(self, build_context, model_path):
        """Copy necessary files for deployment"""
        import shutil
        
        # Create directories
        (build_context / 'app').mkdir(exist_ok=True)
        (build_context / 'models').mkdir(exist_ok=True)
        
        # Copy model
        shutil.copy2(model_path, build_context / 'models' / 'model.pth')
        
        # Create serving application
        serving_app = '''
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model
model = torch.load('/models/model.pth', map_location='cpu')
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Return results
        return jsonify({
            'predictions': probabilities.tolist(),
            'predicted_class': int(torch.argmax(probabilities))
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
'''
        
        (build_context / 'app' / 'serve.py').write_text(serving_app)
        
        # Create requirements.txt
        requirements = '''
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
pillow>=8.0.0
'''
        (build_context / 'requirements.txt').write_text(requirements)
    
    def deploy_to_kubernetes(self, image_name, namespace='default'):
        """Deploy model to Kubernetes cluster"""
        deployment_yaml = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{image_name}-deployment",
                'namespace': namespace
            },
            'spec': {
                'replicas': self.deployment_config.get('replicas', 3),
                'selector': {
                    'matchLabels': {
                        'app': image_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': image_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': image_name,
                            'image': f"{image_name}:latest",
                            'ports': [{'containerPort': 8000}],
                            'resources': {
                                'limits': {
                                    'cpu': self.deployment_config.get('cpu_limit', '1000m'),
                                    'memory': self.deployment_config.get('memory_limit', '2Gi')
                                },
                                'requests': {
                                    'cpu': self.deployment_config.get('cpu_request', '500m'),
                                    'memory': self.deployment_config.get('memory_request', '1Gi')
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        service_yaml = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{image_name}-service",
                'namespace': namespace
            },
            'spec': {
                'selector': {
                    'app': image_name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'type': 'LoadBalancer'
            }
        }
        
        # Save YAML files
        deployment_path = self.model_registry_path / f"{image_name}-deployment.yaml"
        service_path = self.model_registry_path / f"{image_name}-service.yaml"
        
        with open(deployment_path, 'w') as f:
            yaml.dump(deployment_yaml, f)
        
        with open(service_path, 'w') as f:
            yaml.dump(service_yaml, f)
        
        self.logger.info(f"Kubernetes manifests created: {deployment_path}, {service_path}")
        
        return deployment_path, service_path
    
    def setup_monitoring(self, image_name):
        """Setup monitoring for deployed model"""
        monitoring_config = {
            'apiVersion': 'v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': f"{image_name}-monitor",
                'labels': {
                    'app': image_name
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': image_name
                    }
                },
                'endpoints': [{
                    'port': 'http',
                    'interval': '30s',
                    'path': '/metrics'
                }]
            }
        }
        
        # Save monitoring configuration
        monitoring_path = self.model_registry_path / f"{image_name}-monitoring.yaml"
        with open(monitoring_path, 'w') as f:
            yaml.dump(monitoring_config, f)
        
        return monitoring_path

# Edge deployment optimization
class EdgeDeploymentOptimizer:
    def __init__(self, target_device='cpu'):
        self.target_device = target_device
        self.optimization_techniques = []
    
    def optimize_for_edge(self, model, optimization_config):
        """Optimize model for edge deployment"""
        optimized_model = model
        
        # Mobile-specific optimizations
        if optimization_config.get('mobile_optimize', False):
            optimized_model = self._optimize_for_mobile(optimized_model)
        
        # Quantization
        if optimization_config.get('quantization', False):
            optimized_model = self._apply_quantization(optimized_model)
        
        # Pruning
        if optimization_config.get('pruning', False):
            optimized_model = self._apply_pruning(optimized_model, optimization_config['pruning'])
        
        # Convert to TorchScript
        if optimization_config.get('torchscript', True):
            optimized_model = torch.jit.script(optimized_model)
            optimized_model = torch.jit.optimize_for_inference(optimized_model)
        
        return optimized_model
    
    def _optimize_for_mobile(self, model):
        """Apply mobile-specific optimizations"""
        from torch.utils.mobile_optimizer import optimize_for_mobile
        
        model.eval()
        example_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, example_input)
        
        # Apply mobile optimizations
        mobile_model = optimize_for_mobile(traced_model)
        
        return mobile_model
    
    def benchmark_edge_performance(self, model, input_shape, num_runs=100):
        """Benchmark model performance on edge device"""
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results = {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'p95_inference_time_ms': np.percentile(times, 95),
            'p99_inference_time_ms': np.percentile(times, 99)
        }
        
        return results
```

## Monitoring and Maintenance

### Production Model Monitoring

**Comprehensive Monitoring System**:
```python
import torch
import numpy as np
import json
import time
from collections import defaultdict, deque
import threading
import logging
from datetime import datetime, timedelta

class ProductionModelMonitor:
    def __init__(self, model, monitoring_config):
        self.model = model
        self.config = monitoring_config
        self.metrics_buffer = defaultdict(deque)
        self.drift_detector = DataDriftDetector()
        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager(monitoring_config.get('alerts', {}))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Model monitoring stopped")
    
    def log_prediction(self, input_data, prediction, confidence, response_time):
        """Log individual prediction for monitoring"""
        timestamp = datetime.now()
        
        # Store prediction metrics
        self.metrics_buffer['predictions'].append({
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'response_time': response_time
        })
        
        # Store input statistics for drift detection
        if isinstance(input_data, torch.Tensor):
            input_stats = {
                'mean': torch.mean(input_data).item(),
                'std': torch.std(input_data).item(),
                'min': torch.min(input_data).item(),
                'max': torch.max(input_data).item()
            }
            self.metrics_buffer['input_stats'].append({
                'timestamp': timestamp,
                'stats': input_stats
            })
        
        # Check for immediate alerts
        self._check_immediate_alerts(prediction, confidence, response_time)
        
        # Limit buffer size
        max_buffer_size = self.config.get('max_buffer_size', 10000)
        for key in self.metrics_buffer:
            if len(self.metrics_buffer[key]) > max_buffer_size:
                self.metrics_buffer[key].popleft()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        interval = self.config.get('monitoring_interval', 60)  # seconds
        
        while self.monitoring_active:
            try:
                # Compute and check metrics
                current_metrics = self._compute_current_metrics()
                self._check_metrics_alerts(current_metrics)
                
                # Check for data drift
                drift_results = self._check_data_drift()
                if drift_results['drift_detected']:
                    self.alert_manager.send_alert(
                        'data_drift',
                        f"Data drift detected: {drift_results['drift_score']:.4f}",
                        severity='warning'
                    )
                
                # Performance degradation check
                perf_degradation = self._check_performance_degradation()
                if perf_degradation['degradation_detected']:
                    self.alert_manager.send_alert(
                        'performance_degradation',
                        f"Performance degradation detected: {perf_degradation['degradation_percent']:.2f}%",
                        severity='critical'
                    )
                
                # Log current status
                self.logger.info(f"Monitoring check completed: {current_metrics}")
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
            
            time.sleep(interval)
    
    def _compute_current_metrics(self):
        """Compute current performance metrics"""
        if not self.metrics_buffer['predictions']:
            return {}
        
        # Recent predictions (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_predictions = [
            p for p in self.metrics_buffer['predictions']
            if p['timestamp'] > cutoff_time
        ]
        
        if not recent_predictions:
            return {}
        
        # Compute metrics
        confidences = [p['confidence'] for p in recent_predictions]
        response_times = [p['response_time'] for p in recent_predictions]
        
        metrics = {
            'prediction_count': len(recent_predictions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'predictions_per_minute': len(recent_predictions) / 60
        }
        
        return metrics
    
    def _check_immediate_alerts(self, prediction, confidence, response_time):
        """Check for immediate alerting conditions"""
        # Low confidence alert
        min_confidence = self.config.get('min_confidence_threshold', 0.5)
        if confidence < min_confidence:
            self.alert_manager.send_alert(
                'low_confidence',
                f"Low confidence prediction: {confidence:.4f}",
                severity='warning'
            )
        
        # High latency alert
        max_response_time = self.config.get('max_response_time', 1000)  # ms
        if response_time > max_response_time:
            self.alert_manager.send_alert(
                'high_latency',
                f"High response time: {response_time:.2f}ms",
                severity='warning'
            )
    
    def _check_metrics_alerts(self, metrics):
        """Check computed metrics against alert thresholds"""
        if not metrics:
            return
        
        # Check various thresholds
        thresholds = self.config.get('alert_thresholds', {})
        
        for metric_name, threshold_config in thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if 'min_value' in threshold_config and value < threshold_config['min_value']:
                    self.alert_manager.send_alert(
                        f'low_{metric_name}',
                        f"{metric_name} below threshold: {value} < {threshold_config['min_value']}",
                        severity=threshold_config.get('severity', 'warning')
                    )
                
                if 'max_value' in threshold_config and value > threshold_config['max_value']:
                    self.alert_manager.send_alert(
                        f'high_{metric_name}',
                        f"{metric_name} above threshold: {value} > {threshold_config['max_value']}",
                        severity=threshold_config.get('severity', 'warning')
                    )
    
    def _check_data_drift(self):
        """Check for data drift"""
        if len(self.metrics_buffer['input_stats']) < 100:
            return {'drift_detected': False}
        
        # Get recent and historical data
        recent_stats = list(self.metrics_buffer['input_stats'])[-100:]
        historical_stats = list(self.metrics_buffer['input_stats'])[:-100] if len(self.metrics_buffer['input_stats']) > 100 else []
        
        if not historical_stats:
            return {'drift_detected': False}
        
        # Simple drift detection based on mean shift
        recent_means = [s['stats']['mean'] for s in recent_stats]
        historical_means = [s['stats']['mean'] for s in historical_stats[-1000:]]  # Last 1000 historical points
        
        # Two-sample t-test (simplified)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(recent_means, historical_means)
        
        drift_threshold = self.config.get('drift_threshold', 0.05)
        drift_detected = p_value < drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': 1 - p_value,
            'p_value': p_value
        }
    
    def _check_performance_degradation(self):
        """Check for performance degradation"""
        if len(self.metrics_buffer['predictions']) < 1000:
            return {'degradation_detected': False}
        
        # Compare recent vs historical performance
        all_predictions = list(self.metrics_buffer['predictions'])
        recent_predictions = all_predictions[-500:]  # Recent 500 predictions
        historical_predictions = all_predictions[-1000:-500]  # Previous 500
        
        if not historical_predictions:
            return {'degradation_detected': False}
        
        # Compare average confidence
        recent_confidence = np.mean([p['confidence'] for p in recent_predictions])
        historical_confidence = np.mean([p['confidence'] for p in historical_predictions])
        
        degradation_percent = (historical_confidence - recent_confidence) / historical_confidence * 100
        degradation_threshold = self.config.get('degradation_threshold', 5.0)  # 5% degradation
        
        degradation_detected = degradation_percent > degradation_threshold
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_percent': degradation_percent,
            'recent_confidence': recent_confidence,
            'historical_confidence': historical_confidence
        }
    
    def get_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        current_metrics = self._compute_current_metrics()
        drift_results = self._check_data_drift()
        perf_results = self._check_performance_degradation()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'drift_analysis': drift_results,
            'performance_analysis': perf_results,
            'total_predictions': len(self.metrics_buffer['predictions']),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive'
        }
        
        return report

class DataDriftDetector:
    def __init__(self):
        self.reference_data = None
        self.reference_stats = None
    
    def set_reference_data(self, data):
        """Set reference data for drift detection"""
        self.reference_data = data
        self.reference_stats = self._compute_stats(data)
    
    def detect_drift(self, new_data, method='ks_test'):
        """Detect drift using various statistical tests"""
        if self.reference_data is None:
            return {'drift_detected': False, 'message': 'No reference data set'}
        
        if method == 'ks_test':
            return self._ks_test_drift(new_data)
        elif method == 'population_stability_index':
            return self._psi_drift(new_data)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
    
    def _ks_test_drift(self, new_data):
        """Kolmogorov-Smirnov test for drift detection"""
        from scipy import stats
        
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_data.flatten().numpy(),
            new_data.flatten().numpy()
        )
        
        drift_detected = p_value < 0.05  # 5% significance level
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'method': 'ks_test'
        }
    
    def _psi_drift(self, new_data):
        """Population Stability Index for drift detection"""
        # Simplified PSI calculation
        ref_bins = np.percentile(self.reference_data.flatten().numpy(), 
                                np.linspace(0, 100, 11))
        
        ref_counts, _ = np.histogram(self.reference_data.flatten().numpy(), bins=ref_bins)
        new_counts, _ = np.histogram(new_data.flatten().numpy(), bins=ref_bins)
        
        # Normalize to get proportions
        ref_props = ref_counts / ref_counts.sum()
        new_props = new_counts / new_counts.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_props = np.maximum(ref_props, epsilon)
        new_props = np.maximum(new_props, epsilon)
        
        # Calculate PSI
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        
        # PSI thresholds
        if psi < 0.1:
            drift_level = 'no_drift'
        elif psi < 0.2:
            drift_level = 'minor_drift'
        else:
            drift_level = 'major_drift'
        
        return {
            'drift_detected': psi >= 0.1,
            'psi_score': psi,
            'drift_level': drift_level,
            'method': 'psi'
        }
    
    def _compute_stats(self, data):
        """Compute statistical summary of data"""
        return {
            'mean': torch.mean(data).item(),
            'std': torch.std(data).item(),
            'min': torch.min(data).item(),
            'max': torch.max(data).item(),
            'shape': data.shape
        }

class PerformanceTracker:
    def __init__(self):
        self.baseline_metrics = None
        self.performance_history = deque(maxlen=1000)
    
    def set_baseline(self, metrics):
        """Set baseline performance metrics"""
        self.baseline_metrics = metrics
    
    def track_performance(self, current_metrics):
        """Track current performance against baseline"""
        if self.baseline_metrics is None:
            return {'status': 'no_baseline'}
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics
        })
        
        # Compare to baseline
        degradation_analysis = {}
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                degradation = (baseline_value - current_value) / baseline_value * 100
                degradation_analysis[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation_percent': degradation
                }
        
        return degradation_analysis

class AlertManager:
    def __init__(self, alert_config):
        self.alert_config = alert_config
        self.alert_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
    
    def send_alert(self, alert_type, message, severity='info'):
        """Send alert through configured channels"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(log_level, f"Alert [{alert_type}]: {message}")
        
        # Send through configured channels
        for channel_name, channel_config in self.alert_config.items():
            try:
                if channel_name == 'email':
                    self._send_email_alert(alert, channel_config)
                elif channel_name == 'slack':
                    self._send_slack_alert(alert, channel_config)
                elif channel_name == 'webhook':
                    self._send_webhook_alert(alert, channel_config)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel_name}: {str(e)}")
    
    def _send_email_alert(self, alert, config):
        """Send alert via email"""
        # Email implementation would go here
        pass
    
    def _send_slack_alert(self, alert, config):
        """Send alert via Slack"""
        # Slack implementation would go here
        pass
    
    def _send_webhook_alert(self, alert, config):
        """Send alert via webhook"""
        import requests
        
        webhook_url = config.get('url')
        if webhook_url:
            payload = {
                'timestamp': alert['timestamp'].isoformat(),
                'type': alert['type'],
                'message': alert['message'],
                'severity': alert['severity']
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
```

## Key Questions for Review

### Model Optimization
1. **Quantization Trade-offs**: What are the mathematical principles behind quantization, and how do different bit widths affect model accuracy and performance?

2. **Pruning Strategies**: When should structured vs unstructured pruning be used, and how do they impact model architecture and performance?

3. **Knowledge Distillation**: How can knowledge distillation be optimized for different teacher-student architecture combinations?

### Deployment Architecture
4. **Containerization Benefits**: What are the key advantages of containerized model deployment, and how do they address production challenges?

5. **Kubernetes Orchestration**: How can Kubernetes features be leveraged to ensure high availability and scalability of ML services?

6. **Edge Optimization**: What specific optimizations are most effective for deploying models on resource-constrained edge devices?

### Performance Monitoring
7. **Drift Detection**: What statistical methods are most reliable for detecting different types of data and concept drift in production?

8. **Performance Degradation**: How can performance degradation be distinguished from normal variation in model outputs?

9. **Alert Management**: What alerting strategies balance early problem detection with avoiding alert fatigue?

### Production Operations
10. **Model Versioning**: How should model versioning and rollback strategies be implemented in production environments?

11. **A/B Testing**: What are the best practices for conducting A/B tests with ML models in production?

12. **Scalability Planning**: How should infrastructure capacity planning account for ML model performance characteristics?

### Optimization Techniques
13. **Multi-Objective Optimization**: How can optimization techniques balance multiple objectives like accuracy, latency, and model size?

14. **Automated Optimization**: What role can automated techniques like neural architecture search play in production optimization?

15. **Cost Management**: How can deployment costs be optimized while maintaining required performance and reliability levels?

## Conclusion

Model Optimization and Deployment represent the critical bridge between experimental deep learning research and production AI systems that deliver reliable business value through sophisticated optimization techniques, robust deployment strategies, and comprehensive monitoring frameworks that ensure models maintain their effectiveness while meeting stringent performance, scalability, and reliability requirements. The comprehensive exploration of quantization, pruning, knowledge distillation, containerization, orchestration, and monitoring demonstrates how systematic engineering practices can transform research prototypes into production-ready systems that operate effectively at scale.

**Optimization Excellence**: The mathematical foundations and practical implementations of model compression techniques including quantization, pruning, and knowledge distillation provide the tools necessary for creating efficient models that maintain accuracy while significantly reducing computational and storage requirements, enabling deployment across diverse hardware platforms from cloud servers to edge devices.

**Deployment Sophistication**: The comprehensive deployment framework encompassing containerization, Kubernetes orchestration, and cloud-native architectures demonstrates how modern infrastructure technologies can provide the scalability, reliability, and maintainability required for production ML systems while enabling automated deployment, scaling, and management operations.

**Monitoring Intelligence**: The advanced monitoring and alerting systems provide real-time visibility into model performance, data quality, and system health, enabling proactive identification and resolution of issues before they impact users while maintaining detailed audit trails and performance analytics for continuous improvement.

**Production Readiness**: The integrated approach to optimization, deployment, and monitoring creates a complete framework for transitioning from experimental models to production systems that meet enterprise requirements for reliability, security, compliance, and cost-effectiveness while maintaining the flexibility to adapt to changing business needs and technical requirements.

**Operational Excellence**: The systematic methodology for production AI operations including performance tracking, drift detection, automated alerting, and maintenance procedures enables organizations to operate ML systems with the same reliability and efficiency standards applied to other mission-critical business systems while leveraging the unique characteristics and requirements of AI workloads.

Understanding and implementing these model optimization and deployment practices provides practitioners with the essential knowledge for successfully operationalizing deep learning systems that can scale from research prototypes to production applications serving millions of users while maintaining high standards for accuracy, performance, reliability, and cost-effectiveness. This comprehensive approach enables organizations to realize the full business value of their AI investments while building sustainable, maintainable systems that can evolve with changing requirements and technological advances.