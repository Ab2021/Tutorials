# Day 57: Scaling Laws & Compute Optimization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Scaling Law Prediction

```python
import numpy as np
import matplotlib.pyplot as plt

class ScalingLawPredictor:
    def __init__(self, law_type='chinchilla'):
        self.law_type = law_type
        
        # Kaplan scaling law coefficients
        self.kaplan_alpha = 0.076  # Parameter scaling exponent
        self.kaplan_beta = 0.095   # Data scaling exponent
        
        # Chinchilla optimal ratios
        self.chinchilla_tokens_per_param = 20
    
    def predict_loss(self, num_params, num_tokens):
        """Predict loss given model size and data."""
        if self.law_type == 'kaplan':
            # Kaplan scaling law
            loss_params = (num_params / 1e9) ** (-self.kaplan_alpha)
            loss_data = (num_tokens / 1e9) ** (-self.kaplan_beta)
            loss = loss_params * loss_data
        
        elif self.law_type == 'chinchilla':
            # Chinchilla scaling law
            # Optimal: N and D scale equally with compute
            compute = 6 * num_params * num_tokens
            loss = compute ** (-0.05)  # Approximate
        
        return loss
    
    def compute_optimal_allocation(self, compute_budget):
        """Compute optimal model size and data for given budget."""
        # Chinchilla optimal: N ∝ C^0.5, D ∝ C^0.5
        # C = 6 × N × D
        # N_optimal = sqrt(C / (6 × 20))
        
        optimal_params = np.sqrt(compute_budget / (6 * self.chinchilla_tokens_per_param))
        optimal_tokens = self.chinchilla_tokens_per_param * optimal_params
        
        return optimal_params, optimal_tokens
    
    def compare_configurations(self, compute_budget):
        """Compare different model size / data combinations."""
        configs = []
        
        # Kaplan-style (larger model, less data)
        kaplan_params = compute_budget / (6 * 100e9)  # 100B tokens
        configs.append({
            'name': 'Kaplan-style',
            'params': kaplan_params,
            'tokens': 100e9,
            'loss': self.predict_loss(kaplan_params, 100e9)
        })
        
        # Chinchilla optimal
        opt_params, opt_tokens = self.compute_optimal_allocation(compute_budget)
        configs.append({
            'name': 'Chinchilla optimal',
            'params': opt_params,
            'tokens': opt_tokens,
            'loss': self.predict_loss(opt_params, opt_tokens)
        })
        
        # Over-trained (smaller model, more data)
        overtrain_params = opt_params / 2
        overtrain_tokens = compute_budget / (6 * overtrain_params)
        configs.append({
            'name': 'Over-trained',
            'params': overtrain_params,
            'tokens': overtrain_tokens,
            'loss': self.predict_loss(overtrain_params, overtrain_tokens)
        })
        
        return configs

# Usage
predictor = ScalingLawPredictor(law_type='chinchilla')

# Example: 1e24 FLOPs budget
compute_budget = 1e24
configs = predictor.compare_configurations(compute_budget)

for config in configs:
    print(f"{config['name']}:")
    print(f"  Parameters: {config['params']/1e9:.1f}B")
    print(f"  Tokens: {config['tokens']/1e9:.1f}B")
    print(f"  Predicted loss: {config['loss']:.4f}")
```

### 2. Compute Budget Calculator

```python
class ComputeBudgetCalculator:
    def __init__(self):
        self.flops_per_token_per_param = 6  # Forward + backward
    
    def training_compute(self, num_params, num_tokens):
        """Calculate training compute (FLOPs)."""
        return self.flops_per_token_per_param * num_params * num_tokens
    
    def inference_compute(self, num_params, num_output_tokens):
        """Calculate inference compute per request."""
        return 2 * num_params * num_output_tokens  # Forward only
    
    def gpu_hours(self, total_flops, gpu_type='A100'):
        """Convert FLOPs to GPU hours."""
        gpu_flops = {
            'V100': 125e12,    # 125 TFLOPS (FP16)
            'A100': 312e12,    # 312 TFLOPS (FP16)
            'H100': 1000e12    # 1 PFLOPS (FP8)
        }
        
        flops_per_second = gpu_flops[gpu_type]
        seconds = total_flops / flops_per_second
        hours = seconds / 3600
        
        return hours
    
    def cost_estimate(self, gpu_hours, gpu_type='A100', num_gpus=1):
        """Estimate training cost."""
        gpu_cost_per_hour = {
            'V100': 2.5,
            'A100': 4.0,
            'H100': 8.0
        }
        
        cost = gpu_hours * gpu_cost_per_hour[gpu_type] * num_gpus
        
        return cost
    
    def plan_training(self, num_params, num_tokens, gpu_type='A100', num_gpus=2048):
        """Complete training plan."""
        # Compute
        total_flops = self.training_compute(num_params, num_tokens)
        
        # Time
        total_gpu_hours = self.gpu_hours(total_flops, gpu_type)
        wall_clock_hours = total_gpu_hours / num_gpus
        wall_clock_days = wall_clock_hours / 24
        
        # Cost
        total_cost = self.cost_estimate(total_gpu_hours, gpu_type, num_gpus)
        
        return {
            'total_flops': total_flops,
            'gpu_hours': total_gpu_hours,
            'wall_clock_days': wall_clock_days,
            'total_cost': total_cost
        }

# Usage
calculator = ComputeBudgetCalculator()

# LLaMA 70B training plan
plan = calculator.plan_training(
    num_params=70e9,
    num_tokens=1.4e12,
    gpu_type='A100',
    num_gpus=2048
)

print(f"Total FLOPs: {plan['total_flops']:.2e}")
print(f"GPU hours: {plan['gpu_hours']:,.0f}")
print(f"Wall-clock time: {plan['wall_clock_days']:.1f} days")
print(f"Estimated cost: ${plan['total_cost']:,.0f}")
```

### 3. Emergent Abilities Tracker

```python
class EmergentAbilitiesTracker:
    def __init__(self):
        self.abilities = {
            'few_shot_learning': {'threshold': 10e9, 'emerged': False},
            'chain_of_thought': {'threshold': 60e9, 'emerged': False},
            'instruction_following': {'threshold': 100e9, 'emerged': False},
            'code_generation': {'threshold': 10e9, 'emerged': False},
            'multilingual': {'threshold': 1e9, 'emerged': False}
        }
    
    def check_abilities(self, num_params):
        """Check which abilities should emerge at this scale."""
        emerged = []
        
        for ability, info in self.abilities.items():
            if num_params >= info['threshold']:
                emerged.append(ability)
                info['emerged'] = True
        
        return emerged
    
    def plot_emergence(self):
        """Plot emergence thresholds."""
        abilities = list(self.abilities.keys())
        thresholds = [self.abilities[a]['threshold']/1e9 for a in abilities]
        
        plt.figure(figsize=(10, 6))
        plt.barh(abilities, thresholds)
        plt.xlabel('Model Size (Billions of Parameters)')
        plt.title('Emergent Abilities Thresholds')
        plt.xscale('log')
        plt.tight_layout()
        plt.show()
```

### 4. Data Scaling Planner

```python
class DataScalingPlanner:
    def __init__(self):
        self.chinchilla_ratio = 20  # tokens per parameter
    
    def required_tokens(self, num_params, training_regime='optimal'):
        """Calculate required training tokens."""
        if training_regime == 'optimal':
            # Chinchilla optimal
            return num_params * self.chinchilla_ratio
        
        elif training_regime == 'over_train':
            # Over-train by 2x
            return num_params * self.chinchilla_ratio * 2
        
        elif training_regime == 'under_train':
            # GPT-3 style (under-trained)
            return num_params * 2
    
    def data_sources_needed(self, total_tokens):
        """Estimate data sources needed."""
        sources = {
            'CommonCrawl': 250e12,      # 250T tokens
            'C4': 150e12,               # 150T tokens
            'Books': 10e12,             # 10T tokens
            'GitHub': 5e12,             # 5T tokens
            'Wikipedia': 3e12,          # 3T tokens
            'ArXiv': 2e12               # 2T tokens
        }
        
        # Calculate mix
        total_available = sum(sources.values())
        
        if total_tokens > total_available:
            print(f"Warning: Need {total_tokens/1e12:.1f}T tokens but only {total_available/1e12:.1f}T available")
        
        # Proportional allocation
        mix = {}
        for source, available in sources.items():
            mix[source] = min(available, total_tokens * (available / total_available))
        
        return mix
```

### 5. Efficiency Comparison

```python
class EfficiencyComparator:
    def __init__(self):
        pass
    
    def compare_approaches(self, base_params=70e9, base_tokens=1.4e12):
        """Compare different scaling approaches."""
        approaches = {}
        
        # Dense model
        dense_compute = 6 * base_params * base_tokens
        approaches['Dense'] = {
            'params': base_params,
            'active_params': base_params,
            'tokens': base_tokens,
            'compute': dense_compute,
            'inference_cost': 2 * base_params
        }
        
        # MoE (8 experts, 2 active)
        moe_total_params = base_params * 4  # 4x parameters
        moe_active_params = base_params  # Same active params
        moe_compute = 6 * moe_active_params * base_tokens  # Same training compute
        approaches['MoE'] = {
            'params': moe_total_params,
            'active_params': moe_active_params,
            'tokens': base_tokens,
            'compute': moe_compute,
            'inference_cost': 2 * moe_active_params
        }
        
        # Quantized (INT8)
        approaches['Quantized'] = {
            'params': base_params,
            'active_params': base_params,
            'tokens': base_tokens,
            'compute': dense_compute,
            'inference_cost': 2 * base_params * 0.5  # 2x faster
        }
        
        return approaches
    
    def print_comparison(self, approaches):
        """Print comparison table."""
        print(f"{'Approach':<15} {'Total Params':<15} {'Active Params':<15} {'Training Compute':<20} {'Inference Cost':<15}")
        print("-" * 80)
        
        for name, metrics in approaches.items():
            print(f"{name:<15} {metrics['params']/1e9:<15.1f}B {metrics['active_params']/1e9:<15.1f}B {metrics['compute']:<20.2e} {metrics['inference_cost']:<15.2e}")
```

### 6. Scaling Visualization

```python
def visualize_scaling_laws():
    """Visualize scaling laws."""
    # Parameter range
    params = np.logspace(9, 12, 50)  # 1B to 1T
    
    # Kaplan scaling
    kaplan_loss = (params / 1e9) ** (-0.076)
    
    # Chinchilla scaling (with optimal data)
    chinchilla_tokens = params * 20
    chinchilla_loss = ((6 * params * chinchilla_tokens) ** (-0.05))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(params/1e9, kaplan_loss, label='Kaplan')
    plt.loglog(params/1e9, chinchilla_loss, label='Chinchilla')
    plt.xlabel('Parameters (Billions)')
    plt.ylabel('Loss')
    plt.title('Scaling Laws Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.loglog(params/1e9, chinchilla_tokens/1e9, label='Chinchilla Optimal')
    plt.xlabel('Parameters (Billions)')
    plt.ylabel('Training Tokens (Billions)')
    plt.title('Optimal Data Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```
