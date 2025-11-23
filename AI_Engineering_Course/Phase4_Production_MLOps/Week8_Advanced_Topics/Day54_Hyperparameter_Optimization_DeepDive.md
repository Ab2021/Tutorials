# Day 54: Hyperparameter Optimization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Bayesian Optimization with Gaussian Process

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimizer:
    def __init__(self, bounds, n_init=5):
        self.bounds = bounds
        self.n_init = n_init
        
        # Gaussian Process
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Observations
        self.X_observed = []
        self.y_observed = []
    
    def acquisition_function(self, X, xi=0.01):
        """Expected Improvement acquisition function."""
        X = np.array(X).reshape(-1, len(self.bounds))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Best observed value
        y_best = np.max(self.y_observed)
        
        # Expected Improvement
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def propose_location(self):
        """Propose next point to evaluate."""
        # Minimize negative acquisition (maximize acquisition)
        def min_obj(X):
            return -self.acquisition_function(X)
        
        # Random restarts
        best_x = None
        best_acquisition = float('inf')
        
        for _ in range(25):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            
            # Optimize
            res = minimize(
                min_obj,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if res.fun < best_acquisition:
                best_acquisition = res.fun
                best_x = res.x
        
        return best_x
    
    def optimize(self, objective_fn, n_iterations=50):
        """Run Bayesian optimization."""
        # Initial random samples
        for _ in range(self.n_init):
            x = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            y = objective_fn(x)
            
            self.X_observed.append(x)
            self.y_observed.append(y)
        
        # Bayesian optimization loop
        for i in range(n_iterations):
            # Fit GP
            self.gp.fit(self.X_observed, self.y_observed)
            
            # Propose next point
            x_next = self.propose_location()
            
            # Evaluate
            y_next = objective_fn(x_next)
            
            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            print(f"Iteration {i+1}: x={x_next}, y={y_next:.4f}")
        
        # Return best
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

# Usage
def objective(params):
    lr, batch_size = params
    # Train model and return validation accuracy
    model = train_model(lr=lr, batch_size=int(batch_size))
    return model.val_accuracy

bounds = [(1e-5, 1e-3), (16, 128)]  # lr, batch_size
optimizer = BayesianOptimizer(bounds)
best_params, best_score = optimizer.optimize(objective, n_iterations=30)
```

### 2. Population-Based Training

```python
class PopulationBasedTraining:
    def __init__(self, population_size=10, exploit_threshold=0.2):
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.population = []
    
    def initialize_population(self, model_fn, hyperparams_fn):
        """Initialize population with random hyperparameters."""
        for _ in range(self.population_size):
            hyperparams = hyperparams_fn()
            model = model_fn(hyperparams)
            
            self.population.append({
                'model': model,
                'hyperparams': hyperparams,
                'score': 0.0,
                'steps': 0
            })
    
    def exploit_and_explore(self, member, population_scores):
        """Exploit: copy from best. Explore: perturb hyperparams."""
        # Exploit: replace if in bottom 20%
        rank = np.argsort(population_scores)
        if member in rank[:int(self.exploit_threshold * len(rank))]:
            # Copy from top performer
            best_idx = rank[-1]
            best_member = self.population[best_idx]
            
            member['model'].load_state_dict(
                best_member['model'].state_dict()
            )
            member['hyperparams'] = best_member['hyperparams'].copy()
        
        # Explore: perturb hyperparams
        for key in member['hyperparams']:
            if np.random.rand() < 0.25:  # 25% chance to perturb
                # Multiply or divide by 1.2
                factor = 1.2 if np.random.rand() < 0.5 else 1/1.2
                member['hyperparams'][key] *= factor
    
    def train(self, train_fn, eval_fn, total_steps=10000, eval_interval=1000):
        """Train population with PBT."""
        while any(m['steps'] < total_steps for m in self.population):
            # Train each member for eval_interval steps
            for member in self.population:
                if member['steps'] < total_steps:
                    train_fn(
                        member['model'],
                        member['hyperparams'],
                        steps=eval_interval
                    )
                    member['steps'] += eval_interval
                    member['score'] = eval_fn(member['model'])
            
            # Exploit and explore
            scores = [m['score'] for m in self.population]
            for member in self.population:
                self.exploit_and_explore(member, scores)
        
        # Return best
        best_idx = np.argmax([m['score'] for m in self.population])
        return self.population[best_idx]
```

### 3. Learning Rate Schedules

```python
class LearningRateScheduler:
    def __init__(self, optimizer, schedule_type='cosine', **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.step_count = 0
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        if self.schedule_type == 'cosine':
            lr = self._cosine_schedule()
        elif self.schedule_type == 'linear_warmup_cosine':
            lr = self._warmup_cosine_schedule()
        elif self.schedule_type == 'step':
            lr = self._step_schedule()
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _cosine_schedule(self):
        """Cosine annealing."""
        max_lr = self.kwargs['max_lr']
        min_lr = self.kwargs.get('min_lr', 0)
        max_steps = self.kwargs['max_steps']
        
        progress = min(self.step_count / max_steps, 1.0)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(np.pi * progress)
        )
        
        return lr
    
    def _warmup_cosine_schedule(self):
        """Linear warmup + cosine decay."""
        warmup_steps = self.kwargs['warmup_steps']
        max_lr = self.kwargs['max_lr']
        min_lr = self.kwargs.get('min_lr', 0)
        max_steps = self.kwargs['max_steps']
        
        if self.step_count < warmup_steps:
            # Linear warmup
            lr = max_lr * (self.step_count / warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - warmup_steps) / (max_steps - warmup_steps)
            progress = min(progress, 1.0)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        
        return lr
    
    def _step_schedule(self):
        """Step decay."""
        initial_lr = self.kwargs['initial_lr']
        decay_rate = self.kwargs['decay_rate']
        decay_steps = self.kwargs['decay_steps']
        
        lr = initial_lr * (decay_rate ** (self.step_count // decay_steps))
        
        return lr

# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = LearningRateScheduler(
    optimizer,
    schedule_type='linear_warmup_cosine',
    max_lr=1e-3,
    min_lr=1e-5,
    warmup_steps=1000,
    max_steps=10000
)

for step in range(10000):
    # Training step
    loss.backward()
    optimizer.step()
    
    # Update learning rate
    lr = scheduler.step()
```

### 4. Hyperband Implementation

```python
class Hyperband:
    def __init__(self, max_budget, eta=3):
        self.max_budget = max_budget
        self.eta = eta
    
    def optimize(self, get_config_fn, train_fn, eval_fn):
        """Run Hyperband algorithm."""
        # Compute number of brackets
        s_max = int(np.log(self.max_budget) / np.log(self.eta))
        B = (s_max + 1) * self.max_budget
        
        best_config = None
        best_score = float('-inf')
        
        for s in reversed(range(s_max + 1)):
            n = int(np.ceil(B / self.max_budget / (s + 1) * self.eta ** s))
            r = self.max_budget * self.eta ** (-s)
            
            # Generate configurations
            configs = [get_config_fn() for _ in range(n)]
            
            for i in range(s + 1):
                # Train each config for r budget
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                
                scores = []
                for config in configs[:n_i]:
                    train_fn(config, budget=r_i)
                    score = eval_fn(config)
                    scores.append((score, config))
                
                # Keep top eta^(-1) fraction
                scores.sort(reverse=True)
                configs = [config for _, config in scores[:int(n_i / self.eta)]]
                
                # Update best
                if scores[0][0] > best_score:
                    best_score = scores[0][0]
                    best_config = scores[0][1]
        
        return best_config, best_score
```

### 5. Optuna Integration

```python
import optuna

def create_optuna_study(objective_fn, n_trials=100):
    """Create and run Optuna study."""
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 6, 12)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Train and evaluate
        config = {
            'lr': lr,
            'batch_size': batch_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        score = objective_fn(config)
        
        return score
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Best parameters
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value}")
    
    return study
```
