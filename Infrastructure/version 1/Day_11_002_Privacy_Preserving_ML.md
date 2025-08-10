# Day 11.2: Privacy-Preserving ML Infrastructure

## 🔒 Responsible AI, Privacy & Edge Computing - Part 2

**Focus**: Differential Privacy, Federated Learning, Homomorphic Encryption  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master privacy-preserving machine learning techniques and infrastructure
- Learn differential privacy implementation and privacy budget management
- Understand federated learning architecture and secure aggregation protocols
- Analyze homomorphic encryption for private computation on encrypted data

---

## 🔒 Privacy-Preserving ML Theory

### **Privacy-Preserving ML Architecture**

Privacy-preserving ML requires sophisticated techniques to protect sensitive data while maintaining model utility through differential privacy, federated learning, and cryptographic methods.

**Privacy-Preserving Framework:**
```
Privacy-Preserving ML Components:
1. Differential Privacy Layer:
   - Privacy budget management
   - Noise injection mechanisms
   - Privacy accounting systems
   - Utility-privacy trade-off optimization

2. Federated Learning Layer:
   - Secure aggregation protocols
   - Client selection strategies
   - Communication optimization
   - Byzantine fault tolerance

3. Cryptographic Computation Layer:
   - Homomorphic encryption systems
   - Secure multi-party computation
   - Private set intersection
   - Zero-knowledge proofs

4. Privacy Governance Layer:
   - Privacy policy enforcement
   - Consent management
   - Data minimization controls
   - Audit and compliance tracking

Differential Privacy Mathematical Framework:
ε-Differential Privacy: For all datasets D, D' differing by one record:
Pr[M(D) ∈ S] ≤ exp(ε) × Pr[M(D') ∈ S]

Privacy Budget Management:
Total_Privacy_Budget = ε_total
Per_Query_Budget = ε_total / Number_of_Queries

Composition Theorems:
Sequential Composition: ε_total = Σ ε_i
Parallel Composition: ε_total = max(ε_i)

Federated Learning Convergence:
Global_Model = Σ (n_k/n) × Local_Model_k
where n_k = samples at client k, n = total samples

Privacy-Utility Trade-off:
Utility_Loss = f(Privacy_Budget, Data_Sensitivity, Algorithm_Complexity)
```

**Comprehensive Privacy-Preserving ML System:**
```
Privacy-Preserving ML Infrastructure:
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
import secrets
from datetime import datetime
import threading
import queue

class PrivacyMechanism(Enum):
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    FEDERATED_LEARNING = "federated_learning"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_differential_privacy"

class NoiseType(Enum):
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"

@dataclass
class PrivacyBudget:
    epsilon: float
    delta: float
    remaining_epsilon: float
    remaining_delta: float
    allocated_queries: int
    max_queries: int
    creation_timestamp: datetime

@dataclass
class FederatedClient:
    client_id: str
    local_data_size: int
    computation_capacity: float
    communication_bandwidth: float
    trust_score: float
    last_participation: datetime
    privacy_preferences: Dict[str, Any]

class PrivacyPreservingMLInfrastructure:
    def __init__(self):
        self.differential_privacy_engine = DifferentialPrivacyEngine()
        self.federated_learning_coordinator = FederatedLearningCoordinator()
        self.homomorphic_encryption_system = HomomorphicEncryptionSystem()
        self.privacy_budget_manager = PrivacyBudgetManager()
        self.privacy_auditor = PrivacyAuditor()
        self.consent_manager = ConsentManager()
    
    def train_privacy_preserving_model(self, training_config, privacy_requirements):
        """Train model with privacy-preserving techniques"""
        
        training_result = {
            'training_id': self._generate_training_id(),
            'timestamp': datetime.utcnow(),
            'privacy_mechanism': privacy_requirements['mechanism'],
            'privacy_parameters': privacy_requirements.get('parameters', {}),
            'model_performance': {},
            'privacy_guarantees': {},
            'compliance_status': {},
            'training_metrics': {}
        }
        
        try:
            mechanism = PrivacyMechanism(privacy_requirements['mechanism'])
            
            if mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                result = self._train_with_differential_privacy(training_config, privacy_requirements)
                
            elif mechanism == PrivacyMechanism.FEDERATED_LEARNING:
                result = self._train_with_federated_learning(training_config, privacy_requirements)
                
            elif mechanism == PrivacyMechanism.HOMOMORPHIC_ENCRYPTION:
                result = self._train_with_homomorphic_encryption(training_config, privacy_requirements)
                
            else:
                raise ValueError(f"Unsupported privacy mechanism: {mechanism}")
            
            training_result.update(result)
            
            # Validate privacy guarantees
            privacy_validation = self.privacy_auditor.validate_privacy_guarantees(
                training_result, privacy_requirements
            )
            training_result['privacy_validation'] = privacy_validation
            
            # Check compliance
            compliance_check = self._check_privacy_compliance(training_result, privacy_requirements)
            training_result['compliance_status'] = compliance_check
            
            return training_result
            
        except Exception as e:
            logging.error(f"Error in privacy-preserving training: {str(e)}")
            training_result['error'] = str(e)
            return training_result
    
    def _train_with_differential_privacy(self, training_config, privacy_requirements):
        """Train model with differential privacy"""
        
        epsilon = privacy_requirements['parameters'].get('epsilon', 1.0)
        delta = privacy_requirements['parameters'].get('delta', 1e-5)
        noise_type = NoiseType(privacy_requirements['parameters'].get('noise_type', 'laplace'))
        
        # Create privacy budget
        privacy_budget = self.privacy_budget_manager.create_privacy_budget(
            epsilon=epsilon,
            delta=delta,
            max_queries=training_config.get('max_iterations', 100)
        )
        
        # Initialize DP training
        dp_trainer = self.differential_privacy_engine.create_dp_trainer(
            privacy_budget=privacy_budget,
            noise_type=noise_type,
            training_config=training_config
        )
        
        # Train model with differential privacy
        training_result = dp_trainer.train_model(
            training_data=training_config['training_data'],
            model_architecture=training_config['model_architecture'],
            training_parameters=training_config.get('training_parameters', {})
        )
        
        return {
            'model': training_result['model'],
            'training_metrics': training_result['metrics'],
            'privacy_guarantees': {
                'epsilon_spent': privacy_budget.epsilon - privacy_budget.remaining_epsilon,
                'delta_spent': privacy_budget.delta - privacy_budget.remaining_delta,
                'privacy_budget_remaining': privacy_budget.remaining_epsilon,
                'noise_added': training_result.get('noise_statistics', {})
            },
            'model_performance': training_result.get('performance_metrics', {}),
            'privacy_mechanism_details': {
                'mechanism': 'differential_privacy',
                'noise_type': noise_type.value,
                'clipping_threshold': training_result.get('clipping_threshold'),
                'batch_size': training_result.get('batch_size')
            }
        }
    
    def _train_with_federated_learning(self, training_config, privacy_requirements):
        """Train model with federated learning"""
        
        # Initialize federated learning setup
        federated_config = privacy_requirements['parameters']
        
        fl_result = self.federated_learning_coordinator.coordinate_federated_training(
            clients=training_config.get('federated_clients', []),
            global_model_config=training_config['model_architecture'],
            training_rounds=federated_config.get('training_rounds', 10),
            client_selection_strategy=federated_config.get('client_selection', 'random'),
            aggregation_method=federated_config.get('aggregation_method', 'fedavg'),
            privacy_enhancements=federated_config.get('privacy_enhancements', {})
        )
        
        return {
            'model': fl_result['global_model'],
            'training_metrics': fl_result['training_metrics'],
            'privacy_guarantees': {
                'data_locality': True,
                'secure_aggregation': fl_result.get('secure_aggregation_used', False),
                'differential_privacy_noise': fl_result.get('dp_noise_applied', False),
                'client_participation': fl_result['client_participation_stats']
            },
            'model_performance': fl_result.get('performance_metrics', {}),
            'privacy_mechanism_details': {
                'mechanism': 'federated_learning',
                'participating_clients': fl_result['total_participating_clients'],
                'communication_rounds': fl_result['completed_rounds'],
                'aggregation_method': federated_config.get('aggregation_method', 'fedavg')
            }
        }

class DifferentialPrivacyEngine:
    def __init__(self):
        self.noise_generators = {
            NoiseType.LAPLACE: self._generate_laplace_noise,
            NoiseType.GAUSSIAN: self._generate_gaussian_noise,
            NoiseType.EXPONENTIAL: self._generate_exponential_noise
        }
        self.privacy_accountant = PrivacyAccountant()
    
    def create_dp_trainer(self, privacy_budget, noise_type, training_config):
        """Create differential privacy trainer"""
        
        return DPSGDTrainer(
            privacy_budget=privacy_budget,
            noise_generator=self.noise_generators[noise_type],
            privacy_accountant=self.privacy_accountant,
            training_config=training_config
        )
    
    def _generate_laplace_noise(self, scale, shape):
        """Generate Laplace noise for differential privacy"""
        return np.random.laplace(0, scale, shape)
    
    def _generate_gaussian_noise(self, scale, shape):
        """Generate Gaussian noise for differential privacy"""
        return np.random.normal(0, scale, shape)
    
    def _generate_exponential_noise(self, scale, shape):
        """Generate Exponential mechanism noise"""
        return np.random.exponential(scale, shape)

class DPSGDTrainer:
    """Differentially Private Stochastic Gradient Descent Trainer"""
    
    def __init__(self, privacy_budget, noise_generator, privacy_accountant, training_config):
        self.privacy_budget = privacy_budget
        self.noise_generator = noise_generator
        self.privacy_accountant = privacy_accountant
        self.training_config = training_config
        
        # DP-SGD parameters
        self.clipping_threshold = training_config.get('clipping_threshold', 1.0)
        self.batch_size = training_config.get('batch_size', 32)
        self.learning_rate = training_config.get('learning_rate', 0.01)
        
        # Noise scale calculation
        self.noise_scale = self._calculate_noise_scale()
        
    def _calculate_noise_scale(self):
        """Calculate noise scale for DP-SGD"""
        
        # Standard noise scale calculation for DP-SGD
        # σ = (sensitivity × √(2 ln(1.25/δ))) / ε
        sensitivity = self.clipping_threshold
        epsilon = self.privacy_budget.epsilon
        delta = self.privacy_budget.delta
        
        if delta > 0:
            noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
        else:
            # Pure ε-DP case
            noise_scale = sensitivity / epsilon
        
        return noise_scale
    
    def train_model(self, training_data, model_architecture, training_parameters):
        """Train model with DP-SGD"""
        
        # Initialize model
        model = self._initialize_model(model_architecture)
        
        # Training metrics
        training_metrics = {
            'loss_history': [],
            'accuracy_history': [],
            'privacy_spent_per_epoch': [],
            'noise_added_per_batch': []
        }
        
        # Training loop
        epochs = training_parameters.get('epochs', 10)
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_noise = []
            
            # Process data in batches
            for batch_data, batch_labels in self._create_batches(training_data, self.batch_size):
                
                # Forward pass
                predictions = model.forward(batch_data)
                loss = self._calculate_loss(predictions, batch_labels)
                
                # Backward pass with gradient clipping
                gradients = model.backward(loss)
                clipped_gradients = self._clip_gradients(gradients, self.clipping_threshold)
                
                # Add noise to gradients
                noisy_gradients = self._add_noise_to_gradients(clipped_gradients)
                epoch_noise.append(self._calculate_noise_magnitude(noisy_gradients))
                
                # Update model parameters
                model.update_parameters(noisy_gradients, self.learning_rate)
                
                epoch_loss.append(loss)
                
                # Update privacy budget
                self.privacy_accountant.update_privacy_spent(
                    self.privacy_budget, self.noise_scale, self.batch_size
                )
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_loss)
            training_metrics['loss_history'].append(avg_loss)
            training_metrics['noise_added_per_batch'].extend(epoch_noise)
            
            # Calculate accuracy if possible
            accuracy = self._calculate_accuracy(model, training_data)
            training_metrics['accuracy_history'].append(accuracy)
            
            # Record privacy spent
            training_metrics['privacy_spent_per_epoch'].append(
                self.privacy_budget.epsilon - self.privacy_budget.remaining_epsilon
            )
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}, Privacy spent: "
                        f"{self.privacy_budget.epsilon - self.privacy_budget.remaining_epsilon:.4f}")
            
            # Check privacy budget
            if self.privacy_budget.remaining_epsilon <= 0:
                logging.warning("Privacy budget exhausted, stopping training")
                break
        
        return {
            'model': model,
            'metrics': training_metrics,
            'performance_metrics': {
                'final_accuracy': training_metrics['accuracy_history'][-1],
                'final_loss': training_metrics['loss_history'][-1]
            },
            'noise_statistics': {
                'noise_scale': self.noise_scale,
                'average_noise_magnitude': np.mean(training_metrics['noise_added_per_batch']),
                'total_noise_applied': len(training_metrics['noise_added_per_batch'])
            },
            'clipping_threshold': self.clipping_threshold,
            'batch_size': self.batch_size,
            'epochs_completed': min(epoch + 1, epochs)
        }
    
    def _clip_gradients(self, gradients, clipping_threshold):
        """Clip gradients to bound sensitivity"""
        
        clipped_gradients = {}
        
        for layer_name, gradient in gradients.items():
            # Calculate L2 norm of gradient
            gradient_norm = np.linalg.norm(gradient)
            
            # Clip if necessary
            if gradient_norm > clipping_threshold:
                clipped_gradients[layer_name] = gradient * (clipping_threshold / gradient_norm)
            else:
                clipped_gradients[layer_name] = gradient
        
        return clipped_gradients
    
    def _add_noise_to_gradients(self, gradients):
        """Add calibrated noise to gradients"""
        
        noisy_gradients = {}
        
        for layer_name, gradient in gradients.items():
            # Generate noise with appropriate scale
            noise = self.noise_generator(self.noise_scale, gradient.shape)
            noisy_gradients[layer_name] = gradient + noise
        
        return noisy_gradients

class FederatedLearningCoordinator:
    def __init__(self):
        self.client_manager = FederatedClientManager()
        self.aggregation_engine = SecureAggregationEngine()
        self.communication_manager = FLCommunicationManager()
        self.byzantine_detector = ByzantineDetector()
    
    def coordinate_federated_training(self, clients, global_model_config, training_rounds,
                                    client_selection_strategy, aggregation_method, 
                                    privacy_enhancements):
        """Coordinate federated learning training process"""
        
        # Initialize global model
        global_model = self._initialize_global_model(global_model_config)
        
        # Federated training results
        fl_results = {
            'global_model': None,
            'training_metrics': {
                'round_accuracies': [],
                'round_losses': [],
                'client_participation': [],
                'communication_costs': [],
                'aggregation_time': []
            },
            'client_participation_stats': {},
            'completed_rounds': 0,
            'total_participating_clients': 0,
            'secure_aggregation_used': privacy_enhancements.get('secure_aggregation', False),
            'dp_noise_applied': privacy_enhancements.get('differential_privacy', False)
        }
        
        # Track client participation
        client_participation_count = {client.client_id: 0 for client in clients}
        
        for round_num in range(training_rounds):
            logging.info(f"Starting federated training round {round_num + 1}/{training_rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients(clients, client_selection_strategy)
            
            # Distribute global model to selected clients
            client_updates = self._distribute_and_train(selected_clients, global_model, privacy_enhancements)
            
            # Filter out any failed client updates
            valid_updates = [update for update in client_updates if update.get('success', False)]
            
            if not valid_updates:
                logging.warning(f"No valid client updates in round {round_num + 1}")
                continue
            
            # Detect and filter Byzantine clients
            if privacy_enhancements.get('byzantine_resilience', False):
                valid_updates = self.byzantine_detector.filter_byzantine_updates(valid_updates)
            
            # Aggregate client updates
            aggregation_start_time = datetime.utcnow()
            
            if privacy_enhancements.get('secure_aggregation', False):
                aggregated_update = self.aggregation_engine.secure_aggregate(
                    valid_updates, aggregation_method
                )
            else:
                aggregated_update = self._simple_aggregate(valid_updates, aggregation_method)
            
            aggregation_time = (datetime.utcnow() - aggregation_start_time).total_seconds()
            
            # Apply differential privacy to aggregated update if enabled
            if privacy_enhancements.get('differential_privacy', False):
                dp_config = privacy_enhancements['differential_privacy_config']
                aggregated_update = self._apply_dp_noise_to_update(aggregated_update, dp_config)
            
            # Update global model
            global_model = self._update_global_model(global_model, aggregated_update)
            
            # Evaluate global model performance
            round_performance = self._evaluate_global_model(global_model, clients)
            
            # Update metrics
            fl_results['training_metrics']['round_accuracies'].append(round_performance.get('accuracy', 0.0))
            fl_results['training_metrics']['round_losses'].append(round_performance.get('loss', float('inf')))
            fl_results['training_metrics']['client_participation'].append(len(valid_updates))
            fl_results['training_metrics']['aggregation_time'].append(aggregation_time)
            
            # Update client participation stats
            for update in valid_updates:
                client_id = update['client_id']
                client_participation_count[client_id] += 1
            
            fl_results['completed_rounds'] = round_num + 1
            
            logging.info(f"Round {round_num + 1} completed - Accuracy: {round_performance.get('accuracy', 0.0):.4f}, "
                        f"Loss: {round_performance.get('loss', 0.0):.4f}, "
                        f"Participating clients: {len(valid_updates)}")
        
        # Finalize results
        fl_results['global_model'] = global_model
        fl_results['client_participation_stats'] = client_participation_count
        fl_results['total_participating_clients'] = len([count for count in client_participation_count.values() if count > 0])
        
        # Calculate performance metrics
        if fl_results['training_metrics']['round_accuracies']:
            fl_results['performance_metrics'] = {
                'final_accuracy': fl_results['training_metrics']['round_accuracies'][-1],
                'best_accuracy': max(fl_results['training_metrics']['round_accuracies']),
                'final_loss': fl_results['training_metrics']['round_losses'][-1],
                'best_loss': min(fl_results['training_metrics']['round_losses']),
                'average_clients_per_round': np.mean(fl_results['training_metrics']['client_participation']),
                'total_communication_rounds': fl_results['completed_rounds']
            }
        
        return fl_results
    
    def _select_clients(self, clients, selection_strategy):
        """Select clients for federated training round"""
        
        if selection_strategy == 'random':
            # Random selection
            num_clients = min(len(clients), 10)  # Select up to 10 clients
            return np.random.choice(clients, size=num_clients, replace=False).tolist()
        
        elif selection_strategy == 'resource_based':
            # Select based on computation capacity and bandwidth
            sorted_clients = sorted(clients, 
                                  key=lambda c: c.computation_capacity * c.communication_bandwidth, 
                                  reverse=True)
            return sorted_clients[:min(len(clients), 10)]
        
        elif selection_strategy == 'data_size_based':
            # Select clients with more data
            sorted_clients = sorted(clients, key=lambda c: c.local_data_size, reverse=True)
            return sorted_clients[:min(len(clients), 10)]
        
        else:
            # Default to random selection
            return self._select_clients(clients, 'random')
    
    def _distribute_and_train(self, selected_clients, global_model, privacy_enhancements):
        """Distribute global model and coordinate local training"""
        
        client_updates = []
        
        # Use threading for parallel client training simulation
        def train_client(client):
            try:
                # Simulate local training
                local_update = self._simulate_local_training(client, global_model, privacy_enhancements)
                return local_update
            except Exception as e:
                logging.error(f"Error training client {client.client_id}: {str(e)}")
                return {'client_id': client.client_id, 'success': False, 'error': str(e)}
        
        # Execute parallel training
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
            future_to_client = {executor.submit(train_client, client): client for client in selected_clients}
            
            for future in concurrent.futures.as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    client_update = future.result()
                    client_updates.append(client_update)
                except Exception as e:
                    logging.error(f"Client {client.client_id} training failed: {str(e)}")
                    client_updates.append({
                        'client_id': client.client_id, 
                        'success': False, 
                        'error': str(e)
                    })
        
        return client_updates
    
    def _simulate_local_training(self, client, global_model, privacy_enhancements):
        """Simulate local training on client"""
        
        # This is a simplified simulation
        # In practice, this would involve actual model training on client data
        
        local_epochs = 3
        learning_rate = 0.01
        
        # Simulate local model update
        model_update = {}
        
        # Generate simulated gradients based on client data characteristics
        for layer_name in ['layer1', 'layer2', 'output']:
            # Simulate gradient computation
            gradient_shape = (100, 50) if layer_name != 'output' else (50, 10)
            
            # Add some randomness based on client characteristics
            client_factor = client.local_data_size / 1000.0  # Normalize data size
            gradient = np.random.normal(0, 0.1 * client_factor, gradient_shape)
            
            # Apply local differential privacy if enabled
            if privacy_enhancements.get('local_differential_privacy', False):
                ldp_config = privacy_enhancements['local_differential_privacy_config']
                epsilon = ldp_config.get('epsilon', 1.0)
                noise_scale = 2.0 / epsilon  # Laplace noise scale
                gradient += np.random.laplace(0, noise_scale, gradient.shape)
            
            model_update[layer_name] = gradient * learning_rate
        
        # Calculate local training metrics
        local_accuracy = 0.8 + np.random.normal(0, 0.1)  # Simulate accuracy
        local_loss = 0.5 + np.random.normal(0, 0.1)      # Simulate loss
        
        return {
            'client_id': client.client_id,
            'success': True,
            'model_update': model_update,
            'local_accuracy': max(0, min(1, local_accuracy)),
            'local_loss': max(0, local_loss),
            'data_size': client.local_data_size,
            'training_time': np.random.uniform(10, 60),  # Simulate training time
            'communication_cost': len(str(model_update))  # Simulate communication cost
        }
    
    def _simple_aggregate(self, client_updates, aggregation_method):
        """Simple aggregation of client updates"""
        
        if aggregation_method == 'fedavg':
            return self._federated_averaging(client_updates)
        elif aggregation_method == 'weighted_average':
            return self._weighted_averaging(client_updates)
        else:
            return self._federated_averaging(client_updates)  # Default
    
    def _federated_averaging(self, client_updates):
        """FedAvg aggregation"""
        
        # Calculate total data size
        total_data_size = sum(update['data_size'] for update in client_updates)
        
        # Initialize aggregated update
        aggregated_update = {}
        
        # Get layer names from first update
        layer_names = list(client_updates[0]['model_update'].keys())
        
        for layer_name in layer_names:
            # Weighted average based on data size
            weighted_sum = None
            
            for update in client_updates:
                layer_update = update['model_update'][layer_name]
                weight = update['data_size'] / total_data_size
                
                if weighted_sum is None:
                    weighted_sum = weight * layer_update
                else:
                    weighted_sum += weight * layer_update
            
            aggregated_update[layer_name] = weighted_sum
        
        return aggregated_update

class HomomorphicEncryptionSystem:
    def __init__(self):
        self.encryption_schemes = {
            'paillier': PaillierEncryption(),
            'ckks': CKKSEncryption(),
            'bfv': BFVEncryption()
        }
        self.key_manager = HEKeyManager()
    
    def train_on_encrypted_data(self, encrypted_data, model_config, encryption_scheme='paillier'):
        """Train model on encrypted data using homomorphic encryption"""
        
        he_scheme = self.encryption_schemes.get(encryption_scheme)
        if not he_scheme:
            raise ValueError(f"Unsupported encryption scheme: {encryption_scheme}")
        
        # Generate encryption keys if not provided
        public_key, private_key = he_scheme.generate_keypair()
        
        # Initialize encrypted model parameters
        encrypted_model = he_scheme.initialize_encrypted_model(model_config, public_key)
        
        # Training configuration
        training_config = {
            'learning_rate': 0.01,
            'epochs': 5,  # Fewer epochs due to computation overhead
            'batch_size': 32
        }
        
        training_results = {
            'encrypted_model': None,
            'training_metrics': [],
            'computational_overhead': {},
            'privacy_guarantees': {
                'data_never_decrypted': True,
                'computation_on_encrypted_data': True,
                'encryption_scheme': encryption_scheme
            }
        }
        
        # Homomorphic training loop
        for epoch in range(training_config['epochs']):
            epoch_start = datetime.utcnow()
            
            # Process encrypted data in batches
            for batch in self._create_encrypted_batches(encrypted_data, training_config['batch_size']):
                # Forward pass on encrypted data
                encrypted_predictions = he_scheme.encrypted_forward_pass(encrypted_model, batch)
                
                # Compute encrypted loss
                encrypted_loss = he_scheme.encrypted_loss_computation(encrypted_predictions, batch['labels'])
                
                # Compute encrypted gradients
                encrypted_gradients = he_scheme.encrypted_gradient_computation(
                    encrypted_model, batch, encrypted_loss
                )
                
                # Update encrypted model parameters
                encrypted_model = he_scheme.encrypted_parameter_update(
                    encrypted_model, encrypted_gradients, training_config['learning_rate']
                )
            
            epoch_time = (datetime.utcnow() - epoch_start).total_seconds()
            
            # Record training metrics (these would be encrypted in practice)
            training_results['training_metrics'].append({
                'epoch': epoch + 1,
                'training_time_seconds': epoch_time,
                'encrypted_loss': 'encrypted_value',  # Placeholder
                'computation_operations': he_scheme.get_operation_count()
            })
            
            logging.info(f"HE Training Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        
        training_results['encrypted_model'] = encrypted_model
        training_results['computational_overhead'] = {
            'total_training_time': sum(m['training_time_seconds'] for m in training_results['training_metrics']),
            'average_epoch_time': np.mean([m['training_time_seconds'] for m in training_results['training_metrics']]),
            'total_he_operations': sum(m['computation_operations'] for m in training_results['training_metrics'])
        }
        
        return training_results

class PaillierEncryption:
    """Simplified Paillier homomorphic encryption implementation"""
    
    def __init__(self):
        self.operation_count = 0
    
    def generate_keypair(self, key_size=1024):
        """Generate Paillier public/private key pair"""
        
        # This is a simplified implementation
        # In practice, use a proper cryptographic library
        
        import secrets
        
        # Generate large primes (simplified)
        p = self._generate_large_prime(key_size // 2)
        q = self._generate_large_prime(key_size // 2)
        
        n = p * q
        lambda_n = (p - 1) * (q - 1)  # lcm(p-1, q-1) simplified
        
        # Generate g
        g = n + 1  # Simplified choice
        
        # Calculate mu (modular inverse)
        mu = pow(lambda_n, -1, n)  # Simplified
        
        public_key = {'n': n, 'g': g}
        private_key = {'lambda': lambda_n, 'mu': mu, 'n': n}
        
        return public_key, private_key
    
    def encrypt(self, plaintext, public_key):
        """Encrypt plaintext using Paillier encryption"""
        
        n = public_key['n']
        g = public_key['g']
        
        # Generate random r
        r = secrets.randbelow(n)
        
        # Compute ciphertext: c = g^m * r^n mod n^2
        n_squared = n * n
        ciphertext = (pow(g, int(plaintext), n_squared) * pow(r, n, n_squared)) % n_squared
        
        self.operation_count += 1
        return ciphertext
    
    def decrypt(self, ciphertext, private_key):
        """Decrypt ciphertext using Paillier decryption"""
        
        n = private_key['n']
        lambda_n = private_key['lambda']
        mu = private_key['mu']
        
        n_squared = n * n
        
        # Compute L(c^lambda mod n^2)
        u = pow(ciphertext, lambda_n, n_squared)
        l_u = (u - 1) // n
        
        # Compute plaintext
        plaintext = (l_u * mu) % n
        
        self.operation_count += 1
        return plaintext
    
    def homomorphic_add(self, ciphertext1, ciphertext2, public_key):
        """Homomorphic addition of two encrypted values"""
        
        n_squared = public_key['n'] * public_key['n']
        result = (ciphertext1 * ciphertext2) % n_squared
        
        self.operation_count += 1
        return result
    
    def homomorphic_multiply_constant(self, ciphertext, constant, public_key):
        """Homomorphic multiplication by constant"""
        
        n_squared = public_key['n'] * public_key['n']
        result = pow(ciphertext, int(constant), n_squared)
        
        self.operation_count += 1
        return result
    
    def initialize_encrypted_model(self, model_config, public_key):
        """Initialize model with encrypted parameters"""
        
        encrypted_model = {}
        
        # Initialize random encrypted parameters
        for layer_name, layer_config in model_config.items():
            shape = layer_config.get('shape', (10, 10))
            
            # Initialize with small random encrypted values
            encrypted_params = []
            for i in range(shape[0]):
                param_row = []
                for j in range(shape[1]):
                    # Encrypt small random initialization
                    random_val = np.random.normal(0, 0.1)
                    encrypted_val = self.encrypt(random_val * 1000, public_key)  # Scale for integer arithmetic
                    param_row.append(encrypted_val)
                encrypted_params.append(param_row)
            
            encrypted_model[layer_name] = encrypted_params
        
        return encrypted_model
    
    def get_operation_count(self):
        """Get count of homomorphic operations performed"""
        return self.operation_count
    
    def _generate_large_prime(self, bits):
        """Generate a large prime number (simplified)"""
        # This is a very simplified implementation
        # In practice, use proper prime generation
        import random
        
        while True:
            num = random.getrandbits(bits)
            if self._miller_rabin(num):
                return num
    
    def _miller_rabin(self, n, k=5):
        """Miller-Rabin primality test (simplified)"""
        if n < 2:
            return False
        
        # Simple cases
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if n in small_primes:
            return True
        if any(n % p == 0 for p in small_primes):
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Witness loop
        for _ in range(k):
            a = secrets.randbelow(n - 2) + 2
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True

class PrivacyBudgetManager:
    def __init__(self):
        self.active_budgets = {}
        self.budget_history = []
    
    def create_privacy_budget(self, epsilon, delta, max_queries):
        """Create new privacy budget"""
        
        budget = PrivacyBudget(
            epsilon=epsilon,
            delta=delta,
            remaining_epsilon=epsilon,
            remaining_delta=delta,
            allocated_queries=0,
            max_queries=max_queries,
            creation_timestamp=datetime.utcnow()
        )
        
        budget_id = hashlib.sha256(f"{epsilon}{delta}{datetime.utcnow()}".encode()).hexdigest()[:16]
        self.active_budgets[budget_id] = budget
        
        return budget
    
    def allocate_privacy_budget(self, budget_id, required_epsilon, required_delta=0):
        """Allocate privacy budget for a query"""
        
        budget = self.active_budgets.get(budget_id)
        if not budget:
            raise ValueError(f"Privacy budget {budget_id} not found")
        
        if budget.remaining_epsilon < required_epsilon:
            raise ValueError(f"Insufficient privacy budget. Required: {required_epsilon}, Available: {budget.remaining_epsilon}")
        
        if budget.remaining_delta < required_delta:
            raise ValueError(f"Insufficient delta budget. Required: {required_delta}, Available: {budget.remaining_delta}")
        
        # Allocate budget
        budget.remaining_epsilon -= required_epsilon
        budget.remaining_delta -= required_delta
        budget.allocated_queries += 1
        
        # Record allocation
        self.budget_history.append({
            'budget_id': budget_id,
            'allocated_epsilon': required_epsilon,
            'allocated_delta': required_delta,
            'remaining_epsilon': budget.remaining_epsilon,
            'remaining_delta': budget.remaining_delta,
            'timestamp': datetime.utcnow()
        })
        
        return True
```

This comprehensive framework for privacy-preserving ML infrastructure provides the theoretical foundations and practical implementation strategies for protecting sensitive data while maintaining model utility through differential privacy, federated learning, and homomorphic encryption techniques.