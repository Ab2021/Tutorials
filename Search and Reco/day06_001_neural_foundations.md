# Day 6.1: Deep Learning Foundations for Recommendation Systems

## Learning Objectives
- Master fundamental neural network architectures for recommendation systems
- Implement deep feedforward networks for rating prediction and ranking
- Understand embedding layers and their role in recommendation systems
- Design and implement neural network training pipelines for RecSys
- Explore regularization techniques specific to recommendation systems

## 1. Neural Network Fundamentals for RecSys

### Deep Feedforward Networks for Recommendations

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RecSysDataset(Dataset):
    """Dataset class for recommendation system data"""
    
    def __init__(self, interactions: pd.DataFrame, user_features: pd.DataFrame = None, 
                 item_features: pd.DataFrame = None, rating_col: str = 'rating'):
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features
        self.rating_col = rating_col
        
        # Encode user and item IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.interactions['user_encoded'] = self.user_encoder.fit_transform(interactions['user_id'])
        self.interactions['item_encoded'] = self.item_encoder.fit_transform(interactions['item_id'])
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        
        sample = {
            'user_id': torch.tensor(row['user_encoded'], dtype=torch.long),
            'item_id': torch.tensor(row['item_encoded'], dtype=torch.long),
            'rating': torch.tensor(row[self.rating_col], dtype=torch.float32)
        }
        
        # Add user features if available
        if self.user_features is not None:
            user_feat = self.user_features[self.user_features['user_id'] == row['user_id']]
            if not user_feat.empty:
                feat_cols = [col for col in user_feat.columns if col != 'user_id']
                sample['user_features'] = torch.tensor(user_feat[feat_cols].values[0], dtype=torch.float32)
        
        # Add item features if available
        if self.item_features is not None:
            item_feat = self.item_features[self.item_features['item_id'] == row['item_id']]
            if not item_feat.empty:
                feat_cols = [col for col in item_feat.columns if col != 'item_id']
                sample['item_features'] = torch.tensor(item_feat[feat_cols].values[0], dtype=torch.float32)
        
        return sample

class BaseNeuralRecommender(nn.Module, ABC):
    """Base class for neural recommendation models"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    @abstractmethod
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a specific user"""
        return self.user_embedding(torch.tensor([user_id]))
    
    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """Get embedding for a specific item"""
        return self.item_embedding(torch.tensor([item_id]))
    
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate top-k recommendations for a user"""
        self.eval()
        
        with torch.no_grad():
            user_tensor = torch.tensor([user_id])
            item_tensors = torch.arange(self.n_items)
            
            # Get predictions for all items
            scores = []
            for item_id in item_tensors:
                score = self.forward(user_tensor, torch.tensor([item_id]))
                scores.append(score.item())
            
            # Get top-k items
            top_k_indices = np.argsort(scores)[-k:][::-1]
            recommendations = [(idx, scores[idx]) for idx in top_k_indices]
        
        return recommendations

class DeepMatrixFactorization(BaseNeuralRecommender):
    """Deep Matrix Factorization model"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__(n_users, n_items, embedding_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build deep network
        layers = []
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.deep_layers = nn.Sequential(*layers)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize bias terms
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Pass through deep network
        deep_output = self.deep_layers(x)
        
        # Add bias terms
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        
        output = deep_output.squeeze() + user_bias.squeeze() + item_bias.squeeze() + self.global_bias
        
        return output

class NeuralMatrixFactorization(BaseNeuralRecommender):
    """Neural Matrix Factorization (NCF) model combining GMF and MLP"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 mlp_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__(n_users, n_items, embedding_dim)
        
        # Separate embeddings for GMF and MLP paths
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for mlp_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = mlp_dim
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.final_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1)
        
        # Initialize all embeddings
        for embedding in [self.gmf_user_embedding, self.gmf_item_embedding,
                         self.mlp_user_embedding, self.mlp_item_embedding]:
            nn.init.normal_(embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path: element-wise product
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb
        
        # MLP path: concatenation + deep network
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Combine GMF and MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.final_layer(combined)
        
        return output.squeeze()

class WideAndDeepRecommender(BaseNeuralRecommender):
    """Wide & Deep model for recommendations"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 deep_dims: List[int] = [128, 64, 32], n_user_features: int = 0,
                 n_item_features: int = 0, dropout_rate: float = 0.2):
        super().__init__(n_users, n_items, embedding_dim)
        
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        
        # Wide part: linear model with cross features
        wide_input_dim = 2  # user_id and item_id one-hot encoded (simplified)
        if n_user_features > 0:
            wide_input_dim += n_user_features
        if n_item_features > 0:
            wide_input_dim += n_item_features
        
        self.wide_linear = nn.Linear(wide_input_dim, 1)
        
        # Deep part: deep neural network
        deep_input_dim = embedding_dim * 2
        if n_user_features > 0:
            deep_input_dim += n_user_features
        if n_item_features > 0:
            deep_input_dim += n_item_features
        
        deep_layers = []
        for deep_dim in deep_dims:
            deep_layers.extend([
                nn.Linear(deep_input_dim, deep_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            deep_input_dim = deep_dim
        
        deep_layers.append(nn.Linear(deep_input_dim, 1))
        self.deep_layers = nn.Sequential(*deep_layers)
        
        # Feature processing layers
        if n_user_features > 0:
            self.user_feature_layer = nn.Linear(n_user_features, n_user_features)
        if n_item_features > 0:
            self.item_feature_layer = nn.Linear(n_item_features, n_item_features)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                user_features: torch.Tensor = None, item_features: torch.Tensor = None) -> torch.Tensor:
        
        batch_size = user_ids.size(0)
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Wide part input
        wide_input = []
        
        # Simplified one-hot encoding (in practice, use proper sparse features)
        user_onehot = torch.zeros(batch_size, 1)
        item_onehot = torch.zeros(batch_size, 1)
        wide_input.extend([user_onehot, item_onehot])
        
        # Add additional features to wide part
        if user_features is not None and self.n_user_features > 0:
            wide_input.append(user_features)
        if item_features is not None and self.n_item_features > 0:
            wide_input.append(item_features)
        
        wide_input_tensor = torch.cat(wide_input, dim=-1)
        wide_output = self.wide_linear(wide_input_tensor)
        
        # Deep part input
        deep_input = [user_emb, item_emb]
        
        if user_features is not None and self.n_user_features > 0:
            processed_user_feat = self.user_feature_layer(user_features)
            deep_input.append(processed_user_feat)
        if item_features is not None and self.n_item_features > 0:
            processed_item_feat = self.item_feature_layer(item_features)
            deep_input.append(processed_item_feat)
        
        deep_input_tensor = torch.cat(deep_input, dim=-1)
        deep_output = self.deep_layers(deep_input_tensor)
        
        # Combine wide and deep outputs
        output = wide_output + deep_output
        
        return output.squeeze()

class RecSysTrainer:
    """Training pipeline for neural recommendation models"""
    
    def __init__(self, model: BaseNeuralRecommender, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              n_epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, patience: int = 10,
              loss_fn: str = 'mse') -> Dict[str, List[float]]:
        """Train the recommendation model"""
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if 'user_features' in batch and 'item_features' in batch:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    predictions = self.model(user_ids, item_ids, user_features, item_features)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                # Compute loss
                loss = criterion(predictions, ratings)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Load best model
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.training_history
    
    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                if 'user_features' in batch and 'item_features' in batch:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    predictions = self.model(user_ids, item_ids, user_features, item_features)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                loss = criterion(predictions, ratings)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        if self.training_history['val_loss']:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        if self.training_history['val_loss']:
            plt.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def create_sample_data(n_users: int = 1000, n_items: int = 500, n_interactions: int = 10000):
    """Create sample recommendation data"""
    np.random.seed(42)
    
    # Generate interactions
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'rating': np.random.uniform(1, 5, n_interactions)
    })
    
    # Remove duplicates
    interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'])
    
    # Generate user features
    user_features = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.uniform(18, 65, n_users),
        'gender': np.random.choice([0, 1], n_users),
        'income': np.random.uniform(20000, 100000, n_users)
    })
    
    # Generate item features
    item_features = pd.DataFrame({
        'item_id': range(n_items),
        'price': np.random.uniform(10, 1000, n_items),
        'category': np.random.randint(0, 10, n_items),
        'popularity': np.random.uniform(0, 1, n_items)
    })
    
    return interactions, user_features, item_features

def demonstrate_neural_recommenders():
    """Demonstrate neural recommendation models"""
    
    print("🧠 Creating sample recommendation data...")
    interactions, user_features, item_features = create_sample_data()
    
    # Create dataset
    dataset = RecSysDataset(interactions, user_features, item_features)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print(f"📊 Dataset created with {len(dataset)} interactions")
    print(f"🧑‍🤝‍🧑 Users: {dataset.n_users}, 📦 Items: {dataset.n_items}")
    
    # Test different models
    models = {
        'Deep Matrix Factorization': DeepMatrixFactorization(
            dataset.n_users, dataset.n_items, embedding_dim=64, hidden_dims=[128, 64, 32]
        ),
        'Neural Collaborative Filtering': NeuralMatrixFactorization(
            dataset.n_users, dataset.n_items, embedding_dim=64, mlp_dims=[128, 64, 32]
        ),
        'Wide & Deep': WideAndDeepRecommender(
            dataset.n_users, dataset.n_items, embedding_dim=64, deep_dims=[128, 64, 32],
            n_user_features=3, n_item_features=3
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🚀 Training {model_name}...")
        
        trainer = RecSysTrainer(model)
        history = trainer.train(
            train_loader, val_loader,
            n_epochs=50, learning_rate=0.001,
            patience=10
        )
        
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
        results[model_name] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': final_val_loss,
            'model': model,
            'trainer': trainer
        }
        
        print(f"✅ {model_name} training completed!")
        print(f"📈 Final train loss: {results[model_name]['final_train_loss']:.4f}")
        if final_val_loss:
            print(f"📊 Final val loss: {final_val_loss:.4f}")
    
    # Compare results
    print("\n" + "="*50)
    print("🏆 MODEL COMPARISON RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Train Loss: {result['final_train_loss']:.4f}")
        if result['final_val_loss']:
            print(f"  Val Loss: {result['final_val_loss']:.4f}")
    
    # Generate sample recommendations
    print("\n" + "="*50)
    print("🎯 SAMPLE RECOMMENDATIONS")
    print("="*50)
    
    best_model_name = min(results.keys(), key=lambda x: results[x]['final_val_loss'] or float('inf'))
    best_model = results[best_model_name]['model']
    
    print(f"\nUsing best model: {best_model_name}")
    
    # Generate recommendations for first 3 users
    for user_id in range(3):
        recommendations = best_model.recommend(user_id, k=5)
        print(f"\nUser {user_id} recommendations:")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            print(f"  {rank}. Item {item_id}: {score:.4f}")
    
    return results

if __name__ == "__main__":
    results = demonstrate_neural_recommenders()
```

## 2. Advanced Regularization Techniques

### Dropout, Batch Normalization, and RecSys-specific Regularization

```python
class RegularizedNeuralMF(BaseNeuralRecommender):
    """Neural Matrix Factorization with advanced regularization"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64, 32], dropout_rates: List[float] = None,
                 use_batch_norm: bool = True, l2_reg: float = 1e-4,
                 embedding_dropout: float = 0.1):
        super().__init__(n_users, n_items, embedding_dim)
        
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Default dropout rates
        if dropout_rates is None:
            dropout_rates = [0.2] * len(hidden_dims)
        
        # Build network with regularization
        layers = []
        input_dim = embedding_dim * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Bias terms with regularization
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper schemes"""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings with dropout
        user_emb = self.embedding_dropout(self.user_embedding(user_ids))
        item_emb = self.embedding_dropout(self.item_embedding(item_ids))
        
        # Concatenate and process
        x = torch.cat([user_emb, item_emb], dim=-1)
        deep_output = self.network(x)
        
        # Add bias terms
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        
        output = deep_output.squeeze() + user_bias.squeeze() + item_bias.squeeze() + self.global_bias
        
        return output
    
    def compute_l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss"""
        l2_loss = 0
        
        # Embedding regularization
        l2_loss += torch.norm(self.user_embedding.weight, p=2)
        l2_loss += torch.norm(self.item_embedding.weight, p=2)
        l2_loss += torch.norm(self.user_bias.weight, p=2)
        l2_loss += torch.norm(self.item_bias.weight, p=2)
        
        # Network weights regularization
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                l2_loss += torch.norm(module.weight, p=2)
        
        return self.l2_reg * l2_loss

class AdversarialNeuralMF(BaseNeuralRecommender):
    """Neural MF with adversarial training for robustness"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64, 32], epsilon: float = 0.1):
        super().__init__(n_users, n_items, embedding_dim)
        
        self.epsilon = epsilon
        
        # Build network
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adversarial: bool = False) -> torch.Tensor:
        
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        if adversarial and self.training:
            # Add adversarial noise
            user_noise = torch.randn_like(user_emb) * self.epsilon
            item_noise = torch.randn_like(item_emb) * self.epsilon
            
            user_emb = user_emb + user_noise
            item_emb = item_emb + item_noise
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        output = self.network(x)
        
        return output.squeeze()
    
    def adversarial_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                        ratings: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """Compute adversarial training loss"""
        
        # Normal prediction
        normal_pred = self.forward(user_ids, item_ids, adversarial=False)
        normal_loss = F.mse_loss(normal_pred, ratings)
        
        # Adversarial prediction
        adv_pred = self.forward(user_ids, item_ids, adversarial=True)
        adv_loss = F.mse_loss(adv_pred, ratings)
        
        # Combined loss
        total_loss = normal_loss + alpha * adv_loss
        
        return total_loss

class VariationalNeuralMF(BaseNeuralRecommender):
    """Variational Neural Matrix Factorization for uncertainty estimation"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64, 32], kl_weight: float = 1e-3):
        super().__init__(n_users, n_items, embedding_dim)
        
        self.kl_weight = kl_weight
        
        # Variational embeddings (mean and log variance)
        self.user_embedding_mu = nn.Embedding(n_users, embedding_dim)
        self.user_embedding_logvar = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mu = nn.Embedding(n_items, embedding_dim)
        self.item_embedding_logvar = nn.Embedding(n_items, embedding_dim)
        
        # Network
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize variational parameters
        nn.init.normal_(self.user_embedding_mu.weight, std=0.01)
        nn.init.constant_(self.user_embedding_logvar.weight, -3)
        nn.init.normal_(self.item_embedding_mu.weight, std=0.01)
        nn.init.constant_(self.item_embedding_logvar.weight, -3)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for variational inference"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            return mu
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Variational user embeddings
        user_mu = self.user_embedding_mu(user_ids)
        user_logvar = self.user_embedding_logvar(user_ids)
        user_emb = self.reparameterize(user_mu, user_logvar)
        
        # Variational item embeddings
        item_mu = self.item_embedding_mu(item_ids)
        item_logvar = self.item_embedding_logvar(item_ids)
        item_emb = self.reparameterize(item_mu, item_logvar)
        
        # Forward pass
        x = torch.cat([user_emb, item_emb], dim=-1)
        output = self.network(x)
        
        return output.squeeze()
    
    def kl_divergence(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for variational regularization"""
        
        # User KL divergence
        user_mu = self.user_embedding_mu(user_ids)
        user_logvar = self.user_embedding_logvar(user_ids)
        user_kl = -0.5 * torch.sum(1 + user_logvar - user_mu.pow(2) - user_logvar.exp())
        
        # Item KL divergence
        item_mu = self.item_embedding_mu(item_ids)
        item_logvar = self.item_embedding_logvar(item_ids)
        item_kl = -0.5 * torch.sum(1 + item_logvar - item_mu.pow(2) - item_logvar.exp())
        
        return self.kl_weight * (user_kl + item_kl)
    
    def variational_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                        ratings: torch.Tensor) -> torch.Tensor:
        """Compute variational loss (reconstruction + KL)"""
        
        predictions = self.forward(user_ids, item_ids)
        reconstruction_loss = F.mse_loss(predictions, ratings)
        kl_loss = self.kl_divergence(user_ids, item_ids)
        
        return reconstruction_loss + kl_loss
```

## Key Takeaways

1. **Neural Foundations**: Deep neural networks provide powerful non-linear modeling capabilities for recommendation systems

2. **Embedding Layers**: User and item embeddings are fundamental components that capture latent representations

3. **Architecture Diversity**: Different architectures (DMF, NCF, Wide & Deep) serve different purposes and data characteristics

4. **Regularization Importance**: Proper regularization prevents overfitting and improves generalization in neural RecSys

5. **Training Pipeline**: Systematic training with validation, early stopping, and hyperparameter tuning is crucial

6. **Advanced Techniques**: Adversarial training, variational inference, and other advanced techniques enhance model robustness

## Study Questions

### Beginner Level
1. What are the advantages of neural networks over traditional matrix factorization?  
2. How do embedding layers work in recommendation systems?
3. What is the purpose of bias terms in neural recommendation models?
4. Why is regularization important in neural recommendation systems?

### Intermediate Level
1. Compare and contrast Deep MF, NCF, and Wide & Deep architectures
2. How would you handle cold start problems in neural recommendation systems?
3. What are the trade-offs between model complexity and training time?
4. How can you detect and prevent overfitting in neural RecSys?

### Advanced Level
1. Design a multi-task neural architecture for rating prediction and ranking simultaneously
2. Implement a neural model that handles temporal dynamics in user preferences
3. How would you incorporate graph structure into neural recommendation models?
4. Design a variational approach for handling uncertainty in recommendations

## Next Session Preview

Tomorrow we'll explore **Neural Collaborative Filtering and Autoencoders**, covering:
- Advanced neural collaborative filtering architectures
- Denoising autoencoders for recommendation systems
- Variational autoencoders for collaborative filtering
- Convolutional neural networks for sequential recommendations
- Recurrent neural networks for session-based recommendations
- Advanced training techniques and optimization strategies

We'll implement sophisticated neural architectures that push the boundaries of collaborative filtering!