# Day 10: Week 2 Review - Deep Learning Foundations

## Overview
This review consolidates neural networks, CNNs, architectures, and training techniques. We'll implement a complete deep learning pipeline from scratch to deployment.

## Key Concepts Recap

### 1. Neural Networks
**Forward propagation:**
$$ z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} $$
$$ a^{[l]} = g^{[l]}(z^{[l]}) $$

**Backpropagation:**
$$ \frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} (a^{[l-1]})^T $$

**Optimizers:**
- SGD: $\theta_t = \theta_{t-1} - \alpha \nabla L$
- Momentum: $v_t = \beta v_{t-1} + \nabla L$
- Adam: Adaptive learning rates per parameter

### 2. Convolutional Neural Networks
**Convolution:**
$$ Z[i,j] = \sum_m \sum_n X[i+m, j+n] \cdot K[m, n] + b $$

**Output size:**
$$ H_{out} = \frac{H_{in} + 2p - k}{s} + 1 $$

**Key components:**
- Conv layers: Feature extraction
- Pooling: Downsampling
- Batch norm: Stabilization
- Activation: Non-linearity

### 3. CNN Architectures
**Evolution:**
- AlexNet (2012): ReLU, dropout, GPU
- VGG (2014): Deep + small filters
- ResNet (2015): Skip connections
- EfficientNet (2019): Compound scaling

**Design principles:**
- Increasing depth
- Decreasing spatial dimensions
- Increasing channel depth
- Skip connections for very deep networks

### 4. Training Techniques
**Data augmentation:**
- Geometric: Flip, crop, rotate
- Photometric: Color jitter
- Advanced: Mixup, CutMix

**Transfer learning:**
- Feature extraction
- Fine-tuning
- Discriminative LR

**Regularization:**
- Dropout, weight decay
- Label smoothing
- Early stopping

## Complete Image Classification Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class ImageClassificationPipeline:
    """End-to-end image classification pipeline."""
    
    def __init__(self, num_classes=10, model_name='resnet50', pretrained=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model(model_name, pretrained)
        self.model = self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = None
        self.scheduler = None
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def _create_model(self, model_name, pretrained):
        """Create model with custom classifier."""
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif model_name == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(pretrained=pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        elif model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def prepare_data(self, data_dir, batch_size=32, num_workers=4):
        """Prepare data loaders with augmentation."""
        # Training transforms
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5),
        ])
        
        # Validation transforms
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root=f'{data_dir}/train',
            transform=train_transform
        )
        
        val_dataset = torchvision.datasets.ImageFolder(
            root=f'{data_dir}/val',
            transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return self.train_loader, self.val_loader
    
    def configure_training(self, lr=0.001, weight_decay=1e-4, 
                          warmup_epochs=5, total_epochs=100):
        """Configure optimizer and scheduler."""
        # Optimizer with different LR for different layers
        params = [
            {'params': self.model.layer1.parameters(), 'lr': lr * 0.1},
            {'params': self.model.layer2.parameters(), 'lr': lr * 0.1},
            {'params': self.model.layer3.parameters(), 'lr': lr * 0.5},
            {'params': self.model.layer4.parameters(), 'lr': lr * 0.5},
            {'params': self.model.fc.parameters(), 'lr': lr},
        ]
        
        self.optimizer = optim.AdamW(params, weight_decay=weight_decay)
        
        # Warm-up + cosine annealing scheduler
        self.scheduler = self._get_scheduler(warmup_epochs, total_epochs)
    
    def _get_scheduler(self, warmup_epochs, total_epochs):
        """Create warm-up + cosine scheduler."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def mixup_data(self, x, y, alpha=0.2):
        """Mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, use_mixup=True):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Mixup augmentation
            if use_mixup:
                inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Loss
            if use_mixup:
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            
            if use_mixup:
                correct += (lam * predicted.eq(labels_a).sum().float() +
                           (1 - lam) * predicted.eq(labels_b).sum().float())
            else:
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=100, use_mixup=True, save_best=True):
        """Complete training loop."""
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(use_mixup)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if save_best and val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        print(f'\nBest validation accuracy: {best_acc:.4f}')
        return self.history
    
    def plot_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
    
    def predict(self, image_path):
        """Predict single image."""
        from PIL import Image
        
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence

# Usage Example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ImageClassificationPipeline(
        num_classes=10,
        model_name='resnet50',
        pretrained=True
    )
    
    # Prepare data
    train_loader, val_loader = pipeline.prepare_data(
        data_dir='./data',
        batch_size=32
    )
    
    # Configure training
    pipeline.configure_training(
        lr=0.001,
        weight_decay=1e-4,
        warmup_epochs=5,
        total_epochs=100
    )
    
    # Train
    history = pipeline.train(epochs=100, use_mixup=True)
    
    # Plot results
    pipeline.plot_history()
    
    # Predict
    pred_class, confidence = pipeline.predict('test_image.jpg')
    print(f'Predicted class: {pred_class}, Confidence: {confidence:.4f}')
```

## Practice Problems

### Problem 1: Custom Architecture
Implement a custom CNN architecture with:
- 4 convolutional blocks
- Residual connections
- SE blocks
- Global average pooling

### Problem 2: Advanced Augmentation
Implement CutMix augmentation from scratch.

### Problem 3: Multi-GPU Training
Modify the pipeline to support DataParallel or DistributedDataParallel.

## Summary
Week 2 covered neural networks, CNNs, architectures, and training techniques. The complete pipeline demonstrates integration of all concepts for production-ready image classification.

**Next Week:** Object Detection (R-CNN, YOLO, advanced techniques).
