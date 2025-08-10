# Day 29.2: PyTorch Ecosystem Tools - Comprehensive Framework Integration and Specialized Libraries

## Overview

PyTorch Ecosystem Tools represent a comprehensive collection of specialized libraries, frameworks, and utilities that extend PyTorch's core capabilities to address specific domains, deployment scenarios, and advanced use cases through sophisticated integrations that maintain PyTorch's design philosophy while providing domain-specific optimizations, pre-built components, and production-ready solutions. Understanding the breadth and depth of the PyTorch ecosystem, from computer vision libraries like TorchVision and specialized frameworks like PyTorch Lightning to distributed training tools, deployment platforms, and domain-specific extensions, reveals how the modular architecture and open-source nature of PyTorch has fostered an extensive ecosystem that accelerates development, reduces code duplication, and enables practitioners to leverage state-of-the-art implementations across diverse application domains. This comprehensive exploration examines the mathematical foundations and practical implementations of key ecosystem tools including TorchVision for computer vision, TorchAudio for audio processing, PyTorch Lightning for training automation, TorchServe for model serving, and specialized libraries for natural language processing, reinforcement learning, and scientific computing that collectively provide a complete toolkit for modern deep learning development and deployment.

## Computer Vision Ecosystem - TorchVision

### Core Vision Components and Transformations

**Mathematical Foundations of Image Transformations**:
```python
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2  # New transforms API

# Advanced transformation pipeline
class AdvancedAugmentation:
    def __init__(self, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
        # Geometric transformations with mathematical precision
        self.geometric_transforms = v2.Compose([
            # Random perspective with 3D projection matrix
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            # Elastic deformation using displacement fields
            v2.ElasticTransform(alpha=50.0, sigma=5.0, p=0.3),
            # Affine transformations with matrix operations
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), 
                           scale=(0.9, 1.1), shear=10)
        ])
        
        # Color space transformations
        self.color_transforms = v2.Compose([
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
    
    def __call__(self, image):
        # Apply transformations in specific order
        image = self.geometric_transforms(image)
        image = self.color_transforms(image)
        
        # Convert to tensor and normalize
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        image = self.normalize(image)
        
        return image

# Custom transformation with mathematical operations
class FourierNoiseTransform:
    """Add noise in frequency domain"""
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert to frequency domain
            img_freq = torch.fft.fft2(img)
            
            # Add noise in frequency domain
            noise = torch.randn_like(img_freq) * self.noise_factor
            img_freq_noisy = img_freq + noise
            
            # Convert back to spatial domain
            img_noisy = torch.fft.ifft2(img_freq_noisy).real
            
            # Clamp to valid range
            return torch.clamp(img_noisy, 0, 1)
        else:
            return img

# Mixup and CutMix implementations
class MixupCutmix:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def mixup_batch(self, batch, targets):
        """Apply mixup to a batch"""
        batch_size = batch.size(0)
        
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        
        # Permute batch
        indices = torch.randperm(batch_size)
        mixed_batch = lam * batch + (1 - lam) * batch[indices]
        
        # Mix targets
        targets_a, targets_b = targets, targets[indices]
        
        return mixed_batch, targets_a, targets_b, lam
    
    def cutmix_batch(self, batch, targets):
        """Apply CutMix to a batch"""
        batch_size = batch.size(0)
        _, _, H, W = batch.shape
        
        # Sample lambda and compute cut region
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
        cut_rat = torch.sqrt(1.0 - lam).item()
        
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        
        # Compute bounding box
        bbx1 = torch.clamp(torch.tensor(cx - cut_w // 2), 0, W).item()
        bby1 = torch.clamp(torch.tensor(cy - cut_h // 2), 0, H).item()
        bbx2 = torch.clamp(torch.tensor(cx + cut_w // 2), 0, W).item()
        bby2 = torch.clamp(torch.tensor(cy + cut_h // 2), 0, H).item()
        
        # Permute and apply cut
        indices = torch.randperm(batch_size)
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[indices]
        return batch, targets_a, targets_b, lam
```

### Advanced Model Architectures and Pre-trained Models

**Vision Transformer Implementation**:
```python
import torchvision.models as models
from torchvision.models import vision_transformer

# Advanced model loading with customization
class ModelZoo:
    @staticmethod
    def load_pretrained_with_custom_head(model_name, num_classes, freeze_backbone=True):
        """Load pre-trained model with custom classification head"""
        
        if model_name.startswith('vit'):
            # Vision Transformer
            model = getattr(vision_transformer, model_name)(pretrained=True)
            
            # Freeze backbone if requested
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
            # Replace head
            model.heads = torch.nn.Linear(model.hidden_dim, num_classes)
            
        elif model_name.startswith('resnet'):
            # ResNet family
            model = getattr(models, model_name)(pretrained=True)
            
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
            # Replace final layer
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            
        elif model_name.startswith('efficientnet'):
            # EfficientNet
            model = getattr(models, model_name)(pretrained=True)
            
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        
        return model
    
    @staticmethod
    def create_ensemble(model_configs, num_classes):
        """Create ensemble of different architectures"""
        models_list = []
        
        for config in model_configs:
            model = ModelZoo.load_pretrained_with_custom_head(
                config['name'], num_classes, config.get('freeze', True)
            )
            models_list.append(model)
        
        return EnsembleModel(models_list)

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        
        # Equal weights if not provided
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average of predictions
        ensemble_output = sum(w * out for w, out in zip(self.weights, outputs))
        return ensemble_output

# Feature extraction and analysis
class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.get_activation(name))
                self.hooks.append(hook)
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook
    
    def extract_features(self, x):
        self.features.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

### Object Detection and Segmentation

**Advanced Detection Framework**:
```python
import torchvision.ops as ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class AdvancedDetectionSystem:
    def __init__(self, num_classes, backbone='resnet50'):
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Non-Maximum Suppression parameters
        self.nms_threshold = 0.5
        self.score_threshold = 0.05
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def predict(self, image, confidence_threshold=0.5):
        """Make predictions with post-processing"""
        self.model.eval()
        
        with torch.no_grad():
            prediction = self.model([image])
        
        pred = prediction[0]
        
        # Filter by confidence
        keep = pred['scores'] > confidence_threshold
        boxes = pred['boxes'][keep]
        scores = pred['scores'][keep]
        labels = pred['labels'][keep]
        
        # Apply NMS
        keep_nms = ops.nms(boxes, scores, self.nms_threshold)
        
        return {
            'boxes': boxes[keep_nms],
            'scores': scores[keep_nms],
            'labels': labels[keep_nms]
        }
    
    def compute_map(self, predictions, ground_truth, iou_thresholds=None):
        """Compute mean Average Precision"""
        if iou_thresholds is None:
            iou_thresholds = torch.arange(0.5, 1.0, 0.05)
        
        aps = []
        for iou_thresh in iou_thresholds:
            ap = self._compute_ap_at_iou(predictions, ground_truth, iou_thresh)
            aps.append(ap)
        
        return torch.stack(aps).mean()
    
    def _compute_ap_at_iou(self, predictions, ground_truth, iou_threshold):
        """Compute AP at specific IoU threshold"""
        # Implementation of AP calculation
        # This would involve sorting by confidence, computing precision-recall curve
        pass

# Instance segmentation with Mask R-CNN
class InstanceSegmentation:
    def __init__(self, num_classes):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def predict_with_masks(self, image, confidence_threshold=0.5):
        """Predict instances with segmentation masks"""
        self.model.eval()
        
        with torch.no_grad():
            prediction = self.model([image])
        
        pred = prediction[0]
        
        # Filter by confidence
        keep = pred['scores'] > confidence_threshold
        
        return {
            'boxes': pred['boxes'][keep],
            'labels': pred['labels'][keep],
            'scores': pred['scores'][keep],
            'masks': pred['masks'][keep]
        }
```

## Audio Processing Ecosystem - TorchAudio

### Audio Signal Processing and Feature Extraction

**Advanced Audio Transformations**:
```python
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

class AdvancedAudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Spectral transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=sample_rate // 2
        )
        
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
        )
        
        # Time-domain augmentations
        self.time_stretch = T.TimeStretch(hop_length=512, n_freq=1025)
        self.pitch_shift = T.PitchShift(sample_rate=sample_rate, n_steps=2)
        
        # Noise and distortion
        self.vol_aug = T.Vol(gain=1.0, gain_type='amplitude')
        
    def extract_features(self, waveform):
        """Extract comprehensive audio features"""
        features = {}
        
        # Time domain features
        features['rms'] = torch.sqrt(torch.mean(waveform ** 2, dim=1, keepdim=True))
        features['zcr'] = self._zero_crossing_rate(waveform)
        
        # Spectral features
        features['mel_spec'] = self.mel_spectrogram(waveform)
        features['mfcc'] = self.mfcc(waveform)
        features['spectral_centroid'] = self._spectral_centroid(features['mel_spec'])
        features['spectral_rolloff'] = self._spectral_rolloff(features['mel_spec'])
        
        # Chroma features
        features['chroma'] = self._chroma_features(waveform)
        
        return features
    
    def _zero_crossing_rate(self, waveform):
        """Compute zero crossing rate"""
        diff = torch.diff(torch.sign(waveform), dim=1)
        zcr = torch.sum(torch.abs(diff), dim=1, keepdim=True) / (2 * waveform.shape[1])
        return zcr
    
    def _spectral_centroid(self, mel_spec):
        """Compute spectral centroid"""
        freqs = torch.linspace(0, self.sample_rate // 2, mel_spec.shape[1])
        freqs = freqs.unsqueeze(0).unsqueeze(-1)  # Shape: [1, n_mels, 1]
        
        # Weighted average frequency
        magnitude = mel_spec.abs()
        centroid = torch.sum(magnitude * freqs, dim=1) / torch.sum(magnitude, dim=1)
        return centroid
    
    def _spectral_rolloff(self, mel_spec, roll_percent=0.85):
        """Compute spectral rolloff"""
        magnitude = mel_spec.abs()
        cumsum = torch.cumsum(magnitude, dim=1)
        total = cumsum[:, -1:, :]
        
        threshold = roll_percent * total
        rolloff_idx = torch.argmax((cumsum >= threshold).float(), dim=1)
        
        return rolloff_idx.float()
    
    def _chroma_features(self, waveform):
        """Extract chroma features"""
        # Simplified chroma computation
        stft = torch.stft(waveform.squeeze(0), n_fft=2048, 
                         hop_length=512, return_complex=True)
        magnitude = stft.abs()
        
        # Map frequencies to chroma bins (simplified)
        # In practice, this would use more sophisticated frequency mapping
        chroma = torch.zeros(12, magnitude.shape[-1])
        
        freq_bins = torch.fft.fftfreq(2048, 1/self.sample_rate)[:1025]
        for i, freq in enumerate(freq_bins):
            if freq > 0:
                chroma_bin = int(12 * torch.log2(freq / 440.0) % 12)
                chroma[chroma_bin] += magnitude[0, i]
        
        return chroma.unsqueeze(0)
    
    def augment_audio(self, waveform, augment_type='random'):
        """Apply audio augmentations"""
        if augment_type == 'pitch_shift':
            return self.pitch_shift(waveform)
        elif augment_type == 'time_stretch':
            # Random time stretch factor
            rate = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            return self.time_stretch(waveform, rate.item())
        elif augment_type == 'noise':
            noise = torch.randn_like(waveform) * 0.005
            return waveform + noise
        elif augment_type == 'volume':
            gain = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            return self.vol_aug(waveform, gain.item())
        else:
            # Random augmentation
            augmentations = ['pitch_shift', 'time_stretch', 'noise', 'volume']
            aug_type = torch.randint(len(augmentations), (1,)).item()
            return self.augment_audio(waveform, augmentations[aug_type])

# Speech processing pipeline
class SpeechProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Voice Activity Detection
        self.vad = T.Vad(sample_rate=sample_rate)
        
        # Resampling for different models
        self.resample_8k = T.Resample(sample_rate, 8000)
        self.resample_22k = T.Resample(sample_rate, 22050)
        
        # Noise reduction (simplified)
        self.wiener_filter = self._create_wiener_filter()
    
    def _create_wiener_filter(self):
        """Create a simple Wiener filter for noise reduction"""
        class WienerFilter(torch.nn.Module):
            def __init__(self, alpha=0.95):
                super().__init__()
                self.alpha = alpha
                self.noise_psd = None
            
            def forward(self, noisy_spec):
                if self.noise_psd is None:
                    # Estimate noise from first few frames
                    self.noise_psd = torch.mean(noisy_spec[:, :, :10].abs() ** 2, dim=2, keepdim=True)
                
                signal_psd = noisy_spec.abs() ** 2
                wiener_gain = signal_psd / (signal_psd + self.alpha * self.noise_psd)
                
                return noisy_spec * wiener_gain
        
        return WienerFilter()
    
    def preprocess_speech(self, waveform):
        """Complete speech preprocessing pipeline"""
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Voice activity detection
        waveform = self.vad(waveform)
        
        # Convert to spectrogram for filtering
        stft = torch.stft(waveform.squeeze(0), n_fft=1024, 
                         hop_length=256, return_complex=True)
        
        # Apply Wiener filter
        filtered_stft = self.wiener_filter(stft.unsqueeze(0))
        
        # Convert back to time domain
        filtered_waveform = torch.istft(filtered_stft.squeeze(0), 
                                       n_fft=1024, hop_length=256)
        
        return filtered_waveform.unsqueeze(0)
```

## Training Automation - PyTorch Lightning

### Lightning Module Design Patterns

**Advanced Lightning Module**:
```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

class AdvancedLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4, 
                 scheduler_config=None, loss_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {'name': 'cosine'}
        self.loss_weights = loss_weights
        
        # Metrics tracking
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
        # Advanced metrics
        self.train_f1 = torchmetrics.F1Score(num_classes=10, average='macro')
        self.val_f1 = torchmetrics.F1Score(num_classes=10, average='macro')
        
        # Loss tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Multi-task loss if applicable
        if isinstance(logits, dict):
            loss = self._compute_multi_task_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        
        # Metrics
        if not isinstance(logits, dict):
            preds = torch.argmax(logits, dim=1)
            self.train_accuracy(preds, y)
            self.train_f1(preds, y)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        
        # Store for epoch end
        self.training_step_outputs.append(loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if isinstance(logits, dict):
            loss = self._compute_multi_task_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y, weight=self.loss_weights)
            preds = torch.argmax(logits, dim=1)
            self.val_accuracy(preds, y)
            self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_accuracy(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy)
        
        return loss
    
    def on_train_epoch_end(self):
        # Compute epoch statistics
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.log('epoch_train_loss', epoch_loss)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('epoch_val_loss', epoch_loss)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Advanced optimizer configuration
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        config = {'optimizer': optimizer}
        
        # Learning rate scheduler
        if self.scheduler_config['name'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        
        elif self.scheduler_config['name'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        
        return config
    
    def _compute_multi_task_loss(self, logits_dict, targets):
        """Compute weighted multi-task loss"""
        total_loss = 0
        for task_name, task_logits in logits_dict.items():
            task_targets = targets[task_name] if isinstance(targets, dict) else targets
            task_loss = F.cross_entropy(task_logits, task_targets)
            
            # Apply task-specific weights
            weight = self.loss_weights.get(task_name, 1.0) if self.loss_weights else 1.0
            total_loss += weight * task_loss
            
            # Log individual task losses
            self.log(f'{task_name}_loss', task_loss, on_epoch=True)
        
        return total_loss

# Advanced training configuration
class LightningTrainingSystem:
    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        
        # Setup Lightning module
        self.lightning_model = AdvancedLightningModule(
            model=model,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            scheduler_config=config.get('scheduler', {}),
            loss_weights=config.get('loss_weights')
        )
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Setup loggers
        self.loggers = self._setup_loggers()
        
        # Setup trainer
        self.trainer = self._setup_trainer()
    
    def _setup_callbacks(self):
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['checkpoint_dir'],
            filename='{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.get('early_stopping', True):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Custom callbacks
        if 'custom_callbacks' in self.config:
            callbacks.extend(self.config['custom_callbacks'])
        
        return callbacks
    
    def _setup_loggers(self):
        loggers = []
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=self.config['log_dir'],
            name='tensorboard_logs'
        )
        loggers.append(tb_logger)
        
        # Weights & Biases logger (if configured)
        if self.config.get('use_wandb', False):
            wandb_logger = WandbLogger(
                project=self.config['project_name'],
                name=self.config.get('experiment_name')
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def _setup_trainer(self):
        trainer_args = {
            'max_epochs': self.config['max_epochs'],
            'callbacks': self.callbacks,
            'logger': self.loggers,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': self.config.get('num_gpus', 1),
            'precision': self.config.get('precision', 32),
            'gradient_clip_val': self.config.get('gradient_clip_val', 0.5),
            'accumulate_grad_batches': self.config.get('accumulate_grad_batches', 1),
            'val_check_interval': self.config.get('val_check_interval', 1.0),
            'check_val_every_n_epoch': self.config.get('check_val_every_n_epoch', 1),
        }
        
        # Distributed training configuration
        if self.config.get('distributed', False):
            trainer_args['strategy'] = 'ddp'
            trainer_args['sync_batchnorm'] = True
        
        return pl.Trainer(**trainer_args)
    
    def train(self):
        """Start training"""
        self.trainer.fit(
            model=self.lightning_model,
            datamodule=self.data_module
        )
        
        return self.trainer.callback_metrics
    
    def test(self, ckpt_path=None):
        """Run testing"""
        return self.trainer.test(
            model=self.lightning_model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path
        )

# Advanced data module
class AdvancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, 
                 train_transforms=None, val_transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
        # Data statistics for normalization
        self.dims = None
        self.num_classes = None
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load training and validation datasets
            self.train_dataset = CustomDataset(
                self.data_dir, 
                split='train',
                transform=self.train_transforms
            )
            
            self.val_dataset = CustomDataset(
                self.data_dir,
                split='val', 
                transform=self.val_transforms
            )
            
            # Set data properties
            self.dims = self.train_dataset[0][0].shape
            self.num_classes = len(self.train_dataset.classes)
        
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(
                self.data_dir,
                split='test',
                transform=self.val_transforms
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
```

## Model Serving - TorchServe

### Production Model Deployment

**TorchServe Integration**:
```python
import torch
import json
import logging
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class AdvancedModelHandler(BaseHandler):
    """Advanced TorchServe handler with comprehensive features"""
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.model = None
        self.preprocess_transforms = None
        self.postprocess_config = None
        self.class_mapping = None
        self.model_metadata = None
        
    def initialize(self, context):
        """Initialize handler with model and configurations"""
        properties = context.system_properties
        
        # Set device
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) 
            if torch.cuda.is_available() and properties.get("gpu_id") is not None 
            else "cpu"
        )
        
        # Load model
        model_dir = properties.get("model_dir")
        model_file = context.manifest.get("model", {}).get("modelFile", "model.pth")
        model_path = f"{model_dir}/{model_file}"
        
        # Load model with error handling
        try:
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Standard PyTorch model
                self.model = torch.load(model_path, map_location=self.device)
            elif model_path.endswith('.mar'):
                # TorchServe model archive
                self.model = torch.jit.load(model_path, map_location=self.device)
            
            self.model.eval()
            logging.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
        # Load configurations
        self._load_configurations(model_dir)
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        logging.info("Handler initialized successfully")
    
    def _load_configurations(self, model_dir):
        """Load model configurations and metadata"""
        try:
            # Load class mapping
            with open(f"{model_dir}/class_mapping.json", 'r') as f:
                self.class_mapping = json.load(f)
            
            # Load preprocessing config
            with open(f"{model_dir}/preprocess_config.json", 'r') as f:
                preprocess_config = json.load(f)
                self.preprocess_transforms = self._create_transforms(preprocess_config)
            
            # Load postprocessing config
            with open(f"{model_dir}/postprocess_config.json", 'r') as f:
                self.postprocess_config = json.load(f)
                
            # Load model metadata
            with open(f"{model_dir}/model_metadata.json", 'r') as f:
                self.model_metadata = json.load(f)
                
        except FileNotFoundError as e:
            logging.warning(f"Configuration file not found: {e}")
        except Exception as e:
            logging.error(f"Error loading configurations: {e}")
    
    def _create_transforms(self, config):
        """Create preprocessing transforms from configuration"""
        transform_list = []
        
        # Resize
        if 'resize' in config:
            transform_list.append(transforms.Resize(config['resize']))
        
        # Center crop
        if 'center_crop' in config:
            transform_list.append(transforms.CenterCrop(config['center_crop']))
        
        # To tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if 'normalize' in config:
            norm_config = config['normalize']
            transform_list.append(transforms.Normalize(
                mean=norm_config['mean'],
                std=norm_config['std']
            ))
        
        return transforms.Compose(transform_list)
    
    def preprocess(self, data):
        """Preprocess input data"""
        preprocessed_data = []
        
        for item in data:
            try:
                # Handle different input types
                if isinstance(item, dict):
                    if 'data' in item:
                        # Base64 encoded image
                        image_data = item['data']
                        # Decode and process image
                        processed_image = self._process_image_data(image_data)
                    elif 'body' in item:
                        # JSON body
                        processed_image = self._process_json_body(item['body'])
                    else:
                        raise ValueError("Invalid input format")
                else:
                    # Direct image data
                    processed_image = self._process_image_data(item)
                
                # Apply transforms
                if self.preprocess_transforms:
                    processed_image = self.preprocess_transforms(processed_image)
                
                preprocessed_data.append(processed_image)
                
            except Exception as e:
                logging.error(f"Preprocessing error: {str(e)}")
                raise
        
        return torch.stack(preprocessed_data).to(self.device)
    
    def _process_image_data(self, image_data):
        """Process raw image data"""
        from PIL import Image
        import base64
        import io
        
        if isinstance(image_data, str):
            # Base64 encoded
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # Assume PIL Image or tensor
            image = image_data
        
        return image
    
    def _process_json_body(self, body):
        """Process JSON body input"""
        import json
        
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        data = json.loads(body)
        
        # Handle different JSON formats
        if 'image' in data:
            return self._process_image_data(data['image'])
        elif 'data' in data:
            return torch.FloatTensor(data['data'])
        else:
            raise ValueError("Unsupported JSON format")
    
    def inference(self, data):
        """Run model inference"""
        try:
            with torch.no_grad():
                # Model forward pass
                outputs = self.model(data)
                
                # Handle different output types
                if isinstance(outputs, dict):
                    # Multi-output model
                    return outputs
                else:
                    # Single output
                    return outputs
                    
        except Exception as e:
            logging.error(f"Inference error: {str(e)}")
            raise
    
    def postprocess(self, inference_output):
        """Postprocess model outputs"""
        postprocessed_results = []
        
        # Handle batch outputs
        for i in range(inference_output.shape[0]):
            output = inference_output[i]
            
            try:
                if self.postprocess_config:
                    if self.postprocess_config.get('type') == 'classification':
                        result = self._postprocess_classification(output)
                    elif self.postprocess_config.get('type') == 'detection':
                        result = self._postprocess_detection(output)
                    else:
                        result = self._postprocess_default(output)
                else:
                    result = self._postprocess_default(output)
                
                postprocessed_results.append(result)
                
            except Exception as e:
                logging.error(f"Postprocessing error: {str(e)}")
                raise
        
        return postprocessed_results
    
    def _postprocess_classification(self, output):
        """Postprocess classification output"""
        # Apply softmax
        probabilities = torch.nn.functional.softmax(output, dim=0)
        
        # Get top-k predictions
        top_k = self.postprocess_config.get('top_k', 5)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Create result
        result = {
            'predictions': []
        }
        
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            class_name = self.class_mapping.get(str(idx), f'class_{idx}')
            result['predictions'].append({
                'class': class_name,
                'probability': prob,
                'class_index': idx
            })
        
        return result
    
    def _postprocess_detection(self, output):
        """Postprocess object detection output"""
        # Assuming output contains boxes, scores, labels
        result = {
            'detections': []
        }
        
        confidence_threshold = self.postprocess_config.get('confidence_threshold', 0.5)
        
        # Filter by confidence
        valid_detections = output['scores'] > confidence_threshold
        
        boxes = output['boxes'][valid_detections]
        scores = output['scores'][valid_detections] 
        labels = output['labels'][valid_detections]
        
        for box, score, label in zip(boxes, scores, labels):
            class_name = self.class_mapping.get(str(label.item()), f'class_{label.item()}')
            
            result['detections'].append({
                'bbox': box.tolist(),
                'score': score.item(),
                'class': class_name,
                'class_index': label.item()
            })
        
        return result
    
    def _postprocess_default(self, output):
        """Default postprocessing"""
        if output.dim() == 0:
            # Scalar output
            return {'prediction': output.item()}
        else:
            # Vector output
            return {'predictions': output.tolist()}

# Model packaging utility
class ModelPackager:
    @staticmethod
    def create_model_archive(model_name, model_path, handler_path, 
                           config_dir, version="1.0"):
        """Create TorchServe model archive"""
        import subprocess
        import os
        
        # Prepare command
        cmd = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", version,
            "--model-file", model_path,
            "--handler", handler_path,
            "--extra-files", f"{config_dir}/",
            "--export-path", "./model-store/",
            "--archive-format", "no-archive"
        ]
        
        # Execute command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Model archive created successfully: {model_name}.mar")
                return True
            else:
                print(f"Error creating archive: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    @staticmethod
    def deploy_model(model_name, model_archive_path, 
                    management_api="http://localhost:8081"):
        """Deploy model to TorchServe"""
        import requests
        
        # Register model
        register_url = f"{management_api}/models"
        params = {
            'model_name': model_name,
            'url': model_archive_path,
            'initial_workers': 1,
            'synchronous': True
        }
        
        try:
            response = requests.post(register_url, params=params)
            if response.status_code == 200:
                print(f"Model {model_name} deployed successfully")
                return True
            else:
                print(f"Deployment failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Deployment error: {str(e)}")
            return False
```

## Specialized Domain Libraries

### Natural Language Processing - TorchText and HuggingFace Integration

**Advanced NLP Pipeline**:
```python
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from torch.utils.data import Dataset
import torch.nn as nn

class AdvancedNLPSystem:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Data collator for dynamic padding
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
    def create_custom_model(self, model_config):
        """Create custom model architecture"""
        class CustomNLPModel(nn.Module):
            def __init__(self, base_model, num_labels, hidden_dim=768):
                super().__init__()
                self.base_model = base_model
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_labels)
                )
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use pooled output
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                
                return {
                    'loss': loss,
                    'logits': logits,
                    'hidden_states': outputs.hidden_states
                }
        
        # Load base model without classification head
        base_model = AutoModel.from_pretrained(self.model_name)
        custom_model = CustomNLPModel(base_model, self.num_labels)
        
        return custom_model
    
    def setup_training(self, train_dataset, eval_dataset, output_dir):
        """Setup training configuration"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to="tensorboard",
            dataloader_pin_memory=True,
            gradient_checkpointing=True,  # Save memory
            fp16=True,  # Mixed precision
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

### Reinforcement Learning - TorchRL Integration

**RL Environment and Agent Framework**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

class AdvancedDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=64, target_update_freq=1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size, batch_size)
        self.step_count = 0
        
        # Performance tracking
        self.losses = []
        self.rewards_history = []
        
    def _build_network(self):
        """Build the Q-network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def step(self, state, action, reward, next_state, done):
        """Save experience and learn"""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn if enough samples
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self._learn(experiences)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._update_target_network()
    
    def _learn(self, experiences):
        """Update Q-network using batch of experiences"""
        states, actions, rewards, next_states, dones = experiences
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            # Double DQN: use main network to select actions
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track loss
        self.losses.append(loss.item())
    
    def _update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).cuda()
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).cuda()
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).cuda()
        next_states = torch.FloatTensor([e.next_state for e in experiences]).cuda()
        dones = torch.BoolTensor([e.done for e in experiences]).unsqueeze(1).cuda()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)

# Policy Gradient Agent
class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, value_coef=0.5, entropy_coef=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage
        self.memory = PPOMemory()
        
    def act(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()
    
    def update(self):
        """Update policy using PPO"""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Convert to tensors
        old_states = torch.FloatTensor(self.memory.states).to(self.device)
        old_actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Calculate ratio
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            # Calculate advantages
            advantages = rewards - state_values.squeeze()
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Total loss
            loss = -torch.min(surr1, surr2) + self.value_coef * F.mse_loss(state_values.squeeze(), rewards) - self.entropy_coef * dist_entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class PPOMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def add(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
```

## Key Questions for Review

### Ecosystem Understanding
1. **Library Integration**: How do different PyTorch ecosystem libraries integrate with the core framework while maintaining compatibility?

2. **Domain Specialization**: What advantages do specialized libraries like TorchVision and TorchAudio provide over implementing functionality from scratch?

3. **Tool Selection**: What factors should guide the selection of ecosystem tools for specific project requirements?

### Computer Vision Tools
4. **TorchVision Features**: How can TorchVision's transforms, models, and utilities be effectively combined for complex vision tasks?

5. **Model Customization**: What are the best practices for customizing pre-trained models from TorchVision for specific applications?

6. **Performance Optimization**: How can computer vision pipelines be optimized for both training and inference performance?

### Training Frameworks
7. **Lightning Benefits**: What specific advantages does PyTorch Lightning provide over vanilla PyTorch for research and production?

8. **Configuration Management**: How should complex training configurations be managed and versioned in Lightning-based projects?

9. **Scaling Training**: What Lightning features are most important for scaling training across multiple GPUs and nodes?

### Production Deployment
10. **TorchServe Deployment**: What are the key considerations when deploying models using TorchServe in production environments?

11. **Model Versioning**: How can model versioning and A/B testing be implemented using PyTorch ecosystem tools?

12. **Monitoring Integration**: How can model serving be integrated with monitoring and logging systems?

### Specialized Applications
13. **Audio Processing**: What unique challenges does audio processing present, and how does TorchAudio address them?

14. **NLP Integration**: How can PyTorch be effectively integrated with modern NLP libraries like HuggingFace Transformers?

15. **Reinforcement Learning**: What are the key components needed for implementing RL algorithms in PyTorch?

## Conclusion

PyTorch Ecosystem Tools represent a comprehensive and mature collection of specialized libraries, frameworks, and utilities that significantly extend PyTorch's capabilities across diverse domains while maintaining the framework's core principles of flexibility, ease of use, and research-friendly design. The exploration of computer vision tools like TorchVision, audio processing capabilities through TorchAudio, training automation via PyTorch Lightning, production deployment with TorchServe, and specialized domain libraries demonstrates how the PyTorch ecosystem has evolved to provide complete end-to-end solutions for modern deep learning development and deployment.

**Comprehensive Coverage**: The breadth of ecosystem tools covers every aspect of the machine learning workflow, from data preprocessing and model development to training automation, hyperparameter optimization, and production deployment, providing practitioners with battle-tested, optimized implementations that accelerate development while maintaining high quality and performance standards.

**Domain Expertise**: Specialized libraries like TorchVision for computer vision and TorchAudio for audio processing incorporate domain-specific knowledge and best practices that would be difficult and time-consuming to implement independently, enabling researchers and practitioners to focus on novel algorithmic contributions rather than infrastructure development.

**Production Readiness**: Tools like TorchServe and PyTorch Lightning bridge the gap between research experimentation and production deployment by providing robust, scalable infrastructure components that handle the complex operational requirements of real-world ML systems while maintaining compatibility with PyTorch's dynamic graph approach.

**Integration Excellence**: The seamless integration between ecosystem components demonstrates careful architectural design that preserves PyTorch's flexibility while providing higher-level abstractions that reduce boilerplate code and common implementation errors, enabling more productive and reliable development workflows.

**Community Innovation**: The vibrant ecosystem reflects PyTorch's success in fostering community-driven innovation, with contributions from both Facebook AI Research and the broader open-source community creating specialized tools that address emerging needs and incorporate cutting-edge research developments.

Understanding and effectively utilizing PyTorch ecosystem tools enables practitioners to leverage decades of collective expertise and optimization work while maintaining the experimental flexibility that makes PyTorch particularly suitable for research and rapid prototyping. This comprehensive ecosystem provides the foundation for building sophisticated deep learning applications that can scale from research prototypes to production systems while maintaining code clarity, performance optimization, and operational reliability.