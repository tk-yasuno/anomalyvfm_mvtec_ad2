#!/usr/bin/env python3
# AnomalyVFM v1.3 - Multi-Scale LoRA for comprehensive anomaly detection
# 128-256-512 multi-resolution LoRA adaptation with feature fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import numpy as np
import cv2
import time
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from dataset_ad2 import AD2TrainDataset, AD2TestDataset
from pathlib import Path
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class MultiScaleLoRALayer(nn.Module):
    """Multi-Scale Low-Rank Adaptation Layer with scale-specific parameters"""
    
    def __init__(self, input_dim, scale_size, rank=16, alpha=32):
        super().__init__()
        self.scale_size = scale_size
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Scale-specific LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, input_dim))
        
        # Scale-specific normalization and dropout
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Scale-specific attention weights
        self.scale_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, N, D] or [B, D]
        original_x = x
        
        # Apply layer normalization
        normalized_x = self.layer_norm(x)
        
        # Apply LoRA: x + alpha/rank * x @ A @ B
        if len(x.shape) == 3:  # [B, N, D]
            B, N, D = x.shape
            x_flat = normalized_x.reshape(-1, D)  # [B*N, D]
            lora_out = x_flat @ self.lora_A @ self.lora_B  # [B*N, D]
            lora_out = lora_out.reshape(B, N, D)  # [B, N, D]
            
            # Compute scale-specific attention
            attention_weights = self.scale_attention(x_flat).reshape(B, N, 1)
        else:  # [B, D]
            lora_out = normalized_x @ self.lora_A @ self.lora_B
            attention_weights = self.scale_attention(normalized_x).unsqueeze(-1)
            
        # Apply scale-specific attention and dropout
        adapted_features = self.scale * self.dropout(lora_out) * attention_weights
        adapted_x = original_x + adapted_features
        
        return adapted_x, attention_weights.squeeze(-1)

class MultiScaleFeatureFusion(nn.Module):
    """Multi-Scale Feature Fusion with learnable weights"""
    
    def __init__(self, feature_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # Scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Scale-wise projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_scales)
        ])
        
    def forward(self, multi_scale_features, scale_attentions):
        """
        multi_scale_features: List of [B, N, D] or [B, D] features from different scales
        scale_attentions: List of attention weights from each scale
        """
        B = multi_scale_features[0].shape[0]
        
        # Normalize scale weights
        normalized_weights = F.softmax(self.scale_weights, dim=0)
        
        # Project each scale's features
        projected_features = []
        for i, (features, attention) in enumerate(zip(multi_scale_features, scale_attentions)):
            if len(features.shape) == 3:
                # Average pooling for patch features
                pooled_features = features.mean(dim=1)  # [B, D]
            else:
                pooled_features = features
                
            # Apply scale-specific projection
            projected = self.scale_projections[i](pooled_features)
            
            # Weight by scale importance and attention
            if len(attention.shape) > 1:
                attention_weight = attention.mean(dim=1)  # [B]
            else:
                attention_weight = attention
                
            weighted_features = projected * normalized_weights[i] * attention_weight.unsqueeze(-1)
            projected_features.append(weighted_features)
        
        # Concatenate all scale features
        concatenated = torch.cat(projected_features, dim=1)  # [B, D*num_scales]
        
        # Apply fusion layers
        fused_features = self.fusion_layers(concatenated)  # [B, D]
        
        # Apply cross-attention for inter-scale relationships
        scale_stack = torch.stack(projected_features, dim=1)  # [B, num_scales, D]
        attended_features, _ = self.cross_attention(scale_stack, scale_stack, scale_stack)
        attended_mean = attended_features.mean(dim=1)  # [B, D]
        
        # Final fusion
        final_features = fused_features + attended_mean
        
        return final_features

class MultiScaleSyntheticGenerator:
    """Multi-Scale Synthetic Anomaly Generation for different resolution training"""
    
    def __init__(self, scales=[128, 256, 512]):
        self.scales = scales
        
    def generate_multiscale_anomalies(self, normal_images, category, num_anomalies=90):
        """Generate synthetic anomalies at multiple scales"""
        print(f"    Generating multi-scale synthetic anomalies for {category}")
        
        all_scale_anomalies = {}
        
        for scale in self.scales:
            print(f"      Scale {scale}x{scale}: ", end="")
            
            # Resize normal images to current scale
            scaled_normals = []
            for img in normal_images:
                if len(img.shape) == 3:
                    resized = cv2.resize(img, (scale, scale))
                    scaled_normals.append(resized)
            
            # Generate scale-specific anomalies
            scale_anomalies = self.generate_scale_specific_anomalies(
                scaled_normals, scale, num_anomalies // len(self.scales) + 10
            )
            all_scale_anomalies[scale] = scale_anomalies
            print(f"{len(scale_anomalies)} samples")
        
        return all_scale_anomalies
    
    def generate_scale_specific_anomalies(self, normal_images, scale, num_anomalies):
        """Generate anomalies specific to a resolution scale"""
        anomalies = []
        
        # Scale-adaptive parameters
        if scale == 128:
            # Low resolution: focus on large structural anomalies
            texture_ratio = 0.2
            structural_ratio = 0.5
            contextual_ratio = 0.3
            patch_size_range = (10, 30)
            noise_intensity = 25
        elif scale == 256:
            # Medium resolution: balanced approach
            texture_ratio = 0.4
            structural_ratio = 0.3
            contextual_ratio = 0.3
            patch_size_range = (20, 60)
            noise_intensity = 30
        else:  # 512
            # High resolution: focus on fine texture anomalies
            texture_ratio = 0.5
            structural_ratio = 0.2
            contextual_ratio = 0.3
            patch_size_range = (40, 120)
            noise_intensity = 35
        
        # Generate texture anomalies
        texture_count = int(num_anomalies * texture_ratio)
        for i in range(texture_count):
            base_img = normal_images[np.random.randint(len(normal_images))].copy()
            anomaly_img = self.add_texture_anomaly(base_img, patch_size_range, noise_intensity)
            anomalies.append(anomaly_img)
        
        # Generate structural anomalies
        structural_count = int(num_anomalies * structural_ratio)
        for i in range(structural_count):
            base_img = normal_images[np.random.randint(len(normal_images))].copy()
            anomaly_img = self.add_structural_anomaly(base_img, scale)
            anomalies.append(anomaly_img)
        
        # Generate contextual anomalies
        contextual_count = num_anomalies - texture_count - structural_count
        for i in range(contextual_count):
            img1 = normal_images[np.random.randint(len(normal_images))]
            img2 = normal_images[np.random.randint(len(normal_images))]
            anomaly_img = self.add_contextual_anomaly(img1, img2, patch_size_range)
            anomalies.append(anomaly_img)
        
        return anomalies
    
    def add_texture_anomaly(self, img, patch_size_range, noise_intensity):
        """Add texture-based anomaly"""
        h, w = img.shape[:2]
        patch_size = np.random.randint(patch_size_range[0], patch_size_range[1])
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        
        noise = np.random.normal(0, noise_intensity, (patch_size, patch_size, 3))
        img[y:y+patch_size, x:x+patch_size] = np.clip(
            img[y:y+patch_size, x:x+patch_size] + noise, 0, 255
        )
        return img.astype(np.uint8)
    
    def add_structural_anomaly(self, img, scale):
        """Add structural anomaly scaled to resolution"""
        h, w = img.shape[:2]
        shape_type = np.random.choice(['circle', 'rectangle', 'line'])
        
        # Scale-adaptive sizes
        size_factor = scale / 512.0
        
        if shape_type == 'circle':
            center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            radius = int(np.random.randint(8, 40) * size_factor)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.circle(img, center, radius, color, -1)
        elif shape_type == 'rectangle':
            pt1 = (np.random.randint(0, w//2), np.random.randint(0, h//2))
            size = int(np.random.randint(20, 80) * size_factor)
            pt2 = (pt1[0] + size, pt1[1] + size)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(img, pt1, pt2, color, -1)
        else:  # line
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            thickness = max(1, min(100, int(np.random.randint(2, 10) * size_factor)))
            cv2.line(img, pt1, pt2, color, thickness)
        
        return img.astype(np.uint8)
    
    def add_contextual_anomaly(self, img1, img2, patch_size_range):
        """Add contextual anomaly by patch transplantation"""
        img = img1.copy()
        h, w = img.shape[:2]
        
        patch_size = np.random.randint(patch_size_range[0], patch_size_range[1])
        x1 = np.random.randint(0, w - patch_size)
        y1 = np.random.randint(0, h - patch_size)
        
        x2 = np.random.randint(0, img2.shape[1] - patch_size)
        y2 = np.random.randint(0, img2.shape[0] - patch_size)
        
        patch = img2[y2:y2+patch_size, x2:x2+patch_size]
        alpha = np.random.uniform(0.6, 0.8)
        img[y1:y1+patch_size, x1:x1+patch_size] = (
            alpha * patch + (1-alpha) * img[y1:y1+patch_size, x1:x1+patch_size]
        ).astype(np.uint8)
        
        return img

class AnomalyVFMv13:
    """AnomalyVFM v1.3 with Multi-Scale LoRA"""
    
    def __init__(self, device='cuda', scales=[128, 256, 512]):
        self.device = device
        self.scales = scales
        self.num_scales = len(scales)
        
        print("🚀 AnomalyVFM v1.3 - Multi-Scale LoRA")
        print("=" * 50)
        print("🔍 Multi-resolution anomaly detection")
        print(f"📏 Scales: {scales}")
        print("🎯 Scale-specific LoRA adaptation")
        print("🔗 Cross-scale feature fusion")
        print("⚡ DINOv2-Base + Multi-Scale LoRA")
        print("🌟 Comprehensive anomaly coverage")
        print("=" * 50 + "\n")
        
        # Initialize components
        self.synthetic_generator = MultiScaleSyntheticGenerator(scales)
        self.load_multiscale_model()
        
    def load_multiscale_model(self):
        """Load model with multi-scale LoRA configuration"""
        print("🏗️ Loading multi-scale LoRA model...")
        
        # Single DINOv2 backbone (518x518 input)
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.to(self.device)
        
        # Get feature dimension with DINOv2 standard size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 518, 518).to(self.device)
            dummy_features = self.backbone(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
        
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Number of scales: {self.num_scales}")
        
        # Multi-scale LoRA adapters
        self.multiscale_loras = nn.ModuleDict()
        for scale in self.scales:
            # Scale-specific rank (higher for higher resolution)
            scale_rank = max(8, int(16 * (scale / 256)))
            scale_alpha = scale_rank * 2
            
            lora_layers = nn.ModuleList([
                MultiScaleLoRALayer(self.feature_dim, scale, scale_rank, scale_alpha),
                MultiScaleLoRALayer(self.feature_dim, scale, scale_rank, scale_alpha),
                MultiScaleLoRALayer(self.feature_dim, scale, scale_rank, scale_alpha)
            ])
            self.multiscale_loras[str(scale)] = lora_layers.to(self.device)
            print(f"   Scale {scale}: Rank={scale_rank}, Alpha={scale_alpha}")
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(self.feature_dim, self.num_scales).to(self.device)
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        print("✅ Multi-scale LoRA model loaded")
    
    def extract_multiscale_features(self, dataloader, use_lora=True):
        """Extract features at multiple scales with LoRA"""
        all_scale_features = {str(scale): [] for scale in self.scales}
        all_scale_attentions = {str(scale): [] for scale in self.scales}
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Multi-scale extraction", ncols=70):
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 1:
                    images = batch_data[0]
                else:
                    images = batch_data
                
                if not isinstance(images, torch.Tensor):
                    continue
                
                batch_size = images.shape[0]
                
                # Extract features at each scale
                for scale in self.scales:
                    scale_key = str(scale)
                    
                    # Resize images to current scale, then to DINOv2 input size (518)
                    scaled_images = F.interpolate(
                        images, size=(scale, scale), mode='bilinear', align_corners=False
                    )
                    dinov2_images = F.interpolate(
                        scaled_images, size=(518, 518), mode='bilinear', align_corners=False
                    ).to(self.device)
                    
                    # Extract base features using single backbone
                    base_features = self.backbone(dinov2_images)
                    if len(base_features.shape) == 3:
                        base_features = base_features[:, 0, :]  # CLS token
                    
                    # Apply multi-scale LoRA if enabled
                    if use_lora:
                        adapted_features = base_features
                        scale_attentions = []
                        
                        for lora_layer in self.multiscale_loras[scale_key]:
                            adapted_features, attention = lora_layer(adapted_features)
                            # Ensure attention is always 1D per sample
                            if len(attention.shape) > 1:
                                attention = attention.mean(dim=-1)  # Average over last dimension
                            scale_attentions.append(attention.cpu())
                        
                        final_features = F.normalize(adapted_features, dim=1)
                        if scale_attentions:
                            mean_attention = torch.stack(scale_attentions).mean(dim=0)
                        else:
                            mean_attention = torch.ones(batch_size)
                    else:
                        final_features = F.normalize(base_features, dim=1)
                        mean_attention = torch.ones(batch_size)
                    
                    all_scale_features[scale_key].append(final_features.cpu())
                    all_scale_attentions[scale_key].append(mean_attention)
        
        # Concatenate features for each scale
        for scale in self.scales:
            scale_key = str(scale)
            if all_scale_features[scale_key]:
                features_tensor = torch.cat(all_scale_features[scale_key], dim=0)
                # Ensure 2D shape for EmpiricalCovariance
                if len(features_tensor.shape) > 2:
                    features_tensor = features_tensor.view(features_tensor.shape[0], -1)
                all_scale_features[scale_key] = features_tensor.numpy()
                
                attentions_tensor = torch.cat(all_scale_attentions[scale_key], dim=0)
                if len(attentions_tensor.shape) > 1:
                    attentions_tensor = attentions_tensor.view(-1)
                all_scale_attentions[scale_key] = attentions_tensor.numpy()
            else:
                all_scale_features[scale_key] = np.array([])
                all_scale_attentions[scale_key] = np.array([])
        
        return all_scale_features, all_scale_attentions
    
    def train_multiscale_lora(self, train_dataset, category, epochs=10):
        """Train multi-scale LoRA with scale-specific synthetic data"""
        print("🏗️ Generating multi-scale synthetic training data...")
        
        # Load normal images
        normal_images = []
        for i in range(min(50, len(train_dataset))):
            try:
                data = train_dataset[i]
                if isinstance(data, tuple):
                    img = data[0]
                else:
                    img = data
                    
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy() * 255
                normal_images.append(img.astype(np.uint8))
            except Exception as e:
                continue
        
        # Generate multi-scale synthetic anomalies
        multiscale_anomalies = self.synthetic_generator.generate_multiscale_anomalies(
            normal_images, category, 90
        )
        
        # Set up training
        all_params = []
        for scale_key in self.multiscale_loras:
            all_params.extend(list(self.multiscale_loras[scale_key].parameters()))
        all_params.extend(list(self.feature_fusion.parameters()))
        all_params.extend(list(self.anomaly_head.parameters()))
        
        optimizer = optim.AdamW(all_params, lr=1e-4, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"🏋️ Training Multi-Scale LoRA for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Set training mode
            for scale_key in self.multiscale_loras:
                self.multiscale_loras[scale_key].train()
            self.feature_fusion.train()
            self.anomaly_head.train()
            
            # Training with multi-scale data
            batch_size = 6
            for i in range(0, len(normal_images), batch_size//2):
                batch_normal = normal_images[i:i+batch_size//2]
                
                # Collect anomalies from all scales
                batch_anomalies = []
                for scale in self.scales:
                    scale_anomalies = multiscale_anomalies[scale]
                    if i < len(scale_anomalies):
                        batch_anomalies.append(scale_anomalies[i])
                
                if len(batch_anomalies) < len(batch_normal):
                    continue
                
                # Create batch
                batch_images = []
                batch_labels = []
                
                # Add normal images
                for img in batch_normal:
                    if len(img.shape) == 3:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
                    batch_images.append(img_tensor)
                    batch_labels.append(0)
                
                # Add anomaly images  
                for img in batch_anomalies[:len(batch_normal)]:
                    if len(img.shape) == 3:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
                    batch_images.append(img_tensor)
                    batch_labels.append(1)
                
                if len(batch_images) < 2:
                    continue
                
                # Forward pass with multi-scale processing
                optimizer.zero_grad()
                
                multiscale_features = []
                multiscale_attentions = []
                
                for scale in self.scales:
                    scale_key = str(scale)
                    
                    # Resize batch to current scale, then to DINOv2 size (518)
                    resized_batch = []
                    for img_tensor in batch_images:
                        # First resize to target scale
                        scaled = F.interpolate(
                            img_tensor.unsqueeze(0), 
                            size=(scale, scale), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        # Then resize to DINOv2 input size
                        dinov2_sized = F.interpolate(
                            scaled, 
                            size=(518, 518), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        resized_batch.append(dinov2_sized.squeeze(0))
                    
                    if len(resized_batch) == 0:
                        continue
                        
                    batch_tensor = torch.stack(resized_batch).to(self.device)
                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                    
                    # Extract base features using single backbone
                    base_features = self.backbone(batch_tensor)
                    if len(base_features.shape) == 3:
                        base_features = base_features[:, 0, :]
                    
                    # Apply multi-scale LoRA
                    adapted_features = base_features
                    scale_attentions = []
                    
                    for lora_layer in self.multiscale_loras[scale_key]:
                        adapted_features, attention = lora_layer(adapted_features)
                        # Ensure attention is always 1D per sample
                        if len(attention.shape) > 1:
                            attention = attention.mean(dim=1)
                        scale_attentions.append(attention)
                    
                    mean_attention = torch.stack(scale_attentions).mean(dim=0)
                    multiscale_features.append(adapted_features)
                    multiscale_attentions.append(mean_attention)
                
                if len(multiscale_features) == 0:
                    continue
                
                # Multi-scale feature fusion
                fused_features = self.feature_fusion(multiscale_features, multiscale_attentions)
                
                # Anomaly prediction
                predictions = self.anomaly_head(fused_features).squeeze()
                
                # Compute loss
                loss = criterion(predictions, labels_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("✅ Multi-Scale LoRA training completed")
    
    def evaluate_category_multiscale(self, category, data_root='./data/MVTec AD2', use_lora=True):
        """Evaluate category with multi-scale LoRA"""
        print(f"\n{category.upper():=^60}")
        
        start_time = time.time()
        
        # Load datasets
        train_dataset = AD2TrainDataset(data_root, category, image_size=self.scales[-1])  # Use highest resolution
        test_dataset = AD2TestDataset(data_root, category, image_size=self.scales[-1])
        
        train_loader = DataLoader(train_dataset, batch_size=6, shuffle=False, num_workers=2, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=2, drop_last=True)
        
        # Count samples
        normal_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 0)
        anomaly_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 1)
        
        print(f"Train: {len(train_dataset)}, Test: {normal_count} normal + {anomaly_count} anomaly")
        print(f"Multi-scale processing: {self.scales}")
        
        # Train multi-scale LoRA
        if use_lora:
            self.train_multiscale_lora(train_dataset, category, epochs=10)
        
        # Extract multi-scale features
        print("Extracting multi-scale features...")
        train_features, train_attentions = self.extract_multiscale_features(train_loader, use_lora)
        test_features, test_attentions = self.extract_multiscale_features(test_loader, use_lora)
        
        # Multi-scale ensemble anomaly detection
        ensemble_scores = []
        scale_weights = []
        
        for scale in self.scales:
            scale_key = str(scale)
            
            # Skip if no features for this scale
            if len(train_features[scale_key]) == 0 or len(test_features[scale_key]) == 0:
                print(f"   Scale {scale}: Skipped (no features)")
                continue
            
            # Anomaly detection for each scale
            try:
                cov_estimator = EmpiricalCovariance().fit(train_features[scale_key])
                train_distances = cov_estimator.mahalanobis(train_features[scale_key])
                test_distances = cov_estimator.mahalanobis(test_features[scale_key])
                
                # Normalize scores
                train_mean = np.mean(train_distances)
                train_std = np.std(train_distances)
                scale_scores = (test_distances - train_mean) / (train_std + 1e-8)
                
                ensemble_scores.append(scale_scores)
                
                # Compute scale weight based on attention
                avg_attention = np.mean(test_attentions[scale_key]) if len(test_attentions[scale_key]) > 0 else 1.0
                scale_weights.append(avg_attention)
                
                print(f"   Scale {scale}: Avg attention = {avg_attention:.4f}")
            except Exception as e:
                print(f"   Scale {scale}: Error - {str(e)}")
                continue
        
        # Handle ensemble calculation
        if len(ensemble_scores) == 0:
            print("   Warning: No valid scale scores, using fallback")
            final_scores = np.random.random(len(test_dataset))
            scale_weights = np.array([1.0])
        else:
            # Normalize scale weights
            scale_weights = np.array(scale_weights)
            scale_weights = scale_weights / (np.sum(scale_weights) + 1e-8)
            
            # Weighted ensemble
            ensemble_scores = np.array(ensemble_scores)
            final_scores = np.sum(ensemble_scores * scale_weights.reshape(-1, 1), axis=0)
        
        # Get labels that match the processed samples (accounting for drop_last=True)
        actual_test_samples = len(final_scores) if isinstance(final_scores, np.ndarray) else len(test_dataset)
        test_labels = [test_dataset[i][1] for i in range(min(actual_test_samples, len(test_dataset)))]
        test_labels = np.array(test_labels)
        
        # Ensure labels and scores have same length
        min_length = min(len(test_labels), len(final_scores))
        test_labels = test_labels[:min_length]
        final_scores = final_scores[:min_length]
        
        print(f"Final evaluation: {len(test_labels)} samples, {len(final_scores)} scores")
        
        # Compute AUC
        if len(test_labels) > 0 and len(final_scores) > 0 and len(np.unique(test_labels)) > 1:
            image_auc = roc_auc_score(test_labels, final_scores)
        else:
            print("Warning: Cannot compute AUC - insufficient data or no class variation")
            image_auc = 0.5
        
        processing_time = time.time() - start_time
        
        method_status = f"Multi-Scale LoRA ({'-'.join(map(str, self.scales))})" if use_lora else "Baseline"
        print(f"Image-level AUC ({method_status}): {image_auc:.4f}")
        print(f"Scale weights: {scale_weights}")
        print(f"Processing time: {processing_time:.1f}s")
        
        return {
            'category': category,
            'image_auc': image_auc,
            'use_lora': use_lora,
            'scales': self.scales,
            'scale_weights': scale_weights.tolist(),
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'processing_time': processing_time
        }

def main():
    """AnomalyVFM v1.3 with Multi-Scale LoRA evaluation"""
    model = AnomalyVFMv13(scales=[128, 256, 512])
    
    # All 7 categories for multi-scale evaluation
    categories = ['fruit_jelly', 'fabric', 'can', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    results = []
    
    print(f"AnomalyVFM v1.3 Multi-Scale LoRA evaluation for all {len(categories)} categories...\n")
    
    for i, category in enumerate(categories, 1):
        print(f"[{i}/{len(categories)}] Processing {category}")
        
        # Evaluate with multi-scale LoRA
        result_lora = model.evaluate_category_multiscale(category, use_lora=True)
        results.append(result_lora)
        
        # Evaluate baseline for comparison
        result_baseline = model.evaluate_category_multiscale(category, use_lora=False)
        results.append(result_baseline)
    
    # Summary
    print(f"\n{'ANOMALYVFM v1.3 MULTI-SCALE RESULTS':=^80}")
    print(f"{'Category':<12} {'Method':<25} {'AUC':<8} {'Weights':<20} {'Time(s)':<8}")
    print("-" * 80)
    
    multiscale_aucs = []
    baseline_aucs = []
    
    for result in results:
        if result['use_lora']:
            method = f"Multi-Scale LoRA"
            weights = str([f"{w:.2f}" for w in result['scale_weights']])[:18]
            multiscale_aucs.append(result['image_auc'])
        else:
            method = "Baseline"
            weights = "Standard"
            baseline_aucs.append(result['image_auc'])
            
        print(f"{result['category']:<12} {method:<25} {result['image_auc']:.4f}   "
              f"{weights:<20} {result['processing_time']:.1f}")
    
    print("-" * 80)
    print(f"{'AVERAGE':<12} {'Multi-Scale LoRA':<25} {np.mean(multiscale_aucs):.4f}")
    print(f"{'AVERAGE':<12} {'Baseline':<25} {np.mean(baseline_aucs):.4f}")
    
    improvement = np.mean(multiscale_aucs) - np.mean(baseline_aucs)
    print(f"\n{'ANOMALYVFM v1.3 FINAL SUMMARY':=^80}")
    print(f"🔍 Multi-Scale LoRA Integration Results:")
    print(f"   Average Multi-Scale LoRA AUC: {np.mean(multiscale_aucs):.4f}")
    print(f"   Average Baseline AUC: {np.mean(baseline_aucs):.4f}")
    print(f"   Multi-Scale LoRA Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"\n📏 Scales: 128-256-512 multi-resolution processing")
    print(f"🎯 Scale-specific LoRA adaptation with cross-scale fusion")
    print(f"🏆 AnomalyVFM v1.3 with comprehensive anomaly coverage ready")

if __name__ == "__main__":
    main()