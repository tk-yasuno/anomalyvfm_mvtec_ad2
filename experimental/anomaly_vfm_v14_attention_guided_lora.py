#!/usr/bin/env python3
# AnomalyVFM v1.4 - Attention-guided LoRA based on successful v1.1
# Intelligent LoRA application guided by attention mechanisms
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

class AttentionGuidedLoRALayer(nn.Module):
    """Attention-guided Low-Rank Adaptation Layer"""
    
    def __init__(self, input_dim, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, input_dim))
        
        # Attention guidance network
        self.attention_guide = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU(), 
            nn.Linear(input_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Feature importance analyzer
        self.importance_analyzer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [B, N, D] or [B, D]
        original_x = x
        batch_size = x.shape[0]
        
        # Normalize input
        normalized_x = self.layer_norm(x)
        
        # Compute attention guidance weights
        if len(x.shape) == 3:  # [B, N, D]
            B, N, D = x.shape
            x_flat = normalized_x.reshape(-1, D)  # [B*N, D]
            attention_weights = self.attention_guide(x_flat).reshape(B, N, 1)  # [B, N, 1]
            importance_weights = self.importance_analyzer(x_flat).reshape(B, N, D)  # [B, N, D]
        else:  # [B, D]
            attention_weights = self.attention_guide(normalized_x).unsqueeze(-1)  # [B, 1]
            importance_weights = self.importance_analyzer(normalized_x)  # [B, D]
            x_flat = normalized_x
        
        # Apply attention-guided LoRA: x + attention * importance * scale * x @ A @ B
        lora_out = x_flat @ self.lora_A @ self.lora_B
        
        if len(x.shape) == 3:
            lora_out = lora_out.reshape(B, N, D)
            
        # Apply attention and importance weighting
        guided_lora = attention_weights * importance_weights * self.scale * self.dropout(lora_out)
        adapted_x = original_x + guided_lora
        
        # Return adapted features and attention for analysis
        if len(attention_weights.shape) > 2:
            attention_summary = attention_weights.squeeze(-1).mean(dim=1)  # [B]
        else:
            attention_summary = attention_weights.squeeze(-1)  # [B]
            
        return adapted_x, attention_summary

class AnomalyAttentionAnalyzer(nn.Module):
    """Advanced attention analysis for anomaly detection"""
    
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Multi-head attention for feature correlation analysis
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # Anomaly scoring head with attention weighting
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Attention aggregation for ensemble
        self.attention_aggregator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features_list, attention_weights_list):
        """
        features_list: List of [B, D] or [B, N, D] features from each LoRA layer
        attention_weights_list: List of [B] attention weights
        """
        # Ensure all features are 2D [B, D] before stacking
        normalized_features = []
        for features in features_list:
            if len(features.shape) == 3:  # [B, N, D]
                # Take mean over sequence dimension or use CLS token (first token)
                features_2d = features.mean(dim=1)  # [B, D]
            else:  # Already [B, D]
                features_2d = features
            normalized_features.append(features_2d)
        
        # Stack features for cross-attention
        stacked_features = torch.stack(normalized_features, dim=1)  # [B, num_layers, D]
        
        # Apply cross-attention to capture inter-layer relationships
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )  # [B, num_layers, D]
        
        # Weight by attention importance
        attention_weights = torch.stack(attention_weights_list, dim=1).unsqueeze(-1)  # [B, num_layers, 1]
        weighted_features = attended_features * attention_weights
        
        # Aggregate weighted features
        aggregated_features = weighted_features.mean(dim=1)  # [B, D]
        
        # Compute final attention score for ensemble weighting
        final_attention = self.attention_aggregator(aggregated_features).squeeze(-1)  # [B]
        
        return aggregated_features, final_attention

class EnhancedSyntheticGenerator:
    """Enhanced synthetic anomaly generation with attention guidance"""
    
    def __init__(self, image_size=518):
        self.image_size = image_size
        
    def generate_attention_guided_anomalies(self, normal_images, category, num_anomalies=90):
        """Generate synthetic anomalies with attention-based targeting"""
        print(f"    Generating attention-guided synthetic anomalies for {category}")
        
        # Category-specific attention focus
        if category in ['fruit_jelly', 'walnuts']:
            # High texture: focus on texture and surface anomalies
            texture_focus = 0.5
            structural_focus = 0.3
            contextual_focus = 0.2
        elif category in ['fabric', 'vial']:
            # High performance: conservative, focus on structural
            texture_focus = 0.2
            structural_focus = 0.5
            contextual_focus = 0.3
        else:
            # Balanced approach for medium categories
            texture_focus = 0.4
            structural_focus = 0.3
            contextual_focus = 0.3
        
        anomalies = []
        
        # Generate texture-focused anomalies
        texture_count = int(num_anomalies * texture_focus)
        for i in range(texture_count):
            base_img = normal_images[np.random.randint(len(normal_images))].copy()
            anomaly_img = self.add_attention_guided_texture_anomaly(base_img)
            anomalies.append(anomaly_img)
        
        # Generate structural anomalies
        structural_count = int(num_anomalies * structural_focus)
        for i in range(structural_count):
            base_img = normal_images[np.random.randint(len(normal_images))].copy()
            anomaly_img = self.add_attention_guided_structural_anomaly(base_img)
            anomalies.append(anomaly_img)
        
        # Generate contextual anomalies
        remaining_count = num_anomalies - texture_count - structural_count
        for i in range(remaining_count):
            img1 = normal_images[np.random.randint(len(normal_images))]
            img2 = normal_images[np.random.randint(len(normal_images))]
            anomaly_img = self.add_contextual_anomaly(img1, img2)
            anomalies.append(anomaly_img)
        
        print(f"      T={texture_count}, S={structural_count}, C={remaining_count} samples")
        return anomalies
    
    def add_attention_guided_texture_anomaly(self, img):
        """Add texture anomaly guided by attention mechanisms"""
        h, w = img.shape[:2]
        
        # Multi-scale attention targeting
        num_patches = np.random.randint(2, 5)
        
        for _ in range(num_patches):
            # Vary patch size for multi-scale attention
            patch_size = np.random.randint(30, 100)
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            
            # Attention-guided noise with varying intensities
            noise_type = np.random.choice(['gaussian', 'salt_pepper', 'blur'])
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, np.random.randint(20, 50), (patch_size, patch_size, 3))
                img[y:y+patch_size, x:x+patch_size] = np.clip(
                    img[y:y+patch_size, x:x+patch_size] + noise, 0, 255
                )
            elif noise_type == 'salt_pepper':
                mask = np.random.random((patch_size, patch_size)) < 0.1
                img[y:y+patch_size, x:x+patch_size][mask] = np.random.choice([0, 255])
            else:  # blur
                patch = img[y:y+patch_size, x:x+patch_size]
                blurred = cv2.GaussianBlur(patch, (15, 15), 0)
                img[y:y+patch_size, x:x+patch_size] = blurred
        
        return img.astype(np.uint8)
    
    def add_attention_guided_structural_anomaly(self, img):
        """Add structural anomaly with attention guidance"""
        h, w = img.shape[:2]
        
        # Multiple anomaly types for attention diversity
        anomaly_types = ['circle', 'rectangle', 'line', 'irregular']
        chosen_type = np.random.choice(anomaly_types)
        
        if chosen_type == 'circle':
            center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            radius = np.random.randint(15, 80)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.circle(img, center, radius, color, -1)
            
        elif chosen_type == 'rectangle':
            pt1 = (np.random.randint(0, w//2), np.random.randint(0, h//2))
            size = np.random.randint(40, 120)
            pt2 = (min(pt1[0] + size, w-1), min(pt1[1] + size, h-1))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(img, pt1, pt2, color, -1)
            
        elif chosen_type == 'line':
            for _ in range(np.random.randint(2, 5)):
                pt1 = (np.random.randint(0, w), np.random.randint(0, h))
                pt2 = (np.random.randint(0, w), np.random.randint(0, h))
                color = tuple(np.random.randint(0, 256, 3).tolist())
                thickness = np.random.randint(3, 15)
                cv2.line(img, pt1, pt2, color, thickness)
                
        else:  # irregular
            # Create irregular shape using random contours
            points = []
            center_x, center_y = w//2, h//2
            for i in range(np.random.randint(6, 12)):
                angle = 2 * np.pi * i / 10
                radius = np.random.randint(20, 80)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.fillPoly(img, [points], color)
        
        return img.astype(np.uint8)
    
    def add_contextual_anomaly(self, img1, img2):
        """Standard contextual anomaly generation"""
        img = img1.copy()
        h, w = img.shape[:2]
        
        patch_size = np.random.randint(60, 140)
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

class AnomalyVFMv14:
    """AnomalyVFM v1.4 - Attention-guided LoRA based on v1.1"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.image_size = 518
        
        print("🚀 AnomalyVFM v1.4 - Attention-guided LoRA")
        print("=" * 55)
        print("🎯 Based on successful v1.1 architecture")
        print("👁️ Attention-guided LoRA application")
        print("🧠 Intelligent feature importance analysis")
        print("📊 Multi-head cross-attention correlation")
        print("⚡ DINOv2-Base + Guided LoRA")
        print("🎯 Smart attention-based anomaly targeting")
        print("=" * 55 + "\n")
        
        # Initialize components
        self.synthetic_generator = EnhancedSyntheticGenerator(self.image_size)
        self.load_attention_guided_model()
        
    def load_attention_guided_model(self):
        """Load model with attention-guided LoRA configuration"""
        print("🏗️ Loading attention-guided LoRA model...")
        
        # Base DINOv2 model (same as successful v1.1)
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.to(self.device)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
            dummy_features = self.backbone(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
        
        print(f"   Feature dimension: {self.feature_dim}")
        
        # Attention-guided LoRA adapters (same structure as v1.1 but with attention)
        self.attention_loras = nn.ModuleList([
            AttentionGuidedLoRALayer(self.feature_dim, rank=16, alpha=32),
            AttentionGuidedLoRALayer(self.feature_dim, rank=16, alpha=32),
            AttentionGuidedLoRALayer(self.feature_dim, rank=16, alpha=32)
        ]).to(self.device)
        
        # Advanced attention analyzer
        self.attention_analyzer = AnomalyAttentionAnalyzer(self.feature_dim).to(self.device)
        
        print("   Attention-guided LoRA layers: 3x (Rank=16, Alpha=32)")
        print("   Multi-head cross-attention: 8 heads")
        print("✅ Attention-guided LoRA model loaded")
    
    def extract_attention_guided_features(self, dataloader, use_lora=True):
        """Extract features with attention guidance"""
        features = []
        attention_scores = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Attention-guided extraction", ncols=70):
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 1:
                    images = batch_data[0]
                else:
                    images = batch_data
                
                if not isinstance(images, torch.Tensor):
                    continue
                    
                images = images.to(self.device)
                
                # Extract base features
                base_features = self.backbone(images)
                if len(base_features.shape) == 3:
                    base_features = base_features[:, 0, :]  # CLS token
                
                # Apply attention-guided LoRA if enabled
                if use_lora:
                    layer_features = []
                    layer_attentions = []
                    
                    current_features = base_features
                    for attention_lora in self.attention_loras:
                        current_features, attention_weight = attention_lora(current_features)
                        layer_features.append(current_features)
                        layer_attentions.append(attention_weight)
                    
                    # Use attention analyzer for final feature aggregation
                    final_features, final_attention = self.attention_analyzer(layer_features, layer_attentions)
                    final_features = F.normalize(final_features, dim=1)
                    
                    features.append(final_features.cpu())
                    attention_scores.append(final_attention.cpu())
                else:
                    final_features = F.normalize(base_features, dim=1)
                    features.append(final_features.cpu())
                    attention_scores.append(torch.ones(base_features.shape[0]))
        
        features = torch.cat(features, dim=0).numpy()
        attention_scores = torch.cat(attention_scores, dim=0).numpy()
        
        return features, attention_scores
    
    def train_attention_guided_lora(self, train_dataset, category, epochs=10):
        """Train attention-guided LoRA with enhanced synthetic data"""
        print("🏗️ Generating attention-guided synthetic training data...")
        
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
        
        # Generate attention-guided synthetic anomalies
        synthetic_anomalies = self.synthetic_generator.generate_attention_guided_anomalies(
            normal_images, category, 90
        )
        print(f"✅ Generated {len(synthetic_anomalies)} attention-guided synthetic anomaly samples")
        
        # Set up training (same structure as successful v1.1)
        optimizer = optim.AdamW(
            list(self.attention_loras.parameters()) + list(self.attention_analyzer.parameters()),
            lr=1e-4, 
            weight_decay=1e-5
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"🏋️ Training Attention-guided LoRA for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            self.attention_loras.train()
            self.attention_analyzer.train()
            
            # Training loop with attention-guided synthetic data
            for i in range(0, len(synthetic_anomalies), 8):
                batch_normal = normal_images[i:i+4] if i+4 < len(normal_images) else normal_images[:4]
                batch_anomaly = synthetic_anomalies[i:i+4] if i+4 < len(synthetic_anomalies) else synthetic_anomalies[:4]
                
                batch_images = []
                batch_labels = []
                
                for img in batch_normal:
                    if len(img.shape) == 3:
                        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
                    batch_images.append(img)
                    batch_labels.append(0)
                
                for img in batch_anomaly:
                    if len(img.shape) == 3:
                        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
                    batch_images.append(img)
                    batch_labels.append(1)
                
                if len(batch_images) < 2:
                    continue
                
                # Resize images
                resized_images = []
                for img in batch_images:
                    img_resized = F.interpolate(
                        img.unsqueeze(0), 
                        size=(self.image_size, self.image_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    resized_images.append(img_resized.squeeze(0))
                
                batch_tensor = torch.stack(resized_images).to(self.device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Extract features with attention-guided LoRA
                base_features = self.backbone(batch_tensor)
                if len(base_features.shape) == 3:
                    base_features = base_features[:, 0, :]
                
                # Apply attention-guided LoRA layers
                layer_features = []
                layer_attentions = []
                
                current_features = base_features
                for attention_lora in self.attention_loras:
                    current_features, attention_weight = attention_lora(current_features)
                    layer_features.append(current_features)
                    layer_attentions.append(attention_weight)
                
                # Final attention analysis and prediction
                final_features, final_attention = self.attention_analyzer(layer_features, layer_attentions)
                predictions = self.attention_analyzer.anomaly_scorer(final_features).squeeze()
                
                # Compute loss
                loss = criterion(predictions, labels_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("✅ Attention-guided LoRA training completed")
    
    def evaluate_category_attention_guided(self, category, data_root='./data/MVTec AD2', use_lora=True):
        """Evaluate category with attention-guided LoRA"""
        print(f"\n{category.upper():=^60}")
        
        start_time = time.time()
        
        # Load datasets (same as v1.1)
        train_dataset = AD2TrainDataset(data_root, category, image_size=self.image_size)
        test_dataset = AD2TestDataset(data_root, category, image_size=self.image_size)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Count samples
        normal_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 0)
        anomaly_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 1)
        
        print(f"Train: {len(train_dataset)}, Test: {normal_count} normal + {anomaly_count} anomaly")
        print("Attention-guided processing enabled" if use_lora else "Baseline processing")
        
        # Train attention-guided LoRA
        if use_lora:
            self.train_attention_guided_lora(train_dataset, category, epochs=10)
        
        # Extract features with attention guidance
        print("Extracting attention-guided features...")
        train_features, train_attentions = self.extract_attention_guided_features(train_loader, use_lora)
        test_features, test_attentions = self.extract_attention_guided_features(test_loader, use_lora)
        
        # Anomaly detection with attention weighting
        cov_estimator = EmpiricalCovariance().fit(train_features)
        train_distances = cov_estimator.mahalanobis(train_features)
        test_distances = cov_estimator.mahalanobis(test_features)
        
        # Normalize scores
        train_mean = np.mean(train_distances)
        train_std = np.std(train_distances)
        anomaly_scores = (test_distances - train_mean) / (train_std + 1e-8)
        
        # Weight scores by attention (if using LoRA)
        if use_lora:
            # Normalize attention weights
            attention_weights = (test_attentions - np.min(test_attentions)) / (np.max(test_attentions) - np.min(test_attentions) + 1e-8)
            attention_weights = attention_weights * 0.3 + 0.7  # Scale to [0.7, 1.0] to avoid suppressing too much
            final_scores = anomaly_scores * attention_weights
            avg_attention = np.mean(test_attentions)
            print(f"Average attention score: {avg_attention:.4f}")
        else:
            final_scores = anomaly_scores
            avg_attention = 1.0
        
        # Get labels
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        test_labels = np.array(test_labels)
        
        # Compute AUC
        image_auc = roc_auc_score(test_labels, final_scores)
        
        processing_time = time.time() - start_time
        
        method_status = "Attention-guided LoRA" if use_lora else "Baseline"
        print(f"Image-level AUC ({method_status}): {image_auc:.4f}")
        if use_lora:
            print(f"Attention enhancement factor: {avg_attention:.4f}")
        print(f"Processing time: {processing_time:.1f}s")
        
        return {
            'category': category,
            'image_auc': image_auc,
            'use_lora': use_lora,
            'attention_score': avg_attention,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'processing_time': processing_time
        }

def main():
    """AnomalyVFM v1.4 with Attention-guided LoRA evaluation"""
    model = AnomalyVFMv14()
    
    # All 7 categories for attention-guided evaluation
    categories = ['fruit_jelly', 'fabric', 'can', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    results = []
    
    print(f"AnomalyVFM v1.4 Attention-guided LoRA evaluation for all {len(categories)} categories...\n")
    
    for i, category in enumerate(categories, 1):
        print(f"[{i}/{len(categories)}] Processing {category}")
        
        # Evaluate with attention-guided LoRA
        result_lora = model.evaluate_category_attention_guided(category, use_lora=True)
        results.append(result_lora)
        
        # Evaluate baseline for comparison
        result_baseline = model.evaluate_category_attention_guided(category, use_lora=False)
        results.append(result_baseline)
    
    # Summary
    print(f"\n{'ANOMALYVFM v1.4 ATTENTION-GUIDED RESULTS':=^80}")
    print(f"{'Category':<12} {'Method':<25} {'AUC':<8} {'Attention':<10} {'Time(s)':<8}")
    print("-" * 80)
    
    attention_aucs = []
    baseline_aucs = []
    
    for result in results:
        if result['use_lora']:
            method = "Attention-guided LoRA"
            attention = f"{result['attention_score']:.3f}"
            attention_aucs.append(result['image_auc'])
        else:
            method = "Baseline"
            attention = "N/A"
            baseline_aucs.append(result['image_auc'])
            
        print(f"{result['category']:<12} {method:<25} {result['image_auc']:.4f}   "
              f"{attention:<10} {result['processing_time']:.1f}")
    
    print("-" * 80)
    print(f"{'AVERAGE':<12} {'Attention-guided LoRA':<25} {np.mean(attention_aucs):.4f}")
    print(f"{'AVERAGE':<12} {'Baseline':<25} {np.mean(baseline_aucs):.4f}")
    
    improvement = np.mean(attention_aucs) - np.mean(baseline_aucs)
    print(f"\n{'ANOMALYVFM v1.4 FINAL SUMMARY':=^80}")
    print(f"👁️ Attention-guided LoRA Integration Results:")
    print(f"   Average Attention-guided AUC: {np.mean(attention_aucs):.4f}")
    print(f"   Average Baseline AUC: {np.mean(baseline_aucs):.4f}")
    print(f"   Attention-guided Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"\n🎯 Based on successful v1.1 architecture")
    print(f"👁️ Intelligent attention-guided LoRA application")
    print(f"🏆 AnomalyVFM v1.4 with smart anomaly targeting ready")

if __name__ == "__main__":
    main()