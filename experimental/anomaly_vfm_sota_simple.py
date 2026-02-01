#!/usr/bin/env python3
# Simple SOTA Comparison for AnomalyVFM v0.3
import torch
import torch.nn.functional as F
import timm
import numpy as np
import time
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset_ad2 import AD2TrainDataset, AD2TestDataset
import warnings
warnings.filterwarnings('ignore')

class AnomalyVFMSOTASimple:
    """Simple SOTA Comparison based on v0.3 (Best performer)"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.image_size = 518
        print("🚀 AnomalyVFM v0.3 - SOTA Benchmark Comparison")
        print("=" * 50)
        print("📊 MVTec AD2 Paper comparison ready")
        print("🎯 Image-level AUC evaluation")
        print("⚡ DINOv2-Base + Mahalanobis (Best config)")
        print("=" * 50 + "\n")
        
    def load_model(self):
        """Load DINOv2 Base model"""
        self.dino_base = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
        self.dino_base = torch.nn.Sequential(*list(self.dino_base.children())[:-1])
        self.dino_base.to(self.device)
        self.dino_base.eval()
        
    def extract_features(self, dataloader, desc=""):
        """Extract features using v0.3 method"""
        features = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 1:
                    images = batch_data[0]
                else:
                    images = batch_data
                
                if not isinstance(images, torch.Tensor):
                    continue
                    
                images = images.to(self.device)
                
                # DINOv2 features
                feat = self.dino_base(images)
                if len(feat.shape) == 3:
                    feat = feat[:, 0, :]  # CLS token
                feat = F.normalize(feat, dim=1)
                features.append(feat.cpu())
        
        features = torch.cat(features, dim=0).numpy()
        return features
    
    def evaluate_category(self, category, data_root='./data/MVTec AD2'):
        """Evaluate category with v0.3 method"""
        print(f"\n{category.upper():=^50}")
        
        start_time = time.time()
        
        # Load datasets
        train_dataset = AD2TrainDataset(data_root, category, image_size=self.image_size)
        test_dataset = AD2TestDataset(data_root, category, image_size=self.image_size)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Count samples
        normal_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 0)
        anomaly_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i][1] == 1)
        
        print(f"Train: {len(train_dataset)}, Test: {normal_count} normal + {anomaly_count} anomaly")
        
        # Load model and extract features
        self.load_model()
        print("Extracting features...")
        train_features = self.extract_features(train_loader)
        test_features = self.extract_features(test_loader)
        
        # Compute anomaly scores (v0.3 method)
        cov_estimator = EmpiricalCovariance().fit(train_features)
        train_mahal_dist = cov_estimator.mahalanobis(train_features)
        test_mahal_dist = cov_estimator.mahalanobis(test_features)
        
        # Normalize scores
        train_mean = np.mean(train_mahal_dist)
        train_std = np.std(train_mahal_dist)
        anomaly_scores = (test_mahal_dist - train_mean) / (train_std + 1e-8)
        
        # Get labels and compute AUC
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        test_labels = np.array(test_labels)
        
        image_auc = roc_auc_score(test_labels, anomaly_scores)
        processing_time = time.time() - start_time
        
        print(f"Image-level AUC: {image_auc:.4f}")
        print(f"Processing time: {processing_time:.1f}s")
        
        return {
            'category': category,
            'image_auc': image_auc,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'processing_time': processing_time
        }

def main():
    """SOTA comparison evaluation"""
    model = AnomalyVFMSOTASimple()
    
    # All 7 categories for complete SOTA comparison
    categories = ['fruit_jelly', 'fabric', 'can', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    results = []
    
    print(f"Evaluating {len(categories)} categories for SOTA comparison...\n")
    
    for i, category in enumerate(categories, 1):
        print(f"[{i}/{len(categories)}] Processing {category}")
        result = model.evaluate_category(category)
        results.append(result)
    
    # Summary
    print(f"\n{'SOTA COMPARISON RESULTS':=^60}")
    print(f"{'Category':<12} {'AUC':<8} {'Normal':<8} {'Anomaly':<8} {'Time(s)':<8}")
    print("-" * 60)
    
    total_auc = 0
    for result in results:
        total_auc += result['image_auc']
        print(f"{result['category']:<12} {result['image_auc']:.4f}   "
              f"{result['normal_count']:<8} {result['anomaly_count']:<8} {result['processing_time']:.1f}")
    
    avg_auc = total_auc / len(results)
    print("-" * 60)
    print(f"{'AVERAGE':<12} {avg_auc:.4f}")
    
    print(f"\n{'MVTec AD2 PAPER COMPARISON':=^60}")
    print(f"🎯 AnomalyVFM v0.3 Results:")
    print(f"   Method: DINOv2-Base + Mahalanobis Distance")
    print(f"   Image-level AUC: {avg_auc:.4f}")
    print(f"   Categories: {len(categories)} (complete dataset)")
    print(f"   Architecture: Self-supervised Vision Transformer")
    print(f"\n📊 Ready for direct comparison with:")
    print(f"   - MVTec AD2 paper Table results")
    print(f"   - Other SOTA anomaly detection methods")
    print(f"   - Foundation model approaches")
    
    # Per-category analysis
    print(f"\n{'CATEGORY PERFORMANCE ANALYSIS':=^60}")
    high_perf = [r for r in results if r['image_auc'] >= 0.7]
    medium_perf = [r for r in results if 0.5 <= r['image_auc'] < 0.7] 
    low_perf = [r for r in results if r['image_auc'] < 0.5]
    
    print(f"🏆 High Performance (AUC ≥ 0.7): {len(high_perf)} categories")
    for r in high_perf:
        print(f"   {r['category']}: {r['image_auc']:.4f}")
        
    print(f"🥈 Medium Performance (0.5 ≤ AUC < 0.7): {len(medium_perf)} categories")
    for r in medium_perf:
        print(f"   {r['category']}: {r['image_auc']:.4f}")
        
    print(f"📈 Needs Improvement (AUC < 0.5): {len(low_perf)} categories")
    for r in low_perf:
        print(f"   {r['category']}: {r['image_auc']:.4f}")

if __name__ == "__main__":
    main()