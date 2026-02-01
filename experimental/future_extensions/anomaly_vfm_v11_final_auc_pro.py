#!/usr/bin/env python3
# AnomalyVFM v1.1 Final - with AUC-PRO Support
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, auc
from scipy import ndimage
import os
import time
from tqdm import tqdm
from dataset_ad2 import AD2TrainDataset, AD2TestDataset
import timm
from torch.utils.data import DataLoader

# AUC-PRO calculation functions
def compute_per_region_overlap(anomaly_map, gt_mask, threshold=0.5):
    """Compute Per-Region Overlap (PRO) score"""
    binary_anomaly = (anomaly_map > threshold).astype(np.uint8)
    gt_labeled, num_gt_regions = ndimage.label(gt_mask > 0.5)
    
    if num_gt_regions == 0:
        return 0.0
    
    overlap_scores = []
    for region_id in range(1, num_gt_regions + 1):
        gt_region = (gt_labeled == region_id)
        intersection = np.logical_and(gt_region, binary_anomaly)
        
        if np.sum(gt_region) == 0:
            overlap = 0.0
        else:
            overlap = np.sum(intersection) / np.sum(gt_region)
        
        overlap_scores.append(overlap)
    
    return np.mean(overlap_scores)

def calculate_auc_pro(anomaly_maps, gt_masks, num_thresholds=50):
    """Calculate AUC-PRO (Area Under Curve - Per-Region Overlap)"""
    if len(anomaly_maps) == 0 or len(gt_masks) == 0:
        return 0.0, [], []
    
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []
    fprs = []
    
    for threshold in thresholds:
        pro_scores_thresh = []
        fpr_scores_thresh = []
        
        for anomaly_map, gt_mask in zip(anomaly_maps, gt_masks):
            # Calculate PRO score
            pro_score = compute_per_region_overlap(anomaly_map, gt_mask, threshold)
            pro_scores_thresh.append(pro_score)
            
            # Calculate FPR
            binary_pred = (anomaly_map > threshold).astype(np.uint8)
            normal_pixels = (gt_mask == 0)
            
            if np.sum(normal_pixels) > 0:
                false_positives = np.sum(np.logical_and(binary_pred == 1, normal_pixels))
                fpr = false_positives / np.sum(normal_pixels)
            else:
                fpr = 0.0
            
            fpr_scores_thresh.append(fpr)
        
        pro_scores.append(np.mean(pro_scores_thresh))
        fprs.append(np.mean(fpr_scores_thresh))
    
    # Sort by FPR for AUC calculation
    sorted_indices = np.argsort(fprs)
    fprs_sorted = np.array(fprs)[sorted_indices]
    pro_scores_sorted = np.array(pro_scores)[sorted_indices]
    
    # Calculate AUC
    auc_pro = auc(fprs_sorted, pro_scores_sorted)
    return auc_pro, fprs_sorted, pro_scores_sorted

def load_gt_mask_for_image_path(image_path, target_size=(518, 518)):
    """Load GT mask based on image path"""
    try:
        image_path_abs = os.path.abspath(image_path)
        filename = os.path.basename(image_path_abs)
        filename_no_ext = os.path.splitext(filename)[0]
        mask_filename = filename_no_ext + '_mask.png'
        
        dir_path = os.path.dirname(image_path_abs)
        
        if 'bad' in dir_path:
            if '\\\\' in dir_path:
                gt_dir = dir_path.replace('bad', 'ground_truth\\\\bad')
            else:
                gt_dir = dir_path.replace('bad', 'ground_truth/bad')
            
            mask_path = os.path.join(gt_dir, mask_filename)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, target_size)
                    mask = mask.astype(np.float32) / 255.0
                    return mask
        
        return None
        
    except Exception as e:
        return None

class AnomalyVFMv11Final:
    """AnomalyVFM v1.1 Final with AUC-PRO support"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 518
        self.model = None
        self.normal_cov = None
        
    def load_model(self):
        """Load DINOv2 model"""
        print("🤖 Loading DINOv2-Base model...")
        self.model = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                                       pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")
        
    def extract_features_batch(self, dataloader):
        """Extract features from dataloader"""
        features = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Extracting features"):
                if isinstance(batch_data, (tuple, list)):
                    images = batch_data[0]
                else:
                    images = batch_data
                
                images = images.to(self.device)
                feats = self.model(images)
                features.append(F.normalize(feats, dim=1).cpu().numpy())
        
        return np.vstack(features)
    
    def extract_features_single(self, image):
        """Extract features from single image"""
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
            features = F.normalize(features, dim=1)
        
        return features.cpu().numpy()
    
    def fit_normal_distribution(self, train_features):
        """Fit normal distribution"""
        self.normal_cov = EmpiricalCovariance().fit(train_features)
        
    def generate_pixel_anomaly_map(self, image, patch_size=32):
        """Generate pixel-level anomaly map using patches"""
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        h_patches = max(1, H // patch_size)
        w_patches = max(1, W // patch_size)
        
        patch_scores = []
        
        with torch.no_grad():
            for i in range(h_patches):
                for j in range(w_patches):
                    # Extract patch coordinates
                    start_h = i * patch_size
                    end_h = min((i + 1) * patch_size, H)
                    start_w = j * patch_size
                    end_w = min((j + 1) * patch_size, W)
                    
                    # Extract and process patch
                    patch = image[:, :, start_h:end_h, start_w:end_w]
                    patch_resized = F.interpolate(patch, size=(H, W), mode='bilinear')
                    
                    # Get features and compute anomaly score
                    patch_features = self.extract_features_single(patch_resized)
                    
                    if self.normal_cov is not None:
                        score = self.normal_cov.mahalanobis(patch_features)[0]
                    else:
                        score = np.linalg.norm(patch_features)
                    
                    patch_scores.append(score)
        
        # Reshape and resize to original dimensions
        scores_array = np.array(patch_scores).reshape(h_patches, w_patches)
        anomaly_map = cv2.resize(scores_array, (W, H))
        
        # Normalize to 0-1 range
        if np.max(anomaly_map) > np.min(anomaly_map):
            anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        
        return anomaly_map
    
    def evaluate_category_with_auc_pro(self, category, max_anomaly_samples=10):
        """Evaluate category with both Image-AUC and AUC-PRO"""
        print(f"\\n🎯 Evaluating {category}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load datasets  
        train_dataset = AD2TrainDataset('./data/MVTec AD2', category, image_size=self.image_size)
        test_dataset = AD2TestDataset('./data/MVTec AD2', category, image_size=self.image_size)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Load model and extract features
        self.load_model()
        
        print("📊 Extracting training features...")
        train_features = self.extract_features_batch(train_loader)
        
        print("📊 Extracting test features...")
        test_features = self.extract_features_batch(test_loader)
        
        # Fit normal distribution
        self.fit_normal_distribution(train_features)
        
        # Calculate image-level anomaly scores
        train_distances = self.normal_cov.mahalanobis(train_features)
        test_distances = self.normal_cov.mahalanobis(test_features)
        
        # Normalize scores
        train_mean = np.mean(train_distances)
        train_std = np.std(train_distances)
        anomaly_scores = (test_distances - train_mean) / (train_std + 1e-8)
        
        # Get test labels
        test_labels = []
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            test_labels.append(label)
        test_labels = np.array(test_labels)
        
        # Calculate Image-AUC
        if len(np.unique(test_labels)) > 1:
            image_auc = roc_auc_score(test_labels, anomaly_scores)
        else:
            image_auc = 0.0
        
        # Calculate AUC-PRO for anomaly samples
        print("🎯 Calculating AUC-PRO...")
        
        anomaly_indices = np.where(test_labels == 1)[0]
        if len(anomaly_indices) == 0:
            auc_pro_score = 0.0
            valid_samples = 0
        else:
            # Limit samples for efficiency
            selected_indices = anomaly_indices[:max_anomaly_samples]
            
            anomaly_maps = []
            gt_masks = []
            
            for idx in tqdm(selected_indices, desc="Processing anomalies"):
                # Get test sample
                image, _ = test_dataset[idx]
                
                # Generate anomaly map
                anomaly_map = self.generate_pixel_anomaly_map(image)
                
                # Load GT mask
                image_path = test_dataset.get_image_path(idx)
                if image_path:
                    gt_mask = load_gt_mask_for_image_path(image_path, (self.image_size, self.image_size))
                    
                    if gt_mask is not None and np.sum(gt_mask > 0.5) > 0:
                        anomaly_maps.append(anomaly_map)
                        gt_masks.append(gt_mask)
            
            # Calculate AUC-PRO
            if len(anomaly_maps) > 0:
                auc_pro_score, _, _ = calculate_auc_pro(anomaly_maps, gt_masks)
                valid_samples = len(anomaly_maps)
            else:
                auc_pro_score = 0.0
                valid_samples = 0
        
        processing_time = time.time() - start_time
        
        print(f"\\n✅ Results for {category}:")
        print(f"   Image-level AUC: {image_auc:.4f}")
        print(f"   AUC-PRO (Per-Region Overlap): {auc_pro_score:.4f}")
        print(f"   Valid anomaly samples for AUC-PRO: {valid_samples}/{len(anomaly_indices)}")
        print(f"   Processing time: {processing_time:.1f}s")
        
        return {
            'category': category,
            'image_auc': image_auc,
            'auc_pro': auc_pro_score,
            'valid_samples': valid_samples,
            'total_anomalies': len(anomaly_indices),
            'processing_time': processing_time
        }

def main():
    """Test AUC-PRO with AnomalyVFM v1.1 Final"""
    print("🚀 AnomalyVFM v1.1 Final - AUC-PRO Evaluation")
    print("=" * 70)
    
    model = AnomalyVFMv11Final()
    
    # Test with fruit_jelly (known to work)
    categories = ['fruit_jelly']
    results = []
    
    for category in categories:
        result = model.evaluate_category_with_auc_pro(category, max_anomaly_samples=5)
        results.append(result)
    
    # Summary
    print(f"\\n{'ANOMALYVFM v1.1 FINAL - AUC-PRO RESULTS':=^70}")
    print(f"{'Category':<15} {'Image-AUC':<10} {'AUC-PRO':<8} {'Valid/Total':<12} {'Time(s)':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['category']:<15} {result['image_auc']:.4f}     "
              f"{result['auc_pro']:.4f}   {result['valid_samples']}/{result['total_anomalies']:<8} "
              f"{result['processing_time']:.1f}")
    
    print(f"\\n🎯 AnomalyVFM v1.1 with AUC-PRO evaluation completed!")
    print(f"✅ Per-Region Overlap measurement successfully integrated!")

if __name__ == "__main__":
    main()