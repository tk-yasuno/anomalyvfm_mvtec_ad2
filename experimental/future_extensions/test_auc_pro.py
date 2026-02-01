#!/usr/bin/env python3
# AUC-PRO Test Script
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
from dataset_ad2 import AD2TestDataset

def compute_per_region_overlap(anomaly_map, gt_mask, threshold=0.5):
    """Compute Per-Region Overlap (PRO) score"""
    # Binarize anomaly map
    binary_anomaly = (anomaly_map > threshold).astype(np.uint8)
    
    # Label connected components in ground truth
    gt_labeled, num_gt_regions = ndimage.label(gt_mask > 0.5)
    
    if num_gt_regions == 0:
        return 0.0
    
    overlap_scores = []
    
    for region_id in range(1, num_gt_regions + 1):
        # Get current GT region
        gt_region = (gt_labeled == region_id)
        
        # Calculate overlap with predicted anomaly map
        intersection = np.logical_and(gt_region, binary_anomaly)
        
        if np.sum(gt_region) == 0:
            overlap = 0.0
        else:
            overlap = np.sum(intersection) / np.sum(gt_region)
        
        overlap_scores.append(overlap)
    
    return np.mean(overlap_scores)

def calculate_auc_pro(anomaly_maps, gt_masks, num_thresholds=100):
    """Calculate AUC-PRO (Area Under Curve - Per-Region Overlap)"""
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []
    fprs = []
    
    for threshold in thresholds:
        pro_scores_thresh = []
        fpr_scores_thresh = []
        
        for anomaly_map, gt_mask in zip(anomaly_maps, gt_masks):
            # Calculate PRO score for this threshold
            pro_score = compute_per_region_overlap(anomaly_map, gt_mask, threshold)
            pro_scores_thresh.append(pro_score)
            
            # Calculate FPR
            binary_pred = (anomaly_map > threshold).astype(np.uint8)
            
            if np.sum(gt_mask == 0) > 0:
                false_positives = np.sum(np.logical_and(binary_pred == 1, gt_mask == 0))
                total_normals = np.sum(gt_mask == 0)
                fpr = false_positives / total_normals
            else:
                fpr = 0.0
            
            fpr_scores_thresh.append(fpr)
        
        pro_scores.append(np.mean(pro_scores_thresh))
        fprs.append(np.mean(fpr_scores_thresh))
    
    # Sort by FPR for proper AUC calculation
    sorted_indices = np.argsort(fprs)
    fprs_sorted = np.array(fprs)[sorted_indices]
    pro_scores_sorted = np.array(pro_scores)[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc_pro = auc(fprs_sorted, pro_scores_sorted)
    
    return auc_pro, fprs_sorted, pro_scores_sorted

def load_gt_mask_for_image_path(image_path, target_size=(224, 224)):
    """Load GT mask based on image path"""
    try:
        # Convert to absolute path and normalize separators
        image_path = os.path.abspath(image_path).replace('\\', '/')
        
        # Extract filename from image path
        filename = os.path.basename(image_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Construct mask filename
        mask_filename = filename_no_ext + '_mask.png'
        
        # Get directory path and replace bad with ground_truth/bad
        dir_path = os.path.dirname(image_path)
        if '/bad' in dir_path:
            gt_dir = dir_path.replace('/bad', '/ground_truth/bad')
            mask_path = os.path.join(gt_dir, mask_filename).replace('\\', '/')
        else:
            print(f"  Warning: Could not find /bad in path: {dir_path}")
            return None
        
        # Convert back to Windows path format
        mask_path_win = mask_path.replace('/', '\\')
        
        print(f"  Trying mask path: {mask_path_win}")
        
        if os.path.exists(mask_path_win):
            mask = cv2.imread(mask_path_win, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize to target size
                mask = cv2.resize(mask, target_size)
                # Normalize to 0-1
                mask = mask.astype(np.float32) / 255.0
                return mask
        
        return None
        
    except Exception as e:
        print(f"  Error loading GT mask: {e}")
        return None

def test_auc_pro_calculation():
    """Test AUC-PRO calculation with a single category"""
    print("🧪 Testing AUC-PRO Calculation")
    print("=" * 50)
    
    # Test with fruit_jelly category (has known good results)
    category = 'fruit_jelly'
    
    # Load test dataset
    test_dataset = AD2TestDataset('./data/MVTec AD2', category, image_size=224)
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Get anomaly samples only
    anomaly_indices = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label == 1:  # Anomaly
            anomaly_indices.append(i)
    
    print(f"Found {len(anomaly_indices)} anomaly samples")
    
    if len(anomaly_indices) == 0:
        print("❌ No anomaly samples found")
        return
    
    # Generate dummy anomaly maps and load GT masks
    anomaly_maps = []
    gt_masks = []
    
    for idx in anomaly_indices[:5]:  # Test with first 5 anomaly samples
        print(f"Processing sample {idx}...")
        
        # Generate dummy anomaly map (random scores)
        np.random.seed(idx)
        anomaly_map = np.random.rand(224, 224)
        
        # Load corresponding GT mask
        image_path = test_dataset.get_image_path(idx)
        if image_path:
            print(f"  Image path: {os.path.basename(image_path)}")
            
            gt_mask = load_gt_mask_for_image_path(image_path)
            
            if gt_mask is not None and np.sum(gt_mask) > 0:
                anomaly_maps.append(anomaly_map)
                gt_masks.append(gt_mask)
                print(f"  ✅ GT mask loaded, non-zero pixels: {np.sum(gt_mask > 0)}")
            else:
                print(f"  ❌ GT mask not found or empty")
        else:
            print(f"  ❌ Could not get image path for sample {idx}")
    
    print(f"\nValid samples for AUC-PRO: {len(anomaly_maps)}")
    
    if len(anomaly_maps) == 0:
        print("❌ No valid GT masks found for AUC-PRO calculation")
        return
    
    # Calculate AUC-PRO
    print("🎯 Calculating AUC-PRO...")
    start_time = time.time()
    
    auc_pro_score, fprs, pro_scores = calculate_auc_pro(anomaly_maps, gt_masks)
    
    calculation_time = time.time() - start_time
    
    print(f"\n✅ AUC-PRO Calculation Results:")
    print(f"   Category: {category}")
    print(f"   AUC-PRO Score: {auc_pro_score:.4f}")
    print(f"   Valid samples: {len(anomaly_maps)}")
    print(f"   Calculation time: {calculation_time:.2f}s")
    print(f"   FPR range: [{np.min(fprs):.3f}, {np.max(fprs):.3f}]")
    print(f"   PRO range: [{np.min(pro_scores):.3f}, {np.max(pro_scores):.3f}]")
    
    # Test individual PRO calculation
    print(f"\n🔍 Testing individual PRO scores:")
    for i, (anomaly_map, gt_mask) in enumerate(zip(anomaly_maps[:3], gt_masks[:3])):
        pro_05 = compute_per_region_overlap(anomaly_map, gt_mask, threshold=0.5)
        print(f"   Sample {i}: PRO@0.5 = {pro_05:.4f}")

if __name__ == "__main__":
    test_auc_pro_calculation()