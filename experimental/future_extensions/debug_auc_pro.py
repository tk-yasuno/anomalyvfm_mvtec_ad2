#!/usr/bin/env python3
# AUC-PRO Debug Script - Detailed Analysis
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from dataset_ad2 import AD2TestDataset

def load_gt_mask_for_image_path(image_path, target_size=(518, 518)):
    """Load GT mask based on image path with detailed logging"""
    try:
        print(f"  Input image path: {image_path}")
        
        # Convert to absolute path
        image_path_abs = os.path.abspath(image_path)
        print(f"  Absolute path: {image_path_abs}")
        
        filename = os.path.basename(image_path_abs)
        filename_no_ext = os.path.splitext(filename)[0]
        mask_filename = filename_no_ext + '_mask.png'
        
        print(f"  Looking for mask: {mask_filename}")
        
        dir_path = os.path.dirname(image_path_abs)
        print(f"  Directory: {dir_path}")
        
        if '\\\\bad' in dir_path:
            gt_dir = dir_path.replace('\\\\bad', '\\\\ground_truth\\\\bad')
            print(f"  GT directory (Windows): {gt_dir}")
        elif 'bad' in dir_path:
            # Handle both forward and backward slashes
            if '\\\\' in dir_path:
                gt_dir = dir_path.replace('bad', 'ground_truth\\\\bad')
            else:
                gt_dir = dir_path.replace('bad', 'ground_truth/bad')
            print(f"  GT directory (general): {gt_dir}")
        else:
            print(f"  ❌ No 'bad' found in directory path")
            return None
        
        mask_path = os.path.join(gt_dir, mask_filename)
        print(f"  Final mask path: {mask_path}")
        
        if os.path.exists(mask_path):
            print(f"  ✅ Mask file exists")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"  ✅ Mask loaded successfully: {mask.shape}")
                mask = cv2.resize(mask, target_size)
                mask = mask.astype(np.float32) / 255.0
                return mask
            else:
                print(f"  ❌ cv2.imread failed")
        else:
            print(f"  ❌ Mask file does not exist")
        
        return None
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None

def compute_per_region_overlap_debug(anomaly_map, gt_mask, threshold=0.5):
    """Compute Per-Region Overlap with debug info"""
    print(f"    Anomaly map stats: min={np.min(anomaly_map):.4f}, max={np.max(anomaly_map):.4f}, mean={np.mean(anomaly_map):.4f}")
    print(f"    GT mask stats: min={np.min(gt_mask):.4f}, max={np.max(gt_mask):.4f}, mean={np.mean(gt_mask):.4f}")
    print(f"    Non-zero GT pixels: {np.sum(gt_mask > 0.5)}")
    
    # Binarize anomaly map
    binary_anomaly = (anomaly_map > threshold).astype(np.uint8)
    print(f"    Binary anomaly (threshold={threshold}): {np.sum(binary_anomaly)} pixels")
    
    # Label connected components in ground truth
    gt_labeled, num_gt_regions = ndimage.label(gt_mask > 0.5)
    print(f"    GT regions found: {num_gt_regions}")
    
    if num_gt_regions == 0:
        return 0.0
    
    overlap_scores = []
    
    for region_id in range(1, num_gt_regions + 1):
        gt_region = (gt_labeled == region_id)
        gt_pixels = np.sum(gt_region)
        
        intersection = np.logical_and(gt_region, binary_anomaly)
        intersection_pixels = np.sum(intersection)
        
        if gt_pixels == 0:
            overlap = 0.0
        else:
            overlap = intersection_pixels / gt_pixels
        
        print(f"    Region {region_id}: {gt_pixels} pixels, overlap: {intersection_pixels}/{gt_pixels} = {overlap:.4f}")
        overlap_scores.append(overlap)
    
    mean_overlap = np.mean(overlap_scores)
    print(f"    Mean PRO: {mean_overlap:.4f}")
    return mean_overlap

def debug_auc_pro_calculation():
    """Debug AUC-PRO calculation step by step"""
    print("🔍 AUC-PRO Calculation Debug")
    print("=" * 50)
    
    category = 'fruit_jelly'
    test_dataset = AD2TestDataset('./data/MVTec AD2', category, image_size=518)
    
    # Get first anomaly sample
    anomaly_idx = None
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label == 1:
            anomaly_idx = i
            break
    
    if anomaly_idx is None:
        print("❌ No anomaly samples found")
        return
    
    print(f"📊 Analyzing sample {anomaly_idx}")
    
    # Load GT mask
    image_path = test_dataset.get_image_path(anomaly_idx)
    print(f"Image path: {image_path}")
    
    gt_mask = load_gt_mask_for_image_path(image_path)
    if gt_mask is None:
        print("❌ Could not load GT mask")
        return
    
    print(f"✅ GT mask loaded: shape={gt_mask.shape}")
    
    # Create dummy anomaly maps with different characteristics
    print(f"\\n🎯 Testing different anomaly maps:")
    
    # Test 1: Random anomaly map
    print(f"\\n--- Test 1: Random Anomaly Map ---")
    np.random.seed(42)
    random_anomaly_map = np.random.rand(518, 518)
    pro_random = compute_per_region_overlap_debug(random_anomaly_map, gt_mask, threshold=0.5)
    
    # Test 2: High values where GT mask exists
    print(f"\\n--- Test 2: GT-guided Anomaly Map ---")
    guided_anomaly_map = np.random.rand(518, 518) * 0.3  # Base noise
    guided_anomaly_map[gt_mask > 0.5] = 0.8  # High values where anomalies exist
    pro_guided = compute_per_region_overlap_debug(guided_anomaly_map, gt_mask, threshold=0.5)
    
    # Test 3: Inverse (should have low PRO)
    print(f"\\n--- Test 3: Inverse Anomaly Map ---")
    inverse_anomaly_map = np.ones((518, 518)) * 0.8
    inverse_anomaly_map[gt_mask > 0.5] = 0.2  # Low values where anomalies exist
    pro_inverse = compute_per_region_overlap_debug(inverse_anomaly_map, gt_mask, threshold=0.5)
    
    print(f"\\n📊 PRO Score Comparison:")
    print(f"   Random anomaly map: {pro_random:.4f}")
    print(f"   GT-guided anomaly map: {pro_guided:.4f}")
    print(f"   Inverse anomaly map: {pro_inverse:.4f}")
    
    # Analyze GT mask regions
    print(f"\\n🔍 GT Mask Analysis:")
    gt_labeled, num_regions = ndimage.label(gt_mask > 0.5)
    print(f"   Total GT regions: {num_regions}")
    
    for region_id in range(1, min(num_regions + 1, 6)):  # Show first 5 regions
        region_mask = (gt_labeled == region_id)
        region_size = np.sum(region_mask)
        y_coords, x_coords = np.where(region_mask)
        
        if len(y_coords) > 0:
            center_y, center_x = np.mean(y_coords), np.mean(x_coords)
            print(f"   Region {region_id}: {region_size} pixels, center: ({center_y:.1f}, {center_x:.1f})")
    
    print(f"\\n✅ Debug analysis completed!")

if __name__ == "__main__":
    debug_auc_pro_calculation()