# test_new_categories.py - 新規4カテゴリのテスト
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import time
from datetime import datetime

from dataset_ad2 import AD2TrainDataset, AD2TestDataset

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Testing New Categories - Using device: {device}")

if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")


def quick_evaluate_category(root, category, batch_size=64):
    """
    新カテゴリの簡易評価
    """
    print(f"\n{'='*50}")
    print(f"  🎯 Testing: {category.upper()}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # データセット確認
        train_ds = AD2TrainDataset(root, category, image_size=224)
        test_ds = AD2TestDataset(root, category, image_size=224)
        
        print(f"  📊 Found: {len(train_ds)} train, {len(test_ds)} test images")
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  ❌ No data available")
            return 0.0
        
        # DataLoader（高速化のため大きめバッチサイズ）
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, pin_memory=True if device=='cuda' else False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True if device=='cuda' else False)
        
        # 軽量モデルで高速テスト
        print("  🤖 Loading EfficientNet-B0...")
        model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=0)
        model = model.to(device).eval()
        
        # 訓練特徴量抽出
        train_features = []
        with torch.no_grad():
            for batch_idx, images in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                feats = model(images)
                train_features.append(feats.cpu())
                
                if batch_idx % 5 == 0:
                    print(f"    Train batch {batch_idx+1}/{len(train_loader)}")
        
        train_features = torch.cat(train_features, dim=0).numpy()
        
        # テスト特徴量抽出
        test_features = []
        test_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                feats = model(images)
                test_features.append(feats.cpu())
                test_labels.extend(labels.numpy())
                
                if batch_idx % 3 == 0:
                    print(f"    Test batch {batch_idx+1}/{len(test_loader)}")
        
        test_features = torch.cat(test_features, dim=0).numpy()
        test_labels = np.array(test_labels)
        
        # 簡易異常スコア計算
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        # L2距離ベース
        mean_train = np.mean(train_scaled, axis=0)
        scores = np.linalg.norm(test_scaled - mean_train, axis=1)
        
        # AUC計算
        auc = roc_auc_score(test_labels, scores)
        
        elapsed_time = time.time() - start_time
        
        # 結果表示
        normal_count = np.sum(test_labels == 0)
        anomaly_count = np.sum(test_labels == 1)
        
        grade = "🏆 EXCELLENT" if auc >= 0.90 else "🥇 VERY GOOD" if auc >= 0.80 else "🥈 GOOD" if auc >= 0.70 else "🥉 FAIR" if auc >= 0.60 else "📈 POOR"
        
        print(f"\n  📊 Quick Results:")
        print(f"    AUC Score: {auc:.4f}")
        print(f"    Grade: {grade}")
        print(f"    Samples: {normal_count} normal, {anomaly_count} anomaly")
        print(f"    Time: {elapsed_time:.1f}s")
        
        return auc
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0.0
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_new_4_categories():
    """
    新規4カテゴリの簡易テスト
    """
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    new_categories = ["sheet_metal", "vial", "wallplugs", "walnuts"]
    
    print("🔥" * 50)
    print("   Testing New 4 Categories")
    print("🔥" * 50)
    print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
    
    results = []
    total_start = time.time()
    
    for i, category in enumerate(new_categories, 1):
        print(f"\n🎯 [{i}/4] Testing: {category}")
        auc = quick_evaluate_category(root, category)
        results.append((category, auc))
    
    # サマリー
    total_time = time.time() - total_start
    
    print(f"\n{'='*50}")
    print("   NEW CATEGORIES SUMMARY")
    print("="*50)
    
    total_auc = 0
    valid_count = 0
    
    for category, auc in results:
        grade = "🏆" if auc >= 0.90 else "🥇" if auc >= 0.80 else "🥈" if auc >= 0.70 else "🥉" if auc >= 0.60 else "📈"
        print(f"{category:<12}: {auc:.4f} {grade}")
        
        if auc > 0.0:
            total_auc += auc
            valid_count += 1
    
    if valid_count > 0:
        avg_auc = total_auc / valid_count
        print(f"{'AVERAGE':<12}: {avg_auc:.4f} 🌟")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Average: {total_time/len(new_categories):.1f}s per category")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"🖥️  GPU Memory: {memory_used:.2f} GB")
    
    return results


if __name__ == "__main__":
    print("🚀 Testing new 4 categories...")
    results = test_new_4_categories()
    print("\n✅ New category testing complete! 🎉")