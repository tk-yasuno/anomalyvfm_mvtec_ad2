# full_7category_anomalyvfm_demo.py - 全7カテゴリ対応版
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
print(f"🚀 AnomalyVFM Full Demo - Using device: {device}")

if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True


def extract_features_efficient(model, loader, desc=""):
    """
    効率的な特徴量抽出
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if isinstance(data, tuple):
                images = data[0]
            else:
                images = data
                
            images = images.to(device, non_blocking=True)
            
            # 特徴量抽出
            feats = model(images)
            features.append(feats.cpu())
            
            if batch_idx % 5 == 0 and desc:
                print(f"    {desc} batch {batch_idx+1}/{len(loader)}")
    
    return torch.cat(features, dim=0).numpy()


def compute_anomaly_scores_robust(train_features, test_features):
    """
    ロバストな異常スコア計算
    """
    # 標準化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # マハラノビス距離ベースのスコア
    try:
        cov = EmpiricalCovariance().fit(train_scaled)
        mean = cov.location_
        precision = cov.precision_
        
        # テストデータのスコア計算
        scores = []
        for sample in test_scaled:
            diff = sample - mean
            score = float(diff @ precision @ diff.T)
            scores.append(score)
        
        return np.array(scores)
        
    except Exception as e:
        print(f"    Warning: Mahalanobis failed ({str(e)}), using L2 distance")
        
        # フォールバック：シンプルなL2距離
        mean_train = np.mean(train_scaled, axis=0)
        scores = np.linalg.norm(test_scaled - mean_train, axis=1)
        return scores


def evaluate_category_final(root, category, batch_size=32):
    """
    最終版カテゴリ評価
    """
    print(f"\n{'='*60}")
    print(f"  🎯 AnomalyVFM Demo: {category.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # データセット作成 (DINOv2用: 518x518)
        train_ds = AD2TrainDataset(root, category, image_size=518)
        test_ds = AD2TestDataset(root, category, image_size=518)
        
        print(f"  📊 Data: {len(train_ds)} train, {len(test_ds)} test")
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  ❌ No data available")
            return 0.0
        
        # DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, pin_memory=True if device=='cuda' else False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True if device=='cuda' else False)
        
        # モデル: DINOv2-ViT-Base（最新の自己教師学習モデル）
        print("  🤖 Loading DINOv2-ViT-Base...")
        model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
        model = model.to(device).eval()
        
        # 訓練特徴量
        print("  📈 Extracting training features...")
        train_features = extract_features_efficient(model, train_loader, "Train")
        print(f"    Shape: {train_features.shape}")
        
        # テスト特徴量とラベル
        print("  🔍 Extracting test features...")
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
        
        # 異常スコア計算
        print("  ⚡ Computing anomaly scores...")
        scores = compute_anomaly_scores_robust(train_features, test_features)
        
        # AUC計算
        auc = roc_auc_score(test_labels, scores)
        
        # 統計情報
        normal_count = np.sum(test_labels == 0)
        anomaly_count = np.sum(test_labels == 1)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  📊 Results:")
        print(f"    AUC Score: {auc:.4f}")
        print(f"    Normal samples: {normal_count}")
        print(f"    Anomaly samples: {anomaly_count}")
        print(f"    Processing time: {elapsed_time:.1f}s")
        
        # パフォーマンス評価
        if auc >= 0.90:
            grade = "🏆 EXCELLENT"
        elif auc >= 0.80:
            grade = "🥇 VERY GOOD"
        elif auc >= 0.70:
            grade = "🥈 GOOD"
        elif auc >= 0.60:
            grade = "🥉 FAIR"
        else:
            grade = "📈 NEEDS IMPROVEMENT"
            
        print(f"    Performance: {grade}")
        
        return auc
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0.0
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_full_7_category_demo():
    """
    全7カテゴリデモ実行
    """
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    # 全7カテゴリ
    categories = [
        "can", "fabric", "fruit_jelly",          # 既存の3カテゴリ
        "sheet_metal", "vial", "wallplugs", "walnuts"  # 新規4カテゴリ
    ]
    
    print("🔥" * 70)
    print("   🚀 AnomalyVFM MVP - FULL 7-CATEGORY COMPREHENSIVE DEMO 🚀")
    print("🔥" * 70)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Categories: {categories}")
    print(f"🎯 Total categories: {len(categories)}")
    print(f"🖥️  Running on: {device.upper()}")
    
    results = []
    total_start = time.time()
    
    # 各カテゴリの処理
    for i, category in enumerate(categories, 1):
        print(f"\n🎯 [{i}/{len(categories)}] Processing: {category.upper()}")
        auc = evaluate_category_final(root, category)
        results.append((category, auc))
        
        # 中間報告
        if auc >= 0.90:
            print(f"    ✨ Outstanding result! AUC = {auc:.4f}")
        elif auc >= 0.70:
            print(f"    👍 Good result! AUC = {auc:.4f}")
        else:
            print(f"    📊 AUC = {auc:.4f} (needs optimization)")
    
    # 最終結果サマリー
    total_time = time.time() - total_start
    
    print("\n" + "🏁" * 70)
    print("   📊 FINAL COMPREHENSIVE RESULTS - ALL 7 CATEGORIES")
    print("🏁" * 70)
    
    print(f"{'Category':<15} {'AUC':<10} {'Grade':<20} {'Status'}")
    print("-" * 65)
    
    total_auc = 0
    valid_count = 0
    excellent_count = 0
    good_count = 0
    fair_count = 0
    poor_count = 0
    
    for category, auc in results:
        if auc >= 0.90:
            grade = "🏆 EXCELLENT"
            status = "✨ OUTSTANDING"
            excellent_count += 1
        elif auc >= 0.80:
            grade = "🥇 VERY GOOD"
            status = "🔥 GREAT"
            good_count += 1
        elif auc >= 0.70:
            grade = "🥈 GOOD"
            status = "👍 SOLID"
            good_count += 1
        elif auc >= 0.60:
            grade = "🥉 FAIR"
            status = "⚡ OK"
            fair_count += 1
        elif auc > 0.0:
            grade = "📈 POOR"
            status = "🔧 NEEDS WORK"
            poor_count += 1
        else:
            grade = "❌ FAILED"
            status = "🚫 ERROR"
            continue
            
        print(f"{category:<15} {auc:.4f}     {grade:<20} {status}")
        
        if auc > 0.0:
            total_auc += auc
            valid_count += 1
    
    # 統計サマリー
    if valid_count > 0:
        avg_auc = total_auc / valid_count
        print("-" * 65)
        print(f"{'AVERAGE':<15} {avg_auc:.4f}     {'🌟 OVERALL':<20} {'📊 SUMMARY'}")
        
        # 詳細統計
        print(f"\n📈 PERFORMANCE BREAKDOWN:")
        print(f"  🏆 Excellent (≥0.90):    {excellent_count:2d}/{len(categories)} ({excellent_count/len(categories)*100:.1f}%)")
        print(f"  🥇 Very Good+ (≥0.70):   {excellent_count + good_count:2d}/{len(categories)} ({(excellent_count + good_count)/len(categories)*100:.1f}%)")
        print(f"  🥉 Fair+ (≥0.60):        {excellent_count + good_count + fair_count:2d}/{len(categories)} ({(excellent_count + good_count + fair_count)/len(categories)*100:.1f}%)")
        print(f"  ✅ Valid results:        {valid_count:2d}/{len(categories)} ({valid_count/len(categories)*100:.1f}%)")
        
        # 総合評価
        if avg_auc >= 0.85:
            overall_grade = "🌟 OUTSTANDING SYSTEM"
        elif avg_auc >= 0.75:
            overall_grade = "🔥 EXCELLENT SYSTEM"
        elif avg_auc >= 0.65:
            overall_grade = "👍 GOOD SYSTEM"
        elif avg_auc >= 0.55:
            overall_grade = "⚡ DECENT SYSTEM"
        else:
            overall_grade = "🔧 NEEDS IMPROVEMENT"
        
        print(f"\n🎯 OVERALL SYSTEM GRADE: {overall_grade}")
        print(f"   Average AUC: {avg_auc:.4f}")
    
    # パフォーマンス統計
    print(f"\n⏱️  PERFORMANCE METRICS:")
    print(f"  📊 Total evaluation time: {total_time:.1f} seconds")
    print(f"  ⚡ Average time per category: {total_time/len(categories):.1f} seconds")
    print(f"  🏁 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        efficiency = "🔋 EFFICIENT" if memory_used < 4.0 else "🔥 INTENSIVE" if memory_used < 8.0 else "🚨 HEAVY"
        print(f"  🖥️  Peak GPU Memory: {memory_used:.2f} GB ({efficiency})")
    
    print("🔥" * 70)
    
    # 結果保存（詳細版）
    save_detailed_results(results, total_time, avg_auc if valid_count > 0 else 0)
    

def save_detailed_results(results, total_time, avg_auc):
    """
    詳細結果の保存
    """
    filename = f"full_7category_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("AnomalyVFM MVP - Full 7-Category Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device.upper()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for category, auc in results:
            status = "EXCELLENT" if auc >= 0.90 else "GOOD" if auc >= 0.70 else "FAIR" if auc >= 0.60 else "POOR" if auc > 0 else "FAILED"
            f.write(f"{category:<15}: {auc:.4f} ({status})\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"Average AUC: {avg_auc:.4f}\n")
        f.write(f"Total Time: {total_time:.1f} seconds\n")
        f.write(f"Categories: {len(results)}\n")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            f.write(f"Peak GPU Memory: {memory_used:.2f} GB\n")
        
        f.write(f"\nGenerated by AnomalyVFM MVP Full Demo\n")
    
    print(f"📄 Detailed results saved to: {filename}")


if __name__ == "__main__":
    print("🚀 Starting AnomalyVFM Full 7-Category Comprehensive Demo...")
    run_full_7_category_demo()
    print("\n✅ Full 7-Category Demo Complete! 🎉")