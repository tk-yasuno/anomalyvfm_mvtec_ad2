# demo_ad2_anomalyvfm_gpu.py - GPU最適化版
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import time

from dataset_ad2 import AD2TrainDataset, AD2TestDataset

# 警告を抑制
warnings.filterwarnings('ignore')

# GPU最適化設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # GPU設定最適化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def extract_embeddings(model, loader, desc=""):
    """
    GPU最適化された特徴量抽出
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(device, non_blocking=True)
            
            # 特徴量抽出
            features = model(x)
            embeddings.append(features.cpu())
            
            if batch_idx % 10 == 0 and desc:
                print(f"  {desc} batch {batch_idx}/{len(loader)}")
                
    # メモリ効率的に結合
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def compute_gaussian_params(embeddings):
    """
    正規化されたガウスパラメータを計算
    """
    print(f"  Computing Gaussian parameters for {embeddings.shape[0]} samples...")
    
    # 特徴量正規化
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # ガウス分布フィット
    cov_estimator = EmpiricalCovariance(assume_centered=False)
    cov_estimator.fit(embeddings_scaled)
    
    return cov_estimator.location_, cov_estimator.precision_, scaler


def mahalanobis_distance_batch(features, mean, precision):
    """
    バッチ処理でマハラノビス距離を計算
    """
    diff = features - mean
    distances = np.sum((diff @ precision) * diff, axis=1)
    return distances


def evaluate_category_gpu(root, category, batch_size=64):
    """
    GPU最適化された異常検知評価
    """
    print(f"\n{'='*60}")
    print(f"  GPU Anomaly Detection: {category}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # データセット作成
        train_ds = AD2TrainDataset(root, category, image_size=224)
        test_ds = AD2TestDataset(root, category, image_size=224)

        if len(train_ds) == 0:
            print(f"  ❌ ERROR: No training images for {category}")
            return 0.0
        
        if len(test_ds) == 0:
            print(f"  ❌ ERROR: No test images for {category}")
            return 0.0

        # DataLoader（GPU最適化）
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,  # Windowsでは2-4が推奨
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        # 軽量で高性能なEfficientNet-B0モデル
        print("  🚀 Loading EfficientNet-B0 (GPU optimized)...")
        model = timm.create_model(
            "efficientnet_b0.ra_in1k", 
            pretrained=True, 
            num_classes=0,  # 特徴量抽出専用
        )
        model = model.to(device)
        model.eval()

        # 訓練データから特徴量抽出
        print("  📊 Extracting training features...")
        train_features = extract_embeddings(model, train_loader, "Train")
        
        # ガウス分布パラメータ計算
        mean, precision, scaler = compute_gaussian_params(train_features)

        # テストデータで評価
        print("  🔍 Evaluating anomaly detection...")
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x = x.to(device, non_blocking=True)
                
                # 特徴量抽出
                features = model(x)
                features_np = features.cpu().numpy()
                
                # 正規化
                features_scaled = scaler.transform(features_np)
                
                # マハラノビス距離計算（バッチ処理）
                scores = mahalanobis_distance_batch(features_scaled, mean, precision)
                
                all_scores.extend(scores)
                all_labels.extend(y.numpy())
                
                if batch_idx % 5 == 0:
                    print(f"    Processing batch {batch_idx+1}/{len(test_loader)}")

        # AUC計算
        auc_score = roc_auc_score(all_labels, all_scores)
        
        # 統計情報
        normal_scores = [all_scores[i] for i in range(len(all_scores)) if all_labels[i] == 0]
        anomaly_scores = [all_scores[i] for i in range(len(all_scores)) if all_labels[i] == 1]
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  📈 Results for {category}:")
        print(f"    AUC Score: {auc_score:.4f}")
        print(f"    Normal samples: {len(normal_scores)} (scores: {min(normal_scores):.2f} - {max(normal_scores):.2f})")
        print(f"    Anomaly samples: {len(anomaly_scores)} (scores: {min(anomaly_scores):.2f} - {max(anomaly_scores):.2f})")
        print(f"    Evaluation time: {elapsed_time:.1f} seconds")
        
        return auc_score
        
    except Exception as e:
        print(f"  ❌ ERROR in {category}: {str(e)}")
        return 0.0
    
    finally:
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # データルートパス
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    # 単一カテゴリテスト
    category = "can"
    auc = evaluate_category_gpu(root, category)
    
    print(f"\n🎯 Final Result: {category} AUC = {auc:.4f}")
    
    # GPUメモリ使用量表示
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"🖥️  Max GPU Memory Used: {memory_used:.2f} GB")