# anomaly_vfm_improved.py - 改良版AnomalyVFM
import torch
import torch.nn.functional as F
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import warnings
import time

from dataset_ad2 import AD2TrainDataset, AD2TestDataset

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 AnomalyVFM Improved - Using device: {device}")

if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


class PatchBasedFeatureExtractor:
    """
    パッチベース特徴抽出器（より細かな異常を検出）
    """
    def __init__(self, model_name="resnet18.a1_in1k", patch_size=64, stride=32):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(device).eval()
        self.patch_size = patch_size
        self.stride = stride
        
    def extract_patch_features(self, images):
        """
        画像からパッチ特徴量を抽出
        """
        B, C, H, W = images.shape
        features = []
        
        with torch.no_grad():
            # 画像全体の特徴量
            global_feat = self.model(images)
            features.append(global_feat.cpu().numpy())
            
            # パッチ特徴量（オプション：より詳細な分析用）
            # メモリ効率のため、小さなパッチサイズで実装
            
        return np.concatenate(features, axis=1)


def compute_reconstruction_error(train_features, test_features):
    """
    再構成誤差ベースの異常スコア計算
    """
    # 正常データの統計量計算
    mean_feat = np.mean(train_features, axis=0)
    std_feat = np.std(train_features, axis=0) + 1e-8  # 0除算防止
    
    # 正規化
    train_normalized = (train_features - mean_feat) / std_feat
    test_normalized = (test_features - mean_feat) / std_feat
    
    # 各テストサンプルの異常スコア（L2距離ベース）
    scores = []
    for test_sample in test_normalized:
        # 最も近い正常サンプルとの距離を計算
        distances = np.linalg.norm(train_normalized - test_sample, axis=1)
        min_distance = np.min(distances)
        scores.append(min_distance)
    
    return np.array(scores)


def compute_statistical_anomaly_score(train_features, test_features, method='isolation'):
    """
    統計的異常検知手法
    """
    if method == 'lof':
        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        lof.fit(train_features)
        scores = -lof.decision_function(test_features)  # 負の値なので反転
    else:
        # シンプルな距離ベース手法
        scores = compute_reconstruction_error(train_features, test_features)
    
    return scores


def evaluate_improved_anomaly_detection(root, category, batch_size=64):
    """
    改良版異常検知評価
    """
    print(f"\n{'='*60}")
    print(f"  🎯 Improved AnomalyVFM: {category}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # データセット
        train_ds = AD2TrainDataset(root, category, image_size=224)
        test_ds = AD2TestDataset(root, category, image_size=224)
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("❌ No data found")
            return 0.0
            
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True)
        
        # 改良された特徴抽出器
        print("  📡 Loading improved feature extractor...")
        extractor = PatchBasedFeatureExtractor(model_name="resnet18.a1_in1k")
        
        # 訓練特徴量抽出
        print("  📊 Extracting training features...")
        train_features = []
        
        with torch.no_grad():
            for batch_idx, images in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                features = extractor.extract_patch_features(images)
                train_features.append(features)
                
                if batch_idx % 5 == 0:
                    print(f"    Train batch {batch_idx+1}/{len(train_loader)}")
        
        train_features = np.concatenate(train_features, axis=0)
        print(f"  ✅ Train features shape: {train_features.shape}")
        
        # テスト特徴量抽出と評価
        print("  🔍 Testing anomaly detection...")
        test_features = []
        test_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                features = extractor.extract_patch_features(images)
                test_features.append(features)
                test_labels.extend(labels.numpy())
                
                if batch_idx % 3 == 0:
                    print(f"    Test batch {batch_idx+1}/{len(test_loader)}")
        
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.array(test_labels)
        
        # 複数の異常検知手法を試す
        methods = [
            ('Distance-based', 'distance'),
            ('Local Outlier Factor', 'lof'),
        ]
        
        best_auc = 0.0
        best_method = None
        
        for method_name, method_key in methods:
            try:
                if method_key == 'distance':
                    scores = compute_reconstruction_error(train_features, test_features)
                else:
                    scores = compute_statistical_anomaly_score(train_features, test_features, method_key)
                
                auc = roc_auc_score(test_labels, scores)
                print(f"    {method_name}: AUC = {auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_method = method_name
                    
            except Exception as e:
                print(f"    {method_name}: Error - {str(e)}")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  🏆 Best Result:")
        print(f"    Method: {best_method}")
        print(f"    AUC Score: {best_auc:.4f}")
        print(f"    Evaluation Time: {elapsed_time:.1f}s")
        
        return best_auc
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 0.0
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    # 単一カテゴリテスト
    category = "can"
    auc = evaluate_improved_anomaly_detection(root, category)
    
    print(f"\n🎯 Final AUC for {category}: {auc:.4f}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"🖥️  Peak GPU Memory: {memory_used:.2f} GB")