# visualized_anomalyvfm.py - 可視化機能付きAnomalyVFM
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import time
import os
from datetime import datetime

from dataset_ad2 import AD2TrainDataset, AD2TestDataset

warnings.filterwarnings('ignore')

# スタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 AnomalyVFM Visualized - Using device: {device}")

if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")


class AnomalyVFMVisualizer:
    """
    AnomalyVFMの可視化クラス
    """
    
    def __init__(self, save_dir="visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_roc_curve(self, y_true, y_scores, category, save_name=None):
        """
        ROC曲線とAUCを可視化
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(10, 8))
        
        # ROC曲線
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        # 最適閾値の点をプロット
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve - {category.upper()}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # AUCスコアをテキストで追加
        plt.text(0.6, 0.2, f'AUC = {auc:.4f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                fontsize=14, fontweight='bold')
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_roc.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return auc, optimal_threshold
    
    def plot_anomaly_scores_distribution(self, scores, labels, category, save_name=None):
        """
        異常スコアの分布をヒストグラムで可視化
        """
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        plt.figure(figsize=(12, 6))
        
        # ヒストグラム
        plt.hist(normal_scores, bins=50, alpha=0.7, color='blue', 
                label=f'Normal (n={len(normal_scores)})', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, color='red', 
                label=f'Anomaly (n={len(anomaly_scores)})', density=True)
        
        # 統計情報を追加
        normal_mean, normal_std = np.mean(normal_scores), np.std(normal_scores)
        anomaly_mean, anomaly_std = np.mean(anomaly_scores), np.std(anomaly_scores)
        
        plt.axvline(normal_mean, color='blue', linestyle='--', alpha=0.8,
                   label=f'Normal mean = {normal_mean:.3f}')
        plt.axvline(anomaly_mean, color='red', linestyle='--', alpha=0.8,
                   label=f'Anomaly mean = {anomaly_mean:.3f}')
        
        plt.xlabel('Anomaly Score', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Anomaly Score Distribution - {category.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_scores.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'normal_mean': normal_mean, 'normal_std': normal_std,
            'anomaly_mean': anomaly_mean, 'anomaly_std': anomaly_std,
            'separation_ratio': (anomaly_mean - normal_mean) / (normal_std + anomaly_std)
        }
    
    def plot_features_heatmap(self, features, labels, category, max_features=50, save_name=None):
        """
        特徴量のヒートマップ可視化
        """
        # 特徴量数を制限（計算効率のため）
        n_features = min(max_features, features.shape[1])
        selected_features = features[:, :n_features]
        
        # 正常・異常別の平均特徴量
        normal_features = selected_features[labels == 0]
        anomaly_features = selected_features[labels == 1]
        
        normal_mean = np.mean(normal_features, axis=0)
        anomaly_mean = np.mean(anomaly_features, axis=0)
        
        # データフレーム作成
        heatmap_data = pd.DataFrame({
            'Normal': normal_mean,
            'Anomaly': anomaly_mean,
            'Difference': anomaly_mean - normal_mean
        }, index=[f'Feature_{i:03d}' for i in range(n_features)])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Feature Value'})
        plt.title(f'Feature Heatmap - {category.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Sample Type', fontsize=14)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_heatmap.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_tsne_features(self, features, labels, category, save_name=None):
        """
        t-SNEを使用した特徴量の2D可視化
        """
        print(f"  🔄 Computing t-SNE for {category}...")
        
        # サンプル数を制限（t-SNEは計算が重いため）
        max_samples = min(1000, len(features))
        indices = np.random.choice(len(features), max_samples, replace=False)
        
        features_subset = features[indices]
        labels_subset = labels[indices]
        
        # t-SNE実行
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_subset)
        
        plt.figure(figsize=(10, 8))
        
        # 正常・異常を色分けしてプロット
        normal_mask = labels_subset == 0
        anomaly_mask = labels_subset == 1
        
        plt.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, s=50, label=f'Normal (n={np.sum(normal_mask)})')
        plt.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                   c='red', alpha=0.6, s=50, label=f'Anomaly (n={np.sum(anomaly_mask)})')
        
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.title(f't-SNE Feature Visualization - {category.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}_tsne.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, results_dict, save_name="summary_report"):
        """
        結果サマリーレポートを作成
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        categories = list(results_dict.keys())
        aucs = [results_dict[cat]['auc'] for cat in categories]
        
        # 1. AUCバーチャート
        bars = ax1.bar(categories, aucs, color=['red' if auc < 0.6 else 'orange' if auc < 0.8 else 'green' for auc in aucs])
        ax1.set_ylabel('AUC Score', fontsize=12)
        ax1.set_title('AUC Scores by Category', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # バーに値を表示
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 処理時間
        times = [results_dict[cat]['time'] for cat in categories]
        ax2.bar(categories, times, color='skyblue')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Processing Time by Category', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 異常分離度（Normal-Anomaly分離の良さ）
        separations = [results_dict[cat]['separation_ratio'] for cat in categories]
        ax3.bar(categories, separations, color='lightcoral')
        ax3.set_ylabel('Separation Ratio', fontsize=12)
        ax3.set_title('Normal-Anomaly Separation', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. パフォーマンス分類
        performance_counts = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        for auc in aucs:
            if auc >= 0.9: performance_counts['Excellent'] += 1
            elif auc >= 0.7: performance_counts['Good'] += 1
            elif auc >= 0.6: performance_counts['Fair'] += 1
            else: performance_counts['Poor'] += 1
        
        ax4.pie(performance_counts.values(), labels=performance_counts.keys(), 
               autopct='%1.0f%%', colors=['green', 'blue', 'orange', 'red'])
        ax4.set_title('Performance Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_category_with_visualization(root, category, visualizer, batch_size=32):
    """
    可視化付きカテゴリ評価
    """
    print(f"\n{'='*60}")
    print(f"  🎯 AnomalyVFM Visualization: {category.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # データセット (DINOv2用: 518x518)
        train_ds = AD2TrainDataset(root, category, image_size=518)
        test_ds = AD2TestDataset(root, category, image_size=518)
        
        print(f"  📊 Data: {len(train_ds)} train, {len(test_ds)} test")
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  ❌ No data available")
            return None
        
        # DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, pin_memory=True if device=='cuda' else False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True if device=='cuda' else False)
        
        # モデル
        print("  🤖 Loading DINOv2-ViT-Base...")
        model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
        model = model.to(device).eval()
        
        # 特徴量抽出（訓練）
        print("  📈 Extracting training features...")
        train_features = []
        with torch.no_grad():
            for batch_idx, images in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                feats = model(images)
                train_features.append(feats.cpu())
        
        train_features = torch.cat(train_features, dim=0).numpy()
        
        # 特徴量抽出（テスト）
        print("  🔍 Extracting test features...")
        test_features = []
        test_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                feats = model(images)
                test_features.append(feats.cpu())
                test_labels.extend(labels.numpy())
        
        test_features = torch.cat(test_features, dim=0).numpy()
        test_labels = np.array(test_labels)
        
        # 異常スコア計算
        print("  ⚡ Computing anomaly scores...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        try:
            cov = EmpiricalCovariance().fit(train_scaled)
            mean = cov.location_
            precision = cov.precision_
            
            scores = []
            for sample in test_scaled:
                diff = sample - mean
                score = float(diff @ precision @ diff.T)
                scores.append(score)
            scores = np.array(scores)
            
        except:
            # フォールバック
            mean_train = np.mean(train_scaled, axis=0)
            scores = np.linalg.norm(test_scaled - mean_train, axis=1)
        
        # 可視化実行
        print("  🎨 Creating visualizations...")
        save_name = f"{category}_{datetime.now().strftime('%m%d_%H%M')}"
        
        # 1. ROC曲線
        auc, optimal_threshold = visualizer.plot_roc_curve(
            test_labels, scores, category, save_name)
        
        # 2. スコア分布
        score_stats = visualizer.plot_anomaly_scores_distribution(
            scores, test_labels, category, save_name)
        
        # 3. 特徴量ヒートマップ
        visualizer.plot_features_heatmap(
            test_features, test_labels, category, save_name=save_name)
        
        # 4. t-SNE可視化
        visualizer.plot_tsne_features(
            test_features, test_labels, category, save_name)
        
        elapsed_time = time.time() - start_time
        
        # 結果
        print(f"\n  📊 Results:")
        print(f"    AUC Score: {auc:.4f}")
        print(f"    Optimal Threshold: {optimal_threshold:.4f}")
        print(f"    Separation Ratio: {score_stats['separation_ratio']:.3f}")
        print(f"    Processing Time: {elapsed_time:.1f}s")
        
        return {
            'category': category,
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            'separation_ratio': score_stats['separation_ratio'],
            'time': elapsed_time,
            'normal_mean': score_stats['normal_mean'],
            'anomaly_mean': score_stats['anomaly_mean']
        }
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_visualization_demo():
    """
    可視化デモ実行
    """
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    # デモ用に3カテゴリを選択（時間節約）
    demo_categories = ["fruit_jelly", "fabric", "can"]  # 高性能カテゴリを先に
    
    print("🎨" * 60)
    print("   AnomalyVFM - Advanced Visualization Demo")
    print("🎨" * 60)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Categories: {demo_categories}")
    
    # 可視化ツール初期化
    visualizer = AnomalyVFMVisualizer(save_dir="anomaly_visualizations")
    
    results = {}
    total_start = time.time()
    
    # 各カテゴリの処理
    for i, category in enumerate(demo_categories, 1):
        print(f"\n🎯 [{i}/{len(demo_categories)}] Processing: {category}")
        result = evaluate_category_with_visualization(root, category, visualizer)
        
        if result:
            results[category] = result
            
            # 中間結果表示
            grade = "🏆 EXCELLENT" if result['auc'] >= 0.9 else "🥇 VERY GOOD" if result['auc'] >= 0.8 else "🥈 GOOD" if result['auc'] >= 0.7 else "🥉 FAIR"
            print(f"    ✅ {grade} (AUC: {result['auc']:.4f})")
    
    # 総合サマリー作成
    if results:
        print(f"\n📊 Creating comprehensive summary...")
        visualizer.create_summary_report(results, "comprehensive_summary")
    
    total_time = time.time() - total_start
    
    # 最終サマリー
    print(f"\n{'🎨' * 60}")
    print("   VISUALIZATION DEMO COMPLETE")
    print("🎨" * 60)
    
    if results:
        avg_auc = np.mean([r['auc'] for r in results.values()])
        print(f"📊 Average AUC: {avg_auc:.4f}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📁 Visualizations saved in: anomaly_visualizations/")
        
        # 最良結果のハイライト
        best_category = max(results.keys(), key=lambda k: results[k]['auc'])
        best_auc = results[best_category]['auc']
        print(f"🏆 Best performance: {best_category} (AUC: {best_auc:.4f})")
    
    print("🎨" * 60)


if __name__ == "__main__":
    print("🎨 Starting AnomalyVFM Advanced Visualization Demo...")
    run_visualization_demo()
    print("\n✅ Visualization Demo Complete! 🎉")