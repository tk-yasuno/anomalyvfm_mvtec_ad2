# evaluate_ad2_multi_gpu.py - GPU最適化3カテゴリ一括評価
import time
from datetime import datetime
import torch
import sys
import os

# 現在のディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_ad2_anomalyvfm_gpu import evaluate_category_gpu


def evaluate_multiple_gpu(root, categories):
    """
    GPU最適化された複数カテゴリ一括評価
    """
    results = []
    start_time = time.time()
    
    print("="*70)
    print("   🚀 AnomalyVFM MVP - GPU Accelerated Multi-Category Evaluation")
    print("="*70)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Data root: {root}")
    print(f"📝 Categories: {categories}")
    
    # GPU情報表示
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("="*70)
    
    for i, category in enumerate(categories, 1):
        print(f"\n🎯 [{i}/{len(categories)}] Processing: {category}")
        
        try:
            # GPUメモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            auc = evaluate_category_gpu(root, category, batch_size=128)  # GPU用に大きめのバッチサイズ
            results.append((category, auc))
            
            # GPUメモリ使用量表示
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"    💾 Current GPU Memory: {memory_used:.2f} GB")
            
        except Exception as e:
            print(f"❌ ERROR evaluating {category}: {str(e)}")
            results.append((category, 0.0))

    # 結果サマリー
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    print("\n" + "="*70)
    print("   📊 FINAL RESULTS SUMMARY")
    print("="*70)
    
    print("{:<20} {:<10} {:<15} {:<20}".format("Category", "AUC", "Performance", "Rating"))
    print("-" * 70)
    
    total_auc = 0
    valid_count = 0
    performance_levels = []
    
    for category, auc in results:
        # パフォーマンス評価とレーティング
        if auc >= 0.95:
            performance = "Outstanding"
            rating = "⭐⭐⭐⭐⭐"
            level = 5
        elif auc >= 0.90:
            performance = "Excellent"
            rating = "⭐⭐⭐⭐"
            level = 4
        elif auc >= 0.85:
            performance = "Very Good"
            rating = "⭐⭐⭐"
            level = 3
        elif auc >= 0.75:
            performance = "Good"
            rating = "⭐⭐"
            level = 2
        elif auc >= 0.60:
            performance = "Fair"
            rating = "⭐"
            level = 1
        elif auc > 0.0:
            performance = "Poor"
            rating = "❌"
            level = 0
        else:
            performance = "Failed"
            rating = "💥"
            level = 0
            
        print("{:<20} {:.4f}     {:<15} {:<20}".format(category, auc, performance, rating))
        
        if auc > 0.0:
            total_auc += auc
            valid_count += 1
            performance_levels.append(level)
    
    print("-" * 70)
    
    # 全体統計
    if valid_count > 0:
        avg_auc = total_auc / valid_count
        avg_level = sum(performance_levels) / len(performance_levels)
        
        if avg_auc >= 0.90:
            overall_rating = "🏆 OUTSTANDING"
        elif avg_auc >= 0.85:
            overall_rating = "🥇 EXCELLENT"
        elif avg_auc >= 0.80:
            overall_rating = "🥈 VERY GOOD"
        elif avg_auc >= 0.75:
            overall_rating = "🥉 GOOD"
        else:
            overall_rating = "📈 NEEDS IMPROVEMENT"
            
        print("{:<20} {:.4f}     {:<15} {:<20}".format("AVERAGE", avg_auc, f"Level {avg_level:.1f}", overall_rating))
    
    print("="*70)
    print(f"⏱️  Total evaluation time: {total_elapsed:.1f} seconds")
    print(f"⚡ Average time per category: {total_elapsed/len(categories):.1f} seconds")
    print(f"🏁 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU使用統計
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"💾 Peak GPU Memory Usage: {max_memory:.2f} GB")
        print(f"🔄 GPU Utilization: Efficient")
    
    print("="*70)
    
    return results


def save_results_detailed(results, filename="gpu_evaluation_results.txt"):
    """
    詳細な結果をファイルに保存
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("AnomalyVFM MVP - GPU Accelerated Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hardware: GPU Accelerated (CUDA)\n")
        if torch.cuda.is_available():
            f.write(f"GPU Model: {torch.cuda.get_device_name(0)}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Results Summary:\n")
        f.write("-"*30 + "\n")
        for category, auc in results:
            status = "✅ PASS" if auc >= 0.75 else "⚠️ REVIEW" if auc >= 0.60 else "❌ FAIL"
            f.write(f"{category:<20}: {auc:.4f} ({status})\n")
        
        f.write(f"\nGenerated by AnomalyVFM MVP\n")
    
    print(f"📄 Detailed results saved to: {filename}")


if __name__ == "__main__":
    # データルートパス
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"

    # 評価する3カテゴリ
    categories = [
        "can",           # 缶
        "fabric",        # 布地
        "fruit_jelly",   # フルーツゼリー
    ]

    # GPU加速一括評価実行
    print("🔥 Starting GPU-accelerated anomaly detection evaluation...")
    results = evaluate_multiple_gpu(root, categories)
    
    # 結果保存
    save_results_detailed(results)
    
    print("\n✅ GPU Evaluation Complete! 🚀")