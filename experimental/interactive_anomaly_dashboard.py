# interactive_anomaly_dashboard.py - インタラクティブダッシュボード
import torch
import timm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import time

from dataset_ad2 import AD2TrainDataset, AD2TestDataset

warnings.filterwarnings('ignore')
class SimpleDashboard:
    """
    Streamlit非依存の簡易ダッシュボード
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def create_interactive_plots(self, results_data, save_dir="interactive_plots"):
        """
        インタラクティブプロット作成
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        categories = list(results_data.keys())
        aucs = [results_data[cat]['auc'] for cat in categories]
        times = [results_data[cat]['time'] for cat in categories]
        
        # 1. インタラクティブAUCバーチャート
        fig = go.Figure()
        
        colors = ['red' if auc < 0.6 else 'orange' if auc < 0.8 else 'green' for auc in aucs]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=aucs,
            marker_color=colors,
            text=[f'{auc:.3f}' for auc in aucs],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>AUC: %{y:.4f}<br><extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '🎯 AnomalyVFM Performance Dashboard - AUC Scores',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Categories",
            yaxis_title="AUC Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=500
        )
        
        # 閾値線を追加
        fig.add_hline(y=0.9, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (0.9)")
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="Good (0.7)")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                     annotation_text="Fair (0.6)")
        
        fig.write_html(os.path.join(save_dir, "auc_scores_interactive.html"))
        
        # 2. 処理時間 vs AUC散布図
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=times,
            y=aucs,
            mode='markers+text',
            text=categories,
            textposition="top center",
            marker=dict(
                size=15,
                color=aucs,
                colorscale='RdYlGn',
                colorbar=dict(title="AUC Score"),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            hovertemplate='<b>%{text}</b><br>Time: %{x:.1f}s<br>AUC: %{y:.4f}<br><extra></extra>'
        ))
        
        fig2.update_layout(
            title={
                'text': '⚡ Processing Time vs Performance',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Processing Time (seconds)",
            yaxis_title="AUC Score",
            template="plotly_white",
            height=500
        )
        
        fig2.write_html(os.path.join(save_dir, "time_vs_auc_interactive.html"))
        
        # 3. レーダーチャート（総合評価）
        metrics = ['AUC', 'Speed', 'Separation']
        
        fig3 = go.Figure()
        
        for cat in categories:
            # メトリクス正規化
            auc_norm = results_data[cat]['auc']
            speed_norm = max(0, 1 - results_data[cat]['time'] / max(times))  # 時間が短いほど高スコア
            separation_norm = max(0, min(1, (results_data[cat]['separation_ratio'] + 0.5)))  # -0.5~0.5を0~1に変換
            
            values = [auc_norm, speed_norm, separation_norm, auc_norm]  # 最初の値を最後にも追加（閉じるため）
            
            fig3.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],  # 閉じるため
                fill='toself',
                name=cat,
                line=dict(width=2)
            ))
        
        fig3.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title={
                'text': '🌟 Comprehensive Performance Radar',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=600
        )
        
        fig3.write_html(os.path.join(save_dir, "radar_chart_interactive.html"))
        
        print(f"📊 Interactive plots saved to: {save_dir}/")
        print(f"   - auc_scores_interactive.html")
        print(f"   - time_vs_auc_interactive.html") 
        print(f"   - radar_chart_interactive.html")


def create_comprehensive_dashboard():
    """
    包括的ダッシュボード作成
    """
    print("🎨 Creating Comprehensive AnomalyVFM Dashboard...")
    
    # サンプルデータ（実際の結果から）
    sample_results = {
        'fruit_jelly': {
            'auc': 0.9717,
            'time': 90.9,
            'separation_ratio': 0.332,
            'optimal_threshold': 246.99
        },
        'fabric': {
            'auc': 0.6029,
            'time': 106.4,
            'separation_ratio': 0.182,
            'optimal_threshold': 511.21
        },
        'can': {
            'auc': 0.4443,
            'time': 65.2,
            'separation_ratio': -0.117,
            'optimal_threshold': 634.90
        }
    }
    
    dashboard = SimpleDashboard()
    dashboard.create_interactive_plots(sample_results)
    
    # データテーブル作成
    df = pd.DataFrame(sample_results).T
    df.index.name = 'Category'
    df = df.round(4)
    
    print(f"\n📋 Results Summary Table:")
    print("=" * 80)
    print(df.to_string())
    
    # 統計サマリー
    print(f"\n📈 Statistical Summary:")
    print(f"   Average AUC: {df['auc'].mean():.4f}")
    print(f"   Best AUC: {df['auc'].max():.4f} ({df['auc'].idxmax()})")
    print(f"   Worst AUC: {df['auc'].min():.4f} ({df['auc'].idxmin()})")
    print(f"   Average Time: {df['time'].mean():.1f}s")
    print(f"   Fastest: {df['time'].min():.1f}s ({df['time'].idxmin()})")
    
    # CSVエクスポート
    df.to_csv('anomalyvfm_results.csv')
    print(f"\n💾 Results exported to: anomalyvfm_results.csv")
    
    return df


def create_comparison_chart():
    """
    カテゴリ間比較チャート作成
    """
    print("\n📊 Creating category comparison charts...")
    
    # 実際の結果データ
    categories = ['fruit_jelly', 'fabric', 'can']
    aucs = [0.9717, 0.6029, 0.4443]
    times = [90.9, 106.4, 65.2]
    
    # 静的プロット作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUCバーチャート（改良版）
    colors = ['darkgreen' if auc >= 0.9 else 'orange' if auc >= 0.7 else 'lightcoral' for auc in aucs]
    bars = ax1.bar(categories, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # バーに値とグレードを表示
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        grade = 'EXCELLENT' if auc >= 0.9 else 'GOOD' if auc >= 0.7 else 'FAIR'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{auc:.3f}\n({grade})', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('🎯 AUC Performance by Category', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (≥0.9)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good (≥0.7)')
    ax1.legend()
    
    # 2. 処理時間
    ax2.bar(categories, times, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    for i, (cat, time) in enumerate(zip(categories, times)):
        ax2.text(i, time + 2, f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('⚡ Processing Speed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 効率性（AUC/Time）
    efficiency = [auc/time * 100 for auc, time in zip(aucs, times)]
    ax3.bar(categories, efficiency, color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    for i, (cat, eff) in enumerate(zip(categories, efficiency)):
        ax3.text(i, eff + 0.1, f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylabel('Efficiency (AUC/Time × 100)', fontsize=12, fontweight='bold')
    ax3.set_title('🚀 Efficiency Score', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 総合スコア（レーダーチャート風の棒グラフ）
    # 正規化スコア
    auc_norm = [(auc - min(aucs)) / (max(aucs) - min(aucs)) for auc in aucs]
    speed_norm = [(max(times) - time) / (max(times) - min(times)) for time in times]  # 時間は逆転
    overall_score = [(a + s) / 2 for a, s in zip(auc_norm, speed_norm)]
    
    ax4.bar(categories, overall_score, color='mediumpurple', alpha=0.8, edgecolor='indigo', linewidth=1)
    for i, (cat, score) in enumerate(zip(categories, overall_score)):
        ax4.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    ax4.set_ylabel('Overall Score (Normalized)', fontsize=12, fontweight='bold')
    ax4.set_title('🌟 Overall Performance Score', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparison charts created and saved as 'comprehensive_comparison.png'")


if __name__ == "__main__":
    print("🎨 AnomalyVFM Interactive Dashboard Generator")
    print("=" * 60)
    
    # 1. 包括的ダッシュボード作成
    results_df = create_comprehensive_dashboard()
    
    # 2. 比較チャート作成
    create_comparison_chart()
    
    print("\n✅ All visualizations and dashboards created successfully!")
    print("📁 Check the following outputs:")
    print("   - interactive_plots/ (HTML interactive charts)")
    print("   - anomalyvfm_results.csv (Data export)")
    print("   - comprehensive_comparison.png (Static comparison chart)")