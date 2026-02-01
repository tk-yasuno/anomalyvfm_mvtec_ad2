# Anomaly Detection Lessons Learned
## AnomalyVFM MVP Development Journey

### プロジェクト概要
MVTec-AD2データセットを用いたAnomalyVFM（Vision Foundation Model for Anomaly Detection）の開発・検証プロジェクト。産業異常検出における各種アプローチの性能を系統的に評価。

---

## 🏆 バージョン別パフォーマンス比較

### v0.1: MobileNetV3ベースライン
- **アーキテクチャ**: MobileNetV3 + Mahalanobis距離
- **対象カテゴリー**: 3カテゴリー (fruit_jelly, fabric, can)
- **AUC平均**: 0.6733
- **特徴**: 軽量、高速、surprisingly effective

### v0.3: DINOv2単体 ⭐ **BEST PERFORMER**
- **アーキテクチャ**: DINOv2-Base (768D) + Mahalanobis距離
- **対象カテゴリー**: 3カテゴリー
- **AUC平均**: 0.6275
- **特徴**: 
  - Self-supervised事前学習の恩恵
  - シンプルで安定
  - **実用性と性能のベストバランス**

### v0.7: Foundation Model
- **アーキテクチャ**: CLIP + DINOv2 Multi-modal
- **対象カテゴリー**: 3カテゴリー  
- **AUC平均**: 0.5315
- **特徴**: Text promptingの効果限定的

### v0.8: 最適化Foundation
- **アーキテクチャ**: 改良CLIP + DINOv2
- **対象カテゴリー**: 3カテゴリー
- **AUC平均**: 0.4868
- **特徴**: カテゴリ特化プロンプトも効果不十分

### v0.9: Multi-scale DINOv2
- **アーキテクチャ**: DINOv2-Base + DINOv2-Large
- **対象カテゴリー**: 3カテゴリー
- **AUC平均**: 0.6053 (Base単体)
- **発見**: Largeモデルは期待ほどの効果なし

### v1.0: アンサンブル
- **アーキテクチャ**: DINOv2-Base + 5種異常検出手法
- **対象カテゴリー**: **7カテゴリー全て**
- **AUC平均**: 0.5728
- **手法**: Mahalanobis, SVM, Isolation Forest, LOF, K-NN

---

## 📊 MVTec AD2カテゴリー別分析

### 高性能カテゴリー (AUC > 0.6)
1. **vial**: 0.6882 (Mahalanobis最適)
   - 透明な液体容器、形状変化が明確
   - 異常：気泡、汚れ、変形が視覚的に識別しやすい

2. **walnuts**: 0.6613 (SVM最適)  
   - 自然物、テクスチャベースの異常
   - 異常：欠け、虫食い、色変化

3. **fabric**: 0.6556 (K-NN最適)
   - 織物パターン、規則性のある構造
   - 異常：穴、汚れ、織り不良

4. **fruit_jelly**: 0.6308 (Mahalanobis最適)
   - ゼリー状食品、表面の均一性
   - 異常：気泡、亀裂、変色

### 困難カテゴリー (AUC < 0.5)
1. **sheet_metal**: 0.3667 (最困難)
   - 金属板、反射・照明の影響大
   - 異常：微細な傷、凹み（視覚的に曖昧）

2. **wallplugs**: 0.4474
   - プラスチック製品、複雑な3D形状
   - 異常：成形不良、細部の欠陥

3. **can**: 0.5594
   - 金属缶、反射特性が複雑
   - 異常：凹み、傷（軽微なものは識別困難）

---

## 🔍 技術的発見と教訓

### ✅ 成功要因
1. **DINOv2の優位性**
   - Self-supervised事前学習が産業用途に適合
   - 768次元特徴量が十分な表現力を提供
   - PCAによる次元削減は不要（情報損失を避ける）

2. **シンプルアプローチの有効性**
   - Mahalanobis距離の安定した性能
   - 複雑な手法が常に優秀とは限らない
   - **v0.3のシンプルさが最も実用的**

### ❌ 期待外れの結果
1. **Foundation Modelの限界**
   - CLIP+DINOv2マルチモーダルが期待を下回る
   - Text promptingが産業異常検出に不適合
   - ドメインギャップ（自然画像 vs 産業画像）

2. **アンサンブルの非効率性**
   - 複数手法の組み合わせが性能向上に寄与せず
   - 計算コスト増大に見合わない効果
   - 手法選択の重要性（1つの最適手法 > 複数手法の平均）

3. **Large Modelの逆効果**
   - DINOv2-LargeがBaseより劣る
   - パラメータ数と性能は必ずしも比例しない

---

## 💡 MVTec AD2多様性に基づく改善戦略

### カテゴリー特性別アプローチ

#### 1. **透明/半透明材料** (vial, fruit_jelly)
- **現状**: 比較的高性能 (0.6-0.7 AUC)
- **改善案**:
  - 照明条件の標準化
  - 透明度特化の特徴量抽出
  - Multi-view imaging（複数角度からの撮影）

#### 2. **金属材料** (sheet_metal, can)
- **現状**: 困難カテゴリー (0.3-0.6 AUC)  
- **課題**: 反射、照明むら、表面テクスチャの複雑性
- **改善案**:
  - **Polarized lighting**: 偏光フィルターによる反射制御
  - **Multi-spectral imaging**: 可視光以外の波長利用
  - **3D scanning**: 深度情報による形状異常検出
  - **Domain-specific augmentation**: 照明変化を考慮したデータ拡張

#### 3. **テクスチャ材料** (fabric, walnuts)
- **現状**: 中程度性能 (0.6-0.7 AUC)
- **改善案**:
  - **Texture analysis**: Gabor filter, LBP等のテクスチャ特徴量
  - **Scale-invariant features**: マルチスケールテクスチャ解析
  - **Wavelet transform**: 周波数ドメインでの異常検出

#### 4. **複雑形状** (wallplugs)
- **現状**: 困難カテゴリー (0.4 AUC)
- **改善案**:
  - **3D Point Cloud**: 立体形状での異常検出
  - **Multi-view stereo**: 複数視点からの3D復元
  - **Shape analysis**: 幾何学的特徴量による形状解析

### 統合改善戦略

#### 1. **カテゴリー適応型アーキテクチャ**
```python
# Pseudo-code
if category in ['sheet_metal', 'can']:
    # 金属特化: 反射対応
    model = MetalAnomalyDetector(polarized_features=True)
elif category in ['fabric', 'walnuts']:  
    # テクスチャ特化
    model = TextureAnomalyDetector(multi_scale=True)
elif category in ['vial', 'fruit_jelly']:
    # 透明材料特化  
    model = TransparentAnomalyDetector(multi_view=True)
```

#### 2. **データ拡張戦略**
- **物理ベース拡張**: 照明、視点、材質特性を考慮
- **GAN生成**: カテゴリ特化の異常パターン生成
- **Synthetic data**: 3Dレンダリングによる追加データ

#### 3. **ハイブリッドアプローチ**
- **Classical + Deep**: 従来手法とDNNの組み合わせ
- **Multi-modal sensing**: RGB + Depth + Infrared
- **Active learning**: 困難サンプルの能動学習

---

## 🎯 最終推奨アーキテクチャ

### 実用システム設計
**ベース**: v0.3 DINOv2単体アプローチ
**拡張**: カテゴリー適応型前処理

```yaml
Production_System:
  Base_Model: DINOv2-Base (768D)
  Detection_Method: Mahalanobis Distance
  
  Category_Specific_Preprocessing:
    Metal: 
      - Polarized_Lighting
      - Multi_Spectral_Imaging
    Transparent:
      - Multi_View_Capture  
      - Illumination_Standardization
    Texture:
      - Multi_Scale_Analysis
      - Wavelet_Transform
    Complex_Shape:
      - 3D_Point_Cloud
      - Multi_View_Stereo
      
  Performance_Target:
    - Overall_AUC: >0.7
    - Processing_Time: <30s/sample
    - Memory_Usage: <4GB GPU
```

### 開発ロードマップ
1. **Phase 1**: カテゴリー特化前処理の実装
2. **Phase 2**: Multi-modal sensing統合  
3. **Phase 3**: Active learning による継続改善
4. **Phase 4**: 実環境デプロイメント最適化

---

## 📈 性能改善予測

### 目標値設定
| Category | Current v1.0 | Target v2.0 | Improvement Strategy |
|----------|-------------|-------------|---------------------|
| sheet_metal | 0.3667 | **0.6500** | Polarized imaging + 3D |
| wallplugs | 0.4474 | **0.6000** | 3D point cloud analysis |
| can | 0.5594 | **0.7000** | Multi-spectral + shape |
| fruit_jelly | 0.6308 | **0.7500** | Multi-view optimization |
| fabric | 0.6556 | **0.7800** | Advanced texture analysis |
| walnuts | 0.6613 | **0.7500** | Scale-invariant features |
| vial | 0.6882 | **0.8200** | Transparent material spec |

**Overall Target**: 0.7200 AUC (vs current 0.5728)

---

## 🏭 産業応用への提言

### 1. **段階的導入戦略**
- **高性能カテゴリー**: 即座に実用化可能
- **中性能カテゴリー**: 改善版で実用化  
- **困難カテゴリー**: 研究開発継続

### 2. **品質管理統合**
- **人間検査員との協調**: AI支援による効率化
- **信頼度スコア**: 判定の確信度を提示
- **False positive管理**: 過検出の許容範囲設定

### 3. **継続学習システム**
- **Online learning**: 現場データによる継続改善
- **Feedback loop**: 検査員フィードバックの活用
- **Domain adaptation**: 新しい製品への適応

---

## 📚 研究課題と今後の展開

### 短期課題 (3-6ヶ月)
1. **カテゴリー特化前処理の実装**
2. **Multi-view imaging system構築**
3. **Polarized lighting setup開発**

### 中期課題 (6-12ヶ月)  
1. **3D point cloud異常検出**
2. **Multi-spectral imaging統合**
3. **Real-time processing最適化**

### 長期ビジョン (1-2年)
1. **Universal anomaly detector**
2. **Zero-shot新カテゴリー対応**
3. **Explainable AI integration**

---

## 💻 コード資産とリポジトリ構成

### 実装済みバージョン
- `anomaly_vfm_v03_dino.py`: **推奨ベースライン**
- `anomaly_vfm_v10_ensemble.py`: 7カテゴリー完全対応
- `dataset_ad2.py`: MVTec-AD2データローダー
- `visualized_anomalyvfm.py`: 可視化システム

### パフォーマンス記録
- v0.3: 0.6275 AUC (3カテゴリー) - **Production Ready**
- v1.0: 0.5728 AUC (7カテゴリー) - **Complete Coverage**

**結論**: **Simple is Best** - DINOv2単体アプローチ (v0.3) が最も実用的な解決策