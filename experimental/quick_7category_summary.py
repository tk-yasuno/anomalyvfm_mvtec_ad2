# quick_7category_summary.py - 全7カテゴリの概要確認
from dataset_ad2 import AD2TrainDataset, AD2TestDataset

def check_all_categories():
    """
    全7カテゴリのデータ概要を確認
    """
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    all_categories = [
        "can", "fabric", "fruit_jelly",          # 既存3カテゴリ
        "sheet_metal", "vial", "wallplugs", "walnuts"  # 新規4カテゴリ  
    ]
    
    print("🔍 MVTec-AD2 Dataset Overview - All 7 Categories")
    print("=" * 60)
    print(f"{'Category':<15} {'Train':<8} {'Test':<8} {'Normal':<8} {'Anomaly':<8} {'Status'}")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    ready_categories = []
    
    for category in all_categories:
        try:
            train_ds = AD2TrainDataset(root, category)
            test_ds = AD2TestDataset(root, category)
            
            train_count = len(train_ds)
            test_count = len(test_ds)
            
            # テストデータのラベル分布
            normal_count = 0
            anomaly_count = 0
            
            if test_count > 0:
                for i in range(min(test_count, 200)):  # 最初の200個だけチェック（高速化）
                    try:
                        _, label = test_ds[i]
                        if label == 0:
                            normal_count += 1
                        else:
                            anomaly_count += 1
                    except:
                        break
                
                # 全体に比例して推定
                if i > 0:
                    scale = test_count / (i + 1)
                    normal_count = int(normal_count * scale)
                    anomaly_count = int(anomaly_count * scale)
            
            status = "✅ Ready" if train_count > 0 and test_count > 0 else "❌ Issues"
            
            print(f"{category:<15} {train_count:<8} {test_count:<8} {normal_count:<8} {anomaly_count:<8} {status}")
            
            if train_count > 0 and test_count > 0:
                total_train += train_count
                total_test += test_count
                ready_categories.append(category)
                
        except Exception as e:
            print(f"{category:<15} {'Error':<8} {'Error':<8} {'Error':<8} {'Error':<8} ❌ Failed")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_train:<8} {total_test:<8} {'':<8} {'':<8} {len(ready_categories)}/7 Ready")
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Ready categories: {len(ready_categories)}/7")
    print(f"  📈 Total train samples: {total_train:,}")
    print(f"  🔍 Total test samples: {total_test:,}")
    print(f"  🎯 Categories ready for demo: {', '.join(ready_categories)}")
    
    return ready_categories

def create_demo_recommendation():
    """
    デモ実行の推奨事項を作成
    """
    ready_categories = check_all_categories()
    
    print(f"\n🚀 Demo Recommendations:")
    print("=" * 40)
    
    if len(ready_categories) >= 7:
        print("🎉 All 7 categories are ready!")
        print("💡 You can run the full demo:")
        print("   python full_7category_anomalyvfm_demo.py")
    elif len(ready_categories) >= 4:
        print(f"👍 {len(ready_categories)} categories ready for demo")
        print("💡 You can run a multi-category demo")
    else:
        print(f"⚠️  Only {len(ready_categories)} categories available")
        print("💡 Consider single-category testing first")
    
    if ready_categories:
        print(f"\n🎯 Ready categories: {', '.join(ready_categories)}")
    
    # 予想実行時間
    estimated_time = len(ready_categories) * 2.5  # 平均2.5分/カテゴリ
    print(f"⏱️  Estimated demo time: ~{estimated_time:.0f} minutes")

if __name__ == "__main__":
    create_demo_recommendation()