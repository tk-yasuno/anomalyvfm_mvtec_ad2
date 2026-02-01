# single_category_test.py - 単一カテゴリテスト
import torch
from dataset_ad2 import AD2TrainDataset, AD2TestDataset

def test_category_data(category):
    root = r"C:\Users\yasun\MultimodalAD\anomalyvfm_mvtec_ad2\data\MVTec AD2"
    
    print(f"Testing category: {category}")
    
    try:
        train_ds = AD2TrainDataset(root, category)
        test_ds = AD2TestDataset(root, category)
        
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Test samples: {len(test_ds)}")
        
        if len(test_ds) > 0:
            # テストデータのラベル分布確認
            normal_count = sum(1 for _, label in test_ds if label == 0)
            anomaly_count = sum(1 for _, label in test_ds if label == 1)
            print(f"  Normal: {normal_count}, Anomaly: {anomaly_count}")
        
        return len(train_ds) > 0 and len(test_ds) > 0
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        return False

if __name__ == "__main__":
    # 新規4カテゴリをテスト
    new_categories = ["sheet_metal", "vial", "wallplugs", "walnuts"]
    
    print("🔍 Testing data availability for new categories:")
    print("=" * 50)
    
    for category in new_categories:
        success = test_category_data(category)
        status = "✅ Ready" if success else "❌ Issues"
        print(f"{category}: {status}")
        print()
    
    print("Data testing complete!")