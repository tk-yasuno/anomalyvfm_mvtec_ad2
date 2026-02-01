# dataset_ad2.py
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class AD2TrainDataset(Dataset):
    """
    MVTec-AD2 train (normal only)
    root/category/category/train/good/*.jpg or *.png
    """
    def __init__(self, root, category, image_size=518):
        # MVTec-AD2の実際のパス構造: data/MVTec AD2/category/category/train/good/
        train_path = os.path.join(root, category, category, "train", "good")
        
        # .jpg と .png の両方に対応
        jpg_paths = sorted(glob(os.path.join(train_path, "*.jpg")))
        png_paths = sorted(glob(os.path.join(train_path, "*.png")))
        self.paths = jpg_paths + png_paths
        
        print(f"[{category}] Train images: {len(self.paths)}")
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet標準化
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class AD2TestDataset(Dataset):
    """
    MVTec-AD2 test_public (normal + anomaly)
    root/category/category/test_public/good/*.jpg (normal: label=0)
    root/category/category/test_public/bad/*.jpg (anomaly: label=1)
    """
    def __init__(self, root, category, image_size=518):
        self.data_root = root  # Store root for GT mask loading
        self.category = category  # Store category name
        
        # MVTec-AD2の実際のパス構造
        good_path = os.path.join(root, category, category, "test_public", "good")
        bad_path = os.path.join(root, category, category, "test_public", "bad")
        
        # Normal images (label=0)
        good_jpg = sorted(glob(os.path.join(good_path, "*.jpg")))
        good_png = sorted(glob(os.path.join(good_path, "*.png")))
        good_paths = good_jpg + good_png
        
        # Anomaly images (label=1)  
        bad_jpg = sorted(glob(os.path.join(bad_path, "*.jpg")))
        bad_png = sorted(glob(os.path.join(bad_path, "*.png")))
        bad_paths = bad_jpg + bad_png
        
        self.paths = good_paths + bad_paths
        self.labels = [0] * len(good_paths) + [1] * len(bad_paths)
        
        print(f"[{category}] Test images: {len(good_paths)} normal, {len(bad_paths)} anomaly")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet標準化
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
    
    def get_image_path(self, idx):
        """Get the file path of the image at given index"""
        return self.paths[idx] if idx < len(self.paths) else None