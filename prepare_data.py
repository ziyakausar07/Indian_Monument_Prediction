import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# PATHS
BASE_DIR = r"E:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images"
SOURCE_DIR = os.path.join(BASE_DIR, "all_data")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# CONFIG
# Use more images for better accuracy, up to 400 per category to stay balanced.
MAX_PER_CAT = 400 
TRAIN_SPLIT = 0.8 

def prepare_data():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} not found!")
        return

    # Clean folders for fresh start
    for d in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(d):
            print(f"Cleaning: {d}")
            shutil.rmtree(d)
        os.makedirs(d)

    monuments = [m for m in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, m))]
    
    for monument in monuments:
        print(f"Processing {monument}...")
        src_mon_path = os.path.join(SOURCE_DIR, monument)
        train_mon_path = os.path.join(TRAIN_DIR, monument)
        test_mon_path = os.path.join(TEST_DIR, monument)
        
        os.makedirs(train_mon_path, exist_ok=True)
        os.makedirs(test_mon_path, exist_ok=True)
        
        all_images = [f for f in os.listdir(src_mon_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)
        
        # Take max images or all available
        num_to_use = min(len(all_images), MAX_PER_CAT)
        selected_images = all_images[:num_to_use]
        
        # Split 80/20
        split_idx = int(num_to_use * TRAIN_SPLIT)
        train_images = selected_images[:split_idx]
        test_images = selected_images[split_idx:]
        
        # Copy to folders
        for img_name in train_images:
            shutil.copy(os.path.join(src_mon_path, img_name), os.path.join(train_mon_path, img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(src_mon_path, img_name), os.path.join(test_mon_path, img_name))
            
        print(f"  - {monument}: Train={len(train_images)}, Test={len(test_images)}")

    print("\nData preparation finished! Dynamic augmentation will be handled in train.py.")

if __name__ == "__main__":
    prepare_data()

