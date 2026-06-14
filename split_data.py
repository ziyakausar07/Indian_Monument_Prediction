import os
import shutil
import random

base_dir = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images"

source = os.path.join(base_dir, "all_data")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

TRAIN_COUNT = 120
TEST_COUNT = 30

# STEP 1: Delete old train/test
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir)
os.makedirs(test_dir)

print("Splitting dataset (120 Train + 30 Test)...\n")

for class_name in os.listdir(source):
    class_path = os.path.join(source, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total_needed = TRAIN_COUNT + TEST_COUNT

    # Handle small dataset
    if len(images) < total_needed:
        print(f"Warning: {class_name}: Only {len(images)} images available")

        split_index = int(len(images) * 0.8)
        train_imgs = images[:split_index]
        test_imgs = images[split_index:]
    else:
        selected_images = images[:total_needed]
        train_imgs = selected_images[:TRAIN_COUNT]
        test_imgs = selected_images[TRAIN_COUNT:total_needed]

    os.makedirs(os.path.join(train_dir, class_name))
    os.makedirs(os.path.join(test_dir, class_name))

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

    print(f"OK: {class_name} -> Train: {len(train_imgs)}, Test: {len(test_imgs)}")

print("\nData split complete!")