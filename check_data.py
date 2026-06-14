import os

def check(path, name):
    print(f"\n📊 {name} Data:\n")

    for cls in os.listdir(path):
        class_path = os.path.join(path, cls)

        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{cls} → {count} images")


base_dir = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images"

train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "test")

check(train_path, "TRAIN")
check(test_path, "TEST")