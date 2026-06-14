import os
import random
import shutil

train_path = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images\train"

TARGET = 120

print("Balancing Started...\n")

for cls in os.listdir(train_path):
    class_path = os.path.join(train_path, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    # Case 1: More than TARGET -> Remove
    if len(images) > TARGET:
        remove_count = len(images) - TARGET
        remove_imgs = random.sample(images, remove_count)

        for img in remove_imgs:
            os.remove(os.path.join(class_path, img))

        print(f"Removed {remove_count} from {cls}")

    # Case 2: Less than TARGET -> Duplicate
    elif len(images) < TARGET:
        needed = TARGET - len(images)

        for i in range(needed):
            img = random.choice(images)
            src = os.path.join(class_path, img)

            # unique name
            new_name = f"copy_{i}_{img}"
            dst = os.path.join(class_path, new_name)

            shutil.copy(src, dst)

        print(f"Added {needed} to {cls}")

    else:
        print(f"OK: {cls} already {TARGET}")

print("\nBalancing Done!")