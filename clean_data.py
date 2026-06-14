import os
import hashlib
from PIL import Image
import cv2

base_dir = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images\all_data"

MIN_SIZE = 100
BLUR_THRESHOLD = 50   # ⬅️ less strict (important)

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def safe_remove(path):
    try:
        os.remove(path)
    except:
        print("⚠ Skip:", path)

def is_blurry(path):
    img = cv2.imread(path)

    if img is None:
        return True

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < BLUR_THRESHOLD
    except:
        return True

def clean_images():
    print("🔍 Cleaning Started...\n")

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"📂 {class_name}")
        seen = set()

        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)

            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            try:
                # ❌ Blur remove
                if is_blurry(path):
                    print("❌ Blurry:", path)
                    safe_remove(path)
                    continue

                with Image.open(path) as img:
                    w, h = img.size

                    # ❌ Small remove
                    if w < MIN_SIZE or h < MIN_SIZE:
                        print("❌ Small:", path)
                        safe_remove(path)
                        continue

                    # ❌ Duplicate remove
                    file_hash = get_hash(path)
                    if file_hash in seen:
                        print("❌ Duplicate:", path)
                        safe_remove(path)
                        continue
                    seen.add(file_hash)

                    # 🔁 Only convert (NO resize)
                    img = img.convert("RGB")
                    img.save(path)

            except Exception as e:
                print("❌ Error:", path, e)
                safe_remove(path)

        print("✔ Done\n")

    print("✅ Cleaning complete!")

if __name__ == "__main__":
    clean_images()