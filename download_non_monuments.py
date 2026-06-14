import os
import requests
import time

# Paths
base_dir = r"e:\Project\Indian_MonumentsImage_Prediction\IndianMonumentsImagePrediction\Indian-monuments\images"
train_path = os.path.join(base_dir, "train", "non_monument")
test_path = os.path.join(base_dir, "test", "non_monument")
all_data_path = os.path.join(base_dir, "all_data", "non_monument")

# Ensure directories exist
for p in [train_path, test_path, all_data_path]:
    os.makedirs(p, exist_ok=True)

import concurrent.futures

def download_single_image(idx, path_prefix):
    try:
        url = f"https://picsum.photos/224/224?random={idx}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filename = f"non_monument_{idx}.jpg"
            filepath = os.path.join(path_prefix, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading image {idx}: {e}")
    return False

def download_images_fast(count, path_prefix, start_idx=0):
    print(f"Downloading {count} images to {path_prefix} (THREADED)...")
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_single_image, i, path_prefix) for i in range(start_idx, start_idx + count)]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Downloaded {success_count}/{count} images...")
    
    print(f"Finished downloading {success_count} images to {path_prefix}.\n")
    return success_count

if __name__ == "__main__":
    # Download 250 images directly to all_data/non_monument
    # This allows split_data.py to distribute them correctly
    total_count = download_images_fast(250, all_data_path)
    
    print(f"✅ Download complete! Total in all_data: {total_count}")
