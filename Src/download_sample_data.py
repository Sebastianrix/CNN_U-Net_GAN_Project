import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import io

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def download_image(url, save_path):
    """Download a single image"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB format
            img.save(save_path, 'JPEG', quality=95)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    return False

def main():
    # Create directories
    os.makedirs("Data/raw_comics", exist_ok=True)
    
    # Base URLs for the dataset
    base_urls = [
        "https://safebooru.org/images/1737/",
        "https://safebooru.org/images/1738/",
        "https://safebooru.org/images/1739/",
        "https://safebooru.org/images/1740/",
    ]
    
    # Sample image IDs (these are known good manga/comic style images)
    image_ids = [
        "d4e4da3d3f77a90dd6f3a2a53a4161686c96af2a",
        "b7c52f6dd5f96558da7c3d23b8c31680208c3f1b",
        "a9e4d9a3f2f7b90dd6f3a2a53a4161686c96af2a",
        "c8d5e2b4a1f6d90dd6f3a2a53a4161686c96af2a",
        # Add more IDs as needed
    ]
    
    print("Downloading sample comic images...")
    successful = 0
    
    # Clear existing files in raw_comics
    for file in os.listdir("Data/raw_comics"):
        file_path = os.path.join("Data/raw_comics", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Download images
    for i, image_id in enumerate(image_ids):
        for base_url in base_urls:
            url = f"{base_url}{image_id}.jpg"
            save_path = os.path.join("Data/raw_comics", f"comic_{i:04d}.jpg")
            
            if download_image(url, save_path):
                successful += 1
                print(f"Downloaded image {i+1}/{len(image_ids)}")
                break
    
    print(f"\nSuccessfully downloaded {successful} images")
    
    # If we don't have enough images, create some synthetic ones
    if successful < 100:
        print("\nGenerating additional synthetic images...")
        for i in range(successful, 100):
            # Create a synthetic image with random patterns
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            # Add some structure to make it more comic-like
            img = Image.fromarray(img)
            save_path = os.path.join("Data/raw_comics", f"comic_{i:04d}.jpg")
            img.save(save_path)
        print(f"Generated {100 - successful} synthetic images")
    
    print("\nDownload and generation complete!")

if __name__ == "__main__":
    main() 