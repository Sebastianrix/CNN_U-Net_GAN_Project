import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_and_preprocess_images(data_dir, img_size=256):
    """Load and preprocess images from directory"""
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Read image
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, (img_size, img_size))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images.append(img)
    
    return np.array(images)

def create_splits(images, train_size=0.7, val_size=0.15):
    """Create train/validation/test splits"""
    # First split into train and temp
    train_images, temp_images = train_test_split(
        images, train_size=train_size, random_state=42
    )
    
    # Split temp into validation and test
    val_size_adjusted = val_size / (1 - train_size)
    val_images, test_images = train_test_split(
        temp_images, train_size=0.5, random_state=42
    )
    
    return train_images, val_images, test_images

def process_images(images):
    """Process images into grayscale and color components"""
    gray_images = []
    color_images = []
    
    for img in images:
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Split channels
        h, s, v = cv2.split(hsv)
        
        # Create grayscale image (V channel)
        gray = v[..., np.newaxis]
        
        # Create color image (H and S channels)
        color = np.stack([h, s], axis=-1)
        
        gray_images.append(gray)
        color_images.append(color)
    
    return np.array(gray_images), np.array(color_images)

def save_splits(splits, base_dir):
    """Save splits to directory"""
    for name, (gray, color) in splits.items():
        # Create directory
        split_dir = os.path.join(base_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(split_dir, 'gray.npy'), gray)
        np.save(os.path.join(split_dir, 'color.npy'), color)

def main():
    # Parameters
    img_size = 256
    data_dir = 'Data/raw_comics'
    output_dir = 'Data/prepared_data'
    
    # Load images
    print("Loading images...")
    images = load_and_preprocess_images(data_dir, img_size)
    print(f"Processed {len(images)} images")
    
    # Create splits
    train_images, val_images, test_images = create_splits(images)
    
    # Process each split
    splits = {}
    for name, split_images in [
        ('train', train_images),
        ('val', val_images),
        ('test', test_images)
    ]:
        gray, color = process_images(split_images)
        splits[name] = (gray, color)
    
    # Print split information
    print("\nSplit sizes:")
    for name, (gray, color) in splits.items():
        print(f"{name.capitalize()}: {len(gray)}")
    
    # Save splits
    print("\nSaving splits...")
    save_splits(splits, output_dir)
    
    print(f"\nData preparation complete!")
    print(f"Files saved in {output_dir}/")

if __name__ == "__main__":
    main() 