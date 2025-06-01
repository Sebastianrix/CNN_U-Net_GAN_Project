import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

def load_and_process_image(image_path, target_size=(256, 256)):
    """Load and process a single image"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize RGB to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Create grayscale version (already normalized)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        
        return gray, img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def main():
    # Parameters
    comic_dir = "Data/comic_dataset/train"
    output_dir = "Data/prepared_data"
    target_size = (256, 256)
    val_split = 0.1
    test_split = 0.1
    
    # Get all image files recursively
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob(os.path.join(comic_dir, "**", ext), recursive=True))
    
    print(f"Found {len(image_files)} comic book images")
    
    # Process images
    gray_images = []
    color_images = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        gray, color = load_and_process_image(img_path, target_size)
        if gray is not None and color is not None:
            gray_images.append(gray)
            color_images.append(color)
    
    gray_images = np.array(gray_images)
    color_images = np.array(color_images)
    
    print(f"\nProcessed {len(gray_images)} images successfully")
    print(f"Gray shape: {gray_images.shape}")
    print(f"Color shape: {color_images.shape}")
    
    # Create train/val/test splits
    # First split out test set
    gray_temp, gray_test, color_temp, color_test = train_test_split(
        gray_images, color_images, test_size=test_split, random_state=42
    )
    
    # Then split remaining data into train/val
    val_size = val_split / (1 - test_split)
    gray_train, gray_val, color_train, color_val = train_test_split(
        gray_temp, color_temp, test_size=val_size, random_state=42
    )
    
    # Print split sizes
    print("\nDataset splits:")
    print(f"Training: {len(gray_train)} images")
    print(f"Validation: {len(gray_val)} images")
    print(f"Test: {len(gray_test)} images")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving processed data...")
    np.save(os.path.join(output_dir, "comic_input_grayscale_train.npy"), gray_train)
    np.save(os.path.join(output_dir, "comic_output_color_train.npy"), color_train)
    np.save(os.path.join(output_dir, "comic_input_grayscale_val.npy"), gray_val)
    np.save(os.path.join(output_dir, "comic_output_color_val.npy"), color_val)
    np.save(os.path.join(output_dir, "comic_input_grayscale_test.npy"), gray_test)
    np.save(os.path.join(output_dir, "comic_output_color_test.npy"), color_test)
    
    print("\nData preparation completed!")
    print(f"Files saved in {output_dir}/")

if __name__ == "__main__":
    main() 