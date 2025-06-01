import os
import numpy as np
import cv2
from tqdm import tqdm

def convert_to_hsv(data_dir):
    """Convert RGB data to HSV format"""
    print("Loading RGB data...")
    
    # Load training data
    rgb_train = np.load(os.path.join(data_dir, 'comic_output_color_train.npy'))
    gray_train = np.load(os.path.join(data_dir, 'comic_input_grayscale_train.npy'))
    
    # Load validation data
    rgb_val = np.load(os.path.join(data_dir, 'comic_output_color_val.npy'))
    gray_val = np.load(os.path.join(data_dir, 'comic_input_grayscale_val.npy'))
    
    # Load test data
    rgb_test = np.load(os.path.join(data_dir, 'comic_output_color_test.npy'))
    gray_test = np.load(os.path.join(data_dir, 'comic_input_grayscale_test.npy'))
    
    def process_batch(rgb_data, gray_data, desc):
        print(f"\nProcessing {desc} data...")
        hsv_data = []
        
        for img in tqdm(rgb_data, desc=f"Converting {desc} set to HSV"):
            # Convert RGB to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Keep only H and S channels (V will come from grayscale input)
            hsv_data.append(hsv[..., :2])
        
        hsv_data = np.array(hsv_data, dtype=np.float32)
        return hsv_data, gray_data
    
    # Process all sets
    hsv_train, gray_train = process_batch(rgb_train, gray_train, "training")
    hsv_val, gray_val = process_batch(rgb_val, gray_val, "validation")
    hsv_test, gray_test = process_batch(rgb_test, gray_test, "test")
    
    # Print statistics
    print("\nData statistics:")
    print(f"Training set - HSV shape: {hsv_train.shape}, Grayscale shape: {gray_train.shape}")
    print(f"Validation set - HSV shape: {hsv_val.shape}, Grayscale shape: {gray_val.shape}")
    print(f"Test set - HSV shape: {hsv_test.shape}, Grayscale shape: {gray_test.shape}")
    
    # Save HSV data
    output_dir = os.path.join(data_dir, 'hsv')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving HSV data...")
    np.save(os.path.join(output_dir, 'comic_output_hsv_train.npy'), hsv_train)
    np.save(os.path.join(output_dir, 'comic_output_hsv_val.npy'), hsv_val)
    np.save(os.path.join(output_dir, 'comic_output_hsv_test.npy'), hsv_test)
    
    # Copy grayscale data
    print("\nCopying grayscale data...")
    np.save(os.path.join(output_dir, 'comic_input_grayscale_train.npy'), gray_train)
    np.save(os.path.join(output_dir, 'comic_input_grayscale_val.npy'), gray_val)
    np.save(os.path.join(output_dir, 'comic_input_grayscale_test.npy'), gray_test)
    
    print(f"\nConversion complete! HSV data saved in {output_dir}")
    print("\nHSV ranges:")
    print(f"H channel: [{hsv_train[..., 0].min()}, {hsv_train[..., 0].max()}]")
    print(f"S channel: [{hsv_train[..., 1].min()}, {hsv_train[..., 1].max()}]")

if __name__ == "__main__":
    convert_to_hsv('Data/prepared_data') 