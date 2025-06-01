import os
import sys

# Add Src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Src'))

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from colorization_gan_v3 import ColorizeGAN, ColorSpace

def display_results(original, grayscale, generated, index):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale Input')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(generated)
    plt.title('Generated Color')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = 'colorization_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'result_sample_{index}.png'))
    plt.show()
    plt.close()

def main():
    # Create GAN model with the same architecture
    print("Creating model...")
    gan = ColorizeGAN(img_size=256, color_space=ColorSpace.HSV)
    
    # Build the model with a dummy input to initialize weights
    print("\nInitializing model weights...")
    dummy_input = np.zeros((1, 256, 256, 1), dtype=np.float32)
    _ = gan.generator(dummy_input)
    
    # Load weights
    print("\nLoading weights...")
    try:
        gan.generator.load_weights('checkpoints/epoch_10/generator.h5')
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return
    
    # Load test data
    print("\nLoading test data...")
    try:
        test_gray = np.load('Data/prepared_data/comic_input_grayscale_test.npy')
        test_color = np.load('Data/prepared_data/comic_output_color_test.npy')
        
        print(f"Test data loaded: {len(test_gray)} images")
        print(f"Grayscale shape: {test_gray.shape}")
        print(f"Color shape: {test_color.shape}")
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return
    
    # Process a few test samples
    num_samples = 5  # Number of test samples to process
    for i in range(num_samples):
        print(f"\nProcessing test sample {i+1}/{num_samples}...")
        
        try:
            # Get input grayscale image
            gray_input = test_gray[i]
            
            # Generate colors
            gray_batch = np.expand_dims(gray_input, 0)  # Add batch dimension
            generated = gan.generator.predict(gray_batch)
            
            # Create full HSV image
            hsv = np.zeros((*gray_input.shape[:2], 3), dtype=np.uint8)
            hsv[..., 0] = generated[0, ..., 0]  # Hue
            hsv[..., 1] = generated[0, ..., 1]  # Saturation
            hsv[..., 2] = (gray_input[..., 0] * 255.0).astype(np.uint8)  # Value
            
            # Convert to RGB
            generated_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Display and save results
            display_results(
                test_color[i],  # Original
                gray_input[..., 0],  # Grayscale
                generated_rgb,  # Generated
                i
            )
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 