import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import keras
import os
import tensorflow as tf
import sys

# Add Src directory to path for importing our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet_model_lab import rgb_to_lab, lab_to_rgb, lab_loss

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

def test_model(model_path, num_samples=5):
    print(f"\nTesting model: {model_path}")
    
    # Create Results directory if it doesn't exist
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        model = load_model(model_path, custom_objects={'lab_loss': lab_loss})
        print("Model loaded successfully")

        # Load test data
        X_test = np.load("Data/prepared_data/comic_input_grayscale_test.npy")
        y_test = np.load("Data/prepared_data/comic_output_color_test.npy")
        print("Test data loaded")

        # Convert RGB test data to LAB
        print("Converting RGB to LAB...")
        lab_images = rgb_to_lab(y_test)
        
        # Extract L channel and normalize to [-1, 1]
        l_channel = lab_images[:, :, :, 0:1]
        X_test = (l_channel - 50.0) / 50.0
        print("Data preprocessing completed")

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_test[:num_samples], verbose=1)
        
        # Denormalize predictions (scale back to LAB ranges)
        predictions = predictions.clip(-127, 127)  # Ensure valid LAB ranges
        print("Predictions completed")
        
        # Print prediction statistics
        print(f"\nPrediction statistics:")
        print(f"A channel range: [{predictions[..., 0].min():.2f}, {predictions[..., 0].max():.2f}]")
        print(f"B channel range: [{predictions[..., 1].min():.2f}, {predictions[..., 1].max():.2f}]")

        # Plot and save results
        for i in range(num_samples):
            print(f"\nProcessing sample {i+1}...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grayscale
            axes[0].imshow(l_channel[i].squeeze(), cmap='gray')
            axes[0].set_title("Grayscale Input")
            axes[0].axis("off")
            
            # Predicted colorization
            predicted_lab = np.concatenate([l_channel[i], predictions[i]], axis=-1)
            predicted_rgb = lab_to_rgb(predicted_lab)
            
            axes[1].imshow(predicted_rgb)
            axes[1].set_title("Predicted Color")
            axes[1].axis("off")
            
            # Ground truth
            axes[2].imshow(y_test[i])
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")
            
            plt.suptitle(f"Results for LAB U-Net - Sample {i+1}")
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(output_dir, f'sample_{i+1}.png')
            print(f"Saving plot to: {save_path}")
            plt.savefig(save_path)
            plt.close()
            
            # Also save individual images
            plt.imsave(os.path.join(output_dir, f'grayscale_{i+1}.png'), l_channel[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(output_dir, f'predicted_{i+1}.png'), predicted_rgb)
            plt.imsave(os.path.join(output_dir, f'ground_truth_{i+1}.png'), y_test[i])
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test LAB models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Models')
    print(f"Looking for models in: {models_dir}")
    
    models_to_test = [
        os.path.join(models_dir, "best_unet_lab_v8.keras")
    ]

    for model_path in models_to_test:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            test_model(model_path)
        else:
            print(f"Model not found: {model_path}")