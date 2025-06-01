import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import os
import tensorflow as tf
import sys

# Add Src directory to path for importing our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet_model_hsv import rgb_to_hsv, hsv_to_rgb, hsv_loss

# Enable unsafe deserialization for Lambda layers
tf.keras.mixed_precision.set_global_policy('float32')

def test_model(model_path, num_samples=5):
    print(f"\nTesting model: {model_path}")
    
    # Create Results directory if it doesn't exist
    output_dir = "Results/unet_hsv_test"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        model = load_model(model_path, custom_objects={'hsv_loss': hsv_loss}, safe_mode=False)
        print("Model loaded successfully")

        # Load test data
        X_test = np.load("Data/prepared_data/comic_input_grayscale_test.npy")
        y_test = np.load("Data/prepared_data/comic_output_color_test.npy")
        print("Test data loaded")

        # Convert RGB test data to HSV
        print("Converting RGB to HSV...")
        hsv_images = rgb_to_hsv(y_test)
        
        # Use grayscale input as Value channel (already normalized)
        v_channel = X_test
        print("Data preprocessing completed")

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(v_channel[:num_samples], verbose=1)
        print("Predictions completed")
        
        # Print prediction statistics
        print(f"\nPrediction statistics:")
        print(f"H channel range: [{predictions[..., 0].min():.2f}, {predictions[..., 0].max():.2f}]")
        print(f"S channel range: [{predictions[..., 1].min():.2f}, {predictions[..., 1].max():.2f}]")

        # Plot and save results
        for i in range(num_samples):
            print(f"\nProcessing sample {i+1}...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grayscale
            axes[0].imshow(v_channel[i].squeeze(), cmap='gray')
            axes[0].set_title("Grayscale Input")
            axes[0].axis("off")
            
            # Predicted colorization
            predicted_hsv = np.concatenate([predictions[i], v_channel[i]], axis=-1)
            # Ensure HSV values are in the correct range
            predicted_hsv = predicted_hsv.copy()
            predicted_hsv[..., 0] = np.clip(predicted_hsv[..., 0], 0, 179)  # H channel
            predicted_hsv[..., 1] = np.clip(predicted_hsv[..., 1], 0, 255)  # S channel
            predicted_hsv[..., 2] = np.clip(predicted_hsv[..., 2], 0, 1)    # V channel
            predicted_rgb = hsv_to_rgb(predicted_hsv)
            # Ensure RGB values are in [0, 1]
            predicted_rgb = np.clip(predicted_rgb, 0, 255).astype(np.float32) / 255.0
            
            axes[1].imshow(predicted_rgb)
            axes[1].set_title("Predicted Color")
            axes[1].axis("off")
            
            # Ground truth (normalize to [0, 1])
            ground_truth = np.clip(y_test[i], 0, 255).astype(np.float32) / 255.0
            axes[2].imshow(ground_truth)
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")
            
            plt.suptitle(f"Results for HSV U-Net - Sample {i+1}")
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(output_dir, f'sample_{i+1}.png')
            print(f"Saving plot to: {save_path}")
            plt.savefig(save_path)
            plt.close()
            
            # Also save individual images
            plt.imsave(os.path.join(output_dir, f'grayscale_{i+1}.png'), v_channel[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(output_dir, f'predicted_{i+1}.png'), predicted_rgb)
            plt.imsave(os.path.join(output_dir, f'ground_truth_{i+1}.png'), ground_truth)
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test HSV models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Models')
    print(f"Looking for models in: {models_dir}")
    
    models_to_test = [
        os.path.join(models_dir, "best_unet_hsv.keras")
    ]

    for model_path in models_to_test:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            test_model(model_path)
        else:
            print(f"Model not found: {model_path}") 