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
    try:
        # Create output directory for results
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', model_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, custom_objects={'lab_loss': lab_loss})
        print("Model loaded successfully")

        # Load test data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'prepared_data')
        print(f"Loading test data from: {data_dir}")
        
        test_input_path = os.path.join(data_dir, "comic_input_grayscale_test.npy")
        test_output_path = os.path.join(data_dir, "comic_output_color_test.npy")
        
        if not os.path.exists(test_input_path) or not os.path.exists(test_output_path):
            print(f"Error: Test data not found at {data_dir}")
            return
            
        X_test = np.load(test_input_path)
        y_test = np.load(test_output_path)
        print(f"Test data loaded. Shapes - Input: {X_test.shape}, Output: {y_test.shape}")

        # Convert to LAB color space
        print("Converting to LAB color space...")
        lab_images = np.array([rgb_to_lab(img) for img in y_test])
        l_channel = lab_images[:, :, :, 0:1]
        ab_channels = lab_images[:, :, :, 1:]
        
        # Normalize L channel
        X_test = (l_channel - 50.0) / 50.0
        print("Data preprocessing completed")

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_test[:num_samples], verbose=1)
        print("Predictions completed")
        
        # Print prediction statistics
        print(f"\nPrediction statistics:")
        print(f"A channel range: [{predictions[..., 0].min():.2f}, {predictions[..., 0].max():.2f}]")
        print(f"B channel range: [{predictions[..., 1].min():.2f}, {predictions[..., 1].max():.2f}]")

        # Plot results
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
            
    except Exception as e:
        import traceback
        print(f"Error testing model {model_path}:")
        print(traceback.format_exc())

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