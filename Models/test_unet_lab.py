import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import keras
import os
import tensorflow as tf

# Add Src directory to path for importing our model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add Src directory to path
from unet_model_lab import rgb_to_lab, lab_to_rgb, lab_loss, build_unet_lab

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

def scale_output(x):
    return tf.tanh(x) * 127.0

def output_shape(input_shape):
    return input_shape

def test_model(model_path, num_samples=5):
    print(f"\nTesting model: {model_path}")
    try:
        # Create a new model with the same architecture
        model = build_unet_lab((256, 256, 1))
        # Load the weights from the saved model
        model.load_weights(model_path)
        print("Model loaded successfully")

        # Load test data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'prepared_data')
        X_test = np.load(os.path.join(data_dir, "comic_input_grayscale_test.npy"))
        y_test = np.load(os.path.join(data_dir, "comic_output_color_test.npy"))
        print("Test data loaded")

        # Convert ground truth to LAB color space
        y_test_lab = np.array([rgb_to_lab(img) for img in y_test])
        
        # Normalize L channel to [-1, 1] for the input
        X_test = (X_test - 0.5) * 2
        print("Data converted to LAB color space")

        # Predict AB channels
        predictions = model.predict(X_test[:num_samples])
        print("Predictions completed")

        # Plot results
        for i in range(num_samples):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grayscale
            axes[0].imshow(X_test[i].squeeze(), cmap='gray')
            axes[0].set_title("Grayscale Input")
            axes[0].axis("off")
            
            # Predicted colorization
            # Combine L channel with predicted AB channels
            l_channel = y_test_lab[i, :, :, 0:1]  # Using original L for better visualization
            predicted_lab = np.concatenate([l_channel, predictions[i]], axis=-1)
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
            plt.show()
            
            # Print color space ranges for debugging
            print(f"\nSample {i+1} ranges:")
            print(f"L channel range: [{l_channel.min():.2f}, {l_channel.max():.2f}]")
            print(f"Predicted A channel range: [{predictions[i,:,:,0].min():.2f}, {predictions[i,:,:,0].max():.2f}]")
            print(f"Predicted B channel range: [{predictions[i,:,:,1].min():.2f}, {predictions[i,:,:,1].max():.2f}]")
            
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")

if __name__ == "__main__":
    # Test LAB models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Models')
    models_to_test = [
        os.path.join(models_dir, "best_unet_lab.keras"),
        os.path.join(models_dir, "final_unet_lab.keras")
    ]

    for model_path in models_to_test:
        if os.path.exists(model_path):
            test_model(model_path)
        else:
            print(f"Model not found: {model_path}") 