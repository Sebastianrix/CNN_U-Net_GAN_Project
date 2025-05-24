import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import keras
import os
import tensorflow as tf

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

def test_model(model_path, num_samples=5):
    print(f"\nTesting model: {model_path}")
    try:
        # Load model
        model = load_model(model_path)
        print("Model loaded successfully")

        # Load test data
        X_test = np.load("Data/prepared_data/comic_input_grayscale_test.npy")
        y_test = np.load("Data/prepared_data/comic_output_color_test.npy")
        print("Test data loaded")

        # For v1, we don't normalize to [-1, 1] since it uses sigmoid activation
        print("Using original input range [0, 1] for v1 model")

        # Predict
        predictions = model.predict(X_test[:num_samples])
        print("Predictions completed")

        # Plot results
        for i in range(num_samples):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Grayscale input
            axes[0].imshow(X_test[i].squeeze(), cmap="gray")
            axes[0].set_title("Grayscale Input")
            axes[0].axis("off")
            
            # Predicted color
            axes[1].imshow(predictions[i])
            axes[1].set_title("Predicted Color")
            axes[1].axis("off")
            
            # Ground truth
            axes[2].imshow(y_test[i])
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")
            
            plt.suptitle(f"Results for RGB U-Net v1 - Sample {i+1}")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")

# Test v1 model
model_path = "Models/unet_rgb_v1.keras"
test_model(model_path) 