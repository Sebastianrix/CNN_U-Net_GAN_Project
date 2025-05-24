import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import keras
import os
import tensorflow as tf

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

# Define and register the combined loss function
@keras.saving.register_keras_serializable()
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 0.84 * mse + 0.16 * mae

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

        # Normalize input images to [-1, 1] range for better prediction with tanh
        X_test = (X_test - 0.5) * 2
        print("Input data normalized")

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
            
            plt.suptitle(f"Results for {os.path.basename(model_path)} - Sample {i+1}")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")

# Test both best and final models
models_to_test = [
    "Models/best_unet_rgb_v2.keras",
    "Models/final_unet_rgb_v2.keras"
]

for model_path in models_to_test:
    test_model(model_path) 