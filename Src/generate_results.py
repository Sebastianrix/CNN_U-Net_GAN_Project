import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configurations
MODEL_PATH = 'Notebooks/U-NET_RGB_best_model_v2.keras'  
DATA_PATH = 'Data/prepared_data/RGB'        
SAVE_DIR = 'Results'                     

# Makes directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Load test data
x_test = np.load(os.path.join(DATA_PATH, 'comic_input_grayscale_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'comic_output_color_test.npy'))

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Predict on the test set
predictions = model.predict(x_test, batch_size=4)


# Generate results ( two versions: without and with normalized )
for i in range(len(x_test)):
    # 1. Without normalization

    # Grayscale
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_test[i].squeeze(), cmap='gray')
    axes[0].set_title('Grayscale')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(predictions[i])
    axes[1].set_title('Predicted (unnormalized)')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(np.clip(y_test[i], 0, 1))
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'result_unnormalized_{i:04d}.png'))
    plt.close()

    # 2. Normalized prediction to [0, 1])

    # Grayscale
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_test[i].squeeze(), cmap='gray')
    axes[0].set_title('Grayscale')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(np.clip(predictions[i], 0, 1)) 
    axes[1].set_title('Predicted (Clipped)')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(np.clip(y_test[i], 0, 1))
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    # Save the results
    plt.suptitle(f'Results for Image {i+1}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'result_clipped_{i:04d}.png'))
    plt.close()