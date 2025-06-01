import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import sys
import datetime

# Add Src directory to path for importing our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet_model_hsv import build_unet_hsv, hsv_loss, rgb_to_hsv, get_callbacks

def prepare_hsv_data(rgb_images):
    """Prepare training data in HSV color space"""
    print("Converting RGB images to HSV...")
    print(f"Input RGB shape: {rgb_images.shape}, dtype: {rgb_images.dtype}, range: [{rgb_images.min()}, {rgb_images.max()}]")
    
    # Normalize RGB to [0, 1] if needed
    if rgb_images.dtype == np.uint8 or rgb_images.max() > 1.0:
        print("Normalizing RGB values to [0, 1]...")
        rgb_images = rgb_images.astype(np.float32) / 255.0
    
    hsv_images = rgb_to_hsv(rgb_images)
    print(f"HSV shape: {hsv_images.shape}, range: H[{hsv_images[...,0].min()}, {hsv_images[...,0].max()}], S[{hsv_images[...,1].min()}, {hsv_images[...,1].max()}], V[{hsv_images[...,2].min()}, {hsv_images[...,2].max()}]")
    
    # Split into V channel (input) and H,S channels (target)
    v_channel = hsv_images[..., 2:3]  # Value channel
    hs_channels = hsv_images[..., :2]  # Hue and Saturation channels
    
    # Value channel is already normalized to [0, 1]
    X = v_channel
    
    # H and S channels are our targets
    y = hs_channels
    
    print(f"Prepared X shape: {X.shape}, range: [{X.min()}, {X.max()}]")
    print(f"Prepared y shape: {y.shape}, range: [{y.min()}, {y.max()}]")
    
    return X, y

if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Setup directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'Data', 'prepared_data')
    models_dir = os.path.join(project_root, 'Models')
    
    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'hsv_unet_' + timestamp)
    
    # Create necessary directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Models will be saved to: {models_dir}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Load and prepare data
    print("\nLoading training data...")
    X_train = np.load(os.path.join(data_dir, "comic_input_grayscale_train.npy"))
    y_train = np.load(os.path.join(data_dir, "comic_output_color_train.npy"))
    X_test = np.load(os.path.join(data_dir, "comic_input_grayscale_test.npy"))
    y_test = np.load(os.path.join(data_dir, "comic_output_color_test.npy"))
    
    # Convert data to HSV color space
    X_train, y_train = prepare_hsv_data(y_train)
    X_test, y_test = prepare_hsv_data(y_test)
    
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Create and compile model
    print("\nBuilding model...")
    model = build_unet_hsv((256, 256, 1))
    
    # Use Adam optimizer with initial learning rate
    optimizer = Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss=hsv_loss,
        metrics=['mae', 'mse']
    )
    
    # Get callbacks
    callbacks = get_callbacks(
        model_dir=models_dir,
        log_dir=log_dir,
        validation_data=(X_test, y_test)
    )
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!") 