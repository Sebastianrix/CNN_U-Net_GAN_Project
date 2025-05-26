import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import sys

# Add Src directory to path for importing our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet_model_lab import build_unet_lab, lab_loss, rgb_to_lab

def create_data_augmentation():
    """Create a sequential model for data augmentation"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.GaussianNoise(0.01)
    ])

def prepare_lab_data(rgb_images):
    """Convert RGB images to LAB color space and separate channels"""
    print("Converting images to LAB space...")
    lab_images = np.array([rgb_to_lab(img) for img in rgb_images])
    
    # Extract L and ab channels
    l_channel = lab_images[:, :, :, 0:1]
    ab_channels = lab_images[:, :, :, 1:]
    
    print(f"L channel range: [{l_channel.min():.2f}, {l_channel.max():.2f}]")
    print(f"A channel range: [{ab_channels[:,:,:,0].min():.2f}, {ab_channels[:,:,:,0].max():.2f}]")
    print(f"B channel range: [{ab_channels[:,:,:,1].min():.2f}, {ab_channels[:,:,:,1].max():.2f}]")
    
    # Normalize L channel to [-1, 1]
    l_channel = (l_channel - 50.0) / 50.0
    
    return l_channel, ab_channels

if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load and prepare data
    print("Loading training data...")
    data_dir = os.path.join(project_root, 'Data', 'prepared_data')
    X_train = np.load(os.path.join(data_dir, "comic_input_grayscale_train.npy"))
    y_train = np.load(os.path.join(data_dir, "comic_output_color_train.npy"))
    
    # Convert and prepare data
    X_train, y_train = prepare_lab_data(y_train)
    
    print("\nFinal training data shapes and ranges:")
    print(f"Input shape: {X_train.shape}, range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"Target shape: {y_train.shape}, range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Create and compile model
    print("\nBuilding model...")
    model = build_unet_lab((256, 256, 1))
    
    # Use simple Adam optimizer with fixed learning rate
    optimizer = Adam(learning_rate=2e-4)
    
    model.compile(
        optimizer=optimizer,
        loss=lab_loss,
        metrics=['mae']
    )
    
    # Setup callbacks
    models_dir = os.path.join(project_root, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(models_dir, "best_unet_lab_v8.keras"),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=200,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!") 