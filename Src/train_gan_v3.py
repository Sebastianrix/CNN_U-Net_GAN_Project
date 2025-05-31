import os
import numpy as np
import tensorflow as tf
from colorization_gan_v3 import ColorizeGAN, ColorSpace
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

# Configure GPU for optimal performance
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
        # Set TensorFlow to use mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Configuration
BATCH_SIZE = 8  # Restored for GPU training
EPOCHS = 100
IMG_SIZE = 256  # Restored original size
COLOR_SPACE = ColorSpace.HSV
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

def load_data(data_dir):
    """Load and preprocess images"""
    color_images = np.load(os.path.join(data_dir, 'color.npy'))
    gray_images = np.load(os.path.join(data_dir, 'gray.npy'))
    
    # Normalize grayscale to [0, 1]
    gray_images = gray_images.astype('float32') / 255.0
    
    # For HSV, keep color values in their original range
    if COLOR_SPACE == ColorSpace.HSV:
        color_images = color_images.astype('float32')
    else:
        # For RGB/LAB, normalize to [-1, 1] or [0, 1]
        color_images = color_images.astype('float32') / 127.5 - 1.0
    
    return gray_images, color_images

def create_dataset(images, labels, batch_size):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset

def save_samples(epoch, generator, val_gray, val_color, save_dir='samples'):
    """Save sample colorization results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate predictions
    predictions = generator.predict(val_gray[:4])
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    for i in range(4):
        # Original grayscale
        plt.subplot(3, 4, i + 1)
        plt.imshow(val_gray[i, ..., 0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Input (Grayscale)')
        
        # Ground truth
        plt.subplot(3, 4, i + 5)
        if COLOR_SPACE == ColorSpace.HSV:
            # Convert HSV to RGB for display
            hsv = np.concatenate([
                val_color[i],
                val_gray[i, ..., 0:1] * 255.0
            ], axis=-1)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
        else:
            plt.imshow(val_color[i] * 0.5 + 0.5)
        plt.axis('off')
        if i == 0:
            plt.title('Ground Truth')
        
        # Generated
        plt.subplot(3, 4, i + 9)
        if COLOR_SPACE == ColorSpace.HSV:
            # Convert HSV to RGB for display
            hsv = np.concatenate([
                predictions[i],
                val_gray[i, ..., 0:1] * 255.0
            ], axis=-1)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
        else:
            plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
        if i == 0:
            plt.title('Generated')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png'))
    plt.close()

def main():
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Enable memory cleanup
    gc.collect()
    
    # Load data
    print("Loading training data...")
    train_gray, train_color = load_data('Data/prepared_data/train')
    
    print("Loading validation data...")
    val_gray, val_color = load_data('Data/prepared_data/val')
    
    print(f"Training data shapes: X={train_gray.shape}, y={train_color.shape}")
    print(f"Validation data shapes: X={val_gray.shape}, y={val_color.shape}")
    
    # Create datasets with prefetch to optimize memory usage
    train_dataset = create_dataset(train_gray, train_color, BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = create_dataset(val_gray, val_color, BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Clear unneeded variables
    del train_gray, train_color, val_gray, val_color
    gc.collect()
    
    # Initialize GAN
    gan = ColorizeGAN(img_size=IMG_SIZE, color_space=COLOR_SPACE)
    gan.compile()
    
    # Setup logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(LOG_DIR, current_time, 'train')
    val_log_dir = os.path.join(LOG_DIR, current_time, 'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # Training loop
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Training
        train_losses = {
            'disc_loss': [], 'gen_loss': [], 'gp_loss': [],
            'fm_loss': [], 'perc_loss': [], 'hist_loss': []
        }
        
        for batch_gray, batch_color in train_dataset:
            losses = gan.train_step(batch_gray, batch_color)
            for k, v in losses.items():
                train_losses[k].append(float(v))
            
            # Clear memory after each batch
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Log training metrics
        with train_summary_writer.as_default():
            for k, v in train_losses.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)
        
        # Validation
        val_losses = {
            'disc_loss': [], 'gen_loss': [], 'gp_loss': [],
            'fm_loss': [], 'perc_loss': [], 'hist_loss': []
        }
        
        for batch_gray, batch_color in val_dataset:
            losses = gan.train_step(batch_gray, batch_color)
            for k, v in losses.items():
                val_losses[k].append(float(v))
            
            # Clear memory after each batch
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Log validation metrics
        with val_summary_writer.as_default():
            for k, v in val_losses.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)
        
        # Print progress
        print(f"Train - Gen: {np.mean(train_losses['gen_loss']):.4f}, "
              f"Disc: {np.mean(train_losses['disc_loss']):.4f}")
        print(f"Val - Gen: {np.mean(val_losses['gen_loss']):.4f}, "
              f"Disc: {np.mean(val_losses['disc_loss']):.4f}")
        
        # Save samples
        if (epoch + 1) % 5 == 0:
            save_samples(epoch + 1, gan.generator, val_gray, val_color)
            gc.collect()  # Clean up after saving samples
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            gan.generator.save(os.path.join(
                CHECKPOINT_DIR, f'generator_epoch_{epoch+1}.h5'
            ))
            gan.discriminator.save(os.path.join(
                CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1}.h5'
            ))
            gc.collect()  # Clean up after saving models

if __name__ == "__main__":
    main() 