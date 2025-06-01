import os
import numpy as np
import tensorflow as tf
from colorization_gan_v3 import ColorizeGAN, ColorSpace
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import platform

# Detailed GPU detection and setup
print("Python version:", platform.python_version())
print("TensorFlow version:", tf.__version__)
print("Operating System:", platform.system(), platform.release())

# Configure GPU for optimal performance
physical_devices = tf.config.list_physical_devices('GPU')
print("\nGPU Devices found:", physical_devices)

if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"\nEnabled memory growth for {device}")
            
            # Get device details
            device_details = tf.config.experimental.get_device_details(device)
            print("Device details:", device_details)
        
        # Set mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("\nMixed precision policy:", policy.name)
        
        # Test GPU availability with a simple operation
        with tf.device('/GPU:0'):
            print("\nTesting GPU with a simple operation...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("GPU test successful - Matrix multiplication shape:", c.shape)
            
    except RuntimeError as e:
        print(f"\nError configuring GPU: {e}")
else:
    print("\nNo GPU devices found. Running on CPU.")
    print("Available devices:", tf.config.list_physical_devices())

# Configuration
BATCH_SIZE = 8
EPOCHS = 100
IMG_SIZE = 256
COLOR_SPACE = ColorSpace.HSV
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_data(data_dir, data_type='train'):
    """Load and preprocess images"""
    color_images = np.load(os.path.join(data_dir, f'comic_output_color_{data_type}.npy'))
    gray_images = np.load(os.path.join(data_dir, f'comic_input_grayscale_{data_type}.npy'))
    
    # Normalize grayscale to [0, 1]
    gray_images = gray_images.astype('float32') / 255.0
    
    # Convert color images to HSV if needed
    if COLOR_SPACE == ColorSpace.HSV:
        # Process each image individually for RGB to HSV conversion
        hsv_images = []
        for img in color_images:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv_images.append(hsv[..., :2])  # Keep only H and S channels
        color_images = np.array(hsv_images, dtype='float32')
    else:
        # For RGB/LAB, normalize to [-1, 1]
        color_images = color_images.astype('float32') / 127.5 - 1.0
    
    return gray_images, color_images

def create_dataset(images, labels, batch_size):
    """Create TensorFlow dataset with prefetching and caching"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch while processing current batch
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
            # Create full HSV image (H, S from ground truth, V from grayscale)
            hsv = np.zeros((val_gray[i].shape[0], val_gray[i].shape[1], 3), dtype=np.uint8)
            hsv[..., :2] = val_color[i].astype(np.uint8)  # H and S channels
            hsv[..., 2] = (val_gray[i, ..., 0] * 255.0).astype(np.uint8)  # V channel
            # Convert to RGB for display
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
        else:
            plt.imshow(val_color[i] * 0.5 + 0.5)
        plt.axis('off')
        if i == 0:
            plt.title('Ground Truth')
        
        # Generated
        plt.subplot(3, 4, i + 9)
        if COLOR_SPACE == ColorSpace.HSV:
            # Create full HSV image (H, S from predictions, V from grayscale)
            hsv = np.zeros((val_gray[i].shape[0], val_gray[i].shape[1], 3), dtype=np.uint8)
            hsv[..., :2] = predictions[i].astype(np.uint8)  # H and S channels
            hsv[..., 2] = (val_gray[i, ..., 0] * 255.0).astype(np.uint8)  # V channel
            # Convert to RGB for display
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
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
    # Load data
    print("Loading training data...")
    train_gray, train_color = load_data('Data/prepared_data')
    
    print("Loading validation data...")
    val_gray, val_color = load_data('Data/prepared_data')
    
    print(f"Training data shapes: X={train_gray.shape}, y={train_color.shape}")
    print(f"Validation data shapes: X={val_gray.shape}, y={val_color.shape}")
    print(f"Training data range: X=[{np.min(train_gray)}, {np.max(train_gray)}], y=[{np.min(train_color)}, {np.max(train_color)}]")
    print(f"Training data types: X={train_gray.dtype}, y={train_color.dtype}")
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    train_dataset = create_dataset(train_gray, train_color, BATCH_SIZE)
    val_dataset = create_dataset(val_gray, val_color, BATCH_SIZE)
    
    # Create GAN model
    gan = ColorizeGAN(img_size=IMG_SIZE, color_space=COLOR_SPACE)
    gan.compile()
    
    # Training loop
    epochs = tqdm(range(EPOCHS), desc='Epochs')
    for epoch in epochs:
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Training
        batch_losses = []
        batch_iterator = tqdm(enumerate(train_dataset), desc='Batches', total=len(train_gray)//BATCH_SIZE)
        
        for batch_idx, (batch_gray, batch_color) in batch_iterator:
            try:
                # Train step with error handling
                try:
                    with tf.device('/GPU:0'):  # Explicitly use GPU
                        losses = gan.train_step(batch_gray, batch_color)
                    batch_losses.append(losses)
                    
                    # Update progress bar with losses
                    avg_losses = {k: float(np.mean([loss[k] for loss in batch_losses])) 
                                for k in losses.keys()}
                    batch_iterator.set_postfix(avg_losses)
                    
                except tf.errors.ResourceExhaustedError as e:
                    print(f"\nGPU memory exhausted. Error: {e}")
                    print("Attempting to free memory...")
                    gc.collect()
                    tf.keras.backend.clear_session()
                    continue
                    
                except Exception as e:
                    print(f"\nError during training step: {str(e)}")
                    print(f"Batch info:")
                    print(f"- Shapes: X={batch_gray.shape}, y={batch_color.shape}")
                    print(f"- Types: X={batch_gray.dtype}, y={batch_color.dtype}")
                    print(f"- Ranges: X=[{np.min(batch_gray)}, {np.max(batch_gray)}], "
                          f"y=[{np.min(batch_color)}, {np.max(batch_color)}]")
                    raise e
                
            except Exception as e:
                print(f"\nError during batch processing: {str(e)}")
                raise e
        
        # Validation and sample generation
        if (epoch + 1) % 5 == 0:
            try:
                print("\nGenerating samples...")
                save_samples(epoch + 1, gan.generator, val_gray, val_color)
                gc.collect()
            except Exception as e:
                print(f"\nError during sample generation: {str(e)}")
                print("Continuing training...")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            try:
                print("\nSaving checkpoints...")
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}')
                os.makedirs(checkpoint_path, exist_ok=True)
                gan.generator.save(os.path.join(checkpoint_path, 'generator.h5'))
                gan.discriminator.save(os.path.join(checkpoint_path, 'discriminator.h5'))
                gc.collect()
            except Exception as e:
                print(f"\nError saving checkpoints: {str(e)}")
                print("Continuing training...")
        
        # Memory cleanup at end of epoch
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        raise e 