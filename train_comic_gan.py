import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from Src.comic_colorization_gan import ComicColorizeGAN

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Training parameters
BATCH_SIZE = 8
EPOCHS = 100
IMG_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE  # Optimize dataset performance

def load_and_preprocess_data(data_dir):
    """Load and preprocess data for LAB color space"""
    print("Loading data...")
    
    # Load numpy arrays
    gray_images = np.load(os.path.join(data_dir, 'comic_input_grayscale_train.npy'))
    color_images = np.load(os.path.join(data_dir, 'comic_output_color_train.npy'))
    
    print(f"Loaded {len(gray_images)} training images")
    print(f"Grayscale shape: {gray_images.shape}")
    print(f"Color shape: {color_images.shape}")
    
    # Convert color images to LAB
    print("\nConverting to LAB color space...")
    lab_images = []
    for img in tqdm(color_images, desc="Converting to LAB"):
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Keep only a,b channels
        lab_images.append(lab[..., 1:])
    
    lab_images = np.array(lab_images)
    
    # Normalize data
    gray_images = gray_images.astype('float32') / 255.0  # [0, 1]
    lab_images = lab_images.astype('float32')  # Already in correct range
    
    print("\nData preprocessing complete!")
    print(f"Grayscale range: [{np.min(gray_images)}, {np.max(gray_images)}]")
    print(f"LAB range: [{np.min(lab_images)}, {np.max(lab_images)}]")
    
    return gray_images, lab_images

@tf.function
def create_dataset(gray_images, lab_images, batch_size):
    """Create TensorFlow dataset with prefetch and shuffle"""
    # Convert numpy arrays to tensors
    gray_tensor = tf.convert_to_tensor(gray_images, dtype=tf.float32)
    lab_tensor = tf.convert_to_tensor(lab_images, dtype=tf.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((gray_tensor, lab_tensor))
    dataset = dataset.shuffle(10000)  # Larger buffer for better shuffling
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)  # Prefetch next batch while processing current one
    return dataset

def save_sample_images(epoch, generator, val_gray, val_color, save_dir='samples'):
    """Save sample colorization results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select 4 random validation images
    indices = np.random.randint(0, len(val_gray), 4)
    test_gray = val_gray[indices]
    test_color = val_color[indices]
    
    # Generate colorizations
    generated_ab = generator.predict(test_gray, verbose=0)  # Disable progress bar
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    for i in range(4):
        # Create full LAB images
        generated_lab = np.concatenate([
            test_gray[i] * 100.0,  # L channel
            generated_ab[i]        # a,b channels
        ], axis=-1)
        
        original_lab = np.concatenate([
            test_gray[i] * 100.0,  # L channel
            test_color[i]          # a,b channels
        ], axis=-1)
        
        # Convert to RGB
        generated_rgb = cv2.cvtColor(generated_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        original_rgb = cv2.cvtColor(original_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        
        # Plot results
        plt.subplot(3, 4, i + 1)
        plt.imshow(test_gray[i, ..., 0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Input (Grayscale)')
        
        plt.subplot(3, 4, i + 5)
        plt.imshow(original_rgb)
        plt.axis('off')
        if i == 0:
            plt.title('Ground Truth')
        
        plt.subplot(3, 4, i + 9)
        plt.imshow(generated_rgb)
        plt.axis('off')
        if i == 0:
            plt.title('Generated')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png'))
    plt.close()

def main():
    # Load and preprocess data
    train_gray, train_lab = load_and_preprocess_data('Data/prepared_data')
    
    # Create validation split
    val_size = len(train_gray) // 10
    val_gray = train_gray[-val_size:]
    val_lab = train_lab[-val_size:]
    train_gray = train_gray[:-val_size]
    train_lab = train_lab[:-val_size]
    
    # Create datasets
    train_dataset = create_dataset(train_gray, train_lab, BATCH_SIZE)
    val_dataset = create_dataset(val_gray, val_lab, BATCH_SIZE)
    
    # Create GAN model
    print("\nInitializing GAN model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        gan = ComicColorizeGAN(img_size=IMG_SIZE)
        gan.compile()
    
    # Create directories for checkpoints and samples
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training metrics
        metrics = {
            'disc_loss': [],
            'gen_loss': [],
            'l1_loss': [],
            'gp': []
        }
        
        # Train on batches
        for batch_gray, batch_lab in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # Train step
            batch_metrics = gan.train_step(batch_gray, batch_lab)
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k].append(float(v))
        
        # Print epoch metrics
        print(f"\nEpoch {epoch+1} metrics:")
        for k, v in metrics.items():
            avg_value = np.mean(v)
            print(f"{k}: {avg_value:.4f}")
        
        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_sample_images(epoch + 1, gan.generator, val_gray, val_lab)
        
        # Save model weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f'checkpoints/epoch_{epoch+1:03d}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            gan.generator.save_weights(os.path.join(checkpoint_dir, 'generator.h5'))
            gan.discriminator.save_weights(os.path.join(checkpoint_dir, 'discriminator.h5'))
            print(f"\nSaved model weights at epoch {epoch+1}")

if __name__ == "__main__":
    main() 