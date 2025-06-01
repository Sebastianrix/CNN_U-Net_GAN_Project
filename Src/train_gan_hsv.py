import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
from tqdm import tqdm
from colorization_gan_v3 import ColorizeGAN, ColorSpace
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} Physical GPUs")
    except RuntimeError as e:
        print(e)

# Training parameters
BATCH_SIZE = 8
EPOCHS = 100
IMG_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE

def load_data(data_dir):
    """Load HSV and grayscale data"""
    print("Loading training data...")
    hsv_train = np.load(os.path.join(data_dir, 'comic_output_hsv_train.npy'))
    gray_train = np.load(os.path.join(data_dir, 'comic_input_grayscale_train.npy'))
    
    print("Loading validation data...")
    hsv_val = np.load(os.path.join(data_dir, 'comic_output_hsv_val.npy'))
    gray_val = np.load(os.path.join(data_dir, 'comic_input_grayscale_val.npy'))
    
    # Print data statistics
    print(f"\nTraining set:")
    print(f"HSV shape: {hsv_train.shape}, range: [{hsv_train.min()}, {hsv_train.max()}]")
    print(f"Grayscale shape: {gray_train.shape}, range: [{gray_train.min()}, {gray_train.max()}]")
    
    print(f"\nValidation set:")
    print(f"HSV shape: {hsv_val.shape}, range: [{hsv_val.min()}, {hsv_val.max()}]")
    print(f"Grayscale shape: {gray_val.shape}, range: [{gray_val.min()}, {gray_val.max()}]")
    
    return (gray_train, hsv_train), (gray_val, hsv_val)

def create_dataset(gray_images, hsv_images, batch_size, shuffle=True):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((gray_images, hsv_images))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def save_samples(epoch, gan, val_gray, val_hsv, samples_dir):
    """Save sample colorizations"""
    n_samples = min(5, len(val_gray))
    
    plt.figure(figsize=(15, 3))
    for i in range(n_samples):
        # Get sample image
        gray_img = val_gray[i:i+1]
        real_hsv = val_hsv[i:i+1]
        
        # Generate fake colors
        fake_hsv = gan.generator(gray_img, training=False)
        
        # Create full HSV images (adding V channel from grayscale)
        gray_v = gray_img * 255.0
        
        real_full_hsv = np.concatenate([
            real_hsv[..., 0:1],  # H
            real_hsv[..., 1:2],  # S
            gray_v               # V
        ], axis=-1)
        
        fake_full_hsv = np.concatenate([
            fake_hsv[..., 0:1],  # H
            fake_hsv[..., 1:2],  # S
            gray_v               # V
        ], axis=-1)
        
        # Convert to RGB for display
        real_rgb = cv2.cvtColor(real_full_hsv[0].numpy().astype(np.uint8), cv2.COLOR_HSV2RGB)
        fake_rgb = cv2.cvtColor(fake_full_hsv[0].numpy().astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Plot
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(gray_img[0, ..., 0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Input')
        
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.imshow(real_rgb)
        plt.axis('off')
        if i == 0:
            plt.title('Real')
        
        plt.subplot(3, n_samples, i + 1 + 2*n_samples)
        plt.imshow(fake_rgb)
        plt.axis('off')
        if i == 0:
            plt.title('Generated')
    
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch}.png'))
    plt.close()

def main():
    # Setup directories
    base_dir = 'Data/prepared_data/hsv'
    log_dir = 'logs/hsv_gan'
    checkpoint_dir = 'checkpoints/hsv_gan'
    samples_dir = 'samples/hsv_gan'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Load data
    (train_gray, train_hsv), (val_gray, val_hsv) = load_data(base_dir)
    
    # Create datasets
    train_dataset = create_dataset(train_gray, train_hsv, BATCH_SIZE)
    val_dataset = create_dataset(val_gray, val_hsv, BATCH_SIZE, shuffle=False)
    
    # Create GAN model
    print("\nInitializing GAN model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        gan = ColorizeGAN(img_size=IMG_SIZE, color_space=ColorSpace.HSV)
        gan.compile()
    
    # Setup TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training
        epoch_losses = []
        progress_bar = tqdm(train_dataset, desc=f"Training")
        for batch_gray, batch_hsv in progress_bar:
            losses = gan.train_step(batch_gray, batch_hsv)
            epoch_losses.append(losses)
            
            # Update progress bar
            avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) 
                        for k in losses.keys()}
            progress_bar.set_postfix(avg_losses)
        
        # Log losses
        with train_summary_writer.as_default():
            for k, v in avg_losses.items():
                tf.summary.scalar(k, v, step=epoch)
        
        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_samples(epoch + 1, gan, val_gray, val_hsv, samples_dir)
        
        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            os.makedirs(checkpoint_path, exist_ok=True)
            gan.generator.save_weights(os.path.join(checkpoint_path, 'generator.h5'))
            gan.discriminator.save_weights(os.path.join(checkpoint_path, 'discriminator.h5'))
            print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    main() 