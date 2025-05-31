import tensorflow as tf
import numpy as np
import os
import datetime
import argparse
from colorization_gan_v2 import ColorizeGAN, ColorSpace

def prepare_data(color_images, color_space):
    """Prepare data based on color space"""
    if color_space == ColorSpace.RGB:
        return color_images
    elif color_space == ColorSpace.LAB:
        from unet_model_lab import rgb_to_lab
        lab_images = rgb_to_lab(color_images)
        return lab_images[..., 1:]  # Return a,b channels
    else:  # HSV
        from unet_model_hsv import rgb_to_hsv
        hsv_images = rgb_to_hsv(color_images)
        return hsv_images[..., :2]  # Return H,S channels

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Colorization GAN')
    parser.add_argument('--color-space', type=str, default='hsv', choices=['rgb', 'lab', 'hsv'],
                      help='Color space to use for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs')
    args = parser.parse_args()

    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    # Set random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Setup directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"gan_{args.color_space}_{timestamp}"
    
    models_dir = os.path.join('Models', model_name)
    log_dir = os.path.join('logs', model_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    print("\nLoading training data...")
    X_train = np.load("Data/prepared_data/comic_input_grayscale_train.npy")
    y_train = np.load("Data/prepared_data/comic_output_color_train.npy")
    X_test = np.load("Data/prepared_data/comic_input_grayscale_test.npy")
    y_test = np.load("Data/prepared_data/comic_output_color_test.npy")
    
    # Normalize input images to [-1, 1]
    X_train = (X_train / 127.5) - 1
    X_test = (X_test / 127.5) - 1
    
    # Convert to appropriate color space
    color_space = ColorSpace(args.color_space)
    y_train = prepare_data(y_train, color_space)
    y_test = prepare_data(y_test, color_space)
    
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Create GAN model
    print("\nBuilding GAN model...")
    gan = ColorizeGAN(img_size=256, color_space=color_space)
    gan.compile()
    
    # Setup TensorBoard
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Training loop
    print("\nStarting training...")
    num_batches = len(X_train) // args.batch_size
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        total_disc_loss = 0
        total_gen_loss = 0
        
        for batch in range(num_batches):
            start_idx = batch * args.batch_size
            end_idx = start_idx + args.batch_size
            
            # Get batch data
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Train on batch
            losses = gan.train_step(X_batch, y_batch)
            
            total_disc_loss += losses['disc_loss']
            total_gen_loss += losses['total_gen_loss']
            
            # Print progress
            if batch % 10 == 0:
                print(f"Batch {batch+1}/{num_batches} - "
                      f"D_loss: {losses['disc_loss']:.4f}, "
                      f"G_loss: {losses['total_gen_loss']:.4f}")
        
        # Calculate epoch averages
        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches
        
        # Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('discriminator_loss', avg_disc_loss, step=epoch)
            tf.summary.scalar('generator_loss', avg_gen_loss, step=epoch)
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            gan.generator.save(os.path.join(models_dir, f'generator_epoch_{epoch+1}.keras'))
            gan.discriminator.save(os.path.join(models_dir, f'discriminator_epoch_{epoch+1}.keras'))
    
    # Save final model
    gan.generator.save(os.path.join(models_dir, 'generator_final.keras'))
    gan.discriminator.save(os.path.join(models_dir, 'discriminator_final.keras'))
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 