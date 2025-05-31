import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from colorization_gan_v3 import ColorizeGAN, ColorSpace
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_model(model_path, color_space):
    """Load the generator model"""
    try:
        print(f"Loading model from: {model_path}")
        print(f"Using color space: {color_space.value}")
        
        # Create a new GAN instance to get the generator architecture
        print("Creating GAN instance...")
        gan = ColorizeGAN(img_size=256, color_space=color_space)
        generator = gan.generator
        
        # Load weights from the saved model
        print("Loading weights...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        generator.load_weights(model_path)
        print("Model loaded successfully!")
        return generator
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def evaluate_model(generator, X_test, y_test, color_space, output_dir):
    """Evaluate model performance and save sample images"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Print shapes for debugging
        print(f"\nInput shape: {X_test.shape}")
        print(f"Ground truth shape: {y_test.shape}")
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = generator.predict(X_test, verbose=1)
        print(f"Predictions shape: {predictions.shape}")
        
        # Convert predictions to RGB
        print("\nConverting to RGB...")
        rgb_predictions = []
        rgb_ground_truth = []
        
        for i, (pred, gt) in enumerate(zip(predictions, y_test)):
            try:
                if color_space == ColorSpace.HSV:
                    from unet_model_hsv import hsv_to_rgb
                    # Combine predicted H,S with input V
                    rgb_pred = hsv_to_rgb(np.concatenate([
                        pred, X_test[i]], axis=-1))
                    rgb_gt = hsv_to_rgb(np.concatenate([
                        gt, X_test[i]], axis=-1))
                elif color_space == ColorSpace.LAB:
                    from unet_model_lab import lab_to_rgb
                    # Combine L with predicted ab
                    rgb_pred = lab_to_rgb(np.concatenate([
                        X_test[i], pred], axis=-1))
                    rgb_gt = lab_to_rgb(np.concatenate([
                        X_test[i], gt], axis=-1))
                else:  # RGB
                    rgb_pred = (pred + 1) * 127.5
                    rgb_gt = (gt + 1) * 127.5
                
                rgb_predictions.append(rgb_pred)
                rgb_ground_truth.append(rgb_gt)
                
                if i == 0:  # Print first conversion details
                    print(f"First prediction ranges - RGB: [{rgb_pred.min():.2f}, {rgb_pred.max():.2f}]")
            except Exception as e:
                print(f"Error converting prediction {i}: {str(e)}")
                continue
        
        rgb_predictions = np.array(rgb_predictions)
        rgb_ground_truth = np.array(rgb_ground_truth)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        psnr_scores = []
        ssim_scores = []
        
        for i in range(len(rgb_ground_truth)):
            try:
                # Ensure images are in range [0, 1] for metrics
                pred_norm = rgb_predictions[i] / 255.0
                gt_norm = rgb_ground_truth[i] / 255.0
                
                # Calculate PSNR
                psnr_score = psnr(gt_norm, pred_norm, data_range=1.0)
                
                # Calculate SSIM
                ssim_score = ssim(gt_norm, pred_norm, 
                                data_range=1.0, 
                                multichannel=True)
                
                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)
                
                if i == 0:  # Print first metric calculation details
                    print(f"First image PSNR: {psnr_score:.2f}")
                    print(f"First image SSIM: {ssim_score:.4f}")
            except Exception as e:
                print(f"Error calculating metrics for image {i}: {str(e)}")
                continue
        
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        
        print(f"\nAverage PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save sample images
        print("\nSaving sample images...")
        num_samples = min(10, len(X_test))
        
        for i in range(num_samples):
            try:
                # Create comparison image
                fig = plt.figure(figsize=(15, 5))
                
                # Input grayscale
                plt.subplot(1, 3, 1)
                plt.imshow(X_test[i].squeeze(), cmap='gray')
                plt.title('Input')
                plt.axis('off')
                
                # Generated color
                plt.subplot(1, 3, 2)
                plt.imshow(rgb_predictions[i].astype(np.uint8))
                plt.title(f'Generated (PSNR: {psnr_scores[i]:.2f})')
                plt.axis('off')
                
                # Ground truth
                plt.subplot(1, 3, 3)
                plt.imshow(rgb_ground_truth[i].astype(np.uint8))
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.suptitle(f'Sample {i+1}')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
                plt.close()
                
                if i == 0:  # Print first save details
                    print(f"First sample saved: {os.path.join(output_dir, 'sample_1.png')}")
            except Exception as e:
                print(f"Error saving sample {i}: {str(e)}")
                continue
        
        # Save metrics to file
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Color Space: {color_space.value}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"\nPer-image metrics:\n")
            for i in range(len(psnr_scores)):
                f.write(f"Image {i+1} - PSNR: {psnr_scores[i]:.2f}, SSIM: {ssim_scores[i]:.4f}\n")
        
        print(f"\nResults saved to {output_dir}")
        return avg_psnr, avg_ssim
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Test Improved Colorization GAN')
        parser.add_argument('--model-path', type=str, required=True,
                          help='Path to the generator model')
        parser.add_argument('--color-space', type=str, required=True,
                          choices=['rgb', 'lab', 'hsv'],
                          help='Color space used by the model')
        parser.add_argument('--output-dir', type=str, default='test_results_v3',
                          help='Directory to save test results')
        args = parser.parse_args()
        
        # Print arguments
        print("\nArguments:")
        print(f"Model path: {args.model_path}")
        print(f"Color space: {args.color_space}")
        print(f"Output directory: {args.output_dir}")
        
        # Set memory growth for GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"\nFound {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Enabled memory growth for {device}")
        else:
            print("\nNo GPU devices found, using CPU")
        
        # Load test data
        print("\nLoading test data...")
        X_test = np.load("Data/prepared_data/comic_input_grayscale_test.npy")
        y_test = np.load("Data/prepared_data/comic_output_color_test.npy")
        
        # Normalize input images to [-1, 1]
        X_test = (X_test / 127.5) - 1.0
        
        # Convert ground truth based on color space
        color_space = ColorSpace(args.color_space)
        if color_space == ColorSpace.HSV:
            from unet_model_hsv import rgb_to_hsv
            y_test = rgb_to_hsv(y_test)[..., :2]  # Keep only H and S channels
        elif color_space == ColorSpace.LAB:
            from unet_model_lab import rgb_to_lab
            y_test = rgb_to_lab(y_test)[..., 1:]  # Keep only a and b channels
        else:  # RGB
            y_test = y_test / 127.5 - 1.0
        
        print(f"Test data loaded - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Load model
        print("\nLoading model...")
        generator = load_model(args.model_path, color_space)
        
        # Create output directory with timestamp
        timestamp = tf.timestamp().numpy().astype(int)
        output_dir = os.path.join(args.output_dir, f"{color_space.value}_{timestamp}")
        
        # Evaluate model
        psnr_score, ssim_score = evaluate_model(
            generator, X_test, y_test, color_space, output_dir)
        
        print("\nEvaluation complete!")
        print(f"Final PSNR: {psnr_score:.2f}")
        print(f"Final SSIM: {ssim_score:.4f}")
        
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 