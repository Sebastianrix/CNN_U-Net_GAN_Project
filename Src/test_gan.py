import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from colorization_gan_v2 import ColorSpace, ColorizeGAN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_model(model_path, color_space):
    """Load the generator model"""
    try:
        print(f"Loading model from: {model_path}")
        print(f"Using color space: {color_space.value}")
        
        # Enable unsafe deserialization for Lambda layers
        tf.keras.utils.disable_interactive_logging()
        tf.keras.config.enable_unsafe_deserialization()
        
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

def convert_to_rgb(generated_output, grayscale_input, color_space):
    """Convert model output to RGB based on color space"""
    try:
        if color_space == ColorSpace.RGB:
            return generated_output
        elif color_space == ColorSpace.LAB:
            from unet_model_lab import lab_to_rgb
            # Combine L channel from input with predicted ab channels
            lab_image = np.concatenate([grayscale_input, generated_output], axis=-1)
            return lab_to_rgb(lab_image)
        else:  # HSV
            from unet_model_hsv import hsv_to_rgb
            # Print shapes for debugging
            print(f"Generated output shape: {generated_output.shape}")
            print(f"Grayscale input shape: {grayscale_input.shape}")
            
            # Convert grayscale from [-1, 1] to [0, 1] for V channel
            v_channel = (grayscale_input + 1) / 2
            
            # Convert H from [0, 179] to [0, 360] and S from [0, 255] to [0, 1]
            h_channel = generated_output[..., 0:1] * (360/179)  # Scale H to [0, 360]
            s_channel = generated_output[..., 1:2] / 255.0      # Scale S to [0, 1]
            
            # Combine H, S, V channels
            hsv_image = np.concatenate([h_channel, s_channel, v_channel], axis=-1)
            print(f"Combined HSV shape: {hsv_image.shape}")
            print(f"HSV ranges - H: [{hsv_image[..., 0].min():.2f}, {hsv_image[..., 0].max():.2f}], "
                  f"S: [{hsv_image[..., 1].min():.2f}, {hsv_image[..., 1].max():.2f}], "
                  f"V: [{hsv_image[..., 2].min():.2f}, {hsv_image[..., 2].max():.2f}]")
            
            # Convert to RGB
            rgb_image = hsv_to_rgb(hsv_image)
            return rgb_image
    except Exception as e:
        print(f"Error in convert_to_rgb: {str(e)}")
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
        for i, pred in enumerate(predictions):
            try:
                rgb_pred = convert_to_rgb(pred[np.newaxis, ...], X_test[i:i+1], color_space)
                rgb_predictions.append(rgb_pred)
                if i == 0:  # Print first conversion details
                    print(f"First prediction conversion shape: {rgb_pred.shape}")
            except Exception as e:
                print(f"Error converting prediction {i}: {str(e)}")
                continue
        
        rgb_predictions = np.array(rgb_predictions)
        rgb_predictions = np.squeeze(rgb_predictions)
        print(f"Final RGB predictions shape: {rgb_predictions.shape}")
        
        # Ensure both arrays are in the same format
        y_test = y_test.astype(np.float32)
        rgb_predictions = rgb_predictions.astype(np.float32)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        psnr_scores = []
        ssim_scores = []
        
        for i in range(len(y_test)):
            try:
                # Calculate PSNR
                psnr_score = psnr(y_test[i], rgb_predictions[i], data_range=1.0)
                
                # Calculate SSIM for each channel separately and take the mean
                ssim_score = np.mean([
                    ssim(y_test[i, ..., j], rgb_predictions[i, ..., j],
                         data_range=1.0, win_size=3)  # Using smaller window size
                    for j in range(3)  # RGB channels
                ])
                
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
                # Convert grayscale to RGB by repeating the channel
                input_rgb = np.repeat(X_test[i], 3, axis=-1)
                # Rescale input from [-1, 1] to [0, 1]
                input_rgb = (input_rgb + 1) / 2
                
                # Create comparison image
                comparison = np.hstack([
                    input_rgb,
                    rgb_predictions[i],
                    y_test[i]
                ])
                
                # Convert to BGR for OpenCV
                comparison = cv2.cvtColor((comparison * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Add text labels
                height = comparison.shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 255, 255)
                thickness = 1
                
                cv2.putText(comparison, 'Input', (10, height - 10), font, font_scale, color, thickness)
                cv2.putText(comparison, 'Generated', (comparison.shape[1]//3 + 10, height - 10), 
                           font, font_scale, color, thickness)
                cv2.putText(comparison, 'Ground Truth', (2*comparison.shape[1]//3 + 10, height - 10), 
                           font, font_scale, color, thickness)
                
                # Save image
                cv2.imwrite(os.path.join(output_dir, f'sample_{i+1}.png'), comparison)
                
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
        
        print(f"\nResults saved to {output_dir}")
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Test Colorization GAN')
        parser.add_argument('--model-path', type=str, required=True,
                          help='Path to the generator model')
        parser.add_argument('--color-space', type=str, required=True,
                          choices=['rgb', 'lab', 'hsv'],
                          help='Color space used by the model')
        parser.add_argument('--output-dir', type=str, default='test_results',
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
        
        print(f"Test data loaded - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Normalize input images to [-1, 1]
        X_test = (X_test / 127.5) - 1
        
        # Load model
        print("\nLoading model...")
        color_space = ColorSpace(args.color_space)
        generator = load_model(args.model_path, color_space)
        
        # Create output directory with timestamp
        timestamp = tf.timestamp().numpy().astype(int)
        output_dir = os.path.join(args.output_dir, f"{color_space.value}_{timestamp}")
        
        # Evaluate model
        evaluate_model(generator, X_test, y_test, color_space, output_dir)
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 