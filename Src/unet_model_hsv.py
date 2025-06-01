import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import io
import matplotlib.pyplot as plt

def rgb_to_hsv(rgb_image):
    """Convert RGB image to HSV color space"""
    # Ensure the input is in range [0, 1] and float32
    if rgb_image.dtype != np.float32:
        rgb_image = rgb_image.astype(np.float32)
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    
    # Convert to HSV
    if len(rgb_image.shape) == 4:
        hsv_images = []
        for img in rgb_image:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Normalize H to [0, 179] and S to [0, 255]
            hsv_img[..., 0] = np.clip(hsv_img[..., 0], 0, 179)  # H channel
            hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)  # S channel
            hsv_img[..., 2] = np.clip(hsv_img[..., 2], 0, 1)    # V channel
            hsv_images.append(hsv_img)
        return np.array(hsv_images)
    else:
        hsv_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        # Normalize H to [0, 179] and S to [0, 255]
        hsv_img[..., 0] = np.clip(hsv_img[..., 0], 0, 179)  # H channel
        hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)  # S channel
        hsv_img[..., 2] = np.clip(hsv_img[..., 2], 0, 1)    # V channel
        return hsv_img

def hsv_to_rgb(hsv_image):
    """Convert HSV image to RGB color space"""
    # Ensure float32 for OpenCV
    hsv_image = hsv_image.astype(np.float32)
    
    # Ensure proper ranges for H [0, 179], S [0, 255], V [0, 1]
    hsv_image = hsv_image.copy()  # Make a copy to avoid modifying the original
    hsv_image[..., 0] = np.clip(hsv_image[..., 0], 0, 179)  # H channel
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)  # S channel
    hsv_image[..., 2] = np.clip(hsv_image[..., 2], 0, 1)    # V channel
    
    # Convert to RGB
    if len(hsv_image.shape) == 4:
        rgb_images = []
        for img in hsv_image:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            rgb_images.append(rgb_img)
        return np.array(rgb_images)
    else:
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def build_unet_hsv(input_shape):
    """Build U-Net model for HSV color space (predicting H and S channels)"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2))(x)

    # Level 2
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip2 = x
    x = layers.MaxPooling2D((2, 2))(x)

    # Level 3
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip3 = x
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)  # Add dropout to prevent overfitting

    # Decoder
    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, skip3])
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, skip2])
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, skip1])
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final output layer - predict H and S channels
    x = layers.Conv2D(2, (1, 1), padding='same')(x)
    
    # Custom activation for H and S channels
    def hsv_activation(x):
        h = tf.keras.activations.sigmoid(x[..., 0:1]) * 179.0  # H in range [0, 179]
        s = tf.keras.activations.sigmoid(x[..., 1:2]) * 255.0  # S in range [0, 255]
        return tf.concat([h, s], axis=-1)
    
    outputs = layers.Lambda(hsv_activation)(x)

    model = models.Model(inputs, outputs)
    return model

@tf.keras.utils.register_keras_serializable()
def hsv_loss(y_true, y_pred):
    """Custom loss function for HSV color space prediction
    
    Args:
        y_true: Ground truth H and S channels
        y_pred: Predicted H and S channels
    """
    # Split into H and S channels
    h_true, s_true = tf.split(y_true, 2, axis=-1)
    h_pred, s_pred = tf.split(y_pred, 2, axis=-1)
    
    # Circular loss for Hue (considering the circular nature of hue)
    h_diff = tf.minimum(tf.abs(h_true - h_pred), 179.0 - tf.abs(h_true - h_pred))
    h_loss = tf.reduce_mean(h_diff)
    
    # Regular loss for Saturation
    s_loss = tf.reduce_mean(tf.abs(s_true - s_pred))
    
    # Combine losses with more weight on hue
    return 0.7 * h_loss + 0.3 * s_loss

class ColorizeVisualizerCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, num_samples=3):
        super(ColorizeVisualizerCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.num_samples = min(num_samples, len(self.X_val))
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'colorization_samples'))
        
    def plot_to_image(self, figure):
        """Convert matplotlib figure to PNG."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Visualize every 5 epochs
            predictions = self.model.predict(self.X_val[:self.num_samples])
            
            fig = plt.figure(figsize=(15, 5 * self.num_samples))
            for i in range(self.num_samples):
                # Get Value channel from input (already normalized)
                v_channel = self.X_val[i]
                
                # Predicted color
                predicted_hsv = np.concatenate([predictions[i], v_channel], axis=-1)
                predicted_rgb = hsv_to_rgb(predicted_hsv)
                
                # Ground truth
                true_hsv = np.concatenate([self.y_val[i], v_channel], axis=-1)
                true_rgb = hsv_to_rgb(true_hsv)
                
                # Plot results
                plt.subplot(self.num_samples, 3, i*3 + 1)
                plt.imshow(v_channel.squeeze(), cmap='gray')
                plt.title('Input (Value Channel)')
                plt.axis('off')
                
                plt.subplot(self.num_samples, 3, i*3 + 2)
                plt.imshow(predicted_rgb)
                plt.title('Predicted')
                plt.axis('off')
                
                plt.subplot(self.num_samples, 3, i*3 + 3)
                plt.imshow(true_rgb)
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Print color space ranges for debugging
                print(f"\nSample {i+1} ranges:")
                print(f"V channel range: [{v_channel.min():.2f}, {v_channel.max():.2f}]")
                print(f"Predicted H range: [{predictions[i,:,:,0].min():.2f}, {predictions[i,:,:,0].max():.2f}]")
                print(f"Predicted S range: [{predictions[i,:,:,1].min():.2f}, {predictions[i,:,:,1].max():.2f}]")
            
            plt.suptitle(f"Results for HSV U-Net - Epoch {epoch}")
            plt.tight_layout()
            
            # Log the plot to TensorBoard
            with self.file_writer.as_default():
                tf.summary.image("Colorization Progress", self.plot_to_image(fig), step=epoch)

def get_callbacks(model_dir, log_dir, validation_data):
    """Get all callbacks for training"""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "best_unet_hsv.keras"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=30,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        ),
        ColorizeVisualizerCallback(
            validation_data=validation_data,
            log_dir=log_dir
        )
    ] 