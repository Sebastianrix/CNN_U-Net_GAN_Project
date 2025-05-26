import tensorflow as tf
from tensorflow.keras import layers, models
from skimage import color
import numpy as np
import os
import io
import matplotlib.pyplot as plt

def rgb_to_lab(rgb_image):
    """Convert RGB image to LAB color space"""
    # Ensure the input is in range [0, 1]
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    return color.rgb2lab(rgb_image)

def lab_to_rgb(lab_image):
    """Convert LAB image to RGB color space"""
    rgb_image = color.lab2rgb(lab_image)
    # Ensure output is in range [0, 1]
    return np.clip(rgb_image, 0, 1)

def build_unet_lab(input_shape):
    """Build U-Net model for LAB color space (predicting a and b channels)"""
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

    # Final output - scale to proper LAB ranges
    x = layers.Conv2D(2, (1, 1), padding='same')(x)
    x = layers.Activation('tanh')(x)  # Output in [-1, 1]
    outputs = layers.Lambda(lambda x: x * 127.0)(x)  # Scale to [-127, 127]

    model = models.Model(inputs, outputs)
    return model

@tf.keras.utils.register_keras_serializable()
def lab_loss(y_true, y_pred):
    """Simple L1 + L2 loss for LAB color space"""
    # Ensure predictions are in valid range
    y_pred = tf.clip_by_value(y_pred, -127.0, 127.0)
    
    # Basic L1 (MAE) and L2 (MSE) losses
    mae = tf.abs(y_true - y_pred)
    mse = tf.square(y_true - y_pred)
    
    # Combine losses
    return tf.reduce_mean(mae) + 0.5 * tf.reduce_mean(mse)

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
                # Original grayscale (denormalize from [-1, 1] to [0, 100])
                l_channel = (self.X_val[i] * 50.0) + 50.0
                
                # Predicted color
                predicted_lab = np.concatenate([l_channel, predictions[i]], axis=-1)
                predicted_rgb = lab_to_rgb(predicted_lab)
                
                # Ground truth
                true_lab = np.concatenate([l_channel, self.y_val[i]], axis=-1)
                true_rgb = lab_to_rgb(true_lab)
                
                # Plot results
                plt.subplot(self.num_samples, 3, i*3 + 1)
                plt.imshow(l_channel.squeeze(), cmap='gray')
                plt.title('Input (Grayscale)')
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
                print(f"L channel range: [{l_channel.min():.2f}, {l_channel.max():.2f}]")
                print(f"Predicted A channel range: [{predictions[i,:,:,0].min():.2f}, {predictions[i,:,:,0].max():.2f}]")
                print(f"Predicted B channel range: [{predictions[i,:,:,1].min():.2f}, {predictions[i,:,:,1].max():.2f}]")
            
            plt.suptitle(f"Results for LAB U-Net - Epoch {epoch}")
            plt.tight_layout()
            
            # Log the plot to TensorBoard
            with self.file_writer.as_default():
                tf.summary.image("Colorization Progress", self.plot_to_image(fig), step=epoch)

def get_callbacks(model_dir, log_dir, validation_data):
    """Get all callbacks for training"""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "best_unet_lab.keras"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=30,  # Increased patience
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,  # Increased patience
            min_lr=1e-7,  # Lower minimum learning rate
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