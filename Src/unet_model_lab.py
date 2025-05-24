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
    c1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Conv2D(64, (3, 3), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Conv2D(128, (3, 3), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Conv2D(256, (3, 3), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    bn = layers.Conv2D(512, (3, 3), padding='same')(p3)
    bn = layers.BatchNormalization()(bn)
    bn = layers.ReLU()(bn)
    bn = layers.Conv2D(512, (3, 3), padding='same')(bn)
    bn = layers.BatchNormalization()(bn)
    bn = layers.ReLU()(bn)
    bn = layers.Dropout(0.3)(bn)

    # Decoder
    u1 = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(bn)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(256, (3, 3), padding='same')(u1)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)
    c4 = layers.Conv2D(256, (3, 3), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)

    u2 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, (3, 3), padding='same')(u2)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)
    c5 = layers.Conv2D(128, (3, 3), padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)

    u3 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(64, (3, 3), padding='same')(u3)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)
    c6 = layers.Conv2D(64, (3, 3), padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)

    # Output layer: 2 channels for a* and b* values
    outputs = layers.Conv2D(2, (1, 1), activation='tanh')(c6)
    
    # Scale the output to match the a* and b* ranges in LAB space
    outputs = layers.Lambda(lambda x: x * 100)(outputs)

    model = models.Model(inputs, outputs)
    return model

@tf.keras.utils.register_keras_serializable()
def lab_loss(y_true, y_pred):
    """Combined loss function for LAB color space"""
    # MSE loss
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # MAE loss
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    # Combine losses (similar weights to RGB version)
    return 0.84 * mse + 0.16 * mae

class ColorizeVisualizerCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, num_samples=3):
        super(ColorizeVisualizerCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.num_samples = min(num_samples, len(self.X_val))
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'colorization_samples'))
        
        # Convert validation ground truth to LAB
        self.y_val_lab = np.array([rgb_to_lab(img) for img in self.y_val[:self.num_samples]])
        
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
                # Original grayscale
                plt.subplot(self.num_samples, 3, i*3 + 1)
                plt.imshow(self.X_val[i].squeeze(), cmap='gray')
                plt.title('Input (Grayscale)')
                plt.axis('off')
                
                # Predicted color
                l_channel = self.y_val_lab[i, :, :, 0:1]
                predicted_lab = np.concatenate([l_channel, predictions[i]], axis=-1)
                predicted_rgb = lab_to_rgb(predicted_lab)
                
                plt.subplot(self.num_samples, 3, i*3 + 2)
                plt.imshow(predicted_rgb)
                plt.title('Predicted')
                plt.axis('off')
                
                # Ground truth
                plt.subplot(self.num_samples, 3, i*3 + 3)
                plt.imshow(self.y_val[i])
                plt.title('Ground Truth')
                plt.axis('off')
            
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
            patience=15,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
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