import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from enum import Enum

class ColorSpace(Enum):
    RGB = "rgb"
    LAB = "lab"
    HSV = "hsv"

class ColorizeGAN:
    def __init__(self, img_size=256, color_space=ColorSpace.RGB):
        self.img_size = img_size
        self.color_space = color_space
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Initialize GAN model
        self.gan = self._build_gan()
        
    def _build_generator(self):
        """U-Net based generator"""
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        
        # Encoder
        x = self._encoder_block(inputs, 64)  # 128x128
        skip1 = x
        
        x = self._encoder_block(x, 128)      # 64x64
        skip2 = x
        
        x = self._encoder_block(x, 256)      # 32x32
        skip3 = x
        
        x = self._encoder_block(x, 512)      # 16x16
        skip4 = x
        
        # Bottleneck
        x = self._encoder_block(x, 512)      # 8x8
        x = layers.Dropout(0.5)(x)
        
        # Decoder
        x = self._decoder_block(x, skip4, 512)    # 16x16
        x = self._decoder_block(x, skip3, 256)    # 32x32
        x = self._decoder_block(x, skip2, 128)    # 64x64
        x = self._decoder_block(x, skip1, 64)     # 128x128
        
        # Output layer based on color space
        if self.color_space == ColorSpace.RGB:
            x = layers.Conv2D(3, 1, activation='tanh')(x)  # RGB: 3 channels
        elif self.color_space == ColorSpace.LAB:
            x = layers.Conv2D(2, 1)(x)  # LAB: predict a,b channels
            x = layers.Lambda(lambda x: x * 127.0)(x)  # Scale to LAB range
        else:  # HSV
            x = layers.Conv2D(2, 1)(x)  # HSV: predict H,S channels
            # Custom activation for H and S channels
            h = layers.Lambda(lambda x: x[..., 0:1] * 179.0)(x)  # H: [0, 179]
            s = layers.Lambda(lambda x: tf.nn.sigmoid(x[..., 1:2]) * 255.0)(x)  # S: [0, 255]
            x = layers.Concatenate()([h, s])
        
        return models.Model(inputs, x, name='generator')
    
    def _build_discriminator(self):
        """PatchGAN discriminator"""
        input_shape = (self.img_size, self.img_size, 
                      3 if self.color_space == ColorSpace.RGB else 1)
        
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(1, 4, strides=1, padding='same')(x)
        
        return models.Model(inputs, x, name='discriminator')
    
    def _build_gan(self):
        """Combine generator and discriminator"""
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.img_size, self.img_size, 1))
        gen_output = self.generator(gan_input)
        
        # Combine with grayscale input for discriminator if not RGB
        if self.color_space != ColorSpace.RGB:
            gan_output = layers.Concatenate()([gan_input, gen_output])
        
        gan_output = self.discriminator(gan_output)
        
        return models.Model(gan_input, gan_output, name='gan')
    
    def _encoder_block(self, x, filters):
        """Encoder block: Conv -> BN -> LeakyReLU -> Conv -> BN -> LeakyReLU"""
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x
    
    def _decoder_block(self, x, skip, filters):
        """Decoder block: Upsample -> Concat -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    def compile(self, gen_lr=2e-4, disc_lr=2e-4):
        """Compile both generator and discriminator"""
        self.generator.compile(
            optimizer=tf.keras.optimizers.Adam(gen_lr),
            loss=self._get_generator_loss()
        )
        
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(disc_lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )
        
        self.gan.compile(
            optimizer=tf.keras.optimizers.Adam(gen_lr),
            loss=[self._get_generator_loss(), 
                  tf.keras.losses.BinaryCrossentropy(from_logits=True)],
            loss_weights=[100, 1]  # Higher weight for color loss
        )
    
    def _get_generator_loss(self):
        """Get appropriate loss function based on color space"""
        if self.color_space == ColorSpace.RGB:
            return tf.keras.losses.MeanAbsoluteError()
        elif self.color_space == ColorSpace.LAB:
            return self._lab_loss
        else:
            return self._hsv_loss
    
    def _lab_loss(self, y_true, y_pred):
        """Custom loss for LAB color space"""
        # Ensure predictions are in valid range
        y_pred = tf.clip_by_value(y_pred, -127.0, 127.0)
        
        # Calculate color-aware loss
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        return 0.7 * mae + 0.3 * mse
    
    def _hsv_loss(self, y_true, y_pred):
        """Custom loss for HSV color space"""
        # Split into H and S channels
        h_true, s_true = tf.split(y_true, 2, axis=-1)
        h_pred, s_pred = tf.split(y_pred, 2, axis=-1)
        
        # Circular loss for Hue
        h_diff = tf.minimum(tf.abs(h_true - h_pred), 179.0 - tf.abs(h_true - h_pred))
        h_loss = tf.reduce_mean(h_diff)
        
        # Regular loss for Saturation
        s_loss = tf.reduce_mean(tf.abs(s_true - s_pred))
        
        return 0.7 * h_loss + 0.3 * s_loss
    
    @tf.function
    def train_step(self, grayscale_images, color_images):
        """Single training step"""
        batch_size = tf.shape(grayscale_images)[0]
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(grayscale_images, training=True)
            
            # Get discriminator predictions
            real_output = self.discriminator(color_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate discriminator loss
            real_loss = self.discriminator.loss(tf.ones_like(real_output), real_output)
            fake_loss = self.discriminator.loss(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Apply discriminator gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            generated_images = self.generator(grayscale_images, training=True)
            
            # Get discriminator prediction on fake images
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate generator losses
            gen_loss = self.generator.loss(color_images, generated_images)
            adv_loss = self.discriminator.loss(tf.ones_like(fake_output), fake_output)
            total_gen_loss = 100 * gen_loss + adv_loss
        
        # Apply generator gradients
        gen_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'adv_loss': adv_loss,
            'total_gen_loss': total_gen_loss
        } 