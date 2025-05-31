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
        self.gan = self._build_gan()
        
    def _downsample(self, x, filters, size=3, apply_batchnorm=True):
        x = layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False)(x)
        if apply_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x
    
    def _upsample(self, x, skip, filters, size=3, apply_dropout=False):
        x = layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        if apply_dropout:
            x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip])
        return x
    
    def _build_generator(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        
        # Encoder
        d1 = self._downsample(inputs, 64, apply_batchnorm=False)  # 128x128
        d2 = self._downsample(d1, 128)  # 64x64
        d3 = self._downsample(d2, 256)  # 32x32
        d4 = self._downsample(d3, 512)  # 16x16
        d5 = self._downsample(d4, 512)  # 8x8
        
        # Bottleneck
        bottleneck = layers.Conv2D(512, 3, padding='same', use_bias=False)(d5)
        bottleneck = layers.BatchNormalization()(bottleneck)
        bottleneck = layers.LeakyReLU(0.2)(bottleneck)
        
        # Decoder
        u1 = self._upsample(bottleneck, d4, 512, apply_dropout=True)  # 16x16
        u2 = self._upsample(u1, d3, 256, apply_dropout=True)  # 32x32
        u3 = self._upsample(u2, d2, 128)  # 64x64
        u4 = self._upsample(u3, d1, 64)  # 128x128
        
        # Final upsampling without skip connection
        last = layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False)(u4)
        last = layers.BatchNormalization()(last)
        last = layers.ReLU()(last)
        
        # Output layer based on color space
        if self.color_space == ColorSpace.RGB:
            outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(last)
        elif self.color_space == ColorSpace.LAB:
            x = layers.Conv2D(2, 3, padding='same')(last)
            outputs = layers.Lambda(lambda x: x * 127.0)(x)
        else:  # HSV
            x = layers.Conv2D(2, 3, padding='same')(last)
            h = layers.Lambda(lambda x: x[..., 0:1] * 179.0)(x)
            s = layers.Lambda(lambda x: tf.nn.sigmoid(x[..., 1:2]) * 255.0)(x)
            outputs = layers.Concatenate()([h, s])
        
        return models.Model(inputs, outputs, name='generator')
    
    def _build_discriminator(self):
        """PatchGAN discriminator"""
        input_shape = (self.img_size, self.img_size, 
                      3 if self.color_space == ColorSpace.RGB else 2)  # Changed from 1 to 2 for LAB/HSV
        
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
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.img_size, self.img_size, 1))
        gen_output = self.generator(gan_input)
        
        # Remove the concatenation with input for non-RGB
        gan_output = self.discriminator(gen_output)
        
        return models.Model(gan_input, gan_output, name='gan')
    
    def compile(self, gen_lr=2e-4, disc_lr=2e-4):
        """Compile both generator and discriminator"""
        # Generator optimizer and loss
        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr)
        self.gen_loss = self._get_generator_loss()
        
        # Discriminator optimizer
        self.disc_optimizer = tf.keras.optimizers.Adam(disc_lr)
        self.disc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def _get_generator_loss(self):
        if self.color_space == ColorSpace.RGB:
            return tf.keras.losses.MeanAbsoluteError()
        elif self.color_space == ColorSpace.LAB:
            return self._lab_loss
        else:
            return self._hsv_loss
    
    def _lab_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, -127.0, 127.0)
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return 0.7 * mae + 0.3 * mse
    
    def _hsv_loss(self, y_true, y_pred):
        h_true, s_true = tf.split(y_true, 2, axis=-1)
        h_pred, s_pred = tf.split(y_pred, 2, axis=-1)
        
        h_diff = tf.minimum(tf.abs(h_true - h_pred), 179.0 - tf.abs(h_true - h_pred))
        h_loss = tf.reduce_mean(h_diff)
        
        s_loss = tf.reduce_mean(tf.abs(s_true - s_pred))
        
        return 0.7 * h_loss + 0.3 * s_loss
    
    @tf.function
    def train_step(self, grayscale_images, color_images):
        """Single training step"""
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            generated_images = self.generator(grayscale_images, training=True)
            
            # Get discriminator predictions
            real_output = self.discriminator(color_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate discriminator loss
            real_loss = self.disc_loss(tf.ones_like(real_output), real_output)
            fake_loss = self.disc_loss(tf.zeros_like(fake_output), fake_output)
            total_disc_loss = real_loss + fake_loss
            
            # Calculate generator losses
            gen_color_loss = self.gen_loss(color_images, generated_images)
            gen_adv_loss = self.disc_loss(tf.ones_like(fake_output), fake_output)
            total_gen_loss = 100.0 * gen_color_loss + gen_adv_loss
        
        # Calculate and apply discriminator gradients
        disc_gradients = tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        if any(g is not None for g in disc_gradients):
            self.disc_optimizer.apply_gradients(
                [(g, v) for (g, v) in zip(disc_gradients, self.discriminator.trainable_variables) if g is not None]
            )
        
        # Calculate and apply generator gradients
        gen_gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        if any(g is not None for g in gen_gradients):
            self.gen_optimizer.apply_gradients(
                [(g, v) for (g, v) in zip(gen_gradients, self.generator.trainable_variables) if g is not None]
            )
        
        del tape
        
        return {
            'disc_loss': total_disc_loss,
            'gen_color_loss': gen_color_loss,
            'gen_adv_loss': gen_adv_loss,
            'total_gen_loss': total_gen_loss
        } 