import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from enum import Enum

# Disable mixed precision for this model
tf.keras.mixed_precision.set_global_policy('float32')

class ColorSpace(Enum):
    LAB = 'lab'  # Using LAB color space for better color separation

class ComicColorizeGAN:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.vgg = self._build_vgg()
        self.gan = self._build_gan()
        
    def _build_generator(self):
        """U-Net based generator with comic-specific enhancements"""
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        
        # Initial feature extraction
        x = layers.Conv2D(64, 7, padding='same')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        # Encoder with residual blocks and line art preservation
        skips = []
        filters = [128, 256, 512, 512]
        
        # Encoder path
        for f in filters:
            # Store skip connection
            skips.append(x)
            
            # Down-sampling with line art preservation
            x = self._downsample_block(x, f)
            
            # Add residual blocks
            x = self._residual_block(x, f)
            x = self._residual_block(x, f)
            
            # Add attention at higher levels
            if f >= 256:
                x = self._comic_attention_block(x)
        
        # Bottleneck with dilated convolutions for better context
        x = self._dilated_bottleneck(x)
        
        # Decoder path with skip connections
        for f, skip in zip(filters[::-1], skips[::-1]):
            # Up-sampling
            x = layers.Conv2DTranspose(f, 4, strides=2, padding='same')(x)
            
            # Concatenate with skip connection
            x = layers.Concatenate()([x, skip])
            
            # Refine features
            x = self._residual_block(x, f)
            x = self._residual_block(x, f)
            
            # Add attention at higher levels
            if f >= 256:
                x = self._comic_attention_block(x)
        
        # Final processing
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        # Output a and b channels for LAB color space
        x = layers.Conv2D(2, 1, padding='same')(x)
        outputs = layers.Lambda(lambda x: x * 127.0)(x)  # Scale to LAB range
        
        return models.Model(inputs, outputs, name='generator')
    
    def _build_discriminator(self):
        """Multi-scale PatchGAN discriminator with comic-style awareness"""
        def discriminator_block(x, filters, stride=2):
            x = layers.Conv2D(filters, 4, strides=stride, padding='same', use_bias=False)(x)
            x = layers.LayerNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            return x
        
        # Input is 2-channel LAB (a,b channels)
        inputs = layers.Input(shape=(self.img_size, self.img_size, 2))
        
        # Initial feature extraction
        x = discriminator_block(inputs, 64, stride=2)
        
        # Multi-scale feature extraction
        x = discriminator_block(x, 128)
        x = self._comic_attention_block(x)
        
        x = discriminator_block(x, 256)
        x = self._comic_attention_block(x)
        
        x = discriminator_block(x, 512)
        
        # Output layer
        x = layers.Conv2D(1, 4, padding='same')(x)
        
        return models.Model(inputs, x, name='discriminator')
    
    def _build_vgg(self):
        """Modified VGG for comic-style perceptual loss"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Use specific layers that capture both low-level and high-level features
        outputs = [
            vgg.get_layer('block1_conv1').output,  # Edge features
            vgg.get_layer('block2_conv2').output,  # Texture features
            vgg.get_layer('block3_conv4').output,  # Pattern features
            vgg.get_layer('block4_conv4').output   # High-level features
        ]
        
        return models.Model(vgg.input, outputs)
    
    def _build_gan(self):
        """Combine generator and discriminator"""
        self.discriminator.trainable = False
        
        # GAN input is grayscale image
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        
        # Generate colors
        gen_output = self.generator(inputs)
        
        # Discriminator prediction
        disc_output = self.discriminator(gen_output)
        
        return models.Model(inputs, [gen_output, disc_output])
    
    def _downsample_block(self, x, filters):
        """Downsample while preserving line art details"""
        # Main branch
        main = layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        main = layers.LayerNormalization()(main)
        main = layers.ReLU()(main)
        
        # Edge detection branch
        edge = layers.Conv2D(filters, 3, padding='same')(x)
        edge = layers.LayerNormalization()(edge)
        edge = layers.ReLU()(edge)
        edge = layers.MaxPooling2D(2)(edge)
        
        # Combine features
        return layers.Add()([main, edge])
    
    def _residual_block(self, x, filters):
        """Enhanced residual block for comic features"""
        skip = x
        
        # First convolution
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second convolution
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.LayerNormalization()(x)
        
        # Adjust skip connection if needed
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, 1, padding='same')(skip)
        
        # Combine with skip connection
        x = layers.Add()([x, skip])
        return layers.ReLU()(x)
    
    def _comic_attention_block(self, x):
        """Custom attention mechanism for comic-style features"""
        return ComicAttentionBlock()(x)
    
    def _hw_flatten(self, x):
        """Helper function to reshape feature maps for attention"""
        return tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
    
    def _dilated_bottleneck(self, x):
        """Dilated convolutions for better context understanding"""
        skip = x
        filters = x.shape[-1]
        
        # Series of dilated convolutions
        rates = [1, 2, 4, 8]
        outputs = []
        
        for rate in rates:
            y = layers.Conv2D(filters // 4, 3, padding='same', dilation_rate=rate)(x)
            y = layers.LayerNormalization()(y)
            y = layers.ReLU()(y)
            outputs.append(y)
        
        # Concatenate all dilated conv outputs
        x = layers.Concatenate()(outputs)
        
        # Project back to original number of channels
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.LayerNormalization()(x)
        
        # Residual connection
        return layers.Add()([x, skip])
    
    def compile(self, gen_lr=1e-4, disc_lr=4e-4):
        """Compile with specialized loss functions for comic colorization"""
        # Optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=0.0, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=0.0, beta_2=0.9)
        
        # Loss weights
        self.lambda_l1 = 100.0  # L1 loss weight
        self.lambda_perc = 10.0  # Perceptual loss weight
        self.lambda_style = 50.0  # Style loss weight
        self.lambda_gp = 10.0  # Gradient penalty weight
    
    def gradient_penalty(self, real, fake):
        """WGAN-GP gradient penalty"""
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def style_loss(self, real_features, fake_features):
        """Compute style loss using Gram matrices"""
        style_loss = 0.0
        
        for real_feat, fake_feat in zip(real_features, fake_features):
            # Get shapes
            shape = tf.shape(real_feat)
            h = shape[1]
            w = shape[2]
            c = shape[3]
            
            # Reshape features
            real_feat = tf.reshape(real_feat, [-1, h * w, c])
            fake_feat = tf.reshape(fake_feat, [-1, h * w, c])
            
            # Compute Gram matrices
            real_gram = tf.matmul(real_feat, real_feat, transpose_a=True)
            fake_gram = tf.matmul(fake_feat, fake_feat, transpose_a=True)
            
            # Normalize and compute difference
            size = tf.cast(h * w * c, tf.float32)
            style_loss += tf.reduce_mean(tf.abs(real_gram - fake_gram)) / size
        
        return style_loss
    
    @tf.function
    def train_step(self, grayscale_images, color_images):
        """Single training step with comic-specific loss computation"""
        batch_size = tf.shape(grayscale_images)[0]
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake colors
            fake_colors = self.generator(grayscale_images, training=True)
            
            # Discriminator predictions
            real_output = self.discriminator(color_images, training=True)
            fake_output = self.discriminator(fake_colors, training=True)
            
            # Gradient penalty
            gp = self.gradient_penalty(color_images, fake_colors)
            
            # Discriminator loss (WGAN-GP)
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            disc_loss += self.lambda_gp * gp
            
            # Generator losses
            l1_loss = tf.reduce_mean(tf.abs(color_images - fake_colors))
            
            gen_loss = -tf.reduce_mean(fake_output)  # WGAN generator loss
            gen_loss += self.lambda_l1 * l1_loss
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Create gradient-variable pairs, filtering out None gradients
        gen_grads_and_vars = [(g, v) for (g, v) in zip(gen_gradients, self.generator.trainable_variables) if g is not None]
        disc_grads_and_vars = [(g, v) for (g, v) in zip(disc_gradients, self.discriminator.trainable_variables) if g is not None]
        
        # Apply gradients if there are any
        if gen_grads_and_vars:
            self.gen_optimizer.apply_gradients(gen_grads_and_vars)
        if disc_grads_and_vars:
            self.disc_optimizer.apply_gradients(disc_grads_and_vars)
        
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'l1_loss': l1_loss,
            'gp': gp
        } 

class ComicAttentionBlock(layers.Layer):
    """Custom attention mechanism for comic-style features"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        filters = input_shape[-1]
        self.f_conv = layers.Conv2D(filters // 8, 1)
        self.g_conv = layers.Conv2D(filters // 8, 1)
        self.h_conv = layers.Conv2D(filters, 1)
        self.add = layers.Add()
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x):
        # Generate attention maps
        f = self.f_conv(x)
        g = self.g_conv(x)
        h = self.h_conv(x)
        
        # Get shapes
        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        n_channels = shape[3]
        
        # Reshape for attention computation
        f_flat = tf.reshape(f, [-1, height * width, n_channels // 8])
        g_flat = tf.reshape(g, [-1, height * width, n_channels // 8])
        h_flat = tf.reshape(h, [-1, height * width, n_channels])
        
        # Compute attention scores
        s = tf.matmul(g_flat, f_flat, transpose_b=True)
        attention_map = tf.nn.softmax(s)
        
        # Apply attention
        o = tf.matmul(attention_map, h_flat)
        o = tf.reshape(o, [-1, height, width, n_channels])
        
        # Residual connection
        return self.add([x, o]) 