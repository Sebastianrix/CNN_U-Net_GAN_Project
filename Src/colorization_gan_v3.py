import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from enum import Enum
from color_utils import hsv_to_rgb, rgb_to_hsv

class ColorSpace(Enum):
    RGB = "rgb"
    LAB = "lab"
    HSV = "hsv"

class SpectralNormalization(tf.keras.layers.Wrapper):
    """Spectral normalization layer wrapper"""
    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.power_iterations = power_iterations
    
    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        
        if not hasattr(self.layer, 'kernel'):
            raise ValueError('`SpectralNormalization` must wrap a layer that'
                           ' contains a `kernel` for weights')
        
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name='sn_u',
            dtype=self.w.dtype
        )
        
        super(SpectralNormalization, self).build()
    
    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = None
        
        for _ in range(self.power_iterations):
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = tf.nn.l2_normalize(v_)
            
            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = tf.nn.l2_normalize(u_)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)

class SelfAttentionBlock(layers.Layer):
    """Self-attention layer for better global context"""
    def __init__(self, filters, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.f_conv = layers.Conv2D(filters // 8, 1)
        self.g_conv = layers.Conv2D(filters // 8, 1)
        self.h_conv = layers.Conv2D(filters, 1)
        self.softmax = layers.Softmax(axis=-1)
        self.add = layers.Add()
    
    def build(self, input_shape):
        super(SelfAttentionBlock, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        # Get input shape
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        
        # Apply convolutions
        f = self.f_conv(inputs)
        g = self.g_conv(inputs)
        h = self.h_conv(inputs)
        
        # Reshape to matrix multiply
        f_flat = tf.reshape(f, [-1, height * width, self.filters // 8])
        g_flat = tf.reshape(g, [-1, height * width, self.filters // 8])
        h_flat = tf.reshape(h, [-1, height * width, self.filters])
        
        # Attention map
        s = tf.matmul(g_flat, f_flat, transpose_b=True)
        beta = self.softmax(s)
        
        # Apply attention
        o = tf.matmul(beta, h_flat)
        o = tf.reshape(o, [-1, height, width, self.filters])
        
        return self.add([inputs, o])

def ResBlock(x, filters, kernel_size=3):
    """Residual block with instance normalization"""
    skip = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.LayerNormalization()(x)
    
    if skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, padding='same')(skip)
    
    return layers.Add()([x, skip])

class ColorizeGAN:
    def __init__(self, img_size=256, color_space=ColorSpace.HSV):
        self.img_size = img_size
        self.color_space = color_space
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.vgg = self._build_vgg()
        self.gan = self._build_gan()
        
    def _build_generator(self):
        """U-Net based generator with residual blocks and self-attention"""
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        
        # Initial convolution
        x = layers.Conv2D(64, 7, padding='same')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        # Encoder
        skips = []
        filters = [128, 256, 512, 512]
        for f in filters:
            skips.append(x)
            x = layers.Conv2D(f, 4, strides=2, padding='same')(x)
            x = ResBlock(x, f)
            if f >= 256:
                x = SelfAttentionBlock(f)(x)
        
        # Bottleneck
        x = ResBlock(x, 512)
        x = ResBlock(x, 512)
        x = SelfAttentionBlock(512)(x)
        
        # Decoder
        for f, skip in zip(filters[::-1], skips[::-1]):
            x = layers.Conv2DTranspose(f, 4, strides=2, padding='same')(x)
            x = layers.Concatenate()([x, skip])
            x = ResBlock(x, f)
            if f >= 256:
                x = SelfAttentionBlock(f)(x)
        
        # Output layer based on color space
        if self.color_space == ColorSpace.RGB:
            x = layers.Conv2D(3, 7, padding='same', activation='tanh')(x)
        elif self.color_space == ColorSpace.LAB:
            x = layers.Conv2D(2, 7, padding='same')(x)
            x = layers.Lambda(lambda x: x * 127.0)(x)
        else:  # HSV
            x = layers.Conv2D(2, 7, padding='same')(x)
            # Separate H and S predictions with proper activation
            h = layers.Lambda(lambda x: x[..., 0:1])(x)
            s = layers.Lambda(lambda x: x[..., 1:2])(x)
            
            # H: Use periodic activation for hue [0, 179]
            h = layers.Lambda(lambda x: (tf.math.atan2(
                tf.sin(x * 2 * np.pi / 179.0),
                tf.cos(x * 2 * np.pi / 179.0)
            ) / (2 * np.pi) + 0.5) * 179.0)(h)
            
            # S: Use sigmoid for saturation [0, 255]
            s = layers.Lambda(lambda x: tf.nn.sigmoid(x) * 255.0)(s)
            
            x = layers.Concatenate()([h, s])
        
        return models.Model(inputs, x, name='generator')
    
    def _build_discriminator(self):
        """Multi-scale PatchGAN discriminator with spectral normalization"""
        def discriminator_block(x, filters, stride=2):
            x = SpectralNormalization(
                layers.Conv2D(filters, 4, strides=stride, padding='same')
            )(x)
            x = layers.LeakyReLU(0.2)(x)
            return x
        
        # Input shape based on color space
        input_shape = (self.img_size, self.img_size, 
                      3 if self.color_space == ColorSpace.RGB else 2)
        
        inputs = layers.Input(shape=input_shape)
        
        # Multi-scale discrimination
        scales = []
        x = inputs
        num_scales = 3
        
        for i in range(num_scales):
            if i > 0:
                x = layers.AveragePooling2D(pool_size=(2, 2))(x)
            
            # Discriminator network for this scale
            d = x
            d = discriminator_block(d, 64, stride=1 if i > 0 else 2)
            d = discriminator_block(d, 128)
            d = discriminator_block(d, 256)
            d = discriminator_block(d, 512)
            
            # Output for this scale
            d = SpectralNormalization(
                layers.Conv2D(1, 4, padding='same')
            )(d)
            scales.append(d)
        
        return models.Model(inputs, scales, name='discriminator')
    
    def _build_vgg(self):
        """Build VGG model for perceptual loss"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in [
            'block1_conv1', 'block2_conv1',
            'block3_conv1', 'block4_conv1'
        ]]
        return models.Model(vgg.input, outputs)
    
    def _build_gan(self):
        self.discriminator.trainable = False
        inputs = layers.Input(shape=(self.img_size, self.img_size, 1))
        gen_output = self.generator(inputs)
        disc_outputs = self.discriminator(gen_output)
        return models.Model(inputs, [gen_output] + disc_outputs)
    
    def compile(self, gen_lr=1e-4, disc_lr=4e-4):
        """Compile with WGAN-GP loss and additional perceptual losses"""
        # Enable mixed precision for GPU training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Optimizers with original settings
        self.gen_optimizer = tf.keras.optimizers.Adam(
            gen_lr, beta_1=0.0, beta_2=0.9,
            clipnorm=1.0
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.9,
            clipnorm=1.0
        )
        
        # Original loss weights
        self.lambda_gp = 10.0  # Gradient penalty weight
        self.lambda_fm = 10.0  # Feature matching weight
        self.lambda_perc = 10.0  # Perceptual loss weight
        self.lambda_hist = 5.0  # Histogram matching weight
    
    def gradient_penalty(self, real, fake):
        """WGAN-GP gradient penalty"""
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1])
        interpolated = alpha * real + (1 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated)[0]
        
        gradients = tape.gradient(pred, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        return tf.reduce_mean(tf.square(slopes - 1.0))
    
    def feature_matching_loss(self, real_features, fake_features):
        """Feature matching loss across discriminator scales"""
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += tf.reduce_mean(tf.abs(real_feat - fake_feat))
        return loss
    
    def perceptual_loss(self, real_rgb, fake_rgb):
        """VGG-based perceptual loss"""
        real_features = self.vgg(real_rgb)
        fake_features = self.vgg(fake_rgb)
        
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += tf.reduce_mean(tf.abs(real_feat - fake_feat))
        return loss
    
    def histogram_loss(self, real, fake):
        """Color histogram matching loss"""
        def get_hist(x):
            hist = tf.histogram_fixed_width(
                x, [0.0, 255.0], nbins=256, dtype=tf.int32
            )
            hist = tf.cast(hist, tf.float32)
            return hist / tf.reduce_sum(hist)
        
        loss = 0
        for i in range(real.shape[-1]):
            real_hist = get_hist(real[..., i])
            fake_hist = get_hist(fake[..., i])
            loss += tf.reduce_mean(tf.abs(real_hist - fake_hist))
        return loss
    
    @tf.function
    def train_step(self, grayscale_images, color_images):
        """Single training step with improved losses"""
        batch_size = tf.shape(grayscale_images)[0]
        
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_colors = self.generator(grayscale_images, training=True)
            
            # Get discriminator outputs for real and fake
            real_outputs = self.discriminator(color_images, training=True)
            fake_outputs = self.discriminator(fake_colors, training=True)
            
            # Discriminator losses
            disc_loss = 0
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                disc_loss += tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
            
            # Gradient penalty
            gp = self.gradient_penalty(color_images, fake_colors)
            disc_loss += self.lambda_gp * gp
            
            # Generator losses
            gen_loss = 0
            for fake_out in fake_outputs:
                gen_loss -= tf.reduce_mean(fake_out)
            
            # Feature matching loss
            fm_loss = self.feature_matching_loss(real_outputs, fake_outputs)
            gen_loss += self.lambda_fm * fm_loss
            
            # Convert to RGB for perceptual loss if needed
            if self.color_space != ColorSpace.RGB:
                if self.color_space == ColorSpace.LAB:
                    from unet_model_lab import lab_to_rgb
                    fake_rgb = lab_to_rgb(tf.concat([grayscale_images, fake_colors], axis=-1))
                    real_rgb = lab_to_rgb(tf.concat([grayscale_images, color_images], axis=-1))
                else:  # HSV
                    # Create full HSV tensors
                    fake_hsv = tf.concat([fake_colors, grayscale_images * 255.0], axis=-1)
                    real_hsv = tf.concat([color_images, grayscale_images * 255.0], axis=-1)
                    # Convert to RGB
                    fake_rgb = hsv_to_rgb(fake_hsv)
                    real_rgb = hsv_to_rgb(real_hsv)
            else:
                fake_rgb = fake_colors
                real_rgb = color_images
            
            # Perceptual loss
            perc_loss = self.perceptual_loss(real_rgb, fake_rgb)
            gen_loss += self.lambda_perc * perc_loss
            
            # Histogram matching loss
            hist_loss = self.histogram_loss(color_images, fake_colors)
            gen_loss += self.lambda_hist * hist_loss
        
        # Calculate gradients
        disc_vars = self.discriminator.trainable_variables
        gen_vars = self.generator.trainable_variables
        
        disc_gradients = tape.gradient(disc_loss, disc_vars)
        gen_gradients = tape.gradient(gen_loss, gen_vars)
        
        # Apply gradients
        if disc_gradients is not None and any(g is not None for g in disc_gradients):
            disc_grads_and_vars = [(g, v) for g, v in zip(disc_gradients, disc_vars) if g is not None]
            self.disc_optimizer.apply_gradients(disc_grads_and_vars)
        
        if gen_gradients is not None and any(g is not None for g in gen_gradients):
            gen_grads_and_vars = [(g, v) for g, v in zip(gen_gradients, gen_vars) if g is not None]
            self.gen_optimizer.apply_gradients(gen_grads_and_vars)
        
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'gp_loss': gp,
            'fm_loss': fm_loss,
            'perc_loss': perc_loss,
            'hist_loss': hist_loss
        } 