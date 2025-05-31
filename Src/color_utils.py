import tensorflow as tf

def hsv_to_rgb(hsv):
    """Convert HSV to RGB color space using TensorFlow operations"""
    h = hsv[..., 0:1]
    s = hsv[..., 1:2]
    v = hsv[..., 2:3]
    
    # Normalize to [0, 1]
    h = h / 179.0
    s = s / 255.0
    v = v / 255.0
    
    # HSV to RGB conversion
    c = v * s
    x = c * (1 - tf.abs(tf.math.mod(h * 6.0, 2.0) - 1))
    m = v - c
    
    # Calculate RGB components
    h_prime = h * 6.0
    zeros = tf.zeros_like(h)
    ones = tf.ones_like(h)
    
    mask0 = tf.cast(h_prime < 1.0, tf.float32)
    mask1 = tf.cast((h_prime >= 1.0) & (h_prime < 2.0), tf.float32)
    mask2 = tf.cast((h_prime >= 2.0) & (h_prime < 3.0), tf.float32)
    mask3 = tf.cast((h_prime >= 3.0) & (h_prime < 4.0), tf.float32)
    mask4 = tf.cast((h_prime >= 4.0) & (h_prime < 5.0), tf.float32)
    mask5 = tf.cast(h_prime >= 5.0, tf.float32)
    
    r = c * mask0 + x * mask1 + zeros * mask2 + zeros * mask3 + x * mask4 + c * mask5
    g = x * mask0 + c * mask1 + c * mask2 + x * mask3 + zeros * mask4 + zeros * mask5
    b = zeros * mask0 + zeros * mask1 + x * mask2 + c * mask3 + c * mask4 + x * mask5
    
    r = r + m
    g = g + m
    b = b + m
    
    # Scale back to [0, 255]
    rgb = tf.concat([r, g, b], axis=-1) * 255.0
    return rgb

def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space using TensorFlow operations"""
    # Normalize RGB to [0, 1]
    rgb = tf.cast(rgb, tf.float32) / 255.0
    r = rgb[..., 0:1]
    g = rgb[..., 1:2]
    b = rgb[..., 2:3]
    
    # Calculate value (v)
    v = tf.reduce_max(rgb, axis=-1, keepdims=True)
    
    # Calculate saturation (s)
    c_min = tf.reduce_min(rgb, axis=-1, keepdims=True)
    delta = v - c_min
    s = tf.where(v > 0, delta / v, tf.zeros_like(v))
    
    # Calculate hue (h)
    h = tf.zeros_like(v)
    
    # Red is max
    mask_r = tf.cast(tf.equal(v, r), tf.float32)
    h += mask_r * (60.0 * ((g - b) / (delta + 1e-7)))
    
    # Green is max
    mask_g = tf.cast(tf.equal(v, g), tf.float32)
    h += mask_g * (60.0 * (2.0 + (b - r) / (delta + 1e-7)))
    
    # Blue is max
    mask_b = tf.cast(tf.equal(v, b), tf.float32)
    h += mask_b * (60.0 * (4.0 + (r - g) / (delta + 1e-7)))
    
    # Normalize hue
    h = tf.where(h < 0, h + 360.0, h)
    h = h / 2.0  # Convert to [0, 179] range
    
    # Scale back
    h = tf.clip_by_value(h, 0.0, 179.0)
    s = tf.clip_by_value(s * 255.0, 0.0, 255.0)
    v = tf.clip_by_value(v * 255.0, 0.0, 255.0)
    
    return tf.concat([h, s, v], axis=-1) 