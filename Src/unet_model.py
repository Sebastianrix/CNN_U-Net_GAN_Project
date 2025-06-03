import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape, output_channels=3, use_scaling=True):
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
    bn = layers.Dropout(0.3)(bn)  # Dropout to reduce overfitting

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

    # Output layer with flexible output channels depending on color space
    outputs = layers.Conv2D(output_channels, (1, 1), activation='tanh')(c6)

    if use_scaling:
        outputs = layers.Lambda(lambda x: (x + 1.0) / 2.0, dtype='float32')(outputs)
    
    model = models.Model(inputs, outputs)

    return model

def get_callbacks():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    return [reduce_lr, early_stopping]
