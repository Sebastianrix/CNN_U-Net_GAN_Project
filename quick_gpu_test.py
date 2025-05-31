import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\nDevices:", tf.config.list_physical_devices()) 