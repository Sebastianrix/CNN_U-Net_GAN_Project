import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))
print("\nAll Devices:")
print(tf.config.list_physical_devices()) 