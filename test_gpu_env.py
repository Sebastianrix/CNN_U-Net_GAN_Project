import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("\nNum GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))
print("\nCUDA Available:", tf.test.is_built_with_cuda())
print("\nAll Physical Devices:")
print(tf.config.list_physical_devices())

# Try a simple GPU operation
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nGPU Computation Result:")
        print(c)
        print("\nGPU test successful!")
except Exception as e:
    print("\nGPU test failed with error:", str(e)) 