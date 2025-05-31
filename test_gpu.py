import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))
print("\nIs built with CUDA:", tf.test.is_built_with_cuda())

# Try to perform a simple operation on GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nMatrix multiplication result:", c.numpy())
        print("\nGPU test successful! Your GPU is working with TensorFlow.")
except RuntimeError as e:
    print("\nGPU test failed:", str(e)) 