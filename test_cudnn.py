import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)
print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))

# Test if CUDA is available
print("\nCUDA Available:", tf.test.is_built_with_cuda())

# Test if GPU is available
print("GPU Available:", tf.test.is_gpu_available())

# Get device details
print("\nDevice Details:")
for device in tf.config.list_physical_devices():
    print(device)

# Try to perform a simple GPU operation to verify everything works
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nGPU Computation Test Result:")
        print(c)
        print("\nGPU test passed successfully!")
except Exception as e:
    print("\nGPU test failed with error:", str(e)) 