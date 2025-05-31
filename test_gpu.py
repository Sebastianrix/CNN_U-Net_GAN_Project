import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))
print("\nIs built with CUDA:", tf.test.is_built_with_cuda())

print('\nTrying a simple GPU operation...')
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print('GPU computation successful!')
except Exception as e:
    print('Error during GPU computation:', str(e)) 