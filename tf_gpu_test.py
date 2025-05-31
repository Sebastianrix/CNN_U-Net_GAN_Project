import tensorflow as tf
import os
import sys

def print_gpu_info():
    print("\n=== TensorFlow GPU Detection Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # List physical devices
    print("\nPhysical devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")
    
    # Specifically check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nNumber of GPUs detected: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"GPU found: {gpu}")
            
        # Get GPU device details
        try:
            with tf.device('/GPU:0'):
                print("\nGPU device details:")
                print(tf.config.experimental.get_device_details(gpus[0]))
        except Exception as e:
            print(f"\nError getting GPU details: {e}")
    else:
        print("No GPU devices found.")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            # Create and multiply two large matrices
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            print("GPU computation test successful!")
            print(f"Result shape: {c.shape}")
    except Exception as e:
        print(f"GPU computation test failed: {e}")

if __name__ == "__main__":
    print_gpu_info() 