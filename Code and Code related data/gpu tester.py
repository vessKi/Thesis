import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# List physical devices
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")
for gpu in gpus:
    print(f"GPU: {gpu}")

# Check if TensorFlow is built with CUDA and cuDNN
print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Is TensorFlow built with GPU support: {tf.test.is_built_with_gpu_support()}")

# Run a simple computation to check GPU performance
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print(f"Matrix multiplication result: \n{c}")

# Check if TensorFlow can run computations on the GPU
if len(gpus) > 0:
    with tf.device('/GPU:0'):
        d = tf.matmul(a, b)
        print(f"Matrix multiplication result on GPU: \n{d}")

# Check TensorFlow device placement logging
tf.debugging.set_log_device_placement(True)

# Perform a more complex operation to verify GPU usage
if len(gpus) > 0:
    with tf.device('/GPU:0'):
        e = tf.random.normal([10000, 10000])
        f = tf.random.normal([10000, 10000])
        g = tf.matmul(e, f)
        print("Performed a large matrix multiplication on GPU")
