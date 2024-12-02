import tensorflow as tf
import time

def stress_gpu(duration=60):
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU found. Please ensure that you have a CUDA-compatible GPU and TensorFlow installed.")
        return

    device_name = tf.config.list_physical_devices('GPU')[0].name
    print(f"Using GPU: {device_name}")
    print("Starting stress test...")

    matrix_size = 10000

    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        with tf.device('/GPU:0'):
            x = tf.random.normal([matrix_size, matrix_size])
            y = tf.random.normal([matrix_size, matrix_size])
            result = tf.matmul(x, y)

            # print(result)

    print(f"Test completed in {duration} seconds.")

stress_gpu(duration=60)