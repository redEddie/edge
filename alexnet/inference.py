import os
import tensorflow as tf
import time
import numpy as np

USE_GPU = True
GPU_MEM_LIMIT = 512
THREADS = None

if __name__ == '__main__':
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if THREADS:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEM_LIMIT)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    model = tf.keras.models.load_model('model.keras')
    model.summary()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if USE_GPU:
        model.predict(x_test) # pre-prediction process to load cuda libs

    infer_time = []
    for i in range(10):
        print('predict start!')
        start = time.time()
        result = model.predict(x_test)
        end = time.time()
        print(f'prediction took {end - start} seconds.')
        infer_time.append(end - start)

    infer_time = np.array(infer_time)
    print(f'avg: {np.mean(infer_time)}\nmax: {np.max(infer_time)}\nmin: {np.min(infer_time)}')