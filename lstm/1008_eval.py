import os
import tensorflow as tf
import numpy as np
import psutil
import time
import argparse

#####################
# Argument Parsing
#####################
def parse_args():
    parser = argparse.ArgumentParser(description="Train an AlexNet model on the CIFAR-10 dataset.")
    parser.add_argument("--gpu", type=bool, default=True, help="Use GPU for training. (Default = True)")
    parser.add_argument("--gpu_mem_limit", type=int, default=1024, help="Limit GPU memory usage in MB. Set to 0 for no limit. (Default = 1024)")
    parser.add_argument("--memory_growth", type=bool, default=False, help="Enable GPU memory growth. (Default = False)")
    parser.add_argument("--threads", type=int, default=0, help="Number of CPU threads to use (0 for all available, Default = 0).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs. (Default = 10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training. (Default = 128)")
    return parser.parse_args()

args = parse_args()

# Configurations

USE_GPU = args.gpu
GPU_MEM_LIMIT = args.gpu_mem_limit
MEMORY_GROWTH = args.memory_growth
THREADS = args.threads
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


#####################
# CPU Setup
#####################
def setup_cpu_threads(threads):
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    print(f"Logical CPUs: {logical_cores}, Physical CPUs: {physical_cores}")
    
    # number of threads used for parallel execution of different operations
    tf.config.threading.set_inter_op_parallelism_threads(0)

    # number of threads used for parallel execution within a single operation
    if threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
    
    print(f"Inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"Intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")

setup_cpu_threads(THREADS)

#####################
# GPU Setup
#####################
def setup_gpu(use_gpu=True, memory_growth=True, gpu_mem_limit=False):
    if not use_gpu:
        # empty string means no GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Num GPUs Available: {len(gpus)}")

        for gpu in gpus:
            print(f"GPU Details: {tf.config.experimental.get_device_details(gpu)}")
        
        if memory_growth:
            for gpu in gpus:
                # memory growth allows the process to allocate memory only when needed
                tf.config.experimental.set_memory_growth(gpu, True)
        
        try:
            # Set memory limit for the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit)]
            )
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
        
        # Verify logical devices
        logical_devices = tf.config.list_logical_devices('GPU')
        print(f"Logical GPUs: {logical_devices}")
    else:
        print("No GPUs found.")

setup_gpu(USE_GPU, MEMORY_GROWTH, GPU_MEM_LIMIT)

#####################
# Model Load
#####################

def bring_model():
    model = tf.keras.models.load_model('model.keras')
    model.summary()
    return model

model = bring_model()

#####################
# Inference Speed
#####################

def analyze_inference_speed(USE_GPU, model):
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

analyze_inference_speed(USE_GPU, model)