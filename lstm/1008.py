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
# Load CIFAR-10 data from TensorFlow
#####################
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]
    return (x_train, y_train), (x_test, y_test), (img_height, img_width, channel)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test), input_shape = load_cifar10_data()

#####################
# AlexNet Model
#####################
def create_alexnet(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()
    return model

model = create_alexnet(input_shape, 10)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

#####################
# Model Training with Timing
#####################
print("Training model...")
start_time = time.time()  # Start the clock

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

end_time = time.time()  # End the clock
print("Model training completed.")
training_time = end_time - start_time  # Calculate total training time

# Print the training duration
print(f"Training Time: {training_time:.2f} seconds")

# Save model
model.save('alexnet_cifar10.keras')