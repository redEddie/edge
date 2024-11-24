import os
import tensorflow as tf

# configs
USE_GPU = True
GPU_MEM_LIMIT = 512
THREADS = None

BATCH_SIZE = 128
EPOCHS = 10

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

    # modeling(functional API)

    # cifar10 data load
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

    num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)  # category화 
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.models.Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(tf.keras.layers.Conv2D(48, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(96, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS, 
                        validation_data=(x_test, y_test))
    model.save('model.keras')