import tensorflow as tf

# TensorFlow 버전 확인
tf_version = tf.__version__

# CUDA 사용 가능한 하드웨어 리스트
gpu_devices = tf.config.list_physical_devices('GPU')
cuda_enabled = len(gpu_devices) > 0

# cuDNN 사용 가능 여부 확인
cudnn_enabled = False
if cuda_enabled:
    device = gpu_devices[0]
    device_details = tf.config.experimental.get_device_details(device)
    cudnn_enabled = 'compute_capability' in device_details

# 분산 데이터 병렬 처리 가능 여부 확인
strategy = tf.distribute.MirroredStrategy()
distributed_data_parallel_enabled = isinstance(strategy, tf.distribute.MirroredStrategy)

# 결과 출력
print(f"TensorFlow Version: {tf_version}")
print(f"CUDA Enabled: {cuda_enabled}")
print(f"cuDNN Enabled: {cudnn_enabled}")

# 분산 훈련 가능 여부 출력
if isinstance(strategy, tf.distribute.MirroredStrategy):
    print("Distributed training is enabled and you can use multiple GPUs for parallel data processing.")
else:
    print("Distributed training is not enabled. Consider setting up a distribution strategy for multi-GPU training.")
