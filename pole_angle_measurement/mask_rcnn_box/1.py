import tensorflow as tf
version=tf.__version__  #输出tensorflow版本
gpu_ok=tf.test.is_gpu_available()  #输出gpu可否使用（True/False）
print("tf version:",version,"\nuse GPU:",gpu_ok)
tf.test.is_built_with_cuda()  # 判断CUDA是否可用（True/False）

#
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hellow = tf.constant('hellow')
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(log_device_placement=True))
print(sess.run(hellow))
