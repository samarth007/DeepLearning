import tensorflow as tf


kernel=tf.constant([[1,0,1],[1,0,1],[1,0,1]])
# print(kernel)
kernel=tf.reshape(kernel,[*kernel.shape,1,1])
# print(kernel)
kernel=tf.cast(kernel,dtype=tf.float32)
print(kernel)
# tf.nn.conv2d()

