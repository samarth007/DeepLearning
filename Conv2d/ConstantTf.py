from dis import show_code
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

# a=tf.constant(1)
# b=tf.constant(3)
# c=tf.add(a,b)
# with tf.compat.v1.Session() as sess:
#     print(sess.run(c))


aa=tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)   #matrix
ab=tf.constant(32,dtype=tf.float32)  #Scalar
print(aa)
# print(ab)
# print(aa[:,:2])
# print(aa+10)

arr=np.array([[2,3],[4,5]])
print(tf.constant(arr))
