import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# k=tf.constant([[2,6,3],[6,9,1]],dtype=tf.float32,name='kernel')
# i=tf.constant([[21,4,11,6],[9,2,4,1],[11,21,5,3]],dtype=tf.float32,name='Image')

# image=tf.reshape(i,[1,3,4,3])

# tf.squeeze(tf.nn.conv2d(image,kernel,strides=[1,1,1,1],padding='VALID'))

var=tf.Variable(4)
print(var)