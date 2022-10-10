import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

wc=tf.Variable([[2,3,4],[6,8,11]])
init=tf.compat.v1.initialize_all_variables()

with tf.compat.v1.Session() as s:
    s.run(init)
    print(s.run(wc))