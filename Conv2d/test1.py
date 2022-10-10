import tensorflow.compat.v1 as tf   #explicity using version 1.0
tf.disable_v2_behavior()   #disabling version 2.0 behavior


learning_rate = 0.0001
epochs = 10
batch_size = 128
test_valid_size = 128
n_classes = 5
dropout = 0.75


weights = {
'wc1': tf.Variable(tf.random.normal([5, 5, 1, 32])),
'wc2': tf.Variable(tf.random.normal([5, 5, 32, 64])),
'wd1': tf.Variable(tf.random.normal([7*7*64, 1024])),
'out': tf.Variable(tf.random.normal([1024, n_classes]))}
biases = {
'bc1': tf.Variable(tf.random.normal([32])),
'bc2': tf.Variable(tf.random.normal([64])),
'bd1': tf.Variable(tf.random.normal([1024])),
'out': tf.Variable(tf.random.normal([n_classes]))}

x = tf.placeholder(tf.float32, [None, 80, 240, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

layer_1 = tf.nn.conv2d(x, weights['wc1'], strides = [1, 1, 1, 1], padding = 'SAME')
layer_1 = tf.nn.bias_add(layer_1, biases['bc1'])
layer_1 = tf.nn.max_pool(layer_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

layer_2 = tf.nn.conv2d(layer_1, weights['wc2'], strides = [1, 1, 1, 1], padding = 'SAME')
layer_2 = tf.nn.bias_add(layer_2, biases['bc2'])
layer_2 = tf.nn.max_pool(layer_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

layer_3 = tf.reshape(layer_2, [-1, weights['wd1'].get_shape().as_list()[0]])
layer_3 = tf.add(tf.matmul(layer_3, weights['wd1']), biases['bd1'])
layer_3 = tf.nn.relu(layer_3)
layer_3 = tf.nn.dropout(layer_3, dropout)

logits = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   

init = tf.global_variables_initializer()

save_file_2 = './train_model2.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # for epoch in range(epochs):
    #     for batch_features, batch_labels in batches(batch_size, count_X_train, count_y_train):
    #         sess.run(optimizer, feed_dict = {x:batch_features, y:batch_labels, keep_prob:dropout})

    # saver.save(sess, save_file_2)
    # print("")
    # print("trained model_2 saved")
    print(sess.run(weights['wc1']))
    # test_acc = sess.run(accuracy, feed_dict = {x: count_X_test[:test_valid_size], y: count_y_test[:test_valid_size], keep_prob:1.})
    # print('Testing Accuracy: {}'.format(test_acc))


