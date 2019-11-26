import tensorflow as tf

a = tf.constant([[[1,2,3],[14,15,16],[7,8,9],[10,11,12]],[[2,3,4],[25,26,27],[8,9,10],[11,12,13]],[[3,4,5],[36,37,38],[9,10,11],[12,13,14]]])
a_transpose = tf.transpose(a, perm=[2, 1, 0])
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a[:, :, 0]))
    print(sess.run(tf.transpose(a_transpose[:, :, 0])))
