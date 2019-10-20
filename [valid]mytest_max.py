import tensorflow as tf

input_tensor = tf.constant([[1,2,23,4],[5,6,8,9]])

m = tf.reduce_max(
    input_tensor,
    axis=1
)

with tf.Session() as sess:
  test = sess.run(m)
  print(test)