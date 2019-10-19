import tensorflow as tf

const0 = tf.reshape([[0.7,0.12,0.4,0.2],[0.6,0.12,0.4,0.3]],[2,4])
const1 = tf.nn.log_softmax(const0,axis=0)

with tf.Session() as sess:
    const1_result = sess.run(const1)
    print(const1_result)

