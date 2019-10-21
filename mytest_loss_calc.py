import tensorflow as tf

p_tensors = tf.constant([[1,2,3],[98,1,6]])
r_tensors = tf.constant([[4,5,6],[12,8,1]])
loss = tf.matmul(p_tensors, r_tensors, transpose_b=True)
loss = tf.reduce_min(loss)


loss = []
for i,pt in enumerate(p_tensors):
  for j,rt in enumerate(r_tensors):
    loss.append(1 - tf.matmul(pt, rt, transpose_b=True))
loss.sort()
loss = sum(loss[:(min([i,j])-1)]]


with tf.Session() as sess:
  losp = sess.run(loss)
  print(losp)