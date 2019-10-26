import tensorflow as tf
import tokenization

vocab_pass = '../01_raw/cased_L-12_H-768_A-12/vocab.txt'

p_tokens = ['[Subject]','John','and','Michael','[equalTo]','genius','[when]','morning']
predict_ids = tokenization.convert_tokens_to_ids(tokenization.load_vocab(vocab_pass),p_tokens)
r_tokens = ['[Subject]','John','and','Michael','[equalTo]','smart','[when]','afternoon','and','evening']
real_ids = tokenization.convert_tokens_to_ids(tokenization.load_vocab(vocab_pass),r_tokens)
predict_tensor = tf.constant([[1,2,3],[98,1,6],[1,2,4],[22,1,6],[3,2,3],[7,1,6],[0,2,3],[11,1,9]],dtype = float)
real_tensor = tf.constant([[1,2,3],[12,8,1],[1,2,4],[12,8,1],[3,2,3],[12,8,1],[0,2,3],[12,8,1],[1,2,4],[12,8,1]],dtype=float)


def is_special_id(id):
  vocab = tokenization.load_vocab(vocab_pass)
  inv_vocab = {v:k for k,v in vocab.items()}
  special_tokens = []
  for token in vocab.keys():
    if len(token) >= 2:
      if token[0] == '[' and token[-1] == ']':
        special_tokens.append(token)
  if tokenization.convert_ids_to_tokens(inv_vocab, [id])[0] in special_tokens:
    return True
  else:
    return False

def add_segment_ids(ids):
  kg_segment_ids = [0]
  for i in range(len(ids)):
    if is_special_id(ids[i]):
      kg_segment_ids.append(kg_segment_ids[-1]+1)
    else:
      kg_segment_ids.append(kg_segment_ids[-1])
  return (ids, kg_segment_ids[1:])

def other_ids_loss(predict_ids, real_ids, predict_tensor, real_tensor):
  #スペシャルトークンごとにセグメントidを付与する
  (predict_ids, predict_kg_ids) = add_segment_ids(predict_ids)
  (real_ids, real_kg_ids) = add_segment_ids(real_ids)
  #predict_idsのスペシャルid１つに着目しインデックスを取得
  loss = 0
  #predictとrealのidの中から、同じスペシャルトークンを取得し、ペアを作り、ロスを算出する
  #ペアができないものは、すでにスペシャルトークンのロス関数で計算されているので無視する
  for i in range(len(predict_ids)):
    if is_special_id(predict_ids[i]):
      p_tpl1 = set()
      for l in range(len(predict_ids)):
        if predict_ids[l] == predict_ids[i]:
          p_tpl1.add(predict_kg_ids[l])
      p_tensors = []
      normalized_list = []
      for k2 in p_tpl1:
        for l3 in range(len(predict_ids)):
          if predict_kg_ids[l3] == k2:
            if not is_special_id(predict_ids[l3]):
              normalized_list.append(predict_tensor[l3,:])
      normalized_tensor = tf.stack(normalized_list)
      p_tensors.append(tf.math.l2_normalize(tf.reduce_sum(normalized_tensor,axis=0)))

      r_tpl1 = set()
      for l in range(len(real_ids)):
        if real_ids[l] == real_ids[i]:
          r_tpl1.add(real_kg_ids[l])
      r_tensors = []
      normalized_list = []
      for k2 in r_tpl1:
        for l3 in range(len(real_ids)):
          if real_kg_ids[l3] == k2:
            if not is_special_id(real_ids[l3]):
              normalized_list.append(real_tensor[l3,:])
      normalized_tensor = tf.stack(normalized_list)
      r_tensors.append(tf.math.l2_normalize(tf.reduce_sum(normalized_tensor,axis=0)))
      #ロスが最小になる組み合わせを貪欲法で探索
      #ペアができたものはロスに加算して
      #ペアができなかったものはspecial_ids_lossで加算されているので除くだけ
      vec_tmp = []
      for i,pt in enumerate(p_tensors):
        for j,rt in enumerate(r_tensors):
          pt = tf.reshape(pt,[1,3])
          rt = tf.reshape(rt,[1,3])
          inner_product = tf.matmul(pt, rt, transpose_b=True)
          inner_product = tf.reshape(inner_product,[1])
          vec_tmp.append((tf.subtract(tf.constant(1,dtype=float),inner_product),i,j))
      vec_tmp.sort()
      counter = 0
      while counter <= min([i,j]):
        loss += vec_tmp[counter][0]
        vec_tmp = [(loss,i,j) for (loss,i,j) in vec_tmp if i != vec_tmp[counter][1] and j != vec_tmp[counter][2]]
        counter += 1
    
  return loss



loss = other_ids_loss(predict_ids, real_ids, predict_tensor, real_tensor)

with tf.Session() as sess:
  losp = sess.run(loss)
  print(losp)
