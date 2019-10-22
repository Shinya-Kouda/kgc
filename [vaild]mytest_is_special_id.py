import tokenization
import tensorflow as tf

vocab_pass = '../01_raw/cased_L-12_H-768_A-12/vocab.txt'

def is_special_id2(id):
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

vocab = tokenization.load_vocab(vocab_pass)
inv_vocab = {v:k for k,v in vocab.items()}

id = 28995
token2 = tokenization.convert_ids_to_tokens(inv_vocab, [id])
print(token2[0])
print(is_special_id2(id))

id = 28996
token2 = tokenization.convert_ids_to_tokens(inv_vocab, [id])
print(token2[0])
print(is_special_id2(id))