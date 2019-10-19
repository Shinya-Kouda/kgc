from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections#[Memo]新しいデータ型が使える
import json#[Memo]json形式のデータを扱えるようにする
import math#[Memo]数学関数を使える
import os
import random#[Memo]乱数を生成する
import modeling#[Memo]同フォルダ内ファイル。モデリングをする
import optimization#[Memo]同フォルダ内ファイル。最適化をする
import tokenization#[Memo]同フォルダ内ファイル。トークン化する
import six#[Memo]python2と3の差を少し埋めてくれる
import tensorflow as tf

class KGExample(object):#objectクラスを継承

  def __init__(self,
               id,
               nlr_text,
               nlr_tokens,
               kgr_text,
               kgr_tokens
               ):
    self.id = id
    self.nlr_text = nlr_text
    self.nlr_tokens = nlr_tokens
    self.kgr_text = kgr_text
    self.kgr_tokens = kgr_tokens#いらない？

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "id: %s" % (tokenization.printable_text(self.id))
    s += ", nlr_text: %s" % (
        tokenization.printable_text(self.nlr_text))
    s += ", nlr_tokens: [%s]" % (" ".join(self.nlr_tokens))
    s += ", kgr_text: %s" % (
        tokenization.printable_text(self.kgr_text))
    s += ", kgr_tokens: [%s]" % (" ".join(self.kgr_tokens))
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               tokens,
               input_ids,
               input_mask,
               segment_ids
               ):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids


def read_kg_examples(input_file, is_training):
  """Read a KG json file into a list of KGExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]#[Memo]ちょっとわからないけどデータを抽出してる

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for data in input_data:
    
    #文字列を単語の列に変換する
    nlr_text = data["nlr"]
    nlr_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in nlr_text:
      if is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          nlr_tokens.append(c)
        else:
          nlr_tokens[-1] += c
        prev_is_whitespace = False
      char_to_word_offset.append(len(nlr_tokens) - 1)

    #kgrの方も同様の処理
    id=data["id"]
    kgr_text = data["kgr"]
    kgr_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in kgr_text:
      if is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          kgr_tokens.append(c)
        else:
          kgr_tokens[-1] += c
        prev_is_whitespace = False
      char_to_word_offset.append(len(kgr_tokens) - 1)
    
    if is_training:
      #もしkgrにあたるものがなかったらエラー
      if (len(kgr_tokens) < 1):
        raise ValueError(
            "knowledge graph representation is " + str(len(kgr_tokens)))

    #ここまでで得たデータをKGExampleクラスのインスタンスに変換する
    example = KGExample(
        id=id,
        nlr_text=nlr_text,
        nlr_tokens=nlr_tokens,
        kgr_text=kgr_text,
        kgr_tokens=kgr_tokens#いらない？
        )
    examples.append(example)

  return examples

def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_nlr_length,
                                 max_kgr_length,
                                 is_training,
                                 output_fn
                                 ):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000#exampleごとに１ずつ足してく
  max_seq_length = max_nlr_length + max_kgr_length + 3

  #exampleはKGExampleクラスのインスタンス
  for (example_index, example) in enumerate(examples):
    #tokenizerはtokenization.FullTokenizerやtokenization.BasicTokenizerの返り値
    #tokenizerは使わずexample.kgr_tokensでいいと思う
    #kgr_tokens = tokenizer.tokenize(example.kgr_text)
    nlr_tokens = example.nlr_tokens
    kgr_tokens = example.kgr_tokens

    #ナレッジグラフ表現の長さを制限する
    if len(nlr_tokens) > max_nlr_length:
      nlr_tokens = nlr_tokens[0:max_nlr_length]
    if len(kgr_tokens) > max_kgr_length:
      kgr_tokens = kgr_tokens[0:max_kgr_length]

    #InputFeaturesクラスのインスタンスを１つ定義する
    #doc_spansはdoc_tokensでいいと思うけどわからん
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in nlr_tokens:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in kgr_tokens:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    #Falseの場合、例外メッセージを出力する
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #ログ出力
    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info(
          "input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    #InputFeaturesクラスのインスタンスを定義する
    feature = InputFeatures(
        unique_id=unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids
        )

    # Run callback
    #output_fn(feature)
    print(feature.tokens)

    unique_id += 1

class FeatureWriter(object):
  """Writes InputFeature to TF example file."""
  #オブジェクトxxxを定義してxxx.process_feature(feature)を実行すると書き込む
  #そのあとxxx.close()を実行する

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()

train_writer = FeatureWriter(
        filename="converter_test",
        is_training=True)
vocab_file = "../01_raw/cased_L-12_H-768_A-12/vocab.txt"
convert_examples_to_features(examples = read_kg_examples("test.json", True),
                                 tokenizer=tokenization.FullTokenizer(vocab_file=vocab_file,do_lower_case=False),
                                 max_nlr_length=50,
                                 max_kgr_length=30,
                                 is_training=True,
                                 output_fn = train_writer.process_feature
                                 )

#train_writer.close()