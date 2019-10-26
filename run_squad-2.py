# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

#[Memo]python2系のモジュールをpython3系で使うとき__future___を使う
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

#[Memo]tf.flags(or tf.app.flags)は、pythonファイルをコマンドライン実行時に引数を受け取ることができる。
#[Memo]モデルのパラメータを定義するのに使う。
#[Memo]説明付きなので、とても勉強になる！！！

#[Memo]定義
flags = tf.flags
#[Memo]？？
FLAGS = flags.FLAGS
#[Memo](tf.flags.)DEFINE_(type)("(引数名)",(初期値),"(説明)")
## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "KG json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "KG json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
#[todo]数をいい感じに調整する
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
#文書を読み込むsliding windowのストライド
flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")
#max_seq_lengthとはちがう。クエリのほう
flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
#[Memo]trainのバッチサイズと変える意味は？
flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
#[Memo]warmupは、学習の初期に学習率を高めにする方法。ただ以下の設定でどのような挙動になるのかはわからない
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
#1ステップで呼ばれるestimatorの回数
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
#これはなんなのかわからない
flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")
#[Memo]answerの最大の長さ
flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


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
               segment_ids,
               input_tensors
               ):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_tensors = input_tensors


def read_kg_examples(input_file, is_training):
  """Read a KG json file into a list of KGExample."""
  #自然言語表現(Natural Language Representation)と
  #ナレッジグラフ表現(Knowledge Graph Representation)を持つ
  #ngrは、学習用データにはあって予測用データはNone

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
      if (len(data["kgr"]) != 1):
        raise ValueError(
            "knowledge graph representation is =0 or <=2.")

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
  #返り値はないが、この関数を実行するとInputFeaturesクラスのインスタンスが読み込まれる

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
    
    #bertを呼び出して、トークンをエンコードする
    #スペシャルトークンは全部０にする
    ##理由１：スペシャルトークンは本家BERTのvocabにないから
    ##理由２：あとで１つのスペシャルトークンに対するトークンたちのテンソルを足しあげるから
    #input_tensors

    #InputFeaturesクラスのインスタンスを定義する
    feature = InputFeatures(
        unique_id=unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        input_tensors=None
        )

    # Run callback
    output_fn(feature)

    unique_id += 1

#いらない
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)



#いらない
def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


#ここはTransformerにする
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,#たぶんこの設定にしたがってbertを呼び出すということ
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()#Bertの最終層

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])

  #ここをTransformerにする
  """
  output_weights = tf.get_variable(
      #/はスコープの区切りを表す。だからcls/squad/output_weightsはclsスコープのsquadスコープのoutput_weightsという変数を表す
      #変数がないときは定義し、ある時はそれを呼び出す
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      #/はスコープの区切りを表す。だからcls/squad/output_weightsはclsスコープのsquadスコープのoutput_weightsという変数を表す
      #変数がないときは定義し、ある時はそれを呼び出す
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)
  """
  #Transformer層
  #bertの中のtransformerよりずっとスペック低くしている
  transformer_outputs = modeling.transformer_model(input_tensor=final_hidden_matrix,
                              attention_mask=None,
                              hidden_size=5,
                              num_hidden_layers=2,
                              num_attention_heads=2,
                              intermediate_size=20,
                              intermediate_act_fn=modeling.gelu,
                              hidden_dropout_prob=0.1,
                              attention_probs_dropout_prob=0.1,
                              initializer_range=0.02,
                              do_return_all_layers=False)#現状Falseのみ

  #線型層
  output_weights = tf.get_variable(
      #/はスコープの区切りを表す。だからcls/squad/output_weightsはclsスコープのsquadスコープのoutput_weightsという変数を表す
      #変数がないときは定義し、ある時はそれを呼び出す
      "cls/squad/output_weights", [30000, 5],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias = tf.get_variable(
      #/はスコープの区切りを表す。だからcls/squad/output_weightsはclsスコープのsquadスコープのoutput_weightsという変数を表す
      #変数がないときは定義し、ある時はそれを呼び出す
      "cls/squad/output_bias", [30000], initializer=tf.zeros_initializer())
  logits = tf.matmul(transformer_outputs, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  #max
  ids = tf.reduce_max(logits,axis=0)

  #Transformerのテンソルとidを出力。損失を測るのに両方使うため
  return (ids,transformer_outputs)


#引数を与えたらmodel_fnを返す関数
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  #引数を与えたらoutput_specを返す関数
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    #ログ出力
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    #featuresからデータ取得
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    input_tensor = features["input_tensor"]#いらない

    #trainかどうか
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #モデル作成
    (estimated_ids,estimated_tensor) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    #わからない
    tvars = tf.trainable_variables()

    #checkpointの変数を読み込む
    #tpu使わない場合はここで読み込み、使う場合は読み込む関数を作る
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    #ログ出力
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    #output_specを出力。trainかpredictかで変わる
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      #スペシャルトークンかどうか判断する関数
      #special_idの辞書オブジェクトを作って、そこに属すならtrue
      def is_special_id(id):
        vocab = tokenization.load_vocab(FLAGS.vocab_file)
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

      #special ids loss
      #predictのスペシャルトークンAがrealにもある場合、そのトークンによる損失は０、ない場合は１
      #スペシャルトークンAが複数存在する場合、その個数が同じなら、それらのトークンによる損失は０、同じでないときはその個数
      #これをすべてのトークンに渡り加算する
      def special_ids_loss(predict_ids, real_ids):
        p_specials = {p for p in predict_ids if is_special_id(p)}
        r_specials = {r for r in real_ids if is_special_id(r)}
        counter = len(p_specials ^ r_specials)
        return counter
      #other ids loss
      #等しいスペシャルトークンの後ろのトークン同士の差をロスとする
      #１つのトークンはベクトルで表されるので、ベクトルの間の角でロスを定義する
      #角が０の時ロスも０
      #角が９０度のときロスは１
      #角が１８０度の時ロスは２になるようにする
      #等しいトークンが複数あるときは、それらの間でロスが最も低い組み合わせを選ぶ
      #等しいトークンがないときは、ロスには加算されない（special_ids_lossで加算済）
      #等しいトークンがありかつ数が異なるときは、ロスが小さいペアを優先的に結び、ロスが大きいペアは
      #special_ids_lossで加算される
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

      #lossを計算する関数
      #かなり作りこみが必要。たぶん自分で考える必要がある
      def compute_loss(predict_ids, real_ids, predict_tensor, real_tensor):
        loss = special_ids_loss(predict_ids, real_ids) + other_ids_loss(predict_ids, real_ids, predict_tensor, real_tensor)
        return loss

      #loss
      loss = compute_loss(estimated_ids,input_ids,estimated_tensor,input_tensor)

      #オプティマイザー
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      #スペック
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
          
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = transformer_outputs
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


#引数を与えたらinput_fnを返す関数
def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  #paramsを入れたらデータを返す関数
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


#予測結果をこのオブジェクトに格納して、下のwritepredictionで書き出すのでここで定義する
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


#最終的な予測結果をjsonに書き出す関数
def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  #辞書か関数のようなものを作っている
  example_index_to_features = collections.defaultdict(list)#これは何だろう。関数？辞書？
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  #結果のユニークな集合
  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  #予測結果を集めるタプル？
  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  #何かわからない。何らかのモデリング結果ぽい
  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  #全てのサンプルにわたって以下を行う
  for (example_index, example) in enumerate(all_examples):
    #サンプルのインデックスのfeaturesをとってくる
    features = example_index_to_features[example_index]

    #準備っぽい
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score

    #全てのfeatureにわたり以下を行う
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    #もしsquad version1.1だったら以下を挿入
    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    #ソート
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    #
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    #すべてのprelim_predictionsの数だけ以下を行う
    for pred in prelim_predictions:
      #best sizeを超えたらforを抜ける
      if len(nbest) >= n_best_size:
        break

      #featureを一つ抽出
      feature = features[pred.feature_index]
      #featureがnon-null予測だったら以下を行う
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  #jsonファイルとして書く
  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


#最終的なテキストを返す
def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


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

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  #ログ
  tf.logging.set_verbosity(tf.logging.INFO)
  #bertの設定
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  #フラグの評価
  validate_flags_or_throw(bert_config)
  #ディレクトリ作成
  tf.gfile.MakeDirs(FLAGS.output_dir)
  #トークナイザー
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  #tpuあるなら設定
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  #これはtensorflowがversion2のときということかな
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  #モデリングの設定
  #モデルオブジェクトの出力先などもここで設定する
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  #学習するときの事前処理、シャッフルする
  if FLAGS.do_train:
    train_examples = read_kg_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    #たぶん、本データでシャッフルすると、データがでかすぎるということ
    #なので、レコード数が同じで、まだ使わないデータを除いてシャッフルしているんではないかと思う
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  #モデル定義
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  #学習するときのメインの処理
  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(#この関数自身に返り値はないが、別の関数が実行され結果としてInputFeaturesクラスのインスタンスができる
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)#output_fnは関数だからprocess_featureにfeatureを与えなくていい
        #convert_examples_to_featuresのなかでfeatureを与えて実行している模様
        #なので、下でcloseしている
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)#ここで学習している
    #estimatorの使い方は、https://cloud.google.com/tpu/docs/using-estimator-api?hl=ja
    #が参考になる。ここを見る限り、model_fnでモデル構造を決めているようである

  #予測するとき
  if FLAGS.do_predict:
    eval_examples = read_kg_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      #書き込みたいものをforループで集める
      tokens = tokenizer.convert_ids_to_tokens[result["ids"]]
      # unique_id = int(result["unique_ids"])
      # start_logits = [float(x) for x in result["start_logits"].flat]
      # end_logits = [float(x) for x in result["end_logits"].flat]
      # all_results.append(
      #     RawResult(
      #         unique_id=unique_id,
      #         start_logits=start_logits,
      #         end_logits=end_logits))

    # output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    # output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    # output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    # write_predictions(eval_examples, eval_features, all_results,
    #                   FLAGS.n_best_size, FLAGS.max_answer_length,
    #                   FLAGS.do_lower_case, output_prediction_file,
    #                   output_nbest_file, output_null_log_odds_file)

    #tokens,eval_examplesをナレッジグラフのjsonfileとして書き出す

    


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
