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

#[Memo]定義
flags = tf.flags
#[Memo]？？
FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", "../01_raw/cased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

tokens = ["[CLS]","I","am","Sam","[SEP]"]

input_ids = tokenizer.convert_tokens_to_ids(tokens)

print(input_ids)