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
  
    print(kgr_tokens)

  return examples

print(read_kg_examples("test.json", False))