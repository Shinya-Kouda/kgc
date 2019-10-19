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

_DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
doc_spans = []
start_offset = 0

all_doc_tokens = 'adsfasdfasdfsdfzxcvasdfadfzxqwerascvadrqwcaasdfasdfcvaeracfahnklkkmnsdfnakd'
start_offset = 2
max_tokens_for_doc = 40
doc_stride = 10
while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
        length = max_tokens_for_doc
    doc_spans.append(_DocSpan(start=start_offset, length=length))#メインの処理
    if start_offset + length == len(all_doc_tokens):
        break
    start_offset += min(length, doc_stride)

print(doc_spans)