# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_text as text

from tfx.components.trainer.executor import TrainerFnArgs

_TRAIN_BATCH_SIZE = 512
_TRAIN_DATA_SIZE = 51200
_EVAL_BATCH_SIZE = 512
_LABEL_KEY = "sentiment"

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
          filenames,
          compression_type='GZIP')

def _tokenize(stringA):
    """Tokenize the two sentences and insert appropriate tokens"""
    tokenizer = text.BertTokenizer(
            "vocab.txt",
            token_out_type=tf.string,
            )

    stringA = tf.squeeze(stringA)
    idA = tokenizer.tokenize(stringA)
    #idB = tokenizer.tokenize(stringB)
    return idA.merge_dims(-2, -1).to_sparse() 

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  test = tf.constant(['test sentence'])
  tokenizer = text.BertTokenizer(
           "vocab.txt",
            token_out_type=tf.string,
            )
  output = tokenizer.tokenize(test)
  return inputs 
