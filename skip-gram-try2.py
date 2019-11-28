
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wordwVec_model1 import model
import time

import collections
import math
import os
import random
import zipfile
import json

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


row_index = 0
column_index = 0
category_video_dict = {}
category_dict = {}
category_len = 0
category_index = {}
interests_video_dict = {}
interests_dict = {}
interests_len = 0
interests_index = {}
vocabulary_size = 100000
vocalulary_size1 = 0
reverse_dictionary = {}
reverse_dictionary1 = {}
dictionary = {}
dictionary1 = {}
sentences = ''

def get_index_data():
    global sentences
    with open('index_sentences.txt','r') as f:
        sentences = f.readlines()

def read_data(filename):
    with open(filename, 'r') as f:
        content = f.read()
    data = content.strip().split(' ')
    return data

# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
  global vocabulary_size1
  count = [['UNK', -1]]
  count1 = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  count1.extend(collections.Counter(words).most_common())
  vocabulary_size1 = len(count1)
  dictionary = dict()
  dictionary1 = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  for word, _ in count1:
    dictionary1[word] = len(dictionary1)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  reverse_dictionary1 = dict(zip(dictionary1.values(),dictionary1.keys()))
  #print(len(reverse_dictionary1))
  #print(reverse_dictionary1[127652])
  return data, count, dictionary, reverse_dictionary, dictionary1, reverse_dictionary1


def index_data():
   with open('sentences.txt','r') as f:
       contents = f.readlines()

   with open('index_sentences.txt','w') as ff:
       for line in contents:
           line = line.strip().split(' ')
           for word in line:
               ff.write(str(dictionary1[word]))
               ff.write(' ')
           ff.write('\n')


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global row_index
  global column_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)

  if (len(sentences[row_index].strip().split(' ')) - column_index) >= span:
      for idx in range(span):
          buffer.append(int(sentences[row_index].strip().split(' ')[column_index + idx]))
      column_index = column_index + 1
  else:
      while((len(sentences[row_index].strip().split(' ')) - column_index) < span):
          row_index = (row_index + 1) % len(sentences)
          column_index = 0
      for idx in range(span):
          buffer.append(int(sentences[row_index].strip().split(' ')[column_index + idx]))
      column_index = column_index + 1

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]

    while((len(sentences[row_index].strip().split(' ')) - column_index) < span):
        row_index = (row_index + 1) % len(sentences)
        column_index = 0
    for idx in range(span):
        buffer.append(int(sentences[row_index].strip().split(' ')[column_index + idx]))
    column_index = column_index + 1
  return batch, labels

# Step 4: Build and train a skip-gram model.
def process_category(filename):
  global category_video_dict
  global category_dict
  global category_len
  global category_index
  with open(filename,'r') as f:
      category_video_dict = json.load(f)
  for k,v in category_video_dict.items():
      if v not in category_dict:
          category_dict[v] = 1
      else:
          category_dict[v] = category_dict[v] + 1
  category_len = len(category_dict.keys())
  i = 1
  for k,v in category_dict.items():
      category_index[k] = i
      i = i + 1

def process_interests(filename):
  global interests_video_dict
  global interests_dict
  global interests_len
  global interests_index
  with open(filename,'r') as f:
      interests_video_dict = json.load(f)
  for k,v in interests_video_dict.items():
      for s in v:
          if s not in interests_dict:
              interests_dict[s] = 1
          else:
              interests_dict[s] = interests_dict[s] + 1
  interests_len = len(interests_dict.keys())
  i = 1
  for k,v in interests_dict.items():
      interests_index[k] = i
      i= i + 1


def prepare_data():
  global reverse_dictionary
  global reverse_dictionary1
  global dictionary
  global dictionary1
  filename = 'words.txt'
  words = read_data(filename)
  print('Data size', len(words))
  data, count, dictionary, reverse_dictionary, dictionary1, reverse_dictionary1= build_dataset(words)
  print(len(dictionary1.keys()))
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
  index_data()
  get_index_data()
  process_category('re_video_category.json')
  process_interests('re_video_interests.json')

# Step 5: Begin training.
def train():
  batch_size = 128
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 1       # How many words to consider left and right.
  num_skips = 2         # How many times to reuse an input to generate a label.
  #vocabulary_size = 50000
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  num_sampled = 64    # Number of negative examples to sample.
  prepare_data()
  sd1_size = category_len + 1
  sd2_size = interests_len + 1
  num_interests = 3
  num_steps = 600000

  graph = tf.Graph()
  with graph.as_default():
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_inputs_sd1 = tf.placeholder(tf.int32, shape=[batch_size])
      train_inputs_sd2 = tf.placeholder(tf.int32, shape=[batch_size, num_interests])
      train_mask = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
      train_inputs_all = []
      train_inputs_all.append(train_inputs)
      train_inputs_all.append(train_inputs_sd1)
      train_inputs_all.append(train_inputs_sd2)
      train_inputs_all.append(train_mask)
      train_inputs_all.append(valid_dataset)

      word2Vec_model = model(batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, sd2_size, embedding_size, num_sampled)
      embedding_all = word2Vec_model.model_inferences(train_inputs_all)
      loss = word2Vec_model.loss(embedding_all, train_labels)
      # Add variable initializer.
      init = tf.global_variables_initializer()
      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # training process
      with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
          init.run()
          print("Initialized")
          var_list = tf.trainable_variables()
          saver = tf.train.Saver(var_list, max_to_keep = 20)
          average_loss = 0
          batch_inputs = np.ndarray(shape=(batch_size),dtype=np.int32)
          for step in xrange(num_steps):
              batch_inputs, batch_labels = generate_batch(
                  batch_size, num_skips, skip_window)
              #for i in range(batch_size):
              #    if batch_inputs1[i] not in reverse_dictionary:
              #        batch_inputs[i] = 0
              #    else:
              #        batch_inputs[i] = batch_inputs1[i]

              batch_inputs_sd1 = np.ndarray(shape=(batch_size),dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in category_video_dict:
                      batch_inputs_sd1[i] = 0
                  else:
                          #print('inputs',batch_inputs[i])
                          #print('video:',reverse_dictionary[batch_inputs[i]])
                          #print('category:',category_video_dict[reverse_dictionary[batch_inputs[i]]])
                          #print('index:',category_index[category_video_dict[reverse_dictionary[batch_inputs[i]]]])
                      batch_inputs_sd1[i] = category_index[category_video_dict[reverse_dictionary1[batch_inputs[i]]]]
              batch_inputs_sd2 = np.ndarray(shape=(batch_size, num_interests),dtype=np.int32)
              mask = np.ndarray(shape=(batch_size),dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in interests_video_dict:
                      for j in range(num_interests):
                          batch_inputs_sd2[i][j] = 0
                      mask[i] = 1

                  else:
                      interest_list = interests_video_dict[reverse_dictionary1[batch_inputs[i]]]
                      len_interest = len(interest_list)
                      index = 0
                      if len_interest > num_interests:
                          index = num_interests
                      else:
                          index = len_interest
                      mask[i] = index
                      for j in range(index):
                          batch_inputs_sd2[i][j] = interests_index[interest_list[j]]
                      for j in range(index, num_interests):
                          batch_inputs_sd2[i][j] = 0
              feed_dict = {train_inputs: batch_inputs, train_inputs_sd1:batch_inputs_sd1, train_inputs_sd2:batch_inputs_sd2,
                train_mask:mask, train_labels: batch_labels}

                 # We perform one update step by evaluating the optimizer op (including it
                 # in the list of returned values for session.run()
              _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
              #print(loss_val)
              average_loss += loss_val

              if step % 10000 == 0:
                  if step > 0:
                      average_loss /= 10000
                  # The average loss is an estimate of the loss over the last 2000 batches.
                  print("Average loss at step ", step, ": ", average_loss)
                  average_loss = 0

              # Note that this is expensive (~20% slowdown if computed every 500 steps)
              if step % 50000 == 0:
                  '''
                  sim = similarity.eval()
                  for i in xrange(valid_size):
                      valid_word = reverse_dictionary1[valid_examples[i]]
                      top_k = 8  # number of nearest neighbors
                      nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                      log_str = "Nearest to %s:" % valid_word
                      for k in xrange(top_k):
                          close_word = reverse_dictionary1[nearest[k]]
                          log_str = "%s %s," % (log_str, close_word)
                          print(log_str)
                  '''
                  saver.save(session, 'checkpoints-hour1/model.ckpt', global_step = step)
                  print('saved!')
          #final_embeddings = normalized_embeddings.eval()

def get_eval_data():
    data = []
    data_vid = []
    for k, v in dictionary1.items():
        if(k == 'UNK'):
            print('!!!!!!!!!')
        if k != 'UNK':
            data_vid.append(k)
            data.append(v)
    vid_len = len(dictionary1.keys()) - 1
    return data, data_vid, vid_len

def predict():
    batch_size = 1
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    num_sampled = 64
    num_interests = 3
    prepare_data()
    sd1_size = category_len + 1
    sd2_size = interests_len + 1
    data, data_vid, total = get_eval_data()
    eval_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_sd1 = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_sd2 = tf.placeholder(tf.int32, shape=[batch_size, num_interests])
    eval_mask = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_all = []
    eval_inputs_all.append(eval_inputs)
    eval_inputs_all.append(eval_inputs_sd1)
    eval_inputs_all.append(eval_inputs_sd2)
    eval_inputs_all.append(eval_mask)
    word2Vec_model = model(batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, sd2_size, embedding_size, num_sampled)
    embedding_all = word2Vec_model.model_inferences(eval_inputs_all)
    var_list = tf.trainable_variables()
    saver = tf.train.Saver(var_list)
    weight_file = './checkpoints-hour/model.ckpt-450000'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if weight_file is not None:
        saver.restore(sess, weight_file)
        print('Restoring weight file from:%s'%weight_file)
    f = open('result-hour.txt','w+')
    start = time.time()
    vec = []
    batch_inputs = np.ndarray(shape=(batch_size),dtype=np.int32)
    #batch_inputs1 = np.ndarray(shape=(batch_size),dtype=np.int32)
    batch_inputs_sd1 = np.ndarray(shape=(batch_size),dtype=np.int32)
    batch_inputs_sd2 = np.ndarray(shape=(batch_size, num_interests), dtype=np.int32)
    batch_mask = np.ndarray(shape=(batch_size), dtype=np.int32)
    for i in range(total):
        batch_inputs[0] = data[i]
        #if data[i] not in reverse_dictionary:
        #    batch_inputs[0] = 0
        #else:
        #    batch_inputs[0] = data[i]
        if reverse_dictionary1[data[i]] not in category_video_dict:
            batch_inputs_sd1[0] = 0
        else:
            batch_inputs_sd1[0] = category_index[category_video_dict[reverse_dictionary1[data[i]]]]
        
        if reverse_dictionary1[data[i]] not in interests_video_dict:
            for j in range(num_interests):
                batch_inputs_sd2[0][j] = 0
            batch_mask[0] = 1
        else:
            interest_list = interests_video_dict[reverse_dictionary1[data[i]]]
            len_interest = len(interest_list)
            index = 0
            if len_interest > num_interests:
                index = num_interests
            else:
                index = len_interest
            batch_mask[0] = index
            for j in range(index):
                batch_inputs_sd2[0][j] = interests_index[interest_list[j]]
            for j in range(index, num_interests):
                batch_inputs_sd2[0][j] = 0
        feed_dict1 = {eval_inputs:batch_inputs, eval_inputs_sd1:batch_inputs_sd1, eval_inputs_sd2:batch_inputs_sd2, eval_mask:batch_mask}
        embedding_single = sess.run(embedding_all,feed_dict = feed_dict1)
        f.write(data_vid[i]+'\n')
        vec.append(embedding_single)
    vec = np.array(vec)
    np.save('test-hour.npy',vec)
    end = time.time() - start
    print(end)
predict()
