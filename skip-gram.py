
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
import json

import collections
import math
import os
import random
import zipfile
import json
import faiss
import sys
from tensorflow import gfile
from faiss import normalize_L2

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

tags_video_dict = {}
tags_dict = {}
tags_len = 0
tags_index = {}

accountCategory_video_dict = {}
accountCategory_dict = {}
accountCategory_len = 0
accountCategory_index = {}

accountClassify_video_dict = {}
accountClassify_dict = {}
accountClassify_len = 0
accountClassify_index = {}

vocabulary_size = 170000
vocalulary_size1 = 0
reverse_dictionary = {}
reverse_dictionary1 = {}
dictionary = {}
dictionary1 = {}
sentences = ''
test_result = {}

def get_index_data():
    global sentences
    with open('train_data/index_sentences.txt','r') as f:
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
  #content = open('words.txt','r').readlines()[0].strip()
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  count1.extend(collections.Counter(words).most_common())
  #print(count)
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
      #print(unk_count)
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  reverse_dictionary1 = dict(zip(dictionary1.values(),dictionary1.keys()))
  #print(count[0][1])
  s = open('train_data/video_index.json','w')
  s.write(json.dumps(reverse_dictionary1))
  return data, count, dictionary, reverse_dictionary, dictionary1, reverse_dictionary1

#build_dataset('words.txt')

def index_data():
   with open('train_data/sentences.txt','r') as f:
       contents = f.readlines()

   with open('train_data/index_sentences.txt','w') as ff:
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
      if '/' in v:
          category_video_dict[k] = v.split('/')[1]

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

  for k, v in interests_video_dict.items():
      v = v.strip().split(',')
      tmp = []
      for interest in v:
          tmp.append(interest)
      interests_video_dict[k] = tmp

  for k,v in interests_video_dict.items():
      #v = v.strip().split(',')
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
'''
def process_tags(filename):
    global tags_video_dict
    global tags_dict
    global tags_len
    global tags_index
    with open(filename, 'r') as f:
        tags_video_dict = json.load(f)
    for k,v in tags_video_dict.items():
        v = v.strip().split(',')
        tmp = []
        for interest in v:
            tmp.append(interest)
        tags_video_dict[k] = tmp

    for k,v in tags_video_dict.items():
        for s in v:
            if s not in tags_dict:
                tags_dict[s] = 1
            else:
                tags_dict[s] = tags_dict[s] + 1
    tags_len = len(tags_dict.keys())
    i = 1
    for k,v in tags_dict.items():
        tags_index[k] = i
        i = i + 1

def process_accountCategory(filename):
    global accountCategory_video_dict
    global accountCategory_dict
    global accountCategory_len
    global accountCategory_index
    with open(filename,'r') as f:
        accountCategory_video_dict = json.load(f)
    for k,v in accountCategory_video_dict.items():
        if v is None or type(v) is not str:
            continue
        v = v.strip()
        if len(v) == 0:
            continue
        #print('k',k)
        #print('v',v)
        #if v == '':
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #    continue

        if v not in accountCategory_dict:
            accountCategory_dict[v] = 1
        else:
            accountCategory_dict[v] = accountCategory_dict[v] + 1
    accountCategory_len = len(accountCategory_dict.keys())
    i = 1
    #with open('try_accountCategory','w') as f:
    #    f.write(json.dumps(accountCategory_dict))
    for k,v in accountCategory_dict.items():
        accountCategory_index[k] = i
        i = i + 1

def process_accountClassify(filename):
    global accountClassify_video_dict
    global accountClassify_dict
    global accountClassify_len
    global accountClassify_index
    with open(filename,'r') as f:
        accountClassify_video_dict = json.load(f)
    for k,v in accountClassify_video_dict.items():
        if v is None:
            continue
        #print('k_c',k)
        #print('v_c',v)
        if v not in accountClassify_dict:
            accountClassify_dict[v] = 1
        else:
            accountClassify_dict[v] = accountClassify_dict[v] + 1
    accountClassify_len = len(accountClassify_dict.keys())
    i = 1
    for k,v in accountClassify_dict.items():
        accountClassify_index[k] = i
        i = i + 1
'''
def prepare_data():
  global reverse_dictionary
  global reverse_dictionary1
  global dictionary
  global dictionary1
  filename = 'train_data/words.txt'
  words = read_data(filename)
  print('Data size', len(words))
  data, count, dictionary, reverse_dictionary, dictionary1, reverse_dictionary1= build_dataset(words)
  print('all words length:',len(dictionary1.keys()))
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
  index_data()
  get_index_data()
  process_category('preprocess/category.json')
  process_interests('preprocess/interests.json')
  #process_tags('preprocess/tags.json')
  #process_accountCategory('preprocess/accountCategory.json')
  #process_accountClassify('preprocess/accountClassify.json')

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
  sd3_size = tags_len + 1
  sd4_size = accountCategory_len + 1
  sd5_size = accountClassify_len + 1
  num_interests = 3
  num_tags = 3
  num_steps = 600000

  graph = tf.Graph()
  with graph.as_default():
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_inputs_sd1 = tf.placeholder(tf.int32, shape=[batch_size])
      train_inputs_sd2 = tf.placeholder(tf.int32, shape=[batch_size, num_interests])
      #train_inputs_sd3 = tf.placeholder(tf.int32, shape=[batch_size, num_tags])
      #train_inputs_sd4 = tf.placeholder(tf.int32, shape=[batch_size])
      #train_inputs_sd5 = tf.placeholder(tf.int32, shape=[batch_size])

      train_mask_interests = tf.placeholder(tf.int32, shape=[batch_size])
      #train_mask_tags = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
      train_inputs_all = []
      train_inputs_all.append(train_inputs)
      train_inputs_all.append(train_inputs_sd1)
      train_inputs_all.append(train_inputs_sd2)
      train_inputs_all.append(train_mask_interests)
      #train_inputs_all.append(train_inputs_sd3)
      #train_inputs_all.append(train_mask_tags)
      #train_inputs_all.append(train_inputs_sd4)
      #train_inputs_all.append(train_inputs_sd5)
      train_inputs_all.append(valid_dataset)

      word2Vec_model = model(batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, sd2_size, sd3_size, sd4_size, sd5_size,  embedding_size, num_sampled)
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

              batch_inputs_sd1 = np.ndarray(shape=(batch_size),dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in category_video_dict:
                      batch_inputs_sd1[i] = 0
                  else:
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
                      #print('interests:',interest_list)
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
              '''
              batch_inputs_sd3 = np.ndarray(shape=(batch_size, num_tags),dtype=np.int32)
              mask1 = np.ndarray(shape=(batch_size),dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in tags_video_dict:
                      for j in range(num_tags):
                          batch_inputs_sd3[i][j] = 0
                      mask1[i] = 1
                  else:
                      tags_list = tags_video_dict[reverse_dictionary1[batch_inputs[i]]]
                      #print('tags:',tags_list)
                      len_tags = len(tags_list)
                      index = 0
                      if len_tags > num_tags:
                          index = num_tags
                      else:
                          index = len_tags
                      mask1[i] = index
                      for j in range(index):
                          batch_inputs_sd3[i][j] = tags_index[tags_list[j]]
                      for j in range(index, num_tags):
                          batch_inputs_sd3[i][j] = 0

              batch_inputs_sd4 = np.ndarray(shape=(batch_size), dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in accountCategory_video_dict or accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]] not in accountCategory_index:
                      batch_inputs_sd4[i] = 0
                  else:
                      #print(batch_inputs[i])
                      #print(reverse_dictionary1[batch_inputs[i]])
                      # print(accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]])
                      #print(accountCategory_index[accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]]])
                      batch_inputs_sd4[i] = accountCategory_index[accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]]]
                      #print('accountCategory:',accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]])

              batch_inputs_sd5 = np.ndarray(shape=(batch_size), dtype=np.int32)
              for i in range(batch_size):
                  if reverse_dictionary1[batch_inputs[i]] not in accountClassify_video_dict or accountClassify_video_dict[reverse_dictionary1[batch_inputs[i]]] not in accountClassify_index:
                      batch_inputs_sd5[i] = 0
                  else:
                      batch_inputs_sd5[i] = accountClassify_index[accountClassify_video_dict[reverse_dictionary1[batch_inputs[i]]]]
                      #print('accountClassify:',accountClassify_video_dict[reverse_dictionary1[batch_inputs[i]]])
              '''
              feed_dict = {train_inputs: batch_inputs, train_inputs_sd1:batch_inputs_sd1, train_inputs_sd2:batch_inputs_sd2, train_mask_interests:mask, train_labels: batch_labels}
              #learning_rate = tf.train.exponential_decay(0.01, step, 10000, 0.95, staircase=True) 
              #if step == 10000:
              #    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
              #if step == 100000:
              #    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
              #if step == 100000:
              #    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
              #if step == 300000:
              #    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
              if step == 400000:
                  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
              #if step == 500000:
              #    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
                 # We perform one update step by evaluating the optimizer op (including it
                 # in the list of returned values for session.run()
              _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
              average_loss += loss_val

              print('loss at step ', step, ',', loss_val)
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
                  saver.save(session, 'checkpoints-base-LR0.5/model.ckpt', global_step = step)
                  print('saved!')

train()

def get_eval_data():
    data = []
    data_vid = []
    for k, v in dictionary1.items():
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
    #num_tags = 3
    prepare_data()
    sd1_size = category_len + 1
    sd2_size = interests_len + 1
    sd3_size = tags_len + 1
    sd4_size = accountCategory_len + 1
    sd5_size = accountClassify_len + 1
    data, data_vid, total = get_eval_data()
    eval_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_sd1 = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_sd2 = tf.placeholder(tf.int32, shape=[batch_size, num_interests])
    #eval_inputs_sd3 = tf.placeholder(tf.int32, shape=[batch_size, num_tags])
    eval_mask_interests = tf.placeholder(tf.int32, shape=[batch_size])
    #eval_mask_tags = tf.placeholder(tf.int32, shape=[batch_size])
    #eval_inputs_sd4 = tf.placeholder(tf.int32, shape=[batch_size])
    #eval_inputs_sd5 = tf.placeholder(tf.int32, shape=[batch_size])
    eval_inputs_all = []
    eval_inputs_all.append(eval_inputs)
    eval_inputs_all.append(eval_inputs_sd1)
    eval_inputs_all.append(eval_inputs_sd2)
    eval_inputs_all.append(eval_mask_interests)
    #eval_inputs_all.append(eval_inputs_sd3)
    #eval_inputs_all.append(eval_mask_tags)
    #eval_inputs_all.append(eval_inputs_sd4)
    #eval_inputs_all.append(eval_inputs_sd5)

    word2Vec_model = model(batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, sd2_size, sd3_size, sd4_size, sd5_size, embedding_size, num_sampled)
    embedding_all = word2Vec_model.model_inferences(eval_inputs_all)
    var_list = tf.trainable_variables()
    saver = tf.train.Saver(var_list)
    weight_file = './checkpoints-base/model.ckpt-550000'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if weight_file is not None:
        saver.restore(sess, weight_file)
        print('Restoring weight file from:%s'%weight_file)
    f = open('result-vid--base.txt','w+')
    start = time.time()
    vec = []
    batch_inputs = np.ndarray(shape=(batch_size),dtype=np.int32)
    batch_inputs_sd1 = np.ndarray(shape=(batch_size),dtype=np.int32)
    batch_inputs_sd2 = np.ndarray(shape=(batch_size, num_interests), dtype=np.int32)
    #batch_inputs_sd3 = np.ndarray(shape=(batch_size, num_tags), dtype=np.int32)
    #batch_inputs_sd4 = np.ndarray(shape=(batch_size),dtype=np.int32)
    #batch_inputs_sd5 = np.ndarray(shape=(batch_size),dtype=np.int32)
    batch_mask = np.ndarray(shape=(batch_size), dtype=np.int32)
    #batch_mask1 = np.ndarray(shape=(batch_size), dtype=np.int32)
    for i in range(total):
        batch_inputs[0] = data[i]
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
        
        '''
        if reverse_dictionary1[data[i]] not in tags_video_dict:
            for j in range(num_tags):
                batch_inputs_sd3[0][j] = 0
            batch_mask1[0] = 1
        else:
            tags_list = tags_video_dict[reverse_dictionary1[data[i]]]
            len_tags = len(tags_list)
            index = 0
            if len_tags > num_tags:
                index = num_tags
            else:
                index = len_tags
            batch_mask1[0] = index
            for j in range(index):
                batch_inputs_sd3[0][j] = tags_index[tags_list[j]]
            for j in range(index, num_tags):
                batch_inputs_sd3[0][j] = 0

        if reverse_dictionary1[data[i]] not in accountCategory_video_dict or accountCategory_video_dict[reverse_dictionary1[data[i]]] not in accountCategory_index:
            batch_inputs_sd4[0] = 0
        else:
            #print(batch_inputs[i])
            #print(reverse_dictionary1[batch_inputs[i]])
            # print(accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]])
            #print(accountCategory_index[accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]]])
            batch_inputs_sd4[0] = accountCategory_index[accountCategory_video_dict[reverse_dictionary1[data[i]]]]
            #print('accountCategory:',accountCategory_video_dict[reverse_dictionary1[batch_inputs[i]]])

        if reverse_dictionary1[data[i]] not in accountClassify_video_dict or accountClassify_video_dict[reverse_dictionary1[data[i]]] not in accountClassify_index:
            batch_inputs_sd5[0] = 0
        else:
            batch_inputs_sd5[0] = accountClassify_index[accountClassify_video_dict[reverse_dictionary1[data[i]]]]
                #print('accountClassify:',accountClassify_video_dict[reverse_dictionary1[batch_inputs[i]]])
        '''
        feed_dict = {eval_inputs: batch_inputs, eval_inputs_sd1:batch_inputs_sd1, eval_inputs_sd2:batch_inputs_sd2, eval_mask_interests:batch_mask}

        #feed_dict1 = {eval_inputs:batch_inputs, eval_inputs_sd1:batch_inputs_sd1, eval_inputs_sd2:batch_inputs_sd2, eval_mask:batch_mask}
        embedding_single = sess.run(embedding_all,feed_dict = feed_dict)
        f.write(data_vid[i]+'\n')
        vec.append(embedding_single)
    vec = np.array(vec)
    np.save('result-embedding-base.npy',vec)
    end = time.time() - start
    print(end)

#predict()


vec_file = 'result-embedding-base.npy'
vec = np.load(vec_file)
vec = vec[:,0,:]
res = faiss.StandardGpuResources()
normalize_L2(vec)
index_flat = faiss.IndexFlatIP(128)
gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
gpu_index_flat.add(vec)

def get_top_k(user_vec, vid_list):
    try:
        D,I = gpu_index_flat.search(user_vec,50)
        for m in range(1):
            tmp_res = []
            for n in range(0,50):
                tmp_res.append(vid_list[I[m][n]])
    except:
        print('error!!!!!')

    return tmp_res

def NDCG(top_k, truth_list):
    dcg = 0.0
    k = 0
    rank = []
    for i in range(0, len(top_k)):
        if top_k[i] in truth_list:
            dcg = dcg + 1/math.log((i+2),2)
            k = k+1
            rank.append(1)
        else:
            rank.append(0)
    max_dcg = 0.0
    rank = sorted(rank, reverse=True)
    for i in range(1, len(top_k)+1):
        max_dcg = max_dcg + rank[i-1]/math.log(i+1,2)
    if k == 0:
        ndcg = 0.0
    else:
        ndcg = dcg/max_dcg
    return ndcg

def AP(top_k, truth_list):
    rank = []
    flag = False
    for vid in top_k:
        if vid in truth_list:
            rank.append(1)
            flag = True
        else:
            rank.append(0)
    ap = 0.0
    for i in range(1,len(rank)+1):
        ap = ap + rank[i - 1] * float(sum(rank[0:i]))/i
    ap = ap/len(top_k)
    return ap, flag

def Precision(top_k, truth_list):
    n = 0.0
    for vid in truth_list:
        if vid in top_k:
            n = n + 1
    return n/len(top_k)

def Recall(top_k, truth_list):
    n = 0.0
    for vid in top_k:
        if vid in truth_list:
            n = n + 1
    print(n)
    return n/len(truth_list)

def calculate_all():
    all_ap = 0.0
    all_ndcg = 0.0
    #all_precision = 0.0
    #all_recall = 0.0
    for k,v in test_result.items():
        ap, flag = AP(v['predict'], v['hit'])
        all_ap = all_ap + ap
        all_ndcg = all_ndcg + NDCG(v['predict'], v['hit'])
        #precision = all_precision + Precision(v['predict'], v['hit'])
        #recall = all_recall + Recall(v['predict'], v['hit'])
    MAP = all_ap/len(test_result.keys())
    ndcg = all_ndcg/len(test_result.keys())
    #recall = all_recall/len(test_result.keys())
    #precision = all_precision/len(test_result.keys())

    print('MAP_cy_L2:',MAP)
    print('NDCG_cy_L2:',ndcg)
    #print('recall_cy_IP:',recall)
    #print('precision_cy_IP:',precision)

vid_result = []
prepare_data()

def test():
    
    global test_result
    global vid_result
    global vec_fc
    with open('result-vid--base.txt') as f:
        vid = f.readlines()
    vec = np.load('result-embedding-base.npy',allow_pickle=True)
    vid_vec = {}
    print(vec.shape)
    vids = []
    for id in vid:
        id = id.strip()
        vids.append(id)
    print(len(vids))

    i = 0
    for vid in vids:
        vid_vec[vid] = vec[i]
        i = i + 1

   # vec_fc = np.load('norm_weight_fc.npy',allow_pickle=True)
    vid_result = []

    for i in range(1,len(vids)):
        vid_result.append(reverse_dictionary1[i])

    with open('/data/2/chenyao/word2Vec/test_data/20191128-v1.json') as f:
        content = json.load(f)

    print('user_len: ',len(content.keys()))
    test_result = {}
    w = open('/data/3/chenyao/relative_result_base', 'w')
    i = 0
    for k,v in content.items():
        hit = v['hit']
        history = v['history']
        if len(hit) == 0: continue
        vec_all = np.zeros(shape=(1,128),dtype=np.float32)
        num = 0
        for item in history:
            if item in vid_vec:
                num = num + 1
                vec_all = vec_all + vid_vec[item]
        if num == 0:
            continue
        else:
            print(i)
            i = i + 1
            vec_ave = vec_all/num
            temp_res = get_top_k(vec_ave, vid_result)
            tmp = {}
            tmp['predict'] = temp_res
            tmp['hit'] = hit
            w.write(json.dumps({k:tmp})+'\n')
            test_result[k] = tmp

    print('after:',len(test_result.keys()))
    ''''
    with open('/data/3/chenyao/relative_result_cy_IP','r') as f:
        content = f.readlines()
    for line in content:
        line = line.strip()
        line = eval(line)
        for k, v in line.items():
            test_result[k] = v
    '''
    calculate_all()
#test()
