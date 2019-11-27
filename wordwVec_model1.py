import tensorflow as tf
import math
import numpy as np

class model():
    def __init__(self,batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, embedding_size, num_sampled):
        self.reverse_dictionary = reverse_dictionary
        self.reverse_dictionary1 = reverse_dictionary1
        self.vocabulary_size = vocabulary_size
        self.vocabulary_size1 = vocabulary_size1
        self.sd1_size = sd1_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.batch_size = batch_size

    def model_inferences(self, train_inputs_all):
        # Input data.
        train_inputs = train_inputs_all[0]
        train_inputs_sd1 = train_inputs_all[1]
        if len(train_inputs_all) == 3:
            valid_dataset = train_inputs_all[2]
        else:
            valid_dataset = None

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            train_inputs1 = np.ndarray(shape = (self.batch_size), dtype = np.int32)
            for i in range(self.batch_size):
                if train_inputs[i] not in self.reverse_dictionary:
                    train_inputs1[i] = 0
                else:
                    train_inputs1[i] = train_inputs[i] 
            embed = tf.nn.embedding_lookup(embeddings, train_inputs1)

            embeddings_sd1 = tf.Variable(
                tf.random_uniform([self.sd1_size, self.embedding_size], -1.0, 1.0))
            embed_sd1 = tf.nn.embedding_lookup(embeddings_sd1, train_inputs_sd1)

            sd_weights = tf.Variable(
                tf.random_uniform([self.vocabulary_size1, 2], 0, 1.0))
            norm_weight = tf.sqrt(tf.reduce_sum(tf.square(sd_weights), 0, keep_dims=True))
            norm_sd_weights = sd_weights/norm_weight
            embed_weight = tf.nn.embedding_lookup(norm_sd_weights, train_inputs)
            embedding_all = embed_weight[:,0] * embed +  embed_weight[:,1] * embed_sd1

            similarity = None
            # Compute the cosine similarity between minibatch examples and all embeddings.
            if valid_dataset is not None:
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(
                    normalized_embeddings, valid_dataset)
                similarity = tf.matmul(
                    valid_embeddings, normalized_embeddings, transpose_b=True)      
                return embedding_all, similarity
            else:
                return embedding_all

     # Compute the average NCE loss for the batch.
     # tf.nce_loss automatically draws a new sample of the negative labels each
     # time we evaluate the loss.
    def loss(self, embedding_all, train_labels):
    # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
        tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                               stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embedding_all,
                         self.num_sampled, self.vocabulary_size))
        return loss
