import tensorflow as tf
import math
import numpy as np

class model():
    def __init__(self,batch_size, reverse_dictionary, reverse_dictionary1, vocabulary_size, vocabulary_size1, sd1_size, sd2_size, sd3_size, sd4_size, sd5_size, embedding_size, num_sampled):
        self.reverse_dictionary = reverse_dictionary
        self.reverse_dictionary1 = reverse_dictionary1
        self.vocabulary_size = vocabulary_size
        self.vocabulary_size1 = vocabulary_size1
        self.sd1_size = sd1_size
        self.sd2_size = sd2_size
        self.sd3_size = sd3_size
        self.sd4_size = sd4_size
        self.sd5_size = sd5_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.batch_size = batch_size

    def model_inferences(self, train_inputs_all):
        # Input data.
        train_inputs = train_inputs_all[0]
        train_inputs_sd1 = train_inputs_all[1]
        train_inputs_sd2 = train_inputs_all[2]
        train_mask_interests = train_inputs_all[3]
       # train_inputs_sd3 = train_inputs_all[4]
       # train_mask_tags = train_inputs_all[5]
       # train_inputs_sd4 = train_inputs_all[6]
       # train_inputs_sd5 = train_inputs_all[7]

       # if len(train_inputs_all) == 9: 
       #     valid_dataset = train_inputs_all[8]
       # else:
       #     valid_dataset = None

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/gpu:0'):
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
            print('embed_data:',embed.shape)
            embeddings_sd1 = tf.Variable(
                tf.random_uniform([self.sd1_size, self.embedding_size], -1.0, 1.0))
            embed_sd1 = tf.nn.embedding_lookup(embeddings_sd1, train_inputs_sd1)

            embeddings_sd2 = tf.Variable(
                tf.random_uniform([self.sd2_size, self.embedding_size],-1.0, 1.0))

            #embeddings_sd3 = tf.Variable(
            #    tf.random_uniform([self.sd3_size, self.embedding_size],-1.0,1.0))

            #embeddings_sd4 = tf.Variable(
            #    tf.random_uniform([self.sd4_size, self.embedding_size],-1.0,1.0))

            #embeddings_sd5 = tf.Variable(
            #    tf.random_uniform([self.sd5_size, self.embedding_size],-1.0,1.0))
            
            #print('train_inputs_sd4:',train_inputs_sd4)
            #print('sd4 size:', self.sd4_size)
            #print('embeddings_sd4:', embeddings_sd4)
            #embed_sd4 = tf.nn.embedding_lookup(embeddings_sd4, train_inputs_sd4)
            #print(embeddings_sd4.shape)
            #print('embed_sd4:',embed_sd4.shape)
            #embed_sd5 = tf.nn.embedding_lookup(embeddings_sd5, train_inputs_sd5)
            #print('embed_sd5:',embed_sd5.shape)
            #ave_embed_sd2 = np.ndarray(shape=(self.batch_size, self.embedding_size), dtype=tf.float32)
            ave_embed_sd2 = []
            for i in range(self.batch_size):
                embed_sd2_tmp = tf.nn.embedding_lookup(embeddings_sd2, train_inputs_sd2[i])
                mask_flatten = tf.sequence_mask(tf.transpose(train_mask_interests[i]), 3, dtype=tf.float32)
                #print(embed_sd2_tmp)
                #print(mask_flatten)
                mask_embed_sd2 = tf.expand_dims(mask_flatten, axis=1) * embed_sd2_tmp
                tmp = 0
                for j in range(3):
                    tmp = mask_embed_sd2[j,:] + tmp
                embedd_sd2 = tf.expand_dims(tmp/tf.cast(train_mask_interests[i],tf.float32),axis=0)
                #print('embedd_sd2:',embedd_sd2.shape)
                ave_embed_sd2.append(embedd_sd2)
            embed_sd2 = tf.concat(ave_embed_sd2,axis=0)
            print('embed_sd2:',embed_sd2)
            '''
            ave_embed_sd3 = []
            for i in range(self.batch_size):
                embed_sd3_tmp = tf.nn.embedding_lookup(embeddings_sd3, train_inputs_sd3[i])
                #print('look_up_sd3:',embed_sd3_tmp.shape)
                mask_flatten = tf.sequence_mask(tf.transpose(train_mask_tags[i]), 3, dtype=tf.float32)
                mask_embed_sd3 = tf.expand_dims(mask_flatten,axis=1) * embed_sd3_tmp
                tmp = 0
                for j in range(3):
                    tmp = mask_embed_sd3[j,:] + tmp
                embedd_sd3 = tf.expand_dims(tmp/tf.cast(train_mask_tags[i],tf.float32),axis=0)
                #print('embedd_sd3:',embedd_sd3.shape)
                ave_embed_sd3.append(embedd_sd3)
            #print('128:',len(ave_embed_sd3))
            embed_sd3 =  tf.concat(ave_embed_sd3,axis=0)
            print('embed_sd1:',embed_sd1)
            print('embed_sd3:',embed_sd3.shape)
            '''    
            sd_weights = tf.Variable(
                tf.random_uniform([self.vocabulary_size1, 3], 0, 1.0))
            norm_weight = tf.sqrt(tf.reduce_sum(tf.square(sd_weights), 0, keep_dims=True))
            norm_sd_weights = sd_weights/norm_weight
            embed_weight = tf.nn.embedding_lookup(norm_sd_weights, train_inputs)
            embedding_all = embed_weight[:,0] * embed +  embed_weight[:,1] * embed_sd1 + embed_weight[:,2] * embed_sd2 
            #embed_weight[:,3] * embed_sd3 + embed_weight[:,4] * embed_sd4 + embed_weight[:,5] * embed_sd5 
            print('embedding_all:',embedding_all.shape) 
            similarity = None
            '''
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
            '''
            return embedding_all

     # Compute the average NCE loss for the batch.
     # tf.nce_loss automatically draws a new sample of the negative labels each
     # time we evaluate the loss.
    def loss(self, embedding_all, train_labels):
    # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
        tf.truncated_normal([self.vocabulary_size1, self.embedding_size],
                               stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size1]))
        loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embedding_all,
                         self.num_sampled, self.vocabulary_size1))
        return loss
