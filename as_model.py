import tensorflow as tf
import numpy as np
import pandas as pd
import heapq
import json
import time
from OnehotDocumentParser import *
from util import *
import argparse

class AbstractiveSummarizer(object):
    
    def get_encoder(self, context_encoder):
        def bag_of_words_encoder(x, y, x_mask, y_pos, params, tfparams):
            mb_size = tf.shape(x)[0]
                
            l = params['seq_max_len']
            x_emb = tf.matmul(tf.reshape(x, [mb_size*l, -1], tfparams['Xemb']))
            ctx = tf.reduce_sum(tf.reshape(x_emb, [mb_size, l, -1]), 1) * (1 / tf.reduce_sum(x_mask, 1))
            return ctx
        
        def attention_encoder(x, y, x_mask, y_pos, params, tfparams):
            mb_size = tf.shape(x)[0]
                
            l = params['seq_max_len']
            d_x = params['x_embedder_word_dim']
            C = params['summary_context_length']
            V_y = params['y_embedder_vocabulary_size']
            
            x_emb = tf.matmul(tf.reshape(x, shape=[mb_size*l, -1]), tfparams['Xemb'])
            x_emb = tf.reshape(tf.transpose(tf.reshape(x_emb, [mb_size, l, -1]), [0, 2, 1]), [-1, l]) 
            
            y_emb = tf.matmul(tf.reshape(y[:, (y_pos-C)*V_y:y_pos*V_y], shape=[mb_size*C, -1]), tfparams['Yemb'])
            y_emb = tf.matmul(tf.reshape(y_emb, [mb_size, -1]), tf.transpose(tfparams['att_P'], [1, 0]))
            
            p = tf.reduce_sum(tf.reshape((x_emb + tf.reshape(y_emb, [-1, 1])), [mb_size, -1, l]), 1)
            
            p = tf.nn.softmax(p)
            p_masked = p * x_mask
            p_masked = p_masked / tf.reduce_sum(p_masked, 1)
            x_emb = tf.matmul(x_emb, tf.cast(tfparams['att_smoothing'], tf.float32))
            x_emb = tf.reshape(tf.transpose(tf.reshape(x_emb, [mb_size, -1, l]), [0, 2, 1]), [mb_size*l, -1])
            ctx = tf.reduce_sum(tf.reshape(x_emb + tf.reshape(p, [-1, 1]), [mb_size, l, d_x]), 1)
    
            return ctx
        
        if context_encoder == 'bag_of_words':
            return bag_of_words_encoder
        elif context_encoder == 'attention':
            return attention_encoder
        else:
            raise ValueError('Invalid context encoder: ', context_encoder)
            
    def init_tfparams(self, **kwargs):
        def init_shared_params(name, shape, value=None, trainable=True):
            if value is None:
                value = tf.random_normal(shape)
            return tf.Variable(initial_value=value, name=name, trainable=trainable)
        
        def attention_smoothing_window_matrix(Q, l):
            assert l >= Q
            m = np.diagflat(np.ones(l))
            for i in range(1, Q+1):
                m += np.diagflat(np.ones(l-i), k=i)
                m += np.diagflat(np.ones(l-i), k=-i)
            m /= np.sum(m, axis=0)
            return m
                
        params = kwargs.copy()
        np.random.seed(params['seed'])
        
        h = params['hidden_layer_size']
        C = params['summary_context_length']
        l = params['seq_max_len']
        V_x = params['x_embedder_vocabulary_size']
        V_y = params['y_embedder_vocabulary_size']
        d_x = params['x_embedder_word_dim']
        d_y = params['y_embedder_word_dim']
        
        tfparams = {
            'U': init_shared_params('U', (h, C * d_y)),
            'd': init_shared_params('d', (h, 1)),
            'V': init_shared_params('V', (V_y, h)),
            'W': init_shared_params('W', (V_y, d_x)),
            'b': init_shared_params('b', (V_y, 1)),
            'Xemb': init_shared_params('Xemb', (V_x, d_x)),
            'Yemb': init_shared_params('Yemb', (V_y, d_y))
        }
        
        if params['context_encoder'] == 'attention':
            Q = params['attention_smoothing_window_size']
            m = attention_smoothing_window_matrix(Q, l)
            tfparams['att_P'] = init_shared_params('att_P', (d_x, C * d_y))
            tfparams['att_smoothing'] = init_shared_params('att_smoothing', (l, l), value=m, trainable=False)
            
        return params, tfparams
    
    def __init__(self, 
                 # model params
                 seq_max_len=300, 
                 summary_max_len=30, 
                 summary_pred_len=10,
                 summary_context_length=10, 
                 hidden_layer_size=200, 
                 attention_smoothing_window_size=5,
                 context_encoder='attention',
                 x_embedder_vocabulary_size=60,
                 y_embedder_vocabulary_size=60,
                 x_embedder_word_dim=10,
                 y_embedder_word_dim=10,
                 summary_search_beam_size=5,
                 # training params
                 learning_rate=0.001,
                 optimizer='adam',
                 epochs=80,
                 mini_batch_size=64,
                 l2_penalty_coeff=0.001,
                 epsilon_for_log = 1e-8,
                 seed=2016,
                 print_every_batches=50):
        self.params, self.tfparams = self.init_tfparams(
             seq_max_len=seq_max_len, 
             summary_max_len=summary_max_len, 
             summary_pred_len=summary_pred_len,
             summary_context_length=summary_context_length, 
             hidden_layer_size=hidden_layer_size, 
             attention_smoothing_window_size=attention_smoothing_window_size,
             context_encoder=context_encoder,
             x_embedder_vocabulary_size=x_embedder_vocabulary_size,
             y_embedder_vocabulary_size=y_embedder_vocabulary_size,
             x_embedder_word_dim=x_embedder_word_dim,
             y_embedder_word_dim=y_embedder_word_dim,
             summary_search_beam_size=summary_search_beam_size,
             learning_rate=learning_rate,
             optimizer=optimizer,
             epochs=epochs,
             mini_batch_size=mini_batch_size,
             l2_penalty_coeff=l2_penalty_coeff,
             epsilon_for_log = epsilon_for_log,
             seed=seed,
             print_every_batches=print_every_batches
             )
        self.sess = tf.Session()
        
        
    def train(self, generator=None):
        start = time.time()
        # general params
        V_x = self.params['x_embedder_vocabulary_size']
        V_y = self.params['y_embedder_vocabulary_size']
        
        # define input
        x = tf.placeholder(tf.float32, shape=[None, self.params['seq_max_len']*V_x])
        y = tf.placeholder(tf.float32, shape=[None, (self.params['summary_context_length']+self.params['summary_max_len'])*V_y])
        x_mask = tf.placeholder(tf.float32, shape=[None, self.params['seq_max_len']])
        y_mask = tf.placeholder(tf.float32, shape=[None, self.params['summary_max_len']])
        
        
        # define loss
        print 'Start defining the loss'
        C = self.params['summary_context_length']
        for i in range(C, C + self.params['summary_max_len']):
            cur_probs = self.conditional_prob(x, y, x_mask, i, self.params, self.tfparams)
            if i == C:
                conditional_probs = cur_probs
            else:
                conditional_probs = tf.concat([conditional_probs, cur_probs], 1)

        loss = -tf.log(self.params['epsilon_for_log'] + conditional_probs) * y_mask
        loss = tf.reduce_mean(tf.reduce_sum(loss, 1) / tf.cast(tf.reduce_sum(y_mask, 1), tf.float32))
        l2_loss = [tf.reduce_sum(tf.square(v) / 2.0) for k, v in self.tfparams.items()]
        for l in l2_loss:
            loss += self.params['l2_penalty_coeff'] * tf.cast(l, tf.float32)

        if self.params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)
        
        init = tf.global_variables_initializer()
        print 'Loss is defined'
        print 'Time elapsed:', time.time() - start
        
        print 'Start training'
        train_start = time.time()
        self.sess.run(init)
        for epoch in range(self.params['epochs']):
            print 'Epoch ', epoch
            epoch_start = time.time()
            total_loss = 0
            num_batches = generator.get_num_batches()
            for batch_id in range(num_batches):
                if batch_id % self.params['print_every_batches'] == 0:
                    print 'Batch ', batch_id
                batch_x, batch_x_mask, batch_y, batch_y_mask, batch_labels = generator.get_next_batch()
                _, loss_batch = self.sess.run([optimizer, loss], feed_dict={x: batch_x,
                                                                            x_mask: batch_x_mask,
                                                                            y: batch_y,
                                                                            y_mask: batch_y_mask})
                total_loss += loss_batch
                # print 'total loss', total_loss
            
            print "Epoch ", epoch, " - total loss: ", total_loss / float(num_batches)
            print 'Time elapsed:', (time.time() - epoch_start) / 60.0, ' minutes'
            
        print 'Training is done!'
        print 'Time elapsed:', (time.time() - train_start) / 60.0, ' minutes'
                
    
    def predict(self, generator=None, output_filename='output'):
        # batch size is 1
        print 'Start making predictions'
        pred_start = time.time()
        
        params = self.params.copy()
        params['mini_batch_size'] = 1
        
        res = []
        f_pred = open('result/' + output_filename + '-pred-summary.txt', 'w')
        f_labels = open('result/' + output_filename + '-true-summary.txt', 'w')
        for batch_id in range(generator.get_num_batches()):
            batch_x, batch_x_mask, batch_y, batch_y_mask, batch_labels = generator.get_next_batch()
            y_pred_ids = self.summarize(batch_x, batch_x_mask, params, self.tfparams, generator=generator)
            y_pred_words = generator.doc_parser.ids_to_words(y_pred_ids)
            print 'predictions: ', y_pred_words
            print 'true labels: ', batch_labels[0]
            res.append(y_pred_words)
            for i in range(len(y_pred_words)-1):
                f_pred.write(str(y_pred_words[i]) + ' ')
            f_pred.write(y_pred_words[-1]+'\n')
            
            for j in range(len(batch_labels[0])-1):
                f_labels.write(str(batch_labels[0][j]) + ' ')
            f_labels.write(batch_labels[0][-1]+'\n')
        f_pred.close()
        f_labels.close()
        
        print 'Predictions are done!'
        print 'Time elapsed:', (time.time() - pred_start) / 60.0, ' minutes'
        return res
            
    
    def summarize(self, x, x_mask, params, tfparams, generator=None):
        ''' Generate summary for a single text using beam search
        '''
        C = params['summary_context_length']
        k = params['summary_search_beam_size']
        
        # the id of padded word is 1
        id_pad = 1
        
        # initialise the summary and the beams for search
        y = [id_pad] * C
        beams = [(0.0, y)]
        
        for j in range(params['summary_pred_len']):
            new_beams = []
            for (base_score, y) in beams:
                # print base_score, y
                onehot_y = np.array([generator.doc_parser.ids_to_onehot_vector(y)], dtype=np.float32)
                top_k = tf.nn.top_k(self.conditional_distribution(x, onehot_y, x_mask, len(y), params, tfparams), k=k)
                y_probs, y_ids = self.sess.run(top_k)
                for y_id, y_prob in zip(y_ids[0], y_probs[0]):
                    # print y_id, y_prob
                    new_score = base_score - np.log(params['epsilon_for_log'] + y_prob)
                    heapq.heappush(new_beams, (new_score, y + [y_id]))
                    
            beams = heapq.nsmallest(k, new_beams)
            
        best_score, summary = heapq.heappop(beams)
        return summary[C:]
                
        
        
    def conditional_distribution(self, x, y, x_mask, y_pos, params, tfparams):
        mb_size = params['mini_batch_size']
        
        encoder = self.get_encoder(params['context_encoder'])
        C = params['summary_context_length']
        V_y = params['y_embedder_vocabulary_size']
        d_y = params['y_embedder_word_dim']
        
        y_emb = tf.matmul(tf.reshape(y[:, (y_pos-C)*V_y:y_pos*V_y], shape=[mb_size*C, -1]), tfparams['Yemb'])
        y_emb = tf.transpose(tf.reshape(y_emb, shape=[mb_size, C * d_y]))
        h = tf.nn.tanh(tf.matmul(tfparams['U'], y_emb) + tfparams['d']) # broadcast for [size, 1] reshape()

        ctx = encoder(x, y, x_mask, y_pos, params, tfparams)
        
        y_dist = tf.matmul(tfparams['V'], h) + tf.matmul(tfparams['W'], tf.transpose(ctx)) + tfparams['b']
        
        return tf.nn.softmax(tf.transpose(y_dist))


    def conditional_prob(self, x, y, x_mask, y_pos, params, tfparams):
        mb_size = params['mini_batch_size']
        
        V_y = params['y_embedder_vocabulary_size']
        y_indices = tf.cast(tf.argmax(y[:, y_pos*V_y:(y_pos+1)*V_y], 1), tf.int32)
        
        for i in range(mb_size):
            index = tf.constant(i, dtype=tf.int32)
            y_dist = self.conditional_distribution(x, y, x_mask, y_pos, params, tfparams)
            cur = tf.reshape(y_dist[index, y_indices[index]], shape=[1, 1])
            
            if i == 0:
                res = cur
            else:
                res = tf.concat([res, cur], 0)
            
        return tf.reshape(res, shape=[mb_size, 1])
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train abstractive summarizer model')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--word_freq', type=str)
    args = parser.parse_args()
    
    onehot_parser = OnehotDocumentParser(args.word_freq, unk_size=3)
    model = AbstractiveSummarizer(
                 # model params
                 seq_max_len=500, 
                 summary_max_len=30, 
                 summary_pred_len=10,
                 summary_context_length=5, 
                 hidden_layer_size=128, 
                 attention_smoothing_window_size=6,
                 context_encoder='attention',
                 x_embedder_vocabulary_size=onehot_parser.vocab_size,
                 y_embedder_vocabulary_size=onehot_parser.vocab_size,
                 x_embedder_word_dim=64,
                 y_embedder_word_dim=64,
                 summary_search_beam_size=100,
                 # training params
                 learning_rate=0.001,
                 optimizer='adam',
                 epochs=20,
                 mini_batch_size=64,
                 l2_penalty_coeff=0.001,
                 seed=2016)
    train_filename = args.train_file
    bg_train = BatchGenerator(train_filename, batch_size=model.params['mini_batch_size'], document_parser=onehot_parser, params=model.params)
    model.train(generator=bg_train)
    
    pred_filename = args.pred_file
    bg_predict = BatchGenerator(pred_filename, batch_size=1, document_parser=onehot_parser, params=model.params)
    model.predict(generator=bg_predict, output_filename=pred_filename)
    

