import numpy as np
import pandas as pd
import json
import time

class BatchGenerator(object):
    def __init__(self, filename, batch_size=64, generate_y=True, document_parser=None, params=None, seq_len_threshold=500):
        self.seq_len_threshold=seq_len_threshold

        self.batch_size = batch_size
        self.generate_y = generate_y
        self.x, self.y = self.preprocess(filename)
        self.doc_parser = document_parser
        self.ith_batch = 0
        self.num_batches = None
        self.params = params
        
        
        
    def preprocess(self, filename):
        with open(filename, 'r') as json_data:
            data = pd.DataFrame(json.load(json_data))
            data['seq_len'] =  [len(line.split()) for line in data['reviewText']]
            data['summary_len'] = [len(line.split()) for line in data['summary']]
            data = data[(data['seq_len'] <= self.seq_len_threshold) & (data['summary_len'] <= 30)]
            #data = data.iloc[:self.a, :]
        x = data['reviewText'].values
        y = data['summary'].values if self.generate_y else None
        return x, y
    
    def get_num_batches(self):
        if self.num_batches is not None:
            return self.num_batches
        else:
            self.num_batches = int(np.floor(self.x.shape[0] / float(self.batch_size)))
        return self.num_batches
    
    def get_next_batch(self):
        s = time.time()
        batch_x = self.x[self.ith_batch*self.batch_size:(self.ith_batch+1)*self.batch_size]
        batch_x = [line.split() for line in batch_x]
        batch_x_mask = [[1] * len(line) + [0] * (self.params['seq_max_len'] - len(line)) for line in batch_x]
        batch_x = [self.sentence_padding(line, self.params['seq_max_len'], False) for line in batch_x]
        
        
        if self.generate_y:
            batch_labels = self.y[self.ith_batch*self.batch_size:(self.ith_batch+1)*self.batch_size]
            batch_labels = [line.split() for line in batch_labels]
            batch_y_mask = [[1] * len(line) + [0] * (self.params['summary_max_len'] - len(line)) for line in batch_labels]
            #batch_y_mask = [[1] * self.params['summary_max_len'] for line in batch_labels]
            batch_y = [self.sentence_padding(line, self.params['summary_max_len'], True) for line in batch_labels]
            
            #batch_y = np.array(batch_y)[:, :(self.params['summary_max_len']*self.doc_parser.vocab_size)]
            
        # print 'done1:', time.time() - s
        batch_x = [self.doc_parser.word_to_onehot_vector(line) for line in batch_x]
        batch_y = [self.doc_parser.word_to_onehot_vector(line) for line in batch_y]
        
        self.ith_batch += 1
        if self.ith_batch == self.get_num_batches():
            self.ith_batch = 0
            
        if self.generate_y:
            return batch_x, batch_x_mask, batch_y, batch_y_mask, batch_labels
        else:
            return batch_x, batch_x_mask
        
    def sentence_padding(self, sentence, max_len, is_y=False):
        if not is_y:
            return sentence + [self.doc_parser.pad] * (max_len - len(sentence))
        else:
            y_context_padding = [self.doc_parser.pad] * self.params['summary_context_length'] 
            return y_context_padding + sentence + [self.doc_parser.pad] * (max_len - len(sentence))
        
