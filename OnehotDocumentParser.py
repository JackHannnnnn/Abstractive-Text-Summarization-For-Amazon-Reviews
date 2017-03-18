import numpy as np
import time

class OnehotDocumentParser(object):
    def __init__(self, file_name, unk_size=5, pad_token = "<PAD>", unk_token = "<UNK>"):
        self.unk_size = unk_size
        
        self.word_to_id, self.id_to_word, self.vocab_size = self.parse_file(file_name)
        
        self.unk = unk_token
        self.pad = pad_token
    
        self.word_to_id[self.unk] = 0
        self.id_to_word[0] = self.unk 
        
        self.word_to_id[self.pad] = 1
        self.id_to_word[1] = self.pad 


    def parse_file(self, file_name):
        word_to_id = {}
        id_to_word = {}
        index = 2 
        with open(file_name, 'r') as f:
            for line in f:
                split_line = line.split()
                if len(split_line) < 2:
                    continue
                
                word_key = split_line[0]
                word_freq = int(split_line[1])
                if word_freq <= self.unk_size:
                    break
                word_to_id[word_key] = index
                id_to_word[index] = word_key
                index += 1
                
        return word_to_id, id_to_word, index

    
    def word_to_onehot_vector(self, words):
        s = time.time()
        # print 'word len: ', len(words)
        word_ids = [self.word_to_id[word] if word in self.word_to_id else self.word_to_id[self.unk] for word in words]
        res = np.zeros((len(words), self.vocab_size))
        for i, id in enumerate(word_ids):
            res[i, id] = 1
        # print 'elapsed: ', time.time() - s
        return res.flatten().tolist()
    
    def ids_to_onehot_vector(self, ids):
        res = np.zeros((len(ids), self.vocab_size))
        for i, id in enumerate(ids):
            res[i, int(id)] = 1
        return res.flatten().tolist()
        
        
    def ids_to_words(self, ids):
        words = [self.id_to_word[id] for id in ids]
        return words
    