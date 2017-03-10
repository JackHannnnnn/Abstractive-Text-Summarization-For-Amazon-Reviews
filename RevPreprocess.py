'''
This is for 517 nlp project

author: Chaofan Han, An Yan

source data:  http://jmcauley.ucsd.edu/data/amazon/links.html

small dataset used:

Amazon Instant Video 5 core : 37,126 reviews, 9.1M

***Instruction***
python RevPreprocess.py -t 3 reviews_Amazon_Instant_Video_5.json

'''

from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn 
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
import itertools
import sys
import random

# import utils 
from utils import *


UNK_token = '_UNK_'



class TextProcess:

    def __init__(self,args):
        self.full_dataset_path = args.full_dataset
        # self.dev_set = args.dev_set
        self.unk_threshold = args.threshold

        self.oovcount = 0
        self.oovs = dict()
        # frequency table
        self.review_dict = dict()
        self.sum_dict = dict()
        self.reviewsum_dict = dict()

        # a set of dict {review id, review text, review summary}
        self.train_set = set()
        self.test_set = set()
        self.train_list = list()
        self.test_list = list()
        self.dev_set = set()
        self.dev_list = list()

        # dict of summary + rev text. {word: freq} with UNK
        self.unigram_V = {}
        # word freq dict of sum text and review test seperately
        self.sum_V = {}
        self.rev_V={}
        self.train_rev_text = ""
        self.train_sum_text = ""

       

    '''
    split full dataset into 8:1:1
    '''
    def train_test_split(self):
        

        with open(self.full_dataset_path,'r') as f:
            data = f.readlines()
            full_len = len(data)
            train_len = int(full_len * 0.8)
            dev_len = int(full_len * 0.1)
            test_len = int(full_len * 0.1)
            f.close()
            print("review count of the whole data set:",full_len)
            print("training data length:",train_len)
            print("test data length:",test_len)
          
            
            train_data = data[:train_len]
            dev_data = data[train_len:train_len+dev_len]
            test_data = data[-test_len:]

            train_sum_set = set()
            train_rev_set = set()
            print (" begin splitting data")
            t0 = time.time()
            for line in train_data:
               
                temp_dict = dict()
                temp_dict['reviewerID']=json.loads(line.decode('utf-8'))['reviewerID']
                temp_dict['reviewText']=textclean(json.loads(line)['reviewText'])
                temp_dict['summary']=textclean(json.loads(line)['summary'].lower())
                # remove short sentences---hold off
                # if (len(temp_dict['reviewText'])< threshold):
                train_sum_set.add(temp_dict['summary'])
                train_rev_set.add(temp_dict['reviewText'])
             
                    # encode dict with json string in order to be hashable
                temp_json = json.dumps(temp_dict)
                    
                self.train_set.add(temp_json)
                self.train_list.append(temp_dict)

            self.train_sum_text = ' '.join(train_sum_set)
            self.train_rev_text = ' '.join(train_rev_set)


            t1 = time.time()
            print("-----time for splitting out train----",t1-t0)
            for line in test_data:
                temp_dict = dict()
                temp_dict['reviewerID']=json.loads(line.decode('utf-8'))['reviewerID']
                temp_dict['reviewText']=textclean(json.loads(line)['reviewText'])
                temp_dict['summary']=textclean(json.loads(line)['summary'].lower())
                    # encode dict with json string in order to be hashable
                temp_json = json.dumps(temp_dict)
                    
                self.test_set.add(temp_json)
                self.test_list.append(temp_dict)

            for line in dev_data:
                temp_dict = dict()
                temp_dict['reviewerID']=json.loads(line.decode('utf-8'))['reviewerID']
                temp_dict['reviewText']=textclean(json.loads(line)['reviewText'])
                temp_dict['summary']=textclean(json.loads(line)['summary'].lower())
                    # encode dict with json string in order to be hashable
                temp_json = json.dumps(temp_dict)
               
                self.dev_set.add(temp_json)
                self.dev_list.append(temp_dict)

            print("len of train list",len(self.train_list))
            print("len of dev list",len(self.dev_list))
            print("len of test list",len(self.test_list))
            print("------time for splitting test and dev data---",time.time()-t1)



    '''
    write train, dev, test to json files
    '''
    def write_json(self):

        test_str = json.dumps(self.test_list)
        train_str = json.dumps(self.train_list)
        dev_str = json.dumps(self.dev_list)
        # json.dumps thinks that the " is part of a the string, not part of the json formatting.
        # so json.loads(string) before json.dump
        print("writing train data and dev/test data to json file")
        print("find the output as rev.test.json/rev.dev.json and rev.train.json ")
        with open('rev.test.json', 'w') as f1:
  
            json.dump(json.loads(test_str),f1,ensure_ascii=False)
        with open('rev.train.json', 'w') as f2:
  
            json.dump(json.loads(train_str),f2,ensure_ascii=False)
        with open('rev.dev.json', 'w') as f3:
  
            json.dump(json.loads(dev_str),f3,ensure_ascii=False)


    '''get Vocabulary from training set. including UNK
        :param: input tokenized training text including summar and review
        :output: unigrams {unigrams:count}
    '''
    def word_freq(self):
       
        # full_text = ""
        t0 = time.time()
        train_rev_text = self.train_rev_text
        train_sum_text = self.train_sum_text
        
        # for record in self.train_set:
        #     record_j = json.loads(record.decode('utf-8'))
        #     temp_rev = record_j['reviewText']
        #     temp_sum = record_j['summary']
        #     train_sum_text = train_sum_text +" "+temp_sum
        #     train_rev_text = train_rev_text +" " +temp_rev
        # full_text= train_sum_text + " "+train_rev_text

        
        rev_tokens = tokenize(train_rev_text)
        sum_tokens = tokenize(train_sum_text)
        full_tokens = rev_tokens + sum_tokens
        t1 = time.time()
        print("tokenize done:")
        print("len of review tokens:",len(rev_tokens))
        print("len of sum tokens",len(sum_tokens))
        print("len of full tokens", len(full_tokens))
        print("-------------time tokenizing-----",t1-t0)
        # dict for summary and review data
        sum_V = {}
        rev_V = {}
            


        unigram_V = {}
        unigram_V[UNK_token] = 0
        # initial work-count dict population
        for token in full_tokens:
            unigram_V[token]= unigram_V.get(token,0) + 1
       
        # re-assign UNK
        unk_words = set()
        items = unigram_V.iteritems()
        for word, count in items:
            # treat low freq word as UNK
            if count <= self.unk_threshold:
                unk_words.add(word)
                unigram_V[UNK_token] += count
           
        
        unk_words.discard(UNK_token)

        for word in unk_words:
            del unigram_V[word]

        self.unigram_V = unigram_V

        t2 = time.time()
        print("-----time for unigram_V-----",t2-t1)

        replaced_sum_tokens = sum_tokens
        for idx,token in enumerate(replaced_sum_tokens):
            if token in unk_words:
                replaced_sum_tokens[idx]=UNK_token
                sum_V[UNK_token] = sum_V.get(UNK_token,0) +1
            else:

            
                sum_V[token] = sum_V.get(token,0) +1


        t3 = time.time()
        print("-----time for sum_V-----",t3-t2)

        replaced_rev_tokens = rev_tokens
        for idx,token in enumerate(replaced_rev_tokens):
            if token in unk_words:
                replaced_rev_tokens[idx]=UNK_token
                rev_V[UNK_token] = rev_V.get(UNK_token,0) +1
            else:

            
                rev_V[token] = rev_V.get(token,0) +1

        self.sum_V = sum_V
        self.rev_V = rev_V
           


#         replaced_tokens_train = training_tokens
#         for idx, token in enumerate(replaced_tokens_train):            
#             if token in unk_words:                
#                 replaced_tokens_train[idx] = UNK_token
# #             

        return None



           
            
       




    def initTextProcess(self):
        print("begin train-test split reviews")
        self.train_test_split()
        self.write_json()
        print("start generating word freq")
        self.word_freq()
        print self.sum_V[UNK_token]
        return None

    


def main():
    args = get_args()
    tp = TextProcess(args)
    tp.initTextProcess()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("full_dataset",action = "store",
    help = "training set")

    # parser.add_argument("dev_set", action="store",
    #                     help="either dev data or test data. dev for tunning, test can be used only once")
    parser.usage = ("NER_yanan.py [-h] [-n N] full_dataset")
    parser.add_argument("-t", "--threshold", action="store", type=int,
                        default=3, metavar='T',
                        help="threshold value for words to be UNK.")
    
    return parser.parse_args()


if __name__=="__main__":
    main()
