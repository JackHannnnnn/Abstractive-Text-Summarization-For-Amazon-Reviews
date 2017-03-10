from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
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


# remove punctutations and convert to lower
def textclean(s):
    # retain . and , ! ? ;
    for ch in '"#$%&()*+-/:<=>@[\\]^`{|}~,.!?':
        s = string.replace(s,ch,' ')
    return s.lower()

def tokenize(text):
        return nltk.word_tokenize(text)