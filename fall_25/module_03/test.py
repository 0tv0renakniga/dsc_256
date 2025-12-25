from sklearn import linear_model
import gzip
import numpy as np
from collections import defaultdict
import math
import scipy.optimize
import string
import random
import os
import ast
import json
import re

    # Without raw string (requires double backslashes for regex)
pattern_normal = "\\d+"

    # With raw string (single backslashes for regex)
pattern_raw = r"\d+"

    
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
data = []
"""

for d in readGz("train_Category.json.gz"):
    data.append(d)
    # Just use a little data to make things faster...
    if len(data) > 10000:
        break
"""
# Read the first 100 lines of the file
lines = [ast.literal_eval(re.match(pattern_raw, d)) for d in os.popen("head -100 train_Category.json").read().splitlines()]
print(len(lines))
data = []


wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
    r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
    for w in r.split():
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

NW = 500 # dictionary size

words = [x[1] for x in counts[:NW]]

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def featureCat(datum, words, wordId, wordSet):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in string.punctuation])
    for w in r.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat

def testQ8():
    X = [featureCat(d, words, wordId, wordSet) for d in data]
    y = [d['genreID'] for d in data]
    
    Xtrain = X[:9*len(X)//10]
    ytrain = y[:9*len(y)//10]
    Xvalid = X[9*len(X)//10:]
    yvalid = y[9*len(y)//10:]
    
    mod = linear_model.LogisticRegression(C=1)
    mod.fit(Xtrain, ytrain)
    pred = mod.predict(Xvalid)
    correctA = pred == yvalid
    correctA = sum(correctA) / len(correctA)
    print(correctA)

testQ8()