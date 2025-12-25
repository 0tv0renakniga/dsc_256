# %%
import gzip
import math
import numpy as np
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn import linear_model
from dateutil import parser

# %%
import homework4

# %%
def MSE(predictions, y):
    diffs = [(a-b)**2 for (a,b) in zip(predictions, y)]
    return sum(diffs)/len(diffs)

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
def testQ1():
    dset = dataset[:5000]
    
    mostCommonUnigrams, mostCommonBigrams, mostCommonBoth = homework4.counts(dset)
    
    for which, wList in ("unigrams", mostCommonUnigrams), ("bigrams", mostCommonBigrams), ("both", mostCommonBoth):
        clf, theta, predictions, mse = homework4.fitModel(wList, which, dataset)
        print(which + " MSE = " + str(mse))

# %%
testQ1()

# %%
def testQ2():
    dset = dataset[:5000]
    mostCommonUnigrams, _, _ = homework4.counts(dset)
    wordSet = set(mostCommonUnigrams)
    query = dataset[0] # Query review
    similarities = homework4.similarities(query, dset, wordSet)
    
    print(similarities[:10])

# %%
testQ2()

# %%
def testQ3():
    dset = dataset[:5000]
    sentences = homework4.makeUserSentences(dset)

    random.seed(0)
    model10 = homework4.makeWVmodel(sentences)

    scores = model10.wv.similar_by_word(dataset[0]['book_id'])
    
    print(scores)

# %%
testQ3()

# %%
def testQ4():
    dset = dataset[:1000]
    sentences = homework4.makeUserSentences(dset)
    
    random.seed(0)
    model10 = homework4.makeWVmodel(sentences)
    
    itemAverages = defaultdict(list)
    ratingMean = []

    for d in dset:
        itemAverages[d['book_id']].append(d['rating'])
        ratingMean.append(d['rating'])

    for b in itemAverages:
        itemAverages[b] = sum(itemAverages[b]) / len(itemAverages[b])

    ratingMean = sum(ratingMean) / len(ratingMean)

    reviewsPerUser = defaultdict(list)

    for d in dset:
        reviewsPerUser[d['user_id']].append(d)
        
    dset = dset[:200]
    
    alwaysPredictMean = [ratingMean for d in dset]
    labels = [d['rating'] for d in dset]

    simPredictions =\
        [homework4.predictRating(d['user_id'], d['book_id'], model10, ratingMean, itemAverages, reviewsPerUser) for d in dset]

    betterPredictions =\
        [homework4.betterPredictRating(d['user_id'], d['book_id'], model10, ratingMean, itemAverages, reviewsPerUser) for d in dset]
    
    print("MSE (always predict mean, 200 samples) = " + str(MSE(alwaysPredictMean, labels)))
    print("MSE (your solution) = " + str(MSE(simPredictions, labels)))
    print("MSE (better solution) = " + str(MSE(betterPredictions, labels)))

# %%
testQ4()

# %%



