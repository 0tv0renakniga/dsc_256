import gzip
import math
import numpy
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn import linear_model
from dateutil import parser


# In[ ]:


def MSE(predictions, y):
    diffs = [(a-b)**2 for (a,b) in zip(predictions, y)]
    return sum(diffs)/len(diffs)


# In[ ]:


punctuation = set(string.punctuation)


# In[3]:


def counts(dataset):
    wordCount = defaultdict(int)
    for d in dataset:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        ws = r.split()
        ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
        for w in ws + ws2:
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()
    
    mostCommonUnigrams = [x[1] for x in counts if len(x[1].split()) == 1][:1000]
    mostCommonBigrams = [x[1] for x in counts if len(x[1].split()) == 2][:1000]
    mostCommonBoth = [x[1] for x in counts][:1000]
    
    # Bigrams should be strings with a space in the middle, e.g. "like this"
    return mostCommonUnigrams, mostCommonBigrams, mostCommonBoth


# In[5]:


def feature(datum, wordId, wordSet, which):
    # "which" is one of "unigrams", "bigrams", or "both"
    feat = [0]*len(wordSet)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    ws = r.split()
    ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
    for w in ws + ws2:
        if w in wordSet:
            feat[wordId[w]] += 1
    # Do not include an offset term, it'll be done by the library
    return feat


# In[12]:


def fitModel(wList, which, dataset):
    wordId = dict(zip(wList, range(len(wList))))
    wordSet = set(wList)
    X = [feature(d, wordId, wordSet, which) for d in dataset]
    y = [d['rating'] for d in dataset]
    
    clf = linear_model.Ridge(1.0, fit_intercept=True) # MSE + 1.0 l2
    clf.fit(X, y)
    theta = clf.coef_
    predictions = clf.predict(X)
    mse = MSE(predictions, y)
    
    return clf, theta, predictions, mse


# In[13]:


def DF(dataset, wordSet):
    df = defaultdict(int)
    for d in dataset:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in set(r.split()):
            if w in wordSet:
                df[w] += 1
    return df


# In[15]:


def TF(query, wordSet):
    tf = defaultdict(int)
    r = ''.join([c for c in query.lower() if not c in punctuation])
    for w in r.split():
        if w in wordSet:
            # Note = rather than +=, different versions of tf could be used instead
            tf[w] = 1
    return tf


# In[ ]:


def TFIDF(query, df, dataset, wordSet):
    tf = TF(query, wordSet)
    tfidfQuery = [tf[w] * math.log2(len(dataset) / df[w]) for w in wordSet]
    #tfidf = dict(zip(words,[tf[w] * math.log2(len(dataset) / df[w]) for w in words]))
    return tfidfQuery


# In[16]:


def Cosine(x1,x2):
    numer = 0
    norm1 = 0
    norm2 = 0
    for a1,a2 in zip(x1,x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0
####
def similarities(query, dataset, wordSet):
    similarities = []
    df = DF(dataset, wordSet)
    
    # 1. Freeze the order of the features for Vector consistency
    wordSetList = list(wordSet)
    
    # 2. Extract Query info
    if isinstance(query, dict):
        query_text = query['review_text']
        query_id = query.get('review_id', None)
    else:
        query_text = query
        query_id = None
        
    # 3. Compute Query Vector
    tfidfQuery = TFIDF(query_text, df, dataset, wordSetList)
    
    for rev2 in dataset:
        # Skip if it is the exact same object or ID
        if query_id and rev2.get('review_id') == query_id:
            continue
        if rev2 is query:
            continue
            
        # 4. Compute Doc Vector (using SAME wordSetList)
        tfidf2 = TFIDF(rev2['review_text'], df, dataset, wordSetList)
        
        sim = Cosine(tfidfQuery, tfidf2)
        similarities.append((sim, rev2))
        
    # Sort by similarity score descending
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return similarities

###

# In[17]:


def similarities(query, dataset, wordSet):
    similarities = []
    df = DF(dataset, wordSet)
    tfidfQuery = TFIDF(query['review_text'], df, dataset, wordSet)
    for rev2 in dataset:
        if rev2['review_text'] == query['review_text']: continue
        tfidf2 = TFIDF(rev2['review_text'], df, dataset, wordSet)
        similarities.append((Cosine(tfidfQuery, tfidf2), rev2['review_text']))
    return similarities


# In[21]:


def makeUserSentences(dataset):
    # For every user, make a "sentence" (a list of item IDs) containing their user history, ordered by time
    # Returned value should be a list of lists of book_ids
    reviewsPerUser = defaultdict(list)
    
    for d in dataset:
        reviewsPerUser[d['user_id']].append((parser.parse(d['date_added']), d['book_id']))
        
    sentences = []
    
    for u in reviewsPerUser:
        reviewsPerUser[u].sort()
        sentences.append([x[1] for x in reviewsPerUser[u]])
        
    return sentences


# In[ ]:


def makeWVmodel(sentences):
    random.seed(0)
    model10 = Word2Vec(sentences,
                     min_count=1, # Words/items with fewer instances are discarded
                     vector_size=10, # Model dimensionality
                     window=3, # Window size
                     sg=1) # Skip-gram model
    return model10


# In[36]:


def predictRating(user, item, model10, ratingMean, itemAverages, reviewsPerUser):
    # reviewsPerUser maps a user ID to a list of their reviews
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        c = Cosine(model10.wv[item], model10.wv[i2])
        if False:
            if c < 0.01:
                c = 0.01
        similarities.append(c)
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[37]:


def betterPredictRating(user, item, model10, ratingMean, itemAverages, reviewsPerUser):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        c = Cosine(model10.wv[item], model10.wv[i2])
        if True:
            if c < 0.01:
                c = 0.01
        similarities.append(c)
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


###############
# TEST FOR CODE
###############

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


# In[2]:


import homework4
import hw4_reference as reference


# In[3]:


def MSE(predictions, y):
    diffs = [(a-b)**2 for (a,b) in zip(predictions, y)]
    return sum(diffs)/len(diffs)


# In[4]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


def countRight(a,b,epsilon):
    if len(a) != len(b):
        print("It looks like your solution has the wrong length (got " + str(len(a)) + ", expected "
 + str(len(b)) + ")")
        return 0
    a_ = np.array(a).flatten()
    b_ = np.array(b).flatten()
    right = np.abs(a_ - b_) < epsilon
    return float(sum(right) / len(right))


# In[7]:


dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()


# In[47]:


def testQ1():
    mostCommonUnigrams, mostCommonBigrams, mostCommonBoth = homework4.counts(dataset[:10000])
    mostCommonUnigrams_, mostCommonBigrams_, mostCommonBoth_ = reference.counts(dataset[:10000])
    random.seed(0)
    matches = []
    for a,b in (mostCommonUnigrams, mostCommonUnigrams_),\
                (mostCommonBigrams, mostCommonBigrams_),\
                (mostCommonBoth, mostCommonBoth_):
        if len(a) != len(b):
            print("Returned set has wrong length")
            return 0
        for i in range(100):
            x = random.choice(a)
            matches.append(x in b)
    sc1 = sum(matches) / len(matches)
    
    dset = dataset[:5000]
    
    mses = []
    mses_ = []
    
    for which, wList in ("unigrams", mostCommonUnigrams), ("bigrams", mostCommonBigrams), ("both", mostCommonBoth):
        clf, theta, predictions, mse = homework4.fitModel(wList, which, dataset)
        clf, theta, predictions, mse_ = reference.fitModel(wList, which, dataset)
        
        mses.append(mse)
        mses_.append(mse_)

    
    sc2 = countRight(mses, mses_, 0.02)

    print("Are your word lists right: " + str(sc1))
    print("Are your MSEs right: " + str(sc2))
    
    return sc1 + sc2


# In[48]:


#testQ1()


# In[45]:


def testQ2():
    dset = dataset[:5000]
    mostCommonUnigrams, _, _ = reference.counts(dset)
    wordSet = set(mostCommonUnigrams)
    query = dataset[0] # Query review
    similarities = homework4.similarities(query, dset, wordSet)
    similarities_ = reference.similarities(query, dset, wordSet)
    
    return 2.0 * countRight([x[0] for x in similarities[:20]], [x[0] for x in similarities_[:20]], 0.02)


# In[46]:


#testQ2()


# In[38]:


def testQ3():
    dset = dataset[:5000]
    sentences = homework4.makeUserSentences(dset)
    sentences_ = reference.makeUserSentences(dset)
    
    random.seed(0)
    matches = []
    for i in range(100):
        r = random.choice(sentences_)
        if not (r in sentences):
            print("Expected " + str(r) in sentences)
            return 0
    sc = 1.0

    model10 = homework4.makeWVmodel(sentences_)
    model10_ = reference.makeWVmodel(sentences_)

    scores = model10.wv.similar_by_word(dataset[0]['book_id'])
    scores_ = model10_.wv.similar_by_word(dataset[0]['book_id'])
    
    return 2.0 * countRight([x[1] for x in scores[:10]], [x[1] for x in scores_[:10]], 0.03)


# In[39]:


#testQ3()


# In[40]:


def testQ4():
    dset = dataset[:1000]
    sentences = reference.makeUserSentences(dset)
    
    random.seed(0)
    model10 = reference.makeWVmodel(sentences)
    
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
    simPredictions_ =\
        [reference.predictRating(d['user_id'], d['book_id'], model10, ratingMean, itemAverages, reviewsPerUser) for d in dset]
    
    betterPredictions =\
        [homework4.betterPredictRating(d['user_id'], d['book_id'], model10, ratingMean, itemAverages, reviewsPerUser) for d in dset]
    
    print("MSE (always predict mean, 200 samples) = " + str(MSE(alwaysPredictMean, labels)))
    print("MSE (reference solution) = " + str(MSE(simPredictions_, labels)))
    print("MSE (your solution) = " + str(MSE(simPredictions, labels)))
    print("MSE (better solution) = " + str(MSE(betterPredictions, labels)))
    
    sc1 = countRight(simPredictions, simPredictions, 50)
    sc2 = 1.0 * (MSE(betterPredictions, labels) < (0.9*MSE(alwaysPredictMean, labels)))
    
    print("Does your function match the reference (this test is loose due to instability of the problem): " + str(sc1))
    print("Is your better solution better: " + str(sc2))
    
    return sc1 + sc2
