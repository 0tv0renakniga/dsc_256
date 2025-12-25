import gzip
import math
import string
import random
from collections import defaultdict
from dateutil import parser
from sklearn import linear_model
from gensim.models import Word2Vec

# Helper function with explicit loops

def MSE(predictions, y):
    diffs = [(a-b)**2 for (a,b) in zip(predictions, y)]
    return sum(diffs)/len(diffs)

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

def counts(dataset):
    wordCount = defaultdict(int)
    def clean_text(text):
        punctuation = set(string.punctuation)
        chars = []
        for c in text.lower():
            if c not in punctuation:
                chars.append(c)
        r = ''.join(chars)
        return r.split()
    
    for d in dataset:
        ws = clean_text(d['review_text'])
        
        # Count Unigrams
        for w in ws:
            wordCount[w] += 1
            
        # Loop through words to create bigrams
        for i in range(len(ws) - 1):
            bigram = ws[i] + " " + ws[i+1]
            wordCount[bigram] += 1

    counts_list = []
    for w, count in wordCount.items():
        counts_list.append((count, w))
    
    counts_list.sort(reverse=True)
    
    # get top 1000 Unigrams
    mostCommonUnigrams = []
    for count, w in counts_list:
        if len(w.split()) == 1:
            mostCommonUnigrams.append(w)
        if len(mostCommonUnigrams) >= 1000:
            break
            
    # get top 1000 Bigrams
    mostCommonBigrams = []
    for count, w in counts_list:
        if len(w.split()) == 2:
            mostCommonBigrams.append(w)
        if len(mostCommonBigrams) >= 1000:
            break
            
    # get top 1000 of Both
    mostCommonBoth = []
    for count, w in counts_list:
        mostCommonBoth.append(w)
        if len(mostCommonBoth) >= 1000:
            break
    
    return mostCommonUnigrams, mostCommonBigrams, mostCommonBoth

def feature(datum, wordId, wordSet, which):
    feat = [0] * len(wordSet)
    def clean_text(text):
        punctuation = set(string.punctuation)
        chars = []
        for c in text.lower():
            if c not in punctuation:
                chars.append(c)
        r = ''.join(chars)
        return r.split()
    ws = clean_text(datum['review_text'])
    ws2 = []
    for i in range(len(ws) - 1):
        ws2.append(ws[i] + " " + ws[i+1])

    # Update feature vector
    for w in ws + ws2:
        if w in wordSet:
            feat[wordId[w]] += 1
            
    return feat

def fitModel(wList, which, dataset):
    wordId = dict(zip(wList, range(len(wList))))
    wordSet = set(wList)
    
    # Expand list comprehensions for X and y
    X = []
    y = []
    for d in dataset:
        X.append(feature(d, wordId, wordSet, which))
        y.append(d['rating'])
    
    clf = linear_model.Ridge(1.0, fit_intercept=True) 
    clf.fit(X, y)
    theta = clf.coef_
    predictions = clf.predict(X)
    mse = MSE(predictions, y)
    
    return clf, theta, predictions, mse

def DF(dataset, wordSet):
    def clean_text(text):
        punctuation = set(string.punctuation)
        chars = []
        for c in text.lower():
            if c not in punctuation:
                chars.append(c)
        r = ''.join(chars)
        return r.split()
    df = defaultdict(int)
    for d in dataset:
        ws = set(clean_text(d['review_text']))
        for w in ws:
            if w in wordSet:
                df[w] += 1
    return df

def TF(query, wordSet):
    def clean_text(text):
        punctuation = set(string.punctuation)
        chars = []
        for c in text.lower():
            if c not in punctuation:
                chars.append(c)
        r = ''.join(chars)
        return r.split()
    tf = defaultdict(int)
    ws = clean_text(query)
    for w in ws:
        if w in wordSet:
            tf[w] = 1
    return tf

def TFIDF(query, df, dataset, wordSet):
    tf = TF(query, wordSet)
    N = len(dataset)
    tfidfQuery = []
    
    for w in wordSet:
        # Avoid division by zero
        val = tf[w] * math.log2(N / df[w])
        tfidfQuery.append(val)
        
    return tfidfQuery

def Cosine(x1, x2):
    numer = 0
    norm1 = 0
    norm2 = 0
    for a1, a2 in zip(x1, x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0

def similarities(query, dataset, wordSet):
    similarities = []
    df = DF(dataset, wordSet)
    
    wordSetList = list(wordSet)
        
    tfidfQuery = TFIDF(query['review_text'], df, dataset, wordSetList)
    
    for rev2 in dataset:
        if rev2['review_text'] == query['review_text']:
            continue
            
        tfidf2 = TFIDF(rev2['review_text'], df, dataset, wordSetList)
        sim = Cosine(tfidfQuery, tfidf2)
        similarities.append((sim, rev2['review_text']))
        
    return similarities

def makeUserSentences(dataset):
    reviewsPerUser = defaultdict(list)
    
    for d in dataset:
        reviewsPerUser[d['user_id']].append((parser.parse(d['date_added']), d['book_id']))
        
    sentences = []
    for u in reviewsPerUser:
        reviewsPerUser[u].sort()
        # Expand the extraction of book_ids
        user_books = []
        for x in reviewsPerUser[u]:
            user_books.append(x[1])
        sentences.append(user_books)
        
    return sentences

def makeWVmodel(sentences):
    random.seed(0)
    model10 = Word2Vec(sentences,
                     min_count=1, 
                     vector_size=10, 
                     window=3, 
                     sg=1) 
    return model10

def predictRating(user, item, model10, ratingMean, itemAverages, reviewsPerUser):
    ratings = []
    similarities = []
    
    if item not in model10.wv:
        return ratingMean
        
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        if i2 not in model10.wv: continue
            
        ratings.append(d['rating'] - itemAverages[i2])
        c = Cosine(model10.wv[item], model10.wv[i2])
        similarities.append(c)
        
    if sum(similarities) > 0:
        weightedRatings = []
        for x, y in zip(ratings, similarities):
            weightedRatings.append(x * y)
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean

def betterPredictRating(user, item, model10, ratingMean, itemAverages, reviewsPerUser):
    ratings = []
    similarities = []
    
    if item not in model10.wv:
        return ratingMean
        
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        if i2 not in model10.wv: continue

        ratings.append(d['rating'] - itemAverages[i2])
        c = Cosine(model10.wv[item], model10.wv[i2])
        
        if c < 0.5:
            c = 0.5
            
        similarities.append(c)
        
    if sum(similarities) > 0:
        weightedRatings = []
        for x, y in zip(ratings, similarities):
            weightedRatings.append(x * y)
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean