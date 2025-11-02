# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
import random
from sklearn import linear_model
import numpy as np

# %%
import homework3

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
def countRight(a,b,epsilon):
    if len(a) != len(b):
        print("It looks like your solution has the wrong length (got " + str(len(a)) + ", expected "
 + str(len(b)) + ")")
        return 0
    a_ = np.array(a).flatten()
    b_ = np.array(b).flatten()
    right = np.abs(a_ - b_) < epsilon
    return float(sum(right) / len(right))

# %%
# Some data structures that will be useful

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
##################################################
# Rating prediction                              #
##################################################

# %%
trainRatings = [r[2] for r in ratingsTrain]
globalAverage = homework3.getGlobalAverage(trainRatings)

# %%
def testQ1():
    ga = homework3.getGlobalAverage(trainRatings)
    trivialValidMSE = homework3.trivialValidMSE(ratingsValid, ga)
    
    print("average = " + str(ga))
    print("validation MSE = " + str(trivialValidMSE))

# %%
testQ1()

# %%
def iterateN(which, alpha, betaU, betaI, lamb, N):
    for i in range(N):
        alpha = which.alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = which.betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = which.betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
        mse, mseReg = which.msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)
        print("Iteration " + str(i + 1))
        print("  MSE = " + str(mse))
        print("  regularized objective = " + str(mseReg))
    return alpha, betaU, betaI, mse, mseReg

# %%
def testModel(which):
    betaU = {}
    betaI = {}
    for u in ratingsPerUser:
        betaU[u] = 0

    for b in ratingsPerItem:
        betaI[b] = 0

    alpha = globalAverage # Could initialize anywhere, this is a guess
    
    alpha, betaU, betaI, mse, mseReg = iterateN(which, alpha, betaU, betaI, 1.0, 1)
    validMSE = which.validMSE(ratingsValid, alpha, betaU, betaI)
    
    return alpha, betaU, betaI, mse, mseReg, validMSE

# %%
def testQ2():
    alpha, betaU, betaI, mse, mseReg, validMSE = testModel(homework3)
    print("validMSE = " + str(validMSE))

# %%
testQ2()

# %%
def testQ3():
    betaU = {}
    betaI = {}
    for u in ratingsPerUser:
        betaU[u] = 0

    for b in ratingsPerItem:
        betaI[b] = 0

    alpha = globalAverage # Could initialize anywhere, this is a guess
    
    alpha, betaU, betaI = homework3.goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI)
    validMSE = homework3.validMSE(ratingsValid, alpha, betaU, betaI)
    
    print("validMSE = " + str(validMSE))

# %%
testQ3()

# %%
##################################################
# Read prediction                                #
##################################################

# %%
# From baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

# %%
def testQ4():
    readValid, notRead = homework3.generateValidation(allRatings, ratingsValid)
    print("Should be equal: " + str((len(readValid), len(notRead), len(ratingsValid))))

# %%
testQ4()

# %%
def testQ5():
    return1 = homework3.baseLineStrategy(mostPopular, totalRead)
    better = homework3.improvedStrategy(mostPopular, totalRead)
    
    readValid, notRead = homework3.generateValidation(allRatings, ratingsValid)
    
    correctA = homework3.evaluateStrategy(return1, readValid, notRead)
    correctB = homework3.evaluateStrategy(better, readValid, notRead)
    
    print("Accuracy (simple strategy) = " + str(correctA))
    print("Accuracy (better strategy) = " + str(correctB))

# %%
testQ5()

# %%
def testQ6():
    readValid, notRead = homework3.generateValidation(allRatings, ratingsValid)
    
    for (u,b) in list(readValid)[:20] + list(notRead)[:20]:
        a = homework3.jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        print("Jaccard-based predictor for " + str((u,b)) + " = " + str(a))

    # This is slow (so the autograder doesn't run it) but you should run it at home once you have a good solution
    #homework3.writePredictionsRead(ratingsPerItem, ratingsPerUser)

# %%
testQ6()

# %%
##################################################
# Category prediction                            #
##################################################

# %%
data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)
    # Just use a little data to make things faster...
    if len(data) > 10000:
        break

# %%
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
    r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
    for w in r.split():
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

# %%
NW = 500 # dictionary size

# %%
words = [x[1] for x in counts[:NW]]

# %%
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

# %%
def testQ7():
    f1 = homework3.featureCat(data[0], words, wordId, wordSet)
    
    print("Feature vector = " + str(f1))

# %%
testQ7()

# %%
def testQ8():
    X = [homework3.featureCat(d, words, wordId, wordSet) for d in data]
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
    
    X = homework3.betterFeatures(data)
    Xtrain = X[:9*len(X)//10]
    Xvalid = X[9*len(X)//10:]
    
    mod = linear_model.LogisticRegression(C=1)
    mod.fit(Xtrain, ytrain)
    pred = mod.predict(Xvalid)
    correctB = pred == yvalid
    correctB = sum(correctB) / len(correctB)
    
    sc = correctA < (correctB * 0.99)

    data_test = []
    for d in readGz("test_Category.json.gz"):
        data_test.append(d)
    
    Xtest = homework3.betterFeatures(data_test)
    pred_test = mod.predict(Xtest)
    
    homework3.writePredictionsCategory(pred_test)
    
    if sc:
        print("Looks like your solution is better")
    else:
        print("Looks like your solution is not better")

# %%
testQ8()

# %%



