# %%
from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
from sklearn import linear_model
import random

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
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

# %%
##################################################
# Rating prediction                              #
##################################################

# %%
def getGlobalAverage(trainRatings):
    # Return the average rating in the training set

# %%
def trivialValidMSE(ratingsValid, globalAverage):
    # Compute and return the MSE of a trivial model that always returns the global mean computed above

# %%
def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    return newAlpha

# %%
def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    return newBetaU

# %%
def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    return newBetaI

# %%
def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    # Compute the MSE and the mse+regularization term
    return mse, mse + lamb*regularizer

# %%
def validMSE(ratingsValid, alpha, betaU, betaI):
    # Compute the MSE on the validation set
    return validMSE

# %%
def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    # Improve upon your model from the previous question (e.g. by running multiple iterations)
    return alpha, betaU, betaI

# %%
def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bu = 0
        bi = 0
        if u in betaU:
            bu = betaU[u]
        if b in betaI:
            bi = betaI[b]
        _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

    predictions.close()

# %%
##################################################
# Read prediction                                #
##################################################

# %%
def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    return readValid, notRead

# %%
def baseLineStrategy(mostPopular, totalRead):
    return1 = set()

    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break

    return return1

# %%
def improvedStrategy(mostPopular, totalRead):
    return1 = set()

    # Same as above function, just find an item set that'll have higher accuracy

    return return1

# %%
def evaluateStrategy(return1, readValid, notRead):

    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label

    return acc

# %%
def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular
    
    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0

# %%
def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()

# %%
##################################################
# Category prediction                            #
##################################################

# %%
def featureCat(datum, words, wordId, wordSet):
    feat = [0]*len(words)

    # Compute features counting instance of each word in "words"
    # after converting to lower case and removing punctuation
    
    feat.append(1) # offset (put at the end)
    return feat

# %%
def betterFeatures(data):
    
    # Produce better features than those from the above question
    # Return matrix (each row is the feature vector for one entry in the dataset)

    return X

# %%
def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)

# %%
def writePredictionsCategory(pred_test):
    predictions = open("predictions_Category.csv", 'w')
    pos = 0

    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()

# %%



