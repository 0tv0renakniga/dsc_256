import gzip
import json
import dateutil.parser
import random
import numpy as np
from collections import defaultdict
from sklearn import linear_model
import math

def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    if includeCat:
        # My implementation is modular such that this one function concatenates all three features together,
        # depending on which are selected
    if includeReview:
        #
    if includeLength:
        #
    return feat + [1]

def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')

    maxLength = max([len(d['review/text']) for d in dataTrain])

    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]

    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    # (1) Fit the model on the training set
    # (2) Compute validation BER
    # (3) Compute test BER

    return mod, vBER, tBER

def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER

def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER

def Q3(catID, dataTrain, dataValid, dataTest):
    # Your solution here...
    for c in [0.001, 0.01, 0.1, 1, 10]:
        #
    # Return the validBER and testBER for the model that works best on the validation set
    return mod, validBER, testBER

def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength

def Jaccard(s1, s2):
    # Implement
    pass

def mostSimilar(i, N, usersPerItem):
    # Implement...

    # Should be a list of (similarity, itemID) pairs
    return similarities[:N]

def MSE(y, ypred):
    # Implement...
    pass

def getMeanRating(dataTrain):
    # Implement...

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    # Implement...
    return itemAverages

def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Solution for Q6, should return a rating
    return 0

def predictRatingQ7(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Your solution here
    return 0
