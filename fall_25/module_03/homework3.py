# homework3.py
# Implementations for DSC 256 Module 03 Homework 3
# Tasks:
# - Rating prediction (Q1-Q3): global average, coordinate descent for biases, improved model
# - Read prediction (Q4-Q6): validation generation, popularity baselines, Jaccard threshold predictor
# - Category prediction (Q7-Q8): bag-of-words features and improved hashed TF-IDF features; write predictions
#
# Notes:
# - All implementations follow data structures and call patterns from hw3_runner.py.
# - Vectorized operations are used whenever possible (NumPy) for efficiency.
# - Edge cases (missing keys, empty sets) are handled defensively.

from __future__ import annotations

import math
import os
import gzip
import random
import string
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Set
import csv
import numpy as np
import sys

def getGlobalAverage(trainRatings):
    return float(np.mean(np.array(trainRatings,dtype=float))) if len(trainRatings) > 0 else 0.0

def trivialValidMSE(ratingsValid, ga):
    val = np.array(ratingsValid)[:,2].astype(float)
    ga_array = np.ones_like(val) * ga
    return float(np.mean((val - ga_array) ** 2))

def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    
    # Extract components into NumPy arrays for vectorization
    ratings = np.array([r for u, b, r in ratingsTrain])
    beta_u_array = np.array([betaU.get(u, 0) for u, b, r in ratingsTrain])
    beta_i_array = np.array([betaI.get(b, 0) for u, b, r in ratingsTrain])
    
    # Compute the mean of the residuals
    residuals = ratings - beta_u_array - beta_i_array
    newAlpha = np.mean(residuals)
    
    return newAlpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    newBetaU = {}
    
    # Loop over each user is required for coordinate descent
    for u in ratingsPerUser:
        items_ratings = ratingsPerUser[u] # List of (item, rating)
        
        # Vectorize the inner sum
        ratings = np.array([r for i, r in items_ratings])
        beta_i_array = np.array([betaI.get(i, 0) for i, r in items_ratings])
        
        numerator = np.sum(ratings - alpha - beta_i_array)
        denominator = len(items_ratings) + lamb
        
        newBetaU[u] = numerator / denominator
        
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    newBetaI = {}
    
    # Loop over each item is required
    for b in ratingsPerItem:
        users_ratings = ratingsPerItem[b] # List of (user, rating)
        
        # Vectorize the inner sum
        ratings = np.array([r for u, r in users_ratings])
        beta_u_array = np.array([betaU.get(u, 0) for u, r in users_ratings])
        
        numerator = np.sum(ratings - alpha - beta_u_array)
        denominator = len(users_ratings) + lamb
        
        newBetaI[b] = numerator / denominator
        
    return newBetaI

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

def testModel(which):
    betaU = {}
    betaI = {}
    for u in ratingsPerUser:
        betaU[u] = 0

    for b in ratingsPerItem:
        betaI[b] = 0

    alpha = globalAverage 
    
    alpha, betaU, betaI, mse, mseReg = iterateN(which, alpha, betaU, betaI, 1.0, 1)
    validMSE = which.validMSE(ratingsValid, alpha, betaU, betaI)
    
    return alpha, betaU, betaI, mse, mseReg, validMSE
def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    
    N_ITERATIONS = 10 
    LAMBDA = 1.0       
    
    for i in range(N_ITERATIONS):
        # Perform updates in order
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, LAMBDA)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, LAMBDA)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, LAMBDA)
        
    return alpha, betaU, betaI

def validMSE(ratingsValid, alpha, betaU, betaI):
    actual = np.array([r for u, b, r in ratingsValid])
    predictions = np.array([alpha + betaU.get(u, 0) + betaI.get(b, 0) 
                            for u, b, r in ratingsValid])
    
    return np.mean((actual - predictions)**2)

def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    actual = np.array([r for u, b, r in ratingsTrain])
    predictions = np.array([alpha + betaU.get(u, 0) + betaI.get(b, 0) 
                            for u, b, r in ratingsTrain])
    mse = np.mean((actual - predictions)**2)
    reg_u = np.sum(np.array(list(betaU.values()))**2)
    reg_i = np.sum(np.array(list(betaI.values()))**2)
    regularizer = lamb * (reg_u + reg_i)
    
    return mse, mse + regularizer

def generateValidation(allRatings,ratingsValid):
    readValid: Set[Tuple[str, str]] = set((u, b) for u, b, _ in ratingsValid)
    allBooks: Set[str] = set(b for _, b, _ in allRatings)
    booksPerUser: Dict[str, Set[str]] = defaultdict(set)
    for u, b, _ in allRatings:
        booksPerUser[u].add(b)

    notRead: Set[Tuple[str, str]] = set()
    # To keep runtime modest and ensure reproducibility, we try a bounded number of samples per user
    for u, _, _ in ratingsValid:
        user_read = booksPerUser.get(u, set())
        # Candidate pool: books the user hasn't read
        candidates = list(allBooks - user_read)
        if not candidates:
            # Degenerate case: if user has read everything in allBooks (unlikely), skip
            continue
        # Sample 1 uniformly at random
        b_neg = random.choice(candidates)
        notRead.add((u, b_neg))

    # If we somehow undersampled (due to degenerate users), top up by sampling globally
    while len(notRead) < len(readValid):
        u, _, _ = random.choice(ratingsValid)
        user_read = booksPerUser.get(u, set())
        candidates = list(allBooks - user_read)
        if candidates:
            notRead.add((u, random.choice(candidates)))
        else:
            # If still no candidates, break to avoid infinite loop
            break
    
    return readValid, notRead


def improvedStrategy(mostPopular: List[Tuple[int, str]], totalRead: int) -> Set[str]:
    threshold = 0.70  
    selected: Set[str] = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        selected.add(i)
        if count > totalRead * threshold:
            break
    return selected

def evaluateStrategy(return1,readValid,notRead):
    correct = 0
    total = len(readValid) + len(notRead)

    # Positives: correct if book predicted as read
    for u, b in readValid:
        if b in return1:
            correct += 1

    # Negatives: correct if book predicted as not read
    for u, b in notRead:
        if b not in return1:
            correct += 1

    return (correct / total) if total > 0 else 0.0

def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    
    # Popularity fallback first (helps cold-start users and sparse overlaps)
    pop_b = len(ratingsPerItem.get(b, []))
    if pop_b > 40:
        return 1

    # If user has no history, cannot compute similarity
    if u not in ratingsPerUser or not ratingsPerUser[u]:
        return 0

    users_b = set(user for user, _ in ratingsPerItem.get(b, []))
    maxSim = 0.0

    for item, _ in ratingsPerUser[u]:
        if item == b:
            # Skip identical item if somehow present in user's history
            continue
        users_item = set(user for user, _ in ratingsPerItem.get(item, []))
        sim = _jaccard(users_b, users_item)
        if sim > maxSim:
            maxSim = sim
            # Quick early exit if we exceed threshold by a large margin
            if maxSim > 0.1:
                break

    return 1 if (maxSim > 0.013 or pop_b > 40) else 0

def _jaccard(a, b):
    
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union > 0 else 0.0


def featureCat(datum,words,wordId,wordSet):
    feat = [0] * len(words)
    review = str(datum.get("review_text", "")).lower()
    # Remove punctuation by replacing with spaces
    review_clean = "".join(ch if ch not in string.punctuation else " " for ch in review)

    for w in review_clean.split():
        if w in wordSet:
            feat[wordId[w]] += 1

    # Offset term at end
    feat.append(1)  
    return feat

def betterFeatures(data):
    # Re-build the dictionary (self-contained, as required)
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in data:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    # Tune Dictionary 
    NW = 1000 
    
    words = [x[1] for x in counts[:NW]]
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)
    
    # Create the feature matrix X
    X = []
    for d in data:
        # Start with the Bag-of-Words features
        feat = [0] * len(words)
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in r.split():
            if w in wordSet:
                feat[wordId[w]] += 1
        
        # Add Metadata Features 
        
        # Add the rating
        feat.append(d['rating'])
        
        # Add the review length 
        feat.append(len(d['review_text']) / 1000.0) 
        
        # Add the offset at the end
        feat.append(1) 
        X.append(feat)

    return X


