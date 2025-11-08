import gzip
from collections import defaultdict
import random
import numpy as np
import os
import string
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

def Jaccard(s1, s2):
    """
    Computes Jaccard similarity between two sets.
    """
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

# =================================================================
# TASK 1: RATING PREDICTION (Advanced: Matrix Factorization)
# =================================================================

print("Starting Task 1: Rating Prediction...")
print("Training the Matrix Factorization model (alpha + bu + bi + pu*qi)...")

# --- 1. Read Training Data ---
# We read the data once and build all data structures needed
allRatings = []
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set) # Needed for Jaccard (Task 2)
userIDs = {} # Map user string to integer ID
itemIDs = {} # Map item string to integer ID
interactions = [] # List of (user_int_id, item_int_id, rating)

for user, book, r in readCSV("train_Interactions.csv.gz"):
  r = int(r)
  
  # Get integer IDs for users and items for matrix ops
  if user not in userIDs: userIDs[user] = len(userIDs)
  if book not in itemIDs: itemIDs[book] = len(itemIDs)
  u_int = userIDs[user]
  i_int = itemIDs[book]
  
  allRatings.append(r)
  interactions.append((u_int, i_int, r))
  
  # These dicts still use the original string IDs
  ratingsPerUser[user].append((book, r))
  ratingsPerItem[book].append((user, r))
  usersPerItem[book].add(user)

nUsers = len(userIDs)
nItems = len(itemIDs)
alpha = np.mean(allRatings)

# --- 2. Train Biases (alpha, beta_u, beta_i) ---
# We first train the simple bias model, as its results
# are a great starting point for the full model.

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    """ Update all user biases (beta_u) """
    newBetaU = {}
    for u in ratingsPerUser:
        items_ratings = ratingsPerUser[u]
        ratings = np.array([r for i, r in items_ratings])
        beta_i_array = np.array([betaI.get(i, 0) for i, r in items_ratings])
        numerator = np.sum(ratings - alpha - beta_i_array)
        denominator = len(items_ratings) + lamb
        newBetaU[u] = numerator / denominator
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    """ Update all item biases (beta_i) """
    newBetaI = {}
    for b in ratingsPerItem:
        users_ratings = ratingsPerItem[b]
        ratings = np.array([r for u, r in users_ratings])
        beta_u_array = np.array([betaU.get(u, 0) for u, r in users_ratings])
        numerator = np.sum(ratings - alpha - beta_u_array)
        denominator = len(users_ratings) + lamb
        newBetaI[b] = numerator / denominator
    return newBetaI

LAMBDA_BIAS = 1.0
N_ITERATIONS_BIAS = 10
betaU = {u: 0 for u in ratingsPerUser}
betaI = {b: 0 for b in ratingsPerItem}

print(f"Iterating {N_ITERATIONS_BIAS} times to learn biases...")
for i in range(N_ITERATIONS_BIAS):
    # We only update betas, as alpha is stable
    betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, LAMBDA_BIAS)
    betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, LAMBDA_BIAS)
    print(f"  Bias Iteration {i+1}/{N_ITERATIONS_BIAS} complete.")

print("Bias training complete.")

# --- 3. Train Latent Factors (p_u, q_i) using SGD ---
# 
K = 20  # Number of latent factors
GAMMA = 0.01 # Learning rate
LAMBDA_MF = 0.01 # Regularization
EPOCHS = 20 # Number of passes over the data

# Initialize latent factor matrices with small random numbers
P = np.random.normal(scale=1./K, size=(nUsers, K))
Q = np.random.normal(scale=1./K, size=(nItems, K))

# Convert bias dicts to numpy arrays for fast lookup
betaU_np = np.array([betaU.get(u, 0) for u in userIDs])
betaI_np = np.array([betaI.get(i, 0) for i in itemIDs])

print(f"Training {K} latent factors with SGD for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    random.shuffle(interactions) # Shuffle data for SGD
    
    for u_int, i_int, rating in interactions:
        # Get current values
        bu = betaU_np[u_int]
        bi = betaI_np[i_int]
        pu = P[u_int]
        qi = Q[i_int]
        
        # Make prediction and calculate error
        pred = alpha + bu + bi + np.dot(pu, qi)
        error = rating - pred
        
        # Update biases and factors (the "gradient descent" step)
        betaU_np[u_int] += GAMMA * (error - LAMBDA_MF * bu)
        betaI_np[i_int] += GAMMA * (error - LAMBDA_MF * bi)
        P[u_int] += GAMMA * (error * qi - LAMBDA_MF * pu)
        Q[i_int] += GAMMA * (error * pu - LAMBDA_MF * qi)
        
    print(f"  SGD Epoch {epoch+1}/{EPOCHS} complete.")

print("Matrix Factorization training complete.")

# --- 4. Write Rating Predictions ---
def predictRating(user_str, item_str):
    """ Predicts a rating using the full MF model """
    # Get integer IDs, default to -1 if not seen
    u_int = userIDs.get(user_str, -1)
    i_int = itemIDs.get(item_str, -1)
    
    # Start with global average
    pred = alpha
    
    # Add biases if user/item exists
    if u_int != -1: pred += betaU_np[u_int]
    if i_int != -1: pred += betaI_np[i_int]
    
    # Add dot product if *both* exist
    if u_int != -1 and i_int != -1:
        pred += np.dot(P[u_int], Q[i_int])
        
    return pred

print(f"Writing predictions to predictions_Rating.csv...")
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
  if l.startswith("userID"):
    predictions.write(l)
    continue
  u,b = l.strip().split(',')
  prediction = predictRating(u, b)
  predictions.write(u + ',' + b + ',' + str(prediction) + '\n')
predictions.close()
print("Rating predictions finished.")

# =================================================================
# TASK 2: READ PREDICTION (High-Performance Jaccard Model)
# =================================================================
# (This task re-uses data structures from Task 1)

print("\nStarting Task 2: Read Prediction...")
print("Writing 'predictions_Read.csv' using Jaccard model...")

def jaccardThresh(u, b):
    """
    Predicts 'read' (1) if the book 'b' is similar to books
    the user 'u' has already read.
    """
    maxSim = 0
    # Use data from Task 1
    users_b = usersPerItem.get(b, set()) # Users who read book b
    
    if u not in ratingsPerUser:
        if len(users_b) > 40: return 1 # Popularity fallback
        return 0
    
    # For books u has read
    for item_prime, rating in ratingsPerUser[u]: 
        if item_prime == b: continue
        users_item_prime = usersPerItem.get(item_prime, set())
        sim = Jaccard(users_b, users_item_prime)
        maxSim = max(maxSim, sim)
    
    # These thresholds are tuned for high performance
    if maxSim > 0.013 or len(users_b) > 40:
        return 1
    return 0

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
  if l.startswith("userID"):
    predictions.write(l)
    continue
  u,b = l.strip().split(',')
  read_prediction = jaccardThresh(u, b)
  predictions.write(u + ',' + b + ',' + str(read_prediction) + '\n')
predictions.close()
print("Read predictions finished.")

# =================================================================
# TASK 3: CATEGORY PREDICTION (High-Performance TF-IDF Model)
# =================================================================

print("\nStarting Task 3: Category Prediction...")
print("Training Logistic Regression model with TF-IDF...")

# --- 1. Load Training Data ---
data_train = []
for d in readGz("train_Category.json.gz"):
    data_train.append(d)

reviews_train = [d['review_text'] for d in data_train]
y_train = [d['genreID'] for d in data_train]

# --- 2. Build Features (using optimized TF-IDF Vectorizer) ---
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
print("Building TF-IDF feature matrix...")
X_train_sparse = vectorizer.fit_transform(reviews_train)

# --- 3. Train the sklearn Model ---
print("Fitting model...")
mod = linear_model.LogisticRegression(C=1.0, max_iter=1000)
mod.fit(X_train_sparse, y_train)
print("Category model training complete.")

# --- 4. Write Category Predictions ---
print("Writing 'predictions_Category.csv' using TF-IDF model...")
predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")

test_data = list(readGz("test_Category.json.gz"))
reviews_test = [l['review_text'] for l in test_data]
X_test_sparse = vectorizer.transform(reviews_test)
all_predictions = mod.predict(X_test_sparse)

for i, l in enumerate(test_data):
    pred = all_predictions[i]
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(pred) + "\n")

predictions.close()
print("Category predictions finished.")
print("\nAll tasks complete.")

def predictRating(user, item):
    """ Predicts a rating using the trained model """
    bu = betaU.get(user, 0)
    bi = betaI.get(item, 0)
    return alpha + bu + bi

# --- 4. Write Rating Predictions ---
print(f"Writing predictions to predictions_Rating.csv...")
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
  if l.startswith("userID"):
    predictions.write(l)
    continue
  u,b = l.strip().split(',')
  prediction = predictRating(u, b)
  predictions.write(u + ',' + b + ',' + str(prediction) + '\n')
predictions.close()
print("Rating predictions finished.")

# =================================================================
# TASK 2: READ PREDICTION (High-Performance Jaccard Model)
# =================================================================
# (This task was not memory-intensive, so it remains the same)

print("Writing 'predictions_Read.csv' using Jaccard model...")

def jaccardThresh(u, b):
    """
    Predicts 'read' (1) if the book 'b' is similar to books
    the user 'u' has already read.
    """
    maxSim = 0
    users_b = usersPerItem.get(b, set()) # Users who read book b
    
    if u not in ratingsPerUser:
        if len(users_b) > 40: return 1 # Popularity fallback
        return 0
    
    for item_prime, rating in ratingsPerUser[u]: # For books u has read
        if item_prime == b: continue
        users_item_prime = usersPerItem.get(item_prime, set())
        sim = Jaccard(users_b, users_item_prime)
        maxSim = max(maxSim, sim)
    
    if maxSim > 0.013 or len(users_b) > 40:
        return 1
    return 0

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
  if l.startswith("userID"):
    predictions.write(l)
    continue
  u,b = l.strip().split(',')
  read_prediction = jaccardThresh(u, b)
  predictions.write(u + ',' + b + ',' + str(read_prediction) + '\n')
predictions.close()
print("Read predictions finished.")

# =================================================================
# TASK 3: CATEGORY PREDICTION (Vectorized, Memory-Efficient)
# =================================================================

from sklearn.feature_extraction.text import TfidfVectorizer # <-- Import TF-IDF

print("Training Logistic Regression model for Category Prediction...")

# --- 1. Load Training Data ---
data_train = []
for d in readGz("train_Category.json.gz"):
    data_train.append(d)

# Get raw review text and labels
reviews_train = [d['review_text'] for d in data_train]
y_train = [d['genreID'] for d in data_train]

# --- 2. Build Features (using optimized TF-IDF Vectorizer) ---
# This replaces CountVectorizer.
# TfidfVectorizer weighs rare, important words more heavily.
# We'll use more features (5000) since TF-IDF is good at handling them.
vectorizer = TfidfVectorizer(max_features=5000)

print("Building TF-IDF feature matrix...")
# Fit on train data and transform it into a sparse matrix
# This is memory-efficient and builds the dictionary.
X_train_sparse = vectorizer.fit_transform(reviews_train)

# --- 3. Train the sklearn Model ---
print("Fitting model...")
# We'll use a stronger C=1.0. 
# The default 'fit_intercept=True' handles the offset (bias) term,
# so we don't need to add it manually.
mod = linear_model.LogisticRegression(C=1.0, max_iter=1000)
mod.fit(X_train_sparse, y_train)

print("Category model training complete.")

# --- 4. Write Category Predictions ---
print("Writing 'predictions_Category.csv' using TF-IDF model...")
predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")

# Process the test data
test_data = list(readGz("test_Category.json.gz"))
reviews_test = [l['review_text'] for l in test_data]

# Use .transform() (NOT .fit_transform()) to use the *same* dictionary
X_test_sparse = vectorizer.transform(reviews_test)

# Get all predictions at once (much faster)
all_predictions = mod.predict(X_test_sparse)

# Write predictions to file
for i, l in enumerate(test_data):
    pred = all_predictions[i]
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(pred) + "\n")

predictions.close()
print("Category predictions finished.")
print("All tasks complete.")