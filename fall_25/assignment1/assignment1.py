import gzip
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import trim_mean,kurtosis,zscore,skew,iqr
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# load train_Interactions.csv
train_df = pd.DataFrame()
df = pd.read_csv("train_Interactions.csv")

def predictRating_tuned(user, item, ratingMean, reviewsPerUser, usersPerItem,
                        userAverages, itemAverages, k_user=5, k_item=20):
    """
    Predicts a rating using a regularized baseline model.
    """
    mu = ratingMean
    
    # calc user bias with k_user
    num_user_reviews = len(reviewsPerUser.get(user, []))
    user_avg = userAverages.get(user, mu)
    b_u_raw = user_avg - mu
    b_u = b_u_raw * (num_user_reviews / (num_user_reviews + k_user))
    
    # calc item bias with k_item
    num_item_raters = len(usersPerItem.get(item, set()))
    item_avg = itemAverages.get(item, mu)
    b_i_raw = item_avg - mu
    b_i = b_i_raw * (num_item_raters / (num_item_raters + k_item))
    
    # calc final rating
    final_rating = mu + b_u + b_i
    
    return final_rating
df['norm_rating'] = df['rating']/5

ratingMean = df['norm_rating'].mean()

# Calculate user and item average ratings (on the normalized scale)
userAverages = df.groupby('userID')['norm_rating'].mean().to_dict()
itemAverages = df.groupby('bookID')['norm_rating'].mean().to_dict()


reviewsPerUser = df.groupby('userID')['bookID'].apply(list).to_dict()

# Your function uses len(usersPerItem.get(item, set())), so it needs a set.
usersPerItem = df.groupby('bookID')['userID'].apply(set).to_dict()

# This is required by the function signature, even if not used in the body.
itemsPerUser = df.groupby('userID')['bookID'].apply(set).to_dict()

print("Pre-computation complete.")
test_df = pd.read_csv('predictions_Rating.csv',index=False)
predictions = []
for _, row in test_df.iterrows():
    user = row['userID']
    item = row['bookID']
    
    pred_norm = predictRating_tuned(user, item, ratingMean, reviewsPerUser, usersPerItem,
                        userAverages, itemAverages, k_user=5, k_item=20)
    predictions.append(pred_norm)


# Add normalized predictions to the DataFrame
test_df['prediction_normalized'] = predictions
# Convert normalized prediction back to the 0-5 scale
test_df['prediction_0_to_5'] = test_df['prediction_normalized'] * 5.0

# Clip the final 0-5 rating to be within the valid range [0, 5]
test_df['prediction'] = test_df['prediction_0_to_5'].clip(lower=0, upper=5)
clean_test_df = test_df.loc[:, 'userID','bookID','prediction']
#=========================================================
# convert to memory efficent dtypes
train_df['userID'] = df['userID'].astype('category')
train_df['bookID'] = df['bookID'].astype('category')
train_df['rating'] = df['rating'].astype('float32')

# create features
train_df['user_tmean'] = df.groupby('userID')['rating'].transform(trim_mean,proportiontocut=0.25)
train_df['book_tmean'] = df.groupby('bookID')['rating'].transform(trim_mean,proportiontocut=0.25)
train_df['user_kur'] = df.groupby('userID')['rating'].transform(kurtosis)
train_df['book_kur'] = df.groupby('bookID')['rating'].transform(kurtosis)
train_df['user_skew'] = df.groupby('userID')['rating'].transform(skew,bias=False)
train_df['book_skew'] = df.groupby('bookID')['rating'].transform(skew,bias=False)
train_df['user_zscore'] = df.groupby('userID')['rating'].transform(zscore)
train_df['book_zscore'] = df.groupby('bookID')['rating'].transform(zscore)
train_df['book_high_rating'] = df.groupby('bookID')['rating'].transform(lambda x: (x > 3).sum())
train_df['book_mid_rating'] = df.groupby('bookID')['rating'].transform(lambda x: (x == 3).sum())
train_df['book_low_rating'] = df.groupby('bookID')['rating'].transform(lambda x: (x < 3).sum())
train_df['user_high_rating'] = df.groupby('userID')['rating'].transform(lambda x: (x > 3).sum())
train_df['user_mid_rating'] = df.groupby('userID')['rating'].transform(lambda x: (x == 3).sum())
train_df['user_low_rating'] = df.groupby('userID')['rating'].transform(lambda x: (x < 3).sum())

# define pca numeric features for users and books
numeric_features = ['user_tmean', 'book_tmean', 'user_kur','book_kur', 'user_skew', 
    'book_skew', 'user_zscore', 'book_zscore','book_high_rating', 'book_mid_rating', 
    'book_low_rating','user_high_rating', 'user_mid_rating', 'user_low_rating']

# define training set for features
X_train = train_df[numeric_features]

# fill all nan features with 0
X_train = X_train.fillna(0)

# standardize all numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# initalize PCA with 10 components and fit it to the scaled data
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# 5. Create a new dataframe with PCA components
pca_df = pd.DataFrame(
    X_pca, 
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
    index=train_df.index
)
# Create a DataFrame of the component "loadings"
loadings_df = pd.DataFrame(
    pca.components_,  # The "function" weights
    columns=numeric_features,  # Your original feature names
    index=[f'PC{i+1}' for i in range(pca.n_components_)]  # PC1, PC2, ...
)

# add PCA components to the original dataframe
train_df = pd.concat([train_df, pca_df], axis=1)
#pred_df = train_df.loc[:, 'userID','bookID','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
#pred_df = pred_df.reset_index(drop=True)
print("--- PCA Component Loadings (The Custom Function) ---")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

pred_df = pd.read_csv('predictions_Rating.csv',index=False)

# List of user-specific features
user_features = [
    'userID', 
    'user_tmean', 
    'user_kur', 
    'user_skew', 
    'user_zscore', 
    'user_high_rating', 
    'user_mid_rating', 
    'user_low_rating'
]

# List of book-specific features
book_features = [
    'bookID', 
    'book_tmean', 
    'book_kur', 
    'book_skew', 
    'book_zscore', 
    'book_high_rating', 
    'book_mid_rating', 
    'book_low_rating'
]

# Create the user lookup table
# (This assumes 'user_tmean' is the same for 'u1' in all rows)
user_stats = train_df[user_features].drop_duplicates(subset=['userID']).set_index('userID')

# Create the book lookup table
book_stats = train_df[book_features].drop_duplicates(subset=['bookID']).set_index('bookID')

# Calculate global averages for all user features
global_user_stats = user_stats.mean()

# Calculate global averages for all book features
global_book_stats = book_stats.mean()

# Combine them into one simple dictionary for filling
global_fill_values = {**global_user_stats, **global_book_stats}

pred_df_features = pred_df.reset_index(drop=True)

# --- Merge ---
# 1. Merge user stats (using 'userID' as the key)
pred_df_features = pred_df_features.merge(
    user_stats, 
    on='userID', 
    how='left'  # 'how=left' keeps every row in pred_df
)

# 2. Merge book stats (using 'bookID' as the key)
pred_df_features = pred_df_features.merge(
    book_stats, 
    on='bookID', 
    how='left'
)

# --- Fill ---
# 3. Fill all NaNs (from cold starts) with the global averages
pred_df_features = pred_df_features.fillna(global_fill_values)


numeric_features = ['user_tmean', 'book_tmean', 'user_kur','book_kur', 'user_skew', 
    'book_skew', 'user_zscore', 'book_zscore','book_high_rating', 'book_mid_rating', 
    'book_low_rating','user_high_rating', 'user_mid_rating', 'user_low_rating']

X_train = pred_df_features[numeric_features]
# fill all nan features with 0
X_train = X_train.fillna(0)

# standardize all numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# initalize PCA with 10 components and fit it to the scaled data
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# 5. Create a new dataframe with PCA components
pca_df = pd.DataFrame(
    X_pca, 
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
    index=pred_df_features.index
)
# Create a DataFrame of the component "loadings"
loadings_df = pd.DataFrame(
    pca.components_,  # The "function" weights
    columns=numeric_features,  # Your original feature names
    index=[f'PC{i+1}' for i in range(pca.n_components_)]  # PC1, PC2, ...
)

# add PCA components to the original dataframe
#train_df = pd.concat([train_df, pca_df], axis=1)
#pred_df = train_df.loc[:, 'userID','bookID','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
#pred_df = pred_df.reset_index(drop=True)
print("--- PCA Component Loadings (The Custom Function) ---")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")


"""
now you have the PCA components and their loadings. 
You can use these components to predict user ratings for new books. 
The loadings represent the importance of each feature in the PCA space, 
and the components themselves are the actual values of the features in the PCA space. 
You can use these components to predict user ratings for new books 
by applying the PCA transformation to the new book features and then using
blah blah blah... just build this

# This learns: rating = w1*PC1 + w2*PC2 + ... + wn*PCn + intercept
"""