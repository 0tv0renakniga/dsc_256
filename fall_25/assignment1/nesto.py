import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import lightgbm as lgb
from tqdm import tqdm # A progress bar, because this will take a moment

# Load data
train_df = pd.read_csv('train_Interactions.csv')
test_df = pd.read_csv('pairs_Read.csv')

print("Starting Feature Engineering...")

# --- User Features ---
user_features = train_df.groupby('userID')['rating'].agg(
    user_read_count='count',
    user_avg_rating='mean',
    user_std_rating='std'
).reset_index()

# --- Book Features ---
book_features = train_df.groupby('bookID')['rating'].agg(
    book_read_count='count',
    book_avg_rating='mean',
    book_std_rating='std'
).reset_index()

# --- Interaction Features (The "Secret Sauce") ---
# We will use Matrix Factorization (SVD) NOT as the final
# answer, but as a *feature*. We will generate 20 "embedding"
# features for every user and 20 for every book.

# 1. Create the User-Item sparse matrix
print("Building User-Item Matrix...")
# Map user/book IDs to a 0-indexed integer
train_df['user_idx'] = train_df['userID'].astype('category').cat.codes
train_df['book_idx'] = train_df['bookID'].astype('category').cat.codes

# Get mappings back in case we need them (for the test set)
user_map = dict(zip(train_df['userID'].unique(), train_df['user_idx'].unique()))
book_map = dict(zip(train_df['bookID'].unique(), train_df['book_idx'].unique()))

n_users = train_df['user_idx'].nunique()
n_books = train_df['book_idx'].nunique()

# Create the sparse matrix
user_item_matrix = csr_matrix((np.ones(len(train_df)),  # <--- THE FIX
                              (train_df['user_idx'], train_df['book_idx'])),
                              shape=(n_users, n_books))


# 2. Run TruncatedSVD to get embeddings
N_COMPONENTS = 20
print(f"Running SVD with {N_COMPONENTS} components...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
book_factors = svd.components_.T # Shape is (n_books, N_COMPONENTS)

# 3. Create DataFrames for these new features
user_embed_df = pd.DataFrame(user_factors, 
                             columns=[f'user_embed_{i}' for i in range(N_COMPONENTS)])
user_embed_df['user_idx'] = range(n_users)
# Map back to original IDs
user_embed_df['userID'] = user_embed_df['user_idx'].map({v:k for k,v in user_map.items()})


book_embed_df = pd.DataFrame(book_factors, 
                             columns=[f'book_embed_{i}' for i in range(N_COMPONENTS)])
book_embed_df['book_idx'] = range(n_books)
# Map back to original IDs
book_embed_df['bookID'] = book_embed_df['book_idx'].map({v:k for k,v in book_map.items()})

print("Feature Engineering Complete.")

print("Starting Negative Sampling to build training set...")

# 1. Get all unique book IDs
all_book_ids = set(book_features['bookID'])

# 2. Find books each user HAS read (for fast lookup)
user_read_books = train_df.groupby('userID')['bookID'].apply(set)

# 3. Create the negative samples
negative_samples = []
for userID, read_books in tqdm(user_read_books.items(), desc="Sampling negatives"):
    # Find books this user has NOT read
    not_read = all_book_ids - read_books

    # Sample N negative books, where N = number of books they *did* read
    # (This is called 1:1 sampling)
    n_samples = len(read_books)

    # Handle case where user has read almost all books
    if len(not_read) >= n_samples:
        samples = np.random.choice(list(not_read), n_samples, replace=False)
    else:
        # If user has read > half the books, just sample what's left
        samples = list(not_read)

    for bookID in samples:
        negative_samples.append({'userID': userID, 'bookID': bookID, 'read': 0})

# 4. Create our final training DataFrame
neg_df = pd.DataFrame(negative_samples)

# Get our positive samples
pos_df = train_df[['userID', 'bookID']].copy()
pos_df['read'] = 1

# Combine positive and negative
master_train_df = pd.concat([pos_df, neg_df], ignore_index=True)

# --- Merge all features into this new training set ---
print("Merging features into master training set...")
print("Adding 'explicit_rating' as a new feature...")
master_train_df = master_train_df.merge(
    train_df[['userID', 'bookID', 'rating']], 
    on=['userID', 'bookID'], 
    how='left'
)
# Fill the 'NaNs' (from our negative samples) with 0
master_train_df['rating'] = master_train_df['rating'].fillna(0)
# --- END OF NEW PART ---


# --- Merge all other features into this new training set ---
print("Merging all other features into master training set...")
def create_feature_set(df):
    # Base data
    data = df.copy()

    # Merge User/Book features
    data = data.merge(user_features, on='userID', how='left')
    data = data.merge(book_features, on='bookID', how='left')

    # Merge User/Book embeddings
    data = data.merge(user_embed_df, on='userID', how='left')
    data = data.merge(book_embed_df, on='bookID', how='left')

    # --- Create the FINAL magic feature: embedding dot product ---
    # This feature *alone* is often the most predictive
    user_embed_cols = [f'user_embed_{i}' for i in range(N_COMPONENTS)]
    book_embed_cols = [f'book_embed_{i}' for i in range(N_COMPONENTS)]

    # Get the numpy arrays for fast dot product
    user_vecs = data[user_embed_cols].values
    book_vecs = data[book_embed_cols].values

    # Calculate dot product (sum of element-wise multiplication)
    data['embed_dot_product'] = (user_vecs * book_vecs).sum(axis=1)

    # Fill NaNs that result from new users/books (e.g., in test set)
    # A simple '0' or mean fill is usually fine
    data = data.fillna(0) # Fill NaNs with 0

    return data

master_train_df = create_feature_set(master_train_df)

print("Master training set is ready.")

# --- Define Features (X) and Target (y) ---
# All columns EXCEPT the IDs and the target
TARGET = 'read'
# Get all columns, remove what we don't want
features = list(master_train_df.columns)
features.remove('userID')
features.remove('bookID')
features.remove('read')

# Handle columns that might not be in the SVD set
features = [f for f in features if 'user_idx' not in f and 'book_idx' not in f]

X_train = master_train_df[features]
y_train = master_train_df[TARGET]

# --- Train the LightGBM Classifier ---
print("Training LightGBM model...")

# These are good, fast, default parameters
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc', # Area Under Curve is a great metric for this
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42,
    reg_alpha=0.1, # L1 regularization
    reg_lambda=0.1 # L2 regularization
)

# We can use early stopping to prevent overfitting
# (This requires a validation set, but for now we'll skip
# it for simplicity and just train on the full set)
lgb_model.fit(X_train, y_train,
              categorical_feature=[f for f in features if 'embed' not in f])

print("Model training complete.")

# --- Apply Feature Engineering to the TEST set ---
print("Applying all feature engineering to the test set...")
# test_df is 'pairs_Read.csv'
X_test = create_feature_set(test_df)
X_test['rating'] = 0
X_test = X_test[features] # Make sure columns are in the same order

# --- Make Final Predictions ---
print("Making final predictions...")

# Use predict_proba to get the *probability* of '1'
test_probs = lgb_model.predict_proba(X_test)[:, 1]

# --- Set a threshold ---
# 0.5 is the default, but you can tune this!
#PREDICTION_THRESHOLD = 0.5
target_proportion = 0.5 

# Find the 50th percentile of the probabilities
PREDICTION_THRESHOLD = pd.Series(test_probs).quantile(1.0 - target_proportion)

print(f"Using dynamic threshold: {PREDICTION_THRESHOLD:.4f}")
test_df['prediction'] = (test_probs >= PREDICTION_THRESHOLD).astype(int)

# --- Save Results ---
print("Saving predictions...")
print(test_df['prediction'].value_counts())

test_df[['userID', 'bookID', 'prediction']].to_csv('predictions_Read.csv', index=False)

print("All done. 'predictions_Read.csv' is ready.")
