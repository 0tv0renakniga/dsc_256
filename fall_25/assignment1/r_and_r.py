import pandas as pd
import numpy as np
from tqdm import tqdm

def predictRating(user, item, ratingMean, reviewsPerUser, usersPerItem, 
                  itemsPerUser, userAverages, itemAverages):
    """
    Standard Regularized Baseline:
    Prediction = mu + b_u + b_i
    """
    # --- This is the logic from your provided function body ---
    k = 5 # Using the k=5 hardcoded from your example
    mu = ratingMean
    
    # calc user bias (b_u)
    num_user_reviews = len(reviewsPerUser.get(user, []))
    user_avg = userAverages.get(user, mu)
    b_u_raw = user_avg - mu
    b_u = b_u_raw * (num_user_reviews / (num_user_reviews + k))
    
    # calc item bias (b_i)
    num_item_raters = len(usersPerItem.get(item, set()))
    item_avg = itemAverages.get(item, mu)
    b_i_raw = item_avg - mu
    b_i = b_i_raw * (num_item_raters / (num_item_raters + k))
    
    # calc final rating
    final_rating = mu + b_u + b_i
    
    return final_rating
    # --- End of your function logic ---

# 1. Read Training Data
print("Loading train_Interactions.csv...")
df = pd.read_csv("train_Interactions.csv")

# 2. Make Features from Training Set
print("Calculating statistics (this may take a moment)...")

# Global average rating
ratingMean = df['rating'].mean()

# Average rating for each user and item
userAverages = df.groupby('userID')['rating'].mean().to_dict()
itemAverages = df.groupby('bookID')['rating'].mean().to_dict()

# Interaction counts/lists for regularization
reviewsPerUser = df.groupby('userID')['bookID'].apply(list).to_dict()
usersPerItem = df.groupby('bookID')['userID'].apply(set).to_dict()

# This is in your signature, so we compute it
itemsPerUser = df.groupby('userID')['bookID'].apply(set).to_dict()

print("Statistics calculation complete.")

# 3. Read predictions_Rating.csv
print("Loading predictions_Rating.csv...")
test_df = pd.read_csv('predictions_Rating.csv')

# 4. Make Predictions
print("Generating predictions...")
predictions = []

# Use tqdm for a progress bar
for row in tqdm(test_df.itertuples(), total=len(test_df)):
    user = row.userID
    item = row.bookID
    
    # Call your function with all the pre-computed stats
    pred = predictRating(
        user, item, 
        ratingMean, 
        reviewsPerUser, usersPerItem, 
        itemsPerUser, 
        userAverages, itemAverages
    )
    predictions.append(pred)

# 5. Write Predictions
# Add predictions to the dataframe
test_df['prediction'] = predictions

# IMPORTANT: Clip predictions to the valid 0-5 range
test_df['prediction'] = test_df['prediction'].clip(0, 5)

# Save the file in the required format
# This will OVERWRITE your existing file
output_cols = ['userID', 'bookID', 'prediction']
test_df[output_cols].to_csv('predictions_Rating.csv', index=False)

print("\nDone. 'predictions_Rating.csv' has been updated with new predictions.")

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
df = pd.read_csv('train_Interactions.csv')

# Create binary label: 1 if rating > 0 (read), 0 otherwise (not read)
df['read'] = (df['rating'] > 0).astype(int)

# Compute user-level features
user_stats = df.groupby('userID').agg(
    user_read_rate=('read', 'mean'),
    user_num_books=('bookID', 'count')
).reset_index()

# Compute book-level features
book_stats = df.groupby('bookID').agg(
    book_read_rate=('read', 'mean'),
    book_num_users=('userID', 'count')
).reset_index()

# Merge features back into the dataframe
df = df.merge(user_stats, on='userID')
df = df.merge(book_stats, on='bookID')

# Split into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Define features
features = ['user_read_rate', 'user_num_books', 'book_read_rate', 'book_num_users']

# Prepare training data
X_train = train[features]
y_train = train['read']
X_train = sm.add_constant(X_train)  # Add intercept term

# Train logistic regression model
model = sm.Logit(y_train, X_train)
result = model.fit()

# Print model summary for interpretation
print(result.summary())

# Evaluate on test set
X_test = test[features]
y_test = test['read']
X_test = sm.add_constant(X_test)
preds = result.predict(X_test)
acc = ((preds > 0.5).astype(int) == y_test).mean()
print('Test Accuracy:', acc)

# Baseline: Always predict 1 (read)
baseline_acc = (y_test == 1).mean()
print('Baseline Accuracy (always predict 1):', baseline_acc)

# Function to predict for a given userID and bookID
def predict_read(user_id, book_id, result, df, user_stats, book_stats):
    """
    Predicts whether the user has read the book (1) or not (0).
    
    Parameters:
    - user_id: str, the userID
    - book_id: str, the bookID
    - result: fitted statsmodels Logit result object
    - df: pandas DataFrame, the original dataset
    - user_stats: pandas DataFrame, precomputed user statistics
    - book_stats: pandas DataFrame, precomputed book statistics
    
    Returns:
    - int: 1 (read) or 0 (not read)
    """
    # Retrieve user features; default to global means if unseen
    user_data = user_stats[user_stats['userID'] == user_id]
    if user_data.empty:
        user_read_rate = df['read'].mean()
        user_num_books = 0
    else:
        user_read_rate = user_data['user_read_rate'].values[0]
        user_num_books = user_data['user_num_books'].values[0]
    
    # Retrieve book features; default to global means if unseen
    book_data = book_stats[book_stats['bookID'] == book_id]
    if book_data.empty:
        book_read_rate = df['read'].mean()
        book_num_users = 0
    else:
        book_read_rate = book_data['book_read_rate'].values[0]
        book_num_users = book_data['book_num_users'].values[0]
    
    # Prepare input array (with constant/intercept)
    x = np.array([[1.0, user_read_rate, user_num_books, book_read_rate, book_num_users]])
    
    # Predict probability and threshold
    prob = result.predict(x)[0]
    return int(prob > 0.5)

# Example usage (replace with actual IDs)
# prediction = predict_read('u67805239', 'b61372131', result, df, user_stats, book_stats)
# print('Prediction:', prediction)

# 3. Read predictions_Rating.csv
print("Loading predictions_Rating.csv...")
test_df = pd.read_csv('predictions_Read.csv')

# 4. Make Predictions
print("Generating predictions...")
predictions = []

# Use tqdm for a progress bar
for row in tqdm(test_df.itertuples(), total=len(test_df)):
    user = row.userID
    item = row.bookID
    
    # Call your function with all the pre-computed stats
    pred = predict_read(user, item, result, df, user_stats, book_stats)
    predictRating(
        user, item, 
        ratingMean, 
        reviewsPerUser, usersPerItem, 
        itemsPerUser, 
        userAverages, itemAverages
    )
    predictions.append(pred)

# 5. Write Predictions
# Add predictions to the dataframe
test_df['prediction'] = predictions