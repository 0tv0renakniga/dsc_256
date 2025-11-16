import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode

# load train_Interactions.csv
df = pd.read_csv("train_Interactions.csv")

def predictRating_tuned(user, item, ratingMean, reviewsPerUser, usersPerItem,
                        userAverages, itemAverages, k_user=5, k_item=20):
    """
    Predicts a rating using a regularized baseline model.
    """
    mu = ratingMean
    
    # calc user bias with k_user
    num_user_reviews = len(reviewsPerUser.get(user, []))
    user_avg = userAverages.get(user, np.array(mu))
    b_u_raw = user_avg - mu
    b_u = b_u_raw * (num_user_reviews / (num_user_reviews + k_user))
    
    # calc item bias with k_item
    num_item_raters = len(usersPerItem.get(item, set()))
    item_avg = itemAverages.get(item, np.array(mu))
    b_i_raw = item_avg - mu
    b_i = b_i_raw * (num_item_raters / (num_item_raters + k_item))
    user_avg = np.atleast_1d(user_avg)
    item_avg = np.atleast_1d(item_avg)
    # calc final rating
    try:
        if 0.0 in mode(np.concatenate((item_avg, user_avg))):
            return 0
        elif 0.2 in mode(np.concatenate((item_avg, user_avg))) and 0.4 in mode(np.concatenate((item_avg, user_avg))):
            final_rating = 0.3 + np.sum(b_u) + np.sum(b_i)
            return final_rating 
        elif 1 in mode(np.concatenate((item_avg, user_avg))):
            return 0.9
        else:
            final_rating = mu + np.sum(b_u) + np.sum(b_i)
            return final_rating
    except Exception as e:
        print(f"user_avg:{user_avg}")
        print(f"item_avg: {item_avg}")
        avgs = [user_avg, item_avg]
        for i,j in enumerate(avgs):
            if type(j) != np.ndarray:
                avgs[i] = np.array(j)
        avg = np.mean(np.concatenate(avg[0],avg[1]))
        if avg == 1:
            return 1.0
        else:
            final_rating = item_avg + np.sum(b_u) + np.sum(b_i)
            return final_rating
        #print(f"Error: {e}")
        
        #print(f"user_avg:{user_avg}")
        
        #print(f'mu: {mu}')
        #print(f'b_u: {b_u}')
        #print(f"b_i: {b_i}")
        #print(np.dotb_u,b_i))
    # thresholds assume data is normalized 0 to 1
    """
    lower_threshold = 0.2
    upper_threshold = 0.99

    if final_rating > upper_threshold:
        return 5.0
    
    # If the model predicts < 2.5, it's probably trying to say 1
    # (or 0, depending on your scale)
    elif final_rating < lower_threshold:
        return 1.0  # or 0.0 if you prefer
    
    # 3. If it's in the middle, trust it.
    else:
        return final_rating    
    """

    
df['norm_rating'] = df['rating']/5

ratingMean = df['norm_rating'].mean()

# Calculate user and item average ratings (on the normalized scale)
userAverages = df.groupby('userID')['norm_rating'].agg(pd.Series.mode).to_dict()
itemAverages = df.groupby('bookID')['norm_rating'].agg(pd.Series.mode).to_dict()


reviewsPerUser = df.groupby('userID')['bookID'].apply(list).to_dict()

# Your function uses len(usersPerItem.get(item, set())), so it needs a set.
usersPerItem = df.groupby('bookID')['userID'].apply(set).to_dict()

# This is required by the function signature, even if not used in the body.
itemsPerUser = df.groupby('userID')['bookID'].apply(set).to_dict()

print("Pre-computation complete.")
#test_df = pd.read_csv('predictions_Rating.csv')
test_df = pd.read_csv('predictions_Rating.csv')
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
clean_test_df = test_df.loc[:, ['userID','bookID','prediction']]

# --- 2. Set up K-Fold CV ---
print(f"Successfully loaded {len(df)} interactions.")
print(f"shape of train_df: {df.shape}")
N_SPLITS = 5
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# We will store all results here
all_results = []

print(f"Starting {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    print(f"\n--- Processing Fold {fold+1}/{N_SPLITS} ---")
    
    # --- 3. Split data for this fold ---
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]

    # --- 4. "Train" your model (Calculate stats ONLY on train_data) ---
    print("Calculating statistics for this fold...")
    ratingMean = train_data['rating'].mean()
    
    # Convert to dicts for fast lookup
    reviewsPerUser = train_data.groupby('userID')['rating'].apply(list).to_dict()
    usersPerItem = train_data.groupby('bookID')['userID'].apply(set).to_dict()
    userAverages = train_data.groupby('userID')['rating'].mean().to_dict()
    itemAverages = train_data.groupby('bookID')['rating'].mean().to_dict()

    # --- 5. Make predictions on the validation set ---
    print("Making predictions on validation set...")
    for row in tqdm(val_data.itertuples(), total=len(val_data)):
        user = row.userID
        item = row.bookID
        actual_rating = row.rating
        
        # Predict rating
        pred_rating = predictRating_tuned(
            user, item, ratingMean,
            reviewsPerUser, usersPerItem,
            userAverages, itemAverages
        )
        
        # Store the result
        all_results.append({
            'userID': user,
            'bookID': item,
            'actual': actual_rating,
            'predicted': pred_rating,
            'fold': fold + 1
        })

print("\nCross-Validation Complete.")

# --- 6. Convert results to a DataFrame for analysis ---
results_df = pd.DataFrame(all_results)

# It's good practice to clip predictions to the valid 1-5 range
results_df['predicted_clipped'] = results_df['predicted'].clip(0, 5)
print(80*'=')
print('HEAD 20')
print(80*'=')
print(results_df.head(20))
for i in range(6):
    print(80*'=')
    print(f'DESCRIBE actual: {i}')
    print(80*'=')
    print(results_df[results_df['actual']==i].describe())

# --- 1. Calculate Error Metrics ---
# We'll use Squared Error (for RMSE) and Absolute Error (for MAE)
results_df['sq_error'] = (results_df['actual'] - results_df['predicted_clipped'])**2
results_df['abs_error'] = (results_df['actual'] - results_df['predicted_clipped']).abs()

print("\n--- Overall Model Performance ---")
overall_rmse = np.sqrt(results_df['sq_error'].mean())
overall_mae = results_df['abs_error'].mean()

print(f"Overall RMSE: {overall_rmse:.4f}")
print(f"Overall MAE : {overall_mae:.4f}")

# --- 2. THE CORE ANALYSIS: Error by Actual Rating ---
print("\n--- Error by Actual Rating (The 'Trouble' Report) ---")

# Group by the true rating and calculate the average error
error_by_rating = results_df.groupby('actual').agg(
    RMSE=pd.NamedAgg(column='sq_error', aggfunc=lambda x: np.sqrt(x.mean())),
    MAE=pd.NamedAgg(column='abs_error', aggfunc='mean'),
    Count=pd.NamedAgg(column='actual', aggfunc='count')
)

print(error_by_rating)

# --- 3. Visual Analysis (Highly Recommended) ---

print("\nGenerating visualization...")

# A boxplot is perfect for this
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='actual', y='predicted_clipped')
plt.title('Predicted Rating vs. Actual Rating')
plt.xlabel('Actual Rating (What the user gave)')
plt.ylabel('Predicted Rating (What your model guessed)')

# Add 'perfect' reference line
sns.lineplot(x=[0, 1, 2, 3, 4], y=[1, 2, 3, 4, 5], color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.savefig("rating.png")


