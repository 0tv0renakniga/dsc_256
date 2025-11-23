###########################################################
# RATING PREDICTION
###########################################################
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack, csr_matrix
from surprise import SVD, Dataset, Reader

# --- 1. Setup ---
df = pd.DataFrame(train_data)
df['hours_transformed'] = np.log2(df['hours'] + 1)

# --- 2. Ridge Feature (The "Bias" Expert) ---
print("Generating Ridge (Bias) Features...")
ohe = OneHotEncoder(handle_unknown='ignore')
sparse_ids = ohe.fit_transform(df[['userID', 'gameID']])

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
sparse_text = tfidf.fit_transform(df['text'].fillna(''))

X_sparse = hstack([sparse_ids, sparse_text])
y_target = df['hours_transformed'].values

# OOF Ridge
ridge_preds = np.zeros(len(df))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_sparse):
    # Lower alpha to 0.1 to reduce underfitting
    model = Ridge(alpha=0.1, solver='sag', random_state=42)
    model.fit(X_sparse[train_idx], y_target[train_idx])
    ridge_preds[val_idx] = model.predict(X_sparse[val_idx])

df['ridge_score'] = ridge_preds

# --- 3. SVD Feature (The "Interaction" Expert) ---
print("Generating SVD (Interaction) Features...")
# We use Surprise here because it handles explicit rating interactions naturally
svd_preds = np.zeros(len(df))
reader = Reader(rating_scale=(0, df['hours_transformed'].max()))

for train_idx, val_idx in kf.split(df):
    fold_train = df.iloc[train_idx]
    fold_val = df.iloc[val_idx]
    
    data_train = Dataset.load_from_df(fold_train[['userID', 'gameID', 'hours_transformed']], reader)
    trainset = data_train.build_full_trainset()
    
    # Standard SVD params
    algo = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    
    svd_preds[val_idx] = [algo.predict(row['userID'], row['gameID']).est for _, row in fold_val.iterrows()]

df['svd_score'] = svd_preds

# --- 4. The Final Stack (XGBoost) ---
print("Training Ensemble...")

features = ['ridge_score', 'svd_score'] # The two experts
X = df[features]
y = df['hours_transformed']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# We let XGBoost decide how much to trust Bias (Ridge) vs Interaction (SVD)
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

train_p = xgb_model.predict(X_train)
val_p = xgb_model.predict(X_val)

print(f"\nFinal Train MSE: {mean_squared_error(y_train, train_p):.4f}")
print(f"Final Val MSE:   {mean_squared_error(y_val, val_p):.4f}")

# --- 5. Generate Test Submission ---
print("Generating Test Predictions...")
pairs_hours = pd.read_csv('pairs_Hours.csv')

# A. Final Ridge Model (Full Data)
final_ridge = Ridge(alpha=0.1, solver='sag', random_state=42)
final_ridge.fit(X_sparse, y_target)

# B. Final SVD Model (Full Data)
full_data = Dataset.load_from_df(df[['userID', 'gameID', 'hours_transformed']], reader)
full_trainset = full_data.build_full_trainset()
final_svd = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
final_svd.fit(full_trainset)

# C. Predict Test Features
# Ridge Prep
test_ids = ohe.transform(pairs_hours[['userID', 'gameID']])
test_text = csr_matrix((len(pairs_hours), 3000)) # Empty text for test
X_test_sparse = hstack([test_ids, test_text])
pairs_hours['ridge_score'] = final_ridge.predict(X_test_sparse)

# SVD Prep
pairs_hours['svd_score'] = [final_svd.predict(u, g).est for u, g in zip(pairs_hours['userID'], pairs_hours['gameID'])]

# XGBoost Final Predict
final_preds = xgb_model.predict(pairs_hours[features])

# Clip
min_rating = 0
max_rating = df['hours_transformed'].max()
pairs_hours['prediction'] = [max(min_rating, min(max_rating, p)) for p in final_preds]

print(pairs_hours.head())
# pairs_hours.to_csv('predictions_Hours_Ensemble.csv', index=False, columns=['userID', 'gameID', 'prediction'])

###########################################################
# PLAY PREDICTION
###########################################################
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import csr_matrix
import random
import gzip
from collections import defaultdict

def readJSON(path):
  for l in gzip.open(path, 'rt'):
    d = eval(l)
    u = d['userID']
    try:
      g = d['gameID']
    except Exception as e:
      g = None
    yield u,g,d

# define data paths
train_json = 'train.json.gz'
pairs_hours = 'pairs_Hours.csv'
pairs_played = 'pairs_Played.csv'

# create containers for users and games
user_dict = defaultdict(list)
game_dict = defaultdict(list)
train_data = []

# read train.json.gz and populate user_dict and game_dict
for u,g,d in readJSON(train_json):
    user_dict[u].append(g)
    game_dict[g].append(u)
    train_data.append(d)
# --- 1. Load Data ---
df = pd.DataFrame(train_data)

# --- 2. Build the Implicit Interaction Matrix ---
print("Building Implicit Matrix...")
# Map IDs to contiguous integers for matrix operations
unique_users = df['userID'].unique()
unique_items = df['gameID'].unique()

u_map = {u: i for i, u in enumerate(unique_users)}
i_map = {j: i for i, j in enumerate(unique_items)}

row_ind = [u_map[u] for u in df['userID']]
col_ind = [i_map[i] for i in df['gameID']]
data_vals = np.ones(len(row_ind))

# Create Sparse Matrix (Rows=Users, Cols=Games)
# This is the "Implicit" matrix (1 = Played, 0 = Not Played)
interaction_matrix = csr_matrix((data_vals, (row_ind, col_ind)), 
                                shape=(len(unique_users), len(unique_items)))

# --- 3. Implicit SVD (Latent Features) ---
print("Extracting Latent Factors (SVD)...")
# We reduce the matrix to 32 dimensions. 
# This captures "Genre Affinity" and "User Taste" without needing text.
n_components = 32
svd = TruncatedSVD(n_components=n_components, random_state=42)

# Fit on the sparse matrix
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T

# Create fast lookup dictionaries
user_vec_dict = {u_id: user_factors[i] for u_id, i in u_map.items()}
item_vec_dict = {i_id: item_factors[i] for i_id, i in i_map.items()}

# Global Averages for Cold Start
avg_user_vec = np.mean(user_factors, axis=0)
avg_item_vec = np.mean(item_factors, axis=0)

# --- 4. Feature Engineering Function ---
item_popularity = df['gameID'].value_counts().to_dict()
user_activity = df['userID'].value_counts().to_dict()

def extract_features(dataframe):
    # 1. Latent Compatibility (Dot Product of SVD vectors)
    dots = []
    for u, i in zip(dataframe['userID'], dataframe['gameID']):
        u_vec = user_vec_dict.get(u, avg_user_vec)
        i_vec = item_vec_dict.get(i, avg_item_vec)
        dots.append(np.dot(u_vec, i_vec))
    
    dataframe['latent_score'] = dots
    
    # 2. Gravity Features (Pop/Act)
    dataframe['item_pop'] = dataframe['gameID'].map(item_popularity).fillna(0)
    dataframe['user_act'] = dataframe['userID'].map(user_activity).fillna(0)
    dataframe['gravity'] = np.log1p(dataframe['item_pop']) * np.log1p(dataframe['user_act'])
    
    return dataframe[['latent_score', 'item_pop', 'user_act', 'gravity']]

# --- 5. Construct Training Set (Positives + Negatives) ---
print("Constructing Negative Samples...")
pos_df = df[['userID', 'gameID']].copy()
pos_df['label'] = 1

# Negative Sampling
all_games = list(item_popularity.keys())
n_negatives = len(pos_df)

neg_users = df['userID'].sample(n_negatives, replace=True).values
neg_games = random.choices(all_games, k=n_negatives)
neg_df = pd.DataFrame({'userID': neg_users, 'gameID': neg_games})
neg_df['label'] = 0

train_class_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42)

# --- 6. Train XGBoost ---
print("Training Hybrid XGBoost...")
X = extract_features(train_class_df)
y = train_class_df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier(
    n_estimators=300,        # More trees to leverage the new latent features
    max_depth=6,             # Deeper trees to capture interaction between Latent & Gravity
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# --- 7. Validate ---
val_probs = clf.predict_proba(X_val)[:, 1]
roc = roc_auc_score(y_val, val_probs)
print(f"\nValidation ROC AUC: {roc:.4f}")

# Median Accuracy
threshold = np.median(val_probs)
val_preds = (val_probs > threshold).astype(int)
acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy (Median Thresh): {acc:.4f}")

# --- 8. Final Prediction ---
print("\nGenerating Predictions for pairs_Played.csv...")
pairs_played = pd.read_csv('pairs_Played.csv')

# Extract Hybrid Features
X_test = extract_features(pairs_played)

# Predict
test_probs = clf.predict_proba(X_test)[:, 1]

# Ranking Strategy
pairs_played['raw_score'] = test_probs
median_thresh = pairs_played['raw_score'].median()
pairs_played['prediction'] = (pairs_played['raw_score'] > median_thresh).astype(int)

print(pairs_played.head())
pairs_played.to_csv('predictions_Played.csv', index=False, columns=['userID', 'gameID', 'prediction'])