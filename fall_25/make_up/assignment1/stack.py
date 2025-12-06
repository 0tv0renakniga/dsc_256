import gzip
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & HELPERS
# ==========================================
print("Initializing...")

def readJSON():
    try:
        for l in gzip.open("train.json.gz", 'rt'):
            yield eval(l)
    except FileNotFoundError:
        print("Error: train.json.gz not found.")
        return

def parse_year(date_str):
    try:
        # format: "Sep 21, 2015"
        return int(date_str.split(',')[-1].strip())
    except:
        return 2015 # default

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading Data...")
dataset = [d for d in readJSON()]
df = pd.DataFrame(dataset)
df['rating'] = np.log1p(df['hours'])
df['year'] = df['date'].apply(parse_year)

# Load Pairs
pairs = pd.read_csv('pairs_Hours.csv')

# ==========================================
# 3. FEATURE 1: SIMPLE SVD FEATURE
# ==========================================
print("Training Simple SVD...")

users = list(df['userID'].unique())
items = list(df['gameID'].unique())
user_map = {u: i for i, u in enumerate(users)}
item_map = {i: u for i, u in enumerate(items)}

# Create Matrix
rows = [user_map[u] for u in df['userID']]
cols = [item_map[g] for g in df['gameID']]
data = df['rating'].values

R = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

# Compute SVD (k=50)
# We use Scipy's sparse SVD
u, s, vt = svds(R, k=50)
s_diag = np.diag(s)

# Reconstruct Prediction Matrix (Approximation)
# We don't reconstruct the whole thing (Memory Explosion).
# We compute dot products on demand.
user_factors = u @ s_diag  # (n_users, k)
item_factors = vt.T        # (n_items, k)

def get_svd_pred(u_id, g_id, global_mean):
    if u_id not in user_map or g_id not in item_map:
        return global_mean
    u_idx = user_map[u_id]
    g_idx = item_map[g_id]
    
    pred = np.dot(user_factors[u_idx], item_factors[g_idx])
    return pred

# ==========================================
# 4. FEATURE 2: JACCARD SIMILARITY
# ==========================================
print("Calculating Jaccard...")
game_users = df.groupby('gameID')['userID'].apply(set).to_dict()
user_history = df.groupby('userID')['gameID'].apply(list).to_dict()

def get_jaccard_feature(u, g):
    history = user_history.get(u, [])
    target_users = game_users.get(g, set())
    
    if not history or not target_users: return 0.0
    
    # Average Jaccard with history
    scores = []
    for h_g in history:
        if h_g == g: continue
        h_users = game_users.get(h_g, set())
        
        intersect = len(target_users.intersection(h_users))
        union = len(target_users) + len(h_users) - intersect
        if union > 0:
            scores.append(intersect / union)
            
    return np.mean(scores) if scores else 0.0

# ==========================================
# 5. FEATURE 3: KATZ (Co-occurrence)
# ==========================================
print("Calculating Katz...")
# Binary Matrix for Structure
data_bin = np.ones(len(rows))
M = csr_matrix((data_bin, (rows, cols)), shape=(len(users), len(items)))
S = M.T @ M # Co-occurrence

def get_katz_feature(u_id, g_id):
    if u_id not in user_map or g_id not in item_map: return 0.0
    u_idx = user_map[u_id]
    g_idx = item_map[g_id]
    
    # Dot product of User's row in M and Game's row in S
    # Measures how many paths of length 3 connect U to G
    try:
        val = S[g_idx, :].dot(M[u_idx, :].T)[0, 0]
        return np.log1p(val) # Log scale for stability
    except:
        return 0.0

# ==========================================
# 6. STATISTICAL FEATURES
# ==========================================
print("Generating Stats Features...")

# Percentiles
user_hrs = df.groupby('userID')['hours'].sum()
user_pct_map = (user_hrs.rank(pct=True) * 100).to_dict()

game_hrs = df.groupby('gameID')['hours'].sum()
game_pct_map = (game_hrs.rank(pct=True) * 100).to_dict()

# Means & Vars
user_stats = df.groupby('userID')['rating'].agg(['mean', 'var']).to_dict('index')
game_stats = df.groupby('gameID')['rating'].agg(['mean', 'var', 'count']).to_dict('index')

# Game Count Per Year (Velocity)
game_years = df.groupby('gameID')['year'].apply(list)
def get_game_velocity(g_id):
    years = game_years.get(g_id, [])
    if not years: return 0
    span = max(years) - min(years) + 1
    return len(years) / span
game_velocity_map = {g: get_game_velocity(g) for g in game_years.index}

# Topic / Topic Modeling (TF-IDF N-grams)
print("Extracting Topics...")
game_texts = df.groupby('gameID')['text'].apply(lambda x: " ".join(str(s) for s in x))
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50) # Top 50 topics
tfidf_matrix = tfidf.fit_transform(game_texts)
# Reduce to 1D Topic Score (e.g., max tfidf val) as a proxy for "Topical Intensity"
# Or just use the game mean hours as a proxy for genre quality
# Let's map topic to a simplistic "Cluster Mean Rating"
# For Ridge, we need numbers. We'll use "Mean Rating of this Topic"
topic_clusters = np.argmax(tfidf_matrix.toarray(), axis=1)
game_topic_map = dict(zip(game_texts.index, topic_clusters))

# Calculate mean rating per topic
df['topic_id'] = df['gameID'].map(game_topic_map)
topic_means = df.groupby('topic_id')['rating'].mean().to_dict()

# ==========================================
# 7. BUILD FEATURE MATRIX
# ==========================================
print("Building Feature Matrix...")

global_mean = df['rating'].mean()
global_var = df['rating'].var()

def extract_features_df(target_df):
    features = []
    for idx, row in target_df.iterrows():
        u = row['userID']
        g = row['gameID']
        
        # 1. SVD
        f_svd = get_svd_pred(u, g, global_mean)
        
        # 2. Graph
        f_jac = get_jaccard_feature(u, g)
        f_katz = get_katz_feature(u, g)
        
        # 3. Stats
        f_g_pct = game_pct_map.get(g, 50.0)
        f_u_pct = user_pct_map.get(u, 50.0)
        
        g_s = game_stats.get(g, {'mean': global_mean, 'var': global_var, 'count': 0})
        u_s = user_stats.get(u, {'mean': global_mean, 'var': global_var})
        
        f_g_dis = g_s['var']
        f_g_mean = g_s['mean']
        f_u_mean = u_s['mean']
        f_g_cnt = g_s['count']
        
        # 4. Temporal / Topic
        f_g_vel = game_velocity_map.get(g, 0)
        
        t_id = game_topic_map.get(g, 0)
        f_topic_val = topic_means.get(t_id, global_mean)
        
        features.append([
            f_svd, f_jac, f_katz, 
            f_g_pct, f_u_pct, f_g_dis, 
            f_g_mean, f_u_mean, f_g_cnt, 
            f_g_vel, f_topic_val
        ])
    return np.array(features)

# Train Data
X_train = extract_features_df(df)
y_train = df['rating'].values

# ==========================================
# 8. TRAIN RIDGE REGRESSION
# ==========================================
print("Training Ridge Regression...")
# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Alpha=10.0 is a good starting point for regularizing this many correlated features
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)

# ==========================================
# 9. PREDICT & SUBMISSION
# ==========================================
print("Predicting for Submission...")
X_test = extract_features_df(pairs)
X_test_scaled = scaler.transform(X_test)

preds = ridge.predict(X_test_scaled)

# Calibrate to Target Stats (Safe Calibration)
# If variance is too low, boost it slightly, but clamp to safe limits
current_std = preds.std()
target_std = 2.30
boost_factor = target_std / (current_std + 1e-9)
# Don't let it explode like last time. Max boost 1.5x
boost_factor = min(boost_factor, 1.5) 

preds_cal = (preds - preds.mean()) * boost_factor + 3.72
pairs['prediction'] = np.clip(preds_cal, 0, 14)

pairs.to_csv('predictions_Hours.csv', index=False)
print("Done! Saved predictions_Hours.csv")
