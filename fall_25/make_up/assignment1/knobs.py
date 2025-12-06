import gzip
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==========================================
# 1. SETUP REAL VADER
# ==========================================
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading data...")
def readJSON():
    for l in gzip.open("train.json.gz", 'rt'):
        yield eval(l)

dataset = [d for d in readJSON()]
df = pd.DataFrame(dataset)
df['rating'] = np.log1p(df['hours']) # Target

# ==========================================
# 3. GENERATE ALL FEATURES (Matching knobs.py)
# ==========================================
print("Generating Features...")

# --- A. Real VADER Sentiment ---
print("  - Calculating VADER...")
df['sentiment_score'] = df['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Game Sentiment Category
game_sent_stats = df.groupby('gameID')['sentiment_score'].agg(['mean', 'count'])
def get_sentiment_category(row):
    score = row['mean']
    count = row['count']
    if count < 5: return "Unknown"
    if score >= 0.5: return "Overwhelmingly Positive"
    elif score >= 0.1: return "Positive"
    elif score > -0.1: return "Mixed"
    elif score > -0.5: return "Negative"
    else: return "Overwhelmingly Negative"

game_sent_map = game_sent_stats.apply(get_sentiment_category, axis=1).to_dict()
# Map to numeric for correlation check (0=Neg, 1=Mix, 2=Pos, etc or just mean score)
df['game_sentiment_category'] = df['gameID'].map(game_sent_map)

# For correlation, we need numbers. Let's also keep the raw mean.
game_sent_mean = df.groupby('gameID')['sentiment_score'].mean().to_dict()
df['game_sentiment_mean'] = df['gameID'].map(game_sent_mean)

# --- B. TF-IDF Topics (1-2 grams) ---
print("  - Extracting Topics...")
game_texts = df.groupby('gameID')['text'].apply(lambda x: " ".join(str(s) for s in x))
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.8, min_df=2, max_features=1000)
tfidf_matrix = tfidf.fit_transform(game_texts)
# We can't easily correlate "Topics" (strings), so we'll check if knowing the topic reduces variance
# We assign the Topic String to the dataframe
feature_names = np.array(tfidf.get_feature_names_out())
game_topics = {}
for i, gid in enumerate(game_texts.index):
    row = tfidf_matrix[i].toarray().flatten()
    game_topics[gid] = feature_names[row.argmax()]
df['game_topic'] = df['gameID'].map(game_topics)

# --- C. Standard Features ---
print("  - Calculating Stats (Hv, Disagreement, Percentiles)...")
# Hv Score
def calculate_h_index(hours_list):
    sorted_hours = sorted(hours_list, reverse=True)
    h_idx = 0
    for i, h in enumerate(sorted_hours):
        if h >= i + 1: h_idx = i + 1
        else: break
    return h_idx
user_hv = df.groupby('userID')['hours'].apply(list).apply(calculate_h_index).to_dict()
df['user_hv_score'] = df['userID'].map(user_hv)

# Disagreement
game_var = df.groupby('gameID')['rating'].var().fillna(0).to_dict()
df['game_disagreement'] = df['gameID'].map(game_var)

# Percentiles
user_pct = df.groupby('userID')['hours'].sum().rank(pct=True).to_dict()
game_pct = df.groupby('gameID')['hours'].sum().rank(pct=True).to_dict()
df['user_hours_pct'] = df['userID'].map(user_pct)
df['game_hours_pct'] = df['gameID'].map(game_pct)

# Text Len
df['text_len'] = df['text'].str.len().fillna(0)

# --- D. KNN & Katz ---
print("  - Calculating KNN & Katz...")
users = list(df['userID'].unique())
items = list(df['gameID'].unique())
user_map = {u: i for i, u in enumerate(users)}
item_map = {i: u for u, i in enumerate(items)}
rows = [user_map[u] for u in df['userID']]
cols = [item_map[g] for g in df['gameID']]
M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), len(items)))

# Katz (simplified)
S = M.T @ M
def get_katz(row):
    u_idx = user_map[row['userID']]
    g_idx = item_map[row['gameID']]
    return S[g_idx, :].dot(M[u_idx, :].T)[0,0]
df['katz_score'] = df.apply(get_katz, axis=1)

# KNN
knn = NearestNeighbors(metric='cosine', n_neighbors=5).fit(M)
dists, indices = knn.kneighbors(M)
df['knn_sim'] = [1 - np.mean(d) for d in dists[rows]]

# ==========================================
# 4. QUALITY REPORT
# ==========================================
print("\n" + "="*60)
print("FEATURE QUALITY REPORT (Real NLTK VADER)")
print("="*60)

features_to_check = [
    'sentiment_score', 'game_sentiment_mean', # VADER FEATURES
    'game_topic',                             # TOPIC FEATURE
    'user_hv_score', 'game_disagreement', 
    'user_hours_pct', 'game_hours_pct',
    'katz_score', 'knn_sim', 'text_len'
]

# Train Model for Importance (Needs numerics, so we drop topic/category for this part)
numeric_feats = [f for f in features_to_check if f not in ['game_topic', 'game_sentiment_category']]
X = df[numeric_feats].fillna(0)
y = df['rating']
model = HistGradientBoostingRegressor(max_iter=50).fit(X, y)
result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
importances = dict(zip(numeric_feats, result.importances_mean))

global_std = y.std()

print(f"{'Feature Name':<25} | {'Var Reduct':<10} | {'Corr':<8} | {'Importance':<10}")
print("-" * 60)

for feat in features_to_check:
    # 1. Variance Reduction
    # For strings (Topic), just group by value. For numbers, bin them.
    if df[feat].dtype == 'object':
        df['bin'] = df[feat]
    else:
        try:
            df['bin'] = pd.qcut(df[feat], 10, duplicates='drop')
        except:
            df['bin'] = pd.cut(df[feat], 10)
            
    bin_stats = df.groupby('bin')['rating'].agg(['std', 'count'])
    weighted_std = np.average(bin_stats['std'].fillna(global_std), weights=bin_stats['count'])
    reduction = (1 - (weighted_std / global_std)) * 100
    
    # 2. Correlation & Importance (Skip for strings)
    if df[feat].dtype != 'object':
        corr = df[feat].corr(df['rating'])
        imp = importances.get(feat, 0.0)
        print(f"{feat:<25} | {reduction:>9.2f}% | {corr:>8.2f} | {imp:>10.4f}")
    else:
        print(f"{feat:<25} | {reduction:>9.2f}% | {'N/A':>8} | {'N/A':>10}")

print("-" * 60)
print("Look closely at 'game_sentiment_mean' and 'game_topic'.")
print("If Var Reduct is < 1%, VADER isn't helping.")
