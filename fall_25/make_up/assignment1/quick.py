import gzip
import pandas as pd
import numpy as np
from surprise import SVDpp, Dataset, Reader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d

print("1. Loading Data...")
data_list = []
try:
    for d in readJSON("train.json.gz"):
        data_list.append(d)
except FileNotFoundError:
    print("Error: 'train.json.gz' not found.")
    exit()

df_train = pd.DataFrame(data_list)
df_test = pd.read_csv('predictions_Hours.csv')
df_train['text'] = df_train['text'].fillna("")

# ==========================================
# 2. FAST FEATURE EXTRACTION (The Turbo Part)
# ==========================================
print("\n2. Extracting Features (Turbo Mode)...")

# --- A. Fast Genre Matching (Keywords) ---
# Instead of a heavy AI model, we check if the game description contains the genre name.
genres = ["RPG", "Strategy", "Shooter", "Puzzle", "Adventure", "Simulation", "Sports", "Action"]

def get_genre_fast(text):
    text_lower = text.lower()
    for g in genres:
        if g.lower() in text_lower:
            return g
    return "Indie" # Default

# Group text by game to get a "Game Description"
game_texts = df_train.groupby('gameID')['text'].apply(lambda x: " ".join(x.head(10).tolist())[:5000])
# Map keywords
game_genres = game_texts.apply(get_genre_fast).to_dict()
df_train['game_genre'] = df_train['gameID'].map(game_genres).fillna("Indie")

# --- B. Fast Sentiment (Dictionary) ---
# Simple word counting is surprisingly effective for game reviews
pos_words = set(['fun', 'good', 'great', 'love', 'best', 'addictive', 'amazing', 'awesome', 'recommend', 'nice'])
neg_words = set(['boring', 'bad', 'worst', 'hate', 'refund', 'crash', 'broken', 'slow', 'clunky', 'tedious'])

def get_sentiment_fast(text):
    if not text: return 0.5
    tokens = text.lower().split()
    score = 0
    for t in tokens:
        if t in pos_words: score += 1
        if t in neg_words: score -= 1
    # Normalize to 0-1 range (Sigmoid-ish)
    return 0.5 + (np.tanh(score) / 2)

# Apply to all rows (This takes seconds, not hours)
tqdm.pandas(desc="Calculating Sentiment")
df_train['sentiment_score'] = df_train['text'].progress_apply(get_sentiment_fast)

# ==========================================
# 3. SAFE FEATURE ENGINEERING (Anti-Leakage)
# ==========================================
print("\n3. Generating Safe Statistical Features...")

# Calculate Global Stats
total_hours = df_train['hours_transformed'].sum()

# Leave-One-Out Calculations for Train
u_sums = df_train.groupby('userID')['hours_transformed'].transform('sum')
u_cnts = df_train.groupby('userID')['hours_transformed'].transform('count')
g_sums = df_train.groupby('gameID')['hours_transformed'].transform('sum')
g_cnts = df_train.groupby('gameID')['hours_transformed'].transform('count')

# Sentiment Stats
u_sum_sent = df_train.groupby('userID')['sentiment_score'].transform('sum')
u_cnt_sent = df_train.groupby('userID')['sentiment_score'].transform('count')

# --- Training Features ---
df_train['user_optimism'] = (u_sum_sent - df_train['sentiment_score']) / (u_cnt_sent - 1).replace(0, np.nan)
df_train['user_avg_hours'] = (u_sums - df_train['hours_transformed']) / (u_cnts - 1).replace(0, np.nan)
df_train['game_avg_hours'] = (g_sums - df_train['hours_transformed']) / (g_cnts - 1).replace(0, np.nan)

# Fill Cold Start
df_train['user_optimism'] = df_train['user_optimism'].fillna(0.5)
df_train['user_avg_hours'] = df_train['user_avg_hours'].fillna(df_train['hours_transformed'].mean())
df_train['game_avg_hours'] = df_train['game_avg_hours'].fillna(df_train['hours_transformed'].mean())

# Encode Genre
le = LabelEncoder()
df_train['genre_encoded'] = le.fit_transform(df_train['game_genre'])

# --- Test Features (Map Profiles) ---
user_profile_optimism = df_train.groupby('userID')['sentiment_score'].mean()
user_profile_hours = df_train.groupby('userID')['hours_transformed'].mean()
game_profile_hours = df_train.groupby('gameID')['hours_transformed'].mean()
game_profile_genre = df_train.groupby('gameID')['genre_encoded'].first()

df_test['user_optimism'] = df_test['userID'].map(user_profile_optimism).fillna(0.5)
df_test['user_avg_hours'] = df_test['userID'].map(user_profile_hours).fillna(df_train['hours_transformed'].mean())
df_test['game_avg_hours'] = df_test['gameID'].map(game_profile_hours).fillna(df_train['hours_transformed'].mean())

# Handle Genre Mapping (Careful with missing games)
indie_code = le.transform(['Indie'])[0] if 'Indie' in le.classes_ else 0
df_test['genre_encoded'] = df_test['gameID'].map(game_profile_genre).fillna(indie_code)

# ==========================================
# 4. MODEL TRAINING & BLENDING
# ==========================================
print("\n4. Training Models...")

# Model A: SVDpp (The Expert)
# ---------------------------
# If you ALREADY have predictions_SVDpp_Final.csv, you can load it here to save 15 mins.
# df_svd_preds_load = pd.read_csv('predictions_SVDpp_Final.csv')['prediction']
# svd_preds = df_svd_preds_load.values

print("   Training SVDpp (10-15 mins)...")
reader = Reader(rating_scale=(0, 14))
data_s = Dataset.load_from_df(df_train[['userID', 'gameID', 'hours_transformed']], reader)
svd = SVDpp(n_factors=1, n_epochs=30, lr_all=0.002, reg_all=0.08, random_state=42)
svd.fit(data_s.build_full_trainset())
svd_preds = [svd.predict(uid, iid).est for uid, iid in zip(df_test['userID'], df_test['gameID'])]

# Model B: GBR (The Assistant)
# ----------------------------
print("   Training Gradient Boosting...")
features = ['user_optimism', 'user_avg_hours', 'game_avg_hours', 'genre_encoded']
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gbr.fit(df_train[features], df_train['hours_transformed'])
gbr_preds = gbr.predict(df_test[features])

# ==========================================
# 5. FINAL BLEND
# ==========================================
print("\n5. Blending...")
# 70% SVD + 30% GBR
df_test['prediction'] = (0.7 * np.array(svd_preds)) + (0.3 * gbr_preds)

df_test[['userID', 'gameID', 'prediction']].to_csv('predictions_Turbo_Ensemble.csv', index=False)
print("Done! Saved to 'predictions_Turbo_Ensemble.csv'.")
