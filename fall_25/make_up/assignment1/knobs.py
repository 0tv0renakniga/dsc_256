import gzip
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. SETUP VADER
# Ensure you have the lexicon downloaded
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

# ==========================================
# 3. FEATURE 1: VADER SENTIMENT CATEGORY
# ==========================================
print("Calculating VADER Sentiment...")

# A. Get compound score for every review
def get_vader_score(text):
    if not isinstance(text, str): return 0.0
    return sid.polarity_scores(text)['compound']

df['sentiment_score'] = df['text'].apply(get_vader_score)

# B. Aggregate by Game to get the "Game's General Vibe"
game_stats = df.groupby('gameID')['sentiment_score'].agg(['mean', 'count'])

# C. Categorize (Steam-style logic)
def get_sentiment_category(row):
    score = row['mean']
    count = row['count']
    
    if count < 5: return "Unknown"
    
    if score >= 0.5: return "Overwhelmingly Positive"
    elif score >= 0.1: return "Positive"
    elif score > -0.1: return "Mixed"
    elif score > -0.5: return "Negative"
    else: return "Overwhelmingly Negative"

game_stats['category'] = game_stats.apply(get_sentiment_category, axis=1)

# ==========================================
# 4. FEATURE 2: UNIQUE TOPIC (TF-IDF N-Grams)
# ==========================================
print("Extracting Unique Game Topics...")

# A. Combine all text for each game into one massive string
game_texts = df.groupby('gameID')['text'].apply(lambda x: " ".join(str(s) for s in x))

# B. TF-IDF to find words unique to this game vs others
# ngram_range=(1,2) captures "fun" and "open world"
# max_df=0.8 removes words common to >80% of games (like "game", "play")
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.8, min_df=2)

tfidf_matrix = tfidf.fit_transform(game_texts)
feature_names = np.array(tfidf.get_feature_names_out())

# C. Extract Top Term for each game
game_topics = {}
game_ids = game_texts.index.tolist()

for i, game_id in enumerate(game_ids):
    # Get the row for this game
    row = tfidf_matrix[i].toarray().flatten()
    # Find the index of the highest TF-IDF score
    top_idx = row.argmax()
    # Get the actual word/phrase
    game_topics[game_id] = feature_names[top_idx]

# ==========================================
# 5. MERGE BACK TO DATAFRAME
# ==========================================
print("Merging Features...")

# Map the Game-Level features back to the main User-Level interactions
df['game_sentiment_category'] = df['gameID'].map(game_stats['category'])
df['game_topic'] = df['gameID'].map(game_topics)
# Global Uncertainty (Baseline to beat)
global_std = df['hours_transformed'].std()
print(f"Global Uncertainty (Std Dev): {global_std:.4f}")
print("-" * 60)

# ==========================================
# 2. GENERATE CANDIDATE FEATURES
# ==========================================
print("Generating Candidates...")
total_years = (pd.to_datetime(df['date'],errors='coerce').max().year - pd.to_datetime(df['date'],errors='coerce').min().year)
# A. Basic Counts (Popularity/Activity)
df['game_count'] = df.groupby('gameID')['gameID'].transform('count')
df['game_hours'] = df.groupby('gameID')['hours'].transform('sum')

df['user_count'] = df.groupby('userID')['userID'].transform('count')
df['user_hours'] = df.groupby('userID')['hours'].transform('sum')

df['game_count_per_year'] = df.groupby('gameID')['gameID'].transform('count')/total_years
df['user_count_per_year'] = df.groupby('userID')['userID'].transform('count')/total_years

# B. Mean Encodings (The "Prior")
# Note: In a real model, we'd do this with cross-validation to avoid leakage.
# For analysis, doing it globally is fine to see the potential signal.
df['game_mean_hours'] = df.groupby('gameID')['hours_transformed'].transform('mean')
df['user_mean_hours'] = df.groupby('userID')['hours_transformed'].transform('mean')

# C. Interaction Features (The "Category" Candidates)
# e.g. "Is this a Whale playing a Popular game?"
df['high_activity_user'] = df['user_count'] > df['user_count'].median()
df['popular_game'] = df['game_count'] > df['game_count'].median()

# D. Jaccard-ish Similarity (Simplified)
# "How much does this user play games like this one?"
# (Requires more complex logic, let's stick to the stats first)
df['rating'] = np.log1p(df['hours'])

# ==========================================
# 3. FEATURE: Hv-SCORE (User H-Index)
# ==========================================
print("Calculating Hv-scores...")
def calculate_h_index(hours_list):
    # Sort descending: [100, 20, 5, 2, 1]
    sorted_hours = sorted(hours_list, reverse=True)
    h_idx = 0
    for i, h in enumerate(sorted_hours):
        # If the (i+1)th game has >= (i+1) hours, it counts
        if h >= i + 1:
            h_idx = i + 1
        else:
            break
    return h_idx

user_hours = df.groupby('userID')['hours'].apply(list)
user_hv = user_hours.apply(calculate_h_index).to_dict()
df['user_hv_score'] = df['userID'].map(user_hv)

# ==========================================
# 4. FEATURE: DEGREE OF DISAGREEMENT
# ==========================================
print("Calculating Degree of Disagreement...")
# Variance of log-hours
game_variance = df.groupby('gameID')['rating'].var().fillna(0).to_dict()
df['degree_of_disagreement'] = df['gameID'].map(game_variance)

# ==========================================
# 5. FEATURE: SENTIMENT CATEGORY
# ==========================================
# ==========================================
# 3. FEATURE: KATZ MEASURE (Structural)
# ==========================================
print("Calculating Katz-3 Measure...")

# Map IDs to Integers
users = list(df['userID'].unique())
items = list(df['gameID'].unique())
user_map = {u: i for i, u in enumerate(users)}
item_map = {i: u for u, i in enumerate(items)}

rows = [user_map[u] for u in df['userID']]
cols = [item_map[g] for g in df['gameID']]
# Binary Interaction for Structure (1 if played)
data_bin = np.ones(len(rows))

# Sparse Matrix M (Users x Items)
M = csr_matrix((data_bin, (rows, cols)), shape=(len(users), len(items)))

# Compute Item-Item Co-occurrence Matrix (S = M^T * M)
# S[i, j] = Number of users who played both game i and game j
print("  - Computing Co-occurrence Matrix...")
S = M.T @ M

# To compute Katz feature for (User U, Game G):
# Path Count = Sum over all games G' played by U of (SharedUsers(G', G))
# This equals: (UserVector_U * S)[:, G]
# This operation is heavy to do for all pairs at once, so we do it iteratively or batched.
# Ideally: Feature_Matrix = M @ S. 
# M (U x I) @ S (I x I) -> Result (U x I). Result[u, g] is the Katz score.
# Since U*I is too big (e.g. 200k * 10k = 2 billion), we only compute it for the rows we need.

print("  - Extracting Katz Features...")

def get_katz_scores(pairs_df, M_matrix, S_matrix, u_map, i_map):
    scores = []
    # Convert pairs to indices
    u_indices = [u_map.get(u, -1) for u in pairs_df['userID']]
    g_indices = [i_map.get(g, -1) for g in pairs_df['gameID']]
    
    # We can compute efficiently by row slicing
    # For a user u, their vector is M[u, :]. 
    # We want dot product of M[u, :] and S[:, g]
    # S is symmetric, so S[:, g] is same as S[g, :].T
    
    # Optimization: Iterate and compute dot product
    # Since M is CSR, getting rows is fast. S is CSR, getting cols is slow. 
    # Convert S to CSC or just use S[g, :] since it's symmetric.
    
    for u_idx, g_idx in zip(u_indices, g_indices):
        if u_idx == -1 or g_idx == -1:
            scores.append(0)
            continue
        
        # Get games user played (indices)
        user_games = M_matrix.indices[M_matrix.indptr[u_idx]:M_matrix.indptr[u_idx+1]]
        
        # Get co-occurrences for target game g_idx
        # S is Item x Item. We want row g_idx (which is all games g shares users with)
        # S[g_idx, :] gives a sparse row of shared counts.
        # We want the sum of these counts for the games the user played.
        
        # Intersection of (User's Games) and (Games Connected to Target)
        # We can just index S. 
        # But slicing S for specific columns (user_games) is fast if S is CSR and we slice rows?
        # Wait, S[g_idx, user_games] gives the values.
        
        # S is symmetric. Row g_idx contains connections from g to all other games.
        # We sum the values in that row corresponding to user_games.
        
        # Extract the row for the target game
        row_start = S_matrix.indptr[g_idx]
        row_end = S_matrix.indptr[g_idx+1]
        
        # These are the game IDs connected to target G
        connected_games = S_matrix.indices[row_start:row_end]
        connected_counts = S_matrix.data[row_start:row_end]
        
        # We need sum of counts where connected_game is in user_games.
        # This is essentially dot product of two sparse vectors.
        # Ideally: S[g_idx, :].dot(M[u_idx, :].T)
        
        # Faster manual intersection for sorted arrays?
        # Let's trust scipy's dot product.
        
        val = S_matrix[g_idx, :].dot(M_matrix[u_idx, :].T)[0,0]
        scores.append(val)
        
    return scores

# Calculate for Train DF
df['katz_score'] = get_katz_scores(df, M, S, user_map, item_map)

# --- A. PERCENTILES (New Feature) ---
print("Calculating Percentiles...")
# User Percentile (Based on Total Hours Played)
user_sum_hours = df.groupby('userID')['hours'].sum()
user_pct = user_sum_hours.rank(pct=True) * 100
# Clip to 0.1 - 99.9 range
user_pct = user_pct.clip(0.1, 99.9).to_dict()
df['user_hours_pct'] = df['userID'].map(user_pct)

# Game Percentile (Based on Total Hours Played by all users)
game_sum_hours = df.groupby('gameID')['hours'].sum()
game_pct = game_sum_hours.rank(pct=True) * 100
game_pct = game_pct.clip(0.1, 99.9).to_dict()
df['game_hours_pct'] = df['gameID'].map(game_pct)
# ==========================================
# 6. FEATURE: KNN SIMILARITY (Top 5 Users)
# ==========================================
print("Calculating KNN Features (Top 5 Similarity)...")

# Map IDs to Integers
users = list(df['userID'].unique())
items = list(df['gameID'].unique())
user_map = {u: i for i, u in enumerate(users)}
item_map = {i: u for u, i in enumerate(items)}

rows = [user_map[u] for u in df['userID']]
cols = [item_map[g] for g in df['gameID']]
data = df['rating'].values

# Sparse Matrix
matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

# Fit KNN (Cosine Similarity)
# n_neighbors=6 because the 1st neighbor is always self
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
knn.fit(matrix)

# Find Neighbors
distances, indices = knn.kneighbors(matrix)

# Process Results
knn_mean_sim = []
knn_mean_rating = []
rating_lookup = df.set_index(['userID', 'gameID'])['rating'].to_dict()

# We need to map integer indices back to UserIDs
int_to_user = {i: u for u, i in user_map.items()}

for idx_row, row in df.iterrows():
    u_int = user_map[row['userID']]
    target_g = row['gameID']
    
    # Get neighbors for this user
    # indices[u_int] gives [self, n1, n2, n3, n4, n5]
    nbr_indices = indices[u_int][1:] 
    nbr_dists = distances[u_int][1:]
    
    # Calculate Average Similarity (1 - cosine_dist)
    sims = 1 - nbr_dists
    avg_sim = np.mean(sims) if len(sims) > 0 else 0
    knn_mean_sim.append(avg_sim)
    
    # Calculate Average Rating of Neighbors for THIS game
    ratings = []
    weights = []
    
    for n_idx, sim in zip(nbr_indices, sims):
        n_u = int_to_user[n_idx]
        if (n_u, target_g) in rating_lookup:
            ratings.append(rating_lookup[(n_u, target_g)])
            weights.append(sim)
            
    if ratings:
        weighted_avg = np.average(ratings, weights=weights)
        knn_mean_rating.append(weighted_avg)
    else:
        # Fallback: Game Mean
        knn_mean_rating.append(np.nan)

df['top5_similarity'] = knn_mean_sim
df['top5_neighbor_rating'] = knn_mean_rating
# Fill NaN ratings with game mean
df['top5_neighbor_rating'] = df['top5_neighbor_rating'].fillna(df.groupby('gameID')['rating'].transform('mean'))
df['text_len'] = df['text'].str.len()
# ==========================================
# 3. THE VARIANCE SCANNER
# ==========================================
features_to_scan = [
    'game_count', 'user_count', 'game_count_per_year', 'user_count_per_year', 'sentiment_score',

    'game_mean_hours', 'user_mean_hours','user_hours', 'game_hours', 'game_topic',
    'user_hv_score', 'degree_of_disagreement', 'top5_similarity','text_len'
]

results = []

for feat in features_to_scan:
    # 1. Bin the feature (Split into 10 categories)
    try:
        # qcut handles quantiles (e.g., Top 10%, Bottom 10%)
        df[f'{feat}_bin'] = pd.qcut(df[feat], q=10, duplicates='drop')
    except:
        # cut handles linear spacing
        df[f'{feat}_bin'] = pd.cut(df[feat], bins=10)
        
    # 2. Calculate Uncertainty (Std Dev) inside each bin
    # We want the Weighted Average Std Dev to be LOWER than Global Std
    bin_stats = df.groupby(f'{feat}_bin')['log_hours'].agg(['std', 'count', 'mean'])
    
    # Weighted Average Variance (The "Uncertainty Score")
    # Lower is Better
    total_count = bin_stats['count'].sum()
    weighted_std = np.average(bin_stats['std'].fillna(global_std), weights=bin_stats['count'])
    
    # Reduction %
    reduction = (1 - (weighted_std / global_std)) * 100
    
    results.append({
        'Feature': feat,
        'Weighted_Std': weighted_std,
        'Uncertainty_Reduction': reduction
    })
    
    # Print the "Best" Bin for this feature (Where we are most confident)
    best_bin = bin_stats.loc[bin_stats['std'].idxmin()]
    print(f"\nFeature: {feat}")
    print(f"  -> Reduces Global Uncertainty by: {reduction:.2f}%")
    print(f"  -> Most Confident Zone: {best_bin.name}")
    print(f"     (Std: {best_bin['std']:.4f}, Mean Hours: {best_bin['mean']:.2f})")

# ==========================================
# 4. SUMMARY
# ==========================================
res_df = pd.DataFrame(results).sort_values('Uncertainty_Reduction', ascending=False)
print("\n" + "="*60)
print("FINAL RANKING: Which features capture the PDF?")
print("="*60)
print(res_df)
print(df.columns)
