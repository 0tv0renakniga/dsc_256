import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import gzip
import random
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. METADATA & CONTENT EXTRACTOR
# =============================================================================
class FeatureStore:
    def __init__(self):
        self.user_stats = {}  # Map: userID -> [n_games, avg_hours]
        self.game_stats = {}  # Map: gameID -> [n_users, avg_hours]
        self.global_user_stats = [0, 0]
        self.global_game_stats = [0, 0]
        
        # Content
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,1))
        self.game_vectors = None
        self.user_profiles = {}
        self.game_map = {}

    def fit(self, df):
        print("   [Metadata] Computing user/game statistics...")
        # 1. Metadata (Hours & Counts)
        # Fill missing hours with 1.0 just in case
        df['hours'] = df.get('hours', pd.Series(np.ones(len(df)))).fillna(1.0)
        
        # User Stats
        u_group = df.groupby('userID')['hours']
        self.user_stats = u_group.agg(['count', 'mean']).to_dict('index')
        self.global_user_stats = [df.groupby('userID').size().mean(), df['hours'].mean()]
        
        # Game Stats
        g_group = df.groupby('gameID')['hours']
        self.game_stats = g_group.agg(['count', 'mean']).to_dict('index')
        self.global_game_stats = [df.groupby('gameID').size().mean(), df['hours'].mean()]

        # 2. Content (Text)
        print("   [Content] Processing review text...")
        df['text'] = df['text'].fillna('')
        game_docs = df.groupby('gameID')['text'].apply(lambda x: " ".join(x.astype(str)))
        
        self.game_ids = game_docs.index.tolist()
        self.game_map = {g: i for i, g in enumerate(self.game_ids)}
        
        # TF-IDF
        self.game_vectors = self.vectorizer.fit_transform(game_docs.values)
        
        # User Profiles (Average of their games)
        print("   [Content] Building user profiles...")
        user_games = df.groupby('userID')['gameID'].apply(list).to_dict()
        
        for u, games in tqdm(user_games.items(), desc="User Vectors"):
            valid = [self.game_map[g] for g in games if g in self.game_map]
            if valid:
                self.user_profiles[u] = np.asarray(self.game_vectors[valid].mean(axis=0))

    def get_features(self, u, g):
        # Metadata Features
        u_stat = self.user_stats.get(u, {'count': self.global_user_stats[0], 'mean': self.global_user_stats[1]})
        g_stat = self.game_stats.get(g, {'count': self.global_game_stats[0], 'mean': self.global_game_stats[1]})
        
        # Content Similarity
        content_sim = 0.0
        if u in self.user_profiles and g in self.game_map:
            u_vec = self.user_profiles[u]
            g_vec = self.game_vectors[self.game_map[g]].toarray()
            content_sim = float(np.dot(u_vec, g_vec.T))
            
        return [
            u_stat['count'],  # User Activity
            u_stat['mean'],   # User Intensity (Avg Hours)
            g_stat['count'],  # Game Popularity
            g_stat['mean'],   # Game Stickiness (Avg Hours)
            content_sim       # Text Match
        ]

# =============================================================================
# 2. BEHAVIORAL MODELS (SVD + SAR)
# =============================================================================
class BPRLikeSVD:
    def __init__(self, n_factors=32, n_epochs=10):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.user_map = {}; self.item_map = {}
        self.Pu = None; self.Qi = None
        self.bu = None; self.bi = None

    def _sigmoid(self, x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    def fit(self, df):
        print("   [SVD] Training Latent Factors...")
        users = df['userID'].unique(); items = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        
        self.Pu = np.random.normal(0, 0.1, (len(users), self.n_factors))
        self.Qi = np.random.normal(0, 0.1, (len(items), self.n_factors))
        self.bu = np.zeros(len(users)); self.bi = np.zeros(len(items))
        
        u_idxs = df['userID'].map(self.user_map).values
        i_idxs = df['gameID'].map(self.item_map).values
        labels = df['label'].values

        for _ in range(self.n_epochs):
            # Vectorized approximation or simple loop
            indices = np.random.permutation(len(labels))[:1000000] # Subsample for speed if needed
            for idx in indices:
                u, i, y = u_idxs[idx], i_idxs[idx], labels[idx]
                err = y - self._sigmoid(self.bu[u] + self.bi[i] + np.dot(self.Pu[u], self.Qi[i]))
                self.bu[u] += 0.02 * err
                self.bi[i] += 0.02 * err
                self.Pu[u] += 0.02 * (err * self.Qi[i] - 0.01 * self.Pu[u])
                self.Qi[i] += 0.02 * (err * self.Pu[u] - 0.01 * self.Qi[i])

    def predict(self, u, g):
        if u in self.user_map and g in self.item_map:
            u_i, g_i = self.user_map[u], self.item_map[g]
            return self._sigmoid(self.bu[u_i] + self.bi[g_i] + np.dot(self.Pu[u_i], self.Qi[g_i]))
        return 0.5

class SARModel:
    def __init__(self):
        self.item_users = defaultdict(set)
        self.user_items = defaultdict(set)
        
    def fit(self, df):
        print("   [SAR] Indexing co-occurrence...")
        # Only use positive interactions
        pos = df[df['label'] == 1]
        for row in pos.itertuples():
            self.item_users[row.gameID].add(row.userID)
            self.user_items[row.userID].add(row.gameID)
            
    def predict(self, u, g):
        if u not in self.user_items or g not in self.item_users: return 0.0
        hist = self.user_items[u]
        target_users = self.item_users[g]
        if not hist: return 0.0
        
        # Simplified Jaccard
        score = 0
        count = 0
        for h_item in list(hist)[:20]: # Limit history for speed
            if h_item == g: continue
            h_users = self.item_users[h_item]
            intersect = len(target_users.intersection(h_users))
            if intersect > 0:
                score += intersect / (len(target_users) + len(h_users) - intersect)
                count += 1
        return score / count if count > 0 else 0.0

# =============================================================================
# 3. HYBRID RANKER
# =============================================================================
class MetadataHybrid:
    def __init__(self):
        self.features = FeatureStore()
        self.svd = BPRLikeSVD(n_factors=30)
        self.sar = SARModel()
        self.ranker = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.05)

    def fit(self, df_raw):
        print("\n=== Training Metadata Hybrid ===")
        
        # 1. Prepare Data
        # Keep all data for feature extraction (even if negative generation uses subset)
        self.features.fit(df_raw)
        
        # Create Training Set (Pos + Neg)
        df_pos = df_raw[['userID', 'gameID']].copy()
        df_pos['label'] = 1
        
        # Generate Negatives (Ratio 2:1)
        print("   Generating Negatives...")
        users = df_pos['userID'].unique()
        items = list(df_pos['gameID'].unique())
        user_set = df_pos.groupby('userID')['gameID'].apply(set).to_dict()
        
        neg_data = []
        for u in tqdm(users, desc="Negatives"):
            seen = user_set[u]
            for _ in range(int(len(seen) * 2)):
                g = random.choice(items)
                while g in seen: g = random.choice(items)
                neg_data.append({'userID': u, 'gameID': g, 'label': 0})
        
        df_train = pd.concat([df_pos, pd.DataFrame(neg_data)], ignore_index=True)
        df_train = df_train.sample(frac=1.0, random_state=42)
        
        # 2. Train Behavioral Models
        self.svd.fit(df_train)
        self.sar.fit(df_train)
        
        # 3. Extract Features for Ranker
        print("   [Ranker] Building feature matrix...")
        X = self._build_feature_matrix(df_train)
        y = df_train['label'].values
        
        # 4. Train Classifier
        print("   [Ranker] Fitting Gradient Boosting...")
        self.ranker.fit(X, y)
        print(f"   Training AUC: {roc_auc_score(y, self.ranker.predict_proba(X)[:,1]):.4f}")

    def _build_feature_matrix(self, df):
        # Extract all 7 features for every row
        X = []
        # Convert to list of tuples for speed
        data = list(zip(df['userID'], df['gameID']))
        
        for u, g in tqdm(data, desc="Features"):
            # 1. SVD Score
            f1 = self.svd.predict(u, g)
            # 2. SAR Score
            f2 = self.sar.predict(u, g)
            # 3-7. Metadata (Activity, Intensity, Pop, Stickiness, Content)
            meta = self.features.get_features(u, g)
            
            X.append([f1, f2] + meta)
        return np.array(X)

    def predict(self, test_df, output_file):
        print(f"\n=== Predicting for {len(test_df)} pairs ===")
        X_test = self._build_feature_matrix(test_df)
        
        # Get Probabilities
        probs = self.ranker.predict_proba(X_test)[:, 1]
        
        # --- FORCE 50/50 SPLIT ---
        # Since test set is balanced, we set threshold at the median
        threshold = np.median(probs)
        binary_preds = (probs >= threshold).astype(int)
        
        sub = test_df.copy()
        sub['prediction'] = binary_preds
        sub.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file} (Median Threshold={threshold:.4f})")

# =============================================================================
# MAIN
# =============================================================================
def readJSON(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

if __name__ == "__main__":
    try:
        print("Loading data...")
        data = list(readJSON("train.json.gz"))
        df_train = pd.DataFrame(data)
        
        model = MetadataHybrid()
        model.fit(df_train)
        
        pairs = pd.read_csv("pairs_Played.csv")
        model.predict(pairs, "predictions_Played.csv")
        
    except Exception as e:
        print(f"Error: {e}")
