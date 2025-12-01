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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONTENT ENGINE (TF-IDF on Reviews)
# =============================================================================
class ContentEngine:
    """
    Treats every game as a 'document' (collection of all its reviews).
    Treats every user as a 'query' (average of the games they played).
    """
    def __init__(self, max_features=1000):
        # Increased features slightly for better resolution
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.game_vectors = None
        self.user_profiles = {}
        self.game_map = {}
        self.game_ids = []

    def fit(self, df):
        print("   [Content] Vectorizing review text...")
        
        # 1. Aggregate reviews by Game
        # We handle NaN text by filling with empty string
        df['text'] = df['text'].fillna('')
        
        # Groupby gameID and join all reviews into one massive string per game
        game_docs = df.groupby('gameID')['text'].apply(lambda x: " ".join(x.astype(str)))
        
        self.game_ids = game_docs.index.tolist()
        self.game_map = {g: i for i, g in enumerate(self.game_ids)}
        
        # 2. TF-IDF Matrix (Rows=Games, Cols=Words)
        self.game_vectors = self.vectorizer.fit_transform(game_docs.values)
        
        # 3. Build User Profiles
        print("   [Content] Building user profiles...")
        user_games = df.groupby('userID')['gameID'].apply(list).to_dict()
        
        # Pre-calculate for speed
        game_vecs_dense = self.game_vectors
        
        for u, games in tqdm(user_games.items(), desc="User Profiles"):
            valid_indices = [self.game_map[g] for g in games if g in self.game_map]
            if valid_indices:
                # User vector is the Average of the games they played
                # We use sparse indexing for speed
                user_vec = np.asarray(game_vecs_dense[valid_indices].mean(axis=0))
                self.user_profiles[u] = user_vec

    def predict_sim(self, u_id, g_id):
        # Calculate Cosine Similarity
        if u_id in self.user_profiles and g_id in self.game_map:
            u_vec = self.user_profiles[u_id] # Shape (1, 1000)
            g_idx = self.game_map[g_id]
            g_vec = self.game_vectors[g_idx].toarray() # Shape (1, 1000)
            
            # Dot product of normalized vectors is cosine similarity
            # (TF-IDF output is already normalized by default)
            return float(np.dot(u_vec, g_vec.T))
        return 0.0

# =============================================================================
# 2. BPR-LIKE SVD (Latent Factors)
# =============================================================================
class BPRLikeSVD:
    def __init__(self, n_factors=30, n_epochs=15, lr=0.02, reg=0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_map = {}
        self.item_map = {}
        self.Pu = None; self.Qi = None
        self.bu = None; self.bi = None

    def _sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def fit(self, df_pos, df_neg):
        print(f"   [SVD] Training Latent Factors ({self.n_factors})...")
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        
        users = df_all['userID'].unique()
        items = df_all['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        
        n_u, n_i = len(users), len(items)
        self.Pu = np.random.normal(0, 0.1, (n_u, self.n_factors))
        self.Qi = np.random.normal(0, 0.1, (n_i, self.n_factors))
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)

        u_idxs = df_all['userID'].map(self.user_map).values
        i_idxs = df_all['gameID'].map(self.item_map).values
        labels = df_all['label'].values.astype(float)

        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(labels))
            for idx in indices:
                u, i, y = u_idxs[idx], i_idxs[idx], labels[idx]
                dot = np.dot(self.Pu[u], self.Qi[i])
                pred = self._sigmoid(self.bu[u] + self.bi[i] + dot)
                err = y - pred
                
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.Pu[u] += self.lr * (err * self.Qi[i] - self.reg * self.Pu[u])
                self.Qi[i] += self.lr * (err * self.Pu[u] - self.reg * self.Qi[i])

    def predict(self, u_id, g_id):
        if u_id in self.user_map and g_id in self.item_map:
            u = self.user_map[u_id]
            i = self.item_map[g_id]
            return self._sigmoid(self.bu[u] + self.bi[i] + np.dot(self.Pu[u], self.Qi[i]))
        return 0.5

# =============================================================================
# 3. SAR FEATURE EXTRACTOR (Co-occurrence)
# =============================================================================
class SARFeatureExtractor:
    def __init__(self):
        self.item_users = defaultdict(set)
        self.user_items = defaultdict(set)
        self.item_pop = {}

    def fit(self, df_pos):
        print("   [SAR] Computing item-item statistics...")
        for row in df_pos.itertuples():
            self.item_users[row.gameID].add(row.userID)
            self.user_items[row.userID].add(row.gameID)
        
        total = len(df_pos)
        self.item_pop = {g: len(us) / total for g, us in self.item_users.items()}

    def predict_sim(self, u_id, g_id):
        if u_id not in self.user_items or g_id not in self.item_users:
            return 0.0
        
        user_history = self.user_items[u_id]
        target_users = self.item_users[g_id]
        if not user_history: return 0.0
        
        # Jaccard Sim
        intersection_count = 0
        for hist_item in user_history:
            if hist_item == g_id: continue
            hist_users = self.item_users[hist_item]
            overlap = len(target_users.intersection(hist_users))
            if overlap > 0:
                intersection_count += overlap / (len(target_users) + len(hist_users) - overlap)
                
        return intersection_count / len(user_history)

# =============================================================================
# 4. HYBRID RANKER (The Manager)
# =============================================================================
class LTRHybrid:
    def __init__(self):
        self.svd = BPRLikeSVD(n_factors=30)
        self.sar = SARFeatureExtractor()
        self.content = ContentEngine(max_features=800)
        
        # LightGBM Equivalent in sklearn
        self.ranker = HistGradientBoostingClassifier(
            learning_rate=0.05, 
            max_iter=300, 
            max_depth=7, 
            random_state=42
        )
        self.user_activity = {}

    def generate_negatives(self, df_pos, ratio=2):
        print(f"   Generating negatives (Ratio={ratio})...")
        users = df_pos['userID'].unique()
        items = list(df_pos['gameID'].unique())
        user_items_set = df_pos.groupby('userID')['gameID'].apply(set).to_dict()
        
        neg_rows = []
        for u in tqdm(users, desc="Neg Sampling"):
            seen = user_items_set[u]
            for _ in range(int(len(seen) * ratio)):
                g = random.choice(items)
                while g in seen:
                    g = random.choice(items)
                neg_rows.append({'userID': u, 'gameID': g, 'label': 0})
        return pd.DataFrame(neg_rows)

    def fit(self, df_train):
        print("\n=== Training Hybrid Recommender ===")
        # Keep positive examples (label=1)
        df_pos = df_train[['userID', 'gameID', 'text']].copy()
        df_pos['label'] = 1
        
        # Calculate User Activity (Count of games played) for feature
        self.user_activity = df_pos['userID'].value_counts().to_dict()
        
        # 1. Train Sub-Models
        # Content model needs the Text data
        self.content.fit(df_pos) 
        # SAR needs positive interactions
        self.sar.fit(df_pos)
        
        # 2. Generate Training Data for Ranker (Pos + Neg)
        df_neg = self.generate_negatives(df_pos, ratio=2)
        
        # SVD needs balanced data to learn 0 vs 1
        self.svd.fit(df_pos, df_neg)
        
        # 3. Feature Extraction
        print("   [LTR] extracting features for stacking...")
        df_combined = pd.concat([df_pos[['userID', 'gameID', 'label']], df_neg], ignore_index=True)
        # Shuffle
        df_combined = df_combined.sample(frac=1.0, random_state=42)
        
        X = self._extract_features(df_combined)
        y = df_combined['label'].values
        
        # 4. Train Final Ranker
        print("   [LTR] Training Gradient Boosting Ranker...")
        self.ranker.fit(X, y)
        print(f"   Training AUC: {roc_auc_score(y, self.ranker.predict_proba(X)[:, 1]):.4f}")

    def _extract_features(self, df):
        # We process in a loop (list comprehension) for simplicity/compatibility
        u_list = df['userID'].values
        g_list = df['gameID'].values
        
        svd_scores = []
        sar_scores = []
        pop_scores = []
        content_scores = []
        act_scores = []
        
        for u, g in zip(u_list, g_list):
            svd_scores.append(self.svd.predict(u, g))
            sar_scores.append(self.sar.predict_sim(u, g))
            pop_scores.append(self.sar.item_pop.get(g, 0.0))
            content_scores.append(self.content.predict_sim(u, g))
            act_scores.append(self.user_activity.get(u, 0))
            
        return np.column_stack([svd_scores, sar_scores, pop_scores, content_scores, act_scores])

    def predict(self, test_df, output_file="predictions_Played.csv"):
        print(f"\n=== Predicting for {len(test_df)} pairs ===")
        X_test = self._extract_features(test_df)
        probs = self.ranker.predict_proba(X_test)[:, 1]
        
        # --- MEDIAN THRESHOLDING (Force 50/50 Split) ---
        threshold = np.median(probs)
        binary_preds = (probs >= threshold).astype(int)
        
        sub = test_df.copy()
        sub['prediction'] = binary_preds
        sub.to_csv(output_file, index=False)
        print(f"✓ Saved 0/1 predictions to {output_file} (Threshold={threshold:.4f})")

# =============================================================================
# MAIN
# =============================================================================
def readJSON(path):
    # READ EVERYTHING (including text)
    for l in gzip.open(path, 'rt'):
        yield eval(l)

if __name__ == "__main__":
    try:
        print("Loading data...")
        # Load all data into DataFrame
        data = list(readJSON("train.json.gz"))
        df_train = pd.DataFrame(data)
        
        # Initialize and Train
        model = LTRHybrid()
        model.fit(df_train)
        
        # Predict
        pairs = pd.read_csv("pairs_Played.csv")
        model.predict(pairs)
        
    except FileNotFoundError:
        print("Error: Files not found. Ensure train.json.gz and pairs_Played.csv are present.")
