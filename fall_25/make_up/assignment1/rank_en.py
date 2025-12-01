import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from collections import defaultdict
from tqdm import tqdm
import gzip
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONTENT ENGINE WITH LSA (Topics instead of Words)
# =============================================================================
class ContentEngineLSA:
    def __init__(self, max_features=5000, n_components=50):
        # TF-IDF: Higher features to capture detail
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1,2))
        # SVD: Compress to 50 "Topic" dimensions (Dense Embeddings)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.game_vectors = None
        self.user_profiles = {}
        self.game_map = {}

    def fit(self, df):
        print("   [Content] Vectorizing & Decomposing text...")
        df['text'] = df['text'].fillna('')
        game_docs = df.groupby('gameID')['text'].apply(lambda x: " ".join(x.astype(str)))
        
        self.game_ids = game_docs.index.tolist()
        self.game_map = {g: i for i, g in enumerate(self.game_ids)}
        
        # 1. Sparse TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(game_docs.values)
        
        # 2. Dense LSA (Latent Topics)
        self.game_vectors = self.svd.fit_transform(tfidf_matrix)
        
        # 3. Build User Profiles
        print("   [Content] Building user topic profiles...")
        user_games = df.groupby('userID')['gameID'].apply(list).to_dict()
        
        for u, games in tqdm(user_games.items(), desc="User Profiles"):
            valid_indices = [self.game_map[g] for g in games if g in self.game_map]
            if valid_indices:
                # Average Topic Vector
                user_vec = np.mean(self.game_vectors[valid_indices], axis=0)
                self.user_profiles[u] = user_vec

    def predict_sim(self, u_id, g_id):
        # Cosine Similarity on Dense LSA Vectors
        if u_id in self.user_profiles and g_id in self.game_map:
            u_vec = self.user_profiles[u_id]
            g_idx = self.game_map[g_id]
            g_vec = self.game_vectors[g_idx]
            
            # Manual Cosine Sim
            dot = np.dot(u_vec, g_vec)
            norm_u = np.linalg.norm(u_vec)
            norm_g = np.linalg.norm(g_vec)
            
            if norm_u > 0 and norm_g > 0:
                return dot / (norm_u * norm_g)
        return 0.0

# =============================================================================
# 2. BPR-LIKE SVD (Behavioral Factors)
# =============================================================================
class BPRLikeSVD:
    def __init__(self, n_factors=40, n_epochs=20, lr=0.02, reg=0.01):
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
        print("   [SAR] Computing co-occurrence stats...")
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
        
        intersection_count = 0
        for hist_item in user_history:
            if hist_item == g_id: continue
            hist_users = self.item_users[hist_item]
            overlap = len(target_users.intersection(hist_users))
            if overlap > 0:
                intersection_count += overlap / (len(target_users) + len(hist_users) - overlap)
                
        return intersection_count / len(user_history)

# =============================================================================
# 4. ENSEMBLE RANKER (Bagging)
# =============================================================================
class EnsembleHybrid:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.svd = BPRLikeSVD(n_factors=40)
        self.sar = SARFeatureExtractor()
        self.content = ContentEngineLSA(max_features=5000, n_components=60)
        
        # Bagging: We will train multiple GBDT classifiers
        self.rankers = []
        self.user_activity = {}

    def generate_negatives(self, df_pos, ratio=2):
        print(f"   Generating negatives (Ratio={ratio})...")
        users = df_pos['userID'].unique()
        items = list(df_pos['gameID'].unique())
        user_items_set = df_pos.groupby('userID')['gameID'].apply(set).to_dict()
        
        neg_rows = []
        for u in tqdm(users, desc="Neg Sampling"):
            seen = user_items_set[u]
            # Increased ratio for better discriminator training
            for _ in range(int(len(seen) * ratio)):
                g = random.choice(items)
                while g in seen:
                    g = random.choice(items)
                neg_rows.append({'userID': u, 'gameID': g, 'label': 0})
        return pd.DataFrame(neg_rows)

    def fit(self, df_train):
        print("\n=== Training Ensemble Hybrid Recommender ===")
        df_pos = df_train[['userID', 'gameID', 'text']].copy()
        df_pos['label'] = 1
        
        self.user_activity = df_pos['userID'].value_counts().to_dict()
        
        # 1. Train Feature Extractors
        self.content.fit(df_pos) 
        self.sar.fit(df_pos)
        
        # 2. Prepare Training Data
        df_neg = self.generate_negatives(df_pos, ratio=2)
        self.svd.fit(df_pos, df_neg)
        
        print("   [Ensemble] Building Feature Matrix...")
        df_combined = pd.concat([df_pos[['userID', 'gameID', 'label']], df_neg], ignore_index=True)
        
        # Extract Base Features
        X_base = self._extract_features(df_combined)
        y = df_combined['label'].values
        
        # 3. Train Multiple Models (Bagging)
        print(f"   [Ensemble] Training {self.n_models} Gradient Boosters...")
        for i in range(self.n_models):
            # Random seed and subsample for diversity
            clf = HistGradientBoostingClassifier(
                learning_rate=0.05, 
                max_iter=300, 
                max_depth=6, 
                random_state=42 + i,
                validation_fraction=0.1
            )
            # Bagging: Train on 80% sample each time
            mask = np.random.rand(len(X_base)) < 0.8
            clf.fit(X_base[mask], y[mask])
            self.rankers.append(clf)
            print(f"      Model {i+1}/{self.n_models} trained.")

    def _extract_features(self, df):
        u_list = df['userID'].values
        g_list = df['gameID'].values
        
        # Base Features
        svd_scores = np.array([self.svd.predict(u, g) for u, g in zip(u_list, g_list)])
        sar_scores = np.array([self.sar.predict_sim(u, g) for u, g in zip(u_list, g_list)])
        pop_scores = np.array([self.sar.item_pop.get(g, 0.0) for g in g_list])
        content_scores = np.array([self.content.predict_sim(u, g) for u, g in zip(u_list, g_list)])
        act_scores = np.array([self.user_activity.get(u, 0) for u in u_list])
        
        # Feature Crosses (The "Trick")
        # Interaction between Content and Behavior is often a strong signal
        svd_x_content = svd_scores * content_scores
        sar_x_pop = sar_scores * pop_scores
        
        return np.column_stack([
            svd_scores, sar_scores, pop_scores, content_scores, act_scores,
            svd_x_content, sar_x_pop
        ])

    def predict(self, test_df, output_file="predictions_Played.csv"):
        print(f"\n=== Predicting for {len(test_df)} pairs ===")
        X_test = self._extract_features(test_df)
        
        # Average predictions from all models
        probs_sum = np.zeros(len(X_test))
        for clf in self.rankers:
            probs_sum += clf.predict_proba(X_test)[:, 1]
        
        avg_probs = probs_sum / self.n_models
        
        # --- MEDIAN THRESHOLDING (Force 50/50 Split) ---
        threshold = np.median(avg_probs)
        binary_preds = (avg_probs >= threshold).astype(int)
        
        sub = test_df.copy()
        sub['prediction'] = binary_preds
        sub.to_csv(output_file, index=False)
        print(f"✓ Saved 0/1 predictions to {output_file} (Threshold={threshold:.4f})")

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
        
        model = EnsembleHybrid(n_models=5)
        model.fit(df_train)
        
        pairs = pd.read_csv("pairs_Played.csv")
        model.predict(pairs)
        
    except FileNotFoundError:
        print("Error: Files not found. Ensure train.json.gz and pairs_Played.csv are present.")
