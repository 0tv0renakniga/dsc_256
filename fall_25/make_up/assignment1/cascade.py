import gzip
import csv
import json
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from collections import defaultdict
import warnings

# --- Production Imports ---
import lightgbm as lgb
import catboost as cb

# --- NLP Imports (Optional) ---
try:
    # Requires: pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
except ImportError:
    print("Warning: 'sentence_transformers' not found. NLP features will be skipped.")
    NLP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
FILES = {
    'train': 'train.json.gz',         
    'test_hours': 'pairs_Hours.csv', 
    'test_played': 'pairs_Played.csv',
}

# ==============================================================================
# 2. DATA LOADING & NLP (COMBINED)
# ==============================================================================
def load_data_and_extract_metadata(path):
    """
    Performs a single pass load of train.json.gz to get:
    1. raw_data list (for training DataFrame)
    2. game_text_metadata dict (for NLP)
    """
    print(f"[Data] Loading and Extracting Metadata from {path}...")
    raw_data = []
    game_texts = {} 

    with gzip.open(path, 'rt') as f:
        for line in f:
            d = eval(line)
            raw_data.append(d)
            
            # Extract Text for Embeddings
            game_id = d.get('gameID')
            text = d.get('text') or d.get('description', '') 
            
            if game_id and text:
                game_texts[game_id] = text

    return raw_data, game_texts

def generate_game_embeddings(game_texts_dict):
    """Generates 10-dim SBERT embeddings."""
    if not NLP_AVAILABLE or not game_texts_dict:
        print("[NLP] Skipping embedding generation.")
        return {}
        
    print(f"[NLP] Generating SBERT Embeddings for {len(game_texts_dict)} games...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Encode
    ids = list(game_texts_dict.keys())
    texts = list(game_texts_dict.values())
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    
    # 2. Reduce Dimension (384 -> 10)
    pca = PCA(n_components=10, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    return {gid: vec.tolist() for gid, vec in zip(ids, reduced)}

# ==============================================================================
# 3. CORE PROCESSOR (Features + H-Score + NLP)
# ==============================================================================
class DataProcessor:
    def __init__(self, game_embeddings=None):
        self.user_stats = {}
        self.game_stats = {}
        self.signatures_stats = {}
        self.h_scores = {}
        self.game_embeddings = game_embeddings if game_embeddings else {}
        self.global_mean = 0
        self.global_median_h = 0 
        self.u_map = {} 
        self.g_map = {} 

    def fit_transform(self, df):
        self.global_mean = df['hours_trans'].mean()
        
        u_grp = df.groupby('userID')['hours_trans'].agg(['count', 'mean']).rename(columns={'count': 'u_count', 'mean': 'u_mean'})
        g_grp = df.groupby('gameID')['hours_trans'].agg(['count', 'mean']).rename(columns={'count': 'g_count', 'mean': 'g_mean'})
        
        u_grp['u_bin'] = pd.qcut(u_grp['u_count'], q=10, labels=False, duplicates='drop')
        g_grp['g_bin'] = pd.qcut(g_grp['g_count'], q=10, labels=False, duplicates='drop')
        
        df = df.merge(u_grp, on='userID', how='left')
        df = df.merge(g_grp, on='gameID', how='left')
        
        # H-SCORE CALCULATION
        df['residual'] = df['hours_trans'] - df['g_mean']
        u_variance = df.groupby('userID')['residual'].var()
        self.global_median_h = u_variance.median()
        u_variance = u_variance.fillna(self.global_median_h)
        self.h_scores = u_variance.to_dict()
        df['h_score'] = df['userID'].map(self.h_scores)
        
        df['signature'] = df['u_bin'].astype(str) + "_" + df['g_bin'].astype(str)
        
        sig_agg = df.groupby('signature')['hours_trans'].agg(['std', 'mean', 'count'])
        sig_agg['std'] = sig_agg['std'].fillna(100)
        
        conditions = [(sig_agg['count'] < 10), (sig_agg['std'] < 0.7), (sig_agg['std'] <= 1.2)]
        choices = [3, 1, 2]
        sig_agg['tier'] = np.select(conditions, choices, default=3)
        
        self.user_stats = u_grp.to_dict('index')
        self.game_stats = g_grp.to_dict('index')
        self.signatures_stats = sig_agg.to_dict('index')
        
        df = df.merge(sig_agg[['tier', 'std']], on='signature', how='left').rename(columns={'std': 'sig_std'})
        return df

    def get_embedding_features(self, game_ids):
        """Retrieves 10-dim embedding features for a list of game IDs."""
        if not self.game_embeddings: return None
        default_vec = [0.0] * 10
        vecs = [self.game_embeddings.get(g, default_vec) for g in game_ids]
        return np.array(vecs)

# ==============================================================================
# 4. CASCADE REGRESSOR (Goal 1 Logic)
# ==============================================================================
class CascadeRegressor:
    def __init__(self, n_components=10, game_embeddings=None):
        # FIX IS HERE: game_embeddings is now correctly accepted
        self.processor = DataProcessor(game_embeddings)
        self.svd_model = None
        self.svd_U = None
        self.svd_V = None
        self.ensemble_models = []
        self.n_components = n_components
        
    def _train_tier_models(self, df):
        
        # --- NOISE FILTERING for SVD ---
        noise_thresh = df['h_score'].quantile(0.90)
        is_clean = df['h_score'] <= noise_thresh
        print(f"[Train] Noise Filter: Excluding users with Variance > {noise_thresh:.2f} from SVD")
        
        # TIER 2: SVD (Clean Data Only)
        t2_data = df[(df['tier'] == 2) & (is_clean)] 
        
        if not t2_data.empty:
            u_uniques = t2_data['userID'].unique()
            g_uniques = t2_data['gameID'].unique()
            self.processor.u_map = {u: i for i, u in enumerate(u_uniques)}
            self.processor.g_map = {g: i for i, g in enumerate(g_uniques)}
            rows = t2_data['userID'].map(self.processor.u_map)
            cols = t2_data['gameID'].map(self.processor.g_map)
            vals = t2_data['hours_trans']
            R = sp.coo_matrix((vals, (rows, cols)), shape=(len(u_uniques), len(g_uniques)))
            n_comps = min(self.n_components, R.shape[1]-1, R.shape[0]-1)
            if n_comps > 1:
                self.svd_model = TruncatedSVD(n_components=n_comps, random_state=42)
                self.svd_U = self.svd_model.fit_transform(R)
                self.svd_V = self.svd_model.components_
            else: self.svd_model = None
        else: self.svd_model = None

        # TIER 3: ENSEMBLE (All Data + H-Score + NLP)
        t3_data = df[df['tier'] == 3]
        if not t3_data.empty:
            features_df = t3_data[['u_mean', 'g_mean', 'u_count', 'g_count', 'h_score']].reset_index(drop=True)
            emb_matrix = self.processor.get_embedding_features(t3_data['gameID'])
            if emb_matrix is not None:
                emb_df = pd.DataFrame(emb_matrix, columns=[f'emb_{i}' for i in range(10)])
                X_t3 = pd.concat([features_df, emb_df], axis=1)
            else: X_t3 = features_df
            y_t3 = t3_data['hours_trans'].values
            
            print(f"[Train] Tier 3 Ensemble on {len(X_t3)} samples with {X_t3.shape[1]} features")

            lgb_reg = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, verbose=-1)
            cb_reg = cb.CatBoostRegressor(iterations=150, learning_rate=0.05, depth=8, verbose=0, allow_writing_files=False)
            gbr_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
            
            lgb_reg.fit(X_t3, y_t3)
            cb_reg.fit(X_t3, y_t3)
            gbr_reg.fit(X_t3, y_t3)
            self.ensemble_models = [lgb_reg, cb_reg, gbr_reg]
        else: self.ensemble_models = []

    def predict_batch(self, test_df, is_validation=False):
        test_df = test_df.copy()
        
        # 1. MERGE FEATURES
        u_stats = pd.DataFrame.from_dict(self.processor.user_stats, orient='index')
        g_stats = pd.DataFrame.from_dict(self.processor.game_stats, orient='index')
        sig_stats = pd.DataFrame.from_dict(self.processor.signatures_stats, orient='index')
        h_score_series = pd.Series(self.processor.h_scores, name='h_score')

        default_u = {'u_count': 0, 'u_mean': self.processor.global_mean, 'u_bin': 5}
        default_g = {'g_count': 0, 'g_mean': self.processor.global_mean, 'g_bin': 5}

        test_df = test_df.merge(u_stats, left_on='userID', right_index=True, how='left').fillna(default_u)
        test_df = test_df.merge(g_stats, left_on='gameID', right_index=True, how='left').fillna(default_g)
        test_df = test_df.merge(h_score_series, left_on='userID', right_index=True, how='left')
        test_df['h_score'] = test_df['h_score'].fillna(self.processor.global_median_h)
        
        test_df['signature'] = test_df['u_bin'].astype(int).astype(str) + "_" + test_df['g_bin'].astype(int).astype(str)
        test_df = test_df.merge(sig_stats[['tier', 'mean']], left_on='signature', right_index=True, how='left').rename(columns={'mean': 'sig_mean'})
        test_df['tier'] = test_df['tier'].fillna(3).astype(int)
        test_df['sig_mean'] = test_df['sig_mean'].fillna(self.processor.global_mean)

        # 2. PREDICT
        test_df['final_pred'] = np.nan
        
        # Tier 1
        mask_t1 = (test_df['tier'] == 1)
        if mask_t1.any(): test_df.loc[mask_t1, 'final_pred'] = test_df.loc[mask_t1, 'sig_mean']

        # Tier 2
        mask_t2 = (test_df['tier'] == 2)
        if mask_t2.any():
            if self.svd_model is not None:
                u_idx = test_df.loc[mask_t2, 'userID'].map(self.processor.u_map)
                g_idx = test_df.loc[mask_t2, 'gameID'].map(self.processor.g_map)
                valid = (u_idx.notna()) & (g_idx.notna())
                valid_indices = mask_t2 & valid.reindex(test_df.index, fill_value=False)
                
                if valid_indices.any():
                    u_vecs = self.svd_U[u_idx[valid_indices].astype(int)]
                    v_vecs = self.svd_V[:, g_idx[valid_indices].astype(int)].T
                    svd_preds = np.sum(u_vecs * v_vecs, axis=1)
                    test_df.loc[valid_indices, 'final_pred'] = svd_preds
                
                cold_indices = mask_t2 & (~valid.reindex(test_df.index, fill_value=True))
                if cold_indices.any():
                    test_df.loc[cold_indices, 'final_pred'] = test_df.loc[cold_indices, 'sig_mean']
            else:
                test_df.loc[mask_t2, 'final_pred'] = test_df.loc[mask_t2, 'sig_mean']

        # Tier 3
        mask_t3 = (test_df['tier'] == 3) | (test_df['final_pred'].isna())
        if mask_t3.any() and self.ensemble_models:
            t3_subset = test_df.loc[mask_t3]
            feat_base = t3_subset[['u_mean', 'g_mean', 'u_count', 'g_count', 'h_score']].reset_index(drop=True)
            emb_matrix = self.processor.get_embedding_features(t3_subset['gameID'])
            if emb_matrix is not None:
                emb_df = pd.DataFrame(emb_matrix, columns=[f'emb_{i}' for i in range(10)])
                X_feat = pd.concat([feat_base, emb_df], axis=1)
            else: X_feat = feat_base
                
            if not X_feat.empty:
                p1 = self.ensemble_models[0].predict(X_feat)
                p2 = self.ensemble_models[1].predict(X_feat)
                p3 = self.ensemble_models[2].predict(X_feat)
                test_df.loc[mask_t3, 'final_pred'] = (p1 + p2 + p3) / 3.0
            
        test_df['final_pred'] = test_df['final_pred'].fillna(self.processor.global_mean)
        return test_df['final_pred'].values

    def run_cv(self, df_raw, k=5):
        print(f"\n[CV] Starting {k}-Fold Validation...")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        analysis_records = []
        fold = 1
        
        for train_idx, val_idx in kf.split(df_raw):
            print(f"   Processing Fold {fold}/{k}...")
            train_sub = df_raw.iloc[train_idx].copy()
            val_sub = df_raw.iloc[val_idx].copy()
            
            # Use embeddings from the main instance
            self.processor = DataProcessor(self.processor.game_embeddings)
            train_sub = self.processor.fit_transform(train_sub)
            self._train_tier_models(train_sub)
            
            preds = self.predict_batch(val_sub, is_validation=True)
            
            val_sub['prediction'] = preds
            val_sub['mse'] = (val_sub['hours_trans'] - val_sub['prediction']) ** 2
            val_sub['raw_residual'] = val_sub['prediction'] - val_sub['hours_trans']
            
            subset = val_sub[['userID', 'gameID', 'hours_trans', 'prediction', 'mse', 'raw_residual']].copy()
            subset['fold'] = fold
            analysis_records.append(subset)
            print(f"   Fold {fold} MSE: {subset['mse'].mean():.4f}")
            fold += 1
            
        full_analysis = pd.concat(analysis_records)
        print(f"[CV] Overall Average MSE: {full_analysis['mse'].mean():.4f}")
        return full_analysis

    def train_final(self, df):
        print("\n[Final Train] Training on full dataset...")
        # Preserve embeddings when resetting processor
        self.processor = DataProcessor(self.processor.game_embeddings) 
        df = self.processor.fit_transform(df)
        self._train_tier_models(df)


# ==============================================================================
# 5. GOAL 2: CLASSIFIER (Enhanced with H-Score and Embeddings)
# ==============================================================================
def train_play_classifier(all_data, h_scores, global_median_h, embeddings):
    print("[Goal 2] Preparing Classification Data (50/50 split)...")
    
    # 1. Positive and Negative Sampling
    pos_df = pd.DataFrame(all_data)
    pos_df['played'] = 1
    
    played_map = defaultdict(set)
    all_games = pos_df['gameID'].unique()
    for d in all_data: played_map[d['userID']].add(d['gameID'])
        
    neg_data = []
    users = pos_df['userID'].unique()
    target_count = len(pos_df)
    
    while len(neg_data) < target_count:
        u_rand = np.random.choice(users, size=target_count)
        g_rand = np.random.choice(all_games, size=target_count)
        
        new_negs = [{'userID': u, 'gameID': g, 'played': 0} 
                    for u, g in zip(u_rand, g_rand) if (u, g) not in played_map]
        neg_data.extend(new_negs)
        neg_data = neg_data[:target_count] 
    
    train_df = pd.concat([pos_df[['userID', 'gameID', 'played']], pd.DataFrame(neg_data)], ignore_index=True)
    
    # 2. Feature Construction
    u_counts = train_df[train_df['played']==1].groupby('userID').size().to_dict()
    g_counts = train_df[train_df['played']==1].groupby('gameID').size().to_dict()
    
    train_df['u_act'] = train_df['userID'].map(u_counts).fillna(0)
    train_df['g_pop'] = train_df['gameID'].map(g_counts).fillna(0)

    # Add H-Score
    train_df['h_score'] = train_df['userID'].map(h_scores).fillna(global_median_h)
    
    # Define base features
    features = ['u_act', 'g_pop', 'h_score'] 
    
    # Add Embeddings
    if embeddings:
        emb_matrix = np.array([embeddings.get(g, [0.0]*10) for g in train_df['gameID']])
        emb_df = pd.DataFrame(emb_matrix, columns=[f'emb_{i}' for i in range(10)])
        train_df = pd.concat([train_df.reset_index(drop=True), emb_df], axis=1)
        features.extend([f'emb_{i}' for i in range(10)])
        
    # 3. Train LightGBM Classifier
    print(f"[Goal 2] Training Classifier with {len(features)} features...")
    X = train_df[features]
    y = train_df['played']
    
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
    clf.fit(X, y)
    
    return clf, u_counts, g_counts, features

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # A. LOAD DATA & PREP NLP
    raw_data, game_texts = load_data_and_extract_metadata(FILES['train'])
    embeddings = generate_game_embeddings(game_texts)
    
    df_train = pd.DataFrame(raw_data)
    df_train['hours_trans'] = np.log2(df_train['hours'] + 1)
    
    # B. TRAIN CASCADE (Goal 1)
    print("\n--- GOAL 1: REGRESSION CASCADE ---")
    
    # Instantiate CascadeRegressor correctly
    cascade = CascadeRegressor(n_components=10, game_embeddings=embeddings)
    
    # Run CV and generate error report
    error_df = cascade.run_cv(df_train, k=5)
    error_df.to_csv("error_analysis_enhanced.csv", index=False)
    print("\nDetailed Error Analysis saved to 'error_analysis_enhanced.csv'")
    
    # Final Training
    cascade.train_final(df_train)
    
    # Predict Goal 1
    try:
        test_pairs = pd.read_csv(FILES['test_hours'])
        print("[Goal 1] Batch Predicting...")
        preds = cascade.predict_batch(test_pairs)
        test_pairs['prediction'] = preds
        test_pairs.to_csv("predictions_Hours.csv", columns=['userID', 'gameID', 'prediction'], index=False)
        print("Goal 1 Predictions saved.")
    except Exception as e:
        print(f"Goal 1 Prediction Skipped: {e}")

    # C. TRAIN CLASSIFIER (Goal 2)
    print("\n--- GOAL 2: CLASSIFICATION ---")
    
    # Use stats calculated during the final cascade training
    h_scores = cascade.processor.h_scores
    global_median_h = cascade.processor.global_median_h
    
    clf, u_c, g_c, clf_features = train_play_classifier(
        raw_data, 
        h_scores, 
        global_median_h, 
        embeddings
    )
    
    # Predict Goal 2
    try:
        test_played = pd.read_csv(FILES['test_played'])
        
        # 1. Vectorized Feature Construction for TEST SET
        test_played['u_act'] = test_played['userID'].map(u_c).fillna(0)
        test_played['g_pop'] = test_played['gameID'].map(g_c).fillna(0)
        test_played['h_score'] = test_played['userID'].map(h_scores).fillna(global_median_h)
        
        # 2. Add Embeddings
        X_test_base = test_played[['u_act', 'g_pop', 'h_score']].copy().reset_index(drop=True)
        emb_matrix = cascade.processor.get_embedding_features(test_played['gameID'])
        
        if emb_matrix is not None:
            emb_df = pd.DataFrame(emb_matrix, columns=[f'emb_{i}' for i in range(10)])
            X_test = pd.concat([X_test_base, emb_df], axis=1)
        else:
            X_test = X_test_base

        # Ensure column order matches training features list
        X_test = X_test[clf_features]
        
        # 3. Predict Probability & Threshold
        print("[Goal 2] Batch Predicting...")
        probs = clf.predict_proba(X_test)[:, 1]
        test_played['prob'] = probs
        
        median_thresh = test_played['prob'].median()
        test_played['prediction'] = (test_played['prob'] > median_thresh).astype(int)
        
        test_played.to_csv("predictions_Played.csv", columns=['userID', 'gameID', 'prediction'], index=False)
        print("Goal 2 Predictions saved (Binary 0/1 Split).")
        
    except Exception as e:
        print(f"Goal 2 Prediction Skipped: {e}")
