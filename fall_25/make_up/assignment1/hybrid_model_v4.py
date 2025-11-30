import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from tqdm import tqdm
from itertools import combinations
from scipy.optimize import minimize
import gzip
from textblob import TextBlob # Requires: pip install textblob
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline  # <--- NEW
from sklearn.preprocessing import StandardScaler # <--- NEW
from tqdm import tqdm
from itertools import combinations
from scipy.optimize import minimize
import gzip
from textblob import TextBlob
# =============================================================================
# 1. SIGNAL PROCESSING & FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    def __init__(self):
        self.user_stats = {} 
        self.signal_cols = ['sin_month', 'cos_month', 'sin_weekday', 'cos_weekday', 
                            'sentiment', 'user_lorentz']
        
    def fit(self, df):
        """
        Learn user-specific signal profiles from training data.
        Calculates robust 'User Lorentz Bias' from raw hours.
        """
        print("\nFeatureEngineer: Learning user signal profiles...")
        
        # 1. Lorentz Transformation of User History
        # L(x) = log(1 + x^2/2) - Robust to outliers (dampens extreme 1000h+ values)
        if 'hours' in df.columns:
            # Create a local copy to avoid modifying original during calculation
            temp = df[['userID', 'hours']].copy()
            temp['lorentz'] = np.log(1 + (temp['hours'] ** 2) / 2.0)
            
            # Map userID -> Average Lorentz Hours
            self.user_stats = temp.groupby('userID')['lorentz'].mean().to_dict()
            print(f"   ✓ Computed bias profiles for {len(self.user_stats):,} users")
        
    def transform(self, df):
        """
        Generate signal features for a given DataFrame (Train, Val, or Test).
        Returns DataFrame with added signal columns.
        """
        df_new = df.copy()
        
        # --- A. Temporal Signals (Fourier Transform) ---
        # Maps cyclic time (Jan 1 vs Dec 31) to close points in vector space.
        if 'date' in df_new.columns:
            dates = pd.to_datetime(df_new['date'], errors='coerce')
            
            # Annual Cycle (Month 1-12)
            # sin/cos ensures Month 12 is 'close' to Month 1
            months = dates.dt.month
            df_new['sin_month'] = np.sin(2 * np.pi * months / 12.0).fillna(0)
            df_new['cos_month'] = np.cos(2 * np.pi * months / 12.0).fillna(0)
            
            # Weekly Cycle (Weekday 0-6)
            weekdays = dates.dt.weekday
            df_new['sin_weekday'] = np.sin(2 * np.pi * weekdays / 7.0).fillna(0)
            df_new['cos_weekday'] = np.cos(2 * np.pi * weekdays / 7.0).fillna(0)
        else:
            # Fallback for test data if date is missing
            for col in ['sin_month', 'cos_month', 'sin_weekday', 'cos_weekday']:
                df_new[col] = 0.0

        # --- B. Content Signals (Sentiment) ---
        # Distinguishes "Long play time (Love)" vs "Long play time (Addiction/Grind)"
        if 'text' in df_new.columns:
            # Vectorized approximation or efficient apply
            # Note: TextBlob is somewhat slow on huge data.
            # Using a simple lambda with error handling.
            def get_sentiment(text):
                try:
                    return TextBlob(str(text)).sentiment.polarity
                except:
                    return 0.0
            
            # Use tqdm if interactive, else simple apply
            print("   Extracting sentiment features...")
            df_new['sentiment'] = df_new['text'].fillna("").apply(get_sentiment)
        else:
            df_new['sentiment'] = 0.0
            
        # --- C. User Bias Signal (Lorentz) ---
        # Apply pre-computed user bias. If user is new (cold start), use global mean.
        global_bias = np.mean(list(self.user_stats.values())) if self.user_stats else 0
        df_new['user_lorentz'] = df_new['userID'].map(self.user_stats).fillna(global_bias)
        
        return df_new[self.signal_cols]

# =============================================================================
# 2. ENHANCED HYBRID SCORER
# =============================================================================
class HybridScorer:
    def __init__(self, n_factors=50, k_neighbors=20, max_features=500, 
                 nmf_components=30, active_models=None):
        """
        Signal-Aware Hybrid Recommender.
        """
        self.n_factors = n_factors
        self.k = k_neighbors
        self.max_features = max_features
        self.nmf_components = nmf_components
        self.active_models = active_models
        
        # Mappings & Models
        self.user_map = {}
        self.item_map = {}
        self.global_mean = 0
        self.u_factors = None
        self.vt_factors = None
        self.sim_ii = None
        self.knn_user = None
        self.tfidf = None
        self.user_profiles = None
        self.item_profiles = None
        self.nmf_user = None
        self.nmf_item = None
        
        # Signal Processing
        self.fe = FeatureEngineer()
        
        # Meta Model (Pipeline: Scale -> Ridge)
        # Replaces deprecated normalize=True/False
        self.meta_model = make_pipeline(
            StandardScaler(), 
            Ridge(alpha=1.0, fit_intercept=True)
        )
        
    def fit(self, df):
        """Train base models and feature engineer."""
        print("="*60)
        print("TRAINING HYBRID RECOMMENDER (SIGNAL-AWARE)")
        print("="*60)
        
        print("\n1. Initializing & Feature Engineering...")
        self.global_mean = df['hours_transformed'].mean()
        
        # Fit Signal Engineer
        self.fe.fit(df)
        
        # Standard Mappings
        self.users_arr = df['userID'].unique()
        self.items_arr = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(self.users_arr)}
        self.item_map = {g: i for i, g in enumerate(self.items_arr)}
        
        n_users, n_items = len(self.users_arr), len(self.items_arr)
        print(f"   Users: {n_users:,}, Items: {n_items:,}")
        
        # Sparse Matrix
        rows = df['userID'].map(self.user_map).values
        cols = df['gameID'].map(self.item_map).values
        data = df['hours_transformed'].values
        self.R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # --- MODEL A: SVD ---
        print("\n2. Training Model A (SVD)...")
        item_sums = np.array(self.R.sum(axis=0)).flatten()
        item_counts = np.diff(self.R.tocsc().indptr)
        self.item_means_arr = np.zeros(n_items)
        mask = item_counts > 0
        self.item_means_arr[mask] = item_sums[mask] / item_counts[mask]
        
        R_centered = self.R.copy()
        R_centered.data -= self.item_means_arr[cols]
        u, s, vt = svds(R_centered, k=self.n_factors)
        self.u_factors = u @ np.diag(s)
        self.vt_factors = vt
        
        # --- MODEL B: KNN ---
        print("\n3. Training Model B (User-KNN)...")
        self.knn_user = NearestNeighbors(n_neighbors=self.k, metric='cosine', n_jobs=-1)
        self.knn_user.fit(self.R)
        
        # --- MODEL C: Item Similarity ---
        print("\n4. Training Model C (Item-Cosine)...")
        self.sim_ii = cosine_similarity(self.R.T, dense_output=False)
        
        # --- MODEL D: Content ---
        print("\n5. Training Model D (TF-IDF)...")
        df_clean = df.copy()
        df_clean['text'] = df_clean['text'].fillna('')
        self.tfidf = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        item_corpus = [df_clean[df_clean['gameID']==g]['text'].str.cat(sep=' ') for g in self.items_arr]
        self.item_profiles = self.tfidf.fit_transform(item_corpus)
        user_corpus = [df_clean[df_clean['userID']==u]['text'].str.cat(sep=' ') for u in self.users_arr]
        self.user_profiles = self.tfidf.transform(user_corpus)
        
        # --- MODEL E: NMF ---
        print("\n6. Training Model E (NMF)...")
        try:
            nmf = NMF(n_components=self.nmf_components, init='nndsvd', max_iter=100, random_state=42)
            self.nmf_user = nmf.fit_transform(self.R)
            self.nmf_item = nmf.components_.T
        except:
            print("   (Switching to dense NMF - High Memory)")
            R_dense = self.R.toarray()
            self.nmf_user = nmf.fit_transform(R_dense)
            self.nmf_item = nmf.components_.T
            
        print("\n✓ Base models trained.")

    def _get_individual_preds(self, u_idx, i_idx):
        preds = []
        
        # A: SVD
        pA = self.item_means_arr[i_idx] + np.dot(self.u_factors[u_idx], self.vt_factors[:, i_idx])
        preds.append(pA)
        
        # B: KNN
        dists, idxs = self.knn_user.kneighbors(self.R[u_idx], n_neighbors=self.k)
        sims = 1 - dists.flatten()
        ratings = self.R[idxs.flatten(), i_idx].toarray().flatten()
        if ratings.sum() > 0: preds.append(np.dot(sims, ratings) / (sims.sum() + 1e-9))
        else: preds.append(self.global_mean)
            
        # C: ItemSim
        u_vec = self.R[u_idx]
        if u_vec.nnz > 0:
            sim_row = self.sim_ii[i_idx]
            pC = (u_vec @ sim_row.T).toarray()[0][0] / (np.abs(sim_row).sum() + 1e-9)
            if pC == 0: pC = self.global_mean
        else: pC = self.global_mean
        preds.append(pC)
        
        # D: Content
        sim = (self.user_profiles[u_idx] @ self.item_profiles[i_idx].T).toarray()[0][0]
        preds.append(sim * 10.0)
        
        # E: NMF
        preds.append(np.dot(self.nmf_user[u_idx], self.nmf_item[i_idx]))
        
        return preds

    def _get_base_preds_and_signals(self, df):
        # 1. Base Model Predictions
        X_models = []
        y_true = []
        
        print("   Generating base model predictions...")
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            if getattr(row, 'userID') in self.user_map and getattr(row, 'gameID') in self.item_map:
                u_idx = self.user_map[row.userID]
                i_idx = self.item_map[row.gameID]
                X_models.append(self._get_individual_preds(u_idx, i_idx))
                if hasattr(row, 'hours_transformed'):
                    y_true.append(row.hours_transformed)
                else:
                    y_true.append(0)
            else:
                X_models.append([self.global_mean] * 5)
                y_true.append(0)
                
        X_models = np.array(X_models)
        y_true = np.array(y_true)
        
        # 2. Signal Features
        print("   Generating signal features...")
        df_signals = self.fe.transform(df)
        X_signals = df_signals.values
        
        # 3. Stack
        X_full = np.hstack([X_models, X_signals])
        
        return X_full, y_true

    def learn_weights(self, val_df):
        print("\n" + "="*60)
        print("LEARNING META-MODEL WEIGHTS")
        print("="*60)
        
        X_stack, y_val = self._get_base_preds_and_signals(val_df)
        
        print(f"\nFeature Matrix Shape: {X_stack.shape}")
        print("Training Ridge Regression (Wide & Deep)...")
        
        self.meta_model.fit(X_stack, y_val)
        
        # Display Weights
        model_names = ['SVD', 'UserKNN', 'ItemKNN', 'Content', 'NMF']
        signal_names = self.fe.signal_cols
        all_names = model_names + signal_names
        
        # FIX: Access coefficients from the Ridge step of the pipeline
        ridge_model = self.meta_model.named_steps['ridge']
        
        print("\nLearned Weights:")
        for name, w in zip(all_names, ridge_model.coef_):
            print(f"  {name:15s}: {w:.4f}")
            
        preds = self.meta_model.predict(X_stack)
        mse = mean_squared_error(y_val, preds)
        print(f"\n✓ Validation MSE: {mse:.4f}")
        return mse

    def find_best_model_subset(self, val_df, verbose=True):
        print("\n" + "="*60)
        print("OPTIMIZING BASE MODEL SUBSET")
        print("="*60)
        
        X_full, y_true = self._get_base_preds_and_signals(val_df)
        n_base = 5
        X_models = X_full[:, :n_base]
        X_signals = X_full[:, n_base:]
        
        best_mse = float('inf')
        best_subset = list(range(n_base))
        
        for r in range(1, n_base + 1):
            for subset in combinations(range(n_base), r):
                subset = list(subset)
                X_sub = np.hstack([X_models[:, subset], X_signals])
                
                # Use pipeline here too for consistency
                clf = make_pipeline(StandardScaler(), Ridge(fit_intercept=True))
                clf.fit(X_sub, y_true)
                preds = clf.predict(X_sub)
                mse = mean_squared_error(y_true, preds)
                
                if mse < best_mse:
                    best_mse = mse
                    best_subset = subset
                    if verbose:
                        print(f"New Best: Models {subset} -> MSE {mse:.5f}")
                        
        self.active_models = best_subset
        print(f"\n🏆 Best Subset: {self.active_models} (MSE: {best_mse:.5f})")
        return {'best_subset': best_subset}

    def make_test_predictions(self, test_df, output_path='predictions_Hours.csv'):
        print(f"\nGenerating predictions for {len(test_df)} pairs...")
        
        X_full, _ = self._get_base_preds_and_signals(test_df)
        n_base = 5
        X_models = X_full[:, :n_base]
        X_signals = X_full[:, n_base:]
        
        if self.active_models is not None:
            X_models = X_models[:, self.active_models]
            
        X_final = np.hstack([X_models, X_signals])
        
        preds = self.meta_model.predict(X_final)
        preds = np.clip(preds, 0, 14.5)
        
        sub = test_df[['userID', 'gameID']].copy()
        sub['prediction'] = preds
        sub.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
# =============================================================================
# 3. PIPELINE UTILS
# =============================================================================

def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d['userID'], d.get('gameID'), d

def split_train_data(df, test_size=0.2):
    # Simple random split for brevity
    mask = np.random.rand(len(df)) < (1 - test_size)
    return df[mask].copy(), df[~mask].copy()

if __name__ == "__main__":
    # Load Data
    print("Loading data...")
    # NOTE: Ensure train.json.gz is in the directory
    data = [d for u, g, d in readJSON("train.json.gz")]
    df = pd.DataFrame(data)
    
    # Split
    df_train, df_val = split_train_data(df)
    
    # Pipeline
    scorer = HybridScorer(n_factors=50, nmf_components=30)
    scorer.fit(df_train)
    
    # Optimize & Train Meta-Model
    res = scorer.find_best_model_subset(df_val)
    
    # Re-train meta-model on best subset
    # Note: find_best_model_subset sets self.active_models, 
    # so we just need to fit the meta_model on the correct columns now.
    X_full, y_val = scorer._get_base_preds_and_signals(df_val)
    n_base = 5
    X_models = X_full[:, :n_base][:, scorer.active_models]
    X_signals = X_full[:, n_base:]
    X_final = np.hstack([X_models, X_signals])
    
    scorer.meta_model.fit(X_final, y_val)
    
    # Prediction
    pairs_df = pd.read_csv("pairs_Hours.csv")
    scorer.make_test_predictions(pairs_df)
