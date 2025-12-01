import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from tqdm import tqdm
from itertools import combinations
import gzip
from textblob import TextBlob

# =============================================================================
# 1. TEMPORAL SVD (Model F)
# =============================================================================

class TemporalSVD:
    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, n_bins=30):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.n_bins = n_bins
        self.mu = 0
        self.bu = None; self.bi = None; self.bi_bin = None; self.alpha_u = None
        self.Pu = None; self.Qi = None
        self.user_map = {}; self.item_map = {}
        self.u_mean_date = {}
        self.min_date = 0; self.bin_size = 1

    def _get_bin(self, t):
        b = int((t - self.min_date) / self.bin_size)
        return max(0, min(self.n_bins - 1, b))

    def _get_dev(self, u_idx, t):
        t_mean = self.u_mean_date.get(u_idx, self.min_date)
        return (t - t_mean) / (30 * 86400 * 1e9) 

    def fit(self, df):
        print("   Training Temporal SGD (Time-SVD)...")
        if 'date' in df.columns:
            times = pd.to_datetime(df['date'], errors='coerce').astype(np.int64)
        else:
            times = pd.Series(0, index=df.index)
            
        self.min_date = times.min()
        self.max_date = times.max()
        time_span = self.max_date - self.min_date
        if time_span == 0: time_span = 1
        self.bin_size = time_span / self.n_bins
        
        temp_df = pd.DataFrame({'u': df['userID'], 't': times})
        self.u_mean_date = temp_df.groupby('u')['t'].mean().to_dict()
        
        users = df['userID'].unique()
        items = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        self.u_mean_date = {self.user_map[u]: t for u, t in self.u_mean_date.items() if u in self.user_map}

        n_users, n_items = len(users), len(items)
        self.mu = df['hours_transformed'].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.bi_bin = np.zeros((n_items, self.n_bins))
        self.alpha_u = np.zeros(n_users)
        self.Pu = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Qi = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        u_indices = df['userID'].map(self.user_map).values
        i_indices = df['gameID'].map(self.item_map).values
        ratings = df['hours_transformed'].values
        t_values = times.values
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            current_lr = self.lr * (0.9 ** epoch)
            for idx in indices:
                u, i, r, t = u_indices[idx], i_indices[idx], ratings[idx], t_values[idx]
                b_idx = self._get_bin(t)
                dev = self._get_dev(u, t)
                pred = self.mu + self.bu[u] + self.bi[i] + self.bi_bin[i, b_idx] + \
                       self.alpha_u[u] * dev + np.dot(self.Pu[u], self.Qi[i])
                err = r - pred
                self.bu[u] += current_lr * (err - self.reg * self.bu[u])
                self.bi[i] += current_lr * (err - self.reg * self.bi[i])
                self.bi_bin[i, b_idx] += current_lr * (err - self.reg * self.bi_bin[i, b_idx])
                self.alpha_u[u] += current_lr * (err * dev - self.reg * self.alpha_u[u])
                pu_old = self.Pu[u].copy()
                self.Pu[u] += current_lr * (err * self.Qi[i] - self.reg * self.Pu[u])
                self.Qi[i] += current_lr * (err * pu_old - self.reg * self.Qi[i])

    def predict_one(self, u_id, g_id, date_val):
        if u_id not in self.user_map or g_id not in self.item_map: return self.mu
        u = self.user_map[u_id]
        i = self.item_map[g_id]
        try: t = pd.to_datetime(date_val).value
        except: t = self.min_date
        b_idx = self._get_bin(t)
        dev = self._get_dev(u, t)
        return self.mu + self.bu[u] + self.bi[i] + self.bi_bin[i, b_idx] + \
               self.alpha_u[u] * dev + np.dot(self.Pu[u], self.Qi[i])

# =============================================================================
# 2. FEATURE ENGINEER
# =============================================================================

class LowPlaytimeClassifier:
    def __init__(self, max_features=1000):
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
        self.threshold = 2.0 
    def fit(self, df):
        print("   Training 'Bouncer Detector'...")
        y_binary = (df['hours_transformed'] < self.threshold).astype(int)
        texts = df['text'].fillna("").astype(str) if 'text' in df.columns else pd.Series([""] * len(df))
        X_text = self.tfidf.fit_transform(texts)
        sentiments = texts.apply(lambda x: TextBlob(x).sentiment.polarity).values.reshape(-1, 1)
        self.clf.fit(hstack([X_text, sentiments]), y_binary)
    def predict_proba(self, df):
        texts = df['text'].fillna("").astype(str) if 'text' in df.columns else pd.Series([""] * len(df), index=df.index)
        X_text = self.tfidf.transform(texts)
        sentiments = texts.apply(lambda x: TextBlob(x).sentiment.polarity).values.reshape(-1, 1)
        return self.clf.predict_proba(hstack([X_text, sentiments]))[:, 1]

class FeatureEngineer:
    def __init__(self, user_reg_alpha=12):
        self.alpha = user_reg_alpha
        self.item_stats = {} 
        self.user_stats = {}
        self.user_rates = {} 
        self.item_rates = {}
        self.user_raw_means = {}
        self.global_mean = 0
        self.global_raw_mean = 0
        self.bouncer_clf = LowPlaytimeClassifier()
        self.signal_cols = [
            'item_mean_stat', 'user_bias_stat', 'user_rate', 'item_rate', 
            'sentiment', 'log_len', 'year', 'start_year_proxy', 
            'prob_low_playtime', 'is_early_access'
        ]
        
    def fit(self, df):
        print("\nFeatureEngineer: Learning Priors...")
        self.global_mean = df['hours_transformed'].mean()
        self.bouncer_clf.fit(df)
        self.item_stats = df.groupby('gameID')['hours_transformed'].mean().to_dict()
        temp = df.copy()
        temp['item_mean'] = temp['gameID'].map(self.item_stats).fillna(self.global_mean)
        temp['residual'] = temp['hours_transformed'] - temp['item_mean']
        user_sums = temp.groupby('userID')['residual'].sum()
        user_counts = temp.groupby('userID')['residual'].count()
        self.user_stats = (user_sums / (user_counts + self.alpha)).to_dict()
        if 'hours' in df.columns:
            self.user_raw_means = df.groupby('userID')['hours'].mean().to_dict()
            self.global_raw_mean = df['hours'].mean()
            self.user_rates = {u: 1.0/(m+1.0) for u, m in self.user_raw_means.items()}
            item_raw_means = df.groupby('gameID')['hours'].mean()
            self.item_rates = (1.0 / (item_raw_means + 1.0)).to_dict()

    def transform(self, df):
        df_new = df.copy()
        df_new['prob_low_playtime'] = self.bouncer_clf.predict_proba(df_new)
        df_new['item_mean_stat'] = df_new['gameID'].map(self.item_stats).fillna(self.global_mean)
        df_new['user_bias_stat'] = df_new['userID'].map(self.user_stats).fillna(0.0)
        global_rate = 1.0 / (self.global_raw_mean + 1.0) if self.global_raw_mean else 0.01
        df_new['user_rate'] = df_new['userID'].map(self.user_rates).fillna(global_rate)
        df_new['item_rate'] = df_new['gameID'].map(self.item_rates).fillna(global_rate)
        
        texts = df_new['text'].fillna("").astype(str) if 'text' in df_new.columns else pd.Series([""]*len(df_new))
        df_new['sentiment'] = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
        df_new['log_len'] = np.log1p(texts.str.len())
        
        if 'date' in df_new.columns:
            dates = pd.to_datetime(df_new['date'], errors='coerce')
            df_new['year'] = dates.dt.year.fillna(2015)
            user_avg_hours = df_new['userID'].map(self.user_raw_means).fillna(self.global_raw_mean)
            time_shift = pd.to_timedelta(user_avg_hours, unit='h')
            approx_start_date = dates - time_shift
            df_new['start_year_proxy'] = approx_start_date.dt.year.fillna(2015)
        else:
            df_new['year'] = 2015
            df_new['start_year_proxy'] = 2015
        
        df_new['is_early_access'] = df_new['early_access'].astype(float).fillna(0) if 'early_access' in df_new.columns else 0.0
        return df_new[self.signal_cols]

# =============================================================================
# 3. HYBRID SCORER (OPTIMIZED SUBSET SELECTION)
# =============================================================================

class HybridScorer:
    def __init__(self, n_factors=50, k_neighbors=20, max_features=500, nmf_components=30):
        self.n_factors = n_factors
        self.k = k_neighbors
        self.max_features = max_features
        self.nmf_components = nmf_components
        
        self.fe = FeatureEngineer(user_reg_alpha=12)
        self.temp_svd = TemporalSVD(n_factors=20, n_epochs=25, lr=0.005)
        
        # Meta-Model
        self.meta_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=6, l2_regularization=1.0, random_state=42
        )
        self.active_models = None # Indices of models to use
        self.user_map = {}
        self.item_map = {}
        
    def fit(self, df):
        print("="*60)
        print("TRAINING HYBRID SYSTEM (v11 - Optimization Mode)")
        print("="*60)
        self.global_mean = df['hours_transformed'].mean()
        self.fe.fit(df)
        self.temp_svd.fit(df)
        
        self.users_arr = df['userID'].unique()
        self.items_arr = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(self.users_arr)}
        self.item_map = {g: i for i, g in enumerate(self.items_arr)}
        n_users, n_items = len(self.users_arr), len(self.items_arr)
        
        rows = df['userID'].map(self.user_map).values
        cols = df['gameID'].map(self.item_map).values
        data = df['hours_transformed'].values
        self.R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        print("   Training SVD...")
        item_sums = np.array(self.R.sum(axis=0)).flatten()
        item_counts = np.diff(self.R.tocsc().indptr)
        self.item_means_arr = np.zeros(n_items)
        mask = item_counts > 0
        self.item_means_arr[mask] = item_sums[mask] / item_counts[mask]
        R_centered = self.R.copy()
        R_centered.data -= self.item_means_arr[cols]
        u, s, vt = svds(R_centered, k=self.n_factors)
        self.u_factors = u @ np.diag(s); self.vt_factors = vt
        
        print("   Training User-KNN...")
        self.knn_user = NearestNeighbors(n_neighbors=self.k, metric='cosine', n_jobs=-1)
        self.knn_user.fit(self.R)
        
        print("   Training Item-Cosine...")
        self.sim_ii = cosine_similarity(self.R.T, dense_output=False)
        
        print("   Training TF-IDF...")
        df_clean = df.copy()
        df_clean['text'] = df_clean['text'].fillna('')
        self.tfidf = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        item_corpus = [df_clean[df_clean['gameID']==g]['text'].str.cat(sep=' ') for g in self.items_arr]
        self.item_profiles = self.tfidf.fit_transform(item_corpus)
        user_corpus = [df_clean[df_clean['userID']==u]['text'].str.cat(sep=' ') for u in self.users_arr]
        self.user_profiles = self.tfidf.transform(user_corpus)
        
        print("   Training Poisson NMF...")
        try:
            nmf = NMF(n_components=self.nmf_components, init='nndsvda', solver='mu', beta_loss='kullback-leibler', max_iter=200, random_state=42)
            self.nmf_user = nmf.fit_transform(self.R); self.nmf_item = nmf.components_.T
        except:
            nmf = NMF(n_components=self.nmf_components, init='nndsvd', random_state=42)
            self.nmf_user = nmf.fit_transform(self.R.toarray()); self.nmf_item = nmf.components_.T

    def _get_individual_preds(self, u_idx, i_idx, u_id, g_id, date_val):
        preds = []
        # 1. SVD
        if u_idx is not None and i_idx is not None:
            preds.append(self.item_means_arr[i_idx] + np.dot(self.u_factors[u_idx], self.vt_factors[:, i_idx]))
        else: preds.append(self.global_mean)
        # 2. KNN
        if u_idx is not None and i_idx is not None:
            dists, idxs = self.knn_user.kneighbors(self.R[u_idx], n_neighbors=self.k)
            sims = 1 - dists.flatten()
            ratings = self.R[idxs.flatten(), i_idx].toarray().flatten()
            if ratings.sum() > 0: preds.append(np.dot(sims, ratings) / (sims.sum() + 1e-9))
            else: preds.append(self.global_mean)
        else: preds.append(self.global_mean)
        # 3. ItemSim
        if u_idx is not None and i_idx is not None:
            u_vec = self.R[u_idx]
            if u_vec.nnz > 0:
                sim_row = self.sim_ii[i_idx]
                pC = (u_vec @ sim_row.T).toarray()[0][0] / (np.abs(sim_row).sum() + 1e-9)
                if pC == 0: pC = self.global_mean
                preds.append(pC)
            else: preds.append(self.global_mean)
        else: preds.append(self.global_mean)
        # 4. Content
        if u_idx is not None and i_idx is not None:
            sim = (self.user_profiles[u_idx] @ self.item_profiles[i_idx].T).toarray()[0][0]
            preds.append(sim * 10.0)
        else: preds.append(0.0)
        # 5. NMF
        if u_idx is not None and i_idx is not None:
            preds.append(np.dot(self.nmf_user[u_idx], self.nmf_item[i_idx]))
        else: preds.append(self.global_mean)
        # 6. Temporal SGD
        preds.append(self.temp_svd.predict_one(u_id, g_id, date_val))
        
        return preds

    def _get_raw_matrices(self, df):
        """Returns separated [X_models, X_signals] for optimization."""
        X_models = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Predicting Base Models"):
            u_idx = self.user_map.get(getattr(row, 'userID'))
            i_idx = self.item_map.get(getattr(row, 'gameID'))
            X_models.append(self._get_individual_preds(u_idx, i_idx, getattr(row, 'userID'), getattr(row, 'gameID'), getattr(row, 'date', 0)))
        
        df_signals = self.fe.transform(df)
        return np.array(X_models), df_signals.values

    def find_best_model_subset(self, val_df):
        print("\n" + "="*60)
        print("EXHAUSTIVE SUBSET SEARCH (Models A-F)")
        print("="*60)
        
        X_models_all, X_signals = self._get_raw_matrices(val_df)
        y_val = val_df['hours_transformed'].values
        
        model_names = ['SVD', 'UserKNN', 'ItemKNN', 'Content', 'NMF', 'TimeSVD']
        n_models = len(model_names)
        
        best_mse = float('inf')
        best_subset = list(range(n_models))
        
        # Iterate 1 to n_models
        for r in range(1, n_models + 1):
            for subset in combinations(range(n_models), r):
                subset = list(subset)
                
                # Stack: [Selected Models | Signals]
                # Signals are ALWAYS included. We only permute base models.
                X_sub = np.hstack([X_models_all[:, subset], X_signals])
                
                # Train small booster
                # We use slightly fewer trees for speed during search
                clf = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
                clf.fit(X_sub, y_val)
                mse = mean_squared_error(y_val, clf.predict(X_sub))
                
                if mse < best_mse:
                    best_mse = mse
                    best_subset = subset
                    names = [model_names[i] for i in subset]
                    print(f"New Best: {names} -> MSE: {mse:.4f}")
        
        self.active_models = best_subset
        print(f"\n🏆 WINNER: {[model_names[i] for i in best_subset]}")
        print(f"   MSE: {best_mse:.4f}")
        return best_subset

    def apply_best_subset(self, subset):
        self.active_models = subset

    def learn_weights(self, val_df):
        # Optimized retrain with best subset and full parameters
        print("\nRetraining Meta-Model on Best Subset...")
        X_models_all, X_signals = self._get_raw_matrices(val_df)
        
        if self.active_models is not None:
            X_models = X_models_all[:, self.active_models]
        else:
            X_models = X_models_all
            
        X_stack = np.hstack([X_models, X_signals])
        y_val = val_df['hours_transformed'].values
        
        self.meta_model.fit(X_stack, y_val)
        mse = mean_squared_error(y_val, self.meta_model.predict(X_stack))
        print(f"Final Validation MSE: {mse:.4f}")

    def make_test_predictions(self, test_df, output_path='predictions_Hours.csv'):
        print(f"\nGenerating predictions...")
        X_models_all, X_signals = self._get_raw_matrices(test_df)
        
        if self.active_models is not None:
            X_models = X_models_all[:, self.active_models]
        else:
            X_models = X_models_all
            
        X_stack = np.hstack([X_models, X_signals])
        preds = self.meta_model.predict(X_stack)
        preds = np.clip(preds, 0, 14.5)
        
        sub = test_df[['userID', 'gameID']].copy()
        sub['prediction'] = preds
        sub.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")

# =============================================================================
# RUN PIPELINE
# =============================================================================

def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d['userID'], d.get('gameID'), d

if __name__ == "__main__":
    print("Loading data...")
    try:
        data = [d for u, g, d in readJSON("train.json.gz")]
        df_all = pd.DataFrame(data)
        
        # 90/10 Split for Optimization
        mask = np.random.rand(len(df_all)) < 0.90
        df_train = df_all[mask].copy()
        df_val = df_all[~mask].copy()
        
        scorer = HybridScorer(n_factors=50, nmf_components=30)
        scorer.fit(df_train)
        
        # 1. Find Best Subset
        best_subset = scorer.find_best_model_subset(df_val)
        
        # 2. Retrain weights
        scorer.apply_best_subset(best_subset)
        scorer.learn_weights(df_val)
        
        # 3. Predict
        pairs = pd.read_csv("pairs_Hours.csv")
        scorer.make_test_predictions(pairs)
        
    except FileNotFoundError:
        print("Error: train.json.gz not found.")
