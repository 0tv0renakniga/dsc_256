import numpy as np
import pandas as pd
import gzip
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. BASELINE: OPTIMIZED SVD
# =============================================================================
class OptimizedSVD:
    def __init__(self, n_factors=40, n_epochs=30, lr=0.005, reg=0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.mu = 0
        self.bu = None; self.bi = None
        self.Pu = None; self.Qi = None
        self.user_map = {}; self.item_map = {}
        
    def fit(self, df):
        users = df['userID'].unique()
        items = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        n_users, n_items = len(users), len(items)
        
        self.mu = df['hours_transformed'].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.Pu = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Qi = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        u_indices = df['userID'].map(self.user_map).values
        i_indices = df['gameID'].map(self.item_map).values
        ratings = df['hours_transformed'].values
        
        print(f"   [SVD] Training {self.n_factors} factors for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            current_lr = self.lr * (0.95 ** epoch)
            for idx in indices:
                u, i, r = u_indices[idx], i_indices[idx], ratings[idx]
                dot = np.dot(self.Pu[u], self.Qi[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot
                err = r - pred
                self.bu[u] += current_lr * (err - self.reg * self.bu[u])
                self.bi[i] += current_lr * (err - self.reg * self.bi[i])
                pu_old = self.Pu[u].copy()
                self.Pu[u] += current_lr * (err * self.Qi[i] - self.reg * self.Pu[u])
                self.Qi[i] += current_lr * (err * pu_old - self.reg * self.Qi[i])

    def predict(self, df):
        preds = []
        u_map = self.user_map; i_map = self.item_map
        for idx, row in df.iterrows():
            u_id = row['userID']; g_id = row['gameID']
            u = u_map.get(u_id); i = i_map.get(g_id)
            if u is not None and i is not None:
                pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.Pu[u], self.Qi[i])
            elif u is not None: pred = self.mu + self.bu[u]
            elif i is not None: pred = self.mu + self.bi[i]
            else: pred = self.mu
            preds.append(pred)
        return np.array(preds)

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def calculate_features(train_df, full_df):
    """
    Calculates User/Game statistical features based on TRAINING data.
    """
    print("   [Features] Computing Volatility and Deviation stats...")
    
    # 1. Volatility (Std Dev)
    u_std = train_df.groupby('userID')['hours_transformed'].std().fillna(0)
    g_std = train_df.groupby('gameID')['hours_transformed'].std().fillna(0)
    
    # 2. Popularity / Activity Percentiles (Global Context)
    # Use full_df for best percentile estimates
    u_pct = full_df.groupby('userID')['gameID'].count().rank(pct=True)
    g_pct = full_df.groupby('gameID')['userID'].count().rank(pct=True)
    
    # 3. Peer Deviation (The "Hipster" Score)
    # Avg absolute difference between User's rating and the Game's average rating
    g_avg = train_df.groupby('gameID')['hours_transformed'].mean()
    
    # We do this calculation on train_df
    train_w_avg = train_df.merge(g_avg.rename('g_avg'), on='gameID', how='left')
    train_w_avg['diff'] = (train_w_avg['hours_transformed'] - train_w_avg['g_avg']).abs()
    u_dev = train_w_avg.groupby('userID')['diff'].mean().fillna(0)
    
    return {
        'u_std': u_std, 'g_std': g_std,
        'u_pct': u_pct, 'g_pct': g_pct,
        'u_dev': u_dev
    }

def calculate_historical_mrr(model, train_df):
    """
    Calculates how "predictable" a user is based on SVD's performance on the TRAINING set.
    """
    print("   [Features] Computing Historical User MRR...")
    # Predict on training data (self-check)
    train_preds = train_df.copy()
    train_preds['pred'] = model.predict(train_preds)
    
    user_mrr = {}
    
    # For each user, rank their training items
    for uid, group in train_preds.groupby('userID'):
        # Ground Truth: Items above user's own average
        threshold = group['hours_transformed'].mean()
        relevant = set(group[group['hours_transformed'] >= threshold]['gameID'])
        
        if len(relevant) == 0:
            user_mrr[uid] = 0.0
            continue
            
        # SVD Ranking
        ranked_items = group.sort_values('pred', ascending=False)['gameID'].values
        
        mrr = 0
        for rank, item in enumerate(ranked_items):
            if item in relevant:
                mrr = 1.0 / (rank + 1)
                break
        user_mrr[uid] = mrr
        
    return pd.Series(user_mrr)

def add_batch_rank_feature(df, svd_preds):
    """
    Adds a feature representing the Rank of the item within the specific batch (Val or Test).
    "Is this the #1 predicted game for this user in this dataset?"
    """
    temp = df.copy()
    temp['svd_pred'] = svd_preds
    
    # Rank descending (Higher score = Rank 1)
    # We normalize rank by batch size (0.0 to 1.0) to handle different batch sizes
    temp['batch_rank'] = temp.groupby('userID')['svd_pred'].rank(ascending=False, pct=True)
    
    return temp['batch_rank'].values

# =============================================================================
# 3. HARPOON TRAINING
# =============================================================================
def train_feature_weighted_harpoon(train_df, val_df, test_df, svd_model, full_df):
    print("\n--- Training Feature-Weighted Harpoon ---")
    
    # 1. Compute Base SVD Predictions
    train_base = svd_model.predict(train_df)
    val_base = svd_model.predict(val_df)
    test_base = svd_model.predict(test_df)
    
    # 2. Compute Static Features (from Train)
    stats = calculate_features(train_df, full_df)
    u_mrr_map = calculate_historical_mrr(svd_model, train_df)
    
    # 3. Feature Assembly Function
    def get_X(df, base_preds):
        # A. Map Static Features
        # FillNa logic: New users get global median/mean
        f_u_std = df['userID'].map(stats['u_std']).fillna(stats['u_std'].mean())
        f_g_std = df['gameID'].map(stats['g_std']).fillna(stats['g_std'].mean())
        f_u_pct = df['userID'].map(stats['u_pct']).fillna(0.5)
        f_g_pct = df['gameID'].map(stats['g_pct']).fillna(0.5)
        f_u_dev = df['userID'].map(stats['u_dev']).fillna(stats['u_dev'].mean())
        f_u_mrr = df['userID'].map(u_mrr_map).fillna(0.5) # Unknown users = average predictability
        
        # B. Dynamic Batch Ranking (The "Test Rank" Feature)
        # Calculate rank of this item within the current user's batch based on SVD score
        f_rank = add_batch_rank_feature(df, base_preds)
        
        X = pd.DataFrame({
            'u_std': f_u_std,
            'g_std': f_g_std,
            'u_pct': f_u_pct,
            'g_pct': f_g_pct,
            'u_dev': f_u_dev,
            'u_mrr': f_u_mrr,
            'batch_rank': f_rank
        })
        return X

    print("   [Harpoon] Building Matrices...")
    X_train = get_X(train_df, train_base)
    X_val = get_X(val_df, val_base)
    X_test = get_X(test_df, test_base)
    
    # Standardize Features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    
    # 4. Train Ridge on Residuals
    y_resid = train_df['hours_transformed'] - train_base
    
    print("   [Harpoon] Fitting Ridge...")
    ridge = Ridge(alpha=100.0) # Strong regularization for stability
    ridge.fit(X_train_sc, y_resid)
    
    # Show weights
    print("\n   [Harpoon Feature Importance]")
    for name, coef in zip(X_train.columns, ridge.coef_):
        print(f"      {name:<15}: {coef:.4f}")
        
    # 5. Predict
    print("\n   [Harpoon] Applying Corrections...")
    val_corr = ridge.predict(X_val_sc)
    test_corr = ridge.predict(X_test_sc)
    
    val_final = np.clip(val_base + val_corr, 0, 16)
    test_final = np.clip(test_base + test_corr, 0, 15)
    
    val_mse = np.mean((val_df['hours_transformed'] - val_final)**2)
    print(f"   >>> Hybrid Validation MSE: {val_mse:.4f}")
    
    return test_final

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d['userID'], d.get('gameID'), d

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    raw = []
    try:
        for u, g, d in readJSON("train.json.gz"):
            raw.append(d)
        df = pd.DataFrame(raw)
    except:
        print("Using dummy data (File not found)...")
        df = pd.DataFrame({
            'userID': np.random.choice(['u1','u2'], 100),
            'gameID': np.random.choice(['g1','g2'], 100),
            'hours': np.random.rand(100)*10
        })
        
    if 'hours_transformed' not in df.columns:
        if 'hours' in df.columns:
            df['hours_transformed'] = np.log2(df['hours'] + 1)
        else:
             df['hours_transformed'] = df['hours'] # Fallback
            
    try:
        pairs = pd.read_csv("pairs_Hours.csv")
    except:
        pairs = pd.DataFrame(columns=['userID','gameID'])

    # 2. Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.90)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    # Warm Validation
    known_u = set(train_df['userID'])
    known_i = set(train_df['gameID'])
    val_warm = val_df[val_df['userID'].isin(known_u) & val_df['gameID'].isin(known_i)].copy()

    # 3. Train Base SVD
    print("\n[Phase 1] Training SVD Baseline...")
    svd = OptimizedSVD(n_factors=40, reg=0.01)
    svd.fit(train_df)
    
    # 4. Train Harpoon & Predict
    print("\n[Phase 2] Training Feature-Weighted Harpoon...")
    final_preds = train_feature_weighted_harpoon(train_df, val_warm, pairs, svd, df)
    
    # 5. Save
    pairs['prediction'] = final_preds
    pairs.to_csv("predictions_Hours_FeatureWeighted.csv", index=False)
    print("\n-> Saved 'predictions_Hours_FeatureWeighted.csv'")
