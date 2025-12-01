import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import copy

# =============================================================================
# 1. THE WINNING ARCHITECTURE: SIMPLE SVD
# =============================================================================

class SimpleSVD:
    def __init__(self, n_factors=10, n_epochs=30, lr=0.005, reg=0.05):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.mu = 0
        self.bu = None; self.bi = None
        self.Pu = None; self.Qi = None
        self.user_map = {}; self.item_map = {}
        
    def fit(self, df):
        # Mappings
        users = df['userID'].unique()
        items = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        n_users, n_items = len(users), len(items)
        
        # Init
        self.mu = df['hours_transformed'].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.Pu = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Qi = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        # Fast Arrays
        u_indices = df['userID'].map(self.user_map).values
        i_indices = df['gameID'].map(self.item_map).values
        ratings = df['hours_transformed'].values
        
        # Training
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            current_lr = self.lr * (0.95 ** epoch)
            
            for idx in indices:
                u, i, r = u_indices[idx], i_indices[idx], ratings[idx]
                dot = np.dot(self.Pu[u], self.Qi[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot
                err = r - pred
                
                # Standard updates
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
# 2. ENSEMBLE PIPELINE (BAGGING)
# =============================================================================

def mimic_test_split(df, seed):
    # 90/10 Split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.90)
    train = df.iloc[:split].copy()
    val_cand = df.iloc[split:].copy()
    
    # Filter Warm
    train_users = set(train['userID'])
    train_items = set(train['gameID'])
    val_warm = val_cand[
        val_cand['userID'].isin(train_users) & 
        val_cand['gameID'].isin(train_items)
    ].copy()
    return train, val_warm

def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d['userID'], d.get('gameID'), d

if __name__ == "__main__":
    print("Loading Data...")
    try:
        data = [d for u, g, d in readJSON("train.json.gz")]
        df = pd.DataFrame(data)
    except:
        df = pd.read_csv("train_data_1.txt")
        
    pairs = pd.read_csv("pairs_Hours.csv")
    
    n_folds = 5
    ensemble_preds = np.zeros(len(pairs))
    val_mse_scores = []
    
    print(f"\nTraining {n_folds}-Fold Ensemble...")
    
    for k in range(n_folds):
        seed = 42 + k
        train_df, val_df = mimic_test_split(df, seed)
        print(f"\n--- Fold {k+1}/{n_folds} ---")
        
        # Train
        model = SimpleSVD(n_factors=10, n_epochs=30, lr=0.005, reg=0.05)
        model.fit(train_df)
        
        # Validate (Just to check health)
        val_p = model.predict(val_df)
        val_mse = np.mean((val_df['hours_transformed'] - np.clip(val_p, 0, 15))**2)
        val_mse_scores.append(val_mse)
        print(f"   Val MSE: {val_mse:.4f}")
        
        # Predict on Test (Accumulate)
        test_p = model.predict(pairs)
        ensemble_preds += test_p
        
    # Average
    ensemble_preds /= n_folds
    
    # Clip Final
    ensemble_preds = np.clip(ensemble_preds, 0, 14.5)
    
    print("\n" + "="*60)
    print(f"Ensemble Complete.")
    print(f"Avg Single-Model Val MSE: {np.mean(val_mse_scores):.4f}")
    print(f"Expected Ensemble Boost: ~0.05 - 0.10 MSE")
    print("="*60)
    
    pairs['prediction'] = ensemble_preds
    pairs.to_csv("predictions_Hours.csv", index=False)
    print("-> Saved 'predictions_Hours.csv'")
