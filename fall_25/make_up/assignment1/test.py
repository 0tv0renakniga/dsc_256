import numpy as np
import pandas as pd
import gzip
import itertools
import time

# =============================================================================
# 1. CORE ARCHITECTURE: SIMPLE SVD
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
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            current_lr = self.lr * (0.95 ** epoch)
            
            # Vectorized loop unrolling is hard in pure python, sticking to loop
            # for clarity and safety as per original "Winning Architecture"
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
# 2. UTILS
# =============================================================================
def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        yield d['userID'], d.get('gameID'), d

def get_percentile_maps(full_df):
    """Calculates global percentiles for analysis context."""
    user_counts = full_df.groupby('userID')['gameID'].count().rank(pct=True)
    item_counts = full_df.groupby('gameID')['userID'].count().rank(pct=True)
    return user_counts, item_counts

# =============================================================================
# 3. GRID SEARCH ENGINE
# =============================================================================
if __name__ == "__main__":
    # --- A. Load Data ---
    print("Loading Data...")
    try:
        data = [d for u, g, d in readJSON("train.json.gz")]
        df = pd.DataFrame(data)
    except:
        df = pd.read_csv("train_data_1.txt")
        # Ensure we have the target column
        if 'hours_transformed' not in df.columns and 'hours' in df.columns:
            df['hours_transformed'] = np.log2(df['hours'] + 1)

    # --- B. Create Single Validation Split (Fixed Seed) ---
    # We use a single split for grid search speed, rather than 5-fold
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.90)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    # Filter warm (only evaluate on known user/items for fairness)
    known_users = set(train_df['userID'])
    known_items = set(train_df['gameID'])
    val_warm = val_df[
        val_df['userID'].isin(known_users) & 
        val_df['gameID'].isin(known_items)
    ].copy()

    # Calculate Percentiles (for analysis)
    user_pct_map, item_pct_map = get_percentile_maps(df)
    val_warm['user_pct'] = val_warm['userID'].map(user_pct_map)
    val_warm['item_pct'] = val_warm['gameID'].map(item_pct_map)

    # --- C. Define Grid ---
    param_grid = {
        'n_factors': [10, 20, 40],
        'reg': [0.01, 0.05, 0.1],
        'lr': [0.001, 0.005],
        'n_epochs': [30] # Keep fixed to save time, or add [30, 50]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nStarting Grid Search over {len(combinations)} combinations...")
    print(f"Validation Set Size: {len(val_warm)}")
    print("-" * 60)
    print(f"{'Factors':<8} {'Reg':<6} {'LR':<6} | {'Global':<8} {'Pop(High)':<10} {'Pop(Low)':<10} {'Whale':<10}")
    print("-" * 60)

    results = []

    # --- D. Run Grid ---
    for i, params in enumerate(combinations):
        start_time = time.time()
        
        # 1. Train
        model = SimpleSVD(**params)
        model.fit(train_df)
        
        # 2. Predict
        preds = model.predict(val_warm)
        preds = np.clip(preds, 0, 15) # Clip to reasonable range
        
        # 3. Calculate Errors
        sq_err = (val_warm['hours_transformed'] - preds) ** 2
        
        # 4. Segment Metrics
        mse_global = sq_err.mean()
        
        # > Popularity Segments
        mask_pop_high = val_warm['item_pct'] > 0.8
        mask_pop_low = val_warm['item_pct'] <= 0.2
        mse_pop_high = sq_err[mask_pop_high].mean()
        mse_pop_low = sq_err[mask_pop_low].mean()
        
        # > User Activity Segments ("Whales")
        mask_user_high = val_warm['user_pct'] > 0.8
        mse_user_high = sq_err[mask_user_high].mean()
        
        # 5. Store & Print
        res_row = params.copy()
        res_row.update({
            'MSE_Global': mse_global,
            'MSE_Pop_High': mse_pop_high, 
            'MSE_Pop_Low': mse_pop_low,
            'MSE_Whale': mse_user_high
        })
        results.append(res_row)
        
        print(f"{params['n_factors']:<8} {params['reg']:<6} {params['lr']:<6} | "
              f"{mse_global:.4f}   {mse_pop_high:.4f}       {mse_pop_low:.4f}       {mse_user_high:.4f}")

    # --- E. Analyze Results ---
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('MSE_Global')
    
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS (Global MSE)")
    print("="*60)
    print(res_df.head(5).to_string(index=False))
    
    print("\n" + "="*60)
    print("BEST CONFIG FOR POPULAR GAMES (The SVD Weakness)")
    print("="*60)
    print(res_df.sort_values('MSE_Pop_High').head(1).to_string(index=False))

    res_df.to_csv("grid_search_results.csv", index=False)
    print("\n-> Saved full results to 'grid_search_results.csv'")
