import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import copy

# =============================================================================
# 1. THE CHAMPION MODEL: SIMPLE SVD (With Component Breakdown)
# =============================================================================

class SimpleSVD:
    def __init__(self, n_factors=10, n_epochs=30, lr=0.005, reg=0.05):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
        self.mu = 0
        self.bu = None
        self.bi = None
        self.Pu = None
        self.Qi = None
        
        self.user_map = {}
        self.item_map = {}
        
    def fit(self, df, verbose=False):
        # Mappings
        users = df['userID'].unique()
        items = df['gameID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {g: i for i, g in enumerate(items)}
        n_users = len(users)
        n_items = len(items)
        
        # Init Weights
        self.mu = df['hours_transformed'].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.Pu = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Qi = np.random.normal(0, 0.01, (n_items, self.n_factors))
        
        # Fast Access Arrays
        u_indices = df['userID'].map(self.user_map).values
        i_indices = df['gameID'].map(self.item_map).values
        ratings = df['hours_transformed'].values
        
        # SGD Loop
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(ratings))
            current_lr = self.lr * (0.95 ** epoch)
            epoch_loss = 0
            
            for idx in indices:
                u, i, r = u_indices[idx], i_indices[idx], ratings[idx]
                
                # Predict
                dot = np.dot(self.Pu[u], self.Qi[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot
                
                err = r - pred
                epoch_loss += err**2
                
                # Updates
                self.bu[u] += current_lr * (err - self.reg * self.bu[u])
                self.bi[i] += current_lr * (err - self.reg * self.bi[i])
                
                pu_old = self.Pu[u].copy()
                self.Pu[u] += current_lr * (err * self.Qi[i] - self.reg * self.Pu[u])
                self.Qi[i] += current_lr * (err * pu_old - self.reg * self.Qi[i])
                
            rmse = np.sqrt(epoch_loss / len(ratings))
            if verbose and (epoch+1) % 10 == 0:
                print(f"      Epoch {epoch+1}: Train RMSE = {rmse:.4f}")

    def predict(self, df, return_details=False):
        preds = []
        details = []
        
        u_map_get = self.user_map.get
        i_map_get = self.item_map.get
        
        for idx, row in df.iterrows():
            u_id = row['userID']
            g_id = row['gameID']
            
            u = u_map_get(u_id)
            i = i_map_get(g_id)
            
            # Components
            comp_mu = self.mu
            comp_bu = 0.0
            comp_bi = 0.0
            comp_dot = 0.0
            
            if u is not None and i is not None:
                # Warm Start
                comp_bu = self.bu[u]
                comp_bi = self.bi[i]
                comp_dot = np.dot(self.Pu[u], self.Qi[i])
            elif u is not None:
                # User Only
                comp_bu = self.bu[u]
            elif i is not None:
                # Item Only
                comp_bi = self.bi[i]
            
            pred = comp_mu + comp_bu + comp_bi + comp_dot
            preds.append(pred)
            
            if return_details:
                details.append({
                    'mu': comp_mu,
                    'b_u': comp_bu,
                    'b_i': comp_bi,
                    'dot': comp_dot
                })
        
        if return_details:
            return np.array(preds), pd.DataFrame(details)
        return np.array(preds)

# =============================================================================
# 2. UTILS
# =============================================================================

def mimic_test_split(df, seed):
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 90/10 Split
    split_idx = int(len(df) * 0.90)
    train = df.iloc[:split_idx].copy()
    val_cand = df.iloc[split_idx:].copy()
    
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

# =============================================================================
# MAIN K-FOLD LOOP
# =============================================================================

if __name__ == "__main__":
    print("Loading Data...")
    try:
        data = [d for u, g, d in readJSON("train.json.gz")]
        df = pd.DataFrame(data)
    except:
        df = pd.read_csv("train_data_1.txt")
    
    pairs = pd.read_csv("pairs_Hours.csv")
    
    best_mse = float('inf')
    best_model = None
    best_fold_idx = -1
    best_val_df_detailed = None # To store the juicy details
    
    print(f"\nRunning 5-Fold Validation Loop (SimpleSVD)...")
    
    for k in range(5):
        seed = 42 + k
        train_df, val_df = mimic_test_split(df, seed=seed)
        
        print(f"\n--- Fold {k+1} (Seed {seed}) ---")
        print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Train
        model = SimpleSVD(n_factors=10, n_epochs=30, lr=0.005, reg=0.05)
        model.fit(train_df, verbose=True)
        
        # Validate (With Details)
        preds, details = model.predict(val_df, return_details=True)
        preds = np.clip(preds, 0, 15)
        
        mse = np.mean((val_df['hours_transformed'] - preds)**2)
        print(f"   Fold MSE: {mse:.4f}")
        
        if mse < best_mse:
            print(f"   >>> NEW BEST MODEL! (Previous: {best_mse:.4f})")
            best_mse = mse
            best_model = copy.deepcopy(model)
            best_fold_idx = k + 1
            
            # Prepare the Detailed CSV for the winner
            val_out = val_df.copy().reset_index(drop=True)
            val_out['predicted'] = preds
            val_out['error'] = val_out['hours_transformed'] - preds
            val_out['abs_error'] = val_out['error'].abs()
            # Add text preview if available
            if 'text' in val_out.columns:
                val_out['text_preview'] = val_out['text'].fillna("").astype(str).str.slice(0, 50)
            
            # Combine with decomposition details
            best_val_df_detailed = pd.concat([val_out, details], axis=1)

    print("\n" + "="*60)
    print(f"WINNER: Fold {best_fold_idx} with MSE {best_mse:.4f}")
    print("="*60)
    
    # 1. Save The Juicy Details (Best Fold)
    # Sort by worst errors so you see the problems first
    if best_val_df_detailed is not None:
        out_cols = ['userID', 'gameID', 'hours_transformed', 'predicted', 'error', 
                    'mu', 'b_u', 'b_i', 'dot', 'text_preview']
        out_cols = [c for c in out_cols if c in best_val_df_detailed.columns]
        
        best_val_df_detailed.sort_values('abs_error', ascending=False, inplace=True)
        best_val_df_detailed.to_csv("validation_breakdown_best_fold.csv", index=False, columns=out_cols)
        print("-> Saved 'validation_breakdown_best_fold.csv'")
    
    # 2. Predict on Test with Best Model
    print(f"\nGenerating predictions using Model from Fold {best_fold_idx}...")
    test_preds = best_model.predict(pairs)
    test_preds = np.clip(test_preds, 0, 14.5)
    
    pairs['prediction'] = test_preds
    pairs.to_csv("predictions_Hours.csv", index=False)
    print("-> Saved 'predictions_Hours.csv'")
