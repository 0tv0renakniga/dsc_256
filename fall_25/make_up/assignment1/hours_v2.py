import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import copy

# =============================================================================
# 1. ROBUST SVD (Huber Loss + Slow Learning)
# =============================================================================

class RobustSVD:
    def __init__(self, n_factors=10, n_epochs=50, lr=0.002, reg=0.05, huber_delta=1.5):
        """
        SVD with Huber Loss to handle outliers without hard clipping.
        
        Parameters:
        - huber_delta: The threshold where 'MSE' switches to 'MAE'.
                       Errors > 1.5 log-hours are treated as 'outliers' 
                       and get linear updates (safe), not quadratic (explosive).
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.delta = huber_delta # The "Shock Absorber" threshold
        
        self.mu = 0
        self.bu = None; self.bi = None
        self.Pu = None; self.Qi = None
        self.user_map = {}; self.item_map = {}
        
    def fit(self, df, verbose=False):
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
        
        u_indices = df['userID'].map(self.user_map).values
        i_indices = df['gameID'].map(self.item_map).values
        ratings = df['hours_transformed'].values
        
        print(f"   Training RobustSVD (Factors={self.n_factors}, LR={self.lr}, Huber={self.delta})...")
        
        for epoch in range(self.n_epochs):
            # Shuffle ("Chunks" logic handles implicitly via random permutation)
            indices = np.random.permutation(len(ratings))
            
            # Slower Decay (0.98 instead of 0.95) to keep learning alive longer
            current_lr = self.lr * (0.98 ** epoch)
            epoch_loss = 0
            
            for idx in indices:
                u, i, r = u_indices[idx], i_indices[idx], ratings[idx]
                
                # Predict
                dot = np.dot(self.Pu[u], self.Qi[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot
                
                # Raw Error
                err = r - pred
                
                # --- HUBER GRADIENT LOGIC ---
                # Instead of letting err explode (e.g. -10 -> 100 gradient),
                # We cap the gradient magnitude mathematically.
                if abs(err) <= self.delta:
                    # Normal MSE update for small errors
                    grad = err
                    epoch_loss += err**2
                else:
                    # Linear update for huge errors (The "Safe" Path)
                    # Sign of error * delta
                    grad = self.delta * np.sign(err)
                    # Huber loss calculation for reporting
                    epoch_loss += self.delta * (abs(err) - 0.5 * self.delta)

                # Updates using the SAFE gradient
                self.bu[u] += current_lr * (grad - self.reg * self.bu[u])
                self.bi[i] += current_lr * (grad - self.reg * self.bi[i])
                
                pu_old = self.Pu[u].copy()
                self.Pu[u] += current_lr * (grad * self.Qi[i] - self.reg * self.Pu[u])
                self.Qi[i] += current_lr * (grad * pu_old - self.reg * self.Qi[i])
                
            # Report RMSE (converted back from Huber mostly for tracking)
            if verbose and (epoch+1) % 5 == 0:
                print(f"      Epoch {epoch+1}: Approx Loss = {epoch_loss/len(ratings):.4f}")

    def predict(self, df, return_details=False):
        preds = []
        details = []
        u_map = self.user_map; i_map = self.item_map
        
        for idx, row in df.iterrows():
            u_id = row['userID']; g_id = row['gameID']
            u = u_map.get(u_id); i = i_map.get(g_id)
            
            mu=self.mu; bu=0.0; bi=0.0; dot=0.0
            
            if u is not None and i is not None:
                bu=self.bu[u]; bi=self.bi[i]; dot=np.dot(self.Pu[u], self.Qi[i])
            elif u is not None: bu=self.bu[u]
            elif i is not None: bi=self.bi[i]
                
            pred = mu + bu + bi + dot
            preds.append(pred)
            
            if return_details:
                details.append({'mu':mu, 'b_u':bu, 'b_i':bi, 'dot':dot})
                
        if return_details: return np.array(preds), pd.DataFrame(details)
        return np.array(preds)

# =============================================================================
# 2. PIPELINE
# =============================================================================

def mimic_test_split(df, seed):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.90)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

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
    
    best_mse = float('inf')
    best_model = None
    best_fold = -1
    best_breakdown = None
    
    print("\nRunning 5-Fold Robust Loop...")
    
    for k in range(5):
        seed = 100 + k # New seeds
        train_df, val_df = mimic_test_split(df, seed)
        print(f"\n--- Fold {k+1} ---")
        
        # Slower, Longer, Safer Training
        model = RobustSVD(n_factors=10, n_epochs=50, lr=0.002, reg=0.05, huber_delta=1.5)
        model.fit(train_df, verbose=True)
        
        preds, details = model.predict(val_df, return_details=True)
        # We still clip OUTPUT for valid MSE calculation, but training wasn't clipped
        preds_clipped = np.clip(preds, 0, 15)
        
        mse = np.mean((val_df['hours_transformed'] - preds_clipped)**2)
        print(f"   Fold MSE: {mse:.4f}")
        
        if mse < best_mse:
            print(f"   >>> NEW BEST! ({mse:.4f})")
            best_mse = mse
            best_model = copy.deepcopy(model)
            best_fold = k + 1
            
            # Save breakdown for analysis
            val_out = val_df.copy().reset_index(drop=True)
            val_out['predicted'] = preds_clipped
            val_out['actual'] = val_out['hours_transformed']
            val_out['error'] = val_out['actual'] - val_out['predicted']
            val_out['abs_error'] = val_out['error'].abs()
            if 'text' in val_out.columns:
                val_out['text_preview'] = val_out['text'].fillna("").astype(str).str.slice(0,50)
            best_breakdown = pd.concat([val_out, details], axis=1)

    print("\n" + "="*60)
    print(f"WINNER: Fold {best_fold} with MSE {best_mse:.4f}")
    print("="*60)
    
    # Save Breakdown
    if best_breakdown is not None:
        best_breakdown.sort_values('abs_error', ascending=False, inplace=True)
        best_breakdown.to_csv("validation_breakdown_robust.csv", index=False)
        print("-> Saved 'validation_breakdown_robust.csv'")
        
    # Predict
    print(f"\nPredicting with Fold {best_fold} model...")
    test_preds = best_model.predict(pairs)
    test_preds = np.clip(test_preds, 0, 14.5)
    
    pairs['prediction'] = test_preds
    pairs.to_csv("predictions_Hours.csv", index=False)
    print("-> Saved 'predictions_Hours.csv'")
