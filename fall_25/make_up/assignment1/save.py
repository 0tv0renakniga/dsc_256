import gzip
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import random
from tqdm import tqdm

# 1. Data Loading
def readJSON():
    for l in gzip.open("train.json.gz", 'rt'):
        yield eval(l)

# 2. Graph Engineer (Best Feature Set)
class GraphFeatureEngineer:
    def __init__(self, train_pos_df):
        self.df = train_pos_df
        self.game_users = defaultdict(set)
        self.user_games = defaultdict(set)
        self.game_pop = self.df['gameID'].value_counts().to_dict()
        self.user_act = self.df['userID'].value_counts().to_dict()
        
        print("Building Graph...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            self.game_users[row['gameID']].add(row['userID'])
            self.user_games[row['userID']].add(row['gameID'])
            
        # Adamic/Adar Weights (Inverse Log Frequency)
        self.u_weights = {u: 1.0/np.log1p(c+1) for u, c in self.user_act.items()}
        
        # Time
        if 'unixReviewTime' in self.df.columns:
            self.has_time = True
            self.u_last = self.df.groupby('userID')['unixReviewTime'].max().to_dict()
            self.g_avg = self.df.groupby('gameID')['unixReviewTime'].mean().to_dict()
        else:
            self.has_time = False

    def enrich(self, df):
        print("Enriching...")
        df = df.copy()
        
        # Base Logs
        df['pop_log'] = df['gameID'].map(lambda x: np.log1p(self.game_pop.get(x, 0)))
        df['act_log'] = df['userID'].map(lambda x: np.log1p(self.user_act.get(x, 0)))
        
        # Adamic/Adar
        aa_scores = []
        for u, g in tqdm(zip(df['userID'], df['gameID']), total=len(df)):
            target = self.game_users.get(g, set())
            history = self.user_games.get(u, set())
            score = 0
            if target and history:
                for h_game in history:
                    if h_game == g: continue
                    h_users = self.game_users.get(h_game, set())
                    shared = target.intersection(h_users)
                    if shared:
                        # Sum of weights
                        score += sum(self.u_weights.get(v, 0) for v in shared)
            aa_scores.append(score)
        df['adamic_adar'] = aa_scores
        
        # Time Delta
        if self.has_time:
            u_t = df['userID'].map(self.u_last).fillna(0)
            g_t = df['gameID'].map(self.g_avg).fillna(0)
            df['time_delta'] = np.log1p((u_t - g_t).abs())
        else:
            df['time_delta'] = 0
            
        return df

# 3. Main
if __name__ == "__main__":
    print("Loading...")
    train_df = pd.DataFrame([d for d in readJSON()])
    
    # Fix Time if needed
    if 'unixReviewTime' not in train_df.columns and 'date' in train_df.columns:
         train_df['unixReviewTime'] = pd.to_datetime(train_df['date'], errors='coerce').astype(np.int64) // 10**9
    
    # Engineer
    eng = GraphFeatureEngineer(train_df)
    
    # Train Data (Hard Negatives)
    print("Sampling Negatives...")
    users = train_df['userID'].unique()
    all_games = list(train_df['gameID'].unique())
    pos_set = set(zip(train_df['userID'], train_df['gameID']))
    
    train_data = []
    # 5 Random Negatives per Positive
    for u in tqdm(users):
        u_games = train_df[train_df['userID'] == u]['gameID'].tolist()
        for g in u_games:
            train_data.append({'userID': u, 'gameID': g, 'label': 1})
        
        # Sampling
        target = 5 * len(u_games)
        cnt = 0
        while cnt < target:
            g = random.choice(all_games)
            if (u, g) not in pos_set:
                train_data.append({'userID': u, 'gameID': g, 'label': 0})
                cnt += 1
                
    train_enriched = eng.enrich(pd.DataFrame(train_data))
    
    # Fit LGBM Ranker
    print("Training...")
    features = ['pop_log', 'act_log', 'adamic_adar', 'time_delta']
    train_enriched = train_enriched.sort_values('userID')
    q_train = train_enriched.groupby('userID', sort=False).size().values
    
    ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', n_estimators=1000)
    ranker.fit(train_enriched[features], train_enriched['label'], group=q_train)
    
    # Predict
    print("Predicting...")
    pairs = pd.read_csv("pairs_Played.csv")
    test_enriched = eng.enrich(pairs)
    pairs['score'] = ranker.predict(test_enriched[features])
    
    # --- PER USER CALIBRATION ---
    print("Applying Per-User 50/50 Split...")
    final_preds = []
    for u, group in tqdm(pairs.groupby('userID')):
        # Sort by Score
        group = group.sort_values('score', ascending=False)
        # Top 50% = 1
        n = len(group)
        group['prediction'] = 0
        group.iloc[:n//2, group.columns.get_loc('prediction')] = 1
        final_preds.append(group)
        
    final_df = pd.concat(final_preds).sort_index()
    # Ensure alignment with original file by merging on keys if needed, 
    # but concat usually preserves data if we didn't drop rows.
    # Safer to just re-sort by index if we didn't reset index.
    
    # But groupby iteration might shuffle order.
    # Let's map back to original pairs
    pairs_orig = pd.read_csv("pairs_Played.csv")
    final_map = dict(zip(zip(final_df['userID'], final_df['gameID']), final_df['prediction']))
    
    pairs_orig['prediction'] = [final_map.get((u, g), 0) for u, g in zip(pairs_orig['userID'], pairs_orig['gameID'])]
    pairs_orig.to_csv("predictions_Played.csv", index=False)
    print("Done.")
