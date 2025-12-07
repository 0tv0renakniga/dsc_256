import gzip
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import random
from tqdm import tqdm
import warnings

# --- Production Imports ---
import lightgbm as lgb
import catboost as cb

# --- HPF Imports ---
try:
    from hpfrec import HPF
    HPF_AVAILABLE = True
except ImportError:
    print("Warning: 'hpfrec' not found. HPF features will be skipped.")
    HPF_AVAILABLE = False

# --- NLP Imports ---
try:
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
except ImportError:
    print("Warning: 'sentence_transformers' not found. NLP features will be skipped.")
    NLP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
###########################################################################
if __name__ == "__main__":
    # C. GOAL 2: PLAYED (Graph Ranker)
    # We use the raw_data (df) to train the graph
    ranker, eng, features = run_goal2_ranker(df)
    
    try:
        # Load Test Pairs
        pairs = pd.read_csv(FILES['test_played'])
        
        # Enrich Test Data
        test_enriched = eng.enrich(pairs)
        
        # Predict Scores
        print("[Goal 2] Predicting Scores...")
        pairs['score'] = ranker.predict(test_enriched[features])
        
        # Per-User Calibration (Top 50% = 1)
        print("[Goal 2] Applying Per-User 50/50 Split...")
        final_preds = []
        for u, group in tqdm(pairs.groupby('userID')):
            group = group.sort_values('score', ascending=False)
            n = len(group)
            group['prediction'] = 0
            # Assign top half to 1
            if n > 0:
                cutoff = n // 2
                # If n=1, cutoff=0 -> prediction=0. If we want at least one play?
                # Usually balanced means exactly half.
                if cutoff > 0:
                    group.iloc[:cutoff, group.columns.get_loc('prediction')] = 1
                elif n == 1:
                    # Edge case: single prediction.
                    # Ranker score doesn't mean much alone without threshold.
                    # We default to 1 if score is positive? 
                    # Let's trust the sort. If n=1, cutoff=0. Pred is 0.
                    pass
            final_preds.append(group)
            
        final_df = pd.concat(final_preds)
        
        # Re-map to original order to be safe
        pairs_orig = pd.read_csv(FILES['test_played'])
        # Create a map key: (u,g) -> pred
        # Note: final_df might shuffle rows, but (u,g) is unique in test set usually
        pred_map = dict(zip(zip(final_df['userID'], final_df['gameID']), final_df['prediction']))
        
        pairs_orig['prediction'] = [pred_map.get((u, g), 0) for u, g in zip(pairs_orig['userID'], pairs_orig['gameID'])]
        
        pairs_orig.to_csv("predictions_Played.csv", columns=['userID', 'gameID', 'prediction'], index=False)
        print("-> predictions_Played.csv saved.")
        
    except Exception as e:
        print(f"Goal 2 Error: {e}")
