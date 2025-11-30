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

class HybridScorer:
    def __init__(self, n_factors=40, k_neighbors=20, max_features=500, 
                 nmf_components=30, active_models=None):
        """
        Enhanced Hybrid Recommender with 5 base models + subset selection.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors for SVD
        k_neighbors : int
            Number of neighbors for User-User KNN
        max_features : int
            Max TF-IDF features for content model
        nmf_components : int
            Number of components for NMF (NEW Model E)
        active_models : list or None
            List of model indices to use [0,1,2,3,4]. If None, uses all.
            0=SVD, 1=UserUser, 2=ItemItem, 3=Content, 4=NMF
        """
        self.n_factors = n_factors
        self.k = k_neighbors
        self.max_features = max_features
        self.nmf_components = nmf_components
        self.active_models = active_models  # Model selection
        
        # Mappings
        self.user_map = {}
        self.item_map = {}
        self.users_arr = None
        self.items_arr = None
        self.item_means_arr = None
        self.global_mean = 0
        
        # Models
        self.u_factors = None      # SVD User factors
        self.vt_factors = None     # SVD Item factors
        self.sim_ii = None         # Item-Item Similarity
        self.knn_user = None       # User-User KNN
        self.tfidf = None          # TF-IDF vectorizer
        self.user_profiles = None  # User content profiles
        self.item_profiles = None  # Item content profiles
        
        # NEW: Model E - Non-Negative Matrix Factorization
        self.nmf_user = None       # NMF user factors
        self.nmf_item = None       # NMF item factors
        
        # Meta Model
        self.meta_model = Ridge(alpha=1.0, fit_intercept=True, positive=True)
        
    def fit(self, df):
        """Train all base models on the training data."""
        print("="*60)
        print("TRAINING HYBRID RECOMMENDER SYSTEM")
        print("="*60)
        
        print("\n1. Initializing...")
        self.global_mean = df['hours_transformed'].mean()
        print(f"   Global mean: {self.global_mean:.3f}")
        
        # Build mappings
        self.users_arr = df['userID'].unique()
        self.items_arr = df['gameID'].unique()
        self.user_map = {user_id: idx for idx, user_id in enumerate(self.users_arr)}
        self.item_map = {game_id: idx for idx, game_id in enumerate(self.items_arr)}
        
        n_users = len(self.users_arr)
        n_items = len(self.items_arr)
        print(f"   Users: {n_users:,}, Items: {n_items:,}")
        
        # Build sparse rating matrix
        rows = df['userID'].map(self.user_map).values
        cols = df['gameID'].map(self.item_map).values
        data = df['hours_transformed'].values
        self.R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        print(f"   Sparsity: {100 * (1 - self.R.nnz / (n_users * n_items)):.2f}%")
        
        # --- MODEL A: Item-Centered SVD ---
        print("\n2. Training Model A (Item-Centered SVD)...")
        item_sums = np.array(self.R.sum(axis=0)).flatten()
        item_counts = np.diff(self.R.tocsc().indptr)
        self.item_means_arr = np.zeros(n_items)
        mask = item_counts > 0
        self.item_means_arr[mask] = item_sums[mask] / item_counts[mask]
        
        data_centered = data - self.item_means_arr[cols]
        R_centered = csr_matrix((data_centered, (rows, cols)), shape=(n_users, n_items))
        
        u, s, vt = svds(R_centered, k=self.n_factors)
        self.u_factors = u @ np.diag(s)
        self.vt_factors = vt
        print(f"   ✓ Learned {self.n_factors} latent factors")
        
        # --- MODEL B: User-User KNN ---
        print("\n3. Training Model B (User-User KNN)...")
        self.knn_user = NearestNeighbors(n_neighbors=self.k, metric='cosine', n_jobs=-1)
        self.knn_user.fit(self.R)
        print(f"   ✓ Built KNN index with k={self.k}")
        
        # --- MODEL C: Item-Item Similarity ---
        print("\n4. Training Model C (Item-Item Cosine)...")
        R_t = self.R.T.tocsr()
        self.sim_ii = cosine_similarity(R_t, dense_output=False)
        print(f"   ✓ Computed {n_items}x{n_items} similarity matrix")
        
        # --- MODEL D: Content-Based (TF-IDF) ---
        print("\n5. Training Model D (TF-IDF Content)...")
        df_clean = df.copy()
        if 'text' not in df_clean.columns: 
            df_clean['text'] = ''
        df_clean['text'] = df_clean['text'].fillna('')
        
        self.tfidf = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        
        item_text = df_clean.groupby('gameID')['text'].apply(lambda x: ' '.join(x))
        item_corpus = [item_text.get(game_id, '') for game_id in self.items_arr]
        self.item_profiles = self.tfidf.fit_transform(item_corpus)
        
        user_text = df_clean.groupby('userID')['text'].apply(lambda x: ' '.join(x))
        user_corpus = [user_text.get(user_id, '') for user_id in self.users_arr]
        self.user_profiles = self.tfidf.transform(user_corpus)
        print(f"   ✓ Built TF-IDF with {self.max_features} features")
        
        # --- MODEL E: Non-Negative Matrix Factorization (NEW!) ---
        print("\n6. Training Model E (NMF - Non-Negative Factorization)...")
        print("   Why NMF? It learns parts-based, interpretable patterns.")
        print("   Unlike SVD, NMF constraints: all factors ≥ 0")
        print("   Good for: discovering game genres, playstyle clusters")
        
        # NMF requires dense non-negative matrix, so we work with a sample
        # For very large matrices, use sparse NMF or approximate methods
        nmf = NMF(n_components=self.nmf_components, init='nndsvd', 
                  max_iter=200, random_state=42, alpha_W=0.01, alpha_H=0.01)
        
        # Convert to dense (warning: memory intensive for huge datasets)
        # Alternative: use implicit library's AlternatingLeastSquares with confidence weights
        R_dense = self.R.toarray()
        self.nmf_user = nmf.fit_transform(R_dense)  # User factors [n_users x nmf_components]
        self.nmf_item = nmf.components_.T           # Item factors [n_items x nmf_components]
        
        reconstruction_error = nmf.reconstruction_err_
        print(f"   ✓ Learned {self.nmf_components} non-negative components")
        print(f"   Reconstruction error: {reconstruction_error:.3f}")
        
        print("\n" + "="*60)
        print("ALL BASE MODELS TRAINED SUCCESSFULLY")
        print("="*60)

    def _get_individual_preds(self, u_idx, i_idx):
        """
        Calculate predictions from all 5 base models for a single (user, item) pair.
        
        Returns:
        --------
        list of 5 floats: [pA, pB, pC, pD, pE]
        """
        preds = []
        
        # --- A: SVD ---
        base = self.item_means_arr[i_idx]
        pA = base + np.dot(self.u_factors[u_idx], self.vt_factors[:, i_idx])
        preds.append(pA)
        
        # --- B: User-User KNN ---
        dists, n_indices = self.knn_user.kneighbors(self.R[u_idx], n_neighbors=self.k)
        n_indices = n_indices.flatten()
        n_sims = 1 - dists.flatten()
        
        n_ratings = self.R[n_indices, i_idx].toarray().flatten()
        mask = n_ratings > 0
        if mask.sum() > 0:
            pB = np.dot(n_sims[mask], n_ratings[mask]) / n_sims[mask].sum()
        else:
            pB = self.global_mean
        preds.append(pB)
        
        # --- C: Item-Item ---
        u_vec = self.R[u_idx]
        indices = u_vec.indices
        ratings = u_vec.data
        
        if len(indices) > 0:
            sim_row = self.sim_ii[i_idx]
            sims = np.zeros(len(indices))
            for k, hist_item_idx in enumerate(indices):
                sims[k] = sim_row[0, hist_item_idx]
                
            if sims.sum() > 0:
                pC = np.dot(sims, ratings) / sims.sum()
            else:
                pC = self.global_mean
        else:
            pC = self.global_mean
        preds.append(pC)
        
        # --- D: Content (TF-IDF) ---
        sim = (self.user_profiles[u_idx] @ self.item_profiles[i_idx].T).toarray()[0][0]
        pD = sim * 10.0  # Scale factor
        preds.append(pD)
        
        # --- E: NMF (NEW) ---
        # Prediction = user_factors · item_factors
        pE = np.dot(self.nmf_user[u_idx], self.nmf_item[i_idx])
        preds.append(pE)
        
        # Filter by active models if specified
        if self.active_models is not None:
            preds = [preds[i] for i in self.active_models]
        
        return preds

    def _get_base_preds(self, df):
        """
        Extract base predictions for all models on a dataframe.
        
        Returns:
        --------
        X_base : np.array, shape [n_samples, n_models]
        y_true : np.array, shape [n_samples]
        """
        X_stack = []
        y_true = []
        
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting predictions"):
            u, g = row.userID, row.gameID
            if u in self.user_map and g in self.item_map:
                u_idx = self.user_map[u]
                i_idx = self.item_map[g]
                preds = self._get_individual_preds(u_idx, i_idx)
                X_stack.append(preds)
                y_true.append(row.hours_transformed)
        
        return np.array(X_stack), np.array(y_true)

    def learn_weights(self, val_df):
        """Learn meta-model weights using Ridge regression."""
        print("\n" + "="*60)
        print("LEARNING META-MODEL WEIGHTS")
        print("="*60)
        
        X_stack, y_val = self._get_base_preds(val_df)
        
        print(f"\nValidation samples: {len(y_val):,}")
        print(f"Feature matrix shape: {X_stack.shape}")
        
        print("\nTraining Ridge regression...")
        self.meta_model.fit(X_stack, y_val)
        
        model_names = ['SVD', 'UserUser', 'ItemItem', 'Content', 'NMF']
        if self.active_models is not None:
            model_names = [model_names[i] for i in self.active_models]
        
        print("\nLearned Weights:")
        for name, weight in zip(model_names, self.meta_model.coef_):
            print(f"  {name:12s}: {weight:.4f}")
        print(f"  Intercept   : {self.meta_model.intercept_:.4f}")
        
        final_val_preds = self.meta_model.predict(X_stack)
        mse = mean_squared_error(y_val, final_val_preds)
        print(f"\n✓ Validation MSE: {mse:.4f}")
        print("="*60)
        
        return mse

    def find_best_model_subset(self, val_df, method='optimize', verbose=True):
        """
        Test all possible subsets of base models to find optimal combination.
        
        Parameters:
        -----------
        val_df : DataFrame
        method : str, 'optimize' or 'ridge'
        verbose : bool
        
        Returns:
        --------
        dict with best_subset, best_mse, best_weights, etc.
        """
        print("\n" + "="*60)
        print("EXHAUSTIVE MODEL SUBSET SEARCH")
        print("="*60)
        
        # Temporarily use all models
        original_active = self.active_models
        self.active_models = None
        
        X_base, y_true = self._get_base_preds(val_df)
        n_models = X_base.shape[1]
        model_names = ['A_SVD', 'B_UserUser', 'C_ItemItem', 'D_Content', 'E_NMF']
        
        print(f"\nBase predictions shape: {X_base.shape}")
        print(f"Testing all 2^{n_models}-1 = {2**n_models - 1} subsets...\n")
        
        all_results = []
        best_mse = float('inf')
        best_config = None
        
        # Test all non-empty subsets
        for r in range(1, n_models + 1):
            for subset_indices in combinations(range(n_models), r):
                subset_indices = list(subset_indices)
                subset_names = [model_names[i] for i in subset_indices]
                
                X_subset = X_base[:, subset_indices]
                
                if method == 'optimize':
                    mse, weights, bias = self._optimize_subset(X_subset, y_true)
                else:
                    mse, weights, bias = self._ridge_subset(X_subset, y_true)
                
                result = {
                    'subset_indices': subset_indices,
                    'subset_names': subset_names,
                    'mse': mse,
                    'weights': weights,
                    'bias': bias,
                    'n_models': len(subset_indices)
                }
                all_results.append(result)
                
                if mse < best_mse:
                    best_mse = mse
                    best_config = result
                
                if verbose:
                    models_str = ' + '.join([n.split('_')[1] for n in subset_names])
                    print(f"[{r}/{n_models}] {models_str:30s} | MSE: {mse:.5f}")
        
        # Restore original active models
        self.active_models = original_active
        
        # Print results
        print("\n" + "="*60)
        print("🏆 BEST CONFIGURATION FOUND")
        print("="*60)
        print(f"Models: {' + '.join([n.split('_')[1] for n in best_config['subset_names']])}")
        print(f"Validation MSE: {best_config['mse']:.5f}")
        print(f"\nOptimal Weights:")
        for name, weight in zip(best_config['subset_names'], best_config['weights']):
            print(f"  {name:15s}: {weight:.4f}")
        print(f"  Bias/Intercept: {best_config['bias']:.4f}")
        
        # Compare to all models
        all_models_result = [r for r in all_results if r['n_models'] == n_models][0]
        improvement = all_models_result['mse'] - best_config['mse']
        pct_improvement = 100 * improvement / all_models_result['mse']
        
        print(f"\n📊 Comparison:")
        print(f"  All {n_models} models MSE: {all_models_result['mse']:.5f}")
        print(f"  Best subset MSE:  {best_config['mse']:.5f}")
        print(f"  Improvement:      {improvement:.5f} ({pct_improvement:.2f}%)")
        
        if best_config['n_models'] < n_models:
            excluded = set(range(n_models)) - set(best_config['subset_indices'])
            excluded_names = [model_names[i].split('_')[1] for i in excluded]
            print(f"\n⚠️  RECOMMENDATION: Remove {', '.join(excluded_names)}!")
            print(f"    These models add noise and hurt performance.")
        else:
            print(f"\n✓ All models contribute positively. Keep full ensemble.")
        
        print("="*60)
        
        return {
            'best_subset': best_config['subset_indices'],
            'best_subset_names': best_config['subset_names'],
            'best_mse': best_config['mse'],
            'best_weights': best_config['weights'],
            'best_bias': best_config['bias'],
            'all_results': all_results,
            'all_models_mse': all_models_result['mse']
        }
    
    def _optimize_subset(self, X, y):
        """Optimize weights using scipy minimize."""
        n_weights = X.shape[1]
        
        def objective(params):
            weights = params[:n_weights]
            bias = params[n_weights]
            pred = np.clip(X @ weights + bias, 0, 14.5)
            return mean_squared_error(y, pred)
        
        initial = np.ones(n_weights + 1) * 0.5
        initial[-1] = 0
        
        result = minimize(
            objective, initial,
            method='L-BFGS-B',
            bounds=[(-2, 2)] * n_weights + [(-5, 5)]
        )
        
        return result.fun, result.x[:n_weights], result.x[n_weights]
    
    def _ridge_subset(self, X, y):
        """Use Ridge regression for subset."""
        model = Ridge(alpha=1.0, fit_intercept=True, positive=True)
        model.fit(X, y)
        pred = np.clip(model.predict(X), 0, 14.5)
        mse = mean_squared_error(y, pred)
        return mse, model.coef_, model.intercept_

    def apply_best_subset(self, best_subset_indices):
        """Permanently set the model to use only the best subset."""
        self.active_models = best_subset_indices
        print(f"\n✓ Model now uses only: {self.active_models}")

    def make_test_predictions(self, test_df, output_path='predictions_Hours.csv'):
        """Generate predictions for test set."""
        print(f"\n{'='*60}")
        print(f"GENERATING TEST PREDICTIONS")
        print(f"{'='*60}")
        print(f"Test pairs: {len(test_df):,}")
        
        final_preds = []
        
        for row in tqdm(test_df.itertuples(index=False), total=len(test_df)):
            u, g = row.userID, row.gameID
            p = self.global_mean
            
            if u in self.user_map and g in self.item_map:
                u_idx = self.user_map[u]
                i_idx = self.item_map[g]
                base_preds = np.array(self._get_individual_preds(u_idx, i_idx)).reshape(1, -1)
                p = self.meta_model.predict(base_preds)[0]
            
            final_preds.append(p)
        
        final_preds = np.clip(final_preds, 0, 14.5)
        
        sub = test_df[['userID', 'gameID']].copy()
        sub['prediction'] = final_preds
        sub.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")
        print("="*60)


#######
# USAGE PIPELINE
#######

def readJSON(path):
    """Read compressed JSON data."""
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        u = d['userID']
        g = d.get('gameID', None)
        yield u, g, d

def split_train_data(df, test_size=0.2, min_user_games=2, min_game_users=2, random_state=42):
    """Split ensuring no cold-start and no pair overlap."""
    np.random.seed(random_state)
    
    print("\n" + "="*60)
    print("SPLITTING TRAIN/VALIDATION DATA")
    print("="*60)
    
    # Filter minimum interactions
    print("\nFiltering for minimum interactions...")
    temp_df = df.copy()
    
    user_counts = temp_df['userID'].value_counts()
    valid_users = user_counts[user_counts >= min_user_games].index
    temp_df = temp_df[temp_df['userID'].isin(valid_users)]
    
    game_counts = temp_df['gameID'].value_counts()
    valid_games = game_counts[game_counts >= min_game_users].index
    temp_df = temp_df[temp_df['gameID'].isin(valid_games)]
    
    print(f"  Original: {len(df):,} → Filtered: {len(temp_df):,}")
    
    # Split by user
    temp_df = temp_df.reset_index(drop=True)
    train_mask = np.ones(len(temp_df), dtype=bool)
    grouped = temp_df.groupby('userID')
    val_indices_list = []
    
    print("\nSplitting per user...")
    for user, group in grouped:
        n_samples = len(group)
        n_val = int(n_samples * test_size)
        if n_samples - n_val < min_user_games:
            n_val = n_samples - min_user_games
        if n_val > 0:
            val_idx = np.random.choice(group.index, n_val, replace=False)
            val_indices_list.extend(val_idx)
    
    train_mask[val_indices_list] = False
    train_df = temp_df[train_mask].copy()
    val_df = temp_df[~train_mask].copy()
    
    # Fix cold-start items
    train_items = set(train_df['gameID'].unique())
    val_items = set(val_df['gameID'].unique())
    cold_start_items = val_items - train_items
    
    if len(cold_start_items) > 0:
        print(f"\nMoving {len(cold_start_items)} cold-start items to train...")
        rows_to_move = val_df[val_df['gameID'].isin(cold_start_items)]
        train_df = pd.concat([train_df, rows_to_move], ignore_index=True)
        val_df = val_df[~val_df['gameID'].isin(cold_start_items)].copy()
    
    # Verify
    train_pairs = set(zip(train_df['userID'], train_df['gameID']))
    val_pairs = set(zip(val_df['userID'], val_df['gameID']))
    overlap = train_pairs.intersection(val_pairs)
    
    print(f"\n✓ Final split: Train={len(train_df):,}, Val={len(val_df):,}")
    print(f"✓ Pair overlap: {len(overlap)} (should be 0)")
    print("="*60)
    
    return train_df, val_df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID RECOMMENDER SYSTEM - FULL PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = [d for u, g, d in readJSON("train.json.gz")]
    df = pd.DataFrame(data)
    print(f"✓ Loaded {len(df):,} reviews")
    
    # Split
    df_train, df_val = split_train_data(df, test_size=0.2, random_state=42)
    
    # Initialize with all 5 models
    print("\nInitializing HybridScorer with 5 models...")
    scorer = HybridScorer(n_factors=40, k_neighbors=20, max_features=500, nmf_components=30)
    
    # Train
    scorer.fit(df_train)
    
    # Find best subset
    print("\n" + "="*60)
    print("STEP 1: FIND OPTIMAL MODEL SUBSET")
    print("="*60)
    results = scorer.find_best_model_subset(df_val, method='optimize', verbose=True)
    
    # Apply best subset
    print("\n" + "="*60)
    print("STEP 2: RETRAIN WITH BEST SUBSET")
    print("="*60)
    scorer.apply_best_subset(results['best_subset'])
    final_mse = scorer.learn_weights(df_val)
    
    # Make predictions
    print("\n" + "="*60)
    print("STEP 3: GENERATE SUBMISSION")
    print("="*60)
    pairs_hours_df = pd.read_csv("pairs_Hours.csv")
    scorer.make_test_predictions(pairs_hours_df)
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print(f"Best model subset: {[m.split('_')[1] for m in results['best_subset_names']]}")
    print(f"Final validation MSE: {final_mse:.4f}")
    print(f"Improvement over all models: {results['all_models_mse'] - final_mse:.4f}")
    print("="*60)
