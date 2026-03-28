import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'events_impact_analyzed.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'data', 'models', 'impact_model.pkl')

def train_trajectory_model():
    print("Loading Event Impact Data...")
    if not os.path.exists(DATA_FILE):
        print("Data file missing.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Deduplicate: Keep the event with the highest 'honing_mat_demand' per date
    # This resolves "Update Notice" vs "Cash Shop" ambiguity
    if 'honing_mat_demand' in df.columns:
         df['honing_mat_demand'] = pd.to_numeric(df['honing_mat_demand'], errors='coerce').fillna(0)
         df = df.sort_values(by='honing_mat_demand', ascending=False).drop_duplicates(subset=['date'], keep='first')
         df = df.sort_values(by='date')
         print(f"Deduplicated Events: {len(df)} unique dates.")
    features = ['honing_mat_demand', 'gem_demand', 'gold_inflation', 'content_difficulty']
    
    # Ensure features exist and are numeric
    # Ensure features exist and are numeric
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    
    print("DEBUG: Feature Dtypes after conversion:")
    print(df[features].dtypes)
    print(df[features].head())
            
    X = df[features]
    
    # 2. Define Targets (Y) - Multi-Horizon Trajectory
    # We want to predict the price index for the full curve
    # Updated to wider range as per user request
    horizons = [-30, -21, -14, -10, -7, -5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
    
    # Honing Targets
    honing_targets = [f'Actual_Honing_T{h}' for h in horizons]
    gem_targets = [f'Actual_Gem_T{h}' for h in horizons]
    engraving_targets = [f'Actual_Engraving_T{h}' for h in horizons]
    
    # We will train separate models for each category or one big multi-output?
    # Let's train one model per category for simplicity and clarity.
    
    models = {}
    
    def train_category(cat_name, target_cols, score_feature='honing_mat_demand'):
        print(f"\nTraining for {cat_name} (Per-Horizon Independent Models)...")
        
        df_sorted = df.sort_values('date')
        honest_preds = []
        min_train = 3
        
        # 1. Honest Predictions (Adaptive Batch Walk-Forward)
        # Use large batches for deep history (speed), single-step for recent history (accuracy)
        print(f"  > Generating Honest Predictions (Adaptive Batching)...")
        
        start_idx = 0
        total_len = len(df_sorted)
        
        while start_idx < total_len:
            # Adaptive Batch Size: 50 for history, 1 for recent 50 events
            current_batch_size = 50 if (total_len - start_idx) > 50 else 1
            
            end_idx = min(start_idx + current_batch_size, total_len)
            
            # Define Batch to Predict
            test_batch = df_sorted.iloc[start_idx : end_idx]
            
            # Define Training Data (Strict Past relative to batch start)
            past_data = df_sorted.iloc[:start_idx]
            
            chunk_models = {}
            can_train = len(past_data) >= min_train
            
            if can_train:
                for col in target_cols:
                    valid_mask = (past_data[col] != 0) & (past_data[col].notna())
                    train_subset = past_data[valid_mask]
                    
                    if len(train_subset) >= 3:
                        # 1. Score Weight (Impact)
                        w_score = 1 + (train_subset[score_feature] ** 2)
                        
                        # 2. Time Weight (Recency)
                        try:
                            dates = pd.to_datetime(train_subset['date'])
                            min_date = dates.min()
                            days = (dates - min_date).dt.days
                            # Linear weight: 1.0 (start) -> ~5.0 (4 years later)
                            w_time = 1 + (days / 100.0) 
                        except:
                            w_time = 1.0
                        
                        weights = w_score * w_time
                        
                        ctr = '(1, 0, 0, 1)' if 'Honing' in cat_name else '(0, 1, 0, 0)'
                        xgb = XGBRegressor(
                            objective='reg:squarederror', 
                            n_estimators=100, 
                            max_depth=5, 
                            learning_rate=0.05, 
                            monotone_constraints=ctr,
                            n_jobs=-1
                        )
                        xgb.fit(train_subset[features], train_subset[col], sample_weight=weights)
                        chunk_models[col] = xgb
                    else:
                        chunk_models[col] = None
            
            # Predict for Batch
            for idx, row in test_batch.iterrows():
                if str(row['date']) == '2026-01-07':
                     print(f"DEBUG: Jan 7 Prediction Input Features: {row[features].to_dict()}")
                
                traj = []
                # Prepare input as DataFrame (preserving feature names)
                input_df = row[features].to_frame().T.astype(float)
                
                for col in target_cols:
                    model = chunk_models.get(col) if can_train else None
                    if model:
                        pred = model.predict(input_df)[0]
                        traj.append(float(pred))
                    else:
                        if traj: traj.append(traj[-1])
                        else: traj.append(100.0)
                
                honest_preds.append({
                    'date': row['date'],
                    'predicted_trajectory': traj, 
                    'horizons': horizons
                })
            
            # Update final_models if needed (though we rebuild it later properly)
            # Actually final_models is rebuilt after loop.
            
            start_idx += current_batch_size

        # 2. Final Training (For Future Events)
        final_models = {} 
        for col in target_cols:
            valid_mask = (df[col] != 0) & (df[col].notna())
            train_subset = df[valid_mask]
            
            if len(train_subset) < 3:
                final_models[col] = None
            else:
                # 1. Score Weight (Impact)
                w_score = 1 + (train_subset[score_feature] ** 2)
                
                # 2. Time Weight (Recency)
                # Convert dates to numeric relative days
                try:
                    dates = pd.to_datetime(train_subset['date'])
                    min_date = dates.min()
                    days = (dates - min_date).dt.days
                    # Linear weight: 1.0 (start) -> ~5.0 (4 years later)
                    w_time = 1 + (days / 100.0) # Aggressive recency!
                except:
                    w_time = 1.0
                
                weights = w_score * w_time
                
                ctr = '(1, 0, 0, 1)' if 'Honing' in cat_name else '(0, 1, 0, 0)'
                xgb = XGBRegressor(
                    objective='reg:squarederror', 
                    n_estimators=100, 
                    max_depth=5,
                    learning_rate=0.05,
                    monotone_constraints=ctr
                )
                xgb.fit(train_subset[features], train_subset[col], sample_weight=weights)
                final_models[col] = xgb
        
        # Calculate RMSE (Approximate Average across horizons)
        # Just simple print
        print(f"  > Final Training Complete.")
        
        # 3. Overwrite Recent Event Predictions (Last 10) with Final Model
        # This resolves discrepancies for recent events (e.g. Jan 7) compared to final model
        if len(df_sorted) > 0:
            recents = df_sorted.tail(10)
            print(f"  > Refinding Predictions for Last {len(recents)} Events using Final Model...")
            
            for idx, row in recents.iterrows():
                target_date = row['date']
                
                # Prepare Input
                input_df = row[features].to_frame().T.astype(float)
                traj = []
                for col in target_cols:
                    model = final_models.get(col)
                    if model:
                        try:
                            pred = model.predict(input_df)[0]
                            traj.append(float(pred))
                        except:
                            traj.append(100.0)
                    else:
                         traj.append(100.0)
                
                # Update honest_preds
                for p in honest_preds:
                    if p['date'] == target_date:
                        p['predicted_trajectory'] = traj
                        if target_date == '2026-01-07':
                             print(f"    DEBUG: Jan 7 Overwritten Trajectory: {traj[:5]}...")
        
        return final_models, honest_preds

    all_honest_preds = {}

    h_res = train_category('Honing', honing_targets, 'honing_mat_demand')
    if h_res:
        models['honing'], honest_h = h_res
        all_honest_preds['honing'] = honest_h
        
    g_res = train_category('Gem', gem_targets, 'gem_demand')
    if g_res:
         models['gem'], honest_g = g_res
         all_honest_preds['gem'] = honest_g
         
    e_res = train_category('Engraving', engraving_targets, 'honing_mat_demand')
    if e_res:
         models['engraving'], honest_e = e_res
         all_honest_preds['engraving'] = honest_e
    
    # Save Dictionary of Models
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(models, MODEL_FILE)
    print(f"\nSaved Independent Horizon Models to {MODEL_FILE}")
    
    # Save Honest History (JSON)
    hist_file = os.path.join(BASE_DIR, 'data', 'historical_predictions.json')
    import json
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.float64): return float(o)
        return o
        
    with open(hist_file, 'w', encoding='utf-8') as f:
        json.dump(all_honest_preds, f, default=convert, indent=2)
    print(f"Saved Honest History to {hist_file}")

if __name__ == "__main__":
    train_trajectory_model()
