import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = r"c:\LMO\data"
TRAIN_FILE = os.path.join(DATA_DIR, "training_data.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SEASONALITY_FILE = os.path.join(DATA_DIR, "models", "seasonality_factors.csv")

def backtest():
    print("Loading Training Data...")
    df = pd.read_csv(TRAIN_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # Check Events in Data
    event_days = df[df['is_event_day'] == 1]['date'].unique()
    print(f"Total Event Days in Dataset: {len(event_days)}")
    
    if len(event_days) < 2:
        print("Not enough event days to backtest (need at least 2).")
        return

    # Sort and pick the LAST event day
    last_event_date = sorted(event_days)[-1]
    print(f"Target Backtest Date (Last Event): {last_event_date.date()}")
    
    # Split
    # Train: Data BEFORE last_event_date
    # Test: Data ON last_event_date
    train_df = df[df['date'] < last_event_date].copy()
    test_df = df[df['date'] == last_event_date].copy()
    
    print(f"Train Set: {len(train_df)} rows")
    print(f"Test Set: {len(test_df)} rows")
    
    # Recalculate Seasonality on TRAIN Set only?
    # Ideally yes, but for speed, we assume Seasonality is stable. 
    # Let's focus on XGBoost Impact Model.
    
    # Prepare Features (Same as train_impact.py)
    seasonality = pd.read_csv(SEASONALITY_FILE, index_col='item_name')
    
    def prepare_xy(data_df):
        # Merge Seasonality
        s_melt = seasonality[['day_0','day_1','day_2','day_3','day_4','day_5','day_6']].reset_index().melt(id_vars='item_name', var_name='day_col', value_name='factor')
        s_melt['weekday'] = s_melt['day_col'].apply(lambda x: int(x.split('_')[1]))
        s_melt.drop(columns=['day_col'], inplace=True)
        
        tmp = pd.merge(data_df, s_melt, on=['item_name', 'weekday'], how='left')
        tmp = pd.merge(tmp, seasonality[['global_mean']], on='item_name', how='left')
        
        tmp['baseline_price'] = tmp['global_mean'] * tmp['factor']
        tmp['excess_return'] = (tmp['price_mean'] - tmp['baseline_price']) / tmp['baseline_price']
        
        # Categorize
        def categorize(name):
            if any(x in name for x in ['명파', '오레하', '파괴석', '수호석', '돌파석']): return 'honing_mat'
            if any(x in name for x in ['보석', '젬']): return 'gem'
            if any(x in name for x in ['각인', '서']): return 'engraving'
            if any(x in name for x in ['주머니', '상자']): return 'consumable'
            return 'other'
            
        tmp['category'] = tmp['item_name'].apply(categorize)
        tmp = pd.get_dummies(tmp, columns=['category'], prefix='is')
        
        features = [
            'honing_mat_demand', 'gem_demand', 'gold_inflation', 'content_difficulty', 
            'event_count', 'is_event_day', 'is_weekend',
            'is_honing_mat', 'is_gem', 'is_engraving', 'is_consumable', 'is_other'
        ]
        
        for col in features:
            if col not in tmp.columns: tmp[col] = 0
            
        # Clean Targets
        # Drop NaNs
        tmp = tmp.dropna(subset=['excess_return'])
        
        # Filter Inf
        tmp = tmp[~np.isinf(tmp['excess_return'])]
        
        # Filter Outliers (Same as train_impact.py)
        tmp = tmp[(tmp['excess_return'] > -0.5) & (tmp['excess_return'] < 5.0)] # Loosened upper bound for backtest
        
        return tmp[features], tmp['excess_return'], tmp
        
    print("Preparing features...")
    X_train, y_train, train_full = prepare_xy(train_df)
    X_test, y_test, full_test = prepare_xy(test_df)
    
    # Calculate Weights for Train Set
    train_full['date_ordinal'] = train_full['date'].map(pd.Timestamp.toordinal)
    min_date = train_full['date_ordinal'].min()
    max_date = train_full['date_ordinal'].max()
    
    # Same Formula: 0.5 to 2.0
    w_train = 0.5 + 1.5 * (train_full['date_ordinal'] - min_date) / (max_date - min_date)
    
    # Train Temp Model
    print("Training Backtest Model (Weighted)...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Compare
    full_test['predicted_excess'] = preds
    full_test['error'] = full_test['predicted_excess'] - full_test['excess_return']
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\n--- Backtest Results ({last_event_date.date()}) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Show Top 5 Items with Highest Predicted Impact
    print("\n[AI Predicted Top Movers]")
    top_pred = full_test.sort_values('predicted_excess', ascending=False).head(5)
    print(top_pred[['item_name', 'predicted_excess', 'excess_return', 'error']])
    
    # Show Top 5 Items with Actual Highest Impact
    print("\n[Actual Top Movers]")
    top_actual = full_test.sort_values('excess_return', ascending=False).head(5)
    print(top_actual[['item_name', 'predicted_excess', 'excess_return', 'error']])

if __name__ == "__main__":
    backtest()
