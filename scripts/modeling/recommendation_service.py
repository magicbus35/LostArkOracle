import pandas as pd
import numpy as np
import os
import xgboost as xgb
import json
from datetime import datetime

DATA_DIR = r"c:\LMO\data"
MODEL_DIR = os.path.join(DATA_DIR, "models")
SEASONALITY_FILE = os.path.join(MODEL_DIR, "seasonality_factors.csv")
IMPACT_MODEL_FILE = os.path.join(MODEL_DIR, "impact_model.json")
MARKET_FILE = os.path.join(DATA_DIR, "lostark_market_data_all.csv") # For current price

class RecommendationService:
    def __init__(self):
        self.seasonality = None
        self.impact_model = None
        self.last_prices = {}
        self.load_models()
        self.load_current_prices()
        
    def load_models(self):
        if os.path.exists(SEASONALITY_FILE):
            self.seasonality = pd.read_csv(SEASONALITY_FILE, index_col='item_name')
        
        if os.path.exists(IMPACT_MODEL_FILE):
            self.impact_model = xgb.XGBRegressor()
            self.impact_model.load_model(IMPACT_MODEL_FILE)
            
    def load_current_prices(self):
        # Load latest price for each item from the aggregated training data or market data
        # For speed, let's use the seasonality 'global_mean' as a fallback, 
        # but ideally we want the *latest* price.
        # Let's read the last chunk of market data? Too slow.
        # Let's rely on global_mean for now as 'Reference Price'.
        pass

    def categorize_item(self, name):
        if any(x in name for x in ['명파', '오레하', '파괴석', '수호석', '돌파석']): return 'honing_mat'
        if any(x in name for x in ['보석', '젬']): return 'gem'
        if any(x in name for x in ['각인', '서']): return 'engraving'
        if any(x in name for x in ['주머니', '상자']): return 'consumable'
        return 'other'

    def predict_impact(self, event_features):
        """
        event_features: dict with keys like 'honing_mat_demand', 'is_event_day', etc.
        Returns: DataFrame with ['item_name', 'predicted_return', 'score']
        """
        if self.seasonality is None or self.impact_model is None:
            return pd.DataFrame()
            
        items = self.seasonality.index.tolist()
        
        # Prepare Input DataFrame
        rows = []
        for item in items:
            cat = self.categorize_item(item)
            row = {
                'item_name': item,
                'is_honing_mat': 1 if cat == 'honing_mat' else 0,
                'is_gem': 1 if cat == 'gem' else 0,
                'is_engraving': 1 if cat == 'engraving' else 0,
                'is_consumable': 1 if cat == 'consumable' else 0,
                'is_other': 1 if cat == 'other' else 0,
                **event_features
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # XGBoost Prediction
        # Columns must match training order!
        ordered_cols = [
            'honing_mat_demand', 'gem_demand', 'gold_inflation', 'content_difficulty', 
            'event_count', 'is_event_day', 'is_weekend',
            'is_honing_mat', 'is_gem', 'is_engraving', 'is_consumable', 'is_other',
            'honing_mat_supply', 'gem_supply'
        ]
        
        # Ensure cols exist
        for c in ordered_cols:
            if c not in df.columns: df[c] = 0
            
        X = df[ordered_cols]
        preds = self.impact_model.predict(X)
        
        df['predicted_excess_return'] = preds
        
        # Calculate Final Score
        # Score = Excess Return * 100 (Simplified)
        df['score'] = df['predicted_excess_return'] * 100
        
        return df[['item_name', 'predicted_excess_return', 'score']].sort_values('score', ascending=False)

if __name__ == "__main__":
    # Test
    svc = RecommendationService()
    test_event = {
        'honing_mat_demand': 8,
        'gem_demand': 2,
        'gold_inflation': 5,
        'content_difficulty': 3,
        'event_count': 1,
        'is_event_day': 1,
        'is_weekend': 0
    }
    print(svc.predict_impact(test_event).head(10))
