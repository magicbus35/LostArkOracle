import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDS_PATH = os.path.join(BASE_DIR, 'data', 'historical_predictions.json')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'events_impact_analyzed.csv')

def evaluate_directional_accuracy():
    if not os.path.exists(PREDS_PATH) or not os.path.exists(DATA_PATH):
        print("Data files not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    with open(PREDS_PATH, 'r') as f:
        preds = json.load(f)

    # Dictionary: mapping date to predictions for honing
    honing_preds = {}
    if 'honing' in preds:
        for p in preds['honing']:
            date_str = p['date']
            # We want T+7 which is normally index 13 if horizons are [-30, -21, -14, -10, -7, -5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
            # find index of 7 in p['horizons']
            if 'horizons' in p and 7 in p['horizons']:
                idx = p['horizons'].index(7)
                pred_val = p['predicted_trajectory'][idx]
                if pred_val is not None:
                    honing_preds[date_str] = pred_val

    # Match with df
    target_horizon = 'Actual_Honing_T7'
    df_valid = df.dropna(subset=[target_horizon])
    df_valid = df_valid[df_valid[target_horizon] > 0]
    
    actual_ups = 0
    pred_ups = 0
    true_ups = 0
    total = 0
    hits = 0

    THRESHOLD = 101.0  # Consider an actual UP trend as > 1% gain to ignore noise

    for _, row in df_valid.iterrows():
        date_str = str(row['date'])
        if date_str in honing_preds:
            y_actual = row[target_horizon]
            y_pred = honing_preds[date_str]
            
            # Since the model is currently trained uniformly, many might be flat predictions (~100.0)
            # Let's see if the direction matched exactly.
            actual_up = y_actual > THRESHOLD
            pred_up = y_pred > THRESHOLD
            
            if actual_up == pred_up:
                hits += 1
                
            if actual_up:
                actual_ups += 1
            if pred_up:
                pred_ups += 1
            if actual_up and pred_up:
                true_ups += 1
                
            total += 1

    if total == 0:
        print("No valid predictions matched.")
        return

    accuracy = (hits / total) * 100
    recall = (true_ups / actual_ups * 100) if actual_ups > 0 else 0
    precision = (true_ups / pred_ups * 100) if pred_ups > 0 else 0
    
    print(f"Total Valid Events Evaluated: {total}")
    print(f"Global Directional Hit Rate: {accuracy:.2f}%")
    print(f"Recall (Caught Real Spikes): {recall:.2f}% ({true_ups}/{actual_ups})")
    print(f"Precision (When Model said UP, it was UP): {precision:.2f}% ({true_ups}/{pred_ups})")

if __name__ == "__main__":
    evaluate_directional_accuracy()
