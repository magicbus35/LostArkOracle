import pandas as pd
import os
import numpy as np

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MARKET_CSV = os.path.join(DATA_DIR, 'lostark_market_data_all.csv')
EVENTS_CSV = os.path.join(DATA_DIR, 'events.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'lostark_features.csv')

def load_data():
    print("Loading Event Data...")
    # Try Enriched Data First
    v2_path = os.path.join(DATA_DIR, 'events_enriched_v2.csv')
    enriched_path = os.path.join(DATA_DIR, 'events_enriched.csv')
    official_path = os.path.join(DATA_DIR, 'events_official.csv')
    
    events_df = None
    
    if os.path.exists(v2_path):
        print(f"Loading Enriched Event Data (v2) from {v2_path}...")
        events_df = pd.read_csv(v2_path)
        # Header: date,event_type,title,link,keywords,impact_score,sentiment,reasoning
        
        if 'title' in events_df.columns:
            events_df.rename(columns={'title': 'event_name'}, inplace=True)
            
        # Map Sentiment
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        events_df['sentiment_score'] = events_df['sentiment'].map(sentiment_map).fillna(0).astype('int8')
        events_df['impact_score'] = events_df['impact_score'].fillna(0).astype('float32') # Use float for score
        
        # Keywords are in 'keywords' col instead of 'market_impact'
        target_col = 'keywords'
        
    elif os.path.exists(enriched_path):
        print(f"Loading Enriched Event Data (v1) from {enriched_path}...")
        events_df = pd.read_csv(enriched_path) 
        if 'title' in events_df.columns:
            events_df.rename(columns={'title': 'event_name'}, inplace=True)
        
        events_df['sentiment_score'] = 0
        events_df['impact_score'] = 0
        target_col = 'market_impact'
            
    if events_df is not None:
        events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')
        
        # Generate Features
        print("Generating Keyword Features...")
        events_df[target_col] = events_df[target_col].fillna('None')
        events_df['is_gold_nerf'] = events_df[target_col].str.contains('Gold Nerf|Gold Reduction', case=False).astype('int8')
        events_df['is_new_raid'] = events_df[target_col].str.contains('New Raid|Raid', case=False).astype('int8')
        events_df['is_t4'] = events_df[target_col].str.contains('T4|Tier 4', case=False).astype('int8')
        events_df['is_gem'] = events_df[target_col].str.contains('Gem', case=False).astype('int8')
        events_df['is_package'] = events_df[target_col].str.contains('Package', case=False).astype('int8')
        
    elif os.path.exists(official_path):
        print("Enriched data not found. Using basic official events.")
        events_df = pd.read_csv(official_path, names=['date', 'event_type', 'category', 'event_name', 'link'])
        events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')
        if 'category' in events_df.columns:
            events_df.rename(columns={'category': 'event_type'}, inplace=True)
        # Add dummy cols
        for col in ['is_gold_nerf', 'is_new_raid', 'is_t4', 'is_gem', 'is_package', 'sentiment_score', 'impact_score']:
            events_df[col] = 0
            
    else:
        print("Warning: No event data found.")
        events_df = pd.DataFrame(columns=['date', 'event_type', 'event_name'])
        
    print(f"Loaded {len(events_df)} events.")
        
    print("Loading Market Data (This may take a while)...")
    if not os.path.exists(MARKET_CSV):
        print("Market CSV not found.")
        return None, None

    # Optimize types to save memory
    dtypes = {
        'price': 'float32',
        'item_name': 'category'
    }
    
    # Read only necessary columns
    df = pd.read_csv(MARKET_CSV, usecols=['date', 'item_name', 'price'], dtype=dtypes)
    
    print("Parsing/Converting Dates...")
    # Handle mixed format dates (e.g. .000000000 suffix)
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    
    # Drop rows with invalid dates (if any)
    df.dropna(subset=['date'], inplace=True)
    
    # Normalize to midnight for Event Merging only, keep original time for analysis
    df['date_daily'] = df['date'].dt.normalize()
    
    # Do NOT aggregate to daily. Keep 5-min granularity.
    # df = df.groupby(['date', 'item_name'])['price'].mean().reset_index()
    
    return df, events_df

def feature_engineering(market_df, events_df):
    print("Engineering Features (High Resolution)...")
    
    # Merge Events onto Market Data using 'date_daily'
    # We want to know if a specific 5-min tick falls on an Event Day
    merged_df = market_df.merge(events_df, left_on='date_daily', right_on='date', how='left', suffixes=('', '_event'))
    
    # Drop redundant column
    if 'date_event' in merged_df.columns:
        merged_df.drop(columns=['date_event'], inplace=True)
        
    # Fill NaN for days with no events
    # Fill NaN for event features
    feature_cols = ['is_gold_nerf', 'is_new_raid', 'is_t4', 'is_gem', 'is_package', 'sentiment_score', 'impact_score']
    for col in feature_cols:
        if col in merged_df.columns:
            # fill metrics with 0
            if 'score' in col:
                 merged_df[col] = merged_df[col].fillna(0)
            else:
                 merged_df[col] = merged_df[col].fillna(0).astype('int8')

    merged_df['event_name'] = merged_df['event_name'].fillna('None')
    merged_df['event_type'] = merged_df['event_type'].fillna('None')
    # merged_df['event_score'] = merged_df['score'].fillna(0) # Removed as 'score' is not in official data
    
    # Intraday Features (Crucial for "7-8 PM Peak" analysis)
    print("Adding Intraday Features (Hour, DayOfWeek)...")
    merged_df['hour'] = merged_df['date'].dt.hour
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek # 0=Mon, 6=Sun
    
    # Sort for Shift calculation
    merged_df.sort_values(['item_name', 'date'], inplace=True)
    
    # Calculate Target: 7-Day ROI
    # To do this accurately efficiently, we should probably group by item.
    # Also, mixing 5-min data with 7-day ROI is noisy. 
    # Let's Agg to Daily first? User wants "Trend", daily is sufficient.
    # If we keep 35M rows, it's too big. Daily agg is cleaner for "Strategy".
    
    print("Aggregating to Daily for robust ROI calculation...")
    daily_df = merged_df.groupby(['item_name', 'date_daily']).agg({
        'price': 'mean',
        'is_gold_nerf': 'max',
        'is_new_raid': 'max',
        'is_t4': 'max',
        'is_gem': 'max',
        'is_package': 'max',
        'impact_score': 'max',
        'sentiment_score': 'min', # min/max? If negative sentiment exists, capture it. Let's take 'mean' or representative. 
                                  # Actually sentiment is -1, 0, 1. max/min might lose info. Let's take Mean.
        'event_type_id': 'first', # roughly
        'hour': 'first', # drop intraday
        'day_of_week': 'first'
    }).reset_index()
    
    # Fix Sentiment agg (rounding to nearest int for category, or keep as float score)
    # merged_df has sentiment_score.
    # Re-merge event info properly? 
    # Actually, simpler: merged_df already has daily event logic.
    
    # Calculate Future Price (7 Days later)
    # Shift(-7) on daily data? 
    # Need to ensure continuous dates. Reindexing might be needed but for MVP:
    # Just shift(-7) per item group.
    
    print("Calculating 7-Day Future Return...")
    daily_df['future_price_7d'] = daily_df.groupby('item_name')['price'].shift(-7)
    
    # ROI = (Future - Current) / Current
    daily_df['target_return_7d'] = (daily_df['future_price_7d'] - daily_df['price']) / daily_df['price']
    
    # Clean up
    daily_df.dropna(subset=['target_return_7d'], inplace=True)
    
    # Use this daily DF as result
    return daily_df

def main():
    market_df, events_df = load_data()
    if market_df is not None:
        print(f"Market Data Rows (Daily): {len(market_df)}")
        
        processed_df = feature_engineering(market_df, events_df)
        
        # Save processed data
        print(f"Saving to {OUTPUT_CSV}...")
        processed_df.to_csv(OUTPUT_CSV, index=False)
        print("Done!")

if __name__ == "__main__":
    main()
