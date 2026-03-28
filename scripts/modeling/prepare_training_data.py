import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
DATA_DIR = r"c:\LMO\data"
# Find the High Res File dynamically
high_res_files = [f for f in os.listdir(DATA_DIR) if f.startswith("고해상도_") and f.endswith(".csv")]
if high_res_files:
    # Use the latest one (lexicographically sorting YYYYMMDD works)
    target_file = sorted(high_res_files)[-1]
    MARKET_FILE = os.path.join(DATA_DIR, target_file)
    print(f"Using High-Resolution Market File: {target_file}")
else:
    print("Warning: High-Resolution file not found. Falling back to default.")
    MARKET_FILE = os.path.join(DATA_DIR, "lostark_market_data_all.csv")

EVENTS_FILE = os.path.join(DATA_DIR, "events_enriched.csv")
VOLUME_FILE = os.path.join(DATA_DIR, "lostark_volume_history.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "training_data.csv")

def load_events():
    print("Loading Events...")
    df = pd.read_csv(EVENTS_FILE)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['announcement_date'] = pd.to_datetime(df['announcement_date'], errors='coerce')
    
    # Filter valid dates
    df = df.dropna(subset=['date'])
    
    # We want to create a "Daily Event Feature" map
    # A day can have multiple events. We aggregate their scores.
    # Features: count, max_impact, is_update_day
    
    daily_events = df.groupby('date').agg({
        'title': 'count',
        'honing_mat_demand': 'max',
        'gem_demand': 'max',
        'gold_inflation': 'max',
        'content_difficulty': 'max'
    }).reset_index()
    
    daily_events.rename(columns={'title': 'event_count'}, inplace=True)
    daily_events['is_event_day'] = 1
    
    return daily_events

def process_market_data():
    print("Processing Market Data (Chunked)...")
    
    # Define aggregation logic
    # We will process in chunks, aggregate per chunk, then aggregate the results?
    # No, chunks might split the same day. 
    # Better to read, parse date (Day), and accumulate.
    # Since file is 1.6GB, we can't load all.
    # But usually 1.6GB fits in memory (pandas overhead ~5-10GB). 
    # If machine has 16GB RAM, it might be tight but possible.
    # Let's try explicit dtypes to save memory.
    
    chunk_size = 500000
    aggregated_chunks = []
    
    cols = ['date', 'item_name', 'price']
    
    for i, chunk in enumerate(pd.read_csv(MARKET_FILE, usecols=cols, chunksize=chunk_size)):
        if i % 10 == 0: print(f"Processing chunk {i}...")
        
        # Convert date to datetime then to date (floor)
        # Handle mixed formats (some might have nanoseconds)
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce', format='mixed')
        chunk.dropna(subset=['date'], inplace=True),
        chunk['day'] = chunk['date'].dt.date
        
        # Group by day, item_name
        agg = chunk.groupby(['day', 'item_name'])['price'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        aggregated_chunks.append(agg)
        
    print("Concatenating aggregated chunks...")
    all_aggs = pd.concat(aggregated_chunks)
    
    # Final aggregation (in case days were split across chunks)
    # Weighted average for mean is hard. Simplification: Just take mean of means (approx) or re-agg.
    # Re-aggregating:
    # We have mean, count. Global Mean = sum(mean * count) / sum(count)
    print("Finalizing aggregation...")
    all_aggs['price_sum'] = all_aggs['mean'] * all_aggs['count']
    
    final_df = all_aggs.groupby(['day', 'item_name']).agg({
        'price_sum': 'sum',
        'count': 'sum',
        'min': 'min',
        'max': 'max',
        'std': 'mean' # Approx std
    }).reset_index()
    
    final_df['price_mean'] = final_df['price_sum'] / final_df['count']
    final_df.drop(columns=['price_sum'], inplace=True)
    
    # Convert day back to datetime for merging
    final_df['date'] = pd.to_datetime(final_df['day'])
    final_df.drop(columns=['day'], inplace=True)
    
    return final_df

def load_volume():
    if not os.path.exists(VOLUME_FILE): return pd.DataFrame()
    print("Loading Volume History...")
    df = pd.read_csv(VOLUME_FILE)
    df['date'] = pd.to_datetime(df['date'])
    # Check if avg_price exists
    cols = ['date', 'item_name', 'trade_count']
    if 'avg_price' in df.columns:
        cols.append('avg_price')
    return df[cols]

def main():
    # 1. Load Data
    events_df = load_events()
    market_df = process_market_data()
    volume_df = load_volume()
    
    print(f"Market Data: {len(market_df)} rows")
    print(f"Event Data: {len(events_df)} days")
    
    # 2. Load and Prepare Volume Data (Official / Recent)
    volume_df = load_volume()
    if not volume_df.empty:
        # Rename col for compatibility
        volume_prices = volume_df.rename(columns={'avg_price': 'price_mean'})
        volume_prices['min'] = volume_prices['price_mean']
        volume_prices['max'] = volume_prices['price_mean']
        volume_prices['std'] = 0 # Missing volatility in daily avg
        
        # Merge Volume info into Market DF (Join on date/item)
        # But also, if market_df is missing dates that volume_df has, APPEND them.
        
        # 1. Merge Volume counts to existing records
        market_df = pd.merge(market_df, volume_df[['date', 'item_name', 'trade_count']], on=['date', 'item_name'], how='left')
        market_df['trade_count'] = market_df['trade_count'].fillna(0)
        
        # 2. Find new records in Volume DF that are NOT in Market DF
        # Create composite keys
        m_keys = set(zip(market_df['date'], market_df['item_name']))
        
        # Filter volume_prices
        new_records = []
        for idx, row in volume_prices.iterrows():
            if (row['date'], row['item_name']) not in m_keys:
                new_records.append(row)
        
        if new_records:
            print(f"Appending {len(new_records)} recent records from Official API...")
            new_df = pd.DataFrame(new_records)
            # Ensure columns 
            cols = ['date', 'item_name', 'price_mean', 'min', 'max', 'std', 'trade_count']
            market_df = pd.concat([market_df, new_df[cols]], ignore_index=True)
            
    else:
        print("No Volume/Official Data found.")
        market_df['trade_count'] = 0
    
    # 3. Merge Events
    print("Merging Events...")
    # Merge on Date (Event Day)
    full_df = pd.merge(market_df, events_df, on='date', how='left')
    
    # Fill NaNs for non-event days
    full_df['event_count'] = full_df['event_count'].fillna(0)
    full_df['is_event_day'] = full_df['is_event_day'].fillna(0)
    
    # 4. Feature Engineering
    print("Feature Engineering...")
    full_df['weekday'] = full_df['date'].dt.weekday
    full_df['is_weekend'] = full_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    full_df['is_wednesday'] = full_df['weekday'].apply(lambda x: 1 if x == 2 else 0) # Reset Day
    
    # 5. Save
    print(f"Saving to {OUTPUT_FILE}...")
    full_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
