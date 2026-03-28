import pandas as pd
import os
import numpy as np
import warnings

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKET_FILE = os.path.join(BASE_DIR, 'data', 'lostark_market_data_all.csv')
EVENTS_FILE = os.path.join(BASE_DIR, 'data', 'events_enriched.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_impact_analyzed.csv')

def calculate_impact():
    print("Loading Data...")
    if not os.path.exists(MARKET_FILE) or not os.path.exists(EVENTS_FILE):
        print("Data files missing.")
        return

    # Load Market Data
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    # Use 'price' column from high-res data, aggregate to daily average
    market_df = pd.read_csv(MARKET_FILE)
    # Robust Date Parsing & Normalize to Midnight
    market_df['date'] = pd.to_datetime(market_df['date'], errors='coerce').dt.normalize()
    market_df = market_df.dropna(subset=['date'])
    
    # Deduplicate: Group by Date/Item and take Mean
    # Note: High-res data uses 'price', we map it to 'avg_price'
    market_df = market_df.groupby(['date', 'item_name'], as_index=False).agg({
        'price': 'mean',
        'timestamp': 'count' # Proxy for activity volume
    })
    market_df.rename(columns={'price': 'avg_price', 'timestamp': 'trade_count'}, inplace=True)
    
    # Pivot to Price Matrix (Date x Item)
    price_matrix = market_df.pivot(index='date', columns='item_name', values='avg_price')
    
    # Fill Missing Dates (Weekends/Holidays)
    full_idx = pd.date_range(start=market_df['date'].min(), end=market_df['date'].max(), freq='D')
    price_matrix = price_matrix.reindex(full_idx).ffill().bfill() # Continuous Price Series
    price_matrix.index.name = 'date'
    
    # Load Events
    events_df = pd.read_csv(EVENTS_FILE)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce').dt.normalize()
    
    events_df = events_df.dropna(subset=['date'])
    
    # Representative Items
    targets = {
        'Honing': ['위대한 명예의 돌파석', '경이로운 명예의 돌파석', '찬란한 명예의 돌파석', '정제된 파괴강석'],
        'Gem': ['10레벨 멸화의 보석', '9레벨 멸화의 보석', '7레벨 멸화의 보석'],
        'Engraving': ['원한 각인서', '예리한 둔기 각인서'] 
    }
    
    print("Calculating Granular Trajectories with Dynamic Lead Time...")
    
    # 1. Calculate Lead Time
    if 'announcement_date' in events_df.columns:
        events_df['announcement_date'] = pd.to_datetime(events_df['announcement_date'], errors='coerce')
        events_df['lead_time_days'] = (events_df['date'] - events_df['announcement_date']).dt.days
        events_df['lead_time_days'] = events_df['lead_time_days'].fillna(7).astype(int) 
    else:
        events_df['lead_time_days'] = 7
    
    # T-Minus & T-Plus Horizons
    horizons = [-30, -21, -14, -10, -7, -5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
    
    # Initialize Columns
    for h in horizons:
        col_suffix = f"T{h}" 
        events_df[f'Actual_Honing_{col_suffix}'] = 0.0
        events_df[f'Actual_Gem_{col_suffix}'] = 0.0
        events_df[f'Actual_Engraving_{col_suffix}'] = 0.0
    
    # Helper: Fast Lookup from Matrix
    def get_price_index(target_date, item_list):
        # 1. Check Date
        if target_date not in price_matrix.index:
            return 0.0
            
        # 2. Get Valid Items (Intersection)
        valid_items = [i for i in item_list if i in price_matrix.columns]
        if not valid_items: return 0.0
        
        # 3. Get Prices
        prices = price_matrix.loc[target_date, valid_items]
        
        # 4. Average (Composite Index)
        # Note: Mixing T3 and T4 prices might be weird if T4 is 100g and T3 is 10g.
        # Ideally we should pick the DOMINANT item.
        # But for now, Mean is acceptable as usually only one Tier is active/dominant per era.
        if prices.sum() == 0: return 0.0
        return prices.mean()

    # Iterate Events
    for idx, row in events_df.iterrows():
        dt = row['date']
        
        # Base Prices (At Event Day T=0)
        # Actually strictly, Base should be 'Start of Analysis'? 
        # Convention: T=100 usually means Price relative to T=0 (Event Day) OR T=Start (Lead Time).
        # Dashboard uses T=100 relative to... Wait.
        # The previous code plotted "Price Index (T=100)". 
        # If we normalize to T=0, then T=0 is 100.
        # Let's Normalize to T=0 price.
        
        if str(dt.date()) == '2022-03-30':
             print(f"DEBUG: Checking 2022-03-30. Index type: {type(dt)}")
             print(f"DEBUG: Price Matrix Index Sample: {price_matrix.index[:5]}")
             print(f"DEBUG: One item col in matrix? {'위대한 명예의 돌파석' in price_matrix.columns}")
             print(f"DEBUG: In Matrix? {dt in price_matrix.index}")
             dummy_price = get_price_index(dt, targets['Honing'])
             print(f"DEBUG: Honing Price: {dummy_price}")

        base_h = get_price_index(dt, targets['Honing'])
        base_g = get_price_index(dt, targets['Gem'])
        base_e = get_price_index(dt, targets['Engraving'])
        
        for h in horizons:
            target_dt = dt + pd.Timedelta(days=h)
            col = f"T{h}"
            
            # Lookup Target Price
            if base_h > 0:
                p = get_price_index(target_dt, targets['Honing'])
                if p > 0: 
                    events_df.at[idx, f'Actual_Honing_{col}'] = round((p / base_h) * 100, 1)
            
            if base_g > 0:
                p = get_price_index(target_dt, targets['Gem'])
                if p > 0: 
                    events_df.at[idx, f'Actual_Gem_{col}'] = round((p / base_g) * 100, 1)
                    
            if base_e > 0:
                p = get_price_index(target_dt, targets['Engraving'])
                if p > 0: 
                    events_df.at[idx, f'Actual_Engraving_{col}'] = round((p / base_e) * 100, 1)
            
    events_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Updated {OUTPUT_FILE} with Pivot-Accelerated Matching.")
            
    # Legacy Support (T+7 as default) for backward compatibility if needed
    events_df['Actual_Honing_Return'] = events_df['Actual_Honing_T7']
    events_df['Actual_Gem_Return'] = events_df['Actual_Gem_T7']
            
    # Save
    events_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved multi-horizon analysis to {OUTPUT_FILE}")
    print(events_df[['date', 'Actual_Honing_T1', 'Actual_Honing_T7', 'Actual_Honing_T30']].head(5))

if __name__ == "__main__":
    calculate_impact()
