import pandas as pd
import numpy as np
import os
import warnings

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKET_FILE = os.path.join(BASE_DIR, 'data', 'lostark_volume_history.csv')
MARKET_ALL_FILE = os.path.join(BASE_DIR, 'data', 'lostark_market_data_all.csv')
EVENTS_FILE = os.path.join(BASE_DIR, 'data', 'events_impact_analyzed.csv')
BETA_FILE = os.path.join(BASE_DIR, 'data', 'item_betas.csv')

def load_engravings():
    """Load and aggregate engraving data from raw market file."""
    print("Loading Raw Engraving Data...")
    if not os.path.exists(MARKET_ALL_FILE):
        return pd.DataFrame()
        
    chunks = []
    # Use chunksize to handle large file
    for chunk in pd.read_csv(MARKET_ALL_FILE, chunksize=100000):
        # Filter for Engravings
        eng_chunk = chunk[chunk['item_name'].str.contains('각인서', na=False)].copy()
        if not eng_chunk.empty:
            eng_chunk['date'] = pd.to_datetime(eng_chunk['date'], errors='coerce').dt.floor('D') # Normalize to Day
            chunks.append(eng_chunk)
            
    if not chunks:
        return pd.DataFrame()
        
    full_eng = pd.concat(chunks)
    # Aggregate: Avg Price per Day per Item
    agg_eng = full_eng.groupby(['date', 'item_name'])['price'].mean().reset_index()
    agg_eng.rename(columns={'price': 'avg_price'}, inplace=True)
    return agg_eng

def calculate_item_betas():
    print("Loading Data...")
    if not os.path.exists(MARKET_FILE) or not os.path.exists(EVENTS_FILE):
        print("Data files missing.")
        return

    # Load volume history (Materials/Gems)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        market_df = pd.read_csv(MARKET_FILE)
        market_df['date'] = pd.to_datetime(market_df['date'], errors='coerce')
        
    # Load Engravings
    engraving_df = load_engravings()
    if not engraving_df.empty:
        print(f"Loaded {len(engraving_df)} daily engraving records.")
        # Merge
        market_df = pd.concat([market_df, engraving_df], ignore_index=True)
        
    events_df = pd.read_csv(EVENTS_FILE)
    events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')

    # Drop invalid dates
    market_df = market_df.dropna(subset=['date'])
    events_df = events_df.dropna(subset=['date'])

    # --- FULL HISTORY ANALYSIS ---
    # We want to use ALL historical data (T3 + T4) to find patterns.
    # The mapping/translation will happen at the Dashboard level.
    
    print("Analyzing Full History...")
    
    # Filter Events
    events_filtered = events_df.copy()
    
    # Honing Demand Events
    honing_events = events_filtered[events_filtered['honing_mat_demand'] >= 7].copy()
    
    if honing_events.empty:
        print("No strong honing demand events found.")
        return
        
    print(f"Found {len(honing_events)} honing events across history.")
    
    # Define ALL Target Items (T3 + T4 + Engravings)
    target_items = [
        # Fusion Materials (T4)
        '아비도스 융화 재료', '상급 아비도스 융화 재료',
        # Fusion Materials (T3)
        '오레하 융화 재료', '상급 오레하 융화 재료', '최상급 오레하 융화 재료',
        
        # Leapstones (T4)
        '운명의 돌파석', '위대한 운명의 돌파석', 
        # Leapstones (T3)
        '위대한 명예의 돌파석', '경이로운 명예의 돌파석', '찬란한 명예의 돌파석',
        
        # Aux (T4)
        '빙하의 숨결', '용암의 숨결',
        # Aux (T3)
        '태양의 은총', '태양의 축복', '태양의 가호',
        '야금술 : 심화 [13-15]', '재봉술 : 심화 [13-15]',
        
        # Shards (T3/T4)
        '명예의 파편 주머니(소)', '명예의 파편 주머니(중)', '명예의 파편 주머니(대)',
        '운명의 파편 주머니(소)', '운명의 파편 주머니(중)', '운명의 파편 주머니(대)',
        
        # Stones (T3/T4)
        '파괴강석', '정제된 파괴강석', '수호강석', '정제된 수호강석',
        '운명의 파괴석', '운명의 수호석', '운명의 파괴석 결정', '운명의 수호석 결정',
        
        # Relic Engravings (T4/T5)
        '(유물)원한 각인서', '(유물)예리한 둔기 각인서', '(유물)돌격대장 각인서', 
        '(유물)아드레날린 각인서', '(유물)저주받은 인형 각인서', '(유물)타격의 대가 각인서',
        '(유물)각성 각인서', '(유물)전문의 각인서'
    ]
    
    item_stats = []
    
    for item in target_items:
        returns = []
        for _, event in honing_events.iterrows():
            evt_date = event['date']
            
            # Calculate Return (T+1 to T+14 Max)
            item_data = market_df[market_df['item_name'] == item]
            if item_data.empty: continue
            
            # Base price (T-1)
            base_rows = item_data[item_data['date'] == evt_date - pd.Timedelta(days=1)]
            if base_rows.empty: 
                # Try T0
                base_rows = item_data[item_data['date'] == evt_date]
                
            if base_rows.empty: continue
            base_price = base_rows['avg_price'].values[0]
            
            # Look ahead 14 days
            future_window = item_data[(item_data['date'] > evt_date) & (item_data['date'] <= evt_date + pd.Timedelta(days=14))]
            if future_window.empty: continue
            
            # We want MAX return in this window
            max_price = future_window['avg_price'].max()
            
            # Sustainability Filter:
            # User request: "Exclude momentary spikes, but keep sustained low-to-high jumps (e.g. 2g -> 8g)"
            if max_price > base_price * 1.5: # If spiked > 50%
                 window_avg = future_window['avg_price'].mean()
                 # Check if the AVERAGE price of the window sustained at least 25% of the spike's magnitude
                 # e.g. Base=2, Max=8 (Gain=6). Avg must be > 2 + 1.5 = 3.5. 
                 # If Avg is 2.1, it implies it was a 1-day spike and crashed.
                 required_maintenance = base_price + (max_price - base_price) * 0.25
                 if window_avg < required_maintenance:
                     continue # Filter out flash spikes
            
            if base_price > 0:
                ret = (max_price - base_price) / base_price * 100
                returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            # Win Rate: Chance of > 3% profit (tighter spread for high tiers)
            win_rate = sum(r > 3 for r in returns) / len(returns) * 100
            
            item_stats.append({
                'item_name': item,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sample_size': len(returns)
            })
            
    beta_df = pd.DataFrame(item_stats).sort_values('avg_return', ascending=False)
    print("Modern Meta Item Betas:")
    print(beta_df)
    
    beta_df.to_csv(BETA_FILE, index=False)
    print(f"Saved modern betas to {BETA_FILE}")

if __name__ == "__main__":
    calculate_item_betas()
