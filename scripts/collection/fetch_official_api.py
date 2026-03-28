import requests
import os
import json
import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'lostark_volume_history.csv')

# API Config
API_KEY = os.environ.get('LOSTARK_API_KEY')
API_URL = "https://developer-lostark.game.onstove.com/markets/items"

# Categories to Scrape
# 50000: Upgrade Materials (Destruction, Guardian, Leapstones, Shards, Fusion)
# 51000: Battle Items (Potions, Bombs) - Optional, mainly 50000 is key.
# 51100: Food - Optional
# 40000: Engraving Recipes - Good to have.
# 210000: Gems - Crucial for full market analysis.
# Update: Switched to fetch all 3 core categories (Honing, Gems, Engravings)
TARGET_CATEGORIES = [50000, 210000, 40000]

def fetch_market_data():
    if not API_KEY:
        print("Error: LOSTARK_API_KEY not found in .env")
        return

    headers = {
        'accept': 'application/json',
        'authorization': f'bearer {API_KEY}',
        'content-type': 'application/json'
    }

    all_items = []

    # 1. Discover ALL Items in Categories via Pagination
    print("--- 1. Discovering Items ---")
    for cat_code in TARGET_CATEGORIES:
        page_no = 1
        while True:
            payload = {     
                "Sort": "CURRENT_MIN_PRICE",
                "CategoryCode": cat_code,
                "ItemTier": 0, # All Tiers
                "ItemGrade": "",
                "ItemName": "",
                "PageNo": page_no,
                "SortCondition": "ASC"
            }
            
            try:
                # Use SEARCH endpoint to find items
                url = "https://developer-lostark.game.onstove.com/markets/items"
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('Items', [])
                    
                    if not items:
                        print(f"Category {cat_code}: No more items at page {page_no}.")
                        break
                    
                    print(f"Category {cat_code} Page {page_no}: Found {len(items)} items.")
                    
                    for item in items:
                        # Collect ID and Name
                        # Filter T3/T4 for relevance if needed, but user said "All".
                        # We'll stick to T3+ to save requests? Or just all. 
                        # Let's just take all for now, maybe T3+ is safer for rate limits.
                        if item['Grade'] in ['일반', '고급', '희귀', '영웅', '전설', '유물', '고대']: 
                             all_items.append({'Id': item['Id'], 'Name': item['Name']})

                    page_no += 1
                    time.sleep(1.1) # Rate limit safe (60 req/min typically)
                else:
                    print(f"Search Error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                print(f"Search Loop Exception: {e}")
                break

    print(f"Total Items Found: {len(all_items)}")
    
    # Deduplicate by ID
    unique_items = {item['Id']: item for item in all_items}.values()
    print(f"Unique Items to Fetch: {len(unique_items)}")

    # 2. Fetch History for each ID
    print("--- 2. Fetching History ---")
    
    # Check if we need to write header
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        fields = ['item_name', 'date', 'avg_price', 'trade_count']
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for idx, item in enumerate(unique_items):
            item_id = item['Id']
            name = item['Name']
            
            url = f"https://developer-lostark.game.onstove.com/markets/items/{item_id}"
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # data format: [{"Name": "...", "Stats": [...]}]
                    if data:
                        stats = data[0].get('Stats', [])
                        print(f"[{idx+1}/{len(unique_items)}] {name}: {len(stats)} days history.")
                        
                        for stat in stats:
                            writer.writerow({
                                'item_name': name,
                                'date': stat['Date'],
                                # API key naming might differ slightly, checking previous working ver
                                'avg_price': stat['AvgPrice'],
                                'trade_count': stat['TradeCount']
                            })
                        f.flush()
                    else:
                        print(f"[{idx+1}/{len(unique_items)}] {name}: No history data.")
                else:
                    print(f"History Error {name} ({response.status_code})")
                    if response.status_code == 429:
                        print("Rate Limit Hit! Sleeping 60s...")
                        time.sleep(60)
            
            except Exception as e:
                print(f"History Exception {name}: {e}")
            
            time.sleep(1.2) # Conservative rate limit (50 req/min)

if __name__ == "__main__":
    fetch_market_data()
