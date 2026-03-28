import os
import json
import requests
import time
from dotenv import load_dotenv

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, '.env')
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'tradeable_cache.json')

load_dotenv(ENV_PATH)
API_KEY = os.environ.get('LOSTARK_API_KEY')

# In-memory cache
_cache = {}

def load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                _cache = json.load(f)
        except:
            _cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Cache save failed: {e}")

# Initial load
load_cache()

def is_tradeable_api(item_name):
    """
    Checks if item exists in Market via Official API.
    Returns: True (Tradeable), False (Not found/Untradeable)
    """
    if not item_name or item_name == 'None':
        return False
        
    # Check Cache
    if item_name in _cache:
        return _cache[item_name]
    
    if not API_KEY:
        print("Warning: LOSTARK_API_KEY missing. Returning False (Fail-safe).")
        return False

    url = "https://developer-lostark.game.onstove.com/markets/items"
    headers = {
        'accept': 'application/json',
        'authorization': f'bearer {API_KEY}',
        'content-type': 'application/json'
    }
    
    payload = {
        "Sort": "CURRENT_MIN_PRICE",
        "CategoryCode": 0, # All
        "ItemTier": 0,
        "ItemName": item_name,
        "PageNo": 1,
        "SortCondition": "ASC"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # If TotalCount > 0, it's tradeable
            # Some items might have same name but different entity, but usually if it shows up in Market, it IS tradeable.
            # However, exact match is better. The search is 'contains'.
            # We will assume if ANY item matches the name, it's tradeable.
            is_valid = data.get('TotalCount', 0) > 0
            
            _cache[item_name] = is_valid
            save_cache() # Save immediately or periodically
            time.sleep(0.1) # Rate limit protection
            return is_valid
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"API Exception: {e}")
        return False

if __name__ == "__main__":
    # Test
    print(f"Leapstone: {is_tradeable_api('정제된 파괴강석')}")
    print(f"Bound Item: {is_tradeable_api('귀속된 파괴강석')}")
    print(f"Card: {is_tradeable_api('세상을 구하는 빛 카드')}")
