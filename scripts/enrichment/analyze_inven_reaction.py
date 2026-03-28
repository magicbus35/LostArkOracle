import os
import sys
import json
import time
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import from sibling packages
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'scripts'))

from collection.crawl_inven import find_best_reaction_post

load_dotenv(os.path.join(BASE_DIR, '.env'))

# API Config
API_KEY = os.environ.get('PERPLEXITY_API_KEY')
API_URL = "https://api.perplexity.ai/chat/completions"
MECHANISMS_FILE = os.path.join(BASE_DIR, 'docs', 'mechanisms.md')

def load_mechanisms():
    if os.path.exists(MECHANISMS_FILE):
        with open(MECHANISMS_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def call_llm_reaction_analysis(post_title, post_content, mechanisms_text):
    if not API_KEY:
        return {"error": "No API Key"}

    prompt = f"""
    Act as a Data Engineer for Lost Ark (Korean Server).
    You are extracting structured features from a "Live Stream Summary Post" to feed a Price Prediction Model.
    
    ---
    ### KNOWLEDGE BASE (Economic Mechanisms)
    {mechanisms_text}
    ---

    ### INPUT: Summary Post
    Title: {post_title}
    Content:
    {post_content[:4000]} 
    
    ---
    ### TASK
    Extract Boolean/Categorical features indicating confirmed updates.
    
    1. **Gold Nerf**: Was a Gold Reward reduction announced for any raid?
    2. **New Raid**: Was a new Legion Raid or Abyssal Dungeon announced?
    3. **T4 / Reset**: Was a new Gear Tier (Tier 4) or Hard Reset announced?
    4. **Market Impact**: Based on the mechanism, predict if material prices will RISE or FALL.
    
    ### OUTPUT (JSON ONLY)
    {{
        "is_gold_nerf": true/false,
        "is_new_raid": true/false,
        "is_t4_update": true/false,
        "is_progression_event": true/false,
        "key_features": ["List", "of", "major", "features"],
        "predicted_market_direction": "Rise / Fall / Stable",
        "market_reasoning": "Brief explanation focused on supply/demand mechanics.",
        "confidence_score": <Integer 1-10>
    }}
    """
    
    payload = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": "You are a strategic game economist analyzing market sentiment."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, timeout=30)
        if response.status_code == 200:
            raw_content = response.json()['choices'][0]['message']['content'].strip()
            # Cleanup markdown code blocks if present
            raw_content = raw_content.replace('```json', '').replace('```', '')
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                return {"error": "JSON Parse Error", "raw": raw_content}
        else:
            return {"error": f"API Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def analyze_single_event_reaction(event_date, event_name="Custom Event", window_hours=4, max_pages=20):
    print(f"--- Analyzing Reaction for Event: {event_name} ({event_date}) ---")
    
    # 1. Find Best Summary Post
    best_post = find_best_reaction_post(event_date, window_hours=window_hours, max_pages_to_scan=max_pages)
    
    if not best_post:
        print("No suitable summary post found.")
        return
    
    print(f"Target Post Found: {best_post['title']}")
    print(f"Link: {best_post['link']}")
    
    # 2. LLM Analysis
    mechanisms = load_mechanisms()
    analysis = call_llm_reaction_analysis(best_post['title'], best_post['content'], mechanisms)
    
    # 3. Print Result
    print("\n--- LLM Analysis Result ---")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # 4. Save to File for Dashboard
    output_file = os.path.join(BASE_DIR, 'data', 'inven_analysis.json')
    history = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
            
    # Upsert based on date/link
    record = {
        "date": event_date,
        "event_name": event_name,
        "post_title": best_post['title'],
        "post_link": best_post['link'],
        "analysis": analysis,
        "analyzed_at": datetime.now().isoformat()
    }
    
    # Remove old record for same link if exists
    history = [h for h in history if h.get('post_link') != best_post['link']]
    history.append(record)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to {output_file}")
    
    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Event Date (YYYY-MM-DD HH:MM)", required=True)
    parser.add_argument("--name", help="Event Name", default="Live Stream")
    parser.add_argument("--pages", help="Max pages to scan", type=int, default=20)
    parser.add_argument("--window", help="Search window in hours", type=int, default=4)
    args = parser.parse_args()
    
    analyze_single_event_reaction(args.date, args.name, window_hours=args.window, max_pages=args.pages)
