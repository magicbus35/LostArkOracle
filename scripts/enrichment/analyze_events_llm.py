import csv
import os
import requests
import time
import json
import pandas as pd
from dotenv import load_dotenv

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_with_content.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_enriched.csv') # Standard filename
MECHANISMS_FILE = os.path.join(BASE_DIR, 'docs', 'mechanisms.md')

# API Config
API_KEY = os.environ.get('PERPLEXITY_API_KEY')
API_URL = "https://api.perplexity.ai/chat/completions"

def load_mechanisms():
    if os.path.exists(MECHANISMS_FILE):
        with open(MECHANISMS_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def analyze_events():
    if not API_KEY:
        print("Error: PERPLEXITY_API_KEY environment variable not set.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    mechanisms_text = load_mechanisms()
    
    # Load existing processed links
    processed_links = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_links.add(row['link'])

    # Read input
    events = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)

    # Load Stream Context (Inven Posts)
    stream_df = None
    stream_file = os.path.join(BASE_DIR, 'data', 'inven_posts.csv')
    if os.path.exists(stream_file):
        try:
            stream_df = pd.read_csv(stream_file)
            stream_df['date'] = pd.to_datetime(stream_df['date'])
        except Exception as e:
            print(f"Warning: Could not load stream context: {e}")

    print(f"Total events: {len(events)}. Processed: {len(processed_links)}")

    # Open output
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    with open(OUTPUT_FILE, mode, encoding='utf-8', newline='') as f:
        # detailed strategic output
        fieldnames = ['date', 'event_type', 'title', 'link', 
                      'is_pre_announced', 'announcement_date', 
                      'honing_mat_supply', 'honing_mat_demand',
                      'gem_supply', 'gem_demand',
                      'engraving_supply', 'card_supply',
                      'gold_inflation', 'gold_sink',
                      'content_difficulty', 'package_volume', 'package_category',
                      'trajectory_pattern',
                      'target_items', 'confidence_score', 'mechanisms_applied']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        from datetime import datetime
        
        for event in events:
            link = event['link']
            
            # Enforce Cache: Skip if already processed
            if link in processed_links:
                continue
            
            # Filter
            if event['event_type'] not in ['Update', 'Major Update', 'Event', 'Official Notice']:
                 continue

            content = event.get('content', '')
            if len(content) < 50 or content == "Content not found":
                continue # Skip empty

            # Get Context
            event_date_str = event['date']
            context_text = "None"
            
            if stream_df is not None:
                try:
                    current_dt = pd.to_datetime(event_date_str)
                    # Look back 60 days
                    start_dt = current_dt - pd.Timedelta(days=60)
                    mask = (stream_df['date'] >= start_dt) & (stream_df['date'] <= current_dt)
                    recent_streams = stream_df.loc[mask].sort_values('date', ascending=False).head(3)
                    
                    if not recent_streams.empty:
                        context_list = []
                        for _, row in recent_streams.iterrows():
                            context_list.append(f"- [{row['date'].strftime('%Y-%m-%d')}] {row['title']} (Score: {row.get('score', 0)})")
                        context_text = "\n".join(context_list)
                except Exception as e:
                    print(f"Context Error: {e}")

            # LLM Call
            # Reduce content to 2000 to avoid token limits/bad request
            result = call_llm(event['title'], content[:2000], mechanisms_text, context_text)

            # HYBRID OVERRIDE: Check for Hidden Triggers via Keywords (Python-side enforcement)
            # Apply to ALL types to be safe
            keywords_supply = ['보상', '지급', '선물', '기념'] # Reward, Sent, Gift, Celebration
            keywords_demand = ['회수', '조치', '버그 악용', '랭킹'] # Recovery, Action, Exploit
            
            # Debug Content (First 100 chars)
            # print(f"DEBUG Check: {event['title']} - Content: {content[:50]}...")
            
            # Check Supply (Compensation)
            if any(k in content[:2000] for k in keywords_supply):
                # Force create keys if missing due to API error
                if 'honing_mat_supply' not in result: result['honing_mat_supply'] = 0
                
                if result.get('honing_mat_supply', 0) < 5:
                    print(f"  [Override] Detected 'Supply' keywords -> Boosting Supply Score to 8.")
                    result['honing_mat_supply'] = max(result.get('honing_mat_supply', 0), 8)
                    result['mechanisms_applied'] = str(result.get('mechanisms_applied', '')) + " | Python_Keyword_Override_Supply"
                    
            # Check Demand/Correction (Recovery)
            if any(k in content[:2000] for k in keywords_demand):
                 if 'honing_mat_demand' not in result: result['honing_mat_demand'] = 0
                 
                 if result.get('honing_mat_demand', 0) < 5:
                    print(f"  [Override] Detected 'Demand' keywords -> Boosting Demand Score to 8.")
                    result['honing_mat_demand'] = max(result.get('honing_mat_demand', 0), 8)
                    result['mechanisms_applied'] = str(result.get('mechanisms_applied', '')) + " | Python_Keyword_Override_Demand"

            print(f"[{event['title']}] Announced: {result.get('is_pre_announced')} | HoningS: {result.get('honing_mat_supply')} | GemD: {result.get('gem_demand')}")
            
            writer.writerow({
                'date': event['date'],
                'event_type': event['event_type'],
                'title': event['title'],
                'link': link,
                'is_pre_announced': result.get('is_pre_announced', False),
                'announcement_date': result.get('announcement_date', 'None'),
                
                'honing_mat_supply': result.get('honing_mat_supply', 0),
                'honing_mat_demand': result.get('honing_mat_demand', 0),
                'gem_supply': result.get('gem_supply', 0),
                'gem_demand': result.get('gem_demand', 0),
                'engraving_supply': result.get('engraving_supply', 0),
                'card_supply': result.get('card_supply', 0),
                'gold_inflation': result.get('gold_inflation', 0),
                'gold_sink': result.get('gold_sink', 0),
                
                'content_difficulty': result.get('content_difficulty', 0),
                'package_volume': result.get('package_volume', 0),
                'package_category': result.get('package_category', 'None'),
                'trajectory_pattern': result.get('trajectory_pattern', 'general_update'),

                'target_items': clean_target_items(result.get('target_items', 'None')),
                'confidence_score': result.get('confidence_score', 0),
                'mechanisms_applied': result.get('mechanisms_applied', 'None')
            })
            f.flush()
            time.sleep(1)

import sys

# Add scripts root to path to verify imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.market_validator import is_tradeable_api

def clean_target_items(items_str):
    if not items_str or items_str == 'None':
        return 'None'
    
    # 1. Fast Blacklist (Save API Calls)
    forbidden = ['카드', 'Card', '계승', '쐐기', '눈', '나팔', '관문', '경험치', '실링', '귀속', '캐릭터', '슬롯']

    valid_items = []
    
    for item in items_str.split(','):
        item = item.strip()
        if not item: continue
        
        # Check Blacklist
        is_bad = False
        for f in forbidden:
            if f in item:
                is_bad = True
                break
        
        if is_bad:
            continue
            
        # 2. API Verification
        # Only verify if it looks like a real item (not empty/weird)
        if is_tradeable_api(item):
            valid_items.append(item)
        else:
            # Maybe it's a generic name "Leapstones" vs specific "Radiant Leapstone" explains failure.
            # But the user wants STRICT filtering. So if not in market -> remove.
            pass
            
    return ", ".join(valid_items) if valid_items else "None"

def call_llm(title, content, mechanisms_text, context_text, prompt_template=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    if prompt_template:
        # Use provided template, inject variables if they exist in string format
        # Simple f-string won't work if passed as a string object, so we use .format
        # But handling all vars is safe.
        prompt = prompt_template.format(
            title=title, 
            content=content, 
            mechanisms_text=mechanisms_text, 
            context_text=context_text
        )
    else:
        # Default Prompt
        prompt = f"""
        Act as a Senior Economist for Lost Ark (Korean Server).
        You are analyzing a past game update to predict market movements AS OF THAT DATE.
        Do NOT use future knowledge. Use only the provided text and mechanism rules.
    
        ---
        ### KNOWLEDGE BASE (Economic Mechanisms)
        {mechanisms_text}
        ---
        
        ### RECENT STREAM CONTEXT (Fact Check Only)
        Check if this update was mentioned in these recent streams.
        {context_text}
        ---
    
        ### TASKS (Role: Micro-Economic Analyst)
        Analyze the update and output **GRANULAR IMPACT SCORES (0-10)** for specific item categories.
        
        **A. Honing Materials (Destruction/Guardian/Leapstones/Auxiliaries)**
        - `honing_mat_supply`: Events giving mats, Frog, Shop.
        - `honing_mat_demand`: New Character, Hyper Express (Critical), Soft Reset. 
          *CRITICAL*: Auxiliary items (Metallurgy/Tailoring/Solar) are EXTREMELY sensitive to Express events. If Hyper Express -> Score 9-10.
    
        **B. Advanced Spec (Gems / Accessories)**
        - `gem_supply`: Event Gems, Cube Ticket Increase.
        - `gem_demand`: New Class Release (Score 8-10 allowed), Tier Expansion.
          *NOTE*: For general updates or Express events NOT involving a new class, Gem Demand is usually moderate (Score 3-5). Do NOT overestimate.
        
        **C. Books & Cards (Engravings / Cards)**
        - `engraving_supply`: **Gold Toad (High Impact)**, Event Books.
        - `card_supply`: Card Packs in Event/Shop.
    
        **D. Gold Economy (Currency)**
        - `gold_inflation`: Raid Gold Buff, Free Gold.
        - `gold_sink`: **Systems burning gold** (Elixir, Transcendence, Quality).
    
        **F. Trajectory Pattern Classification**
        Classify the event into one of these price movement patterns for ML modeling:
        - `raid_update`: New Content. Pattern: Announce -> Update(Spike) -> Weekend(Peak) -> Saturation.
        - `economy_shock`: Shop/Gold Toad/Package. Pattern: Release(Impact) -> 3 Days(Reaction) -> End Date(Recovery).
        - `system_change`: Reset/Nerf. Pattern: Announce(Panic) -> Update -> Long-term Stabilization.
        - `general_update`: Bug fix/Small patch. Pattern: Low Impact.
        
        **G. HIDDEN TRIGGERS (CRITICAL for "Notices")**
        - If the input is an "Official Notice" or "Known Issues":
          - Look for **"Compensation" (보상)**, **"Sent" (지급)**, **"Recovery" (회수)**.
          - Example: "We sent compensation for the bug." -> This is a SUPPLY Event. Score `honing_mat_supply` or `gem_supply` accordingly (5-8).
          - Example: "We are recovering exploits." -> This is a DEMAND/Correction Event.
          - **DO NOT SCORE 0** if there is explicit mention of items being given to players.

        ---
        ### INPUT UPDATE
        Title: {title}
        Content Snippet:
        {content}
    
        ---
        ### OUTPUT (JSON ONLY)
        {{
            "mechanisms_applied": "Rule used. (MUST BE WRITTEN IN KOREAN language unconditionally. Ex: '신규 레이드 출시에 따른 수요 증가 예상')",
            "is_pre_announced": <Boolean>,
            "announcement_date": "YYYY-MM-DD or None",
            
            "honing_mat_supply": <int 0-10>,
            "honing_mat_demand": <int 0-10>,
            
            "gem_supply": <int 0-10>,
            "gem_demand": <int 0-10>,
            
            "engraving_supply": <int 0-10>,
            "card_supply": <int 0-10>,
            
            "gold_inflation": <int 0-10>,
            "gold_sink": <int 0-10>,
            
            "content_difficulty": <int 0-10>,
            "package_volume": <int 0-10>,
            "package_category": "String",
            
            "trajectory_pattern": "raid_update" | "economy_shock" | "system_change" | "general_update",

            "target_items": "Comma-separated list (KOREAN ONLY, e.g. 명예의 파편, 10레벨 멸화의 보석).",
            "confidence_score": <int 1-10>
        }}
        """
    
    payload = {
        "model": "sonar-pro", 
        "messages": [
            {"role": "system", "content": "You are a rigid output machine. Output ONLY valid JSON. No Markdown. No Intro."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            raw_content = response.json()['choices'][0]['message']['content'].strip()
            raw_content = raw_content.replace('```json', '').replace('```', '')
            try:
                # Robust extraction using Regex
                # Robust extraction using Regex
                import re
                # Match either {...} OR [...] (dotall for multiline)
                json_match = re.search(r'(\{.*\}|\[.*\])', raw_content, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(0)
                    return json.loads(clean_json)
                else:
                     # Fallback if no braces found
                    return json.loads(raw_content)
            except json.JSONDecodeError:
                print(f"JSON Parse Error. Raw: {raw_content}")
                return {"mechanisms_applied": "Error", "strategy_analysis": "JSON Parse Error", "peak_timing": "Error", "target_items": "None", "confidence_score": 0}
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return {"mechanisms_applied": "Error", "strategy_analysis": f"API Error {response.status_code}", "peak_timing": "Error", "target_items": "None", "confidence_score": 0}
    except Exception as e:
        print(f"Exception during API call: {e}")
        return {"mechanisms_applied": "Error", "strategy_analysis": f"Exception {str(e)}", "peak_timing": "Error", "target_items": "None", "confidence_score": 0}

if __name__ == "__main__":
    analyze_events()
if __name__ == "__main__":
    analyze_events()
