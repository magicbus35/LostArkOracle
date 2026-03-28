import os
import re
import sys
import pandas as pd
import csv
from datetime import datetime, timedelta

# Add path to import sibling scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from collection.crawl_inven import find_best_reaction_post
from enrichment.analyze_events_llm import call_llm, clean_target_items

# Paths
BASE_DIR = os.path.dirname(parent_dir)
EVENTS_FILE = os.path.join(BASE_DIR, 'data', 'events_with_content.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_enriched.csv')

def extract_broadcast_time(content, title, anchor_date=None):
    """
    Extracts broadcast start time from text.
    Uses anchor_date (Notice Date) to infer year if missing.
    """
    # 1. Normalize
    text = (title + " " + content).replace('\n', ' ')
    
    # Analyze Header Year first (if passed)
    if anchor_date:
        if isinstance(anchor_date, str):
            try:
                base_dt = datetime.strptime(anchor_date, "%Y-%m-%d")
                base_year = base_dt.year
                base_month = base_dt.month
            except:
                base_year = datetime.now().year
                base_month = datetime.now().month
        else:
            base_year = anchor_date.year
            base_month = anchor_date.month
    else:
        base_year = datetime.now().year
        base_month = datetime.now().month

    # Regex 1: Explicit Date format YYYY.MM.DD HH:MM
    match = re.search(r'(\d{4})\.(\d{1,2})\.(\d{1,2}).*?(\d{1,2}):(\d{2})', text)
    if match:
        y, m, d, hh, mm = map(int, match.groups())
        return datetime(y, m, d, hh, mm)
        
    # Regex 2: Korean format (YYYY년) MM월 DD일 ... HH시
    # We capture the full string first to check for '오후'
    match = re.search(r'(?:(\d{4})년\s*)?(\d{1,2})월\s*(\d{1,2})일.*?(오전|오후)?\s*(\d{1,2})시', text)
    if match:
        y_str, m_str, d_str, ampm, hh_str = match.groups()
        m, d, hh = int(m_str), int(d_str), int(hh_str)
        
        # Handle PM
        if ampm == '오후' and hh < 12:
            hh += 12
        elif ampm == '오전' and hh == 12:
            hh = 0
        
        if y_str:
            y = int(y_str)
        else:
            # Infer Year from Anchor
            # If notice in Dec (12) mentions Jan (1), it's Next Year (Base+1)
            # If notice in Jan (1) mentions Dec (12), it's Last Year (Base-1) -> Rare for future events, but possible for recap
            # Standard case: Year is Base Year
            y = base_year
            
            if base_month == 12 and m == 1:
                y += 1
            elif base_month == 1 and m == 12:
                # If notice is Jan 2026 discussing Dec event, likely Dec 2025
                y -= 1
        
        return datetime(y, m, d, hh, 0)

    # Regex 3: "Next Week" relative patterns (Advanced - maybe later)
    # Regex 4: Simple "MM/DD HH:MM"
    match = re.search(r'(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})', text)
    if match:
        m, d, hh, mm = map(int, match.groups())
        y = base_year
        if base_month == 12 and m == 1: y += 1
        elif base_month == 1 and m == 12: y -= 1
        return datetime(y, m, d, hh, mm)

    return None

def main():
    if not os.path.exists(EVENTS_FILE):
        print("No events file found.")
        return

    # Load processed dates to avoid duplicates
    processed_dates = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if 'announcement_date' in existing_df.columns:
                # Ensure we capture YYYY-MM-DD format as strings, ONLY for Stream Summaries
                # We don't want a regular patch on 2026-01-02 to block the 2026-01-02 Broadcast summary.
                stream_mask = existing_df['event_type'] == 'Stream Summary'
                processed_dates = set(existing_df[stream_mask]['announcement_date'].astype(str).unique())
            print(f"DEBUG: Found {len(processed_dates)} already processed broadcast dates.")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    df = pd.read_csv(EVENTS_FILE)
    df = df[df['date'] >= '2026-01-01'] # HOTFIX: Process only recent 2026 data
    print(f"DEBUG: Loaded {len(df)} rows from {EVENTS_FILE}")
    
    # Filter for Broadcast keywords
    keywords = ['방송', '라이브', 'LOA ON', '로아온', '쇼케이스', '프리뷰']
    blacklist = ['당첨자', '리샤의 편지', '설정집', '모험가'] # Exclude these
    
    # Normalize titles to string
    df['title'] = df['title'].astype(str)
    
    # Apply Positive Filter
    broadcast_events = df[df['title'].apply(lambda x: any(k in x for k in keywords))]
    
    # Apply Negative Filter (Blacklist)
    broadcast_events = broadcast_events[~broadcast_events['title'].apply(lambda x: any(b in x for b in blacklist))]
    
    print(f"DEBUG: Found {len(broadcast_events)} potential broadcast notices (after blacklist).")
    
    for idx, row in broadcast_events.iterrows():
        title = row['title']
        content = str(row.get('content', ''))
        notice_date = row['date'] # "YYYY-MM-DD"
        
        # Extract Time with Anchor
        broadcast_sys_time = extract_broadcast_time(content, title, anchor_date=notice_date)
        
        if not broadcast_sys_time:
            print(f"Skipping '{title}': Could not parse broadcast time.")
            continue
            
        # IDEMPOTENCY CHECK
        broadcast_date_str = broadcast_sys_time.strftime("%Y-%m-%d")
        if broadcast_date_str in processed_dates:
            print(f"Skipping '{title}' ({broadcast_date_str}): Already processed.")
            continue
            
        print(f"\n[Processing] {title} @ {broadcast_sys_time}")
        
        # MANUAL LINK OVERRIDE CHECK
        # Load manual links if available to bypass crawler
        manual_links = {}
        manual_file = os.path.join(BASE_DIR, 'data', 'manual_links.csv')
        if os.path.exists(manual_file):
            try:
                m_df = pd.read_csv(manual_file)
                # Map date -> {title, link}
                # Date format in CSV must match broadcast_date_str (YYYY-MM-DD)
                for _, m_row in m_df.iterrows():
                    manual_links[str(m_row['target_date'])] = {'title': m_row['title'], 'link': m_row['link']}
            except:
                pass

        best_post = None
        if broadcast_date_str in manual_links:
            print(f"  [Override] Using Manual Link for {broadcast_date_str}")
            override_data = manual_links[broadcast_date_str]
            
            # Fetch Content for Analysis
            from collection.crawl_inven import fetch_post_content
            content = fetch_post_content(override_data['link'])
            
            best_post = {
                'title': override_data['title'],
                'link': override_data['link'],
                'content': content,
                'date': broadcast_date_str, # Use target date as proxy
                'score': 9999
            }
        else:
            # 1. Trigger Inven Crawler (6h window)
            print("  -> Hunting for Best Summary on Inven...")
            best_post = find_best_reaction_post(
                broadcast_sys_time.strftime("%Y-%m-%d %H:%M"), 
                window_hours=6,
                max_pages_to_scan=200
            )
        
        if best_post is None:
            print("  -> No summary found.")
            continue
            
        # 2. Analyze with LLM (Roadmap Decomposition)
        print(f"  -> Analyzing Summary: {best_post['title']}")
        
        # Context texts
        mechanisms_text = "Standard Rules" 
        context_text = "Previous streams..."

        ROADMAP_PROMPT = """
        Act as a Senior Economist for Lost Ark.
        You are analyzing a Developer Stream Summary to identify the **Roadmap of Future Updates**.
        
        ### TASK
        Break down the stream content into a **JSON List** of distinct economic events.
        1. **The Stream Itself** (Immediate Impact): Sentiment/Hype on the broadcast day.
        2. **Future Updates** (Roadmap Items): Specific milestones mentioned (e.g., "New Raid next month", "Tier 4 in Winter").
        
        ### MECHANISMS (Summary)
        - **Honing Materials**: Supply up (Express/Pass), Demand up (New Class/Reset).
        - **Gems**: Demand up (New Class). Supply up (Events).
        - **Gold**: Sink (Elixir/Transcendence), Inflation (Raid Gold Buff).
        
        ### OUTPUT FORMAT (JSON LIST)
        [
            {{
                "event_label": "Stream Hype / Summer Update Preview",
                "estimated_date": "{title}",  // Use the Stream Date for the immediate impact event
                "event_type": "Stream Summary",
                "impact_scores": {{ ... 12 scores ... }},
                "target_items": "..."
            }},
            {{
                "event_label": "New Raid: Kazeros",
                "estimated_date": "YYYY-MM-DD", // Best guess from text (e.g. 'Next Month' -> Add 30 days to stream date)
                "event_type": "Roadmap Item",
                "impact_scores": {{ ... 12 scores ... }},
                "target_items": "..."
            }}
        ]
        
        ### INPUT
        Stream Date: {title} (Use this as baseline for future dates)
        Content:
        {content}
        """

        try:
            # We pass the prompt template to the modified call_llm
            # Note: We must ensure the template uses {title}, {content} etc. matching the function signature.
            # But wait, call_llm injects mechanisms_text etc. 
            # Let's adjust the prompt variables to match what call_llm provides: {title}, {content}, {mechanisms_text}, {context_text}
            
            FINAL_TEMPLATE = """
            Act as a Senior Economist for Lost Ark.
            You are analyzing a Developer Stream Summary to identify the **Roadmap of Future Updates**.
            
            ### CONTEXT
            Stream Date/Title: {title}
            
            ### TASK
            Break down the stream content into a **JSON List** of distinct economic events.
            1. **The Stream Itself** (Immediate Impact): Sentiment/Hype on the broadcast day.
            2. **Future Updates** (Roadmap Items): Specific milestones mentioned (e.g., "New Raid next month", "Tier 4 in Winter").
            
            ### MECHANISMS
            {mechanisms_text}
            
            ### OUTPUT FORMAT (JSON LIST)
            [
                {{
                    "event_label": "Stream Name / Immediate Effect",
                    "estimated_date": "YYYY-MM-DD",  // The Stream Date
                    "event_type": "Stream Summary",
                    "honing_mat_supply": 0-10, "honing_mat_demand": 0-10,
                    "gem_supply": 0-10, "gem_demand": 0-10,
                    "engraving_supply": 0-10, "card_supply": 0-10,
                    "gold_inflation": 0-10, "gold_sink": 0-10,
                    "content_difficulty": 0-10, "package_volume": 0-10, "package_category": "None",
                    "trajectory_pattern": "raid_update", // raid_update, economy_shock, system_change, general_update
                    "target_items": "명예의 파편, 최상급 오레하 융화 재료", // MUST use exact Korean in-game names. No English.
                    "confidence_score": 8,
                    "mechanisms_applied": "Reasoning..."
                }},
                {{
                    "event_label": "Future Update Name",
                    "estimated_date": "YYYY-MM-DD", // Infer from text (e.g. 'September Update')
                    "event_type": "Roadmap Item",
                    "honing_mat_supply": 0-10, ... (same structure) ...
                }}
            ]
            
            ### CONTENT
            {content}
            """
            
            results = call_llm(
                title=f"{broadcast_sys_time.strftime('%Y-%m-%d')}", # Pass date as title for reference
                content=best_post['content'][:5000], 
                mechanisms_text=mechanisms_text, 
                context_text=context_text,
                prompt_template=FINAL_TEMPLATE
            )
            
            # Normalize to list
            if isinstance(results, dict):
                results = [results]
                
        except Exception as e:
            print(f"LLM Error: {e}")
            continue

        # 3. Save to Enrichment CSV
        modes = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
        
        fieldnames = ['date', 'event_type', 'title', 'link', 
                      'is_pre_announced', 'announcement_date', 
                      'honing_mat_supply', 'honing_mat_demand',
                      'gem_supply', 'gem_demand',
                      'engraving_supply', 'card_supply',
                      'gold_inflation', 'gold_sink',
                      'content_difficulty', 'package_volume', 'package_category',
                      'trajectory_pattern',
                      'target_items', 'confidence_score', 'mechanisms_applied']
                      
        with open(OUTPUT_FILE, modes, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if modes == 'w': writer.writeheader()
            
            for res in results:
                # Map LLM fields to CSV fields
                # Fallbacks for safety
                row_data = {
                    'date': res.get('estimated_date', broadcast_sys_time.strftime("%Y-%m-%d")),
                    'event_type': res.get('event_type', 'Stream Summary'),
                    'title': f"[{res.get('event_type','Event')}] {res.get('event_label', title)}", 
                    'link': best_post['link'],
                    
                    # Pre-announced Logic
                    # If it's the stream itself, it's NOT pre-announced (it's happening now).
                    # If it's a roadmap item, it IS pre-announced (announced today, happens later).
                    'is_pre_announced': True if res.get('event_type') == 'Roadmap Item' else False,
                    'announcement_date': broadcast_sys_time.strftime("%Y-%m-%d"),
                    
                    'honing_mat_supply': res.get('honing_mat_supply', 0),
                    'honing_mat_demand': res.get('honing_mat_demand', 0),
                    'gem_supply': res.get('gem_supply', 0),
                    'gem_demand': res.get('gem_demand', 0),
                    'engraving_supply': res.get('engraving_supply', 0),
                    'card_supply': res.get('card_supply', 0),
                    'gold_inflation': res.get('gold_inflation', 0),
                    'gold_sink': res.get('gold_sink', 0),
                    
                    'content_difficulty': res.get('content_difficulty', 0),
                    'package_volume': res.get('package_volume', 0),
                    'package_category': res.get('package_category', 'None'),
                    'trajectory_pattern': res.get('trajectory_pattern', 'general_update'),

                    'target_items': clean_target_items(res.get('target_items', 'None')),
                    'confidence_score': res.get('confidence_score', 0),
                    'mechanisms_applied': res.get('mechanisms_applied', 'Analysis')
                }
                writer.writerow(row_data)
                print(f"  -> Saved: {row_data['title']} ({row_data['date']})")

if __name__ == "__main__":
    main()
