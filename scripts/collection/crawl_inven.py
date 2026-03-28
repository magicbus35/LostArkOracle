import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
from datetime import datetime, timedelta
import re
import argparse
import sys

# Configuration
# BOARD_IDS = ["4811", "6271"] # 4811 is stale/broken on Page 1 (July 2025?)
BOARD_IDS = ["6271"] # Focus on Tips/Info (High Quality, Active)
BASE_URL_TEMPLATE = "https://www.inven.co.kr/board/lostark/{board_id}"
# my=chuchu : 10+ Recommended Posts (The "10-Choo" board)
PARAMS = {"my": "chuchu"} 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'inven_posts.csv')

# Strict Keyword Filtering System
# ... (Keywords remain same)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'inven_posts.csv')

# Strict Keyword Filtering System
EVENT_KEYWORDS = ['로아온', '라방', '방송', '라이브', '디렉터', '쇼케이스', '특별', '공지', '소통', '금강선', '전재학']
TYPE_KEYWORDS = ['요약', '정리', '분석', '후기', '내용', '중계', '속보', '피셜']

def clean_text(text):
    return text.strip().replace('\n', ' ').replace('\t', ' ')

def parse_inven_date(date_str, reference_year=None):
    """
    Parses Inven date string into datetime object.
    Formats: 'HH:mm' (Today), 'MM-DD' (Current Year), 'YYYY-MM-DD' (Older)
    reference_year: Year hint for MM-DD dates.
    """
    now = datetime.now()
    if reference_year is None: 
        reference_year = now.year

    try:
        if ':' in date_str and '-' not in date_str: # HH:mm -> Today
            hour, minute = map(int, date_str.split(':'))
            return datetime(now.year, now.month, now.day, hour, minute)
        elif re.match(r'\d{2}-\d{2}', date_str): # MM-DD -> Implicit Year
            month, day = map(int, date_str.split('-'))
            # Heuristic: If we are in Jan 2026, and date is 12-20, it's probably 2025.
            # But let's respect reference_year if provided.
            dt = datetime(reference_year, month, day)
            if dt > now + timedelta(days=1): # Future date? Must be last year
                 dt = datetime(reference_year - 1, month, day)
            return dt
        elif re.match(r'\d{4}-\d{2}-\d{2}', date_str): # YYYY-MM-DD
            return datetime.strptime(date_str, "%Y-%m-%d")
        else:
            return now # Fallback
    except:
        return now

def fetch_post_content(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.select_one('#powerbbsContent')
        if content_div:
            return clean_text(content_div.get_text())
        return "Content not found"
    except:
        return "Error"

def get_page_date_range(board_id, page, upper_bound_date=None):
    """
    Returns (latest, oldest) dates on the page for a specific board.
    """
    try:
        url = BASE_URL_TEMPLATE.format(board_id=board_id)
        params = PARAMS.copy()
        params['p'] = page
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.select('.board-list tr')
        
        dates = []
        if upper_bound_date is None:
            upper_bound_date = datetime.now()

        for row in rows:
            # Skip Notices (Sticky posts)
            # Inven structure: Notices often have 'notice' class OR they are in a different section.
            # But safer check: check for 'notice' in class list of TR
            classes = row.get('class', [])
            if 'notice' in classes:
                continue
                
            # Additional Check: Some notices don't have 'notice' class but have a specific icon or structure.
            # Usually notices have a strongly typed '공지' text in the category or title.
            # Let's check the date specifically.
            
            date_tag = row.select_one('.date')
            if date_tag:
                date_str = date_tag.get_text(strip=True)
                dt = parse_inven_date(date_str)
                
                # Check for pinned post anomalies:
                # If page > 50 and date is Today/Tomorrow/Future, it's definitely a Pinned Post/Ad.
                if page > 50 and dt > datetime.now() - timedelta(days=7):
                    # Likely a sticky info post
                    continue
                    
                # Enforce Monotonicity locally?
                # No, just collect all valid dates.
                
                # Enforce Monotonicity (Sanity check against upper bound)
                # But upper_bound might be loose.
                # Just appending for now.
                
                while dt > upper_bound_date + timedelta(days=1):
                     dt = dt.replace(year=dt.year - 1)
                     
                dates.append(dt)
        
        if not dates:
            return None, None
            
        return max(dates), min(dates)
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        return None, None

def find_start_page(board_id, target_dt):
    """
    Finds the start page using Dynamic Binary Search with Robust Year Inference.
    Goal: Find a page where: latest_date >= target_dt >= oldest_date.
    """
    print(f"Searching Board {board_id} for {target_dt.date()} using Binary Search...")
    
    # Track known dates: [Page] -> (Latest, Oldest)
    known_dates = {}
    
    # --- Phase 1: Bracketing (Exponential Expansion) ---
    low = 1
    high = 10
    limit_reached = False
    
    # For Bracketing, we always go deeper, so previous page's oldest is the valid upper bound.
    current_upper_bound = datetime.now()
    known_dates[0] = (current_upper_bound, current_upper_bound) # Dummy Page 0
    
    while True:
        # Check High bound
        latest, oldest = get_page_date_range(board_id, high, upper_bound_date=current_upper_bound)
        
        if not latest:
            # High went past end of board.
            print(f"  [Bracket] Page {high} is empty/out of bounds. Stopping expansion.")
            limit_reached = True
            break
        
        known_dates[high] = (latest, oldest)
        print(f"  [Bracket] Check Page {high}: {latest.date()} ~ {oldest.date()} (Bound: {current_upper_bound.date()})")
        
        # Prepare for next deeper step
        if oldest < current_upper_bound:
             current_upper_bound = oldest
        
        if target_dt >= oldest:
            # Found range: [Low, High]
            print(f"  [Bracket] Found range: Page {low} ~ {high}")
            break
            
        # Target is older -> Go deeper
        low = high
        high *= 2
        
        if high > 3000: 
            high = 3000
            break

    # --- Phase 2: Binary Search ---
    best_guess = low
    
    print(f"  [Binary] Entering Bisection {low}~{high}")

    while low <= high:
        mid = (low + high) // 2

        # KEY FIX: Upper Bound for 'Mid' should come from the SHALLOWER side ('Low').
        closest_shallower_page = max([p for p in known_dates.keys() if p < mid], default=0)
        context_upper_bound = known_dates[closest_shallower_page][1] # Oldest date of shallower page

        latest, oldest = get_page_date_range(board_id, mid, upper_bound_date=context_upper_bound)
        
        if not latest:
            high = mid - 1
            continue

        known_dates[mid] = (latest, oldest)
        print(f"  [Binary] Check Page {mid}: {latest.date()} ~ {oldest.date()} (Context Page {closest_shallower_page}: {context_upper_bound.date()})")
        
        if oldest <= target_dt <= latest:
            print(f"  -> Target found on Page {mid}!")
            return mid
            
        if target_dt > latest:
            # Target is NEWER -> Need Lower Page Number
            high = mid - 1
        elif target_dt < oldest:
            # Target is OLDER -> Need Higher Page Number
            low = mid + 1
            best_guess = low
            
    print(f"  -> Exact match missed. Best guess: Page {best_guess}")
    return best_guess 

def find_best_reaction_post(target_date_str, window_hours=4, max_pages_to_scan=20):
    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        # Fallback if only date provided
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")

    end_dt = target_dt + timedelta(hours=window_hours)
    
    all_candidates = []
    
    for board_id in BOARD_IDS:
        print(f"\n--- Scanning Board {board_id} ---")
        
        # Smart Seek Start Page
        start_page = find_start_page(board_id, target_dt)
        start_page = max(1, start_page - 2)
        
        print(f"Starting detailed scan from Page {start_page}...")
        
        # Scan pages
        for page in range(start_page, start_page + max_pages_to_scan):
            try:
                url = BASE_URL_TEMPLATE.format(board_id=board_id)
                params = PARAMS.copy()
                params['p'] = page
                response = requests.get(url, params=params, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                rows = soup.select('.board-list tr')
                
                if not rows: break
                print(f"Scanning Page {page}...")
                
                for row in rows:
                    subject_tag = row.select_one('.subject-link')
                    if not subject_tag: continue
                    title = subject_tag.get_text(strip=True)
                    
                    # --- TWO-STEP SCORING LOGIC (Step 1 Only) ---
                    # User Request: "1. Score Title first. If passed, THEN check recommendations."
                    # Implementation: Calculate Title Score. If < 100, reject immediately.
                    
                    def calculate_title_score(text):
                        score = 0
                        text_lower = text.lower()
                        
                        # Context: Must be Broadcast/Event related
                        # Expanded keywords based on user feedback (e.g. '로아온', '방송')
                        stream_keywords = ['로아온', '방송', '라이브', '디렉터', '쇼케이스', '전재학', '금강선', '특별', '공지', 'preview', 'on air', '기록', '속보', '캠프']
                        
                        # Type: Must be Summary/Report
                        # Expanded type keywords
                        type_keywords = ['요약', '정리', '후기', '내용', '중계', '발표', '텍스트', '스크립트', '모음']
                        
                        has_stream = any(k in text_lower for k in stream_keywords)
                        has_type = any(k in text_lower for k in type_keywords)
                        
                        if has_stream and has_type:
                            score = 100 # Passing Grade
                        
                            # Bonus points for specificity
                            if '로아온' in text_lower: score += 50
                            if '요약' in text_lower: score += 30
                            if 'on air' in text_lower: score += 20
                            
                        return score

                    writer_tag = row.select_one('.user')
                    author = writer_tag.get_text(strip=True) if writer_tag else ""
                    
                    # PRIORITY AUTHOR CHECK
                    is_priority_author = (author == '가지있는나무')
                    
                    # 1. Title Score
                    title_score = calculate_title_score(title)
                    
                    # If Priority Author, bypass title score check (or ensure it passes)
                    if is_priority_author:
                        title_score = max(title_score, 100) # Ensure it passes filter
                        print(f"  [Priority] Found Author '{author}'! Bypass checks.")
                    
                    if title_score < 100:
                        continue

                    # If passed Step 1, proceed to collect
                    reco_tag = row.select_one('.reco')
                    try:
                        reco_count = int(reco_tag.get_text(strip=True)) if reco_tag else 0
                    except:
                        reco_count = 0
                        
                    date_tag = row.select_one('.date')
                    date_str = date_tag.get_text(strip=True) if date_tag else ""
                    
                    post_dt = parse_inven_date(date_str, reference_year=target_dt.year)
                    if post_dt.year != target_dt.year: # Fix year drift
                         if post_dt.month == target_dt.month and post_dt.day == target_dt.day:
                              post_dt = post_dt.replace(year=target_dt.year)
                              
                    # Check Time Window (Still important)
                    diff_hours = (post_dt - target_dt).total_seconds() / 3600
                    if not (-24 <= diff_hours <= 24):
                        continue

                    print(f"  [Pass] {title} | Author: {author} | Score: {title_score} | Reco: {reco_count}")
                    
                    time.sleep(random.uniform(0.1, 0.3))
                    link = subject_tag['href']
                    
                    # Fetch content for length score (Secondary metric, kept for sorting)
                    content = fetch_post_content(link)
                    
                    # LENGTH CHECK
                    # If priority author, ALLOW short content (Trust the author) - user request
                    if len(content) < 100 and not is_priority_author: 
                        print(f"    -> Skipped (Too short)")
                        continue
                        
                    final_score = title_score + (reco_count * 0.5)
                    
                    if is_priority_author:
                        final_score += 9999 # Massive Bonus to guarantee selection
                        
                    formatted_date = post_dt.strftime("%Y-%m-%d %H:%M")
                        
                    all_candidates.append({
                        'title': title,
                        'link': link,
                        'reco': reco_count,
                        'content': content,
                        'date': formatted_date,
                        'score': final_score,
                        'board': board_id,
                        'is_target': True
                    })

            except Exception as e:
                print(f"Error page {page}: {e}")
                
            time.sleep(random.uniform(0.5, 1.0))

    if not all_candidates:
        print("No suitable summary posts found.")
        return None
        
    # Save candidates to CSV for Dashboard
    df = pd.DataFrame(all_candidates)
    df = df.sort_values(by='score', ascending=False)
    
    # Append if exists, or overwrite? 
    # For now, append to keep history, or overwrite for fresh view?
    # User flow suggests selecting *from* candidates. Overwrite is cleaner for "Current Focus".
    # But user might want history. Let's append but remove duplicates.
    
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            combined = pd.concat([existing_df, df]).drop_duplicates(subset=['link'])
            combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        except:
             df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    best_post = df.iloc[0]
    print(f"\n[Winner] {best_post['title']} (Score: {best_post['score']:.1f}) | Board: {best_post['board']}")
    return best_post


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target Event Date (YYYY-MM-DD HH:MM)", default="2026-01-16 20:00")
    parser.add_argument("--pages", help="Max pages to scan AFTER seek", type=int, default=10) # Reduced default
    parser.add_argument("--window", help="Search window in hours", type=int, default=8)
    args = parser.parse_args()
    
    find_best_reaction_post(args.date, window_hours=args.window, max_pages_to_scan=args.pages)
