import requests
import re
import csv
import time
import os
from datetime import datetime

# Output file
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'events_official.csv')

def crawl_official_notices(max_pages=100):
    base_url = "https://lostark.game.onstove.com/News/Notice/List"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
def crawl_official_notices(max_pages=100):
    base_url = "https://lostark.game.onstove.com/News/Notice/List"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Load Existing Data into Dict for Update Checking
    # Key: Link, Value: Dict (Row)
    notice_db = {}
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    notice_db[row['link']] = row
            print(f"Loaded {len(notice_db)} existing notices.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            
    updates_count = 0
    new_count = 0
    consecutive_unchanged = 0
    stop_threshold = 3  # Stop after seeing 3 unchanged normal items
    stop_crawl = False
    
    print(f"Starting crawl of official notices (Max Pages: {max_pages})...")
    
    for page in range(1, max_pages + 1):
        if stop_crawl:
            break
            
        print(f"Scanning Page {page}/{max_pages}...")
        try:
            params = {'page': page}
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"Failed to fetch page {page}: {response.status_code}")
                continue
                
            html = response.text
            
            li_pattern = r'<li class="(.*?)">(.*?)</li>'
            list_items = re.findall(li_pattern, html, re.DOTALL)
            
            if not list_items:
                print("No list items found (Regex mismatch?).")
                continue

            for item_cls, content in list_items:
                is_pinned = 'notice' in item_cls.lower() or 'noti' in item_cls.lower()
                
                # Link & ID
                link_match = re.search(r'href="/News/Notice/(?:Notice)?Views/(\d+)\?.*?"', content)
                if not link_match:
                    continue
                notice_id = link_match.group(1)
                full_link = f"https://lostark.game.onstove.com/News/Notice/Views/{notice_id}"
                
                # Title
                title_match = re.search(r'<span class="list__title">(.*?)</span>', content)
                title = title_match.group(1).strip() if title_match else "Unknown Title"
                
                # Category
                cat_match = re.search(r'<span class="icon.*?>(.*?)</span>', content)
                category = cat_match.group(1).strip() if cat_match else "General"
                
                # Date
                date_match = re.search(r'<div class="list__date"[^>]*>(.*?)</div>', content)
                date_str = date_match.group(1).strip() if date_match else "Unknown Date"
                
                try:
                    date_obj = datetime.strptime(date_str, "%Y.%m.%d")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                except:
                    formatted_date = date_str
                
                current_notice = {
                    'date': formatted_date,
                    'category': category,
                    'title': title,
                    'link': full_link
                }
                
                # LOGIC: Check vs DB
                if full_link in notice_db:
                    existing_title = notice_db[full_link]['title']
                    if existing_title != title:
                        print(f"[Update] Title changed: '{existing_title}' -> '{title}'")
                        notice_db[full_link] = current_notice # Update DB
                        updates_count += 1
                        consecutive_unchanged = 0 # Reset counter
                    else:
                        # Exact duplicate (Link + Title matches)
                        if not is_pinned:
                            consecutive_unchanged += 1
                        else:
                            print(f"[Skip] Pinned Notice: '{title}' (Already in DB)")
                            
                        if consecutive_unchanged >= stop_threshold:
                            print(f"Hit {stop_threshold} consecutive unchanged items. Stopping crawl.")
                            stop_crawl = True
                            break
                else:
                    # New Item
                    print(f"[New] {title}")
                    notice_db[full_link] = current_notice
                    new_count += 1
                    consecutive_unchanged = 0 # Reset
                    
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Exception on page {page}: {e}")
            
    # Save Dictionary back to CSV (Sorted by Date Descending)
    sorted_notices = sorted(notice_db.values(), key=lambda x: x['date'], reverse=True)
    
    print(f"Crawl complete. New: {new_count}, Updates: {updates_count}. Total: {len(sorted_notices)}")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'category', 'title', 'link'])
        writer.writeheader()
        writer.writerows(sorted_notices)
        
    print(f"Saved to {OUTPUT_FILE}")
    
    # Trigger Downstream AI Pipeline automatically if new data was found
    if new_count > 0 or updates_count > 0:
        print("\n[Auto-Trigger] New notices detected. Launching downstream AI pipeline...")
        import subprocess
        import sys
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pipeline_scripts = [
            os.path.join(base_dir, 'scripts', 'collection', 'scrape_notice_content.py'),
            os.path.join(base_dir, 'scripts', 'enrichment', 'analyze_events_llm.py'),
            os.path.join(base_dir, 'scripts', 'modeling', 'calculate_actual_impact.py'),
            os.path.join(base_dir, 'scripts', 'modeling', 'train_impact.py')
        ]
        
        for script in pipeline_scripts:
            print(f"  -> Executing: {os.path.basename(script)}")
            try:
                # Use the same python executable environment
                result = subprocess.run([sys.executable, script], check=True, text=True, capture_output=True)
                print(f"     [Success] {os.path.basename(script)}")
                # Optional: Print subset of output for debugging if needed
                # print(result.stdout[:200] + "...") 
            except subprocess.CalledProcessError as e:
                print(f"     [Error] Pipeline halted at {os.path.basename(script)}:")
                print(e.stderr)
                break
                
        print("\n✅ Autonomous Pipeline Execution Completed.")
    else:
        print("\nNo new official notices detected. Downstream pipeline skipped.")

if __name__ == "__main__":
    crawl_official_notices()
