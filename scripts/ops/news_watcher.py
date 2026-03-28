
import requests
import time
import re
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# You would install beautifulsoup4 usually, but we can do regex for simple scraping to avoid dependency hell in this env if bs4 isn't guaranteed.
# Actually, let's try to be robust. Is bs4 available? Usually standard in data envs. 
# If not, I'll fallback to regex.

def fetch_latest_news():
    url = "https://lostark.game.onstove.com/News/Notice/List"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"[News] Error fetching news: {response.status_code}")
            return None
        
        html = response.text
        
        # Try a more robust pattern based on the unique URL structure
        # Matches: <a href="/News/Notice/Views/12345?page=..."> ... <span class="list__title"> ... <span class="text">TITLE</span>
        # But let's just grab the text inside the anchor that looks like a title
        
        # Pattern: look for the NoticeViews URL, then grab the title text which is usually in a span inside
        # Let's find all notice IDs first, as that is reliable
        notice_ids = re.findall(r'/News/Notice/Views/(\d+)\?', html)
        
        if notice_ids:
            latest_id = notice_ids[0]
            # Now try to find the title associated with this ID
            # It's usually following the href.
            # Let's just grab the whole block around this ID
            block_match = re.search(f'/News/Notice/Views/{latest_id}\?.*?(<span class="list__title">.*?)(</a>|</div>)', html, re.DOTALL)
            
            title = "Unknown Title"
            if block_match:
                block_html = block_match.group(1)
                # Remove tags
                clean_text = re.sub(r'<[^>]+>', ' ', block_html).strip()
                # Remove extra spaces
                title = re.sub(r'\s+', ' ', clean_text)
                
            return {"id": latest_id, "title": title, "url": f"https://lostark.game.onstove.com/News/Notice/Views/{latest_id}"}
        
        return None

    except Exception as e:
        print(f"[News] Exception: {e}")
        return None

def analyze_with_perplexity(news_item, api_key):
    """
    Sends the news to Perplexity API for analysis.
    """
    print(f"\n[Analyst] Analyzing news: '{news_item['title']}'...")
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prompt engineering for the analyst
    prompt = f"""
    Analyze the following Lost Ark news title for its potential impact on the in-game market economy (Gold price, Item prices).
    
    News Title: "{news_item['title']}"
    
    Respond in JSON format:
    {{
        "score": <int 1-10>,
        "reason": "<short explanation>"
    }}
    SCORE GUIDE:
    1-3: Routine/Minor (Maintenance check, trivial events)
    4-6: Moderate (New skins, small balance patches)
    7-8: High (New Raid, New Class, Gold Sink updates)
    9-10: Critical (Economy overhaul, Major controversy)
    """
    
    payload = {
        "model": "sonar-pro", # or sonar-reasoning-pro
        "messages": [
            {"role": "system", "content": "You are an expert Lost Ark market analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Simple parsing of the JSON response from text (or use regex if model is chatty)
            # Improving robustness by looking for the json block
            import json
            try:
                # remove markdown code blocks if present
                clean_content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_content)
                return data.get('score', 0), data.get('reason', 'No reason provided')
            except:
                # Fallback if model didn't return pure JSON
                return 5, f"Parse Error. Raw: {content[:50]}..."
        else:
            print(f"[API Error] {response.status_code}: {response.text}")
            return 0, "API Request Failed"
            
    except Exception as e:
        print(f"[API Exception] {e}")
        return 0, f"Exception: {e}"

def main():
    print("--- Lost Ark News Watcher Started (Ctrl+C to stop) ---")
    
    # Check for API Key
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️  WARNING: 'PERPLEXITY_API_KEY' environment variable not found.")
        print("   Please set it using: $env:PERPLEXITY_API_KEY='pplx-...'")
        print("   Running in LIMITED mode (Fetching only, No Analysis)")
    
    last_id = None
    
    # Loop for demonstration (run once here, but designed for loop)
    # while True: 
    news = fetch_latest_news()
    if news:
        print(f"[Watcher] Latest News: {news['title']} (ID: {news['id']})")
        
        if last_id is None or news['id'] != last_id:
            if api_key:
                score, reason = analyze_with_perplexity(news, api_key)
                print(f"[Perplexity] Impact Score: {score}/10")
                print(f"[Perplexity] Reason: {reason}")
                
                if score >= 7:
                    print(f"🚨 ALERT: Significant Market News Detected! 🚨")
            else:
                print("ℹ️  Skipping analysis (No API Key).")
            
            last_id = news['id']
    else:
        print("[Watcher] Could not find latest news.")
    
    # time.sleep(300)
