import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import random

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_official.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'events_with_content.csv')

def scrape_content():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Check if output exists to resume or skip
    seen_links = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_links.add(row['link'])
    
    import pandas as pd
    try:
        events_df = pd.read_csv(INPUT_FILE)
        events = []
        for _, row in events_df.iterrows():
            events.append({
                'date': str(row['date']),
                'event_type': row.get('event_type', 'Official Notice'),
                'original_category': row.get('category', row.get('original_category', '공지')),
                'title': str(row['title']),
                'link': str(row['link'])
            })
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    print(f"Total events to process: {len(events)}")
    
    # Open output for appending if exists, else write header
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    with open(OUTPUT_FILE, mode, encoding='utf-8', newline='') as f:
        fieldnames = ['date', 'event_type', 'original_category', 'title', 'link', 'content']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        
        count = 0
        for event in events:
            link = event['link']
            if link in seen_links:
                continue

            print(f"Scraping: {event['title']} ({link})")
            
            try:
                response = requests.get(link, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Selector identified: div.fr-view
                    content_div = soup.select_one('div.fr-view')
                    if content_div:
                        text = content_div.get_text(separator='\n', strip=True)
                    else:
                        text = "Content not found"
                else:
                    text = f"Error: {response.status_code}"
            except Exception as e:
                text = f"Error: {str(e)}"
            
            event['content'] = text
            writer.writerow(event)
            f.flush() # Ensure write
            
            count += 1
            if count % 10 == 0:
                print(f"Scraped {count} items...")
            
            # Polymorphism: Sleep random time to be polite
            time.sleep(random.uniform(0.1, 0.3))

    print("Scraping completed.")

if __name__ == "__main__":
    scrape_content()
