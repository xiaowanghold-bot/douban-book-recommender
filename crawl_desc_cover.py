#!/usr/bin/env python
"""Crawl book descriptions and covers from Douban"""
import requests, re, time, os, random, json, sys
from datetime import datetime
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DETAIL_FILE = os.path.join(DATA_DIR, 'raw', 'Books_detail.csv')
DESC_OUTPUT = os.path.join(DATA_DIR, 'processed', 'book_descriptions.json')
COVER_OUTPUT = os.path.join(DATA_DIR, 'processed', 'book_covers.json')
COVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'covers')
PROGRESS_FILE = os.path.join(DATA_DIR, 'raw', 'desc_crawl_progress.txt')
MAX_BOOKS = 500

os.makedirs(COVER_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

def load_books():
    df = pd.read_csv(DETAIL_FILE, encoding='utf-8-sig')
    success = df[df['crawl_status'] == 'success'].sort_values('Votes', ascending=False)
    return success['ID'].astype(int).tolist()[:MAX_BOOKS]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_progress(idx):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(idx))

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def extract_description(html):
    patterns = [
        r'<div class="intro">\s*<p>(.*?)</p>',
        r'id="link-report".*?<div class="intro">\s*<p>(.*?)</p>',
        r'<div class="intro">(.*?)</div>',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL)
        if m:
            text = re.sub(r'<.*?>', '', m.group(1)).strip()
            text = re.sub(r'\s+', ' ', text)[:600]
            if len(text) > 15:
                return text
    return None

def extract_cover(html, bid):
    m = re.search(r'src="([^"]*img\d+\.doubanio\.com[^"]*view/subject/[^"]*\.(?:jpg|png|webp))"', html)
    if not m:
        m = re.search(r'<img[^>]*src="([^"]*doubanio\.com[^"]*)"', html)
    if not m:
        return None
    cover_url = m.group(1)
    try:
        img_resp = requests.get(cover_url, headers=HEADERS, timeout=15)
        if img_resp.status_code == 200:
            if '.png' in cover_url:
                ext = 'png'
            elif '.webp' in cover_url:
                ext = 'webp'
            else:
                ext = 'jpg'
            fname = f'{bid}.{ext}'
            fpath = os.path.join(COVER_DIR, fname)
            with open(fpath, 'wb') as f:
                f.write(img_resp.content)
            return fname
    except:
        pass
    return None

def crawl():
    print('=' * 60)
    print('  Douban Book Description and Cover Crawler')
    print(f'  Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    book_ids = load_books()
    print(f'Total books to process: {len(book_ids)}')
    start_idx = load_progress()
    print(f'Resume from index: {start_idx}')
    descriptions = load_json(DESC_OUTPUT)
    covers = load_json(COVER_OUTPUT)
    print(f'Existing: {len(descriptions)} descriptions, {len(covers)} covers')
    new_desc = 0
    new_covers = 0
    for i in range(start_idx, len(book_ids)):
        bid = book_ids[i]
        bid_str = str(bid)
        if bid_str in descriptions and bid_str in covers:
            continue
        try:
            url = f'https://book.douban.com/subject/{bid}/'
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
            html = resp.text
            if bid_str not in descriptions:
                desc = extract_description(html)
                if desc:
                    descriptions[bid_str] = desc
                    new_desc += 1
            if bid_str not in covers:
                cover = extract_cover(html, bid)
                if cover:
                    covers[bid_str] = cover
                    new_covers += 1
        except Exception as e:
            pass
        if (i - start_idx + 1) % 30 == 0:
            pct = (i - start_idx + 1) / (len(book_ids) - start_idx) * 100
            print(f'[{pct:5.1f}%] {i+1}/{len(book_ids)} desc:{len(descriptions)} covers:{len(covers)} new_desc:{new_desc} new_covers:{new_covers}')
            save_progress(i + 1)
            save_json(DESC_OUTPUT, descriptions)
            save_json(COVER_OUTPUT, covers)
        time.sleep(random.uniform(0.8, 1.5))
    save_progress(len(book_ids))
    save_json(DESC_OUTPUT, descriptions)
    save_json(COVER_OUTPUT, covers)
    print(f'\nDone! Total descriptions: {len(descriptions)}, Total covers: {len(covers)}')
    print(f'New this session: {new_desc} descriptions, {new_covers} covers')

if __name__ == '__main__':
    crawl()