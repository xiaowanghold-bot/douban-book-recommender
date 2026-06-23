#!/usr/bin/env python
import requests, re, time, os, random, json, sys
from datetime import datetime
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
DETAIL_FILE = os.path.join(DATA_DIR, 'raw', 'Books_detail.csv')
COVER_OUTPUT = os.path.join(DATA_DIR, 'processed', 'book_covers.json')
COVER_DIR = os.path.join(BASE, 'app', 'covers')
PROGRESS_FILE = os.path.join(DATA_DIR, 'raw', 'cover_crawl_progress.txt')
MAX_BOOKS = 3000
MIN_IMAGE_SIZE = 3000

os.makedirs(COVER_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'image/webp,*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://book.douban.com/',
}

def load_books():
    df = pd.read_csv(DETAIL_FILE, encoding='utf-8-sig')
    success = df[df['crawl_status'] == 'success'].sort_values('Votes', ascending=False)
    return success['ID'].astype(int).tolist()[:MAX_BOOKS]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as fp:
            return int(fp.read().strip())
    old = os.path.join(DATA_DIR, 'raw', 'desc_crawl_progress.txt')
    if os.path.exists(old):
        with open(old, 'r', encoding='utf-8') as fp:
            return int(fp.read().strip())
    return 0

def save_progress(idx):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as fp:
        fp.write(str(idx))

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    return {}

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)

def download_cover_cdn(book_id):
    for server in [1, 2, 3, 9]:
        url = 'https://img{}.doubanio.com/view/subject/s/public/s{}.jpg'.format(server, book_id)
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and len(r.content) > MIN_IMAGE_SIZE:
                if b'default' not in r.content[:200].lower():
                    return r.content, 'jpg'
        except:
            continue
    for ext in ['png', 'webp']:
        for server in [1, 2, 3, 9]:
            url = 'https://img{}.doubanio.com/view/subject/s/public/s{}.{}'.format(server, book_id, ext)
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code == 200 and len(r.content) > MIN_IMAGE_SIZE:
                    return r.content, ext
            except:
                continue
    return None, None

def crawl():
    print('=' * 60)
    print('  Douban Cover CDN Crawler v2')
    print('  Start: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('=' * 60)
    book_ids = load_books()
    print('Total books: {}'.format(len(book_ids)))
    start_idx = load_progress()
    print('Resume from: {}'.format(start_idx))
    covers = load_json(COVER_OUTPUT)
    print('Existing covers: {}'.format(len(covers)))
    new_covers = 0
    consecutive_failures = 0
    for i in range(start_idx, len(book_ids)):
        bid = book_ids[i]
        bid_str = str(bid)
        if bid_str in covers:
            continue
        try:
            img_bytes, ext = download_cover_cdn(bid)
            if img_bytes:
                fname = '{}.{}'.format(bid, ext)
                fpath = os.path.join(COVER_DIR, fname)
                with open(fpath, 'wb') as fp:
                    fp.write(img_bytes)
                covers[bid_str] = fname
                new_covers += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            processed = i - start_idx + 1
            if processed % 100 == 0:
                total = len(book_ids) - start_idx
                pct = min(100, processed / max(1, total) * 100)
                print('[{:.1f}%] {}/{} covers={} new={}'.format(pct, i+1, len(book_ids), len(covers), new_covers))
                save_progress(i + 1)
                save_json(COVER_OUTPUT, covers)
            delay = random.uniform(0.25, 0.6)
            if consecutive_failures > 30:
                print('  Many failures ({}), pausing 30s...'.format(consecutive_failures))
                time.sleep(30)
                consecutive_failures = 0
            time.sleep(delay)
        except KeyboardInterrupt:
            print('Interrupted at {}! Saving...'.format(i))
            save_progress(i)
            save_json(COVER_OUTPUT, covers)
            return
        except Exception as e:
            consecutive_failures += 1
            time.sleep(0.5)
    save_progress(len(book_ids))
    save_json(COVER_OUTPUT, covers)
    print('Done! Total covers: {} (+{})'.format(len(covers), new_covers))

if __name__ == '__main__':
    crawl()
